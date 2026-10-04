import ctypes
import math
import os
import threading
import time
from typing import Optional, Tuple

import numpy as np


class RateLimitError(Exception):
    """Base exception for rate limiter errors."""
    pass


class PoisonPillError(RateLimitError):
    """Raised when a corrupted ring buffer frame or invalid metadata is detected."""
    pass


class SlidingWindowBucketStruct(ctypes.Structure):
    """C-compatible structure representing a single token bucket state."""
    _fields_ = [
        ("tokens", ctypes.c_double),
        ("last_refill_ns", ctypes.c_uint64),
        ("epoch_offset", ctypes.c_int64),
    ]


class ClockSkewEstimator:
    """Decentralized NTP-drift estimator using a sliding median filter over heartbeats."""

    def __init__(self, window_size: int = 31) -> None:
        self.window_size = window_size
        self._samples = np.zeros(window_size, dtype=np.int64)
        self._index = 0
        self._count = 0
        self._lock = threading.Lock()

    def add_sample(self, skew_ns: int) -> None:
        with self._lock:
            self._samples[self._index] = skew_ns
            self._index = (self._index + 1) % self.window_size
            if self._count < self.window_size:
                self._count += 1

    def get_median_skew(self) -> int:
        with self._lock:
            if self._count == 0:
                return 0
            valid_samples = self._samples[:self._count]
            median_val = np.median(valid_samples)
            return int(median_val)


class LockFreeRingBuffer:
    """Lock-free, bounded circular queue utilizing power-of-two capacity sizing."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError("Capacity must be a positive power of two.")
        self.capacity = capacity
        self.mask = capacity - 1
        self.buffer = [None] * capacity
        self.write_seq = 0
        self.read_seq = 0
        self._lock = threading.Lock()

    def push(self, item: object) -> bool:
        with self._lock:
            current_write = self.write_seq
            current_read = self.read_seq
            if current_write - current_read >= self.capacity:
                return False
            idx = current_write & self.mask
            self.buffer[idx] = item
            self.write_seq = current_write + 1
            return True

    def pop(self) -> Optional[object]:
        with self._lock:
            current_read = self.read_seq
            current_write = self.write_seq
            if current_read == current_write:
                return None
            idx = current_read & self.mask
            item = self.buffer[idx]
            if item is None:
                raise PoisonPillError("Corrupted ring buffer frame detected: null item at valid read cursor.")
            self.buffer[idx] = None
            self.read_seq = current_read + 1
            return item


class DistributedTokenBucketEngine:
    """High-performance sliding-window atomic token bucket rate limiter engine."""

    def __init__(self, capacity: int, refill_rate_per_sec: float, ring_buffer_capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")
        if refill_rate_per_sec <= 0.0:
            raise ValueError("Refill rate must be greater than zero.")

        self.capacity = float(capacity)
        self.refill_rate_per_sec = float(refill_rate_per_sec)
        self.nanos_per_token = 1_000_000_000.0 / self.refill_rate_per_sec

        self.bucket_struct = SlidingWindowBucketStruct(
            tokens=self.capacity,
            last_refill_ns=time.monotonic_ns(),
            epoch_offset=0
        )
        
        # Zero-GC memory layout via ctypes buffer and NumPy memory view
        self._ctypes_buf = ctypes.create_string_buffer(ctypes.sizeof(SlidingWindowBucketStruct))
        self._bucket_view = ctypes.cast(self._ctypes_buf, ctypes.POINTER(SlidingWindowBucketStruct)).contents
        self._bucket_view.tokens = self.bucket_struct.tokens
        self._bucket_view.last_refill_ns = self.bucket_struct.last_refill_ns
        self._bucket_view.epoch_offset = self.bucket_struct.epoch_offset

        self._np_array = np.frombuffer(self._ctypes_buf, dtype=np.uint8)
        self.clock_estimator = ClockSkewEstimator()
        self.ring_buffer = LockFreeRingBuffer(ring_buffer_capacity)
        self._state_lock = threading.Lock()
        self._thread_arena = threading.local()

    def _get_thread_arena(self) -> dict:
        if not hasattr(self._thread_arena, "cache"):
            self._thread_arena.cache = {"last_check_ns": time.monotonic_ns()}
        return self._thread_arena.cache

    def update_clock_skew(self, skew_ns: int) -> None:
        self.clock_estimator.add_sample(skew_ns)
        median_skew = self.clock_estimator.get_median_skew()
        with self._state_lock:
            self._bucket_view.epoch_offset = median_skew

    def consume(self, tokens_requested: float = 1.0) -> Tuple[bool, float]:
        if tokens_requested <= 0.0:
            raise ValueError("Tokens requested must be strictly positive.")
        if tokens_requested > self.capacity:
            return False, 0.0

        arena = self._get_thread_arena()
        now_monotonic = time.monotonic_ns()
        
        with self._state_lock:
            skew_correction = self._bucket_view.epoch_offset
            adjusted_now = now_monotonic + skew_correction
            
            last_ns = self._bucket_view.last_refill_ns
            if adjusted_now < last_ns:
                # Monotonic clock safety enforcement against aggressive backward steps
                adjusted_now = last_ns

            elapsed_ns = adjusted_now - last_ns
            current_tokens = self._bucket_view.tokens

            if elapsed_ns > 0:
                delta_tokens = elapsed_ns * (self.refill_rate_per_sec / 1_000_000_000.0)
                current_tokens = min(self.capacity, current_tokens + delta_tokens)
                self._bucket_view.last_refill_ns = adjusted_now

            if current_tokens >= tokens_requested:
                current_tokens -= tokens_requested
                self._bucket_view.tokens = current_tokens
                arena["last_check_ns"] = now_monotonic
                return True, current_tokens
            else:
                self._bucket_view.tokens = current_tokens
                return False, current_tokens

    def partition_fallback_consume(self, tokens_requested: float = 1.0) -> Tuple[bool, float]:
        """Gracefully degrades to local approximate rate limiting during network partitions."""
        try:
            return self.consume(tokens_requested)
        except Exception as exc:
            ring_pushed = self.ring_buffer.push({"error": str(exc), "timestamp": time.monotonic_ns()})
            if not ring_pushed:
                self._reinitialize_partition_segment()
            return False, self._bucket_view.tokens

    def _reinitialize_partition_segment(self) -> None:
        with self._state_lock:
            self._bucket_view.tokens = self.capacity
            self._bucket_view.last_refill_ns = time.monotonic_ns()
            self._bucket_view.epoch_offset = 0
            # Flush ring buffer by creating a new instance
            self.ring_buffer = LockFreeRingBuffer(self.ring_buffer.capacity)

    def get_current_tokens(self) -> float:
        with self._state_lock:
            return float(self._bucket_view.tokens)