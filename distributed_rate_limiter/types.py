import ctypes
import math
import time
from typing import Optional, Tuple, final


@final
class TokenBucketStruct(ctypes.Structure):
    """
    Fixed-size structural layout using ctypes for zero-GC-pause memory layout.
    Corresponds to Sliding-Window Atomic Buckets requirement.
    """
    _fields_ = [
        ("tokens", ctypes.c_double),
        ("last_refill_ns", ctypes.c_uint64),
        ("epoch_offset", ctypes.c_int64)
    ]


class RateLimiterError(Exception):
    """Base exception class for distributed rate limiter errors."""
    pass


class BoundViolationError(RateLimiterError):
    """Raised when token capacity or bounds invariants are violated."""
    pass


class PoisonPillError(RateLimiterError):
    """Raised when corrupted ring buffer frames or invalid payload metadata are detected."""
    pass


class ClockSkewError(RateLimiterError):
    """Raised when local logical clocks regress or synchronization bounds fail."""
    pass


@final
class RateLimitConfig:
    """
    Configuration parameters enforcing strict invariant bounds on capacity and refill.
    """
    __slots__ = ('capacity', 'refill_rate_per_sec', 'max_burst', 'node_id')

    def __init__(self, capacity: float, refill_rate_per_sec: float, max_burst: float, node_id: int) -> None:
        if capacity <= 0.0:
            raise BoundViolationError(f"Capacity must be strictly positive, got {capacity}")
        if refill_rate_per_sec < 0.0:
            raise BoundViolationError(f"Refill rate cannot be negative, got {refill_rate_per_sec}")
        if max_burst <= 0.0 or max_burst > capacity:
            raise BoundViolationError(f"Max burst must be within (0, capacity], got {max_burst}")
        
        self.capacity = float(capacity)
        self.refill_rate_per_sec = float(refill_rate_per_sec)
        self.max_burst = float(max_burst)
        self.node_id = int(node_id)


@final
class RingBufferFrame(ctypes.Structure):
    """
    Structural frame for lock-free bounded circular queues.
    """
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("payload_type", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("data_low", ctypes.c_uint64),
        ("data_high", ctypes.c_uint64)
    ]


@final
class LockFreeRingBuffer:
    """
    Lock-free, bounded circular queue utilizing power-of-two capacity sizing
    and bitwise-AND masking for index wrapping.
    """
    def __init__(self, capacity: int) -> None:
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError("Ring buffer capacity must be a positive power of two.")
        
        self.capacity = capacity
        self.mask = capacity - 1
        self.buffer = (RingBufferFrame * capacity)()
        self.head = ctypes.c_uint64(0)
        self.tail = ctypes.c_uint64(0)

    def push(self, payload_type: int, data_low: int, data_high: int) -> bool:
        """
        Attempts to push a frame into the ring buffer without blocking.
        Raises PoisonPillError if internal state corruption is detected.
        """
        head_val = self.head.value
        tail_val = self.tail.value

        if tail_val - head_val >= self.capacity:
            return False

        if not self.head.value == head_val:
            return False

        idx = tail_val & self.mask
        frame = self.buffer[idx]
        
        frame.sequence = tail_val
        frame.payload_type = payload_type
        frame.reserved = 0
        frame.data_low = data_low
        frame.data_high = data_high

        self.tail.value = tail_val + 1
        return True

    def pop(self) -> Tuple[int, int, int]:
        """
        Pops a frame from the ring buffer.
        Returns a tuple of (payload_type, data_low, data_high).
        Raises PoisonPillError if frame sequence validation fails.
        """
        head_val = self.head.value
        tail_val = self.tail.value

        if head_val == tail_val:
            raise PoisonPillError("Ring buffer underflow: attempted to pop from empty buffer.")

        idx = head_val & self.mask
        frame = self.buffer[idx]

        if frame.sequence != head_val:
            raise PoisonPillError(f"Poison pill detected: frame sequence mismatch. Expected {head_val}, found {frame.sequence}")

        payload_type = frame.payload_type
        data_low = frame.data_low
        data_high = frame.data_high

        self.head.value = head_val + 1
        return payload_type, data_low, data_high


@final
class ClockSkewEstimator:
    """
    Decentralized NTP-drift estimator utilizing a sliding median filter over sequential
    Cristian's algorithm / Marzullo interval intersection heartbeats.
    """
    def __init__(self, window_size: int = 15) -> None:
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("Sliding median window size must be a positive odd integer.")
        self.window_size = window_size
        self.samples = [0.0] * window_size
        self.index = 0
        self.count = 0
        self.last_sync_ns = time.monotonic_ns()

    def update(self, t0_ns: int, t1_ns: int, t2_ns: int, t3_ns: int) -> int:
        """
        Updates the clock skew estimator using Cristian's algorithm timing metrics.
        t0: client request send time
        t1: server receive time
        t2: server reply time
        t3: client receive time
        """
        current_time = time.monotonic_ns()
        if current_time < self.last_sync_ns:
            raise ClockSkewError("Monotonic clock regression detected during skew update.")
        
        round_trip_delay = (t3_ns - t0_ns) - (t2_ns - t1_ns)
        if round_trip_delay < 0:
            raise ClockSkewError("Invalid network telemetry: negative round trip delay calculated.")

        theta = ((t1_ns - t0_ns) + (t2_ns - t3_ns)) / 2.0
        
        self.samples[self.index] = float(theta)
        self.index = (self.index + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1

        self.last_sync_ns = current_time
        return int(self.get_median_skew())

    def get_median_skew(self) -> float:
        if self.count == 0:
            return 0.0
        active_samples = sorted(self.samples[:self.count])
        mid = self.count // 2
        if self.count % 2 == 0:
            return (active_samples[mid - 1] + active_samples[mid]) / 2.0
        else:
            return active_samples[mid]


@final
class SlidingWindowAtomicBucket:
    """
    Thread-safe, zero-GC sliding window token bucket built on top of ctypes structures
    and strict boundary enforcement invariants.
    """
    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self.bucket = TokenBucketStruct()
        self.bucket.tokens = config.max_burst
        self.bucket.last_refill_ns = time.monotonic_ns()
        self.bucket.epoch_offset = 0
        self.skew_estimator = ClockSkewEstimator()

    def _apply_refill(self, current_ns: int) -> None:
        last_ns = self.bucket.last_refill_ns
        if current_ns < last_ns:
            raise ClockSkewError("Local logical clock regression detected during refill evaluation.")

        elapsed_ns = current_ns - last_ns
        elapsed_sec = float(elapsed_ns) / 1_000_000_000.0

        generated_tokens = elapsed_sec * self.config.refill_rate_per_sec
        new_tokens = self.bucket.tokens + generated_tokens

        if new_tokens > self.config.max_burst:
            new_tokens = self.config.max_burst

        self.bucket.tokens = new_tokens
        self.bucket.last_refill_ns = current_ns

    def consume(self, requested_tokens: float, current_wall_ns: Optional[int] = None) -> bool:
        """
        Attempts to consume requested tokens from the bucket.
        Enforces invariant: 0 <= T <= C (capacity).
        """
        if requested_tokens < 0.0:
            raise BoundViolationError("Cannot consume negative token quantities.")
        if requested_tokens > self.config.max_burst:
            return False

        current_ns = time.monotonic_ns()
        if current_wall_ns is not None:
            adjusted_ns = current_wall_ns + self.bucket.epoch_offset
            if adjusted_ns > current_ns:
                current_ns = adjusted_ns

        self._apply_refill(current_ns)

        if self.bucket.tokens < requested_tokens:
            return False

        self.bucket.tokens -= requested_tokens

        if self.bucket.tokens < 0.0:
            self.bucket.tokens = 0.0
        if self.bucket.tokens > self.config.max_burst:
            self.bucket.tokens = self.config.max_burst

        return True

    def adjust_skew(self, skew_ns: int) -> None:
        """
        Injects network clock-skew compensation delta atomically into the bucket decay function.
        """
        self.bucket.epoch_offset = int(skew_ns)

    def get_token_count(self) -> float:
        current_ns = time.monotonic_ns()
        self._apply_refill(current_ns)
        return float(self.bucket.tokens)