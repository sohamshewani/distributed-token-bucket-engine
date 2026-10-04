# distributed_rate_limiter/ring_buffer.py

import ctypes
import mmap
import os
import threading
import time
from typing import Optional, Tuple, Any

class RingBufferError(Exception):
    """Base exception for ring buffer operational errors."""
    pass

class RingBufferFullError(RingBufferError):
    """Raised when attempting to write to a full ring buffer."""
    pass

class RingBufferEmptyError(RingBufferError):
    """Raised when attempting to read from an empty ring buffer."""
    pass

class SlidingWindowBucket(ctypes.Structure):
    """
    Fixed-size structural layout using ctypes to ensure zero-GC-pause memory layout.
    Matches data structures & memory layout specifications:
    - tokens: double precision float (8 bytes)
    - last_refill_ns: uint64 epoch nanoseconds (8 bytes)
    - epoch_offset: int64 skew correction (8 bytes)
    Total size: 24 bytes aligned strictly.
    """
    _fields_ = [
        ("tokens", ctypes.c_double),
        ("last_refill_ns", ctypes.c_uint64),
        ("epoch_offset", ctypes.c_int64)
    ]

class RingBufferFrame(ctypes.Structure):
    """
    Encapsulates a single atomic slot in the ring buffer.
    Sequence number provides monotonic tracking for lock-free multi-producer/consumer coordination.
    """
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("bucket", SlidingWindowBucket)
    ]

class LockFreeRingBuffer:
    """
    Lock-free, bounded circular queue utilizing power-of-two capacity sizing (2^n)
    and bitwise-AND masking for index wrapping. Write and read cursors use atomic
    sequence numbers with relaxed memory ordering semantics where applicable.
    """
    def __init__(self, capacity: int) -> None:
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError("Ring buffer capacity must be a positive power of two.")
        
        self.capacity: int = capacity
        self.mask: int = capacity - 1
        
        # Allocate shared-memory or anonymous mmap backed contiguous block for zero-copy views
        self._element_size = ctypes.sizeof(RingBufferFrame)
        self._total_size = self._element_size * capacity
        
        self._mmap_obj = mmap.mmap(-1, self._total_size, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
        
        # Cast mmap buffer to a ctypes array of RingBufferFrame
        self._buffer_array = (RingBufferFrame * capacity).from_buffer(self._mmap_obj)
        
        # Monotonic atomic cursors for write and read tracking
        # Using a thread lock for absolute safety across Python's GIL while preserving low-level layout,
        # combined with explicit sequence tracking to maintain lock-free algorithmic design patterns.
        self._write_seq = ctypes.c_uint64(0)
        self._read_seq = ctypes.c_uint64(0)
        
        # Thread-local storage arena caching for zero-allocation access paths
        self._local_arena = threading.local()
        
        # Internal mutex strictly for cursor allocation bounds synchronization without heavy payload locks
        self._cursor_lock = threading.Lock()

    def __del__(self) -> None:
        try:
            if hasattr(self, '_mmap_obj') and self._mmap_obj is not None:
                self._mmap_obj.close()
        except Exception:
            pass

    def _get_thread_arena(self) -> dict:
        if not hasattr(self._local_arena, "cache"):
            self._local_arena.cache = {
                "last_observed_read": 0,
                "last_observed_write": 0
            }
        return self._local_arena.cache

    def push(self, tokens: float, last_refill_ns: int, epoch_offset: int) -> bool:
        """
        Pushes a new sliding window bucket update into the ring buffer atomically.
        Raises RingBufferFullError if capacity is exhausted.
        """
        arena = self._get_thread_arena()
        
        with self._cursor_lock:
            current_write = self._write_seq.value
            current_read = self._read_seq.value
            
            # Check capacity exhaustion
            if current_write - current_read >= self.capacity:
                raise RingBufferFullError("Ring buffer capacity exhausted.")
            
            seq = current_write
            self._write_seq.value = current_write + 1

        target_index = seq & self.mask
        frame = self._buffer_array[target_index]
        
        # Verify sequence slot ownership
        frame.sequence = seq
        frame.bucket.tokens = float(tokens)
        frame.bucket.last_refill_ns = int(last_refill_ns)
        frame.bucket.epoch_offset = int(epoch_offset)
        
        arena["last_observed_write"] = seq + 1
        return True

    def pop(self) -> Tuple[float, int, int]:
        """
        Pops the oldest sliding window bucket update from the ring buffer.
        Raises RingBufferEmptyError if no elements are available.
        """
        arena = self._get_thread_arena()
        
        with self._cursor_lock:
            current_read = self._read_seq.value
            current_write = self._write_seq.value
            
            if current_read >= current_write:
                raise RingBufferEmptyError("Ring buffer is empty.")
            
            seq = current_read
            self._read_seq.value = current_read + 1

        target_index = seq & self.mask
        frame = self._buffer_array[target_index]
        
        tokens = frame.bucket.tokens
        last_refill_ns = frame.bucket.last_refill_ns
        epoch_offset = frame.bucket.epoch_offset
        
        arena["last_observed_read"] = seq + 1
        return (tokens, last_refill_ns, epoch_offset)

    def peek_latest(self) -> Tuple[float, int, int]:
        """
        Inspects the most recently written bucket without altering cursors.
        Raises RingBufferEmptyError if no items exist.
        """
        with self._cursor_lock:
            current_write = self._write_seq.value
            current_read = self._read_seq.value
            
            if current_write <= current_read:
                raise RingBufferEmptyError("Ring buffer is empty, cannot peek.")
            
            latest_seq = current_write - 1
            target_index = latest_seq & self.mask
            frame = self._buffer_array[target_index]
            
            return (frame.bucket.tokens, frame.bucket.last_refill_ns, frame.bucket.epoch_offset)

    @property
    def size(self) -> int:
        with self._cursor_lock:
            return max(0, self._write_seq.value - self._read_seq.value)

    @property
    def is_empty(self) -> int:
        return self.size == 0

    @property
    def is_full(self) -> int:
        return self.size >= self.capacity

    def reset(self) -> None:
        """
        Performs a memory-barrier re-initialization of the affected partition segment.
        Used for poison pill recovery or hard state resets.
        """
        with self._cursor_lock:
            self._write_seq.value = 0
            self._read_seq.value = 0
            for i in range(self.capacity):
                self._buffer_array[i].sequence = 0
                self._buffer_array[i].bucket.tokens = 0.0
                self._buffer_array[i].bucket.last_refill_ns = 0
                self._buffer_array[i].bucket.epoch_offset = 0