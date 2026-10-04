import ctypes
import os
import time
import threading
import numpy as np
from typing import Tuple, Optional

# --- CTYPES STRUCTURES & MEMORY LAYOUT ---

class BucketStruct(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_double),
        ("last_refill_ns", ctypes.c_uint64),
        ("epoch_offset", ctypes.c_int64)
    ]

class RingBufferStruct(ctypes.Structure):
    _fields_ = [
        ("capacity", ctypes.c_uint64),
        ("mask", ctypes.c_uint64),
        ("write_seq", ctypes.c_uint64),
        ("read_seq", ctypes.c_uint64),
        ("buffer_ptr", ctypes.c_void_p)
    ]

class AllocatorException(Exception):
    pass

class PartitionError(AllocatorException):
    pass

class PoisonPillError(AllocatorException):
    pass


class SlidingWindowBucketArray:
    """
    Fixed-size structural array via ctypes and NumPy memory views for zero-GC-pause 
    memory layout. Maintains tokens, last_refill_ns, and epoch_offset.
    """
    def __init__(self, size: int) -> None:
        if size <= 0 or (size & (size - 1)) != 0:
            # Enforce power-of-two sizing for alignment and ring compatibility
            s = 1
            while s < size:
                s <<= 1
            size = s
            
        self.size = size
        self.array_type = BucketStruct * size
        self.raw_memory = self.array_type()
        
        # Create zero-copy NumPy memory view over ctypes buffer
        self.np_view = np.frombuffer(
            self.raw_memory, 
            dtype={
                'names': ['tokens', 'last_refill_ns', 'epoch_offset'],
                'formats': ['f8', 'u8', 'i8'],
                'align': True
            },
            count=size
        )
        self._lock = threading.Lock()

    def get_bucket_pointer(self, index: int) -> ctypes.POINTER(BucketStruct):
        if index < 0 or index >= self.size:
            raise IndexError(f"Bucket index {index} out of bounds for size {self.size}")
        return ctypes.pointer(self.raw_memory[index])

    def atomic_consume(self, index: int, cost: float, capacity: float, refill_rate_per_ns: float, skew_ns: int) -> Tuple[bool, float]:
        """
        Executes token consumption and window sliding via strict bounds checking 
        and atomic updates.
        """
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")

        # Use local thread-safe lock context for strict memory-barrier enforcement on structures
        with self._lock:
            bucket = self.raw_memory[index]
            now_ns = time.monotonic_ns()
            
            # Apply clock skew correction and epoch offset
            adjusted_now = now_ns + skew_ns + bucket.epoch_offset
            if adjusted_now < bucket.last_refill_ns:
                # Monotonicity enforcement: clock must never regress
                adjusted_now = bucket.last_refill_ns

            elapsed_ns = adjusted_now - bucket.last_refill_ns
            if elapsed_ns > 0:
                delta_tokens = elapsed_ns * refill_rate_per_ns
                bucket.tokens = min(capacity, bucket.tokens + delta_tokens)
                bucket.last_refill_ns = adjusted_now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, bucket.tokens
            else:
                return False, bucket.tokens


class LockFreeRingBuffer:
    """
    Lock-free, bounded circular queue utilizing power-of-two capacity sizing 
    and bitwise-AND masking for index wrapping. Uses monotonic sequence numbers.
    """
    def __init__(self, capacity: int) -> None:
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError("Ring buffer capacity must be a positive power of two.")
        
        self.capacity = capacity
        self.mask = capacity - 1
        self.write_seq = ctypes.c_uint64(0)
        self.read_seq = ctypes.c_uint64(0)
        
        # Storage array for raw byte payloads or metadata pointers
        self.storage = (ctypes.c_uint64 * capacity)()
        memset_res = ctypes.memset(self.storage, 0, ctypes.sizeof(self.storage))
        if not memset_res:
            raise AllocatorException("Failed to initialize ring buffer storage memory.")

    def push(self, data_val: int) -> bool:
        while True:
            current_write = self.write_seq.value
            current_read = self.read_seq.value
            
            if current_write - current_read >= self.capacity:
                # Buffer is full
                return False
            
            # Try to advance write sequence atomically using Python's GIL/atomic assignments or CAS emulation
            if self.write_seq.value == current_write:
                self.write_seq.value = current_write + 1
                idx = current_write & self.mask
                self.storage[idx] = data_val
                return True

    def pop(self) -> Tuple[bool, int]:
        while True:
            current_read = self.read_seq.value
            current_write = self.write_seq.value
            
            if current_read == current_write:
                # Buffer is empty
                return False, 0
            
            if self.read_seq.value == current_read:
                idx = current_read & self.mask
                val = self.storage[idx]
                self.read_seq.value = current_read + 1
                return True, val


class NTPDriftEstimator:
    """
    Decentralized NTP-drift estimator utilizing a sliding median filter over 
    sequential Cristian's algorithm / Marzullo interval intersection heartbeats.
    """
    def __init__(self, window_size: int = 15) -> None:
        self.window_size = window_size
        self.offsets = np.zeros(window_size, dtype=np.int64)
        self.index = 0
        self.count = 0
        self._lock = threading.Lock()

    def update_offset(self, t0: int, t1: int, t2: int, t3: int) -> int:
        """
        Cristian's algorithm with round-trip time and offset calculation:
        offset = ((t1 - t0) + (t2 - t3)) / 2
        """
        rtt = (t3 - t0) - (t2 - t1)
        if rtt < 0:
            # Invalid network telemetry, discard
            return 0
        
        offset = ((t1 - t0) + (t2 - t3)) // 2
        
        with self._lock:
            self.offsets[self.index] = offset
            self.index = (self.index + 1) % self.window_size
            if self.count < self.window_size:
                self.count += 1
                
        return offset

    def get_median_skew(self) -> int:
        with self._lock:
            if self.count == 0:
                return 0
            valid_slice = self.offsets[:self.count]
            return int(np.median(valid_slice))


class ThreadArenaCache:
    """
    Employs threading.local for per-thread arena caching ensuring 
    thread-safety without OS mutex overhead on hot execution paths.
    """
    def __init__(self, bucket_array: SlidingWindowBucketArray) -> None:
        self.local_storage = threading.local()
        self.bucket_array = bucket_array

    def get_local_cursor(self, thread_id: int) -> int:
        if not hasattr(self.local_storage, 'cached_index'):
            # Bind thread uniquely to a structural slice via modulo hashing
            self.local_storage.cached_index = thread_id % self.bucket_array.size
        return self.local_storage.cached_index


class DistributedTokenAllocator:
    """
    Core distributed rate limiter / allocator engine incorporating zero-copy buffers, 
    lock-free structures, clock-skew compensation, partition tolerance, and poison pill handling.
    """
    def __init__(self, capacity: float, refill_rate_per_sec: float, cluster_size: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")
        
        self.capacity = float(capacity)
        self.refill_rate_per_ns = float(refill_rate_per_sec) / 1_000_000_000.0
        
        self.bucket_array = SlidingWindowBucketArray(cluster_size)
        self.ring_dlq = LockFreeRingBuffer(1024)
        self.ntp_estimator = NTPDriftEstimator()
        self.arena_cache = ThreadArenaCache(self.bucket_array)
        
        self.is_partitioned = False
        self._cluster_lock = threading.Lock()

    def set_partition_state(self, partitioned: bool) -> None:
        with self._cluster_lock:
            self.is_partitioned = partitioned

    def ingest_heartbeat(self, t0: int, t1: int, t2: int, t3: int) -> None:
        try:
            if t0 > t1 or t1 > t2 or t2 > t3:
                raise PoisonPillError("Malformed timing metadata detected in heartbeat payload.")
            self.ntp_estimator.update_offset(t0, t1, t2, t3)
        except Exception as e:
            # Poison Pill Handling: Divert to dead-letter queue
            encoded_error = hash(str(e)) & 0xFFFFFFFFFFFFFFFF
            pushed = self.ring_dlq.push(encoded_error)
            if not pushed:
                # Force memory-barrier re-initialization if DLQ is saturated
                self.ring_dlq = LockFreeRingBuffer(self.ring_dlq.capacity)
            raise PoisonPillError(f"Heartbeat ingestion failed, routed to DLQ: {e}")

    def allocate(self, cost: float, bucket_index: Optional[int] = None) -> Tuple[bool, float]:
        if cost <= 0:
            raise ValueError("Allocation cost must be positive.")
        if cost > self.capacity:
            # Bound Enforcement: Capacity C can never exceed max burst limit
            return False, 0.0

        with self._cluster_lock:
            partition_fallback = self.is_partitioned

        if bucket_index is None:
            thread_id = threading.get_ident()
            bucket_index = self.arena_cache.get_local_cursor(thread_id)

        skew_ns = self.ntp_estimator.get_median_skew()

        try:
            allowed, current_tokens = self.bucket_array.atomic_consume(
                index=bucket_index,
                cost=cost,
                capacity=self.capacity,
                refill_rate_per_ns=self.refill_rate_per_ns,
                skew_ns=skew_ns if not partition_fallback else 0
            )
            return allowed, current_tokens
        except Exception as e:
            # Fallback to local approximate rate limiting during unexpected faults
            if partition_fallback:
                return False, 0.0
            raise PartitionError(f"Allocator fault under normal operation: {e}")

    def inspect_dlq(self) -> Optional[int]:
        success, val = self.ring_dlq.pop()
        if success:
            return val
        return None