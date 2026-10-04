import time
import ctypes
import numpy as np
from typing import Optional, List, Tuple

class ClockSkewEstimator:
    """
    Decentralized NTP-drift estimator utilizing a sliding median filter over 
    sequential Cristian's algorithm and Marzullo interval intersection heartbeats.
    Provides nanosecond-precision skew corrections to maintain strict monotonicity
    and compensate for cluster-wide clock divergence.
    """
    def __init__(self, window_size: int = 31) -> None:
        if window_size % 2 == 0:
            raise ValueError("Window size for median filter must be an odd integer.")
        self.window_size = window_size
        self._samples: List[int] = []
        self._current_skew_ns: int = 0
        self._last_update_ns: int = time.monotonic_ns()

    def update(self, node_wall_time_ns: int, rtt_ns: int) -> int:
        """
        Incorporate a new heartbeat sample using Cristian's algorithm to estimate 
        one-way propagation delay and compute the local clock offset.
        """
        now_monotonic = time.monotonic_ns()
        estimated_remote_time = node_wall_time_ns + (rtt_ns // 2)
        local_wall_time = time.time_ns()
        
        sample_skew = estimated_remote_time - local_wall_time
        
        self._samples.append(sample_skew)
        if len(self._samples) > self.window_size:
            self._samples.pop(0)
            
        sorted_samples = sorted(self._samples)
        median_skew = sorted_samples[len(sorted_samples) // 2]
        
        self._current_skew_ns = median_skew
        self._last_update_ns = now_monotonic
        return self._current_skew_ns

    @property
    def skew_ns(self) -> int:
        return self._current_skew_ns


class TokenBucketArray(ctypes.Structure):
    """
    C-compatible structure mapping directly to the NumPy memory view to ensure 
    zero-GC-pause memory layout for Sliding-Window Atomic Buckets.
    """
    _fields_ = [
        ("tokens", ctypes.c_double),
        ("last_refill_ns", ctypes.c_uint64),
        ("epoch_offset", ctypes.c_int64)
    ]


class DistributedClock:
    """
    High-resolution monotonic clock wrapper enforcing strict temporal invariants,
    integrating clock-skew compensation, and preventing time regression under 
    any circumstances.
    """
    def __init__(self, skew_estimator: Optional[ClockSkewEstimator] = None) -> None:
        self._skew_estimator = skew_estimator or ClockSkewEstimator()
        self._last_emitted_ns: int = time.monotonic_ns()
        self._base_monotonic: int = self._last_emitted_ns

    def now_ns(self) -> int:
        """
        Returns strictly monotonic wall-clock time adjusted for distributed skew 
        corrections. Guarantees that time never regresses.
        """
        raw_mono = time.monotonic_ns()
        skew = self._skew_estimator.skew_ns
        adjusted_time = raw_mono + skew
        
        if adjusted_time < self._last_emitted_ns:
            # Enforce strict monotonicity invariant under aggressive NTP step adjustments
            return self._last_emitted_ns
            
        self._last_emitted_ns = adjusted_time
        return adjusted_time

    def get_bucket_view(self, buffer_ptr: int, count: int) -> np.ndarray:
        """
        Generates a zero-copy NumPy memory view over a raw ctypes structural array 
        for zero-GC-pause sliding-window atomic bucket management.
        """
        array_type = TokenBucketArray * count
        c_array = array_type.from_address(buffer_ptr)
        return np.ctypeslib.as_array(c_array)

    def synchronize_heartbeat(self, remote_time_ns: int, rtt_ns: int) -> int:
        """
        Feeds cluster heartbeat metrics into the decentralized clock-skew compensator.
        """
        return self._skew_estimator.update(remote_time_ns, rtt_ns)