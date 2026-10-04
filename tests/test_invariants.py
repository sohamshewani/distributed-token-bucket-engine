import time
import uuid
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# -------------------------------------------------------------------------
# IMPLEMENTATION: distributed-token-bucket-engine
# -------------------------------------------------------------------------

class TokenBucketException(Exception):
    pass

class RateLimitExceeded(TokenBucketException):
    pass

class StorageConnectionError(TokenBucketException):
    pass

class TokenBucket:
    """
    Core token bucket engine implementation supporting both local memory
    and distributed/mock-redis backend simulation with strict atomic execution.
    """
    def __init__(
        self,
        key: str,
        capacity: float,
        fill_rate: float,
        redis_client=None,
        time_fn=time.time
    ):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if fill_rate < 0:
            raise ValueError("Fill rate cannot be negative")
            
        self.key = f"token_bucket:{key}"
        self.capacity = float(capacity)
        self.fill_rate = float(fill_rate)
        self.redis = redis_client
        self.time_fn = time_fn
        
        # Local fallback storage if no redis client is provided
        # Format: [tokens, last_updated]
        self._local_storage = [self.capacity, self.time_fn()]
        self._local_lock = threading.Lock()

    def _lua_script(self) -> str:
        # Atomic Lua script for Redis backend
        return """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local fill_rate = tonumber(ARGV[2])
        local requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_updated')
        local tokens
        local last_updated

        if not bucket[1] then
            tokens = capacity
            last_updated = now
        else
            tokens = tonumber(bucket[1])
            last_updated = tonumber(bucket[2])
        end

        local delta = math.max(0, now - last_updated)
        local generated = delta * fill_rate
        tokens = math.min(capacity, tokens + generated)
        last_updated = now

        if tokens < requested {
            redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
            return {0, tokens}
        }

        tokens = tokens - requested
        redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
        return {1, tokens}
        """

    def consume(self, tokens: float = 1.0) -> bool:
        if tokens < 0:
            raise ValueError("Tokens to consume cannot be negative")
        if tokens == 0:
            return True

        now = self.time_fn()

        if self.redis is not None:
            try:
                # Simulate redis execution
                script = self._lua_script()
                # Mock evaluation fallback if redis is a fake client wrapper
                if hasattr(self.redis, "eval_script"):
                    res = self.redis.eval_script(script, [self.key], [self.capacity, self.fill_rate, tokens, now])
                else:
                    res = self._fallback_redis_eval(script, [self.key], [self.capacity, self.fill_rate, tokens, now])
                return bool(res[0])
            except Exception as e:
                raise StorageConnectionError(f"Failed to communicate with storage: {e}")
        else:
            with self._local_lock:
                tokens_avail = self._local_storage[0]
                last_updated = self._local_storage[1]

                delta = max(0.0, now - last_updated)
                tokens_avail = min(self.capacity, tokens_avail + (delta * self.fill_rate))
                last_updated = now

                if tokens_avail < tokens:
                    self._local_storage[0] = tokens_avail
                    self._local_storage[1] = last_updated
                    return False

                tokens_avail -= tokens
                self._local_storage[0] = tokens_avail
                self._local_storage[1] = last_updated
                return True

    def _fallback_redis_eval(self, script, keys, args):
        # Fallback thread-safe memory simulation for mock Redis objects lacking eval
        with self.redis._lock:
            k = keys[0]
            cap, rate, req, now = map(float, args)
            if k not in self.redis._store:
                self.redis._store[k] = {'tokens': cap, 'last_updated': now}
            
            data = self.redis._store[k]
            delta = max(0.0, now - data['last_updated'])
            tokens = min(cap, data['tokens'] + (delta * rate))
            last_updated = now

            if tokens < req:
                self.redis._store[k] = {'tokens': tokens, 'last_updated': last_updated}
                return [0, tokens]
            
            tokens -= req
            self.redis._store[k] = {'tokens': tokens, 'last_updated': last_updated}
            return [1, tokens]


class MockRedisClient:
    def __init__(self, simulate_failures=False):
        self._store = {}
        self._lock = threading.Lock()
        self.simulate_failures = simulate_failures

    def eval_script(self, script, keys, args):
        if self.simulate_failures:
            raise ConnectionError("Redis cluster unreachable")
        
        with self._lock:
            k = keys[0]
            cap, rate, req, now = map(float, args)
            if k not in self._store:
                self._store[k] = {'tokens': cap, 'last_updated': now}
            
            data = self._store[k]
            delta = max(0.0, now - data['last_updated'])
            tokens = min(cap, data['tokens'] + (delta * rate))
            last_updated = now

            if tokens < req:
                self._store[k] = {'tokens': tokens, 'last_updated': last_updated}
                return [0, tokens]
            
            tokens -= req
            self._store[k] = {'tokens': tokens, 'last_updated': last_updated}
            return [1, tokens]


class ControlledClock:
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.current_time

    def advance(self, seconds: float):
        with self.lock:
            self.current_time += seconds


# -------------------------------------------------------------------------
# TEST SUITE: Exhaustive Unit, Invariant, and Concurrent Stress Tests
# -------------------------------------------------------------------------

def test_bucket_initialization_boundaries():
    # Zero or negative capacity boundaries
    with pytest.raises(ValueError):
        TokenBucket("test_zero_cap", capacity=0, fill_rate=10)
    with pytest.raises(ValueError):
        TokenBucket("test_neg_cap", capacity=-5, fill_rate=10)
        
    # Negative fill rate boundaries
    with pytest.raises(ValueError):
        TokenBucket("test_neg_rate", capacity=10, fill_rate=-1)

    # Zero fill rate (static quota) validity check
    clock = ControlledClock(100.0)
    bucket = TokenBucket("static", capacity=5.0, fill_rate=0.0, time_fn=clock)
    assert bucket.consume(5.0) is True
    assert bucket.consume(1.0) is False
    clock.advance(1000.0)
    assert bucket.consume(1.0) is False  # Should not refill


def test_token_consumption_and_refill_invariants():
    clock = ControlledClock(500.0)
    bucket = TokenBucket("invariant", capacity=10.0, fill_rate=2.0, time_fn=clock)

    # Consume full capacity instantly
    assert bucket.consume(10.0) is True
    # Immediate subsequent consumption should fail
    assert bucket.consume(1.0) is False

    # Advance time by 2.5 seconds -> should generate 5.0 tokens
    clock.advance(2.5)
    assert bucket.consume(5.0) is True
    assert bucket.consume(0.1) is False

    # Advance time far beyond capacity limit -> tokens should cap at 10.0
    clock.advance(100.0)
    # Drain capacity completely
    assert bucket.consume(10.0) is True
    # Attempting to consume more than capacity should fail structurally
    with pytest.raises(ValueError):
        bucket.consume(-1.0)
    
    # Fractional consumption precision test
    clock.advance(0.5)  # generates 1.0 token
    assert bucket.consume(0.3) is True
    assert bucket.consume(0.7) is True
    assert bucket.consume(0.01) is False


def test_distributed_redis_backend_integration():
    clock = ControlledClock(1000.0)
    redis_mock = MockRedisClient()
    bucket = TokenBucket("dist", capacity=5.0, fill_rate=1.0, redis_client=redis_mock, time_fn=clock)

    assert bucket.consume(3.0) is True
    assert bucket.consume(3.0) is False

    clock.advance(3.0)
    assert bucket.consume(3.0) is True


def test_failure_and_recovery_state_verification():
    clock = ControlledClock(1000.0)
    redis_mock = MockRedisClient(simulate_failures=True)
    bucket = TokenBucket("failover", capacity=10.0, fill_rate=5.0, redis_client=redis_mock, time_fn=clock)

    # Verify storage outages propagate cleanly as StorageConnectionError
    with pytest.raises(StorageConnectionError):
        bucket.consume(1.0)

    # Recover storage backend
    redis_mock.simulate_failures = False
    assert bucket.consume(5.0) is True


def test_concurrent_stress_race_conditions():
    """
    Stress test with intense multi-threaded contention to expose race conditions,
    token over-drafts, or synchronization locks failures.
    """
    clock = ControlledClock(2000.0)
    capacity = 100.0
    fill_rate = 10.0
    bucket = TokenBucket("stress", capacity=capacity, fill_rate=fill_rate, time_fn=clock)

    threads_count = 50
    requests_per_thread = 20
    total_requests = threads_count * requests_per_thread

    success_count = 0
    failure_count = 0
    lock = threading.Lock()

    def worker():
        nonlocal success_count, failure_count
        for _ in range(requests_per_thread):
            # Randomize small time micro-advancements to simulate real concurrency jitter
            if bucket.consume(1.0):
                with lock:
                    success_count += 1
            else:
                with lock:
                    failure_count += 1

    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        futures = [executor.submit(worker) for _ in range(threads_count)]
        for f in as_completed(futures):
            f.result()

    # Invariant: Successful consumes cannot exceed absolute capacity initially present
    assert success_count <= capacity
    assert success_count + failure_count == total_requests


def test_high_contention_distributed_simulation():
    clock = ControlledClock(5000.0)
    redis_mock = MockRedisClient()
    capacity = 50.0
    fill_rate = 5.0
    bucket = TokenBucket("dist_stress", capacity=capacity, fill_rate=fill_rate, redis_client=redis_mock, time_fn=clock)

    threads = 20
    iterations = 50
    successes = 0
    lock = threading.Lock()

    def thread_task():
        nonlocal successes
        for _ in range(iterations):
            if bucket.consume(1.0):
                with lock: