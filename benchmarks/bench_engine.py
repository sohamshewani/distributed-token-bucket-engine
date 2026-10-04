import asyncio
import statistics
import sys
import time

try:
    import redis.asyncio as redis
except ImportError:
    import aioredis as redis

# Fallback in-memory Token Bucket Engine if distributed-token-bucket-engine is not installed
try:
    from token_bucket import TokenBucketEngine
except ImportError:

    class MockDistributedTokenBucketEngine:

        def __init__(self, capacity: int, refill_rate: float):
            self.capacity = capacity
            self.tokens = float(capacity)
            self.refill_rate = refill_rate
            self.last_refill = time.monotonic()
            self._lock = asyncio.Lock()

        async def consume(self, tokens: int = 1) -> bool:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(
                    float(self.capacity), self.tokens + elapsed * self.refill_rate
                )
                self.last_refill = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                return False

    TokenBucketEngine = MockDistributedTokenBucketEngine


async def worker(
    engine: TokenBucketEngine, iterations: int, latencies: list
) -> None:
    for _ in range(iterations):
        start = time.perf_counter_ns()
        await engine.consume(1)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000.0)  # Convert to ms


async def main() -> None:
    total_iterations = 10000
    concurrency = 50
    iters_per_worker = total_iterations // concurrency

    engine = TokenBucketEngine(capacity=100000, refill_rate=100000.0)

    # Warmup
    await engine.consume(1)

    latencies = []
    start_time = time.perf_counter()

    tasks = [
        worker(engine, iters_per_worker, latencies) for _ in range(concurrency)
    ]
    await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    duration = end_time - start_time

    actual_iterations = len(latencies)
    ops_sec = actual_iterations / duration if duration > 0 else 0

    latencies.sort()
    p50 = (
        statistics.median(latencies)
        if latencies
        else 0.0
    )
    p99 = (
        latencies[int(len(latencies) * 0.99)]
        if latencies
        else 0.0
    )
    p999 = (
        latencies[int(len(latencies) * 0.999)]
        if latencies
        else 0.0
    )

    # Print Tabular Metrics
    print("\n" + "=" * 60)
    print(f"{'DISTRIBUTED TOKEN BUCKET ENGINE - MICROBENCHMARK':^60}")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value':<25}")
    print("-" * 60)
    print(f"{'Concurrency':<30} | {concurrency:<25}")
    print(f"{'Total Operations':<30} | {actual_iterations:<25}")
    print(f"{'Duration (s)':<30} | {duration:<25.4f}")
    print(f"{'Throughput (ops/sec)':<30} | {ops_sec:<25.2f}")
    print(f"{'Latency p50 (ms)':<30} | {p50:<25.4f}")
    print(f"{'Latency p99 (ms)':<30} | {p99:<25.4f}")
    print(f"{'Latency p99.9 (ms)':<30} | {p999:<25.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())