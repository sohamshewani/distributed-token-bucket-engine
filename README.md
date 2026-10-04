# distributed-token-bucket-engine
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/sohamshewani/distributed-token-bucket-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sohamshewani/distributed-token-bucket-engine/actions) [![Security: Hardened](https://img.shields.io/badge/Security-Hardened-blue.svg)](https://github.com/sohamshewani/distributed-token-bucket-engine) [![Stars](https://img.shields.io/github/stars/sohamshewani/distributed-token-bucket-engine?style=flat)](https://github.com/sohamshewani/distributed-token-bucket-engine)

## 1. Executive Summary & Problem Space

`distributed-token-bucket-engine` is an industrial-grade, ultra-low-latency distributed rate-limiting and token-bucket allocation engine written in Python. In hyperscale distributed environments, traditional centralized rate limiters introduce severe bottlenecks: single points of failure, network round-trip time (RTT) overhead on every critical path request, and lock contention across cluster nodes.

This engine solves these foundational distributed systems bottlenecks by implementing a decentralized, locally-buffered token bucket model. Using local lock-free or fine-grained lock rings (`ring_buffer.py`), deterministic high-resolution monotonic timekeeping (`clock.py`), and asynchronous synchronization drivers (`driver.py`), the engine allows individual worker nodes to make sub-millisecond local allocation decisions while safely synchronizing global consumption limits in the background. It prevents burst anomalies, protects downstream services from cascading failures, and strictly enforces distributed concurrency limits without sacrificing throughput.

---

## 2. Structural ASCII Architecture / Dataflow Diagram

The system is organized into decoupled layers separating timekeeping, state allocation, transport buffering, and execution engine logic:

```
 +-----------------------------------------------------------------------+
 |                     Client / Ingress Request Layer                    |
 +-----------------------------------------------------------------------+
                                     |
                                     v
 +-----------------------------------------------------------------------+
 |                        Engine Core (engine.py)                        |
 |   +------------------------+      +-------------------------------+   |
 |   | Policy Evaluation      |----->| Allocator Strategy (allocator)|   |
 |   +------------------------+      +-------------------------------+   |
 +-----------------------------------------------------------------------+
            |                                         |
            v                                         v
 +-----------------------+                 +-----------------------------+
 |  Monotonic Time (clock)|                 | Ring Buffer (ring_buffer.py)|
 |  - Drift Compensation |                 | - Circular Write/Read Ptrs  |
 |  - High-Res Resolution|                 | - Lock-Free Batch Queuing   |
 +-----------------------+                 +-----------------------------+
                                                      |
                                                      v
                                           +-----------------------------+
                                           | Transport Driver (driver.py)|
                                           | - Network / IPC Sync        |
                                           | - Batch State Reconcile     |
                                           +-----------------------------+
```

---

## 3. Concurrency Model & Memory Invariants

### Thread Safety & Lock Contention Prevention
The engine uses thread-safe primitives designed to eliminate global interpreter lock (GIL) bottlenecks where possible and isolate critical sections:
- **Lock-Free Ring Buffers (`ring_buffer.py`)**: Implements bounded circular queues with atomic head/tail pointer increments, ensuring multi-producer, single-consumer (MPSC) or single-producer, multi-consumer (SPMC) safety without blocking the data plane during high concurrency.
- **Granular Allocators (`allocator.py`)**: Employs fine-grained thread locks per bucket namespace rather than global locks, isolating contention domains across distinct rate-limited routes or tenants.

### Memory Ownership & Garbage Reduction
- **Pre-Allocated Memory Structures**: Buffers and types (`types.py`) reuse internal memory blocks to avoid runtime allocations and mitigate garbage collection (GC) pause spikes in latency-sensitive paths.
- **Deterministic Cleanup**: Timestamps and state records are safely mutated in place using strict ownership transfers across thread boundaries via the transport driver.

---

## 4. Performance Characteristics & Algorithmic Complexity Table

| Operation | Path | Time Complexity | Space Complexity | P99 Latency Estimate |
| :--- | :--- | :--- | :--- | :--- |
| **Token Acquisition** | Fast Path (Local Buffer) | $O(1)$ | $O(1)$ | $< 50\mu s$ |
| **Ring Push / Pop** | Buffer Queue | $O(1)$ | $O(N)$ (Bounded) | $< 10\mu s$ |
| **Clock Resolution** | Timekeeper | $O(1)$ | $O(1)$ | $< 1\mu s$ |
| **Global State Sync** | Driver Batch Merge | $O(K)$ ($K$ = batch size) | $O(K)$ | $< 2ms$ |

---

## 5. Security Architecture & Threat Model

- **Input Boundary Sanitization**: All incoming rate parameters, token weights, and client identifiers undergo strict schema validation via `types.py` to prevent integer overflows, negative token injections, and malformed allocations.
- **Memory Corruption Protections**: Bounds checking is rigorously enforced across all circular ring buffer index calculations (`ring_buffer.py`) to prevent buffer overreads/overwrites.
- **Denial of Service (DoS) Mitigation**: The engine protects its own control plane by bounding internal synchronization queues, dropping or shedding load gracefully when downstream drivers experience network degradation, thereby preventing memory exhaustion attacks.

---

## 6. Build, Verification & Benchmarking Recipes

Execute the following terminal commands to set up the environment, run the invariant verification suite, and execute performance benchmarks.

### Prerequisites & Installation
```bash
# Clone the repository
git clone https://github.com/sohamshewani/distributed-token-bucket-engine.git
cd distributed-token-bucket-engine

# Create a virtual environment and install dependencies via pyproject.toml
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,benchmarks]
```

### Run Verification & Invariant Tests
```bash
# Run the complete test suite focusing on concurrency invariants
pytest tests/test_invariants.py -v --tb=short
```

### Execute Performance Benchmarks
```bash
# Run micro-benchmarks against the engine core
python benchmarks/bench_engine.py