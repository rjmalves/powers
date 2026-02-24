---
status: approved
review_priority: 4-low
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.1 (Memory Budget Overview)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.2 (Memory Layout Strategy)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.3 (NUMA-Aware Allocation)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.4 (Memory Pool for Temporary Allocations)"
  - "DATA_MODEL_SPECIFICATION.md §6.7 (NUMA-Aware Memory Management)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from architecture §24 and data model §6.7 with ferroMPI migration"
  - date: 2026-02-23
    description: "P4 review rewrite. Fixed review_priority (2-high → 4-low). Stripped 6 Rust code blocks (~350 lines: MemoryManager/SolverWorkspace, numa allocators, first-touch pattern, NumaPartitionedScenarios, SharedWindow usage, MemoryPool/PooledBuffer). Replaced fabricated memory budget with derivable numbers from approved specs (solver-workspaces.md §1.2, cut-management-impl.md §4.2, scenario-generation.md §2.3). Removed wrong concurrency primitives (Arc/RwLock/ThreadLocal) — OpenMP shared data model is the approved pattern. Removed NUMA-partitioned scenarios (contradicts SharedWindow approach in shared-memory-aggregation.md §1). Removed memory pool (speculative — hot-path allocation avoidance covered by solver-workspaces.md §1). Removed duplicated SharedWindow code (references shared-memory-aggregation.md instead). Reorganized around data ownership model, derivable budget, NUMA principles, and growth analysis. Cross-references expanded from 4 to 12."
---

# Memory Architecture

## Purpose

This spec defines the memory architecture for POWE.RS: data ownership categories, per-rank memory budget derived from approved specifications, NUMA-aware allocation principles, and memory growth analysis across SDDP iterations. This spec aggregates memory-relevant requirements from [Solver Workspaces](../03-architecture/solver-workspaces.md), [Shared Memory Aggregation](./shared-memory-aggregation.md), and [Communication Patterns](./communication-patterns.md) into a unified memory perspective.

## 1. Data Ownership Model

### 1.1 Ownership Categories

All runtime data falls into one of four ownership categories, each with distinct allocation and access patterns:

| Category                 | Owner                           | Access During Training                           | Allocation Strategy                                    | Examples                                                                 |
| ------------------------ | ------------------------------- | ------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------ |
| **Shared read-only**     | Node (via SharedWindow) or rank | Read by all threads, never modified after init   | SharedWindow leader allocation or per-rank replication | Opening tree, input case data, PAR parameters, Cholesky factors          |
| **Thread-local mutable** | OpenMP thread                   | Exclusive read/write by owning thread            | First-touch on owning thread's NUMA node               | Solver workspace, solution buffers, basis cache, cut accumulation buffer |
| **Rank-local growing**   | MPI rank                        | Read by all threads, written at stage boundaries | Pre-allocated with growth capacity                     | Cut pool (grows each iteration)                                          |
| **Temporary**            | OpenMP thread                   | Allocated/freed within a single solve            | Pre-allocated in workspace, reused                     | RHS patch buffer, scratch arrays                                         |

### 1.2 Concurrency Model

POWE.RS uses OpenMP for intra-rank threading (see [Hybrid Parallelism §5](./hybrid-parallelism.md)). The data sharing model follows OpenMP semantics:

| OpenMP Clause                    | Data Category        | Implication                                                                            |
| -------------------------------- | -------------------- | -------------------------------------------------------------------------------------- |
| `shared`                         | Shared read-only     | All threads see the same pointer; no synchronization needed for reads                  |
| `private` / indexed by thread ID | Thread-local mutable | Each thread accesses its own workspace by `omp_get_thread_num()`                       |
| `shared` with barrier            | Rank-local growing   | Written between parallel regions (single-threaded merge), read during parallel regions |

No `Arc`, `RwLock`, or `Mutex` is needed — OpenMP's shared data model, implicit barriers, and thread-ID-indexed arrays provide the necessary semantics without Rust synchronization primitives.

## 2. Per-Rank Memory Budget

### 2.1 Derivable Components

The following memory estimates are derived from production-scale dimensions in approved specs. Reference configuration: 160 hydros, 130 thermals, 120 stages, 200 forward passes, 200 openings, 15K cut capacity, 16 threads per rank.

| Component                 |        Size | Derivation                                                                                     | Source                                                                |
| ------------------------- | ----------: | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Opening tree              |      ~30 MB | 200 openings × 120 stages × 160 entities × 8 bytes                                             | [Scenario Generation §2.3](../03-architecture/scenario-generation.md) |
| Solver workspaces (×16)   |     ~912 MB | 16 threads × ~57 MB per workspace (HiGHS: solver instance + buffers + per-stage basis cache)   | [Solver Workspaces §1.2](../03-architecture/solver-workspaces.md)     |
| Cut pool (at capacity)    |     ~250 MB | 15,000 cuts × 120 stages × ~16,660 bytes per cut ÷ 120 = ~250 MB active in memory at any stage | [Cut Management Impl §4.2](../03-architecture/cut-management-impl.md) |
| Input case data           |      ~20 MB | System entities, PAR parameters, correlation factors, block/exchange factors                   | [Internal Structures](../02-data-model/internal-structures.md)        |
| Forward pass state        |      ~20 MB | 200 trajectories × ~9 KB state vector (1,120 doubles + metadata) × ~1 stage buffered           | [Training Loop §5.1](../03-architecture/training-loop.md)             |
| MPI communication buffers |      ~10 MB | Send/receive buffers for `MPI_Allgatherv` (cuts: ~3.3 MB, states: ~1.75 MB per stage)          | [Communication Patterns §2](./communication-patterns.md)              |
| **Total per rank**        | **~1.2 GB** |                                                                                                |                                                                       |

### 2.2 SharedWindow Savings

When ranks on the same node share read-only data via `SharedWindow<T>` (see [Shared Memory Aggregation §1](./shared-memory-aggregation.md)), the per-node memory footprint is reduced:

| Data                | Per-Rank (replicated) | SharedWindow (1 copy/node) | Savings (4 ranks/node) |
| ------------------- | --------------------: | -------------------------: | ---------------------: |
| Opening tree        |                ~30 MB |                     ~30 MB |                 ~90 MB |
| Input case data     |                ~20 MB |                     ~20 MB |                 ~60 MB |
| **Total shareable** |                ~50 MB |                     ~50 MB |                ~150 MB |

The savings are modest at production scale because the dominant memory consumer is the solver workspaces (~912 MB), which are inherently thread-local and cannot be shared. The cut pool (§2.3) is a larger sharing candidate if memory becomes constrained.

### 2.3 Memory Growth

The cut pool is the only component that grows during training. All other allocations are fixed at initialization.

| Iteration | Active Cuts (approx) | Cut Pool Memory | Notes                                   |
| --------- | -------------------: | --------------: | --------------------------------------- |
| 1         |                  200 |           ~3 MB | $M$ cuts (one per forward pass)         |
| 50        |               ~5,000 |          ~80 MB | Before cut selection starts pruning     |
| 100       |              ~10,000 |         ~160 MB | Cut selection bounds active count       |
| 200       |              ~15,000 |         ~250 MB | At capacity; dominated cuts deactivated |

The cut pool pre-allocates slots at initialization (per [Solver Abstraction §5](../03-architecture/solver-abstraction.md)), so memory is allocated up front, not dynamically during training. The growth shown above is logical (slots being populated), not physical (heap allocations).

### 2.4 Scaling with Problem Size

Memory scales primarily with the number of hydro plants (state dimension) and the number of stages:

| Dimension           | Effect on Memory                                                                       |
| ------------------- | -------------------------------------------------------------------------------------- |
| Hydro count (×2)    | Cut coefficients double → cut pool doubles. Solver workspace grows proportionally.     |
| Stage count (×2)    | Per-stage basis cache doubles. Cut pool doubles (separate pool per stage).             |
| Forward passes (×2) | Forward state buffers double. Cut pool growth rate doubles (more cuts per iteration).  |
| Threads (×2)        | Solver workspaces double (linear scaling). Dominant memory cost at high thread counts. |
| Openings (×2)       | Opening tree doubles. Modest impact (~60 MB → ~120 MB at production scale).            |

## 3. NUMA-Aware Allocation

### 3.1 Principles

Modern HPC nodes have multiple NUMA domains. Memory access latency varies significantly between local and remote NUMA domains (typical: 1.5-3× slower for remote access). POWE.RS follows three NUMA principles:

**Principle 1 — Thread-owns-workspace**: Each solver workspace is allocated by the thread that will use it, ensuring first-touch allocation on the thread's local NUMA node. The workspace is never accessed by other threads.

**Principle 2 — One rank per NUMA domain**: The recommended deployment is one MPI rank per NUMA domain (see [Hybrid Parallelism §4.4](./hybrid-parallelism.md) and [SLURM Deployment](./slurm-deployment.md)). This ensures that all threads within a rank share the same NUMA domain, making shared read-only data (case data, opening tree) local to all threads.

**Principle 3 — First-touch initialization**: Large arrays (solution buffers, basis cache) are initialized by the owning thread within an OpenMP parallel region, not by the main thread. This ensures the OS places memory pages on the NUMA node where they will be accessed.

### 3.2 NUMA Initialization Sequence

The initialization of per-thread resources follows the sequence documented in [Solver Workspaces §1.3](../03-architecture/solver-workspaces.md):

1. Main thread determines NUMA topology via `ferrompi::slurm` helpers (see [Hybrid Parallelism §1.2](./hybrid-parallelism.md))
2. OpenMP parallel region is entered
3. Each thread creates its solver instance (first-touch allocates solver internals on local NUMA)
4. Each thread initializes its per-stage basis cache, solution buffers, and scratch arrays
5. Parallel region ends (implicit barrier)
6. Main thread verifies all workspaces are initialized

### 3.3 Cache Line Alignment

All per-thread data structures must be padded to cache line boundaries (64 bytes) to prevent false sharing between adjacent threads' workspaces. This applies to:

- Solver workspace array entries (indexed by thread ID)
- Cut accumulation buffers (see [Synchronization §3.1](./synchronization.md))
- Per-thread solve statistics counters

### 3.4 NUMA Latency Reference

Representative NUMA latency characteristics for multi-socket server nodes:

| Access Pattern             | Typical Latency | Impact on LP Solve   |
| -------------------------- | --------------- | -------------------- |
| Local NUMA                 | ~80 ns          | Baseline             |
| Adjacent NUMA              | ~120 ns (1.5×)  | Moderate slowdown    |
| Remote NUMA (cross-socket) | ~200 ns (2.5×)  | Significant slowdown |

These values are representative of AMD EPYC and Intel Xeon platforms. Actual latencies vary by hardware. The key insight is that remote NUMA access can slow LP solves by 2-3× when solver working data (LU factorization, pricing vectors) is allocated on the wrong NUMA node — motivating Principle 1 (thread-owns-workspace).

## 4. Hot-Path Allocation Avoidance

### 4.1 Requirement

No heap allocation (`malloc`/`new`/`Vec::push` beyond capacity) is permitted during the SDDP hot path — the forward pass LP solves, backward pass LP solves, and cut accumulation. All buffers are pre-allocated at initialization and reused.

### 4.2 Pre-Allocated Components

| Component                | Pre-allocated At | Reused During                         | Reference                                                         |
| ------------------------ | ---------------- | ------------------------------------- | ----------------------------------------------------------------- |
| Solver instance          | Initialization   | Every LP solve (all iterations)       | [Solver Workspaces §1.2](../03-architecture/solver-workspaces.md) |
| Primal/dual buffers      | Initialization   | Solution extraction after each solve  | [Solver Workspaces §1.2](../03-architecture/solver-workspaces.md) |
| RHS patch buffer         | Initialization   | Scenario patching before each solve   | [Solver Workspaces §1.2](../03-architecture/solver-workspaces.md) |
| Cut pool slots           | Initialization   | Cut insertion (slot reuse via bitmap) | [Solver Abstraction §5](../03-architecture/solver-abstraction.md) |
| Cut accumulation buffers | Initialization   | Per-thread cut collection each stage  | [Synchronization §3.1](./synchronization.md)                      |
| MPI send/recv buffers    | Initialization   | `MPI_Allgatherv` each stage           | [Communication Patterns §2](./communication-patterns.md)          |

### 4.3 Allocation Monitoring

In debug/test builds, an allocation tracker can detect unexpected hot-path allocations by hooking the global allocator and flagging any allocation between the start and end of an LP solve. This is a development tool, not a production feature.

## Cross-References

- [Solver Workspaces §1](../03-architecture/solver-workspaces.md) — Detailed workspace contents, sizing, NUMA-aware initialization, basis cache
- [Solver Abstraction §5](../03-architecture/solver-abstraction.md) — Cut pool preallocation, slot assignment, capacity management
- [Shared Memory Aggregation §1](./shared-memory-aggregation.md) — SharedWindow leader allocation, opening tree sharing, cut pool sharing trade-offs
- [Communication Patterns §2](./communication-patterns.md) — Data payload sizes for MPI buffers
- [Communication Patterns §5](./communication-patterns.md) — SharedWindow capabilities and shared data candidates
- [Synchronization §3](./synchronization.md) — Cut accumulation buffers, cache line alignment
- [Hybrid Parallelism §1.2](./hybrid-parallelism.md) — ferrompi capabilities, SLURM/NUMA detection
- [Hybrid Parallelism §4.4](./hybrid-parallelism.md) — NUMA binding policy
- [Scenario Generation §2.3](../03-architecture/scenario-generation.md) — Opening tree size and memory layout
- [Cut Management Impl §4.2](../03-architecture/cut-management-impl.md) — Per-cut wire size and cut pool sizing
- [Training Loop §5.1](../03-architecture/training-loop.md) — State vector dimensions
- [Internal Structures](../02-data-model/internal-structures.md) — In-memory data model for system entities
