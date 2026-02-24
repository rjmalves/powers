---
status: approved
review_priority: 4-low
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.1 (Synchronization Points)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.2 (Synchronization Summary)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.3 (Thread Synchronization Within Rank)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.4 (Lock-Free Cut Aggregation)"
  - "DATA_MODEL_SPECIFICATION.md §6.3 (Synchronization Points)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from monolith docs (T-022)"
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): The backward pass has a hard synchronization barrier at each stage boundary — all threads must complete cut construction at stage t before proceeding to stage t-1. The forward pass has no such per-stage barrier (fully parallel trajectories). Validate that existing sync point design accounts for this asymmetry during P4 review. See sddp-algorithm.md §3.4."
  - date: 2026-02-23
    description: "P4 review rewrite. Fixed review_priority (2-high → 4-low). Stripped 4 Rust code blocks (~140 lines: OutcomeSync, SpinBarrier, CutAccumulator, ThreadLocal<T>). Fixed forward→backward transition from 'no synchronization' to MPI_Allgatherv trial point collection (per training-loop.md §5.2). Fixed backward pass MPI from comm.gather() + comm.broadcast() to MPI_Allgatherv (per training-loop.md §6.3). Removed fabricated '800 MB state-gathering' claim. Removed custom spin barrier — thread coordination uses OpenMP implicit barriers (per hybrid-parallelism.md §5). Replaced OutcomeSync model with behavioral cut accumulation pattern. Updated post-forward aggregation to 4 quantities (per work-distribution.md §1.4). Removed Init→Forward barrier (no evidence in approved specs). Cross-references expanded from 4 to 10."
---

# Synchronization

## Purpose

This spec defines the synchronization architecture for POWE.RS: the complete set of MPI synchronization points during SDDP iterations, the forward-to-backward transition, per-stage barrier semantics in the backward pass, and thread coordination within a rank. This spec details the synchronization mechanics that [Training Loop](../03-architecture/training-loop.md) and [Work Distribution](./work-distribution.md) describe at the algorithmic and distribution levels.

## 1. MPI Synchronization Points

### 1.1 Synchronization Summary

The following table lists all MPI synchronization points in a single SDDP iteration. There are exactly three collective calls per iteration (one post-forward, one per backward stage, one for convergence).

| Phase              | MPI Operation    | Data Exchanged                                              | Direction       |
| ------------------ | ---------------- | ----------------------------------------------------------- | --------------- |
| Forward → Backward | `MPI_Allgatherv` | Visited states (trial points) from all forward trajectories | All ranks ↔ all |
| Backward stage $t$ | `MPI_Allgatherv` | New cuts generated at stage $t$ by each rank                | All ranks ↔ all |
| Post-backward      | `MPI_Allreduce`  | Convergence statistics (4 scalars — see §1.3)               | All ranks ↔ all |

### 1.2 Forward Pass: No Per-Stage Synchronization

The forward pass has **no per-stage synchronization barrier**. Each thread solves its assigned trajectories independently from stage $t = 1$ to $T$. No MPI communication occurs during the forward pass itself — each rank processes its contiguous block of scenarios without coordinating with other ranks.

### 1.3 Forward-to-Backward Transition

After all forward trajectories complete within a rank, the rank contributes its visited states to an `MPI_Allgatherv` call. After this call, every rank has the complete set of trial points from all forward trajectories across all ranks. This is the **only** synchronization point between the forward and backward passes.

The post-forward `MPI_Allreduce` aggregates convergence statistics:

| Quantity                            | Reduction | Purpose                                |
| ----------------------------------- | --------- | -------------------------------------- |
| First-stage LP objective            | `MPI_MIN` | Lower bound (monotonically increasing) |
| Total forward cost (sum)            | `MPI_SUM` | Upper bound mean computation           |
| Total forward cost (sum of squares) | `MPI_SUM` | Upper bound variance computation       |
| Trajectory count                    | `MPI_SUM` | Denominator for mean/variance          |

See [Work Distribution §1.4](./work-distribution.md) and [Convergence Monitoring §3](../03-architecture/convergence-monitoring.md).

### 1.4 Backward Pass: Per-Stage Barrier

The backward pass has a **hard synchronization barrier at each stage boundary**. At each stage $t$ (walking from $T$ down to 2):

1. All ranks evaluate their assigned trial points and generate cuts (parallel across ranks and threads)
2. `MPI_Allgatherv` collects all new cuts from all ranks — after this call, every rank has the complete set of new cuts for stage $t$
3. All ranks add the new cuts to stage $t-1$'s cut pool
4. Only then do ranks proceed to stage $t-1$

The `MPI_Allgatherv` acts as an implicit barrier — no rank can proceed to stage $t-1$ until all ranks have contributed their cuts for stage $t$. This barrier is mandatory because the cuts generated at stage $t$ must be available when solving backward LPs at stage $t-1$ (the backward pass uses $V_{t+1}^k$, the current iteration's approximation). See [Training Loop §6.3](../03-architecture/training-loop.md) and [Work Distribution §2.2](./work-distribution.md).

### 1.5 Iteration Boundary

No explicit MPI synchronization is required between iterations. The convergence monitor evaluates stopping rules locally on each rank using the aggregated statistics from §1.3. All ranks reach the same termination decision deterministically (same data, same rules).

An optional checkpoint barrier (`MPI_Barrier`) may occur every $N$ iterations if checkpointing is enabled — see [Checkpointing](./checkpointing.md).

## 2. Thread Coordination Within Rank

Thread coordination within a rank uses OpenMP synchronization primitives, accessed via the C FFI wrapper described in [Hybrid Parallelism §5](./hybrid-parallelism.md).

### 2.1 Forward Pass Thread Coordination

No explicit thread synchronization during the forward pass. Each thread owns complete trajectories (thread-trajectory affinity) and solves them independently. The OpenMP parallel region's implicit barrier at the end ensures all threads have completed before the rank proceeds to the `MPI_Allgatherv` for trial point collection.

### 2.2 Backward Pass Thread Coordination

At each backward stage $t$, threads within a rank coordinate in two phases:

**Phase 1 — Parallel evaluation**: Each thread evaluates its assigned trial points, solving all openings sequentially per trial point. Each thread accumulates its generated cuts in a thread-local buffer (no shared writes, no contention). See §3.

**Phase 2 — Collection and MPI**: The OpenMP parallel region ends (implicit barrier ensures all threads have finished). The rank's main thread collects cuts from all thread-local buffers and participates in the inter-rank `MPI_Allgatherv`.

This two-phase pattern repeats for each stage from $T$ down to 2.

### 2.3 Synchronization Primitives

| Primitive               | Usage                                                     | Source                                             |
| ----------------------- | --------------------------------------------------------- | -------------------------------------------------- |
| OpenMP implicit barrier | End of parallel region — ensures all threads complete     | Automatic at `}` of `#pragma omp parallel`         |
| OpenMP explicit barrier | If needed within a parallel region (not expected in v1.0) | `#pragma omp barrier` via C FFI                    |
| Thread-local storage    | Per-thread cut buffers, solver workspaces                 | OpenMP thread ID indexing into pre-allocated array |

The approved architecture does not require custom spin barriers or lock-free data structures for thread synchronization. OpenMP's implicit barrier at the end of parallel regions provides the necessary synchronization point between the parallel evaluation phase and the single-threaded MPI collection phase.

## 3. Cut Accumulation Pattern

### 3.1 Thread-Local Accumulation

During the backward pass parallel evaluation phase, each thread accumulates cuts in its own buffer. The buffers are indexed by OpenMP thread ID and pre-allocated at initialization. This design has zero contention during the hot loop — each thread writes exclusively to its own buffer.

| Property          | Value                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------- |
| Buffer allocation | Pre-allocated per thread at initialization                                            |
| Buffer indexing   | OpenMP thread ID (`omp_get_thread_num()`)                                             |
| Contention        | None — pure thread-local writes                                                       |
| Cache alignment   | Each buffer starts on a cache line boundary (64 bytes) to prevent false sharing       |
| Capacity          | Sized for expected cuts per thread per stage ($\lceil M / N_{\text{threads}} \rceil$) |

### 3.2 Collection and Merge

After the OpenMP parallel region ends (implicit barrier), the rank's main thread collects all cuts from the per-thread buffers into a single contiguous array. This array is then used as the send buffer for `MPI_Allgatherv`.

### 3.3 False Sharing Prevention

Adjacent thread buffers must not share cache lines. If two threads' buffers share a cache line, a write by one thread invalidates the cache line for the other, causing expensive cache coherency traffic. Pre-allocating each buffer at a cache-line-aligned address (64-byte boundary) eliminates this. See [Memory Architecture](./memory-architecture.md) for cache-line alignment conventions.

## 4. Asymmetry Summary

| Property                      | Forward Pass                                         | Backward Pass                                                 |
| ----------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| Per-stage MPI synchronization | None                                                 | `MPI_Allgatherv` at every stage                               |
| Thread coordination           | None (independent trajectories)                      | Implicit OpenMP barrier between evaluation and collection     |
| Data flow direction           | Each thread accumulates independently                | Thread-local → rank-local merge → inter-rank `MPI_Allgatherv` |
| Parallelism grain             | Trajectory (coarse)                                  | Trial point (coarse), openings sequential within trial point  |
| Post-phase aggregation        | `MPI_Allgatherv` (states) + `MPI_Allreduce` (bounds) | Convergence check after all stages complete                   |

## Cross-References

- [Training Loop §4.3](../03-architecture/training-loop.md) — Forward pass parallel distribution, post-forward `MPI_Allreduce`
- [Training Loop §5.2](../03-architecture/training-loop.md) — State extraction and `MPI_Allgatherv` for trial point collection
- [Training Loop §6.3](../03-architecture/training-loop.md) — Backward pass stage synchronization barrier, `MPI_Allgatherv` for cuts
- [SDDP Algorithm §3.4](../01-math/sddp-algorithm.md) — Backward pass hard barrier, forward pass no barrier, thread-trajectory affinity
- [Work Distribution §1.4](./work-distribution.md) — Post-forward `MPI_Allreduce` with 4 convergence quantities
- [Work Distribution §2.2](./work-distribution.md) — Per-stage backward pass execution (6 steps including barrier)
- [Hybrid Parallelism §5](./hybrid-parallelism.md) — OpenMP C FFI wrapper primitives, implicit barriers
- [Convergence Monitoring](../03-architecture/convergence-monitoring.md) — Stopping rule evaluation using aggregated statistics
- [Communication Patterns](./communication-patterns.md) — ferrompi collectives: `MPI_Allgatherv`, `MPI_Allreduce`
- [Memory Architecture](./memory-architecture.md) — Cache-line alignment conventions, NUMA-local allocation
