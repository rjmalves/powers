---
status: approved
review_priority: 4-low
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §21.1 (Forward Pass Distribution)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §21.2 (Backward Pass Distribution)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §21.3 (Work Distribution Implementation)"
  - "DATA_MODEL_SPECIFICATION.md §6.5 (Backward Pass Computation Modes)"
  - "DATA_MODEL_SPECIFICATION.md §6.10 (Dynamic Work Distribution)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): The same thread that runs forward pass K must also run backward pass K (thread-trajectory affinity). This preserves cache locality for solver basis and scenario data, and eliminates cross-thread data handoff. Within the backward pass, each thread solves its branching scenarios sequentially to keep the solver state hot. Validate this pattern against the existing dynamic dispatch design during P4 review. See sddp-algorithm.md §3.4."
  - date: 2026-02-23
    description: "P4 review rewrite. Fixed review_priority (2-high → 4-low). Stripped all Rust code (5 blocks: backward_pass_sequential, WorkDistributor, execute_backward_pass, Rank0Executor, WorkerExecutor — ~235 lines). Replaced forward pass dynamic dispatch architecture with static contiguous block distribution (per approved training-loop.md §4.3 and hybrid-parallelism.md §3). Replaced backward pass parallel-over-openings with sequential opening evaluation per thread (per approved training-loop.md §6.2-§6.3). Removed rank 0 dispatcher/worker architecture, MPI message protocol, and dedicated dispatcher thread (none of this exists in approved architecture). Moved pipelined backward pass to deferred-features.md C.18. Removed training.backward_pass.mode config (not in approved configuration-reference.md). Condensed scenario-based rationale. Cross-references expanded from 5 to 12."
---

# Work Distribution

## Purpose

This spec defines how POWE.RS distributes computational work across MPI ranks and OpenMP threads during the SDDP training loop: forward pass scenario distribution with thread-trajectory affinity, backward pass trial point distribution with per-stage synchronization, and the load balancing strategy. This spec details the distribution mechanics that [Training Loop](../03-architecture/training-loop.md) describes at the algorithmic level.

## 1. Forward Pass Distribution

### 1.1 Static Contiguous Block Assignment

Forward pass scenarios are distributed across MPI ranks using **static contiguous block assignment**. Given $M$ total forward trajectories and $R$ MPI ranks, each rank receives a contiguous block of $\lfloor M/R \rfloor$ or $\lceil M/R \rceil$ scenario indices:

| Parameter                                    | Value                                                                                           |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Distribution method                          | Static contiguous blocks                                                                        |
| Scenarios per rank                           | $\lfloor M/R \rfloor$ or $\lceil M/R \rceil$ (remainder distributed to first $M \bmod R$ ranks) |
| Inter-rank communication during forward pass | None — each rank processes independently                                                        |
| Post-forward aggregation                     | `MPI_Allreduce` for lower bound and upper bound statistics                                      |

**Why static, not dynamic dispatch**: Forward pass scenarios have nearly identical per-stage LP solve cost (same LP structure, same cut count, same constraint dimensions). The variance in solve time comes from simplex iterations, which is small relative to the total. Static distribution avoids the complexity and latency of a dispatcher/worker protocol while achieving near-perfect load balance.

### 1.2 Thread-Trajectory Affinity Within Rank

Within each rank, the rank's assigned scenarios are parallelized across OpenMP threads using **thread-trajectory affinity**: each thread owns one or more complete forward trajectories and solves all stages sequentially ($t = 1$ to $T$) for each assigned trajectory. This preserves:

- **Cache locality** — solver basis, scenario data, and LP coefficients remain warm in the thread's cache lines across stages
- **Warm-start continuity** — the solver basis from stage $t$ warm-starts stage $t+1$ within the same trajectory
- **Implementation simplicity** — no cross-thread data handoff during the forward pass

OpenMP distributes trajectories to threads using `schedule(dynamic,1)`. Dynamic scheduling within a rank absorbs the small per-trajectory solve time variance while maintaining thread-trajectory affinity (each trajectory is fully executed by the thread that picks it up).

### 1.3 Batch Processing

When a rank's assigned scenario count exceeds its thread count ($M_r > N_{\text{threads}}$), threads process trajectories in batches. Between batches, the thread saves and restores forward pass state (solver basis, visited states, scenario realization) at stage boundaries. See [Training Loop §4.3](../03-architecture/training-loop.md) and [SDDP Algorithm §3.4](../01-math/sddp-algorithm.md).

### 1.4 Post-Forward Aggregation

After all ranks complete their trajectories, a single `MPI_Allreduce` aggregates:

| Quantity                            | Reduction                                    | Purpose                                |
| ----------------------------------- | -------------------------------------------- | -------------------------------------- |
| First-stage LP objective            | `MPI_MIN` or first-stage deterministic value | Lower bound (monotonically increasing) |
| Total forward cost (sum)            | `MPI_SUM`                                    | Upper bound mean computation           |
| Total forward cost (sum of squares) | `MPI_SUM`                                    | Upper bound variance computation       |
| Trajectory count                    | `MPI_SUM`                                    | Denominator for mean/variance          |

This is a single collective call — no point-to-point messaging, no dispatcher coordination.

## 2. Backward Pass Distribution

### 2.1 Trial Point Collection

Before the backward pass begins, the visited states from all forward trajectories must be available to all ranks. This is accomplished via `MPI_Allgatherv`: each rank contributes its forward pass visited states, and all ranks receive the complete set.

The trial points are then distributed across ranks for the backward pass using the same static contiguous block assignment as the forward pass (§3).

### 2.2 Per-Stage Execution

The backward pass walks stages in reverse order ($t = T$ down to 2). At each stage $t$:

**Step 1 — Distribute trial points**: The trial points for stage $t$ (visited states from all forward trajectories) are divided across ranks using static contiguous blocks.

**Step 2 — Evaluate openings**: Each rank processes its assigned trial points. For each trial point, the rank evaluates **all** $N_{\text{openings}}$ noise vectors from the fixed opening tree. Within a rank, trial points are distributed across OpenMP threads, and each thread evaluates its assigned trial points' openings **sequentially** — not in parallel. This sequential evaluation preserves solver warm-start across openings (the LP structure is identical, only the RHS changes between openings).

**Step 3 — Aggregate into cuts**: Each thread aggregates its per-opening results into cuts using the configured risk measure (expectation or CVaR).

**Step 4 — Synchronize cuts**: `MPI_Allgatherv` collects all new cuts from all ranks. After this call, every rank has the complete set of new cuts for stage $t$.

**Step 5 — Update FCF**: All ranks add the new cuts to stage $t-1$'s cut pool. This ensures that when stage $t-1$ is processed in the next loop iteration, the freshly computed cuts from stage $t$ are available (sequential backward pass using $V_{t+1}^k$).

**Step 6 — Barrier**: The `MPI_Allgatherv` in step 4 acts as an implicit barrier — no rank can proceed to stage $t-1$ until all ranks have contributed their cuts for stage $t$.

### 2.3 Why Sequential Opening Evaluation

The openings for a given trial point are evaluated sequentially by the owning thread (not parallelized across threads). This is a deliberate design choice:

| Concern                | Sequential openings                      | Parallel openings                                     |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------- |
| **Solver warm-start**  | Preserved — same LP, only RHS changes    | Lost — would require separate solver per opening      |
| **Memory**             | 1 solver per thread                      | $N_{\text{openings}}$ solvers per thread (infeasible) |
| **Cache locality**     | Hot — LP data stays in L1/L2             | Cold — thrashing across $N_{\text{openings}}$ LPs     |
| **Parallelism source** | Trial points across threads (sufficient) | Openings within a trial point                         |

With production-scale parameters (e.g., $M = 200$ forward passes, $R = 8$ ranks, $N_{\text{threads}} = 16$ per rank), there are 200 trial points per stage distributed across 128 total threads — approximately 1.5 trial points per thread per stage, which is sufficient parallelism.

## 3. Distribution Arithmetic

### 3.1 Contiguous Block Assignment

Given $N$ items to distribute across $R$ ranks:

| Rank $r$           | Start index                                                                         | Count                 |
| ------------------ | ----------------------------------------------------------------------------------- | --------------------- |
| $r < N \bmod R$    | $r \times \lceil N/R \rceil$                                                        | $\lceil N/R \rceil$   |
| $r \geq N \bmod R$ | $(N \bmod R) \times \lceil N/R \rceil + (r - N \bmod R) \times \lfloor N/R \rfloor$ | $\lfloor N/R \rfloor$ |

This produces at most a difference of 1 between the largest and smallest block — optimal balance.

### 3.2 MPI Collective Parameters

For `MPI_Allgatherv`, each rank must know the counts and displacements for all ranks:

| Parameter       | Computation                                 | Type       |
| --------------- | ------------------------------------------- | ---------- |
| `sendcount`     | Number of items this rank contributes       | `i32`      |
| `recvcounts[r]` | Number of items rank $r$ contributes (§3.1) | `Vec<i32>` |
| `displs[r]`     | Cumulative sum of `recvcounts[0..r]`        | `Vec<i32>` |

These are computed once per iteration (or once at startup if $M$ is fixed) and reused for both forward state gathering and backward cut gathering.

## 4. Load Balancing

### 4.1 Sources of Imbalance

| Source                              | Magnitude                                    | Mitigation                                                                                                         |
| ----------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| LP simplex iteration count variance | Small (~5-15% per solve)                     | Dynamic scheduling within rank absorbs this                                                                        |
| Problem heterogeneity across stages | Negligible — same LP structure per stage     | None needed                                                                                                        |
| Cut count growth across iterations  | All ranks add the same cuts simultaneously   | None needed — symmetric                                                                                            |
| NUMA memory access latency          | Significant if threads cross NUMA boundaries | Addressed by NUMA-local workspace allocation (see [Solver Workspaces §1](../03-architecture/solver-workspaces.md)) |

### 4.2 Why Static Distribution is Sufficient

SDDP forward pass scenarios have nearly identical computational cost because:

1. **Same LP structure** — all scenarios at stage $t$ solve the same LP (same constraints, same variable count, same cut count). Only the RHS differs.
2. **Solver warm-start** — all threads warm-start from the same per-stage basis cache, leading to similar iteration counts.
3. **No branching variance** — unlike branch-and-bound, simplex iteration count variance is bounded.

The backward pass has slightly more variance (trial points at different positions in state space may require different simplex iteration counts), but this is absorbed by dynamic OpenMP scheduling within each rank.

### 4.3 Dynamic Scheduling Within Rank

OpenMP uses `schedule(dynamic,1)` for distributing trial points across threads within a rank. This means:

- Threads pick up the next available trial point when they finish the current one
- No pre-assignment — if one trial point takes longer (more simplex iterations), other threads continue with remaining trial points
- Chunk size of 1 provides finest-grained load balancing within the rank

This two-level approach (static across ranks, dynamic within rank) provides optimal load balance without the complexity and latency overhead of cross-rank dynamic dispatch.

## 5. Design Rationale: Scenario-Based Distribution

The backward pass distributes work by **scenario (trial point)**, not by **state value**. Each thread owns a set of trial points and evaluates all openings for each.

The alternative — state-based distribution — would first deduplicate trial points (multiple scenarios may visit similar states), then distribute unique states across ranks. This was rejected because:

1. **Warm-start loss**: State-based distribution breaks the thread-trajectory affinity that makes the solver basis from the forward pass reusable in the backward pass. Without warm-start, LP solve time increases substantially.
2. **Synchronization cost**: Collecting and deduplicating states across all ranks requires additional communication before the backward pass can begin.
3. **Marginal benefit**: State deduplication typically saves a small percentage of LP solves — far less than the warm-start loss.
4. **Cache locality**: Scenario-based distribution keeps each thread's LP data in its cache hierarchy. State-based distribution would scatter work across threads with no cache affinity.

> **Deferred**: State deduplication as an optimization (merging near-identical trial points within a rank) is documented in [Deferred Features §C.17](../06-deferred/deferred-features.md).

> **Deferred**: Pipelined backward pass (using $V_{t+1}^{k-1}$ from the previous iteration to overlap computation with communication) is documented in [Deferred Features §C.18](../06-deferred/deferred-features.md).

## Cross-References

- [Hybrid Parallelism](./hybrid-parallelism.md) — ferrompi + OpenMP architecture, static distribution across ranks, thread-trajectory affinity
- [Training Loop §4.3](../03-architecture/training-loop.md) — Forward pass parallel distribution: contiguous blocks to ranks, thread-trajectory affinity
- [Training Loop §6.2-§6.3](../03-architecture/training-loop.md) — Backward pass: per-stage execution, trial point distribution, MPI_Allgatherv
- [SDDP Algorithm §3.4](../01-math/sddp-algorithm.md) — Thread-trajectory affinity, backward sync barriers, forward pass state saving
- [Solver Workspaces §1](../03-architecture/solver-workspaces.md) — Thread-local solver infrastructure, per-stage basis cache, NUMA-local allocation
- [Cut Management Implementation §4](../03-architecture/cut-management-impl.md) — MPI cut synchronization protocol and wire format
- [Scenario Generation §2.3](../03-architecture/scenario-generation.md) — Fixed opening tree: the $N_{\text{openings}}$ noise vectors evaluated in the backward pass
- [Synchronization](./synchronization.md) — Sync points, per-stage barrier semantics
- [Communication Patterns](./communication-patterns.md) — ferrompi collectives: `MPI_Allreduce`, `MPI_Allgatherv`
- [Shared Memory Aggregation](./shared-memory-aggregation.md) — Hierarchical cut aggregation within node
- [Memory Architecture](./memory-architecture.md) — NUMA-aware allocation, memory budget for per-rank resources
- [Deferred Features](../06-deferred/deferred-features.md) — State deduplication (C.17), pipelined backward pass (C.18)
