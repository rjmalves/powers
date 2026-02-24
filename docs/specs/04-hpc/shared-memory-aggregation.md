---
status: approved
review_priority: 4-low
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §6.4 (Hierarchical Cut Aggregation)"
  - "DATA_MODEL_SPECIFICATION.md §6.6 (Intra-Node Shared Memory)"
  - "DATA_MODEL_SPECIFICATION.md §6.8 (Performance Monitoring)"
  - "DATA_MODEL_SPECIFICATION.md §6.11 (Shared Memory Scenario Storage)"
  - "DATA_MODEL_SPECIFICATION.md §6.12 (Two-Level Cut Aggregation)"
  - "DATA_MODEL_SPECIFICATION.md §6.13 (Reproducibility Guarantees)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from monolith docs (T-022); migrated all MPI references to ferrompi crate API"
  - date: 2026-02-23
    description: "P4 review rewrite. Fixed review_priority (2-high → 4-low). Stripped 8 Rust code blocks (~350 lines: AggregationNode, hierarchical_aggregate, SharedFcf, scenario_seed, generate_scenarios_distributed, SharedScenarioStorage, aggregate_single_cut, replicated_cut_selection, ReproducibilityChecker). Removed hierarchical tree aggregation — approved architecture uses flat MPI_Allgatherv (per synchronization.md §1.1). Removed two-level reduce+broadcast — replaced with symmetric MPI_Allgatherv (per work-distribution.md §2.2). Removed SharedFCF as primary architecture — reframed as optimization candidate (per communication-patterns.md §5.2). Removed fabricated memory numbers (18.6 GB FCF). Removed scenario_seed code — references scenario-generation.md §2.2 instead. Removed unapproved reproducibility config block. Removed load balance analysis code. Reorganized around three focused topics: SharedWindow usage patterns, reproducibility guarantees, performance monitoring. Cross-references expanded from 5 to 12."
---

# Shared Memory and Aggregation

## Purpose

This spec defines the intra-node shared memory usage patterns, reproducibility guarantees, and performance monitoring for POWE.RS. It extends [Communication Patterns §5](./communication-patterns.md) with detailed `SharedWindow<T>` allocation and access patterns, consolidates reproducibility requirements from across the approved specs, and defines performance monitoring interpretation. For the MPI collective operations and their semantics, see [Communication Patterns](./communication-patterns.md) and [Synchronization](./synchronization.md).

## 1. Intra-Node Shared Memory

### 1.1 SharedWindow\<T\> Usage Model

ferrompi's `SharedWindow<T>` allows MPI ranks on the same physical node to share memory regions. POWE.RS uses a **leader allocation pattern**: one rank per node (the leader, determined by rank 0 within the intra-node communicator from `split_shared_memory()`) allocates the shared region; other ranks on the same node access it via pointer dereference.

| Role     | Allocation                   | Read Access                | Write Access                            |
| -------- | ---------------------------- | -------------------------- | --------------------------------------- |
| Leader   | Allocates full region        | Direct pointer dereference | Direct write, then `window.fence()`     |
| Follower | Allocates size 0 (no memory) | Direct pointer dereference | Not permitted (leader writes on behalf) |

The `SharedWindow<T>` type provides RAII semantics — `Drop` automatically frees the MPI window, eliminating resource leaks.

### 1.2 Shared Data: Opening Tree

The opening tree (fixed noise vectors for the backward pass — see [Scenario Generation §2.3](../03-architecture/scenario-generation.md)) is the primary candidate for `SharedWindow<T>`:

| Property        | Value                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| Size            | $N_{\text{openings}} \times T \times N_{\text{entities}} \times 8$ bytes (~30 MB at production scale) |
| Access pattern  | Read-only during training (generated once before first iteration)                                     |
| Sharing benefit | Avoid replicating ~30 MB per rank on same node (4 ranks = 90 MB saved)                                |
| Write phase     | Leader generates during initialization, `window.fence()` ensures visibility                           |
| Memory layout   | Opening-major ordering for backward pass locality (see scenario-generation.md §2.3)                   |

**Generation protocol:**

1. Create intra-node communicator via `comm.split_shared_memory()`
2. Leader allocates `SharedWindow<f64>` with total opening tree size; followers allocate size 0
3. Distribute generation work across ranks within the node (each rank generates its assigned portion via contiguous block assignment)
4. Each rank writes its portion directly to the shared window at the correct offset
5. `window.fence()` ensures all writes are visible to all ranks on the node
6. All ranks read any opening via direct pointer dereference (zero-copy)

### 1.3 Shared Data: Input Case Data

Static input data (hydro parameters, thermal parameters, system topology) loaded during initialization is read-only during training. Sharing via `SharedWindow<T>` avoids per-rank replication:

| Data                   | Approximate Size | Shareable? | Notes                             |
| ---------------------- | ---------------- | ---------- | --------------------------------- |
| System entity data     | ~10 MB           | Yes        | Read-only after initialization    |
| PAR model parameters   | ~5 MB            | Yes        | Preprocessed contiguous arrays    |
| Correlation factors    | ~2 MB            | Yes        | Cholesky factors, one per profile |
| Block/exchange factors | ~1 MB            | Yes        | Loaded once, read-only            |

Whether these smaller data structures justify `SharedWindow<T>` overhead (allocation, fence, pointer indirection) depends on the number of ranks per node. With 4+ ranks per node, the savings compound.

### 1.4 Shared Data: Cut Pool (Optimization Candidate)

The cut pool grows each iteration and is the largest runtime data structure. Sharing the cut pool via `SharedWindow<T>` would eliminate per-rank replication within a node:

| Property            | Baseline (replicated)                   | Shared (optimization)                    |
| ------------------- | --------------------------------------- | ---------------------------------------- |
| Memory per node     | Cut pool size × ranks per node          | Cut pool size × 1                        |
| Read access         | Local memory (fastest)                  | Shared window pointer (may cross NUMA)   |
| Write access        | Each rank updates locally               | Leader writes, fence, followers read     |
| Forward pass read   | Hot-path: evaluating cuts at each stage | Potentially slower if shared across NUMA |
| Backward pass write | Each rank adds new cuts                 | Leader integrates after `MPI_Allgatherv` |

> **Design point**: The baseline approach (each rank maintains its own cut pool, synchronized via `MPI_Allgatherv`) is simple and correct. Shared memory cut pool is an optimization for memory-constrained nodes, with a latency trade-off. The quantification of this trade-off depends on the production-scale cut pool size — see [Memory Architecture](./memory-architecture.md). See also [Communication Patterns §5.2](./communication-patterns.md).

## 2. Intra-Node Cut Aggregation

### 2.1 Baseline: Flat MPI_Allgatherv

The approved architecture uses flat `MPI_Allgatherv` for cut distribution: every rank sends its cuts and every rank receives all cuts. There is no hierarchical aggregation, no root-based reduction, and no point-to-point messaging. See [Synchronization §1.1](./synchronization.md) and [Communication Patterns §1.2](./communication-patterns.md).

### 2.2 Optimization: Two-Level Aggregation

For deployments with many ranks per node (8+), a two-level aggregation can reduce inter-node `MPI_Allgatherv` volume by first aggregating cuts within a node:

1. **Level 1 (Intra-node)**: Ranks on the same node contribute cuts to shared memory via `SharedWindow<T>`. The node leader collects and deduplicates if applicable.
2. **Level 2 (Inter-node)**: Node leaders participate in `MPI_Allgatherv` across nodes, exchanging the aggregated per-node cut sets.
3. **Distribute within node**: Node leaders write received remote cuts to shared memory; `window.fence()` ensures visibility.

This reduces the `MPI_Allgatherv` participant count from $R$ total ranks to $R / R_{\text{node}}$ node leaders, reducing collective latency for large rank counts. However, it adds intra-node synchronization overhead.

> **Design decision**: Two-level aggregation is an optimization for large-scale deployments (64+ ranks). For the initial implementation with moderate rank counts (4-16 ranks), flat `MPI_Allgatherv` is sufficient. Profiling should guide whether to enable this optimization. See [Communication Patterns §3.2](./communication-patterns.md) for bandwidth analysis showing communication fraction < 2% even on Ethernet.

## 3. Reproducibility Guarantees

### 3.1 Requirement

Given the same inputs and random seed, POWE.RS must produce **bit-for-bit identical** results regardless of:

| Must Be Independent Of            | Mechanism                                                                     |
| --------------------------------- | ----------------------------------------------------------------------------- |
| Number of MPI ranks               | Deterministic seeding, contiguous block distribution, deterministic cut slots |
| Number of OpenMP threads per rank | Thread-local accumulation, OpenMP barrier, single-threaded merge              |
| Execution timing/ordering         | Identity-based seeding, deterministic `MPI_Allgatherv` rank ordering          |

### 3.2 Component-Level Reproducibility

Each component of the SDDP pipeline has specific reproducibility mechanisms:

| Component                | Mechanism                                                                   | Reference                                                                       |
| ------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Scenario noise           | Deterministic seed from `(base_seed, iteration, scenario, stage)`           | [Scenario Generation §2.2](../03-architecture/scenario-generation.md)           |
| Opening tree             | Deterministic seed from `(base_seed, opening_index, stage)`                 | [Scenario Generation §2.3](../03-architecture/scenario-generation.md)           |
| Forward pass assignment  | Contiguous block distribution by rank index                                 | [Work Distribution §3.1](./work-distribution.md)                                |
| Backward pass assignment | Contiguous block distribution by rank index                                 | [Work Distribution §2.1](./work-distribution.md)                                |
| Cut slot positions       | Deterministic from `(iteration, forward_pass_index)`                        | [Cut Management Implementation §4.3](../03-architecture/cut-management-impl.md) |
| Cut synchronization      | `MPI_Allgatherv` receives in rank order (0, 1, ..., R-1)                    | [Communication Patterns §6.1](./communication-patterns.md)                      |
| Cut selection            | Deterministic algorithm on identical data (all ranks compute independently) | [Cut Management Implementation §6](../03-architecture/cut-management-impl.md)   |
| LP constraint order      | Cuts added by slot index (iteration order)                                  | [Solver Abstraction §5](../03-architecture/solver-abstraction.md)               |

### 3.3 Floating-Point Considerations

**MPI reductions**: `MPI_Allreduce` with `Op::Sum` may produce different results depending on the reduction tree shape (non-associativity of floating-point addition). For convergence statistics this is acceptable — the upper bound is a statistical estimate. See [Communication Patterns §6.2](./communication-patterns.md).

**OpenMP reductions**: Thread-local accumulation followed by single-threaded merge (see [Synchronization §3](./synchronization.md)) produces deterministic results because the merge order is fixed (thread 0, thread 1, ..., thread $N-1$). This avoids the non-determinism of `#pragma omp reduction(+:sum)` where summation order depends on thread scheduling.

**Cut coefficient aggregation**: For single-cut formulation, per-opening results are aggregated within each thread (sequential — deterministic), then merged across threads (fixed order — deterministic), then exchanged via `MPI_Allgatherv` (rank order — deterministic). The final averaging is performed locally by each rank on the full set of results (identical data — deterministic). No floating-point non-determinism is introduced at any step.

### 3.4 Verification

Reproducibility can be verified by running the same case with different rank/thread configurations and comparing:

| Quantity          | Comparison                 | Expected Result        |
| ----------------- | -------------------------- | ---------------------- |
| Lower bound trace | Bit-for-bit across configs | Identical              |
| Upper bound mean  | Within FP tolerance        | ~1e-12 relative error  |
| Final cut pool    | Bit-for-bit across configs | Identical coefficients |
| Policy output     | Bit-for-bit across configs | Identical              |

Upper bound mean may have small FP differences due to `MPI_Allreduce` tree shape variation, but the lower bound and cut pool (which determine the policy) must be exact.

## 4. Performance Monitoring

### 4.1 Approved Timing Outputs

POWE.RS writes per-iteration and per-rank timing data to Parquet files during training. The schemas are defined in [Output Schemas §6.2-§6.3](../02-data-model/output-schemas.md):

**Per-iteration** (`training/timing/iterations.parquet`): forward solve, backward solve, cut computation, cut selection, MPI communication, I/O, overhead.

**Per-rank** (`training/timing/mpi_ranks.parquet`): forward time, backward time, communication time, idle time, LP solve count, scenarios processed.

### 4.2 Diagnostic Interpretation

| Symptom                                      | Metric to Check                         | Likely Cause                                        | Mitigation                                                       |
| -------------------------------------------- | --------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| Idle time >> 0 on some ranks                 | `idle_time_ms` variance across ranks    | Load imbalance                                      | Check LP iteration count variance; consider NUMA placement       |
| Communication time growing across iterations | `communication_time_ms` trend           | Cut count growth increases `MPI_Allgatherv` payload | Enable cut selection to bound active cut count                   |
| Forward time >> backward time                | `forward_time_ms` vs `backward_time_ms` | Many forward passes, few trial points per rank      | Expected for high $M / R$ ratio                                  |
| Backward time growing across iterations      | `backward_solve_ms` trend               | More cuts in FCF → larger LPs → longer solves       | Enable cut selection; check for dominated cuts                   |
| Communication fraction > 10%                 | `communication_time_ms / total`         | Insufficient compute per rank                       | Reduce rank count (fewer, larger ranks); check network bandwidth |

### 4.3 Load Balance Assessment

The primary load balance indicator is the ratio of maximum to minimum `forward_time_ms` (or `backward_time_ms`) across ranks within an iteration. With static contiguous block distribution and homogeneous LP structure, this ratio should be close to 1.0:

| Imbalance Ratio | Assessment  | Action                                          |
| --------------- | ----------- | ----------------------------------------------- |
| < 1.1           | Excellent   | None needed                                     |
| 1.1 - 1.2       | Acceptable  | Monitor — may improve with NUMA tuning          |
| 1.2 - 1.5       | Notable     | Check NUMA placement, check LP iteration counts |
| > 1.5           | Significant | Investigate — likely NUMA or hardware issue     |

## Cross-References

- [Communication Patterns §5](./communication-patterns.md) — `SharedWindow<T>` capabilities and shared data candidates
- [Communication Patterns §6](./communication-patterns.md) — Deterministic communication, floating-point reduction
- [Communication Patterns §3](./communication-patterns.md) — Communication volume analysis, bandwidth requirements
- [Synchronization §1.1](./synchronization.md) — Three MPI collective operations per iteration
- [Synchronization §3](./synchronization.md) — Thread-local cut accumulation pattern
- [Scenario Generation §2.2](../03-architecture/scenario-generation.md) — Deterministic seed derivation, reproducible sampling
- [Scenario Generation §2.3](../03-architecture/scenario-generation.md) — Fixed opening tree generation and memory layout
- [Cut Management Implementation §4](../03-architecture/cut-management-impl.md) — Wire format, deterministic slot assignment
- [Work Distribution §3.1](./work-distribution.md) — Contiguous block distribution arithmetic
- [Solver Abstraction §5](../03-architecture/solver-abstraction.md) — Cut pool preallocation, slot assignment
- [Memory Architecture](./memory-architecture.md) — Memory budget, NUMA-aware allocation, shared memory savings quantification
- [Output Schemas §6.2-§6.3](../02-data-model/output-schemas.md) — Timing output Parquet schemas
