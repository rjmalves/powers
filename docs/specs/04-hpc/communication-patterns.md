---
status: approved
review_priority: 4-low
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.1 (MPI Communication Summary)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.2 (MPI 4.0 Persistent Collectives)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.4 (Hybrid Shared Memory Architecture)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.5 (Communication Volume Analysis)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.6 (Asynchronous Communication Overlap)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.7 (Communication Performance Targets)"
  - "DATA_MODEL_SPECIFICATION.md §6.2 (Message Structures)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from monolith docs (T-022); migrated all MPI references to ferrompi crate API"
  - date: 2026-02-23
    description: "P4 review rewrite. Fixed review_priority (2-high → 4-low). Stripped 3 Rust code blocks (~165 lines: CutSyncManager, CutMessage/FcfUpdateMessage/PersistentComm, backward_pass_with_overlap). Removed master/worker broadcast pattern — replaced with symmetric MPI_Allgatherv (per synchronization.md §1.1, work-distribution.md §2.2). Removed #[repr(C)] CutMessage — wire format is compact binary per cut-management-impl.md §4.2. Removed SharedWindow<T> for FCF storage and window.fence() — not in approved architecture. Removed async overlap (pipelined backward pass deferred to C.18). Removed fabricated memory numbers (168 GB/22.5 GB). Corrected communication volume with derivation from cut-management-impl.md §4.2. Persistent collectives reframed as optimization opportunity, not mandate. Cross-references expanded from 4 to 12."
---

# Communication Patterns

## Purpose

This spec defines the MPI communication patterns used by POWE.RS during SDDP training: the collective operations, their data payloads, wire formats, communication volume analysis, and optimization opportunities. All MPI operations use the ferrompi crate — no raw C FFI for MPI. This spec details the communication mechanics that [Synchronization](./synchronization.md) defines at the protocol level and [Cut Management Implementation §4](../03-architecture/cut-management-impl.md) defines for cut wire format.

## 1. MPI Operations

### 1.1 Operations Summary

POWE.RS uses exactly three MPI collective operations during SDDP training. All use ferrompi's safe, generic API with `Communicator` handles that are `Send + Sync`.

| Operation        | ferrompi API                                          | When                    | Data                          | Frequency                |
| ---------------- | ----------------------------------------------------- | ----------------------- | ----------------------------- | ------------------------ |
| `MPI_Allgatherv` | `comm.allgatherv(&send, &mut recv, &counts, &displs)` | Forward → backward      | Visited states (trial points) | Once per iteration       |
| `MPI_Allgatherv` | `comm.allgatherv(&send, &mut recv, &counts, &displs)` | Backward stage boundary | New cuts at stage $t$         | Once per stage ($T - 1$) |
| `MPI_Allreduce`  | `comm.allreduce(&send, &mut recv, Op::Sum)`           | Post-forward            | Convergence statistics        | Once per iteration       |

Additionally, initialization uses standard (non-iterative) collectives:

| Operation     | ferrompi API                 | When       | Data                     |
| ------------- | ---------------------------- | ---------- | ------------------------ |
| `MPI_Bcast`   | `comm.bcast(&mut buf, root)` | Startup    | Configuration, case data |
| `MPI_Barrier` | `comm.barrier()`             | Checkpoint | Synchronization only     |

### 1.2 No Point-to-Point Messaging

The approved architecture uses only collective operations — no point-to-point (`Send`/`Recv`) communication. The symmetric `MPI_Allgatherv` pattern ensures all ranks have identical data after each synchronization point, eliminating the need for a master/worker protocol.

## 2. Data Payloads

### 2.1 Trial Point Payload (Forward → Backward)

After the forward pass, each rank contributes its visited states to `MPI_Allgatherv`. The payload per trial point consists of the state vector:

| Component       | Type    | Size per trial point              |
| --------------- | ------- | --------------------------------- |
| Storage volumes | `[f64]` | $N_{\text{hydro}} \times 8$ bytes |
| AR inflow lags  | `[f64]` | $\sum_h P_h \times 8$ bytes       |
| Stage index     | `u32`   | 4 bytes                           |

At production scale ($N_{\text{hydro}} = 160$, average $P_h = 6$ lags, $M = 200$ trajectories, $T = 120$ stages):

- State dimension: $160 + 160 \times 6 = 1{,}120$ doubles = 8,960 bytes per trial point
- Trial points per stage: 200
- Total payload: $200 \times 8{,}964 \approx 1.75$ MB per stage, or $1.75 \times 120 \approx 210$ MB for all stages

The `MPI_Allgatherv` counts and displacements are computed from the contiguous block assignment (see [Work Distribution §3.1](./work-distribution.md)).

### 2.2 Cut Payload (Backward Stage Boundary)

After generating cuts at each backward stage, ranks exchange cuts via `MPI_Allgatherv`. The wire format is a compact binary representation optimized for bandwidth — see [Cut Management Implementation §4.2](../03-architecture/cut-management-impl.md) for the complete specification.

| Field              | Type    | Size (production scale)              |
| ------------------ | ------- | ------------------------------------ |
| Slot index         | `u32`   | 4 bytes                              |
| Iteration          | `u32`   | 4 bytes                              |
| Forward pass index | `u32`   | 4 bytes                              |
| Intercept          | `f64`   | 8 bytes                              |
| Coefficients       | `[f64]` | $D_{\text{state}} \times 8$ bytes    |
| **Total per cut**  |         | **~16,660 bytes** (at $D = 2{,}080$) |

At production scale with $M = 200$ forward passes and $R = 16$ ranks, each rank generates $\lfloor 200/16 \rfloor = 12\text{-}13$ cuts per stage. The `MPI_Allgatherv` payload is $200 \times 16{,}660 \approx 3.3$ MB per stage.

### 2.3 Convergence Statistics Payload (Post-Forward)

The `MPI_Allreduce` aggregates 4 scalars using `Op::Sum`:

| Quantity                            | Type  | Reduction | Purpose                                |
| ----------------------------------- | ----- | --------- | -------------------------------------- |
| First-stage LP objective            | `f64` | `MPI_MIN` | Lower bound (monotonically increasing) |
| Total forward cost (sum)            | `f64` | `MPI_SUM` | Upper bound mean computation           |
| Total forward cost (sum of squares) | `f64` | `MPI_SUM` | Upper bound variance computation       |
| Trajectory count                    | `f64` | `MPI_SUM` | Denominator for mean/variance          |

Total payload: 32 bytes. See [Work Distribution §1.4](./work-distribution.md) and [Convergence Monitoring §3](../03-architecture/convergence-monitoring.md).

> **Note**: The lower bound uses `MPI_MIN` while the other quantities use `MPI_SUM`. If ferrompi does not support mixed reduction operations in a single `allreduce`, this may require two separate calls (one `MPI_MIN` for lower bound, one `MPI_SUM` for the other 3 scalars) or a custom reduction operation.

## 3. Communication Volume Analysis

### 3.1 Per-Iteration Budget

Reference configuration: $R = 16$ ranks, $T = 120$ stages, $M = 200$ forward passes, $D_{\text{state}} = 2{,}080$.

| Operation                | Per-stage | Per-iteration        | Notes                              |
| ------------------------ | --------- | -------------------- | ---------------------------------- |
| Trial point `allgatherv` | —         | ~210 MB (once)       | All stages' visited states at once |
| Cut `allgatherv`         | ~3.3 MB   | ~393 MB (119 stages) | Per cut-management-impl.md §4.2    |
| Convergence `allreduce`  | —         | 32 bytes (once)      | 4 scalars                          |
| **Total per iteration**  |           | **~603 MB**          |                                    |

### 3.2 Bandwidth Requirements

On InfiniBand HDR (200 Gb/s = 25 GB/s):

- 603 MB takes ~24 ms at wire speed
- With protocol overhead (~50%), ~48 ms per iteration
- At 200 iterations total: ~9.6 seconds of communication
- Typical training time: 30-60 minutes → communication fraction: **< 1%**

On 100 Gbps Ethernet (12.5 GB/s):

- 603 MB takes ~48 ms at wire speed
- With TCP/RDMA overhead (~100%), ~96 ms per iteration
- At 200 iterations: ~19.2 seconds → communication fraction: **~1-2%**

SDDP's communication-to-computation ratio is low. The LP solve time dominates.

## 4. Persistent Collectives

### 4.1 Optimization Opportunity

MPI 4.0 persistent collectives (`MPI_Allgatherv_init`, `MPI_Allreduce_init`) allow pre-negotiating communication patterns at initialization and reusing them across iterations. This amortizes setup cost over the ~100-200 iterations of an SDDP training run.

| Aspect                  | Standard Collective  | Persistent Collective    |
| ----------------------- | -------------------- | ------------------------ |
| Setup cost per call     | Protocol negotiation | None (pre-negotiated)    |
| Subsequent call latency | Full negotiation     | Reduced                  |
| Buffer requirements     | Any buffer per call  | Fixed buffers at init    |
| ferrompi support        | `comm.allgatherv()`  | `comm.allgatherv_init()` |

### 4.2 Applicability to SDDP

The three collective operations in §1.1 are candidates for persistent collectives:

| Operation                | Persistent candidate? | Notes                                                                       |
| ------------------------ | --------------------- | --------------------------------------------------------------------------- |
| Cut `allgatherv`         | Yes                   | Same pattern every stage, buffer sizes vary per iteration (cut count grows) |
| Convergence `allreduce`  | Yes                   | Fixed 32-byte payload, identical every iteration                            |
| Trial point `allgatherv` | Conditional           | Only if $M$ is fixed across iterations; if adaptive, buffer sizes change    |

> **Implementation note**: Persistent collectives require fixed buffer addresses at initialization. If the cut count per rank varies across iterations (which it may, due to cut selection), the send buffer must be pre-allocated at the maximum expected size. This is consistent with the cut pool preallocation strategy in [Solver Abstraction §5](../03-architecture/solver-abstraction.md).

### 4.3 Design Decision

Whether to use persistent or standard collectives is an **implementation choice**, not an architectural requirement. The approved synchronization model ([Synchronization §1.1](./synchronization.md)) specifies the collective operations and their semantics but does not mandate persistence. The decision should be based on profiling: if communication accounts for less than 5% of total training time (§3.2), the 5-10x speedup from persistent collectives yields marginal absolute improvement.

## 5. Intra-Node Shared Memory

### 5.1 SharedWindow\<T\>

ferrompi's `SharedWindow<T>` enables ranks on the same physical node to share memory regions without replication. This is used for large read-only data structures that would otherwise be duplicated across ranks within a node.

| Capability            | ferrompi API                     | Use Case                                          |
| --------------------- | -------------------------------- | ------------------------------------------------- |
| Window creation       | `SharedWindow::new(comm, count)` | Allocate shared region on intra-node communicator |
| Intra-node grouping   | `comm.split_shared_memory()`     | Identify co-located ranks                         |
| Read access           | Direct pointer dereference       | Zero-copy reads from shared region                |
| Write synchronization | `window.fence()`                 | Ensure visibility of writes across ranks          |

### 5.2 Shared Data Candidates

| Data Structure         | Per-Rank Size (production) | Shareable? | Rationale                                                      |
| ---------------------- | -------------------------- | ---------- | -------------------------------------------------------------- |
| Scenario noise vectors | Large (opening tree)       | Yes        | Read-only during training, identical across ranks on same node |
| Input case data        | Moderate                   | Yes        | Read-only after initialization                                 |
| Cut pool               | Large (grows each iter)    | Partial    | Read-heavy in forward pass, written at stage boundaries only   |
| Solver workspace       | Per-thread                 | No         | Thread-local mutable state, must not be shared                 |

The memory savings from `SharedWindow<T>` are quantified in [Memory Architecture](./memory-architecture.md).

> **Design point**: The extent to which `SharedWindow<T>` is used for the cut pool depends on the access pattern analysis in [Shared Memory Aggregation](./shared-memory-aggregation.md). The baseline approach (each rank maintains its own cut pool, synchronized via `MPI_Allgatherv`) is simple and correct; shared memory is an optimization to reduce memory footprint on memory-constrained nodes.

## 6. Deterministic Communication

### 6.1 Reproducibility Invariant

All MPI collective operations in the SDDP training loop are deterministic: given the same inputs and rank count, every rank produces identical results after synchronization. This is critical for the SDDP correctness requirement that all ranks have identical FCFs (see [Cut Management Implementation §4.3](../03-architecture/cut-management-impl.md)).

Determinism sources:

- **Cut slot assignment** — Computed from `(iteration, forward_pass_index)`, deterministic across all ranks
- **Contiguous block distribution** — Forward pass scenarios assigned by rank index, reproducible
- **MPI_Allgatherv ordering** — Receives data in rank order (rank 0, rank 1, ..., rank $R-1$)

### 6.2 Floating-Point Reduction

`MPI_Allreduce` with `Op::Sum` may produce different results depending on reduction tree shape (non-associativity of floating-point addition). For convergence statistics (§2.3), this variance is acceptable — the upper bound is already a statistical estimate. For the lower bound (`MPI_MIN`), the operation is exact.

## Cross-References

- [Synchronization §1.1](./synchronization.md) — Three collective operations per iteration, their timing and semantics
- [Synchronization §1.4](./synchronization.md) — Per-stage barrier via `MPI_Allgatherv` implicit synchronization
- [Work Distribution §1.4](./work-distribution.md) — Post-forward `MPI_Allreduce` with 4 convergence quantities
- [Work Distribution §2.2](./work-distribution.md) — Per-stage backward pass execution, `MPI_Allgatherv` for cuts
- [Work Distribution §3](./work-distribution.md) — Contiguous block assignment arithmetic, `MPI_Allgatherv` parameters
- [Cut Management Implementation §4](../03-architecture/cut-management-impl.md) — Wire format, deterministic slot assignment, synchronization protocol
- [Hybrid Parallelism §1.2](./hybrid-parallelism.md) — ferrompi capabilities table, `SharedWindow<T>`, `split_shared_memory()`
- [Convergence Monitoring §3](../03-architecture/convergence-monitoring.md) — Cross-rank bound aggregation
- [Training Loop §5.2](../03-architecture/training-loop.md) — `MPI_Allgatherv` for trial point collection
- [Training Loop §6.3](../03-architecture/training-loop.md) — `MPI_Allgatherv` for cut distribution
- [Shared Memory Aggregation](./shared-memory-aggregation.md) — Intra-node shared memory patterns, hierarchical cut aggregation
- [Memory Architecture](./memory-architecture.md) — Memory budget, shared memory savings quantification
