---
status: approved
review_priority: 3-medium
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §15 (15.1-15.4)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from monolith docs"
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): Generic constraints that involve state variables (storage volumes) produce duals that contribute to cut coefficients (see cut-management.md §2). The mapping from generic constraint duals to state variable coefficients is static (determined at input loading time) and should be precomputed once into a lookup structure, then reused across all iterations. This preprocessing step must be designed during P3 review. See sddp-algorithm.md §3.4."
  - date: 2026-02-23
    description: "P3 review — full rewrite. Fix review_priority (2→3). Strip all Rust code (4 blocks, ~310 lines) and replace with behavioral descriptions and tables. §1 rewritten: FCF structure as behavioral description aligned with solver-abstraction.md §5 (cut pool preallocation, deterministic slot assignment, activity bitmap). §2 rewritten: cut selection strategies aligned with approved cut-management.md §7 (Level-1, LML1, Dominated Detection) — removed 5 misaligned strategies from old Rust enum. §3 rewritten: removed custom binary serialization (PWRSCUTS magic bytes) — replaced with FlatBuffers reference per approved binary-formats.md §3. §4 rewritten: MPI cut synchronization as behavioral description. Added §5 generic constraint dual preprocessing (from sddp-algorithm.md review note). Added §6 cut activity tracking. Updated cross-references (3→12)."
---

# Cut Management Implementation

## Purpose

This spec defines how the mathematical cut management concepts from [Cut Management](../01-math/cut-management.md) are implemented in the POWE.RS architecture: the Future Cost Function (FCF) runtime structure, how cut selection strategies operate on the pre-allocated cut pool, cut serialization for checkpoint/resume via FlatBuffers, cross-rank cut synchronization via MPI, generic constraint dual preprocessing, and cut activity tracking.

For the mathematical foundations (cut definition, dual extraction, aggregation, validity, convergence), see [Cut Management](../01-math/cut-management.md). For the cut pool memory layout requirements and FlatBuffers schema, see [Binary Formats §3](../02-data-model/binary-formats.md). For how cuts enter the solver LP, see [Solver Abstraction §5](../03-architecture/solver-abstraction.md).

## 1. Future Cost Function Structure

The FCF is the runtime data structure that holds the piecewise-linear outer approximation of the cost-to-go function. It consists of one cut pool per stage, each using the pre-allocated layout defined in [Binary Formats §3.4](../02-data-model/binary-formats.md).

### 1.1 Per-Stage Cut Pool

Each stage's cut pool is a pre-allocated, fixed-capacity collection of Benders cuts. The pool is shared read-only across threads during the forward pass and written by a single thread per stage during the backward pass (with a synchronization barrier at each stage boundary — see [Training Loop §6.3](./training-loop.md)).

| Component           | Description                                                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coefficient storage | Contiguous dense `[f64]` arrays, one per cut, of length `state_dimension`. Stored in a flat buffer indexed by slot index for O(1) access. Cache-line aligned (64 bytes) for efficient CSR assembly. |
| Intercept array     | Separate contiguous `[f64]` array of intercepts (α values), one per slot. Separated from coefficients to avoid polluting cache during coefficient copy loops.                                       |
| Activity bitmap     | Bit array tracking which slots contain active cuts. An active count is maintained alongside the bitmap to avoid scanning.                                                                           |
| Populated count     | Number of slots that contain valid cuts (active or inactive). Slots beyond this count are empty (pre-allocated but unused).                                                                         |
| Cut metadata        | Per-slot metadata: iteration generated, forward pass index, last-active iteration (for LML1), cumulative active count (for Level-1 and dominated detection).                                        |

### 1.2 Deterministic Slot Assignment

Cut slots are computed, not allocated at runtime. The slot for a new cut is determined by a pure function:

```
slot = warm_start_count + iteration × forward_passes + forward_pass_index
```

This eliminates thread-safety concerns and non-determinism — the same inputs always produce the same slot. Combined with RNG state serialization, this enables bit-for-bit reproducible checkpoint/resume.

### 1.3 Capacity and Sizing

The pool is fully pre-allocated at initialization:

| Parameter              | Formula / Value                                           |
| ---------------------- | --------------------------------------------------------- |
| Capacity per stage     | `warm_start_cuts + max_iterations × forward_passes`       |
| Production example     | 5,000 + 50 × 200 = 15,000 slots                           |
| Per-cut memory         | `state_dimension × 8` bytes (coefficients) + metadata     |
| Per-stage total (prod) | 15,000 × 2,080 × 8 ≈ 238 MB coefficients + ~1 MB metadata |
| All stages (prod)      | 120 × 238 MB ≈ 28 GB per rank                             |

See [Binary Formats §4.3](../02-data-model/binary-formats.md) for the full sizing breakdown.

### 1.4 FCF-Level Operations

The FCF provides the following operations to the training loop:

| Operation         | Description                                                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Add cut           | Write a new cut's coefficients, intercept, and metadata into the deterministic slot. Set the slot as active in the bitmap. O(1).         |
| Get active cuts   | Return the active cut indices and coefficient/intercept data for a given stage. Used by the solver workspace for `addRows` CSR assembly. |
| Evaluate at state | Compute max over all active cuts: $\max_k \{ \alpha_k + \beta_k^\top x \}$. Used for upper bound evaluation and dominated cut detection. |
| Run selection     | Apply the configured cut selection strategy (§2) to deactivate cuts. Updates the activity bitmap.                                        |
| Aggregate stats   | Return total cuts, active cuts per stage, cuts added this iteration — for logging and convergence monitoring.                            |

## 2. Cut Selection Strategies

Three cut selection strategies are implemented, corresponding to the three strategies defined in [Cut Management §7](../01-math/cut-management.md). Selection runs periodically (every `check_frequency` iterations) and only deactivates cuts — it never deletes them, preserving slot indices for reproducibility.

### 2.1 Level-1

A cut is **Level-1 active** if it was binding at least once during the entire algorithm execution. At each selection check, cuts with a cumulative active count of zero are deactivated.

| Tracking data  | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `active_count` | Per-cut counter, incremented each time the cut is binding at an LP solution |

This is the least aggressive strategy — it retains any cut that was ever useful. Convergence guarantee is preserved (see [Cut Management §8](../01-math/cut-management.md)).

### 2.2 Limited Memory Level-1 (LML1)

Each cut is timestamped with the most recent iteration at which it was binding. At each selection check, cuts whose timestamp is older than `memory_window` iterations are deactivated.

| Tracking data      | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `last_active_iter` | Per-cut timestamp, updated each time the cut is binding              |
| `memory_window`    | Configurable parameter: number of iterations to retain inactive cuts |

More aggressive than Level-1 — forgets cuts that haven't been active recently. Convergence guarantee is preserved.

### 2.3 Dominated Cut Detection

A cut is dominated if at every visited state, some other cut achieves a higher value. At each selection check:

1. For each active cut $k$, evaluate $\alpha_k + \beta_k^\top \hat{x}$ at all visited states $\hat{x}$
2. For each visited state, compute the maximum value across all other active cuts
3. If cut $k$ is dominated at every visited state (within tolerance $\epsilon$), deactivate it

| Tracking data      | Description                                                                       |
| ------------------ | --------------------------------------------------------------------------------- |
| `domination_count` | Per-cut counter: number of visited states where this cut was dominated            |
| Visited states     | The set of trial points from forward passes, stored in the policy's `StageStates` |

This is the most aggressive strategy — directly identifies cuts providing no value at any known operating point. Computational cost is $\mathcal{O}(|\text{active cuts}| \times |\text{visited states}|)$ per stage per check.

### 2.4 Selection Configuration

| Parameter         | Description                                                 | Applies to |
| ----------------- | ----------------------------------------------------------- | ---------- |
| `method`          | Selection strategy: `"level1"`, `"lml1"`, or `"domination"` | All        |
| `threshold`       | Activity threshold $\epsilon$ (recommended: 0)              | All        |
| `check_frequency` | Iterations between selection runs                           | All        |
| `memory_window`   | Iterations to retain inactive cuts                          | LML1 only  |

See [Configuration Reference](../05-config/configuration-reference.md) for the JSON schema.

## 3. Cut Serialization

Cut persistence uses FlatBuffers, as defined in [Binary Formats §3.1](../02-data-model/binary-formats.md). The serialization supports three execution modes with different reproducibility guarantees:

| Mode       | What is serialized                                   | What is restored                             | Reproducibility         |
| ---------- | ---------------------------------------------------- | -------------------------------------------- | ----------------------- |
| Fresh      | Nothing — training starts empty                      | —                                            | Deterministic from seed |
| Warm-start | All cuts (active + inactive), slot indices, metadata | Cut pool populated from policy; new RNG seed | Different from original |
| Resume     | All cuts + activity flags + RNG state + basis        | Exact state at checkpoint iteration          | Bit-for-bit identical   |

### 3.1 Checkpoint Write

At checkpoint boundaries (configured interval or graceful shutdown), the following data is serialized per stage:

| Data                         | FlatBuffers Table       | Rationale                                       |
| ---------------------------- | ----------------------- | ----------------------------------------------- |
| All cuts (active + inactive) | `StageCuts`             | LP row structure must be reconstructable        |
| Activity flags               | `BendersCut.is_active`  | Which cuts to load into solver on resume        |
| Slot indices                 | `BendersCut.slot_index` | Row mapping for reproducibility                 |
| Cut metadata                 | `BendersCut` fields     | Iteration, forward pass index, domination count |
| Visited states               | `StageStates`           | Needed for dominated cut detection on resume    |
| Solver basis                 | `StageBasis`            | Exact warm-start on resume                      |

**Write strategy**: Rank 0 serializes the shared cut pool to the policy directory ([Binary Formats §3.2](../02-data-model/binary-formats.md)). Each stage is written as a separate `.bin` file. Compression (Zstd) is optional for archival.

### 3.2 Checkpoint Read (Resume / Warm-Start)

On resume, the FlatBuffers data is deserialized into the runtime cut pool layout:

1. Read the `StageCuts` FlatBuffer for each stage
2. Copy coefficient vectors from the FlatBuffers `[double]` arrays into the runtime contiguous layout (cache-line aligned)
3. Reconstruct the activity bitmap from `is_active` flags
4. Set `populated_count` and `warm_start_count`
5. Restore RNG state from `PolicyMetadata` (resume mode only)
6. Load cached basis from `StageBasis` into per-stage basis cache (see [Solver Workspaces §1.2](./solver-workspaces.md))

The FlatBuffers zero-copy property means step 2 can read coefficient data directly from the memory-mapped file without parsing — but the data must still be copied into the NUMA-local, cache-aligned runtime layout for hot-path performance.

## 4. Cross-Rank Cut Synchronization

After each backward pass stage, new cuts generated by each MPI rank must be distributed to all other ranks so that every rank has the complete FCF for the next iteration's forward pass.

### 4.1 Synchronization Protocol

At each stage boundary during the backward pass:

1. **Count** — Each rank determines how many new cuts it generated at this stage
2. **Size exchange** — `MPI_Allgather` of cut counts (one integer per rank)
3. **Data exchange** — `MPI_Allgatherv` of serialized cut data (coefficients + intercepts + metadata)
4. **Integration** — Each rank deserializes received cuts and writes them into the local cut pool at the deterministic slot positions

### 4.2 Serialization for MPI

The MPI wire format is a compact binary representation optimized for bandwidth, not the full FlatBuffers format (which includes schema metadata overhead). For each cut:

| Field              | Type    | Size (production)        |
| ------------------ | ------- | ------------------------ |
| Slot index         | `u32`   | 4 bytes                  |
| Iteration          | `u32`   | 4 bytes                  |
| Forward pass index | `u32`   | 4 bytes                  |
| Intercept          | `f64`   | 8 bytes                  |
| Coefficients       | `[f64]` | 2,080 × 8 = 16,640 bytes |
| **Total per cut**  |         | **~16,660 bytes**        |

At production scale with 200 forward passes per iteration and 16 ranks, each rank generates ~12-13 cuts per stage. The `MPI_Allgatherv` payload is ~200 cuts × 16,660 bytes ≈ 3.3 MB per stage — modest for InfiniBand interconnects.

### 4.3 Deterministic Integration

Because slot indices are computed from `(iteration, forward_pass_index)` and forward passes are distributed to ranks in contiguous blocks, each rank can compute the correct slot position for every received cut without negotiation. All ranks end up with identical cut pools after synchronization.

**Invariant**: After `MPI_Allgatherv` at stage $t$, every rank has the same set of active cuts at stage $t$, in the same slot positions, with the same coefficients. This is necessary for the forward pass at the next iteration to produce identical results regardless of which rank evaluates which trajectory.

## 5. Generic Constraint Dual Preprocessing

When generic constraints ([LP Formulation §10](../01-math/lp-formulation.md)) involve state variables (storage volumes), their dual multipliers contribute to cut coefficients. The mapping from generic constraint duals to state variable cut coefficients is **static** — it depends only on the constraint structure, which is determined at input loading time and does not change during training.

### 5.1 Preprocessing Step

During initialization (after input loading, before training begins):

1. For each generic constraint $c$ that references at least one state variable, extract the coefficient of each state variable in that constraint
2. Build a **dual-to-cut mapping** — a sparse structure that, for each state variable $j$, stores the list of `(constraint_index, coefficient)` pairs from generic constraints

This mapping is computed once and shared read-only across all threads.

### 5.2 Runtime Usage

During cut coefficient computation in the backward pass ([Training Loop §7.2](./training-loop.md)):

1. Extract the standard cut coefficients from water balance and AR lag constraint duals
2. For each generic constraint dual $\pi^{gen}_c$, look up the precomputed mapping and add $\pi^{gen}_c \times \text{coefficient}_{c,j}$ to the cut coefficient for each state variable $j$ that appears in constraint $c$

This is equivalent to a sparse matrix-vector multiply: `β_generic = G' × π_gen`, where `G` is the sparse state-variable participation matrix for generic constraints, precomputed at initialization.

### 5.3 Interaction with FPHA Duals

The FPHA hyperplane duals follow the same pattern — their contribution to storage cut coefficients is via a fixed mapping (the $\frac{1}{2} \gamma_v^m$ factor from [Cut Management §2](../01-math/cut-management.md)). Unlike generic constraints, the FPHA mapping changes per stage (different hydros may use different production models at different stages), but within a stage it is static. Both FPHA and generic constraint contributions can use the same precomputed lookup structure.

## 6. Cut Activity Tracking

After each LP solve, the training loop must update cut activity counters for the selection strategies to function. This happens in step 6 of the stage solve workflow ([Solver Workspaces §1.4](./solver-workspaces.md)).

### 6.1 Binding Detection

A cut is binding if its dual multiplier (shadow price) in the LP solution is positive. After solving the LP:

1. Extract the dual values for the cut rows (the bottom region of the LP, per [Solver Abstraction §2.2](./solver-abstraction.md))
2. For each cut row with a positive dual (above a configurable tolerance), mark the corresponding cut as active

### 6.2 Counter Updates

| Strategy  | Update per binding cut                                                 |
| --------- | ---------------------------------------------------------------------- |
| Level-1   | Increment `active_count` for the cut                                   |
| LML1      | Set `last_active_iter` to the current iteration                        |
| Dominated | Reset `domination_count` to 0 (the cut is not dominated at this state) |

These updates happen on the thread-local solver's cut rows. Since each thread processes its own trajectories and the backward pass has per-stage synchronization barriers, no locking is needed — each thread updates the activity data for the cuts it evaluated, and the results are reconciled during the cut synchronization step (§4).

## Cross-References

- [Cut Management (Math)](../01-math/cut-management.md) — Cut definition, dual extraction, aggregation, validity, selection theory, convergence guarantee
- [Solver Abstraction](./solver-abstraction.md) — Cut pool design (§5), LP layout convention (§2), how active cuts enter the solver LP (§5.4)
- [Solver Workspaces](./solver-workspaces.md) — Stage solve workflow (§1.4) where cuts are loaded and activity is tracked, cut loading cost analysis (§1.10)
- [Binary Formats](../02-data-model/binary-formats.md) — FlatBuffers schema (§3.1), policy directory structure (§3.2), cut pool memory layout requirements (§3.4), checkpoint reproducibility (§4)
- [Training Loop](./training-loop.md) — Forward/backward pass that drives cut generation (§6), dual extraction for cut coefficients (§7)
- [Convergence Monitoring](./convergence-monitoring.md) — Uses FCF statistics (lower bound from stage 1 cuts)
- [Risk Measures](../01-math/risk-measures.md) — CVaR modifies aggregation weights during cut generation
- [LP Formulation](../01-math/lp-formulation.md) — Generic constraints (§10) whose duals contribute to cut coefficients
- [Scenario Generation](./scenario-generation.md) — Fixed opening tree (§2.3) that defines backward pass branchings
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — MPI rank topology for cut synchronization
- [Checkpointing](../04-hpc/checkpointing.md) — Checkpoint trigger logic and graceful shutdown that invoke cut serialization
- [Configuration Reference](../05-config/configuration-reference.md) — `cut_selection` JSON schema parameters
