---
status: approved
review_priority: 3-medium
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.4.9 (Thread-Local Solver Infrastructure)"
  - "DATA_MODEL_SPECIFICATION.md §5.5 (5.5.1-5.5.5)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from DATA_MODEL_SPECIFICATION.md §5.4.9, §5.5.1-5.5.5"
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): Two critical performance concerns: (1) LP rebuild cost — memory constraints prevent keeping all stage LPs with full cut sets resident simultaneously; must minimize rebuild cost through cut preallocation, basis persistence, and incremental constraint updates. This is the critical performance path. (2) Forward pass state save/restore — when M > N (more forward passes than threads), need efficient context-switching of solver state at stage boundaries. Validate workspace design against these constraints during P3 review. See sddp-algorithm.md §3.4."
  - date: 2026-02-23
    description: "P3 review — full rewrite. Fix review_priority (2→3). Strip all Rust code (8 blocks, ~480 lines) and replace with behavioral descriptions and tables. Rewrite §1 workspace structure as behavioral description with memory budget table. Rewrite §1.2 NUMA initialization as requirements. Rewrite §1.3 warm-start as behavioral contract. Rewrite §1.4 workspace manager as behavioral description. Condense §1.6 SolverPool comparison to brief note. Rewrite §2.2 scaling data as requirements table. Remove §2.5 FlatBuffers schema (premature given open scaling decision in solver-abstraction.md §3). Add CLP cross-reference. Add stage template (CSC) reference. Align LpProblem framing with adopted stage template architecture."
  - date: 2026-02-23
    description: "Review feedback: (1) Replace single cached basis with per-stage basis cache (T slots). (2) Drop adjacent-stage warm-start heuristic — only exact stage match or cold start. (3) Update all sizing estimates with production dimensions from lp_sizing.py (8360 cols, 80628 rows, 120 stages). (4) Clarify solution buffers as single-solve scratch space. (5) Add basis lifecycle paragraph (in-memory cache vs FlatBuffers checkpoint). (6) Add open point §1.9 — structural LP homogeneity across stages for performance optimization."
  - date: 2026-02-23
    description: "Review feedback: (7) Add §1.10 open point — cut loading cost analysis with production-scale estimates (2.5:1 loading-to-solving ratio in forward pass), two-level storage consideration for cuts (coherent with basis lifecycle in §1.2), and 4 deferred mitigation strategies. Cross-referenced from solver-abstraction.md §11.2."
  - date: 2026-02-23
    description: "Review fixes: (1) §2.5 rewritten — replaced parallel 7-step workflow with augmentation table mapping scaling operations onto §1.4 steps; added scaling cross-reference note to §1.4. (2) §2.4 formula notation fixed (asterisk-based → unicode β̃). (3) §2.3 row_scale clarified with multiplier convention and D_r/D_c diagonal matrix explanation. (4) §2.3 scaling factors storage clarified — computed once per stage template, stored alongside template as shared read-only data. (5) §2.5 stale diagram placeholder removed."
---

# Solver Workspaces & LP Scaling

## Purpose

This spec defines the thread-local solver workspace infrastructure and the LP scaling specification. These components bridge the [Solver Abstraction Layer](./solver-abstraction.md) with the HPC execution layer, ensuring each OpenMP thread has an exclusive, NUMA-local solver instance with pre-allocated buffers for the SDDP hot path. For solver-specific implementations, see [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md).

## 1. Thread-Local Solver Infrastructure

### 1.1 Design Rationale

LP solvers (HiGHS, CLP, CPLEX) are **not thread-safe**. Each OpenMP thread requires its own solver instance. The workspace pattern provides:

1. **Exclusive ownership** — Each thread owns one solver instance for the entire SDDP run (no pool-based borrow/return). This eliminates synchronization overhead and enables optimal warm-starting.
2. **Per-stage basis cache** — The workspace stores one basis per stage. The basis from solving stage `t` in iteration `i` is reused to warm-start stage `t` in iteration `i+1`, without external basis save/restore. For checkpoint/restart, the cache is serialized to FlatBuffers (see §1.2).
3. **Pre-allocated buffers** — All solution extraction, RHS patching, and basis storage buffers are allocated once at initialization and reused for every solve, eliminating hot-path allocations.
4. **NUMA-local memory** — Each workspace is allocated on the NUMA node local to its owning thread, maximizing memory bandwidth during LP solves.

### 1.2 Workspace Contents

Each thread-local workspace contains the following components:

| Component             | Description                                                                                                                                                                                                                                                                                                   | Sizing (production: 8,360 cols, 80,628 rows)      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Solver instance       | Opaque solver handle (`void*` for HiGHS, `Clp_Simplex*` for CLP). Persists for the entire run. Includes LP matrix storage, working arrays (pricing, steepest edge weights), and LU factorization workspace. Does **not** include the LP data itself — that is loaded per stage via `passModel`/`loadProblem`. | ~15 MB (see breakdown below)                      |
| RHS patch buffer      | Pre-allocated arrays for batch bound modification during scenario patching (row indices + lower/upper values). Sized to the maximum scenario-dependent value count across stages (~2,240 values at production scale).                                                                                         | ~54 KB                                            |
| Primal buffer         | Pre-allocated array for primal solution extraction. These are **single-solve scratch buffers** — the SDDP algorithm copies the values it needs (state variables, generation, etc.) after each solve. The buffer is overwritten on the next solve.                                                             | 8,360 × 8 = ~65 KB                                |
| Dual buffer           | Pre-allocated array for dual solution extraction. Same single-solve semantics as primal buffer.                                                                                                                                                                                                               | 80,628 × 8 = ~630 KB                              |
| Reduced cost buffer   | Pre-allocated array for reduced cost extraction. Same single-solve semantics as primal buffer.                                                                                                                                                                                                                | 8,360 × 8 = ~65 KB                                |
| Per-stage basis cache | Array of `T` basis slots (one per stage). Each slot stores the basis from the last successful solve at that stage, used for warm-starting the same stage in the next iteration. See §1.5.                                                                                                                     | T × (num_cols + num_rows) × 1–4 bytes (see below) |
| Solve statistics      | Accumulated per-thread counters (see §1.6).                                                                                                                                                                                                                                                                   | ~64 bytes                                         |
| NUMA node ID          | Which NUMA node this workspace's memory is allocated on.                                                                                                                                                                                                                                                      | scalar                                            |

**Cache line padding**: Workspaces should be padded to cache line boundaries (64 bytes) to prevent false sharing between adjacent workspaces in an array.

**Solver instance breakdown** (production scale, from [HiGHS §5](./solver-highs-impl.md) and [CLP §6](./solver-clp-impl.md)):

| Sub-component     |       Size | Notes                                               |
| ----------------- | ---------: | --------------------------------------------------- |
| LP matrix storage |      ~5 MB | nnz × 16 bytes (value + index, CSC format)          |
| Working arrays    |      ~2 MB | (rows + cols) × 3 × 8 bytes (pricing, edge weights) |
| LU factorization  |   ~5–10 MB | Varies with sparsity; dominates solver memory       |
| Internal metadata |      ~1 MB | Iteration buffers, scaling arrays, options          |
| **Total**         | **~15 MB** | Consistent across HiGHS and CLP                     |

**Per-stage basis sizing** (production: T = 120 stages, 88,988 status entries per basis):

| Solver | Status size | Per basis | 120 stages | Notes                           |
| ------ | ----------: | --------: | ---------: | ------------------------------- |
| CLP    |      1 byte |    ~87 KB |     ~10 MB | `unsigned char` per status code |
| HiGHS  |     4 bytes |   ~348 KB |     ~41 MB | `HighsInt` per status code      |

**Total memory footprint per workspace**:

| Component               | Per Thread (CLP) | Per Thread (HiGHS) | 16 Threads (HiGHS) |
| ----------------------- | ---------------: | -----------------: | -----------------: |
| Solver instance         |           ~15 MB |             ~15 MB |            ~240 MB |
| Solution buffers        |          ~815 KB |            ~815 KB |             ~13 MB |
| Per-stage basis cache   |           ~10 MB |             ~41 MB |            ~656 MB |
| **Total per workspace** |       **~26 MB** |         **~57 MB** |        **~912 MB** |

At 16 threads per rank (production configuration), the total workspace memory is under 1 GB per rank — well within acceptable bounds for production HPC nodes (256+ GB RAM). The per-stage basis cache is the dominant cost with HiGHS due to its 4-byte status codes.

> **Note**: All sizing estimates use production-scale dimensions from `lp_sizing.py` (160 hydros, 130 thermals, 120 stages, 15K cut capacity). See [Production Scale Reference §3](../00-overview/production-scale-reference.md) for the full configuration.

**Basis lifecycle**: The per-stage basis cache is the in-memory warm-start mechanism. For checkpoint/restart with reproducibility, the basis is serialized to FlatBuffers (`StageBasis` table in [Binary Formats §3.1](../02-data-model/binary-formats.md)) at checkpoint boundaries. On resume, the checkpoint basis is loaded into the per-stage cache, restoring warm-start capability. The in-memory cache and the on-disk checkpoint are two lifecycle stages of the same data — the cache is the hot-path runtime structure, the checkpoint is the durable persistence format.

### 1.3 NUMA-Aware Initialization

Workspace initialization follows the first-touch NUMA allocation policy:

1. Each workspace **must be created by the thread that will own it** — not by the main thread. This ensures all allocations (solver internals, buffers) land on the NUMA node local to the owning thread.
2. During initialization, each thread pins to its assigned NUMA node, creates its solver instance, and fills all buffers with zeros (first-touch).
3. After initialization, the workspace array is immutable in structure — threads access their workspace by thread ID with no synchronization.

**Initialization sequence** (within an OpenMP parallel region):

1. Each thread computes its NUMA node: `numa_node = thread_id / threads_per_numa`
2. Pin to NUMA node (NUMA bind guard)
3. Create solver instance via `Highs_create()` or `Clp_newModel()`
4. Configure solver with SDDP-tuned settings (see solver impl specs §4)
5. Allocate and zero-fill all buffers — solution buffers, RHS patch buffer, and per-stage basis cache (T slots, all initially empty) — ensuring first-touch on the local NUMA node
6. Store workspace in shared array at position `thread_id`

### 1.4 Stage Solve Workflow

Each workspace provides a unified solve entry point that handles the full stage solve lifecycle. This is the main interface used by the forward and backward pass:

**Solve sequence**:

1. **Load stage template** (if stage changed) — Bulk-load the pre-assembled CSC stage template ([Solver Abstraction §11](./solver-abstraction.md)) into the solver via `passModel`/`loadProblem`.
2. **Add active cuts** (if stage changed) — Batch-add active cuts from the cut pool via `addRows` in CSR format.
3. **Patch scenario values** — Write scenario-dependent values (incoming storage, AR lag fixing, noise fixing) into the RHS patch buffer, then apply via batch bound modification.
4. **Set basis** (if warm-starting) — Look up the cached basis for the target stage from the per-stage basis cache. If a basis exists for this stage, apply it to the solver.
5. **Solve** — Call the solver. Retry logic is encapsulated within the solver implementation ([Solver Abstraction §7](./solver-abstraction.md)).
6. **Extract solution** — Copy primal values, dual values, and reduced costs into pre-allocated buffers. Extract the basis and store it in the per-stage cache at the solved stage's slot, overwriting any previous basis for that stage.
7. **Update statistics** — Increment solve count, iteration count, timing.

Steps 1–2 are skipped for **within-stage solves** (multiple backward pass scenarios at the same stage, or multiple forward passes at the same stage within a batch). Only steps 3–7 are needed, since the structural LP and cuts are identical.

When LP scaling is active, steps 1–3 and 6 are augmented with scaling transformations. See §2.5 for the mapping.

### 1.5 Warm-Start Eligibility

Each workspace maintains a per-stage basis cache — an array of `T` basis slots, one per stage. The basis produced by solving stage `t` in iteration `i` is stored in slot `t` and reused to warm-start stage `t` in iteration `i+1`. This provides high-quality warm-starts because only the scenario-dependent RHS values change between iterations at the same stage; the structural LP and cut set are identical or nearly identical (new cuts may have been added).

| Condition                                  | Strategy   | Rationale                                                                                |
| ------------------------------------------ | ---------- | ---------------------------------------------------------------------------------------- |
| Basis exists in cache for the target stage | Warm-start | Basis from the previous iteration at this stage is near-optimal — only RHS changed.      |
| No cached basis for the target stage       | Cold start | First iteration, or basis was invalidated by a solver error. Solver starts from scratch. |

On solver error, the basis for the affected stage is invalidated (slot set to empty). The next solve at that stage will use cold start. Other stages' cached bases are unaffected.

**Backward pass reuse**: During the backward pass, a thread solves multiple scenarios at the same stage sequentially. The basis from the first scenario solve at stage `t` is stored in slot `t` and reused for subsequent scenarios at the same stage within the same backward pass. Since only the RHS changes between scenarios, this provides excellent warm-starts within a single backward sweep.

### 1.6 Solve Statistics

Each workspace accumulates per-thread statistics that are aggregated after the parallel region:

| Counter            | Description                                              |
| ------------------ | -------------------------------------------------------- |
| Total solves       | Number of LP solves performed by this thread             |
| Warm starts        | Number of solves that used a cached basis                |
| Cold starts        | Number of solves without a cached basis                  |
| Retries            | Number of solves that required retry escalation          |
| Simplex iterations | Total simplex iterations across all solves               |
| Total solve time   | Cumulative wall-clock time spent in solver calls         |
| Max solve time     | Maximum single-solve wall-clock time (outlier detection) |

Aggregation across threads uses simple summation for all counters except max solve time (which uses the maximum across threads).

### 1.7 Workspace Manager

A workspace manager owns the array of all thread-local workspaces within an MPI rank:

- Creates one workspace per OpenMP thread at initialization (within a parallel region for first-touch NUMA).
- Provides indexed access by thread ID — `workspace[thread_id]`. No locking required since each thread accesses only its own workspace.
- Provides aggregated statistics across all workspaces (called after parallel regions for logging).
- The workspace array is structurally immutable after initialization — the number of workspaces equals the number of OpenMP threads and does not change.

> **Note on SolverPool alternative**: A pool-based pattern (borrow/return from a shared pool of N < threads solver instances) was considered and rejected. The persistent ownership pattern is preferred for SDDP because: (1) warm-starting requires per-stage basis affinity between iterations, which pools break; (2) the memory overhead of one workspace per thread (~57 MB × 16 threads ≈ 912 MB per rank) is acceptable on HPC nodes; (3) pool synchronization would add overhead on the hot path.

### 1.8 Thread Safety Invariants

The workspace design relies on clear ownership boundaries between shared and thread-local data:

| Data                     | Ownership    | Access During Forward Pass   | Access During Backward Pass      |
| ------------------------ | ------------ | ---------------------------- | -------------------------------- |
| Stage LP templates (CSC) | Shared       | Read-only (immutable)        | Read-only (immutable)            |
| Cut pool coefficients    | Shared       | Read-only (immutable in fwd) | Write by single thread per stage |
| Cut pool activity bitmap | Shared       | Read-only                    | Write by single thread per stage |
| Solver instance          | Thread-local | Exclusive write              | Exclusive write                  |
| RHS patch buffer         | Thread-local | Exclusive write              | Exclusive write                  |
| Solution buffers         | Thread-local | Exclusive write              | Exclusive write                  |
| Per-stage basis cache    | Thread-local | Exclusive read/write         | Exclusive read/write             |

The key invariant: **no locking is required on the hot path**. Stage templates are immutable. The cut pool is only modified during the backward pass, which is sequential per stage with a synchronization barrier at each stage boundary ([Training Loop §4](./training-loop.md)). Each solver instance is exclusively owned by one thread.

### 1.9 Open Point — Structural LP Homogeneity

> **Open point — structural LP homogeneity across stages**: A potential performance optimization is to make all stage LPs structurally identical — same number of variables and constraints — even when system elements are not yet operative or have been decommissioned. Elements that are inactive at a given stage would be represented with zero bounds on their generation, exchange, and flow variables, preserving the LP structure. This would enable:
>
> - **Cheaper stage transitions**: The same CSC template structure can be reused across stages, with only coefficient/bound patches instead of different structural templates.
> - **Better basis reuse**: A basis from one stage could potentially warm-start a structurally identical stage, since the basis dimensions match exactly.
> - **Deeper solver integration**: With a fixed LP structure, the solver's factorization workspace can be sized once and reused without reallocation.
>
> **Trade-off**: This approach could reduce modeling flexibility (adding/removing system elements requires restructuring all stages) and may increase LP size at stages where many elements are inactive. It should be evaluated only if profiling shows that stage-template switching is a significant cost. Deferred until performance data is available.

### 1.10 Open Point — Cut Loading Cost and Two-Level Storage

> **Open point — cut loading dominates stage transitions**: Under Option A (full LP rebuild per stage transition), the cut loading cost via `addRows` is significantly larger than the basis warm-start cost that §1.2 already addresses with a two-level storage design (in-memory cache + FlatBuffers checkpoint). The cut data dwarfs the basis data:
>
> | Data                     | Per-stage size (production)      | Frequency              |
> | ------------------------ | -------------------------------- | ---------------------- |
> | Basis (HiGHS)            | ~348 KB                          | Every solve            |
> | Active cuts (worst case) | 15,000 × ~25 KB/row ≈ 238–366 MB | Every stage transition |
>
> At production scale, cut loading in the forward pass takes ~5 ms per stage transition (memory bandwidth + solver `addRows` overhead), while the LP solve itself targets ~2 ms with warm-start — a **2.5:1 ratio of loading to solving**. Over 120 stages per forward pass, this amounts to ~600 ms for cut loading vs ~240 ms for solving.
>
> **Two-level storage consideration**: The basis lifecycle (§1.2) uses two storage levels — a hot-path in-memory per-stage cache (mutable, NUMA-local, updated every solve) and a cold-path FlatBuffers checkpoint (immutable, on-disk, written at checkpoint boundaries). The cuts, being a much larger bottleneck, warrant the same coherent treatment. A hot-path in-memory cut representation optimized for `addRows` (pre-assembled CSR blocks per stage, shared read-only across threads) could reduce the per-stage-transition assembly cost. The cold-path FlatBuffers cut persistence already exists (see [Binary Formats §3](../02-data-model/binary-formats.md)).
>
> However, a per-stage in-memory cut cache sized for all threads is constrained by RAM: 120 stages × 238 MB = ~28 GB per rank for the cut pool alone, which is the current shared cut pool design. The question is whether the cut pool's memory layout can be structured so that the data passed to `addRows` requires minimal or zero intermediate assembly — essentially making the `addRows` call use pre-built CSR arrays directly from the shared cut pool.
>
> **Mitigation strategies** (all deferred until profiling):
>
> 1. **Pre-assembled CSR cut blocks** — Structure the cut pool so that active cuts for each stage are stored in a contiguous CSR-ready layout. The `addRows` call passes pointers directly into the shared cut pool without intermediate copies. The solver still copies internally, but one copy layer is eliminated.
> 2. **CLP cloning** ([CLP Implementation](./solver-clp-impl.md)) — `makeBaseModel()`/`setToBaseModel()` could restore a pre-loaded LP (template + cuts) rather than rebuilding from scratch. But this requires per-stage per-thread base models (~250 MB × 120 stages per thread), which is infeasible at full scale.
> 3. **Incremental cut updates** — If structural LP homogeneity (§1.9) is adopted, a thread transitioning between stages could potentially keep the LP loaded and only patch the cut differences (remove old stage's cuts, add new stage's cuts) rather than full rebuild. This interacts with bound toggling (rejected in [Decision 4](./SOLVER_ARCHITECTURE_DECISIONS.md)) and would need a different mechanism.
> 4. **Reduced active cut count** — In early iterations, the active cut set is much smaller than the worst case. Cut selection strategies ([Cut Management §4](../01-math/cut-management.md)) that aggressively prune inactive cuts reduce the per-stage `addRows` data volume.
>
> This analysis does not change the Option A baseline decision, but it identifies cut loading as the dominant cost in stage transitions and motivates the solver-specific optimizations anticipated in [Solver Abstraction §11.3](./solver-abstraction.md).

## 2. LP Scaling Specification

### 2.1 Motivation

Production SDDP LPs often suffer from numerical ill-conditioning due to:

- Variables spanning wide magnitudes (e.g., storage in hm³ at 10⁴ vs generation slacks at 10⁻²)
- Constraints with mixed coefficient magnitudes (water balance coefficients vs penalty bounds)
- The future cost variable θ dominating other variables in magnitude

LP scaling transforms the problem to improve solver numerical stability by normalizing coefficient magnitudes.

> **Open point — scaling strategy**: The choice between single-phase scaling (compute once from template) and two-phase scaling (re-scale after cuts accumulate) is deferred until profiling. See [Solver Abstraction §3](./solver-abstraction.md) for the full discussion. This section defines the scaling mechanics that apply regardless of the chosen strategy.

### 2.2 Scaling Transformation

Given original LP: min c'x s.t. Ax ≤ b, x ≥ 0

Apply diagonal scaling matrices D_c (column/variable scaling) and D_r (row/constraint scaling):

| Quantity  | Original | Scaled            |
| --------- | -------- | ----------------- |
| Variables | x        | x̃ = D_c⁻¹ x       |
| Objective | c        | c̃ = D_c × c       |
| Matrix    | A        | Ã = D_r × A × D_c |
| RHS       | b        | b̃ = D_r × b       |

**Solution unscaling** (convert solver output back to physical units):

| Quantity      | Scaled → Physical |
| ------------- | ----------------- |
| Primal values | x = D_c × x̃       |
| Row duals     | π = D_r × π̃       |
| Reduced costs | rc = D_c⁻¹ × rc̃   |

### 2.3 Scaling Data Requirements

The scaling factors are computed once per stage from the stage template and stored alongside the template as shared, read-only data. They are not duplicated per workspace — all threads within a rank reference the same scaling factors for a given stage. The factors persist for the entire run:

| Field     | Type      | Description                                                                                                                                        |
| --------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| stage_id  | integer   | Which stage these factors apply to                                                                                                                 |
| col_scale | float64[] | Column scaling factors (one per variable). Multiplier convention: x_physical = col_scale[j] × x̃_scaled.                                            |
| row_scale | float64[] | Row scaling factors (one per constraint). Multiplier convention: row_scale[i] is applied to both sides of constraint i (Ã_i = row_scale[i] × A_i). |
| method    | enum      | Scaling method used: None, GeometricMean, Equilibration, SolverAuto                                                                                |
| applied   | boolean   | Whether scaling was actually applied (false if problem was already well-conditioned)                                                               |

Scaling factors are chosen so that the scaled matrix Ã = D_r × A × D_c has coefficient magnitudes closer to unity. Both D_r and D_c are diagonal matrices with row_scale and col_scale on their diagonals respectively.

**Scaling methods**:

| Method        | Description                                                | When to Use                          |
| ------------- | ---------------------------------------------------------- | ------------------------------------ |
| None          | No scaling applied                                         | Well-conditioned problems            |
| GeometricMean | Iterative geometric mean normalization of rows and columns | Recommended default for SDDP         |
| Equilibration | Scale so max abs(a_ij) = 1 in each row and column          | Alternative to geometric mean        |
| SolverAuto    | Delegate to solver's internal scaling                      | When POWE.RS does not manage scaling |

**Diagnostic metrics** (optional, for monitoring scaling quality):

| Metric          | Description                                         |
| --------------- | --------------------------------------------------- |
| max_coef_before | Maximum absolute coefficient before scaling         |
| min_coef_before | Minimum absolute nonzero coefficient before scaling |
| ratio_before    | max/min ratio before scaling (condition indicator)  |
| max_coef_after  | Maximum absolute coefficient after scaling          |
| min_coef_after  | Minimum absolute nonzero coefficient after scaling  |
| ratio_after     | max/min ratio after scaling (improvement indicator) |

### 2.4 Scaling Impact on Cut Coefficients

Cut coefficients must be stored in **physical (unscaled) units** for solver-agnostic portability. Scaling transformations are applied at solve time and immediately reversed.

**Transformation for cut coefficients**:

If the solver operates in scaled space and produces scaled duals π̃, the physical cut coefficients are:

β_physical = W' × D_r × π̃

where W is the technology matrix linking state variables to constraints.

When adding a cut to a scaled LP, the physical cut coefficients must be transformed to scaled space:

β̃[j] = β_physical[j] × col_scale[j]

This transformation is applied during the `addRows` step (§1.4 step 2) and reversed when extracting solution duals (§1.4 step 6).

**Key invariant**: The cut pool always stores cuts in physical units. Scaling is a transient transformation applied only within the solver workspace.

### 2.5 Scaling Integration with Stage Solve Workflow

When scaling is active, the stage solve workflow (§1.4) is augmented at specific steps. The following table maps each scaling operation to the §1.4 step it modifies:

| §1.4 Step                | Scaling Augmentation                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Load stage template   | After loading, apply pre-computed scaling factors (D_r, D_c) to transform the LP into scaled space.                                        |
| 2. Add active cuts       | Transform cut coefficients from physical to scaled space (§2.4) before passing to `addRows`.                                               |
| 3. Patch scenario values | RHS values are patched in scaled space: multiply each bound by the corresponding row_scale[i].                                             |
| 4–5. Set basis, Solve    | No change — the basis is scale-independent (status codes, not values), and the solver operates entirely in scaled space.                   |
| 6. Extract solution      | After extracting raw solver output, unscale: primal ×= D_c, duals ×= D_r, reduced costs ×= D_c⁻¹. Cut coefficients use the unscaled duals. |
| 7. Update statistics     | No change.                                                                                                                                 |

**Interaction with solver-internal scaling**: If POWE.RS manages its own scaling (augmentations above), solver-internal scaling should be disabled to avoid double-scaling. If POWE.RS delegates scaling to the solver (`SolverAuto` method), all scaling augmentations are skipped — the solver handles scaling internally. See the solver-specific scaling configuration in [HiGHS Implementation §4.2](./solver-highs-impl.md) and [CLP Implementation §4.2](./solver-clp-impl.md).

## Cross-References

- [Solver Abstraction](./solver-abstraction.md) — Interface contract (§4), LP layout convention (§2), cut pool design (§5), stage LP templates in CSC form (§11), scaling open point (§3)
- [HiGHS Implementation](./solver-highs-impl.md) — HiGHS-specific solver configuration, batch bound operations, scaling options
- [CLP Implementation](./solver-clp-impl.md) — CLP-specific solver configuration, mutable pointer patching, scaling options
- [LP Formulation](../01-math/lp-formulation.md) — Constraint structure that defines LP dimensions and row/column layout
- [Cut Management](../01-math/cut-management.md) — Cut generation algorithms that produce coefficients stored in physical units
- [Training Loop](./training-loop.md) — Forward/backward pass orchestration driving the stage solve workflow (§1.4)
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — OpenMP threading model requiring thread-local solver workspaces
- [Memory Architecture](../04-hpc/memory-architecture.md) — NUMA topology and first-touch allocation policy for workspace buffers
- [Binary Formats](../02-data-model/binary-formats.md) — Cut pool CSR layout (§3.4), LP rebuild analysis (§A)
- [Configuration Reference](../05-config/configuration-reference.md) — Solver configuration parameters (tolerances, scaling method selection)
