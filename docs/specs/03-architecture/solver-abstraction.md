---
status: approved
review_priority: 3-medium
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.4 (5.4.1-5.4.8)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from DATA_MODEL_SPECIFICATION.md §5.4.1-5.4.8"
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): LP rebuild cost is a critical performance constraint. Cannot keep all stage LPs with full cut sets in memory simultaneously — must rebuild LPs and add cuts when transitioning between stages. The design must minimize this rebuild cost through strategies such as cut preallocation, basis persistence, and incremental constraint updates. Validate that the pre-allocated cut slot design and basis storage adequately address this during P3 review. See sddp-algorithm.md §3.4."
  - date: 2026-02-23
    description: "P3 review — first rewrite: fix review_priority (2→3). Strip all Rust code (8 blocks, ~350 lines) and replace with behavioral descriptions and tables. Rewrite §3 Core Solver Interface as behavioral contract. Rewrite §5 Error Types as categorized table with SDDP-level actions. Rewrite §6 Retry Logic — behavioral contract only. Rewrite §7 Dual Normalization — keep sign convention, strip Rust. Rewrite §8 Basis Storage — behavioral description. Rewrite §9 Compile-Time Selection — strip Cargo.toml/cfg detail."
  - date: 2026-02-23
    description: "P3 review — architectural decisions (SOLVER_ARCHITECTURE_DECISIONS.md). Five decisions adopted: (1) Pre-assembled stage LP templates — structural LP assembled once per stage at initialization in CSR form, loaded via bulk passModel/loadProblem at each stage transition. (2) LP layout convention — state variables contiguous at column start, state-linking constraints at row top, cuts at row bottom. (3) Option A baseline with solver-specific optimizations anticipated — CLP clone via C++ wrapper, mutable pointer access, pre-assembled CSR. (4) Cut preallocation applies to cut pool only, not solver LP — bound toggling removed, only active cuts added via addRows. (5) Dual-solver validation — interface designed against both HiGHS and CLP as first-class reference implementations. Rewrote §1, §2, §4, §10, §11. Added new §2 LP Layout Convention. Updated §9 for dual-solver. Added CLP Implementation cross-reference."
  - date: 2026-02-23
    description: "P3 review — refinements from second review round. (1) §2.2 Row Layout: expanded top region from state-linking only to all cut-relevant constraints — FPHA hyperplanes and generic volume constraints whose duals contribute to cut coefficients are now included; renamed range to n_cut_relevant; added sub-region table. (2) §3 Scaling: added open point on single-phase vs two-phase scaling strategy — deferred until profiling. (3) §5.4 Cut Loading: added open question on selective active-only addRows vs bulk load with bound deactivation — deferred until benchmarking."
  - date: 2026-02-23
    description: "CSC correction — Stage LP templates changed from CSR to CSC (column-major) form. Investigation of HiGHS source (Highs.cpp:353) and CLP source (ClpModel.cpp:334) revealed both solvers internally store LP matrices in column-major format and transpose CSR→CSC on every passModel/loadProblem call. Updated §1 (design rationale), §3 (hierarchy diagram), §4.1 (interface contract table), §11.1 (template description and array names), §11.2 (rebuild step 1), §11.3 (optimization table). Stage templates now store CSC arrays (col_starts, row_indices, values). Cut addition via addRows remains CSR — this is the native format for row-wise insertion in both solvers."
---

# Solver Abstraction Layer

## Purpose

This spec defines the multi-solver abstraction layer: the unified interface through which the SDDP algorithm interacts with LP solvers, including the solver interface contract, LP layout convention, cut pool design, error categories, retry logic contract, dual normalization, basis storage, compile-time solver selection, stage LP template strategy, and solver-specific optimization paths.

For thread-local solver infrastructure and LP scaling, see [Solver Workspaces & LP Scaling](./solver-workspaces.md). For solver-specific implementations, see [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md).

## 1. Design Rationale

The SDDP algorithm must be solver-agnostic. The solver interface abstracts LP solver details behind a unified contract, designed and validated against both HiGHS and CLP to avoid single-solver bias.

Key design decisions:

1. **Dual-solver validation** — The interface is designed against both HiGHS (C API) and CLP (C API + thin C++ wrappers) as first-class reference implementations. Every operation in the interface contract (§3) must be naturally expressible through both solver APIs. This prevents designing an interface that maps cleanly to one solver but awkwardly to another.
2. **Compile-time solver selection** — The active solver is selected at compile time to avoid virtual dispatch overhead on the hot path (millions of LP solves)
3. **Encapsulated retry logic** — Each solver handles its own numerical difficulties internally; the SDDP algorithm only sees success or failure
4. **Cuts stored in physical units** — Scaling transformations are applied at solve time, not stored in the cut pool
5. **Pre-assembled stage templates** — Each stage's structural LP is assembled once at initialization in solver-ready CSC form (both HiGHS and CLP use column-major internally), minimizing per-iteration rebuild cost

## 2. LP Layout Convention

The LP row and column ordering is standardized to enable performance optimizations across the hot path: slice-based state transfer, fast RHS patching, efficient dual extraction for cut coefficients, and favorable basis persistence.

### 2.1 Column Layout (Variables)

State variables are placed contiguously at the beginning of the variable vector:

| Column Range                     | Contents                                                      | Rationale                                        |
| -------------------------------- | ------------------------------------------------------------- | ------------------------------------------------ |
| `[0, n_hydro)`                   | Storage volumes $v_h$                                         | Contiguous for slice-based state transfer        |
| `[n_hydro, n_hydro + n_ar_lags)` | AR inflow lags $a_{h,\ell}$                                   | Contiguous with storage for complete state slice |
| `[n_state, n_state + 1)`         | Future cost variable $\theta$                                 | Single variable, fixed position                  |
| `[n_state + 1, n_cols)`          | All other decision variables (generation, flow, slacks, etc.) | Order follows LP formulation convention          |

**Key benefit**: The complete state vector (storage + AR lags) occupies columns `[0, n_state)`. Transferring state from one stage to the next is a single contiguous slice copy — no gathering from scattered column positions.

### 2.2 Row Layout (Constraints)

Constraints are ordered in three regions:

| Region     | Row Range                                      | Contents                                                                                                      | Lifecycle                                            |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Top**    | `[0, n_cut_relevant)`                          | All constraints whose duals contribute to cut coefficient computation (see below)                             | Static structure per stage; RHS patched per scenario |
| **Middle** | `[n_cut_relevant, n_structural)`               | All other structural constraints: load balance, generation/flow bounds, penalty bounds, remaining constraints | Static per stage                                     |
| **Bottom** | `[n_structural, n_structural + n_active_cuts)` | Benders cuts (active only)                                                                                    | Rebuilt each iteration via batch `addRows`           |

**Top region — cut-relevant constraints (contiguous for dual extraction):**

The top region groups all constraints whose dual multipliers are needed for Benders cut coefficient computation. Placing them contiguously enables a single slice read from the dual vector rather than gathering from scattered positions. The constraints in this region include:

| Sub-region                 | Contents                                                                                 | Duals Used For                                                                                                                   |
| -------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| State-linking              | Water balance (hydro storage), AR lag fixing                                             | Cut coefficients for state variables ($\pi^{wb}_h$, $\pi^{lag}_{h,\ell}$)                                                        |
| FPHA hyperplanes           | Piecewise-linear production function constraints (when FPHA model is active for a hydro) | Cut coefficients involving volume-dependent production (duals propagate storage marginal value through production approximation) |
| Generic volume constraints | User-defined constraints that reference storage volume as a variable                     | Cut coefficients for any constraint that creates a shadow price on state variables                                               |

The exact sub-region boundaries within the top region depend on the stage's active constraints (e.g., FPHA rows only exist for stages where hydros use the FPHA production model). The key invariant is: **all rows whose duals are needed for cut computation occupy a contiguous prefix of the constraint vector**. This must be validated against the [LP Formulation](../01-math/lp-formulation.md) during implementation — any constraint type whose dual feeds into the cut coefficient formula must be placed in this region.

**Key benefits**:

- **Cut-relevant constraints at top** — All duals needed for cut coefficient computation are in the first `n_cut_relevant` rows. Extracting them is a contiguous slice read from the dual vector — no index gathering required.
- **Cuts at bottom** — Everything above the cut boundary is identical across iterations within a stage. A cached basis from the previous iteration applies directly to the structural portion (rows `[0, n_structural)`). Only the new cut rows need their basis status initialized.
- **Structural middle region** — All non-cut-relevant, non-cut constraints. Their ordering within this region follows the [LP Formulation](../01-math/lp-formulation.md) convention.

### 2.3 Interaction with Basis Persistence

When warm-starting from a cached basis after adding new cuts:

1. The basis status codes for rows `[0, n_structural)` are reused directly from the cached basis
2. New cut rows (appended at the bottom) are initialized with status **Basic** — meaning the slack variable for each new cut constraint is in the basis. The solver will price these rows in and pivot as needed during the solve.
3. If cuts were removed since the cached basis was saved, the basis is truncated to match the new row count

This is possible precisely because cuts are at the bottom — the structural portion of the basis is position-stable across iterations.

## 3. Abstraction Hierarchy

The solver abstraction separates concerns into four layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SOLVER ABSTRACTION HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Stage LP Template (Data Holder)                    │  │
│  │  - Pre-assembled structural LP in CSC form (built once per stage)     │  │
│  │  - Solver-agnostic problem representation                             │  │
│  │  - Row/column layout follows §2 convention                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      LP Scaling (Pre-processing)                      │  │
│  │  - Row/column scaling factors for numerical stability                 │  │
│  │  - Transforms problem ↔ scaled space                                  │  │
│  │  - Persisted for cut coefficient computation                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       LP Solver (Execution)                           │  │
│  │  - Solve → success or error                                           │  │
│  │  - Encapsulates retry logic, warm-start, basis management             │  │
│  │  - Compile-time selected (HiGHS or CLP as reference impls)            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     LP Solution (Result Data)                         │  │
│  │  - Primal values, dual values (constraint & variable)                 │  │
│  │  - Objective value, solve status                                      │  │
│  │  - Basis information (optional, for warm-starting)                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Open point — scaling strategy**: Two approaches are under consideration:
>
> - **Single-phase scaling** — Compute row/column scaling factors once from the stage template and apply them for the entire run. Simpler, deterministic, but may not adapt to numerical conditioning changes as cuts accumulate.
> - **Two-phase scaling** — Apply an initial scaling from the template, then optionally re-scale after cuts are added if numerical diagnostics indicate poor conditioning. More adaptive, but adds complexity (scaling factors must be composed, cut coefficients may need re-transformation).
>
> The trade-offs depend on the practical conditioning behavior of SDDP LPs as cuts accumulate. This decision is deferred until profiling with realistic problem instances reveals whether single-phase scaling is sufficient or re-scaling is needed. See [Solver Workspaces](./solver-workspaces.md) for current scaling design.

**Stage Template vs Transient Solver State:**

| Aspect    | Stage LP Template (Static Data)                      | Solver State (Transient)      |
| --------- | ---------------------------------------------------- | ----------------------------- |
| Lifetime  | Built once at initialization, persists for algorithm | Per-solve invocation          |
| Ownership | Shared read-only across threads                      | Thread-local                  |
| Contents  | CSC matrix, bounds, objective, coefficients          | Basis factors, working arrays |
| Memory    | One per stage, pre-assembled once                    | Created/reused per solve      |

This separation enables the key optimization: template data is read-only and shared, while each thread maintains its own solver workspace for warm-starting. See [Solver Workspaces](./solver-workspaces.md) for the thread-local workspace design.

## 4. Solver Interface Contract

The solver interface defines the behavioral contract that all solver implementations must satisfy. The SDDP algorithm interacts with solvers exclusively through this interface.

### 4.1 Required Operations

| Operation        | Input                             | Output                    | Description                                                                                                          |
| ---------------- | --------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Load model       | Stage LP template (CSC)           | —                         | Bulk-load the pre-assembled structural LP into the solver via `passModel`/`loadProblem`. Fast memcpy-like operation. |
| Add cut rows     | Active cuts in CSR format (batch) | —                         | Batch-add all active Benders cuts via a single `addRows` call. Cuts are appended at the bottom per §2.2.             |
| Patch RHS/bounds | Set of (index, value) pairs       | —                         | Update scenario-dependent values (inflows, state fixing constraints) without structural LP changes.                  |
| Solve            | —                                 | Solution or error         | Solve the loaded LP. Handles retries internally, extracts solution with normalized duals.                            |
| Solve with basis | Basis from prior solve            | Solution or error         | Set basis for warm-start, then solve. Reduces simplex iterations.                                                    |
| Reset            | —                                 | —                         | Clear all internal solver state (caches, basis, factorization).                                                      |
| Get basis        | —                                 | Current basis             | Extract the current simplex basis for caching.                                                                       |
| Statistics       | —                                 | Accumulated solve metrics | Total solves, total iterations, retries, failures, timing.                                                           |

### 4.2 Contract Guarantees

- **Thread safety** — Each solver instance is exclusively owned by one thread. No concurrent access.
- **Retry encapsulation** — The SDDP algorithm never sees retry attempts. It calls solve and receives either a valid solution or a terminal error (see §6 and §7).
- **Dual normalization** — All returned dual values use the canonical sign convention (see §8). Solver-specific sign differences are handled internally.
- **Solver identification** — Each implementation exposes a name string for logging and diagnostics.

### 4.3 Dual-Solver Validation

Every operation in the interface contract above must be verifiable against both reference solver APIs:

| Operation           | HiGHS C API                       | CLP C API / C++                                               |
| ------------------- | --------------------------------- | ------------------------------------------------------------- |
| Load model          | `Highs_passModel`                 | `Clp_loadProblem`                                             |
| Add cut rows        | `Highs_addRows` (CSR batch)       | `Clp_addRows` (CSR batch)                                     |
| Patch RHS           | `Highs_changeRowsBoundsBySet`     | Mutable `double*` via `Clp_rowLower()`/`Clp_rowUpper()`       |
| Patch column bounds | `Highs_changeColsBoundsBySet`     | Mutable `double*` via `Clp_columnLower()`/`Clp_columnUpper()` |
| Solve               | `Highs_run`                       | `Clp_dual` (warm-start) or `Clp_initialSolve` (cold)          |
| Set/get basis       | `Highs_setBasis`/`Highs_getBasis` | `Clp_copyinStatus`/`Clp_statusArray`                          |
| Reset               | `Highs_clearSolver`               | Reconstruct or `Clp_initialSolve`                             |

See [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md) for solver-specific details.

## 5. Cut Pool Design

> **Placeholder** — The cut storage layout diagram (`../../diagrams/exports/svg/data/cut-storage-layout.svg`) will be revised after the text review is complete.

### 5.1 Scope Clarification: Cut Pool vs Solver LP

Two distinct concepts must be clearly separated:

| Concept       | What                                                                                | Where                                      | Preallocation                                                                       |
| ------------- | ----------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Cut pool**  | In-memory shared data structure holding all cuts (active + inactive) for all stages | Shared across threads, one per MPI rank    | Yes — deterministic slot assignment, activity bitmap, contiguous dense coefficients |
| **Solver LP** | Transient LP loaded into a thread-local solver instance for a single solve          | Thread-local, rebuilt per stage transition | No — only active cuts are added via `addRows`                                       |

Under the adopted LP rebuild strategy (§10), the solver LP is transient — it is constructed, used, and discarded at every stage transition. Pre-allocating inactive cut rows in the solver LP would add memory pressure and solver overhead (larger factorization) with no benefit. Only the cut pool uses preallocation.

### 5.2 Cut Pool Preallocation

The cut pool pre-allocates capacity for all cuts that could be generated during the entire training run:

**Capacity formula:**

```
capacity = warm_start_cuts + max_iterations × forward_passes_per_iteration
```

**Example** (production-scale):

| Parameter                       | Value                             |
| ------------------------------- | --------------------------------- |
| Warm-start cuts                 | 5,000                             |
| Max iterations × forward passes | 50 × 200 = 10,000                 |
| **Total capacity per stage**    | **15,000**                        |
| State dimension                 | 2,080                             |
| Memory per stage                | 15,000 × 2,080 × 8 bytes ≈ 238 MB |
| Total for 120 stages            | ~28 GB per rank                   |

**Deterministic slot assignment:** Cut slots are computed, not allocated at runtime. The slot for a new cut is:

```
slot = warm_start_count + iteration × forward_passes + forward_pass_index
```

This is a pure function with no side effects — the same inputs always produce the same slot. This eliminates thread-safety concerns and non-determinism.

An activity bitmap tracks which slots are currently active. The count of active slots is maintained alongside the bitmap to avoid scanning.

### 5.3 Mathematical Basis

A Benders cut has the form: $\theta \geq \alpha + \boldsymbol{\beta}' \mathbf{x}$, where $\theta$ is the future cost variable, $\alpha$ is the cut intercept, $\boldsymbol{\beta}$ is the vector of cut coefficients, and $\mathbf{x}$ is the state vector. Rearranging to standard LP row form: $\theta - \boldsymbol{\beta}' \mathbf{x} \geq \alpha$.

For details on cut generation and selection, see [Cut Management](../01-math/cut-management.md). For the in-memory layout requirements (contiguous dense coefficients, CSR-friendly, cache-aligned), see [Binary Formats §3.4](../02-data-model/binary-formats.md).

### 5.4 How Active Cuts Enter the Solver LP

> **Open question — selective vs bulk cut loading**: Two strategies are under consideration for loading cuts into the solver LP at each stage transition:
>
> - **Selective addition (current baseline)** — Only active cuts are added via `addRows`. Inactive cuts are excluded entirely. The solver sees a smaller LP (fewer rows, smaller factorization). This is the natural fit for Option A rebuild, where the LP is transient anyway.
> - **Bulk load with bound deactivation** — All cuts (active + inactive) are loaded in a single bulk operation. Inactive cuts are "deactivated" by setting their lower bound to $-\infty$ (making the constraint non-binding). The solver's presolve step eliminates these redundant rows before factorization. This approach simplifies the cut loading logic (one bulk call, no bitmap filtering) and may interact favorably with solver-internal optimizations, but increases the nominal LP size and depends on presolve being effective.
>
> The choice depends on: (1) whether presolve overhead for redundant rows is negligible compared to the bitmap-filtered CSR assembly cost, (2) whether presolve is even enabled in our configuration (currently disabled for warm-start compatibility), and (3) the ratio of active to total cuts in practice. This decision is deferred until benchmarking with realistic cut pool sizes.

The current baseline approach loads active cuts via a single batch `addRows` call in CSR format:

1. Query the cut pool activity bitmap for the current stage
2. Assemble CSR arrays from the active cuts: `row_starts`, `column_indices`, `coefficient_values`, `row_lower_bounds` (intercepts $\alpha$), `row_upper_bounds` ($+\infty$)
3. Call the solver's batch row-addition API once (e.g., `Highs_addRows`, `Clp_addRows`)

Since all cuts have dense coefficients (every state variable participates), the `column_indices` array is a repeating pattern `[0, 1, ..., n_state, θ_col]` that can be precomputed once and reused.

**Cut deactivation**: Under the rebuild strategy, cuts are not deactivated in the solver LP — they simply aren't included in the next rebuild's `addRows` call. Deactivation happens in the cut pool (the activity bitmap is updated), and the next LP rebuild naturally excludes deactivated cuts.

## 6. Error Categories

The solver interface returns a categorized error when a solve fails. The SDDP algorithm uses the error category to determine its response. Solver-internal errors (e.g., factorization failures) are resolved by the retry logic (§7) before reaching this level.

| Error Category       | Meaning                                     | SDDP Response                                               | May Have Partial Solution |
| -------------------- | ------------------------------------------- | ----------------------------------------------------------- | :-----------------------: |
| Infeasible           | No feasible solution exists                 | Data error — check bounds, constraints. Hard stop.          |            No             |
| Unbounded            | Objective is unbounded below                | Modeling error — check objective signs. Hard stop.          |            No             |
| Numerical difficulty | Solver retries exhausted without resolution | Log warning; may proceed with partial solution if available |            Yes            |
| Time limit exceeded  | Per-solve time budget exhausted             | Log warning; may proceed with best solution found           |            Yes            |
| Iteration limit      | Solver iteration limit hit                  | Log warning; may proceed with best solution found           |            Yes            |
| Internal error       | Unrecoverable solver-internal failure       | Log and hard stop                                           |            No             |

**Infeasibility diagnostics**: When available, the solver provides an infeasibility ray (proof of infeasibility) or unbounded direction for debugging.

**Numerical recovery hints**: When a numerical difficulty error is returned, the solver may suggest recovery actions: apply scaling, try cold start, tighten tolerances, or report that the problem is inherently ill-conditioned.

## 7. Retry Logic Contract

Retry logic is **encapsulated within each solver implementation**. The SDDP algorithm never sees retry details — it only receives the final result.

### 7.1 Behavioral Contract

Each solver implementation defines its own retry sequence based on its failure modes. The retry logic must satisfy these behavioral requirements:

| Requirement           | Description                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| Maximum attempts      | A configurable upper bound on retry attempts (default: 5)                                         |
| Time budget           | A configurable wall-clock budget for all attempts combined                                        |
| Escalating strategies | Retries should escalate from least-disruptive (clear basis) to most-disruptive (switch algorithm) |
| Final disposition     | After exhausting all retries, return a terminal error with the best partial solution if available |
| Logging               | Each retry attempt is logged at debug level with the strategy used and outcome                    |

### 7.2 Typical Retry Escalation

The following is the general pattern; solver-specific details are in the respective implementation specs:

1. Clear warm-start basis (cold start)
2. Disable presolve
3. Switch to alternative algorithm (e.g., interior point instead of simplex)
4. Relax solver tolerances

For solver-specific retry strategies, see [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md).

## 8. Dual Variable Normalization

Different LP solvers report dual multipliers with different sign conventions. The solver implementation **must** normalize duals to the canonical form before returning to the SDDP algorithm.

**Canonical sign convention:**

| Constraint Form                | Positive Dual Means                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| $\mathbf{a}'\mathbf{x} \leq b$ | Increasing RHS $b$ increases objective (shadow price $\partial z^*/\partial b > 0$) |
| $\mathbf{a}'\mathbf{x} \geq b$ | Sign flipped relative to $\leq$ form                                                |
| $\mathbf{a}'\mathbf{x} = b$    | Unrestricted sign                                                                   |

This normalization is critical for correct cut coefficient computation. A sign error in duals produces cuts that point in the wrong direction, leading to divergence of the algorithm.

## 9. Basis Storage for Warm-Starting

The solver interface supports saving and restoring a simplex basis for warm-starting subsequent solves. The basis captures which variables and constraints are basic (in the basis), at lower bound, at upper bound, free, or fixed.

**Basis status values:**

| Status   | Meaning                                   |
| -------- | ----------------------------------------- |
| At lower | Variable is at its lower bound            |
| Basic    | Variable is in the basis (between bounds) |
| At upper | Variable is at its upper bound            |
| Free     | Variable is free (superbasic)             |
| Fixed    | Variable is fixed (lower = upper)         |

A basis consists of one status value per column (variable) and one per row (constraint).

**Key design decisions:**

- Bases are stored in the **original problem space** (not presolved), ensuring portability across solver versions and presolve strategies
- The forward pass basis at each stage is retained for warm-starting the backward pass at the same stage (see [Training Loop §4.4](./training-loop.md))
- Basis storage uses compact representation (one byte per variable/constraint)
- Basis structure splits naturally at the cut boundary per §2.3: structural rows are position-stable, cut rows are appended/truncated

## 10. Compile-Time Solver Selection

The active solver implementation is selected at compile time via Cargo feature flags. Only one solver is active per build. This design:

- **Eliminates virtual dispatch** — The concrete solver type is known at compile time, enabling inlining and monomorphization
- **Simplifies the hot path** — No trait object indirection on the millions-of-solves critical path
- **Supports multiple backends** — HiGHS and CLP (open-source reference implementations), CPLEX and Gurobi (commercial) can be selected without code changes

HiGHS and CLP are first-class reference implementations — the solver abstraction is designed and tested against both. Commercial solvers (CPLEX, Gurobi) are compile-time selectable but are not primary validation targets.

## 11. Stage LP Template and Rebuild Strategy

### 11.1 Pre-Assembled Stage LP Templates

At initialization, each stage's structural LP (matrix, bounds, objective — everything except cuts and scenario-dependent RHS) is assembled once into a canonical CSC (column-major) representation called the **stage template**. This template:

- Is built from the resolved internal structures (see [Internal Structures](../02-data-model/internal-structures.md)) with full clarity and correctness — this is the initialization phase where data clarity matters more than performance
- Follows the column and row layout convention defined in §2
- Stores the CSC arrays (`col_starts`, `row_indices`, `values`, `col_lower`, `col_upper`, `row_lower`, `row_upper`, `objective`) in solver-ready format — CSC is used because both reference solvers (HiGHS and CLP) internally store LP matrices in column-major format; passing CSC avoids per-stage-transition transposition overhead
- Is shared read-only across all threads within an MPI rank

### 11.2 LP Rebuild Strategy

**Adopted decision: Option A — full rebuild per stage transition, using pre-assembled templates.**

Memory constraints prevent keeping all stage LPs with their full cut sets resident simultaneously. At each stage transition, each thread:

1. **Load template** — Bulk-load the pre-assembled stage template into the solver via `passModel`/`loadProblem`. Since the template is already in CSC form (the native internal format for both HiGHS and CLP), this is a fast bulk memory operation — no per-constraint construction loops and no format transposition.
2. **Add active cuts** — Batch-add all active cuts from the cut pool via a single `addRows` call in CSR format (see §5.4).
3. **Patch scenario values** — Update the ~2,240 scenario-dependent values (incoming storage as RHS of water balance, AR lag fixing constraint RHS, current-stage noise fixing) via batch bound modification.
4. **Warm-start** — Apply the cached basis from the previous iteration's solve at this stage (structural rows reused directly, new cut rows set to Basic per §2.3).
5. **Solve** — Solve the LP.

See [Binary Formats §3, §A](../02-data-model/binary-formats.md) for the full analysis, memory estimates, and solver API survey that informed the Option A decision.

### 11.3 Solver-Specific Optimization Paths

Option A is the **portable baseline** that works identically across all solver backends. The spec explicitly anticipates solver-specific optimizations that could reduce rebuild cost further:

| Optimization                 | Solver | Mechanism                                                                                                         | Status                                                       |
| ---------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Pre-assembled CSC templates  | All    | Stage template in solver-ready CSC → fast `passModel`/`loadProblem` with no format transposition                  | **Adopted** (§11.1)                                          |
| Direct memory patching       | CLP    | Mutable `double*` pointers (`Clp_rowLower()`, `Clp_columnLower()`) for zero-copy in-place RHS patching            | Anticipated — see [CLP Implementation](./solver-clp-impl.md) |
| LP template cloning          | CLP    | C++ `makeBaseModel()`/`setToBaseModel()` or copy constructor via thin C wrapper                                   | Anticipated — could reduce rebuild to clone + cut addition   |
| Pre-assembled CSR cut blocks | All    | Cut pool layout (binary-formats §3.4) is CSR-friendly → `addRows` uses pre-built data (CSR is native for addRows) | Anticipated                                                  |

These optimizations do not change the interface contract (§4) — they are internal to each solver implementation.

### 11.4 Within-Stage Incremental Updates

Within the same stage (e.g., multiple backward pass scenarios at stage $t$, or multiple forward passes at the same stage within a batch), only scenario-dependent values change — the structural LP and cuts are identical. In this case, the solver skips steps 1–2 and only performs:

- **Patch scenario values** (step 3)
- **Solve** with implicit warm-start from the previous solve's basis (step 5)

This is significantly faster than a full rebuild and is the common case in the backward pass.

## Cross-References

- [Solver Architecture Decisions](./SOLVER_ARCHITECTURE_DECISIONS.md) — Full discussion and rationale for the 5 architectural decisions adopted in this spec
- [Solver Workspaces & LP Scaling](./solver-workspaces.md) — Thread-local solver infrastructure and LP scaling specification
- [HiGHS Implementation](./solver-highs-impl.md) — HiGHS-specific implementation: retry strategy, batch operations, memory footprint
- [CLP Implementation](./solver-clp-impl.md) — CLP-specific implementation: C++ wrapper strategy, mutable pointer access, cloning optimization path
- [LP Formulation](../01-math/lp-formulation.md) — Constraint structure that the solver operates on
- [Cut Management](../01-math/cut-management.md) — How cuts are generated; this spec handles how they are stored in the cut pool and loaded into the solver LP
- [Training Loop](./training-loop.md) — Forward pass (parallel solve) and backward pass (cut addition) that drive solver invocations
- [Binary Formats](../02-data-model/binary-formats.md) — FlatBuffers schema for cut persistence, cut pool memory layout (§3.4), LP rebuild strategy analysis (§3, §A)
- [Internal Structures](../02-data-model/internal-structures.md) — Logical in-memory data model from which stage templates are built
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — OpenMP threading model that requires thread-local solvers
- [Memory Architecture](../04-hpc/memory-architecture.md) — NUMA-aware allocation for solver workspaces
- [Configuration Reference](../05-config/configuration-reference.md) — Solver configuration parameters
