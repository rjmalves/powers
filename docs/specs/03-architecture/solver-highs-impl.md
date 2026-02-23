---
status: approved
review_priority: 3-medium
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.4.10 (HiGHS Implementation Guidelines)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from DATA_MODEL_SPECIFICATION.md §5.4.10"
  - date: 2026-02-23
    description: "P3 review — full rewrite. Fix review_priority (2→3). Strip all Rust code (~350 lines, 8 blocks) and replace with behavioral descriptions and tables. Remove §4 (warm-starting with bound-toggled cuts) per Decision 4 — bound toggling removed from solver interface. Remove §8 (thread-local batch buffers) — implementation detail moved to solver-workspaces.md. Restructure to match solver-clp-impl.md structure: §1 Architecture Alignment, §2 Solver Interface Mapping, §3 Retry Strategy, §4 Configuration, §5 Memory Footprint, §6 HiGHS-Specific Considerations. Add complete C API function mapping for all interface operations. Add status code tables. Add infeasibility diagnostics. Add format flexibility note (supports both CSR and CSC via a_format parameter). Align all cross-references with solver architecture decisions."
  - date: 2026-02-23
    description: "CSC correction — Investigation of HiGHS source (Highs.cpp:353) and CLP source (ClpModel.cpp:334) revealed both solvers internally store LP matrices in column-major (CSC) format. HiGHS transposes CSR→CSC inside passModel via ensureColwise(); CLP transposes via reverseOrderedCopyOf(). Updated §1 table, §2.1, §6.1, §6.5 to reflect that stage templates should be stored in CSC form to avoid per-stage-transition transposition overhead. Removed incorrect claims that CSR could be passed directly 'with no transposition needed'."
---

# HiGHS Implementation

## Purpose

This spec provides implementation guidance specific to HiGHS integration as the first open-source LP solver reference implementation for POWE.RS. It complements the [Solver Abstraction Layer](./solver-abstraction.md) with HiGHS-specific patterns for the C API, batch bound operations for scenario patching, retry strategy, basis management, memory footprint, and SDDP-tuned configuration. For thread-local workspace management, see [Solver Workspaces](./solver-workspaces.md).

HiGHS and CLP are both **first-class reference implementations** — the solver abstraction interface is designed and validated against both (see [Solver Architecture Decisions](./SOLVER_ARCHITECTURE_DECISIONS.md), Decision 5).

## 1. Architecture Alignment

| POWE.RS Concept   | HiGHS Equivalent                                               | Notes                                             |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| Stage LP template | CSC arrays passed to `Highs_passLp`                            | CSC preferred — HiGHS internally converts CSR→CSC |
| Solver instance   | `void*` (opaque handle from `Highs_create`)                    | One per thread, persists for entire run           |
| Basis             | `HighsInt[]` via `Highs_setBasis` / `Highs_getBasis`           | Separate column and row status arrays             |
| Solution          | `Highs_getSolution` (col_value, col_dual, row_value, row_dual) | Copied into caller-provided arrays                |
| Solve             | `Highs_run(highs)`                                             | Single entry point for all solve types            |
| Objective value   | `Highs_getObjectiveValue(highs)`                               | Scalar return                                     |
| Iteration count   | `Highs_getSimplexIterationCount(highs)`                        | Scalar return                                     |

### 1.1 HiGHS API Layer

HiGHS exposes a pure C API through `Highs_c_api.h` that is directly callable from Rust via `extern "C"` FFI. Unlike CLP, there is no need for a C++ wrapper — all required operations are available through the C API.

| Layer                       | API                                          | Access From Rust          | Capabilities                                                                                |
| --------------------------- | -------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------- |
| **C API** (`Highs_c_api.h`) | Pure C functions operating on opaque `void*` | Direct FFI (`extern "C"`) | Load, solve, add rows, batch bound changes, basis, tolerances, options, solution extraction |

## 2. Solver Interface Mapping

This section maps each operation from the [Solver Abstraction §4](./solver-abstraction.md) to the specific HiGHS API calls.

### 2.1 Load Model

`Highs_passLp` loads a complete LP in a single call:

```
Highs_passLp(highs, num_col, num_row, num_nz,
             a_format, sense, offset,
             col_cost, col_lower, col_upper,
             row_lower, row_upper,
             a_start, a_index, a_value)
```

**Format flexibility**: Unlike CLP (which requires column-major only), HiGHS accepts both formats via the `a_format` parameter:

| `a_format` value                | Format             | Description                        |
| ------------------------------- | ------------------ | ---------------------------------- |
| `kHighsMatrixFormatColwise` (1) | Column-major (CSC) | Column starts, row indices, values |
| `kHighsMatrixFormatRowwise` (2) | Row-major (CSR)    | Row starts, column indices, values |

Since the stage LP templates ([Solver Abstraction §11](./solver-abstraction.md)) store the structural matrix in CSC form, the HiGHS implementation passes them with `a_format = kHighsMatrixFormatColwise`. This avoids per-stage-transition transposition overhead — HiGHS internally stores the LP matrix in column-major (CSC) format, so passing CSC directly skips the internal `ensureColwise()` transposition that would occur if CSR were passed.

### 2.2 Add Cut Rows

`Highs_addRows` adds multiple rows in a single batch call:

```
Highs_addRows(highs, num_new_row,
              lower, upper,
              num_new_nz, starts, index, value)
```

The `starts`/`index`/`value` arrays follow the standard CSR (row-major) format. This matches the cut pool's CSR-friendly storage layout (see [Binary Formats §3.4](../02-data-model/binary-formats.md)).

### 2.3 Patch Scenario-Dependent Values

HiGHS provides batch bound modification via index-set APIs. Unlike CLP's mutable pointer access (zero-copy direct writes), HiGHS requires function calls for each batch of updates.

| Operation           | HiGHS C API                                                              | Access Pattern                         |
| ------------------- | ------------------------------------------------------------------------ | -------------------------------------- |
| Patch row bounds    | `Highs_changeRowsBoundsBySet(highs, num_set_entries, set, lower, upper)` | Batch: one call for all row updates    |
| Patch column bounds | `Highs_changeColsBoundsBySet(highs, num_set_entries, set, lower, upper)` | Batch: one call for all column updates |

**Performance implication**: Scenario patching (the ~2,240 RHS updates per solve — incoming storage, AR lag fixing, noise fixing) should be assembled into a single batch call rather than individual per-row calls. The index array (`set`) identifies which rows to update; the `lower`/`upper` arrays provide the new bounds. Pre-allocating these batch buffers in the thread-local workspace (see [Solver Workspaces](./solver-workspaces.md)) eliminates allocation overhead on the hot path.

The LP layout convention ([Solver Abstraction §2](./solver-abstraction.md)) places state-linking constraints at the top, so the row indices for scenario patching form a contiguous range starting at 0 — enabling efficient sequential index generation.

**Comparison with CLP**: CLP's mutable `double*` pointers enable zero-copy in-place writes with no function call overhead per element. HiGHS's batch API is slightly higher overhead but provides bounds checking and maintains solver-internal invariants. For the ~2,240 updates typical in SDDP, the difference is expected to be negligible relative to solve time.

### 2.4 Solve

HiGHS uses a single entry point for all solve operations:

| Scenario      | HiGHS Call         | When Used                                                                          |
| ------------- | ------------------ | ---------------------------------------------------------------------------------- |
| **Any solve** | `Highs_run(highs)` | All solves — warm-start, cold-start. HiGHS automatically uses basis if one is set. |

HiGHS determines its solve strategy internally based on:

- Whether a basis has been set (via `Highs_setBasis`) → warm-start
- The configured algorithm (`simplex_strategy` option) → dual or primal simplex
- Whether presolve is enabled → full preprocessing or direct solve

**Status interpretation** after solve:

| `Highs_getModelStatus(highs)` | Constant                                 | Maps To (Solver Abstraction §6)          |
| ----------------------------- | ---------------------------------------- | ---------------------------------------- |
| 7                             | `kHighsModelStatusOptimal`               | Success                                  |
| 8                             | `kHighsModelStatusInfeasible`            | `Infeasible`                             |
| 9                             | `kHighsModelStatusUnboundedOrInfeasible` | `Infeasible` or `Unbounded` (ambiguous)  |
| 10                            | `kHighsModelStatusUnbounded`             | `Unbounded`                              |
| 13                            | `kHighsModelStatusTimeLimit`             | `TimeLimitExceeded`                      |
| 14                            | `kHighsModelStatusIterationLimit`        | `IterationLimit`                         |
| 4                             | `kHighsModelStatusSolveError`            | `NumericalDifficulty` or `InternalError` |
| 15                            | `kHighsModelStatusUnknown`               | `NumericalDifficulty`                    |

For status 9 (`UnboundedOrInfeasible`), use `Highs_getDualRay` and `Highs_getPrimalRay` to disambiguate.

### 2.5 Solution Extraction

HiGHS copies solution values into caller-provided arrays:

| Data                | HiGHS Call                                                           | Array Size                 |
| ------------------- | -------------------------------------------------------------------- | -------------------------- |
| Primal + dual (all) | `Highs_getSolution(highs, col_value, col_dual, row_value, row_dual)` | `num_col` + `num_row` each |
| Objective value     | `Highs_getObjectiveValue(highs)`                                     | scalar                     |
| Simplex iterations  | `Highs_getSimplexIterationCount(highs)`                              | scalar                     |

**Key difference from CLP**: HiGHS copies values into pre-allocated caller-owned arrays, while CLP returns mutable pointers into solver-internal memory. The HiGHS approach is safer (no pointer invalidation risk) but requires pre-allocated buffers. These buffers should be part of the thread-local workspace (see [Solver Workspaces](./solver-workspaces.md)).

**Dual normalization**: HiGHS's dual sign convention must be verified against the canonical sign convention defined in [Solver Abstraction §8](./solver-abstraction.md). If HiGHS reports duals with a different sign for $\geq$ constraints, the HiGHS implementation must negate the appropriate dual values before returning them to the SDDP algorithm. This is a critical correctness requirement — sign errors in duals produce divergent cuts.

### 2.6 Basis Management

HiGHS uses separate arrays for column and row basis status:

```
Highs_setBasis(highs, col_status, row_status)
Highs_getBasis(highs, col_status, row_status)
```

Both arrays are `HighsInt[]` (one integer per variable/constraint).

**Status codes**:

| Code | Constant                    | Meaning        | Maps To (Solver Abstraction §9) |
| ---- | --------------------------- | -------------- | ------------------------------- |
| 0    | `kHighsBasisStatusLower`    | At lower bound | At lower                        |
| 1    | `kHighsBasisStatusBasic`    | Basic          | Basic                           |
| 2    | `kHighsBasisStatusUpper`    | At upper bound | At upper                        |
| 3    | `kHighsBasisStatusZero`     | Free (zero)    | Free                            |
| 4    | `kHighsBasisStatusNonbasic` | Nonbasic       | At lower (default nonbasic)     |

**Key difference from CLP**: HiGHS uses separate column and row status arrays (`HighsInt[]`), while CLP uses a combined `unsigned char[]` array (`status[0..numcols-1]` = columns, `status[numcols..numcols+numrows-1]` = rows). The HiGHS representation is more natural for the basis management described in [Solver Abstraction §2.3](./solver-abstraction.md).

**Interaction with LP layout convention**: Per [Solver Abstraction §2.3](./solver-abstraction.md), the structural portion of the basis is position-stable across iterations. When warm-starting after adding cuts, the implementation:

1. Prepares the column status array (unchanged from cached basis)
2. Prepares the row status array: cached structural rows + new cut rows set to `kHighsBasisStatusBasic` (1)
3. Calls `Highs_setBasis` with both arrays
4. Calls `Highs_run` for warm-start solve

### 2.7 Reset

`Highs_clearSolver(highs)` clears all solver state (basis, factorization, solution) while keeping the model loaded. This is useful for retry strategies that need to discard the current basis without reloading the LP.

For a full reset (discard everything): `Highs_destroy(highs)` + `Highs_create()`. This is only needed at shutdown or in extreme recovery scenarios.

### 2.8 Infeasibility Diagnostics

| Diagnostic                       | HiGHS Call                                                     | Notes                                                                   |
| -------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Dual ray (infeasibility proof)   | `Highs_getDualRay(highs, &has_dual_ray, dual_ray_value)`       | `has_dual_ray` indicates availability. Array pre-allocated by caller.   |
| Primal ray (unboundedness proof) | `Highs_getPrimalRay(highs, &has_primal_ray, primal_ray_value)` | `has_primal_ray` indicates availability. Array pre-allocated by caller. |

## 3. Retry Strategy

The retry strategy follows the behavioral contract defined in [Solver Abstraction §7](./solver-abstraction.md). HiGHS-specific retry escalation:

| Attempt | Strategy         | HiGHS Actions                                                                                                             |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 1       | Clear basis      | `Highs_clearSolver(highs)` — discard cached basis, re-factorize from scratch on next `Highs_run`                          |
| 2       | Enable presolve  | `Highs_setStringOptionValue(highs, "presolve", "on")` — presolve may simplify a degenerate problem                        |
| 3       | Switch to primal | `Highs_setIntOptionValue(highs, "simplex_strategy", 4→1)` — primal simplex may handle different degeneracy patterns       |
| 4       | Relax tolerances | `Highs_setDoubleOptionValue(highs, "primal_feasibility_tolerance", 1e-6)` + dual tolerance — looser than default          |
| 5       | Switch to IPM    | `Highs_setStringOptionValue(highs, "solver", "ipm")` — interior point method, completely different algorithm, last resort |

After each successful retry, the implementation restores default settings for the next solve. After all retries are exhausted, return a terminal error with the best partial solution if available (call `Highs_getSolution` — HiGHS may have a feasible but suboptimal solution).

## 4. Configuration

### 4.1 SDDP-Tuned Settings

| Setting                    | HiGHS Option                              | Value        | Rationale                                                                                                 |
| -------------------------- | ----------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------- |
| Solver algorithm           | `"solver"` → `"simplex"`                  | `"simplex"`  | Simplex required for basis warm-starting                                                                  |
| Simplex strategy           | `"simplex_strategy"` → `4`                | 4 (dual)     | Dual simplex is the standard for SDDP — cut addition modifies RHS, which is a bound change in dual        |
| Presolve                   | `"presolve"` → `"off"`                    | `"off"`      | Disabled for warm-start compatibility; see open point in [Solver Abstraction §3](./solver-abstraction.md) |
| Parallel                   | `"parallel"` → `"off"`                    | `"off"`      | Each solver instance is single-threaded (thread safety via exclusive ownership)                           |
| Output                     | `"output_flag"` → `0`                     | 0 (off)      | Quiet for production; millions of solves per run                                                          |
| Primal tolerance           | `"primal_feasibility_tolerance"` → `1e-7` | 1e-7         | Match CLP defaults for cross-solver reproducibility                                                       |
| Dual tolerance             | `"dual_feasibility_tolerance"` → `1e-7`   | 1e-7         | Match CLP defaults                                                                                        |
| Max iterations (per solve) | `"simplex_iteration_limit"` → value       | Configurable | Per-solve iteration limit; prevents runaway solves                                                        |
| Max time (per solve)       | `"time_limit"` → value                    | Configurable | Per-solve time limit                                                                                      |

### 4.2 Scaling Note

HiGHS manages scaling internally. The relevant options are:

| Option                     | Values                            | Description                          |
| -------------------------- | --------------------------------- | ------------------------------------ |
| `"simplex_scale_strategy"` | 0 (off), 1-4 (various strategies) | Controls simplex scaling; 0 disables |

The scaling strategy interacts with the open point in [Solver Abstraction §3](./solver-abstraction.md) (single-phase vs two-phase scaling). If POWE.RS manages its own scaling (applied at the template level), HiGHS internal scaling should be set to 0 to avoid double-scaling. If POWE.RS delegates scaling to the solver, HiGHS's default strategy is a reasonable starting point.

## 5. Memory Footprint

| Component              | Formula                     | Example (1120 states, 15K cuts) |
| ---------------------- | --------------------------- | ------------------------------- |
| LP matrix storage      | nnz × 16 bytes              | ~5 MB                           |
| Working arrays         | (rows + cols) × 3 × 8 bytes | ~1 MB                           |
| Basis storage          | (rows + cols) × 4 bytes     | ~200 KB                         |
| Factor storage         | Varies with sparsity        | ~5–10 MB                        |
| **Total per instance** |                             | **~15 MB**                      |
| **192 threads**        |                             | **~2.9 GB**                     |

This is within acceptable bounds for production HPC nodes (256+ GB RAM). The memory footprint is comparable to CLP (~15 MB without cloning).

**Note on basis storage**: HiGHS uses `HighsInt` (4 bytes) per basis status entry, vs CLP's `unsigned char` (1 byte). This increases basis storage by ~4x but is negligible relative to the LP matrix and factorization.

## 6. HiGHS-Specific Considerations

### 6.1 Format Flexibility

HiGHS's `Highs_passLp` accepts both CSR (`kHighsMatrixFormatRowwise = 2`) and CSC (`kHighsMatrixFormatColwise = 1`) formats. However, HiGHS internally stores the LP matrix in column-major (CSC) format — when CSR is passed, `Highs_passLp` internally calls `ensureColwise()` to transpose it (see `Highs.cpp:353`). Since stage LP templates are stored in CSC form, the HiGHS implementation passes them directly with `a_format = kHighsMatrixFormatColwise`, avoiding this per-stage-transition transposition overhead. Both solvers (HiGHS and CLP) use CSC internally, so using CSC templates is the common format that works natively with both.

`Highs_addRows` for cut addition uses CSR format (row starts, column indices, values), which is the same for both solvers.

### 6.2 Dual Sign Convention

HiGHS's dual sign convention must be verified against the canonical sign convention defined in [Solver Abstraction §8](./solver-abstraction.md). If HiGHS reports duals with a different sign for $\geq$ constraints, the HiGHS implementation must negate the appropriate dual values before returning them to the SDDD algorithm. This is a critical correctness requirement — sign errors in duals produce divergent cuts.

### 6.3 Thread Safety

Each `void*` HiGHS instance is **not thread-safe** — it must be exclusively owned by one thread. This aligns with the solver abstraction's thread-safety requirement ([Solver Abstraction §4.2](./solver-abstraction.md)). Each OpenMP thread creates its own instance via `Highs_create()` at initialization and destroys it at shutdown via `Highs_destroy`.

### 6.4 Solution Copy Semantics

Unlike CLP (which returns mutable pointers into solver internals), HiGHS copies solution values into caller-provided arrays via `Highs_getSolution`. This is inherently safer — no pointer invalidation risk — but requires pre-allocated buffers in the thread-local workspace. The buffers are sized once at initialization (to `num_col` and `num_row` of the largest stage LP) and reused for all solves.

### 6.5 No Cloning Optimization Path

HiGHS does not expose LP template cloning through its C API (no equivalent of CLP's `makeBaseModel`/`setToBaseModel`). The Option A baseline (full `Highs_passLp` rebuild per stage transition) is the only available strategy. Since stage templates store the matrix in CSC form — HiGHS's native internal format — the `Highs_passLp` call is a fast bulk memory operation with no format conversion overhead.

## Cross-References

- [Solver Abstraction](./solver-abstraction.md) — Interface contract (§4), LP layout convention (§2), cut pool design (§5), error categories (§6), retry contract (§7), dual normalization (§8), basis storage (§9), stage templates (§11)
- [Solver Architecture Decisions](./SOLVER_ARCHITECTURE_DECISIONS.md) — Decision 5 (dual-solver validation) that established HiGHS as a first-class reference implementation
- [CLP Implementation](./solver-clp-impl.md) — Companion solver implementation for cross-validation
- [Solver Workspaces & LP Scaling](./solver-workspaces.md) — Thread-local workspace infrastructure that owns the HiGHS instance
- [LP Formulation](../01-math/lp-formulation.md) — Constraint structure that HiGHS operates on
- [Cut Management](../01-math/cut-management.md) — How cuts are generated; this spec handles how they are loaded via `Highs_addRows`
- [Training Loop](./training-loop.md) — Forward/backward pass orchestration driving solver invocations
- [Binary Formats](../02-data-model/binary-formats.md) — Cut pool CSR layout (§3.4), LP rebuild analysis (§A)
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — OpenMP threading model requiring one HiGHS instance per thread
- [Memory Architecture](../04-hpc/memory-architecture.md) — NUMA-aware allocation for solver workspaces
- [Configuration Reference](../05-config/configuration-reference.md) — Solver configuration parameters
