---
status: approved
review_priority: 3-medium
source_sections:
  - "New spec — created during P3 review per SOLVER_ARCHITECTURE_DECISIONS.md Decision 5"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-23
    description: "Created as companion to solver-highs-impl.md. CLP is a first-class reference implementation per Decision 5 (dual-solver validation). Covers C API baseline, mutable pointer optimization, C++ wrapper strategy for cloning, retry strategy, basis management, memory footprint, and configuration."
  - date: 2026-02-23
    description: "CSC correction — Updated §2.1 and §7.1 to reflect that stage LP templates now store structural matrices in CSC (column-major) form, which is CLP's native internal format. Removed text about CSR→CSC transposition at initialization — templates arrive in CLP's native format directly."
---

# CLP Implementation

## Purpose

This spec provides implementation guidance specific to CLP (Coin-OR Linear Programming) integration as the second open-source LP solver reference implementation for POWE.RS. It complements the [Solver Abstraction Layer](./solver-abstraction.md) with CLP-specific patterns for the C API baseline, mutable pointer optimization for scenario patching, the C++ wrapper strategy for LP template cloning, retry strategy, basis management, memory footprint, and SDDP-tuned configuration. For thread-local workspace management, see [Solver Workspaces](./solver-workspaces.md).

CLP and HiGHS are both **first-class reference implementations** — the solver abstraction interface is designed and validated against both (see [Solver Architecture Decisions](./SOLVER_ARCHITECTURE_DECISIONS.md), Decision 5).

## 1. Architecture Alignment

| POWE.RS Concept   | CLP Equivalent                                                          | Notes                                                  |
| ----------------- | ----------------------------------------------------------------------- | ------------------------------------------------------ |
| Stage LP template | CSC arrays passed to `Clp_loadProblem`                                  | CSC is CLP's native internal format — no transposition |
| Solver instance   | `Clp_Simplex*` (opaque handle from `Clp_newModel`)                      | One per thread, persists for entire run                |
| Basis             | `unsigned char[]` via `Clp_statusArray` / `Clp_copyinStatus`            | Combined row+column status in single array             |
| Solution          | `Clp_primalColumnSolution`, `Clp_dualRowSolution`, `Clp_objectiveValue` | Mutable `double*` pointers into solver internals       |
| Solve (warm)      | `Clp_dual(model, 0)`                                                    | Dual simplex with warm-start from current basis        |
| Solve (cold)      | `Clp_initialDualSolve(model)`                                           | Dual simplex with full initialization                  |
| LP cloning        | `ClpSimplex::makeBaseModel()` / `setToBaseModel()` (C++ only)           | Requires thin C wrapper — see §5                       |
| Raw C++ access    | `Clp_getClpSimplex(model)` → cast to `ClpSimplex*`                      | Escape hatch for C++ features not exposed in the C API |

### 1.1 CLP API Layers

CLP exposes two API layers relevant to POWE.RS:

| Layer                           | API                                                 | Access From Rust                                        | Capabilities                                                                                |
| ------------------------------- | --------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **C API** (`Clp_C_Interface.h`) | Pure C functions operating on opaque `Clp_Simplex*` | Direct FFI (`extern "C"`)                               | Load, solve, add rows, mutable pointer access, basis, tolerances, scaling                   |
| **C++ API** (`ClpSimplex.hpp`)  | Class methods on `ClpSimplex`                       | Via thin C wrapper functions compiled as C++ and linked | Template cloning (`makeBaseModel`/`setToBaseModel`), copy constructor, `setPersistenceFlag` |

The C API is sufficient for the Option A baseline. The C++ API enables the anticipated cloning optimization (§5). The bridge between them is `Clp_getClpSimplex(model)`, which returns the underlying `ClpSimplex*`.

## 2. Solver Interface Mapping

This section maps each operation from the [Solver Abstraction §4](./solver-abstraction.md) to the specific CLP API calls.

### 2.1 Load Model

`Clp_loadProblem` loads a complete LP in column-major sparse format:

```
Clp_loadProblem(model, numcols, numrows,
                start, index, value,       // CSR/CSC matrix
                collb, colub, obj,         // column bounds + objective
                rowlb, rowub)              // row bounds
```

**Format note**: `Clp_loadProblem` expects **column-major** (CSC) format (column starts, row indices, values). The stage LP templates (see [Solver Abstraction §11](./solver-abstraction.md)) store the structural matrix in CSC form — this is CLP's native internal format (CLP stores the LP matrix as a `ClpPackedMatrix` in column-ordered form). The CLP implementation passes the template CSC arrays directly to `Clp_loadProblem` with no format transposition needed.

### 2.2 Add Cut Rows

`Clp_addRows` adds multiple rows in a single batch call:

```
Clp_addRows(model, number, rowLower, rowUpper,
            rowStarts, columns, elements)
```

The `rowStarts`/`columns`/`elements` arrays follow the standard CSR (row-major) format. This is the natural format for cut addition since cut pool storage is already CSR-friendly (see [Binary Formats §3.4](../02-data-model/binary-formats.md)).

### 2.3 Patch Scenario-Dependent Values

CLP's mutable pointer access is a **key differentiator** from HiGHS. The C API functions `Clp_rowLower()`, `Clp_rowUpper()`, `Clp_columnLower()`, `Clp_columnUpper()`, and `Clp_objective()` return writable `double*` pointers directly into the solver's internal arrays.

| Operation          | CLP C API                               | Access Pattern                                   |
| ------------------ | --------------------------------------- | ------------------------------------------------ |
| Patch row lower    | `Clp_rowLower(model)[row_idx] = val`    | Direct memory write, zero function call overhead |
| Patch row upper    | `Clp_rowUpper(model)[row_idx] = val`    | Direct memory write                              |
| Patch column lower | `Clp_columnLower(model)[col_idx] = val` | Direct memory write                              |
| Patch column upper | `Clp_columnUpper(model)[col_idx] = val` | Direct memory write                              |
| Patch objective    | `Clp_objective(model)[col_idx] = val`   | Direct memory write                              |

**Performance implication**: Scenario patching (the ~2,240 RHS updates per solve — incoming storage, AR lag fixing, noise fixing) can be done as direct memory writes with no function call overhead per element. The LP layout convention ([Solver Abstraction §2](./solver-abstraction.md)) places state-linking constraints at the top, so these patches target a contiguous prefix of the row arrays — cache-friendly sequential writes.

Alternatively, CLP provides bulk array replacement via `Clp_chgRowLower(model, rowLower)` and `Clp_chgRowUpper(model, rowUpper)`, which copy entire arrays. This is only useful if the majority of values change — for SDDP scenario patching where only ~2,240 of potentially 10,000+ rows change, direct pointer writes to specific indices are more efficient.

**Safety note**: These mutable pointers are valid only while the solver model is unchanged (no `Clp_addRows`, `Clp_loadProblem`, etc.). After any structural modification, the pointers must be re-obtained. Within the SDDP hot path, structural modifications only occur at stage transitions (load template + add cuts), after which new pointers are obtained for the patching phase.

### 2.4 Solve

| Scenario              | CLP Call                         | When Used                                                                                                   |
| --------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Warm-start dual**   | `Clp_dual(model, 0)`             | Default for all solves after basis is set. The `ifValuesPass=0` parameter means use current basis directly. |
| **Cold-start dual**   | `Clp_initialDualSolve(model)`    | First solve after `Clp_loadProblem`, or after retry clears basis.                                           |
| **Cold-start primal** | `Clp_initialPrimalSolve(model)`  | Retry strategy fallback.                                                                                    |
| **Barrier**           | `Clp_initialBarrierSolve(model)` | Retry strategy last resort.                                                                                 |

**Status interpretation** after solve:

| `Clp_status(model)` | Meaning                    | Maps To (Solver Abstraction §6)          |
| ------------------- | -------------------------- | ---------------------------------------- |
| 0                   | Optimal                    | Success                                  |
| 1                   | Primal infeasible          | `Infeasible`                             |
| 2                   | Dual infeasible            | `Unbounded`                              |
| 3                   | Stopped on iterations/time | `IterationLimit` or `TimeLimitExceeded`  |
| 4                   | Stopped due to errors      | `NumericalDifficulty` or `InternalError` |

Use `Clp_secondaryStatus(model)` for further disambiguation when `Clp_status` returns 3 or 4. Use `Clp_isAbandoned(model)` to distinguish numerical difficulties from clean iteration/time limits.

### 2.5 Solution Extraction

CLP returns mutable `double*` pointers to its internal solution arrays:

| Data                    | CLP Call                          | Array Size |
| ----------------------- | --------------------------------- | ---------- |
| Primal values (columns) | `Clp_primalColumnSolution(model)` | `numcols`  |
| Dual values (rows)      | `Clp_dualRowSolution(model)`      | `numrows`  |
| Reduced costs (columns) | `Clp_dualColumnSolution(model)`   | `numcols`  |
| Objective value         | `Clp_objectiveValue(model)`       | scalar     |
| Simplex iterations      | `Clp_numberIterations(model)`     | scalar     |

Since these are pointers into solver-owned memory, the values must be copied out before any subsequent solver operation that could invalidate them (e.g., the next solve).

**Dual normalization**: CLP uses the convention where the dual row solution `Clp_dualRowSolution(model)` returns row duals that follow the same sign convention as the solver abstraction's canonical form ([Solver Abstraction §8](./solver-abstraction.md)). This must be verified empirically for $\leq$ vs $\geq$ constraint forms and documented in the implementation.

### 2.6 Basis Management

CLP uses a compact combined status array for both rows and columns:

```
unsigned char* status = Clp_statusArray(model);
// Layout: status[0..numcols-1] = column status, status[numcols..numcols+numrows-1] = row status
```

**Status codes** (2 bits used from each byte):

| Code | Meaning        | Maps To (Solver Abstraction §9) |
| ---- | -------------- | ------------------------------- |
| 0    | Free           | Free                            |
| 1    | Basic          | Basic                           |
| 2    | At upper bound | At upper                        |
| 3    | At lower bound | At lower                        |
| 4    | Superbasic     | Free (superbasic)               |
| 5    | Fixed          | Fixed                           |

**Setting basis**: `Clp_copyinStatus(model, statusArray)` copies a status array into the solver. The array must be sized `numcols + numrows`.

**Getting basis**: `Clp_statusArray(model)` returns a mutable pointer to the internal status array. Copy the values before any structural change.

**Interaction with LP layout convention**: Per [Solver Abstraction §2.3](./solver-abstraction.md), the structural portion of the basis (rows `[0, n_structural)`) is position-stable across iterations. When warm-starting after adding cuts, the implementation:

1. Copies the cached basis (truncated or extended to match the new row count)
2. Sets new cut rows to status `1` (Basic — slack in basis)
3. Calls `Clp_copyinStatus` with the assembled array
4. Calls `Clp_dual(model, 0)` for warm-start solve

### 2.7 Reset

CLP does not have a direct equivalent of HiGHS's `clearSolver`. Two approaches:

- **Reconstruct**: `Clp_deleteModel(model)` + `Clp_newModel()`. Clean but discards all state.
- **Re-solve cold**: Call `Clp_initialDualSolve(model)` which performs a full re-factorization from scratch, effectively resetting the solver's internal state while keeping the LP loaded.

For the retry strategy (§3), re-solving cold is preferred since it avoids reloading the LP.

### 2.8 Infeasibility Diagnostics

| Diagnostic        | CLP Call                      | Notes                                                              |
| ----------------- | ----------------------------- | ------------------------------------------------------------------ |
| Infeasibility ray | `Clp_infeasibilityRay(model)` | Returns `NULL` if unavailable. Caller must free via `Clp_freeRay`. |
| Unbounded ray     | `Clp_unboundedRay(model)`     | Returns `NULL` if unavailable. Caller must free via `Clp_freeRay`. |

## 3. Retry Strategy

The retry strategy follows the behavioral contract defined in [Solver Abstraction §7](./solver-abstraction.md). CLP-specific retry escalation:

| Attempt | Strategy            | CLP Actions                                                                                                |
| ------- | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1       | Cold dual restart   | Call `Clp_initialDualSolve(model)` instead of `Clp_dual` — discards cached basis and re-factorizes         |
| 2       | Switch to primal    | Call `Clp_initialPrimalSolve(model)` — primal simplex may handle different degeneracy patterns             |
| 3       | Adjust perturbation | `Clp_setPerturbation(model, 50)` to enable perturbation, then `Clp_dual(model, 0)` — helps with degeneracy |
| 4       | Relax tolerances    | `Clp_setPrimalTolerance(model, 1e-6)` + `Clp_setDualTolerance(model, 1e-6)` — looser than default          |
| 5       | Barrier             | `Clp_initialBarrierSolve(model)` — completely different algorithm, last resort                             |

After each successful retry, the implementation restores default settings for the next solve. After all retries are exhausted, return a terminal error with the best partial solution if available (check `Clp_primalColumnSolution` and `Clp_dualRowSolution` — they may contain useful values even after a failed solve).

## 4. Configuration

### 4.1 SDDP-Tuned Settings

| Setting          | CLP Call                                 | Value        | Rationale                                                                                                 |
| ---------------- | ---------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------- |
| Algorithm        | `Clp_setAlgorithm(model, -1)`            | -1 (dual)    | Dual simplex is the standard for SDDP — cut addition modifies RHS, which is a bound change in the dual    |
| Scaling          | `Clp_scaling(model, 0)`                  | 0 (off)      | Disabled for warm-start compatibility; see open point in [Solver Abstraction §3](./solver-abstraction.md) |
| Log level        | `Clp_setLogLevel(model, 0)`              | 0 (none)     | Quiet for production; millions of solves per run                                                          |
| Primal tolerance | `Clp_setPrimalTolerance(model, 1e-7)`    | 1e-7         | Match HiGHS defaults for cross-solver reproducibility                                                     |
| Dual tolerance   | `Clp_setDualTolerance(model, 1e-7)`      | 1e-7         | Match HiGHS defaults                                                                                      |
| Max iterations   | `Clp_setMaximumIterations(model, value)` | Configurable | Per-solve iteration limit; prevents runaway solves                                                        |
| Max seconds      | `Clp_setMaximumSeconds(model, value)`    | Configurable | Per-solve time limit                                                                                      |
| Perturbation     | `Clp_setPerturbation(model, 100)`        | 100 (auto)   | Auto-perturb if degenerate; override to 50 (force on) in retry                                            |

### 4.2 Scaling Note

CLP supports four scaling modes via `Clp_scaling(model, mode)`:

| Mode | Name        | Description              |
| ---- | ----------- | ------------------------ |
| 0    | Off         | No scaling               |
| 1    | Equilibrium | Row/column equilibration |
| 2    | Geometric   | Geometric mean scaling   |
| 3    | Auto        | CLP chooses              |

The scaling strategy interacts with the open point in [Solver Abstraction §3](./solver-abstraction.md) (single-phase vs two-phase scaling). If POWE.RS manages its own scaling (applied at the template level), CLP internal scaling should remain off to avoid double-scaling. If POWE.RS delegates scaling to the solver, CLP's auto mode (3) is a reasonable default.

## 5. C++ Wrapper Strategy (Anticipated Optimization)

The C API is sufficient for the Option A baseline (full rebuild per stage transition). The C++ API enables a potentially faster alternative: **LP template cloning**.

### 5.1 Cloning Mechanism

`ClpSimplex` provides built-in support for saving and restoring a "base model":

- `makeBaseModel()` — Saves a copy of the current model state (matrix, bounds, objective, but normally without cuts)
- `setToBaseModel(nullptr)` — Resets the model to the saved base state
- `ClpSimplex(const ClpSimplex&)` — Copy constructor for full model duplication

**Potential SDDP usage**: After loading a stage template via `Clp_loadProblem`, call `makeBaseModel()` to snapshot it. At each stage transition, call `setToBaseModel(nullptr)` to restore the structural LP instantly (without re-parsing CSR arrays), then add cuts via `Clp_addRows` and patch scenario values via mutable pointers.

### 5.2 Thin C Wrapper

Since Rust FFI works with `extern "C"` functions, the C++ cloning methods must be exposed through a thin C wrapper:

The wrapper provides a minimal set of C functions that:

1. Accept the raw `ClpSimplex*` pointer (obtained via `Clp_getClpSimplex(model)`)
2. Call the C++ method
3. Return any result through C-compatible types

**Required wrapper functions**:

| Wrapper Function                     | C++ Method Called              | Purpose                                                                         |
| ------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------- |
| `ClpEx_makeBaseModel(ptr)`           | `ptr->makeBaseModel()`         | Save structural LP as base                                                      |
| `ClpEx_setToBaseModel(ptr)`          | `ptr->setToBaseModel(nullptr)` | Reset to saved base state                                                       |
| `ClpEx_setPersistenceFlag(ptr, val)` | `ptr->setPersistenceFlag(val)` | Control memory reuse behavior (0=normal, 1=reuse if bigger, 2=reuse with extra) |

The wrapper is compiled as a C++ translation unit and linked into the Rust binary. The Rust side declares the wrapper functions as `extern "C"` and calls them through the raw pointer obtained from `Clp_getClpSimplex`.

### 5.3 Status and Trade-offs

| Aspect       | Option A Baseline (C API only)           | Cloning (C++ wrapper)                                                                     |
| ------------ | ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| Rebuild cost | `Clp_loadProblem` (parse CSR + allocate) | `setToBaseModel` (restore saved arrays)                                                   |
| Complexity   | Minimal — pure C FFI                     | Additional build step, thin C++ wrapper                                                   |
| Portability  | Works with any CLP build                 | Requires C++ linkage, tight version coupling                                              |
| Memory       | Template stored outside solver           | Base model stored inside each solver instance                                             |
| Status       | **Adopted baseline**                     | **Anticipated optimization** — implement and benchmark when performance data is available |

## 6. Memory Footprint

| Component                    | Formula                                     | Example (1120 states, 15K cuts)                         |
| ---------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| LP matrix storage            | nnz x 16 bytes                              | ~5 MB                                                   |
| Working arrays (simplex)     | (rows + cols) x 3 x 8 bytes                 | ~1 MB                                                   |
| Basis storage                | (rows + cols) x 1 byte                      | ~50 KB                                                  |
| Factorization                | Varies with sparsity, typically 2-5x matrix | ~5-10 MB                                                |
| Base model copy (if cloning) | ~same as LP matrix + bounds                 | ~7 MB                                                   |
| **Total per instance**       |                                             | **~15 MB** (without cloning), **~22 MB** (with cloning) |
| **192 threads**              |                                             | **~2.9 GB** / **~4.2 GB**                               |

This is within acceptable bounds for production HPC nodes (256+ GB RAM). The cloning overhead (~7 MB per instance) is modest relative to the cut pool memory (~28 GB per rank for 120 stages).

## 7. CLP-Specific Considerations

### 7.1 Column-Major Format

`Clp_loadProblem` expects column-major (CSC) format, which matches the stage LP template format directly — templates store their structural matrix in CSC form (see [Solver Abstraction §11](./solver-abstraction.md)). No format transposition is needed at stage transitions.

`Clp_addRows` for cut addition accepts row-major (CSR) format (row starts, column indices, elements), which matches the cut pool's CSR-friendly storage layout. No transposition needed for cut addition either.

### 7.2 Dual Sign Convention

CLP's dual sign convention must be verified against the canonical sign convention defined in [Solver Abstraction §8](./solver-abstraction.md). If CLP reports duals with a different sign for $\geq$ constraints, the CLP implementation must negate the appropriate dual values before returning them to the SDDP algorithm. This is a critical correctness requirement — sign errors in duals produce divergent cuts.

### 7.3 Thread Safety

Each `Clp_Simplex*` instance is **not thread-safe** — it must be exclusively owned by one thread. This aligns with the solver abstraction's thread-safety requirement ([Solver Abstraction §4.2](./solver-abstraction.md)). Each OpenMP thread creates its own `Clp_Simplex*` via `Clp_newModel()` at initialization and destroys it at shutdown via `Clp_deleteModel`.

### 7.4 Persistence Flag

CLP's `setPersistenceFlag(int value)` controls memory reuse behavior:

| Value | Behavior                                           |
| ----- | -------------------------------------------------- |
| 0     | Normal — allocate/deallocate as needed             |
| 1     | Reuse arrays if bigger needed (avoid reallocation) |
| 2     | As 1 but allocate a bit extra (amortized growth)   |

For SDDP where the LP is rebuilt many times with similar (but not identical) sizes, `setPersistenceFlag(2)` reduces allocation overhead by keeping internal arrays sized for the largest LP seen so far. This is a CLP-specific micro-optimization accessible via the C++ wrapper (§5.2).

## Cross-References

- [Solver Abstraction](./solver-abstraction.md) — Interface contract (§4), LP layout convention (§2), cut pool design (§5), error categories (§6), retry contract (§7), dual normalization (§8), basis storage (§9), stage templates (§11)
- [Solver Architecture Decisions](./SOLVER_ARCHITECTURE_DECISIONS.md) — Decision 5 (dual-solver validation) that motivated this spec
- [HiGHS Implementation](./solver-highs-impl.md) — Companion solver implementation for cross-validation
- [Solver Workspaces & LP Scaling](./solver-workspaces.md) — Thread-local workspace infrastructure that owns the `Clp_Simplex*` instance
- [LP Formulation](../01-math/lp-formulation.md) — Constraint structure that CLP operates on
- [Cut Management](../01-math/cut-management.md) — How cuts are generated; this spec handles how they are loaded via `Clp_addRows`
- [Training Loop](./training-loop.md) — Forward/backward pass orchestration driving solver invocations
- [Binary Formats](../02-data-model/binary-formats.md) — Cut pool CSR layout (§3.4), LP rebuild analysis (§A)
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — OpenMP threading model requiring one CLP instance per thread
- [Memory Architecture](../04-hpc/memory-architecture.md) — NUMA-aware allocation for solver workspaces
- [Configuration Reference](../05-config/configuration-reference.md) — Solver configuration parameters
