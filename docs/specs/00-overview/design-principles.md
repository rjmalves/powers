---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §1 (1.1-1.4)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Design Principles

## Purpose

This spec defines the foundational design principles governing the POWE.RS data model: format selection criteria, key design goals, the critical declaration order invariance requirement, and the reference to the LP subproblem formulation. All other data model and architecture specs build on these principles.

## 1. Format Selection Criteria

| Data Type                          | Recommended Format | Rationale                                                                         |
| ---------------------------------- | ------------------ | --------------------------------------------------------------------------------- |
| Configuration & Parameters         | JSON               | Human-readable, easily editable, small size                                       |
| Entity Registries                  | JSON               | Structured objects with relationships                                             |
| Time Series Data                   | Parquet            | Columnar, compressed, efficient for large data                                    |
| Policy Data (Cuts/States/Vertices) | FlatBuffers        | Zero-copy deserialization, cache-friendly dense arrays, in-memory during training |
| Simulation Results                 | Parquet            | High volume, per-entity indexing                                                  |
| Dictionaries/Metadata              | CSV                | Human-readable, small, universal                                                  |

## 2. Key Design Goals

1. **Separation of Concerns**: Static system data vs. dynamic algorithm data vs. stochastic data
2. **Scalability**: File formats that scale to production sizes without memory explosion
3. **Reproducibility**: All inputs deterministically produce same outputs
4. **Warm-start Support**: Efficient serialization/deserialization of algorithm state
5. **Distributed I/O**: Rank 0 loads, broadcasts to workers (or parallel loading where beneficial)
6. **Declaration Order Invariance**: Results must not depend on the order entities are declared in input files

## 3. Declaration Order Invariance (Critical Requirement)

> **⚠️ CRITICAL**: The optimization results MUST be identical regardless of the order in which entities are declared in input files. This is a fundamental correctness requirement.

**Principle**: If a user declares hydros A and B, runs the program, then exchanges the declaration order (B before A), the numerical results must be **bit-for-bit identical** (given the same random seed and same IDs).

**What determines identity**: The **entity ID** is the sole identifier. Two runs are equivalent if:

- All entity IDs are the same
- All entity properties are the same
- All relationships (by ID) are the same
- The random seed is the same

**What must NOT affect results**:

- Order of entities in JSON arrays (`hydros`, `thermals`, `buses`, `lines`)
- Order of rows in Parquet tables
- Order of constraints in `generic_constraints.json`
- Order of stages in `stages.json` (sorted by ID internally)
- Order of blocks within a stage (sorted by ID internally)
- Order of correlation blocks or entities within correlation blocks

### 3.1 Implementation Requirements

1. **Canonical Ordering**: After loading, all entity collections must be sorted by ID before any processing
2. **Deterministic Iteration**: All iterations over entities must use the canonical (sorted by ID) order
3. **LP Variable Ordering**: LP variables must be created in canonical order (by entity ID, then by block ID)
4. **LP Constraint Ordering**: LP constraints must be added in canonical order
5. **Random Number Generation**: Scenario generation must iterate entities in canonical order
6. **Cut Coefficients**: Cut coefficient ordering must follow the canonical state variable order

> **Note**: During implementation, one can assume that all the inputs are mostly sorted, so use sorting algorithms that have grater performance if all the inputs are already sorted by ID, ideally returning in few cycles if the data is already sorted by ID.

### 3.2 Validation Requirements

The test suite must include order-invariance tests that:

1. Run the same case with entities in different declaration orders
2. Verify bit-for-bit identical results (costs, decisions, cuts)

### 3.3 Canonical Ordering Example

```rust
// Canonical ordering example
impl System {
    /// Sort all entity collections by ID for order-invariant processing
    pub fn canonicalize(&mut self) {
        self.buses.sort_by_key(|b| b.id);
        self.lines.sort_by_key(|l| l.id);
        self.hydros.sort_by_key(|h| h.id);
        self.thermals.sort_by_key(|t| t.id);
    }
}

impl GenericConstraints {
    /// Sort constraints by ID for order-invariant processing
    pub fn canonicalize(&mut self) {
        self.constraints.sort_by_key(|c| c.id);
    }
}
```

### 3.4 Why This Matters

- **Debugging**: Users can reorganize input files without worrying about result changes
- **Version Control**: Reordering entities for readability doesn't create spurious diffs in results
- **Correctness**: Non-deterministic behavior from ordering is a bug, not a feature
- **Parallelism**: MPI ranks must agree on ordering without communication

## 4. LP Subproblem Formulation Reference

> **Complete Formulation**: See [MATHEMATICAL_FORMULATIONS.md](../../MATHEMATICAL_FORMULATIONS.md) for the authoritative mathematical specification of the SDDP algorithm and LP subproblem.

This data model specification focuses on **data structures and file formats**. The complete LP formulation, including:

- SDDP algorithm (forward/backward passes, convergence)
- Objective function and constraints
- Hydro production function models (constant, FPHA)
- Block formulation variants (parallel, chronological)
- Stochastic inflow modeling (PAR(p))
- Cut generation and aggregation
- Risk measures and advanced features

is documented in the mathematical formulations document.

**Key Cross-References**:

| This Document                                                           | Mathematical Formulations                                        | Description                 |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------- |
| hydros.json → productivity                                              | [Hydro Production Models](../01-math/hydro-production-models.md) | Constant productivity model |
| hydros.json → fpha\_\*                                                  | [Hydro Production Models](../01-math/hydro-production-models.md) | FPHA coefficients           |
| config.json → block_mode                                                | [Block Formulations](../01-math/block-formulations.md)           | Block formulation variant   |
| scenarios/inflow_seasonal_stats.parquet, inflow_ar_coefficients.parquet | [PAR Inflow Model](../01-math/par-inflow-model.md)               | PAR(p) model parameters     |
| config.json → inflow_non_negativity                                     | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md)      | Inflow treatment method     |
| stages.json → transitions                                               | [Discount Rate](../01-math/discount-rate.md)                     | Discount rate               |
| policy/cuts/                                                            | [Cut Management](../01-math/cut-management.md)                   | Cut coefficients            |

**Variable/Constraint Sizing**: See [Production Scale Reference](./production-scale-reference.md) for production-scale LP dimensions.

## 5. Implementation Language & FFI Strategy

### 5.1 Decision

POWE.RS is implemented in **Rust** with `unsafe` FFI boundaries for LP solver interaction, MPI communication (ferroMPI), and OpenMP thread management.

This decision was made with full awareness of the trade-offs. The discussion below documents the arguments for and against, so that future contributors understand why Rust was chosen and where the friction points are.

### 5.2 The Core Tension

POWE.RS is a system whose performance-critical inner loop calls C/C++ LP solver libraries. The natural question is: should the host language be C++ (same ecosystem as the solvers) or Rust (stronger safety guarantees for the parallel orchestration around the solvers)?

The concern is not theoretical. LP solver C++ APIs offer capabilities that their C APIs do not:

| Capability              | C++ API                                                                     | C API                                          |
| ----------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Model cloning           | Copy constructors (efficient deep copy of solver state)                     | Not available — must rebuild or extract/reload |
| Hot-start               | `markHotStart`/`solveFromHotStart` (CLP)                                    | Not available                                  |
| Change tracking         | `whatsChanged_` bitflags — solver skips unnecessary re-initialization (CLP) | Not available                                  |
| Mutable internal access | Direct struct access, placement new, custom allocators                      | Through accessor functions                     |
| Factorization control   | Fine-grained control over when to refactor                                  | Implicit only                                  |

See [Binary Formats Appendix A](../02-data-model/binary-formats.md) for the complete C vs C++ API comparison across HiGHS and CLP.

### 5.3 What Rust Can and Cannot Do

**Rust `unsafe` provides the same raw memory primitives as C/C++.** There is no performance penalty from choosing Rust — `unsafe` blocks generate identical machine code:

- `std::ptr::copy_nonoverlapping` = `memcpy`
- `std::ptr::copy` = `memmove`
- `std::alloc::alloc` / `dealloc` = `malloc` / `free`
- Raw pointer arithmetic, casting, reinterpretation — all available
- `#[repr(C)]` structs have identical layout to C structs

The limitation is not about what Rust _can_ do, but about **which solver APIs are accessible**. Rust calls C functions naturally via `extern "C"`. Calling C++ APIs requires either:

1. A thin C wrapper that exposes the C++ functionality through C-compatible functions
2. The `cxx` crate (limited to what it supports)
3. Direct use of the C API only (losing C++-exclusive features)

POWE.RS uses approach (3) — C APIs only — for solver portability. This means C++-exclusive features like model cloning, hot-start, and change tracking are not available through the standard solver abstraction. See [Solver Abstraction §9](../03-architecture/solver-abstraction.md) for the compile-time solver selection design.

### 5.4 Why This Does Not Block Performance

The adopted LP construction strategy is **Option A: rebuild per stage** (see [Binary Formats §A.2](../02-data-model/binary-formats.md)). Each thread owns one solver instance, reconfigures it per (stage, block), and solves. The per-solve workflow is:

1. Load full constraint matrix via bulk API (`passModel` / `loadProblem`)
2. Add active cuts via batch `addRows` in CSR format
3. Patch scenario-dependent values (RHS, bounds) — O(m+n) element modifications
4. Load basis from previous iteration for warm-start — O(m+n)
5. Solve (simplex pivots dominate runtime)

Steps 1-4 are cheap compared to step 5. The model clone pattern (available only via C++ APIs) would save steps 1-2 by copying a pre-built template, but:

- LP solve time dominates construction time at production scale
- Warm-starting from a stored basis converges in few pivots (the scenario perturbation is small)
- Different stages have different operative entities and block structures, limiting template reuse across stages

**What we store and restore is solver-agnostic.** The data persisted across iterations — basis status arrays, column values and bounds, row values and bounds, primal and dual solutions — are standard optimization framework concepts, not solver-internal structures. We deliberately avoid storing solver-internal factorization structures (LU decomposition, pivot sequences, working memory), which are solver-specific and non-portable. This keeps the warm-start mechanism clean across solver backends.

### 5.5 Enlarged Unsafe Boundary

The `unsafe` boundary in POWE.RS is **not limited to FFI function calls**. Performance-critical memory operations that would benefit from bypassing Rust's borrow checker are explicitly permitted within `unsafe` blocks:

- **Bulk memory copies** of LP data (coefficient arrays, bound vectors, solution vectors) using raw pointer operations when the safe alternative would require unnecessary allocations or copies
- **Pre-allocated buffer reuse** with raw pointer writes into thread-local scratch space, avoiding repeated allocation/deallocation in the solve loop
- **Direct manipulation of solver-owned memory** when the C API returns mutable pointers (e.g., CLP's mutable `double*` for row/column bounds), writing values in-place without intermediate buffers
- **Zero-copy views** into contiguous data structures (cut pool, scenario buffers) where creating safe slices would require lifetime gymnastics that add no real safety value
- **NUMA-aware allocation** with explicit placement and first-touch initialization patterns that require raw pointer manipulation

The principle is: **the `unsafe` boundary encloses all performance-critical memory operations that interact with solver data, not just the FFI call sites.** This includes preparing data before solver calls and extracting results after them. The safe Rust boundary sits above this — at the level of the training loop, scenario pipeline orchestration, and checkpoint/resume logic, where the parallelism and state management complexity actually lives.

This approach gives POWE.RS the memory manipulation efficiency of C++ in the solver-adjacent code while retaining compile-time safety guarantees in the ~80% of code that orchestrates the algorithm.

### 5.6 Arguments For and Against

**Why Rust holds for this project:**

- **Parallel correctness.** SDDP with MPI + threads is one of the hardest patterns to get right. Data races in cut aggregation, shared FCF access across MPI shared-memory windows, and scenario buffer management are catastrophic bugs that are nearly impossible to reproduce. Rust catches these at compile time in non-FFI code. In C++, a data race in the cut pool can silently produce wrong cuts, and the only symptom is degraded convergence — weeks to diagnose.
- **The solver interaction is contained.** The `unsafe` FFI + memory manipulation code is concentrated in a single module (~1,000-2,000 lines). The rest of the system — scenario pipeline, training loop, checkpoint/resume, MPI orchestration, input loading, validation — benefits from safety.
- **`unsafe` gives the same power as C++.** Raw memory operations, pointer arithmetic, and direct buffer manipulation are all available. The difference is that these operations are explicitly marked and auditable, not silently mixed with safe code.
- **The C API gap is manageable.** The modify-in-place + warm-start pattern works well through C APIs. The missing C++ features (cloning, hot-start, change tracking) provide incremental improvements, not fundamental capabilities.

**Where C++ would have a genuine advantage:**

- **Solver ecosystem is C++ native.** HiGHS, CLP, CPLEX, Gurobi all have C++ as their primary API. Bug reports, performance tuning, and internal documentation assume C++. Staying in the solver's language eliminates an entire class of friction.
- **No FFI translation layer.** Every solver call in Rust goes through an `extern "C"` boundary. In C++, you call solver methods directly. While the overhead per call is negligible, the ergonomic cost of maintaining bindings, handling error codes instead of exceptions, and wrapping opaque pointers adds friction throughout development.
- **C++-exclusive solver features.** If a future solver backend would benefit significantly from copy constructors, hot-start, or change tracking, accessing these from Rust requires writing and maintaining a C shim — additional code that must track upstream solver API changes.

### 5.7 Revisiting This Decision

This decision should be revisited if:

1. **Profiling shows that LP construction time (steps 1-4) becomes a significant fraction of total runtime.** This would indicate that the rebuild-per-stage approach is too expensive and model cloning (C++ API) would provide meaningful speedup.
2. **The `unsafe` boundary grows beyond ~15% of the codebase**, suggesting that the safety benefits of Rust are being eroded by the volume of unsafe code needed.
3. **A solver backend requires C++-exclusive features for correctness** (not just performance), making the C API insufficient.

None of these conditions are expected at current production scale, but they should be monitored as the system evolves.

## Cross-References

- [Notation Conventions](./notation-conventions.md) — Mathematical notation and symbol definitions used across all specs
- [Production Scale Reference](./production-scale-reference.md) — Production-scale LP dimensions and performance targets
- [LP Formulation](../01-math/lp-formulation.md) — Complete LP subproblem formulation
- [Input Directory Structure](../02-data-model/input-directory-structure.md) — File layout implementing these format choices
- [Validation Architecture](../03-architecture/validation-architecture.md) — Validation of order invariance and other requirements
