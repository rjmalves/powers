# T-020b: Extract Solver Abstraction Layer Spec

## Epic

Epic 4: Architecture Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the complete multi-solver abstraction layer into a dedicated spec. This is one of the most architecturally significant components in the system: it defines how the SDDP algorithm interacts with LP solvers (HiGHS, CPLEX, Gurobi) through a unified trait hierarchy with compile-time solver selection via Cargo feature flags.

The source content (~1,450 lines from DATA_MODEL §5.4-5.5) was previously buried inside T-015 (binary-formats.md) under "internal data structures". It must be extracted as its own top-level architecture spec because:

1. **It defines a complete abstraction boundary** — the `LpSolver` trait is the single interface between the SDDP algorithm and any LP solver
2. **Compile-time solver selection** via feature flags is a key architectural decision that affects build configuration, CI/CD, and deployment
3. **Pre-allocated cut constraint design** is deeply tied to SDDP algorithm correctness (deterministic slot assignment, bound toggling vs row insertion)
4. **Thread-local solver workspaces** bridge the architecture and HPC layers
5. **LP scaling** affects both correctness (cut coefficients) and performance (solver stability)
6. **Multiple solver implementations** will be needed for production (HiGHS for open-source, CPLEX/Gurobi for commercial)

## Acceptance Criteria

- [ ] `docs/specs/03-architecture/solver-abstraction.md` extracted from DATA_MODEL §5.4 (5.4.1-5.4.10) and §5.5 (5.5.1-5.5.5)
- [ ] Covers ALL subsections (see Technical Details below)
- [ ] All Rust trait definitions, struct definitions, and code examples preserved verbatim
- [ ] All diagrams and tables preserved
- [ ] Cross-references updated: links to lp-formulation.md (cut math), training-loop.md (forward/backward pass), hybrid-parallelism.md (thread model), memory-architecture.md (NUMA allocation)
- [ ] If content exceeds 500 lines, split into `solver-abstraction.md` (trait + selection + errors + cuts) and `solver-workspaces.md` (thread-local infra + LP scaling + HiGHS impl)
- [ ] Correct frontmatter with `review_priority: 2-high` (must verify correctness before coding)

## Files to Create

- `docs/specs/03-architecture/solver-abstraction.md`
- `docs/specs/03-architecture/solver-workspaces.md` (if split needed for 500-line limit)

## Technical Details

Source: `DATA_MODEL_SPECIFICATION.md` §5.4, §5.5

### Content from §5.4 — Solver Interface Specification (~1,250 lines)

- **§5.4.1 Trait Hierarchy** — `LpProblem` (data holder) → `LpScaling` (preprocessing) → `LpSolver` (execution) → `LpSolution` (result data). Diagram showing the 4-layer stack. Static data vs transient solver state separation.

- **§5.4.2 Core Solver Trait** — The `LpSolver` trait definition: `solve()`, `solve_with_basis()`, `update_and_solve()`, `reset()`, `statistics()`. This is `Send + Sync` because solver instances are thread-local (not shared). The SDDP algorithm ONLY interacts through this trait.

- **§5.4.3 Pre-allocated Cut Constraint Design** — Full preallocation of cut rows at LP construction, bound toggling (`-∞` = inactive, `α` = active), `CutSlotManager` with deterministic O(1) slot computation, capacity calculation formula, `BitVec` active bitmap.

- **§5.4.4 Solver Error Types** — `SolverError` enum: `Infeasible` (with ray), `Unbounded` (with direction), `NumericalDifficulty` (with partial solution + recovery suggestion), `TimeLimit`, `IterationLimit`, `Internal`. `NumericalRecoverySuggestion` enum.

- **§5.4.5 Solver-Specific Retry Logic** — Encapsulated within each solver implementation (SDDP never sees retries). `HighsRetryConfig` example: clear basis → disable presolve → switch to IPM → relax tolerances.

- **§5.4.6 Dual Variable Normalization** — Canonical sign convention across solvers. `normalize_dual()` method converting solver-specific conventions to canonical form.

- **§5.4.7 Basis Storage for Warm-Starting** — `Basis` struct: `column_status`, `row_status` as `Vec<BasisStatus>`. Stored in original (not presolved) space for cross-solve portability.

- **§5.4.8 Compile-Time Solver Selection** — Cargo feature flags: `solver-highs` (default), `solver-cplex`, `solver-gurobi`. `ActiveSolver` type alias via `#[cfg(feature = ...)]`. `create_solver()` factory function.

- **§5.4.9 Thread-Local Solver Infrastructure** — `ThreadSolverWorkspace`: solver instance + RHS buffer + solution storage + basis cache + stats + NUMA node. NUMA-aware allocation with first-touch policy. `WorkspaceManager` for all threads. Thread safety invariants for `LpProblem` (read-only during parallel forward pass, single-thread mutation during backward pass). Batch bound operations for hot path performance. Comparison with §6.9.6 `SolverPool` pattern.

- **§5.4.10 HiGHS Implementation Guidelines** — `HighsSolver` struct, full `LpSolver` trait implementation, model loading, basis management, warm-start with bound-toggled cuts, retry strategy implementation, memory footprint per instance (~15 MB), batch `change_rows_bounds` API.

### Content from §5.5 — LP Scaling Specification (~200 lines)

- **§5.5.1 Scaling Transformation** — Mathematical definition: column scaling `x̃ = D_c⁻¹ x`, row scaling `Ã = D_r × A × D_c`. Solution back-transformation for primals, duals, reduced costs.

- **§5.5.2 Scaling Data Structures** — `LpScaling` struct with `column_scale`, `row_scale`, `is_scaled` flag.

- **§5.5.3 Scaling Impact on Cut Coefficients** — CRITICAL: cuts are stored in physical units, scaling applied at solve time. Transformation formulas for cut coefficients and RHS.

- **§5.5.4 Scaling Workflow Integration** — 4-step workflow: construct LP → compute scaling → solve (scaled) → extract solution (unscaled). Diagram.

- **§5.5.5 FlatBuffers Schema for Scaling Persistence** — Scaling factors persisted for checkpoint/resume consistency.

### Suggested Split (if needed)

If total content exceeds 500 lines:

**solver-abstraction.md** (~400 lines):

- §5.4.1 Trait hierarchy
- §5.4.2 Core solver trait
- §5.4.3 Pre-allocated cut design
- §5.4.4 Error types
- §5.4.5 Retry logic
- §5.4.6 Dual normalization
- §5.4.7 Basis storage
- §5.4.8 Compile-time selection

**solver-workspaces.md** (~350 lines):

- §5.4.9 Thread-local solver infrastructure
- §5.4.10 HiGHS implementation guidelines
- §5.5.1-5.5.5 LP scaling specification

### Cross-References

This spec is central and connects to many others:

- `01-math/lp-formulation.md` — constraint structure that the solver operates on
- `01-math/cut-management.md` — how cuts are generated (this spec handles how they're stored/enabled in the LP)
- `03-architecture/training-loop.md` — forward pass (parallel solve), backward pass (cut addition)
- `04-hpc/hybrid-parallelism.md` — OpenMP threading model that requires thread-local solvers
- `04-hpc/memory-architecture.md` — NUMA-aware allocation for solver workspaces
- `02-data-model/binary-formats.md` — FlatBuffers schema for scaling persistence, LP structure serialization
- `05-config/configuration-reference.md` — solver config parameters (threads_per_solve, tolerances note)

## Definition of Done

File(s) created with complete solver abstraction content, all trait definitions and code preserved, valid cross-references, under 500 lines each. Frontmatter marked with `review_priority: 2-high`.
