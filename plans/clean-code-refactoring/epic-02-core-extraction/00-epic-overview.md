# Epic 2: Core Extraction

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 3 weeks (2 sprints)
> **Status**: ✅ Complete

---

## ⚠️ CRITICAL REMINDER

Before starting any work in this epic, read the [Critical Principles](../00-master-plan.md#️-critical-principles-correctness-first-then-performance) section.

**Algorithm correctness is non-negotiable.** Every change in this epic must pass golden tests. If you encounter unexpected results, **STOP and ask for clarification**.

---

## Summary

This epic extracts focused, reusable functions from the monolithic `subproblem.rs` (6,631 lines) into the new `src/model/` module. We focus on **solution extraction** and **constraint building**—the most self-contained pieces that can be extracted without modifying algorithm flow.

**Key principle**: Extract, don't rewrite. The logic remains identical; we're only improving organization and reducing function sizes.

---

## Scope

### Included

1. **Solution Extraction Refactoring**
   - Extract `get_*_from_solution()` functions into cohesive module
   - Create `SolutionExtractor` struct with clear API
   - Reduce parameter counts through context structs
   - **Design for future SoA migration**: APIs should support both current `collect()` pattern AND future `copy_into_slice()` pattern

2. **Constraint Building Extraction**
   - Extract constraint generation into `src/model/constraints/`
   - Separate by constraint type (hydro balance, bus balance, AR dynamics, etc.)
   - Create builder patterns for constraint construction
   - **Design for future preallocation**: constraint builders should support preallocated buffers

3. **Variable Index Management**
   - Extract variable indexing logic
   - Create clear mapping structs

### Excluded

- Forward/backward pass logic (Epic 3)
- State management changes (Epic 4)  
- Memory pool implementation (Epic 5)
- Solver interface changes (keep existing HiGHS interaction)

---

## Dependencies

- **Requires**: 
  - Epic 1 complete (golden tests, timing module, module skeleton)
- **Enables**:
  - Epic 3: Algorithm Separation (cleaner subproblem interface)
  - Epic 5: Memory Optimization (identified allocation points)

---

## Acceptance Criteria

- [ ] `src/model/solution_extract.rs` contains all solution extraction logic
- [ ] `src/model/constraints/` contains constraint generation by type
- [ ] All extracted functions are ≤50 lines
- [ ] All extracted functions have ≤4 parameters
- [ ] `subproblem.rs` calls into new modules (facade pattern)
- [ ] **Solution extraction APIs support `extract_into()` pattern** for future SoA migration
- [ ] Golden tests pass (bit-for-bit identical output)
- [ ] No performance regression (benchmark within 5%)
- [ ] All existing tests pass

### Correctness Verification

After EVERY change:
- [ ] `cargo test` passes
- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] `cargo bench` shows no regression

---

## Technical Approach

### Extraction Strategy

We use the **Facade Pattern**: existing functions in `subproblem.rs` become thin wrappers that delegate to new modules. This minimizes risk by:

1. Keeping call sites unchanged
2. Allowing incremental extraction
3. Making rollback easy if issues arise

### Future-Ready Solution Extraction Design

The current code uses `clone_from_slice()` into preallocated `Realization` fields. The new `SolutionExtractor` must preserve this AND be ready for future SoA migration:

**Current Pattern** (preserved):
```rust
fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    let first = *self.variables.deficit.first().unwrap();
    let last = *self.variables.deficit.last().unwrap() + 1;
    realization.deficit.clone_from_slice(&solution.colvalue[first..last]);
}
```

**New Pattern** (SoA-ready):
```rust
impl SolutionExtractor {
    /// Extract deficit values into the target slice.
    /// This is the low-level API that works with any slice, enabling SoA layouts.
    #[inline]
    pub fn extract_deficit_into(&self, solution: &Solution, target: &mut [f64]) {
        let range = self.indices.deficit_range();
        target.copy_from_slice(&solution.colvalue[range]);
    }
    
    /// Extract deficit into Realization (current API, calls extract_into).
    pub fn extract_deficit(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_deficit_into(solution, &mut realization.deficit);
    }
}
```

This dual-API approach:
1. **Preserves current behavior** via `extract_X()` methods
2. **Enables future SoA** via `extract_X_into()` methods that take raw slices
3. **Zero overhead** since `extract_X()` just calls `extract_X_into()`

### Complete Solution Extraction Functions

Based on `src/subproblem.rs` lines 1569-2046, the following extraction functions exist and must ALL be extracted:

#### Primal Variables (from `solution.colvalue`)

| Function | Target Field | Source Variables |
|----------|-------------|------------------|
| `get_deficit_from_solution` | `realization.deficit` | `variables.deficit` |
| `get_net_exchange_from_solution` | `realization.exchange` | `variables.direct_exchange`, `variables.reverse_exchange` |
| `get_thermal_gen_from_solution` | `realization.thermal_generation` | `variables.thermal_gen` |
| `get_spillage_from_solution` | `realization.spillage` | `variables.spillage` |
| `get_turbined_flow_from_solution` | `realization.turbined_flow` | `variables.turbined_flow` |
| `get_final_storage_from_solution` | `realization.final_storage` | `variables.stored_volume` |
| `get_load_from_solution` | `realization.loads` | `variables.load` |
| `get_inflow_from_solution` | `realization.inflow` | `variables.inflow` |

#### Dual Variables (from `solution.rowdual`)

| Function | Target Field | Source Constraints |
|----------|-------------|-------------------|
| `get_water_values_from_solution` | `realization.water_value` | `constraints.hydro_balance` |
| `get_marginal_cost_from_solution` | `realization.marginal_cost` | `constraints.load_balance` |
| `get_lag_duals_from_solution` | `realization.load_lag_duals`, `realization.inflow_lag_duals` | `constraints.load_lag_constraints`, `constraints.inflow_lag_constraints` |

#### State Population

| Function | Target Fields | Source |
|----------|--------------|--------|
| `populate_initial_state_fields` | `realization.initial_storage`, `realization.inflow_lags` | `self.state.coefficients()`, `self.inflow_lag_data` |

#### Objective Value

| Function | Target Fields | Source |
|----------|--------------|--------|
| `get_current_stage_objective` | `realization.current_stage_objective` | Computed from `total_stage_objective` and `solution` |

### Constraint Extraction Design (Also SoA-Ready)

Similarly, constraint builders should support preallocated buffers:

```rust
pub trait ConstraintBuilder {
    /// Build constraints, writing indices to preallocated buffers.
    /// Returns the number of constraints written.
    fn build_into(
        &self,
        context: &ConstraintContext,
        row_indices: &mut [usize],
        col_indices: &mut [usize],
        values: &mut [f64],
    ) -> usize;
}
```

### Module Structure After Epic

```
src/model/
├── mod.rs
├── solution_extract.rs     # SolutionExtractor, all get_*_from_solution
│   ├── SolutionExtractor   # Main struct
│   ├── VariableIndices     # Index ranges for primal variables
│   └── ConstraintIndices   # Index ranges for dual extraction
├── variable_indices.rs     # VariableIndices struct (detailed)
└── constraints/
    ├── mod.rs
    ├── hydro_balance.rs    # Hydro balance constraints
    ├── bus_balance.rs      # Bus/load balance constraints
    ├── thermal.rs          # Thermal generation constraints
    ├── ar_dynamics.rs      # AR model lag constraints
    └── bounds.rs           # Variable bounds
```

---

## Sprints

### [Sprint 1: Solution Extraction](./sprint-01/00-sprint-overview.md) (Week 1-2)

Focus on extracting solution extraction logic—the simplest extraction with clear boundaries.

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-006 | Analyze subproblem.rs extraction points | 2 | ⬜ |
| T-007 | Create VariableIndices and ConstraintIndices structs | 3 | ⬜ |
| T-008 | Create SolutionExtractor scaffold with dual API | 3 | ⬜ |
| T-009 | Extract deficit, exchange, thermal extraction | 3 | ⬜ |
| T-010 | Extract spillage, turbined flow, storage extraction | 3 | ⬜ |
| T-011 | Extract load, inflow extraction | 2 | ⬜ |
| T-012 | Extract dual extractions (water value, marginal cost, lag duals) | 3 | ⬜ |
| T-013 | Extract state population and objective calculation | 2 | ⬜ |

**Sprint 1 Points**: 21

### [Sprint 2: Constraint Extraction](./sprint-02/00-sprint-overview.md) (Week 2-3)

Extract constraint building into organized modules.

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-014 | Create constraints module structure | 2 | ⬜ |
| T-015 | Extract hydro balance constraints | 3 | ⬜ |
| T-016 | Extract bus balance constraints | 3 | ⬜ |
| T-017 | Extract AR dynamics constraints | 3 | ⬜ |
| T-018 | Extract bound constraints | 2 | ⬜ |
| T-019 | Refactor subproblem.rs to use new modules | 3 | ⬜ |

**Sprint 2 Points**: 16

---

## Estimated Effort

- **Duration**: 2 sprints (3 weeks)
- **Story Points**: 37
- **Risk Level**: Medium (touching core code, but not algorithm logic)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Extraction changes behavior | Medium | **CRITICAL** | Golden tests after every extraction |
| Hidden dependencies in subproblem.rs | Medium | Medium | Careful analysis in T-006 |
| Performance regression from indirection | Low | Medium | Benchmark after each sprint |
| Extraction too entangled | Low | High | Stop and ask if extraction proves difficult |
| Future SoA migration blocked | Low | Medium | Dual API design from the start |

---

## Definition of Done

- [ ] All Sprint 1 tickets complete
- [ ] All Sprint 2 tickets complete
- [ ] `src/model/` contains extracted logic
- [ ] `subproblem.rs` uses new modules via facade
- [ ] All functions ≤50 lines, ≤4 parameters
- [ ] **`extract_X_into()` methods exist** for all solution extraction
- [ ] Golden tests pass
- [ ] Benchmarks show no regression
- [ ] All tests pass
- [ ] Code reviewed and merged
