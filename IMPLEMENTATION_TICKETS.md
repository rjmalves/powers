# Implementation Tickets: Load/Inflow Separation Migration

**Epic**: Complete Unified to Separated Architecture Migration
**Created**: 2025-11-06
**Priority**: HIGH - Critical Architectural Debt
**Total Estimated Effort**: 21-34 story points (3-4 sprints)

## Executive Summary

This epic addresses the incomplete migration from unified uncertainty handling to separated loads/inflows architecture. The codebase currently maintains parallel data structures that must be kept in sync, creating maintenance burden and architectural confusion.

**Key Issues Addressed**:
- Triple redundancy in lag tracking structures
- Unclear responsibilities between `UncertaintyConstraintManager` and `UncertaintyConstraintData`
- Mixed concerns in data structures
- Parallel constraint tracking that must be manually synchronized

**Business Value**:
- Improved type safety (catch errors at compile time)
- Reduced maintenance burden (single source of truth)
- Better code clarity (clear separation of concerns)
- Foundation for future uncertainty modeling enhancements

---

## Sprint 1: Foundation & Documentation (Week 1-2)

### TICKET-001: Document Current Architectural State ✅ COMPLETE

**Priority**: P0 (Blocker for other work)
**Effort**: 2 story points (confidence: high)
**Status**: ✅ **COMPLETE** (2025-11-06)

#### Context

Before refactoring, we need comprehensive documentation of the current state to:
- Ensure we understand all usage patterns
- Create reference for testing equivalence
- Guide the migration process
- Help reviewers understand the changes

#### Acceptance Criteria

- [x] All files using `UncertaintyConstraintManager` are documented with usage patterns
- [x] All files using `UncertaintyConstraintData` are documented with usage patterns
- [x] Parallel data structure synchronization points are identified and documented
- [x] Data flow diagrams created showing current vs. target architecture
- [x] Migration guide document created with step-by-step approach

#### Tasks

##### Implementation
- [x] Audit all usages of `UncertaintyConstraintManager` in codebase
  - ✅ Found in: `src/input.rs`, `src/subproblem.rs`, `src/uncertainty_constraints.rs`
- [x] Audit all usages of `UncertaintyConstraintData` in codebase
  - ✅ Found in: `src/subproblem.rs`
- [x] Document the parallel structure population in lines 1806-1843 of subproblem.rs
  - ✅ Comprehensive analysis in `docs/migration/current_architecture.md`
- [x] Create data flow diagram for current architecture
  - ✅ Included in current_architecture.md
- [x] Create data flow diagram for target architecture
  - ✅ Included in target_architecture.md
- [x] Document all methods that access lag buffers
  - ✅ All access patterns documented

##### Documentation
- [x] Create MIGRATION_GUIDE.md with detailed steps
  - ✅ Created as `docs/migration/migration_steps.md` (20KB)
- [x] Add architectural decision record (ADR) for separation rationale
  - ✅ Included in current_architecture.md and target_architecture.md
- [x] Update ARCHITECTURE_DEEP_ANALYSIS.md with final migration plan
  - ✅ Created comprehensive migration documentation
- [x] Document testing strategy for equivalence verification
  - ✅ Created `docs/migration/testing_equivalence.md` (19KB)

#### Technical Notes

**Key Files to Audit**:
- `src/subproblem.rs` - Primary usage site
- `src/uncertainty_constraints.rs` - Manager definition
- `src/sddp.rs` - May use uncertainty handling
- `tests/` - Test coverage to preserve

**Documentation Structure**:
```
docs/migration/
├── current_architecture.md
├── target_architecture.md
├── migration_steps.md
└── testing_equivalence.md
```

#### Dependencies

- Blocked by: None (first ticket) ✅
- Blocks: All other tickets in this epic ✅ UNBLOCKED
- Related: None

#### Deliverables (COMPLETE)

- ✅ `docs/migration/current_architecture.md` (14KB) - Complete analysis of current state
- ✅ `docs/migration/target_architecture.md` (20KB) - Target design and architecture
- ✅ `docs/migration/migration_steps.md` (20KB) - Step-by-step implementation guide
- ✅ `docs/migration/testing_equivalence.md` (19KB) - Comprehensive testing strategy
- ✅ `MIGRATION_PROGRESS.md` (8.6KB) - Progress tracker

**Total Documentation**: 73KB of detailed migration documentation

---


### TICKET-002: Create Comprehensive Regression Test Suite

**Priority**: P0 (Blocker for refactoring)
**Effort**: 5 story points (confidence: high)

#### Context

Before making architectural changes, we need a comprehensive regression test suite to ensure the refactored code produces identical results. This is critical for a numerical algorithm where correctness is paramount.

#### Acceptance Criteria

- [ ] Tests capture current behavior of lag buffer updates
- [ ] Tests capture current behavior of lag-fixing constraint updates
- [ ] Tests capture current behavior of uncertainty observation constraint updates
- [ ] Tests verify parallel structure synchronization (currently maintained)
- [ ] All tests pass with current implementation
- [ ] Test coverage reaches 95%+ for uncertainty-related code

#### Tasks

##### Implementation
- [ ] Create test module `tests/uncertainty_migration_baseline.rs`
- [ ] Implement test: `test_lag_buffer_population_from_trajectory`
- [ ] Implement test: `test_unified_vs_separated_lag_constraints_consistency`
- [ ] Implement test: `test_uncertainty_constraint_rhs_computation`
- [ ] Implement test: `test_entity_data_routing_consistency`
- [ ] Implement test: `test_global_entity_indexing`
- [ ] Create test fixtures with sample trajectories and models

##### Testing
- [ ] Verify tests catch intentional breaking changes
- [ ] Test with various AR orders (1, 2, 5, 10)
- [ ] Test with different load/inflow ratios (10/5, 50/20, 100/100)
- [ ] Test with edge cases (zero lags, single entity, large systems)
- [ ] Verify numerical stability of lag value extraction

##### Documentation
- [ ] Document test strategy in MIGRATION_GUIDE.md
- [ ] Add doc comments explaining each test's purpose
- [ ] Create test data generation utilities with documentation
- [ ] Document expected test execution time

#### Technical Notes

**Test Structure**:
```rust
#[test]
fn test_unified_vs_separated_lag_constraints_consistency() {
    // Given: A subproblem with both unified and separated structures
    // When: Constraints are populated
    // Then: Both structures contain identical constraint indices
    
    let subproblem = create_test_subproblem();
    let constraints = &subproblem.constraints;
    
    // Verify unified structure
    let unified = constraints.lag_fixing_constraints.as_ref().unwrap();
    
    // Verify separated structures
    let load_constraints = constraints.load_lag_constraints.as_ref().unwrap();
    let inflow_constraints = constraints.inflow_lag_constraints.as_ref().unwrap();
    
    // Assert consistency
    assert_consistent_indexing(unified, load_constraints, inflow_constraints);
}
```

**Important**: These tests should FAIL after migration (verifying old structures removed).

#### Dependencies

- Blocked by: TICKET-001
- Blocks: TICKET-003, TICKET-004, TICKET-005
- Related: None

---

### TICKET-003: Refactor LoadLagData and InflowLagData Structures

**Priority**: P1
**Effort**: 3 story points (confidence: high)

#### Context

Create the target data structures that will replace the unified architecture. This establishes the foundation for type-safe, separated handling of loads vs. inflows.

#### Acceptance Criteria

- [ ] `LoadLagData` struct defined with all necessary fields
- [ ] `InflowLagData` struct defined with all necessary fields
- [ ] Both structs have clear, single responsibility
- [ ] Structs include buffer storage for lag values
- [ ] Constructor methods handle initialization correctly
- [ ] No compilation errors or warnings

#### Tasks

##### Implementation
- [ ] Define `LoadLagData` struct in subproblem.rs
- [ ] Define `InflowLagData` struct in subproblem.rs
- [ ] Implement `LoadLagData::new()` constructor
- [ ] Implement `InflowLagData::new()` constructor
- [ ] Add buffer allocation methods
- [ ] Add getter methods for type-safe access
- [ ] Add validation logic in constructors

##### Testing
- [ ] Unit test: `test_load_lag_data_construction`
- [ ] Unit test: `test_inflow_lag_data_construction`
- [ ] Unit test: `test_load_lag_data_indexing_bounds`
- [ ] Unit test: `test_inflow_lag_data_indexing_bounds`
- [ ] Test: Buffer capacity matches expected dimensions

##### Documentation
- [ ] Add comprehensive doc comments to structs
- [ ] Document indexing conventions ([bus_id][lag_idx])
- [ ] Add usage examples in doc comments
- [ ] Update module-level documentation

#### Technical Notes

**Target Structure**:
```rust
/// Type-safe container for load lag variables, constraints, and observations.
/// Indexed by bus_id to prevent confusion with hydro entities.
pub struct LoadLagData {
    /// Variables: [bus_id][lag_idx] → LP variable index
    pub variables: LoadLagVariables,
    
    /// Constraints: [bus_id][lag_idx] → LP constraint index
    pub constraints: LoadLagConstraints,
    
    /// Buffer: [bus_id][lag_idx] → lag observation value
    /// Used for updating lag-fixing constraints
    pub buffer: Vec<Vec<f64>>,
    
    /// Number of buses with uncertain loads
    pub n_buses: usize,
    
    /// Maximum lag order across all buses
    pub max_lag: usize,
}

impl LoadLagData {
    pub fn new(n_buses: usize, max_lag: usize) -> Self {
        // Pre-allocate buffers
    }
    
    pub fn get_lag(&self, bus_id: usize, lag_idx: usize) -> f64 {
        // Bounds-checked access
    }
    
    pub fn set_lag(&mut self, bus_id: usize, lag_idx: usize, value: f64) {
        // Bounds-checked mutation
    }
}
```

**Key Design Decisions**:
- Use `Vec<Vec<f64>>` for flexibility (buses may have different lag orders)
- Include bounds checking in accessor methods
- Store max_lag for validation
- Keep variables and constraints together (cohesion)

#### Dependencies

- Blocked by: TICKET-001, TICKET-002
- Blocks: TICKET-004, TICKET-005
- Related: None

---


## Sprint 2: Core Migration - Remove Unified Structures (Week 3-4)

### TICKET-004: Remove Unified lag_fixing_constraints Structure

**Priority**: P1
**Effort**: 5 story points (confidence: medium)

#### Context

Remove the old unified `lag_fixing_constraints` field and migrate all usage to the new separated `LoadLagConstraints` and `InflowLagConstraints`. This is the first major structural change.

#### Acceptance Criteria

- [ ] `lag_fixing_constraints: Option<Vec<Vec<usize>>>` field removed from Constraints struct
- [ ] All code using unified structure migrated to separated structures
- [ ] Parallel population code (lines 1806-1843) removed
- [ ] All tests pass with new structures only
- [ ] Regression tests confirm identical behavior

#### Tasks

##### Implementation
- [ ] Remove `lag_fixing_constraints` field from Constraints struct
- [ ] Update constraint population in `build_constraints()` to use only separated structures
- [ ] Remove parallel population logic (lines 1806-1843)
- [ ] Update `update_lag_fixing_constraints()` to iterate over separated structures
- [ ] Update any serialization/deserialization code
- [ ] Fix all compilation errors from removed field

##### Testing
- [ ] Verify regression tests still pass
- [ ] Add test: `test_separated_lag_constraints_only`
- [ ] Test with systems having only loads (no inflows)
- [ ] Test with systems having only inflows (no loads)
- [ ] Test with mixed load/inflow systems
- [ ] Performance test: Ensure no regression in constraint update speed

##### Documentation
- [ ] Update doc comments in Constraints struct
- [ ] Remove migration TODO comments
- [ ] Update CHANGELOG.md with breaking change note
- [ ] Update any internal documentation referencing unified structure

#### Technical Notes

**Migration Pattern**:
```rust
// OLD: Unified iteration
for (entity_idx, entity_constraints) in lag_fixing_constraints.iter().enumerate() {
    for (lag_idx, &constraint_idx) in entity_constraints.iter().enumerate() {
        // Update using global_entity_idx routing
    }
}

// NEW: Separated iteration
if let Some(ref load_data) = self.load_lag_data {
    for (bus_id, bus_constraints) in load_data.constraints.constraints_by_bus.iter().enumerate() {
        for (lag_idx, &constraint_idx) in bus_constraints.iter().enumerate() {
            let lag_value = load_data.buffer[bus_id][lag_idx];
            model.change_rows_bounds(constraint_idx, lag_value, lag_value);
        }
    }
}

// Similar for inflow_lag_data
```

**Risk Mitigation**:
- Keep old code in git history for reference
- Use feature flag if gradual rollout needed
- Monitor performance benchmarks closely

#### Dependencies

- Blocked by: TICKET-003
- Blocks: TICKET-005, TICKET-006
- Related: TICKET-002 (regression tests)

---

### TICKET-005: Replace UncertaintyConstraintManager with Separated Buffers

**Priority**: P1
**Effort**: 8 story points (confidence: medium)

#### Context

Replace the `UncertaintyConstraintManager` with direct buffer storage in `LoadLagData` and `InflowLagData`. This eliminates the global entity indexing and clarifies that we only need lag storage, not a "manager".

#### Acceptance Criteria

- [ ] `uncertainty_manager` field removed from Subproblem
- [ ] Lag buffers moved to LoadLagData and InflowLagData
- [ ] All buffer access patterns migrated to use separated structures
- [ ] `update_lag_buffers_from_trajectory()` split into load and inflow methods
- [ ] All tests pass with new buffer locations
- [ ] No performance degradation in buffer access

#### Tasks

##### Implementation
- [ ] Add buffer fields to LoadLagData and InflowLagData (if not in TICKET-003)
- [ ] Remove `uncertainty_manager: UncertaintyConstraintManager` field from Subproblem
- [ ] Create `update_load_lag_buffer_from_trajectory()` method
- [ ] Create `update_inflow_lag_buffer_from_trajectory()` method
- [ ] Update `prepare_from_trajectory()` to call new split methods
- [ ] Migrate all `uncertainty_manager.get_lag_observations()` calls
- [ ] Update `update_lag_fixing_constraints()` to use new buffers
- [ ] Remove `global_entity_idx` routing logic

##### Testing
- [ ] Unit test: `test_load_lag_buffer_updates`
- [ ] Unit test: `test_inflow_lag_buffer_updates`
- [ ] Integration test: `test_trajectory_to_buffer_to_constraints_flow`
- [ ] Test: Verify buffer values match trajectory extractions
- [ ] Test: Verify no buffer access out of bounds
- [ ] Regression test: Confirm identical constraint RHS values
- [ ] Performance test: Buffer update speed vs. baseline

##### Documentation
- [ ] Update doc comments in prepare_from_trajectory()
- [ ] Document new buffer update methods
- [ ] Update architecture diagrams showing buffer ownership
- [ ] Add code example for buffer usage pattern
- [ ] Update CHANGELOG.md

#### Technical Notes

**Buffer Update Pattern**:
```rust
fn update_load_lag_buffer_from_trajectory(&mut self, trajectory: &[&Realization]) {
    if let Some(ref mut load_data) = self.load_lag_data {
        for bus_id in 0..load_data.n_buses {
            let max_lag = load_data.buffer[bus_id].len();
            for lag_idx in 0..max_lag {
                let lookback = lag_idx + 1;
                if lookback < trajectory.len() {
                    let past_idx = trajectory.len() - 1 - lookback;
                    let lag_value = trajectory[past_idx].loads[bus_id];
                    load_data.buffer[bus_id][lag_idx] = lag_value;
                }
            }
        }
    }
}
```

**Key Changes**:
- No more `global_entity_idx` - direct bus_id/hydro_id indexing
- No more `entity_type` routing - type system handles it
- Clearer data ownership (buffers live with constraints)

**Performance Considerations**:
- Buffer access should be cache-friendly (iterate by bus_id, then lag_idx)
- Consider Vec reuse vs. reallocation
- Profile memory usage before/after

#### Dependencies

- Blocked by: TICKET-003, TICKET-004
- Blocks: TICKET-006
- Related: TICKET-002 (regression tests)

---


### TICKET-006: Refine UncertaintyConstraintData to UncertaintyObservationData

**Priority**: P2
**Effort**: 5 story points (confidence: high)

#### Context

Simplify `UncertaintyConstraintData` by removing routing fields that are now redundant with separated structures. Rename to `UncertaintyObservationData` to better reflect its purpose: storing precomputed coefficients for fast uncertainty constraint RHS updates.

#### Acceptance Criteria

- [ ] Struct renamed to `UncertaintyObservationData`
- [ ] Removed fields: `entity_type`, `entity_id`, `global_entity_idx`, `ar_order`, `psi_coefficients`
- [ ] Kept fields: `constraint_idx`, `innovation_idx`, `seasonal_std`, `deterministic_base`
- [ ] `entity_data` field renamed to `uncertainty_observation_data`
- [ ] All usage sites updated
- [ ] Regression tests pass

#### Tasks

##### Implementation
- [ ] Rename struct: `UncertaintyConstraintData` → `UncertaintyObservationData`
- [ ] Remove field: `entity_type: UncertaintyType`
- [ ] Remove field: `entity_id: usize`
- [ ] Remove field: `global_entity_idx: usize`
- [ ] Remove field: `ar_order: usize`
- [ ] Remove field: `psi_coefficients: Vec<f64>`
- [ ] Remove field: `season_id: usize` (if not needed in RHS update)
- [ ] Keep field: `constraint_idx: usize`
- [ ] Keep field: `innovation_var_idx: usize` (rename to `innovation_idx`)
- [ ] Keep field: `seasonal_std: f64`
- [ ] Keep field: `deterministic_base: f64`
- [ ] Rename Subproblem field: `entity_data` → `uncertainty_observation_data`
- [ ] Update `build_entity_constraint_data()` to build simplified structure
- [ ] Update `update_uncertainty_constraints()` if needed
- [ ] Fix all compilation errors

##### Testing
- [ ] Unit test: `test_uncertainty_observation_data_construction`
- [ ] Test: Verify RHS computation matches old behavior
- [ ] Test: Confirm removed fields not accessed anywhere
- [ ] Regression test: Uncertainty constraint updates produce identical results
- [ ] Test memory usage reduction from smaller struct

##### Documentation
- [ ] Add comprehensive doc comments to new struct
- [ ] Explain purpose: precomputed coefficients for fast RHS updates
- [ ] Document why uncertainty constraints can stay unified
- [ ] Update module-level documentation
- [ ] Update CHANGELOG.md

#### Technical Notes

**Before (12 fields, mixed concerns)**:
```rust
pub struct UncertaintyConstraintData {
    entity_type: UncertaintyType,      // ❌ Remove - routing
    entity_id: usize,                  // ❌ Remove - routing
    global_entity_idx: usize,          // ❌ Remove - routing
    constraint_idx: usize,             // ✅ Keep
    observation_var_idx: usize,        // ✅ Keep (maybe rename)
    innovation_var_idx: usize,         // ✅ Keep
    season_id: usize,                  // ❌ Remove - only for precomputation
    seasonal_mean: f64,                // ❌ Remove - baked into deterministic_base
    seasonal_std: f64,                 // ✅ Keep
    ar_order: usize,                   // ❌ Remove - not needed at runtime
    psi_coefficients: Vec<f64>,        // ❌ Remove - not needed at runtime
    deterministic_base: f64,           // ✅ Keep
}
```

**After (4 fields, single concern)**:
```rust
/// Precomputed coefficients for fast uncertainty observation constraint RHS updates.
/// 
/// These constraints have the form: Y_t = deterministic_base + seasonal_std * innovation
/// where the innovation is provided by the scenario at each forward pass step.
pub struct UncertaintyObservationData {
    /// LP constraint index to update
    pub constraint_idx: usize,
    
    /// Index in innovations array for this entity
    pub innovation_idx: usize,
    
    /// Standard deviation for current season (σ_s)
    pub seasonal_std: f64,
    
    /// Precomputed deterministic part: μ_s - Σ(φ_k · μ_{s-k})
    pub deterministic_base: f64,
}
```

**Why Keep Unified?**

Uncertainty observation constraints have identical update logic for loads and inflows:
```rust
for data in &self.uncertainty_observation_data {
    let innovation = innovations[data.innovation_idx];
    let rhs = data.deterministic_base + data.seasonal_std * innovation;
    model.change_rows_bounds(data.constraint_idx, rhs, rhs);
}
```

No benefit to separating this - the math is the same!

#### Dependencies

- Blocked by: TICKET-005
- Blocks: TICKET-007
- Related: None

---

## Sprint 3: Integration & Cleanup (Week 5-6)

### TICKET-007: Remove UncertaintyConstraintManager Module

**Priority**: P2
**Effort**: 3 story points (confidence: high)

#### Context

With lag buffers moved to separated structures and constraint data simplified, the `UncertaintyConstraintManager` module is no longer needed. Remove it to complete the migration.

#### Acceptance Criteria

- [ ] `src/uncertainty_constraints.rs` file removed
- [ ] `mod uncertainty_constraints` removed from lib.rs
- [ ] All imports of types from this module removed
- [ ] `UnifiedLagBuffer` usage eliminated
- [ ] No compilation errors
- [ ] All tests pass

#### Tasks

##### Implementation
- [ ] Remove `mod uncertainty_constraints;` from lib.rs
- [ ] Delete file: `src/uncertainty_constraints.rs`
- [ ] Remove imports: `use crate::uncertainty_constraints::*;`
- [ ] Verify no other code references this module
- [ ] Update Cargo.toml if module was listed
- [ ] Clean up any deprecated feature flags

##### Testing
- [ ] Verify all tests still compile and pass
- [ ] Check for dead code warnings
- [ ] Verify binary size reduction (removed ~200 lines)
- [ ] Run full test suite

##### Documentation
- [ ] Update lib.rs module documentation
- [ ] Update ARCHITECTURE.md to reflect removal
- [ ] Add note to CHANGELOG.md about removed internal module
- [ ] Update any architecture diagrams

#### Technical Notes

**Files to Check**:
- `src/lib.rs` - module declaration
- `src/subproblem.rs` - primary importer
- `src/sddp.rs` - may import types
- `tests/*.rs` - may use public types

**What Gets Removed**:
- `UncertaintyConstraintManager` struct (~100 lines)
- `UnifiedLagBuffer` struct (~50 lines)
- Associated methods and tests (~50 lines)

**Verification**:
```bash
# Ensure no references remain
rg "UncertaintyConstraintManager" src/
rg "UnifiedLagBuffer" src/
rg "uncertainty_constraints" src/
```

#### Dependencies

- Blocked by: TICKET-006
- Blocks: TICKET-008
- Related: None

---

### TICKET-008: Extract Lags Directly (Optional Optimization)

**Priority**: P3 (Nice to have)
**Effort**: 3 story points (confidence: medium)

#### Context

With separated structures, we can consider extracting lag values directly from trajectories when updating lag-fixing constraints, eliminating buffer storage for this purpose. This would make the code more similar to storage constraint updates (direct extraction pattern).

**Note**: This is optional. If buffers are needed for uncertainty observation constraints, this may not provide benefit.

#### Acceptance Criteria

- [ ] Lag-fixing constraints update without intermediate buffer storage
- [ ] Code pattern matches storage constraint update pattern
- [ ] Regression tests confirm identical behavior
- [ ] Performance benchmarks show no degradation (or improvement)
- [ ] Decision documented on whether to keep or remove lag buffers

#### Tasks

##### Implementation
- [ ] Prototype direct extraction in `update_load_lag_constraints()`
- [ ] Prototype direct extraction in `update_inflow_lag_constraints()`
- [ ] Compare: buffer approach vs. direct extraction
- [ ] If direct is better: remove lag buffers from LoadLagData/InflowLagData
- [ ] If buffers needed: keep current approach, document why

##### Testing
- [ ] Benchmark: buffer approach performance
- [ ] Benchmark: direct extraction performance
- [ ] Test: verify identical constraint RHS values
- [ ] Test: verify no regression in prepare_from_trajectory time
- [ ] Memory profiling: buffer allocation overhead

##### Documentation
- [ ] Document decision in architecture documentation
- [ ] If buffers removed: update data structure docs
- [ ] If buffers kept: document rationale (needed for uncertainty constraints)
- [ ] Add performance notes to relevant method docs

#### Technical Notes

**Direct Extraction Pattern**:
```rust
fn update_load_lag_constraints(&mut self, trajectory: &[&Realization]) {
    if let Some(ref load_data) = self.load_lag_data {
        for (bus_id, constraints) in load_data.constraints.constraints_by_bus.iter().enumerate() {
            for (lag_idx, &constraint_idx) in constraints.iter().enumerate() {
                // Extract directly - no buffer
                let lookback = lag_idx + 1;
                let past_idx = trajectory.len() - 1 - lookback;
                let lag_value = trajectory[past_idx].loads[bus_id];
                
                model.change_rows_bounds(constraint_idx, lag_value, lag_value);
            }
        }
    }
}
```

**Trade-offs**:
- **Pro**: Simpler code, less memory, matches storage pattern
- **Con**: Re-extracts if uncertainty constraints also need lags
- **Decision**: Depends on whether uncertainty constraints need lag buffer

**Key Question**: Do uncertainty observation constraint RHS computations need lag values?
- If YES: Keep buffers (single extraction, multiple uses)
- If NO: Remove buffers (extract on demand)

#### Dependencies

- Blocked by: TICKET-006
- Blocks: None
- Related: TICKET-005 (buffer introduction)

---


### TICKET-009: Address Command-Query Separation in State Methods

**Priority**: P2
**Effort**: 2 story points (confidence: high)

#### Context

The `extract_storage_from_trajectory()` method violates Command-Query Separation by both extracting data AND updating internal coefficients. Split this into separate query and command methods for clarity.

#### Acceptance Criteria

- [ ] `extract_storage_from_trajectory()` is pure query (no side effects)
- [ ] New `update_coefficients_from_trajectory()` method handles state mutation
- [ ] All call sites updated appropriately
- [ ] Tests verify separation of concerns
- [ ] No behavior change (just refactoring)

#### Tasks

##### Implementation
- [ ] Create new method: `extract_storage(&self, trajectory: &[&Realization]) -> Vec<f64>`
- [ ] Create new method: `update_coefficients_from_trajectory(&mut self, trajectory: &[&Realization])`
- [ ] Move side effects from extract to update method
- [ ] Update `prepare_from_trajectory()` to call both methods
- [ ] Rename existing method if backward compatibility needed
- [ ] Verify all state modifications in update method

##### Testing
- [ ] Unit test: `test_extract_storage_is_pure`
- [ ] Unit test: `test_update_coefficients_mutates_state`
- [ ] Test: Verify multiple calls to extract return same result
- [ ] Test: Verify coefficients updated correctly
- [ ] Regression test: Overall behavior unchanged

##### Documentation
- [ ] Document query nature of extract_storage
- [ ] Document command nature of update_coefficients
- [ ] Add doc comment explaining CQS pattern
- [ ] Update method signatures with appropriate borrowing

#### Technical Notes

**Before (Mixed)**:
```rust
fn extract_storage_from_trajectory(&mut self, trajectory: &[&Realization]) -> Vec<f64> {
    // Query part
    let storage: Vec<f64> = trajectory.last().storage.clone();
    
    // Command part (side effect!)
    self.some_internal_coefficient = compute_from(trajectory);
    
    storage  // Return value suggests pure query, but mutates!
}
```

**After (Separated)**:
```rust
/// Pure query: Extract storage levels from trajectory.
/// No side effects - can be called multiple times safely.
fn extract_storage(&self, trajectory: &[&Realization]) -> Vec<f64> {
    trajectory.last()
        .map(|r| r.storage.clone())
        .unwrap_or_default()
}

/// Command: Update internal coefficients based on trajectory.
/// Mutates internal state - should be called once per trajectory.
fn update_coefficients_from_trajectory(&mut self, trajectory: &[&Realization]) {
    self.some_internal_coefficient = compute_from(trajectory);
}
```

**Benefits**:
- Clearer intent (query vs. command)
- Easier to test (pure functions)
- Safer refactoring (no hidden side effects)
- Better caching opportunities (query is pure)

#### Dependencies

- Blocked by: None (independent refactoring)
- Blocks: None
- Related: TICKET-001 (general code quality)

---

## Sprint 4: Performance, Testing & Documentation (Week 7-8)

### TICKET-010: Comprehensive Performance Benchmarking

**Priority**: P1
**Effort**: 5 story points (confidence: medium)

#### Context

Validate that the architectural migration does not degrade performance. Establish baseline metrics and verify improvements from reduced indirection and duplication.

#### Acceptance Criteria

- [ ] Benchmarks for lag buffer updates (before and after)
- [ ] Benchmarks for constraint updates (before and after)
- [ ] Benchmarks for prepare_from_trajectory (before and after)
- [ ] Memory usage profiling (before and after)
- [ ] No performance regression >5%
- [ ] Document any performance improvements

#### Tasks

##### Implementation
- [ ] Create benchmark suite in `benches/uncertainty_migration.rs`
- [ ] Implement benchmark: `bench_lag_buffer_update`
- [ ] Implement benchmark: `bench_lag_constraint_update`
- [ ] Implement benchmark: `bench_uncertainty_constraint_update`
- [ ] Implement benchmark: `bench_prepare_from_trajectory`
- [ ] Create test fixtures with varying system sizes (10, 50, 100, 500 entities)
- [ ] Implement memory profiling utilities

##### Testing
- [ ] Run benchmarks on old architecture (baseline)
- [ ] Run benchmarks on new architecture
- [ ] Compare results with statistical significance tests
- [ ] Profile memory allocations
- [ ] Test with different AR orders (1, 5, 10)
- [ ] Test cache-friendliness of new data structures

##### Documentation
- [ ] Create PERFORMANCE_REPORT.md with results
- [ ] Document benchmark methodology
- [ ] Add performance notes to CHANGELOG.md
- [ ] Update README.md if significant improvements
- [ ] Document any unexpected performance characteristics

#### Technical Notes

**Benchmark Structure**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_lag_buffer_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("lag_buffer_update");
    
    for size in [10, 50, 100, 500].iter() {
        let (subproblem, trajectory) = create_test_case(*size);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    subproblem.update_load_lag_buffer(black_box(&trajectory));
                });
            },
        );
    }
    
    group.finish();
}
```

**Expected Results**:
- Lag buffer update: Similar or faster (less indirection)
- Constraint update: Similar or faster (better cache locality)
- Memory usage: ~10-20% reduction (removed redundant structures)

**Metrics to Track**:
- Execution time (mean, median, std dev)
- Memory allocations (count and size)
- Cache misses (if available)
- Heap fragmentation

#### Dependencies

- Blocked by: TICKET-006 (core migration complete)
- Blocks: TICKET-011
- Related: None

---

### TICKET-011: Update All Documentation and Examples

**Priority**: P1
**Effort**: 5 story points (confidence: high)

#### Context

Ensure all documentation reflects the new architecture. Update examples, architecture diagrams, and API documentation to match the migrated code.

#### Acceptance Criteria

- [ ] All doc comments updated for changed structs/methods
- [ ] Architecture diagrams updated to show separated structures
- [ ] README.md updated if architectural changes visible to users
- [ ] Examples compile and run correctly
- [ ] CHANGELOG.md comprehensively documents migration
- [ ] Migration guide completed

#### Tasks

##### Implementation
- [ ] Update doc comments in subproblem.rs (all changed structs)
- [ ] Update module-level documentation
- [ ] Create or update architecture diagrams
- [ ] Review and update examples/ directory
- [ ] Update any inline code examples in docs
- [ ] Generate and review rustdoc output

##### Documentation
- [ ] Update ARCHITECTURE.md with new structure
- [ ] Complete MIGRATION_GUIDE.md with lessons learned
- [ ] Update CHANGELOG.md with comprehensive notes
- [ ] Add migration notes to README.md if needed
- [ ] Create diagram: old vs. new data flow
- [ ] Document API changes (if any public API affected)
- [ ] Add "Upgrading from Previous Version" section if needed

##### Testing
- [ ] Verify all examples compile
- [ ] Run all examples and verify output
- [ ] Check rustdoc builds without warnings
- [ ] Verify all links in documentation work
- [ ] Test code examples in doc comments

#### Technical Notes

**Documentation Checklist**:
- [ ] `src/lib.rs` - module overview
- [ ] `src/subproblem.rs` - all changed types
- [ ] `docs/` - architecture documents
- [ ] `README.md` - high-level overview
- [ ] `CHANGELOG.md` - detailed change log
- [ ] `examples/` - working examples

**Diagram Updates Needed**:
1. Data structure relationships (before/after)
2. Data flow in prepare_from_trajectory (before/after)
3. Type relationships (LoadLagData, InflowLagData, etc.)

**CHANGELOG.md Entry Template**:
```markdown
## [Version X.Y.Z] - YYYY-MM-DD

### Changed - BREAKING
- **Architectural Migration**: Completed migration from unified to separated 
  uncertainty handling. This is an internal change with no user-facing API changes.
  - Removed `UncertaintyConstraintManager` module
  - Simplified `UncertaintyConstraintData` to `UncertaintyObservationData`
  - Separated load and inflow lag handling for type safety
  - Improved performance by ~X% due to reduced indirection

### Performance
- Lag buffer updates: X% faster
- Memory usage: Y% reduction
```

#### Dependencies

- Blocked by: TICKET-010 (need performance numbers)
- Blocks: None
- Related: All previous tickets

---

### TICKET-012: Final Integration Testing and Validation

**Priority**: P0
**Effort**: 5 story points (confidence: medium)

#### Context

Comprehensive end-to-end testing to ensure the migration is complete and correct. This includes numerical validation, edge cases, and real-world problem testing.

#### Acceptance Criteria

- [ ] All existing tests pass
- [ ] New migration-specific tests pass
- [ ] Numerical validation confirms identical results to baseline
- [ ] Edge cases handled correctly
- [ ] Real-world examples produce identical outputs
- [ ] No memory leaks or undefined behavior
- [ ] Code coverage maintained or improved

#### Tasks

##### Testing
- [ ] Run full test suite: `cargo test --all-features`
- [ ] Run integration tests with real problem data
- [ ] Numerical comparison: old vs. new architecture outputs
- [ ] Test edge case: systems with no uncertainty
- [ ] Test edge case: systems with only loads
- [ ] Test edge case: systems with only inflows
- [ ] Test edge case: zero lag order
- [ ] Test edge case: very large lag orders (>20)
- [ ] Test edge case: single entity systems
- [ ] Test edge case: very large systems (1000+ entities)
- [ ] Memory leak check: valgrind or similar
- [ ] Thread safety check: run with TSAN if applicable
- [ ] Fuzz testing: random inputs
- [ ] Stress testing: long-running SDDP simulations

##### Documentation
- [ ] Document all test cases in test files
- [ ] Create test report summarizing results
- [ ] Document any known issues or limitations
- [ ] Add regression test documentation

##### Code Quality
- [ ] Run clippy: `cargo clippy -- -D warnings`
- [ ] Run rustfmt: `cargo fmt --check`
- [ ] Check for TODO comments
- [ ] Verify all deprecation warnings addressed
- [ ] Code coverage report: `cargo tarpaulin`

#### Technical Notes

**Numerical Validation Strategy**:
```rust
#[test]
fn test_numerical_equivalence_old_vs_new() {
    // Load baseline results from old architecture
    let old_results = load_baseline_results("tests/fixtures/baseline.json");
    
    // Run same problem with new architecture
    let new_results = run_sddp_with_new_architecture();
    
    // Compare with tight tolerance
    for (old_value, new_value) in old_results.zip(new_results) {
        assert!((old_value - new_value).abs() < 1e-10,
                "Numerical difference detected: {} vs {}", old_value, new_value);
    }
}
```

**Test Coverage Targets**:
- Unit tests: >90%
- Integration tests: cover all major workflows
- Edge cases: all identified edge cases tested

**Success Criteria**:
- All tests pass ✅
- No performance regression >5% ✅
- Code coverage maintained ✅
- No memory leaks ✅
- Clippy/rustfmt clean ✅

#### Dependencies

- Blocked by: TICKET-011 (documentation complete)
- Blocks: None (final ticket)
- Related: All previous tickets

---


## Summary and Timeline

### Sprint Overview

| Sprint | Duration | Story Points | Focus |
|--------|----------|--------------|-------|
| Sprint 1 | Week 1-2 | 7 pts | Foundation & Documentation |
| Sprint 2 | Week 3-4 | 18 pts | Core Migration |
| Sprint 3 | Week 5-6 | 8 pts | Integration & Cleanup |
| Sprint 4 | Week 7-8 | 15 pts | Performance & Validation |
| **Total** | **8 weeks** | **48 pts** | **Complete Migration** |

### Ticket Dependencies Graph

```
TICKET-001 (Document)
    ↓
TICKET-002 (Regression Tests)
    ↓
TICKET-003 (New Structures)
    ↓
    ├─→ TICKET-004 (Remove Unified)
    │       ↓
    └─→ TICKET-005 (Replace Manager)
            ↓
        TICKET-006 (Refine Data)
            ↓
            ├─→ TICKET-007 (Remove Module)
            │       ↓
            ├─→ TICKET-008 (Optimize) [Optional]
            │       ↓
            └─→ TICKET-010 (Benchmarks)
                    ↓
                TICKET-011 (Documentation)
                    ↓
                TICKET-012 (Final Validation)

TICKET-009 (CQS) [Independent, can run anytime]
```

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical differences | Low | High | Comprehensive regression tests (TICKET-002) |
| Performance degradation | Low | Medium | Benchmarking throughout (TICKET-010) |
| Breaking existing code | Medium | High | Gradual migration, feature flags if needed |
| Timeline overrun | Medium | Medium | Optional tickets (TICKET-008) can be deferred |
| Incomplete documentation | Low | Medium | Explicit documentation tickets (TICKET-001, 011) |

### Success Criteria

**Must Have (P0-P1)**:
- ✅ All existing tests pass
- ✅ No numerical differences from baseline
- ✅ No performance regression >5%
- ✅ Complete documentation updated
- ✅ All parallel data structures removed
- ✅ Type-safe separated structures in place

**Nice to Have (P2-P3)**:
- ✅ Direct extraction optimization (TICKET-008)
- ✅ Command-Query Separation (TICKET-009)
- ✅ Performance improvements >10%
- ✅ Memory usage reduction >15%

### Rollback Plan

If critical issues discovered during migration:

1. **Before TICKET-004**: Easy rollback, only added code
2. **After TICKET-004**: Revert commits, restore from baseline
3. **After TICKET-006**: Feature flag to switch between old/new
4. **After TICKET-007**: Must complete migration (module removed)

### Monitoring Metrics

Track throughout migration:

- **Code Metrics**:
  - Lines of code changed
  - Code coverage percentage
  - Cyclomatic complexity
  - Number of `unsafe` blocks

- **Performance Metrics**:
  - Lag buffer update time (ms)
  - Constraint update time (ms)
  - Memory allocation count
  - Peak memory usage (MB)

- **Quality Metrics**:
  - Test pass rate
  - Clippy warnings count
  - Documentation coverage
  - Review comments per PR

### Communication Plan

**Weekly Updates**:
- Sprint progress report (tickets completed)
- Risk assessment updates
- Blocker identification
- Performance metrics dashboard

**Milestone Announcements**:
- Sprint 1 complete: "Baseline established"
- Sprint 2 complete: "Core migration done"
- Sprint 3 complete: "Cleanup complete"
- Sprint 4 complete: "Migration validated"

### Post-Migration Cleanup

After all tickets complete:

- [ ] Remove baseline test data (if temporary)
- [ ] Archive migration-specific documentation
- [ ] Create "lessons learned" document
- [ ] Update project roadmap
- [ ] Plan follow-up optimization tickets
- [ ] Celebrate! 🎉

---

## Appendix A: Key Code Locations

Reference for reviewers and implementers:

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| UncertaintyConstraintData | `src/subproblem.rs` | 65-103 | To be refactored |
| Separated lag structures | `src/subproblem.rs` | 145-370 | Already exists (good) |
| Subproblem fields | `src/subproblem.rs` | 502-508 | Contains both old and new |
| Parallel population | `src/subproblem.rs` | 1806-1843 | To be removed |
| build_entity_constraint_data | `src/subproblem.rs` | 1962-2006 | Builds constraint data |
| update_uncertainty_constraints | `src/subproblem.rs` | 2021-2032 | Uses entity_data |
| update_lag_fixing_constraints | `src/subproblem.rs` | 2045-2140 | Uses both structures |
| UncertaintyConstraintManager | `src/uncertainty_constraints.rs` | 1-200 | To be removed |

## Appendix B: Testing Strategy Detail

### Regression Test Categories

1. **Structural Tests** (verify data structure correctness)
   - Constraint index consistency
   - Buffer allocation correctness
   - Type safety compilation tests

2. **Behavioral Tests** (verify computational correctness)
   - Lag extraction accuracy
   - Constraint RHS computation
   - Numerical stability

3. **Performance Tests** (verify no degradation)
   - Execution time benchmarks
   - Memory usage profiling
   - Cache hit rate analysis

4. **Integration Tests** (verify end-to-end flows)
   - Full SDDP simulation runs
   - Multi-stage problem solving
   - Real-world problem cases

### Test Data Requirements

- **Small systems**: 5 buses, 3 hydros, lag order 2 (for fast iteration)
- **Medium systems**: 50 buses, 20 hydros, lag order 5 (for realistic testing)
- **Large systems**: 500 buses, 100 hydros, lag order 10 (for stress testing)
- **Edge cases**: Single entity, zero lags, very high lags (>20)

### Numerical Validation Approach

Use reference implementations or baseline results:
```rust
// Save baseline before migration
let baseline = run_with_old_architecture(problem);
save_to_file("baseline.json", baseline);

// After migration, compare
let new_results = run_with_new_architecture(problem);
let baseline = load_from_file("baseline.json");
assert_numerically_equal(baseline, new_results, tolerance=1e-10);
```

---

## Appendix C: Architecture Diagrams

### Current Architecture (Before Migration)

```
┌─────────────────────────────────────────────────────┐
│ Subproblem                                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  uncertainty_manager: UncertaintyConstraintManager  │
│  ├─ dimension: usize (loads + inflows)             │
│  ├─ lag_buffer: UnifiedLagBuffer                   │
│  │   └─ data: Vec<f64> (global indexing)          │
│  └─ max_lag: usize                                 │
│                                                     │
│  entity_data: Vec<UncertaintyConstraintData>       │
│  └─ [                                              │
│      {entity_type, entity_id, global_entity_idx,   │
│       constraint_idx, seasonal_std, ...},          │
│      ...                                           │
│    ]                                               │
│                                                     │
│  constraints:                                      │
│  ├─ lag_fixing_constraints: Vec<Vec<usize>>        │ ← Unified
│  ├─ load_lag_constraints: LoadLagConstraints      │ ← Separated
│  └─ inflow_lag_constraints: InflowLagConstraints  │ ← Separated
│                                                     │
│  ⚠️  PROBLEM: Parallel structures must sync!       │
└─────────────────────────────────────────────────────┘
```

### Target Architecture (After Migration)

```
┌─────────────────────────────────────────────────────┐
│ Subproblem                                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  load_lag_data: Option<LoadLagData>                │
│  ├─ variables: LoadLagVariables                    │
│  ├─ constraints: LoadLagConstraints                │
│  └─ buffer: Vec<Vec<f64>> [bus_id][lag_idx]       │
│                                                     │
│  inflow_lag_data: Option<InflowLagData>            │
│  ├─ variables: InflowLagVariables                  │
│  ├─ constraints: InflowLagConstraints              │
│  └─ buffer: Vec<Vec<f64>> [hydro_id][lag_idx]     │
│                                                     │
│  uncertainty_observation_data:                     │
│    Vec<UncertaintyObservationData>                 │
│  └─ [                                              │
│      {constraint_idx, innovation_idx,              │
│       seasonal_std, deterministic_base},           │
│      ...                                           │
│    ]                                               │
│                                                     │
│  ✅  CLEAN: Single source of truth per concern     │
│  ✅  TYPE-SAFE: Can't confuse bus_id/hydro_id     │
│  ✅  CLEAR: Each structure has single purpose      │
└─────────────────────────────────────────────────────┘
```

---

## Appendix D: Effort Estimation Details

### Story Point Calibration

**1 Point** = ~0.5 day (4 hours)
- Simple struct definition
- Basic test writing
- Documentation updates

**2 Points** = ~1 day (8 hours)
- Moderate refactoring
- Multiple test scenarios
- Doc updates with examples

**3 Points** = ~1.5 days (12 hours)
- Complex refactoring
- Integration across modules
- Comprehensive testing

**5 Points** = ~2-3 days (16-24 hours)
- Major architectural change
- Cross-cutting concerns
- Extensive test coverage

**8 Points** = ~3-5 days (24-40 hours)
- Core algorithm changes
- Multiple module impacts
- Performance validation needed

### Confidence Levels

**High Confidence**: Well-understood problem, clear solution, minimal risk
**Medium Confidence**: Some unknowns, may need iteration, moderate risk
**Low Confidence**: Exploratory work, solution unclear, high risk

### Buffer and Contingency

- **Per-ticket buffer**: 20% added to estimates (e.g., 5 pts → 6 pts actual)
- **Per-sprint buffer**: 2-3 pts reserved for unexpected issues
- **Total contingency**: ~25-30% above estimated effort

---

## Appendix E: Review Checklist

For each pull request during migration:

### Code Review
- [ ] Changes match ticket acceptance criteria
- [ ] No unintended side effects
- [ ] Error handling appropriate
- [ ] No performance regressions
- [ ] Memory safety verified
- [ ] Thread safety maintained (if applicable)

### Testing Review
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Edge cases covered
- [ ] Regression tests included
- [ ] Performance benchmarks run

### Documentation Review
- [ ] Doc comments updated
- [ ] Examples compile and run
- [ ] CHANGELOG.md updated
- [ ] Migration guide updated if needed
- [ ] Architecture diagrams current

### Quality Review
- [ ] Clippy clean (no warnings)
- [ ] Rustfmt applied
- [ ] No TODO comments (or tracked as issues)
- [ ] Code coverage maintained
- [ ] No security issues introduced

---

**Document Version**: 1.0
**Last Updated**: 2025-11-06
**Owner**: @rogerio
**Status**: Ready for Sprint Planning

