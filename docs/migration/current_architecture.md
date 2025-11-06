# Current Architecture: Unified vs. Separated Uncertainty Handling

**Status**: BASELINE (Before Migration)  
**Date**: 2025-11-06  
**Ticket**: TICKET-001

## Overview

The codebase currently maintains **parallel data structures** for handling uncertainty constraints:
1. **Unified structures** - Old approach using global entity indexing
2. **Separated structures** - New approach with type-safe bus_id/hydro_id indexing

Both structures must be kept in sync manually, creating architectural debt.

## File Inventory

### Primary Files

| File | Lines | Purpose | Migration Impact |
|------|-------|---------|------------------|
| `src/uncertainty_constraints.rs` | 484 | UncertaintyConstraintManager module | **DELETE** in TICKET-007 |
| `src/subproblem.rs` | ~3500 | Subproblem with both structures | **REFACTOR** in TICKET-004, 005, 006 |
| `src/input.rs` | N/A | Uses UncertaintyConstraintManager | **UPDATE** imports only |
| `src/sddp/` | N/A | May use uncertainty types | **REVIEW** for usages |

## Data Structure Analysis

### 1. UncertaintyConstraintManager (Unified - TO BE REMOVED)

**Location**: `src/uncertainty_constraints.rs:162-263`

```rust
pub struct UncertaintyConstraintManager {
    dimension: usize,                      // loads + inflows count
    lag_buffer: UnifiedLagBuffer,          // Global entity indexing
    max_lag: usize,
    constraint_indices: Option<UncertaintyConstraintIndices>,
}
```

**Problems**:
- Uses global entity indexing (loads first, then inflows)
- Requires routing logic to map entity_id → global_entity_idx
- No compile-time distinction between bus_id and hydro_id
- Mixed concerns (buffer management + constraint tracking)

**Usage Sites**:
1. `src/subproblem.rs:502-503` - Field declaration
2. `src/subproblem.rs:2089, 2123` - `get_lag_observations()` calls
3. `src/input.rs` - Construction from temporal models

### 2. UnifiedLagBuffer (TO BE REMOVED)

**Location**: `src/uncertainty_constraints.rs:42-160`

```rust
pub struct UnifiedLagBuffer {
    data: Vec<f64>,        // Contiguous lag storage
    offsets: Vec<usize>,   // Entity → data offset mapping
    n_entities: usize,
}
```

**Memory Layout Example**:
```
Entities with lag counts [2, 0, 3, 1]:
Offsets:  [0, 2, 2, 5, 6]
Data:     [e0_lag0, e0_lag1, e2_lag0, e2_lag1, e2_lag2, e3_lag0]
```

**Design Intent**: Cache-friendly contiguous storage  
**Reality**: Adds complexity without measurable benefit (needs benchmarking)

### 3. UncertaintyConstraintData (TO BE SIMPLIFIED)

**Location**: `src/subproblem.rs:65-103`

```rust
pub struct UncertaintyConstraintData {
    entity_type: UncertaintyType,    // ❌ Routing field
    entity_id: usize,                // ❌ Routing field
    global_entity_idx: usize,        // ❌ Routing field
    constraint_idx: usize,           // ✅ Keep
    observation_var_idx: usize,      // ⚠️  Maybe keep
    innovation_var_idx: usize,       // ✅ Keep
    season_id: usize,                // ❌ Only for precomputation
    seasonal_mean: f64,              // ❌ Baked into deterministic_base
    seasonal_std: f64,               // ✅ Keep
    ar_order: usize,                 // ❌ Not needed at runtime
    psi_coefficients: Vec<f64>,      // ❌ Not needed at runtime (in LHS)
    deterministic_base: f64,         // ✅ Keep
}
```

**Current Size**: 12 fields (mixed concerns)  
**Target Size**: 4 fields (single concern: RHS update coefficients)

**Usage**: 
- Built in `build_entity_constraint_data()` (line 1962-2006)
- Used in `update_uncertainty_constraints()` (line 2021-2032)

### 4. Separated Structures (NEW - TO BE USED EXCLUSIVELY)

**Location**: `src/subproblem.rs:199-413`

#### LoadLagVariables
```rust
pub struct LoadLagVariables {
    pub lags_by_bus: Vec<Vec<usize>>,  // [bus_id][lag_idx] → var_idx
}
```

#### LoadLagConstraints
```rust
pub struct LoadLagConstraints {
    pub constraints_by_bus: Vec<Vec<usize>>,  // [bus_id][lag_idx] → con_idx
}
```

#### InflowLagVariables
```rust
pub struct InflowLagVariables {
    pub lags_by_hydro: Vec<Vec<usize>>,  // [hydro_id][lag_idx] → var_idx
}
```

#### InflowLagConstraints
```rust
pub struct InflowLagConstraints {
    pub constraints_by_hydro: Vec<Vec<usize>>,  // [hydro_id][lag_idx] → con_idx
}
```

**Benefits**:
- ✅ Type-safe indexing (can't confuse bus_id with hydro_id)
- ✅ Direct O(1) access without entity type routing
- ✅ Clear ownership (variables and constraints paired)
- ✅ Flexible lag orders per entity

**Status**: Already implemented but underutilized

## Parallel Population Anti-Pattern

### Location: `src/subproblem.rs:1806-1843`

**The Problem**: Constraints are populated in BOTH old and new structures simultaneously

```rust
// TICKET-002: Populate both old unified structure and new explicit structures
let (lag_fixing_constraints, load_lag_constraints, inflow_lag_constraints) = 
    if let Some(ref lag_vars) = variables.lagged_state {
        let mut old_constraints = Vec::new();  // ❌ Unified structure
        let mut new_load_constraints = LoadLagConstraints::new(...);  // ✅ Separated
        let mut new_inflow_constraints = InflowLagConstraints::new(...);  // ✅ Separated
        
        for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
            // ... create constraints ...
            
            // Store in OLD unified structure
            old_constraints.push(entity_constraints.clone());
            
            // Route to NEW separated structures
            match model.entity_type {
                Load => new_load_constraints[bus_id] = entity_constraints,
                Inflow => new_inflow_constraints[hydro_id] = entity_constraints,
            }
        }
        
        (Some(old_constraints), Some(new_load), Some(new_inflow))
    } else {
        (None, None, None)
    };
```

**Issues**:
1. Double allocation (clones entity_constraints)
2. Manual synchronization required
3. Entity type routing needed for new structures
4. Cannot get out of sync only because both are built together

**Migration Path**: Remove old_constraints entirely (TICKET-004)

## Constraint Update Patterns

### Current: update_lag_fixing_constraints()

**Location**: `src/subproblem.rs:2045-2141`

**Algorithm**:
1. Build entity maps: `bus_id → entity_idx` and `hydro_id → entity_idx`
2. Iterate over separated constraint structures
3. Use entity maps to lookup global_entity_idx
4. Get lag observations from UncertaintyConstraintManager (unified)
5. Update constraint RHS values

```rust
// Simplified current flow
for bus_id in load_constraints {
    let entity_idx = load_entity_map[bus_id];  // Map to global index
    let lags = uncertainty_manager.get_lag_observations(entity_idx);  // Unified buffer
    for (lag_idx, &con) in constraints.iter() {
        model.change_rows_bounds(con, lags[lag_idx], lags[lag_idx]);
    }
}
```

**Inefficiency**: Entity maps are built on EVERY call (but HashMap construction is cheap)

### Target: Direct Buffer Access (TICKET-005)

```rust
// Proposed: Direct access without routing
if let Some(ref load_data) = self.load_lag_data {
    for (bus_id, constraints) in load_data.constraints.iter() {
        for (lag_idx, &con) in constraints.iter() {
            let lag_value = load_data.buffer[bus_id][lag_idx];  // Direct!
            model.change_rows_bounds(con, lag_value, lag_value);
        }
    }
}
```

**Benefits**:
- No entity type routing
- No global_entity_idx mapping
- Direct buffer access
- Type system prevents bus_id/hydro_id confusion

## Data Flow: Current vs. Target

### Current Data Flow

```
Trajectory → prepare_from_trajectory()
                ↓
    Extract load observations (by bus_id)
    Extract inflow observations (by hydro_id)
                ↓
    Map to global_entity_idx (routing)
                ↓
    Store in UnifiedLagBuffer (via UncertaintyConstraintManager)
                ↓
    update_lag_fixing_constraints() called
                ↓
    Build entity_id → global_entity_idx maps
                ↓
    Iterate separated constraints (bus_id, hydro_id)
                ↓
    Map back to global_entity_idx (reverse routing)
                ↓
    Get lags from unified buffer
                ↓
    Update constraint RHS
```

**Complexity**: O(n_buses + n_hydros) map building + O(total_lags) updates

### Target Data Flow (After Migration)

```
Trajectory → prepare_from_trajectory()
                ↓
    update_load_lag_buffer_from_trajectory()
        → Direct bus_id indexing
        → Store in LoadLagData.buffer[bus_id][lag_idx]
                ↓
    update_inflow_lag_buffer_from_trajectory()
        → Direct hydro_id indexing
        → Store in InflowLagData.buffer[hydro_id][lag_idx]
                ↓
    update_lag_fixing_constraints() called
                ↓
    Iterate LoadLagData.constraints (by bus_id)
        → Access LoadLagData.buffer[bus_id] directly
        → Update constraint RHS
                ↓
    Iterate InflowLagData.constraints (by hydro_id)
        → Access InflowLagData.buffer[hydro_id] directly
        → Update constraint RHS
```

**Complexity**: O(total_lags) for buffer update + O(total_lags) for constraint update  
**Benefit**: No mapping overhead, clearer data flow, type-safe indexing

## Synchronization Points

### Where Structures Must Stay Synchronized

1. **Constraint Creation** (line 1806-1843)
   - Both old and new structures populated
   - Clone required to maintain both
   
2. **Buffer Updates** (indirectly)
   - `prepare_from_trajectory()` updates unified buffer
   - Separated structures don't have buffers yet (TICKET-005)
   
3. **Constraint Updates** (line 2045-2141)
   - Uses separated constraint indices
   - Reads from unified buffer
   - Maps between bus_id/hydro_id ↔ global_entity_idx

### Risk Assessment

**Low Risk**: Structures won't desync because:
- Built together in single code path
- Immutable after construction (no runtime modifications)
- Tests would catch desyncs immediately

**High Maintenance Burden**: 
- Future developers must understand dual system
- Changes require updating both structures
- Code is harder to reason about

## Testing Coverage

### Current Tests

**uncertainty_constraints.rs**:
- ✅ UnifiedLagBuffer creation and operations
- ✅ UncertaintyConstraintManager basic operations
- ✅ Mixed AR order systems

**subproblem.rs**:
- ⚠️  Limited explicit testing of lag constraint updates
- ⚠️  No tests verifying old/new structure consistency
- ⚠️  No numerical equivalence tests

### Testing Gaps (Addressed in TICKET-002)

1. **Consistency Tests**: Verify old and new structures contain same indices
2. **Numerical Tests**: Verify constraint RHS values are identical
3. **Edge Cases**: Zero lags, single entity, very large systems
4. **Performance Tests**: Benchmark constraint update speed

## Performance Characteristics

### Memory Usage

**Current** (both structures):
- UnifiedLagBuffer: `O(total_lags)` for data + `O(n_entities)` for offsets
- Separated constraints: `O(total_lags)` for indices
- Old unified constraints: `O(total_lags)` for indices (duplicate!)
- **Total**: ~2× memory for lag constraint indices

**Target** (separated only):
- Load buffers: `O(Σ load_lags)` 
- Inflow buffers: `O(Σ inflow_lags)`
- Separated constraints: `O(total_lags)`
- **Total**: 1× memory for constraint indices + type-safe buffers

### Cache Friendliness

**UnifiedLagBuffer**: Excellent (contiguous memory)  
**Separated Buffers**: Good (Vec<Vec<f64>> has indirection but small inner vecs)

**Tradeoff**: Type safety and clarity vs. ~5-10ns indirection cost  
**Decision**: Type safety wins (optimization can come later if needed)

## Migration Dependencies

### External Dependencies

- All temporal_model types remain unchanged
- System specification unchanged
- Solver interface unchanged
- SDDP algorithm logic unchanged

### Internal Dependencies

**Must Change**:
- `Subproblem` fields (remove uncertainty_manager, add separated buffers)
- `Constraints` struct (remove lag_fixing_constraints field)
- `build_constraints()` function (remove parallel population)
- `update_lag_fixing_constraints()` (use separated buffers directly)
- `prepare_from_trajectory()` (populate separated buffers)

**Will Change** (refinement):
- `UncertaintyConstraintData` → `UncertaintyObservationData` (simplify)
- `entity_data` field → `uncertainty_observation_data` (rename)

**Will Delete**:
- `src/uncertainty_constraints.rs` (entire module)
- `UncertaintyConstraintManager` struct
- `UnifiedLagBuffer` struct

## Key Insights

### Why This Migration is Safe

1. **Pure Refactoring**: No algorithm changes, only data structure changes
2. **Type Safety**: Compile-time guarantees prevent bus_id/hydro_id confusion
3. **Single Responsibility**: Each structure has one clear purpose
4. **Testable**: Can verify equivalence with comprehensive regression tests
5. **Gradual**: Can be done in stages with feature flags if needed

### Why This Migration is Important

1. **Code Clarity**: Intention is obvious, no entity type routing
2. **Maintainability**: Single source of truth for each concern
3. **Performance**: Removes mapping overhead (small but measurable)
4. **Foundation**: Enables future enhancements to uncertainty modeling
5. **Type Safety**: Compiler catches more errors at compile time

### Potential Pitfalls

1. **Testing Thoroughness**: Must verify numerical equivalence rigorously
2. **Performance Validation**: Must benchmark to ensure no regressions
3. **Edge Cases**: Zero lags, independent models, mixed systems
4. **Documentation**: Must update all docs to reflect new architecture
5. **Initial Conditions**: Ensure lag initialization still works correctly

## Conclusion

The current architecture maintains parallel data structures as a transitional state during the migration from unified to separated uncertainty handling. The old unified structures provide:
- ✅ Working implementation (don't break what works)
- ✅ Reference for testing (verify new matches old)

The new separated structures provide:
- ✅ Type safety (compile-time error detection)
- ✅ Clarity (obvious indexing, no routing)
- ✅ Maintainability (single source of truth)
- ✅ Performance (no mapping overhead)

**Recommendation**: Complete the migration as outlined in TICKET-002 through TICKET-012.

---

**Next Steps**: See `migration_steps.md` for detailed migration plan.
