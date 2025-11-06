# Target Architecture: Fully Separated Uncertainty Handling

**Status**: TARGET (After Migration)  
**Date**: 2025-11-06  
**Ticket**: TICKET-001

## Overview

The target architecture **completely eliminates unified structures** in favor of type-safe, separated structures for loads and inflows. This provides compile-time guarantees, clearer code, and better maintainability.

## Core Principles

1. **Type Safety**: Compiler prevents bus_id/hydro_id confusion
2. **Single Responsibility**: Each structure has one clear purpose
3. **Direct Access**: No entity type routing or mapping needed
4. **Explicit Ownership**: Clear who owns what data
5. **Performance**: No unnecessary indirection or mapping overhead

## New Data Structures

### 1. LoadLagData (NEW in TICKET-003)

**Purpose**: Type-safe container for all load lag-related data

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
        Self {
            variables: LoadLagVariables::new(n_buses),
            constraints: LoadLagConstraints::new(n_buses),
            buffer: vec![Vec::new(); n_buses],
            n_buses,
            max_lag,
        }
    }
    
    pub fn get_lag(&self, bus_id: usize, lag_idx: usize) -> f64 {
        self.buffer[bus_id][lag_idx]
    }
    
    pub fn set_lag(&mut self, bus_id: usize, lag_idx: usize, value: f64) {
        self.buffer[bus_id][lag_idx] = value;
    }
    
    pub fn update_from_trajectory(&mut self, trajectory: &[&Realization]) {
        for bus_id in 0..self.n_buses {
            let max_lag = self.buffer[bus_id].len();
            for lag_idx in 0..max_lag {
                let lookback = lag_idx + 1;
                if lookback < trajectory.len() {
                    let past_idx = trajectory.len() - 1 - lookback;
                    let lag_value = trajectory[past_idx].loads[bus_id];
                    self.buffer[bus_id][lag_idx] = lag_value;
                }
            }
        }
    }
}
```

**Key Features**:
- **Cohesion**: Variables, constraints, and buffer live together
- **Type Safety**: `bus_id` is distinct from `hydro_id` by context
- **Direct Access**: `buffer[bus_id][lag_idx]` - no routing needed
- **Flexible**: Each bus can have different lag orders

### 2. InflowLagData (NEW in TICKET-003)

**Purpose**: Type-safe container for all inflow lag-related data

```rust
/// Type-safe container for inflow lag variables, constraints, and observations.
/// Indexed by hydro_id to prevent confusion with load entities.
pub struct InflowLagData {
    /// Variables: [hydro_id][lag_idx] → LP variable index
    pub variables: InflowLagVariables,
    
    /// Constraints: [hydro_id][lag_idx] → LP constraint index
    pub constraints: InflowLagConstraints,
    
    /// Buffer: [hydro_id][lag_idx] → lag observation value
    /// Used for updating lag-fixing constraints
    pub buffer: Vec<Vec<f64>>,
    
    /// Number of hydros with uncertain inflows
    pub n_hydros: usize,
    
    /// Maximum lag order across all hydros
    pub max_lag: usize,
}

impl InflowLagData {
    pub fn new(n_hydros: usize, max_lag: usize) -> Self {
        Self {
            variables: InflowLagVariables::new(n_hydros),
            constraints: InflowLagConstraints::new(n_hydros),
            buffer: vec![Vec::new(); n_hydros],
            n_hydros,
            max_lag,
        }
    }
    
    pub fn get_lag(&self, hydro_id: usize, lag_idx: usize) -> f64 {
        self.buffer[hydro_id][lag_idx]
    }
    
    pub fn set_lag(&mut self, hydro_id: usize, lag_idx: usize, value: f64) {
        self.buffer[hydro_id][lag_idx] = value;
    }
    
    pub fn update_from_trajectory(&mut self, trajectory: &[&Realization]) {
        for hydro_id in 0..self.n_hydros {
            let max_lag = self.buffer[hydro_id].len();
            for lag_idx in 0..max_lag {
                let lookback = lag_idx + 1;
                if lookback < trajectory.len() {
                    let past_idx = trajectory.len() - 1 - lookback;
                    let lag_value = trajectory[past_idx].inflows[hydro_id];
                    self.buffer[hydro_id][lag_idx] = lag_value;
                }
            }
        }
    }
}
```

**Key Features**: Same as LoadLagData but for hydro entities

### 3. UncertaintyObservationData (SIMPLIFIED in TICKET-006)

**Purpose**: Precomputed coefficients for fast uncertainty constraint RHS updates

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

**Simplification**: From 12 fields → 4 fields
- ❌ Removed: `entity_type`, `entity_id`, `global_entity_idx` (routing)
- ❌ Removed: `ar_order`, `psi_coefficients` (not needed at runtime)
- ❌ Removed: `season_id`, `seasonal_mean` (only for precomputation)
- ❌ Removed: `observation_var_idx` (not needed for RHS update)
- ✅ Kept: Only what's needed for RHS update

**Why Keep Unified?**: 
Uncertainty observation constraints have identical update logic for loads and inflows:
```rust
for data in &self.uncertainty_observation_data {
    let innovation = innovations[data.innovation_idx];
    let rhs = data.deterministic_base + data.seasonal_std * innovation;
    model.change_rows_bounds(data.constraint_idx, rhs, rhs);
}
```
No benefit to separating this - the math is the same!

## Updated Subproblem Structure

### Before (Current)
```rust
pub struct Subproblem {
    // ... other fields ...
    
    // OLD: Unified approach (TO BE REMOVED)
    pub uncertainty_manager: UncertaintyConstraintManager,
    
    // Mixed: Both old and new (TRANSITION STATE)
    pub entity_data: Vec<UncertaintyConstraintData>,  // 12 fields each
    
    pub constraints: Constraints {
        lag_fixing_constraints: Option<Vec<Vec<usize>>>,  // OLD
        load_lag_constraints: Option<LoadLagConstraints>,  // NEW
        inflow_lag_constraints: Option<InflowLagConstraints>,  // NEW
        // ...
    },
}
```

### After (Target)
```rust
pub struct Subproblem {
    // ... other fields ...
    
    // NEW: Type-safe separated structures
    pub load_lag_data: Option<LoadLagData>,
    pub inflow_lag_data: Option<InflowLagData>,
    
    // SIMPLIFIED: Only RHS update coefficients
    pub uncertainty_observation_data: Vec<UncertaintyObservationData>,
    
    pub constraints: Constraints {
        // lag_fixing_constraints removed - indices now in LoadLagData/InflowLagData
        // load_lag_constraints removed - moved to LoadLagData
        // inflow_lag_constraints removed - moved to InflowLagData
        // ...
    },
}
```

**Benefits**:
- ✅ Clearer ownership (LoadLagData owns load lag data)
- ✅ No redundant structures
- ✅ Smaller memory footprint
- ✅ Type-safe access patterns

## Method Changes

### prepare_from_trajectory() - BEFORE
```rust
fn prepare_from_trajectory(&mut self, trajectory: &[&Realization]) {
    // Extract storage (unchanged)
    let storage = self.extract_storage_from_trajectory(trajectory);
    self.update_storage_constraints(&storage);
    
    // OLD: Update unified buffer via entity routing
    for (entity_idx, model) in temporal_models.enumerate() {
        match model.entity_type {
            Load => {
                let bus_id = model.entity_id;
                let obs = trajectory[...].loads[bus_id];
                self.uncertainty_manager.update_lag_buffer(entity_idx, obs);
            }
            Inflow => {
                let hydro_id = model.entity_id;
                let obs = trajectory[...].inflows[hydro_id];
                self.uncertainty_manager.update_lag_buffer(entity_idx, obs);
            }
        }
    }
    
    // Update constraints using unified buffer
    self.update_lag_fixing_constraints();
}
```

### prepare_from_trajectory() - AFTER
```rust
fn prepare_from_trajectory(&mut self, trajectory: &[&Realization]) {
    // Extract storage (unchanged)
    let storage = self.extract_storage_from_trajectory(trajectory);
    self.update_storage_constraints(&storage);
    
    // NEW: Direct update of separated buffers
    if let Some(ref mut load_data) = self.load_lag_data {
        load_data.update_from_trajectory(trajectory);
    }
    
    if let Some(ref mut inflow_data) = self.inflow_lag_data {
        inflow_data.update_from_trajectory(trajectory);
    }
    
    // Update constraints using separated buffers
    self.update_lag_fixing_constraints();
}
```

**Benefits**:
- ✅ No entity type routing
- ✅ Direct trajectory → buffer flow
- ✅ Each data structure handles its own updates
- ✅ Clearer separation of concerns

### update_lag_fixing_constraints() - BEFORE
```rust
fn update_lag_fixing_constraints(&mut self) {
    // Build entity maps for routing
    let load_entity_map: HashMap<usize, usize> = self.entity_data
        .iter()
        .filter(|d| d.entity_type == Load)
        .map(|d| (d.entity_id, d.global_entity_idx))
        .collect();
    
    let inflow_entity_map: HashMap<usize, usize> = self.entity_data
        .iter()
        .filter(|d| d.entity_type == Inflow)
        .map(|d| (d.entity_id, d.global_entity_idx))
        .collect();
    
    if let Some(model) = self.model.as_mut() {
        // Update load constraints
        if let Some(load_constraints) = &self.constraints.load_lag_constraints {
            for bus_id in 0..load_constraints.len() {
                let entity_idx = load_entity_map[&bus_id];  // Routing!
                let lags = self.uncertainty_manager.get_lag_observations(entity_idx);
                for (lag_idx, &con) in load_constraints[bus_id].iter() {
                    model.change_rows_bounds(con, lags[lag_idx], lags[lag_idx]);
                }
            }
        }
        
        // Similar for inflow constraints...
    }
}
```

### update_lag_fixing_constraints() - AFTER
```rust
fn update_lag_fixing_constraints(&mut self) {
    if let Some(model) = self.model.as_mut() {
        // Update load constraints - direct access!
        if let Some(ref load_data) = self.load_lag_data {
            for (bus_id, constraints) in load_data.constraints.constraints_by_bus.iter().enumerate() {
                for (lag_idx, &constraint_idx) in constraints.iter().enumerate() {
                    let lag_value = load_data.buffer[bus_id][lag_idx];  // Direct!
                    model.change_rows_bounds(constraint_idx, lag_value, lag_value);
                }
            }
        }
        
        // Update inflow constraints - direct access!
        if let Some(ref inflow_data) = self.inflow_lag_data {
            for (hydro_id, constraints) in inflow_data.constraints.constraints_by_hydro.iter().enumerate() {
                for (lag_idx, &constraint_idx) in constraints.iter().enumerate() {
                    let lag_value = inflow_data.buffer[hydro_id][lag_idx];  // Direct!
                    model.change_rows_bounds(constraint_idx, lag_value, lag_value);
                }
            }
        }
    }
}
```

**Benefits**:
- ✅ No HashMap construction
- ✅ No entity routing logic
- ✅ Direct buffer access
- ✅ Clearer iteration (bus_id or hydro_id, not entity_idx)
- ✅ Type system prevents confusion

## Data Flow Diagram

### Complete Flow: Scenario Realization → Constraint Update

```
┌─────────────────────────────────────────────────────┐
│ Forward Pass: realize_uncertainties()              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 1. Sample innovations from SAA                     │
│    innovations = [ε_load[0], ..., ε_inflow[0], ...]│
│                                                     │
│ 2. Update uncertainty observation constraints      │
│    for data in uncertainty_observation_data:       │
│        RHS = data.deterministic_base +             │
│              data.seasonal_std * innovations[i]    │
│        model.change_rows_bounds(...)               │
│                                                     │
│ 3. Update lag-fixing constraints                   │
│    → load_lag_data.update_constraints(model)       │
│    → inflow_lag_data.update_constraints(model)     │
│                                                     │
│ 4. Solve LP                                        │
│    solution = solver.solve()                       │
│                                                     │
│ 5. Extract realized observations                   │
│    loads = solution[load_vars]                     │
│    inflows = solution[inflow_vars]                 │
│                                                     │
│ 6. Update trajectory with realization              │
│    trajectory.push(Realization { loads, inflows }) │
│                                                     │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│ Next Stage: prepare_from_trajectory()              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 1. Update load lag buffer                          │
│    load_lag_data.update_from_trajectory():         │
│        for bus_id in 0..n_buses:                   │
│            for lag_idx in 0..ar_order:             │
│                lookback = lag_idx + 1              │
│                past_idx = len - 1 - lookback       │
│                buffer[bus_id][lag_idx] =           │
│                    trajectory[past_idx].loads[bus] │
│                                                     │
│ 2. Update inflow lag buffer                        │
│    inflow_lag_data.update_from_trajectory():       │
│        (similar, using hydro_id and inflows)       │
│                                                     │
│ 3. Update storage constraints                      │
│    (unchanged from current implementation)         │
│                                                     │
└─────────────────────────────────────────────────────┘
         ↓
    [Repeat for next stage]
```

## Type Safety Examples

### Compile-Time Error Prevention

```rust
// ❌ OLD: Can accidentally mix bus_id and hydro_id
let entity_idx = 5;  // Is this a bus or hydro? Who knows!
let lag = uncertainty_manager.get_lag_observations(entity_idx);

// ✅ NEW: Compiler enforces correct usage
let bus_id = 5;
let lag = load_lag_data.buffer[bus_id][0];  // Clearly a bus!

let hydro_id = 2;
let lag = inflow_lag_data.buffer[hydro_id][0];  // Clearly a hydro!

// This won't compile if you try to mix them:
let lag = load_lag_data.buffer[hydro_id][0];  // Context makes this wrong
```

### Iterator Type Safety

```rust
// ❌ OLD: Entity type checking at runtime
for (entity_idx, lags) in unified_lags.iter().enumerate() {
    let model = temporal_models[entity_idx];
    match model.entity_type {  // Runtime check!
        Load => { /* handle load */ },
        Inflow => { /* handle inflow */ },
    }
}

// ✅ NEW: Type determined by structure
for (bus_id, lags) in load_lag_data.buffer.iter().enumerate() {
    // Guaranteed to be a bus at compile time
}

for (hydro_id, lags) in inflow_lag_data.buffer.iter().enumerate() {
    // Guaranteed to be a hydro at compile time
}
```

## Performance Characteristics

### Memory Usage

**Current** (with both structures):
```
UncertaintyConstraintManager: 
    - UnifiedLagBuffer.data: 8 bytes × total_lags
    - UnifiedLagBuffer.offsets: 8 bytes × (n_entities + 1)
    - metadata: ~24 bytes

Separated Constraints:
    - LoadLagConstraints: 8 bytes × load_lags
    - InflowLagConstraints: 8 bytes × inflow_lags

Unified Constraints (OLD):
    - lag_fixing_constraints: 8 bytes × total_lags (DUPLICATE!)

UncertaintyConstraintData:
    - ~200 bytes × n_entities (12 fields, including Vec)

TOTAL: ~(24 + 200) bytes × n_entities + 24 bytes × total_lags
```

**Target** (separated only):
```
LoadLagData:
    - variables: 8 bytes × load_lags
    - constraints: 8 bytes × load_lags
    - buffer: 8 bytes × load_lags
    - metadata: ~24 bytes

InflowLagData:
    - variables: 8 bytes × inflow_lags
    - constraints: 8 bytes × inflow_lags
    - buffer: 8 bytes × inflow_lags
    - metadata: ~24 bytes

UncertaintyObservationData:
    - ~32 bytes × n_entities (4 fields, no Vec)

TOTAL: ~32 bytes × n_entities + 24 bytes × total_lags
```

**Savings**: ~170 bytes per entity + no duplicate constraint indices

### Execution Time

**Current**:
- `update_lag_fixing_constraints()`: 
  - HashMap construction: O(n_entities)
  - HashMap lookups: O(1) amortized
  - Buffer access: O(1)
  - Constraint updates: O(total_lags)
  - **Total**: O(n_entities + total_lags)

**Target**:
- `update_lag_fixing_constraints()`:
  - No HashMap construction
  - Direct buffer access: O(1)
  - Constraint updates: O(total_lags)
  - **Total**: O(total_lags)

**Savings**: Removes O(n_entities) HashMap construction (typically 10-100 entities)

### Cache Performance

**Current**: 
- UnifiedLagBuffer: Excellent (contiguous)
- Entity routing: Poor (HashMap lookups, cache misses)

**Target**:
- Separated buffers: Good (Vec<Vec<f64>>, minor indirection)
- No routing: Excellent (direct iteration)

**Net**: Slightly worse buffer layout, much better access pattern

## Migration Path Summary

### Phase 1: Foundation (TICKET-001, 002, 003)
1. Document current state
2. Create regression test suite
3. Implement LoadLagData and InflowLagData structures

### Phase 2: Core Migration (TICKET-004, 005, 006)
4. Remove unified lag_fixing_constraints
5. Replace UncertaintyConstraintManager with separated buffers
6. Simplify UncertaintyConstraintData to UncertaintyObservationData

### Phase 3: Cleanup (TICKET-007, 008, 009)
7. Remove uncertainty_constraints.rs module
8. (Optional) Extract lags directly without buffer
9. Refactor Command-Query Separation

### Phase 4: Validation (TICKET-010, 011, 012)
10. Comprehensive performance benchmarking
11. Update all documentation
12. Final integration testing

## Success Metrics

### Must Achieve
- ✅ All existing tests pass
- ✅ No numerical differences (tolerance < 1e-10)
- ✅ No performance regression > 5%
- ✅ Code coverage maintained (>90%)
- ✅ Clippy clean with `-D warnings`
- ✅ All docs updated

### Nice to Have
- ✅ Performance improvement > 10%
- ✅ Memory usage reduction > 15%
- ✅ Code reduction > 200 lines
- ✅ Zero compiler warnings

## Conclusion

The target architecture provides:

1. **Type Safety**: Compiler prevents entity confusion
2. **Clarity**: Obvious data flow, no routing logic
3. **Performance**: No unnecessary mapping overhead
4. **Maintainability**: Single source of truth per concern
5. **Foundation**: Enables future enhancements

**Key Insight**: The migration removes architectural debt while maintaining or improving performance and correctness.

---

**Next**: See `migration_steps.md` for step-by-step implementation guide.
