# Quick Reference: Explicit Lag Separation Implementation

This is a quick reference guide for developers implementing the explicit load/inflow lag separation epic.

## At a Glance

**Problem:** Unified lag storage loses type information → heuristic-based code → bugs  
**Solution:** Separate `LoadLagVariables` and `InflowLagVariables` → type-safe → correct  
**Impact:** Fixes critical bug, improves performance, prevents future issues

## Code Changes Summary

### Before (Unified - Buggy)

```rust
pub struct Variables {
    pub lagged_state: Option<Vec<Vec<usize>>>, // ❌ Mixed loads + inflows
}

// Usage requires filtering
for (entity_idx, lags) in lagged_state.iter().enumerate() {
    if entity_is_inflow(entity_idx) { // ❌ Heuristic needed
        // Process inflow...
    }
}
```

### After (Explicit - Correct)

```rust
pub struct Variables {
    pub load_lags: Option<LoadLagVariables>,    // ✅ Explicit loads
    pub inflow_lags: Option<InflowLagVariables>, // ✅ Explicit inflows
}

// Usage is direct and clear
for hydro_id in 0..n_hydros {
    let lags = inflow_lags.get_lags(hydro_id); // ✅ Direct access
    // Process inflow...
}
```

## Key Data Structures

### LoadLagVariables

```rust
#[derive(Clone, Debug)]
pub struct LoadLagVariables {
    pub lags_by_bus: Vec<Vec<usize>>, // [bus_id][lag_idx] → var_idx
}

impl LoadLagVariables {
    pub fn new(buses_count: usize) -> Self;
    pub fn get_lags(&self, bus_id: usize) -> &[usize];
    pub fn get_lag_var(&self, bus_id: usize, lag_idx: usize) -> usize;
    pub fn total_lag_count(&self) -> usize;
}
```

### InflowLagVariables

```rust
#[derive(Clone, Debug)]
pub struct InflowLagVariables {
    pub lags_by_hydro: Vec<Vec<usize>>, // [hydro_id][lag_idx] → var_idx
}

impl InflowLagVariables {
    pub fn new(hydros_count: usize) -> Self;
    pub fn get_lags(&self, hydro_id: usize) -> &[usize];
    pub fn get_lag_var(&self, hydro_id: usize, lag_idx: usize) -> usize;
    pub fn total_lag_count(&self) -> usize;
}
```

## Common Patterns

### Pattern 1: Iterate All Entities of One Type

```rust
// ❌ Before: Filter mixed entities
for (entity_idx, lags) in lagged_state.iter().enumerate() {
    if is_inflow(entity_idx) {
        process_inflow(lags);
    }
}

// ✅ After: Direct iteration
if let Some(inflow_lags) = &variables.inflow_lags {
    for hydro_id in 0..n_hydros {
        let lags = inflow_lags.get_lags(hydro_id);
        process_inflow(lags);
    }
}
```

### Pattern 2: Access Specific Entity's Lags

```rust
// ❌ Before: Find entity in mixed list
let entity_idx = find_entity_index(entity_id, entity_type);
let lags = &lagged_state[entity_idx];

// ✅ After: Direct access by ID
let lags = match entity_type {
    Load => load_lags.get_lags(bus_id),
    Inflow => inflow_lags.get_lags(hydro_id),
};
```

### Pattern 3: Add Cuts with Lag Coefficients

```rust
// ❌ Before: Heuristic matching (BUGGY!)
let mut hydro_count = 0;
for (entity_idx, lags) in lag_vars.iter().enumerate() {
    if lags.len() == expected_hydro_lag_count(hydro_count) {
        // Hope this is a hydro...
        for lag_var in lags {
            factors.push((lag_var, -coef[coef_idx]));
            coef_idx += 1;
        }
        hydro_count += 1;
    }
}

// ✅ After: Explicit hydro iteration (CORRECT!)
if let Some(inflow_lags) = &variables.inflow_lags {
    for hydro_id in 0..n_hydros {
        let lags = inflow_lags.get_lags(hydro_id);
        for &lag_var in lags {
            factors.push((lag_var, -coef[coef_idx]));
            coef_idx += 1;
        }
    }
}
```

### Pattern 4: Extract Duals from Solution

```rust
// ❌ Before: Extract then filter
let mut all_duals = Vec::new();
for entity_constraints in lag_constraints.iter() {
    let duals = extract_duals(entity_constraints);
    all_duals.push(duals);
}
let (load_duals, inflow_duals) = filter_by_type(all_duals);

// ✅ After: Extract directly by type
let mut load_duals = vec![Vec::new(); n_buses];
let mut inflow_duals = vec![Vec::new(); n_hydros];

if let Some(load_constraints) = &self.load_lag_constraints {
    for bus_id in 0..n_buses {
        load_duals[bus_id] = load_constraints.get_constraints(bus_id)
            .iter()
            .map(|&idx| solution.rowdual[idx])
            .collect();
    }
}

// Similar for inflow_duals...
```

## Migration Checklist

When migrating code to use explicit structures:

- [ ] Replace `lagged_state` iteration with separate `load_lags`/`inflow_lags` loops
- [ ] Replace entity_idx with bus_id or hydro_id
- [ ] Remove entity type checking/filtering logic
- [ ] Use direct access methods: `get_lags()`, `get_lag_var()`
- [ ] Update error messages to reference explicit entity types
- [ ] Add tests for both load and inflow cases separately
- [ ] Verify no heuristics or assumptions about entity ordering
- [ ] Check that AR(0) entities (no lags) are handled correctly

## Testing Patterns

### Test Both Entity Types Separately

```rust
#[test]
fn test_load_lags() {
    let load_lags = LoadLagVariables::new(3);
    // Test load-specific behavior...
}

#[test]
fn test_inflow_lags() {
    let inflow_lags = InflowLagVariables::new(2);
    // Test inflow-specific behavior...
}
```

### Test Mixed AR Orders (The Bug Trigger)

```rust
#[test]
fn test_mixed_ar_orders() {
    // System where both loads and inflows have AR(1)
    let system = SystemBuilder::new()
        .add_load(bus_id: 0, ar_order: 1)
        .add_inflow(hydro_id: 0, ar_order: 1)
        .build();
    
    let subproblem = create_subproblem(&system);
    
    let load_lag = subproblem.variables.load_lags.get_lag_var(0, 0);
    let inflow_lag = subproblem.variables.inflow_lags.get_lag_var(0, 0);
    
    // These MUST be different!
    assert_ne!(load_lag, inflow_lag);
}
```

### Test Empty Cases (AR(0))

```rust
#[test]
fn test_no_lags() {
    let system = create_system_with_ar0_models();
    let subproblem = create_subproblem(&system);
    
    // No lags → fields should be None
    assert!(subproblem.variables.load_lags.is_none());
    assert!(subproblem.variables.inflow_lags.is_none());
}
```

## Performance Tips

### ✅ Do: Use Direct Access

```rust
// Fast: O(1) access
let lag = inflow_lags.get_lag_var(hydro_id, lag_idx);
```

### ❌ Don't: Recreate Entity Mapping

```rust
// Slow: Recreating what we just removed
let entity_type = determine_type(entity_id); // ❌ Bad!
match entity_type {
    Load => load_lags.get_lags(entity_id),
    Inflow => inflow_lags.get_lags(entity_id),
}
```

### ✅ Do: Iterate Once Per Type

```rust
// Fast: Two simple loops
for bus_id in 0..n_buses { /* process loads */ }
for hydro_id in 0..n_hydros { /* process inflows */ }
```

### ❌ Don't: Mix Types in Single Loop

```rust
// Slow: Introduces branching
for entity_id in 0..n_entities {
    if is_load(entity_id) { /* ... */ }
    else { /* ... */ }
}
```

## Common Mistakes

### ❌ Mistake 1: Using entity_idx instead of entity_id

```rust
// Wrong: entity_idx is position in mixed list
let lags = inflow_lags.lags_by_hydro[entity_idx]; // ❌

// Right: use hydro_id which is the actual ID
let lags = inflow_lags.lags_by_hydro[hydro_id]; // ✅
```

### ❌ Mistake 2: Assuming entity order

```rust
// Wrong: Assumes entities are ordered
let first_entity_lags = lagged_state[0]; // Is this load or inflow? ❌

// Right: Access by explicit type and ID
let bus_0_lags = load_lags.lags_by_bus[0]; // ✅
let hydro_0_lags = inflow_lags.lags_by_hydro[0]; // ✅
```

### ❌ Mistake 3: Forgetting to handle None cases

```rust
// Wrong: Assumes lags always exist
let lags = variables.inflow_lags.lags_by_hydro[0]; // ❌ Panic if None!

// Right: Check Option first
if let Some(inflow_lags) = &variables.inflow_lags {
    let lags = inflow_lags.get_lags(0); // ✅
}
```

## Validation During Migration

Enable validation feature to catch migration bugs:

```bash
# In Cargo.toml
[features]
migration_validation = []

# Run tests with validation
cargo test --features migration_validation
```

This validates that old and new structures contain identical data during transition.

## Quick Wins

After migrating a function:
1. ✅ Remove entity type checking → simpler code
2. ✅ Remove entity index mapping → fewer bugs
3. ✅ Remove filtering loops → better performance
4. ✅ Add clear error messages → easier debugging

## When In Doubt

1. **Ask:** Is this a load or an inflow?
2. **Use:** The appropriate explicit structure
3. **Access:** By the entity's actual ID (bus_id or hydro_id)
4. **Test:** Both types separately

## Resources

- Full architecture analysis: `docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md`
- Detailed tickets: `tickets/TICKET-00X-*.md`
- Bug report: `BUG_FIX_PAR_LOWER_BOUND.md`

---

**Remember:** The goal is to make the code's structure match the problem's structure. Loads ≠ Inflows mathematically, so Loads ≠ Inflows in the code!
