# Architecture Deep Analysis: Unified vs. Separated Uncertainty Handling

**Date:** 2025-11-06  
**Reviewer:** AI Code Review Agent  
**Requested by:** @rogerio  
**Focus:** Legacy unified architecture remnants and separation concerns

---

## Executive Summary

🔴 **CRITICAL ARCHITECTURAL ISSUE IDENTIFIED**

Your observations reveal a **fundamental architectural inconsistency**: The codebase is caught **halfway through a migration** from a unified uncertainty architecture to a separated loads/inflows architecture. This creates:

1. **❌ Redundant Data Structures**: Three different ways to track the same information
2. **❌ Unclear Responsibilities**: `UncertaintyConstraintManager` and `UncertaintyConstraintData` have overlapping, confused roles
3. **❌ Maintenance Burden**: Parallel structures must be kept in sync (see lines 1806-1828)
4. **❌ Performance Waste**: Unnecessary indirection and data duplication

**Your diagnosis is 100% correct.** This is not just a naming issue - it's incomplete architectural migration that needs resolution.

---

## The Three Parallel Data Structures Problem

### Current Reality: Triple Redundancy

The codebase maintains **THREE separate tracking systems** for lag-related information:

#### 1. **`UncertaintyConstraintManager`** (uncertainty_constraints.rs)
```rust
pub struct UncertaintyConstraintManager {
    dimension: usize,                      // Total entities count
    lag_buffer: UnifiedLagBuffer,          // Stores lag observations
    max_lag: usize,                        // Max AR order
    constraint_indices: Option<...>,       // Constraint indices
}

// Unified lag buffer with global entity indexing
pub struct UnifiedLagBuffer {
    data: Vec<f64>,        // [load0_lag0, load0_lag1, ..., inflow0_lag0, ...]
    offsets: Vec<usize>,   // Offsets for each entity
    n_entities: usize,     // loads + inflows
}
```

**Purpose (original):** Unified handling of all uncertain entities  
**Status:** 🟡 Still used but conflicts with new separation

#### 2. **Separate Lag Structures** (subproblem.rs:145-370)
```rust
// NEW separated structures
pub struct LoadLagVariables {
    pub lags_by_bus: Vec<Vec<usize>>,  // [bus_id][lag_idx] → var_idx
}

pub struct InflowLagVariables {
    pub lags_by_hydro: Vec<Vec<usize>>,  // [hydro_id][lag_idx] → var_idx
}

pub struct LoadLagConstraints {
    pub constraints_by_bus: Vec<Vec<usize>>,  // [bus_id][lag_idx] → constraint_idx
}

pub struct InflowLagConstraints {
    pub constraints_by_hydro: Vec<Vec<usize>>,  // [hydro_id][lag_idx] → constraint_idx
}
```

**Purpose (new):** Type-safe separation of loads vs. inflows  
**Status:** ✅ Clean design, should be the future

#### 3. **`UncertaintyConstraintData`** (subproblem.rs:65-103)
```rust
pub struct UncertaintyConstraintData {
    entity_type: UncertaintyType,      // Load or Inflow (redundant!)
    entity_id: usize,                  // bus_id or hydro_id (ambiguous!)
    global_entity_idx: usize,          // For unified indexing
    constraint_idx: usize,             // LP constraint index
    observation_var_idx: usize,        // LP variable index
    innovation_var_idx: usize,         // LP variable index
    season_id: usize,                  // Current season
    seasonal_mean: f64,                // μ_s
    seasonal_std: f64,                 // σ_s
    ar_order: usize,                   // p
    psi_coefficients: Vec<f64>,        // [ψ_1, ψ_2, ..., ψ_p]
    deterministic_base: f64,           // μ_s - Σ(φ_k·μ_{s-k})
}
```

**Purpose (hybrid):** Fast constraint updates with precomputed coefficients  
**Status:** 🔴 Confused - mixes unified and separated concerns

### The Smoking Gun: Parallel Population

**Lines 1806-1843** reveal the architectural confusion:

```rust
// Create lag-fixing constraints for explicit lag variables
// TICKET-002: Populate both old unified structure and new explicit structures
let (lag_fixing_constraints, load_lag_constraints, inflow_lag_constraints) = 
    if let Some(ref lag_vars) = variables.lagged_state {
        let mut old_constraints = Vec::new();           // OLD unified
        let mut new_load_constraints = ...;             // NEW separated
        let mut new_inflow_constraints = ...;           // NEW separated
        
        for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
            // ... create constraints ...
            
            // Store in old unified structure
            old_constraints.push(entity_constraints.clone());  // ❌ Duplicate!
            
            // Route to appropriate new structure based on entity type
            let model = &temporal_models[entity_idx];
            match model.entity_type {
                Load => new_load_constraints.constraints_by_bus[bus_id] = ...,
                Inflow => new_inflow_constraints.constraints_by_hydro[hydro_id] = ...,
            }
        }
    }
```

**This is maintaining TWO parallel constraint index structures!**

---

## Issue #1: `UncertaintyConstraintManager` - Unclear Role

### Original Intent (Before Separation)

The `UncertaintyConstraintManager` was designed for **unified handling**:
- Single lag buffer for all entities
- Global entity indexing (loads first, then inflows)
- Constraint indices for all entities together

### Current Usage: Confused Identity

**Where it's used:**

1. **Lag Buffer Storage** (lines 688-738):
```rust
fn update_lag_buffers_from_trajectory(&mut self, trajectory: &[&Realization]) {
    for data in &self.entity_data {  // Uses unified entity_data
        // Extract lags from trajectory
        self.uncertainty_manager.set_initial_lags(data.global_entity_idx, &lags);
    }
}
```

2. **Lag Constraint Updates** (lines 2088-2090):
```rust
let lag_obs = self.uncertainty_manager.get_lag_observations(entity_idx);
model.change_rows_bounds(constraint_idx, lag_obs[lag_idx], lag_obs[lag_idx]);
```

3. **NOT used for**: Actual constraint RHS computation! (That uses `entity_data`)

### The Problem

**`UncertaintyConstraintManager` is now just a lag buffer wrapper** but:
- It pretends to be a "constraint manager" (misleading name)
- It uses global entity indexing (conflicts with separated design)
- It's accessed indirectly through `entity_data.global_entity_idx` (unnecessary indirection)

### What It Should Be

If we're committed to load/inflow separation:

```rust
// Option A: Separate lag buffers (consistent with new design)
pub struct LoadLagBuffer {
    data_by_bus: Vec<Vec<f64>>,  // [bus_id][lag_idx] → lag_value
}

pub struct InflowLagBuffer {
    data_by_hydro: Vec<Vec<f64>>,  // [hydro_id][lag_idx] → lag_value
}

// In Subproblem:
pub struct Subproblem {
    load_lag_buffer: Option<LoadLagBuffer>,
    inflow_lag_buffer: Option<InflowLagBuffer>,
    // Remove: uncertainty_manager
}
```

**OR**

```rust
// Option B: Keep unified buffer but clarify purpose
pub struct LagObservationBuffer {  // Renamed from UncertaintyConstraintManager
    buffer: UnifiedLagBuffer,
    // Remove all constraint-related fields
}
```

---

## Issue #2: `UncertaintyConstraintData` - Mixed Responsibilities

### What It Actually Contains

Looking at the fields (lines 65-103):

**1. Entity Identification (for unified indexing):**
- `entity_type: UncertaintyType` ← Why? We have separated structures!
- `entity_id: usize` ← Ambiguous: bus_id or hydro_id?
- `global_entity_idx: usize` ← Only for unified buffer access

**2. LP Structure Indices:**
- `constraint_idx: usize`
- `observation_var_idx: usize`
- `innovation_var_idx: usize`

**3. Precomputed Coefficients (for fast RHS computation):**
- `season_id: usize`
- `seasonal_mean: f64`
- `seasonal_std: f64`
- `ar_order: usize`
- `psi_coefficients: Vec<f64>`
- `deterministic_base: f64`

### The Confusion

This structure mixes **THREE different concerns**:

1. **Entity routing** (Load vs. Inflow) ← Should be handled by type system
2. **LP structure** (variable/constraint indices) ← Could be in separated structures
3. **Coefficient computation** (seasonal parameters) ← This is the real value!

### Where It's Used

**Only place:** `update_uncertainty_constraints()` (lines 2021-2032):

```rust
fn update_uncertainty_constraints(&mut self, innovations: &[f64]) {
    for data in &self.entity_data {
        let innovation = innovations[data.global_entity_idx];  // ← Unified indexing
        let stochastic_term = data.seasonal_std * innovation;
        let rhs = data.deterministic_base + stochastic_term;
        
        model.change_rows_bounds(data.constraint_idx, rhs, rhs);
    }
}
```

**Analysis:**
- Uses `global_entity_idx` for innovations array ← Unified design
- Uses precomputed `seasonal_std` and `deterministic_base` ← This is valuable!
- Doesn't distinguish Load vs. Inflow ← Good! They're handled uniformly
- But conflicts with separated lag constraints approach

### What It Should Be

The **only real value** in `UncertaintyConstraintData` is the **precomputed coefficients** for fast RHS updates.

```rust
// Option A: Keep coefficient data, remove routing
pub struct PrecomputedConstraintCoefficients {
    constraint_idx: usize,          // Which constraint to update
    seasonal_std: f64,              // σ for this entity in this season
    deterministic_base: f64,        // μ - Σ(φ_k·μ_{s-k})
    // Remove: entity_type, entity_id, global_entity_idx
}

// Separate by type
pub struct LoadConstraintCoefficients {
    coefficients_by_bus: Vec<PrecomputedConstraintCoefficients>,
}

pub struct InflowConstraintCoefficients {
    coefficients_by_hydro: Vec<PrecomputedConstraintCoefficients>,
}
```

**OR**

```rust
// Option B: Keep unified for uncertainty constraints (they ARE uniform)
// But separate lag handling (which is NOT uniform)
pub struct UncertaintyObservationData {
    constraint_idx: usize,
    innovation_idx: usize,  // Index in innovations array
    seasonal_std: f64,
    deterministic_base: f64,
    // This can stay unified - uncertainty constraints are the same for loads/inflows!
}
```

---

## Issue #3: Lag Buffer Redundancy (From Previous Analysis)

### The Flow Today

```
prepare_from_trajectory():
  ├─ update_lag_buffers_from_trajectory()
  │   └─ Extract lags from trajectory → Store in uncertainty_manager
  │
  ├─ state.extract_storage_from_trajectory()
  │   └─ Extract storage from trajectory → Return directly
  │
  └─ update_lag_fixing_constraints()
      └─ Read lags from uncertainty_manager → Update model

Why extract lags to buffer, then read from buffer?
Storage goes directly from trajectory → model update!
```

### The Inconsistency

**For storage:**
```rust
let storage = self.state.extract_storage_from_trajectory(trajectory);
self.update_storage_constraints(&storage);
// Direct: trajectory → extracted value → model update
```

**For lags:**
```rust
self.update_lag_buffers_from_trajectory(trajectory);  // trajectory → buffer
self.update_lag_fixing_constraints();                  // buffer → model
// Indirect: trajectory → buffer → model update
```

### Why This Happened

The lag buffer is **actually needed** for `update_uncertainty_constraints()`:

```rust
// During realize_and_solve(), uncertainty constraints need lag values:
// Y_t = deterministic_base + σ·η + Σ[ψ_k · Y_{t-k}]
//                                     ↑ These come from lag buffer
```

**But** for lag-fixing constraints (Y_{t-k} = constant), the buffer is redundant.

---

## The Root Cause: Incomplete Migration

### The History (Based on Code Evidence)

**Phase 1: Original Unified Design**
- Single `UncertaintyConstraintManager` for all entities
- Global entity indexing
- `entity_data: Vec<UncertaintyConstraintData>` with unified iteration
- Works well for **uncertainty observation constraints** (they're uniform!)

**Phase 2: Separation Attempt (Partial)**
- Added `LoadLagVariables`, `InflowLagVariables` (good!)
- Added `LoadLagConstraints`, `InflowLagConstraints` (good!)
- **BUT kept** unified `uncertainty_manager` and `entity_data`
- **Result:** Parallel structures that must be kept in sync

**Phase 3: Current State (Halfway)**
- New separated structures exist but aren't used consistently
- Old unified structures still in place
- Code maintains BOTH (see lines 1806-1843)
- Architectural confusion

### The Evidence: "TICKET-002" Comment

Line 1806:
```rust
// TICKET-002: Populate both old unified structure and new explicit structures
```

This comment **proves** the code knows it's maintaining parallel structures!

---

## Proposed Architecture: Complete the Migration

### Principle: Separate What's Different, Unify What's the Same

**Observation:** 
- **Lag constraints** are different (loads vs. inflows have different semantics)
- **Uncertainty observation constraints** are the same (Y = base + σ·η + Σψ·lag)

### Recommended Design

```rust
pub struct Subproblem {
    // === SEPARATED: Lag Handling ===
    // Different semantics: Load lags affect demand, inflow lags affect supply
    
    load_lag_data: Option<LoadLagData>,
    inflow_lag_data: Option<InflowLagData>,
}

pub struct LoadLagData {
    // Type-safe: indexed by bus_id
    variables: LoadLagVariables,          // From existing code
    constraints: LoadLagConstraints,      // From existing code
    buffer: Vec<Vec<f64>>,                // [bus_id][lag_idx] → value
}

pub struct InflowLagData {
    // Type-safe: indexed by hydro_id
    variables: InflowLagVariables,        // From existing code
    constraints: InflowLagConstraints,    // From existing code  
    buffer: Vec<Vec<f64>>,                // [hydro_id][lag_idx] → value
}

pub struct Subproblem {
    // === UNIFIED: Uncertainty Observation Constraints ===
    // Same semantics: Y[i] = base + σ·η for all uncertain entities
    
    /// Precomputed data for fast uncertainty constraint RHS updates
    /// Can stay unified because the update logic is identical for loads/inflows
    uncertainty_constraint_data: Vec<UncertaintyObservationData>,
}

pub struct UncertaintyObservationData {
    constraint_idx: usize,      // Which LP constraint to update
    innovation_idx: usize,      // Index in innovations array
    seasonal_std: f64,          // σ for RHS computation
    deterministic_base: f64,    // Precomputed deterministic part
    // Removed: entity_type, entity_id, global_entity_idx (not needed here)
}
```

### Key Benefits

1. **Type Safety**: Can't confuse bus_id with hydro_id
2. **Clear Separation**: Load lags ≠ Inflow lags (different data structures)
3. **Appropriate Unification**: Uncertainty constraints are truly uniform
4. **No Redundancy**: Each piece of information stored once
5. **Clear Responsibilities**: Each struct has single, clear purpose

---

## Migration Path

### Phase 1: Remove `lag_fixing_constraints` (Old Unified)

```rust
// In Constraints struct, remove:
pub lag_fixing_constraints: Option<Vec<Vec<usize>>>,  // ❌ DELETE

// Keep only:
pub load_lag_constraints: Option<LoadLagConstraints>,   // ✅ KEEP
pub inflow_lag_constraints: Option<InflowLagConstraints>, // ✅ KEEP
```

**Impact:** Lines 1806-1843 (parallel population) can be deleted.

### Phase 2: Replace `UncertaintyConstraintManager` with Separated Buffers

```rust
// Replace:
pub uncertainty_manager: UncertaintyConstraintManager,  // ❌ DELETE

// With:
pub load_lag_buffer: Option<Vec<Vec<f64>>>,    // ✅ ADD
pub inflow_lag_buffer: Option<Vec<Vec<f64>>>,  // ✅ ADD
```

**Impact:** 
- `update_lag_buffers_from_trajectory()` splits into two methods
- `update_lag_fixing_constraints()` uses separated buffers directly

### Phase 3: Refine `UncertaintyConstraintData` → `UncertaintyObservationData`

```rust
// Rename and refine:
pub struct UncertaintyObservationData {  // Renamed from UncertaintyConstraintData
    constraint_idx: usize,
    innovation_idx: usize,
    seasonal_std: f64,
    deterministic_base: f64,
    // Removed: entity_type, entity_id, global_entity_idx, ar_order, psi_coefficients
}

// Rename field:
pub uncertainty_observation_data: Vec<UncertaintyObservationData>,  // Better name
```

**Why keep this unified?**
Because `update_uncertainty_constraints()` treats all entities identically:
```rust
for data in &self.uncertainty_observation_data {
    let innovation = innovations[data.innovation_idx];
    let rhs = data.deterministic_base + data.seasonal_std * innovation;
    model.change_rows_bounds(data.constraint_idx, rhs, rhs);
}
```

This is **genuinely uniform** - loads and inflows use same update logic.

### Phase 4: Extract Lags Directly (Optional Optimization)

Consider removing lag buffer for constraint updates entirely:

```rust
fn update_load_lag_constraints(&mut self, trajectory: &[&Realization]) {
    for bus_id in 0..n_buses {
        let constraints = self.load_lag_data.constraints.get_constraints(bus_id);
        for (lag_idx, &constraint_idx) in constraints.iter().enumerate() {
            // Extract lag directly from trajectory (no buffer)
            let lookback = lag_idx + 1;
            let past_idx = trajectory.len() - 1 - lookback;
            let lag_value = trajectory[past_idx].loads[bus_id];
            
            model.change_rows_bounds(constraint_idx, lag_value, lag_value);
        }
    }
}
```

But **keep lag buffer** for uncertainty constraints if they need lag terms in RHS.

---

## Addressing Your Specific Concerns

### Concern 1: "Lag buffer could extract everything from trajectory"

**Your observation:** ✅ **CORRECT for lag-fixing constraints**

The lag buffer is redundant when updating lag-fixing constraints. We could extract directly.

**However:** The buffer **is needed** for uncertainty observation constraints that include lag terms in RHS. So we have options:

1. **Keep buffer only for uncertainty constraints** (remove for lag-fixing)
2. **Extract directly everywhere** (duplicate extraction logic)
3. **Use buffer everywhere** (current approach, but document why)

**My recommendation:** Option 1 - separate concerns clearly.

### Concern 2: "`extract_storage_from_trajectory` does more than extract"

**Your observation:** ✅ **CORRECT - violates Command-Query Separation**

This should be split:
```rust
fn extract_storage(&self, trajectory: &[&Realization]) -> Vec<f64> {
    // Pure query - no side effects
}

fn update_coefficients_from_trajectory(&mut self, trajectory: &[&Realization]) {
    // Command - updates internal state
}
```

### Concern 3: "Responsibilities of `uncertainty_manager` and `entity_data` are not well defined"

**Your observation:** ✅ **ABSOLUTELY CORRECT**

**Current confusion:**
- `uncertainty_manager`: Named as if it manages constraints, actually just holds lag buffer
- `entity_data`: Named as generic "entity data", actually holds **precomputed coefficients**

**Both names are misleading!**

### Concern 4: "Could contain only a small part if architecture was unified"

**Your observation:** ✅ **CORRECT - Most fields are redundant**

After proper separation:

**`UncertaintyObservationData` needs only:**
- `constraint_idx` - which constraint to update
- `innovation_idx` - which innovation value to use
- `seasonal_std` - for σ·η computation
- `deterministic_base` - precomputed constant

**Everything else** (`entity_type`, `entity_id`, `global_entity_idx`, `ar_order`, `psi_coefficients`) is:
- Either redundant (routing info)
- Or belongs elsewhere (lag structure info)

---

## Impact Assessment

### Current Architecture: 🔴 Technical Debt

**Correctness:** ✅ Works (maintained carefully)  
**Performance:** ⚠️ Slight overhead (indirection, duplication)  
**Clarity:** ❌ Very confusing (halfway through migration)  
**Maintainability:** ❌ High burden (parallel structures must sync)  
**Type Safety:** ⚠️ Weak (entity_type runtime checks)

### After Complete Migration: ✅ Clean Architecture

**Correctness:** ✅ Same or better (type safety catches errors)  
**Performance:** ✅ Better (less indirection, no duplication)  
**Clarity:** ✅✅ Much clearer (separation of concerns)  
**Maintainability:** ✅✅ Easy (single source of truth)  
**Type Safety:** ✅✅ Strong (compile-time checks)

---

## Recommendations

### Priority 1: CRITICAL - Complete the Migration 🔴

**This is not optional.** The current halfway state will cause bugs.

**Immediate actions:**

1. **Document current state** with clear warnings:
```rust
// TODO(ARCHITECTURAL-DEBT): This code maintains parallel data structures
// for historical reasons. We are migrating from unified to separated
// load/inflow handling. See ARCHITECTURE_DEEP_ANALYSIS.md
```

2. **Create migration ticket** with clear scope:
```
[ARCH-MIGRATION] Complete load/inflow separation
- Remove lag_fixing_constraints (old unified structure)
- Replace UncertaintyConstraintManager with separated buffers
- Refine UncertaintyConstraintData to UncertaintyObservationData
- Update all access patterns
- Verify tests pass
Effort: 8-13 story points
Risk: Medium (touches core data structures)
Value: HIGH (eliminates architectural debt)
```

### Priority 2: Address Naming Issues

Current names are **actively misleading**:

| Current Name | Problem | Better Name |
|---|---|---|
| `UncertaintyConstraintManager` | Not a "manager", just a buffer | `LagObservationBuffer` or remove entirely |
| `UncertaintyConstraintData` | Generic name, specific purpose | `UncertaintyObservationCoefficients` |
| `entity_data` | Vague | `observation_constraint_data` |
| `uncertainty_manager` | Misleading scope | Remove after migration |

### Priority 3: Performance Optimization

After clean separation, optimize hot paths:

1. **Eliminate buffer for lag-fixing constraints** (extract directly)
2. **Keep buffer only for uncertainty constraints** (if needed)
3. **Pre-allocate all buffers** (avoid reallocations)
4. **Use slice operations** instead of indexed access

---

## Testing Strategy

### Verification Tests (Before Migration)

```rust
#[test]
fn test_parallel_structures_in_sync() {
    // Verify lag_fixing_constraints matches load/inflow_lag_constraints
    // This test should FAIL after migration (old structure removed)
}

#[test]
fn test_entity_data_routing_consistency() {
    // Verify entity_type, entity_id, global_entity_idx are consistent
    // This test should be DELETED after migration (no routing needed)
}
```

### Regression Tests (After Migration)

```rust
#[test]
fn test_lag_constraint_updates_correctness() {
    // Verify separated lag updates produce same results as unified
}

#[test]
fn test_uncertainty_constraint_updates_correctness() {
    // Verify observation constraint updates still work
}

#[test]
fn test_type_safety_compilation() {
    // These should fail to compile after migration:
    // let bus_lags = inflow_lag_buffer[bus_id];  // ❌ Type error!
    // let hydro_lags = load_lag_buffer[hydro_id];  // ❌ Type error!
}
```

---

## Conclusion

### Your Analysis: ✅ ABSOLUTELY CORRECT

1. **Unified architecture abandoned** - Confirmed. Halfway through migration.
2. **Responsibilities unclear** - Confirmed. Mixed concerns in both structures.
3. **Could be simplified** - Confirmed. Most data is redundant or misplaced.
4. **Separation would help** - Confirmed. Type safety and clarity would improve.

### The Real Problem: Incomplete Migration

This isn't just about naming or redundancy. The codebase is **stuck between two architectures**:

- **Old:** Unified entity handling (loads + inflows together)
- **New:** Separated handling (loads vs. inflows)

**Neither is fully implemented.** Both coexist, creating confusion.

### Priority Assessment

This is **HIGH PRIORITY** architectural debt:

- **Risk:** Medium (currently works but fragile)
- **Impact:** High (touches core preprocessing logic)
- **Effort:** Medium (clear migration path exists)
- **Value:** Very High (eliminates confusion, enables future development)

### Next Steps

1. **This week**: Add documentation warnings about parallel structures
2. **This sprint**: Create detailed migration ticket with acceptance criteria
3. **Next sprint**: Execute migration in phases with comprehensive testing
4. **Following sprint**: Performance optimization on clean architecture

---

## References

**Code Locations:**
- `src/subproblem.rs:65-103` - UncertaintyConstraintData definition
- `src/subproblem.rs:145-370` - Separated lag structures (NEW)
- `src/subproblem.rs:502-508` - uncertainty_manager and entity_data fields
- `src/subproblem.rs:1806-1843` - Parallel population (SMOKING GUN)
- `src/subproblem.rs:1962-2006` - build_entity_constraint_data
- `src/subproblem.rs:2021-2032` - update_uncertainty_constraints
- `src/subproblem.rs:2045-2140` - update_lag_fixing_constraints
- `src/uncertainty_constraints.rs:1-200` - UncertaintyConstraintManager

**Previous Reviews:**
- `ARCHITECTURE_REVIEW.md` - Preprocessing analysis
- `PREPARE_FROM_TRAJECTORY_REVIEW.md` - State update pattern
- Previous session discussions about unified → separated migration

---

**Reviewed by:** AI Code Review Agent  
**Confidence:** Very High (based on thorough code analysis and historical context)  
**Severity:** 🔴 **HIGH** - Architectural debt requiring resolution  
**Recommendation:** Complete the migration - current halfway state is unsustainable
