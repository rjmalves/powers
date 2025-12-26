# [TICKET-003] Implement Subproblem cut slot infrastructure

> **Epic**: [Epic 1: HiGHS Constraint Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-001](./ticket-001-highs-api-bindings.md), [TICKET-002](./ticket-002-sizing-info-extension.md)  
> **Blocks**: [TICKET-004](../sprint-02/ticket-004-cut-addition-update.md), [TICKET-005](../sprint-02/ticket-005-cut-removal.md)

## Context

### Background

The Subproblem struct needs infrastructure to track preallocated cut slots in the HiGHS model. This includes the row index of the first cut, total preallocated count, and a method to preallocate the constraint matrix at training start.

### Relation to Epic

This ticket adds the core data structures and preallocation method that Sprint 2 will use for cut addition/removal.

### Current State

`Subproblem` in `src/subproblem.rs` has:
- `model: Option<solver::Model>` - HiGHS model ✅
- `add_cut_constraint_to_model()` - Uses dynamic `add_row()` (to be replaced)
- No cut slot tracking fields ❌
- No preallocation method ❌

## Files to Read Before Starting

- `src/subproblem.rs` - Subproblem struct definition (find struct fields)
- `src/solver.rs` - Model API (lines 430-580)
- `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - Implementation strategy (lines 119-195)

## Specification

### New Fields on Subproblem

```rust
/// Index of first preallocated cut row in HiGHS model
first_preallocated_cut_row: usize,

/// Total number of preallocated cut slots
num_preallocated_cuts: usize,

/// Next available slot for sequential allocation
next_available_cut_slot: usize,

/// Mapping from slot index to cut ID (None = slot free)
cut_slot_to_id: Vec<Option<usize>>,

/// Stack of freed slots for reuse (LIFO)
free_cut_slots: Vec<usize>,
```

### New Method

**`preallocate_cut_constraints(max_cuts: usize, state_dimension: usize) -> Result<(), String>`**

1. Create placeholder constraints with:
   - Bounds: `[-∞, ∞]` (inactive by default)
   - Coefficients: `0.0` for alpha and all storage variables
   - Sparsity pattern: alpha + state_dimension storage variables

2. Call `model.add_rows_batch()` with CSR format

3. Store metadata in new fields

### Inputs

- `max_cuts: usize` - Number of slots to preallocate
- `state_dimension: usize` - Number of state variables (for coefficient pattern)

### Outputs

- `Result<(), String>` - Ok on success, error message on failure

### Error Handling

- Return `Err` if model is not initialized
- Return `Err` if HiGHS batch add fails

## Acceptance Criteria

- [ ] New fields added to Subproblem struct
- [ ] `preallocate_cut_constraints()` method implemented
- [ ] Method creates correct CSR format for batch add
- [ ] Placeholder constraints are inactive (bounds `[-∞, ∞]`)
- [ ] Examples 01 and 07 still pass (method not yet called)

## Implementation Guide

### Suggested Approach

1. Locate Subproblem struct definition in `src/subproblem.rs`
2. Add new fields with default values
3. Implement `preallocate_cut_constraints()` method
4. Ensure existing code still compiles (method not called yet)

### Key Files to Modify

- `src/subproblem.rs`: 
  - Add fields to struct definition
  - Add `preallocate_cut_constraints()` method
  - Initialize new fields in constructors

### Code Template

```rust
// Add to Subproblem struct definition

/// Index of first preallocated cut row in HiGHS model.
/// Set by `preallocate_cut_constraints()`.
first_preallocated_cut_row: usize,

/// Total number of preallocated cut slots.
num_preallocated_cuts: usize,

/// Next available slot for sequential allocation.
/// Incremented when no free slots available.
next_available_cut_slot: usize,

/// Mapping from slot index to cut ID.
/// `None` indicates slot is free.
cut_slot_to_id: Vec<Option<usize>>,

/// Stack of freed slots for reuse (LIFO order).
/// When a cut is removed, its slot is pushed here.
free_cut_slots: Vec<usize>,
```

```rust
// Add method to impl Subproblem

/// Preallocate cut constraint slots in the HiGHS model.
///
/// Creates placeholder constraints with relaxed bounds `[-∞, ∞]` that are
/// effectively inactive. Cuts are later added by modifying coefficients
/// and tightening bounds.
///
/// # Arguments
///
/// * `max_cuts` - Number of cut slots to preallocate
/// * `state_dimension` - State dimension (for coefficient pattern)
///
/// # Returns
///
/// `Ok(())` on success, or error message if preallocation fails.
///
/// # Example
///
/// ```ignore
/// subproblem.preallocate_cut_constraints(200, 156)?;
/// // Now subproblem has 200 inactive cut constraint slots
/// ```
pub fn preallocate_cut_constraints(
    &mut self,
    max_cuts: usize,
    state_dimension: usize,
) -> Result<(), String> {
    let model = self.model.as_mut()
        .ok_or("Model not initialized")?;
    
    // Get current row count (cuts will be appended after)
    let first_cut_row = model.num_rows()
        .map_err(|_| "Failed to get row count")?;
    
    // Prepare bounds: [-∞, ∞] makes constraints inactive
    let lower_bounds = vec![f64::NEG_INFINITY; max_cuts];
    let upper_bounds = vec![f64::INFINITY; max_cuts];
    
    // Each cut has (state_dimension + 1) non-zeros:
    // - 1 coefficient for alpha variable
    // - state_dimension coefficients for storage variables
    let nnz_per_cut = state_dimension + 1;
    let total_nnz = max_cuts * nnz_per_cut;
    
    // Build CSR format
    let mut astart: Vec<i32> = Vec::with_capacity(max_cuts + 1);
    let mut aindex: Vec<i32> = Vec::with_capacity(total_nnz);
    let mut avalue: Vec<f64> = Vec::with_capacity(total_nnz);
    
    // Get variable indices
    let alpha_idx = self.variables.alpha as i32;
    
    for cut_idx in 0..max_cuts {
        // Row start index
        astart.push((cut_idx * nnz_per_cut) as i32);
        
        // Alpha variable coefficient (placeholder 0.0)
        aindex.push(alpha_idx);
        avalue.push(0.0);
        
        // Storage variable coefficients (placeholder 0.0)
        for hydro_id in 0..state_dimension {
            if hydro_id < self.variables.stored_volume.len() {
                aindex.push(self.variables.stored_volume[hydro_id] as i32);
                avalue.push(0.0);
            }
        }
    }
    // Final row start (points past last element)
    astart.push(total_nnz as i32);
    
    // Add all rows at once
    model.add_rows_batch(
        max_cuts,
        &lower_bounds,
        &upper_bounds,
        &astart,
        &aindex,
        &avalue,
    ).map_err(|e| format!("HiGHS batch add failed: {:?}", e))?;
    
    // Store metadata
    self.first_preallocated_cut_row = first_cut_row;
    self.num_preallocated_cuts = max_cuts;
    self.next_available_cut_slot = 0;
    self.cut_slot_to_id = vec![None; max_cuts];
    self.free_cut_slots = Vec::with_capacity(max_cuts / 4); // Expect 25% reuse
    
    Ok(())
}
```

### Initialization in Constructors

Find Subproblem constructors and add default initialization:

```rust
first_preallocated_cut_row: 0,
num_preallocated_cuts: 0,
next_available_cut_slot: 0,
cut_slot_to_id: Vec::new(),
free_cut_slots: Vec::new(),
```

### Pitfalls to Avoid

- ⚠️ State dimension may differ from `stored_volume.len()` for inflow states
- ⚠️ Ensure `variables.alpha` and `variables.stored_volume` are valid
- ⚠️ CSR format: `astart` length is `num_rows + 1`
- ⚠️ Need `num_rows()` method on Model (verify it exists)

## Testing Requirements

### Unit Tests

For now, validation via integration only. Unit tests deferred to Sprint 2.

### Integration Tests

- [ ] Examples 01 and 07 still work (preallocation not yet called)
- [ ] Code compiles without warnings

## Documentation Requirements

- [ ] Doc comments on new fields explaining purpose
- [ ] Doc comment on `preallocate_cut_constraints()` with example

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: CSR format construction and field initialization require care

## Definition of Done

- [ ] New fields added to Subproblem
- [ ] Constructors initialize new fields
- [ ] `preallocate_cut_constraints()` implemented
- [ ] Code compiles
- [ ] Examples still work
