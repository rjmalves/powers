# [TICKET-001] Add HiGHS batch row and coefficient change API bindings

> **Epic**: [Epic 1: HiGHS Constraint Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-003](./ticket-003-subproblem-cut-slots.md)

## Context

### Background

The current HiGHS binding in `src/solver.rs` only supports single-row addition via `Highs_addRow`. To preallocate cut constraints efficiently, we need batch row addition via `Highs_addRows` and coefficient modification via `Highs_changeCoeff`.

### Relation to Epic

This ticket provides the low-level HiGHS API bindings required for constraint preallocation.

### Current State

`src/solver.rs` has:
- `add_row()` / `try_add_row()`: Single row addition ✅
- `change_rows_bounds()`: Bound modification ✅
- `change_coefficient()`: **Missing** ❌
- `add_rows_batch()`: **Missing** ❌

## Files to Read Before Starting

- `src/solver.rs` - Current HiGHS bindings (lines 430-580)
- `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - API reference in Appendix A

## Specification

### Inputs

**`add_rows_batch()`**:
- `num_rows: usize` - Number of rows to add
- `lower_bounds: &[f64]` - Row lower bounds (length = num_rows)
- `upper_bounds: &[f64]` - Row upper bounds (length = num_rows)
- `astart: &[i32]` - CSR row start indices (length = num_rows + 1)
- `aindex: &[i32]` - Column indices (length = total non-zeros)
- `avalue: &[f64]` - Coefficient values (length = total non-zeros)

**`change_coefficient()`**:
- `row: usize` - Row index
- `col: usize` - Column index
- `value: f64` - New coefficient value

### Outputs

- `Result<(), HighsStatus>` for both methods

### Behavior

- `add_rows_batch()`: Calls `Highs_addRows` with provided sparse matrix data
- `change_coefficient()`: Calls `Highs_changeCoeff` to update a single matrix element

### Error Handling

- Return `Err(HighsStatus::Error)` if HiGHS returns error status
- Validate array lengths in debug builds

## Acceptance Criteria

- [ ] `add_rows_batch()` method added to `Model` impl
- [ ] `change_coefficient()` method added to `Model` impl
- [ ] Both methods have doc comments with examples
- [ ] Examples 01 and 07 still pass (no behavioral change)

## Implementation Guide

### Suggested Approach

1. Study existing `try_add_row()` implementation (lines 511-533)
2. Add `change_coefficient()` first (simpler, single FFI call)
3. Add `add_rows_batch()` with CSR format handling
4. Test by temporarily calling from existing code

### Key Files to Modify

- `src/solver.rs`: Add new methods to `impl Model` block (~line 440)

### Patterns to Follow

- Follow `try_add_row()` pattern for error handling
- Use `highs_call!` macro for FFI calls
- Use `c()` helper for int conversion

### Code Template

```rust
// src/solver.rs - Add after line 580

/// Change a single coefficient in the constraint matrix.
///
/// # Arguments
///
/// * `row` - Row index (0-based)
/// * `col` - Column index (0-based)  
/// * `value` - New coefficient value
///
/// # Example
///
/// ```ignore
/// model.change_coefficient(5, 10, 1.5)?;  // Set A[5,10] = 1.5
/// ```
pub fn change_coefficient(
    &mut self,
    row: usize,
    col: usize,
    value: f64,
) -> Result<(), HighsStatus> {
    unsafe {
        highs_call!(Highs_changeCoeff(
            self.highs.mut_ptr(),
            c(row),
            c(col),
            value
        ))
    }?;
    Ok(())
}

/// Add multiple rows at once (batch operation).
///
/// Uses CSR (Compressed Sparse Row) format for efficient sparse matrix transfer.
///
/// # Arguments
///
/// * `num_rows` - Number of rows to add
/// * `lower_bounds` - Lower bounds for each row
/// * `upper_bounds` - Upper bounds for each row
/// * `astart` - CSR row start indices (length = num_rows + 1)
/// * `aindex` - Column indices for non-zeros
/// * `avalue` - Coefficient values for non-zeros
///
/// # Example
///
/// ```ignore
/// // Add 2 rows: x0 + 2*x1 >= 5 and 3*x1 + x2 >= 7
/// model.add_rows_batch(
///     2,
///     &[5.0, 7.0],           // lower bounds
///     &[f64::INFINITY; 2],   // upper bounds
///     &[0, 2, 4],            // astart: row 0 has 2 NZ, row 1 has 2 NZ
///     &[0, 1, 1, 2],         // aindex: col indices
///     &[1.0, 2.0, 3.0, 1.0], // avalue: coefficients
/// )?;
/// ```
pub fn add_rows_batch(
    &mut self,
    num_rows: usize,
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    astart: &[c_int],
    aindex: &[c_int],
    avalue: &[f64],
) -> Result<(), HighsStatus> {
    debug_assert_eq!(lower_bounds.len(), num_rows);
    debug_assert_eq!(upper_bounds.len(), num_rows);
    debug_assert_eq!(astart.len(), num_rows + 1);
    
    let num_nz = *astart.last().unwrap_or(&0) as usize;
    debug_assert_eq!(aindex.len(), num_nz);
    debug_assert_eq!(avalue.len(), num_nz);
    
    unsafe {
        highs_call!(Highs_addRows(
            self.highs.mut_ptr(),
            c(num_rows),
            lower_bounds.as_ptr(),
            upper_bounds.as_ptr(),
            c(num_nz),
            astart.as_ptr(),
            aindex.as_ptr(),
            avalue.as_ptr()
        ))
    }?;
    Ok(())
}
```

### Pitfalls to Avoid

- ⚠️ Don't forget `c()` conversion for integer parameters
- ⚠️ CSR format: `astart` has length `num_rows + 1`, not `num_rows`
- ⚠️ Ensure `c_int` type matches HiGHS expectation (check `highs_sys` types)

## Testing Requirements

### Unit Tests

Not required for this ticket (pure FFI wrapper). Validation via integration.

### Integration Tests

- [ ] Examples 01 and 07 still produce correct results
- [ ] No compiler warnings in `src/solver.rs`

## Documentation Requirements

- [ ] Add doc comments with `# Arguments`, `# Example`, `# Errors` sections
- [ ] Update module-level docs if needed

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward FFI binding following existing patterns

## Definition of Done

- [ ] Implementation complete
- [ ] Doc comments added
- [ ] Code compiles without warnings
- [ ] Examples still work
