# [T-009] Extract Hydro Solution Extraction Methods

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Solution Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-008](./ticket-008-solution-extractor-scaffold.md)
> **Blocks**: [T-011](./ticket-011-extract-remaining-solutions.md)

---

## ⚠️ CRITICAL: Behavioral Equivalence

This ticket implements the actual extraction logic for hydro-related variables. The implementation **must be behaviorally identical** to the existing functions in `subproblem.rs`.

Run golden tests after EVERY method implementation.

---

## Files to Read Before Starting

- `src/model/solution_extract.rs` - SolutionExtractor scaffold from T-008
- `src/subproblem.rs:1912-1946` - Existing hydro extraction functions
- `src/subproblem.rs:1970-1980` - Water values extraction
- `src/solver.rs` - Solution struct

---

## Context

### Background

Hydro-related extractions are the core of the SDDP algorithm for hydroelectric systems. These include:
- **Spillage**: Water spilled (not used for generation)
- **Turbined flow**: Water passed through turbines
- **Final storage**: Reservoir level at end of stage
- **Water values**: Dual variables on hydro balance (economic value of water)

### Current Functions (from subproblem.rs)

```rust
fn get_spillage_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    let first = *self.variables.spillage.first().unwrap();
    let last = *self.variables.spillage.last().unwrap() + 1;
    realization.spillage.clone_from_slice(&solution.colvalue[first..last]);
}

fn get_turbined_flow_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    let first = *self.variables.turbined_flow.first().unwrap();
    let last = *self.variables.turbined_flow.last().unwrap() + 1;
    realization.turbined_flow.clone_from_slice(&solution.colvalue[first..last]);
}

fn get_final_storage_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    let first = *self.variables.stored_volume.first().unwrap();
    let last = *self.variables.stored_volume.last().unwrap() + 1;
    realization.final_storage.clone_from_slice(&solution.colvalue[first..last]);
}

fn get_water_values_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    let first = *self.constraints.hydro_balance.first().unwrap();
    let last = *self.constraints.hydro_balance.last().unwrap() + 1;
    realization.water_value.clone_from_slice(&solution.rowdual[first..last]);
}
```

---

## Specification

### Methods to Implement

Replace the `todo!()` in `SolutionExtractor` with actual implementations:

#### 1. Spillage Extraction

```rust
#[inline]
pub fn extract_spillage_into(&self, solution: &Solution, target: &mut [f64]) {
    let range = self.var_indices.spillage_range();
    target.copy_from_slice(&solution.colvalue[range]);
}

#[inline]
pub fn extract_spillage(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_spillage_into(solution, &mut realization.spillage);
}
```

#### 2. Turbined Flow Extraction

```rust
#[inline]
pub fn extract_turbined_flow_into(&self, solution: &Solution, target: &mut [f64]) {
    let range = self.var_indices.turbined_flow_range();
    target.copy_from_slice(&solution.colvalue[range]);
}

#[inline]
pub fn extract_turbined_flow(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_turbined_flow_into(solution, &mut realization.turbined_flow);
}
```

#### 3. Final Storage Extraction

```rust
#[inline]
pub fn extract_final_storage_into(&self, solution: &Solution, target: &mut [f64]) {
    let range = self.var_indices.stored_volume_range();
    target.copy_from_slice(&solution.colvalue[range]);
}

#[inline]
pub fn extract_final_storage(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_final_storage_into(solution, &mut realization.final_storage);
}
```

#### 4. Water Values Extraction (Dual)

```rust
#[inline]
pub fn extract_water_values_into(&self, solution: &Solution, target: &mut [f64]) {
    let range = self.con_indices.hydro_balance_range();
    target.copy_from_slice(&solution.rowdual[range]);
}

#[inline]
pub fn extract_water_values(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_water_values_into(solution, &mut realization.water_value);
}
```

### Behavior

- **Input**: LP solution with `colvalue` (primal) and `rowdual` (dual) arrays
- **Output**: Values copied into target slices
- **Pattern**: Use precomputed ranges from `VariableIndices`/`ConstraintIndices`
- **Performance**: Must be as fast as original (range-based copy)

---

## Acceptance Criteria

- [ ] `extract_spillage_into` and `extract_spillage` implemented
- [ ] `extract_turbined_flow_into` and `extract_turbined_flow` implemented
- [ ] `extract_final_storage_into` and `extract_final_storage` implemented
- [ ] `extract_water_values_into` and `extract_water_values` implemented
- [ ] All methods use `#[inline]` attribute
- [ ] Unit tests verify extraction correctness
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass

### Correctness Verification

- [ ] Extracted values match what original functions would produce
- [ ] Range calculations match original first..last+1 pattern
- [ ] No off-by-one errors

---

## Implementation Guide

### Suggested Approach

1. **Open `src/model/solution_extract.rs`**

2. **Implement spillage extraction**:
   ```rust
   #[inline]
   pub fn extract_spillage_into(&self, solution: &Solution, target: &mut [f64]) {
       let range = self.var_indices.spillage_range();
       target.copy_from_slice(&solution.colvalue[range]);
   }
   
   #[inline]
   pub fn extract_spillage(&self, solution: &Solution, realization: &mut Realization) {
       self.extract_spillage_into(solution, &mut realization.spillage);
   }
   ```

3. **Repeat for turbined_flow, final_storage, water_values**

4. **Add unit tests**:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       fn create_test_solution(n_cols: usize, n_rows: usize) -> Solution {
           Solution {
               colvalue: (0..n_cols).map(|i| i as f64).collect(),
               coldual: vec![0.0; n_cols],
               rowvalue: vec![0.0; n_rows],
               rowdual: (0..n_rows).map(|i| (i as f64) * 10.0).collect(),
           }
       }
       
       #[test]
       fn test_extract_spillage_into() {
           // Create indices where spillage is at positions 5..8
           let var_indices = /* mock VariableIndices with spillage_range = 5..8 */;
           let con_indices = /* mock ConstraintIndices */;
           let extractor = SolutionExtractor::new(var_indices, con_indices);
           
           let solution = create_test_solution(10, 5);
           let mut target = vec![0.0; 3];
           
           extractor.extract_spillage_into(&solution, &mut target);
           
           assert_eq!(target, vec![5.0, 6.0, 7.0]);
       }
       
       #[test]
       fn test_extract_water_values_into() {
           // Create indices where hydro_balance is at positions 2..5
           let var_indices = /* mock */;
           let con_indices = /* mock with hydro_balance_range = 2..5 */;
           let extractor = SolutionExtractor::new(var_indices, con_indices);
           
           let solution = create_test_solution(10, 10);
           let mut target = vec![0.0; 3];
           
           extractor.extract_water_values_into(&solution, &mut target);
           
           // rowdual values are i * 10.0
           assert_eq!(target, vec![20.0, 30.0, 40.0]);
       }
   }
   ```

5. **Run tests**:
   ```bash
   cargo test solution_extract
   ```

6. **Run golden tests**:
   ```bash
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/solution_extract.rs` | Implement 4 extraction methods |

### Patterns to Follow

- Use `self.var_indices.X_range()` for primal extractions
- Use `self.con_indices.X_range()` for dual extractions
- Use `copy_from_slice` (same as `clone_from_slice` for `f64`)
- All `_into` methods work with raw slices
- High-level methods delegate to `_into` variants

### Pitfalls to Avoid

- ⚠️ Don't change the extraction logic—copy exactly from original
- ⚠️ Don't forget `#[inline]` on all methods
- ⚠️ Ensure target slice length matches range length (caller responsibility)
- ⚠️ Use `rowdual` for water values (dual), not `colvalue`

---

## Testing Requirements

### Unit Tests

- [ ] Test `extract_spillage_into` with mock data
- [ ] Test `extract_turbined_flow_into` with mock data
- [ ] Test `extract_final_storage_into` with mock data
- [ ] Test `extract_water_values_into` with mock data (uses rowdual)
- [ ] Test that high-level methods call low-level correctly

### Integration Tests

Not needed at this stage—golden tests provide integration coverage.

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Update method doc comments (if needed)
- [ ] Ensure inline documentation explains the mapping

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward implementation following established pattern; main work is testing

---

## Definition of Done

- [ ] All 4 extraction methods implemented
- [ ] Unit tests written and passing
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass
- [ ] Code documented
- [ ] No changes to `subproblem.rs`
