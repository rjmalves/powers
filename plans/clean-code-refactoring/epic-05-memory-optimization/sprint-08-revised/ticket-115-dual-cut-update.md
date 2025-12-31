# [T-115] Implement Dual Cut Update (Problem + Model)

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-113, T-114
> **Blocks**: T-117
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/subproblem.rs` - Current cut update methods
- `src/cut.rs` - BendersCut structure

---

## Context

### Background

During backward pass, when a cut is selected for stage t-1, we must update **both**:
1. **Problem[t-1]**: For next iteration's Model
2. **Model[t-1]**: For current backward pass (stage t-2 needs it)

---

## Specification

```rust
impl Subproblem {
    /// Update a cut in BOTH Problem and Model.
    ///
    /// Called during backward pass. Updates Problem (for next iteration)
    /// and active Model (for current backward pass continuation).
    pub fn update_cut_dual(
        &mut self,
        slot: usize,
        coefficients: &[f64],
        rhs: f64,
    ) -> Result<(), String> {
        if slot >= self.num_preallocated_cuts {
            return Err(format!("Cut slot {} out of range", slot));
        }
        
        if coefficients.len() != self.cut_var_indices.len() {
            return Err(format!("Coefficient count mismatch"));
        }
        
        // 1. Update Problem (for next iteration)
        self.update_cut_in_problem(slot, coefficients, rhs)?;
        
        // 2. Update Model (for current backward pass)
        if let Some(ref mut model) = self.model {
            self.update_cut_in_model_inner(model, slot, coefficients, rhs)?;
        }
        
        Ok(())
    }
    
    fn update_cut_in_problem(&mut self, slot: usize, coeffs: &[f64], rhs: f64) -> Result<(), String> {
        let row = self.first_preallocated_cut_row + slot;
        
        for (i, &var_idx) in self.cut_var_indices.iter().enumerate() {
            self.problem.change_coefficient(row, var_idx, coeffs[i])?;
        }
        
        self.problem.change_row_bounds(row, rhs, f64::INFINITY);
        Ok(())
    }
    
    fn update_cut_in_model_inner(
        &self,
        model: &mut solver::Model,
        slot: usize,
        coeffs: &[f64],
        rhs: f64,
    ) -> Result<(), String> {
        let row = self.first_preallocated_cut_row + slot;
        
        for (i, &var_idx) in self.cut_var_indices.iter().enumerate() {
            model.change_coefficient(row, var_idx, coeffs[i])
                .map_err(|e| format!("Model coefficient failed: {:?}", e))?;
        }
        
        model.change_rows_bounds(row, rhs, f64::INFINITY);
        Ok(())
    }
    
    /// Deactivate a cut in BOTH Problem and Model.
    pub fn deactivate_cut_dual(&mut self, slot: usize) -> Result<(), String> {
        if slot >= self.num_preallocated_cuts {
            return Err(format!("Cut slot {} out of range", slot));
        }
        
        let row = self.first_preallocated_cut_row + slot;
        
        self.problem.change_row_bounds(row, f64::NEG_INFINITY, f64::INFINITY);
        
        if let Some(ref mut model) = self.model {
            model.change_rows_bounds(row, f64::NEG_INFINITY, f64::INFINITY);
        }
        
        Ok(())
    }
}
```

---

## Acceptance Criteria

- [ ] `update_cut_dual()` updates both Problem and Model
- [ ] `deactivate_cut_dual()` implemented
- [ ] Slot and coefficient validation
- [ ] Works when Model exists (backward pass)
- [ ] Works when Model doesn't exist (pre-iteration setup)
- [ ] Unit tests passing

---

## Testing Requirements

```rust
#[test]
fn test_update_cut_dual_both() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(10, 1).unwrap();
    subproblem.create_iteration_model(false).unwrap();
    
    let coeffs = vec![1.0; subproblem.cut_var_indices.len()];
    subproblem.update_cut_dual(0, &coeffs, 100.0).unwrap();
    
    // Problem updated
    let row = subproblem.first_preallocated_cut_row;
    assert_eq!(subproblem.problem.row_lower[row], 100.0);
    
    // Model also updated (verified by solving)
}

#[test]
fn test_update_cut_without_model() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(10, 1).unwrap();
    
    // No model created
    let coeffs = vec![1.0; subproblem.cut_var_indices.len()];
    subproblem.update_cut_dual(0, &coeffs, 100.0).unwrap();
    
    // Problem updated
    let row = subproblem.first_preallocated_cut_row;
    assert_eq!(subproblem.problem.row_lower[row], 100.0);
}

#[test]
fn test_deactivate_cut_dual() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(10, 1).unwrap();
    subproblem.create_iteration_model(false).unwrap();
    
    let coeffs = vec![1.0; subproblem.cut_var_indices.len()];
    subproblem.update_cut_dual(0, &coeffs, 100.0).unwrap();
    subproblem.deactivate_cut_dual(0).unwrap();
    
    let row = subproblem.first_preallocated_cut_row;
    assert_eq!(subproblem.problem.row_lower[row], f64::NEG_INFINITY);
}

#[test]
fn test_update_cut_validation() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(10, 1).unwrap();
    
    // Invalid slot
    let coeffs = vec![1.0; subproblem.cut_var_indices.len()];
    assert!(subproblem.update_cut_dual(100, &coeffs, 100.0).is_err());
    
    // Wrong coefficient count
    let wrong_coeffs = vec![1.0; 1];
    assert!(subproblem.update_cut_dual(0, &wrong_coeffs, 100.0).is_err());
}
```

---

## Effort Estimate

**Points**: 3
**Confidence**: High

---

## Definition of Done

- [ ] Dual update methods implemented
- [ ] Validation complete
- [ ] Tests passing
- [ ] PR merged
