# [T-110] Implement `Problem::create_model()` for Non-Consuming Model Creation

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-111, T-112, T-113
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/solver.rs` - Current Problem and Model implementation
- `docs/architecture/SOLVER.md` - Design rationale

---

## Context

### Background

Currently `Problem::optimise()` consumes `self`. For the per-iteration Model architecture, we need to keep Problem as the source of truth and create fresh Models from it.

### Goal

Add `create_model(&self)` that creates a HiGHS Model without consuming the Problem.

---

## Specification

```rust
impl Problem {
    /// Create a Model for solving without consuming the Problem.
    ///
    /// The Problem remains valid and can create additional Models or be modified.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Iteration 1
    /// let mut model = problem.create_model(Sense::Minimise)?;
    /// model.solve();
    /// drop(model);  // HiGHS freed
    ///
    /// // Modify problem for iteration 2
    /// problem.change_row_bounds(cut_row, rhs, f64::INFINITY);
    /// let mut model2 = problem.create_model(Sense::Minimise)?;
    /// ```
    pub fn create_model(&self, sense: Sense) -> Result<Model, HighsStatus>;
}
```

---

## Implementation

```rust
pub fn create_model(&self, sense: Sense) -> Result<Model, HighsStatus> {
    let mut highs = HighsPtr::default();
    highs.make_quiet();
    
    let (astart, aindex, avalue) = self.to_compressed_matrix_form();
    
    unsafe {
        highs_call!(Highs_passLp(
            highs.mut_ptr(),
            c(self.num_col),
            c(self.num_row),
            c(self.num_nz),
            MATRIX_FORMAT_COLUMN_WISE,
            OBJECTIVE_SENSE_MINIMIZE,
            self.offset,
            self.col_cost.as_ptr(),
            self.col_lower.as_ptr(),
            self.col_upper.as_ptr(),
            self.row_lower.as_ptr(),
            self.row_upper.as_ptr(),
            astart.as_ptr(),
            aindex.as_ptr(),
            avalue.as_ptr()
        ))
    }?;
    
    let mut model = Model { highs };
    model.set_sense(sense);
    Ok(model)
}
```

---

## Acceptance Criteria

- [ ] `create_model(&self)` implemented
- [ ] Problem not consumed after call
- [ ] Multiple Models can be created from same Problem
- [ ] Modifications to Problem don't affect existing Models
- [ ] Unit tests passing

---

## Testing Requirements

```rust
#[test]
fn test_create_model_does_not_consume() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..=10.0);
    problem.add_row(5.0.., [(0, 1.0)]);
    
    let _ = problem.create_model(Sense::Minimise).unwrap();
    assert_eq!(problem.num_col, 1);  // Still accessible
}

#[test]
fn test_create_model_independent() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..=10.0);
    problem.add_row(5.0.., [(0, 1.0)]);
    
    let mut model1 = problem.create_model(Sense::Minimise).unwrap();
    model1.solve();
    
    problem.row_lower[0] = 8.0;  // Modify problem
    
    // model1 unaffected
    assert!((model1.get_solution().colvalue[0] - 5.0).abs() < 1e-9);
    
    // New model gets modification
    let mut model2 = problem.create_model(Sense::Minimise).unwrap();
    model2.solve();
    assert!((model2.get_solution().colvalue[0] - 8.0).abs() < 1e-9);
}
```

---

## Effort Estimate

**Points**: 5
**Confidence**: High

---

## Definition of Done

- [ ] Method implemented
- [ ] All tests passing
- [ ] Documentation complete
- [ ] PR merged
