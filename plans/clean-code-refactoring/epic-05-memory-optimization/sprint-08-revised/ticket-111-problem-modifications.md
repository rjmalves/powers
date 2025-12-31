# [T-111] Add Problem Modification Methods

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-110
> **Blocks**: T-113, T-115
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/solver.rs` - Current Problem structure, `columns` field format

---

## Context

### Background

Cuts must be updated in Problem so they persist to next iteration. Problem needs methods for modifying row bounds and coefficients.

---

## Specification

```rust
impl Problem {
    /// Change bounds of a row constraint.
    pub fn change_row_bounds(&mut self, row: usize, lower: f64, upper: f64) {
        assert!(row < self.num_row);
        self.row_lower[row] = lower;
        self.row_upper[row] = upper;
    }
    
    /// Change multiple row bounds at once.
    pub fn change_row_bounds_batch(&mut self, rows: &[usize], lowers: &[f64], uppers: &[f64]) {
        for (i, &row) in rows.iter().enumerate() {
            self.change_row_bounds(row, lowers[i], uppers[i]);
        }
    }
    
    /// Change a coefficient in the constraint matrix.
    pub fn change_coefficient(&mut self, row: usize, col: usize, value: f64) -> Result<(), String>;
    
    /// Change bounds of a column variable.
    pub fn change_col_bounds(&mut self, col: usize, lower: f64, upper: f64);
}
```

### Coefficient Change Implementation

Problem stores `columns[col] = (Vec<row_indices>, Vec<values>)`:

```rust
pub fn change_coefficient(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
    if col >= self.num_col || row >= self.num_row {
        return Err("Index out of bounds".into());
    }
    
    let (ref mut row_indices, ref mut values) = self.columns[col];
    let row_i32 = row as c_int;
    
    if let Some(pos) = row_indices.iter().position(|&r| r == row_i32) {
        if value == 0.0 {
            row_indices.remove(pos);
            values.remove(pos);
            self.num_nz -= 1;
        } else {
            values[pos] = value;
        }
    } else if value != 0.0 {
        let insert_pos = row_indices.iter()
            .position(|&r| r > row_i32)
            .unwrap_or(row_indices.len());
        row_indices.insert(insert_pos, row_i32);
        values.insert(insert_pos, value);
        self.num_nz += 1;
    }
    
    Ok(())
}
```

---

## Acceptance Criteria

- [ ] `change_row_bounds` implemented
- [ ] `change_row_bounds_batch` implemented
- [ ] `change_col_bounds` implemented
- [ ] `change_coefficient` with sparse handling
- [ ] Modifications reflected in created Models
- [ ] Unit tests passing

---

## Testing Requirements

```rust
#[test]
fn test_change_row_bounds_reflected() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..=100.0);
    problem.add_row(5.0.., [(0, 1.0)]);
    
    problem.change_row_bounds(0, 10.0, f64::INFINITY);
    
    let mut model = problem.create_model(Sense::Minimise).unwrap();
    model.solve();
    assert!((model.get_solution().colvalue[0] - 10.0).abs() < 1e-9);
}

#[test]
fn test_change_coefficient_add_remove() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..);
    problem.add_row(5.0.., [(0, 1.0)]);
    
    assert_eq!(problem.num_nz, 1);
    
    problem.change_coefficient(0, 0, 0.0).unwrap();  // Remove
    assert_eq!(problem.num_nz, 0);
    
    problem.change_coefficient(0, 0, 2.0).unwrap();  // Add back
    assert_eq!(problem.num_nz, 1);
}
```

---

## Effort Estimate

**Points**: 3
**Confidence**: High

---

## Definition of Done

- [ ] All methods implemented
- [ ] Sparse matrix handling correct
- [ ] Tests passing
- [ ] PR merged
