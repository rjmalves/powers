# [T-112] Implement StoredBasis and Basis Transfer

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-110
> **Blocks**: T-113, T-114
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/solver.rs` - Current `Basis` struct and methods

---

## Context

### Background

With per-iteration Model creation, basis must be stored externally for optional warm-starting. Training uses basis; simulation doesn't (for reproducibility from loaded FCF).

---

## Specification

### New Type

```rust
/// Basis status stored for cross-iteration warm-starting.
/// 
/// # Optional Usage
/// 
/// - Training: Basis cached and applied for warm-starting
/// - Simulation: Basis not used (cold-start for reproducibility)
#[derive(Clone, Debug, Default)]
pub struct StoredBasis {
    pub colstatus: Vec<usize>,
    pub rowstatus: Vec<usize>,
}

impl StoredBasis {
    pub fn new() -> Self { Self::default() }
    
    pub fn with_capacity(num_cols: usize, num_rows: usize) -> Self {
        Self {
            colstatus: Vec::with_capacity(num_cols),
            rowstatus: Vec::with_capacity(num_rows),
        }
    }
    
    pub fn is_compatible(&self, num_cols: usize, num_rows: usize) -> bool {
        self.colstatus.len() == num_cols && self.rowstatus.len() == num_rows
    }
    
    pub fn is_empty(&self) -> bool {
        self.colstatus.is_empty() && self.rowstatus.is_empty()
    }
    
    pub fn clear(&mut self) {
        self.colstatus.clear();
        self.rowstatus.clear();
    }
}
```

### New Methods on Model

```rust
impl Model {
    /// Extract basis into StoredBasis for optional reuse.
    pub fn get_stored_basis(&self) -> StoredBasis {
        let basis = self.get_basis();
        StoredBasis {
            colstatus: basis.colstatus,
            rowstatus: basis.rowstatus,
        }
    }
    
    /// Extract basis into existing buffer (allocation-free).
    pub fn get_stored_basis_into(&self, stored: &mut StoredBasis);
    
    /// Apply stored basis for warm-starting.
    /// 
    /// # Returns
    /// 
    /// `Ok(())` on success, `Err` if dimensions don't match.
    pub fn apply_stored_basis(&mut self, basis: &StoredBasis) -> Result<(), HighsStatus> {
        if !basis.is_compatible(self.num_cols(), self.num_rows()) {
            return Err(HighsStatus::Error);
        }
        self.try_set_basis(Some(&basis.colstatus), Some(&basis.rowstatus))
    }
}
```

---

## Acceptance Criteria

- [ ] `StoredBasis` type implemented with `clear()` method
- [ ] `get_stored_basis()` implemented
- [ ] `get_stored_basis_into()` implemented
- [ ] `apply_stored_basis()` with compatibility check
- [ ] Basis transfer works across Models
- [ ] Unit tests passing

---

## Testing Requirements

```rust
#[test]
fn test_stored_basis_compatibility() {
    let mut basis = StoredBasis::new();
    basis.colstatus = vec![0; 10];
    basis.rowstatus = vec![0; 5];
    
    assert!(basis.is_compatible(10, 5));
    assert!(!basis.is_compatible(10, 6));
}

#[test]
fn test_stored_basis_clear() {
    let mut basis = StoredBasis::new();
    basis.colstatus = vec![0; 10];
    basis.rowstatus = vec![0; 5];
    
    basis.clear();
    assert!(basis.is_empty());
}

#[test]
fn test_warm_start_across_models() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..=100.0);
    problem.add_row(5.0.., [(0, 1.0)]);
    
    let mut model1 = problem.create_model(Sense::Minimise).unwrap();
    model1.solve();
    let basis = model1.get_stored_basis();
    drop(model1);
    
    problem.change_row_bounds(0, 6.0, f64::INFINITY);
    
    let mut model2 = problem.create_model(Sense::Minimise).unwrap();
    model2.apply_stored_basis(&basis).unwrap();
    model2.solve();
    
    assert_eq!(model2.status(), HighsModelStatus::Optimal);
}

#[test]
fn test_apply_basis_dimension_mismatch() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..);
    problem.add_row(0.0.., [(0, 1.0)]);
    
    let mut model = problem.create_model(Sense::Minimise).unwrap();
    
    let bad_basis = StoredBasis {
        colstatus: vec![0; 5],  // Wrong size
        rowstatus: vec![0; 1],
    };
    
    assert!(model.apply_stored_basis(&bad_basis).is_err());
}
```

---

## Effort Estimate

**Points**: 3
**Confidence**: High

---

## Definition of Done

- [ ] StoredBasis type with all methods
- [ ] Basis transfer methods on Model
- [ ] Tests passing
- [ ] PR merged
