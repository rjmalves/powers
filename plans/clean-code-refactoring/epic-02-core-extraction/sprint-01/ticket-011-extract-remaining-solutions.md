# [T-011] Extract Remaining Solution Extractions

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Solution Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-009](./ticket-009-extract-hydro-solution.md), [T-010](./ticket-010-extract-thermal-solution.md)
> **Blocks**: Sprint 2 tickets

---

## ⚠️ CRITICAL: Complex Extractions

This ticket implements the remaining extraction methods, including **non-contiguous index patterns** (load, inflow) and **complex lag dual extraction**. Pay careful attention to the original implementation patterns.

Run golden tests after EVERY method implementation.

---

## Files to Read Before Starting

- `src/model/solution_extract.rs` - Current SolutionExtractor with T-009/T-010 methods
- `src/subproblem.rs:1948-1968` - Load and inflow extraction (non-contiguous)
- `src/subproblem.rs:1992-2034` - Lag duals extraction (complex)
- `src/solver.rs` - Solution struct

---

## Context

### Background

The remaining extractions have different patterns:

1. **Load/Inflow**: Non-contiguous indices (indexed loop, not slice copy)
2. **Lag duals**: Complex structure with nested vectors per entity
3. These require different extraction strategies than simple range copies

### Current Functions (from subproblem.rs)

```rust
fn get_load_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    // Non-contiguous: loop over individual indices
    for (i, &var_idx) in self.variables.load.iter().enumerate() {
        realization.loads[i] = solution.colvalue[var_idx];
    }
}

fn get_inflow_from_solution(&mut self, solution: &Solution, realization: &mut Realization) {
    // Non-contiguous: loop over individual indices
    for (h, &var_idx) in self.variables.inflow.iter().enumerate() {
        realization.inflow[h] = solution.colvalue[var_idx];
    }
}

fn get_lag_duals_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    // Clear existing lag duals
    realization.load_lag_duals.clear();
    realization.inflow_lag_duals.clear();

    // Extract load lag duals directly by bus_id
    if let Some(load_constraints) = &self.constraints.load_lag_constraints {
        let buses_count = load_constraints.constraints_by_bus.len();
        realization.load_lag_duals.resize(buses_count, Vec::new());

        for bus_id in 0..buses_count {
            let constraints = load_constraints.get_constraints(bus_id);
            realization.load_lag_duals[bus_id] = constraints
                .iter()
                .map(|&idx| solution.rowdual[idx])
                .collect();
        }
    }

    // Extract inflow lag duals directly by hydro_id
    if let Some(inflow_constraints) = &self.constraints.inflow_lag_constraints {
        let hydros_count = inflow_constraints.constraints_by_hydro.len();
        realization.inflow_lag_duals.resize(hydros_count, Vec::new());

        for hydro_id in 0..hydros_count {
            let constraints = inflow_constraints.get_constraints(hydro_id);
            realization.inflow_lag_duals[hydro_id] = constraints
                .iter()
                .map(|&idx| solution.rowdual[idx])
                .collect();
        }
    }
}
```

---

## Specification

### Methods to Implement

#### 1. Load Extraction (Non-contiguous)

```rust
/// Extract load observation values into target slice.
///
/// Uses individual index lookups because load variables may not be contiguous.
#[inline]
pub fn extract_load_into(&self, solution: &Solution, target: &mut [f64]) {
    for (i, &var_idx) in self.var_indices.load_indices().iter().enumerate() {
        target[i] = solution.colvalue[var_idx];
    }
}

#[inline]
pub fn extract_load(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_load_into(solution, &mut realization.loads);
}
```

#### 2. Inflow Extraction (Non-contiguous)

```rust
/// Extract inflow observation values into target slice.
///
/// Uses individual index lookups because inflow variables may not be contiguous.
#[inline]
pub fn extract_inflow_into(&self, solution: &Solution, target: &mut [f64]) {
    for (h, &var_idx) in self.var_indices.inflow_indices().iter().enumerate() {
        target[h] = solution.colvalue[var_idx];
    }
}

#[inline]
pub fn extract_inflow(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_inflow_into(solution, &mut realization.inflow);
}
```

#### 3. Lag Duals Extraction (Complex)

```rust
/// Extract lag dual values from solution.
///
/// This method populates both `load_lag_duals` and `inflow_lag_duals` in the realization.
/// The structure is `Vec<Vec<f64>>` indexed by entity_id, with inner vec containing
/// duals for each lag order.
///
/// # Note
///
/// This method does not have an `_into` variant because the output structure
/// is complex (nested vectors with allocations). Future SoA migration may
/// require a different approach.
pub fn extract_lag_duals(&self, solution: &Solution, realization: &mut Realization) {
    // Clear existing lag duals
    realization.load_lag_duals.clear();
    realization.inflow_lag_duals.clear();
    
    // Extract load lag duals
    let num_buses = self.con_indices.num_buses();
    if num_buses > 0 {
        realization.load_lag_duals.resize(num_buses, Vec::new());
        for bus_id in 0..num_buses {
            let constraints = self.con_indices.load_lag_constraints(bus_id);
            realization.load_lag_duals[bus_id] = constraints
                .iter()
                .map(|&idx| solution.rowdual[idx])
                .collect();
        }
    }
    
    // Extract inflow lag duals
    let num_hydros = self.con_indices.num_hydros();
    if num_hydros > 0 {
        realization.inflow_lag_duals.resize(num_hydros, Vec::new());
        for hydro_id in 0..num_hydros {
            let constraints = self.con_indices.inflow_lag_constraints(hydro_id);
            realization.inflow_lag_duals[hydro_id] = constraints
                .iter()
                .map(|&idx| solution.rowdual[idx])
                .collect();
        }
    }
}
```

### Update extract_all_primals and extract_all_duals

Now that all methods are implemented, remove the `todo!()` and verify the convenience methods work:

```rust
pub fn extract_all_primals(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_deficit(solution, realization);
    if self.has_exchange() {
        self.extract_exchange(solution, realization);
    }
    if self.has_thermal() {
        self.extract_thermal_gen(solution, realization);
    }
    self.extract_spillage(solution, realization);
    self.extract_turbined_flow(solution, realization);
    self.extract_final_storage(solution, realization);
    self.extract_load(solution, realization);
    self.extract_inflow(solution, realization);
}

pub fn extract_all_duals(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_water_values(solution, realization);
    self.extract_marginal_costs(solution, realization);
    self.extract_lag_duals(solution, realization);
}
```

---

## Acceptance Criteria

- [ ] `extract_load_into` and `extract_load` implemented
- [ ] `extract_inflow_into` and `extract_inflow` implemented
- [ ] `extract_lag_duals` implemented (no `_into` variant)
- [ ] `extract_all_primals` fully functional (no todo!())
- [ ] `extract_all_duals` fully functional (no todo!())
- [ ] Unit tests for load/inflow extraction
- [ ] Unit tests for lag duals extraction
- [ ] All `todo!()` removed from SolutionExtractor
- [ ] `cargo build` succeeds without todo warnings
- [ ] `cargo test` passes
- [ ] Golden tests pass

### Correctness Verification

- [ ] Load extraction matches original per-index pattern
- [ ] Inflow extraction matches original per-index pattern
- [ ] Lag duals structure matches original (Vec<Vec<f64>>)
- [ ] Entities without lags have empty inner vectors

---

## Implementation Guide

### Suggested Approach

1. **Implement load extraction**:
   ```rust
   #[inline]
   pub fn extract_load_into(&self, solution: &Solution, target: &mut [f64]) {
       for (i, &var_idx) in self.var_indices.load_indices().iter().enumerate() {
           target[i] = solution.colvalue[var_idx];
       }
   }
   
   #[inline]
   pub fn extract_load(&self, solution: &Solution, realization: &mut Realization) {
       self.extract_load_into(solution, &mut realization.loads);
   }
   ```

2. **Implement inflow extraction** (same pattern as load):
   ```rust
   #[inline]
   pub fn extract_inflow_into(&self, solution: &Solution, target: &mut [f64]) {
       for (h, &var_idx) in self.var_indices.inflow_indices().iter().enumerate() {
           target[h] = solution.colvalue[var_idx];
       }
   }
   
   #[inline]
   pub fn extract_inflow(&self, solution: &Solution, realization: &mut Realization) {
       self.extract_inflow_into(solution, &mut realization.inflow);
   }
   ```

3. **Implement lag duals extraction**:
   ```rust
   pub fn extract_lag_duals(&self, solution: &Solution, realization: &mut Realization) {
       realization.load_lag_duals.clear();
       realization.inflow_lag_duals.clear();
       
       let num_buses = self.con_indices.num_buses();
       if num_buses > 0 {
           realization.load_lag_duals.resize(num_buses, Vec::new());
           for bus_id in 0..num_buses {
               let constraints = self.con_indices.load_lag_constraints(bus_id);
               realization.load_lag_duals[bus_id] = constraints
                   .iter()
                   .map(|&idx| solution.rowdual[idx])
                   .collect();
           }
       }
       
       let num_hydros = self.con_indices.num_hydros();
       if num_hydros > 0 {
           realization.inflow_lag_duals.resize(num_hydros, Vec::new());
           for hydro_id in 0..num_hydros {
               let constraints = self.con_indices.inflow_lag_constraints(hydro_id);
               realization.inflow_lag_duals[hydro_id] = constraints
                   .iter()
                   .map(|&idx| solution.rowdual[idx])
                   .collect();
           }
       }
   }
   ```

4. **Verify convenience methods work** - they should now have all dependencies

5. **Add unit tests**:
   ```rust
   #[test]
   fn test_extract_load_non_contiguous() {
       // Mock indices: load variables at positions [2, 5, 8] (non-contiguous)
       let var_indices = create_indices_with_load_indices(vec![2, 5, 8]);
       let extractor = SolutionExtractor::new(var_indices, mock_con_indices());
       
       // Solution colvalue = [0,1,2,3,4,5,6,7,8,9]
       let solution = create_test_solution(10, 5);
       let mut target = vec![0.0; 3];
       
       extractor.extract_load_into(&solution, &mut target);
       
       // Should extract values at indices 2, 5, 8
       assert_eq!(target, vec![2.0, 5.0, 8.0]);
   }
   
   #[test]
   fn test_extract_lag_duals_empty() {
       // Extractor with no lag constraints
       let extractor = create_extractor_without_lags();
       
       let solution = create_test_solution(10, 10);
       let mut realization = mock_realization();
       
       extractor.extract_lag_duals(&solution, &mut realization);
       
       assert!(realization.load_lag_duals.is_empty());
       assert!(realization.inflow_lag_duals.is_empty());
   }
   
   #[test]
   fn test_extract_lag_duals_with_data() {
       // Mock: 2 buses, bus 0 has lags at constraints [3,4], bus 1 has lag at [5]
       let con_indices = create_con_indices_with_load_lags(vec![
           vec![3, 4],  // bus 0
           vec![5],     // bus 1
       ]);
       let extractor = SolutionExtractor::new(mock_var_indices(), con_indices);
       
       // rowdual = [0, 10, 20, 30, 40, 50, ...]
       let solution = create_test_solution(10, 10);
       let mut realization = mock_realization();
       
       extractor.extract_lag_duals(&solution, &mut realization);
       
       assert_eq!(realization.load_lag_duals.len(), 2);
       assert_eq!(realization.load_lag_duals[0], vec![30.0, 40.0]);
       assert_eq!(realization.load_lag_duals[1], vec![50.0]);
   }
   ```

6. **Run all tests and golden tests**:
   ```bash
   cargo test
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/solution_extract.rs` | Implement remaining methods, remove todo!() |

### Patterns to Follow

- Use indexed loop for non-contiguous variables
- Use `iter().map().collect()` for lag duals (matches original)
- Clear vectors before populating (matches original)
- Use `resize()` to set vector length (matches original)

### Pitfalls to Avoid

- ⚠️ Don't assume load/inflow indices are contiguous
- ⚠️ Don't forget to clear lag duals vectors before populating
- ⚠️ Don't change the collection pattern for lag duals
- ⚠️ Verify `num_buses()` and `num_hydros()` return correct values
- ⚠️ Empty lag constraints should result in empty inner vectors

---

## Testing Requirements

### Unit Tests

- [ ] Test `extract_load_into` with non-contiguous indices
- [ ] Test `extract_inflow_into` with non-contiguous indices
- [ ] Test `extract_lag_duals` with no lags (empty result)
- [ ] Test `extract_lag_duals` with load lags only
- [ ] Test `extract_lag_duals` with inflow lags only
- [ ] Test `extract_lag_duals` with both load and inflow lags
- [ ] Test `extract_all_primals` runs without error
- [ ] Test `extract_all_duals` runs without error

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Document non-contiguous index patterns
- [ ] Document why `extract_lag_duals` has no `_into` variant
- [ ] Update any TODO comments in documentation

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Moderate complexity with non-contiguous patterns and nested structures

---

## Definition of Done

- [ ] All extraction methods implemented
- [ ] All `todo!()` removed
- [ ] Unit tests passing
- [ ] Golden tests passing
- [ ] `cargo build` without warnings
- [ ] SolutionExtractor fully functional
- [ ] Ready for Sprint 2 integration
