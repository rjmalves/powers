# [T-010] Extract Thermal and Exchange Solution Extraction Methods

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Solution Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-008](./ticket-008-solution-extractor-scaffold.md)
> **Blocks**: [T-011](./ticket-011-extract-remaining-solutions.md)

---

## ⚠️ CRITICAL: Behavioral Equivalence

This ticket implements extraction logic for thermal generation and power exchange. These methods handle **optional fields** and **computed values** (exchange = direct - reverse).

Run golden tests after EVERY method implementation.

---

## Files to Read Before Starting

- `src/model/solution_extract.rs` - SolutionExtractor scaffold from T-008
- `src/subproblem.rs:1862-1910` - Deficit and exchange extraction
- `src/subproblem.rs:1898-1910` - Thermal generation extraction
- `src/solver.rs` - Solution struct

---

## Context

### Background

Thermal and exchange extractions have additional complexity:

1. **Thermal generation**: Optional (some systems have no thermal plants)
2. **Deficit**: Always present (unmet load)
3. **Exchange**: Computed as `direct_exchange - reverse_exchange`

### Current Functions (from subproblem.rs)

```rust
fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    let first = *self.variables.deficit.first().unwrap();
    let last = *self.variables.deficit.last().unwrap() + 1;
    realization.deficit.clone_from_slice(&solution.colvalue[first..last]);
}

fn get_net_exchange_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    if !self.variables.direct_exchange.is_empty() {
        let direct_first = *self.variables.direct_exchange.first().unwrap();
        let direct_last = *self.variables.direct_exchange.last().unwrap() + 1;
        let reverse_first = *self.variables.reverse_exchange.first().unwrap();
        let reverse_last = *self.variables.reverse_exchange.last().unwrap() + 1;
        
        realization.exchange.clone_from_slice(&solution.colvalue[direct_first..direct_last]);
        realization.exchange
            .iter_mut()
            .zip(&solution.colvalue[reverse_first..reverse_last])
            .for_each(|(direct, reverse)| *direct -= *reverse);
    }
}

fn get_thermal_gen_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    if !self.variables.thermal_gen.is_empty() {
        let first = *self.variables.thermal_gen.first().unwrap();
        let last = *self.variables.thermal_gen.last().unwrap() + 1;
        realization.thermal_generation.clone_from_slice(&solution.colvalue[first..last]);
    }
}
```

---

## Specification

### Methods to Implement

#### 1. Deficit Extraction

```rust
#[inline]
pub fn extract_deficit_into(&self, solution: &Solution, target: &mut [f64]) {
    let range = self.var_indices.deficit_range();
    target.copy_from_slice(&solution.colvalue[range]);
}

#[inline]
pub fn extract_deficit(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_deficit_into(solution, &mut realization.deficit);
}
```

#### 2. Exchange Extraction (Complex: direct - reverse)

```rust
/// Extract net exchange values (direct - reverse) into target slice.
///
/// # Panics
///
/// Panics if exchange variables don't exist. Use `has_exchange()` to check first.
#[inline]
pub fn extract_exchange_into(&self, solution: &Solution, target: &mut [f64]) {
    let direct_range = self.var_indices.direct_exchange_range()
        .expect("exchange_into called without exchange variables");
    let reverse_range = self.var_indices.reverse_exchange_range()
        .expect("exchange_into called without exchange variables");
    
    // Copy direct values first
    target.copy_from_slice(&solution.colvalue[direct_range]);
    
    // Subtract reverse values
    target.iter_mut()
        .zip(&solution.colvalue[reverse_range])
        .for_each(|(direct, reverse)| *direct -= *reverse);
}

#[inline]
pub fn extract_exchange(&self, solution: &Solution, realization: &mut Realization) {
    if self.has_exchange() {
        self.extract_exchange_into(solution, &mut realization.exchange);
    }
}
```

#### 3. Thermal Generation Extraction (Optional)

```rust
/// Extract thermal generation values into target slice.
///
/// # Panics
///
/// Panics if thermal variables don't exist. Use `has_thermal()` to check first.
#[inline]
pub fn extract_thermal_gen_into(&self, solution: &Solution, target: &mut [f64]) {
    let range = self.var_indices.thermal_gen_range()
        .expect("thermal_gen_into called without thermal variables");
    target.copy_from_slice(&solution.colvalue[range]);
}

#[inline]
pub fn extract_thermal_gen(&self, solution: &Solution, realization: &mut Realization) {
    if self.has_thermal() {
        self.extract_thermal_gen_into(solution, &mut realization.thermal_generation);
    }
}
```

#### 4. Marginal Costs Extraction (Dual)

```rust
#[inline]
pub fn extract_marginal_costs_into(&self, solution: &Solution, target: &mut [f64]) {
    let range = self.con_indices.load_balance_range();
    target.copy_from_slice(&solution.rowdual[range]);
}

#[inline]
pub fn extract_marginal_costs(&self, solution: &Solution, realization: &mut Realization) {
    self.extract_marginal_costs_into(solution, &mut realization.marginal_cost);
}
```

### Behavior

- **Deficit**: Always present, simple range copy
- **Exchange**: Computed as direct - reverse; guard with `has_exchange()`
- **Thermal**: Optional; guard with `has_thermal()`
- **Marginal costs**: Dual values from load balance constraints

---

## Acceptance Criteria

- [ ] `extract_deficit_into` and `extract_deficit` implemented
- [ ] `extract_exchange_into` and `extract_exchange` implemented with subtraction logic
- [ ] `extract_thermal_gen_into` and `extract_thermal_gen` implemented with optional handling
- [ ] `extract_marginal_costs_into` and `extract_marginal_costs` implemented
- [ ] High-level methods handle optional fields (no panic if not present)
- [ ] Low-level `_into` methods panic if called without variables (expect pattern)
- [ ] Unit tests verify extraction correctness
- [ ] Unit tests verify exchange subtraction
- [ ] Golden tests pass

### Correctness Verification

- [ ] Exchange = direct - reverse (same as original)
- [ ] Optional fields handled identically to original
- [ ] No panics when optional fields are missing (high-level API)

---

## Implementation Guide

### Suggested Approach

1. **Implement deficit extraction** (simplest):
   ```rust
   #[inline]
   pub fn extract_deficit_into(&self, solution: &Solution, target: &mut [f64]) {
       let range = self.var_indices.deficit_range();
       target.copy_from_slice(&solution.colvalue[range]);
   }
   
   #[inline]
   pub fn extract_deficit(&self, solution: &Solution, realization: &mut Realization) {
       self.extract_deficit_into(solution, &mut realization.deficit);
   }
   ```

2. **Implement thermal extraction** (optional):
   ```rust
   #[inline]
   pub fn extract_thermal_gen_into(&self, solution: &Solution, target: &mut [f64]) {
       let range = self.var_indices.thermal_gen_range()
           .expect("thermal_gen_into called without thermal variables");
       target.copy_from_slice(&solution.colvalue[range]);
   }
   
   #[inline]
   pub fn extract_thermal_gen(&self, solution: &Solution, realization: &mut Realization) {
       if self.has_thermal() {
           self.extract_thermal_gen_into(solution, &mut realization.thermal_generation);
       }
   }
   ```

3. **Implement exchange extraction** (complex):
   ```rust
   #[inline]
   pub fn extract_exchange_into(&self, solution: &Solution, target: &mut [f64]) {
       let direct_range = self.var_indices.direct_exchange_range()
           .expect("exchange_into called without exchange variables");
       let reverse_range = self.var_indices.reverse_exchange_range()
           .expect("exchange_into called without exchange variables");
       
       // Copy direct values
       target.copy_from_slice(&solution.colvalue[direct_range]);
       
       // Subtract reverse values in-place
       target.iter_mut()
           .zip(&solution.colvalue[reverse_range])
           .for_each(|(direct, reverse)| *direct -= *reverse);
   }
   
   #[inline]
   pub fn extract_exchange(&self, solution: &Solution, realization: &mut Realization) {
       if self.has_exchange() {
           self.extract_exchange_into(solution, &mut realization.exchange);
       }
   }
   ```

4. **Implement marginal costs extraction**:
   ```rust
   #[inline]
   pub fn extract_marginal_costs_into(&self, solution: &Solution, target: &mut [f64]) {
       let range = self.con_indices.load_balance_range();
       target.copy_from_slice(&solution.rowdual[range]);
   }
   
   #[inline]
   pub fn extract_marginal_costs(&self, solution: &Solution, realization: &mut Realization) {
       self.extract_marginal_costs_into(solution, &mut realization.marginal_cost);
   }
   ```

5. **Add unit tests**:
   ```rust
   #[test]
   fn test_extract_exchange_subtraction() {
       // Mock solution where colvalue = [0,1,2,3,4,5,6,7,8,9]
       // direct_range = 2..5, reverse_range = 5..8
       // direct = [2,3,4], reverse = [5,6,7]
       // result = [2-5, 3-6, 4-7] = [-3, -3, -3]
       
       let solution = create_test_solution(10, 5);
       let var_indices = create_indices_with_exchange(2..5, 5..8);
       let extractor = SolutionExtractor::new(var_indices, mock_con_indices());
       
       let mut target = vec![0.0; 3];
       extractor.extract_exchange_into(&solution, &mut target);
       
       assert_eq!(target, vec![-3.0, -3.0, -3.0]);
   }
   
   #[test]
   fn test_extract_thermal_when_not_present() {
       // Extractor without thermal variables
       let var_indices = create_indices_without_thermal();
       let extractor = SolutionExtractor::new(var_indices, mock_con_indices());
       
       let solution = create_test_solution(10, 5);
       let mut realization = mock_realization();
       
       // Should not panic, should not modify realization.thermal_generation
       extractor.extract_thermal_gen(&solution, &mut realization);
       
       // thermal_generation should be unchanged
       assert!(realization.thermal_generation.iter().all(|&v| v == 0.0));
   }
   ```

6. **Run tests**:
   ```bash
   cargo test solution_extract
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/solution_extract.rs` | Implement 4 extraction methods |

### Patterns to Follow

- Use `expect()` in `_into` methods for required optional fields
- Use `if self.has_X()` guard in high-level methods
- Maintain the exact subtraction pattern for exchange

### Pitfalls to Avoid

- ⚠️ Don't change the exchange calculation (direct - reverse, not reverse - direct)
- ⚠️ Don't panic in high-level methods—guard with `has_X()`
- ⚠️ Do panic in `_into` methods if called incorrectly—use `expect()`
- ⚠️ Verify `iter_mut().zip()` produces same result as original

---

## Testing Requirements

### Unit Tests

- [ ] Test `extract_deficit_into` with mock data
- [ ] Test `extract_exchange_into` verifies subtraction
- [ ] Test `extract_thermal_gen_into` with mock data
- [ ] Test `extract_marginal_costs_into` with mock data
- [ ] Test high-level methods skip when fields not present
- [ ] Test `_into` methods panic when called incorrectly

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Document panic behavior in `_into` methods
- [ ] Document that high-level methods are safe to call even without optional fields

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Similar to T-009 but with optional handling complexity

---

## Definition of Done

- [ ] All 4 extraction methods implemented
- [ ] Optional handling correct
- [ ] Exchange subtraction verified
- [ ] Unit tests passing
- [ ] Golden tests passing
- [ ] Code documented
