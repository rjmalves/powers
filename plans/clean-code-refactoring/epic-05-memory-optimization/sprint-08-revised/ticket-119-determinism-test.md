# [T-119] Determinism Test: Results Identical With/Without Basis

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-118
> **Blocks**: T-122
> **Priority**: 3 (Validation)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `tests/golden_tests.rs` - Existing golden test patterns
- T-118 implementation (simulation mode)

---

## Context

### Background

This is a **critical validation** test. It verifies that solver results are **identical** regardless of whether basis warm-starting is used.

### Why This Matters

When users load a persisted FCF and run simulation:
- They don't have the basis from training
- Results must match what they would have gotten with basis

If results differ, the architecture is broken.

---

## Specification

### Test Design

Run the same problem twice:
1. With basis warm-starting (training mode)
2. Without basis (simulation mode)

Verify:
- Same optimal objective values
- Same optimal solution values
- Same dual values (for cuts)

### Test Implementation

```rust
#[test]
fn test_determinism_with_without_basis() {
    let system = load_test_system("05-large-scale-brazilian");
    let config = TrainingConfig {
        max_iterations: 10,
        num_forward_passes: 5,
        seed: 42,  // Fixed seed for reproducibility
        // ...
    };
    
    // Run 1: Training with basis
    let mut algorithm1 = create_algorithm(&system);
    algorithm1.train(&config).unwrap();
    
    // Extract forward pass results
    let results_with_basis = run_forward_pass(&mut algorithm1, true);
    
    // Run 2: Simulate without basis
    let mut algorithm2 = create_algorithm(&system);
    // Apply same cuts from training
    algorithm2.load_cuts_from(&algorithm1);
    
    // Clear any cached basis and run simulation mode
    for handler in &mut algorithm2.handlers {
        handler.subproblem.clear_cached_basis();
    }
    let results_without_basis = run_forward_pass(&mut algorithm2, false);
    
    // Verify identical results
    assert_results_equal(&results_with_basis, &results_without_basis);
}

fn run_forward_pass(algorithm: &mut SddpAlgorithm, use_basis: bool) -> ForwardPassResults {
    algorithm.create_iteration_models(use_basis).unwrap();
    
    let mut objectives = Vec::new();
    let mut solutions = Vec::new();
    
    for stage in 0..algorithm.num_stages {
        algorithm.handlers[stage].realize_and_solve(...);
        objectives.push(algorithm.handlers[stage].get_objective());
        solutions.push(algorithm.handlers[stage].get_solution().clone());
    }
    
    algorithm.finalize_iteration(false);
    
    ForwardPassResults { objectives, solutions }
}

fn assert_results_equal(a: &ForwardPassResults, b: &ForwardPassResults) {
    const TOLERANCE: f64 = 1e-9;
    
    assert_eq!(a.objectives.len(), b.objectives.len());
    
    for (i, (obj_a, obj_b)) in a.objectives.iter().zip(&b.objectives).enumerate() {
        assert!(
            (obj_a - obj_b).abs() < TOLERANCE,
            "Objective mismatch at stage {}: {} vs {} (diff: {})",
            i, obj_a, obj_b, (obj_a - obj_b).abs()
        );
    }
    
    for (i, (sol_a, sol_b)) in a.solutions.iter().zip(&b.solutions).enumerate() {
        for (j, (val_a, val_b)) in sol_a.iter().zip(sol_b).enumerate() {
            assert!(
                (val_a - val_b).abs() < TOLERANCE,
                "Solution mismatch at stage {} var {}: {} vs {}",
                i, j, val_a, val_b
            );
        }
    }
}
```

---

## Acceptance Criteria

- [ ] Test runs with/without basis on same problem
- [ ] Objective values match within 1e-9
- [ ] Solution values match within 1e-9
- [ ] Test runs on multiple problem sizes
- [ ] Test documents any numerical differences (should be none)

---

## Why Results Should Be Identical

The simplex algorithm finds the **unique optimal solution** (assuming non-degeneracy) regardless of starting point. Basis only affects:
- Number of iterations (performance)
- Path to optimal (not relevant for results)

The final solution is determined by:
- Constraint matrix
- Bounds
- Objective coefficients

All of which are identical with/without basis.

---

## Edge Cases to Test

1. **Degenerate problems**: Multiple optimal solutions
   - May have different solutions but same objective
   - Test should check objective equality, allow solution variation

2. **Near-feasibility tolerance**: Slight numerical differences
   - Use tolerance in comparisons

3. **Different cut orders**: Verify cuts are applied identically

---

## Effort Estimate

**Points**: 3
**Confidence**: High

---

## Definition of Done

- [ ] Determinism test implemented
- [ ] Runs on test problems
- [ ] Passes with tolerance 1e-9
- [ ] Documents any findings
- [ ] PR merged
