# [T-122] Golden Test Validation

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: All previous tickets
> **Blocks**: None
> **Priority**: 3 (Final Validation)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `tests/golden_tests.rs` - Existing golden tests
- All Sprint 8 implementations

---

## Context

### Background

Golden tests verify that the refactored architecture produces **identical numerical results** to the pre-refactoring implementation. This is the final gate before merging.

---

## Specification

### Test Process

1. Run all existing golden tests with new architecture
2. Verify outputs match saved golden files exactly
3. If differences found, investigate and document

### Golden Tests to Run

- [ ] `golden_single_stage` - Single stage problem
- [ ] `golden_two_stage` - Simple two-stage problem  
- [ ] `golden_multi_stage` - Multi-stage with cuts
- [ ] `golden_with_uncertainty` - Uncertainty handling
- [ ] `golden_large_scale` - 60-stage Brazilian system

### Verification

```rust
#[test]
fn test_golden_outputs_unchanged() {
    let test_cases = [
        "single_stage",
        "two_stage",
        "multi_stage",
        "with_uncertainty",
        "large_scale",
    ];
    
    for case in &test_cases {
        let result = run_sddp_case(case);
        let golden = load_golden_output(case);
        
        assert_eq!(
            result.lower_bound,
            golden.lower_bound,
            "Lower bound mismatch for {}: got {}, expected {}",
            case, result.lower_bound, golden.lower_bound
        );
        
        assert_eq!(
            result.upper_bound,
            golden.upper_bound,
            "Upper bound mismatch for {}",
            case
        );
        
        // Check cut coefficients
        for (stage, (result_cuts, golden_cuts)) in 
            result.cuts.iter().zip(&golden.cuts).enumerate() 
        {
            assert_eq!(
                result_cuts.len(),
                golden_cuts.len(),
                "Cut count mismatch at stage {}",
                stage
            );
            
            for (cut_idx, (rc, gc)) in 
                result_cuts.iter().zip(golden_cuts).enumerate() 
            {
                assert!(
                    (rc.rhs - gc.rhs).abs() < 1e-9,
                    "Cut RHS mismatch at stage {} cut {}",
                    stage, cut_idx
                );
            }
        }
    }
}
```

---

## Acceptance Criteria

- [ ] All existing golden tests pass
- [ ] No numerical differences detected
- [ ] If differences found, documented and explained
- [ ] Test runs complete successfully
- [ ] CI pipeline green

---

## What to Do If Tests Fail

1. **Investigate difference source**
   - Is it numerical precision?
   - Is it algorithm change?
   - Is it a bug?

2. **If precision difference < 1e-9**
   - Document as acceptable
   - Update golden file if appropriate

3. **If algorithmic difference**
   - STOP - investigate before proceeding
   - May indicate bug in refactoring

4. **If bug found**
   - Fix before merging
   - Re-run all tests

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Running existing tests, not creating new ones

---

## Definition of Done

- [ ] All golden tests executed
- [ ] All tests pass
- [ ] Any differences documented
- [ ] CI pipeline green
- [ ] PR ready for final review
