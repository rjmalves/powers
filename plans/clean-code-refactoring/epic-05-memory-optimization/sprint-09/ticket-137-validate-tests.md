# [T-137] Validate All Tests Pass with New Default Allocator

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: T-136
> **Blocks**: T-140

---

## Context

### Background

After making the new allocator the default (T-136), we must ensure all existing tests pass. Different allocators can expose latent bugs or cause subtle behavioral differences.

### Current State

- 589+ tests exist in the test suite
- Tests pass with glibc allocator (current default)
- New allocator is now the default

## Specification

### Test Scope

1. **Unit tests**: All `cargo test` tests
2. **Integration tests**: All tests in `tests/` directory
3. **Golden tests**: Numerical validation tests
4. **Doc tests**: Examples in documentation

### Expected Behavior

- All tests that pass with glibc should pass with new allocator
- Numerical results should be identical (allocator shouldn't affect math)
- No new warnings or errors

## Acceptance Criteria

- [ ] `cargo test` passes (all unit and integration tests)
- [ ] `cargo test --release` passes
- [ ] Golden tests produce identical results
- [ ] No new warnings introduced
- [ ] Test run time similar to baseline (±10%)

## Implementation Guide

### Suggested Approach

1. Run full test suite with new default allocator
2. Compare to baseline (glibc) test results
3. Investigate any failures
4. Document any expected differences

### Commands

```bash
# Run all tests with new default
cargo test 2>&1 | tee test_results.log

# Run release tests
cargo test --release 2>&1 | tee test_results_release.log

# Count results
grep -E "^test .* ok$" test_results.log | wc -l
grep -E "^test .* FAILED$" test_results.log

# Compare to baseline (run with system allocator)
cargo test --no-default-features 2>&1 | tee test_baseline.log
diff <(grep "^test " test_results.log | sort) <(grep "^test " test_baseline.log | sort)
```

### Golden Test Verification

```bash
# Run specific golden tests
cargo test --release golden -- --nocapture

# Check for numerical differences
cargo test --release test_golden_case_ -- --nocapture
```

### Pitfalls to Avoid

- ⚠️ Some tests may have timing dependencies - run multiple times
- ⚠️ Parallel test execution may show different ordering
- ⚠️ Check for flaky tests that might coincidentally fail

## Testing Requirements

### Full Suite

- [ ] All 589+ tests pass
- [ ] No new test failures vs baseline
- [ ] Release and debug builds both pass

### Comparison

- [ ] Same tests pass/fail as with glibc
- [ ] Test timing within 10% of baseline

## Documentation Requirements

- [ ] Document any test adjustments needed
- [ ] Note any expected behavioral differences

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Running existing tests, investigating failures if any

## Definition of Done

- [ ] Full test suite passes with new allocator
- [ ] Comparison to baseline documented
- [ ] Any issues investigated and resolved
- [ ] Ready for CI integration
