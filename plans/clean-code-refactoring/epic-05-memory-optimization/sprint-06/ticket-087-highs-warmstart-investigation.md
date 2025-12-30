# [T-087] Investigate HiGHS Warm-Start API and Basis Reuse

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 6: HiGHS Solver Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-093

---

## Context

### Background

DHAT profiling revealed that `HFactor::setupGeneral` accounts for **44.9% of all heap allocations** (39.6 GB) during SDDP training. This function is called during every `Highs_run()` invocation to set up the factorization matrix for the simplex solver.

In SDDP training, we solve the same LP model thousands of times with only bound changes between solves. If HiGHS can reuse the previous basis and skip full factorization setup, we could eliminate a significant portion of these allocations.

### Relation to Epic

Part of Epic 5's goal to achieve zero-allocation hot paths and minimize memory churn.

### Current State

- Every call to `Model::solve()` → `Highs_run()` triggers full factorization setup
- We already use `Model::try_set_basis()` for basis warm-starting between stages
- It's unclear if this prevents `HFactor::setupGeneral` allocations

## Specification

### Investigation Tasks

1. **Review HiGHS documentation** for warm-start capabilities:
   - `simplex_strategy` options
   - `simplex_warm_start` option
   - Basis persistence between solves

2. **Analyze HiGHS source code** (if needed):
   - When is `HFactor::setupGeneral` called?
   - What conditions skip factorization setup?
   - What does `Highs_setBasis()` actually do?

3. **Experiment with options**:
   - Test with `simplex_warm_start = on`
   - Test with explicit `Highs_setBasis()` before `Highs_run()`
   - Measure allocation impact with DHAT

4. **Document findings**:
   - Create `docs/HIGHS_WARM_START_INVESTIGATION.md`
   - Include API usage patterns that minimize allocations
   - Document any limitations or caveats

### Expected Outputs

- Investigation document with findings
- Recommended HiGHS options for minimal allocations
- Code changes (if warm-start is effective) or rationale for why not

### Behavior

- **If warm-start is effective**: Implement the pattern in `Model::solve()`
- **If warm-start is not effective**: Document why and alternative approaches
- **If partially effective**: Document which scenarios benefit

### Error Handling

- If HiGHS API is unclear, check HiGHS GitHub issues and documentation
- If warm-start causes numerical issues, document and avoid

## Acceptance Criteria

- [x] HiGHS warm-start documentation reviewed
- [x] At least 3 different warm-start approaches tested with DHAT
- [x] Findings documented in `docs/HIGHS_WARM_START_INVESTIGATION.md`
- [x] If warm-start reduces allocations: implementation proposed
- [x] If warm-start doesn't help: rationale documented
- [x] No numerical divergence in any tested configuration

**Status**: ✅ Complete

**Key Finding**: HFactor allocations are inherent to HiGHS internal implementation and cannot be eliminated via API. Warm-starting is already properly implemented via `try_set_basis()`. Current HiGHS options are already optimized (presolve off, scaling off, threading off).

## Implementation Guide

### Suggested Approach

1. **Create test harness for DHAT profiling**:
   ```rust
   #[test]
   #[ignore] // Run manually with DHAT
   fn test_highs_allocation_patterns() {
       let mut model = Model::new();
       // Setup small LP
       for _ in 0..100 {
           model.change_row_bounds(0, 0.0, 1.0);
           model.solve();
       }
   }
   ```

2. **Test baseline allocations**:
   ```bash
   cargo test --release test_highs_allocation_patterns -- --ignored
   valgrind --tool=dhat ./target/release/deps/powers_rs-*
   ```

3. **Test with `simplex_warm_start`**:
   ```rust
   model.set_option("simplex_warm_start", "on");
   ```

4. **Test with explicit basis setting**:
   ```rust
   let basis = model.get_basis();
   model.change_row_bounds(0, 0.0, 1.0);
   model.set_basis(&basis);
   model.solve();
   ```

5. **Test with `clear_solver()` vs without**:
   - Does calling `clear_solver()` affect factorization reuse?

### Key Files to Read

- `src/solver.rs` - Current HiGHS wrapper implementation
- `src/subproblem.rs:set_default_solver_options()` - Current options
- HiGHS documentation: https://ergo-code.github.io/HiGHS/

### Patterns to Follow

- Document-driven investigation (write findings as you go)
- Reproducible benchmarks with DHAT

### Pitfalls to Avoid

- ⚠️ Don't change production code until investigation is complete
- ⚠️ Always verify numerical correctness with golden tests
- ⚠️ DHAT overhead is ~20x slower; use small test cases for iteration

## Testing Requirements

### Investigation Tests

- [ ] Test warm-start with bound changes only
- [ ] Test warm-start with new row additions (cuts)
- [ ] Test warm-start after `clear_solver()`
- [ ] Compare allocation counts between approaches

### Validation Tests

- [ ] Golden tests pass with any configuration changes
- [ ] Benchmark shows no performance regression

## Documentation Requirements

- [ ] Create `docs/HIGHS_WARM_START_INVESTIGATION.md`
- [ ] Include DHAT comparison tables
- [ ] Document recommended configuration
- [ ] Update `docs/MEMORY_BEHAVIOR.md` with findings

## Dependencies

- **Blocked By**: None
- **Blocks**: T-093 (DHAT profiling)
- **Related**: T-088 (batch bounds), T-092 (threading)

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Investigation work with uncertain outcomes; may require reading HiGHS source

## Definition of Done

- [ ] Investigation complete
- [ ] Findings documented
- [ ] Recommendation made (implement or not)
- [ ] If implementing: changes proposed in separate ticket
- [ ] No test regressions
