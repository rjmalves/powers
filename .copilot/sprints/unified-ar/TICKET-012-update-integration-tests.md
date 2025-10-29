# TICKET-012: Update Integration Tests and Examples

**Sprint:** 3 - Cleanup  
**Phase:** 3 - Clean Up Dependencies  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

With the unified AR model in place and obsolete code removed, all integration tests and examples need to be updated to use the new architecture. This ticket ensures that:

1. All examples run correctly with the unified model
2. Integration tests validate the unified architecture
3. Test coverage is maintained or improved
4. Documentation examples are current

This is critical for validating that the refactoring achieves its goals.

## Acceptance Criteria

- [ ] Given example 06 (PAR deterministic), when run, then it produces correct output matching baseline
- [ ] Given example 07 (PAR stochastic), when run, then it produces correct output matching baseline
- [ ] Given all integration tests, when run, then they pass with unified model
- [ ] Given test coverage, when measured, then it is ≥ previous coverage
- [ ] Given documentation examples, when reviewed, then they reflect unified architecture
- [ ] Performance: Examples should run 15-25% faster with unified model

## Tasks

### Implementation

- [ ] Update example 06 (PAR deterministic model):
  - Verify JSON inputs compatible with unified model
  - Run and capture baseline output
  - Update any configuration files if needed
- [ ] Update example 07 (PAR with inflow state):
  - Verify JSON inputs compatible with unified model
  - Verify StorageAndInflowState still works with UnifiedInflowModel
  - Run and capture baseline output
  - Update README explaining unified model usage with both state types
- [ ] Update integration tests in `tests/`:
  - `test_factory_multi_node_prestudy.rs` - Multi-node PreStudy (PAR)
  - `test_par_estimation.rs` - PAR estimation
  - `test_par_validation.rs` - PAR validation
  - `test_scenario_generation_integration.rs` - Scenario generation with AR
  - Tests covering both StorageState and StorageAndInflowState with unified AR
  - Any other tests referencing state types or AR models
- [ ] Add new integration tests for unified model:
  - Test independent vs AR(1) vs AR(2) produce correct results
  - Test lag buffer updates across stages
  - Test residual extraction in realizations
  - Test cut generation with AR dynamics
  - Test both StorageState and StorageAndInflowState work with unified AR
  - Test that both state types produce valid cuts and convergence
- [ ] Update benchmark tests in `benches/`:
  - Ensure benchmarks run with unified model
  - Add specific benchmarks for realize_uncertainties speedup
- [ ] Update unit tests for modified modules:
  - Tests in `src/subproblem.rs` test module
  - Tests in `src/state.rs` test module
  - Any tests in `src/sddp/` that depend on state structure

### Testing

- [ ] Regression test: Run all examples and compare output to baseline
  - Example 01-05 should be unchanged
  - Example 06-07 should work correctly (may have numerical differences)
- [ ] Numerical validation: Verify lower bound convergence matches expected
- [ ] Performance test: Measure example runtimes, expect 15-25% improvement on AR examples
- [ ] Coverage test: Run `cargo tarpaulin` or coverage tool, ensure ≥ previous coverage
- [ ] Stress test: Run large example (05-large-scale-brazilian) to ensure scalability
- [ ] Memory test: Profile memory usage, ensure no regressions

### Documentation

- [ ] Update example READMEs to explain unified AR model
- [ ] Add section to main README explaining AR model representation
- [ ] Update quickstart guide with unified model example
- [ ] Add troubleshooting section for common unified model issues
- [ ] Update CHANGELOG.md with "Updated: All examples use UnifiedInflowModel"

## Technical Notes

### Example Update Checklist

For each example:

- [ ] Run with old code, capture output (baseline)
- [ ] Run with new code, capture output (updated)
- [ ] Compare outputs:
  - Lower bound values (should match within tolerance)
  - Policy values (may differ slightly due to numerical improvements)
  - Solve times (should be faster)
  - Memory usage (should be similar or better)
- [ ] Document any intentional differences

### Baseline Comparison

**Acceptable Differences:**

- Lower bound: ± 0.1% (numerical precision improvements)
- Solve time: -15% to -25% (expected speedup)
- Memory: -20% (from removed redundancy)

**Unacceptable Differences:**

- Lower bound: > 1% difference (indicates bug)
- Solve time: slower (indicates performance regression)
- Solve failures: any failures (indicates correctness issue)

### Integration Test Updates

**Key Tests to Update:**

1. **test_factory_multi_node_prestudy.rs**

   - Tests PAR model with multi-node PreStudy
   - Update to use UnifiedInflowModel
   - Verify lag initialization from PreStudy nodes

2. **test_par_estimation.rs**

   - Tests PAR coefficient estimation
   - Update to residual space only
   - Verify estimated coefficients are stationary

3. **test_scenario_generation_integration.rs**

   - Tests scenario generation with AR
   - Update to use UnifiedInflowModel
   - Verify scenarios follow AR dynamics

4. **test_sddp_algorithm.rs**
   - Tests full SDDP algorithm
   - Update to use UnifiedInflowModel
   - Verify convergence behavior

### New Integration Tests to Add

```rust
#[test]
fn test_unified_model_independent_vs_ar() {
    // Verify independent model (AR(0)) produces different results than AR(1)
    // Setup two identical systems except AR coefficients
    // Run both and compare outputs
}

#[test]
fn test_lag_buffer_updates() {
    // Verify lag buffer correctly updated across forward pass
    // Check that Z'_{t-1} from stage t becomes Z'_{t-2} at stage t+1
}

#[test]
fn test_residual_observation_consistency() {
    // Verify Y_t = μ_s + σ_s * Z'_t holds in solution
    // Extract both observation and residual from realization
    // Check transformation is correct
}

#[test]
fn test_cut_generation_with_ar_duals() {
    // Verify cut generation includes AR lag duals
    // Generate cut in backward pass
    // Check gradient includes lag dual contributions
}
```

### Example Directory Structure

```
examples/
├── 01-deterministic/          (unchanged)
├── 02-stochastic/             (unchanged)
├── 03-multistage/             (unchanged)
├── 04-cascade/                (unchanged)
├── 05-large-scale-brazilian/  (unchanged)
├── 06-par-model/              (updated: verify compatibility)
│   ├── README.md              (update with unified model explanation)
│   └── ...
└── 07-par-model-with-inflow-state/  (updated: verify both state types work)
    ├── README.md              (update: explain state choice tradeoffs)
    └── ...                    (verify unified AR works with StorageAndInflowState)
```

**Consider renaming 07:** Name is still appropriate since it demonstrates StorageAndInflowState usage with unified AR model. Update README to explain when to use StorageAndInflowState vs StorageState.

### Performance Validation

**Metrics to capture:**

- Subproblem solve time (per iteration)
- Forward pass time (per iteration)
- Backward pass time (per iteration)
- Total algorithm time
- Memory usage (peak and average)

**Expected improvements:**

- realize_uncertainties: 15-25% faster
- Overall algorithm: 10-15% faster (smaller proportion due to solver time dominance)
- Memory: ~20% reduction from removed redundancy

### Coverage Maintenance

**Current coverage areas (maintain or improve):**

- State trait implementations: → Should stay same (both state types still exist)
- UnifiedInflowModel: → New coverage area
- Subproblem operations: → Should stay same or improve
- SDDP algorithm: → Should stay same
- Input validation: → Should stay same
- PAR generation: → May decrease slightly from removed observation space code

**Target coverage:** ≥ 80% (or maintain current level)

## Dependencies

- **Blocked by**:
  - TICKET-008 (realize_uncertainties must work)
  - TICKET-010 (State trait interface refactored)
  - TICKET-011 (PAR generator updated)
- **Blocks**: None (last cleanup ticket)
- **Related**: TICKET-013 (performance optimization builds on validated tests)

## References

- `tests/` - Integration test directory
- `examples/` - Example directory
- `benches/` - Benchmark directory
- UNIFIED_AR_ROADMAP.md - Section 2.3 (Update tests and benchmarks)

## Validation Checklist

Before marking this ticket as done:

- [ ] All examples run successfully
- [ ] All integration tests pass
- [ ] All unit tests pass
- [ ] Benchmarks run and show expected speedup
- [ ] Coverage maintained or improved (run coverage tool)
- [ ] Example 06 output matches baseline (within tolerance)
- [ ] Example 07 output matches baseline (within tolerance)
- [ ] Performance improvement measured and documented
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Code reviewed by at least one team member
