# SDDP Testing Strategy - Implementation Tickets

**Document Version**: 1.0  
**Date**: 2025-11-06  
**Based On**: TESTING_STRATEGY.md  
**Total Estimated Duration**: 4-6 weeks

---

## Table of Contents

1. [Sprint Overview](#sprint-overview)
2. [Phase 1: Foundation (Week 1-2)](#phase-1-foundation-week-1-2)
3. [Phase 2: Mathematical Validation (Week 3)](#phase-2-mathematical-validation-week-3)
4. [Phase 3: Integration Tests (Week 4-5)](#phase-3-integration-tests-week-4-5)
5. [Phase 4: End-to-End & Performance (Week 6)](#phase-4-end-to-end--performance-week-6)
6. [Phase 5: Documentation & Polish (Ongoing)](#phase-5-documentation--polish-ongoing)

---

## Sprint Overview

### Sprint 1 (Week 1): Fix Foundations
- **Goal**: Get all test fixtures compiling and establish working baseline
- **Tickets**: TEST-001 through TEST-006
- **Key Deliverable**: All fixtures compile and run

### Sprint 2 (Week 2): Critical Unit Tests
- **Goal**: Add missing critical unit tests
- **Tickets**: TEST-007 through TEST-012
- **Key Deliverable**: 35+ new critical unit tests

### Sprint 3 (Week 3): Mathematical Validation
- **Goal**: Establish algorithm correctness tests
- **Tickets**: TEST-013 through TEST-018
- **Key Deliverable**: 25 mathematical validation tests

### Sprint 4 (Week 4): Algorithm Integration
- **Goal**: Rebuild core algorithm integration tests
- **Tickets**: TEST-019 through TEST-024
- **Key Deliverable**: 20 algorithm integration tests

### Sprint 5 (Week 5): Feature Integration
- **Goal**: Test AR models, risk measures, and system features
- **Tickets**: TEST-025 through TEST-030
- **Key Deliverable**: 40 feature integration tests

### Sprint 6 (Week 6): E2E & Performance
- **Goal**: Complete test pyramid with E2E and benchmarks
- **Tickets**: TEST-031 through TEST-036
- **Key Deliverable**: E2E tests, benchmarks, and CI/CD pipeline

---

## Phase 1: Foundation (Week 1-2)

### TEST-001: Fix Core Test Fixtures

**Context**

The test fixtures in `tests/fixtures/` are broken due to API changes in `System`, `Hydro`, and `Bus` structures. Many field names have changed (e.g., `min_volume` → `min_storage`), and required fields have been added. This is blocking all integration tests.

**Acceptance Criteria**

- [ ] All fixtures in `tests/fixtures/systems.rs` compile without errors
- [ ] All fixtures in `tests/fixtures/benchmarks.rs` compile without errors
- [ ] `simple_2stage_reservoir.rs` compiles and can be instantiated
- [ ] At least 3 basic integration tests can run using the fixed fixtures
- [ ] No warnings related to deprecated field usage

**Tasks**

### Implementation

- [ ] Update `Hydro` struct instantiations: replace `min_volume`/`max_volume` with `min_storage`/`max_storage`
- [ ] Add required fields to `Bus` struct: `hydro_ids`, `source_line_ids`, `target_line_ids`
- [ ] Update `System` construction to match current API
- [ ] Fix `train()` method signatures (verify parameter order and types)
- [ ] Update `Config` struct initialization if needed
- [ ] Run `cargo check --tests` to verify all fixtures compile

### Testing

- [ ] Verify each fixture can be instantiated in a simple test
- [ ] Test that `create_simple_2stage_system()` produces valid SDDP instance
- [ ] Test that `create_deterministic_2stage()` can run 1 training iteration
- [ ] Add regression test to catch future API breaks in fixtures

### Documentation

- [ ] Add comments documenting required fields for each structure
- [ ] Create migration guide notes if field mappings are non-obvious
- [ ] Update fixture documentation with current API examples

**Technical Notes**

- Start with `tests/fixtures/systems.rs` as it's used by most other fixtures
- Use `git grep "min_volume"` to find all occurrences needing updates
- Check `src/system.rs` for the canonical field names and required fields
- The `Bus` struct now tracks connected components via ID vectors
- May need to update `initial_condition` handling if API changed

**Dependencies**

- Blocked by: None (foundational work)
- Blocks: All integration test tickets
- Related: TEST-002 (fixture enhancement)

**Estimated Effort**

3 story points (2-3 days, confidence: high)

---

### TEST-002: Create Test Utility Library

**Context**

Multiple test files need common assertion helpers for checking convergence properties, vector equality, and mathematical invariants. Creating a shared utility library will reduce code duplication and improve test maintainability.

**Acceptance Criteria**

- [ ] `tests/utils/assertions.rs` module exists with documented assertion functions
- [ ] `tests/utils/validation.rs` module exists with result validation helpers
- [ ] At least 3 existing tests refactored to use the new utilities
- [ ] All utility functions have doc comments with examples
- [ ] Utilities support customizable error messages

**Tasks**

### Implementation

- [ ] Create `tests/utils/` directory structure
- [ ] Implement `assert_monotonic_non_decreasing(values, tolerance)` function
- [ ] Implement `assert_cut_validity(cut, state, objective)` function
- [ ] Implement `assert_vector_eq(a, b, epsilon)` function with element-wise comparison
- [ ] Implement `assert_relative_eq(a, b, rel_tolerance)` for percentage-based checks
- [ ] Add `validate_convergence(result, max_gap)` helper
- [ ] Add `validate_trajectory(trajectory)` for state continuity checks
- [ ] Create `tests/utils/mod.rs` to re-export utilities

### Testing

- [ ] Unit test for `assert_monotonic_non_decreasing` with passing case
- [ ] Unit test for `assert_monotonic_non_decreasing` with failing case
- [ ] Unit test for `assert_vector_eq` with various epsilon values
- [ ] Unit test for `assert_cut_validity` with valid and invalid cuts
- [ ] Integration test using utilities in a real SDDP test

### Documentation

- [ ] Add doc comments to each assertion function
- [ ] Include usage examples in doc comments
- [ ] Create `tests/utils/README.md` documenting available utilities
- [ ] Add section to main test strategy doc referencing utilities

**Technical Notes**

- Use `approx` crate for floating-point comparisons if not already available
- Assertions should provide helpful error messages showing actual vs expected
- Consider using custom types for tolerances (absolute vs relative)
- Monotonicity check should handle both strict and non-strict variants
- Vector equality should report which element first failed and by how much

**Dependencies**

- Blocked by: None
- Blocks: TEST-007, TEST-013 (will benefit from these utilities)
- Related: All testing tickets

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-003: Add Critical Unit Tests for cut.rs

**Context**

The `cut.rs` module only has 4 tests, but cut correctness is fundamental to SDDP. We need tests for cut evaluation precision, coefficient validation, domination detection, and numerical stability. These are critical for ensuring the algorithm's mathematical correctness.

**Acceptance Criteria**

- [ ] At least 12 new unit tests added to `src/cut.rs`
- [ ] Cut evaluation at training state matches objective (most critical test)
- [ ] Storage coefficient signs validated (negative for minimization)
- [ ] Numerical precision tests cover extreme values
- [ ] Kahan summation validated if used for cut aggregation
- [ ] All tests pass and increase coverage of `cut.rs` by >50%

**Tasks**

### Implementation

- [ ] Add `test_cut_evaluation_matches_lp_objective()` - validate cut height at training state
- [ ] Add `test_cut_coefficient_signs()` - storage coefficients should be negative
- [ ] Add `test_cut_evaluation_numerical_precision()` - test with large/small values
- [ ] Add `test_cut_domination_detection()` - identify when one cut dominates another
- [ ] Add `test_kahan_summation_in_cut_aggregation()` - verify numerical stability
- [ ] Add `test_cut_with_zero_coefficients()` - edge case for inactive hydros
- [ ] Add `test_cut_with_large_state_dimension()` - scalability to 50+ hydros
- [ ] Add `test_cut_evaluation_empty_state()` - edge case handling
- [ ] Add `test_cut_clone_and_equality()` - proper derive implementations
- [ ] Add `test_cut_serialization()` - if cuts are saved/loaded
- [ ] Add `test_cut_interpolation()` - if relevant to implementation
- [ ] Add `test_cut_numerical_stability_small_coefficients()` - precision edge case

### Testing

- [ ] All new tests pass in isolation
- [ ] Tests pass in parallel execution
- [ ] Tests complete in < 10ms each
- [ ] Run with `--release` to catch optimization issues
- [ ] Verify tests catch actual bugs by temporarily breaking implementation

### Documentation

- [ ] Add module-level doc comment explaining cut testing strategy
- [ ] Document the mathematical property being tested in each test
- [ ] Add inline comments explaining non-obvious test setups
- [ ] Update TESTING_STRATEGY.md progress tracking

**Technical Notes**

- Most critical test: cut height at training state must equal objective value
- Use `approx` crate's `assert_abs_diff_eq!` with epsilon = 1e-4 for LP solver tolerance
- Storage coefficients negative because: ∂V/∂storage < 0 (more water → less cost)
- For numerical precision: test with values like 1e8 storage, 1e-8 coefficients
- Kahan summation prevents catastrophic cancellation in dot products
- Consider property-based testing with `proptest` for coefficient generation
- Cut domination: cut A dominates B if A(x) >= B(x) for all feasible x

**Dependencies**

- Blocked by: TEST-001 (needs working fixtures for realistic cuts)
- Blocks: TEST-013 (mathematical validation builds on these)
- Related: TEST-004 (risk measure tests also validate cuts)

**Estimated Effort**

3 story points (2-3 days, confidence: high)

---

### TEST-004: Add Critical Unit Tests for risk_measure.rs

**Context**

Risk measures (CVaR, expectation, worst-case) are core to SDDP but only have 3 tests. We need to validate that CVaR bounds expectation, probability adjustments are correct, and risk measures satisfy monotonicity properties.

**Acceptance Criteria**

- [ ] At least 8 new unit tests added to `src/risk_measure.rs`
- [ ] CVaR >= Expectation property validated for cost minimization
- [ ] Probability weight adjustments tested for CVaR tail scenarios
- [ ] All three risk measures (CVaR, Expectation, WorstCase) tested
- [ ] Edge cases covered: α=0.01, α=0.99, single scenario, equal costs
- [ ] All tests pass with clear failure messages

**Tasks**

### Implementation

- [ ] Add `test_cvar_probability_adjustment()` - tail scenarios get higher weight
- [ ] Add `test_cvar_bounds_expectation()` - CVaR_α(X) >= E[X] for costs
- [ ] Add `test_expectation_risk_is_identity()` - probabilities unchanged
- [ ] Add `test_worst_case_selects_maximum()` - all weight on worst scenario
- [ ] Add `test_cvar_alpha_parameter_validation()` - reject invalid α
- [ ] Add `test_risk_measure_monotonicity()` - worse scenarios → higher risk
- [ ] Add `test_cvar_extreme_alpha_values()` - α→0 and α→1 limits
- [ ] Add `test_risk_measures_with_equal_probabilities()` - uniform distribution
- [ ] Add `test_risk_measures_with_single_scenario()` - degenerate case
- [ ] Add `test_risk_measure_numerical_stability()` - many scenarios, close values

### Testing

- [ ] Tests cover all three risk measure variants
- [ ] Tests use realistic cost distributions (e.g., from SDDP results)
- [ ] Property tests verify mathematical inequalities hold
- [ ] Edge case tests don't panic or produce NaN
- [ ] Tests complete quickly (<5ms each)

### Documentation

- [ ] Add doc comments explaining each risk measure mathematically
- [ ] Document the α parameter's meaning for CVaR
- [ ] Add examples showing typical usage of each risk measure
- [ ] Reference academic literature for CVaR formulation if complex

**Technical Notes**

- CVaR_α bounds expectation: E[X | X >= VaR_α(X)] >= E[X]
- For minimization: CVaR is worse (higher) than expectation
- Probability adjustment for CVaR: tail scenarios scaled by (1/α) * p_i
- α=0.5 gives median-based risk measure
- α→0 approaches expectation, α→1 approaches worst-case
- Numerical stability: avoid division by very small α values
- Test with costs like [10, 20, 30, 40, 50] and uniform probabilities
- WorstCase is max(costs) regardless of probabilities

**Dependencies**

- Blocked by: None (pure math, no fixture dependencies)
- Blocks: TEST-026 (risk measure integration tests)
- Related: TEST-003 (cut tests also deal with expected values)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-005: Add Critical Unit Tests for system.rs

**Context**

The `system.rs` module has only 1 test despite being foundational. We need tests for system validation (ID uniqueness, reference validity, cascade topology), dimension queries, and edge cases like empty systems.

**Acceptance Criteria**

- [ ] At least 15 new unit tests added to `src/system.rs`
- [ ] System validation rules enforced: unique IDs, valid references, acyclic cascades
- [ ] Dimension query methods tested: `n_hydros()`, `n_buses()`, etc.
- [ ] Edge cases covered: empty system, duplicate IDs, invalid references
- [ ] All tests pass and catch actual validation errors
- [ ] Test coverage for system.rs increases to >70%

**Tasks**

### Implementation

- [ ] Add `test_system_validation_unique_ids()` - reject duplicate hydro/bus IDs
- [ ] Add `test_system_validation_valid_references()` - all bus_ids exist
- [ ] Add `test_hydro_cascade_topology_acyclic()` - no cycles in downstream IDs
- [ ] Add `test_bus_hydro_connection_validity()` - hydro.bus_id references valid bus
- [ ] Add `test_line_bus_connection_validity()` - line.from_bus/to_bus exist
- [ ] Add `test_system_dimension_queries()` - n_hydros(), n_buses() correct
- [ ] Add `test_empty_system_creation()` - system with no components is valid
- [ ] Add `test_system_with_single_hydro()` - minimal valid system
- [ ] Add `test_cascade_upstream_downstream_consistency()` - bidirectional references match
- [ ] Add `test_bus_component_id_vectors()` - hydro_ids, line_ids populated correctly
- [ ] Add `test_system_clone_and_equality()` - proper derive implementations
- [ ] Add `test_system_from_json_roundtrip()` - serialize/deserialize preserves structure
- [ ] Add `test_thermal_plant_validation()` - if thermals exist
- [ ] Add `test_transmission_line_validation()` - capacity limits non-negative
- [ ] Add `test_system_pretty_print()` - debug representation useful

### Testing

- [ ] Each test exercises one validation rule or edge case
- [ ] Negative tests verify errors are produced for invalid systems
- [ ] Tests use builder pattern for readability
- [ ] Tests don't depend on external files
- [ ] All tests complete in <10ms

### Documentation

- [ ] Document all validation rules in module-level comments
- [ ] Add examples of valid and invalid system configurations
- [ ] Document the cascade topology requirements (DAG structure)
- [ ] Update README if system validation is user-facing

**Technical Notes**

- Use `System::validate()` method if it exists, or create it
- ID uniqueness: no two hydros/buses should share an ID
- Reference validity: bus_id in Hydro must exist in System.buses
- Cascade topology: directed acyclic graph (topological sort should succeed)
- Empty system: valid for testing but may be rejected by SDDP builder
- Consider using `anyhow::Context` for detailed error messages
- Validation should happen at construction time, not later
- Test both construction-time validation and explicit `validate()` call

**Dependencies**

- Blocked by: TEST-001 (needs fixture updates for reference)
- Blocks: Many integration tests depend on valid systems
- Related: TEST-006 (SDDP algorithm tests use systems)

**Estimated Effort**

3 story points (2-3 days, confidence: medium - depends on existing validation code)

---

### TEST-006: Add Critical Algorithm Correctness Tests to sddp/mod.rs

**Context**

The main SDDP algorithm module has 34 tests but lacks critical tests for infeasibility handling, scenario branching, risk measure integration, and parallel forward passes. These are essential for proving algorithm correctness.

**Acceptance Criteria**

- [ ] At least 10 new unit tests added to `src/sddp/mod.rs` or test files
- [ ] Infeasibility handling tested (graceful error or recovery)
- [ ] Backward pass scenario branching validated (K scenarios sampled)
- [ ] Risk measure probability adjustment tested in cut aggregation
- [ ] Parallel forward passes produce consistent results
- [ ] Convergence detection tested (stops at gap threshold)
- [ ] All tests pass reliably in parallel execution

**Tasks**

### Implementation

- [ ] Add `test_forward_pass_with_infeasible_subproblem()` - error handling
- [ ] Add `test_backward_pass_branching_scenarios()` - K scenarios sampled at each node
- [ ] Add `test_risk_measure_probability_adjustment()` - CVaR adjusts probabilities in cuts
- [ ] Add `test_cut_aggregation_with_risk()` - risk-adjusted cuts match formula
- [ ] Add `test_multistage_value_function_consistency()` - θ_t bounds V_{t+1}
- [ ] Add `test_parallel_forward_passes()` - multiple passes produce valid results
- [ ] Add `test_convergence_detection()` - algorithm stops when gap < tolerance
- [ ] Add `test_lower_bound_computation()` - LB from first-stage objective + cuts
- [ ] Add `test_upper_bound_from_simulation()` - UB from forward pass costs
- [ ] Add `test_iteration_result_tracking()` - all iterations recorded correctly

### Testing

- [ ] Tests use realistic SDDP instances from fixtures
- [ ] Tests complete in reasonable time (<1s each)
- [ ] Parallel execution tests use deterministic seeds
- [ ] Infeasibility test verifies graceful error messages
- [ ] Risk measure tests compare expected vs actual cut coefficients

### Documentation

- [ ] Document SDDP algorithm flow in module comments
- [ ] Explain forward/backward pass interaction
- [ ] Document convergence criteria (gap, max iterations)
- [ ] Add references to SDDP literature for algorithms used

**Technical Notes**

- Infeasibility: should return `SDDPError::InfeasibleSubproblem` with details
- Backward pass branching: sample K scenarios at each trajectory point
- Risk-adjusted cut: E_ρ[∂V/∂x] where ρ adjusts probabilities
- Parallel forward passes: use different RNG streams per thread
- Lower bound: objective of root node LP after adding cuts
- Upper bound: mean cost of simulation forward passes
- Convergence: typically |UB - LB| < ε or iterations >= max_iterations
- Value function consistency: θ_t should lower-bound next stage value
- Use `rayon` for parallel forward passes if not already

**Dependencies**

- Blocked by: TEST-001 (needs fixtures), TEST-002 (uses utilities)
- Blocks: TEST-019 (algorithm integration tests build on these)
- Related: TEST-003 (cut tests), TEST-004 (risk tests)

**Estimated Effort**

3 story points (2-3 days, confidence: medium - some algorithmic complexity)

---

## Phase 2: Mathematical Validation (Week 3)

### TEST-013: Implement Cut Validity Mathematical Properties

**Context**

Mathematical property tests verify SDDP invariants independent of specific problems. The most fundamental property is that a Benders cut evaluated at its training state should equal the objective value. We need property-based tests that run across all benchmark problems.

**Acceptance Criteria**

- [ ] Property test: cut height at training state equals objective (tolerance 1e-4)
- [ ] Property test: storage coefficients are non-positive (∂V/∂storage <= 0)
- [ ] Property test: dual feasibility via complementary slackness
- [ ] Tests run on all benchmark fixtures (deterministic, stochastic, PAR)
- [ ] Clear failure messages identify which property failed and on which problem
- [ ] All property tests pass consistently

**Tasks**

### Implementation

- [ ] Create `tests/mathematical/cut_properties.rs` module
- [ ] Implement `property_cut_equals_objective_at_training_state()` test
- [ ] Implement `property_storage_coefficients_negative()` test
- [ ] Implement `property_dual_complementary_slackness()` test
- [ ] Create `all_benchmarks()` helper returning all test problems
- [ ] Add property validation to each benchmark run
- [ ] Use test utilities from TEST-002 for assertions

### Testing

- [ ] Test passes on deterministic 2-stage problem
- [ ] Test passes on stochastic 3-stage problem
- [ ] Test passes on PAR(2) system
- [ ] Test catches violations if cut generation is broken
- [ ] Test completes in <5s across all benchmarks

### Documentation

- [ ] Document mathematical formulation of each property
- [ ] Explain why property must hold (theoretical basis)
- [ ] Add references to SDDP theory papers
- [ ] Include examples of what violations would indicate

**Technical Notes**

- Cut validity: height = rhs - coefficients' * state should equal objective
- Use epsilon = 1e-4 to account for LP solver tolerances
- Storage coefficients: negative because water has positive value
- Dual complementary slackness: dual * slack ≈ 0 for all constraints
- Extract training state and objective from cut metadata if available
- May need to enhance Cut struct to store training state/objective
- Property tests should be independent (each can fail separately)

**Dependencies**

- Blocked by: TEST-001, TEST-002, TEST-003
- Blocks: None (validates existing functionality)
- Related: TEST-014 (convergence properties)

**Estimated Effort**

2 story points (1-2 days, confidence: medium)

---

### TEST-014: Implement Convergence Mathematical Properties

**Context**

SDDP's fundamental theoretical guarantee is monotonically non-decreasing lower bounds. We need tests that verify convergence properties: monotonic LB, feasible UB, and gap closure over iterations.

**Acceptance Criteria**

- [ ] Property test: lower bound never decreases across iterations
- [ ] Property test: upper bound comes from feasible solutions
- [ ] Property test: gap decreases on average for stochastic problems
- [ ] Tests allow small tolerance (1e-6) for LP solver numerical noise
- [ ] Tests track and report convergence metrics
- [ ] All convergence tests pass on benchmark problems

**Tasks**

### Implementation

- [ ] Create `tests/mathematical/convergence_properties.rs` module
- [ ] Implement `property_monotonic_lower_bound()` test
- [ ] Implement `property_upper_bound_feasible()` test
- [ ] Implement `property_gap_decreases_on_average()` test
- [ ] Add helper to extract iteration history from training results
- [ ] Use `assert_monotonic_non_decreasing()` from utilities
- [ ] Add statistical test for gap closure (compare early vs late iterations)

### Testing

- [ ] Test monotonic LB on deterministic problem (should be exact)
- [ ] Test monotonic LB on stochastic problem (small tolerance needed)
- [ ] Test gap closure on 100-iteration run
- [ ] Test catches violations if cut generation is buggy
- [ ] Test handles edge cases (1 iteration, no gap improvement)

### Documentation

- [ ] Explain SDDP convergence theory in module comments
- [ ] Document why monotonic LB is guaranteed
- [ ] Explain sampling error in stochastic problems
- [ ] Reference convergence rate literature

**Technical Notes**

- Monotonic LB: LB[i] >= LB[i-1] - tolerance for i in 1..iterations
- Tolerance 1e-6 accounts for LP solver basis changes
- Upper bound feasibility: verify no constraint violations in simulation
- Gap closure: compare average gap in first 20% vs last 20% of iterations
- Deterministic problems should converge exactly (gap → 0)
- Stochastic problems have sampling error (gap may not → 0)
- Track: initial gap, final gap, iterations to 5% gap
- Consider plotting convergence curves for visualization

**Dependencies**

- Blocked by: TEST-001, TEST-002, TEST-006
- Blocks: None
- Related: TEST-013 (cut properties), TEST-019 (integration convergence)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-015: Implement AR Model Chain Rule Properties

**Context**

For AR models, cut coefficients for lagged states must follow the chain rule: ∂V/∂Y_{t-j} = (λ^hydro + λ^AR) * ψ_j. This is critical for correct temporal coupling in PAR models. We need tests that validate this mathematical relationship.

**Acceptance Criteria**

- [ ] Property test: AR lag coefficients match chain rule formula
- [ ] Property test: PAR to AR transformation (ψ = φ * σ_t/σ_{t-j}) is correct
- [ ] Tests cover heterogeneous AR orders (AR(0), AR(1), AR(2) mixed)
- [ ] Tests validate coefficient computation for all lag indices
- [ ] Tolerance appropriate for numerical precision (1e-5)
- [ ] All AR property tests pass

**Tasks**

### Implementation

- [ ] Create `tests/mathematical/ar_properties.rs` module
- [ ] Implement `property_ar_lag_coefficient_chain_rule()` test
- [ ] Implement `property_par_psi_transformation()` test
- [ ] Implement `property_deterministic_noise_base()` test
- [ ] Create fixtures with AR(1) and AR(2) models
- [ ] Extract duals from LP solution for validation
- [ ] Compute expected coefficients and compare to actual

### Testing

- [ ] Test on single-reservoir AR(1) system
- [ ] Test on multi-reservoir heterogeneous AR orders
- [ ] Test seasonal PAR transformation with multiple periods
- [ ] Test catches errors in ψ computation
- [ ] Test handles AR(0) case (no lags)

### Documentation

- [ ] Document chain rule derivation in comments
- [ ] Explain PAR seasonal adjustment formula
- [ ] Reference PAR model literature
- [ ] Add diagram showing temporal coupling if helpful

**Technical Notes**

- Chain rule: cut lag coef = (water_dual + ar_dual) * psi[lag_idx]
- ψ_i = φ_i * (σ_t / σ_{t-i}) for PAR models
- Need access to LP duals for hydro balance and AR constraints
- May need to enhance Subproblem to expose AR constraint duals
- Heterogeneous: each hydro can have different AR order
- Zero-order (AR(0)): no lag terms, only current state
- Deterministic base: μ_t - Σ(φ_i * μ_{t-i})
- Precision: use epsilon = 1e-5 for cut coefficients

**Dependencies**

- Blocked by: TEST-001 (needs AR fixtures)
- Blocks: TEST-025 (AR integration tests)
- Related: TEST-007 (temporal model unit tests)

**Estimated Effort**

3 story points (2-3 days, confidence: medium - requires AR understanding)

---

### TEST-016: Implement Risk Measure Mathematical Properties

**Context**

Risk measures must satisfy mathematical properties: CVaR bounds expectation, monotonicity (worse scenarios → higher risk), and consistency with theoretical definitions. Property tests ensure these hold across different cost distributions.

**Acceptance Criteria**

- [ ] Property test: CVaR_α(X) >= E[X] for all α in (0,1) for cost minimization
- [ ] Property test: Risk measure monotonicity (scenario A > B → ρ(A) >= ρ(B))
- [ ] Property test: Expectation is probability-weighted average
- [ ] Property test: WorstCase is maximum regardless of probabilities
- [ ] Tests use diverse cost distributions (uniform, skewed, bimodal)
- [ ] All risk property tests pass

**Tasks**

### Implementation

- [ ] Create `tests/mathematical/risk_properties.rs` module
- [ ] Implement `property_cvar_bounds_expectation()` test
- [ ] Implement `property_risk_measure_monotonicity()` test
- [ ] Implement `property_expectation_weighted_average()` test
- [ ] Implement `property_worst_case_is_maximum()` test
- [ ] Generate diverse cost distributions for testing
- [ ] Test all α values: 0.1, 0.25, 0.5, 0.75, 0.9

### Testing

- [ ] Test with uniform cost distribution
- [ ] Test with skewed distribution (long tail)
- [ ] Test with equal costs (degenerate case)
- [ ] Test with single scenario (edge case)
- [ ] Test catches violations of mathematical properties

### Documentation

- [ ] Document CVaR definition and properties
- [ ] Explain monotonicity requirement for coherent risk measures
- [ ] Reference risk measure theory literature
- [ ] Provide intuition for why properties must hold

**Technical Notes**

- CVaR definition: E[X | X >= VaR_α(X)] for costs
- For minimization: CVaR >= Expectation (more conservative)
- Monotonicity: if cost_a[i] >= cost_b[i] for all i, then ρ(A) >= ρ(B)
- Expectation: Σ(p_i * cost_i)
- WorstCase: max(costs), ignore probabilities
- Test with costs like [10, 20, 30, 40, 50] and various probability distributions
- Numerical precision: floating-point equality needs epsilon
- CVaR for α→0 approaches expectation, α→1 approaches worst-case

**Dependencies**

- Blocked by: TEST-004 (risk measure unit tests)
- Blocks: TEST-026 (risk integration tests)
- Related: TEST-014 (convergence with risk)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-017: Implement Numerical Stability Tests

**Context**

Numerical precision is critical for LP-based algorithms. We need tests for Kahan summation accuracy, cut evaluation with extreme values, and LP solver tolerance handling. These catch precision issues before they cause convergence problems.

**Acceptance Criteria**

- [ ] Test: Kahan summation vs naive summation with cancellation
- [ ] Test: Cut evaluation with extreme values (1e8, 1e-8)
- [ ] Test: LP solver consistency (same problem, same result)
- [ ] Test: Floating-point precision in cut aggregation
- [ ] Tests demonstrate precision improvements from stable algorithms
- [ ] All numerical stability tests pass

**Tasks**

### Implementation

- [ ] Create `tests/mathematical/numerical_stability.rs` module
- [ ] Implement `test_kahan_summation_vs_naive()` with cancellation case
- [ ] Implement `test_cut_evaluation_extreme_values()` with large states
- [ ] Implement `test_lp_solver_consistency()` with repeated solves
- [ ] Implement `test_floating_point_cut_aggregation()` with many cuts
- [ ] Create examples demonstrating catastrophic cancellation
- [ ] Verify Kahan sum is actually used in critical code paths

### Testing

- [ ] Classic numerical test: [1e10, 1.0, -1e10, 1.0] sums to 2.0
- [ ] Cut evaluation: 100-dim state with 1e8 values, 1e-8 coefficients
- [ ] LP solve: same problem twice, objectives match to 1e-6
- [ ] Aggregation: sum 1000 cuts with similar values
- [ ] Tests catch precision loss if Kahan sum disabled

### Documentation

- [ ] Explain why Kahan summation is necessary
- [ ] Document numerical precision requirements
- [ ] Reference numerical analysis literature
- [ ] Provide examples of precision loss without stable algorithms

**Technical Notes**

- Kahan summation prevents catastrophic cancellation in summations
- Critical for dot products with many terms (cut evaluation, aggregation)
- Extreme value test: state [1e8; 100], coef [-1e-8; 100] should compute accurately
- LP solver tolerance: typically 1e-6 for primal/dual feasibility
- Cancellation occurs when subtracting nearly-equal large numbers
- Use reference implementations from numerical libraries
- Consider `f64::mul_add()` for fused multiply-add precision
- Test both `--debug` and `--release` builds (optimization affects precision)

**Dependencies**

- Blocked by: TEST-003 (cut tests)
- Blocks: None
- Related: TEST-013 (cut properties require precision)

**Estimated Effort**

2 story points (1-2 days, confidence: medium)

---

### TEST-018: Create Mathematical Test Summary Report

**Context**

After implementing mathematical property tests, we need a summary showing which properties are validated and their pass rates across benchmarks. This provides confidence in algorithm correctness and identifies any issues.

**Acceptance Criteria**

- [ ] Summary report generated showing all mathematical properties tested
- [ ] Report shows pass/fail status for each property on each benchmark
- [ ] Report includes performance metrics (test execution time)
- [ ] Report integrated into test suite output
- [ ] Documentation updated with property validation status

**Tasks**

### Implementation

- [ ] Create test helper to collect property test results
- [ ] Generate summary table: properties × benchmarks
- [ ] Add execution time tracking for each property test
- [ ] Create formatted report (markdown or terminal output)
- [ ] Add `--report` flag to test runner if needed

### Testing

- [ ] Report generation works with all properties passing
- [ ] Report highlights failures clearly
- [ ] Report format is readable in CI output
- [ ] Report can be saved to file for documentation

### Documentation

- [ ] Add report to TESTING_STRATEGY.md showing validation complete
- [ ] Document how to run property tests and generate report
- [ ] Update README with "mathematically validated" badge/claim
- [ ] Include sample report in documentation

**Technical Notes**

- Use test framework hooks to collect results
- Consider custom test runner or post-processing test output
- Track: property name, benchmark name, pass/fail, duration
- Format as markdown table for easy inclusion in docs
- Color code pass/fail in terminal output
- Generate report in CI and archive as artifact
- Include git commit hash in report for traceability

**Dependencies**

- Blocked by: TEST-013 through TEST-017
- Blocks: None
- Related: All mathematical validation tests

**Estimated Effort**

1 story point (1 day, confidence: high)

---

## Phase 3: Integration Tests (Week 4-5)

### TEST-019: Create Algorithm Integration Test Structure

**Context**

Integration tests for the SDDP algorithm need organized structure. Create the directory hierarchy and base fixtures for testing forward/backward passes, training loops, and convergence. This establishes the foundation for all algorithm integration tests.

**Acceptance Criteria**

- [ ] Directory structure created: `tests/integration/algorithm/`
- [ ] Base fixture module for algorithm tests created
- [ ] At least 3 shared test fixtures available
- [ ] Documentation explaining integration test organization
- [ ] Template test file demonstrating structure
- [ ] All structure compiles and dummy tests pass

**Tasks**

### Implementation

- [ ] Create `tests/integration/` directory
- [ ] Create `tests/integration/algorithm/` subdirectory
- [ ] Create `tests/integration/algorithm/mod.rs` with re-exports
- [ ] Create `tests/integration/fixtures.rs` with common setups
- [ ] Implement `setup_simple_algorithm()` fixture
- [ ] Implement `setup_stochastic_algorithm()` fixture
- [ ] Implement `setup_par_algorithm()` fixture
- [ ] Create example test demonstrating fixture usage

### Testing

- [ ] Each fixture can be instantiated
- [ ] Fixtures produce valid SDDP instances
- [ ] Example test runs successfully
- [ ] Fixtures can be used in parallel tests

### Documentation

- [ ] Document integration test organization in README
- [ ] Add comments explaining each fixture's purpose
- [ ] Create integration test writing guide
- [ ] Update TESTING_STRATEGY.md with structure

**Technical Notes**

- Follow same pattern as existing `tests/fixtures/` but for integration
- Fixtures should be more complex than unit test fixtures
- Include multi-stage, stochastic scenarios
- Keep fixtures deterministic (fixed seeds)
- Consider using builder pattern for flexibility
- Each subdirectory (`algorithm/`, `ar_models/`, etc.) gets its own `mod.rs`

**Dependencies**

- Blocked by: TEST-001 (fixture fixes)
- Blocks: TEST-020 through TEST-024 (algorithm tests)
- Related: All integration test phases

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-020: Implement Forward Pass Integration Tests

**Context**

Forward pass tests verify that state initialization, uncertainty realization, LP solving, and state extraction work together correctly. These are critical for ensuring the simulation component of SDDP is correct.

**Acceptance Criteria**

- [ ] At least 4 integration tests for forward pass created
- [ ] Test: State initialization from initial_condition works correctly
- [ ] Test: Uncertainty realization applied to subproblems
- [ ] Test: Cost accumulation across stages is correct
- [ ] Test: Trajectory storage captures all necessary information
- [ ] All tests pass consistently

**Tasks**

### Implementation

- [ ] Create `tests/integration/algorithm/forward_pass.rs`
- [ ] Implement `test_forward_pass_state_initialization()`
- [ ] Implement `test_forward_pass_uncertainty_realization()`
- [ ] Implement `test_forward_pass_cost_accumulation()`
- [ ] Implement `test_forward_pass_trajectory_storage()`
- [ ] Implement `test_forward_pass_state_continuity()` - state propagates correctly
- [ ] Use assertion utilities for state validation

### Testing

- [ ] Test with deterministic 2-stage problem
- [ ] Test with stochastic 3-stage problem
- [ ] Verify costs sum correctly across stages
- [ ] Verify state continuity: end state of t = start state of t+1
- [ ] Tests run in <500ms each

### Documentation

- [ ] Document forward pass algorithm flow
- [ ] Explain what trajectory stores and why
- [ ] Add diagrams showing state propagation
- [ ] Reference SDDP algorithm description

**Technical Notes**

- Forward pass: sample scenario, solve stages sequentially, record trajectory
- State continuity: storage_t+1 = storage_t + inflow - generation - spillage
- Trajectory must store: states, decisions, costs, duals for backward pass
- Uncertainty realization: scenario inflows/loads applied as constraint RHS
- Cost accumulation: sum stage costs plus terminal value
- Use `assert_vector_eq` for state comparisons with appropriate epsilon

**Dependencies**

- Blocked by: TEST-019 (integration structure)
- Blocks: TEST-022 (training loop needs forward pass)
- Related: TEST-021 (backward pass)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

I'll continue with more tickets in the next message, as this response is getting long. Would you like me to continue creating tickets for:
- TEST-021: Backward Pass Integration Tests
- TEST-022: Training Loop Integration Tests
- TEST-023: Convergence Integration Tests
- TEST-024: Algorithm Integration Summary
- Phase 4 tickets (E2E, Performance, CI/CD)
- Phase 5 tickets (Documentation & Polish)?


### TEST-021: Implement Backward Pass Integration Tests

**Context**

Backward pass is the core of SDDP's learning mechanism. It samples scenarios at each trajectory point, generates Benders cuts from LP duals, and adds them to future cost functions. Integration tests must verify the complete flow including scenario branching, cut generation, and FCF updates.

**Acceptance Criteria**

- [ ] At least 4 integration tests for backward pass created
- [ ] Test: Scenario branching samples K scenarios at each trajectory point
- [ ] Test: Cut generation produces valid cuts from LP duals
- [ ] Test: Dual extraction matches expected values for known problems
- [ ] Test: FCF (Future Cost Function) receives and stores generated cuts
- [ ] All tests pass consistently and independently

**Tasks**

### Implementation

- [ ] Create `tests/integration/algorithm/backward_pass.rs`
- [ ] Implement `test_backward_pass_scenario_branching()` - verify K scenarios sampled
- [ ] Implement `test_backward_pass_cut_generation()` - one cut per realization
- [ ] Implement `test_backward_pass_dual_extraction()` - duals become cut coefficients
- [ ] Implement `test_backward_pass_fcf_update()` - cuts added to correct nodes
- [ ] Implement `test_backward_pass_cut_count()` - verify expected number of cuts generated
- [ ] Use fixture trajectory from forward pass test

### Testing

- [ ] Test with 2-stage problem (simple case)
- [ ] Test with 3-stage problem (multiple backward stages)
- [ ] Test with different K values (1, 5, 10 scenarios)
- [ ] Verify cut coefficients match LP duals
- [ ] Verify FCF cut count increases after backward pass
- [ ] Tests run in <1s each

### Documentation

- [ ] Document backward pass algorithm flow
- [ ] Explain scenario tree sampling strategy
- [ ] Document dual-to-cut transformation
- [ ] Add diagram showing backward pass through stages

**Technical Notes**

- Backward pass: traverse trajectory backwards, sample K scenarios at each point
- At each node: solve subproblems, extract duals, generate cuts
- Duals: storage balance → cut coefficient for storage state
- Duals: AR constraints → cut coefficients for lagged states
- Cut RHS: E[future_value] under sampled scenarios
- Scenario sampling: use deterministic seed for reproducibility
- Cut should satisfy: height(training_state) = objective
- FCF update: add cut to node's future cost function

**Dependencies**

- Blocked by: TEST-019 (integration structure), TEST-020 (forward pass)
- Blocks: TEST-022 (training loop needs both passes)
- Related: TEST-003 (cut unit tests), TEST-013 (cut properties)

**Estimated Effort**

3 story points (2-3 days, confidence: high)

---

### TEST-022: Implement Training Loop Integration Tests

**Context**

The training loop orchestrates forward passes, backward passes, bound computation, and convergence checking. Integration tests verify the complete iteration cycle and that bounds are computed correctly from the passes.

**Acceptance Criteria**

- [ ] At least 3 integration tests for training loop created
- [ ] Test: Each iteration executes forward passes → backward pass → bounds
- [ ] Test: Lower bound computed from first-stage objective + cuts
- [ ] Test: Upper bound computed from forward pass costs
- [ ] Test: Iteration results recorded with all required metrics
- [ ] Test: Convergence detection works (gap threshold, max iterations)
- [ ] All tests pass reliably

**Tasks**

### Implementation

- [ ] Create `tests/integration/algorithm/training_loop.rs`
- [ ] Implement `test_training_iteration_structure()` - verify pass sequence
- [ ] Implement `test_lower_bound_computation()` - LB from root LP objective
- [ ] Implement `test_upper_bound_computation()` - UB from simulation
- [ ] Implement `test_iteration_result_tracking()` - all metrics recorded
- [ ] Implement `test_convergence_detection_gap()` - stops when gap < threshold
- [ ] Implement `test_convergence_detection_max_iter()` - stops at max iterations

### Testing

- [ ] Test with deterministic problem (converges exactly)
- [ ] Test with stochastic problem (sampling error in UB)
- [ ] Test early stopping on convergence
- [ ] Test hitting max iterations without convergence
- [ ] Verify all iteration data is accessible
- [ ] Tests complete in reasonable time (<5s each)

### Documentation

- [ ] Document training loop structure
- [ ] Explain bound computation formulas
- [ ] Document convergence criteria
- [ ] Add flowchart of training iteration

**Technical Notes**

- Training iteration: M forward passes → 1 backward pass → compute bounds
- Lower bound: solve root node LP with current cuts, get objective
- Upper bound: mean cost of M forward pass simulations
- Gap: |UB - LB| or relative: |UB - LB| / |LB|
- Convergence: gap < tolerance OR iterations >= max_iterations
- Store per-iteration: LB, UB, gap, time, cut count
- Forward passes use different scenarios (or same with different seeds)
- Backward pass samples K scenarios at each trajectory point
- Root LP solve: includes all cuts added so far

**Dependencies**

- Blocked by: TEST-020 (forward pass), TEST-021 (backward pass)
- Blocks: TEST-023 (convergence tests)
- Related: TEST-006 (algorithm unit tests)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-023: Implement Convergence Integration Tests

**Context**

Convergence tests verify that the algorithm exhibits expected behavior over multiple iterations: monotonic lower bounds, improving upper bounds, and gap closure. These are higher-level tests that validate the algorithm's mathematical properties in practice.

**Acceptance Criteria**

- [ ] At least 3 integration tests for convergence created
- [ ] Test: Lower bound monotonically increases (with tolerance)
- [ ] Test: Upper bound improves over iterations (stochastic case)
- [ ] Test: Gap closes for deterministic problems
- [ ] Test: Algorithm terminates on convergence or max iterations
- [ ] Tests use realistic problem instances
- [ ] All tests pass consistently

**Tasks**

### Implementation

- [ ] Create `tests/integration/algorithm/convergence.rs`
- [ ] Implement `test_monotonic_lower_bound_integration()` - LB never decreases
- [ ] Implement `test_upper_bound_improvement()` - UB generally improves
- [ ] Implement `test_gap_closure_deterministic()` - gap → 0 for deterministic
- [ ] Implement `test_convergence_rate()` - iterations to X% gap
- [ ] Use `assert_monotonic_non_decreasing()` utility
- [ ] Track convergence metrics across iterations

### Testing

- [ ] Test on deterministic 2-stage problem (exact convergence)
- [ ] Test on stochastic 3-stage problem (approximate convergence)
- [ ] Run for 50+ iterations to observe trends
- [ ] Verify monotonicity with appropriate tolerance (1e-6)
- [ ] Check convergence rate matches expected bounds
- [ ] Tests may take 5-10s (longer problems)

### Documentation

- [ ] Explain expected convergence behavior
- [ ] Document typical convergence rates for problem classes
- [ ] Reference SDDP convergence theory
- [ ] Include example convergence plots (if visualizing)

**Technical Notes**

- Monotonic LB is a fundamental SDDP guarantee
- Use tolerance 1e-6 for LP solver numerical noise
- Deterministic problems: gap should → 0 (no sampling error)
- Stochastic problems: gap has sampling error floor
- Upper bound: may fluctuate due to sampling, but trend improves
- Gap closure: compare first 20% vs last 20% of iterations
- Convergence rate: problem-dependent, track for regression
- Statistical tests for UB improvement (t-test on first vs last)
- Consider visualizing convergence curves for documentation

**Dependencies**

- Blocked by: TEST-022 (training loop)
- Blocks: None
- Related: TEST-014 (convergence properties), TEST-032 (E2E convergence)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-024: Create Algorithm Integration Test Summary

**Context**

After implementing algorithm integration tests, create a summary showing test coverage, pass rates, and any gaps. This provides visibility into what's been validated and what remains.

**Acceptance Criteria**

- [ ] Summary document or report generated
- [ ] Shows all algorithm integration tests and their status
- [ ] Identifies any gaps in coverage
- [ ] Includes execution time metrics
- [ ] Integrated into test documentation

**Tasks**

### Implementation

- [ ] Create test summary script or document
- [ ] List all algorithm integration tests
- [ ] Categorize by component (forward, backward, training, convergence)
- [ ] Add coverage metrics (lines, branches)
- [ ] Generate execution time report

### Testing

- [ ] Summary accurately reflects test suite state
- [ ] Summary updates automatically with new tests
- [ ] Report generation doesn't break CI

### Documentation

- [ ] Add summary to TESTING_STRATEGY.md
- [ ] Document how to generate summary
- [ ] Highlight any coverage gaps
- [ ] Update README with testing status

**Technical Notes**

- Use `cargo tarpaulin` or similar for coverage metrics
- Parse test output to collect execution times
- Generate markdown table for documentation
- Consider badges for test pass rate
- Track metrics over time for trends

**Dependencies**

- Blocked by: TEST-020, TEST-021, TEST-022, TEST-023
- Blocks: None
- Related: TEST-018 (mathematical test summary)

**Estimated Effort**

1 story point (1 day, confidence: high)

---

## Phase 4: Feature Integration Tests (Week 5)

### TEST-025: Implement AR Model Integration Tests

**Context**

AR (AutoRegressive) models add temporal correlation to inflows and loads. Integration tests must verify that PAR models are correctly integrated into cut generation, lag constraints are properly updated, and heterogeneous AR orders work across multiple hydros.

**Acceptance Criteria**

- [ ] At least 10 integration tests for AR models created
- [ ] Test: PAR cut generation includes lag coefficients
- [ ] Test: Lag constraint updates in subproblem
- [ ] Test: Heterogeneous AR orders (mixed AR(0), AR(1), AR(2))
- [ ] Test: Mixed entities (some hydros AR, some deterministic)
- [ ] Test: Seasonal adjustment in PAR models
- [ ] All AR integration tests pass

**Tasks**

### Implementation

- [ ] Create `tests/integration/ar_models/` directory
- [ ] Create `tests/integration/ar_models/par_cut_generation.rs`
- [ ] Implement `test_par_cut_with_storage_only()` - baseline (no AR)
- [ ] Implement `test_par_cut_with_ar1_inflows()` - single lag
- [ ] Implement `test_par_cut_with_ar2_inflows()` - two lags
- [ ] Implement `test_par_cut_chain_rule_validation()` - coefficient formula
- [ ] Implement `test_par_cut_seasonal_adjustment()` - ψ transformation
- [ ] Create `tests/integration/ar_models/heterogeneous_orders.rs`
- [ ] Implement `test_mixed_ar_orders()` - different orders per hydro
- [ ] Implement `test_state_dimension_consistency()` - state size matches
- [ ] Implement `test_heterogeneous_lag_dual_extraction()` - correct duals
- [ ] Create `tests/integration/ar_models/mixed_entities.rs`
- [ ] Implement `test_loads_ar_inflows_deterministic()` - loads AR, inflows not
- [ ] Implement `test_loads_deterministic_inflows_ar()` - opposite case
- [ ] Implement `test_both_loads_and_inflows_ar()` - full complexity

### Testing

- [ ] Test on single reservoir with AR(1)
- [ ] Test on single reservoir with AR(2)
- [ ] Test on 3 reservoirs with heterogeneous orders [0, 1, 2]
- [ ] Verify cut coefficients match chain rule formula
- [ ] Verify state dimension = n_hydros + sum(ar_orders)
- [ ] Test with seasonal PAR (multiple periods)
- [ ] Tests complete in <2s each

### Documentation

- [ ] Document AR model integration architecture
- [ ] Explain lag constraint formulation
- [ ] Document chain rule for cut coefficients
- [ ] Reference PAR model theory papers
- [ ] Add example with AR model configuration

**Technical Notes**

- PAR: Periodic AutoRegressive model with seasonal parameters
- Chain rule: ∂V/∂Y_{t-j} = (λ^hydro + λ^AR) * ψ_j
- ψ_j = φ_j * (σ_t / σ_{t-j}) for PAR seasonal adjustment
- Heterogeneous: each hydro can have different AR order
- State layout: [storage_1, ..., storage_n, lag_1,1, lag_1,2, ..., lag_n,p]
- Lag constraints: Y_t = μ + Σ(φ_i * Y_{t-i}) + ε_t
- Dual extraction: need duals from both hydro balance and AR constraints
- Numerical precision: use epsilon = 1e-5 for coefficient comparisons

**Dependencies**

- Blocked by: TEST-019 (integration structure), TEST-015 (AR properties)
- Blocks: TEST-032 (E2E PAR tests)
- Related: TEST-007 (temporal model unit tests if created)

**Estimated Effort**

5 story points (3-4 days, confidence: medium - AR complexity)

---

### TEST-026: Implement Risk Measure Integration Tests

**Context**

Risk measures (CVaR, Expectation, WorstCase) affect probability weights in cut generation. Integration tests verify that risk measures are correctly applied during backward pass and that cuts reflect risk-adjusted probabilities.

**Acceptance Criteria**

- [ ] At least 6 integration tests for risk measures created
- [ ] Test: CVaR cuts are more conservative than expectation cuts
- [ ] Test: Probability weights correctly adjusted for CVaR
- [ ] Test: Expectation gives uniform weights
- [ ] Test: WorstCase gives all weight to worst scenario
- [ ] Test: Risk measure affects convergence behavior
- [ ] All risk integration tests pass

**Tasks**

### Implementation

- [ ] Create `tests/integration/risk_measures/` directory
- [ ] Create `tests/integration/risk_measures/cvar_cuts.rs`
- [ ] Implement `test_cvar_vs_expectation_cuts()` - compare cut conservatism
- [ ] Implement `test_cvar_probability_weights()` - verify tail boost
- [ ] Implement `test_cvar_convergence()` - still monotonic
- [ ] Create `tests/integration/risk_measures/expectation_baseline.rs`
- [ ] Implement `test_expectation_uniform_weights()` - no adjustment
- [ ] Implement `test_expectation_convergence_baseline()` - standard SDDP
- [ ] Create `tests/integration/risk_measures/worst_case.rs`
- [ ] Implement `test_worst_case_single_scenario_weight()` - all weight on max
- [ ] Implement `test_worst_case_conservative_policy()` - very conservative

### Testing

- [ ] Compare cuts generated with different risk measures
- [ ] Verify CVaR cuts have steeper slopes (more conservative)
- [ ] Run convergence with each risk measure
- [ ] Check lower bounds: WorstCase >= CVaR >= Expectation
- [ ] Tests use same problem instance for comparison
- [ ] Tests complete in <3s each

### Documentation

- [ ] Document risk measure integration in SDDP
- [ ] Explain how risk affects cut generation
- [ ] Document typical use cases for each risk measure
- [ ] Reference risk-averse SDDP literature

**Technical Notes**

- Risk measure applied during cut aggregation in backward pass
- CVaR: tail scenarios (above α quantile) get boosted probability
- Expectation: default SDDP, no probability adjustment
- WorstCase: all weight on maximum cost scenario
- CVaR cuts: E_ρ[∂V/∂x] where ρ adjusts probabilities
- Lower bound ordering: WorstCase >= CVaR >= Expectation (for costs)
- All should maintain monotonic LB property
- α parameter typical values: 0.1, 0.25, 0.5
- Use same seed for fair comparison across risk measures

**Dependencies**

- Blocked by: TEST-019 (integration structure), TEST-016 (risk properties)
- Blocks: None
- Related: TEST-004 (risk unit tests)

**Estimated Effort**

3 story points (2-3 days, confidence: high)

---

### TEST-027: Implement Scenario Generation Integration Tests

**Context**

Scenario generation (SAA - Sample Average Approximation) creates discrete scenario trees from continuous distributions. Integration tests verify scenario sampling, correlation application, and branching structure match specifications.

**Acceptance Criteria**

- [ ] At least 6 integration tests for scenario generation created
- [ ] Test: SAA sampling produces correct number of scenarios
- [ ] Test: Scenario probabilities sum to 1.0
- [ ] Test: Correlation applied correctly to multi-site inflows
- [ ] Test: Branching structure matches specification
- [ ] Test: Scenarios are reproducible with fixed seed
- [ ] All scenario integration tests pass

**Tasks**

### Implementation

- [ ] Create `tests/integration/scenarios/` directory
- [ ] Create `tests/integration/scenarios/saa_sampling.rs`
- [ ] Implement `test_saa_scenario_count()` - correct number generated
- [ ] Implement `test_saa_probability_sum()` - probabilities sum to 1
- [ ] Implement `test_saa_reproducibility()` - fixed seed gives same scenarios
- [ ] Create `tests/integration/scenarios/correlation.rs`
- [ ] Implement `test_correlation_matrix_application()` - Cholesky decomposition
- [ ] Implement `test_correlated_multisite_inflows()` - spatial correlation
- [ ] Implement `test_correlation_preserves_marginals()` - mean/std maintained
- [ ] Create `tests/integration/scenarios/branching.rs`
- [ ] Implement `test_scenario_tree_structure()` - branching at specified stages
- [ ] Implement `test_scenario_node_probabilities()` - conditional probabilities

### Testing

- [ ] Test with uniform distribution (easy to validate)
- [ ] Test with normal distribution (typical case)
- [ ] Test with correlation matrix (2x2, 3x3)
- [ ] Verify Cholesky decomposition applied correctly
- [ ] Test scenario tree with branching at stage 2
- [ ] Tests complete in <1s each

### Documentation

- [ ] Document scenario generation algorithm
- [ ] Explain SAA method and convergence properties
- [ ] Document correlation matrix requirements
- [ ] Add example with correlated multi-site system

**Technical Notes**

- SAA: Sample Average Approximation with M scenarios
- Scenario probability: typically 1/M for uniform sampling
- Correlation: apply Cholesky decomposition to transform independent samples
- Branching: scenarios diverge at specified stages
- Reproducibility: use deterministic RNG seed
- Marginals: correlation preserves mean and std of each site
- Scenario tree: nodes represent decision points, branches are uncertainties
- Typical M values: 10 (testing), 100 (small problems), 1000+ (production)

**Dependencies**

- Blocked by: TEST-019 (integration structure)
- Blocks: None
- Related: Existing scenario generation unit tests

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-028: Implement System Feature Integration Tests

**Context**

Power system features like cascade hydros, transmission networks, and thermal plants require integration testing to verify component interactions. Tests ensure water balance in cascades, power flow constraints in transmission, and optimal thermal dispatch.

**Acceptance Criteria**

- [ ] At least 7 integration tests for system features created
- [ ] Test: Cascade water balance (upstream release → downstream inflow)
- [ ] Test: Cascade value propagation (downstream water more valuable)
- [ ] Test: Transmission capacity constraints enforced
- [ ] Test: Transmission losses (if modeled)
- [ ] Test: Thermal dispatch with cost curves
- [ ] All system integration tests pass

**Tasks**

### Implementation

- [ ] Create `tests/integration/system/` directory
- [ ] Create `tests/integration/system/cascade_hydros.rs`
- [ ] Implement `test_cascade_water_balance()` - release becomes inflow
- [ ] Implement `test_cascade_value_propagation()` - downstream more valuable
- [ ] Implement `test_cascade_delay()` - if travel time modeled
- [ ] Create `tests/integration/system/transmission.rs`
- [ ] Implement `test_transmission_capacity_constraints()` - flow <= capacity
- [ ] Implement `test_transmission_cost_in_objective()` - transmission cost included
- [ ] Implement `test_transmission_losses()` - if lossy lines modeled
- [ ] Create `tests/integration/system/thermal_dispatch.rs`
- [ ] Implement `test_thermal_minimum_generation()` - min <= gen <= max
- [ ] Implement `test_thermal_cost_curve()` - piecewise linear or quadratic
- [ ] Implement `test_thermal_vs_hydro_dispatch()` - optimal mix

### Testing

- [ ] Test 2-reservoir cascade with known solution
- [ ] Test 2-bus system with transmission line
- [ ] Test thermal plant with linear cost curve
- [ ] Verify optimal dispatch minimizes cost
- [ ] Verify physical constraints satisfied
- [ ] Tests complete in <2s each

### Documentation

- [ ] Document cascade hydro modeling
- [ ] Document transmission network formulation
- [ ] Document thermal plant modeling
- [ ] Add example with multi-reservoir cascade

**Technical Notes**

- Cascade: turbined_upstream + spilled_upstream = natural_inflow_downstream + inflow_downstream
- Travel time: delay between turbining and arrival (advanced feature)
- Transmission: power_flow <= line_capacity (may be bidirectional)
- Transmission losses: delivered power < sent power
- Thermal: cost typically piecewise linear in generation
- Thermal minimum: technical minimum generation constraint
- Optimal mix: hydro (free fuel) preferred over thermal (costly fuel)
- Terminal effect: downstream storage more valuable (affects all downstream stages)

**Dependencies**

- Blocked by: TEST-019 (integration structure), TEST-005 (system unit tests)
- Blocks: TEST-032 (E2E with complex systems)
- Related: None

**Estimated Effort**

3 story points (2-3 days, confidence: medium - depends on feature complexity)

---

### TEST-029: Implement Multi-Reservoir Integration Tests

**Context**

Multi-reservoir systems have complex interactions through cascades, shared buses, and competition for water. Integration tests verify that these interactions are correctly modeled and that the algorithm handles multi-reservoir complexity.

**Acceptance Criteria**

- [ ] At least 4 integration tests for multi-reservoir systems created
- [ ] Test: 3-reservoir cascade converges correctly
- [ ] Test: Parallel reservoirs on same bus
- [ ] Test: Mixed cascade and parallel configuration
- [ ] Test: Water values propagate correctly through system
- [ ] All multi-reservoir tests pass

**Tasks**

### Implementation

- [ ] Create `tests/integration/system/multi_reservoir.rs`
- [ ] Implement `test_three_reservoir_cascade()` - linear cascade
- [ ] Implement `test_parallel_reservoirs_same_bus()` - competition for demand
- [ ] Implement `test_mixed_cascade_parallel()` - complex topology
- [ ] Implement `test_water_value_propagation()` - duals through cascade
- [ ] Use realistic reservoir sizes and productivity factors

### Testing

- [ ] Test with 3-reservoir linear cascade
- [ ] Test with 2-reservoir parallel configuration
- [ ] Test with 5-reservoir complex topology
- [ ] Verify convergence in <100 iterations
- [ ] Verify physical constraints satisfied
- [ ] Tests may take 5-10s (larger problems)

### Documentation

- [ ] Document multi-reservoir modeling approach
- [ ] Add diagram of test system topologies
- [ ] Explain water value propagation theory
- [ ] Include example multi-reservoir system

**Technical Notes**

- Cascade: upstream decisions affect downstream state
- Parallel: reservoirs compete to serve same demand
- Water value: dual of storage constraint (marginal value of water)
- Downstream water more valuable due to terminal effect
- Topology affects cut coefficients and convergence
- Use realistic reservoir parameters for meaningful tests
- Consider using actual system data (anonymized)

**Dependencies**

- Blocked by: TEST-019 (integration structure), TEST-028 (system features)
- Blocks: TEST-032 (E2E multi-reservoir)
- Related: TEST-025 (AR on multi-reservoir)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-030: Create Feature Integration Test Summary

**Context**

After implementing feature integration tests (AR, risk, scenarios, system), create a summary showing coverage and any remaining gaps. This closes out Phase 3 integration testing.

**Acceptance Criteria**

- [ ] Summary document generated for feature integration tests
- [ ] All feature tests listed and categorized
- [ ] Coverage gaps identified
- [ ] Execution time metrics included
- [ ] Integration with overall test documentation

**Tasks**

### Implementation

- [ ] Create feature integration test summary
- [ ] Categorize tests: AR models, risk measures, scenarios, system
- [ ] List coverage: what's tested, what's not
- [ ] Generate execution time report
- [ ] Add to TESTING_STRATEGY.md

### Testing

- [ ] Summary accurately reflects test suite
- [ ] All feature areas covered
- [ ] Any gaps documented and justified

### Documentation

- [ ] Update TESTING_STRATEGY.md with results
- [ ] Document any deferred tests
- [ ] Update README with testing status
- [ ] Include coverage badges if appropriate

**Technical Notes**

- Summarize ~40 feature integration tests
- Typical gaps: exotic features, edge cases, performance
- Document rationale for any intentionally omitted tests
- Track metrics for regression detection

**Dependencies**

- Blocked by: TEST-025 through TEST-029
- Blocks: None
- Related: TEST-018 (math summary), TEST-024 (algorithm summary)

**Estimated Effort**

1 story point (1 day, confidence: high)

---

## Phase 5: End-to-End & Performance (Week 6)

### TEST-031: Implement End-to-End Deterministic Tests

**Context**

E2E tests validate the complete algorithm on full problem instances from input to output. Deterministic tests use known optimal solutions or analytical results to validate algorithm correctness.

**Acceptance Criteria**

- [ ] At least 3 E2E deterministic tests created
- [ ] Test: Single-stage deterministic (trivial, analytical solution)
- [ ] Test: Two-stage deterministic (converges exactly)
- [ ] Test: Three-stage deterministic with known solution
- [ ] Tests verify solution within tolerance of known optimum
- [ ] All E2E deterministic tests pass

**Tasks**

### Implementation

- [ ] Create `tests/e2e/` directory
- [ ] Create `tests/e2e/deterministic_2stage.rs`
- [ ] Implement `test_deterministic_2stage_exact_convergence()` - gap → 0
- [ ] Implement `test_deterministic_2stage_known_solution()` - compare to analytical
- [ ] Create `tests/e2e/deterministic_3stage.rs`
- [ ] Implement `test_deterministic_3stage_convergence()` - converges in <50 iterations
- [ ] Create problem instances with known solutions (hand-computed or from literature)
- [ ] Verify solution within 1% of known optimum

### Testing

- [ ] Run with sufficient iterations for convergence
- [ ] Verify gap < $10 for deterministic problems
- [ ] Compare solution to analytical result
- [ ] Check convergence speed (iterations to converge)
- [ ] Tests may take 10-30s each

### Documentation

- [ ] Document E2E test problems and their characteristics
- [ ] Include analytical solutions or references
- [ ] Document expected convergence behavior
- [ ] Add problem diagrams/descriptions

**Technical Notes**

- Deterministic: single scenario path (no sampling error)
- Should converge exactly: gap → 0
- Known solutions: from literature, hand calculation, or other solvers
- Typical convergence: <20 iterations for 2-stage, <50 for 3-stage
- Tolerance: within 0.1-1% of known optimum
- Use realistic problem sizes (not toy problems)
- Consider canonical examples from SDDP literature

**Dependencies**

- Blocked by: TEST-001 (fixtures), all Phase 3 tests
- Blocks: None
- Related: TEST-032 (stochastic E2E)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---


### TEST-032: Implement End-to-End Stochastic Tests

**Context**

Stochastic E2E tests validate the algorithm on realistic problems with uncertainty. These tests verify convergence behavior, solution quality, and handling of sampling error in multi-stage stochastic problems.

**Acceptance Criteria**

- [ ] At least 4 E2E stochastic tests created
- [ ] Test: Three-stage stochastic converges to reasonable gap
- [ ] Test: PAR model E2E with temporal correlation
- [ ] Test: Multi-reservoir cascade with stochastic inflows
- [ ] Test: CVaR risk measure E2E
- [ ] Tests verify convergence and solution quality
- [ ] All E2E stochastic tests pass

**Tasks**

### Implementation

- [ ] Create `tests/e2e/stochastic_3stage.rs`
- [ ] Implement `test_stochastic_3stage_convergence()` - converges to <5% gap
- [ ] Implement `test_stochastic_3stage_solution_quality()` - verify simulation results
- [ ] Create `tests/e2e/par_model_e2e.rs`
- [ ] Implement `test_par2_inflow_convergence()` - PAR(2) model E2E
- [ ] Implement `test_par_seasonal_adjustment_e2e()` - seasonal PAR
- [ ] Create `tests/e2e/multi_reservoir.rs`
- [ ] Implement `test_three_reservoir_cascade_e2e()` - complex cascade
- [ ] Create `tests/e2e/cvar_risk_e2e.rs`
- [ ] Implement `test_cvar_risk_averse_convergence()` - CVaR E2E

### Testing

- [ ] Run with 50-100 iterations
- [ ] Verify relative gap < 5% at convergence
- [ ] Verify monotonic lower bound
- [ ] Compare solution to benchmark if available
- [ ] Run simulation to validate policy
- [ ] Tests may take 30-60s each

### Documentation

- [ ] Document E2E test problem specifications
- [ ] Include problem diagrams and descriptions
- [ ] Document expected convergence characteristics
- [ ] Reference benchmark problems if using standard cases

**Technical Notes**

- Stochastic: sampling error prevents exact convergence
- Target gap: 2-5% relative gap is typical
- Iterations: 50-100 for reasonable convergence
- PAR model: verify temporal correlation preserved
- Multi-reservoir: verify cascade interactions correct
- CVaR: more conservative than expectation
- Simulation: run out-of-sample scenarios to validate policy
- Typical problem size: 12-52 stages, 3-10 reservoirs, 50-200 scenarios

**Dependencies**

- Blocked by: TEST-031 (deterministic E2E), all Phase 3 tests
- Blocks: None
- Related: TEST-025 (AR integration), TEST-026 (risk integration)

**Estimated Effort**

3 story points (2-3 days, confidence: medium - problem setup complexity)

---

### TEST-033: Implement Performance Benchmark Suite

**Context**

Performance benchmarks track algorithm speed and scalability over time. The benchmark suite prevents performance regressions and validates that optimizations actually improve performance.

**Acceptance Criteria**

- [ ] Criterion benchmark suite created with at least 4 benchmarks
- [ ] Benchmark: Tiny problem (baseline overhead)
- [ ] Benchmark: Small problem (quick regression check)
- [ ] Benchmark: Medium problem (realistic performance)
- [ ] Benchmark: Large problem (scalability test)
- [ ] Benchmarks run successfully and produce reports
- [ ] Baseline performance metrics documented

**Tasks**

### Implementation

- [ ] Setup Criterion benchmarking framework (if not already)
- [ ] Create `benches/sddp_benchmarks.rs`
- [ ] Implement `benchmark_tiny_deterministic()` - 2 stages, 1 hydro
- [ ] Implement `benchmark_small_stochastic()` - 3 stages, 2 hydros, 10 scenarios
- [ ] Implement `benchmark_medium_par1()` - 12 stages, 5 hydros, 50 scenarios, PAR(1)
- [ ] Implement `benchmark_large_par2()` - 52 stages, 10 hydros, 100 scenarios, PAR(2)
- [ ] Document expected performance for each benchmark
- [ ] Add benchmark CI job (may be separate from test CI)

### Testing

- [ ] Benchmarks compile and run
- [ ] Benchmarks complete in reasonable time (<5 min total)
- [ ] Benchmark results are reproducible
- [ ] Baseline performance documented
- [ ] Performance comparison works

### Documentation

- [ ] Document benchmark problems
- [ ] Document expected performance baselines
- [ ] Add benchmark results to BENCHMARK_RESULTS.md
- [ ] Document how to run benchmarks
- [ ] Explain performance targets and why

**Technical Notes**

- Use Criterion for statistical benchmarking
- Tiny: <50ms (measures overhead)
- Small: <500ms (quick check)
- Medium: <10s (realistic)
- Large: <5 min (scalability)
- Run with `cargo bench`
- Criterion produces statistical analysis and trends
- Consider flamegraphs for profiling
- Document hardware used for baseline (CPU, RAM)
- Track performance over git history

**Dependencies**

- Blocked by: TEST-001 (fixtures)
- Blocks: None
- Related: TEST-034 (regression tests use benchmarks)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-034: Implement Regression Test Suite

**Context**

Regression tests use fixed seeds and historical baselines to detect when algorithm behavior changes unexpectedly. These tests catch performance regressions, convergence degradation, and solution quality changes.

**Acceptance Criteria**

- [ ] At least 3 regression tests created with baselines
- [ ] Test: Deterministic 2-stage baseline (known LB)
- [ ] Test: Stochastic convergence speed baseline (iterations to 5% gap)
- [ ] Test: PAR(2) solution quality baseline (LB within range)
- [ ] Tests use fixed seeds for reproducibility
- [ ] Baselines documented with tolerances
- [ ] All regression tests pass

**Tasks**

### Implementation

- [ ] Create `tests/regression/` directory
- [ ] Create `tests/regression/convergence_baselines.rs`
- [ ] Implement `regression_deterministic_2stage()` - fixed LB baseline
- [ ] Implement `regression_stochastic_convergence_speed()` - fixed iterations baseline
- [ ] Implement `regression_par2_solution_quality()` - fixed LB range
- [ ] Document baseline values and how they were established
- [ ] Add baselines to `tests/regression/baselines.json` or code constants
- [ ] Use fixed seeds throughout

### Testing

- [ ] Tests pass with current implementation
- [ ] Tests fail if baselines intentionally degraded
- [ ] Tests allow appropriate tolerance (±2-5%)
- [ ] Tests are reproducible across runs
- [ ] Tests complete in <30s total

### Documentation

- [ ] Document regression testing approach
- [ ] Document how baselines were established
- [ ] Document tolerance rationale
- [ ] Document how to update baselines (when intentional changes made)

**Technical Notes**

- Fixed seed: ensures same random numbers every run
- Baseline: historical performance from known-good version
- Tolerance: typically 2-5% to avoid false positives
- Deterministic baseline: exact LB value (e.g., 523.45 ± 1.0)
- Convergence speed: iterations to 5% gap (e.g., 35 ± 10 iterations)
- Solution quality: LB range (e.g., 1245.67 ± 50.0)
- Update baselines when algorithm intentionally improved
- Document baseline provenance (git commit, date, hardware)

**Dependencies**

- Blocked by: TEST-031, TEST-032, TEST-033
- Blocks: None
- Related: All testing phases

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-035: Implement Memory and Resource Tests

**Context**

Memory and resource tests ensure the algorithm doesn't leak memory, doesn't use excessive memory, and properly manages resources like cut pools. These tests catch resource issues before they become production problems.

**Acceptance Criteria**

- [ ] At least 3 memory/resource tests created
- [ ] Test: Memory usage stays within bounds for large problems
- [ ] Test: Cut pool memory doesn't grow unbounded
- [ ] Test: No memory leaks detected
- [ ] Tests use realistic problem sizes
- [ ] All resource tests pass

**Tasks**

### Implementation

- [ ] Create `tests/performance/` directory
- [ ] Create `tests/performance/memory_usage.rs`
- [ ] Implement `test_memory_usage_large_state()` - 50 hydros AR(2)
- [ ] Implement `test_cut_pool_memory_growth()` - verify cut selection working
- [ ] Implement `test_no_memory_leaks()` - repeated runs don't grow
- [ ] Add memory measurement utilities
- [ ] Set reasonable memory limits (e.g., <500MB for test problems)

### Testing

- [ ] Tests run on realistic problem sizes
- [ ] Memory measurements are accurate
- [ ] Tests detect actual memory issues
- [ ] Tests complete in reasonable time
- [ ] Tests don't fail spuriously due to GC timing

### Documentation

- [ ] Document memory testing approach
- [ ] Document expected memory usage for problem sizes
- [ ] Document cut pool management strategy
- [ ] Add memory profiling guide if needed

**Technical Notes**

- Measure memory with platform-specific APIs or external tools
- Cut pool: should be bounded by cut selection strategy
- Memory leak: repeated runs with same problem shouldn't grow
- Large state: 50 hydros × AR(2) = 150-dim state
- Expected memory: typically <100MB for test problems, <1GB for production
- Cut pool size: typically <10,000 cuts with selection
- Consider using valgrind or similar for leak detection
- May need `#[ignore]` for slow memory tests

**Dependencies**

- Blocked by: TEST-001 (fixtures)
- Blocks: None
- Related: TEST-033 (performance benchmarks)

**Estimated Effort**

2 story points (1-2 days, confidence: medium - platform-specific)

---

### TEST-036: Setup CI/CD Test Pipeline

**Context**

CI/CD pipeline automates testing on every commit and PR. Setup includes fast CI for immediate feedback, full CI for PRs, and nightly tests for comprehensive validation and benchmarks.

**Acceptance Criteria**

- [ ] GitHub Actions workflows created
- [ ] Fast CI suite (<30s) runs on every push
- [ ] Full CI suite (~2min) runs on PRs
- [ ] Nightly test suite runs comprehensive tests and benchmarks
- [ ] Test results visible in PR status checks
- [ ] Coverage reports generated and tracked
- [ ] All CI pipelines working

**Tasks**

### Implementation

- [ ] Create `.github/workflows/fast-tests.yml`
- [ ] Configure fast CI: unit tests + critical integration tests
- [ ] Create `.github/workflows/full-tests.yml`
- [ ] Configure full CI: all tests except long-running E2E
- [ ] Create `.github/workflows/nightly-tests.yml`
- [ ] Configure nightly: full suite + benchmarks + E2E
- [ ] Setup test result reporting
- [ ] Setup coverage reporting (Codecov or similar)
- [ ] Add status badges to README

### Testing

- [ ] Fast CI completes in <30s
- [ ] Full CI completes in <3min
- [ ] Nightly completes in <30min
- [ ] Failed tests block PR merge
- [ ] Coverage reports generated
- [ ] CI runs on multiple platforms if needed (Linux, macOS)

### Documentation

- [ ] Document CI/CD pipeline structure
- [ ] Document how to run each test suite locally
- [ ] Document how to interpret CI results
- [ ] Add CI status badges to README
- [ ] Document how to update CI configuration

**Technical Notes**

- Fast CI: `cargo test --lib --tests -- --test-threads=4` (unit + quick integration)
- Full CI: `cargo test --all-features` (everything except `#[ignore]`)
- Nightly: `cargo test --all-features -- --include-ignored` + `cargo bench`
- Use GitHub Actions cache for dependencies
- Consider matrix builds for multiple Rust versions
- Test timeout: 5min for full CI, 60min for nightly
- Coverage: tarpaulin, grcov, or Codecov
- Badge examples: ![Tests](badge-url), ![Coverage](coverage-url)

**Dependencies**

- Blocked by: All previous test tickets (needs tests to run)
- Blocks: None
- Related: All testing infrastructure

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

## Phase 6: Documentation & Polish (Ongoing)

### TEST-037: Create Comprehensive Testing Documentation

**Context**

Testing documentation helps contributors understand the testing strategy, write new tests, and maintain test quality. Comprehensive docs reduce onboarding time and ensure consistent test quality.

**Acceptance Criteria**

- [ ] Testing guide created for contributors
- [ ] Test writing examples provided
- [ ] Test fixture usage documented
- [ ] Test utilities documented with examples
- [ ] Mathematical property testing explained
- [ ] Documentation complete and reviewed

**Tasks**

### Implementation

- [ ] Create `docs/TESTING_GUIDE.md`
- [ ] Document test organization and structure
- [ ] Provide test writing templates and examples
- [ ] Document test fixtures and builders
- [ ] Document test utilities and assertions
- [ ] Explain mathematical property testing approach
- [ ] Add examples for each test category
- [ ] Create quick-start guide for writing tests

### Testing

- [ ] Documentation reviewed by team
- [ ] Examples compile and run
- [ ] New contributors can follow guide
- [ ] All test categories covered

### Documentation

- [ ] Add testing guide to main docs
- [ ] Link from CONTRIBUTING.md
- [ ] Add to README table of contents
- [ ] Include in documentation website if exists

**Technical Notes**

- Cover all test types: unit, integration, E2E, property, performance
- Provide copy-paste templates for common test patterns
- Explain when to use each test type
- Document test naming conventions
- Explain fixture design philosophy
- Include anti-patterns to avoid
- Link to relevant test examples in codebase

**Dependencies**

- Blocked by: All previous test tickets (needs complete test suite)
- Blocks: None
- Related: TEST-038 (architecture docs)

**Estimated Effort**

2 story points (1-2 days, confidence: high)

---

### TEST-038: Update Architecture Documentation

**Context**

Testing reveals architectural patterns and decisions. Update architecture documentation to reflect testing insights, document test hooks, and explain testability considerations in the design.

**Acceptance Criteria**

- [ ] Architecture docs updated with testing considerations
- [ ] Test hooks and interfaces documented
- [ ] Testability patterns explained
- [ ] Module interaction diagrams updated
- [ ] All architecture docs reviewed

**Tasks**

### Implementation

- [ ] Update `docs/ARCHITECTURE.md` (or create if missing)
- [ ] Add section on testability considerations
- [ ] Document test hooks and dependency injection points
- [ ] Update module diagrams showing test boundaries
- [ ] Document mock/fake implementations
- [ ] Explain fixture design patterns
- [ ] Add testing architecture diagram

### Testing

- [ ] Architecture docs reviewed
- [ ] Diagrams are accurate and helpful
- [ ] Test boundaries are clear

### Documentation

- [ ] Link architecture docs from README
- [ ] Cross-reference with testing guide
- [ ] Include in documentation website

**Technical Notes**

- Testability: loose coupling, dependency injection, clear interfaces
- Test hooks: points where mocks can be injected
- Boundaries: where unit tests end and integration tests begin
- Fixtures: how to create test systems and scenarios
- Mocks: mock solver for unit tests
- Consider architecture decision records (ADRs) for major testing decisions

**Dependencies**

- Blocked by: TEST-037 (testing guide provides context)
- Blocks: None
- Related: All architectural decisions

**Estimated Effort**

2 story points (1-2 days, confidence: medium)

---

### TEST-039: Create Test Fixture Builders and Helpers

**Context**

Builder fixtures make test writing easier and more maintainable. Create fluent builder APIs for common test scenarios, reducing boilerplate and improving test readability.

**Acceptance Criteria**

- [ ] SystemBuilder created with fluent API
- [ ] SddpBuilder wrapper created for test setup
- [ ] ScenarioBuilder created for scenario generation
- [ ] At least 3 existing tests refactored to use builders
- [ ] All builders documented with examples
- [ ] Builders support common variations

**Tasks**

### Implementation

- [ ] Create `tests/fixtures/builders/` directory
- [ ] Create `tests/fixtures/builders/system_builder.rs`
- [ ] Implement `SystemBuilder` with methods: `with_hydros()`, `with_cascade()`, `with_transmission()`
- [ ] Create `tests/fixtures/builders/sddp_builder.rs`
- [ ] Implement `SddpTestBuilder` wrapping algorithm builder
- [ ] Create `tests/fixtures/builders/scenario_builder.rs`
- [ ] Implement `ScenarioBuilder` for easy scenario creation
- [ ] Refactor 3-5 existing tests to use builders
- [ ] Add doc comments and examples to each builder

### Testing

- [ ] Builders can create valid systems
- [ ] Builders support chaining
- [ ] Builders have sensible defaults
- [ ] Example tests demonstrate builder usage
- [ ] Tests using builders are more readable

### Documentation

- [ ] Add builder examples to testing guide
- [ ] Document builder patterns
- [ ] Show before/after refactoring examples
- [ ] Document common builder configurations

**Technical Notes**

- Fluent API: methods return `self` for chaining
- Defaults: sensible defaults for optional fields
- Variations: easy to create test variants
- Example: `SystemBuilder::new().with_cascade(3).with_transmission().build()`
- Consider typed builder pattern for compile-time validation
- Keep builders simple and focused

**Dependencies**

- Blocked by: TEST-001 (fixture fixes), TEST-037 (testing guide)
- Blocks: None
- Related: All test tickets (builders improve all tests)

**Estimated Effort**

3 story points (2-3 days, confidence: high)

---

### TEST-040: Implement Test Coverage Analysis and Reporting

**Context**

Test coverage analysis identifies untested code and tracks coverage over time. Setup automated coverage reporting to maintain and improve test coverage.

**Acceptance Criteria**

- [ ] Coverage tool configured (tarpaulin or grcov)
- [ ] Coverage reports generated locally and in CI
- [ ] Coverage thresholds defined
- [ ] Coverage badge added to README
- [ ] Coverage reports archived in CI
- [ ] Coverage tracked over time

**Tasks**

### Implementation

- [ ] Setup tarpaulin or grcov for coverage
- [ ] Create coverage reporting script
- [ ] Add coverage job to CI
- [ ] Configure coverage thresholds (e.g., >80%)
- [ ] Setup Codecov or Coveralls integration
- [ ] Add coverage badge to README
- [ ] Document how to generate coverage reports locally

### Testing

- [ ] Coverage reports generate successfully
- [ ] Coverage metrics are accurate
- [ ] Coverage trends tracked in CI
- [ ] Uncovered code identified correctly

### Documentation

- [ ] Document coverage tool usage
- [ ] Document coverage thresholds and why
- [ ] Add coverage interpretation guide
- [ ] Link coverage reports from docs

**Technical Notes**

- Tools: cargo-tarpaulin (easy), grcov (more features)
- Command: `cargo tarpaulin --out Html --output-dir coverage/`
- Thresholds: typically 70-80% for new code, higher for critical paths
- Exclude: generated code, trivial code, test utilities
- CI integration: upload to Codecov for trend tracking
- Badge: ![Coverage](https://codecov.io/gh/user/repo/branch/main/graph/badge.svg)
- Focus on branch coverage, not just line coverage

**Dependencies**

- Blocked by: Most test tickets (need tests to measure coverage)
- Blocks: None
- Related: TEST-036 (CI/CD pipeline)

**Estimated Effort**

1 story point (1 day, confidence: high)

---

### TEST-041: Create Test Maintenance Guide

**Context**

Test suites require ongoing maintenance as code evolves. Create a guide for maintaining tests, updating fixtures, handling flaky tests, and keeping tests fast and reliable.

**Acceptance Criteria**

- [ ] Test maintenance guide created
- [ ] Guide covers fixture updates
- [ ] Guide covers handling flaky tests
- [ ] Guide covers test performance optimization
- [ ] Guide covers test refactoring
- [ ] Guide reviewed and integrated into docs

**Tasks**

### Implementation

- [ ] Create `docs/TEST_MAINTENANCE.md`
- [ ] Document fixture update process
- [ ] Document flaky test diagnosis and fixes
- [ ] Document test performance profiling
- [ ] Document when to refactor tests
- [ ] Document test smell patterns to avoid
- [ ] Add examples of good and bad test maintenance

### Testing

- [ ] Guide reviewed by team
- [ ] Examples are realistic and helpful
- [ ] Common issues covered

### Documentation

- [ ] Link from testing guide
- [ ] Add to contributor documentation
- [ ] Include in onboarding materials

**Technical Notes**

- Fixture updates: when API changes, update builders first
- Flaky tests: usually timing issues or non-deterministic behavior
- Test performance: profile with `cargo test -- --nocapture` and timing logs
- Refactoring: when tests become hard to read or maintain
- Test smells: large setup, unclear assertions, testing implementation not behavior
- Maintenance frequency: review tests quarterly, update as needed

**Dependencies**

- Blocked by: TEST-037 (testing guide), TEST-039 (builders)
- Blocks: None
- Related: All documentation tickets

**Estimated Effort**

1 story point (1 day, confidence: high)

---

### TEST-042: Final Testing Strategy Review and Update

**Context**

After implementing the testing strategy, review and update the TESTING_STRATEGY.md document to reflect actual implementation, lessons learned, and any deviations from the plan.

**Acceptance Criteria**

- [ ] TESTING_STRATEGY.md reviewed and updated
- [ ] Actual test counts match or documented if different
- [ ] Lessons learned documented
- [ ] Future work section updated
- [ ] Document marked as "IMPLEMENTED"

**Tasks**

### Implementation

- [ ] Review TESTING_STRATEGY.md against actual implementation
- [ ] Update test counts and metrics
- [ ] Document any deviations from plan
- [ ] Add lessons learned section
- [ ] Update future work section
- [ ] Mark strategy as implemented with date
- [ ] Archive original strategy as TESTING_STRATEGY_ORIGINAL.md if significant changes

### Testing

- [ ] All sections reflect actual implementation
- [ ] Metrics are accurate
- [ ] Links and references are valid

### Documentation

- [ ] TESTING_STRATEGY.md is up-to-date
- [ ] Lessons learned captured
- [ ] Future work clearly identified
- [ ] Document serves as reference for testing approach

**Technical Notes**

- Compare planned vs actual: test counts, effort, duration
- Document wins: what worked well, what was underestimated
- Document challenges: what was harder than expected
- Document discoveries: insights gained during implementation
- Update metrics: actual test suite execution time, coverage, etc.
- Preserve history: keep original strategy for comparison

**Dependencies**

- Blocked by: All previous test tickets (needs complete implementation)
- Blocks: None
- Related: All testing work

**Estimated Effort**

1 story point (1 day, confidence: high)

---

## Summary of All Tickets

### Phase 1: Foundation (Week 1-2) - 6 tickets
- TEST-001: Fix Core Test Fixtures (3 days)
- TEST-002: Create Test Utility Library (2 days)
- TEST-003: Add Critical Unit Tests for cut.rs (3 days)
- TEST-004: Add Critical Unit Tests for risk_measure.rs (2 days)
- TEST-005: Add Critical Unit Tests for system.rs (3 days)
- TEST-006: Add Critical Algorithm Tests for sddp/mod.rs (3 days)

**Total Phase 1: ~16 days (3 weeks with testing/docs)**

### Phase 2: Mathematical Validation (Week 3) - 6 tickets
- TEST-013: Cut Validity Properties (2 days)
- TEST-014: Convergence Properties (2 days)
- TEST-015: AR Model Chain Rule Properties (3 days)
- TEST-016: Risk Measure Properties (2 days)
- TEST-017: Numerical Stability Tests (2 days)
- TEST-018: Mathematical Test Summary Report (1 day)

**Total Phase 2: ~12 days**

### Phase 3: Integration Tests (Week 4-5) - 12 tickets
- TEST-019: Algorithm Integration Structure (2 days)
- TEST-020: Forward Pass Integration (2 days)
- TEST-021: Backward Pass Integration (3 days)
- TEST-022: Training Loop Integration (2 days)
- TEST-023: Convergence Integration (2 days)
- TEST-024: Algorithm Integration Summary (1 day)
- TEST-025: AR Model Integration (4 days)
- TEST-026: Risk Measure Integration (3 days)
- TEST-027: Scenario Generation Integration (2 days)
- TEST-028: System Feature Integration (3 days)
- TEST-029: Multi-Reservoir Integration (2 days)
- TEST-030: Feature Integration Summary (1 day)

**Total Phase 3: ~27 days**

### Phase 4: End-to-End & Performance (Week 6) - 6 tickets
- TEST-031: E2E Deterministic Tests (2 days)
- TEST-032: E2E Stochastic Tests (3 days)
- TEST-033: Performance Benchmark Suite (2 days)
- TEST-034: Regression Test Suite (2 days)
- TEST-035: Memory and Resource Tests (2 days)
- TEST-036: CI/CD Test Pipeline (2 days)

**Total Phase 4: ~13 days**

### Phase 5: Documentation & Polish (Ongoing) - 6 tickets
- TEST-037: Comprehensive Testing Documentation (2 days)
- TEST-038: Update Architecture Documentation (2 days)
- TEST-039: Test Fixture Builders (3 days)
- TEST-040: Coverage Analysis and Reporting (1 day)
- TEST-041: Test Maintenance Guide (1 day)
- TEST-042: Final Testing Strategy Review (1 day)

**Total Phase 5: ~10 days**

---

## Grand Total: 42 Tickets, ~78 days of effort

**Recommended Sprint Organization:**

- **6 sprints × 2 weeks** = 12 weeks calendar time
- **With 1-2 developers working in parallel** = 6-8 weeks calendar time
- **Buffer for reviews, debugging, and unexpected issues** = +20%
- **Realistic timeline: 8-10 weeks (2-2.5 months)**

---

## Prioritization Notes

**Must Have (Critical Path):**
- Phase 1: All tickets (foundation)
- Phase 2: TEST-013, TEST-014 (core properties)
- Phase 3: TEST-019-024 (algorithm integration)
- Phase 4: TEST-031, TEST-036 (E2E and CI)

**Should Have (High Value):**
- Phase 2: TEST-015, TEST-016 (AR and risk properties)
- Phase 3: TEST-025-027 (feature integration)
- Phase 4: TEST-033, TEST-034 (benchmarks and regression)
- Phase 5: TEST-037, TEST-039 (docs and builders)

**Nice to Have (Polish):**
- Phase 3: TEST-028-029 (system features)
- Phase 4: TEST-032, TEST-035 (advanced E2E and memory)
- Phase 5: TEST-038, TEST-040-042 (documentation polish)

---

## Quick Start for Implementation

**Week 1 Immediate Tasks:**
1. Start with TEST-001 (Fix fixtures) - blocks everything
2. Parallel: TEST-002 (Utilities) - used by many tests
3. Then TEST-003 (Cut tests) - most critical

**First Sprint Goal:**
Complete Phase 1 (foundation) to unblock integration testing.

**Success Metrics:**
- All fixtures compile and work
- 35+ new critical unit tests added
- Test utilities library established
- Fast test suite runs in <30s

---

**End of Implementation Tickets Document**

For questions or clarifications on any ticket, refer to the original TESTING_STRATEGY.md document or consult with the team lead.

