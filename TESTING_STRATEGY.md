# SDDP Testing Strategy

**Document Version**: 1.0  
**Date**: 2025-11-06  
**Author**: SDDP Optimization Expert

---

## Executive Summary

This document defines the comprehensive testing strategy for the POWE.RS SDDP implementation. After analyzing ~23K lines of source code and ~20K lines of test code, I've identified that while unit tests are passing, the integration test suite has significant technical debt from multiple refactorings. This strategy provides a roadmap to establish a robust, maintainable test suite aligned with SDDP algorithmic principles.

**Key Findings:**
- **Unit Tests**: 357 tests across 20 modules - well-distributed but could benefit from reorganization
- **Integration Tests**: 45 test files with many broken due to API changes in `System`, `Hydro`, `Bus` structures
- **Test Fixtures**: Good foundation but need updating for current API
- **Coverage Gaps**: Missing tests for convergence properties, numerical stability, and algorithm correctness

**Testing Philosophy:**
> "Test what SDDP promises: monotonic lower bounds, finite convergence, correct dual values, and valid cuts."

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Testing Pyramid for SDDP](#2-testing-pyramid-for-sddp)
3. [Unit Testing Strategy](#3-unit-testing-strategy)
4. [Integration Testing Strategy](#4-integration-testing-strategy)
5. [Mathematical Validation Tests](#5-mathematical-validation-tests)
6. [Performance & Regression Tests](#6-performance--regression-tests)
7. [Test Fixtures & Data](#7-test-fixtures--data)
8. [Execution Plan](#8-execution-plan)

---

## 1. Current State Analysis

### 1.1 Source Code Structure

**Total Source Code**: ~23,100 lines across 25 modules

**Core SDDP Modules** (lines of code):
- `sddp/mod.rs`: 3,653 lines - Main algorithm (forward/backward passes, training loop)
- `sddp/builder.rs`: 1,447 lines - Builder pattern for algorithm construction  
- `sddp/instance.rs`: 73 lines - Instance configuration
- `subproblem.rs`: Large module - LP subproblem formulation
- `fcf.rs`: Future cost function and cut management
- `state.rs`: State representation (storage + AR lags)
- `cut.rs`: Benders cut data structures
- `solver.rs`: LP solver interface (HiGHS)

**Supporting Modules**:
- `temporal_model.rs`: PAR model handling
- `scenario_generator.rs`: SAA scenario generation
- `correlation_applicator.rs`: Multi-site correlation
- `risk_measure.rs`: CVaR, expectation, worst-case
- `graph.rs`: Stage graph structure
- `system.rs`: Power system components (hydros, thermals, buses, lines)

### 1.2 Current Unit Test Distribution

**Unit Tests by Module** (357 total):

```
subproblem.rs      : 103 tests  ← Largest, core LP formulation
state.rs           : 26 tests   ← State handling and AR lags
sddp/builder.rs    : 34 tests   ← Builder API
sddp/mod.rs        : 34 tests   ← Algorithm execution
utils/mod.rs       : 23 tests   ← Utility functions
initial_condition  : 17 tests   ← Initial state setup
fcf.rs             : 17 tests   ← Cut management
input_validation   : 16 tests   ← Input validation
utils/simd.rs      : 15 tests   ← SIMD optimizations
input.rs           : 14 tests   ← JSON parsing
solver.rs          : 13 tests   ← Solver interface
correlation_app.   : 10 tests   ← Correlation handling
temporal_model     : 8 tests    ← PAR model
error.rs           : 8 tests    ← Error handling
graph.rs           : 5 tests    ← Graph operations
cut.rs             : 4 tests    ← Cut evaluation
scenario.rs        : 4 tests    ← Scenario structures
risk_measure.rs    : 3 tests    ← Risk measures
cli.rs             : 2 tests    ← CLI parsing
system.rs          : 1 test     ← System structures
```

**Key Observations:**
1. `subproblem.rs` has excellent coverage (103 tests) for the hot path
2. `state.rs` comprehensive tests for AR model integration (26 tests)
3. High-level algorithm tests relatively few (34) for complexity
4. `system.rs` under-tested (1 test) - needs validation

### 1.3 Integration Test Analysis

**Integration Test Files**: 45 files, ~20,729 lines

**Test Categories**:

**AR Model Tests** (7 files):
- `test_ar_cut_validation.rs` - Convergence with AR models
- `test_ar_lag_cut_coefficients.rs` - Cut coefficient correctness
- `test_ar_model_integration.rs` - Mixed load/inflow AR
- `test_ar_psi_validation.rs` - ψ coefficient validation

**Cut Management** (4 files):
- `test_cut.rs` - Basic cut operations
- `test_cut_pool.rs` - Cut pool management
- `test_cut_selection_integration.rs` - Cut selection strategies
- `test_batch_cut_selection.rs` - Batch operations

**SDDP Algorithm** (4 files):
- `test_sddp_algorithm.rs` - Core algorithm
- `test_sddp_instance_builder.rs` - Builder API
- `test_sddp_par_e2e.rs` - End-to-end PAR
- `test_sddp_thread_config.rs` - Threading

**Scenario/Uncertainty** (4 files):
- `test_scenario.rs`
- `test_scenario_generation.rs`
- `test_scenario_generation_integration.rs`
- `test_scenario_validation.rs`

**Validation/Error** (6 files):
- `test_input_validation.rs`
- `test_numerical_validation.rs`
- `test_error_messages.rs`
- `test_input_error_paths.rs`
- `test_explicit_constraints_validation.rs`

**Other** (20+ files for specific features)

### 1.4 Test Fixture Organization

**Fixture Modules** (`tests/fixtures/`):

```
mod.rs                      - Re-exports
benchmarks.rs               - Hydrothermal benchmarks with known solutions
mock_solver.rs              - Mock solver for testing
oos.rs                      - Out-of-sample validation
scenarios.rs                - Scenario generation helpers
simple_2stage_reservoir.rs  - Simple 2-stage case
subproblems.rs              - Subproblem utilities
systems.rs                  - System configurations
validation.rs               - Validation helpers
```

**Fixture Quality**:
- ✅ Good documentation in `benchmarks.rs` (convergence principles)
- ✅ Separation of concerns
- ❌ Many fixtures broken (old API field names)
- ❌ Missing: heterogeneous AR orders, multi-stage cascade

### 1.5 Critical Issues

**Compilation Errors** (many tests broken):

1. **API Changes**:
   - `train()` signature changed
   - Missing fields in `Config`, `Bus`, `Hydro`
   - Field renames in `Hydro` structure

2. **Type Removals**:
   - `TemporalModelInputWrapper` removed
   - `LegacyTemporalModelInput` removed

3. **Structure Changes**:
   - `Bus` requires `hydro_ids`, `source_line_ids`, `target_line_ids`
   - `Hydro` field naming changed

**Test Coverage Gaps**:

**Mathematical Properties NOT Tested**:
- ❌ Cut validity at generation point (should equal objective)
- ❌ Monotonic lower bound convergence
- ❌ Upper bound convergence
- ❌ Dual feasibility
- ❌ State coefficient chain rule for AR

**Numerical Stability NOT Tested**:
- ❌ Kahan summation correctness
- ❌ Floating-point precision in cuts
- ❌ LP solver tolerance handling

**Algorithm Correctness NOT Systematically Tested**:
- ❌ Forward pass state propagation
- ❌ Backward pass branching
- ❌ Risk measure integration
- ❌ Multistage value approximation

---

## 2. Testing Pyramid for SDDP

### 2.1 SDDP Testing Pyramid Structure

Traditional testing pyramid doesn't fully apply to optimization algorithms. For SDDP, we need a specialized pyramid:

```
                    ╱╲
                   ╱  ╲
                  ╱ E2E╲         ← 5-10 tests: Full algorithm convergence
                 ╱      ╲
                ╱────────╲
               ╱          ╲
              ╱ Algorithm  ╲    ← 30-50 tests: Forward/backward passes, risk
             ╱   Integration╲
            ╱────────────────╲
           ╱                  ╲
          ╱  Mathematical      ╲ ← 50-100 tests: Cut validity, dual correctness
         ╱     Validation       ╲
        ╱────────────────────────╲
       ╱                          ╲
      ╱    Component Unit Tests    ╲ ← 300-500 tests: Subproblem, state, solver
     ╱______________________________╲
```

### 2.2 Test Layer Definitions

#### Layer 1: Component Unit Tests (70% of tests)

**Purpose**: Validate individual components in isolation

**Characteristics**:
- Fast (< 10ms each)
- No external dependencies (mock solver when needed)
- Test single functions/methods
- High code coverage

**Example Modules**:
- `subproblem.rs`: LP formulation correctness
- `state.rs`: State extraction and updates
- `cut.rs`: Cut evaluation
- `solver.rs`: Solver interface
- `temporal_model.rs`: PAR coefficient computation

**Test Examples**:
```rust
#[test]
fn test_cut_evaluation_at_generation_point() {
    // Cut should equal objective at the state where it was generated
    let state = vec![50.0, 10.0];  
    let cut = BendersCut { rhs: 100.0, coefficients: vec![-2.0, -3.0] };
    let height = cut.eval_height(&state);
    assert_abs_diff_eq!(height, 100.0 - (-2.0*50.0 + -3.0*10.0), epsilon = 1e-6);
}

#[test]
fn test_par_psi_coefficient_computation() {
    // ψ_i = φ_i for standard AR (no seasonal adjustment in this test)
    let par = PeriodicAR::new(vec![0.6, 0.3], vec![1.0; 12], vec![1.0; 12]);
    let psi = par.compute_psi_coefficients(0);
    assert_abs_diff_eq!(psi[0], 0.6);
    assert_abs_diff_eq!(psi[1], 0.3);
}
```

#### Layer 2: Mathematical Validation Tests (15% of tests)

**Purpose**: Verify SDDP mathematical properties hold

**Characteristics**:
- Medium speed (10ms - 1s)
- Test mathematical invariants
- Independent of specific problem instances
- Focus on algorithm correctness

**Properties to Test**:

1. **Cut Validity**:
   ```rust
   // Cut evaluated at its generation state should equal the objective
   for all cuts: height(state_at_generation) ≈ objective
   ```

2. **Monotonic Lower Bound**:
   ```rust
   // Lower bound should never decrease
   for i in 1..iterations: LB[i] >= LB[i-1] - tolerance
   ```

3. **Dual Feasibility**:
   ```rust
   // Duals should satisfy complementary slackness
   for all constraints: dual * slack ≈ 0
   ```

4. **Chain Rule for AR States**:
   ```rust
   // Cut coefficient for lagged inflow
   ∂V/∂Y_{t-j} = (λ^hydro + λ^AR) * ψ_j
   ```

5. **Risk Measure Properties**:
   ```rust
   // CVaR >= Expectation >= Min (for costs)
   assert!(cvar_cost >= expectation_cost);
   assert!(expectation_cost >= min_cost);
   ```

**Test Examples**:
```rust
#[test]
fn test_cut_validity_at_generation_point() {
    // Mathematical property: cut height at training state equals objective
    let (mut sddp, saa) = create_simple_system();
    sddp.train(1, 1, &saa).unwrap();
    
    for node in sddp.graph().nodes() {
        for cut in node.fcf().cuts() {
            let height = cut.eval_height(&cut.training_state());
            let obj = cut.training_objective();
            assert_abs_diff_eq!(height, obj, epsilon = 1e-4);
        }
    }
}

#[test]
fn test_monotonic_lower_bound_convergence() {
    let (mut sddp, saa) = create_stochastic_system();
    let result = sddp.train(50, 10, &saa).unwrap();
    
    let iterations = result.iterations();
    for i in 1..iterations.len() {
        let lb_prev = iterations[i-1].lower_bound;
        let lb_curr = iterations[i].lower_bound;
        assert!(
            lb_curr >= lb_prev - 1e-6,
            "LB decreased: {} → {}", lb_prev, lb_curr
        );
    }
}
```

#### Layer 3: Algorithm Integration Tests (10% of tests)

**Purpose**: Test interactions between SDDP components

**Characteristics**:
- Slower (1s - 10s)
- Test component interactions
- Use real LP solver
- Verify end-to-end flows

**Scenarios to Test**:

1. **Forward Pass**:
   - State initialization
   - Uncertainty realization
   - LP solve
   - State extraction
   - Cost accumulation

2. **Backward Pass**:
   - Scenario branching
   - Cut generation
   - Cut coefficient computation (chain rule)
   - Cut addition to FCF

3. **Risk Measures**:
   - Probability adjustment
   - Expected value vs CVaR
   - Cut aggregation with risk

4. **AR Model Integration**:
   - Lag constraint updates
   - Dual extraction from AR constraints
   - Cut coefficients for lagged states

**Test Examples**:
```rust
#[test]
fn test_forward_pass_state_propagation() {
    // State should propagate correctly through stages
    let (mut sddp, saa) = create_3stage_system();
    let trajectory = sddp.forward_pass(&saa.scenarios()[0]).unwrap();
    
    // Check state continuity
    for t in 1..trajectory.len() {
        let prev_storage = trajectory[t-1].final_storage();
        let curr_initial = trajectory[t].initial_storage();
        assert_vector_eq!(prev_storage, curr_initial, epsilon = 1e-8);
    }
}

#[test]
fn test_backward_pass_cut_addition() {
    let (mut sddp, saa) = create_2stage_system();
    
    // Forward pass
    let trajectory = sddp.forward_pass(&saa.scenarios()[0]).unwrap();
    
    // Backward pass should add cuts
    let cuts_before = sddp.graph().node(1).fcf().cut_count();
    sddp.backward_pass(&trajectory, &saa).unwrap();
    let cuts_after = sddp.graph().node(1).fcf().cut_count();
    
    assert!(cuts_after > cuts_before, "Backward pass should add cuts");
}
```

#### Layer 4: End-to-End Tests (5% of tests)

**Purpose**: Validate complete algorithm convergence

**Characteristics**:
- Slowest (10s - 60s)
- Real-world problem instances
- Known optimal solutions (small problems)
- Convergence to within tolerance

**Test Scenarios**:

1. **Deterministic 2-Stage**:
   - Should converge exactly (no sampling error)
   - Known analytical solution

2. **Stochastic 3-Stage**:
   - Converge to narrow gap
   - Validate simulation results

3. **Multi-Reservoir Cascade**:
   - Upstream/downstream coupling
   - Storage water values

4. **AR Model End-to-End**:
   - PAR(2) inflows
   - Heterogeneous AR orders
   - Convergence with temporal correlation

**Test Examples**:
```rust
#[test]
fn test_deterministic_2stage_exact_convergence() {
    let (mut sddp, saa) = create_deterministic_2stage();
    let result = sddp.train(20, 1, &saa).unwrap();
    
    // Deterministic problem should converge tightly
    let final_gap = result.final_gap();
    assert!(final_gap < 10.0, "Gap should be < $10: {}", final_gap);
    
    // Known optimal: $500
    let lb = result.final_lower_bound();
    assert!((lb - 500.0).abs() < 50.0, "Should be near $500");
}

#[test]
fn test_par2_inflow_convergence() {
    let (mut sddp, saa) = create_par2_system();
    let result = sddp.train(100, 20, &saa).unwrap();
    
    // Should converge despite AR dynamics
    let rel_gap = result.relative_gap().unwrap();
    assert!(rel_gap < 0.05, "Relative gap should be < 5%");
    
    // Lower bound should be monotonic
    assert_monotonicity(&result, 1e-6);
}
```

### 2.3 Test Execution Speed Targets

| Layer                | Tests | Avg Time | Total Time | % of Suite |
|----------------------|-------|----------|------------|------------|
| Component Unit       | 400   | 5ms      | 2s         | 70%        |
| Mathematical Valid.  | 80    | 100ms    | 8s         | 15%        |
| Algorithm Integr.    | 50    | 1s       | 50s        | 10%        |
| End-to-End           | 10    | 5s       | 50s        | 5%         |
| **Total**            | **540**| -       | **~110s**  | **100%**   |

**CI/CD Implications**:
- Full suite: ~2 minutes (acceptable for CI)
- Fast subset (unit + math): ~10s (for quick feedback)
- PR gate: Run fast subset + critical E2E tests (~30s)

---

## 3. Unit Testing Strategy

### 3.1 Module-by-Module Unit Test Plan

#### 3.1.1 `subproblem.rs` (Current: 103 tests ✅)

**Status**: Well-tested, maintain current coverage

**Current Coverage** (examples):
- LP model construction
- Constraint RHS updates
- Solution extraction
- Variable indexing
- Deficit/spillage/thermal extraction

**Recommended Additions** (5-10 tests):
```rust
#[test]
fn test_lag_constraint_update_efficiency() {
    // Verify lag constraints only updated when lags change
    // REFACTOR-002 tracking
}

#[test]
fn test_uncertainty_observation_constraint_indexing() {
    // Verify correct innovation indices for loads vs inflows
}

#[test]
fn test_heterogeneous_ar_constraint_layout() {
    // Mixed AR(0), AR(1), AR(2) across hydros
}
```

**Keep**: Current comprehensive coverage of LP operations

#### 3.1.2 `state.rs` (Current: 26 tests ✅)

**Status**: Good coverage of AR state handling

**Current Coverage**:
- State extraction from trajectory
- State coefficient updates
- Heterogeneous AR orders
- Cut evaluation with lags

**Recommended Additions** (3-5 tests):
```rust
#[test]
fn test_state_layout_memory_efficiency() {
    // Verify flattened state has no redundancy
    // Should be: n_hydros + sum(ar_orders)
}

#[test]
fn test_state_extraction_numerical_stability() {
    // Ensure extraction doesn't accumulate floating-point error
}
```

**Keep**: Comprehensive AR order handling tests

#### 3.1.3 `cut.rs` (Current: 4 tests ⚠️)

**Status**: UNDER-TESTED - needs expansion

**Current Tests**: Basic cut creation and evaluation

**Critical Missing Tests** (10-15 new tests):
```rust
#[test]
fn test_cut_evaluation_matches_lp_objective() {
    // Most critical test: cut at training state = objective
}

#[test]
fn test_cut_coefficient_signs() {
    // Storage coefficients should be negative (water has value)
}

#[test]
fn test_cut_evaluation_numerical_precision() {
    // Test with extreme values, check precision
}

#[test]
fn test_cut_domination_detection() {
    // Given two cuts, detect if one dominates
}

#[test]
fn test_kahan_summation_in_cut_aggregation() {
    // Verify Kahan sum used for numerical stability
}

#[test]
fn test_cut_with_zero_coefficients() {
    // Edge case: some hydros inactive
}

#[test]
fn test_cut_with_large_state_dimension() {
    // 50+ hydros with AR(2) → 150+ dimension state
}
```

**Priority**: HIGH - cut correctness is fundamental

#### 3.1.4 `fcf.rs` (Current: 17 tests ✅)

**Status**: Good coverage of cut management

**Current Coverage**:
- Cut addition
- Cut pool updates
- Cut selection
- Dominance tracking

**Recommended Additions** (5 tests):
```rust
#[test]
fn test_cut_pool_memory_management() {
    // Verify old cuts properly removed/freed
}

#[test]
fn test_active_cut_selection_performance() {
    // O(k) not O(n) for k active from n total
}
```

**Keep**: Current cut pool tests

#### 3.1.5 `sddp/mod.rs` (Current: 34 tests)

**Status**: Good basic coverage, needs more edge cases

**Current Coverage**:
- Training result structures
- Forward/backward with default system
- Gap calculations

**Critical Missing Tests** (10-15 tests):
```rust
#[test]
fn test_forward_pass_with_infeasible_subproblem() {
    // How does algorithm handle infeasibility?
}

#[test]
fn test_backward_pass_branching_scenarios() {
    // Verify K scenarios sampled at each node
}

#[test]
fn test_risk_measure_probability_adjustment() {
    // CVaR adjusts probabilities correctly
}

#[test]
fn test_cut_aggregation_with_risk() {
    // Risk-adjusted cuts match theoretical formula
}

#[test]
fn test_multistage_value_function_consistency() {
    // θ_t bounds V_{t+1}(x_t, ξ)
}

#[test]
fn test_parallel_forward_passes() {
    // Multiple forward passes in parallel produce consistent results
}

#[test]
fn test_convergence_detection() {
    // Algorithm stops when gap < tolerance
}
```

**Priority**: HIGH - core algorithm correctness

#### 3.1.6 `solver.rs` (Current: 13 tests ✅)

**Status**: Adequate for wrapper, keep as-is

**Keep**: HiGHS interface tests

#### 3.1.7 `temporal_model.rs` (Current: 8 tests)

**Status**: Needs AR coefficient validation

**Missing Tests** (5-8 tests):
```rust
#[test]
fn test_par_to_ar_transformation() {
    // ψ_i = φ_i * (σ_t / σ_{t-i})
}

#[test]
fn test_deterministic_noise_base_computation() {
    // μ_t - Σ(φ_i * μ_{t-i})
}

#[test]
fn test_seasonal_coefficient_extraction() {
    // Correct period wrapping
}

#[test]
fn test_ar_stationarity_check() {
    // Warn if |φ| > 1 (non-stationary)
}
```

**Priority**: MEDIUM - affects cut coefficients

#### 3.1.8 `risk_measure.rs` (Current: 3 tests ⚠️)

**Status**: UNDER-TESTED

**Critical Missing Tests** (8-10 tests):
```rust
#[test]
fn test_cvar_probability_adjustment() {
    // Tail scenarios get higher weight
}

#[test]
fn test_cvar_bounds_expectation() {
    // CVaR_α(X) >= E[X] for costs
}

#[test]
fn test_expectation_risk_is_identity() {
    // Probabilities unchanged
}

#[test]
fn test_worst_case_selects_maximum() {
    // All weight on worst scenario
}

#[test]
fn test_cvar_alpha_parameter_validation() {
    // 0 < α < 1
}
```

**Priority**: HIGH - risk measures affect convergence

#### 3.1.9 `system.rs` (Current: 1 test ❌)

**Status**: CRITICALLY UNDER-TESTED

**Critical Missing Tests** (15-20 tests):
```rust
#[test]
fn test_system_validation_rules() {
    // All IDs unique, references valid, no cycles
}

#[test]
fn test_hydro_cascade_topology() {
    // Downstream IDs form valid DAG
}

#[test]
fn test_bus_hydro_connection_validity() {
    // All hydro bus_ids exist
}

#[test]
fn test_line_bus_connection_validity() {
    // from_bus, to_bus exist
}

#[test]
fn test_system_dimension_queries() {
    // n_hydros(), n_buses(), etc. correct
}

#[test]
fn test_empty_system_creation() {
    // System with no hydros should be valid
}

#[test]
fn test_duplicate_id_detection() {
    // Should error on duplicate hydro/bus IDs
}
```

**Priority**: HIGH - foundation for all tests

#### 3.1.10 Other Modules

**`scenario_generator.rs`**: Good (scenario generation tests exist)
**`graph.rs`**: Adequate (5 tests for DAG operations)
**`input.rs`**: Good (14 tests for JSON parsing)
**`correlation_applicator.rs`**: Good (10 tests)
**`utils/simd.rs`**: Excellent (15 tests for SIMD operations)

### 3.2 Unit Test Organization

**Current Organization** (embedded in source):
```
src/
├── subproblem.rs
│   └── mod tests { ... }  ← 103 tests here
├── state.rs
│   └── mod tests { ... }  ← 26 tests here
└── ...
```

**Recommended**: Keep current approach
- ✅ Tests close to implementation
- ✅ Easy to find relevant tests
- ✅ Encourages testing while coding

**Enhancement**: Add test documentation
```rust
//! # Testing Strategy for Subproblem
//!
//! This module's tests are organized into:
//! 1. **LP Construction** (tests 1-20): Model building
//! 2. **Constraint Updates** (tests 21-45): RHS updates
//! 3. **Solution Extraction** (tests 46-80): Getting LP solutions
//! 4. **Edge Cases** (tests 81-103): Boundary conditions

#[cfg(test)]
mod tests {
    // ... tests here
}
```

### 3.3 Unit Test Priorities

**Phase 1: Critical Gaps** (2-3 days):
1. `cut.rs`: Add 12 fundamental cut tests
2. `risk_measure.rs`: Add 8 risk measure tests
3. `system.rs`: Add 15 system validation tests
4. `sddp/mod.rs`: Add 10 algorithm correctness tests

**Phase 2: Coverage Enhancement** (2-3 days):
5. `temporal_model.rs`: Add 6 PAR tests
6. `subproblem.rs`: Add 5 edge case tests
7. `state.rs`: Add 3 numerical stability tests

**Phase 3: Polish** (1-2 days):
8. Add test documentation headers
9. Refactor test helpers into `test_utils.rs`
10. Ensure all modules have >= 5 tests

**Target**: 450 unit tests (up from 357)

---

## 4. Integration Testing Strategy

### 4.1 Current State Assessment

**Total Integration Tests**: 45 files
**Status**: Many broken due to API changes
**Action Required**: Rewrite/update most tests

### 4.2 New Integration Test Organization

**Proposed Structure**:
```
tests/
├── integration/
│   ├── algorithm/           ← Forward/backward/training
│   │   ├── forward_pass.rs
│   │   ├── backward_pass.rs
│   │   ├── training_loop.rs
│   │   └── convergence.rs
│   ├── ar_models/           ← PAR model integration
│   │   ├── par_cut_generation.rs
│   │   ├── lag_dynamics.rs
│   │   ├── heterogeneous_orders.rs
│   │   └── mixed_entities.rs
│   ├── risk_measures/       ← Risk integration
│   │   ├── cvar_cuts.rs
│   │   ├── expectation_baseline.rs
│   │   └── worst_case.rs
│   ├── scenarios/           ← Scenario generation
│   │   ├── saa_sampling.rs
│   │   ├── correlation.rs
│   │   └── branching.rs
│   └── system/              ← Power system features
│       ├── cascade_hydros.rs
│       ├── transmission.rs
│       └── thermal_dispatch.rs
├── e2e/                     ← End-to-end tests
│   ├── deterministic_2stage.rs
│   ├── stochastic_3stage.rs
│   ├── par_model_e2e.rs
│   └── multi_reservoir.rs
├── fixtures/                ← Reusable test data
│   ├── mod.rs
│   ├── systems.rs
│   ├── benchmarks.rs
│   └── builders.rs
└── utils/                   ← Test utilities
    ├── mod.rs
    ├── assertions.rs        ← Custom assert macros
    └── validation.rs        ← Result validation helpers
```

### 4.3 Integration Test Categories

#### 4.3.1 Algorithm Integration Tests

**`tests/integration/algorithm/forward_pass.rs`**:
```rust
#[test]
fn test_forward_pass_state_initialization() {
    // Initial state from initial_condition propagates correctly
}

#[test]
fn test_forward_pass_uncertainty_realization() {
    // Scenario inflows/loads applied to constraints
}

#[test]
fn test_forward_pass_cost_accumulation() {
    // Stage costs sum correctly
}

#[test]
fn test_forward_pass_trajectory_storage() {
    // All realizations stored for backward pass
}
```

**`tests/integration/algorithm/backward_pass.rs`**:
```rust
#[test]
fn test_backward_pass_scenario_branching() {
    // K scenarios sampled at each trajectory point
}

#[test]
fn test_backward_pass_cut_generation() {
    // One cut per trajectory realization
}

#[test]
fn test_backward_pass_dual_extraction() {
    // Duals from LP match cut coefficients
}

#[test]
fn test_backward_pass_fcf_update() {
    // Cuts added to correct future cost functions
}
```

**`tests/integration/algorithm/training_loop.rs`**:
```rust
#[test]
fn test_training_iteration_structure() {
    // Each iteration: forward passes → backward pass → bounds
}

#[test]
fn test_training_convergence_criteria() {
    // Stops when gap < tolerance or max iterations
}

#[test]
fn test_training_result_tracking() {
    // Iterations, bounds, timing all recorded
}
```

**`tests/integration/algorithm/convergence.rs`**:
```rust
#[test]
fn test_monotonic_lower_bound() {
    // Fundamental SDDP property
}

#[test]
fn test_upper_bound_improvement() {
    // Simulation cost should generally improve
}

#[test]
fn test_gap_closure() {
    // |UB - LB| should decrease over iterations
}
```

#### 4.3.2 AR Model Integration Tests

**`tests/integration/ar_models/par_cut_generation.rs`**:
```rust
#[test]
fn test_par_cut_with_storage_only() {
    // Baseline: no AR dynamics
}

#[test]
fn test_par_cut_with_ar1_inflows() {
    // Cut coefficients include lagged inflow terms
}

#[test]
fn test_par_cut_chain_rule_validation() {
    // ∂V/∂Y_{t-j} = (λ^hydro + λ^AR) * ψ_j
}

#[test]
fn test_par_cut_seasonal_adjustment() {
    // ψ_i = φ_i * (σ_t / σ_{t-i}) applied correctly
}
```

**`tests/integration/ar_models/heterogeneous_orders.rs`**:
```rust
#[test]
fn test_mixed_ar_orders() {
    // Hydro 0: AR(0), Hydro 1: AR(1), Hydro 2: AR(2)
}

#[test]
fn test_state_dimension_consistency() {
    // State size = n_hydros + sum(ar_orders)
}

#[test]
fn test_heterogeneous_lag_dual_extraction() {
    // Each hydro contributes correct number of lag duals
}
```

**`tests/integration/ar_models/mixed_entities.rs`**:
```rust
#[test]
fn test_loads_ar_inflows_independent() {
    // Loads with AR(1), inflows deterministic
}

#[test]
fn test_loads_independent_inflows_ar() {
    // Opposite case
}

#[test]
fn test_both_loads_and_inflows_ar() {
    // Full complexity case
}
```

#### 4.3.3 Risk Measure Integration Tests

**`tests/integration/risk_measures/cvar_cuts.rs`**:
```rust
#[test]
fn test_cvar_vs_expectation_cuts() {
    // CVaR cuts should be more conservative
}

#[test]
fn test_cvar_probability_weights() {
    // Tail scenarios get boosted weight
}

#[test]
fn test_cvar_convergence() {
    // Should still converge monotonically
}
```

#### 4.3.4 System Integration Tests

**`tests/integration/system/cascade_hydros.rs`**:
```rust
#[test]
fn test_cascade_water_balance() {
    // Upstream release becomes downstream inflow
}

#[test]
fn test_cascade_value_propagation() {
    // Downstream water more valuable (terminal effect)
}
```

**`tests/integration/system/transmission.rs`**:
```rust
#[test]
fn test_transmission_capacity_constraints() {
    // Power flow bounded by line capacity
}

#[test]
fn test_transmission_cost_in_objective() {
    // Transmission cost included in stage cost
}
```

### 4.4 Fixture Modernization Plan

**Priority 1: Fix Broken Fixtures** (1-2 days)

Update `tests/fixtures/systems.rs`:
```rust
// OLD (broken):
Hydro {
    id: 0,
    min_volume: 0.0,  // Field doesn't exist!
    max_volume: 100.0,
    ...
}

// NEW (working):
Hydro {
    id: 0,
    min_storage: 0.0,  // Correct field name
    max_storage: 100.0,
    ...
    // Add all required fields
}
```

**Priority 2: Create Builder Fixtures** (2 days)

New file: `tests/fixtures/builders.rs`:
```rust
/// Builder for test systems with fluent API
pub struct TestSystemBuilder {
    n_hydros: usize,
    n_buses: usize,
    has_transmission: bool,
    ar_orders: Vec<usize>,
}

impl TestSystemBuilder {
    pub fn new() -> Self { ... }
    
    pub fn with_hydros(mut self, n: usize) -> Self {
        self.n_hydros = n;
        self
    }
    
    pub fn with_ar_order(mut self, hydro_id: usize, order: usize) -> Self {
        self.ar_orders[hydro_id] = order;
        self
    }
    
    pub fn build(self) -> System { ... }
}

// Usage in tests:
let system = TestSystemBuilder::new()
    .with_hydros(3)
    .with_ar_order(0, 2)
    .with_ar_order(1, 1)
    .build();
```

**Priority 3: Canonical Benchmarks** (2 days)

Update `tests/fixtures/benchmarks.rs`:
```rust
/// Canonical test cases with known properties

/// Trivial: 1 stage, 1 hydro, deterministic
/// Expected cost: Can compute by hand
pub fn trivial_deterministic() -> (SddpAlgorithm, SAA);

/// Simple: 2 stages, 1 hydro, 2 scenarios
/// Expected: Converge in < 10 iterations
pub fn simple_stochastic() -> (SddpAlgorithm, SAA);

/// Medium: 3 stages, 2 hydros cascade, 10 scenarios
/// Expected: Converge in < 50 iterations
pub fn medium_cascade() -> (SddpAlgorithm, SAA);

/// Complex: 12 stages, 5 hydros, PAR(2), 100 scenarios
/// Expected: Converge in < 200 iterations
pub fn complex_par2() -> (SddpAlgorithm, SAA);
```

### 4.5 Integration Test Execution Plan

**Phase 1: Core Algorithm** (3-4 days)
- Write 20 tests in `integration/algorithm/`
- Fix fixtures to make them compile
- Validate forward/backward passes work

**Phase 2: AR Models** (3-4 days)
- Write 15 tests in `integration/ar_models/`
- Validate cut coefficients for AR lags
- Test heterogeneous orders

**Phase 3: System Features** (2-3 days)
- Write 10 tests in `integration/system/`
- Cascade hydros, transmission, thermal

**Phase 4: Risk & Scenarios** (2 days)
- Write 8 tests for risk measures
- Write 6 tests for scenario generation

**Target**: ~60 integration tests (replacing broken 45)

---

## 5. Mathematical Validation Tests

### 5.1 Purpose

These tests verify that SDDP's mathematical properties hold, independent of specific problem instances. They test *algorithm correctness*, not just *code correctness*.

### 5.2 Property-Based Tests

#### 5.2.1 Cut Validity Properties

**Property 1: Cut Evaluation at Training State**
```rust
/// Mathematical property: A cut evaluated at its training state equals the objective
/// 
/// Formulation: θ_t ≥ E[V_{t+1}] - π'(x_t - x̄_t)
/// At x_t = x̄_t: θ_t = E[V_{t+1}]
#[test]
fn property_cut_equals_objective_at_training_state() {
    for benchmark in all_benchmarks() {
        let (mut sddp, saa) = benchmark();
        sddp.train(10, 5, &saa).unwrap();
        
        for node in sddp.graph().nodes() {
            for cut in node.fcf().cuts() {
                let state = cut.training_state();
                let height = cut.eval_height(&state);
                let objective = cut.training_objective();
                
                assert_abs_diff_eq!(
                    height, 
                    objective, 
                    epsilon = 1e-4,
                    "Cut height at training state should equal objective"
                );
            }
        }
    }
}
```

**Property 2: Storage Coefficient Signs**
```rust
/// Water has positive value → storage coefficients should be negative
/// ∂V/∂storage < 0 (more stored water → less expected future cost)
#[test]
fn property_storage_coefficients_negative() {
    let (mut sddp, saa) = create_benchmark_with_binding_storage();
    sddp.train(20, 10, &saa).unwrap();
    
    for node in sddp.graph().nodes() {
        for cut in node.fcf().cuts() {
            for (hydro_id, coef) in cut.storage_coefficients() {
                assert!(
                    coef <= 0.0,
                    "Storage coefficient for hydro {} should be <= 0, got {}",
                    hydro_id, coef
                );
            }
        }
    }
}
```

**Property 3: Dual Feasibility**
```rust
/// Complementary slackness: dual * slack ≈ 0
#[test]
fn property_dual_complementary_slackness() {
    let (mut sddp, saa) = create_simple_system();
    let trajectory = sddp.forward_pass(&saa.scenarios()[0]).unwrap();
    
    for realization in &trajectory {
        for (constraint_id, dual, slack) in realization.constraint_info() {
            let product = dual * slack;
            assert!(
                product.abs() < 1e-6,
                "Complementary slackness violated: dual={}, slack={}, product={}",
                dual, slack, product
            );
        }
    }
}
```

#### 5.2.2 Convergence Properties

**Property 4: Monotonic Lower Bound**
```rust
/// Fundamental SDDP property: lower bound never decreases
/// Proof: Each iteration adds cuts that improve or maintain approximation
#[test]
fn property_monotonic_lower_bound() {
    for benchmark in all_benchmarks() {
        let (mut sddp, saa) = benchmark();
        let result = sddp.train(50, 10, &saa).unwrap();
        
        let iterations = result.iterations();
        for i in 1..iterations.len() {
            let lb_prev = iterations[i-1].lower_bound;
            let lb_curr = iterations[i].lower_bound;
            
            assert!(
                lb_curr >= lb_prev - 1e-6,  // Small tolerance for LP solver noise
                "Lower bound decreased: iter {} = {:.6}, iter {} = {:.6}",
                i-1, lb_prev, i, lb_curr
            );
        }
    }
}
```

**Property 5: Upper Bound is Feasible**
```rust
/// Upper bound (simulation) gives feasible solution
#[test]
fn property_upper_bound_feasible() {
    let (mut sddp, saa) = create_stochastic_system();
    let result = sddp.train(30, 10, &saa).unwrap();
    
    // Simulate to get upper bound
    let sim_result = sddp.simulate(&saa, 100).unwrap();
    
    // All trajectories should be feasible
    for trajectory in sim_result.trajectories() {
        for realization in trajectory {
            assert!(
                realization.is_feasible(),
                "Simulation produced infeasible solution"
            );
        }
    }
}
```

**Property 6: Gap Closure (Stochastic)**
```rust
/// For stochastic problems: gap should decrease on average
#[test]
fn property_gap_decreases_on_average() {
    let (mut sddp, saa) = create_stochastic_system();
    let result = sddp.train(100, 20, &saa).unwrap();
    
    let iterations = result.iterations();
    
    // Compare first 20% to last 20%
    let early_avg = iterations[..20].iter()
        .map(|it| it.upper_bound - it.lower_bound)
        .sum::<f64>() / 20.0;
        
    let late_avg = iterations[80..].iter()
        .map(|it| it.upper_bound - it.lower_bound)
        .sum::<f64>() / 20.0;
    
    assert!(
        late_avg < early_avg,
        "Gap should decrease: early avg = {:.2}, late avg = {:.2}",
        early_avg, late_avg
    );
}
```

#### 5.2.3 AR Model Mathematical Properties

**Property 7: Chain Rule for Lag Coefficients**
```rust
/// Cut coefficient for lagged inflow Y_{t-j}:
/// ∂V/∂Y_{t-j} = (λ^hydro + λ^AR) * ψ_j
/// 
/// Where:
/// - λ^hydro: dual from hydro balance (direct impact)
/// - λ^AR: dual from AR constraint (indirect through temporal coupling)
/// - ψ_j: transformed AR coefficient (observation space)
#[test]
fn property_ar_lag_coefficient_chain_rule() {
    let (mut sddp, saa) = create_system_with_ar2_inflows();
    let trajectory = sddp.forward_pass(&saa.scenarios()[0]).unwrap();
    
    // Get cut from backward pass
    let cuts = sddp.backward_pass(&trajectory, &saa).unwrap();
    
    for cut in cuts {
        for (hydro_id, lag_idx) in cut.lag_indices() {
            let cut_coef = cut.lag_coefficient(hydro_id, lag_idx);
            
            // Recompute from duals
            let lambda_hydro = trajectory.water_value(hydro_id);
            let lambda_ar = trajectory.ar_dual(hydro_id);
            let psi = saa.temporal_model(hydro_id).psi(lag_idx);
            
            let expected_coef = (lambda_hydro + lambda_ar) * psi;
            
            assert_abs_diff_eq!(
                cut_coef,
                expected_coef,
                epsilon = 1e-5,
                "Chain rule violated for hydro {}, lag {}",
                hydro_id, lag_idx
            );
        }
    }
}
```

**Property 8: PAR to AR Transformation**
```rust
/// ψ_i = φ_i * (σ_t / σ_{t-i}) for PAR models
#[test]
fn property_par_psi_transformation() {
    let phi = vec![0.6, 0.3];
    let seasonal_std = vec![1.0, 1.5, 2.0]; // 3 periods
    let seasonal_mean = vec![100.0, 150.0, 200.0];
    
    let par = PeriodicAR::new(phi.clone(), seasonal_mean, seasonal_std.clone());
    
    for period in 0..3 {
        let psi = par.compute_psi(period);
        
        for (i, &phi_i) in phi.iter().enumerate() {
            let period_lag = (period + 3 - (i+1)) % 3;
            let sigma_t = seasonal_std[period];
            let sigma_t_minus = seasonal_std[period_lag];
            
            let expected_psi = phi_i * (sigma_t / sigma_t_minus);
            
            assert_abs_diff_eq!(
                psi[i],
                expected_psi,
                epsilon = 1e-10,
                "PAR transformation incorrect"
            );
        }
    }
}
```

#### 5.2.4 Risk Measure Properties

**Property 9: CVaR Bounds Expectation**
```rust
/// For costs: CVaR_α(X) >= E[X] for all α ∈ (0,1)
#[test]
fn property_cvar_bounds_expectation() {
    let costs = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let probs = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    
    let expectation = costs.iter()
        .zip(&probs)
        .map(|(c, p)| c * p)
        .sum::<f64>();
    
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let cvar = compute_cvar(&costs, &probs, alpha);
        
        assert!(
            cvar >= expectation - 1e-10,
            "CVaR({}) = {:.2} should be >= E[X] = {:.2}",
            alpha, cvar, expectation
        );
    }
}
```

**Property 10: Risk Measure Monotonicity**
```rust
/// If scenario A is worse than B in all states, ρ(A) >= ρ(B)
#[test]
fn property_risk_measure_monotonicity() {
    let costs_a = vec![10.0, 20.0, 30.0];
    let costs_b = vec![5.0, 15.0, 25.0];  // Strictly better
    let probs = vec![0.33, 0.34, 0.33];
    
    for risk in [Expectation, CVaR(0.25), WorstCase] {
        let risk_a = risk.evaluate(&costs_a, &probs);
        let risk_b = risk.evaluate(&costs_b, &probs);
        
        assert!(
            risk_a >= risk_b,
            "Risk monotonicity violated: ρ(A)={}, ρ(B)={}",
            risk_a, risk_b
        );
    }
}
```

### 5.3 Numerical Stability Tests

#### 5.3.1 Floating-Point Precision

**Test 1: Kahan Summation Accuracy**
```rust
#[test]
fn test_kahan_summation_vs_naive() {
    // Classic numerical analysis problem
    let values = vec![1e10, 1.0, -1e10, 1.0]; // Should sum to 2.0
    
    // Naive sum loses precision
    let naive_sum = values.iter().sum::<f64>();
    
    // Kahan sum maintains precision
    let kahan_sum = kahan_sum(&values);
    
    assert_abs_diff_eq!(kahan_sum, 2.0, epsilon = 1e-10);
    // naive_sum might be 0.0 or other wrong value
}
```

**Test 2: Cut Evaluation Precision**
```rust
#[test]
fn test_cut_evaluation_extreme_values() {
    // Large state values, small coefficients
    let state = vec![1e8; 100];
    let coefficients = vec![-1e-8; 100];
    let rhs = 1000.0;
    
    let cut = BendersCut { rhs, coefficients };
    
    // height = rhs - coef' * state
    //        = 1000 - (-1e-8) * 100 * 1e8
    //        = 1000 - (-100)
    //        = 1100
    
    let height = cut.eval_height(&state);
    
    assert_abs_diff_eq!(height, 1100.0, epsilon = 1e-3);
}
```

#### 5.3.2 LP Solver Tolerances

**Test 3: LP Solver Consistency**
```rust
#[test]
fn test_lp_solver_consistency() {
    // Solve same problem twice, should get same result
    let system = create_simple_system();
    let subproblem = Subproblem::new(/*...*/);
    
    let sol1 = subproblem.solve().unwrap();
    let sol2 = subproblem.solve().unwrap();
    
    assert_abs_diff_eq!(
        sol1.objective, 
        sol2.objective, 
        epsilon = 1e-6
    );
}
```

### 5.4 Test Utilities for Mathematical Validation

**File**: `tests/utils/assertions.rs`

```rust
/// Assert monotonicity of a sequence
pub fn assert_monotonic_non_decreasing(values: &[f64], tolerance: f64) {
    for i in 1..values.len() {
        assert!(
            values[i] >= values[i-1] - tolerance,
            "Non-monotonic at index {}: {} -> {}",
            i, values[i-1], values[i]
        );
    }
}

/// Assert cut validity at training state
pub fn assert_cut_validity(cut: &BendersCut, state: &[f64], objective: f64) {
    let height = cut.eval_height(state);
    assert_abs_diff_eq!(
        height,
        objective,
        epsilon = 1e-4,
        "Cut invalid at training state"
    );
}

/// Assert vector equality with element-wise tolerance
pub fn assert_vector_eq(a: &[f64], b: &[f64], epsilon: f64) {
    assert_eq!(a.len(), b.len(), "Vector lengths differ");
    for (i, (ai, bi)) in a.iter().zip(b).enumerate() {
        assert!(
            (ai - bi).abs() < epsilon,
            "Element {} differs: {} vs {}",
            i, ai, bi
        );
    }
}
```

### 5.5 Mathematical Test Priorities

**Phase 1: Fundamental Properties** (2 days)
- Cut validity (Property 1)
- Monotonic LB (Property 4)
- Storage coefficient signs (Property 2)

**Phase 2: AR Model Properties** (2 days)
- Chain rule (Property 7)
- PAR transformation (Property 8)

**Phase 3: Risk & Convergence** (1 day)
- CVaR properties (Properties 9-10)
- Gap closure (Property 6)

**Target**: 25 mathematical validation tests

---

## 6. Performance & Regression Tests

### 6.1 Benchmark Suite

**Purpose**: Track performance over time, prevent regressions

**Benchmark Problems**:

1. **Tiny** (baseline): 2 stages, 1 hydro, deterministic
   - Expected solve time: < 50ms
   - Use for: Overhead measurement

2. **Small**: 3 stages, 2 hydros, 10 scenarios
   - Expected solve time: < 500ms
   - Use for: Quick regression checks

3. **Medium**: 12 stages, 5 hydros, 50 scenarios, PAR(1)
   - Expected solve time: < 10s
   - Use for: Realistic performance

4. **Large**: 120 stages, 20 hydros, 100 scenarios, PAR(2)
   - Expected solve time: < 5 minutes
   - Use for: Scalability testing

**Implementation**:
```rust
// benches/sddp_benchmarks.rs (using criterion)

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_tiny_deterministic(c: &mut Criterion) {
    let (mut sddp, saa) = create_tiny_deterministic();
    
    c.bench_function("tiny_deterministic", |b| {
        b.iter(|| {
            let mut sddp_clone = sddp.clone();
            sddp_clone.train(black_box(10), black_box(1), &saa).unwrap()
        });
    });
}

fn benchmark_small_stochastic(c: &mut Criterion) {
    let (mut sddp, saa) = create_small_stochastic();
    
    c.bench_function("small_stochastic", |b| {
        b.iter(|| {
            let mut sddp_clone = sddp.clone();
            sddp_clone.train(black_box(20), black_box(5), &saa).unwrap()
        });
    });
}

criterion_group!(
    benches,
    benchmark_tiny_deterministic,
    benchmark_small_stochastic,
    benchmark_medium_par1,
    benchmark_large_par2
);
criterion_main!(benches);
```

### 6.2 Regression Test Suite

**File**: `tests/regression/convergence_baselines.rs`

```rust
/// Regression tests that check algorithm convergence hasn't degraded
/// 
/// These tests use fixed seeds and check that:
/// 1. Final lower bound is within range of historical values
/// 2. Convergence speed (iterations to 5% gap) hasn't increased
/// 3. Solution quality (simulation cost) is comparable

#[test]
fn regression_deterministic_2stage() {
    let (mut sddp, saa) = create_deterministic_2stage_fixed_seed();
    let result = sddp.train(20, 1, &saa).unwrap();
    
    // Historical baseline (from previous runs)
    const EXPECTED_LB: f64 = 523.45;
    const TOLERANCE: f64 = 10.0;
    
    let final_lb = result.final_lower_bound();
    
    assert!(
        (final_lb - EXPECTED_LB).abs() < TOLERANCE,
        "Regression detected: LB = {:.2}, expected {:.2} ± {:.2}",
        final_lb, EXPECTED_LB, TOLERANCE
    );
}

#[test]
fn regression_stochastic_convergence_speed() {
    let (mut sddp, saa) = create_stochastic_3stage_fixed_seed();
    let result = sddp.train(100, 10, &saa).unwrap();
    
    // Historical: converges to 5% gap in ~35 iterations
    const EXPECTED_ITERS_TO_5PCT: usize = 35;
    const TOLERANCE: usize = 10;
    
    let iterations = result.iterations();
    let iters_to_5pct = iterations.iter()
        .position(|it| {
            let gap = it.upper_bound - it.lower_bound;
            let rel_gap = gap / it.lower_bound.abs();
            rel_gap < 0.05
        })
        .unwrap_or(iterations.len());
    
    assert!(
        iters_to_5pct <= EXPECTED_ITERS_TO_5PCT + TOLERANCE,
        "Convergence slower than baseline: {} iters vs {} expected",
        iters_to_5pct, EXPECTED_ITERS_TO_5PCT
    );
}

#[test]
fn regression_par2_solution_quality() {
    let (mut sddp, saa) = create_par2_system_fixed_seed();
    let result = sddp.train(50, 20, &saa).unwrap();
    
    // Historical lower bound
    const EXPECTED_LB: f64 = 1245.67;
    const TOLERANCE: f64 = 50.0;
    
    let final_lb = result.final_lower_bound();
    
    assert!(
        final_lb >= EXPECTED_LB - TOLERANCE,
        "Solution quality degraded: LB = {:.2}, expected >= {:.2}",
        final_lb, EXPECTED_LB - TOLERANCE
    );
}
```

### 6.3 Memory Profiling Tests

**Purpose**: Ensure memory usage doesn't grow unexpectedly

```rust
#[test]
fn test_memory_usage_large_state() {
    // 50 hydros with AR(2) → 150-dimensional state
    let (mut sddp, saa) = create_system_with_50_hydros_ar2();
    
    let memory_before = get_memory_usage();
    
    sddp.train(10, 5, &saa).unwrap();
    
    let memory_after = get_memory_usage();
    let memory_used_mb = (memory_after - memory_before) / 1_000_000;
    
    // Should fit in reasonable memory (< 500 MB for this problem)
    assert!(
        memory_used_mb < 500,
        "Memory usage too high: {} MB",
        memory_used_mb
    );
}

#[test]
fn test_cut_pool_memory_growth() {
    // Verify cut pool doesn't grow unbounded
    let (mut sddp, saa) = create_medium_system();
    
    sddp.train(100, 10, &saa).unwrap();
    
    let total_cuts: usize = sddp.graph()
        .nodes()
        .map(|n| n.fcf().cut_count())
        .sum();
    
    // With cut selection, should be < 10,000 cuts for this problem
    assert!(
        total_cuts < 10_000,
        "Cut pool too large: {} cuts",
        total_cuts
    );
}
```

### 6.4 Continuous Integration Tests

**Fast CI Suite** (runs on every commit):
- Unit tests (2s)
- Fast integration tests (10s)
- Basic convergence test (5s)
- **Total: ~20s**

**Full CI Suite** (runs on PR):
- All unit tests
- All integration tests
- Mathematical validation tests
- Regression tests
- **Total: ~2 minutes**

**Nightly Tests**:
- Full benchmark suite
- Large-scale tests
- Memory profiling
- **Total: ~30 minutes**

**CI Configuration** (`.github/workflows/test.yml`):
```yaml
name: Tests

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
      - name: Run fast tests
        run: cargo test --lib --tests -- --test-threads=4
        timeout-minutes: 2
  
  full-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
      - name: Run full test suite
        run: cargo test --all-features
        timeout-minutes: 5
```

---

## 7. Test Fixtures & Data

### 7.1 Fixture Design Principles

1. **Simplicity**: Start with minimal complexity, add incrementally
2. **Predictability**: Use fixed seeds, known solutions
3. **Reusability**: Builder pattern for common variations
4. **Documentation**: Each fixture explains what it tests

### 7.2 Fixture Hierarchy

```
tests/fixtures/
├── mod.rs                    # Re-exports
├── minimal.rs                # Simplest possible systems
├── canonical.rs              # Standard benchmark problems
├── edge_cases.rs             # Boundary conditions
├── builders/
│   ├── mod.rs
│   ├── system_builder.rs     # Fluent API for systems
│   ├── sddp_builder.rs       # Fluent API for SDDP
│   └── scenario_builder.rs   # Fluent API for scenarios
└── data/
    ├── known_solutions.json  # Problems with analytical solutions
    └── regression_baselines.json  # Historical performance data
```

### 7.3 Minimal Fixtures

**`tests/fixtures/minimal.rs`**:

```rust
/// Absolute minimum: 1 stage, 1 hydro, no uncertainty
pub fn one_stage_one_hydro() -> System {
    System {
        buses: vec![Bus {
            id: 0,
            deficit_cost: 1000.0,
            hydro_ids: vec![0],
            thermal_ids: vec![],
            load: 0.0,
            source_line_ids: vec![],
            target_line_ids: vec![],
        }],
        hydros: vec![Hydro {
            id: 0,
            bus_id: 0,
            min_storage: 0.0,
            max_storage: 100.0,
            productivity: 1.0,
            // ... all fields
        }],
        thermals: vec![],
        lines: vec![],
    }
}

/// Minimal with uncertainty: 2 stages, 1 hydro, 2 inflow scenarios
pub fn two_stage_two_scenarios() -> (System, SAA) {
    let system = /* ... */;
    let saa = SAA::new(vec![
        Scenario { inflows: vec![10.0, 15.0], probability: 0.5 },
        Scenario { inflows: vec![20.0, 25.0], probability: 0.5 },
    ]);
    (system, saa)
}

/// Minimal with AR: 2 stages, 1 hydro, AR(1) inflows
pub fn two_stage_ar1() -> (System, TemporalModel, SAA) {
    let system = /* ... */;
    let temporal = PeriodicAR::new(vec![0.6], /*...*/);
    let saa = /* ... */;
    (system, temporal, saa)
}
```

### 7.4 Canonical Benchmarks

**`tests/fixtures/canonical.rs`**:

```rust
/// Three-Stage Hydrothermal (Pereira & Pinto 1991 style)
/// 
/// **Problem Structure**:
/// - 3 stages (12 months)
/// - 2 reservoirs in cascade
/// - 1 thermal plant
/// - Stochastic inflows (10 scenarios)
/// 
/// **Known Properties**:
/// - Converges in < 30 iterations
/// - Final gap < 5%
/// - Lower bound ≈ $1,200
pub fn three_stage_cascade() -> (SddpAlgorithm, SAA) {
    SddpAlgorithm::builder()
        .system_factory(|| create_cascade_system())
        .num_stages(3)
        .scenario_tree(create_10_scenario_tree())
        .seed(42)
        .build_with_saa()
        .unwrap()
}

/// Known Analytical Solution: Deterministic Linear Reservoir
/// 
/// Single reservoir, linear cost, deterministic inflows.
/// Analytical solution: Dual variables = marginal costs
pub fn deterministic_linear() -> (SddpAlgorithm, SAA, f64) {
    let analytical_cost = 523.45;  // Computed by hand
    let (sddp, saa) = /* ... */;
    (sddp, saa, analytical_cost)
}
```

### 7.5 Edge Case Fixtures

**`tests/fixtures/edge_cases.rs`**:

```rust
/// Empty reservoir: Start with zero storage
pub fn empty_reservoir() -> System;

/// Full reservoir: Start at max storage
pub fn full_reservoir() -> System;

/// Infeasible problem: Demand > Generation capacity + Deficit
pub fn infeasible_system() -> System;

/// Zero inflow: Test with no water arriving
pub fn zero_inflow() -> SAA;

/// Extreme AR coefficient: φ = 0.99 (nearly non-stationary)
pub fn extreme_ar_coefficient() -> TemporalModel;

/// Heterogeneous AR orders: [AR(0), AR(1), AR(2), AR(3)]
pub fn heterogeneous_ar_orders() -> (System, Vec<TemporalModel>);
```

### 7.6 Builder Fixtures

**`tests/fixtures/builders/system_builder.rs`**:

```rust
pub struct SystemBuilder {
    n_buses: usize,
    n_hydros: usize,
    n_thermals: usize,
    cascade: bool,
    transmission: bool,
}

impl SystemBuilder {
    pub fn new() -> Self {
        Self {
            n_buses: 1,
            n_hydros: 1,
            n_thermals: 0,
            cascade: false,
            transmission: false,
        }
    }
    
    pub fn with_cascade(mut self, n_hydros: usize) -> Self {
        self.n_hydros = n_hydros;
        self.cascade = true;
        self
    }
    
    pub fn with_transmission(mut self) -> Self {
        self.n_buses = 2;
        self.transmission = true;
        self
    }
    
    pub fn build(self) -> System {
        // Construct system based on settings
        unimplemented!()
    }
}

// Usage:
let system = SystemBuilder::new()
    .with_cascade(3)
    .with_transmission()
    .build();
```

### 7.7 Fixture Update Plan

**Phase 1: Fix Broken Fixtures** (2 days)
- Update all fixtures to current API
- Ensure all compile and run
- Document any API changes

**Phase 2: Add Missing Fixtures** (2 days)
- Heterogeneous AR orders
- Edge cases
- Known solutions

**Phase 3: Create Builders** (2 days)
- `SystemBuilder`
- `SddpBuilder` wrapper
- `ScenarioBuilder`

**Phase 4: Document** (1 day)
- Add comments to all fixtures
- Create fixture guide
- Include usage examples

---

## 8. Execution Plan

### 8.1 Overview

This plan provides a phased approach to rebuild the testing infrastructure over ~4-6 weeks. Each phase has clear deliverables and can be executed incrementally.

### 8.2 Phase 1: Foundation (Week 1-2)

**Goal**: Fix compilation, establish working baseline

#### Tasks

**1.1 Fix Broken Fixtures** (3 days)
- [ ] Update `tests/fixtures/systems.rs` to current API
  - Fix `Hydro` field names (`min_storage` vs `min_volume`)
  - Add missing `Bus` fields (`hydro_ids`, `source_line_ids`, etc.)
  - Update `System` construction
- [ ] Update `tests/fixtures/benchmarks.rs`
  - Fix `train()` method calls (correct signature)
  - Update `Config` struct initialization
- [ ] Fix `tests/fixtures/simple_2stage_reservoir.rs`
- [ ] Verify all fixtures compile and run

**1.2 Critical Unit Tests** (4 days)
- [ ] `cut.rs`: Add 12 fundamental tests
  - Cut validity at training state
  - Coefficient signs
  - Evaluation precision
  - Domination detection
- [ ] `risk_measure.rs`: Add 8 tests
  - CVaR bounds expectation
  - Probability adjustment
  - Monotonicity
- [ ] `system.rs`: Add 15 validation tests
  - ID uniqueness
  - Reference validity
  - Topology checks

**1.3 Test Utilities** (2 days)
- [ ] Create `tests/utils/assertions.rs`
  - `assert_monotonic_non_decreasing()`
  - `assert_cut_validity()`
  - `assert_vector_eq()`
- [ ] Create `tests/utils/validation.rs`
  - Result validation helpers
  - Convergence checkers

**Deliverables**:
- ✅ All fixtures compile
- ✅ 35 new critical unit tests passing
- ✅ Test utility library established
- ✅ Baseline test suite runs in < 30s

### 8.3 Phase 2: Mathematical Validation (Week 3)

**Goal**: Establish algorithm correctness tests

#### Tasks

**2.1 Cut Properties** (2 days)
- [ ] Property 1: Cut validity at training state
- [ ] Property 2: Storage coefficient signs
- [ ] Property 3: Dual feasibility
- [ ] Test on all benchmarks

**2.2 Convergence Properties** (2 days)
- [ ] Property 4: Monotonic lower bound
- [ ] Property 5: Upper bound feasibility
- [ ] Property 6: Gap closure
- [ ] Create convergence assertion helpers

**2.3 AR Model Properties** (2 days)
- [ ] Property 7: Chain rule for lags
- [ ] Property 8: PAR to AR transformation
- [ ] Test heterogeneous orders

**2.4 Risk Properties** (1 day)
- [ ] Property 9: CVaR bounds
- [ ] Property 10: Monotonicity

**Deliverables**:
- ✅ 25 mathematical validation tests
- ✅ Property-based test framework
- ✅ Validates SDDP theory holds

### 8.4 Phase 3: Integration Tests (Week 4-5)

**Goal**: Rebuild integration test suite

#### Tasks

**3.1 Algorithm Integration** (4 days)
- [ ] Create `tests/integration/algorithm/` directory
- [ ] Forward pass tests (4 tests)
  - State initialization
  - Uncertainty realization
  - Cost accumulation
  - Trajectory storage
- [ ] Backward pass tests (4 tests)
  - Scenario branching
  - Cut generation
  - Dual extraction
  - FCF update
- [ ] Training loop tests (3 tests)
- [ ] Convergence tests (3 tests)

**3.2 AR Model Integration** (3 days)
- [ ] Create `tests/integration/ar_models/` directory
- [ ] PAR cut generation tests (4 tests)
- [ ] Heterogeneous orders tests (3 tests)
- [ ] Mixed entities tests (3 tests)

**3.3 System Features** (2 days)
- [ ] Create `tests/integration/system/` directory
- [ ] Cascade hydros tests (3 tests)
- [ ] Transmission tests (2 tests)
- [ ] Thermal dispatch tests (2 tests)

**3.4 Risk & Scenarios** (2 days)
- [ ] Create `tests/integration/risk_measures/` directory
- [ ] CVaR tests (3 tests)
- [ ] Expectation baseline tests (2 tests)
- [ ] Scenario generation tests (3 tests)

**Deliverables**:
- ✅ 40 new integration tests
- ✅ Organized test directory structure
- ✅ All major features covered

### 8.5 Phase 4: End-to-End & Performance (Week 6)

**Goal**: Complete test pyramid

#### Tasks

**4.1 End-to-End Tests** (2 days)
- [ ] Create `tests/e2e/` directory
- [ ] Deterministic 2-stage test (analytical solution)
- [ ] Stochastic 3-stage test
- [ ] PAR model E2E test
- [ ] Multi-reservoir cascade test

**4.2 Benchmark Suite** (2 days)
- [ ] Setup Criterion benchmarks
- [ ] Create 4 benchmark problems (tiny, small, medium, large)
- [ ] Establish baseline performance metrics
- [ ] Document expected performance

**4.3 Regression Tests** (2 days)
- [ ] Create `tests/regression/` directory
- [ ] Convergence baseline tests (3 tests)
- [ ] Solution quality tests (2 tests)
- [ ] Memory usage tests (2 tests)

**4.4 CI/CD Setup** (1 day)
- [ ] Configure GitHub Actions
- [ ] Fast CI suite (20s)
- [ ] Full CI suite (2min)
- [ ] Nightly benchmarks

**Deliverables**:
- ✅ 10 E2E tests
- ✅ Benchmark suite running
- ✅ Regression tests established
- ✅ CI/CD pipeline operational

### 8.6 Phase 5: Documentation & Polish (Ongoing)

**Goal**: Ensure maintainability

#### Tasks

**5.1 Test Documentation** (2 days)
- [ ] Add module-level test strategy comments
- [ ] Document fixture usage patterns
- [ ] Create testing guide for contributors
- [ ] Update TESTING_STRATEGY.md with actuals

**5.2 Fixture Enhancement** (2 days)
- [ ] Create builder fixtures
- [ ] Add canonical benchmarks
- [ ] Document known solutions
- [ ] Create edge case fixtures

**5.3 Refactoring** (1 day)
- [ ] Consolidate duplicate test utilities
- [ ] Extract common test patterns
- [ ] Optimize slow tests

**Deliverables**:
- ✅ Comprehensive test documentation
- ✅ Easy-to-use test fixtures
- ✅ Contributor testing guide

### 8.7 Success Metrics

**Quantitative Targets**:
- Unit tests: 450+ (up from 357)
- Integration tests: 60+ (replacing broken 45)
- E2E tests: 10+
- Mathematical validation: 25+
- Total suite execution: < 3 minutes
- Fast subset: < 30 seconds

**Qualitative Goals**:
- ✅ All tests compile and pass
- ✅ Mathematical properties validated
- ✅ SDDP algorithm correctness proven
- ✅ No broken fixtures
- ✅ Clear test organization
- ✅ Easy to add new tests
- ✅ CI/CD fully operational

### 8.8 Risk Mitigation

**Risk 1: API Changes During Test Rebuild**
- **Mitigation**: Use builder pattern for fixtures to isolate API changes
- **Strategy**: Update builders, not individual tests

**Risk 2: Long Test Execution Times**
- **Mitigation**: Implement test parallelization
- **Strategy**: Profile and optimize slowest tests

**Risk 3: Flaky Integration Tests**
- **Mitigation**: Use fixed seeds for all stochastic elements
- **Strategy**: Document sources of randomness

**Risk 4: Fixture Maintenance Burden**
- **Mitigation**: Create builder-based fixtures
- **Strategy**: Minimize hard-coded test data

### 8.9 Maintenance Strategy

**Weekly**:
- Run full test suite locally before merging
- Review new test additions for quality
- Check test execution time trends

**Monthly**:
- Update regression baselines if algorithm improves
- Review and prune obsolete tests
- Refactor test utilities

**Per Release**:
- Validate all benchmarks against known solutions
- Update performance baselines
- Review test coverage reports

### 8.10 Priority Matrix

```
                    High Impact
                        ↑
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    │   Phase 1         │   Phase 2         │
    │   Foundation      │   Math Validation │
L   │   DO FIRST        │   DO SECOND       │
o   ├───────────────────┼───────────────────┤
w   │   Phase 5         │   Phase 3         │
    │   Documentation   │   Integration     │
E   │   DO LAST         │   DO THIRD        │
f   │                   │                   │
f   └───────────────────┴───────────────────┘
o                Low Urgency →  High Urgency
r
t
```

**Execution Order**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

### 8.11 Next Steps

**Immediate Actions** (This Week):
1. Fix `tests/fixtures/systems.rs` to compile
2. Fix `tests/fixtures/benchmarks.rs` to compile
3. Add 5 critical tests to `cut.rs`
4. Add 3 monotonic LB tests to `sddp/mod.rs`

**Quick Win Test** (To Validate Strategy):
```rust
// tests/validation_test.rs
#[test]
fn quick_win_monotonic_lower_bound() {
    let (mut sddp, saa) = create_simple_2stage_system();
    let result = sddp.train(20, 5, &saa).unwrap();
    
    assert_monotonic_non_decreasing(
        &result.iterations().map(|it| it.lower_bound).collect::<Vec<_>>(),
        1e-6
    );
}
```

**First Week Goal**: Get this test passing and establish pattern for others.

---

## Appendix A: Test Organization Reference

**Directory Structure**:
```
tests/
├── integration/
│   ├── algorithm/
│   ├── ar_models/
│   ├── risk_measures/
│   ├── scenarios/
│   └── system/
├── e2e/
├── regression/
├── fixtures/
│   ├── builders/
│   └── data/
└── utils/
```

**Naming Conventions**:
- Unit tests: `test_function_name()` in source files
- Integration tests: `test_feature_integration()`
- Properties: `property_mathematical_invariant()`
- Benchmarks: `benchmark_problem_size()`
- Regression: `regression_known_baseline()`

---

## Appendix B: Key Test Patterns

**Pattern 1: Mathematical Property Test**
```rust
#[test]
fn property_name() {
    for benchmark in all_benchmarks() {
        let (sddp, saa) = benchmark();
        // ... test property holds
    }
}
```

**Pattern 2: Integration Test**
```rust
#[test]
fn test_component_interaction() {
    let system = create_test_system();
    // Setup
    // Execute
    // Validate interaction
}
```

**Pattern 3: Regression Test**
```rust
#[test]
fn regression_known_baseline() {
    const BASELINE: f64 = 123.45;
    const TOLERANCE: f64 = 1.0;
    // ... compare to baseline
}
```

---

## Summary

This testing strategy provides:
1. **Clear organization** for 540+ tests across the pyramid
2. **Phased execution plan** deliverable in 4-6 weeks
3. **Mathematical validation** of SDDP algorithm correctness
4. **Maintainable fixtures** using builder patterns
5. **CI/CD integration** with fast feedback loops

The strategy balances **thoroughness** (covering all SDDP properties) with **pragmatism** (phased, incremental delivery). It ensures your SDDP implementation is not just correct code, but correct optimization algorithms.

**Start with Phase 1** to fix the foundation, then build upward through the testing pyramid.

