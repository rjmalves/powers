# Remaining AR Model Tickets - Quick Reference

**Updated 2025-10-13**: Added Sprint 1.5 (Scenario Generation Pipeline) with 7 new tickets (AR-6.1 through AR-6.7) to address input schema conflicts and implement CEPEL-compliant 4-stage pipeline.

## Sprint 1.5: Scenario Generation Pipeline (AR-6.1 to AR-6.7) - 20 days

### AR-6.1: Input Schema Refactor (3 days) - P0 CRITICAL

**Objective**: Refactor input schema to separate marginal, innovation, and temporal specifications
**Key Tasks**:

- Remove ambiguous `distribution` field from NoiseModel
- Add `marginal_distribution`, `innovation_distribution`, `temporal_model` fields
- Integrate LogNormal3 into marginal (eliminate NonNegativityMethod)
- Backward compatibility with schema version field
- Update JSON schemas

**Acceptance Criteria**:

- Clear separation of marginal vs innovation distributions
- LogNormal3 parameters appear only once
- Schema supports CEPEL pipeline
- Backward compatibility maintained
- All tests passing

---

### AR-6.2: Base Noise Generator (2 days) - P0 CRITICAL

**Objective**: Generate independent standard normal samples Z ~ N(0,1)
**Key Tasks**:

- Implement Standard method (direct sampling)
- Stub k-means, QMC, LHS methods (future)
- Validation (num_scenarios > 0, num_entities > 0)
- Statistical testing (mean ≈ 0, std ≈ 1)

**Acceptance Criteria**:

- Generate 1000 scenarios × 10 entities in <10ms
- Same seed → identical samples
- Statistical properties verified (χ² test)
- Output: Vec<Vec<f64>> [scenario][entity]

---

### AR-6.3: Correlation Application (2 days) - P0 CRITICAL

**Objective**: Apply Cholesky decomposition to introduce correlation: W = L×Z
**Key Tasks**:

- Reuse CholeskyFactor from AR-5.6
- Apply per-season correlation blocks
- Preserve N(0,1) marginals
- Handle uncorrelated entities (skip transformation)

**Acceptance Criteria**:

- Correlation structure matches input (ρ within 0.03)
- Marginal properties preserved (mean=0, std=1)
- Transform 1000 scenarios × 10 entities in <20ms
- Statistical validation (Fisher z-transform)

---

### AR-6.4: Marginal Transformation (2 days) - P0 CRITICAL

**Objective**: Transform correlated normals to target marginal distributions
**Key Tasks**:

- Normal transformation: X = μ + σW
- LogNormal3 transformation: X = γ + exp(μ + σW)
- Handle overflow (clamp exponent)
- Preserve correlation (Gaussian copula)

**Acceptance Criteria**:

- Marginal moments match theoretical (within 95% CI)
- LogNormal3 always non-negative
- Correlation preserved (Spearman for LogNormal3)
- Transform 1000 scenarios × 10 entities in <15ms

---

### AR-6.5: AR Temporal Dynamics (3 days) - P0 CRITICAL

**Objective**: Apply autoregressive dynamics: Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
**Key Tasks**:

- Implement AR(1) through AR(p)
- Lag buffer management (shift and update)
- Non-negativity enforcement (clamp to 0)
- Statistical validation (ACF tests)

**Acceptance Criteria**:

- AR recurrence relation correct
- Sample ACF matches theoretical (within 0.08)
- Lag updates correct for multi-stage
- Apply 1000 scenarios × 10 entities in <25ms

---

### AR-6.6: Scenario Pipeline Integration (5 days) - P0 CRITICAL

**Objective**: Wire up 4-stage pipeline into ScenarioGenerator
**Key Tasks**:

- Orchestrate: base noise → correlation → marginal → AR
- Convert to SAA format (backward compatible)
- Factory method from RecourseInput
- Deprecate old NoiseGenerator
- Multi-stage lag propagation

**Acceptance Criteria**:

- Full pipeline working end-to-end
- SAA format compatible with existing SDDP
- Examples 01-04 produce identical results (seed-controlled)
- Generate 1000 scenarios × 10 entities × 12 stages in <200ms
- Backward compatibility maintained

---

### AR-6.7: Scenario Validation & Benchmarking (3 days) - P1 HIGH

**Objective**: Statistical validation and performance benchmarking
**Key Tasks**:

- Statistical tests: marginals, correlation, AR autocorrelation
- Hypothesis testing (chi-square, K-S, confidence intervals)
- Performance benchmarks (baseline vs full pipeline)
- Regression tests (existing examples unchanged)

**Acceptance Criteria**:

- All statistical tests pass (95% CI)
- Performance meets targets (<200ms for 12 stages)
- Regression tests confirm backward compatibility
- Validation methodology documented

---

## Sprint 2: State & Process (AR-7 to AR-11) - 10 days

### AR-7: StorageWithInflowState Implementation (3 days) - P0 CRITICAL

---

## Sprint 3: Algorithm Integration (AR-12 to AR-17) - 10 days

### AR-12: Subproblem AR Constraints (3 days) - P0 CRITICAL

### AR-8: State Trait Refactoring (2 days) - P0 CRITICAL

**Objective**: Handle concatenated coefficients for extended state
**Key Tasks**:

- Ensure State::coefficients() works for extended state
- Cut evaluation with extended state
- Subproblem construction with extended state
- Backward compatibility for StorageState

**Acceptance Criteria**:

- State trait supports extended coefficients
- Cut evaluation unchanged (uses coefficients())
- Subproblem accepts any State implementation
- Tests with both state types

---

### AR-9: AR Stochastic Process Implementation (2 days) - P0 CRITICAL

**Objective**: AutoRegressive struct with conditional sampling
**Key Tasks**:

- AutoRegressive struct (coefficients, innovation_dist, lag_order)
- Implement StochasticProcess trait (sample_conditional)
- Conditional sampling: sample(ε) given lags
- Integration with process factory

**Acceptance Criteria**:

- AR process samples correctly
- Conditional on lag state
- Innovation distribution configurable
- Tests with AR(1), AR(2), AR(3)

---

### AR-10: Pre-study Nodes for Lag Initialization (2 days) - P1 HIGH

**Objective**: Multiple historical nodes before decision stages
**Key Tasks**:

- Extend graph for pre-study nodes
- Deterministic walk through pre-study
- Initialize lags from historical data
- Validation (p pre-study nodes for AR(p))

**Acceptance Criteria**:

- Pre-study nodes in graph
- Lags initialized from history
- Deterministic transitions work
- Integration tests

---

### AR-11: State Transition with Lag Update (1 day) - P0 CRITICAL

**Objective**: Shift and update lags during state transitions
**Key Tasks**:

- Implement lag shift: [Xₜ, Xₜ₋₁, ..., Xₜ₋ₚ₊₁]
- State transition updates lags
- Handle multiple resources (per-hydro lags)
- Validation (lag buffer size constant)

**Acceptance Criteria**:

- Lag shift correct
- State transitions update lags
- Multi-resource support
- Tests verify lag buffer

---

## Sprint 3: Algorithm Integration (AR-12 to AR-17) - 10 days

**Objective**: Extend subproblem LP to include lag variables and AR equation constraints
**Key Tasks**:

- Add lag variables to LP (w[r][lag])
- Add AR constraint: w[r][0] = phi1 \* w[r][1] + ... + epsilon
- Extract duals for both storage and lags
- Update objective function with lag coefficients

**Acceptance Criteria**:

- Lag variables created correctly
- AR equation enforced as constraint
- Dual extraction works for extended state
- Tests with AR(1) and AR(2) subproblems

---

### AR-13: Dual Variable Extraction (2 days) - P0 CRITICAL

**Objective**: Extract dual variables for both storage and lag variables to build cut coefficients
**Key Tasks**:

- Query solver for storage duals (existing)
- Query solver for lag duals (new)
- Concatenate in correct order matching state.coefficients()
- Validation: dual dimensions match state dimensions

**Acceptance Criteria**:

- Dual extraction for extended state works
- Duals match state coefficient order
- Solver interface unchanged for storage-only

---

### AR-14: Cut Generation with Extended State (2 days) - P0 CRITICAL

**Objective**: Generate Benders cuts with coefficients for both storage and lag variables
**Key Tasks**:

- Extend BendersCut to handle extended coefficients
- Cut intercept computation unchanged
- Cut evaluation uses full state vector
- FCF stores cuts with correct dimensions

**Acceptance Criteria**:

- Cuts generated with R + Σp coefficients
- Cut evaluation correct for extended state
- Backward compatibility with storage-only cuts

---

### AR-15: Forward Pass AR Integration (2 days) - P0 CRITICAL

**Objective**: Integrate AR sampling and state transitions into SDDP forward pass
**Key Tasks**:

- Use conditional sampling when state has lags
- State transition with lag updates
- Handle pre-study nodes (deterministic walk)
- Policy evaluation with extended state

**Acceptance Criteria**:

- Forward pass works with AR processes
- Lag states propagate correctly through stages
- Pre-study nodes processed correctly
- Policy values computed correctly

---

### AR-16: ~~Scenario Generation for AR (2 days)~~ - **REMOVED**

**Status**: **SUPERSEDED by Sprint 1.5 (AR-6.1 through AR-6.7)**

This ticket has been replaced by a comprehensive 7-ticket scenario generation pipeline refactor that addresses input schema conflicts and implements CEPEL-compliant 4-stage generation. See Sprint 1.5 above.

---

### AR-17: State Factory Integration (1 day) - P1 HIGH

**Objective**: Wire up state factory to create appropriate state type based on noise models
**Key Tasks**:

- Detect AR models in recourse
- Create StorageWithInflowState if AR present
- Create StorageState otherwise (backward compat)
- Factory pattern for state construction

**Acceptance Criteria**:

- Automatic state type selection
- No manual type specification needed
- Backward compatibility maintained

---

## Sprint 4: Validation & Production (AR-18 to AR-21) - 8 days

### AR-18: AR(1) Validation Test (2 days) - P0 CRITICAL

**Objective**: End-to-end test with known AR(1) problem, validate correctness
**Key Tasks**:

- Create simple AR(1) test problem
- Known analytical solution or SDDP.jl comparison
- Verify policy values, bounds, autocorrelation
- Integration test with full SDDP run

**Acceptance Criteria**:

- Policy value within 1% of expected
- Lower bound converges correctly
- Sample autocorrelation matches theoretical
- All SDDP components working together

---

### AR-19: PAR Support Implementation (2 days) - P1 HIGH

**Objective**: Add periodic AR support for seasonal coefficient variation
**Key Tasks**:

- Extend AutoRegressive with season_id → coefficients map
- Seasonal coefficient selection in sampling
- Validation for seasonal stationarity
- Example with monthly seasonal patterns

**Acceptance Criteria**:

- PAR processes sample correctly
- Coefficients vary by season
- Stationarity checked per season
- Tests with 2-season and 12-season PAR

---

### AR-20: Performance Benchmarking (2 days) - P2 MEDIUM

**Objective**: Measure AR overhead, validate performance expectations
**Key Tasks**:

- Memory profiling (storage-only vs AR(1) vs AR(2))
- Speed benchmarks (iteration time, convergence)
- Scaling tests (R, p, T)
- Comparison with baseline

**Acceptance Criteria**:

- Memory overhead < 2× for AR(1)
- Computation overhead < 10%
- Convergence improvement 20-30% on correlated problems
- Performance report document

---

### AR-21: Documentation and Examples (2 days) - P1 HIGH

**Objective**: Complete user-facing documentation and example problems
**Key Tasks**:

- User guide: How to use AR models
- API documentation for new types
- Example problem with AR(1)
- Migration guide from independent noise

**Acceptance Criteria**:

- Complete user guide (AR-MODEL-GUIDE.md)
- Example 06-ar-hydro/ with documented AR setup
- API docs for all public AR types
- Migration guide covers common scenarios

---

## Ticket Priority Legend

- **P0 CRITICAL**: Blocks core functionality, must complete
- **P1 HIGH**: Important for production use, should complete
- **P2 MEDIUM**: Nice-to-have, can defer if needed

## Dependencies Summary

### Sprint 1.5 Critical Path (NEW)

AR-6.1 (Schema) → AR-6.2 (Base Noise) → AR-6.3 (Correlation) → AR-6.4 (Marginal) → AR-6.5 (AR Dynamics) → AR-6.6 (Integration) → AR-6.7 (Validation)

**This sprint MUST complete before Sprint 2.** The refactored scenario generation pipeline addresses fundamental input schema conflicts and ensures proper CEPEL-compliant implementation.

### Sprint 2 Dependencies

AR-6.7 (Scenario pipeline complete) → AR-7, AR-8, AR-9, AR-10, AR-11 (can proceed in parallel after AR-6.7)

### Sprint 3 Critical Path

AR-12 (Subproblem) → AR-13 (Duals) → AR-14 (Cuts) → AR-15 (Forward Pass)

These 4 tickets are the **critical path** for AR integration. AR-17 can proceed in parallel after AR-7/AR-8 complete.

### Sprint 4 Validation

AR-18 (Validation) depends on all Sprint 3 tickets. AR-19, AR-20, AR-21 can proceed after AR-18.

---

## Estimated Total Effort

- Sprint 1 (Foundation): 8 days ✅ **COMPLETE**
- **Sprint 1.5 (Scenario Pipeline): 20 days ⏳ NOT STARTED** (NEW)
- Sprint 2 (State & Process): 10 days ⏳ NOT STARTED
- Sprint 3 (Integration): 10 days ⏳ NOT STARTED (reduced from 12, AR-16 removed)
- Sprint 4 (Validation): 8 days ⏳ NOT STARTED

**Total**: 56 days (~11 weeks with reviews, buffer)  
**Previous Total**: 38 days (8 weeks)  
**Increase**: +18 days (+47%) due to comprehensive scenario generation refactor

**Rationale for Increase**: The additional time investment prevents technical debt and ensures production-ready implementation following CEPEL best practices. The alternative (proceeding with conflicting input schema) would require substantial refactoring later with higher risk of breaking changes.

---

**Created**: 2025-01-10  
**Updated**: 2025-10-13 (Added Sprint 1.5, updated dependencies, revised effort estimates)  
**Status**: Working document for ticket creation
