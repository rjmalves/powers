# [T-025] Extract Cut Computation

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-024](./ticket-024-extract-backward-pass.md)
> **Blocks**: [T-026](./ticket-026-backward-timing-integration.md)

---

## ⚠️ CRITICAL: Mathematical Correctness Non-Negotiable

This ticket extracts cut computation to a dedicated module. Cut computation is the **mathematical core** of SDDP—the Benders cuts approximate the future cost function.

**ANY change to cut coefficients will cause different convergence and different solutions.**

If ANY test fails or numerical output differs, **STOP IMMEDIATELY**.

---

## Files to Read Before Starting

- `src/sddp/mod.rs:688-820` - `compute_cut_data_for_backward_step()`
- `src/algorithm/backward_pass.rs` - Backward pass from T-024
- `src/fcf.rs` - `CutData` struct definition
- `src/state.rs` - State coefficient computation
- `src/risk_measure.rs` - Risk-adjusted probabilities

---

## Context

### Background

Cut computation takes the dual values from subproblem solves and computes Benders cut coefficients. The cut has the form:

```
θ >= α + β₁s₁ + β₂s₂ + ... + βₙsₙ
```

Where:
- `θ` is the future cost approximation variable
- `α` is the cut intercept
- `βᵢ` are the state variable coefficients (from dual values)
- `sᵢ` are the state variables (reservoir levels, AR states, etc.)

### Current Implementation

The cut computation is embedded in `compute_cut_data_for_backward_step()`:

```rust
fn compute_cut_data_for_backward_step(
    &mut self,
    id: usize,
    past_node_ids: &[usize],
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: &scenario::ScenarioTree,
    iteration: usize,
    forward_pass_idx: usize,
) -> Result<(fcf::CutData, BackwardPhase1Timing), String> {
    // 1. Get forward trajectory state
    // 2. Solve all branching scenarios
    // 3. Compute cut coefficients from duals
    // 4. Apply risk measure
    // 5. Return CutData
}
```

### Target Structure

```rust
// src/algorithm/cut_computation.rs
pub fn compute_cut(
    branching_results: &[BranchingResult],
    risk_measure: &dyn RiskMeasure,
    state: &dyn State,
) -> Result<CutData, String>
```

---

## Specification

### Create `src/algorithm/cut_computation.rs`

```rust
//! Benders cut computation for SDDP algorithm.
//!
//! This module handles the mathematical computation of Benders cuts from
//! subproblem dual values. The cut approximates the expected future cost
//! as a linear function of the current state.
//!
//! # Cut Formula
//!
//! The cut has the form: `θ >= α + Σᵢ βᵢsᵢ`
//!
//! Where:
//! - `α` (intercept) = E[z* - π*·h(s)]
//! - `βᵢ` (coefficients) = E[πᵢ]
//! - `z*` is the optimal objective value
//! - `π*` are the dual values on state-dependent constraints
//! - `h(s)` is the RHS contribution from state
//!
//! # Risk Measures
//!
//! The expectation E[·] can be replaced with risk measures like CVaR.
//! The risk measure adjusts probabilities before computing the cut.
//!
//! # Numerical Precision
//!
//! Cut computation uses `f64` throughout. No rounding or truncation is
//! applied—the exact computed values are stored.

use crate::fcf::CutData;
use crate::risk_measure::RiskMeasure;
use crate::state::State;
use crate::subproblem::Realization;
use crate::utils;

/// Result from a single branching scenario solve.
///
/// Contains all information needed to compute the cut contribution
/// from this scenario.
#[derive(Debug, Clone)]
pub struct BranchingResult {
    /// Optimal objective value for this scenario.
    pub objective: f64,

    /// Dual values on state-dependent constraints.
    /// Length must match number of state coefficients.
    pub state_duals: Vec<f64>,

    /// RHS contribution from state for this scenario.
    pub state_rhs_contribution: f64,

    /// Probability of this scenario (before risk adjustment).
    pub probability: f64,
}

/// Compute a Benders cut from branching results.
///
/// # Arguments
///
/// * `branching_results` - Results from all branching scenario solves
/// * `risk_measure` - Risk measure for probability adjustment
/// * `num_state_coefficients` - Number of state variables
/// * `iteration` - Current iteration number (for cut metadata)
/// * `stage_id` - Stage ID (for cut metadata)
///
/// # Returns
///
/// * `Ok(CutData)` - The computed cut
/// * `Err(String)` - Error if computation fails
///
/// # Algorithm
///
/// 1. Extract objective values from all branchings
/// 2. Adjust probabilities using risk measure
/// 3. Compute cut intercept: α = Σⱼ pⱼ(zⱼ - πⱼ·sⱼ)
/// 4. Compute cut coefficients: βᵢ = Σⱼ pⱼπⱼᵢ
/// 5. Create CutData with computed values
pub fn compute_cut(
    branching_results: &[BranchingResult],
    risk_measure: &dyn RiskMeasure,
    num_state_coefficients: usize,
    iteration: usize,
    stage_id: usize,
) -> Result<CutData, String> {
    if branching_results.is_empty() {
        return Err("Cannot compute cut from empty branching results".to_string());
    }

    // Extract objectives for risk measure
    let objectives: Vec<f64> = branching_results
        .iter()
        .map(|r| r.objective)
        .collect();

    // Get uniform probabilities (before risk adjustment)
    let num_scenarios = branching_results.len();
    let base_probabilities = utils::uniform_prob_by_count(num_scenarios);

    // Adjust probabilities using risk measure
    let adjusted_probabilities = risk_measure.adjust_probabilities(
        &base_probabilities,
        &objectives,
    );

    // Compute cut coefficients
    let coefficients = compute_cut_coefficients(
        branching_results,
        &adjusted_probabilities,
        num_state_coefficients,
    );

    // Compute cut intercept
    let intercept = compute_cut_intercept(
        branching_results,
        &adjusted_probabilities,
    );

    // Create CutData
    // NOTE: The actual CutData constructor may differ - adjust as needed
    Ok(CutData {
        coefficients,
        intercept,
        iteration,
        stage_id,
        // Additional fields as needed by fcf::CutData
    })
}

/// Compute cut coefficients from branching results.
///
/// Each coefficient βᵢ = Σⱼ pⱼπⱼᵢ where:
/// - pⱼ is the (risk-adjusted) probability of scenario j
/// - πⱼᵢ is the dual value for state variable i in scenario j
fn compute_cut_coefficients(
    branching_results: &[BranchingResult],
    probabilities: &[f64],
    num_coefficients: usize,
) -> Vec<f64> {
    let mut coefficients = vec![0.0; num_coefficients];

    for (result, &prob) in branching_results.iter().zip(probabilities.iter()) {
        for (i, &dual) in result.state_duals.iter().enumerate() {
            if i < num_coefficients {
                coefficients[i] += prob * dual;
            }
        }
    }

    coefficients
}

/// Compute cut intercept from branching results.
///
/// Intercept α = Σⱼ pⱼ(zⱼ - πⱼ·sⱼ) where:
/// - pⱼ is the (risk-adjusted) probability of scenario j
/// - zⱼ is the objective value for scenario j
/// - πⱼ·sⱼ is the state RHS contribution for scenario j
fn compute_cut_intercept(
    branching_results: &[BranchingResult],
    probabilities: &[f64],
) -> f64 {
    let mut intercept = 0.0;

    for (result, &prob) in branching_results.iter().zip(probabilities.iter()) {
        let contribution = result.objective - result.state_rhs_contribution;
        intercept += prob * contribution;
    }

    intercept
}

/// Extract branching result from a solved realization.
///
/// This converts a solved `Realization` into the data needed for cut computation.
pub fn extract_branching_result(
    realization: &Realization,
    state: &dyn State,
    probability: f64,
) -> BranchingResult {
    // Extract state duals from realization
    // NOTE: The exact field names may differ - adjust as needed
    let state_duals = state.extract_duals(realization);

    // Compute state RHS contribution
    let state_rhs_contribution = state.compute_rhs_contribution(realization);

    BranchingResult {
        objective: realization.total_stage_objective,
        state_duals,
        state_rhs_contribution,
        probability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_cut_coefficients_uniform() {
        let results = vec![
            BranchingResult {
                objective: 100.0,
                state_duals: vec![1.0, 2.0],
                state_rhs_contribution: 10.0,
                probability: 0.5,
            },
            BranchingResult {
                objective: 200.0,
                state_duals: vec![3.0, 4.0],
                state_rhs_contribution: 20.0,
                probability: 0.5,
            },
        ];
        let probs = vec![0.5, 0.5];

        let coefficients = compute_cut_coefficients(&results, &probs, 2);

        // Expected: [0.5*1.0 + 0.5*3.0, 0.5*2.0 + 0.5*4.0] = [2.0, 3.0]
        assert_eq!(coefficients.len(), 2);
        assert!((coefficients[0] - 2.0).abs() < 1e-10);
        assert!((coefficients[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_cut_intercept() {
        let results = vec![
            BranchingResult {
                objective: 100.0,
                state_duals: vec![1.0],
                state_rhs_contribution: 10.0,
                probability: 0.5,
            },
            BranchingResult {
                objective: 200.0,
                state_duals: vec![3.0],
                state_rhs_contribution: 20.0,
                probability: 0.5,
            },
        ];
        let probs = vec![0.5, 0.5];

        let intercept = compute_cut_intercept(&results, &probs);

        // Expected: 0.5*(100-10) + 0.5*(200-20) = 0.5*90 + 0.5*180 = 45 + 90 = 135
        assert!((intercept - 135.0).abs() < 1e-10);
    }

    #[test]
    fn test_branching_result_creation() {
        let result = BranchingResult {
            objective: 150.0,
            state_duals: vec![1.0, 2.0, 3.0],
            state_rhs_contribution: 25.0,
            probability: 0.25,
        };

        assert_eq!(result.objective, 150.0);
        assert_eq!(result.state_duals.len(), 3);
        assert_eq!(result.probability, 0.25);
    }
}
```

### Update `src/algorithm/mod.rs`

```rust
pub mod cut_computation;

pub use cut_computation::{BranchingResult, compute_cut, extract_branching_result};
```

---

## Acceptance Criteria

- [ ] `src/algorithm/cut_computation.rs` created
- [ ] `BranchingResult` struct defined
- [ ] `compute_cut()` function implemented
- [ ] `compute_cut_coefficients()` helper extracted
- [ ] `compute_cut_intercept()` helper extracted
- [ ] Unit tests for cut computation
- [ ] Module exported from `algorithm/mod.rs`
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass** (CRITICAL - cut computation must be identical)

### Critical Verification

- [ ] Cut coefficients are IDENTICAL to original computation
- [ ] Cut intercept is IDENTICAL to original computation
- [ ] Risk measure application is IDENTICAL

---

## Implementation Guide

### Suggested Approach

1. **Find the exact cut computation logic** in `sddp/mod.rs`:
   ```bash
   grep -n "coefficients\|intercept\|cut_data" src/sddp/mod.rs | head -30
   ```

2. **Understand the CutData structure**:
   ```bash
   grep -A 20 "struct CutData" src/fcf.rs
   ```

3. **Create `cut_computation.rs`** with the `BranchingResult` struct

4. **Implement `compute_cut_coefficients()`** - copy exact logic

5. **Implement `compute_cut_intercept()`** - copy exact logic

6. **Implement `compute_cut()`** - orchestrate the computation

7. **Write unit tests** to verify math is correct

8. **Run golden tests** to verify end-to-end correctness

### Key Files to Modify/Create

| File | Action |
|------|--------|
| `src/algorithm/cut_computation.rs` | CREATE |
| `src/algorithm/mod.rs` | MODIFY (add export) |
| `src/algorithm/backward_pass.rs` | MODIFY (use cut_computation) |

### Mathematical Verification

The cut formula must be:

```
θ ≥ α + β'(s - s₀)

Where:
α = Σⱼ pⱼ zⱼ*  (expected objective under risk measure)
βᵢ = Σⱼ pⱼ πⱼᵢ (expected dual for state variable i)
```

Verify by comparing computed cuts with original implementation.

### Pitfalls to Avoid

- ⚠️ Do NOT change the mathematical formula
- ⚠️ Do NOT round or truncate coefficients
- ⚠️ Do NOT change probability adjustment order
- ⚠️ Ensure dual values are extracted from correct constraints
- ⚠️ Ensure state RHS contribution is computed correctly

---

## Testing Requirements

### Unit Tests

- [ ] Test `compute_cut_coefficients()` with known values
- [ ] Test `compute_cut_intercept()` with known values
- [ ] Test `compute_cut()` with mock risk measure
- [ ] Test edge cases (single branching, many branchings)

### Golden Tests (CRITICAL)

- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Verify cut coefficients match before/after

---

## Documentation Requirements

- [ ] Module-level docs explaining cut formula
- [ ] Function-level docs with mathematical notation
- [ ] Document `BranchingResult` fields
- [ ] Note numerical precision requirements

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Mathematical extraction requires care; good test coverage needed

---

## Definition of Done

- [ ] `cut_computation.rs` created
- [ ] Cut formula implemented correctly
- [ ] Unit tests passing
- [ ] Module exports updated
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass**
- [ ] Documentation complete
- [ ] Code reviewed for mathematical correctness
