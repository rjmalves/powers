# T-011: Expose first-stage branching costs

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: [T-010](./ticket-010-build-display-context.md)
> **Blocks**: [T-012](./ticket-012-integrate-training-loop.md)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Lines 1079-1134: `eval_first_stage_bound()` function
- `src/sddp/mod.rs` - Lines 2345-2368: `eval_first_stage_bound()` free function
- `src/algorithm/backward_pass.rs` - Where backward pass is executed
- `src/display/context.rs` - DisplayContext.first_stage_branching_costs field

## Context

### Background

The first-stage branching costs from the backward pass are the true indicator of policy quality. Currently, `eval_first_stage_bound()` computes the risk-adjusted expected cost but discards the individual branching scenario costs. We need to surface these for display.

### Current State

```rust
fn eval_first_stage_bound(
    branching_realizations: &[subproblem::Realization],
    risk_measure: &dyn risk_measure::RiskMeasure,
) -> Result<f64, String> {
    let costs: Vec<f64> = branching_realizations
        .iter()
        .map(|r| r.total_stage_objective)
        .collect();
    // ... returns risk-adjusted mean, discards individual costs
}
```

We need to return both the bound AND the individual costs.

## Specification

### Modify eval_first_stage_bound Return Type

```rust
/// Result from first-stage evaluation.
#[derive(Debug, Clone)]
pub struct FirstStageResult {
    /// Risk-adjusted expected cost (the lower bound).
    pub bound: f64,
    
    /// Individual branching scenario costs.
    pub branching_costs: Vec<f64>,
    
    /// Probabilities used (uniform, risk-adjusted).
    pub probabilities: Vec<f64>,
}

fn eval_first_stage_bound(
    branching_realizations: &[subproblem::Realization],
    risk_measure: &dyn risk_measure::RiskMeasure,
) -> Result<FirstStageResult, String> {
    let costs: Vec<f64> = branching_realizations
        .iter()
        .map(|r| r.total_stage_objective)
        .collect();
    
    let num_branchings = costs.len();
    
    EVAL_PROBS_BUFFER.with(|buf| {
        let mut probabilities = buf.borrow_mut();
        probabilities.clear();
        probabilities.resize(num_branchings, 0.0);
        utils::fill_uniform_probabilities(&mut probabilities);

        let adjusted_probabilities =
            risk_measure.adjust_probabilities(&probabilities, &costs);
        let bound = utils::dot_product(adjusted_probabilities, &costs);
        
        Ok(FirstStageResult {
            bound,
            branching_costs: costs,
            probabilities: adjusted_probabilities.to_vec(),
        })
    })
}
```

### Update Call Sites

In `SddpTrainHandler::eval_first_stage_bound()`:

```rust
pub fn eval_first_stage_bound(
    &mut self,
    id: usize,
    past_node_ids: &[usize],
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: &scenario::ScenarioTree,
) -> Result<(FirstStageResult, BranchingsTiming), String> {
    // ... existing code ...
    
    let result = eval_first_stage_bound(
        branching_node_data,
        node_data_graph
            .get_node(id)
            .ok_or_else(|| ...)?
            .data
            .risk_measure
            .as_ref(),
    )?;

    Ok((result, branchings_timing))
}
```

### Update Training Loop

In the training loop where `eval_first_stage_bound` is called:

```rust
// Current
let (lower_bound, _timing) = handler.eval_first_stage_bound(...)?;

// New
let (first_stage_result, _timing) = handler.eval_first_stage_bound(...)?;
let lower_bound = first_stage_result.bound;
let first_stage_branching_costs = first_stage_result.branching_costs;
```

### Store in IterationResult

```rust
pub struct IterationResult {
    // ... existing fields ...
    
    /// Individual first-stage branching scenario costs.
    pub first_stage_branching_costs: Vec<f64>,
}
```

## Acceptance Criteria

- [x] `FirstStageResult` struct defined with all fields
- [x] `eval_first_stage_bound` returns `FirstStageResult`
- [x] All call sites updated to use new return type
- [x] `IterationResult` includes `first_stage_branching_costs`
- [x] Training loop populates the field
- [x] Existing tests updated and passing
- [x] No performance regression (costs already computed, just not discarded)

## Implementation Guide

### Step 1: Define FirstStageResult

Add struct definition in `src/sddp/mod.rs`.

### Step 2: Update eval_first_stage_bound (free function)

Modify return type and implementation.

### Step 3: Update SddpTrainHandler::eval_first_stage_bound

Propagate the new return type.

### Step 4: Update training loop

Extract and store branching costs.

### Step 5: Add to IterationResult

New field with initialization.

### Step 6: Update DisplayContext builder

Use the new field.

## Pitfalls to Avoid

- ⚠️ Don't clone vectors unnecessarily - move ownership where possible
- ⚠️ Existing tests may check return type - update them
- ⚠️ The free function and method have same name - update both
- ⚠️ Thread-local buffer for probabilities is fine to copy out

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_first_stage_result_fields() {
    let realizations = create_mock_realizations(5);
    let risk_measure = crate::risk_measure::factory("expectation");
    
    let result = eval_first_stage_bound(&realizations, risk_measure.as_ref()).unwrap();
    
    assert_eq!(result.branching_costs.len(), 5);
    assert_eq!(result.probabilities.len(), 5);
    assert!(result.bound > 0.0);
}

#[test]
fn test_first_stage_costs_match_total_objectives() {
    let realizations = create_mock_realizations_with_costs(&[100.0, 110.0, 120.0]);
    let risk_measure = crate::risk_measure::factory("expectation");
    
    let result = eval_first_stage_bound(&realizations, risk_measure.as_ref()).unwrap();
    
    assert_eq!(result.branching_costs, vec![100.0, 110.0, 120.0]);
    // For expectation with uniform probabilities, bound = mean
    assert!((result.bound - 110.0).abs() < 1e-10);
}
```

### Integration Tests

- [ ] Training run produces iteration results with populated first_stage_branching_costs
- [ ] JSON output includes first-stage statistics

## Documentation Requirements

- [ ] Doc comments on `FirstStageResult` explaining each field
- [ ] Update any existing docs about `eval_first_stage_bound`

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Refactoring existing function. Logic already computes what we need.

## Definition of Done

- [x] `FirstStageResult` struct added
- [x] Return type changed and propagated
- [x] `IterationResult` extended
- [x] All existing tests passing (660 tests)
- [x] Test fixtures updated (2 files)
- [x] Code formatted with rustfmt

## Implementation Summary

**Status**: ✅ Complete

**Files Modified**:
- `src/sddp/mod.rs` - Added FirstStageResult struct (lines 91-107), modified eval_first_stage_bound return type
- `src/algorithm/backward_pass.rs` - Updated to capture first-stage costs from FirstStageResult
- `src/algorithm/context.rs` - Added first_stage_branching_costs field to BackwardPassResult
- `src/algorithm/processor.rs` - Updated trait signature to return FirstStageResult
- `src/algorithm/coordinator.rs` - Updated implementation to return FirstStageResult
- `src/display/context.rs` - Updated from_iteration() to compute first_stage_stats from actual costs
- `src/display/context.rs` (tests) - Fixed mock IterationResult fixture
- `src/output/parquet/writer.rs` (tests) - Fixed mock IterationResult fixture

**Test Results**: All 660 tests passing

**Notes**: Implementation exactly followed the ticket specification. The costs were already being computed but discarded. Now they flow through BackwardPassResult → IterationResult → DisplayContext for rendering.
