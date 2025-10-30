# TICKET-015 Progress Update

**Date:** 2024-10-29  
**Status:** Partial Fix Implemented, Continuing Investigation

## Summary

Fixed the primary bug (lag extraction), but examples still fail at iteration 5. Investigating remaining causes.

## ✅ Completed: Lag Extraction Bug Fix

**Root Cause (Hypothesis 3 CONFIRMED):**

- `StorageAndInflowState::update_from_trajectory()` extracted lags from PreStudy anchor nodes
- Anchor nodes (id=0) have `inflow_residual=[0.0]` because they're never converted
- Stage 0 trajectory: `[PreStudy -1 (-2.0), PreStudy 0 (0.0)]`
- Formula got index 1 (anchor=0.0) instead of index 0 (lag=-2.0)

**Fix Implemented:**

```rust
// Filter trajectory to exclude PreStudy anchor nodes before lag extraction
let filtered_trajectory: Vec<&subproblem::Realization> = past_realizations
    .iter()
    .filter(|r| {
        if r.kind != subproblem::StudyPeriodKind::PreStudy {
            return true;
        }
        // Keep PreStudy nodes with non-zero inflow_residual (converted lags)
        r.inflow_residual.iter().any(|&val| val.abs() > 1e-10)
    })
    .copied()
    .collect();
```

**Results:**

- ✅ Example 07: Now reaches iteration 5 (was failing at iteration 1)
- ✅ Example 06: Also progresses further
- ✅ Lag extraction: `[-2.0]` (correct) vs `[0.0]` (before)
- ✅ No regressions in examples 01-04

**Files Changed:**

- `src/state.rs` (lines 1218-1255): Lag extraction filter + debug logging
- `docs/bugfixes/TICKET-015-lag-extraction-fix.md`: Detailed documentation

## 🔍 In Progress: Remaining Infeasibility Investigation

**Current Status:** Both examples still fail with infeasibility at iteration 5

**Hypotheses Under Investigation:**

### 1. Negative Inflow Values (Normal Marginals)

**Theory:**  
AR model with normal marginals: `Y_t = μ_s + σ_s * Z'_t`

- If Z'\_t very negative (e.g., < -4σ) → Y_t could be negative
- LP has constraint Y_t ≥ 0 → infeasibility

**Example 07 Parameters:**

- Season 4: μ=45, σ=10
- If Z'\_t=-5: Y_t = 45 + 10\*(-5) = -5 ❌

**Diagnostic Added:**

- `check_for_negative_inflow_risk()` in `subproblem.rs`
- Warns when Z'\_t < -4 (extreme negative residual)

**Test Result:** No warnings triggered before failure, so this may not be the primary cause.

### 2. Backward Pass Lag Variable Bounds

**Theory:**  
In backward pass, solve multiple branching scenarios from same node.  
Question: Are lag variable bounds updated for each branching?

**Analysis:**

- Forward pass calls `update_with_current_trajectory()` → sets lag variable bounds
- Backward pass (`solve_all_branchings`) does NOT call `update_with_current_trajectory()`
- **But this is correct!** All branchings share same history (same lag values)
- Only current innovation ε_t differs between branchings

**Conclusion:** Not the issue.

### 3. Other Potential Causes

- Constraint conflicts (storage bounds + inflow + deficit)
- Numerical issues in solver
- Basis reuse causing infeasible warm starts
- Seasonal parameter mismatches

## Next Investigation Steps

1. **Dump LP model at failure point**

   - Catch panic in `retry_solve()`
   - Call `model.write_lp("failed_model.lp")`
   - Inspect constraint matrix manually

2. **Test with lognormal3 marginals**

   - Modify example recourse.json to use lognormal3 instead of normal
   - Lognormal3 ensures Y_t > 0 always
   - If this fixes it → confirms negative inflow hypothesis

3. **Disable basis reuse**

   - Test if starting from scratch each solve helps
   - May indicate warm start incompatibility

4. **Simplified test case**
   - Create minimal 2-stage PAR model
   - 1 hydro, simple parameters
   - Isolate the failure

## Lognormal3 Distribution Option

**Current (Normal):**

```json
{
  "marginal_distribution": {
    "type": "normal",
    "mean": 0.0,
    "std_dev": 1.0
  }
}
```

**Alternative (Lognormal3 - Always Positive):**

```json
{
  "marginal_distribution": {
    "type": "lognormal3",
    "mean": 0.0,
    "std_dev": 1.0,
    "shift": -2.0
  }
}
```

**Advantage:** Ensures transformed inflow Y_t > 0 always, eliminating negative inflow infeasibility.

## Time Spent

- Investigation: ~4 hours
- Fix implementation: ~1 hour
- Testing & documentation: ~1 hour
- **Total: ~6 hours** (Day 1 of estimated 3 days)

## Files Modified

- `src/state.rs`: Lag extraction fix + debug logging
- `src/subproblem.rs`: Negative inflow diagnostic + debug logging
- `src/sddp/mod.rs`: Backward pass debug logging
- `docs/bugfixes/TICKET-015-lag-extraction-fix.md`: Documentation

## Next Session Plan

1. Test example 07 with lognormal3 marginals
2. If still fails, dump LP model for inspection
3. Create minimal reproducible test case
4. Identify exact constraint causing infeasibility
