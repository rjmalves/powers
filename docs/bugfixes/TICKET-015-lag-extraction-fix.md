# TICKET-015: PAR Example Infeasibility - Lag Extraction Bug Fix

## Summary

**Date:** 2024-01-XX  
**Status:** Partial Fix Implemented  
**Impact:** High - Fixes initial infeasibility in PAR examples 06 & 07

## Problem

Examples 06 and 07 (PAR models) were failing with infeasibility immediately or after few iterations. Investigation revealed incorrect lag extraction at stage 0 for StorageAndInflowState.

## Root Cause

**Location:** `src/state.rs::StorageAndInflowState::update_from_trajectory()`

**Issue:** PreStudy anchor nodes (id=0) were included in the trajectory used for lag extraction. Anchor nodes are used for initial storage but are NOT converted for AR lags (their `inflow_residual` remains `[0.0]`).

**Failure Scenario:**

- Stage 0 trajectory: `[PreStudy -1, PreStudy 0]`
  - PreStudy -1: Converted initial condition, e.g., Y=30 → Z'=-2.0 ✅
  - PreStudy 0: Anchor node (unconverted), inflow_residual=[0.0] ❌
- Extraction formula: `hist_idx = traj_len - 1 - lag_idx`
  - For AR(1), lag_idx=0: hist_idx = 2-1-0 = 1
  - Extracted: trajectory[1] = 0.0 (anchor) ❌
  - **Should extract:** trajectory[0] = -2.0 (converted lag) ✅

**Why This Causes Infeasibility:**

- AR constraint: `Z'_t - 0.7 * Z'_{t-1} = ε_t`
- With wrong lag: `Z'_0 - 0.7 * 0.0 = ε_0` → lag contribution is 0.0, breaks AR dynamics
- With correct lag: `Z'_0 - 0.7 * (-2.0) = ε_0` → lag contribution is -1.4, AR dynamics work correctly

## Solution

**File:** `src/state.rs`  
**Method:** `StorageAndInflowState::update_from_trajectory()`

Filter the trajectory to exclude PreStudy anchor nodes before extracting lags:

```rust
let filtered_trajectory: Vec<&subproblem::Realization> = past_realizations
    .iter()
    .filter(|r| {
        // Keep all Study/PostStudy nodes
        if r.kind != subproblem::StudyPeriodKind::PreStudy {
            return true;
        }
        // For PreStudy nodes, keep only if they have non-zero inflow_residual
        // (Anchor node has all zeros because it's never converted)
        r.inflow_residual.iter().any(|&val| val.abs() > 1e-10)
    })
    .copied()
    .collect();
```

**Key Insight:** Anchor nodes are identifiable by having all `inflow_residual` values equal to 0.0, since they skip the conversion step in `src/sddp/mod.rs` (line 892: `if id >= 0 { continue; }`).

## Results

**Before Fix:**

- Example 07: Fails on iteration 1 (immediate infeasibility)
- Example 06: Fails on iteration 1 (immediate infeasibility)

**After Fix:**

- Example 07: Progresses to iteration 5 before infeasibility ✅
- Example 06: Progresses further before infeasibility ✅
- Lag extraction at stage 0: `[-2.0]` (correct) instead of `[0.0]` (wrong) ✅

## Remaining Issues

Both examples still fail with infeasibility later in training (iteration 5). This suggests additional root causes:

**Possible Secondary Issues:**

1. Backward pass branching scenarios may not update lag variable bounds correctly
2. Other constraint conflicts unrelated to lag extraction
3. Numerical stability issues with extreme innovation values
4. Solver tolerances or basis reuse issues

**Next Investigation Steps:**

1. Verify backward pass updates lag variable bounds before each branching solve
2. Check for other constraints that might conflict with AR dynamics
3. Add constraint feasibility analysis tool to identify conflicting constraints
4. Test with different solvers (currently using HiGHS)

## Testing

**Validation:**

```bash
# Run example 07 test (progresses to iteration 5)
cargo test --test test_scenario_generation_integration test_example_07_par_with_inflow_state -- --include-ignored

# Run example 06 test
cargo test --test test_scenario_generation_integration test_example_06_par_model -- --include-ignored
```

**Expected:** Both tests progress further than before, but still fail later. Full fix requires addressing secondary issues.

## Debug Logging

Temporary debug logging added (gated by `cfg!(debug_assertions)`):

- `src/state.rs`: Trajectory filtering, lag extraction, lag variable bounds
- `src/subproblem.rs`: AR constraint RHS updates, lag buffer updates

**Cleanup Required:** Remove or gate behind feature flag before production release.

## References

- **Ticket:** `TICKET-015-fix-par-example-infeasibility.md`
- **Hypothesis:** Hypothesis 3 (Lag buffer updates) - CONFIRMED
- **Related Files:**
  - `src/state.rs` (fix location)
  - `src/sddp/mod.rs` (PreStudy node initialization)
  - `src/subproblem.rs` (AR constraint updates)
- **Examples:**
  - `examples/06-par-model/`
  - `examples/07-par-model-with-inflow-state/`

## Lessons Learned

1. **PreStudy node semantics:** Anchor nodes (id=0) are for storage initialization only, NOT for AR lag values
2. **Two state types:** StorageState (lags in buffer) vs StorageAndInflowState (lags as LP variables) have different update logic
3. **Trajectory composition:** Includes both PreStudy and Study nodes, requires filtering for lag extraction
4. **Debug logging strategy:** `cfg!(debug_assertions)` provides zero production overhead while enabling deep investigation

## Authorship

- **Investigator:** AI Agent (GitHub Copilot)
- **Methodology:** HPC Developer Protocol - "Correctness before performance"
- **Tools:** Debug logging, test execution, code analysis
- **Duration:** TICKET-015 Investigation Phase (Day 1 of 3)
