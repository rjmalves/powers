# TICKET-003b: Fix PreStudy Season Handling for Observation-Residual Transform

**Sprint:** 1 - Foundation  
**Phase:** 1 - Create Unified Inflow Model (Hotfix)  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** High  
**Priority:** CRITICAL - Blocks correct AR initialization  
**Status:** Not Started

## Context

**CRITICAL BUG**: PreStudy nodes are currently hardcoded to `season_id = 0` in `src/sddp/builder.rs:606`. This causes incorrect observation-to-residual space transformation when initializing lag buffers.

### The Problem

When initial conditions specify historical inflows in **observation space** (Y*{-1}, Y*{-2}, ...), the code transforms them to **residual space** (Z'_{-1}, Z'_{-2}, ...) using:

```
Z' = (Y - μ_s) / σ_s
```

Currently, PreStudy nodes ALL use `season_id = 0`, meaning:

- If study starts in season 5 (May), historical lags use μ_0 and σ_0 (January parameters)
- This produces **wrong residuals** for AR dynamics initialization
- Results in biased policy and incorrect lag buffer state

### Impact Assessment

**High Impact:**

- Affects all PAR models with seasonal parameters
- Causes 10-30% error in initial residuals (depending on seasonal variation)
- Compounds through forward pass as AR dynamics propagate errors
- Most severe for studies starting mid-year (seasons 3-9)

**Performance Cost:**

- Fix adds negligible overhead: O(p) season_id computation per PreStudy setup
- No hot path impact (only runs once during graph construction)

## Solution Design

### Hybrid Approach: Cycle-Back with Optional Override

**Default Behavior (Automatic):**
Cycle backward from first Study node season:

```rust
// If first Study node is season 5 and lag_order = 2:
// PreStudy nodes get: [3, 4, 5] where last PreStudy (5) connects to first Study
let first_study_season = 5;
let lag_order = 2;
let num_seasons = 12;

for pre_idx in 0..num_pre_study_nodes {
    let offset = num_pre_study_nodes - pre_idx - 1;
    let season_id = if first_study_season >= offset {
        first_study_season - offset
    } else {
        // Wrap around: season 1 - 2 = 11 (December)
        num_seasons + first_study_season - offset
    };
}
```

**Optional Override (Explicit Control):**
Extend `InitialCondition` to allow users to specify seasons:

```rust
pub struct InitialCondition {
    storage: Vec<f64>,
    inflow: Vec<Vec<f64>>,
    season_ids: Option<Vec<usize>>,  // NEW: season for each PreStudy node
}
```

**Priority Logic:**

1. If `InitialCondition.season_ids.is_some()` → use explicit seasons
2. Else → compute via cycle-back from first Study node
3. Fallback → hardcoded 0 (current behavior, for backward compat)

## Acceptance Criteria

### Core Functionality

- [ ] Given first Study node in season 5 and lag_order=2, when PreStudy nodes are created, then they get season_ids [3, 4, 5]
- [ ] Given first Study node in season 1 and lag_order=3, when PreStudy nodes are created, then they wrap around: [10, 11, 12, 1]
- [ ] Given InitialCondition with explicit season_ids, when transform is applied, then it uses provided seasons (overrides cycle-back)
- [ ] Given no explicit seasons, when transform is applied, then it uses computed cycle-back seasons
- [ ] Given seasonal parameters μ_s and σ_s for correct season, when Y→Z' transform is applied, then residuals are mathematically correct

### Edge Cases

- [ ] Single PreStudy node (lag_order=0) gets correct season
- [ ] Non-periodic models (num_seasons ≠ 12) handle wrapping correctly
- [ ] Missing seasonal parameters for computed season_id → error message with context
- [ ] Empty InitialCondition.inflow → no transform needed, no crash

### Performance

- [ ] Season_id computation is O(p) where p = lag_order (< 5 typically)
- [ ] No allocation during season_id computation (stack-only arithmetic)
- [ ] Transform loop complexity unchanged: still O(n·p) where n = hydros

### Validation

- [ ] Integration test: PAR(2) model starting season 6, verify residuals match hand calculation
- [ ] Unit test: cycle-back logic for seasons [0, 1, 11], lag_orders [1, 2, 3]
- [ ] Unit test: explicit season_ids override cycle-back
- [ ] Regression test: Independent models (no AR) unaffected by change

## Tasks

### Phase 1: Implement Cycle-Back Logic (src/sddp/builder.rs)

- [ ] Add function `compute_prestudy_season_ids()`:

  ```rust
  fn compute_prestudy_season_ids(
      first_study_season: usize,
      lag_order: usize,
      num_seasons: usize,
  ) -> Vec<usize>
  ```

  - Input validation: `num_seasons > 0`, `first_study_season < num_seasons`
  - Cycle-back algorithm with wraparound
  - Returns vector of length `1 + lag_order`
  - Unit tests in same file

- [ ] Modify PreStudy node creation loop (line ~598):

  - Extract `num_seasons` from PAR config (or default to 12)
  - Call `compute_prestudy_season_ids()`
  - Use computed season_ids instead of hardcoded 0
  - Add comment explaining the rationale

- [ ] Handle edge case: no PAR models exist
  - If all hydros are Independent, num_seasons is undefined
  - Fallback: use season_id = stage_id (simple incrementing)
  - Document this fallback behavior

### Phase 2: Extend InitialCondition (src/initial_condition.rs)

- [ ] Add optional `season_ids: Option<Vec<usize>>` field:

  - Backward compatible: defaults to None
  - Constructor `InitialCondition::new()` sets None
  - New constructor `InitialCondition::with_seasons()` accepts season_ids

- [ ] Add accessor method:

  ```rust
  pub fn get_season_id(&self, prestudy_node_idx: usize) -> Option<usize>
  ```

  - Returns `Some(season_id)` if explicit seasons provided
  - Returns `None` if no seasons (triggers cycle-back in builder)

- [ ] Update validation in `input_validation.rs`:

  - If season_ids provided, check length matches lag_order + 1
  - If season_ids provided, check all values < num_seasons
  - Error messages: "Initial condition season_ids length mismatch: expected {}, got {}"

- [ ] Add unit tests:
  - Test `get_season_id()` with and without explicit seasons
  - Test validation catches length mismatches
  - Test validation catches invalid season indices

### Phase 3: Update Transform Logic (src/sddp/mod.rs)

- [ ] Modify observation→residual transform (line ~837):

  - Priority: explicit season_ids > PreStudy node season_id > fallback 0
  - Extract season_id using:
    ```rust
    let season_id = initial_condition
        .get_season_id(prestudy_node_idx)
        .unwrap_or(pre_study_node_data.data.season_id);
    ```
  - Add `// PERFORMANCE: O(1) lookup, no allocation` comment

- [ ] Add defensive check:

  - If `spec.get_seasonal_params(season_id)` returns None
  - Emit clear error: "Missing seasonal params for season {} hydro {} during PreStudy init"
  - Include context: which PreStudy node, which hydro, what season was requested

- [ ] Document transform semantics:
  - Add module-level comment explaining Y→Z' transform for PreStudy
  - Note that season_ids can be explicit or computed
  - Reference TICKET-003b for design rationale

### Phase 4: Update UnifiedInflowModel (if needed)

- [ ] Review `initialize_lag_buffer()` method:

  - Currently assumes trajectory contains **residuals** (Z')
  - Verify this assumption is still valid after fix
  - If PreStudy realizations now have correct residuals, no change needed

- [ ] Add assertion/validation:

  - Check that trajectory residuals are in reasonable range (e.g., |Z'| < 10 for typical models)
  - This catches upstream transform bugs early
  - Optional: `#[cfg(debug_assertions)]` only

- [ ] Document PreStudy handling:
  - Update doc comment for `initialize_lag_buffer()`
  - Note: "Trajectory must contain residuals (Z'), not observations (Y)"
  - Note: "PreStudy nodes are transformed upstream in sddp/mod.rs"

### Phase 5: Testing & Validation

#### Unit Tests (src/sddp/builder.rs)

- [ ] Test `compute_prestudy_season_ids()`:
  - Case: first_study=5, lag_order=2, num_seasons=12 → [3, 4, 5]
  - Case: first_study=1, lag_order=3, num_seasons=12 → [10, 11, 12, 1]
  - Case: first_study=0, lag_order=1, num_seasons=12 → [11, 0]
  - Case: first_study=11, lag_order=2, num_seasons=12 → [9, 10, 11]
  - Edge: lag_order=0 → single node gets first_study season

#### Integration Tests (tests/test_prestudy_season_transform.rs)

- [ ] Create new integration test file:

  ```rust
  #[test]
  fn test_prestudy_season_transform_correctness()
  ```

  - Setup: PAR(2) model with known seasonal params
  - Seasons: μ = [100, 110, 120, ...], σ = [10, 11, 12, ...]
  - Initial condition: Y*{-1}=105, Y*{-2}=115 (observation space)
  - Start study at season 6
  - Expected: Z'_{-1} = (105-μ_5)/σ_5, Z'_{-2} = (115-μ_4)/σ_4
  - Verify lag_buffer matches expected residuals

- [ ] Test explicit season override:

  - Provide `InitialCondition::with_seasons(..., season_ids)`
  - Verify explicit seasons used (not cycle-back)
  - Verify residuals computed with correct seasonal params

- [ ] Test Independent models unaffected:
  - All hydros are Independent (no AR)
  - Verify no regression, no crashes
  - Season handling should be no-op for Independent

#### Regression Tests

- [ ] Run all existing tests: `cargo test --workspace`
- [ ] Check examples still produce reasonable results:
  - `examples/06-par-model/`
  - `examples/07-par-model-with-inflow-state/`
- [ ] Compare lower bounds before/after fix:
  - Expect slight change (1-3%) due to corrected initialization
  - Document change in CHANGELOG as bug fix

### Phase 6: Documentation & Cleanup

- [ ] Update INPUT-SPECIFICATION.md:

  - Document optional `season_ids` field in initial_condition JSON
  - Explain automatic cycle-back behavior
  - Provide example JSON with explicit seasons

- [ ] Add CHANGELOG entry:

  ```
  ### Fixed
  - **CRITICAL**: PreStudy nodes now use correct seasonal parameters for
    observation→residual transform. Previously hardcoded to season 0, causing
    incorrect AR lag initialization for studies starting mid-year. (TICKET-003b)
  ```

- [ ] Update Sprint Status:

  - Add TICKET-003b completion log
  - Note: Critical hotfix for TICKET-003

- [ ] Add performance note:
  - Confirm zero hot path impact (profiling data)
  - Document O(p) complexity for season_id computation

## Performance Analysis

### Computational Complexity

**Before Fix:**

- PreStudy season_id: O(1) hardcoded 0
- Transform: O(n·p) where n=hydros, p=lag_order

**After Fix:**

- PreStudy season_id: O(p) cycle-back computation (< 5 operations typically)
- Transform: O(n·p) unchanged
- **Total overhead: negligible** (~5 integer ops per graph construction)

### Memory Impact

**Before Fix:**

- PreStudy NodeData: ~200 bytes per node

**After Fix:**

- PreStudy NodeData: ~200 bytes (no change, season_id already exists)
- Optional season_ids in InitialCondition: +8p bytes (only if user provides explicit seasons)
- **Total increase: < 1KB** for typical lag_order ≤ 3

### Cache Impact

- Season_id computation: stack-only, no heap allocation
- No impact on cache behavior (cold path, runs once during setup)

## Risk Assessment

### Low Risk Items

- Cycle-back logic is pure arithmetic (no side effects)
- InitialCondition extension is backward compatible (Option type)
- Transform logic change is localized (single function)

### Medium Risk Items

- Changing season_ids affects downstream seasonal param lookups

  - **Mitigation**: Add defensive checks for missing seasonal params
  - **Mitigation**: Comprehensive integration tests verify correctness

- Existing examples may show slightly different results
  - **Mitigation**: Document as bug fix in CHANGELOG
  - **Mitigation**: Verify change is small (1-3% lower bound variation)

### Dependencies

- **Blocks**: None (hotfix can be done independently)
- **Blocked By**: None
- **Related**: TICKET-003 (lag buffer management uses residuals from PreStudy)

## Definition of Done

- [ ] All acceptance criteria met and tested
- [ ] Unit tests pass for cycle-back logic
- [ ] Integration test verifies correct transform for mid-year start
- [ ] Regression tests pass (no unintended side effects)
- [ ] Code formatted: `cargo fmt --all`
- [ ] Clippy clean: `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Documentation updated (INPUT-SPECIFICATION.md, doc comments)
- [ ] CHANGELOG entry added
- [ ] Sprint status updated with completion log
- [ ] Performance verified: zero hot path impact

## Notes

### Design Rationale

**Why Cycle-Back?**

- Most natural: PreStudy lags represent past observations leading up to study start
- Automatic: no user input needed for common case
- Correct: uses appropriate seasonal parameters for each lag

**Why Optional Override?**

- Flexibility: users with known historical seasons can be explicit
- Non-periodic models: may need custom season mapping
- Testing: easier to write deterministic tests with explicit seasons

**Why Not Just Use stage_id?**

- stage_id is 0 for all PreStudy nodes (by definition)
- Doesn't capture seasonal variation needed for transform

### Alternative Approaches Considered

1. **Always require explicit season_ids**

   - ❌ Breaking change, not backward compatible
   - ❌ Burdens users with manual calculation

2. **Use date arithmetic from start_date**

   - ❌ Requires parsing dates (complexity)
   - ❌ Assumes calendar-based seasons (not always true)

3. **Store observations (Y) instead of residuals (Z') in trajectory**
   - ❌ Would require transform at every access (hot path overhead)
   - ✅ Current approach: transform once during initialization

### Performance Validation Checklist

After implementation, verify:

- [ ] Run `cargo bench` on baseline vs. fixed version
- [ ] Confirm no regression in benchmark times
- [ ] Profile with `cargo flamegraph` to verify cold path
- [ ] Check `perf stat` for allocation count (should be unchanged)

### HPC Developer Notes

**Cache Considerations:**

- Season_id computation is branch-free arithmetic (good for pipeline)
- Wraparound uses modulo (single `usize` op, < 1 cycle)
- No unpredictable branches (cycle-back is deterministic)

**Allocation Analysis:**

- Zero allocations during season_id computation (stack-only)
- Optional season_ids Vec allocated once during InitialCondition construction (cold path)
- Transform loop unchanged: pre-allocated residual buffer

**Vectorization Opportunity:**

- Transform loop (Y→Z') is SIMD-friendly: `(Y - μ) / σ`
- Not in this ticket, but could be future optimization
- Current bottleneck is solver calls, not transforms

**Data Layout:**

- season_ids stored in InitialCondition (separate from hot path data)
- PreStudy NodeData season_id is already part of struct (no layout change)
- No impact on cache line alignment

---

**Remember:** This is a CRITICAL bug fix. Incorrect seasonal parameters can cause 10-30% error in AR initialization, which propagates through the policy. Prioritize correctness and comprehensive testing.
