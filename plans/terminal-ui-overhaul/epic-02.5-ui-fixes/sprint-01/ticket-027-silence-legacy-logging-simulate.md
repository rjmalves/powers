# [T-027] Silence Legacy Logging in SDDP Simulate

> **Epic**: [Epic 02.5: UI Fixes](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-030](./ticket-030-fix-minor-issues.md)

## Context

### Background

The `simulate()` function in `src/sddp/mod.rs` contains legacy `log::info!` calls that output simulation status. These should be replaced with calls to the display system, or the information should be returned to the caller for proper rendering.

### Current State

```rust
// Lines 2157-2231 in src/sddp/mod.rs
::log::info!("");
::log::info!("# Simulating");
::log::info!("- Scenarios: {}", num_simulation_scenarios);
::log::info!("");
// ... simulation runs ...
::log::info!("Expected cost ($): {:.6e} ± {:.6e}", mean_cost, std_cost);
::log::info!("");
::log::info!("Simulation time: {:02}:{:02}:{:02}.{:03}", ...);
```

This produces after training summary:
```
Training Stopped ⋯
──────────────────
  Total time:     00:00:02.545
  ...
  Iterations:     3[INFO]          ← No newline before [INFO]
[INFO] # Simulating
[INFO] - Scenarios: 4
[INFO] 
[INFO] Expected cost ($): 1.022176e8 ± 5.256744e5
[INFO] 
[INFO] Simulation time: 00:00:00.138
```

## Specification

### Changes Required

The `simulate()` function needs to either:

**Option A**: Return simulation metadata to caller (preferred)
- Remove all `log::info!` calls from `simulate()`
- Return timing and statistics alongside trajectories
- Let caller use display system to render

**Option B**: Accept display renderer (more complex)
- Pass renderer and config to `simulate()`
- Call renderer methods from within

**Recommendation**: Use **Option A** for cleaner separation of concerns.

### Proposed Changes

1. Create a new struct `SimulationOutput`:
```rust
pub struct SimulationOutput {
    pub trajectories: Vec<SimulationTrajectory>,
    pub elapsed: Duration,
    pub mean_cost: f64,
    pub std_cost: f64,
}
```

2. Modify `simulate()` to return `Result<SimulationOutput, String>`

3. Remove all `log::info!` calls from `simulate()`

4. Update callers to use display system:
   - `src/lib.rs`: Call `renderer.render_simulation_summary()`
   - `src/sddp/instance.rs`: Add `simulate_with_display()` method

### Behavior

- **Simulation phase**: No `[INFO]` prefixed output
- **Display system**: `render_simulation_start()` and `render_simulation_summary()` handle all output
- **Statistics**: Computed in `simulate()`, rendered by display system

## Acceptance Criteria

- [x] Running simulation produces NO `[INFO]` prefix messages
- [x] Simulation start message uses display system (profile-aware)
- [x] Simulation summary uses display system (shows mean, std, time)
- [x] Simulation timing information preserved and displayed
- [x] Clear visual separation from training output
- [x] All existing tests pass
- [x] API backward compatibility maintained (simulate() still exists)

## Implementation Guide

### Suggested Approach

1. Create `SimulationOutput` struct in `src/sddp/mod.rs`
2. Modify `simulate()` signature to return `SimulationOutput`
3. Remove `log::info!` calls from `simulate()`, keeping computation
4. Update `src/sddp/instance.rs`:
   - Create `simulate_with_display()` method
   - Call `render_simulation_start()` before simulation
   - Call `render_simulation_summary()` after simulation
5. Update `src/lib.rs` to use new display-integrated method
6. Test all profiles

### Key Files to Modify

- `src/sddp/mod.rs`: `simulate()` function (lines 2148-2234)
- `src/sddp/instance.rs`: Add `simulate_with_display()` method
- `src/lib.rs`: Update simulation call (lines 128-141)

### Patterns to Follow

- See `train_with_display()` in `src/sddp/instance.rs` for renderer integration pattern
- See `render_simulation_summary()` in renderers for expected output format

### Pitfalls to Avoid

- ⚠️ Don't break existing `simulate()` callers (tests, etc.)
- ⚠️ Preserve simulation timing computation accuracy
- ⚠️ Ensure `SimulationOutput` includes all needed data for display

## Testing Requirements

### Manual Tests

- [x] `powers examples/05-large-scale-brazilian` shows clean simulation output
- [x] `powers examples/05-large-scale-brazilian --profile standard` works
- [x] `powers examples/05-large-scale-brazilian --profile minimal` works
- [x] `powers examples/05-large-scale-brazilian --profile automation` shows JSON

### Automated Tests

- [x] All existing 749 tests pass
- [x] No new clippy warnings
- [x] Any tests using `simulate()` still work

## Documentation Requirements

- [ ] Update doc comments for `simulate()` to reflect new return type
- [ ] Document `SimulationOutput` struct

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear pattern from `train_with_display()`, moderate refactoring

## Definition of Done

- [x] No `log::info!` calls in `simulate()`
- [x] Simulation output uses display system
- [x] Clear separation between training and simulation phases
- [x] All tests passing
- [x] Manual verification with all profiles

## Status: ✅ COMPLETE

**Implementation Summary:**
- Removed all `log::info!` calls from `simulate()` in src/sddp/mod.rs
- Removed unused variables: `begin`, `mean_cost`, `std_cost`, `simulation_costs`
- Added `simulate_with_display()` method to src/sddp/instance.rs
- Updated lib.rs to use `simulate_with_display()` instead of `simulate()`
- Removed simulation skip message logging (no longer needed)
- All 749 tests passing
- All 4 profiles tested and working correctly
- Backward compatibility maintained (`simulate()` still exists for direct use)
