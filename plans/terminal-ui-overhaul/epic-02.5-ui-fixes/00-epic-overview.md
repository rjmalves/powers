# Epic 02.5: UI Output Fixes

## Summary

Fix critical display issues discovered after Epic 02 implementation. The current output mixes old `log::info!` messages with the new display system, produces duplicate table headers, has broken table alignment, and shows incorrect statistics. This epic will consolidate output to use the new display system exclusively during training and simulation phases.

## Problem Statement

### Current Broken Output

```
[INFO] 
[INFO] POWE.RS - Power Optimization for the World of Energy - in pure RuSt   ← OLD LOGGING
[INFO] --------------------------------------------------------------------
[INFO] 
[INFO] Reading input files from 'examples/05-large-scale-brazilian'           ← OLD LOGGING
╭───────────────────────────────────────────────────────────────────────────────────╮
│ POWE.RS - Power Optimization for the World of Energy                              │  ← NEW DISPLAY
│ Training: 3 iterations × 4 forward passes | Cut selection: enabled               │
╰───────────────────────────────────────────────────────────────────────────────────╯
┌─────┬────────────────┬────────────────┬────────────────┬───────┬─────────────────┐
│ Iter │ Lower Bound ($) │ Simul Cost ($) │ 1st Stage ($)  │ Gap % │ Time (fwd/bwd)  │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤┌─────┬...  ← DUPLICATE HEADER
│   1 │     1.02e8     │     1.02e8     │     1.02e8     │  0.3% │ 0.121s / 0.561s │
│     │ μ=1.02e8 σ=5.26e5 [1.01e8..1.03e8] n=4                                    │  ← MISALIGNED
```

### Issues Identified

| # | Issue | Severity | Root Cause |
|---|-------|----------|------------|
| 1 | **Mixed output systems** | HIGH | Old `log::info!` calls not removed from `src/lib.rs` and `src/sddp/mod.rs` |
| 2 | **Duplicate table headers** | HIGH | `render_table_header()` called in `train_with_display()` AND inside `render_iteration()` for first iteration |
| 3 | **Misaligned statistics rows** | HIGH | Statistics continuation row doesn't span table columns correctly |
| 4 | **Progress bar stacking (minimal)** | MEDIUM | Missing flush after carriage return, terminal not overwriting |
| 5 | **Negative std dev in summary** | LOW | Incorrect calculation: `best_upper_bound - statistical_upper_bound` can be negative |
| 6 | **Missing newlines** | LOW | Training summary runs into `[INFO]` simulation messages |

## Scope

### Included

- Remove/silence legacy `log::info!` messages during training phase
- Remove/silence legacy `log::info!` messages during simulation phase  
- Fix duplicate table header rendering
- Fix statistics row alignment in advanced renderer
- Fix minimal profile progress bar terminal behavior
- Fix negative std dev display in training summary
- Add proper newlines between sections
- Integrate simulation with new display system

### Excluded

- New features (deferred to Epic 03)
- Automation profile changes (already works correctly)
- Performance optimizations
- Additional styling/polish

## Dependencies

- **Requires**: Epic 02 complete (display components and renderers implemented)
- **Enables**: Epic 03 (clean foundation for simulation display)

## Acceptance Criteria

- [x] Running `powers examples/05-large-scale-brazilian` produces clean output with NO `[INFO]` prefix messages
- [x] Table header appears exactly once per training run
- [x] Statistics continuation rows align with table structure
- [x] Minimal profile progress bar updates in-place (no stacking)
- [x] Training summary shows correct positive std dev
- [x] Clear visual separation between training and simulation phases
- [x] All 4 profiles (advanced, standard, minimal, automation) produce correct output
- [x] All existing tests pass (749 tests)
- [x] No performance regression

## Technical Approach

### Phase 1: Silence Legacy Logging (T-026, T-027)

Replace `log::info!` calls with new display system calls:

1. **src/lib.rs**: 
   - Remove greeting banner (display header handles this)
   - Remove "Reading input files" (not needed with visual header)
   - Replace simulation messages with renderer calls
   - Replace final timing message with new output

2. **src/sddp/mod.rs** (simulate function):
   - Remove `log::info!` calls inside `simulate()`
   - Return simulation timing to caller for display

### Phase 2: Fix Table Rendering (T-028)

1. Remove `render_table_header()` call from `train_with_display()`
2. Let `render_iteration()` handle table header on first iteration only
3. Ensure consistent behavior across all table-based renderers

### Phase 3: Fix Statistics Alignment (T-029)

1. Fix `render_stats_continuation()` in AdvancedRenderer
2. Ensure continuation row uses proper column spanning
3. Consider alternative layout (inline stats instead of continuation row)

### Phase 4: Fix Progress Bar and Minor Issues (T-030)

1. Add `stdout.flush()` after minimal progress output
2. Fix std dev calculation in training summary
3. Add proper newlines between sections

## Estimated Effort

- **Sprint**: 1 sprint (focused bug fixes)
- **Story Points**: 13 total
- **Duration**: 2-3 days

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking automation profile | Medium | High | Test JSON output before/after |
| Regression in test suite | Low | Medium | Run full test suite after each ticket |
| Terminal compatibility issues | Low | Medium | Test on multiple terminal types |

## Definition of Done

- [x] All 6 identified issues fixed
- [x] Manual testing with all 4 display profiles
- [x] All 749 existing tests pass
- [x] No new clippy warnings
- [x] Code formatted with rustfmt

## Status: ✅ COMPLETE

**Sprint 01 Summary:**
All 5 tickets completed successfully in one implementation session:

- **T-026**: Removed legacy logging from lib.rs (greeting, file reading, total time)
- **T-027**: Removed legacy logging from simulate, added `simulate_with_display()` 
- **T-028**: Fixed duplicate table header by removing explicit call in `train_with_display()`
- **T-029**: Fixed statistics row alignment with proper width calculations
- **T-030**: Fixed negative std dev, added trailing newlines, fixed progress bar termination

**Final Output Quality:**
- Clean, professional output across all profiles
- No `[INFO]` messages during training/simulation
- Proper spacing and alignment throughout
- All display profiles working correctly
- Zero test regressions
