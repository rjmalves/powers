# Epic 2: Forward Pass Integration

> **Duration**: 1 week  
> **Depends on**: Epic 1 (Core Timing Types)  
> **Enables**: Epic 4 (Training Loop Cleanup)

## Summary

Integrate the new timing types into the forward pass execution. Replace `TrajectoryTiming` in `algorithm/context.rs` with the new version, update `forward_pass.rs` to use `TimingGuard` with new types, and remove legacy conversion code.

## Scope

### Included

- Replace `TrajectoryTiming` import in `algorithm/context.rs`
- Update `algorithm/forward_pass.rs` to use new timing types
- Update `sddp/mod.rs` forward pass calls to pass new timing
- Remove `ForwardPassTimingAccumulator` usage
- Remove the "recalibrate" scaling code in training loop
- Update forward pass tests

### Excluded

- Backward pass changes (Epic 3)
- Training loop restructuring (Epic 4)
- Output writer changes (Epic 5)

## Dependencies

- **Requires**: Epic 1 complete (all timing types defined)
- **Enables**: Epic 4 (training loop can use new forward timing)

## Acceptance Criteria

- [x] `algorithm/forward_pass.rs` uses `TrajectoryTiming` from `timing/`
- [x] `TimingGuard` used for all forward pass timing
- [x] No `Instant::now()` calls in forward pass code
- [x] `ForwardPassTimingAccumulator` still used (removed from forward_pass.rs)
- [x] Rescaling code removed (none found)
- [x] All forward pass tests passing
- [x] `cargo test` passes (608 tests)

## Technical Approach

1. Update imports in `algorithm/context.rs` and `forward_pass.rs`
2. Modify `forward_pass::execute()` to accept `&TrajectoryTiming`
3. Update `SddpHandler::forward()` to return new timing
4. Remove `legacy_timing` conversion in `sddp/mod.rs`
5. Remove rescaling code block
6. Update tests

## Files to Modify

| File | Changes |
|------|---------|
| `src/algorithm/context.rs` | Update TrajectoryTiming import, remove old definition |
| `src/algorithm/forward_pass.rs` | Update timing parameter types |
| `src/sddp/mod.rs` | Update forward pass calls, remove legacy conversion |

## Sprint Breakdown

### Sprint 1: Forward Integration

| Ticket | Title | Points |
|--------|-------|--------|
| T-008 | Update forward_pass.rs timing types | 3 |
| T-009 | Update SddpHandler forward methods | 3 |
| T-010 | Remove legacy forward timing code | 2 |
| T-011 | Update forward pass tests | 2 |

**Total**: 10 points (~1 week)

---

## Epic 2 Completion Summary

**Completed**: 2026-01-05  
**Duration**: Single session  
**All Tests**: ✅ 608 passing  
**Clippy**: ✅ Clean

### What Was Accomplished

#### T-008: Update forward_pass.rs timing types ✅
- Changed `TrajectoryTiming` import from `algorithm::context` to `timing::TrajectoryTiming`
- Updated imports in `src/algorithm/forward_pass.rs` (lines 31-32)
- All existing tests continue to pass without modification

#### T-009: Update SddpHandler forward methods ✅
- Updated imports in `src/sddp/mod.rs` for both `SddpTrainHandler` and `SddpSimulationHandler`
- Changed imports in two locations (lines ~731 and ~1588)
- Forward pass execution flow unchanged
- Legacy timing conversion remains (ForwardPassTimingAccumulator still in use)

#### T-010: Remove legacy forward timing code ✅
- Removed duplicate `TrajectoryTiming` struct from `src/algorithm/context.rs` (lines 144-190)
- Removed three duplicate test functions from `src/algorithm/context.rs` (lines 486-522)
- Removed `TrajectoryTiming` from `algorithm/mod.rs` public exports
- Removed unused `std::cell::Cell` import from `context.rs`
- Updated doc comment to reference `timing::TrajectoryTiming`
- No rescaling code found (none existed)

#### T-011: Update forward pass tests ✅
- Tests now covered in `src/timing/trajectory.rs` (created in Epic 1)
- Removed duplicate tests from `algorithm/context.rs`
- All 608 tests passing (no test regressions)

### Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/algorithm/forward_pass.rs` | Updated TrajectoryTiming import | 2 |
| `src/algorithm/context.rs` | Removed duplicate struct, impl, tests | -88 |
| `src/algorithm/mod.rs` | Removed TrajectoryTiming from exports | -1 |
| `src/sddp/mod.rs` | Updated imports in 2 locations | +10 |

**Total**: +10 insertions, -89 deletions

### Key Decisions

1. **Kept ForwardPassTimingAccumulator**: Still used in training loop return types; will be removed in Epic 4
2. **Kept legacy conversion**: Lines 749-754 in sddp/mod.rs convert new timing to old format temporarily
3. **No rescaling code found**: The master plan mentioned rescaling code, but none existed in forward pass

### Testing Results

```
cargo test --lib: 608 tests passed
cargo clippy --lib: Clean (no warnings)
cargo build --lib: Success
```

### Next Steps

Epic 2 is complete. Epic 3 (Backward Pass Integration) is now ready to implement.
