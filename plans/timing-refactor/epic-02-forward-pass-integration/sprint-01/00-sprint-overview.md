# Sprint 1: Forward Pass Integration

> **Epic**: [Epic 2: Forward Pass Integration](../00-epic-overview.md)  
> **Duration**: 1 week  
> **Total Points**: 10

## Goals

- Replace `TrajectoryTiming` in `algorithm/context.rs` with import from `timing/`
- Update `forward_pass.rs` to use new timing types consistently
- Update `SddpHandler::forward()` to work with new timing
- Remove legacy `ForwardPassTimingAccumulator` and conversion code
- Remove the timing "recalibrate" rescaling logic

## Tickets

| ID | Title | Points | Dependencies | Assignable |
|----|-------|--------|--------------|------------|
| T-008 | [Update forward_pass.rs timing types](./ticket-008-update-forward-pass-timing-types.md) | 3 | Epic 1 complete | Yes |
| T-009 | [Update SddpHandler forward methods](./ticket-009-update-sddp-handler-forward-methods.md) | 3 | T-008 | Yes |
| T-010 | [Remove legacy forward timing code](./ticket-010-remove-legacy-forward-timing.md) | 2 | T-009 | Yes |
| T-011 | [Update forward pass tests](./ticket-011-update-forward-pass-tests.md) | 2 | T-010 | Yes |

## Dependencies

- **From Epic 1**: `TrajectoryTiming` struct in `src/timing/trajectory.rs`
- **From Epic 1**: `NewForwardTiming` struct in `src/timing/forward.rs`

## Key Files

Files to modify:
- `src/algorithm/context.rs` - Remove `TrajectoryTiming` definition, update imports
- `src/algorithm/forward_pass.rs` - Update timing parameter and aggregation
- `src/sddp/mod.rs` - Update forward pass calls, remove legacy conversion

Files to read before starting:
- `src/timing/trajectory.rs` - New TrajectoryTiming definition
- `src/timing/forward.rs` - NewForwardTiming hierarchy
- `plans/timing-refactor/00-master-plan.md` - Architecture overview

## Risks

- **Import path changes**: Ensure all references to `TrajectoryTiming` are updated
- **Method signature compatibility**: The new `TrajectoryTiming` has same methods so should be compatible

## Definition of Done

- [ ] All 4 tickets complete
- [ ] No `TrajectoryTiming` definition in `algorithm/context.rs`
- [ ] `forward_pass::execute()` uses `timing::TrajectoryTiming`
- [ ] Legacy conversion code removed from `sddp/mod.rs`
- [ ] Rescaling code removed
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean
