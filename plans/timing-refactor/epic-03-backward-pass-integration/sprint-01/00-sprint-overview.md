# Sprint 1: Backward Pass Integration

> **Epic**: [Epic 3: Backward Pass Integration](../00-epic-overview.md)  
> **Duration**: 1 week  
> **Total Points**: 12

## Goals

- Replace `BackwardPassTimingAccumulator` with `NewBackwardTiming` from timing module
- Update `processor.rs` timing types to align with new hierarchy
- Update `coordinator.rs` phase timing to use `TimingGuard` with new types
- Remove legacy timing structs from backward pass code
- Consolidate backward pass timing into single source of truth

## Tickets

| ID | Title | Points | Dependencies | Assignable |
|----|-------|--------|--------------|------------|
| T-012 | [Update backward_pass.rs timing types](./ticket-012-update-backward-pass-timing-types.md) | 3 | Epic 1 complete | Yes |
| T-013 | [Update processor.rs timing](./ticket-013-update-processor-timing.md) | 2 | T-012 | Yes |
| T-014 | [Update coordinator.rs phase timing](./ticket-014-update-coordinator-phase-timing.md) | 3 | T-013 | Yes |
| T-015 | [Remove legacy backward timing types](./ticket-015-remove-legacy-backward-timing.md) | 2 | T-014 | Yes |
| T-016 | [Update backward pass tests](./ticket-016-update-backward-pass-tests.md) | 2 | T-015 | Yes |

## Dependencies

- **From Epic 1**: `NewBackwardTiming` struct in `src/timing/backward.rs`
- **From Epic 1**: `BackwardPhase1Timing`, `BackwardPhase2Timing`, `BackwardPhase3Timing`

## Key Files

Files to modify:
- `src/algorithm/backward_pass.rs` - Replace `BackwardPassTimingAccumulator` with new types
- `src/algorithm/processor.rs` - Update `CutComputationTiming` and phase result types
- `src/algorithm/coordinator.rs` - Update phase timing to use new types
- `src/algorithm/context.rs` - Remove `BackwardStageTiming` (after migration)

Files to read before starting:
- `src/timing/backward.rs` - New backward timing hierarchy
- `src/timing/output.rs` - `BackwardTimingOutput` definition
- `plans/timing-refactor/00-master-plan.md` - Backward pass execution flow

## Key Mapping

Current types → New types:

| Current Location | Current Type | New Type |
|------------------|--------------|----------|
| `backward_pass.rs` | `BackwardPassTimingAccumulator` | `timing::NewBackwardTiming` |
| `backward_pass.rs` | `BackwardPassTimingSnapshot` | `timing::BackwardTimingOutput` |
| `processor.rs` | `CutComputationTiming` | Inline to `BackwardPhase1Timing` |
| `processor.rs` | `FirstStageTiming` | Keep (simple, local) |
| `context.rs` | `BackwardStageTiming` | Remove (use new types) |

## Risks

- **Phase timing accumulation**: New types accumulate with `Cell<Duration>`, old types use `Duration`
- **Trait signatures**: `BackwardStageProcessor` returns `CutComputationTiming` - may need update

## Definition of Done

- [ ] All 5 tickets complete
- [ ] No `BackwardPassTimingAccumulator` in `backward_pass.rs`
- [ ] `NewBackwardTiming` used for backward pass timing
- [ ] Phase timing uses new types with `TimingGuard`
- [ ] Legacy types removed
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean
