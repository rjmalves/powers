# Sprint 1: Training Loop Cleanup

> **Epic**: [Epic 4: Training Loop Cleanup](../00-epic-overview.md)  
> **Duration**: 1 week  
> **Total Points**: 14

## Goals

- Create `NewIterationTiming` at iteration start
- Replace all `Instant::now()` / `.elapsed()` patterns with `TimingGuard`
- Update `IterationResult` to use `IterationTimingOutput`
- Remove all legacy timing structs from `sddp/mod.rs`
- Verify zero timing pollution in training loop

## Tickets

| ID | Title | Points | Dependencies | Assignable |
|----|-------|--------|--------------|------------|
| T-017 | [Create IterationTiming in training loop](./ticket-017-create-iteration-timing.md) | 2 | Epic 2, Epic 3 | Yes |
| T-018 | [Replace forward timing code with guards](./ticket-018-replace-forward-timing.md) | 3 | T-017 | Yes |
| T-019 | [Replace backward timing code with guards](./ticket-019-replace-backward-timing.md) | 3 | T-017 | Yes |
| T-020 | [Update IterationResult struct](./ticket-020-update-iteration-result.md) | 2 | T-018, T-019 | Yes |
| T-021 | [Remove legacy timing structs from sddp](./ticket-021-remove-legacy-timing.md) | 3 | T-020 | Yes |
| T-022 | [Verify zero timing pollution](./ticket-022-verify-zero-pollution.md) | 1 | T-021 | Yes |

## Dependencies

- **From Epic 2**: Forward pass integrated with new timing types
- **From Epic 3**: Backward pass integrated with new timing types
- **From Epic 1**: `NewIterationTiming`, `ForwardTimingOutput`, `BackwardTimingOutput`

## Key Files

Files to modify:
- `src/sddp/mod.rs` - Training loop, timing struct removal
- `src/sddp/instance.rs` - May need updates if it references timing types

Files to read before starting:
- `src/timing/iteration.rs` - `NewIterationTiming` struct
- `src/timing/output.rs` - Output types (`IterationTimingOutput`)
- `plans/timing-refactor/00-master-plan.md` - Training loop flow diagram

## Legacy Types to Remove

| Type | Location | Status |
|------|----------|--------|
| `ForwardPassTiming` | `sddp/mod.rs:36` | Remove |
| `BackwardPassTiming` | `sddp/mod.rs:46` | Remove |
| `ForwardPassTimingAccumulator` | `sddp/mod.rs:58` | Remove |
| `BackwardPassTimingAccumulator` | `sddp/mod.rs:138` | Remove |
| `BranchingsTiming` | Keep (used by handler) | Keep |
| `BackwardPhase1Timing` | Keep (new timing module) | Keep |

## Risks

- **Breaking changes**: `IterationResult` field changes affect output writers (Epic 5)
- **Parallel execution**: Ensure timing guards work correctly in parallel sections
- **Hidden timing code**: May find timing code in unexpected places

## Definition of Done

- [ ] All 6 tickets complete
- [ ] `grep -n "Instant::now" src/sddp/mod.rs` returns no results
- [ ] `grep -n "\.elapsed()" src/sddp/mod.rs` returns no results
- [ ] No `ForwardPassTiming` or `BackwardPassTiming` in `sddp/mod.rs`
- [ ] `IterationResult` uses `IterationTimingOutput`
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean
