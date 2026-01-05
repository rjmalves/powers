# Epic 4: Training Loop Cleanup

> **Duration**: 1 week  
> **Depends on**: Epic 2, Epic 3  
> **Enables**: Epic 5 (Output & Cleanup)

## Summary

Restructure the training loop in `sddp/mod.rs` to achieve "zero timing pollution". All timing code should be encapsulated in the timing module or use `TimingGuard` RAII patterns. Remove all `Instant::now()` / `.elapsed()` patterns from the training loop.

## Scope

### Included

- Create `IterationTiming` at iteration start
- Replace all `Instant::now()` with `TimingGuard`
- Remove all timing-related local variables from training loop
- Update `IterationResult` to use new timing output types
- Remove legacy timing structs from `sddp/mod.rs`
- Remove `ForwardPassTiming`, `BackwardPassTiming` from `sddp/mod.rs`
- Remove `ForwardPassTimingAccumulator`, `BackwardPassTimingAccumulator`
- Remove `BackwardPhase1Timing`, `BranchingsTiming`, `StepTiming`

### Excluded

- Output writer changes (Epic 5)

## Dependencies

- **Requires**: Epic 2, Epic 3 complete
- **Enables**: Epic 5 (output writers can use new types)

## Acceptance Criteria

- [ ] `grep -n "Instant::now" src/sddp/mod.rs` returns no results
- [ ] `grep -n "elapsed()" src/sddp/mod.rs` returns no results
- [ ] All timing structs removed from `sddp/mod.rs`
- [ ] Training loop uses `IterationTiming` exclusively
- [ ] `IterationResult` uses `IterationTimingOutput`
- [ ] All SDDP tests passing
- [ ] `cargo test` passes

## Technical Approach

1. Create `IterationTiming` at iteration start
2. Replace timing blocks with `TimingGuard` patterns
3. Update `IterationResult` struct definition
4. Remove all legacy timing types from `sddp/mod.rs`
5. Verify with grep that no timing pollution remains

## Files to Modify

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Major refactor of training loop, remove structs |
| `src/sddp/builder.rs` | Update if any timing references exist |

## Sprint Breakdown

### Sprint 1: Training Loop Cleanup

| Ticket | Title | Points |
|--------|-------|--------|
| T-017 | Create IterationTiming in training loop | 2 |
| T-018 | Replace forward timing code with guards | 3 |
| T-019 | Replace backward timing code with guards | 3 |
| T-020 | Update IterationResult struct | 2 |
| T-021 | Remove legacy timing structs from sddp | 3 |
| T-022 | Verify zero timing pollution | 1 |

**Total**: 14 points (~1 week)
