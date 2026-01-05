# Epic 1: Core Timing Types

> **Duration**: 1 week  
> **Depends on**: None  
> **Enables**: Epic 2, Epic 3

## Summary

Define the new hierarchical timing types in `src/timing/` that will replace all legacy timing structs. This epic establishes the foundation - no integration with existing code yet.

## Scope

### Included

- Define `IterationTiming` struct (top-level)
- Define `ForwardTiming`, `ForwardPreprocessingTiming`, `ForwardParallelTiming`, `ForwardPostprocessingTiming`
- Define `BackwardTiming`, `BackwardPhase1Timing`, `BackwardPhase2Timing`, `BackwardPhase3Timing`
- Define `TrajectoryTiming` (per-trajectory accumulator)
- Define `TrainingTiming` (full training run)
- Define output conversion types (plain `Duration` versions)
- Implement `compute_aggregates()` for parallel section statistics
- Implement `to_output()` conversion methods
- Add unit tests for all new types

### Excluded

- Integration with forward/backward pass (Epic 2, 3)
- Training loop changes (Epic 4)
- Output writer changes (Epic 5)
- Removal of legacy types (Epic 4, 5)

## Acceptance Criteria

- [ ] All new timing structs compile and have doc comments
- [ ] `IterationTiming::new(num_forward_passes)` preallocates trajectory Vec
- [ ] `ForwardParallelTiming::compute_aggregates()` calculates avg, max, overhead
- [ ] `to_output()` methods convert Cell<Duration> to Duration
- [ ] Unit tests cover: construction, aggregation, conversion
- [ ] `cargo test` passes with new types

## Technical Approach

1. Create new files in `src/timing/`: `iteration.rs`, `forward.rs`, `backward.rs`, `trajectory.rs`, `output.rs`, `aggregation.rs`
2. Keep existing `guard.rs` (TimingGuard already works)
3. Update `mod.rs` to export new types
4. All accumulator types use `Cell<Duration>` for interior mutability
5. Output types use plain `Duration` for serialization

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/timing/iteration.rs` | Create |
| `src/timing/forward.rs` | Create |
| `src/timing/backward.rs` | Create |
| `src/timing/trajectory.rs` | Create |
| `src/timing/output.rs` | Create |
| `src/timing/mod.rs` | Modify (add exports) |

## Sprint Breakdown

### Sprint 1: Core Types

| Ticket | Title | Points |
|--------|-------|--------|
| T-001 | Create TrajectoryTiming struct | 2 |
| T-002 | Create ForwardTiming hierarchy | 3 |
| T-003 | Create BackwardTiming hierarchy | 2 |
| T-004 | Create IterationTiming struct | 2 |
| T-005 | Implement aggregation methods | 3 |
| T-006 | Create output conversion types | 2 |
| T-007 | Add unit tests | 2 |

**Total**: 16 points (~1 week)
