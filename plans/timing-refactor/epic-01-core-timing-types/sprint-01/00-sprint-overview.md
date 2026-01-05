# Sprint 1: Core Types

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Duration**: 1 week

## Goals

- Define all new timing structs in `src/timing/`
- Implement aggregation and conversion methods
- Full test coverage for new types

## Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| [T-001](./ticket-001-trajectory-timing.md) | Create TrajectoryTiming struct | 2 | None |
| [T-002](./ticket-002-forward-timing.md) | Create ForwardTiming hierarchy | 3 | T-001 |
| [T-003](./ticket-003-backward-timing.md) | Create BackwardTiming hierarchy | 2 | None |
| [T-004](./ticket-004-iteration-timing.md) | Create IterationTiming struct | 2 | T-002, T-003 |
| [T-005](./ticket-005-aggregation-methods.md) | Implement aggregation methods | 3 | T-002 |
| [T-006](./ticket-006-output-types.md) | Create output conversion types | 2 | T-004 |
| [T-007](./ticket-007-unit-tests.md) | Add comprehensive unit tests | 2 | T-006 |

## Parallelization

```
T-001 ──────┐
            ├──► T-002 ──► T-004 ──► T-006 ──► T-007
T-003 ──────┘       │
                    └──► T-005 ──────────────────┘
```

- T-001 and T-003 can be done in parallel
- T-005 can be done in parallel with T-004, T-006
- T-007 waits for all others

## Definition of Done

- [ ] All tickets complete
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] No clippy warnings on new code
- [ ] Doc comments on all public types
