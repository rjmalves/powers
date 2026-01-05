# Timing Infrastructure Refactor

Consolidate ~15 scattered timing structs into a unified, hierarchical timing system with zero pollution of business logic.

## Quick Links

- [Master Plan](./00-master-plan.md) - Architecture, schema definitions, and design decisions

## Epics

### Epic 1: Core Timing Types (Week 1) ✅ COMPLETE
- [x] [Epic Overview](./epic-01-core-timing-types/00-epic-overview.md)
- [x] [Sprint 1: Core Types](./epic-01-core-timing-types/sprint-01/00-sprint-overview.md)
  - [x] [T-001: Create TrajectoryTiming struct](./epic-01-core-timing-types/sprint-01/ticket-001-trajectory-timing.md)
  - [x] [T-002: Create ForwardTiming hierarchy](./epic-01-core-timing-types/sprint-01/ticket-002-forward-timing.md)
  - [x] [T-003: Create BackwardTiming hierarchy](./epic-01-core-timing-types/sprint-01/ticket-003-backward-timing.md)
  - [x] [T-004: Create IterationTiming struct](./epic-01-core-timing-types/sprint-01/ticket-004-iteration-timing.md)
  - [x] [T-005: Implement aggregation methods](./epic-01-core-timing-types/sprint-01/ticket-005-aggregation-methods.md)
  - [x] [T-006: Create output conversion types](./epic-01-core-timing-types/sprint-01/ticket-006-output-types.md)
  - [x] [T-007: Add comprehensive unit tests](./epic-01-core-timing-types/sprint-01/ticket-007-unit-tests.md)

**Status**: All 7 tickets complete. 42 new tests added. All 611 tests passing. Clippy clean.

### Epic 2: Forward Pass Integration (Week 2) ✅ COMPLETE
- [x] [Epic Overview](./epic-02-forward-pass-integration/00-epic-overview.md)
- [x] [Sprint 1: Forward Integration](./epic-02-forward-pass-integration/sprint-01/00-sprint-overview.md)
  - [x] [T-008: Update forward_pass.rs timing types](./epic-02-forward-pass-integration/sprint-01/ticket-008-update-forward-pass-timing-types.md)
  - [x] [T-009: Update SddpHandler forward methods](./epic-02-forward-pass-integration/sprint-01/ticket-009-update-sddp-handler-forward-methods.md)
  - [x] [T-010: Remove legacy forward timing code](./epic-02-forward-pass-integration/sprint-01/ticket-010-remove-legacy-forward-timing.md)
  - [x] [T-011: Update forward pass tests](./epic-02-forward-pass-integration/sprint-01/ticket-011-update-forward-pass-tests.md)

**Status**: All 4 tickets complete. All 608 tests passing. Clippy clean.

### Epic 3: Backward Pass Integration (Week 3) ✅ COMPLETE
- [x] [Epic Overview](./epic-03-backward-pass-integration/00-epic-overview.md)
- [x] [Sprint 1: Backward Integration](./epic-03-backward-pass-integration/sprint-01/00-sprint-overview.md)
  - [x] [T-012: Update backward_pass.rs timing types](./epic-03-backward-pass-integration/sprint-01/ticket-012-update-backward-pass-timing-types.md)
  - [x] [T-013: Update processor.rs timing](./epic-03-backward-pass-integration/sprint-01/ticket-013-update-processor-timing.md)
  - [x] [T-014: Update coordinator.rs phase timing](./epic-03-backward-pass-integration/sprint-01/ticket-014-update-coordinator-phase-timing.md)
  - [x] [T-015: Remove legacy backward timing types](./epic-03-backward-pass-integration/sprint-01/ticket-015-remove-legacy-backward-timing.md)
  - [x] [T-016: Update backward pass tests](./epic-03-backward-pass-integration/sprint-01/ticket-016-update-backward-pass-tests.md)

**Status**: All 5 tickets complete. All 608 tests passing. Clippy clean.

### Epic 4: Training Loop Cleanup (Week 4)
- [ ] [Epic Overview](./epic-04-training-loop-cleanup/00-epic-overview.md)
- [ ] [Sprint 1: Training Loop Cleanup](./epic-04-training-loop-cleanup/sprint-01/00-sprint-overview.md)
  - [ ] [T-017: Create IterationTiming in training loop](./epic-04-training-loop-cleanup/sprint-01/ticket-017-create-iteration-timing.md)
  - [ ] [T-018: Replace forward timing code with guards](./epic-04-training-loop-cleanup/sprint-01/ticket-018-replace-forward-timing.md)
  - [ ] [T-019: Replace backward timing code with guards](./epic-04-training-loop-cleanup/sprint-01/ticket-019-replace-backward-timing.md)
  - [ ] [T-020: Update IterationResult struct](./epic-04-training-loop-cleanup/sprint-01/ticket-020-update-iteration-result.md)
  - [ ] [T-021: Remove legacy timing structs from sddp](./epic-04-training-loop-cleanup/sprint-01/ticket-021-remove-legacy-timing.md)
  - [ ] [T-022: Verify zero timing pollution](./epic-04-training-loop-cleanup/sprint-01/ticket-022-verify-zero-pollution.md)

**Status**: 🔜 Ready to implement. All 6 tickets defined.

### Epic 5: Output & Cleanup (Week 5)
- [ ] [Epic Overview](./epic-05-output-and-cleanup/00-epic-overview.md)
- [ ] [Sprint 1: Output & Cleanup](./epic-05-output-and-cleanup/sprint-01/00-sprint-overview.md)
  - [ ] [T-023: Update CSV training output](./epic-05-output-and-cleanup/sprint-01/ticket-023-update-csv-output.md)
  - [ ] [T-024: Update Parquet training output](./epic-05-output-and-cleanup/sprint-01/ticket-024-update-parquet-output.md)
  - [ ] [T-025: Remove old timing/metrics.rs types](./epic-05-output-and-cleanup/sprint-01/ticket-025-remove-old-metrics.md)
  - [ ] [T-026: Dead code cleanup](./epic-05-output-and-cleanup/sprint-01/ticket-026-dead-code-cleanup.md)
  - [ ] [T-027: Update documentation](./epic-05-output-and-cleanup/sprint-01/ticket-027-update-documentation.md)
  - [ ] [T-028: Final verification](./epic-05-output-and-cleanup/sprint-01/ticket-028-final-verification.md)

**Status**: ⏸️ Blocked by Epic 4. All 6 tickets defined.

## Dependency Graph

```
Epic 1: Core Timing Types ✅
         │
         ├──────────────────┐
         ▼                  ▼
Epic 2: Forward ✅    Epic 3: Backward ✅
(4 tickets)           (5 tickets)
         │                  │
         └────────┬─────────┘
                  ▼
         Epic 4: Training Loop
         (6 tickets) 🔜
                  │
                  ▼
         Epic 5: Output & Cleanup
         (6 tickets) ⏸️
```

## Success Criteria

- [x] All timing types in `src/timing/` (Epic 1 ✅)
- [x] Zero `Instant::now()` in forward pass (Epic 2 ✅)
- [x] Zero `Instant::now()` in backward pass (Epic 3 ✅)
- [ ] Zero `Instant::now()` in training loop (Epic 4)
- [ ] No timing redistribution code (Epic 5)
- [ ] All tests passing
- [ ] No performance regression

## Progress

| Epic | Status | Tickets | Tests | Notes |
|------|--------|---------|-------|-------|
| 1. Core Types | ✅ Complete | 7/7 | 42 | All new types defined |
| 2. Forward Integration | ✅ Complete | 4/4 | 608 | All tickets done |
| 3. Backward Integration | ✅ Complete | 5/5 | 608 | All tickets done |
| 4. Training Loop | 🔜 Ready | 0/6 | - | Tickets T-017 to T-022 defined |
| 5. Output & Cleanup | ⏸️ Blocked | 0/6 | - | Tickets T-023 to T-028 defined |

## Ticket Summary

| ID | Title | Epic | Points | Status |
|----|-------|------|--------|--------|
| T-001 | Create TrajectoryTiming struct | 1 | 2 | ✅ |
| T-002 | Create ForwardTiming hierarchy | 1 | 3 | ✅ |
| T-003 | Create BackwardTiming hierarchy | 1 | 2 | ✅ |
| T-004 | Create IterationTiming struct | 1 | 2 | ✅ |
| T-005 | Implement aggregation methods | 1 | 3 | ✅ |
| T-006 | Create output conversion types | 1 | 2 | ✅ |
| T-007 | Add comprehensive unit tests | 1 | 3 | ✅ |
| T-008 | Update forward_pass.rs timing types | 2 | 3 | ✅ |
| T-009 | Update SddpHandler forward methods | 2 | 3 | ✅ |
| T-010 | Remove legacy forward timing code | 2 | 2 | ✅ |
| T-011 | Update forward pass tests | 2 | 2 | ✅ |
| T-012 | Update backward_pass.rs timing types | 3 | 3 | ✅ |
| T-013 | Update processor.rs timing | 3 | 2 | ✅ |
| T-014 | Update coordinator.rs phase timing | 3 | 3 | ✅ |
| T-015 | Remove legacy backward timing types | 3 | 2 | ✅ |
| T-016 | Update backward pass tests | 3 | 2 | ✅ |
| T-017 | Create IterationTiming in training loop | 4 | 2 | ⬜ |
| T-018 | Replace forward timing code with guards | 4 | 3 | ⬜ |
| T-019 | Replace backward timing code with guards | 4 | 3 | ⬜ |
| T-020 | Update IterationResult struct | 4 | 2 | ⬜ |
| T-021 | Remove legacy timing structs from sddp | 4 | 3 | ⬜ |
| T-022 | Verify zero timing pollution | 4 | 1 | ⬜ |
| T-023 | Update CSV training output | 5 | 2 | ⬜ |
| T-024 | Update Parquet training output | 5 | 2 | ⬜ |
| T-025 | Remove old timing/metrics.rs types | 5 | 2 | ⬜ |
| T-026 | Dead code cleanup | 5 | 1 | ⬜ |
| T-027 | Update documentation | 5 | 2 | ⬜ |
| T-028 | Final verification | 5 | 1 | ⬜ |

## Estimated Timeline

| Week | Epic | Deliverable | Status |
|------|------|-------------|--------|
| 1 | Core Timing Types | New timing structs with tests | ✅ Complete |
| 2 | Forward Pass Integration | Forward pass using new timing | ✅ Complete |
| 3 | Backward Pass Integration | Backward pass using new timing | ✅ Complete |
| 4 | Training Loop Cleanup | Zero timing pollution in training loop | 🔜 Ready |
| 5 | Output & Cleanup | Complete refactor, output writers updated | ⏸️ Blocked |

**Total Duration**: 5 weeks (3 weeks complete, 2 weeks remaining)
**Total Tickets**: 28 (16 complete, 12 remaining)
**Total Story Points**: 65 (38 complete, 27 remaining)
