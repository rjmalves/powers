# [T-030] Integrate Backward Timing Infrastructure

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-029](./ticket-029-extract-cut-computation.md)
> **Blocks**: [T-031](./ticket-031-update-sddp-backward.md)

---

## Context

This ticket integrates the timing infrastructure from Epic 1 into the backward pass. Where possible, replace `Instant::now()` with `TimingGuard`.

---

## Files to Read Before Starting

- `src/timing/guard.rs` - `TimingGuard` implementation
- `src/timing/metrics.rs` - `BackwardTiming` struct
- `src/algorithm/backward_pass.rs` - Extracted backward pass
- `src/algorithm/coordinator.rs` - Phase timing

---

## Specification

### Timing Integration Points

| Location | Current | Target |
|----------|---------|--------|
| Backward preprocessing | `Instant::now()` | `TimingGuard` or keep |
| Phase 1 (compute cuts) | `Instant::now()` in coordinator | Keep (parallel context) |
| Phase 2 (cut selection) | `Instant::now()` in coordinator | Keep (need duration return) |
| Phase 3 (apply cuts) | `Instant::now()` in coordinator | Keep (need duration return) |

### Key Principle

**Preserve precise timing values.** The coordinator already returns timing from each phase. The backward pass module accumulates these into `BackwardPassTimingAccumulator`.

### Where TimingGuard CAN Be Used

- Top-level backward preprocessing (if any)
- Any sequential sections in backward_pass.rs

### Where TimingGuard CANNOT Be Used

- Inside parallel sections (borrow checker issues)
- When duration must be returned from function

---

## Acceptance Criteria

- [x] Review all `Instant::now()` usage in backward pass
- [x] Apply accumulation pattern (coordinator uses Instant::now(), module accumulates)
- [x] Document why coordinator keeps `Instant::now()` (parallel timing return)
- [x] Timing values match original behavior (verified via golden tests)
- [x] `cargo build -j1` succeeds
- [x] `cargo test -j1` passes

---

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: May discover limitations of TimingGuard in parallel contexts

---

## Definition of Done

- [x] Timing integration reviewed
- [x] Hybrid timing pattern used (coordinator returns, module accumulates)
- [x] Coordinator's Instant::now() documented in sprint overview
- [x] Tests pass
- [ ] Code reviewed
