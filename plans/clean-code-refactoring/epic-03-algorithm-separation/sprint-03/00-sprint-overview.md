# Sprint 3: Backward Pass Extraction

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Duration**: 1.5-2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ ARCHITECTURAL DECISION: Timing Separation

All timing in this sprint follows the pattern from T-021:

**Timing is passed as a separate parameter, not inside context structs.**

```rust
// CORRECT:
pub fn execute<P: BackwardStageProcessor>(
    processor: &mut P,
    ctx: &BackwardPassContext,           // No timing inside
    timing: &BackwardPassTimingAccumulator,  // Separate, uses Cell<Duration>
) -> Result<BackwardPassResult, String> {
    {
        let _guard = TimingGuard::new(&timing.model_preprocessing);
        // Can access processor and ctx without borrow conflicts!
    }
}
```

---

## Goals

1. **Primary**: Extract backward pass loop to `src/algorithm/backward_pass.rs`
2. **Primary**: Extract cut computation to `src/algorithm/cut_computation.rs`
3. **Primary**: Integrate timing via `TimingGuard` (timing separate from context)
4. **Validation**: Bit-for-bit identical outputs, `sddp/mod.rs` reduced significantly

---

## Prerequisites

From Sprint 2:
- ✅ `ParallelHandlerCoordinator` implemented
- ✅ `BackwardStageProcessor` trait defined
- ✅ Coordinator integrated into `train()`
- ✅ Timing separation applied to all contexts

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-028](./ticket-028-extract-backward-loop.md) | Extract backward pass loop | 5 | Yes | Sprint 2 | ⬜ |
| [T-029](./ticket-029-extract-cut-computation.md) | Extract cut computation logic | 3 | Yes | T-028 | ⬜ |
| [T-030](./ticket-030-backward-timing.md) | Verify backward timing integration | 3 | Yes | T-029 | ⬜ |
| [T-031](./ticket-031-update-sddp-backward.md) | Update sddp/mod.rs to use backward_pass module | 3 | Yes | T-030 | ⬜ |
| [T-032](./ticket-032-verify-timing.md) | Verify timing integration end-to-end | 2 | Yes | T-031 | ⬜ |

**Total Points**: 16

---

## Timing Pattern

All timing in backward pass uses this pattern:

```rust
/// Timing accumulator with Cell for TimingGuard compatibility.
#[derive(Debug, Clone, Default)]
pub struct BackwardPassTimingAccumulator {
    pub preprocessing: Cell<Duration>,
    pub model_preprocessing: Cell<Duration>,
    pub solver: Cell<Duration>,
    pub cut_selection: Cell<Duration>,
    // ... other fields
}

// Usage with TimingGuard:
{
    let _guard = TimingGuard::new(&timing.model_preprocessing);
    // Work here is timed
    processor.compute_cuts_parallel(&stage_ctx)?;
}
```

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| `sddp/mod.rs` lines | 3,827 | ~3,300 (-500) |
| Backward pass location | inline in train() | `algorithm/backward_pass.rs` |
| Timing mechanism | `Instant::now()` | `TimingGuard` |
| Feature-gating | None | Works via TimingGuard |

---

## Verification Protocol

After EVERY ticket:

```bash
cargo build -j1 && RUST_TEST_THREADS=1 cargo test -j1 && ./scripts/golden-tests.sh verify
```

Verify timing feature-gating:
```bash
cargo build -j1 --features timing
cargo build -j1  # without timing feature
```

---

## Definition of Done

- [ ] Backward pass extracted to `algorithm/backward_pass.rs`
- [ ] Cut computation in `algorithm/cut_computation.rs`
- [ ] All timing uses `TimingGuard` (no raw `Instant::now()`)
- [ ] Timing passed as separate parameter (not in context)
- [ ] Feature-gating works
- [ ] `sddp/mod.rs` reduced by ~500 lines
- [ ] Golden tests pass
- [ ] Benchmarks within 5%
- [ ] Code reviewed
