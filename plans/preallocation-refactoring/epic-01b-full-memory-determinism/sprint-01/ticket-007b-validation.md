# [TICKET-007b] Validation and testing

> **Epic**: [Epic 1b: Full Memory Determinism](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: All previous tickets  
> **Blocks**: None

## Context

### Background

After completing the deterministic slot implementation, we need to validate:
1. **Correctness**: Results identical to before
2. **Memory**: Zero dynamic allocations during training
3. **Performance**: No regression (maintain ≥23% improvement)
4. **Simplicity**: Code is simpler than before

## Files to Read Before Starting

- `src/subproblem.rs` - Verify slot tracking removed
- `src/cut.rs` - Verify slot_index field works

## Specification

### Validation 1: Correctness

Run both examples and verify identical results:

```bash
# Example 01 - Deterministic (no cut selection)
cargo run --release -- run examples/01-deterministic

# Example 07 - With cut selection
cargo run --release -- run examples/07-par-model-with-inflow-state
```

Compare output to baseline (before Epic 1b changes):
- Lower bound should be identical
- Convergence should follow same pattern
- Iteration count should match

### Validation 2: Memory Determinism

Run with valgrind massif to verify no allocations during training:

```bash
cargo build --release

valgrind --tool=massif --massif-out-file=massif.out \
    ./target/release/powers run examples/07-par-model-with-inflow-state

ms_print massif.out | head -100
```

**Expected**: Flat memory profile after initialization. No allocation spikes during iterations.

### Validation 3: Performance

Run performance benchmark:

```bash
hyperfine --warmup 2 --runs 5 \
    'cargo run --release -- run examples/07-par-model-with-inflow-state'
```

**Expected**: ≥23% improvement maintained (compared to pre-Epic-1 baseline).

### Validation 4: Code Simplicity

Verify removed complexity:

```bash
# Should return 0 results (fields removed)
grep -c "free_cut_slots\|cut_slot_to_id\|next_available_cut_slot" src/subproblem.rs

# Should return 0 results (method removed)
grep -c "allocate_cut_slot" src/subproblem.rs
```

Count lines of code:
```bash
wc -l src/subproblem.rs
# Compare to before Epic 1b - should be ~50 lines fewer
```

### Validation 5: Panic on Overflow

Verify that exceeding preallocated slots panics (not falls back):

```rust
#[test]
#[should_panic(expected = "exceeds preallocated count")]
fn test_slot_overflow_panics() {
    // Create subproblem with 10 slots
    // Try to add cut for iteration 100 → should panic
}
```

## Acceptance Criteria

- [x] Example 01 produces identical results
- [x] Example 07 produces identical results
- [x] Valgrind massif shows flat memory profile during training
- [x] Performance ≥23% improvement maintained
- [x] Slot tracking code fully removed
- [x] Overflow panics (not falls back)

## Implementation Guide

### Step 1: Run correctness tests

```bash
cargo run --release -- run examples/01-deterministic 2>&1 | tee output_01.txt
cargo run --release -- run examples/07-par-model-with-inflow-state 2>&1 | tee output_07.txt
```

Compare to baseline outputs.

### Step 2: Run memory profiling

```bash
cargo build --release
valgrind --tool=massif --massif-out-file=massif.out \
    ./target/release/powers run examples/07-par-model-with-inflow-state
ms_print massif.out > massif_report.txt
```

Examine for allocation spikes during training.

### Step 3: Run performance benchmark

```bash
hyperfine --warmup 2 --runs 5 \
    'cargo run --release -- run examples/07-par-model-with-inflow-state'
```

Record timing.

### Step 4: Verify code cleanup

```bash
# Verify removed items
grep -E "free_cut_slots|cut_slot_to_id|next_available_cut_slot|allocate_cut_slot" src/subproblem.rs
# Should return nothing

# Line count
wc -l src/subproblem.rs
```

### Step 5: Document results

Create a brief summary of:
- Baseline vs. new performance
- Memory profile characteristics
- Code complexity reduction

## Testing Requirements

### Manual Tests

- [ ] Example 01 runs successfully
- [ ] Example 07 runs successfully
- [ ] Memory profile is flat

### Automated Tests

- [ ] All existing tests pass (`cargo test`)
- [ ] New unit tests for deterministic slot calculation pass

## Documentation Requirements

- [ ] Update README.md in plan directory with Epic 1b completion status
- [ ] Add notes about deterministic slot formula to code comments

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Mostly running commands and verifying output

## Definition of Done

- [x] All validations pass
- [x] Results documented
- [x] Epic 1b marked complete in README.md
