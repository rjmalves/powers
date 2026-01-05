# [T-022] Verify zero timing pollution

> **Epic**: [Epic 4: Training Loop Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-021](./ticket-021-remove-legacy-timing.md)  
> **Blocks**: None (enables Epic 5)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Verify no timing pollution
- `plans/timing-refactor/00-master-plan.md` - Success metrics

## Context

### Background

This is the final verification ticket for Epic 4. It ensures that the "zero timing pollution" goal has been achieved - the training loop in `sddp/mod.rs` should have no direct timing code, only use of the new timing infrastructure.

### Success Criteria from Master Plan

- Zero `Instant::now()` in `sddp/mod.rs`
- Zero `.elapsed()` in `sddp/mod.rs`
- All timing types in `src/timing/`
- No timing redistribution (recalibrate/scaling code)

## Specification

### Verification Commands

Run these commands and verify expected output:

```bash
# No Instant::now() in sddp/mod.rs
grep -n "Instant::now" src/sddp/mod.rs
# Expected: no output (or only in BranchingsTiming internal code if kept)

# No .elapsed() in sddp/mod.rs  
grep -n "\.elapsed()" src/sddp/mod.rs
# Expected: no output (or only in BranchingsTiming internal code if kept)

# No legacy timing structs
grep -n "ForwardPassTiming\|BackwardPassTiming\|ForwardPassTimingAccumulator\|BackwardPassTimingAccumulator" src/sddp/mod.rs
# Expected: no output

# All timing types in timing module
grep -rn "pub struct.*Timing" src/ | grep -v "src/timing" | grep -v test
# Expected: only BranchingsTiming in sddp/mod.rs, RealizeUncertaintiesTiming in subproblem.rs
```

### Allowed Exceptions

| Pattern | Location | Reason |
|---------|----------|--------|
| `BranchingsTiming` | `sddp/mod.rs` | Used for first-stage bound evaluation |
| `RealizeUncertaintiesTiming` | `subproblem.rs` | Solver-level timing detail |
| `Instant::now` | Inside `BranchingsTiming` methods | Internal to timing struct |

## Acceptance Criteria

- [ ] `grep -n "Instant::now" src/sddp/mod.rs` returns no results (excluding allowed exceptions)
- [ ] `grep -n "\.elapsed()" src/sddp/mod.rs` returns no results (excluding allowed exceptions)
- [ ] No legacy timing structs in `sddp/mod.rs`
- [ ] `cargo clippy` clean
- [ ] `cargo test` passes
- [ ] Documentation of verification results

## Implementation Guide

### Step 1: Run verification commands

Execute all grep commands and capture output.

### Step 2: Document findings

Create a verification checklist:
```
Verification Date: YYYY-MM-DD
Verified By: [Name]

✓ No Instant::now() in training loop: PASS
✓ No .elapsed() in training loop: PASS
✓ Legacy timing structs removed: PASS
✓ All tests passing: PASS
✓ Clippy clean: PASS
```

### Step 3: Fix any violations

If any violations found:
1. Identify the offending code
2. Replace with appropriate timing infrastructure
3. Re-run verification

### Step 4: Run full test suite

```bash
cargo test -p powers-rs
cargo clippy -p powers-rs --all-targets
```

### Pitfalls to Avoid

- ⚠️ `BranchingsTiming` is allowed to use `Instant::now` internally
- ⚠️ Don't forget to check `src/sddp/builder.rs` and `src/sddp/instance.rs`
- ⚠️ Look for indirect timing via helper functions

## Testing Requirements

### Verification Tests

All verification commands must pass.

### Regression Tests

```bash
cargo test -p powers-rs
```

### Performance Tests

- Run a benchmark to confirm no performance regression
- Compare with pre-refactor baseline if available

## Documentation Requirements

- [ ] Document verification results (inline in this ticket or separate report)
- [ ] Confirm Epic 4 is complete

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Pure verification, no code changes expected
