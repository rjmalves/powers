# [T-049] Verify FCF Refactoring End-to-End

> **Epic**: [Epic 4: State Simplification](../../00-epic-overview.md)
> **Sprint**: [Sprint 2: FCF Graph Wrapper Removal](./00-sprint-overview.md)
> **Dependencies**: [T-047](./ticket-047-update-coordinator-fcf-access.md), [T-048](./ticket-048-update-output-fcf-access.md)
> **Blocks**: None (sprint completion)

## Files to Read Before Starting

- `docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md` - Expected benefits
- Sprint 2 tickets (T-045 through T-048)

---

## Context

### Background

This ticket verifies the complete FCF Mutex removal refactoring. All code changes are complete; this ticket runs comprehensive verification.

### Expected State

- `Arc<Mutex<FutureCostFunction>>` replaced with `FutureCostFunction`
- All `.lock().unwrap()` calls removed
- Borrow checker enforces safe access

---

## Specification

### Verification Steps

1. **Full Build**: Clean build with all features
2. **Full Test Suite**: All tests pass
3. **Golden Tests**: Bit-for-bit identical outputs
4. **Performance Check**: No regression (expected slight improvement)
5. **Code Audit**: No remaining lock calls on FCF

---

## Acceptance Criteria

- [ ] `cargo build -j1` succeeds
- [ ] `cargo build -j1 --features timing` succeeds
- [ ] `RUST_TEST_THREADS=1 cargo test -j1` all tests pass
- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] No `.lock()` calls on FCF types remain
- [ ] No `Arc<Mutex<` on FCF types remain

---

## Implementation Guide

### Verification Commands

```bash
# 1. Clean build
cargo clean && cargo build -j1

# 2. Build with timing feature
cargo build -j1 --features timing

# 3. Run all tests
RUST_TEST_THREADS=1 cargo test -j1

# 4. Golden tests
./scripts/golden-tests.sh verify

# 5. Verify no remaining lock calls
grep -rn "\.lock()" src/ | grep -i "fcf\|future_cost"
# Should return empty

# 6. Verify no remaining Arc<Mutex<
grep -rn "Arc<Mutex<" src/ | grep -i "fcf\|FutureCostFunction"
# Should return empty
```

### Performance Verification

```bash
# Run example and compare timing
time cargo run --release -- run examples/01-hydro-scheduling

# Compare with pre-refactoring baseline if available
```

### Expected Performance Impact

From analysis:
- Lock acquisition/release: ~20-50 CPU cycles saved per `.lock()` call
- Memory barriers removed: slight improvement
- Cache line bouncing eliminated

Realistic expectation: **negligible to 1% improvement** (locks were never contended)

---

## Testing Requirements

### Full Test Suite

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All doc tests pass

### Golden Tests

- [ ] All example outputs bit-for-bit identical

### Code Audit

- [ ] `grep` confirms no remaining FCF lock patterns
- [ ] Manual review of key files confirms clean code

---

## Documentation Requirements

- [ ] Update CHANGELOG.md with refactoring note
- [ ] Document performance findings (if measurable)

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Verification only, no code changes

---

## Definition of Done

- [ ] Full build succeeds
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] No remaining lock patterns on FCF
- [ ] CHANGELOG updated
- [ ] Sprint 2 complete
