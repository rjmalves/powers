# [T-094] Remove `reuse_forward_basis()` Code Entirely

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-095
> **Priority**: 1 (Sprint 6 Follow-up)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Contains the function and its call sites
- `docs/HIGHS_WARM_START_INVESTIGATION.md` - Investigation findings
- `docs/DHAT_SPRINT6_ANALYSIS.md` - Validation that disabling it achieved 95% HFactor reduction

---

## Context

### Background

Sprint 6 DHAT analysis confirmed that `reuse_forward_basis()` was **counterproductive**. When this function was disabled, HFactor::setupGeneral allocations dropped from 39.58 GB to 2.00 GB (**95% reduction**).

The function attempted to warm-start backward branching solves by reusing the forward pass basis. However, because cuts are added between passes, the basis has mismatched row counts, triggering HiGHS "alien basis" handling which forces full factorization rebuilds.

### Current State

The function is currently **commented out** at line 1219-1221:
```rust
// Note: reuse_forward_basis temporarily disabled pending investigation
// reuse_forward_basis(
//     &mut subproblem_node.data,
//     _node_forward_realization,
// )?;
```

The function definition exists at lines 2455-2481.

### Why Remove Completely

1. **Proven counterproductive**: DHAT validated the hypothesis
2. **Dead code**: Currently commented out, will never be re-enabled
3. **Maintenance burden**: Code that exists but isn't used creates confusion
4. **Clean codebase**: Removing validates the Sprint 6 decision permanently

---

## Specification

### Tasks

1. Delete the `reuse_forward_basis()` function (lines 2455-2481)
2. Delete the commented-out call site (lines 1219-1221)
3. Remove the `_node_forward_realization` parameter prefix (make it `node_forward_realization` or remove if unused)
4. Remove any related imports if they become unused

### Behavior

- No behavioral changes expected (function was already disabled)
- All tests must continue to pass
- DHAT profile should remain unchanged

### Error Handling

- N/A (removing dead code)

---

## Acceptance Criteria

- [ ] `reuse_forward_basis()` function deleted from `src/sddp/mod.rs`
- [ ] Commented-out call site deleted
- [ ] No unused imports or dead code warnings introduced
- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes (all 567+ tests)
- [ ] `cargo clippy -- -D warnings` passes

---

## Implementation Guide

### Suggested Approach

1. Open `src/sddp/mod.rs`
2. Delete lines 2455-2481 (the `reuse_forward_basis` function)
3. Delete lines 1219-1221 (the commented-out call)
4. Check if `_node_forward_realization` parameter is used elsewhere in the function:
   - If used: remove the underscore prefix
   - If unused: consider if it should be removed (may be needed for other purposes)
5. Run `cargo build --release` to verify no compilation errors
6. Run `cargo clippy` to check for unused imports
7. Run `cargo test` to verify no regressions

### Key Files to Modify

- `src/sddp/mod.rs`: Delete function and call site

### Patterns to Follow

- Clean deletion without leaving orphaned comments
- Preserve any necessary context in commit message

### Pitfalls to Avoid

- ⚠️ Don't accidentally delete other functions nearby
- ⚠️ Verify the parameter `_node_forward_realization` isn't used elsewhere in the containing function
- ⚠️ Don't remove the `Realization` type import if it's used elsewhere

---

## Testing Requirements

### Compilation Tests

- [ ] `cargo build --release` succeeds
- [ ] `cargo clippy -- -D warnings` passes
- [ ] No new warnings introduced

### Unit Tests

- [ ] All existing tests pass (`cargo test`)

### Integration Tests

- [ ] Golden tests pass (numerical correctness preserved)

---

## Documentation Requirements

- [ ] Update `docs/HIGHS_WARM_START_INVESTIGATION.md` to note function was removed
- [ ] Commit message should reference Sprint 6 findings

---

## Dependencies

- **Blocked By**: None
- **Blocks**: T-095 (Document HiGHS basis reuse guidelines)
- **Related**: Sprint 6 T-087 (warm-start investigation)

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple code deletion with clear scope

---

## Definition of Done

- [ ] Function and call site deleted
- [ ] All tests passing
- [ ] No clippy warnings
- [ ] Documentation updated
- [ ] PR merged
