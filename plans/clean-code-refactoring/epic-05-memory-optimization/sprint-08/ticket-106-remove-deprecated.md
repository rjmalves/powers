# [T-106] Remove Deprecated Allocation Code Paths

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8: Validation and Documentation](./00-sprint-overview.md)
> **Dependencies**: T-105
> **Blocks**: None

---

## Context

### Background

Throughout Epic 5, new zero-allocation APIs were created alongside existing allocation-heavy paths. Now that optimization is complete, deprecated paths should be removed.

### Known Deprecated Code

1. `uniform_prob_by_count()` - Replaced by `fill_uniform_probabilities()`
2. Old `CutData` allocation path (if still present)
3. Any `#[deprecated]` marked functions
4. Commented-out old code

## Specification

### Tasks

1. **Find all deprecated markers**
2. **Identify unused allocation functions**
3. **Remove deprecated code**
4. **Update any remaining callers**

### Process

1. Search for `#[deprecated]`
2. Search for `// DEPRECATED` or `// TODO: remove`
3. Search for `#[allow(dead_code)]` that may hide unused paths
4. Compile with warnings enabled to find unused code

## Acceptance Criteria

- [ ] All `#[deprecated]` functions removed
- [ ] No dead code related to allocation
- [ ] All tests still pass
- [ ] No functional changes

## Implementation Guide

### Suggested Approach

1. **Find deprecated code**:
   ```bash
   rg "#\[deprecated" src/
   rg "DEPRECATED" src/
   rg "TODO.*remove" src/
   ```

2. **Check for unused functions**:
   ```bash
   cargo build --release 2>&1 | grep "unused"
   ```

3. **Remove incrementally**:
   - Remove one function at a time
   - Run tests after each removal
   - Ensure no callers remain

4. **Update CHANGELOG**:
   ```markdown
   ## Removed
   
   - `uniform_prob_by_count()` - Use `fill_uniform_probabilities()` instead
   ```

### Key Files to Check

- `src/utils/mod.rs` - Probability functions
- `src/memory/buffers.rs` - Buffer APIs
- `src/state.rs` - State allocation
- `src/fcf.rs` - Cut handling

### Pitfalls to Avoid

- ⚠️ Ensure no external users depend on removed functions
- ⚠️ Check if functions are used in tests
- ⚠️ Don't remove code that's still needed

## Testing Requirements

- [ ] All tests pass after removal
- [ ] No compiler warnings about removed code

## Documentation Requirements

- [ ] Update CHANGELOG with removed items

## Effort Estimate

**Points**: 2
**Confidence**: High
