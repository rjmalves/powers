# [T-095] Document HiGHS Basis Reuse Guidelines

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: T-094
> **Blocks**: None
> **Priority**: 1 (Sprint 6 Follow-up)

## Files to Read Before Starting

- `docs/HIGHS_WARM_START_INVESTIGATION.md` - Current investigation document
- `docs/DHAT_SPRINT6_ANALYSIS.md` - Validation results
- `src/solver.rs` - HiGHS wrapper with `set_basis()` and `try_set_basis()` methods

---

## Context

### Background

The Sprint 6 investigation revealed that HiGHS basis reuse has specific requirements that were violated by `reuse_forward_basis()`. This ticket documents those requirements so future developers understand when basis reuse IS appropriate and when it causes problems.

### Key Learnings

1. **Alien Basis Handling**: When `setBasis()` receives a basis with different row count than the model, HiGHS treats it as "alien" and triggers `formSimplexLpBasisAndFactor()`, which rebuilds the full factorization
2. **Row Count Mismatch**: Adding cuts between forward and backward passes changes row count, invalidating the basis
3. **When Basis Reuse Works**: Same model dimensions, no structural changes between solves

---

## Specification

### Tasks

Update `docs/HIGHS_WARM_START_INVESTIGATION.md` with:

1. **"When to Use Basis Reuse" section** explaining valid use cases
2. **"When NOT to Use Basis Reuse" section** with anti-patterns
3. **Code examples** showing correct vs incorrect usage
4. **Reference to DHAT validation** proving the findings

### Deliverables

A clear, actionable guide that helps developers:
- Understand when `Model::set_basis()` is beneficial
- Avoid the "alien basis" trap
- Make informed decisions about warm-starting

---

## Acceptance Criteria

- [ ] `docs/HIGHS_WARM_START_INVESTIGATION.md` updated with guidelines section
- [ ] Guidelines include specific conditions for valid basis reuse
- [ ] Anti-patterns documented with explanations
- [ ] Code examples provided
- [ ] DHAT validation results referenced

---

## Implementation Guide

### Suggested Content Structure

Add a new section "## Basis Reuse Guidelines" with:

```markdown
## Basis Reuse Guidelines

### When Basis Reuse IS Appropriate

Basis reuse via `Model::set_basis()` is beneficial when:

1. **Model dimensions are unchanged**: Same number of rows and columns
2. **Only RHS/bounds changed**: Objective coefficients, constraint bounds, variable bounds
3. **Same constraint structure**: No rows added, removed, or reordered

Example valid use case:
```rust
// Same model, different RHS values
model.change_rows_bounds(row, new_lb, new_ub);
// Basis from previous solve is still valid
model.solve();  // Will warm-start automatically
```

### When Basis Reuse Causes Problems

**DO NOT** use `set_basis()` when:

1. **Row count changed**: Cuts added/removed between solves
2. **Column count changed**: Variables added/removed
3. **Constraint structure changed**: Different sparsity pattern

What happens:
- HiGHS detects dimension mismatch
- Basis marked as "alien"
- Triggers `formSimplexLpBasisAndFactor()`
- Full factorization rebuild (defeats warm-start purpose)

### SDDP-Specific Guidance

In SDDP training:
- **Between stages (same node)**: Basis reuse MAY work if no cuts added
- **Between forward and backward passes**: Basis reuse DOES NOT work (cuts added)
- **Between iterations**: Basis reuse DOES NOT work (model structure evolves)

### Validation Evidence

Sprint 6 DHAT profiling confirmed:
- With `reuse_forward_basis()`: 39.58 GB HFactor allocations
- Without `reuse_forward_basis()`: 2.00 GB HFactor allocations
- **95% reduction** by NOT using mismatched basis
```

### Key Files to Modify

- `docs/HIGHS_WARM_START_INVESTIGATION.md`: Add guidelines section

---

## Testing Requirements

### Documentation Review

- [ ] Guidelines are clear and actionable
- [ ] Examples compile (if included as code)
- [ ] No contradictions with existing documentation

---

## Documentation Requirements

- [ ] Guidelines section added to investigation document
- [ ] Cross-reference to DHAT analysis
- [ ] Consider adding to main MEMORY_BEHAVIOR.md as well

---

## Dependencies

- **Blocked By**: T-094 (Remove reuse_forward_basis code)
- **Blocks**: None
- **Related**: T-096, T-097 (HiGHS investigation tickets)

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Documentation task with clear content requirements

---

## Definition of Done

- [ ] Guidelines section complete
- [ ] Content reviewed for accuracy
- [ ] Examples are correct
- [ ] PR merged
