# [TICKET-014] Design SubproblemBlock structure

> **Epic**: [Epic 3: Handler-Level SoA Blocks](../00-epic-overview.md)  
> **Sprint**: [Sprint 2](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-012](../sprint-01/ticket-012-implement-realization-block.md)  
> **Blocks**: [TICKET-015](./ticket-015-implement-subproblem-block.md)

## Context

### Background

Similar to RealizationBlock, subproblem hot data could benefit from SoA layout. However, subproblems contain HiGHS models which are complex objects. This ticket evaluates what data (if any) should be in a SubproblemBlock.

### Relation to Epic

Design decision that may or may not result in implementation.

## Files to Read Before Starting

- `src/subproblem.rs` - Subproblem structure (~6,400 lines)
- `src/sddp/mod.rs` - Subproblem access patterns

## Specification

### Analysis Questions

1. **What subproblem data is accessed in hot path?**
   - Variable values after solve
   - Dual values for cut generation
   - Constraint bounds updates

2. **What data is large and could benefit from SoA?**
   - Variable vectors (primal solution)
   - Dual vectors (for cuts)
   - Lag data buffers

3. **What data is better kept with Subproblem object?**
   - HiGHS Model (complex, internal state)
   - Variable indices
   - Constraint metadata

### Design Options

**Option A: No SubproblemBlock**
- Keep subproblems in graph
- Benefit from RealizationBlock only
- Simpler, less risk

**Option B: Minimal SubproblemBlock**
- Extract only solution vectors
- Keep models in graph
- Moderate benefit, moderate risk

**Option C: Full SubproblemBlock**
- Extract all hot data
- Complex, high risk
- Maximum benefit

### Recommendation

Start with Option A (no SubproblemBlock). The RealizationBlock captures most cache benefit. Subproblem access is dominated by HiGHS solve time, not memory access.

If profiling shows subproblem access as bottleneck, implement Option B.

## Acceptance Criteria

- [ ] Analysis of subproblem hot data complete
- [ ] Decision documented (proceed or skip)
- [ ] If proceed: structure designed (like TICKET-011)
- [ ] If skip: rationale documented

## Implementation Guide

### Profiling to Inform Decision

```bash
# Profile cache behavior during backward pass
perf stat -e cache-references,cache-misses \
  ./target/release/powers run examples/07-par-model-with-inflow-state

# Look at backward pass specifically
perf record -g ./target/release/powers run examples/07-par-model-with-inflow-state
perf report --no-children | head -50
```

### Decision Tree

```
Is subproblem memory access in top 10 hot functions?
├── No → Skip SubproblemBlock (Option A)
└── Yes → What data is accessed most?
    ├── Solution vectors → Option B (solution block)
    └── Model operations → No block benefit (HiGHS internal)
```

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Analysis and design, not implementation

## Definition of Done

- [ ] Analysis complete
- [ ] Decision documented
- [ ] If proceeding: design ready for TICKET-015
- [ ] If skipping: TICKET-015 marked N/A
