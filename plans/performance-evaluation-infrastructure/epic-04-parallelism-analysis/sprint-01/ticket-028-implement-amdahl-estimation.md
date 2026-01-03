# [T-028] Implement Amdahl estimation

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-027
> **Blocks**: T-030

## Context

### Background
Amdahl's law estimation helps quantify serial fraction and predict scaling limits from measured data.

### Relation to Epic
Provides analytical insight for scaling collector and dashboard.

### Current State
No Amdahl estimation implemented.

## Specification

### Inputs
- Speedup data per thread count (from T-027)

### Outputs
- Serial fraction estimate per thread count pair and aggregate
- Predicted maximum speedup given measured serial fraction

### Behavior
- Use Amdahl formula to estimate serial fraction from observed speedup and thread count
- Compute aggregate estimate (e.g., using highest thread count data)
- Include confidence indicator based on variance across thread counts

### Error Handling
- Handle invalid/zero speedup data gracefully
- Warn when thread counts insufficient for estimation

## Acceptance Criteria
- [ ] Serial fraction estimated and included in scaling JSON
- [ ] Predicted max speedup calculated
- [ ] Confidence indicator provided

## Implementation Guide

### Suggested Approach
1. Add functions to `analyzers/scaling.py` for Amdahl estimation using speedup data.
2. Expose in scaling collector payload.
3. Write docstring with formula references.

### Key Files to Modify
- `profiling/powers_profile/analyzers/scaling.py`
- `profiling/tests/test_scaling.py`

### Patterns to Follow
- Use float math with guard for divide-by-zero

### Pitfalls to Avoid
- ⚠️ Over-interpreting noisy data; include warnings when variance high

## Testing Requirements

### Unit Tests
- [ ] Amdahl calculation with typical data
- [ ] Behavior when input speedup insufficient

## Documentation Requirements
- [ ] Document formulas in code comments

## Dependencies
- **Blocked By**: T-027
- **Blocks**: T-030
- **Related**: T-031

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Small analytical functions.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Documentation updated
