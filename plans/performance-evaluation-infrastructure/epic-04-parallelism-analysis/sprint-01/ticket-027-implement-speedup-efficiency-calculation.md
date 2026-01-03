# [T-027] Implement speedup/efficiency calculation

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-026
> **Blocks**: T-028, T-030

## Context

### Background
Need to compute speedup and efficiency metrics from scaling run results to quantify parallel performance.

### Relation to Epic
Core analysis feeding scaling collector JSON and dashboards.

### Current State
No metrics computed from scaling runs.

## Specification

### Inputs
- Timing results per thread count (mean duration)
- Reference single-thread duration (T1)

### Outputs
- Speedup per thread count (T1 / Tn)
- Efficiency per thread count (speedup / n)
- Optional geometric mean metrics

### Behavior
- Compute metrics for each thread count; handle missing T1 gracefully
- Provide helper functions to summarize best speedup/efficiency
- Flag regressions (speedup < previous at higher thread count)

### Error Handling
- Handle zero/invalid durations with clear errors
- Skip metrics for thread counts lacking data

## Acceptance Criteria
- [ ] Speedup and efficiency computed for all thread counts
- [ ] Regressions flagged where speedup decreases at higher thread count
- [ ] Handles missing/zero durations without crash

## Implementation Guide

### Suggested Approach
1. Add functions in `analyzers/scaling.py` to compute metrics from timing data.
2. Include optional smoothing (median) using collected iterations.
3. Return structured dict ready for JSON serialization.

### Key Files to Modify
- `profiling/powers_profile/analyzers/scaling.py`
- `profiling/tests/test_scaling.py`

### Patterns to Follow
- Pure functions for testability

### Pitfalls to Avoid
- ⚠️ Division by zero when T1 missing or zero
- ⚠️ Mislabeling efficiency > 1 due to measurement noise; clamp if necessary

## Testing Requirements

### Unit Tests
- [ ] Speedup/efficiency computations for typical data
- [ ] Behavior with missing T1
- [ ] Regression flag logic

## Documentation Requirements
- [ ] Document formulas in code comments/docstrings

## Dependencies
- **Blocked By**: T-026
- **Blocks**: T-028, T-030
- **Related**: T-031

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Deterministic computations.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Documentation updated
