# [T-040] Add comparison dashboard view

> **Epic**: [Epic 5: Visualization & Reporting Dashboard](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-034, T-035, T-036, T-037
> **Blocks**: None

## Context

### Background
Need dashboard view that compares baseline vs target runs across all domains with deltas and visuals.

### Relation to Epic
Completes comparison capability and supports regression analysis.

### Current State
Single-run dashboard exists; no comparison tab.

## Specification

### Inputs
- Baseline and target run data (all domains)
- Paths to comparison artifacts (diff flamegraph, memory deltas, scaling deltas)

### Outputs
- Comparison tab in dashboard with delta tables and charts
- Summary cards for improvements/regressions

### Behavior
- Show side-by-side metrics with delta percentages and color coding
- Embed diff flamegraph when available
- Plot delta bars for key metrics (total time, peak RSS, speedup, hotspot changes)
- Allow selection of baseline/target from available runs (if CLI supports)

### Error Handling
- Handle missing artifacts gracefully with warnings
- Warn when schema versions differ between runs

## Acceptance Criteria
- [ ] Comparison tab renders with deltas and color coding
- [ ] Diff flamegraph embedded when available
- [ ] Handles missing data sections gracefully

## Implementation Guide

### Suggested Approach
1. Extend dashboard generator to accept two-run input and render comparison tab.
2. Reuse markdown report comparison logic (T-038) for data shaping.
3. Color deltas based on sign and threshold.

### Key Files to Modify
- `profiling/powers_profile/reporters/dashboard.py`
- `profiling/powers_profile/cli.py`

### Patterns to Follow
- Consistent color semantics with CLI summary

### Pitfalls to Avoid
- ⚠️ Overly heavy HTML size; lazy-load comparison tab if needed

## Testing Requirements

### Unit Tests
- [ ] Render comparison tab with fixture baseline/target data
- [ ] Missing data path shows warnings

### Documentation Requirements
- [ ] Document comparison dashboard usage

## Dependencies
- **Blocked By**: T-034, T-035, T-036, T-037
- **Blocks**: None
- **Related**: T-018, T-025, T-031

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Requires combining multiple data sources and artifacts.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
