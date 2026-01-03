# [T-034] Add timing visualization charts

> **Epic**: [Epic 5: Visualization & Reporting Dashboard](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-033
> **Blocks**: T-040

## Context

### Background
Timing data needs visual representation (phase breakdown and timeline) in the dashboard.

### Relation to Epic
Provides timing tab content for dashboard and reports.

### Current State
Base dashboard template exists; no timing charts implemented.

## Specification

### Inputs
- Timing collector data (phase durations, timeline events)

### Outputs
- Timing tab with phase breakdown bar chart and timeline line/step chart

### Behavior
- Render phase breakdown (stacked bar or grouped) showing total per phase
- Render iteration timeline (duration vs iteration)
- Include hover tooltips and units (ms)
- Display summary metrics (total time, fastest/slowest phase)

### Error Handling
- Handle missing timing data gracefully with placeholder message

## Acceptance Criteria
- [ ] Timing tab shows phase breakdown and timeline charts
- [ ] Units labeled (ms)
- [ ] Handles runs without timing data (shows message)

## Implementation Guide

### Suggested Approach
1. Add Plotly figures for bar and line charts in dashboard module.
2. Reuse color palette across charts.
3. Add helper to format durations.

### Key Files to Modify
- `profiling/powers_profile/reporters/dashboard.py`
- `profiling/tests/test_dashboard.py`

### Patterns to Follow
- Keep charts responsive; avoid heavy data for large iterations

### Pitfalls to Avoid
- ⚠️ Overlapping labels; rotate or abbreviate phase names

## Testing Requirements

### Unit Tests
- [ ] Render with sample timing data
- [ ] Missing data path shows placeholder

### Documentation Requirements
- [ ] Update dashboard docs to mention timing charts

## Dependencies
- **Blocked By**: T-033
- **Blocks**: T-040
- **Related**: T-039

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Plotly charts with straightforward data.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
