# [T-036] Add scaling visualization charts

> **Epic**: [Epic 5: Visualization & Reporting Dashboard](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-033
> **Blocks**: T-040

## Context

### Background
Parallel scaling data needs charts to show speedup and efficiency versus thread count.

### Relation to Epic
Provides parallel tab content and enables readability of scaling metrics.

### Current State
Base dashboard exists; no scaling charts implemented.

## Specification

### Inputs
- Scaling collector data (thread counts, speedup, efficiency, Amdahl estimates)

### Outputs
- Scaling tab with speedup and efficiency charts, Amdahl summary

### Behavior
- Plot speedup vs thread count line chart with ideal line
- Plot efficiency vs thread count
- Show Amdahl serial fraction and predicted max speedup as summary cards
- Include hover tooltips and proper labels

### Error Handling
- Handle missing scaling data gracefully

## Acceptance Criteria
- [ ] Speedup and efficiency charts rendered with ideal line overlay
- [ ] Amdahl summary displayed
- [ ] Handles missing data with placeholder text

## Implementation Guide

### Suggested Approach
1. Build Plotly line charts with markers for measured data and ideal speedup.
2. Add text annotations for key points (best speedup/efficiency).
3. Reuse color palette.

### Key Files to Modify
- `profiling/powers_profile/reporters/dashboard.py`
- `profiling/tests/test_dashboard.py`

### Patterns to Follow
- Keep axes logarithmic option optional for high thread counts

### Pitfalls to Avoid
- ⚠️ Cluttered tooltips; keep concise

## Testing Requirements

### Unit Tests
- [ ] Render with fixture scaling data
- [ ] Placeholder path when data missing

### Documentation Requirements
- [ ] Update dashboard docs for scaling tab

## Dependencies
- **Blocked By**: T-033
- **Blocks**: T-040
- **Related**: T-034, T-035

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Straightforward Plotly charts.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
