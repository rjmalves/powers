# [T-035] Add memory visualization charts

> **Epic**: [Epic 5: Visualization & Reporting Dashboard](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-033
> **Blocks**: T-040

## Context

### Background
Memory profiling data needs dashboards for RSS timeline, DHAT summary, Massif peak, and cache metrics.

### Relation to Epic
Provides memory tab content.

### Current State
Base dashboard exists; no memory charts implemented.

## Specification

### Inputs
- Memory collector data (RSS samples, DHAT, Massif, Cachegrind summaries)

### Outputs
- Memory tab with RSS timeline, peak markers, DHAT hotspot table, cache miss summary

### Behavior
- Plot RSS over time with peak annotation
- Show Massif peak snapshot info
- Show DHAT hotspots table (alloc site, bytes, blocks)
- Show cache miss bars (I1, D1, LL) and branch stats

### Error Handling
- Handle missing data for any sub-tool with warnings/placeholder text

## Acceptance Criteria
- [ ] Memory tab displays RSS chart with peak
- [ ] DHAT hotspot table rendered
- [ ] Massif peak and cache summary shown
- [ ] Graceful handling when a tool was skipped

## Implementation Guide

### Suggested Approach
1. Add Plotly line chart for RSS with annotations.
2. Add tables for DHAT hotspots and cache stats.
3. Use consistent color palette with timing tab.

### Key Files to Modify
- `profiling/powers_profile/reporters/dashboard.py`
- `profiling/tests/test_dashboard.py`

### Patterns to Follow
- Keep tables sortable if feasible (JS) or simple static

### Pitfalls to Avoid
- ⚠️ Overcrowding memory tab; keep layout clean

## Testing Requirements

### Unit Tests
- [ ] Render with fixture memory data
- [ ] Missing data path shows placeholders

### Documentation Requirements
- [ ] Update dashboard docs to describe memory visuals

## Dependencies
- **Blocked By**: T-033
- **Blocks**: T-040
- **Related**: T-034, T-036

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Multiple chart types but straightforward data.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
