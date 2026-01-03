# [T-039] Implement rich CLI summary

> **Epic**: [Epic 5: Visualization & Reporting Dashboard](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-038
> **Blocks**: None

## Context

### Background
Need improved terminal output summarizing profiling results with colors and emphasis for regressions.

### Relation to Epic
Satisfies acceptance criterion for rich CLI summaries.

### Current State
Basic summary command exists from Epic 1; lacks rich visualization and cross-domain data.

## Specification

### Inputs
- Profiling run data (latest or specified run)
- Optional comparison data
- Flags for verbosity and domain filters

### Outputs
- Rich-formatted console output with tables/panels and color-coded deltas

### Behavior
- Display summary cards for total time, peak RSS, best speedup, hotspot count
- Show tables for CPU hotspots, memory peaks, scaling metrics
- Use color coding (green improvements, red regressions)
- Support comparison mode highlighting deltas

### Error Handling
- Handle missing data sections gracefully
- Provide guidance if run data not found

## Acceptance Criteria
- [ ] CLI summary shows rich output across domains
- [ ] Color-coded improvements/regressions implemented
- [ ] Works for single run and comparison mode

## Implementation Guide

### Suggested Approach
1. Extend `summary` command to use Rich panels/tables with delta formatting.
2. Reuse markdown reporter logic for section ordering but format for console.
3. Add toggles for terse/full modes.

### Key Files to Modify
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/reporters/markdown.py` (shared formatting helpers)

### Patterns to Follow
- Maintain consistent colors/icons across outputs

### Pitfalls to Avoid
- ⚠️ Overly wide tables; wrap or abbreviate values

## Testing Requirements

### Unit Tests
- [ ] Rendering paths with single run data
- [ ] Rendering paths with comparison data
- [ ] Behavior when data missing

### Documentation Requirements
- [ ] Update CLI help and README examples

## Dependencies
- **Blocked By**: T-038
- **Blocks**: None
- **Related**: T-011, T-031

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Presentation logic leveraging existing data and helpers.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
