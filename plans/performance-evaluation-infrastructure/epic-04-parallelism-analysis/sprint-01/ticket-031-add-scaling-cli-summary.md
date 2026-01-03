# [T-031] Add scaling CLI summary

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-030
> **Blocks**: T-032

## Context

### Background
Users need quick scaling insights in the terminal without opening JSON or dashboards.

### Relation to Epic
Completes CLI usability for scaling analysis.

### Current State
No CLI summary exists for scaling results.

## Specification

### Inputs
- Scaling collector output (from latest run or specified run-id)
- Optional `--run-id`, `--threads` filter

### Outputs
- Rich terminal table showing speedup/efficiency per thread count and Amdahl estimate
- Optional markdown export

### Behavior
- Load scaling_data.json via storage helper
- Render table sorted by thread count with speedup/efficiency columns and highlights for regressions
- Show summary cards (best speedup, best efficiency, serial fraction)
- Respect filters and handle missing data gracefully

### Error Handling
- Friendly errors when scaling data absent for run
- Warn when perf contention metrics missing

## Acceptance Criteria
- [ ] `powers-profile scaling --summary` or `powers-profile summary` shows scaling table
- [ ] Highlights regressions and best metrics
- [ ] Optional markdown export works

## Implementation Guide

### Suggested Approach
1. Add CLI command or option leveraging Rich tables.
2. Reuse summary command structure from Epic 1.
3. Include colors for regressions (red) and improvements (green).

### Key Files to Modify
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/reporters/markdown.py`

### Patterns to Follow
- Use same console styling as other summaries

### Pitfalls to Avoid
- ⚠️ Wide tables exceeding terminal width; wrap or abbreviate

## Testing Requirements

### Unit Tests
- [ ] Summary rendering with complete data
- [ ] Behavior when data missing or filtered

### Integration Tests
- [ ] CLI invocation using fixture scaling data

## Documentation Requirements
- [ ] Update CLI help and README with scaling summary usage

## Dependencies
- **Blocked By**: T-030
- **Blocks**: T-032
- **Related**: Epic 5 dashboard

## Effort Estimate
**Points**: 2
**Confidence**: High
**Rationale**: Presentation built on existing data.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
