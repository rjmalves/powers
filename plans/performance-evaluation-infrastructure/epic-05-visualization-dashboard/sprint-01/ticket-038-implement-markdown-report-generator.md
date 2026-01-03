# [T-038] Implement markdown report generator

> **Epic**: [Epic 5: Visualization & Reporting Dashboard](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-033
> **Blocks**: T-039

## Context

### Background
Need human-readable markdown reports summarizing profiling runs for quick sharing without HTML.

### Relation to Epic
Complements dashboard and CLI summary outputs.

### Current State
No markdown generation exists.

## Specification

### Inputs
- Profiling run data (single run) and optional comparison data
- Config: include sections, verbosity

### Outputs
- `report.md` with summary metrics, tables, and links to artifacts

### Behavior
- Generate sections: Run info, Timing, CPU, Memory, Parallel (when available)
- Include key metrics tables and bullet highlights (regressions/improvements)
- Link to artifacts (SVGs, JSON) using relative paths
- Support comparison mode (baseline vs target) with delta tables

### Error Handling
- Handle missing data sections gracefully with notes
- Validate output path

## Acceptance Criteria
- [ ] `report.md` generated for single run
- [ ] Supports comparison mode with delta tables
- [ ] Links to artifacts included

## Implementation Guide

### Suggested Approach
1. Add reporter module `reporters/markdown.py` (or extend existing) for markdown generation.
2. Use simple templates for sections and tables.
3. Integrate with CLI (`run`, `compare`, `summary`).

### Key Files to Modify
- `profiling/powers_profile/reporters/markdown.py`
- `profiling/powers_profile/cli.py`

### Patterns to Follow
- Keep markdown readable in terminals and rendered views

### Pitfalls to Avoid
- ⚠️ Broken relative links; compute based on output dir

## Testing Requirements

### Unit Tests
- [ ] Generate markdown from fixture data and snapshot compare
- [ ] Missing data sections produce placeholders

### Documentation Requirements
- [ ] Document markdown report command/flags

## Dependencies
- **Blocked By**: T-033
- **Blocks**: T-039
- **Related**: T-040

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Template work with data mapping.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
