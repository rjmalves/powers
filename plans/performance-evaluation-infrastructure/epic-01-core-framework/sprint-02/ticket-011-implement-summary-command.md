# [T-011] Implement summary command

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-009, T-010
> **Blocks**: T-013

## Context

### Background
Users need a quick terminal view of the latest profiling run without opening JSON files or dashboards.

### Relation to Epic
Delivers `powers-profile summary` CLI promised in acceptance criteria.

### Current State
Run storage/history will be added in T-010; no summary command exists.

## Specification

### Inputs
- Optional `--run-id` to select a specific run; defaults to latest
- Optional `--output` to write markdown summary

### Outputs
- Rich-formatted table in terminal
- Optional `summary.md` file when requested

### Behavior
- Load run metadata and key metrics (total time, collectors present, git SHA, system info)
- Display warnings for missing collectors or outdated schema versions
- Support filter flags (e.g., `--collectors cpu,memory` to show subset)
- Exit non-zero if requested run not found

### Error Handling
- Handle missing history file with actionable guidance
- Detect corrupted JSON and surface path causing issue
- Validate run id exists before rendering

## Acceptance Criteria
- [ ] `powers-profile summary` shows last run with key metrics
- [ ] Supports `--run-id` selection
- [ ] Optional markdown output file generated when requested
- [ ] Errors on missing/corrupt history are user-friendly

## Implementation Guide

### Suggested Approach
1. Add Typer command `summary` leveraging Rich tables/panels.
2. Load latest run from `history.json` (or specific id) via storage helper.
3. Render system/git info + collector summaries.
4. Add optional markdown rendering using simple template.

### Key Files to Modify
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/reporters/markdown.py` (optional helper)
- `profiling/powers_profile/utils/paths.py`

### Patterns to Follow
- Use Rich formatting consistent with other commands
- Keep output concise (fit typical terminal width)

### Pitfalls to Avoid
- ⚠️ Assuming latest symlink always exists
- ⚠️ Failing silently on missing collectors

## Testing Requirements

### Unit Tests
- [ ] Summary with single collector present
- [ ] Summary with multiple collectors sorted by name
- [ ] Error path when history missing

### Integration Tests
- [ ] CLI invocation after sample run renders expected text

## Documentation Requirements
- [ ] Update CLI help and README examples

## Dependencies
- **Blocked By**: T-009, T-010
- **Blocks**: T-013
- **Related**: T-008, T-012

## Effort Estimate
**Points**: 2
**Confidence**: High
**Rationale**: Mostly presentation using existing stored data.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] CLI help and README updated
