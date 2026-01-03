# [T-041] Write Quick Start guide

> **Epic**: [Epic 6: Integration & Documentation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: Epics 1-5 complete
> **Blocks**: T-042, T-043, T-047

## Context

### Background
Need concise onboarding instructions for using the profiling suite end-to-end.

### Relation to Epic
Forms base documentation referenced by other guides.

### Current State
No finalized docs in `docs/profiling/`.

## Specification

### Inputs
- Final CLI commands and config defaults
- Output directory structure

### Outputs
- `docs/profiling/QUICK_START.md` with prerequisites, install, first run, viewing results

### Behavior
- Include install steps (`pip install -e .`), prerequisites (perf, valgrind, FlameGraph)
- Provide first-run example for timing/cpu/memory/parallel as applicable
- Show how to view summary and dashboard
- Add troubleshooting snippet for permissions

### Error Handling
- N/A (docs) but include warnings for common pitfalls

## Acceptance Criteria
- [ ] QUICK_START.md contains prerequisites, install, run, view, troubleshoot sections
- [ ] Commands copy-pastable
- [ ] Links to other docs (Tools Reference, Analysis Guide)

## Implementation Guide

### Suggested Approach
1. Draft outline then fill with commands and expected outputs.
2. Keep concise (<2 pages) with numbered steps.
3. Cross-link to other documents.

### Key Files to Modify
- `docs/profiling/QUICK_START.md`

### Patterns to Follow
- Use markdown code blocks for commands

### Pitfalls to Avoid
- ⚠️ Overlong explanations; keep action-oriented

## Testing Requirements

### Documentation Checks
- [ ] Validate commands run on fresh env

## Documentation Requirements
- [ ] QUICK_START.md added/updated

## Dependencies
- **Blocked By**: Epics 1-5 complete
- **Blocks**: T-042, T-043, T-047
- **Related**: T-013

## Effort Estimate
**Points**: 2
**Confidence**: High
**Rationale**: Documentation based on completed features.

## Definition of Done
- [ ] QUICK_START.md merged
- [ ] Links added to plan/README
