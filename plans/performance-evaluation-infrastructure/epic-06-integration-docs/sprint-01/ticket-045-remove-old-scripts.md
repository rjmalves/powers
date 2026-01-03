# [T-045] Remove old scripts

> **Epic**: [Epic 6: Integration & Documentation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-044
> **Blocks**: T-047

## Context

### Background
Legacy scripts (monitor_rss.py, plot_rss.py, compare_allocators.sh) should be removed or migrated to avoid confusion.

### Relation to Epic
Cleanup required to finalize migration to new profiling suite.

### Current State
Scripts exist in `scripts/` and docs reference them.

## Specification

### Inputs
- Baseline artifacts and new tooling paths
- Inventory of legacy scripts and doc references

### Outputs
- Removed/migrated scripts; updated docs/changelog

### Behavior
- Remove or deprecate legacy scripts superseded by new collectors
- Update docs to point to new commands
- Add changelog entry noting removal

### Error Handling
- Ensure removal does not break other tooling; provide migration notes

## Acceptance Criteria
- [ ] Legacy scripts removed or clearly deprecated
- [ ] Docs updated to reference new workflow
- [ ] Changelog updated

## Implementation Guide

### Suggested Approach
1. Identify scripts to remove; replace with pointers to new CLI commands.
2. Update docs referencing old scripts.
3. Add CHANGELOG entry under Unreleased/Next version.

### Key Files to Modify
- `scripts/*` (remove/deprecate)
- `docs/*` referencing scripts
- `CHANGELOG.md`

### Patterns to Follow
- Provide migration notes where removal could surprise users

### Pitfalls to Avoid
- ⚠️ Removing scripts still used in CI (verify first)

## Testing Requirements

### Validation
- [ ] Ensure docs build (if any) and examples use new commands

## Documentation Requirements
- [ ] Update references and changelog

## Dependencies
- **Blocked By**: T-044
- **Blocks**: T-047
- **Related**: T-019

## Effort Estimate
**Points**: 2
**Confidence**: Medium
**Rationale**: Cleanup with doc updates.

## Definition of Done
- [ ] Scripts handled
- [ ] Docs/changelog updated
- [ ] Users guided to new workflow
