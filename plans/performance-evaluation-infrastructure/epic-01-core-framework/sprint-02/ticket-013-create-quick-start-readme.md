# [T-013] Create quick start README

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: All Sprint 2 tickets
> **Blocks**: None

## Context

### Background
Developers need concise onboarding instructions to use the profiling framework after Sprint 2 is complete.

### Relation to Epic
Fulfills Epic 1 acceptance criterion for README with quick start instructions.

### Current State
No profiling README exists; only plan documentation is available.

## Specification

### Inputs
- Finalized CLI commands (`run`, `summary`, `history`)
- Storage layout and config defaults

### Outputs
- `profiling/README.md` (or similar) with quick start guide
- Example commands for timing collector

### Behavior
- Document installation (`pip install -e .`), prerequisites, and environment setup
- Provide first-run example with timing collector
- Document where outputs are stored and how to view summary/history
- Include troubleshooting tips for common errors

### Error Handling
- N/A (documentation), but note common failure modes (missing perf, permissions)

## Acceptance Criteria
- [ ] README includes install steps, first run, summary/history usage
- [ ] Paths to outputs and history documented
- [ ] Troubleshooting section for missing tools/permissions
- [ ] Links to schemas/tests for contributors

## Implementation Guide

### Suggested Approach
1. Draft README with sections: Prerequisites, Install, Run, View Results, Troubleshooting.
2. Include minimal examples and expected outputs.
3. Add table linking collectors and config keys.
4. Cross-link to plan files where helpful.

### Key Files to Modify
- `profiling/README.md` (new)
- `profiling/pyproject.toml` (ensure project name in docs)

### Patterns to Follow
- Keep commands copy-pastable
- Use consistent terminology with plan

### Pitfalls to Avoid
- ⚠️ Overlong README; keep focused on getting started
- ⚠️ Forgetting to mention output locations

## Testing Requirements

### Documentation Checks
- [ ] Validate commands execute in a fresh env (dry run where possible)

## Documentation Requirements
- [ ] README added with clear examples
- [ ] Update plan quick links if path differs

## Dependencies
- **Blocked By**: All Sprint 2 tickets
- **Blocks**: None
- **Related**: T-011, T-012

## Effort Estimate
**Points**: 2
**Confidence**: High
**Rationale**: Documentation-only after functionality exists.

## Definition of Done
- [ ] README merged
- [ ] Commands verified
- [ ] References added to plan
