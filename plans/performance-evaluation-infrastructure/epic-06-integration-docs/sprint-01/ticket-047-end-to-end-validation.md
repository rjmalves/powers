# [T-047] End-to-end validation

> **Epic**: [Epic 6: Integration & Documentation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-041, T-042, T-043, T-044
> **Blocks**: None

## Context

### Background
Final validation ensures all commands, collectors, and dashboards work together on at least two environments (WSL2 + bare-metal/cloud).

### Relation to Epic
Finalizes release readiness for profiling suite.

### Current State
Components developed across epics; no holistic validation executed.

## Specification

### Inputs
- Installed profiling suite
- Baseline artifacts
- Test environments (WSL2 dev, bare-metal or cloud)

### Outputs
- Validation report summarizing results, issues, and follow-ups

### Behavior
- Run full suite on two environments; record pass/fail per command (run, summary, history, dashboard, compare, scaling)
- Validate JSON schema compliance using sample outputs
- Capture performance runtime (ensure <30 minutes target)
- Log issues and remediation steps

### Error Handling
- Document failures with reproducible steps; do not silently skip

## Acceptance Criteria
- [ ] Validation executed on two environments with results recorded
- [ ] All commands verified or issues logged with owners
- [ ] Schema validation performed
- [ ] Runtime recorded and within target or justified

## Implementation Guide

### Suggested Approach
1. Prepare checklist covering all commands and domains.
2. Execute on WSL2 and bare-metal/cloud; capture logs and outputs.
3. Write short validation report in `docs/profiling/VALIDATION.md`.

### Key Files to Modify
- `docs/profiling/VALIDATION.md`
- `docs/profiling/README.md` or index linking report

### Patterns to Follow
- Checklist with pass/fail markers

### Pitfalls to Avoid
- ⚠️ Skipping slow commands; must measure runtime

## Testing Requirements

### Validation Tasks
- [ ] Run suite on WSL2 and bare-metal/cloud
- [ ] Verify dashboards open without errors
- [ ] Validate JSON schemas

## Documentation Requirements
- [ ] Validation report added and linked

## Dependencies
- **Blocked By**: T-041, T-042, T-043, T-044
- **Blocks**: None
- **Related**: Success metrics in master plan

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Execution across environments and reporting.

## Definition of Done
- [ ] Validation report completed
- [ ] Issues logged
- [ ] Links added
