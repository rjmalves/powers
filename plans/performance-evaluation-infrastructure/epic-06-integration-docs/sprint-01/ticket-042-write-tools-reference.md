# [T-042] Write Tools Reference

> **Epic**: [Epic 6: Integration & Documentation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-041
> **Blocks**: T-047

## Context

### Background
Users need detailed reference for all collectors, analyzers, and reporters with configuration options.

### Relation to Epic
Forms core documentation for feature set.

### Current State
No consolidated tools reference exists.

## Specification

### Inputs
- Finalized collector/analyzer/reporter options
- CLI flag definitions and defaults

### Outputs
- `docs/profiling/TOOLS_REFERENCE.md` documenting all commands/options

### Behavior
- Document collectors (cpu, memory, parallel, timing) with options and required tools
- Document analyzers/reporters and output paths
- Include sample config snippets (TOML)
- Add troubleshooting for each tool

### Error Handling
- N/A (docs), but include warnings for permissions/tools missing

## Acceptance Criteria
- [ ] TOOLS_REFERENCE.md lists all collectors/analyzers/reporters and options
- [ ] Config examples included
- [ ] Troubleshooting per tool provided

## Implementation Guide

### Suggested Approach
1. Structure by domain (CPU, Memory, Parallel, Timing).
2. Include tables for flags/options and defaults.
3. Link to Quick Start and Analysis Guide.

### Key Files to Modify
- `docs/profiling/TOOLS_REFERENCE.md`

### Patterns to Follow
- Table format for options with description and defaults

### Pitfalls to Avoid
- ⚠️ Out-of-date flags; sync with CLI help

## Testing Requirements

### Documentation Checks
- [ ] Validate examples match CLI

## Documentation Requirements
- [ ] TOOLS_REFERENCE.md completed and linked

## Dependencies
- **Blocked By**: T-041
- **Blocks**: T-047
- **Related**: T-019

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Comprehensive documentation effort.

## Definition of Done
- [ ] Reference merged
- [ ] Links added to README
