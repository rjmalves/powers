# [T-009] Implement JSON reporter

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-003, T-007
> **Blocks**: T-010, T-011, T-012

## Context

### Background
Profiling runs must be serialized into machine-readable JSON that follows the defined schemas and versioning.

### Relation to Epic
Provides the first reporter to persist collector output and metadata for later analysis.

### Current State
Schemas are defined; no reporter exists to emit structured JSON files.

## Specification

### Inputs
- `ProfilingRun` or equivalent data structure
- Output directory path
- Optional pretty-print flag

### Outputs
- `results.json` adhering to schema version
- Optionally `results.pretty.json` when pretty flag enabled

### Behavior
- Serialize run metadata, system/git info, collector results, and analyzer outputs
- Include schema version and timestamp in top-level object
- Validate before writing; fail if schema mismatch
- Return path(s) to generated files

### Error Handling
- Raise errors on missing required fields or invalid types
- Fail fast if output directory is not writable
- Surface JSON encoding issues with clear messages

## Acceptance Criteria
- [ ] Reporter writes schema-compliant JSON file
- [ ] Supports pretty-print toggle
- [ ] Validates collector payloads exist before write
- [ ] Returns generated path for downstream commands

## Implementation Guide

### Suggested Approach
1. Implement `reporters/json_export.py` with `Reporter` interface.
2. Use `dataclasses.asdict` + custom serialization for Paths/Enums.
3. Add schema validation helper (lightweight) before write.
4. Expose reporter selection via CLI `run` command.

### Key Files to Modify
- `profiling/powers_profile/reporters/json_export.py`
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/schemas/run.py`

### Patterns to Follow
- Match serialization style from schemas tests
- Keep file names deterministic (results.json)

### Pitfalls to Avoid
- ⚠️ Writing partial data when validation fails
- ⚠️ Forgetting to create output directories

## Testing Requirements

### Unit Tests
- [ ] Serialize sample run with collector data
- [ ] Pretty-print toggle writes second file
- [ ] Validation failure on missing required field

### Integration Tests
- [ ] End-to-end `powers-profile run` writes results.json

## Documentation Requirements
- [ ] Docstring on reporter module
- [ ] Update CLI help text for output options

## Dependencies
- **Blocked By**: T-003, T-007
- **Blocks**: T-010, T-011, T-012
- **Related**: T-008

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Straightforward serialization using existing schemas.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] Reporter exposed via CLI
- [ ] Documentation updated
