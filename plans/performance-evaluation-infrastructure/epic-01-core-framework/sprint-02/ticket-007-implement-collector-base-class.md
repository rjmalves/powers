# [T-007] Implement Collector base class

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-001, T-003
> **Blocks**: T-008, T-009, T-012

## Context

### Background
Collector abstractions are needed so CPU, memory, timing, and parallel collectors share a consistent interface and lifecycle.

### Relation to Epic
Part of Epic 1 Sprint 2 delivering first working collector and shared infrastructure.

### Current State
CLI, schemas, config loading, and utilities exist from Sprint 1; no collector base or plugin discovery is present yet.

## Specification

### Inputs
- Target binary path (`Path`)
- Argument list (`List[str]`)
- Collector-specific config (`Mapping[str, Any]`)
- Optional environment overrides

### Outputs
- `CollectorResult` instance with raw artifact paths and parsed payload

### Behavior
- Provide `Collector` ABC with `collect` and `parse_output` abstract methods
- Provide `CollectorResult` factory helpers to attach metadata (collector name, version, schema_version)
- Include lifecycle hooks: `prepare_run(output_dir)`, `execute`, `finalize`
- Provide plugin discovery helper to enumerate collectors in `collectors/`

### Error Handling
- Surface subprocess failures with stderr/stdout captured
- Raise descriptive errors on missing binaries or bad config keys
- Ensure partial outputs are cleaned up on failure

## Acceptance Criteria
- [ ] `collectors/base.py` defines `Collector` ABC with typed methods
- [ ] Plugin discovery returns available collectors
- [ ] CollectorResult encapsulates metadata + parsed data
- [ ] Errors propagate with actionable messages

## Implementation Guide

### Suggested Approach
1. Define `Collector` ABC with `collect`, `parse_output`, and optional `prepare_run` hooks.
2. Implement `CollectorRegistry` helper for discovery/lookup by name.
3. Add `CollectorResult` dataclass with helper constructors.
4. Wire discovery into CLI `run` command path.

### Key Files to Modify
- `profiling/powers_profile/collectors/base.py`
- `profiling/powers_profile/cli.py` (wire discovery)
- `profiling/powers_profile/schemas/results.py`

### Patterns to Follow
- Follow existing schema dataclasses from Sprint 1
- Match Typer CLI patterns already used

### Pitfalls to Avoid
- ⚠️ Do not swallow subprocess errors
- ⚠️ Avoid mutable default args in collector configs

## Testing Requirements

### Unit Tests
- [ ] Collector registry returns expected names
- [ ] Dummy collector implements ABC and raises if `parse_output` missing
- [ ] Error on missing binary path

### Integration Tests
- [ ] CLI `run` with dummy collector resolves via registry

## Documentation Requirements
- [ ] Docstrings for `Collector` and `CollectorResult`
- [ ] Update module-level README if present

## Dependencies
- **Blocked By**: T-001, T-003
- **Blocks**: T-008, T-009, T-012
- **Related**: T-010

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Straightforward abstractions with limited external integration.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] Docstrings updated
- [ ] CLI resolves collectors via registry
