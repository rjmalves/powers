# [T-008] Implement timing collector

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-007, T-004, T-005, T-006
> **Blocks**: T-012, T-013

## Context

### Background
We need a first working collector that extracts timing data from POWE.RS runs to validate the framework end-to-end.

### Relation to Epic
Delivers the first collector promised in Epic 1, enabling `powers-profile run --collectors timing`.

### Current State
CLI, schemas, system/git info, and config loading are available; collector base is pending T-007.

## Specification

### Inputs
- Target binary path (POWE.RS executable)
- CLI args passed through
- Config section `collectors.timing` (regex patterns, output dir)

### Outputs
- `CollectorResult` with timing events list, summary stats, and raw log capture path

### Behavior
- Execute target binary, capture stdout/stderr to file
- Parse timing lines (e.g., `TIMING:<phase>:<millis>` or JSON if available) into structured events
- Aggregate totals per phase and overall duration
- Attach git/system metadata from Sprint 1 utilities

### Error Handling
- If no timing markers found, mark collector status as `warning` with message
- Propagate subprocess failures with exit code and stderr excerpt
- Validate config keys; reject unknown keys with clear error

## Acceptance Criteria
- [ ] `powers-profile run --collectors timing` produces `timing.json`
- [ ] Timing events parsed with phase name, duration ms, timestamp order preserved
- [ ] Summary totals computed (total_ms, phase breakdown)
- [ ] Raw stdout/stderr saved alongside parsed JSON

## Implementation Guide

### Suggested Approach
1. Add `timing.py` collector implementing `Collector` ABC.
2. Define regex patterns for timing lines; make configurable in `default.toml`.
3. Use `subprocess.run` with tee to file for stdout/stderr.
4. Build `CollectorResult` with parsed events and write JSON via schema helper.

### Key Files to Modify
- `profiling/powers_profile/collectors/timing.py`
- `profiling/powers_profile/config/default.toml`
- `profiling/powers_profile/schemas/results.py`

### Patterns to Follow
- Reuse logging/error handling from T-007 base
- Keep parsing tolerant to whitespace and ordering

### Pitfalls to Avoid
- ⚠️ Blocking on large stdout; use streaming to file
- ⚠️ Assuming single timing format; allow configurable regex patterns

## Testing Requirements

### Unit Tests
- [ ] Parse sample stdout with multiple timing lines
- [ ] Handle missing timing markers (warning status)
- [ ] Config override of regex pattern

### Integration Tests
- [ ] Run against fixture binary/script emitting timing lines

## Documentation Requirements
- [ ] Add docstring explaining timing format expectations
- [ ] Update README quick start snippet with timing collector usage

## Dependencies
- **Blocked By**: T-007, T-004, T-005, T-006
- **Blocks**: T-012, T-013
- **Related**: T-009, T-010

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Requires subprocess handling and flexible parsing.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] Timing collector documented
- [ ] CLI command works end-to-end
