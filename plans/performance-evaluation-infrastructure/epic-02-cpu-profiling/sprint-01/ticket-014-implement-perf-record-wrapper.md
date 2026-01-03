# [T-014] Implement perf record wrapper

> **Epic**: [Epic 2: CPU & Execution Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-007
> **Blocks**: T-015, T-016, T-017

## Context

### Background
We need a robust wrapper around `perf record`/`perf script` to capture CPU samples with consistent options and artifact paths.

### Relation to Epic
Foundation for CPU collector; produces raw perf data consumed by FlameGraph and hotspot parsing.

### Current State
Collector base exists; no perf integration yet.

## Specification

### Inputs
- Target binary path and args
- Config: sampling frequency, events, duration, output directory
- Environment: optional `PERF_RECORD_ARGS`, `PERF_EVENT_PARANOID` guidance

### Outputs
- `perf.data` binary file
- `stacks.txt` from `perf script`
- Metadata JSON describing perf command and environment

### Behavior
- Run `perf record` with default events (`cycles:u`), frequency (e.g., 99 Hz), and output path under run directory
- Run `perf script` to emit folded stacks input for FlameGraph
- Detect missing permissions and provide remediation message
- Allow configurable perf events and frequency via config

### Error Handling
- If `perf` unavailable or lacks permissions, raise clear error with setup instructions
- Validate user-supplied events/frequency; reject unsafe/unsupported values
- Clean up partial artifacts on failure

## Acceptance Criteria
- [ ] `perf.data` and `stacks.txt` produced for a sample run
- [ ] Configurable sampling frequency/events
- [ ] Errors include instructions for setting `perf_event_paranoid`
- [ ] Metadata JSON saved with command options

## Implementation Guide

### Suggested Approach
1. Add `perf.py` helper under `collectors` or `utils` to invoke perf.
2. Use `subprocess.run` with `check=True` and capture stderr for diagnostics.
3. Create helper to detect permission issues (exit code/messages) and suggest `sudo sysctl -w kernel.perf_event_paranoid=1`.
4. Write metadata file to output directory for traceability.

### Key Files to Modify
- `profiling/powers_profile/collectors/perf.py` (new helper)
- `profiling/powers_profile/config/default.toml`
- `profiling/powers_profile/collectors/base.py`

### Patterns to Follow
- Reuse storage paths from Epic 1
- Keep commands logged for reproducibility

### Pitfalls to Avoid
- ⚠️ Running perf without checking availability/permissions
- ⚠️ Overwriting existing perf artifacts

## Testing Requirements

### Unit Tests
- [ ] Build perf command with default and custom options
- [ ] Detect missing perf binary (mock `shutil.which`)
- [ ] Detect permission error message mapping to guidance

### Integration Tests
- [ ] Dry-run mode that skips execution but writes metadata (for CI)

## Documentation Requirements
- [ ] Docstring for perf helper
- [ ] Note default perf settings in README/config comments

## Dependencies
- **Blocked By**: T-007
- **Blocks**: T-015, T-016, T-017
- **Related**: T-018

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: System tool integration with error handling.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] Perf artifacts produced for sample run
- [ ] Guidance for permissions documented
