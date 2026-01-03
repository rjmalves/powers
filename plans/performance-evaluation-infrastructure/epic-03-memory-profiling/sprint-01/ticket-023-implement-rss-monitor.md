# [T-023] Implement RSS monitor

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-007
> **Blocks**: T-024

## Context

### Background
Need lightweight RSS tracking to replace legacy scripts and capture physical memory usage during runs.

### Relation to Epic
Completes memory profiling coverage beyond valgrind tools.

### Current State
Old `scripts/monitor_rss.py` exists but not integrated; no collector wiring.

## Specification

### Inputs
- Target binary and args
- Config: polling interval, pid selection, output path

### Outputs
- `rss_data.json` with timestamped RSS samples and summary stats (min/max/mean)

### Behavior
- Launch target process and poll RSS via `/proc/<pid>/statm` or psutil
- Record timestamps relative to start and RSS bytes
- Detect steady-state vs growth (optional simple heuristic)
- Stop polling when process exits; handle child processes if needed

### Error Handling
- Handle permission errors when reading /proc
- Timeout if process hangs; ensure cleanup
- Validate polling interval > 0

## Acceptance Criteria
- [ ] RSS samples recorded for run with summary statistics
- [ ] Handles short-lived processes without crash
- [ ] Warning when /proc unavailable

## Implementation Guide

### Suggested Approach
1. Reuse/port logic from `scripts/monitor_rss.py` into collector module.
2. Use thread to poll RSS while process runs; ensure termination on exit.
3. Serialize samples to JSON and integrate with collector schema.

### Key Files to Modify
- `profiling/powers_profile/collectors/rss.py`
- `profiling/powers_profile/schemas/results.py`
- `profiling/powers_profile/config/default.toml`

### Patterns to Follow
- Keep polling lightweight; avoid high-frequency overhead

### Pitfalls to Avoid
- ⚠️ Leaving orphaned child process if wrapper fails
- ⚠️ Incorrect units (pages vs bytes)

## Testing Requirements

### Unit Tests
- [ ] Polling logic handles short processes (mocked)
- [ ] Summary stats computed correctly
- [ ] Error on invalid polling interval

### Integration Tests
- [ ] Run against fixture program allocating memory

## Documentation Requirements
- [ ] Document RSS monitor usage and limitations

## Dependencies
- **Blocked By**: T-007
- **Blocks**: T-024
- **Related**: T-020, T-021, T-022

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Process management and polling logic.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Documentation updated
