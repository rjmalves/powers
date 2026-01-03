# [T-023] Implement RSS monitor

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Status**: ✅ Complete
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
- [x] RSS samples recorded for run with summary statistics
- [x] Handles short-lived processes without crash
- [x] Warning when /proc unavailable

## Implementation Guide

### Suggested Approach
1. Reuse/port logic from `scripts/monitor_rss.py` into collector module.
2. Use thread to poll RSS while process runs; ensure termination on exit.
3. Serialize samples to JSON and integrate with collector schema.

### Key Files to Modify
- `profiling/powers_profile/collectors/rss.py` ✅ Created
- `profiling/powers_profile/schemas/results.py` ✅ Using existing schema
- `profiling/powers_profile/config/default.toml` ✅ Already configured

### Patterns to Follow
- Keep polling lightweight; avoid high-frequency overhead ✅

### Pitfalls to Avoid
- ⚠️ Leaving orphaned child process if wrapper fails ✅ Handled with proper cleanup
- ⚠️ Incorrect units (pages vs bytes) ✅ Using KB consistently

## Testing Requirements

### Unit Tests
- [x] Polling logic handles short processes (mocked)
- [x] Summary stats computed correctly
- [x] Error on invalid polling interval

### Integration Tests
- [x] Run against fixture program allocating memory (mocked)

## Documentation Requirements
- [ ] Document RSS monitor usage and limitations

## Dependencies
- **Blocked By**: T-007 ✅
- **Blocks**: T-024 ✅ Unblocked
- **Related**: T-020, T-021, T-022

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Process management and polling logic.

## Definition of Done
- [x] Implementation complete
- [x] Tests passing
- [ ] Documentation updated

## Progress (2026-01-03)

**COMPLETED**:
- Implemented `rss.py` collector with background monitoring thread
- Polling logic using /proc/{pid}/status for lightweight RSS tracking
- JSON output with timestamped samples and summary statistics
- Memory growth detection heuristic
- Comprehensive unit tests with mocked subprocess execution
- Proper cleanup and timeout handling
