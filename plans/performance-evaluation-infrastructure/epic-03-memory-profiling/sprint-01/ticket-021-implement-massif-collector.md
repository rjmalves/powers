# [T-021] Implement Massif collector

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Status**: ✅ Complete
> **Dependencies**: T-007, T-003
> **Blocks**: T-024

## Context

### Background
Massif tracks heap usage over time; we need to capture snapshots and parse peaks.

### Relation to Epic
Provides temporal memory profile required for dashboards and comparisons.

### Current State
No Massif integration or parsing exists.

## Specification

### Inputs
- Target binary and args
- Config: output file path, time unit (ms), snapshot limit

### Outputs
- `massif.out` file
- Parsed JSON with peak usage, snapshots timeline

### Behavior
- Run `valgrind --tool=massif --massif-out-file=<path> --time-unit=ms`
- Use `ms_print` or parser to extract peak and per-snapshot data
- Summarize peak heap, stack, bytes per snapshot, time indices

### Error Handling
- Validate valgrind availability
- Gracefully handle missing snapshots or malformed output
- Provide warning status when time-unit unsupported

## Acceptance Criteria
- [ ] Massif run produces massif.out and parsed JSON timeline
- [ ] Peak memory identified with snapshot id
- [ ] Metadata includes valgrind version and options

## Implementation Guide

### Suggested Approach
1. Add `collectors/massif.py` invoking valgrind and parsing ms_print output.
2. Extract snapshot list with time/bytes to JSON arrays.
3. Store artifacts under memory collector directory.

### Key Files to Modify
- `profiling/powers_profile/collectors/massif.py`
- `profiling/powers_profile/schemas/results.py`
- `profiling/powers_profile/config/default.toml`

### Patterns to Follow
- Use subprocess with explicit locale
- Keep parser tolerant of valgrind version differences

### Pitfalls to Avoid
- ⚠️ Large massif files; avoid loading entire file if not needed
- ⚠️ Misinterpreting units (bytes vs KB)

## Testing Requirements

### Unit Tests
- [ ] Parse fixture massif output to extract peak and snapshots
- [ ] Handle missing data gracefully

### Integration Tests
- [ ] Run against small fixture program to generate massif.out

## Documentation Requirements
- [ ] Document Massif config options and output fields

## Dependencies
- **Blocked By**: T-007, T-003
- **Blocks**: T-024
- **Related**: T-020, T-022

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Parsing non-trivial text output.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Documentation updated
