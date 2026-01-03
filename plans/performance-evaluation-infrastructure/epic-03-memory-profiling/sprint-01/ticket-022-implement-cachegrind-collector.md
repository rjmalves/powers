# [T-022] Implement Cachegrind collector

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-007, T-003
> **Blocks**: T-024

## Context

### Background
Cachegrind provides cache miss and branch prediction statistics; we need structured extraction for reporting.

### Relation to Epic
Adds cache efficiency metrics to memory profiling domain.

### Current State
No Cachegrind integration or parsers implemented.

## Specification

### Inputs
- Target binary and args
- Config: cache options, output file path

### Outputs
- `cachegrind.out` file
- Parsed JSON with instruction counts, cache misses (I1, D1, LL), branch stats

### Behavior
- Run `valgrind --tool=cachegrind --cachegrind-out-file=<path>`
- Use `cg_annotate` or parser to extract aggregate statistics
- Capture top functions by cache misses (top N configurable)

### Error Handling
- Handle valgrind absence with clear message
- Skip hotspot extraction when cg_annotate unavailable (warning)
- Validate config values before run

## Acceptance Criteria
- [ ] Cachegrind run produces output and parsed JSON metrics
- [ ] Top functions by cache misses listed
- [ ] Metadata includes valgrind version and options

## Implementation Guide

### Suggested Approach
1. Add `collectors/cachegrind.py` invoking valgrind.
2. Parse `cg_annotate` output into summary metrics and hotspots.
3. Store artifacts under memory collector directory.

### Key Files to Modify
- `profiling/powers_profile/collectors/cachegrind.py`
- `profiling/powers_profile/schemas/results.py`
- `profiling/powers_profile/config/default.toml`

### Patterns to Follow
- Similar parsing approach as perf hotspots (T-017)

### Pitfalls to Avoid
- ⚠️ Large output; limit top N functions
- ⚠️ Units confusion (misses vs percent)

## Testing Requirements

### Unit Tests
- [ ] Parse fixture cg_annotate output
- [ ] Handle missing cg_annotate (warning)

### Integration Tests
- [ ] Run against small fixture to produce cachegrind.out

## Documentation Requirements
- [ ] Document Cachegrind options and output fields

## Dependencies
- **Blocked By**: T-007, T-003
- **Blocks**: T-024
- **Related**: T-020, T-021

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Similar to perf parsing with valgrind specifics.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Documentation updated
