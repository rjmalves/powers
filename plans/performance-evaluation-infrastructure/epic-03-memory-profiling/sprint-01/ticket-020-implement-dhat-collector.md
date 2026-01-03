# [T-020] Implement DHAT collector

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-007, T-003
> **Blocks**: T-024

## Context

### Background
DHAT provides heap allocation profiling; we need to integrate and parse its output into structured JSON.

### Relation to Epic
One of the core memory tools; feeds unified memory collector.

### Current State
No DHAT integration; schemas exist for collector results.

## Specification

### Inputs
- Target binary and args
- Config: dhat output path, include_child option

### Outputs
- `dhat.out` raw file
- Parsed `dhat.json` with allocation metrics (bytes, blocks, hotspots)

### Behavior
- Run `valgrind --tool=dhat --dhat-out-file=<path>` around target
- Parse dhat.out using `dhat.py` or manual parser to extract heap profiles
- Summarize top allocation sites and totals
- Capture valgrind version in metadata

### Error Handling
- Fail gracefully when valgrind missing
- Handle dhat output parsing errors with clear message
- Allow skip on unsupported platforms with warning status

## Acceptance Criteria
- [ ] DHAT run produces dhat.out and parsed JSON
- [ ] Top allocation hotspots captured with sizes and counts
- [ ] Metadata includes valgrind version and command line

## Implementation Guide

### Suggested Approach
1. Add `collectors/dhat.py` wrapping valgrind invocation.
2. Parse dhat.out via Python (valgrind-provided parser or regex) into structured dict.
3. Store artifacts under memory collector directory.

### Key Files to Modify
- `profiling/powers_profile/collectors/dhat.py`
- `profiling/powers_profile/config/default.toml`
- `profiling/powers_profile/schemas/results.py`

### Patterns to Follow
- Reuse collector base error handling
- Keep parsing resilient to whitespace

### Pitfalls to Avoid
- ⚠️ Running without ensuring output directory exists
- ⚠️ Consuming entire dhat.out in memory if large; stream parse if needed

## Testing Requirements

### Unit Tests
- [ ] Parse fixture dhat.out to produce hotspots
- [ ] Error on missing valgrind binary (mocked)

### Integration Tests
- [ ] Run against small fixture program to generate dhat.out

## Documentation Requirements
- [ ] Document DHAT usage and config keys

## Dependencies
- **Blocked By**: T-007, T-003
- **Blocks**: T-024
- **Related**: T-021, T-022

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: External tool integration with parsing complexity.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Documentation updated
