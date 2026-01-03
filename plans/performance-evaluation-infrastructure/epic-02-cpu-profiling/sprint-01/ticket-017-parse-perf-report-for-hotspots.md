# [T-017] Parse perf report for hotspots

> **Epic**: [Epic 2: CPU & Execution Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014
> **Blocks**: T-016, T-018

## Context

### Background
We need structured hotspot data (top functions by samples) from `perf report` or `perf script` output.

### Relation to Epic
Feeds CPU collector JSON summary and dashboard hotspot tables.

### Current State
No parser exists for perf output; only raw data from `perf record`/`perf script`.

## Specification

### Inputs
- `perf.data` and/or `perf script` output
- Config: number of hotspots to return (default 20)

### Outputs
- List of hotspot entries `{symbol, samples, percent, dso}`
- Optional call stack snippets for top entries

### Behavior
- Invoke `perf report --stdio` with filters to aggregate by symbol
- Parse output into structured entries sorted by sample percentage
- Support optional demangling flag (`--demangle`)
- Provide summarized totals (total samples, top percent coverage)

### Error Handling
- Handle missing symbols gracefully (unknown/anon entries)
- Propagate errors when `perf report` fails or not found
- Validate integer limits for hotspot count

## Acceptance Criteria
- [ ] Hotspot list extracted with symbol, percent, sample count
- [ ] Configurable top-N limit
- [ ] Demangle option supported
- [ ] Parser covered by unit tests with fixture output

## Implementation Guide

### Suggested Approach
1. Run `perf report --stdio` with `--percentage absolute` or similar for consistent output.
2. Parse lines using regex to extract percent, samples, symbol, DSO.
3. Provide helper function returning list/dict for JSON serialization.
4. Integrate into CPU collector payload.

### Key Files to Modify
- `profiling/powers_profile/collectors/perf.py`
- `profiling/powers_profile/collectors/cpu.py`
- `profiling/tests/fixtures/perf_report.txt`

### Patterns to Follow
- Keep parser tolerant to whitespace and demangled names

### Pitfalls to Avoid
- ⚠️ Locale-specific decimal separators; force C locale
- ⚠️ Large perf reports; limit to top N for speed

## Testing Requirements

### Unit Tests
- [ ] Parse fixture perf report to extract top entries
- [ ] Verify demangle flag toggles expected command
- [ ] Handle unknown symbols without crash

### Integration Tests
- [ ] CPU collector uses parser output in JSON payload

## Documentation Requirements
- [ ] Document hotspot fields in schema comments

## Dependencies
- **Blocked By**: T-014
- **Blocks**: T-016, T-018
- **Related**: T-015

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Deterministic parsing with fixtures.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] Integrated into CPU collector
