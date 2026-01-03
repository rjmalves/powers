# [T-016] Implement CPU collector

> **Epic**: [Epic 2: CPU & Execution Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014, T-015, T-007
> **Blocks**: T-018, T-019

## Context

### Background
Need a collector that orchestrates perf sampling, flamegraph generation, and CPU metrics packaging into JSON.

### Relation to Epic
Central deliverable enabling `powers-profile run --collectors cpu`.

### Current State
Perf and FlameGraph helpers will exist from T-014/T-015; collector glue missing.

## Specification

### Inputs
- Target binary and args
- CPU collector config (events, freq, flamegraph options, sample duration)

### Outputs
- `cpu_data.json` with hotspots summary, perf metadata, flamegraph path
- Artifacts: `perf.data`, `stacks.txt`, `folded.txt`, `flamegraph.svg`

### Behavior
- Invoke perf wrapper to collect samples
- Generate flamegraph via helper
- Compute summary metrics (sample count, hottest symbols, event totals)
- Record tool versions and command lines
- Respect `--collectors cpu` flag in CLI run

### Error Handling
- Bubble up perf/FlameGraph errors with context
- Mark collector status `warning` when perf unsupported (WSL2) but continue run
- Validate config keys and default values

## Acceptance Criteria
- [ ] `powers-profile run --collectors cpu` produces cpu_data.json and flamegraph.svg
- [ ] Summary hotspots list included (top N)
- [ ] Configurable perf events/frequency applied
- [ ] Collector status reflects warnings when perf unavailable

## Implementation Guide

### Suggested Approach
1. Create `collectors/cpu.py` implementing `Collector`.
2. Use perf + flamegraph helpers; store artifacts under collector directory.
3. Build JSON payload with hotspots (from T-017), tool versions, command lines.
4. Register CPU collector in registry and CLI.

### Key Files to Modify
- `profiling/powers_profile/collectors/cpu.py`
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/schemas/results.py`

### Patterns to Follow
- Mirror timing collector structure for consistency
- Keep artifact paths relative to run directory

### Pitfalls to Avoid
- ⚠️ Hardcoding perf events unsuitable for WSL2
- ⚠️ Missing cleanup of intermediate files on failure

## Testing Requirements

### Unit Tests
- [ ] Collector builds commands with defaults and overrides
- [ ] Warning status set when perf unsupported (mocked)
- [ ] JSON payload includes artifact paths and hotspots placeholder

### Integration Tests
- [ ] Run CPU collector against small fixture to produce SVG

## Documentation Requirements
- [ ] Docstring for CPU collector
- [ ] Update CLI help for cpu collector options

## Dependencies
- **Blocked By**: T-014, T-015, T-007
- **Blocks**: T-018, T-019
- **Related**: T-017

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Orchestration of helpers and schema wiring.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] CLI wiring verified
- [ ] Docs updated
