# [T-015] Implement FlameGraph integration

> **Epic**: [Epic 2: CPU & Execution Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014
> **Blocks**: T-016, T-018

## Context

### Background
We need automatic FlameGraph generation from perf stack samples to visualize call stack hotspots.

### Relation to Epic
Provides key visualization artifact for CPU profiling and differential comparisons.

### Current State
Perf wrapper not integrated with FlameGraph scripts; no SVG generation.

## Specification

### Inputs
- Folded stack file from `perf script` (stacks.txt)
- FlameGraph script paths (configurable)
- Config options: width, color scheme, min threshold

### Outputs
- `flamegraph.svg`
- `folded.txt` (intermediate)
- Metadata JSON for generation settings

### Behavior
- Convert perf script output to folded stacks via `stackcollapse-perf.pl`
- Run `flamegraph.pl` with configurable width/color/min threshold
- Validate FlameGraph scripts installed; provide install instructions if missing
- Support optional differential mode input (used later in T-018)

### Error Handling
- Clear errors when FlameGraph scripts missing
- Fail fast on malformed perf script output
- Clean up intermediate files on failure when appropriate

## Acceptance Criteria
- [ ] `flamegraph.svg` generated from perf samples
- [ ] Configurable width/color/min threshold
- [ ] Metadata file captures script versions/paths
- [ ] Errors guide user to install FlameGraph scripts

## Implementation Guide

### Suggested Approach
1. Add helper module `visualizers/flamegraph.py` or similar.
2. Locate FlameGraph scripts via config or env var; verify executable.
3. Invoke `stackcollapse-perf.pl` then `flamegraph.pl` using subprocess.
4. Store outputs under CPU collector directory.

### Key Files to Modify
- `profiling/powers_profile/visualizers/flamegraph.py`
- `profiling/powers_profile/config/default.toml`
- `profiling/powers_profile/collectors/perf.py`

### Patterns to Follow
- Log commands executed for reproducibility
- Keep outputs deterministic by fixing width/color defaults

### Pitfalls to Avoid
- ⚠️ Assuming FlameGraph scripts on PATH
- ⚠️ Losing stderr output; capture for troubleshooting

## Testing Requirements

### Unit Tests
- [ ] Path resolution for FlameGraph scripts
- [ ] Command assembly with custom options
- [ ] Error on missing scripts

### Integration Tests
- [ ] Run against sample folded stacks to produce SVG (can be small fixture)

## Documentation Requirements
- [ ] Document required FlameGraph installation and config keys

## Dependencies
- **Blocked By**: T-014
- **Blocks**: T-016, T-018
- **Related**: T-017

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: External scripts with path resolution and error handling.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] SVG generated in sample run
- [ ] Docs updated
