# [T-018] Implement differential flamegraph

> **Epic**: [Epic 2: CPU & Execution Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-015, T-016
> **Blocks**: T-019

## Context

### Background
Differential flamegraphs highlight regressions between two runs, enabling quick hotspot comparison.

### Relation to Epic
Completes comparison capability promised in acceptance criteria.

### Current State
FlameGraph generation exists (T-015); no diff support or CLI wiring.

## Specification

### Inputs
- Two folded stack files (baseline and target)
- Config: color palette for regressions/improvements, output path

### Outputs
- `flamegraph-diff.svg`
- Metadata JSON noting source runs and options

### Behavior
- Use `difffolded.pl` (FlameGraph) to generate diff folded stacks
- Run `flamegraph.pl --negate` or appropriate diff mode to highlight changes
- Integrate with `powers-profile compare` to produce diff when CPU data present
- Embed diff SVG path into comparison output JSON

### Error Handling
- Handle missing FlameGraph diff scripts with clear message
- Validate both folded stack inputs exist before running
- Return warning when folded stacks incompatible (e.g., mismatched symbols)

## Acceptance Criteria
- [ ] `powers-profile compare` produces `flamegraph-diff.svg` when CPU data available
- [ ] Colors show regressions vs improvements
- [ ] Metadata tracks baseline/target run ids
- [ ] Clear errors when diff scripts missing

## Implementation Guide

### Suggested Approach
1. Add helper for diff generation using FlameGraph scripts.
2. Extend CLI compare command to locate folded stacks for two runs.
3. Produce diff SVG and store under comparison output directory.
4. Update comparison JSON to reference diff SVG path.

### Key Files to Modify
- `profiling/powers_profile/visualizers/flamegraph.py`
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/analyzers/comparison.py`

### Patterns to Follow
- Mirror single-run flamegraph generation flow
- Keep output naming deterministic (`flamegraph-diff.svg`)

### Pitfalls to Avoid
- ⚠️ Missing folded stacks when CPU collector was skipped
- ⚠️ Color scheme confusing; choose clear regression colors

## Testing Requirements

### Unit Tests
- [ ] Diff command built correctly with provided folded stacks
- [ ] Error when input files missing

### Integration Tests
- [ ] Compare two fixture folded stacks and generate SVG

## Documentation Requirements
- [ ] Document diff generation in compare command help

## Dependencies
- **Blocked By**: T-015, T-016
- **Blocks**: T-019
- **Related**: T-014, T-017

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: CLI integration plus external scripts.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] Diff SVG generated in comparison
- [ ] Docs updated
