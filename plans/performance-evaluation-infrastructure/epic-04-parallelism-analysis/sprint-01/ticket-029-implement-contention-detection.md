# [T-029] Implement contention detection

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-026, T-014
> **Blocks**: T-030

## Context

### Background
Understanding lock contention and synchronization overhead is crucial for scaling; perf can surface futex/wait events.

### Relation to Epic
Adds contention insights to scaling results.

### Current State
No contention metrics collected during scaling runs.

## Specification

### Inputs
- Target binary executed under scaling runner
- Optional perf event selection for contention (e.g., `sched:sched_stat_blocked`, `futex`) 

### Outputs
- Contention metrics JSON (wait time, count per event, optional top contended locks if available)
- Perf raw data artifacts if collected

### Behavior
- Optionally wrap scaling runs with perf events focused on contention (config flag)
- Parse perf script/report to extract wait time totals per thread or lock symbol
- Aggregate into per-thread-count contention summary

### Error Handling
- If perf unavailable, set warning status and skip without failing scaling run
- Guard against high overhead by allowing opt-out per config

## Acceptance Criteria
- [ ] Contention metrics collected when enabled in config
- [ ] Warnings emitted when perf unavailable or permissions lacking
- [ ] Metrics included in scaling JSON with per-thread-count breakdown

## Implementation Guide

### Suggested Approach
1. Extend scaling runner to optionally run perf with contention events.
2. Parse perf output (reuse helper from Epic 2) for wait times/counts.
3. Store metrics alongside timing data for each thread count.

### Key Files to Modify
- `profiling/powers_profile/collectors/scaling_runner.py`
- `profiling/powers_profile/collectors/perf.py`
- `profiling/powers_profile/schemas/results.py`

### Patterns to Follow
- Keep contention optional to limit overhead

### Pitfalls to Avoid
- ⚠️ Excessive overhead when collecting contention events
- ⚠️ Mixing contention metrics across thread counts incorrectly

## Testing Requirements

### Unit Tests
- [ ] Contention parsing with fixture perf output
- [ ] Warning path when perf unavailable

### Integration Tests
- [ ] Scaling run with contention enabled produces metrics

## Documentation Requirements
- [ ] Document config flags for contention collection

## Dependencies
- **Blocked By**: T-026, T-014
- **Blocks**: T-030
- **Related**: T-017

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: perf integration with analysis per thread count.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Documentation updated
