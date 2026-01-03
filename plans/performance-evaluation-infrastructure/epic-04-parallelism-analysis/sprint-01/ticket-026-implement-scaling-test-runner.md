# [T-026] Implement scaling test runner

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-007, T-010
> **Blocks**: T-027, T-029, T-030

## Context

### Background
Need automated harness to run target binary across a set of thread counts and collect timing data.

### Relation to Epic
Foundation for scaling analysis and efficiency calculations.

### Current State
No scaling harness; CLI accepts no scaling command yet.

## Specification

### Inputs
- Thread counts list (e.g., `1,2,4,8,16,32`)
- Target binary and args
- Config: warmup iterations, repetitions, environment vars (RAYON_NUM_THREADS)

### Outputs
- Raw timing results per thread count (JSON)
- Logs per run

### Behavior
- For each thread count: set env, optional warmup, run N iterations, capture durations
- Store per-iteration durations; compute mean/stddev/min/max
- Support optional pinning/affinity flags (future NUMA prep) placeholder
- Respect global timeout

### Error Handling
- Abort with clear message if binary fails for any thread count
- Continue to next count when non-critical errors flagged with warning option
- Validate thread list >0 and unique

## Acceptance Criteria
- [ ] Scaling harness runs across configured thread counts and records per-iteration times
- [ ] Supports warmup and repetitions configuration
- [ ] Environment variables applied per run
- [ ] Outputs JSON for downstream analysis

## Implementation Guide

### Suggested Approach
1. Add `scaling_runner.py` under `collectors/parallel` or `analyzers`.
2. Use high-resolution timer (time.perf_counter) and ensure monotonic timing.
3. Serialize results per thread count to structured dict.
4. Expose via CLI `scaling` command or collector hook.

### Key Files to Modify
- `profiling/powers_profile/collectors/scaling_runner.py`
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/config/default.toml`

### Patterns to Follow
- Use consistent storage layout from Epic 1

### Pitfalls to Avoid
- ⚠️ Including warmup in measured iterations
- ⚠️ Not resetting env between runs

## Testing Requirements

### Unit Tests
- [ ] Runner handles list of thread counts and records durations
- [ ] Warmup iterations excluded from measurements
- [ ] Error raised on invalid thread list

### Integration Tests
- [ ] Run against fixture workload (fast command) across small thread set

## Documentation Requirements
- [ ] Document scaling command/config keys

## Dependencies
- **Blocked By**: T-007, T-010
- **Blocks**: T-027, T-029, T-030
- **Related**: T-031

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Requires orchestrating multiple runs and timing.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] CLI wiring verified
