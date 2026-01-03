# [T-030] Create parallel collector

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-026, T-027, T-028, T-029
> **Blocks**: T-031, T-032

## Context

### Background
Need a collector that orchestrates scaling runs, contention metrics, and analysis into a single JSON payload.

### Relation to Epic
Delivers `powers-profile scaling` command output and stores artifacts.

### Current State
Scaling runner and analytics exist separately; collector integration missing.

## Specification

### Inputs
- Thread counts, repetitions, warmup settings
- Flags for contention collection and perf options

### Outputs
- `scaling_data.json` with timing stats, speedup, efficiency, Amdahl estimates, contention metrics
- Logs per thread count run

### Behavior
- Run scaling harness, compute metrics, attach contention data when enabled
- Store artifacts under `parallel/` collector directory
- Record warnings when data missing/unavailable
- Register collector for CLI `run --collectors parallel` and dedicated `scaling` command

### Error Handling
- If any run fails, mark status and continue with remaining thread counts when possible
- Validate config (thread list, repetitions) before execution

## Acceptance Criteria
- [ ] `powers-profile scaling --threads ...` writes scaling_data.json
- [ ] JSON includes timing stats, speedup, efficiency, Amdahl, contention (if enabled)
- [ ] Collector handles partial failures with status per thread count

## Implementation Guide

### Suggested Approach
1. Create `collectors/parallel.py` using scaling runner and analysis helpers.
2. Integrate with storage/history and CLI.
3. Ensure schema alignment for JSON payload and reporter.

### Key Files to Modify
- `profiling/powers_profile/collectors/parallel.py`
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/schemas/results.py`

### Patterns to Follow
- Mirror CPU/memory collectors for structure

### Pitfalls to Avoid
- ⚠️ Mixing data across runs; tie metrics to run_id and thread list

## Testing Requirements

### Unit Tests
- [ ] Collector aggregates metrics correctly
- [ ] Handles contention optional path
- [ ] Error handling for failed thread count

### Integration Tests
- [ ] Scaling command produces expected JSON for fixture workload

## Documentation Requirements
- [ ] Update CLI help and config docs for scaling collector

## Dependencies
- **Blocked By**: T-026, T-027, T-028, T-029
- **Blocks**: T-031, T-032
- **Related**: Epic 5 scaling charts

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Orchestration of existing helpers.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
