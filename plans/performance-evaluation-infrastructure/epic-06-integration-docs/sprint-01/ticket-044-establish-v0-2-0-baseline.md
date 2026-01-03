# [T-044] Establish v0.2.0 baseline

> **Epic**: [Epic 6: Integration & Documentation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: Epics 1-5 complete
> **Blocks**: T-045, T-047

## Context

### Background
Need committed baseline profiling results for current POWE.RS release to enable regression detection.

### Relation to Epic
Provides reference data used by comparison workflows and documentation.

### Current State
No standardized baseline run committed.

## Specification

### Inputs
- Built POWE.RS binary (release)
- Profiling suite commands across CPU, memory, parallel, timing
- Target environment specs (recorded)

### Outputs
- Baseline artifacts under `profiling_results/baselines/v0.2.0/`
- Metadata describing hardware, git SHA, config

### Behavior
- Run full profiling suite with stable config (document seeds/problem size)
- Store results, flamegraphs, dashboards, markdown reports in baseline directory
- Record environment (CPU/RAM/OS), commit SHA, command args
- Update history index referencing baseline

### Error Handling
- If run unstable/flaky, rerun and note variance
- Ensure artifacts small enough for repo (prune large raw files if needed)

## Acceptance Criteria
- [ ] Baseline artifacts committed under baselines/v0.2.0
- [ ] Metadata file includes system info and config
- [ ] History updated to include baseline run_id

## Implementation Guide

### Suggested Approach
1. Build release binary; run `powers-profile run --suite full` with consistent args.
2. Store outputs under baseline path (copy from run directory).
3. Record README in baseline folder summarizing environment and commands.

### Key Files to Modify
- `profiling_results/baselines/v0.2.0/*` (new)
- `profiling_results/history.json`
- `docs/profiling/BASELINES.md` (if needed)

### Patterns to Follow
- Keep artifacts minimal but sufficient (JSON, SVG, dashboard, markdown)

### Pitfalls to Avoid
- ⚠️ Committing huge perf.data; prefer compressed or summarized data

## Testing Requirements

### Validation
- [ ] Verify baseline run can be loaded by CLI summary/dashboard

## Documentation Requirements
- [ ] Document baseline location and creation steps

## Dependencies
- **Blocked By**: Epics 1-5 complete
- **Blocks**: T-045, T-047
- **Related**: T-019

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Execution of profiling suite and artifact organization.

## Definition of Done
- [ ] Baseline created and documented
- [ ] History updated
- [ ] Artifacts committed
