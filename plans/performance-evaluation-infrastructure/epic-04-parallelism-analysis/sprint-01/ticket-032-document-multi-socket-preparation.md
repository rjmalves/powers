# [T-032] Document multi-socket preparation

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: All above
> **Blocks**: None

## Context

### Background
Running scaling on high-core, multi-socket systems (e.g., 192-core c7a.48xlarge) requires guidance on NUMA, pinning, and contention considerations.

### Relation to Epic
Ensures scalability analysis is usable on production-scale hardware.

### Current State
No documentation for multi-socket preparation exists.

## Specification

### Inputs
- Knowledge from scaling runner and perf contention implementation
- Target environments: bare-metal 32 cores, AWS 192 cores

### Outputs
- Documentation section detailing setup steps and best practices

### Behavior
- Describe NUMA considerations, core pinning strategies, and environment variables
- Provide example commands for taskset/numactl usage
- Recommend thread count sets for large systems
- Include troubleshooting tips for perf on multi-socket hosts

### Error Handling
- N/A (documentation), but include warnings about misconfiguration impacts

## Acceptance Criteria
- [ ] Documentation covers NUMA/pinning guidance and example commands
- [ ] Recommendations for thread count selection up to 192 cores
- [ ] Links to scaling collector config

## Implementation Guide

### Suggested Approach
1. Add section to `docs/profiling/ANALYSIS_GUIDE.md` or new `SCALING_GUIDE.md`.
2. Include examples with `taskset`, `numactl`, `RAYON_NUM_THREADS`.
3. Document interpretation of efficiency at high core counts.

### Key Files to Modify
- `docs/profiling/ANALYSIS_GUIDE.md` or `SCALING_GUIDE.md`
- Plan README references

### Patterns to Follow
- Concise command snippets with explanations

### Pitfalls to Avoid
- ⚠️ Assuming uniform memory access; highlight NUMA caveats

## Testing Requirements

### Documentation Checks
- [ ] Validate commands syntactically

## Documentation Requirements
- [ ] Docs updated and linked from plan

## Dependencies
- **Blocked By**: All above
- **Blocks**: None
- **Related**: Epic 6 documentation tasks

## Effort Estimate
**Points**: 2
**Confidence**: High
**Rationale**: Documentation leveraging implemented tools.

## Definition of Done
- [ ] Documentation merged
- [ ] References added to plan
