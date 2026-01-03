# [T-024] Create unified memory collector

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Status**: ✅ Complete
> **Dependencies**: T-020, T-021, T-022, T-023
> **Blocks**: T-025

## Context

### Background
We need a single memory collector entry point that orchestrates DHAT, Massif, Cachegrind, and RSS as configured.

### Relation to Epic
Delivers `powers-profile run --collectors memory` end-to-end.

### Current State
Individual tool collectors exist after prior tickets; no unified orchestration.

## Specification

### Inputs
- Config specifying enabled memory tools and their options
- Target binary and args

### Outputs
- `memory_data.json` combining outputs from all selected tools
- Artifacts from each tool stored under memory directory

### Behavior
- Determine which sub-collectors to run based on config/CLI options
- Execute selected collectors sequentially (valgrind tools) with clear status per tool
- Aggregate results into combined JSON with tool-specific sections and summary metrics (peak RSS, peak heap, cache misses)
- Record warnings if any tool skipped or fails

### Error Handling
- If a sub-collector fails, mark status and continue others unless critical
- Validate config to avoid running conflicting tools simultaneously when not supported

## Acceptance Criteria
- [ ] `powers-profile run --collectors memory` runs configured tools and outputs combined JSON
- [ ] Per-tool status and metadata captured
- [ ] Skipped/failed tools recorded with reasons

## Implementation Guide

### Suggested Approach
1. Add `collectors/memory.py` that orchestrates DHAT/Massif/Cachegrind/RSS modules.
2. Provide CLI options to select tools (e.g., `--memory-tools dhat,massif`).
3. Aggregate results into schema-compliant payload.
4. Integrate with storage/reporter pipeline.

### Key Files to Modify
- `profiling/powers_profile/collectors/memory.py`
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/config/default.toml`

### Patterns to Follow
- Similar orchestration style to CPU collector (T-016)

### Pitfalls to Avoid
- ⚠️ Running multiple valgrind tools in parallel (should be sequential)
- ⚠️ Missing status flags for skipped tools

## Testing Requirements

### Unit Tests
- [ ] Config parsing for selected tools
- [ ] Aggregation merges tool payloads with status flags
- [ ] Failure in one tool does not crash overall collector

### Integration Tests
- [ ] Run with subset of tools enabled and verify outputs

## Documentation Requirements
- [ ] Document CLI options for selecting memory tools

## Dependencies
- **Blocked By**: T-020, T-021, T-022, T-023
- **Blocks**: T-025
- **Related**: Epic 5 visualization

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Orchestration and aggregation with existing collectors.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] CLI wiring verified
- [ ] Docs updated
