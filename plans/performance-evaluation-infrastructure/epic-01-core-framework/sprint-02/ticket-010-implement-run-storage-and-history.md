# [T-010] Implement run storage and history

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-003, T-005
> **Blocks**: T-011, T-012, T-013

## Context

### Background
Profiling runs need deterministic storage with history tracking to enable comparisons and summaries.

### Relation to Epic
Provides persistence layer for runs and an index (`history.json`) enabling `summary` and `history` commands.

### Current State
Output directory structure is specified but not implemented; no history index exists.

## Specification

### Inputs
- `ProfilingRun` results (paths to artifacts, metadata)
- Output root (default `profiling_results/`)

### Outputs
- Run directory under `profiling_results/runs/<timestamp>_<sha>`
- `history.json` append-only index with run metadata

### Behavior
- Create run directory with subfolders per collector and reporter outputs
- Write copy of config, system info, git info into run folder
- Update `history.json` with run_id, git SHA, timestamp, collectors used, duration summary
- Maintain stable symlink `profiling_results/runs/latest` to most recent run

### Error Handling
- Fail if output root not writable
- Prevent duplicate run_id collisions by generating unique IDs
- Roll back partial directory on failure

## Acceptance Criteria
- [ ] Runs stored under timestamped directories with collector artifacts
- [ ] `history.json` records run metadata chronologically
- [ ] `latest` symlink updated atomically
- [ ] Storage layout documented

## Implementation Guide

### Suggested Approach
1. Implement storage helper module (e.g., `utils/paths.py` or new `storage.py`).
2. Define run_id format `<YYYYMMDD_HHMMSS>_<shortsha>`.
3. Write history index update function with file lock to avoid corruption.
4. Wire storage into `powers-profile run` flow after collectors finish.

### Key Files to Modify
- `profiling/powers_profile/utils/paths.py`
- `profiling/powers_profile/cli.py`
- `profiling/powers_profile/schemas/run.py`

### Patterns to Follow
- Use pathlib and atomic writes (write temp then replace)
- Keep JSON schema alignment for history entries

### Pitfalls to Avoid
- ⚠️ Overwriting previous runs due to non-unique IDs
- ⚠️ Leaving broken symlink on failure

## Testing Requirements

### Unit Tests
- [ ] Storage path generation uses timestamp + SHA
- [ ] History append preserves order
- [ ] Symlink update is atomic

### Integration Tests
- [ ] Running two sequential profiles creates two entries in `history.json`

## Documentation Requirements
- [ ] Document storage layout in README/QUICK_START
- [ ] Add comments in storage helper

## Dependencies
- **Blocked By**: T-003, T-005
- **Blocks**: T-011, T-012, T-013
- **Related**: T-009

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Filesystem work with care for atomicity.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests added and passing
- [ ] Storage documented
- [ ] CLI writes history entries
