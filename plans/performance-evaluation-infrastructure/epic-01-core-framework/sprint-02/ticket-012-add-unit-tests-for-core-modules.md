# [T-012] Add unit tests for core modules

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-007, T-008, T-009
> **Blocks**: T-013

## Context

### Background
Core modules (collectors, reporters, storage) need regression coverage to stabilize the new framework.

### Relation to Epic
Satisfies Epic 1 acceptance criterion for unit tests across core modules.

### Current State
Minimal or no tests exist for collectors, reporters, and storage logic.

## Specification

### Inputs
- Fixtures for sample timing output and sample run metadata
- Temporary directories for storage tests

### Outputs
- Pytest suite covering collectors, reporters, storage, and CLI wiring

### Behavior
- Add tests validating collector registry, timing parsing, JSON reporter serialization, and history updates
- Ensure CLI commands wire components together (run, summary) using Typer test runner
- Target coverage > 80% for new modules

### Error Handling
- Tests should assert proper error messages on invalid config, missing files, or subprocess failures

## Acceptance Criteria
- [ ] Tests cover collector base + timing collector
- [ ] Tests cover JSON reporter serialization
- [ ] Tests cover history file updates and symlink behavior
- [ ] CLI tests for `run` and `summary` pass
- [ ] Coverage for core modules > 80%

## Implementation Guide

### Suggested Approach
1. Add pytest fixtures for sample stdout and temp directories.
2. Use `typer.testing.CliRunner` for CLI invocation.
3. Mock subprocess for timing collector parsing tests.
4. Add coverage config in `pyproject.toml` if needed.

### Key Files to Modify
- `profiling/tests/test_collectors.py`
- `profiling/tests/test_reporters.py`
- `profiling/tests/test_storage.py`
- `profiling/tests/test_cli.py`

### Patterns to Follow
- Match naming/style from existing tests in repo
- Prefer pure functions for ease of testing

### Pitfalls to Avoid
- ⚠️ Flaky tests due to time-dependent IDs; inject deterministic run_id where needed
- ⚠️ Tests requiring external tools; mock instead for timing collector

## Testing Requirements

### Unit Tests
- [ ] Collector registry and ABC enforcement
- [ ] Timing parsing with edge cases
- [ ] JSON reporter serialization/validation
- [ ] History append and symlink update

### Integration Tests
- [ ] CLI `run` and `summary` end-to-end using fixtures

### Performance Tests
- [ ] Ensure tests complete within seconds (no heavy subprocesses)

## Documentation Requirements
- [ ] Document how to run tests in README or CONTRIBUTING note

## Dependencies
- **Blocked By**: T-007, T-008, T-009
- **Blocks**: T-013
- **Related**: T-010, T-011

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Multiple modules and CLI paths to cover.

## Definition of Done
- [ ] Tests implemented
- [ ] Coverage target met
- [ ] All tests passing in CI/local
- [ ] Documentation updated
