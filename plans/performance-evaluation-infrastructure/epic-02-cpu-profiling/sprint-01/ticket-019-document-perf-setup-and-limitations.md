# [T-019] Document perf setup and limitations

> **Epic**: [Epic 2: CPU & Execution Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014, T-015, T-016, T-017, T-018
> **Blocks**: None

## Context

### Background
perf requires permissions and setup that vary by environment (WSL2 vs bare-metal). Clear documentation reduces support burden.

### Relation to Epic
Completes Epic 2 acceptance criteria for documented perf setup and limitations.

### Current State
No user-facing guidance exists for perf installation or troubleshooting.

## Specification

### Inputs
- Confirmed commands from T-014/T-016 implementations
- Known limitations on WSL2 and cloud instances

### Outputs
- Docs section covering install, permissions, common errors, troubleshooting
- CLI help updates referencing docs

### Behavior
- Document installation commands (`linux-tools-$(uname -r)`, `apt install perf`), FlameGraph scripts clone
- Document `perf_event_paranoid` settings and `sudo` requirements
- Provide WSL2 notes (sampling restrictions) and bare-metal notes
- Add troubleshooting table mapping errors to fixes

### Error Handling
- N/A (docs), but include warning callouts for privilege requirements

## Acceptance Criteria
- [ ] Documentation includes install steps, permissions, WSL2 notes
- [ ] Troubleshooting table with common perf errors
- [ ] CLI help links to docs

## Implementation Guide

### Suggested Approach
1. Add section to `docs/profiling/TOOLS_REFERENCE.md` or similar.
2. Include commands for checking perf availability and setting capabilities.
3. Add note to CLI `--help` pointing to docs URL/path.

### Key Files to Modify
- `docs/profiling/TOOLS_REFERENCE.md` (new or updated)
- `profiling/powers_profile/cli.py` (help text)

### Patterns to Follow
- Concise steps with commands users can copy
- Highlight warnings using markdown callouts

### Pitfalls to Avoid
- ⚠️ Assuming root access; provide alternatives
- ⚠️ Ignoring kernel version mismatches for perf tools

## Testing Requirements

### Documentation Checks
- [ ] Commands validated on WSL2 and bare-metal notes

## Documentation Requirements
- [ ] Docs updated with perf guidance
- [ ] CLI help references docs

## Dependencies
- **Blocked By**: T-014, T-015, T-016, T-017, T-018
- **Blocks**: None
- **Related**: Epic 6 documentation tasks

## Effort Estimate
**Points**: 2
**Confidence**: High
**Rationale**: Documentation using completed implementation details.

## Definition of Done
- [ ] Documentation merged
- [ ] Help text updated
- [ ] References added to plan
