# Sprint 2: CLI Integration and SDDP Loop Connection

> **Epic**: [Foundation](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: Not Started

## Goals

- **Primary**: Connect display system to SDDP training loop
- **Secondary**: Add CLI flags for profile selection
- **Tertiary**: Extend config.json schema for display settings

## Tickets

| ID | Title | Points | Dependencies | Assignable |
|----|-------|--------|--------------|------------|
| T-008 | [Extend CLI with display flags](./ticket-008-extend-cli-flags.md) | 3 | Sprint 1 | Yes |
| T-009 | [Extend config schema for display section](./ticket-009-extend-config-schema.md) | 2 | T-008 | Yes |
| T-010 | [Build DisplayContext from iteration data](./ticket-010-build-display-context.md) | 4 | Sprint 1 | Yes |
| T-011 | [Expose first-stage branching costs](./ticket-011-expose-first-stage-costs.md) | 3 | T-010 | Yes |
| T-012 | [Integrate display system with training loop](./ticket-012-integrate-training-loop.md) | 4 | T-010, T-011 | No (sequential) |
| T-013 | [Add integration tests for display output](./ticket-013-integration-tests.md) | 2 | T-012 | Yes |

**Total Points**: 18

## Dependencies

- **From Sprint 1**: All core types, AutomationRenderer, terminal detection
- **To Epic 2**: Foundation complete; renderers can be implemented

## Parallel Work Opportunities

- T-008 and T-010 can proceed in parallel
- T-009 follows T-008
- T-011 follows T-010
- T-012 requires T-010 and T-011

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| SDDP loop changes disrupt integration | Medium | Minimize changes to loop structure; add hooks |
| First-stage costs not easily accessible | Low | Already computed in `eval_first_stage_bound`; surface them |
| CLI precedence logic complex | Low | Follow existing pattern for log_level/log_format |

## Definition of Done

- [ ] All 6 tickets complete and merged
- [ ] `cargo run -- examples/04-cascade --profile automation` works
- [ ] `--no-color` disables ANSI codes
- [ ] `--quiet` produces minimal output
- [ ] Config file `display.profile` setting respected
- [ ] First-stage bound appears in output
- [ ] Integration tests pass in CI
