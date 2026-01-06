# Sprint 1: Core Types and Terminal Backend

> **Epic**: [Foundation](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: Not Started

## Goals

- **Primary**: Establish core display type system (`DisplayProfile`, `DisplayContext`, `CostStatistics`)
- **Secondary**: Implement terminal capability detection with `crossterm`
- **Tertiary**: Create working `AutomationRenderer` (JSON output)

## Tickets

| ID | Title | Points | Dependencies | Assignable |
|----|-------|--------|--------------|------------|
| T-001 | [Add crossterm dependency and display module structure](./ticket-001-add-crossterm-dependency.md) | 2 | None | Yes |
| T-002 | [Define DisplayProfile and DisplayConfig types](./ticket-002-define-display-profile.md) | 3 | T-001 | Yes |
| T-003 | [Implement CostStatistics aggregation](./ticket-003-implement-cost-statistics.md) | 2 | T-001 | Yes |
| T-004 | [Define DisplayContext with all metrics fields](./ticket-004-define-display-context.md) | 3 | T-002, T-003 | Yes |
| T-005 | [Implement terminal capability detection](./ticket-005-terminal-detection.md) | 3 | T-001 | Yes |
| T-006 | [Define DisplayRenderer trait](./ticket-006-define-display-renderer.md) | 2 | T-004 | Yes |
| T-007 | [Implement AutomationRenderer (JSON output)](./ticket-007-automation-renderer.md) | 3 | T-006 | Yes |

**Total Points**: 18

## Dependencies

- **From Previous Sprint**: N/A (first sprint)
- **To Next Sprint**: All core types and automation renderer ready for integration

## Parallel Work Opportunities

The following tickets can be worked on in parallel:
- T-002 and T-003 (after T-001)
- T-005 can proceed independently after T-001

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| `crossterm` API unfamiliar | Medium | Allocate time for learning; start with simple detection |
| `CostStatistics` edge cases (empty, single value) | Low | Comprehensive test cases |

## Definition of Done

- [ ] All 7 tickets complete and merged
- [ ] `cargo build` succeeds with new display module
- [ ] `cargo test` passes including new unit tests
- [ ] `AutomationRenderer` produces valid JSON for sample data
- [ ] Terminal detection correctly identifies interactive vs piped
- [ ] Code coverage for new modules > 80%
