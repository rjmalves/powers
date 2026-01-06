# Sprint 2: Advanced and Standard Renderers

> **Epic**: [Training Display](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: Not Started

## Goals

- **Primary**: Implement AdvancedRenderer with full metrics and styling
- **Secondary**: Implement StandardRenderer with simplified layout
- **Tertiary**: Polish and visual consistency across renderers

## Tickets

| ID | Title | Points | Dependencies | Assignable |
|----|-------|--------|--------------|------------|
| T-020 | [Implement AdvancedRenderer header](./ticket-020-advanced-header.md) | 2 | Sprint 1 | Yes |
| T-021 | [Implement AdvancedRenderer iteration row](./ticket-021-advanced-iteration.md) | 5 | T-020 | Yes |
| T-022 | [Implement AdvancedRenderer training summary](./ticket-022-advanced-summary.md) | 3 | T-021 | Yes |
| T-023 | [Implement StandardRenderer](./ticket-023-standard-renderer.md) | 4 | Sprint 1 | Yes |
| T-024 | [Add target gap progress visualization](./ticket-024-target-gap-progress.md) | 2 | T-021 | Yes |
| T-025 | [Polish and visual consistency review](./ticket-025-visual-polish.md) | 2 | T-022, T-023 | Yes |

**Total Points**: 18

## Dependencies

- **From Sprint 1**: All components (colors, table, progress, indicators)
- **To Epic 3**: Full training display ready

## Parallel Work Opportunities

- T-020 and T-023 can proceed in parallel
- T-024 can proceed once T-021 structure is in place

## Sample Advanced Output (Target)

```
╭─────────────────────────────────────────────────────────────────────────────────╮
│ POWE.RS - Power Optimization for the World of Energy                           │
│ Training: 8 iterations × 4 forward passes | Cut selection: enabled             │
╰─────────────────────────────────────────────────────────────────────────────────╯

┌─────┬────────────────┬────────────────┬────────────────┬───────┬─────────────────┐
│ Iter│ Lower Bound ($)│ Simul Cost ($) │ 1st Stage ($)  │ Gap % │ Time (fwd/bwd)  │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   1 │   1.0148e+05   │   1.2823e+05   │   1.0148e+05   │ 26.4↓ │ 0.018s / 0.034s │
│     │                │ μ=1.28e5 σ=3.2e3 [1.24e5..1.35e5] n=4                    │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   2 │   1.2041e+05 ▲ │   1.2982e+05   │   1.2041e+05   │  7.8↓ │ 0.004s / 0.037s │
│     │ +18.6%         │ μ=1.30e5 σ=2.8e3 [1.26e5..1.34e5] n=4                    │
└─────┴────────────────┴────────────────┴────────────────┴───────┴─────────────────┘

Training Complete
─────────────────
  Total time:     00:00:00.511
  Final bound:    1.2413e+05
  Policy cost:    1.2720e+05 ± 3.00e+03
  Final gap:      2.47%
  Total cuts:     32
```

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Complex multi-line row formatting | Medium | Build incrementally; test each component |
| Terminal width constraints | Medium | Use percentage widths; truncate gracefully |

## Definition of Done

- [ ] All 6 tickets complete and merged
- [ ] AdvancedRenderer matches sample output
- [ ] StandardRenderer provides simplified view
- [ ] Target gap progress bar works (when configured)
- [ ] Visual consistency across profiles
- [ ] All tests passing
