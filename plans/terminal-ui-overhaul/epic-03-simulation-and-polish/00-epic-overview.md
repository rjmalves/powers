# Epic 3: Simulation Display and Polish

> **Master Plan**: [Terminal UI Overhaul](../00-master-plan.md)
> **Duration**: 1 sprint (~2 weeks)
> **Status**: Not Started

## Summary

Complete the terminal UI overhaul with enhanced simulation display, error/warning visualization, and final documentation. This epic delivers the polished, production-ready display system.

## Scope

### Included

- Enhanced simulation summary with rich statistics
- Error message visualization (highlighted boxes, context)
- Warning message styling (for numerical issues, solver problems)
- Display system documentation
- Example configuration updates
- CHANGELOG entry
- Performance validation (logging overhead)

### Excluded

- Smart print throttling (prepared but not implemented)
- Async logging (deferred to future optimization)

## Dependencies

- **Requires**: Epic 2 (Training Display) - all renderers complete
- **Enables**: Feature complete for release

## Acceptance Criteria

- [ ] Simulation summary shows full cost distribution statistics
- [ ] Simulation summary includes percentiles (p5, p50, p95) in advanced mode
- [ ] Error messages render with red background/border
- [ ] Warning messages render with yellow styling
- [ ] All renderers handle simulation correctly
- [ ] Documentation covers all configuration options
- [ ] Examples updated with display configuration
- [ ] CHANGELOG.md updated
- [ ] Performance: <1ms overhead per iteration

## Technical Approach

### Enhanced Simulation Summary (Advanced)

```
╭────────────────────────────────────────────────────────────────────╮
│ Simulation Complete: 32 scenarios in 00:00:00.084                  │
├────────────────────────────────────────────────────────────────────┤
│ Expected Cost:    1.2745e+05 ± 2.61e+03                           │
│ Range:            [1.2134e+05 .. 1.3412e+05]                      │
│ Percentiles:      p5=1.22e+05  p50=1.27e+05  p95=1.32e+05         │
│ CV:               2.05%                                            │
╰────────────────────────────────────────────────────────────────────╯
```

### Error Visualization

```
┌──────────────────────────────────────────────────────────────────┐
│ ❌ ERROR                                                          │
├──────────────────────────────────────────────────────────────────┤
│ Solver returned infeasible for stage 5, scenario 12              │
│                                                                  │
│ Context:                                                         │
│   Storage: [45.2, 78.1, 23.4]                                   │
│   Inflow:  [12.3, 8.7, 5.2]                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Warning Styling

```
⚠️  Warning: Solver required 847 iterations (threshold: 500)
    Stage 3, iteration 15 - consider reviewing constraints
```

## Sprint 1 Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-026 | Implement enhanced simulation summary | 3 | Epic 2 |
| T-027 | Add percentile calculation to CostStatistics | 2 | T-026 |
| T-028 | Implement error message rendering | 2 | Epic 2 |
| T-029 | Implement warning message rendering | 2 | T-028 |
| T-030 | Update documentation for display system | 3 | All above |
| T-031 | Update examples with display configuration | 2 | T-030 |
| T-032 | Performance validation and optimization | 2 | All above |

**Total Points**: 16

## Risks

| Risk | Mitigation |
|------|------------|
| Percentile calculation needs sorting | Use efficient algorithm; cache if needed |
| Error context may be large | Truncate with "..." if exceeds width |

## Definition of Done

- [ ] All tickets complete and merged
- [ ] Simulation display rich and informative
- [ ] Errors and warnings visually distinguished
- [ ] Documentation complete
- [ ] Examples updated
- [ ] Performance validated
- [ ] Feature ready for release
