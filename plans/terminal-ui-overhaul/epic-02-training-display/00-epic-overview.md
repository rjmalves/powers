# Epic 2: Training Display

> **Master Plan**: [Terminal UI Overhaul](../00-master-plan.md)
> **Duration**: 2 sprints (~4 weeks)
> **Status**: Not Started

## Summary

Implement the rich visual renderers for training output: `AdvancedRenderer`, `StandardRenderer`, and `MinimalRenderer`. This epic delivers the beautiful, informative terminal output that transforms the user experience.

## Scope

### Included

- `AdvancedRenderer`: Full metrics with colors, box-drawing tables, real-time statistics, trend indicators, cost distributions
- `StandardRenderer`: Key metrics with simplified layout, colors, box-drawing
- `MinimalRenderer`: Progress bar with ETA, final summary only
- Reusable display components: tables, progress bars, trend indicators, statistics formatters
- Color utilities using crossterm
- Header rendering for all profiles
- Training summary rendering with enhanced statistics

### Excluded (Deferred to Epic 3)

- Simulation display enhancements
- Error/warning visual treatment
- Documentation and examples

## Dependencies

- **Requires**: Epic 1 (Foundation) - core types, traits, terminal detection, automation renderer
- **Enables**: Epic 3 (Simulation & Polish)

## Acceptance Criteria

- [ ] `AdvancedRenderer` displays all metrics from master plan sample output
- [ ] Forward cost statistics shown per iteration (μ, σ, min, max)
- [ ] First-stage branching statistics shown
- [ ] Gap percentage with colored trend indicators (↓↑→)
- [ ] Box-drawing table borders render correctly
- [ ] Colors applied appropriately (green=good, red=bad, yellow=caution)
- [ ] `StandardRenderer` shows simplified version
- [ ] `MinimalRenderer` shows progress bar with percentage and ETA
- [ ] All renderers handle terminal width gracefully (truncation/wrapping)
- [ ] Unit tests for formatting functions
- [ ] Visual regression: output matches design mockups

## Technical Approach

### Component Design

```
src/display/
├── components/
│   ├── mod.rs
│   ├── table.rs       # Box-drawing table builder
│   ├── progress.rs    # Progress bar with ETA
│   ├── indicators.rs  # Trend arrows, status icons
│   ├── statistics.rs  # Cost/timing formatters
│   └── color.rs       # Crossterm color utilities
├── renderers/
│   ├── advanced.rs    # Full implementation
│   ├── standard.rs    # Simplified implementation
│   └── minimal.rs     # Progress bar implementation
```

### Table Component

```rust
pub struct TableBuilder {
    columns: Vec<Column>,
    rows: Vec<Row>,
    borders: BorderStyle,
}

pub enum BorderStyle {
    None,
    Ascii,      // +--+
    Rounded,    // ╭──╮
    Heavy,      // ┏━━┓
}
```

### Progress Bar

```rust
pub struct ProgressBar {
    current: usize,
    total: usize,
    width: u16,
    style: ProgressStyle,
}

impl ProgressBar {
    pub fn render(&self) -> String {
        // [████████░░░░░░░░░░░░] 40% | 2/5 iter | ETA: 00:01:23
    }
}
```

### Color Utilities

```rust
use crossterm::style::{Color, Stylize};

pub fn color_by_trend(trend: GapTrend, text: &str, enabled: bool) -> String {
    if !enabled {
        return text.to_string();
    }
    match trend {
        GapTrend::Improving => text.green().to_string(),
        GapTrend::Worsening => text.red().to_string(),
        GapTrend::Stable => text.yellow().to_string(),
        GapTrend::Unknown => text.to_string(),
    }
}
```

## Estimated Effort

- **Sprint 1**: Components (table, progress, indicators, colors) + MinimalRenderer
- **Sprint 2**: AdvancedRenderer + StandardRenderer + polish

**Total**: ~35-40 story points across 2 sprints

## Risks

| Risk | Mitigation |
|------|------------|
| Unicode box-drawing breaks on some terminals | Provide ASCII fallback, detect Unicode support |
| Table column widths hard to balance | Use percentage-based widths relative to terminal |
| Progress bar ETA inaccurate early | Show "calculating..." for first few iterations |

## Definition of Done

- [ ] All three renderers fully implemented
- [ ] All components tested
- [ ] Visual output matches design in master plan
- [ ] Works on Linux, macOS (Windows nice-to-have)
- [ ] Code reviewed and merged
