# Master Plan: Terminal UI Overhaul

## Executive Summary

Transform POWE.RS terminal output from basic streaming logs into a rich, informative display system inspired by modern CLI tools like `htop`, Python's `rich` library, and Claude Code. The new system will provide real-time convergence visualization, detailed timing breakdowns, cost distribution statistics, and multiple display profiles while maintaining backward compatibility through an automation-friendly JSON output mode.

## Goals & Non-Goals

### Goals

1. **Rich Training Display**: Show per-iteration metrics with colors, progress indicators, convergence trends, and real-time forward cost statistics
2. **First-Stage Quality Metrics**: Display risk-adjusted expected cost and branching scenario statistics from backward pass
3. **Profile System**: Support `advanced` (default), `standard`, `minimal`, and `automation` display profiles
4. **Hybrid Mode**: Streaming output that works with pipes while supporting rich formatting when interactive
5. **Convergence Visualization**: Visual indicators for gap improvement trends and optional target gap progress
6. **Enhanced Simulation Summary**: Enriched post-simulation statistics display
7. **Configuration Flexibility**: Both CLI flags and `config.json` settings for display preferences
8. **Terminal Detection**: Automatic fallback to plain output in non-interactive terminals

### Non-Goals (Explicit Scope Exclusions)

- **Full TUI Mode**: No interactive dashboard with cursor control (future consideration)
- **Theming System**: No dark/light themes or custom color schemes
- **Mouse Support**: Terminal interaction is display-only
- **Async Logging Infrastructure**: No buffered/async writes (defer to future optimization)
- **Smart Print Throttling**: Print every iteration for now (infrastructure prepared for future)

## Architecture Overview

### Current State

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
│ SDDP Train  │────▶│ LogContext   │────▶│ TerminalFormatter  │────▶ stdout
│ Loop        │     │ (thread-local)│     │ (simple table)     │
└─────────────┘     └──────────────┘     └────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ JsonFormatter│────▶ file/stdout
                    └──────────────┘
```

**Limitations**:
- `LogContext` has limited fields (iteration, bounds, basic timing)
- `TerminalFormatter` produces fixed table format
- No profile selection mechanism
- No rich formatting (colors are basic, no box drawing)
- No real-time statistics aggregation

### Target State

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ SDDP Train  │────▶│ DisplayContext   │────▶│ DisplayRenderer │
│ Loop        │     │ (rich metrics)   │     │ (trait)         │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                    ┌──────────────────────────────────┼────────────────┐
                    │                                  │                │
                    ▼                                  ▼                ▼
          ┌─────────────────┐              ┌────────────────┐  ┌──────────────┐
          │ AdvancedRenderer│              │ MinimalRenderer│  │ JsonRenderer │
          │ (rich, colors,  │              │ (progress bar) │  │ (structured) │
          │  box drawing)   │              └────────────────┘  └──────────────┘
          └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ TerminalBackend │ (crossterm for colors, detection)
          └─────────────────┘
```

### Key Design Decisions

1. **Use `crossterm` for terminal control**: Cross-platform, well-maintained, used by `ratatui`. Provides color support, terminal detection, and styled output without full TUI overhead.

2. **Trait-based renderer abstraction**: `DisplayRenderer` trait allows profile-specific implementations while sharing common infrastructure.

3. **Rich `DisplayContext` replacing `LogContext`**: New context struct carries all metrics needed for any profile, computed once per iteration.

4. **Preserve `log` crate integration**: Continue using `log::info!` macros but enhance the formatter to recognize structured display contexts.

5. **Conditional compilation not needed**: All profiles compiled in; runtime selection via configuration.

6. **Print decision parameter**: Each iteration receives `should_print: bool` (always `true` for now) to prepare for future smart throttling.

## Technical Approach

### Core Abstractions

#### `DisplayProfile` (enum)
```rust
pub enum DisplayProfile {
    Advanced,   // Full metrics, real-time stats, visual indicators
    Standard,   // Key metrics, simplified timing  
    Minimal,    // Progress bar + final summary
    Automation, // JSON lines, no ANSI
}
```

#### `DisplayContext` (struct)
```rust
pub struct DisplayContext {
    // Iteration info
    pub iteration: usize,
    pub total_iterations: usize,
    pub should_print: bool,
    
    // Convergence metrics
    pub lower_bound: f64,
    pub previous_lower_bound: Option<f64>,
    pub target_gap: Option<f64>,
    
    // Forward pass metrics
    pub forward_costs: Vec<f64>,
    pub forward_cost_stats: CostStatistics,
    pub forward_timing: ForwardTimingOutput,
    
    // Backward pass metrics  
    pub backward_timing: BackwardTimingOutput,
    pub first_stage_bound: f64,
    pub first_stage_branching_costs: Vec<f64>,
    pub first_stage_stats: CostStatistics,
    
    // Cut management
    pub cuts_added: usize,
    pub cuts_removed: usize,
    pub cuts_active: usize,
    
    // Iteration timing
    pub iteration_time: Duration,
    pub elapsed_total: Duration,
}
```

#### `CostStatistics` (struct)
```rust
pub struct CostStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}
```

#### `DisplayRenderer` (trait)
```rust
pub trait DisplayRenderer: Send + Sync {
    fn render_header(&self, config: &DisplayConfig) -> String;
    fn render_iteration(&self, ctx: &DisplayContext) -> String;
    fn render_training_summary(&self, result: &TrainingResult) -> String;
    fn render_simulation_summary(&self, trajectories: &[SimulationTrajectory]) -> String;
    fn supports_color(&self) -> bool;
}
```

### Data Flow

```
1. Iteration completes
        │
        ▼
2. Compute DisplayContext (aggregate stats, compute trends)
        │
        ▼
3. Check should_print flag
        │
        ├── false ──▶ Skip rendering
        │
        ▼ true
4. Get active DisplayRenderer based on profile
        │
        ▼
5. renderer.render_iteration(ctx) ──▶ styled string
        │
        ▼
6. Write to stdout (via logging or direct)
```

### Display Elements by Profile

| Element | Advanced | Standard | Minimal | Automation |
|---------|----------|----------|---------|------------|
| Iteration number | ✓ | ✓ | Progress % | ✓ (JSON) |
| Lower bound | ✓ colored | ✓ | — | ✓ |
| Simulation cost (mean) | ✓ | ✓ | — | ✓ |
| Forward cost stats (μ/σ/min/max) | ✓ | — | — | ✓ |
| First-stage bound | ✓ | ✓ | — | ✓ |
| First-stage branching stats | ✓ | — | — | ✓ |
| Gap % | ✓ colored | ✓ | — | ✓ |
| Gap trend indicator (↓↑→) | ✓ | — | — | — |
| Target gap progress bar | ✓ (if set) | — | — | — |
| Forward timing | ✓ | ✓ | — | ✓ |
| Backward timing | ✓ | ✓ | — | ✓ |
| Detailed timing breakdown | ✓ | — | — | ✓ |
| Cuts info (+/-/active) | ✓ | — | — | ✓ |
| Solver calls | ✓ | — | — | ✓ |
| Box-drawing borders | ✓ | ✓ | — | — |
| Colors | ✓ | ✓ | ✓ | — |
| Progress bar | — | — | ✓ | — |

### Sample Output (Advanced Profile)

```
╭─────────────────────────────────────────────────────────────────────────────────╮
│ POWE.RS - Power Optimization for the World of Energy                           │
│ Training: 8 iterations × 4 forward passes | Cut selection: enabled             │
╰─────────────────────────────────────────────────────────────────────────────────╯

┌─────┬────────────────┬────────────────┬────────────────┬───────┬─────────────────┐
│ Iter│ Lower Bound ($)│ Simul Cost ($) │ 1st Stage ($)  │ Gap % │ Time (fwd/bwd)  │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   1 │   1.0148e+05   │   1.2823e+05   │   1.0148e+05   │ 26.4↓ │ 0.018s / 0.034s │
│     │                │ μ=1.28e5 σ=3.2e3 [1.24e5..1.35e5]                        │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   2 │   1.2041e+05 ▲ │   1.2982e+05   │   1.2041e+05   │  7.8↓ │ 0.004s / 0.037s │
│     │ +18.6%         │ μ=1.30e5 σ=2.8e3 [1.26e5..1.34e5]                        │
└─────┴────────────────┴────────────────┴────────────────┴───────┴─────────────────┘

Training Summary
────────────────
  Total time:     00:00:00.511
  Final bound:    1.2413e+05
  Policy cost:    1.2720e+05 ± 3.00e+03
  Final gap:      2.47%
  Total cuts:     32
```

## Phases & Milestones

| Phase | Epic | Description | Duration | Milestone |
|-------|------|-------------|----------|-----------|
| 1 | Foundation | Core abstractions, profile system, terminal backend | 2 sprints | Profile selection works, basic styled output |
| 2 | Training Display | Rich iteration rendering, all metrics, convergence viz | 2 sprints | Full advanced profile working for training |
| 3 | Simulation & Polish | Simulation display, error handling, documentation | 1 sprint | Feature complete, documented |

**Total estimated duration**: 5 sprints (~10 weeks)

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `crossterm` API changes | Low | Medium | Pin version, abstract behind internal trait |
| Performance overhead from formatting | Medium | Low | Profile shows <1ms/iteration acceptable; defer async if needed |
| Breaking existing log parsers | Medium | Medium | `automation` profile provides stable JSON format |
| Complex terminal detection edge cases | Medium | Low | Conservative fallback to plain text |
| First-stage metrics not exposed cleanly | Low | High | Already computed in `eval_first_stage_bound`; need to surface |

## Success Metrics

- [ ] All 4 profiles implemented and selectable via CLI/config
- [ ] Advanced profile shows all specified metrics with colors and indicators
- [ ] Automation profile produces valid JSON lines parseable by external tools
- [ ] Non-interactive terminal detection works (CI/CD, pipes)
- [ ] No performance regression in iteration loop (benchmark shows <1ms overhead)
- [ ] Existing examples produce visually improved output
- [ ] Documentation covers all configuration options

## Dependencies

### New Crate Dependencies

```toml
[dependencies]
crossterm = "0.27"  # Terminal styling, colors, detection
```

### Internal Dependencies

- Timing infrastructure (`src/timing/`) - already provides detailed metrics
- SDDP module (`src/sddp/mod.rs`) - source of iteration data
- Logging module (`src/logging/`) - will be extended, not replaced

## File Structure Changes

```
src/
├── display/                    # NEW: Display system
│   ├── mod.rs                 # Module exports
│   ├── config.rs              # DisplayConfig, DisplayProfile
│   ├── context.rs             # DisplayContext, CostStatistics
│   ├── renderer.rs            # DisplayRenderer trait
│   ├── renderers/
│   │   ├── mod.rs
│   │   ├── advanced.rs        # AdvancedRenderer
│   │   ├── standard.rs        # StandardRenderer
│   │   ├── minimal.rs         # MinimalRenderer
│   │   └── automation.rs      # AutomationRenderer (JSON)
│   ├── components/            # Reusable display components
│   │   ├── mod.rs
│   │   ├── table.rs           # Table with box drawing
│   │   ├── progress.rs        # Progress bar
│   │   ├── indicators.rs      # Trend arrows, status icons
│   │   └── statistics.rs      # Cost stats formatting
│   └── terminal.rs            # Terminal detection, color support
├── logging/
│   ├── ...                    # Existing (preserved)
│   └── display_integration.rs # NEW: Bridge to display system
└── cli.rs                     # Extended with --profile, --no-color, --quiet
```

## Configuration Schema Changes

### `config.json` additions

```json
{
  "logging": {
    "level": "info",
    "format": "terminal",
    "outputs": [{"type": "terminal"}]
  },
  "display": {
    "profile": "advanced",
    "color": "auto",
    "target_gap": null
  }
}
```

### CLI additions

```
--profile <PROFILE>    Display profile: advanced, standard, minimal, automation
--no-color             Disable colored output
--quiet                Equivalent to --profile minimal
```

## Migration Path

1. **Phase 1**: New display system added alongside existing logging
2. **Phase 2**: Default output switches to new system; `--legacy` flag available
3. **Phase 3**: Legacy flag removed after validation period

For this implementation, we proceed directly to Phase 2 behavior (new system as default) since breaking changes are acceptable.

---

## Addendum: Epic 2.6 - Robust Table Alignment (Added 2026-01-07)

### Context

After Epic 2.5 was completed, persistent table alignment issues were identified. An attempted migration to the `tabled` crate was abandoned due to fundamental incompatibility with progressive rendering requirements.

### Root Cause Analysis

The core issue is Rust's `format!` macro behavior: when content exceeds the specified field width, the field **expands** instead of truncating. This causes rows with larger content to be wider than expected, breaking alignment with other rows.

### Solution: Robust Manual Table with Safety Guarantees

Instead of using an external library, we fix the manual approach with:

1. **Generous Width Budget**: Add 25-35% slack to each column
2. **Explicit Truncation**: Never allow format expansion - truncate content first
3. **Centralized Configuration**: Single source of truth for column widths
4. **ANSI-Aware Measurement**: Strip ANSI codes before calculating display width
5. **Comprehensive Tests**: Validate alignment invariants to prevent regression

### Key Architectural Decision

The `tabled` crate was evaluated but rejected because:
- `tabled` builds complete tables, not individual rows
- Current architecture requires progressive rendering (header once, then rows incrementally)
- Changing to full-table-rebuild would alter output behavior and user experience

### Impact on Phases

| Phase | Epic | Description | Duration | Milestone |
|-------|------|-------------|----------|-----------|
| 1 | Foundation | Core abstractions, profile system, terminal backend | 2 sprints | ✅ COMPLETE |
| 2 | Training Display | Rich iteration rendering, all metrics, convergence viz | 2 sprints | ✅ COMPLETE |
| 2.5 | UI Fixes | Fix duplicate headers, silence legacy logging | 1 sprint | ✅ COMPLETE |
| 2.6 | **Robust Table Alignment** | **Fix alignment with truncation safety** | **1 sprint** | **🔧 NEW** |
| 3 | Simulation & Polish | Simulation display, error handling, documentation | 1 sprint | Pending |

### Dependencies Removed

The `tabled` crate dependency is removed:

```toml
# REMOVED from Cargo.toml:
# tabled = { version = "0.20", default-features = false, features = ["std", "ansi"] }
```

### New Files

```
src/display/components/
└── table_format.rs    # NEW: Robust cell formatting with truncation safety
```

### Tickets

| ID | Title | Points |
|----|-------|--------|
| T-031 | Remove tabled dependency and revert related code | 2 |
| T-032 | Create robust table formatting utilities | 5 |
| T-033 | Migrate AdvancedRenderer to new utilities | 3 |
| T-034 | Migrate StandardRenderer to new utilities | 2 |
| T-035 | Add comprehensive alignment tests | 2 |

**Total**: 14 points (1 sprint)
