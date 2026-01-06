# T-020: Implement AdvancedRenderer header

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: Sprint 1 complete (all components)
> **Blocks**: T-021, T-022

## Files to Read Before Starting

- `src/display/renderer.rs` - DisplayRenderer trait
- `src/display/config.rs` - DisplayConfig struct
- `src/display/components/table.rs` - Table builder (for border style)
- `src/display/components/color.rs` - Color utilities
- Master plan sample output - Target header appearance

## Context

### Background

The AdvancedRenderer header provides a welcoming branded introduction to training output. It shows the POWE.RS name, training configuration summary, and sets the visual tone for the detailed metrics that follow.

### Current State

Sprint 1 completed all display components. Now we build the AdvancedRenderer starting with the header section.

## Specification

### Target Header Output

```
╭─────────────────────────────────────────────────────────────────────────────────╮
│ POWE.RS - Power Optimization for the World of Energy                           │
│ Training: 8 iterations × 4 forward passes | Cut selection: enabled             │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

### Create/Complete `src/display/renderers/advanced.rs`

#### AdvancedRenderer Struct

```rust
/// Advanced display renderer with full metrics, colors, and visual indicators.
///
/// This is the default and most informative display profile, showing:
/// - Branded header with configuration summary
/// - Per-iteration metrics with colors and trend indicators
/// - Forward cost statistics (μ, σ, min, max)
/// - First-stage bound and branching statistics
/// - Gap percentage with trend arrows
/// - Detailed timing breakdown
/// - Training and simulation summaries
pub struct AdvancedRenderer {
    /// Color configuration
    color_config: ColorConfig,
    /// Terminal width for sizing
    terminal_width: u16,
    /// Border style for tables
    border_style: BorderStyle,
}
```

#### Header Method

```rust
impl AdvancedRenderer {
    /// Create a new AdvancedRenderer
    pub fn new(color_config: ColorConfig, terminal_width: u16) -> Self {
        Self {
            color_config,
            terminal_width,
            border_style: BorderStyle::Rounded,
        }
    }
    
    /// Render the branded header box
    fn render_header_box(&self, config: &DisplayConfig) -> String {
        // Build header content
        // Use rounded border style for header box
    }
}

impl DisplayRenderer for AdvancedRenderer {
    fn render_header(&self, config: &DisplayConfig) -> String {
        self.render_header_box(config)
    }
    
    // ... other methods (stub for now, implemented in T-021, T-022)
}
```

### Header Content

Line 1: Product name and tagline
```
POWE.RS - Power Optimization for the World of Energy
```

Line 2: Training configuration summary
```
Training: {iterations} iterations × {forward_passes} forward passes | Cut selection: {enabled/disabled}
```

### Box Dimensions

- Width: Adapts to terminal width (minimum 60, maximum terminal_width - 2)
- Uses rounded border style (`╭`, `╮`, `╰`, `╯`)
- Content is left-aligned with 1 space padding

## Acceptance Criteria

- [ ] `AdvancedRenderer` struct created with necessary fields
- [ ] `render_header` returns properly formatted header box
- [ ] Box adapts to terminal width
- [ ] Product name is emphasized (bold or color)
- [ ] Configuration summary is accurate
- [ ] Uses rounded box-drawing characters
- [ ] Falls back to ASCII if Unicode not supported
- [ ] Unit tests verify output format

## Implementation Guide

### Step 1: Create advanced.rs skeleton

```rust
//! Advanced display renderer with full metrics and visual styling.

use crate::display::{
    config::DisplayConfig,
    context::DisplayContext,
    renderer::DisplayRenderer,
    components::{
        color::{ColorConfig, SemanticColor, colorize, bold},
        table::BorderStyle,
    },
};

pub struct AdvancedRenderer {
    color_config: ColorConfig,
    terminal_width: u16,
    border_style: BorderStyle,
}

impl AdvancedRenderer {
    pub fn new(color_config: ColorConfig, terminal_width: u16) -> Self {
        Self {
            color_config,
            terminal_width,
            border_style: BorderStyle::Rounded,
        }
    }
}
```

### Step 2: Implement render_header_box

```rust
fn render_header_box(&self, config: &DisplayConfig) -> String {
    let width = self.terminal_width.min(85).max(60) as usize;
    let inner_width = width - 4; // Account for "│ " and " │"
    
    // Build content lines
    let line1 = "POWE.RS - Power Optimization for the World of Energy";
    let line1_styled = bold(line1, &self.color_config);
    
    let line2 = format!(
        "Training: {} iterations × {} forward passes | Cut selection: {}",
        config.max_iterations,
        config.forward_passes,
        if config.cut_selection { "enabled" } else { "disabled" }
    );
    
    // Pad lines to inner_width
    let line1_padded = format!("{:<width$}", line1_styled, width = inner_width);
    let line2_padded = format!("{:<width$}", line2, width = inner_width);
    
    // Build box
    let top = format!("╭{}╮", "─".repeat(width - 2));
    let row1 = format!("│ {} │", line1_padded);
    let row2 = format!("│ {} │", line2_padded);
    let bottom = format!("╰{}╯", "─".repeat(width - 2));
    
    format!("{}\n{}\n{}\n{}\n", top, row1, row2, bottom)
}
```

### Step 3: Implement stub methods

```rust
impl DisplayRenderer for AdvancedRenderer {
    fn render_header(&self, config: &DisplayConfig) -> String {
        self.render_header_box(config)
    }
    
    fn render_iteration(&self, ctx: &DisplayContext) -> String {
        // TODO: Implement in T-021
        todo!("Implement in T-021")
    }
    
    fn render_training_summary(&self, result: &TrainingResult) -> String {
        // TODO: Implement in T-022
        todo!("Implement in T-022")
    }
    
    fn render_simulation_summary(&self, trajectories: &[SimulationTrajectory]) -> String {
        // TODO: Implement in T-022
        todo!("Implement in T-022")
    }
    
    fn supports_color(&self) -> bool {
        self.color_config.enabled
    }
}
```

### Step 4: Update renderers/mod.rs

```rust
pub mod advanced;
pub use advanced::AdvancedRenderer;
```

### Patterns to Follow

- Use the color utilities for styling
- Keep box drawing logic inline (simpler than using TableBuilder for a 2-line box)
- Handle width calculation carefully

### Pitfalls to Avoid

- ⚠️ ANSI escape codes affect string length - use plain text for width calculation
- ⚠️ Don't exceed terminal width
- ⚠️ Ensure box characters align properly

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_render_header_contains_brand() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig {
        max_iterations: 8,
        forward_passes: 4,
        cut_selection: true,
        ..Default::default()
    };
    
    let header = renderer.render_header(&config);
    assert!(header.contains("POWE.RS"));
    assert!(header.contains("Power Optimization"));
}

#[test]
fn test_render_header_shows_config() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig {
        max_iterations: 8,
        forward_passes: 4,
        cut_selection: true,
        ..Default::default()
    };
    
    let header = renderer.render_header(&config);
    assert!(header.contains("8 iterations"));
    assert!(header.contains("4 forward passes"));
    assert!(header.contains("Cut selection: enabled"));
}

#[test]
fn test_render_header_cut_selection_disabled() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig {
        cut_selection: false,
        ..Default::default()
    };
    
    let header = renderer.render_header(&config);
    assert!(header.contains("Cut selection: disabled"));
}

#[test]
fn test_render_header_has_box_borders() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig::default();
    
    let header = renderer.render_header(&config);
    assert!(header.contains("╭"));
    assert!(header.contains("╮"));
    assert!(header.contains("╰"));
    assert!(header.contains("╯"));
}

#[test]
fn test_supports_color() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(true), 80);
    assert!(renderer.supports_color());
}
```

## Documentation Requirements

- [ ] Doc comments on AdvancedRenderer struct
- [ ] Doc comment showing expected header output
- [ ] Module-level docs explaining the advanced profile

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple box rendering with known content. Main complexity is width handling.

## Definition of Done

- [ ] Implementation complete
- [ ] Header renders correctly
- [ ] Box adapts to terminal width
- [ ] Colors applied when enabled
- [ ] Tests passing
- [ ] Documentation complete
- [ ] PR reviewed and merged
