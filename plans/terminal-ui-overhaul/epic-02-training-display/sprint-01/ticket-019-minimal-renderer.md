# T-019: Implement MinimalRenderer

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-017 (progress bar component)
> **Blocks**: None (can proceed independently)

## Files to Read Before Starting

- `src/display/renderer.rs` - DisplayRenderer trait
- `src/display/context.rs` - DisplayContext struct
- `src/display/components/progress.rs` - Progress bar component
- `src/display/components/color.rs` - Color utilities

## Context

### Background

The MinimalRenderer provides the simplest output profile: a progress bar during training with ETA, followed by a summary when complete. This is ideal for users who want minimal distraction while training runs, or for environments with limited terminal capabilities.

### Current State

The DisplayRenderer trait exists from Epic 1. The progress bar component (T-017) provides the core visualization. Now we need to implement the full MinimalRenderer.

## Specification

### Create/Complete `src/display/renderers/minimal.rs`

#### MinimalRenderer Struct

```rust
/// Minimal display renderer showing only progress bar and final summary.
///
/// Output format during training:
/// ```text
/// Training: [████████░░░░░░░░░░░░] 40% | 4/10 iter | ETA: 00:01:23
/// ```
///
/// Output format for summary:
/// ```text
/// Training Complete
/// ─────────────────
///   Final bound: 1.2413e+05
///   Final gap:   2.47%
///   Total time:  00:00:00.511
/// ```
pub struct MinimalRenderer {
    /// Color configuration
    color_config: ColorConfig,
    /// Progress bar configuration
    progress_config: ProgressBarConfig,
    /// Terminal width for sizing
    terminal_width: u16,
}
```

#### Implementation

```rust
impl MinimalRenderer {
    /// Create a new MinimalRenderer
    pub fn new(color_config: ColorConfig, terminal_width: u16) -> Self;
}

impl DisplayRenderer for MinimalRenderer {
    fn render_header(&self, config: &DisplayConfig) -> String {
        // Minimal: No header during training (progress bar is enough)
        String::new()
    }
    
    fn render_iteration(&self, ctx: &DisplayContext) -> String {
        // Show progress bar with percentage, count, and ETA
    }
    
    fn render_training_summary(&self, result: &TrainingResult) -> String {
        // Show "Training Complete" with key metrics
    }
    
    fn render_simulation_summary(&self, trajectories: &[SimulationTrajectory]) -> String {
        // Minimal simulation summary
    }
    
    fn supports_color(&self) -> bool {
        self.color_config.enabled
    }
}
```

### Iteration Output Format

Single-line progress that overwrites itself:

```
Training: [████████░░░░░░░░░░░░] 40% | 4/10 iter | ETA: 00:01:23
```

**Note**: Use `\r` (carriage return) to return to start of line, not newline. This creates a "live updating" effect.

### Training Summary Format

```
Training Complete ✓
─────────────────
  Final bound: 1.2413e+05
  Final gap:   2.47%
  Total time:  00:00:00.511
```

### Simulation Summary Format

```
Simulation Complete ✓
─────────────────────
  Trajectories: 100
  Mean cost:    1.2720e+05 ± 3.00e+03
```

## Acceptance Criteria

- [ ] `MinimalRenderer` implements `DisplayRenderer` trait
- [ ] Progress bar renders correctly during iteration
- [ ] Progress bar uses carriage return for overwrite effect
- [ ] Training summary shows bound, gap, and time
- [ ] Simulation summary shows trajectory count and mean cost
- [ ] Colors applied when enabled
- [ ] Graceful output when colors disabled
- [ ] Unit tests for all render methods

## Implementation Guide

### Step 1: Create minimal.rs

```rust
//! Minimal display renderer with progress bar only.

use crate::display::{
    config::DisplayConfig,
    context::DisplayContext,
    renderer::DisplayRenderer,
    components::{
        color::ColorConfig,
        progress::{ProgressBar, ProgressBarConfig},
        statistics::format_duration_compact,
    },
};

pub struct MinimalRenderer {
    color_config: ColorConfig,
    progress_config: ProgressBarConfig,
    terminal_width: u16,
}

impl MinimalRenderer {
    pub fn new(color_config: ColorConfig, terminal_width: u16) -> Self {
        Self {
            color_config,
            progress_config: ProgressBarConfig {
                width: (terminal_width / 2).min(40),
                show_percentage: true,
                show_eta: true,
                show_count: true,
                ..Default::default()
            },
            terminal_width,
        }
    }
}
```

### Step 2: Implement render_iteration

```rust
fn render_iteration(&self, ctx: &DisplayContext) -> String {
    if !ctx.should_print {
        return String::new();
    }
    
    // Create progress bar for this iteration
    let mut bar = ProgressBar::new(ctx.total_iterations, self.progress_config.clone());
    bar.set(ctx.iteration);
    
    // Note: In actual usage, we'd track start time across iterations.
    // For now, we can use elapsed_total to approximate ETA.
    
    let progress = bar.render(&self.color_config);
    
    // Use carriage return to overwrite previous line
    format!("\rTraining: {}", progress)
}
```

### Step 3: Implement render_training_summary

```rust
fn render_training_summary(&self, result: &TrainingResult) -> String {
    let checkmark = if self.color_config.enabled {
        colorize("✓", SemanticColor::Good, &self.color_config)
    } else {
        "✓".to_string()
    };
    
    let title = format!("\nTraining Complete {}", checkmark);
    let separator = "─".repeat(17);
    
    let bound = format_cost(result.lower_bound, true);
    let gap = format!("{:.2}%", result.gap_percentage);
    let time = format_duration_compact(result.total_time);
    
    format!(
        "{}\n{}\n  Final bound: {}\n  Final gap:   {}\n  Total time:  {}",
        title, separator, bound, gap, time
    )
}
```

### Step 4: Implement render_simulation_summary

Similar pattern to training summary but for simulation results.

### Step 5: Update renderers/mod.rs

```rust
pub mod minimal;
pub use minimal::MinimalRenderer;
```

### Patterns to Follow

- Use components for formatting (don't duplicate)
- Keep output concise (minimal profile philosophy)
- Use `\r` for live-updating progress

### Pitfalls to Avoid

- ⚠️ Don't forget the leading `\r` for overwrite effect
- ⚠️ Final summary should include `\n` to move past progress bar
- ⚠️ Handle case where terminal_width is very narrow

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_minimal_render_iteration() {
    let renderer = MinimalRenderer::new(ColorConfig::new(false), 80);
    let ctx = DisplayContext {
        iteration: 3,
        total_iterations: 10,
        should_print: true,
        // ... other fields with defaults
    };
    
    let output = renderer.render_iteration(&ctx);
    assert!(output.starts_with("\rTraining:"));
    assert!(output.contains("30%"));  // 3/10 = 30%
    assert!(output.contains("3/10 iter"));
}

#[test]
fn test_minimal_render_iteration_skip() {
    let renderer = MinimalRenderer::new(ColorConfig::new(false), 80);
    let ctx = DisplayContext {
        should_print: false,
        // ... other fields
    };
    
    assert!(renderer.render_iteration(&ctx).is_empty());
}

#[test]
fn test_minimal_render_training_summary() {
    let renderer = MinimalRenderer::new(ColorConfig::new(false), 80);
    let result = TrainingResult {
        lower_bound: 124130.0,
        gap_percentage: 2.47,
        total_time: Duration::from_millis(511),
        // ... other fields
    };
    
    let output = renderer.render_training_summary(&result);
    assert!(output.contains("Training Complete"));
    assert!(output.contains("Final bound:"));
    assert!(output.contains("Final gap:"));
    assert!(output.contains("Total time:"));
}

#[test]
fn test_minimal_render_header_empty() {
    let renderer = MinimalRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig::default();
    assert!(renderer.render_header(&config).is_empty());
}

#[test]
fn test_supports_color() {
    let renderer_color = MinimalRenderer::new(ColorConfig::new(true), 80);
    assert!(renderer_color.supports_color());
    
    let renderer_no_color = MinimalRenderer::new(ColorConfig::new(false), 80);
    assert!(!renderer_no_color.supports_color());
}
```

### Integration Tests

- [ ] Verify output looks correct in actual terminal
- [ ] Test progress bar overwrites correctly (manual visual test)

## Documentation Requirements

- [ ] Doc comments on MinimalRenderer struct
- [ ] Doc comments on all public methods
- [ ] Example output in module docs

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward implementation using existing components. Main complexity is progress bar state management.

## Definition of Done

- [ ] Implementation complete
- [ ] Implements DisplayRenderer trait
- [ ] Progress bar updates correctly
- [ ] Training summary displays properly
- [ ] All tests passing
- [ ] Documentation complete
- [ ] PR reviewed and merged
