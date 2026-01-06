# T-024: Add target gap progress visualization

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-021 (AdvancedRenderer iteration row)
> **Blocks**: T-025 (visual polish)

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - AdvancedRenderer
- `src/display/components/progress.rs` - Progress bar component
- `src/display/config.rs` - DisplayConfig (target_gap field)
- `src/display/context.rs` - DisplayContext

## Context

### Background

When users configure a target gap for training (e.g., "stop when gap < 5%"), it's helpful to show progress toward that goal. This ticket adds an optional progress bar visualization that appears in the AdvancedRenderer when a target gap is configured.

### Current State

AdvancedRenderer shows gap percentage with trends. There's no visual indication of progress toward a target.

## Specification

### Target Gap Progress Bar

When `config.target_gap` is set (not None), show a progress bar after the iteration table:

```
Target Gap Progress: [████████░░░░░░░░░░░░] 40% → 5.0%
                     Current: 12.3% → Target: 5.0%
```

Or inline at the end of the header:

```
╭─────────────────────────────────────────────────────────────────────────────────╮
│ POWE.RS - Power Optimization for the World of Energy                           │
│ Training: 8 iterations × 4 forward passes | Cut selection: enabled             │
│ Target: ≤5.0% gap [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%                           │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

### Progress Calculation

The target gap progress is calculated as:
- 0% at start (unknown gap, or gap >> target)
- 100% when gap ≤ target

Formula:
```rust
fn gap_progress(current_gap: f64, initial_gap: Option<f64>, target_gap: f64) -> f64 {
    let initial = initial_gap.unwrap_or(100.0); // Assume 100% if unknown
    
    if current_gap <= target_gap {
        return 100.0;
    }
    
    if initial <= target_gap {
        return 100.0; // Already at target
    }
    
    // How far have we come from initial toward target?
    let total_distance = initial - target_gap;
    let distance_covered = initial - current_gap;
    
    ((distance_covered / total_distance) * 100.0).clamp(0.0, 100.0)
}
```

### DisplayContext Extension

Add tracking for initial gap:

```rust
pub struct DisplayContext {
    // ... existing fields ...
    
    /// Initial gap percentage (first iteration)
    pub initial_gap: Option<f64>,
    
    /// Target gap percentage (if configured)
    pub target_gap: Option<f64>,
}
```

### Update Locations

1. **Header**: Show target in header box (if configured)
2. **Iteration**: Optionally show progress after each iteration
3. **Summary**: Show final progress toward target

## Acceptance Criteria

- [ ] Gap progress calculation implemented correctly
- [ ] Progress bar appears when target_gap is configured
- [ ] Progress bar does not appear when target_gap is None
- [ ] Header shows target gap if configured
- [ ] Progress updates each iteration
- [ ] 100% shown when target achieved
- [ ] Works with both large and small target values
- [ ] Unit tests for progress calculation

## Implementation Guide

### Step 1: Add helper function for progress

```rust
// In src/display/components/progress.rs or new file

/// Calculate progress toward a target gap
pub fn gap_progress(current_gap: f64, initial_gap: Option<f64>, target_gap: f64) -> f64 {
    let initial = initial_gap.unwrap_or(100.0);
    
    if current_gap <= target_gap {
        return 100.0;
    }
    
    if initial <= target_gap {
        return 100.0;
    }
    
    let total_distance = initial - target_gap;
    let distance_covered = initial - current_gap;
    
    ((distance_covered / total_distance) * 100.0).clamp(0.0, 100.0)
}
```

### Step 2: Update render_header_box in AdvancedRenderer

```rust
fn render_header_box(&self, config: &DisplayConfig) -> String {
    let width = self.terminal_width.min(85).max(60) as usize;
    let inner_width = width - 4;
    
    let line1 = "POWE.RS - Power Optimization for the World of Energy";
    let line1_styled = bold(line1, &self.color_config);
    
    let line2 = format!(
        "Training: {} iterations × {} forward passes | Cut selection: {}",
        config.max_iterations,
        config.forward_passes,
        if config.cut_selection { "enabled" } else { "disabled" }
    );
    
    let mut lines = vec![
        format!("╭{}╮", "─".repeat(width - 2)),
        format!("│ {:<inner$} │", line1_styled, inner = inner_width),
        format!("│ {:<inner$} │", line2, inner = inner_width),
    ];
    
    // Add target gap line if configured
    if let Some(target) = config.target_gap {
        let line3 = format!("Target: ≤{:.1}% gap", target);
        lines.push(format!("│ {:<inner$} │", line3, inner = inner_width));
    }
    
    lines.push(format!("╰{}╯", "─".repeat(width - 2)));
    
    lines.join("\n") + "\n"
}
```

### Step 3: Add progress indicator to iteration output

In `render_iteration`, after the main row:

```rust
fn render_iteration(&self, ctx: &DisplayContext) -> String {
    // ... existing iteration rendering ...
    
    // Add target progress if configured
    if let (Some(target), Some(initial)) = (ctx.target_gap, ctx.initial_gap) {
        let current_gap = ctx.gap_percentage();
        let progress = gap_progress(current_gap, Some(initial), target);
        
        if progress < 100.0 {
            let bar = simple_progress(progress / 100.0, 20, ProgressStyle::Block);
            lines.push(format!("Target Progress: {} {:.0}% → {:.1}%", 
                bar, progress, target));
        } else {
            let achieved = colorize("Target Achieved! ✓", SemanticColor::Good, &self.color_config);
            lines.push(achieved);
        }
    }
    
    lines.join("\n")
}
```

### Step 4: Update training summary

```rust
fn render_training_summary(&self, result: &TrainingResult) -> String {
    // ... existing summary ...
    
    // Add target gap result if configured
    if let Some(target) = result.target_gap {
        if result.gap_percentage <= target {
            let msg = format!("Target gap of {:.1}% achieved!", target);
            lines.push(colorize(&msg, SemanticColor::Good, &self.color_config));
        } else {
            let msg = format!("Target gap of {:.1}% not reached (final: {:.2}%)", 
                target, result.gap_percentage);
            lines.push(colorize(&msg, SemanticColor::Caution, &self.color_config));
        }
    }
    
    lines.join("\n")
}
```

### Patterns to Follow

- Only show target-related UI when target is configured
- Use green color when target achieved
- Show clear progress toward goal

### Pitfalls to Avoid

- ⚠️ Don't show progress bar when no target configured
- ⚠️ Handle initial_gap = None (first iteration)
- ⚠️ Handle target already achieved at start

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_gap_progress_at_start() {
    // Initial gap 50%, target 5%, current 50% = 0% progress
    assert_eq!(gap_progress(50.0, Some(50.0), 5.0), 0.0);
}

#[test]
fn test_gap_progress_halfway() {
    // Initial 50%, target 5%, current 27.5% = 50% progress
    // (50 - 27.5) / (50 - 5) = 22.5 / 45 = 50%
    assert!((gap_progress(27.5, Some(50.0), 5.0) - 50.0).abs() < 0.1);
}

#[test]
fn test_gap_progress_achieved() {
    // Current gap at or below target = 100%
    assert_eq!(gap_progress(4.0, Some(50.0), 5.0), 100.0);
    assert_eq!(gap_progress(5.0, Some(50.0), 5.0), 100.0);
}

#[test]
fn test_gap_progress_no_initial() {
    // No initial gap defaults to 100%
    let progress = gap_progress(50.0, None, 5.0);
    assert!(progress > 0.0); // Some progress from assumed 100%
}

#[test]
fn test_gap_progress_already_at_target() {
    // Already at target = 100%
    assert_eq!(gap_progress(3.0, Some(3.0), 5.0), 100.0);
}

#[test]
fn test_gap_progress_clamped() {
    // Gap worse than initial shouldn't go negative
    assert_eq!(gap_progress(60.0, Some(50.0), 5.0), 0.0);
}

#[test]
fn test_header_shows_target_when_configured() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig {
        target_gap: Some(5.0),
        ..Default::default()
    };
    
    let header = renderer.render_header(&config);
    assert!(header.contains("Target:"));
    assert!(header.contains("5.0%"));
}

#[test]
fn test_header_no_target_when_not_configured() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig {
        target_gap: None,
        ..Default::default()
    };
    
    let header = renderer.render_header(&config);
    assert!(!header.contains("Target:"));
}
```

## Documentation Requirements

- [ ] Doc comments on gap_progress function
- [ ] Example of target gap progress visualization
- [ ] Note about when progress bar appears

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Clear calculation logic. Uses existing progress bar component. Limited scope.

## Definition of Done

- [ ] Gap progress calculation implemented
- [ ] Header shows target when configured
- [ ] Progress bar displays during iteration
- [ ] Summary shows target achievement status
- [ ] Works correctly at edge cases
- [ ] Tests passing
- [ ] Documentation complete
- [ ] PR reviewed and merged
