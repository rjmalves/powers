# T-017: Implement progress bar component

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014 (color utilities)
> **Blocks**: T-019 (MinimalRenderer), T-024 (target gap progress)

## Files to Read Before Starting

- `src/display/components/color.rs` - Color utilities
- `src/display/terminal.rs` - Terminal width detection

## Context

### Background

Progress bars provide quick visual feedback for long-running operations. The MinimalRenderer uses a progress bar as its primary output, and the AdvancedRenderer uses a smaller variant for target gap progress.

### Current State

No progress bar implementation exists. MinimalRenderer needs this component to function.

## Specification

### Create `src/display/components/progress.rs`

#### Types

```rust
/// Style variants for progress bars
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressStyle {
    /// Standard block characters: [████░░░░░░]
    #[default]
    Block,
    /// ASCII compatible: [====------]
    Ascii,
    /// Thin bar: [▓▓▓▓░░░░░░]
    Thin,
}

/// Configuration for progress bar rendering
pub struct ProgressBarConfig {
    /// Total width of the bar (including brackets)
    pub width: u16,
    /// Style of fill characters
    pub style: ProgressStyle,
    /// Show percentage after bar
    pub show_percentage: bool,
    /// Show ETA after bar
    pub show_eta: bool,
    /// Show iteration count (e.g., "3/10")
    pub show_count: bool,
}

impl Default for ProgressBarConfig {
    fn default() -> Self {
        Self {
            width: 30,
            style: ProgressStyle::Block,
            show_percentage: true,
            show_eta: true,
            show_count: true,
        }
    }
}

/// Progress bar state for rendering
pub struct ProgressBar {
    /// Current progress value
    pub current: usize,
    /// Total expected value
    pub total: usize,
    /// Start time for ETA calculation
    pub started_at: std::time::Instant,
    /// Configuration
    pub config: ProgressBarConfig,
}
```

#### Functions

```rust
impl ProgressBar {
    /// Create a new progress bar
    pub fn new(total: usize, config: ProgressBarConfig) -> Self;
    
    /// Update the current progress
    pub fn set(&mut self, current: usize);
    
    /// Increment progress by 1
    pub fn inc(&mut self);
    
    /// Get the completion fraction (0.0 to 1.0)
    pub fn fraction(&self) -> f64;
    
    /// Calculate estimated time remaining
    pub fn eta(&self) -> Option<std::time::Duration>;
    
    /// Render the progress bar as a string
    pub fn render(&self, color_config: &ColorConfig) -> String;
    
    /// Render a compact version (just the bar, no extras)
    pub fn render_bar_only(&self, width: u16) -> String;
}

/// Create a simple inline progress indicator
/// Returns: "[████████░░░░░░░░░░░░] 40%"
pub fn simple_progress(fraction: f64, width: u16, style: ProgressStyle) -> String;
```

### Fill Characters by Style

| Style | Filled | Empty |
|-------|--------|-------|
| Block | █ | ░ |
| Ascii | = | - |
| Thin | ▓ | ░ |

### Output Format (Full)

```
[████████░░░░░░░░░░░░] 40% | 4/10 iter | ETA: 00:01:23
```

### ETA Calculation

```rust
fn calculate_eta(&self) -> Option<Duration> {
    if self.current == 0 {
        return None; // Can't calculate yet
    }
    
    let elapsed = self.started_at.elapsed();
    let rate = self.current as f64 / elapsed.as_secs_f64();
    
    if rate <= 0.0 {
        return None;
    }
    
    let remaining = self.total.saturating_sub(self.current);
    let eta_secs = remaining as f64 / rate;
    
    Some(Duration::from_secs_f64(eta_secs))
}
```

## Acceptance Criteria

- [ ] ProgressBar struct with all methods implemented
- [ ] All three styles render correctly
- [ ] ETA calculation reasonably accurate after a few iterations
- [ ] Returns "calculating..." or similar for first iteration
- [ ] Respects width constraints
- [ ] Colors applied for filled portion
- [ ] Unit tests for rendering and ETA

## Implementation Guide

### Step 1: Add to components/mod.rs

```rust
pub mod progress;
```

### Step 2: Implement ProgressBar struct

```rust
impl ProgressBar {
    pub fn new(total: usize, config: ProgressBarConfig) -> Self {
        Self {
            current: 0,
            total,
            started_at: std::time::Instant::now(),
            config,
        }
    }
    
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0; // Avoid division by zero
        }
        (self.current as f64 / self.total as f64).clamp(0.0, 1.0)
    }
}
```

### Step 3: Implement render

```rust
pub fn render(&self, color_config: &ColorConfig) -> String {
    let bar = self.render_bar_only(self.config.width);
    let mut parts = vec![bar];
    
    if self.config.show_percentage {
        parts.push(format!("{:3.0}%", self.fraction() * 100.0));
    }
    
    if self.config.show_count {
        parts.push(format!("{}/{} iter", self.current, self.total));
    }
    
    if self.config.show_eta {
        if let Some(eta) = self.eta() {
            parts.push(format!("ETA: {}", format_duration_compact(eta)));
        } else if self.current == 0 {
            parts.push("ETA: calculating...".to_string());
        }
    }
    
    parts.join(" | ")
}
```

### Step 4: Implement render_bar_only

```rust
pub fn render_bar_only(&self, width: u16) -> String {
    let (filled_char, empty_char) = match self.config.style {
        ProgressStyle::Block => ('█', '░'),
        ProgressStyle::Ascii => ('=', '-'),
        ProgressStyle::Thin => ('▓', '░'),
    };
    
    let inner_width = width.saturating_sub(2) as usize; // Account for [ ]
    let filled = (self.fraction() * inner_width as f64).round() as usize;
    let empty = inner_width.saturating_sub(filled);
    
    format!("[{}{}]",
        filled_char.to_string().repeat(filled),
        empty_char.to_string().repeat(empty))
}
```

### Patterns to Follow

- Use `saturating_sub` to avoid underflow
- Clamp fraction to 0.0..1.0
- Use `Instant::now()` for timing

### Pitfalls to Avoid

- ⚠️ Handle total=0 case (100% complete immediately)
- ⚠️ ETA can be wildly inaccurate early - show "calculating..."
- ⚠️ Unicode box characters may not render on all terminals

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_progress_fraction() {
    let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
    assert_eq!(bar.fraction(), 0.0);
    bar.set(5);
    assert_eq!(bar.fraction(), 0.5);
    bar.set(10);
    assert_eq!(bar.fraction(), 1.0);
}

#[test]
fn test_progress_fraction_zero_total() {
    let bar = ProgressBar::new(0, ProgressBarConfig::default());
    assert_eq!(bar.fraction(), 1.0);
}

#[test]
fn test_render_bar_block_style() {
    let mut bar = ProgressBar::new(10, ProgressBarConfig {
        width: 12,
        style: ProgressStyle::Block,
        ..Default::default()
    });
    bar.set(5);
    let rendered = bar.render_bar_only(12);
    assert_eq!(rendered, "[█████░░░░░]");
}

#[test]
fn test_render_bar_ascii_style() {
    let mut bar = ProgressBar::new(10, ProgressBarConfig {
        width: 12,
        style: ProgressStyle::Ascii,
        ..Default::default()
    });
    bar.set(5);
    let rendered = bar.render_bar_only(12);
    assert_eq!(rendered, "[=====-----]");
}

#[test]
fn test_simple_progress() {
    let result = simple_progress(0.5, 12, ProgressStyle::Ascii);
    assert_eq!(result, "[=====-----]");
}
```

### ETA Tests

```rust
#[test]
fn test_eta_none_at_start() {
    let bar = ProgressBar::new(10, ProgressBarConfig::default());
    assert!(bar.eta().is_none());
}

// Note: ETA calculation tests are time-sensitive. Use mock time or 
// focus on testing the formula with known values.
```

## Documentation Requirements

- [ ] Module-level docs with usage examples
- [ ] Doc comments on all public types and methods
- [ ] Example showing full progress bar rendering

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Progress bar logic is straightforward, but ETA calculation and edge cases add complexity.

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] All three styles work
- [ ] ETA reasonably accurate
- [ ] Documentation complete
- [ ] PR reviewed and merged
