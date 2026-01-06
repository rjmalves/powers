# T-014: Implement color utilities with crossterm

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: Epic 1 complete
> **Blocks**: T-015, T-016, T-017, T-018, T-019

## Files to Read Before Starting

- `src/display/mod.rs` - Module structure
- `src/display/terminal.rs` - Terminal detection (color support)
- `src/display/config.rs` - DisplayConfig for color settings

## Context

### Background

All display components need consistent color styling. This ticket creates the foundational color utilities that wrap `crossterm` styling with POWE.RS-specific semantic colors and conditional application based on terminal capabilities.

### Current State

Epic 1 established the display module structure with terminal detection. Now we need the actual color utilities that other components will use.

## Specification

### Create `src/display/components/mod.rs`

```rust
//! Reusable display components for building rich terminal output.

pub mod color;
// Future: pub mod indicators;
// Future: pub mod progress;
// Future: pub mod statistics;
// Future: pub mod table;
```

### Create `src/display/components/color.rs`

#### Types

```rust
/// Semantic color categories for display elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticColor {
    /// Good/positive values (e.g., improving gap)
    Good,
    /// Bad/negative values (e.g., worsening gap)
    Bad,
    /// Caution/neutral values
    Caution,
    /// Informational text
    Info,
    /// Muted/secondary text
    Muted,
    /// Emphasis/highlight
    Emphasis,
    /// Standard text (no special styling)
    Normal,
}

/// Color configuration for styled output
pub struct ColorConfig {
    /// Whether colors are enabled
    pub enabled: bool,
}
```

#### Functions

```rust
/// Apply semantic color to text
pub fn colorize(text: &str, color: SemanticColor, config: &ColorConfig) -> String;

/// Apply color for trend direction
pub fn color_trend(text: &str, improving: bool, config: &ColorConfig) -> String;

/// Apply color for percentage values (green if low, red if high)
pub fn color_gap_percentage(gap: f64, text: &str, config: &ColorConfig) -> String;

/// Format a value with optional color based on comparison
pub fn color_comparison(
    current: f64,
    previous: Option<f64>,
    formatted: &str,
    config: &ColorConfig,
) -> String;

/// Style text as bold
pub fn bold(text: &str, config: &ColorConfig) -> String;

/// Style text as dim/muted
pub fn dim(text: &str, config: &ColorConfig) -> String;
```

### Color Mapping

| SemanticColor | crossterm Color |
|---------------|-----------------|
| Good | Green |
| Bad | Red |
| Caution | Yellow |
| Info | Cyan |
| Muted | DarkGrey |
| Emphasis | White + Bold |
| Normal | (no color) |

### Gap Percentage Thresholds

- gap < 5%: Good (green)
- gap < 20%: Caution (yellow)
- gap >= 20%: Bad (red)

## Acceptance Criteria

- [ ] `src/display/components/color.rs` created with all specified functions
- [ ] Colors only applied when `config.enabled == true`
- [ ] All functions return plain text when colors disabled
- [ ] Uses `crossterm::style::Stylize` for styling
- [ ] Unit tests cover enabled and disabled scenarios
- [ ] No panics on any input

## Implementation Guide

### Step 1: Create components directory

```bash
mkdir -p src/display/components
```

### Step 2: Create mod.rs

Create `src/display/components/mod.rs` with module declarations.

### Step 3: Update display/mod.rs

Add `pub mod components;` to exports.

### Step 4: Implement color.rs

```rust
use crossterm::style::{Color, Stylize};

impl ColorConfig {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }
}

pub fn colorize(text: &str, color: SemanticColor, config: &ColorConfig) -> String {
    if !config.enabled {
        return text.to_string();
    }
    
    match color {
        SemanticColor::Good => text.green().to_string(),
        SemanticColor::Bad => text.red().to_string(),
        SemanticColor::Caution => text.yellow().to_string(),
        SemanticColor::Info => text.cyan().to_string(),
        SemanticColor::Muted => text.dark_grey().to_string(),
        SemanticColor::Emphasis => text.white().bold().to_string(),
        SemanticColor::Normal => text.to_string(),
    }
}
```

### Patterns to Follow

- Keep functions pure (no side effects)
- Accept `&str` input, return `String` output
- Always check `config.enabled` first

### Pitfalls to Avoid

- ⚠️ Don't use `println!` or write directly - just return styled strings
- ⚠️ Ensure crossterm `Stylize` trait is in scope
- ⚠️ Test with actual terminal to verify colors render

## Testing Requirements

### Unit Tests

- [ ] `colorize` returns plain text when disabled
- [ ] `colorize` returns styled text when enabled (check ANSI codes present)
- [ ] `color_gap_percentage` applies correct color for thresholds (4%, 15%, 25%)
- [ ] `color_comparison` handles `None` previous value
- [ ] `color_comparison` shows green when improving, red when worsening
- [ ] `bold` and `dim` work correctly

### Test Strategy

Since ANSI codes are embedded in strings, tests can verify:
- String contains expected escape sequences when enabled
- String equals input when disabled

```rust
#[test]
fn test_colorize_disabled() {
    let config = ColorConfig::new(false);
    assert_eq!(colorize("test", SemanticColor::Good, &config), "test");
}

#[test]
fn test_colorize_enabled_contains_ansi() {
    let config = ColorConfig::new(true);
    let result = colorize("test", SemanticColor::Good, &config);
    assert!(result.contains("\x1b[")); // ANSI escape
    assert!(result.contains("test"));
}
```

## Documentation Requirements

- [ ] Module-level docs explaining purpose
- [ ] Doc comments on all public types and functions
- [ ] Examples in doc comments for `colorize` and `color_gap_percentage`

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Straightforward wrapper functions around crossterm. Well-defined color mappings.

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Documentation complete
- [ ] `cargo clippy` clean
- [ ] PR reviewed and merged
