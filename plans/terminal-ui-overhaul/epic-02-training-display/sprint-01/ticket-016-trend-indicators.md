# T-016: Implement trend indicators component

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014 (color utilities)
> **Blocks**: T-021, T-024

## Files to Read Before Starting

- `src/display/components/color.rs` - Color utilities for styling indicators
- `src/display/context.rs` - DisplayContext for trend data

## Context

### Background

The advanced profile shows visual trend indicators next to key metrics to help users quickly understand if training is progressing well. This component provides the arrows, icons, and bound improvement indicators used throughout the display.

### Current State

No visual indicators exist. Users must compare numbers manually to understand trends.

## Specification

### Create `src/display/components/indicators.rs`

#### Types

```rust
/// Direction of a trend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving (gap decreasing, bound increasing)
    Improving,
    /// Worsening (gap increasing, bound decreasing)
    Worsening,
    /// Stable (minimal change)
    Stable,
    /// Unknown (no previous value to compare)
    Unknown,
}

/// Configuration for trend calculation
pub struct TrendConfig {
    /// Threshold for considering a change significant (percentage)
    pub significance_threshold: f64,
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            significance_threshold: 0.1, // 0.1% change is significant
        }
    }
}
```

#### Functions

```rust
/// Get trend arrow based on direction
/// Returns: "↓" (improving), "↑" (worsening), "→" (stable), "" (unknown)
pub fn trend_arrow(direction: TrendDirection) -> &'static str;

/// Get trend arrow with color applied
pub fn trend_arrow_colored(
    direction: TrendDirection,
    color_config: &ColorConfig,
) -> String;

/// Calculate trend direction for gap (lower is better)
pub fn gap_trend(current: f64, previous: Option<f64>, config: &TrendConfig) -> TrendDirection;

/// Calculate trend direction for bound (higher is better)
pub fn bound_trend(current: f64, previous: Option<f64>, config: &TrendConfig) -> TrendDirection;

/// Get bound change indicator
/// Returns: "▲" (increased), "▼" (decreased), "─" (stable), "" (no change)
pub fn bound_change_indicator(current: f64, previous: Option<f64>) -> &'static str;

/// Format bound change with percentage and indicator
/// Returns: "▲ +18.6%" or "▼ -5.2%" or ""
pub fn format_bound_change(
    current: f64,
    previous: Option<f64>,
    color_config: &ColorConfig,
) -> String;

/// Get status icon for training state
pub fn status_icon(converged: bool) -> &'static str;
```

### Symbol Mappings

| Indicator | Improving | Worsening | Stable | Unknown |
|-----------|-----------|-----------|--------|---------|
| Gap Arrow | ↓ (green) | ↑ (red) | → (yellow) | (empty) |
| Bound Arrow | ▲ (green) | ▼ (red) | ─ (yellow) | (empty) |

### Status Icons

| State | Icon |
|-------|------|
| Converged | ✓ (green) |
| Not Converged | ⋯ (yellow) |

## Acceptance Criteria

- [ ] All indicator functions implemented
- [ ] Unicode symbols render correctly
- [ ] Colors applied via color utilities
- [ ] Trend calculation respects significance threshold
- [ ] Handles None previous values (returns Unknown)
- [ ] Unit tests cover all scenarios

## Implementation Guide

### Step 1: Add to components/mod.rs

```rust
pub mod indicators;
```

### Step 2: Implement TrendDirection

```rust
impl TrendDirection {
    /// Returns true if this trend is positive
    pub fn is_positive(&self) -> bool {
        matches!(self, TrendDirection::Improving)
    }
}
```

### Step 3: Implement trend_arrow

```rust
pub fn trend_arrow(direction: TrendDirection) -> &'static str {
    match direction {
        TrendDirection::Improving => "↓",
        TrendDirection::Worsening => "↑",
        TrendDirection::Stable => "→",
        TrendDirection::Unknown => "",
    }
}
```

### Step 4: Implement gap_trend

```rust
pub fn gap_trend(current: f64, previous: Option<f64>, config: &TrendConfig) -> TrendDirection {
    let Some(prev) = previous else {
        return TrendDirection::Unknown;
    };
    
    if prev == 0.0 {
        return if current < 0.0 { 
            TrendDirection::Improving 
        } else if current > 0.0 { 
            TrendDirection::Worsening 
        } else { 
            TrendDirection::Stable 
        };
    }
    
    let change_pct = ((current - prev) / prev.abs()) * 100.0;
    
    if change_pct < -config.significance_threshold {
        TrendDirection::Improving  // Gap decreased
    } else if change_pct > config.significance_threshold {
        TrendDirection::Worsening  // Gap increased
    } else {
        TrendDirection::Stable
    }
}
```

### Step 5: Implement bound_trend

Similar logic but inverted (higher is better for bounds).

### Step 6: Implement format_bound_change

```rust
pub fn format_bound_change(
    current: f64,
    previous: Option<f64>,
    color_config: &ColorConfig,
) -> String {
    let Some(prev) = previous else {
        return String::new();
    };
    
    if prev == 0.0 {
        return String::new();
    }
    
    let change_pct = ((current - prev) / prev.abs()) * 100.0;
    let indicator = bound_change_indicator(current, Some(prev));
    let formatted = format!("{} {:+.1}%", indicator, change_pct);
    
    let direction = bound_trend(current, Some(prev), &TrendConfig::default());
    trend_arrow_colored(direction, color_config); // For coloring logic
    
    colorize(&formatted, 
        if change_pct > 0.0 { SemanticColor::Good } else { SemanticColor::Bad },
        color_config)
}
```

### Patterns to Follow

- Use static string slices for symbols (no allocation)
- Apply colors via the color utilities, not directly
- Handle edge cases (zero, NaN) gracefully

### Pitfalls to Avoid

- ⚠️ Unicode symbols may not render on all terminals - this is acceptable
- ⚠️ Don't divide by zero when calculating percentage change
- ⚠️ Be careful with signs: for gap, decreasing is good; for bound, increasing is good

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_trend_arrow() {
    assert_eq!(trend_arrow(TrendDirection::Improving), "↓");
    assert_eq!(trend_arrow(TrendDirection::Worsening), "↑");
    assert_eq!(trend_arrow(TrendDirection::Stable), "→");
    assert_eq!(trend_arrow(TrendDirection::Unknown), "");
}

#[test]
fn test_gap_trend_improving() {
    let config = TrendConfig::default();
    // Gap went from 20% to 15% = improving
    assert_eq!(gap_trend(15.0, Some(20.0), &config), TrendDirection::Improving);
}

#[test]
fn test_gap_trend_worsening() {
    let config = TrendConfig::default();
    // Gap went from 10% to 15% = worsening
    assert_eq!(gap_trend(15.0, Some(10.0), &config), TrendDirection::Worsening);
}

#[test]
fn test_gap_trend_stable() {
    let config = TrendConfig { significance_threshold: 1.0 };
    // Gap changed by 0.5% which is below 1% threshold
    assert_eq!(gap_trend(10.05, Some(10.0), &config), TrendDirection::Stable);
}

#[test]
fn test_gap_trend_unknown() {
    let config = TrendConfig::default();
    assert_eq!(gap_trend(10.0, None, &config), TrendDirection::Unknown);
}

#[test]
fn test_bound_trend_improving() {
    let config = TrendConfig::default();
    // Bound increased = improving
    assert_eq!(bound_trend(120000.0, Some(100000.0), &config), TrendDirection::Improving);
}

#[test]
fn test_bound_change_indicator() {
    assert_eq!(bound_change_indicator(120.0, Some(100.0)), "▲");
    assert_eq!(bound_change_indicator(80.0, Some(100.0)), "▼");
    assert_eq!(bound_change_indicator(100.0, None), "");
}
```

## Documentation Requirements

- [ ] Module-level docs explaining indicator system
- [ ] Doc comments with examples for each function
- [ ] Note about Unicode symbol rendering

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple logic with clear specifications. Unicode symbols are straightforward.

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Symbols render correctly in test terminal
- [ ] PR reviewed and merged
