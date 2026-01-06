# T-015: Implement statistics formatter component

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014 (color utilities)
> **Blocks**: T-019, T-021

## Files to Read Before Starting

- `src/display/context.rs` - CostStatistics struct
- `src/display/components/color.rs` - Color utilities
- `src/timing/mod.rs` - Duration formatting patterns

## Context

### Background

The advanced and standard renderers need to display cost statistics (mean, std dev, min, max) and timing information in a consistent, compact format. This component provides formatting functions for these common display patterns.

### Current State

`CostStatistics` exists from Epic 1. Now we need display formatting for terminal output with appropriate precision and units.

## Specification

### Create `src/display/components/statistics.rs`

#### Types

```rust
/// Options for formatting statistics
pub struct StatisticsFormat {
    /// Use scientific notation for large numbers
    pub scientific: bool,
    /// Number of significant digits
    pub precision: usize,
    /// Include sample count
    pub show_count: bool,
}

impl Default for StatisticsFormat {
    fn default() -> Self {
        Self {
            scientific: true,
            precision: 2,
            show_count: true,
        }
    }
}
```

#### Functions

```rust
/// Format a single cost value with appropriate precision
/// Returns something like "1.23e+05" or "123,456"
pub fn format_cost(value: f64, scientific: bool) -> String;

/// Format cost statistics in compact form
/// Returns: "μ=1.28e5 σ=3.2e3 [1.24e5..1.35e5] n=4"
pub fn format_cost_stats(stats: &CostStatistics, format: &StatisticsFormat) -> String;

/// Format cost statistics on one line with explicit labels
/// Returns: "mean: 1.28e5 | std: 3.2e3 | range: [1.24e5, 1.35e5]"
pub fn format_cost_stats_labeled(stats: &CostStatistics, format: &StatisticsFormat) -> String;

/// Format a duration in compact form
/// Returns: "0.034s" or "1.23m" or "2h 15m"
pub fn format_duration_compact(duration: std::time::Duration) -> String;

/// Format timing as forward/backward pair
/// Returns: "0.018s / 0.034s"
pub fn format_timing_pair(forward: std::time::Duration, backward: std::time::Duration) -> String;

/// Format a percentage value with sign
/// Returns: "+18.6%" or "-5.2%"
pub fn format_percentage_change(value: f64) -> String;

/// Format gap percentage
/// Returns: "26.4%" or "2.47%"
pub fn format_gap(gap_percentage: f64) -> String;
```

### Formatting Rules

#### Cost Values

- Values >= 1e4: Use scientific notation (e.g., `1.23e+05`)
- Values < 1e4: Use fixed with 2 decimal places (e.g., `1234.56`)
- Always show sign for percentage changes

#### Duration

| Duration | Format |
|----------|--------|
| < 1s | `0.XXXs` (3 decimal places) |
| < 60s | `X.XXs` (2 decimal places) |
| < 60m | `Xm XXs` |
| >= 60m | `Xh XXm` |

#### Gap Percentage

- 1 decimal place
- No leading zeros (not `00.5%`, use `0.5%`)

## Acceptance Criteria

- [ ] All formatting functions implemented
- [ ] Scientific notation for large costs
- [ ] Compact duration formatting
- [ ] Greek letters (μ, σ) for statistics display
- [ ] Handles edge cases (zero, negative, infinity, NaN)
- [ ] Unit tests for all functions
- [ ] Consistent precision across similar values

## Implementation Guide

### Step 1: Add to components/mod.rs

```rust
pub mod statistics;
```

### Step 2: Implement format_cost

```rust
pub fn format_cost(value: f64, scientific: bool) -> String {
    if !value.is_finite() {
        return if value.is_nan() { "NaN".to_string() } 
               else if value > 0.0 { "+∞".to_string() }
               else { "-∞".to_string() };
    }
    
    if scientific || value.abs() >= 1e4 {
        format!("{:.2e}", value)
    } else {
        format!("{:.2}", value)
    }
}
```

### Step 3: Implement format_cost_stats

```rust
pub fn format_cost_stats(stats: &CostStatistics, format: &StatisticsFormat) -> String {
    let mut parts = vec![
        format!("μ={}", format_cost(stats.mean, format.scientific)),
        format!("σ={}", format_cost(stats.std_dev, format.scientific)),
        format!("[{}..{}]", 
            format_cost(stats.min, format.scientific),
            format_cost(stats.max, format.scientific)),
    ];
    
    if format.show_count {
        parts.push(format!("n={}", stats.count));
    }
    
    parts.join(" ")
}
```

### Step 4: Implement duration formatting

Use `Duration::as_secs_f64()` for calculations.

### Patterns to Follow

- Use `format!` macro for string building
- Handle non-finite values explicitly
- Keep output compact for table cells

### Pitfalls to Avoid

- ⚠️ Don't assume values are always positive
- ⚠️ Watch for precision loss with very large/small values
- ⚠️ Test with actual `CostStatistics` values from the solver

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_format_cost_scientific() {
    assert_eq!(format_cost(123456.0, true), "1.23e+05");
    assert_eq!(format_cost(0.00123, true), "1.23e-03");
}

#[test]
fn test_format_cost_fixed() {
    assert_eq!(format_cost(1234.56, false), "1234.56");
}

#[test]
fn test_format_cost_edge_cases() {
    assert_eq!(format_cost(f64::NAN, true), "NaN");
    assert_eq!(format_cost(f64::INFINITY, true), "+∞");
    assert_eq!(format_cost(f64::NEG_INFINITY, true), "-∞");
}

#[test]
fn test_format_duration_compact() {
    assert_eq!(format_duration_compact(Duration::from_millis(34)), "0.034s");
    assert_eq!(format_duration_compact(Duration::from_secs(65)), "1m 05s");
    assert_eq!(format_duration_compact(Duration::from_secs(3725)), "1h 02m");
}

#[test]
fn test_format_cost_stats() {
    let stats = CostStatistics {
        mean: 128000.0,
        std_dev: 3200.0,
        min: 124000.0,
        max: 135000.0,
        count: 4,
    };
    let result = format_cost_stats(&stats, &StatisticsFormat::default());
    assert!(result.contains("μ=1.28e+05"));
    assert!(result.contains("σ=3.20e+03"));
    assert!(result.contains("n=4"));
}

#[test]
fn test_format_percentage_change() {
    assert_eq!(format_percentage_change(18.6), "+18.6%");
    assert_eq!(format_percentage_change(-5.2), "-5.2%");
    assert_eq!(format_percentage_change(0.0), "+0.0%");
}
```

## Documentation Requirements

- [ ] Module-level docs with examples
- [ ] Doc comments on all public functions
- [ ] Examples showing typical output formats

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Straightforward formatting logic. Clear specifications for all formats.

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Handles edge cases gracefully
- [ ] PR reviewed and merged
