# T-003: Implement CostStatistics aggregation

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-001](./ticket-001-add-crossterm-dependency.md)
> **Blocks**: [T-004](./ticket-004-define-display-context.md)

## Files to Read Before Starting

- `src/display/mod.rs` - Module structure (from T-001)
- `src/utils.rs` - Existing `mean()`, `standard_deviation()` functions
- `src/sddp/mod.rs` - Lines ~2075-2085 where cost statistics are computed

## Context

### Background

The display system needs to show real-time statistics for forward pass costs and first-stage branching scenario costs. This ticket creates a reusable `CostStatistics` struct that computes and holds descriptive statistics from a slice of costs.

### Current State

`src/utils.rs` has `mean()` and `standard_deviation()` functions. These will be reused.

## Specification

### CostStatistics Struct

```rust
/// Descriptive statistics for a collection of cost values.
///
/// Used to summarize forward pass costs and first-stage branching scenario costs
/// in a compact, displayable format.
#[derive(Debug, Clone, Default)]
pub struct CostStatistics {
    /// Arithmetic mean of costs.
    pub mean: f64,
    
    /// Sample standard deviation.
    pub std_dev: f64,
    
    /// Minimum cost value.
    pub min: f64,
    
    /// Maximum cost value.
    pub max: f64,
    
    /// Number of cost values.
    pub count: usize,
}
```

### Factory Method

```rust
impl CostStatistics {
    /// Compute statistics from a slice of costs.
    ///
    /// # Arguments
    ///
    /// * `costs` - Slice of cost values (f64)
    ///
    /// # Returns
    ///
    /// `CostStatistics` with computed values. Returns default (zeros) if slice is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use powers_rs::display::CostStatistics;
    ///
    /// let costs = vec![100.0, 110.0, 105.0, 115.0];
    /// let stats = CostStatistics::from_costs(&costs);
    /// assert_eq!(stats.count, 4);
    /// assert!((stats.mean - 107.5).abs() < 1e-10);
    /// ```
    pub fn from_costs(costs: &[f64]) -> Self {
        if costs.is_empty() {
            return Self::default();
        }
        
        let count = costs.len();
        let mean = crate::utils::mean(costs);
        let std_dev = crate::utils::standard_deviation(costs);
        
        let min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        Self {
            mean,
            std_dev,
            min,
            max,
            count,
        }
    }
    
    /// Check if statistics represent an empty dataset.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Compute range (max - min).
    #[inline]
    pub fn range(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.max - self.min
        }
    }
    
    /// Compute coefficient of variation (std_dev / mean).
    ///
    /// Returns 0.0 if mean is zero or near-zero to avoid division issues.
    #[inline]
    pub fn cv(&self) -> f64 {
        if self.mean.abs() < 1e-10 {
            0.0
        } else {
            self.std_dev / self.mean.abs()
        }
    }
}
```

### Formatting Support

```rust
impl CostStatistics {
    /// Format as compact string for display.
    ///
    /// # Example output
    /// 
    /// "μ=1.28e5 σ=3.2e3 [1.24e5..1.35e5]"
    pub fn format_compact(&self) -> String {
        if self.is_empty() {
            return "n/a".to_string();
        }
        
        format!(
            "μ={:.2e} σ={:.1e} [{:.2e}..{:.2e}]",
            self.mean, self.std_dev, self.min, self.max
        )
    }
    
    /// Format as detailed string with count.
    ///
    /// # Example output
    ///
    /// "μ=1.28e5 σ=3.2e3 [1.24e5..1.35e5] n=4"
    pub fn format_detailed(&self) -> String {
        if self.is_empty() {
            return "n/a".to_string();
        }
        
        format!(
            "μ={:.2e} σ={:.1e} [{:.2e}..{:.2e}] n={}",
            self.mean, self.std_dev, self.min, self.max, self.count
        )
    }
}
```

## Acceptance Criteria

- [ ] `CostStatistics` struct with all fields
- [ ] `from_costs()` computes correct statistics
- [ ] Empty slice returns default (zero) values
- [ ] Single element slice: mean=value, std_dev=0, min=max=value
- [ ] `format_compact()` produces expected output
- [ ] `format_detailed()` produces expected output with count
- [ ] Helper methods `is_empty()`, `range()`, `cv()` work correctly
- [ ] Unit tests cover edge cases

## Implementation Guide

### Step 1: Create struct

Define `CostStatistics` in `src/display/context.rs`.

### Step 2: Implement from_costs

Use existing utils functions for mean/std_dev, compute min/max with fold.

### Step 3: Implement formatting

Use scientific notation (`{:.2e}`) for consistency with existing output.

### Step 4: Write comprehensive tests

## Pitfalls to Avoid

- ⚠️ Don't divide by zero when computing CV with near-zero mean
- ⚠️ Handle empty slices gracefully (return defaults, not panic)
- ⚠️ Use `f64::INFINITY` and `NEG_INFINITY` for initial min/max (not first element)

## Testing Requirements

### Unit Tests

- [ ] Test `from_costs` with typical data [100.0, 110.0, 105.0, 115.0]
- [ ] Test `from_costs` with empty slice
- [ ] Test `from_costs` with single element
- [ ] Test `from_costs` with identical values (std_dev should be 0)
- [ ] Test `from_costs` with negative values (valid for some contexts)
- [ ] Test `is_empty()` returns correct boolean
- [ ] Test `range()` computation
- [ ] Test `cv()` with normal values
- [ ] Test `cv()` with zero mean
- [ ] Test `format_compact()` output format
- [ ] Test `format_detailed()` includes count

### Edge Cases

- [ ] Very large values (1e15) - no overflow
- [ ] Very small values (1e-15) - no underflow
- [ ] Mix of very different magnitudes

## Documentation Requirements

- [ ] Doc comments on struct explaining purpose
- [ ] Doc comments on each field
- [ ] Doc comments with examples on `from_costs()`
- [ ] Doc comments on formatting methods with example output

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple statistics computation with existing utility functions. Well-defined behavior.

## Definition of Done

- [ ] All methods implemented
- [ ] All tests passing
- [ ] Type exported from `src/display/mod.rs`
- [ ] PR reviewed and merged
