# T-004: Define DisplayContext with all metrics fields

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-002](./ticket-002-define-display-profile.md), [T-003](./ticket-003-implement-cost-statistics.md)
> **Blocks**: [T-006](./ticket-006-define-display-renderer.md)

## Files to Read Before Starting

- `src/display/config.rs` - DisplayConfig (from T-002)
- `src/display/context.rs` - CostStatistics (from T-003)
- `src/timing/output.rs` - ForwardTimingOutput, BackwardTimingOutput, IterationTimingOutput
- `src/sddp/mod.rs` - IterationResult struct (~lines 77-90)
- `src/logging/context.rs` - Current LogContext (being replaced)

## Context

### Background

`DisplayContext` is the central data structure passed to renderers. It contains all metrics needed for any display profile. This replaces the limited `LogContext` which only carried basic iteration info.

### Current State

`LogContext` has: iteration, lower_bound, simulation_cost, forward_time, backward_time, total_time.

We need: all of the above plus forward cost distribution, first-stage metrics, cut info, trends, and more.

## Specification

### DisplayContext Struct

```rust
use std::time::Duration;
use crate::timing::{ForwardTimingOutput, BackwardTimingOutput};

/// Complete context for rendering iteration display.
///
/// Contains all metrics needed by any display profile. Computed once per iteration
/// and passed to the active renderer.
#[derive(Debug, Clone)]
pub struct DisplayContext {
    // ═══════════════════════════════════════════════════════════════════════════
    // Iteration identification
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Current iteration number (1-based).
    pub iteration: usize,
    
    /// Total planned iterations.
    pub total_iterations: usize,
    
    /// Whether this iteration should produce output.
    /// Prepared for future smart throttling; always true for now.
    pub should_print: bool,
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Convergence metrics
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Current lower bound from backward pass.
    pub lower_bound: f64,
    
    /// Lower bound from previous iteration (None for first iteration).
    pub previous_lower_bound: Option<f64>,
    
    /// Target gap for convergence (optional).
    /// If set, enables progress visualization toward this target.
    pub target_gap: Option<f64>,
    
    /// Current optimality gap as percentage.
    /// Computed as (simulation_cost - lower_bound) / lower_bound * 100.
    pub gap_percent: f64,
    
    /// Gap trend indicator.
    pub gap_trend: GapTrend,
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Forward pass metrics
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Individual forward pass costs for this iteration.
    pub forward_costs: Vec<f64>,
    
    /// Statistics computed from forward_costs.
    pub forward_cost_stats: CostStatistics,
    
    /// Detailed forward pass timing.
    pub forward_timing: ForwardTimingOutput,
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Backward pass metrics
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Detailed backward pass timing.
    pub backward_timing: BackwardTimingOutput,
    
    /// Risk-adjusted expected cost from first stage evaluation.
    /// This is the true policy quality indicator.
    pub first_stage_bound: f64,
    
    /// Individual branching scenario costs from first stage.
    /// Empty until backward pass completes first stage.
    pub first_stage_branching_costs: Vec<f64>,
    
    /// Statistics computed from first_stage_branching_costs.
    pub first_stage_stats: CostStatistics,
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Cut management
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Number of cuts added this iteration.
    pub cuts_added: usize,
    
    /// Number of cuts removed this iteration (by cut selection).
    pub cuts_removed: usize,
    
    /// Number of cuts returned from purge pool this iteration.
    pub cuts_returned: usize,
    
    /// Total active cuts after this iteration.
    pub cuts_active: usize,
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Timing
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Total iteration wall-clock time.
    pub iteration_time: Duration,
    
    /// Cumulative elapsed time since training start.
    pub elapsed_total: Duration,
    
    /// Total solver calls this iteration (forward + backward).
    pub solver_calls: usize,
}

/// Gap trend direction indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GapTrend {
    /// Gap is decreasing (improving) - shown as ↓
    Improving,
    
    /// Gap is increasing (worsening) - shown as ↑
    Worsening,
    
    /// Gap is stable (within tolerance) - shown as →
    #[default]
    Stable,
    
    /// Not enough data to determine trend (first iteration)
    Unknown,
}
```

### Builder Pattern

```rust
impl DisplayContext {
    /// Create a new DisplayContext with required fields.
    pub fn new(iteration: usize, total_iterations: usize) -> Self {
        Self {
            iteration,
            total_iterations,
            should_print: true,
            lower_bound: 0.0,
            previous_lower_bound: None,
            target_gap: None,
            gap_percent: 0.0,
            gap_trend: GapTrend::Unknown,
            forward_costs: Vec::new(),
            forward_cost_stats: CostStatistics::default(),
            forward_timing: ForwardTimingOutput::default(),
            backward_timing: BackwardTimingOutput::default(),
            first_stage_bound: 0.0,
            first_stage_branching_costs: Vec::new(),
            first_stage_stats: CostStatistics::default(),
            cuts_added: 0,
            cuts_removed: 0,
            cuts_returned: 0,
            cuts_active: 0,
            iteration_time: Duration::ZERO,
            elapsed_total: Duration::ZERO,
            solver_calls: 0,
        }
    }
    
    /// Compute gap percentage from current bounds.
    pub fn compute_gap(&mut self) {
        let sim_cost = self.forward_cost_stats.mean;
        if self.lower_bound.abs() > 1e-10 {
            self.gap_percent = ((sim_cost - self.lower_bound) / self.lower_bound.abs()) * 100.0;
        } else {
            self.gap_percent = f64::INFINITY;
        }
    }
    
    /// Compute gap trend from previous iteration.
    pub fn compute_trend(&mut self, previous_gap: Option<f64>) {
        const TOLERANCE: f64 = 0.1; // 0.1 percentage points
        
        self.gap_trend = match previous_gap {
            None => GapTrend::Unknown,
            Some(prev) => {
                let delta = self.gap_percent - prev;
                if delta < -TOLERANCE {
                    GapTrend::Improving
                } else if delta > TOLERANCE {
                    GapTrend::Worsening
                } else {
                    GapTrend::Stable
                }
            }
        };
    }
    
    /// Progress toward target gap as fraction [0, 1].
    ///
    /// Returns None if no target_gap is set.
    /// Returns 1.0 if current gap <= target gap.
    pub fn target_progress(&self) -> Option<f64> {
        self.target_gap.map(|target| {
            if target <= 0.0 {
                1.0
            } else {
                (1.0 - self.gap_percent / target).clamp(0.0, 1.0)
            }
        })
    }
}
```

### GapTrend Display

```rust
impl GapTrend {
    /// Symbol for display.
    pub fn symbol(&self) -> &'static str {
        match self {
            GapTrend::Improving => "↓",
            GapTrend::Worsening => "↑",
            GapTrend::Stable => "→",
            GapTrend::Unknown => " ",
        }
    }
    
    /// ANSI color code (green for improving, red for worsening).
    pub fn color_code(&self) -> &'static str {
        match self {
            GapTrend::Improving => "\x1b[32m",  // Green
            GapTrend::Worsening => "\x1b[31m",  // Red
            GapTrend::Stable => "\x1b[33m",     // Yellow
            GapTrend::Unknown => "",
        }
    }
}
```

## Acceptance Criteria

- [ ] `DisplayContext` struct with all fields documented
- [ ] `GapTrend` enum with 4 variants
- [ ] `DisplayContext::new()` constructor
- [ ] `compute_gap()` method works correctly
- [ ] `compute_trend()` classifies improving/worsening/stable
- [ ] `target_progress()` returns correct fraction
- [ ] `GapTrend::symbol()` returns correct Unicode arrows
- [ ] Unit tests for gap computation edge cases

## Implementation Guide

### Step 1: Define GapTrend enum

Simple enum with symbol() and color_code() methods.

### Step 2: Define DisplayContext

Large struct - organize with section comments for readability.

### Step 3: Implement methods

Start with new(), then compute methods.

### Step 4: Add tests

## Pitfalls to Avoid

- ⚠️ Division by zero in `compute_gap()` when `lower_bound ≈ 0`
- ⚠️ `gap_percent` should be positive when simulation > lower (the normal case)
- ⚠️ `target_progress` should handle negative gaps (better than target)

## Testing Requirements

### Unit Tests

- [ ] Test `compute_gap` with typical values
- [ ] Test `compute_gap` with near-zero lower bound
- [ ] Test `compute_trend` first iteration (Unknown)
- [ ] Test `compute_trend` improving (gap decreases)
- [ ] Test `compute_trend` worsening (gap increases)
- [ ] Test `compute_trend` stable (within tolerance)
- [ ] Test `target_progress` with target set
- [ ] Test `target_progress` with no target (None)
- [ ] Test `target_progress` when gap exceeds target (0.0)
- [ ] Test `target_progress` when gap meets target (1.0)
- [ ] Test `GapTrend::symbol()` returns correct arrows

## Documentation Requirements

- [ ] Section comments organizing struct fields
- [ ] Doc comments on every field explaining source/meaning
- [ ] Doc comments on methods with examples
- [ ] Module-level overview in context.rs

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Many fields but straightforward. Most logic is simple computation.

## Definition of Done

- [ ] All types implemented
- [ ] All methods implemented and tested
- [ ] Types exported from `src/display/mod.rs`
- [ ] PR reviewed and merged
