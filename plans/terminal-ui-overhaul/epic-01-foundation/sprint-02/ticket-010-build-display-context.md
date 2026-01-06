# T-010: Build DisplayContext from iteration data

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: Sprint 1 complete (DisplayContext type exists)
> **Blocks**: [T-011](./ticket-011-expose-first-stage-costs.md), [T-012](./ticket-012-integrate-training-loop.md)

## Files to Read Before Starting

- `src/display/context.rs` - DisplayContext struct (from T-004)
- `src/sddp/mod.rs` - Lines 1850-2015: training loop where iteration data is computed
- `src/timing/output.rs` - IterationTimingOutput struct
- `src/sddp/mod.rs` - IterationResult struct (~lines 77-90)

## Context

### Background

The `DisplayContext` needs to be populated from actual iteration data. This ticket creates a builder or factory that transforms SDDP iteration results into a `DisplayContext` ready for rendering.

### Current State

At the end of each iteration, the training loop has:
- `lower_bound` from `eval_first_stage_bound()`
- `forward_costs` from parallel forward passes
- `timing` (`NewIterationTiming`)
- Cut statistics

These need to be assembled into a `DisplayContext`.

## Specification

### DisplayContext Builder

```rust
impl DisplayContext {
    /// Build from SDDP iteration results.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number (1-based)
    /// * `total_iterations` - Total planned iterations
    /// * `iteration_result` - Completed iteration result
    /// * `previous_lower_bound` - Lower bound from previous iteration
    /// * `elapsed_total` - Time since training started
    /// * `target_gap` - Optional target gap from config
    ///
    /// # Returns
    ///
    /// Fully populated DisplayContext ready for rendering.
    pub fn from_iteration(
        iteration: usize,
        total_iterations: usize,
        iteration_result: &IterationResult,
        previous_lower_bound: Option<f64>,
        elapsed_total: Duration,
        target_gap: Option<f64>,
    ) -> Self {
        // Compute forward cost statistics
        let forward_cost_stats = CostStatistics::from_costs(&iteration_result.forward_costs);
        
        // Compute previous gap for trend
        let previous_gap = previous_lower_bound.map(|prev| {
            let prev_sim = forward_cost_stats.mean; // approximation
            if prev.abs() > 1e-10 {
                ((prev_sim - prev) / prev.abs()) * 100.0
            } else {
                f64::INFINITY
            }
        });
        
        let mut ctx = Self {
            iteration,
            total_iterations,
            should_print: true, // Always true for now
            
            lower_bound: iteration_result.lower_bound,
            previous_lower_bound,
            target_gap,
            gap_percent: 0.0, // Computed below
            gap_trend: GapTrend::Unknown,
            
            forward_costs: iteration_result.forward_costs.clone(),
            forward_cost_stats,
            forward_timing: iteration_result.timing.forward.clone(),
            
            backward_timing: iteration_result.timing.backward.clone(),
            first_stage_bound: iteration_result.lower_bound, // Same for now
            first_stage_branching_costs: Vec::new(), // Populated by T-011
            first_stage_stats: CostStatistics::default(),
            
            cuts_added: iteration_result.num_cuts_added,
            cuts_removed: iteration_result.num_cuts_removed,
            cuts_returned: iteration_result.num_cuts_returned,
            cuts_active: iteration_result.num_active_cuts,
            
            iteration_time: iteration_result.timing.total,
            elapsed_total,
            solver_calls: iteration_result.timing.solver_calls,
        };
        
        ctx.compute_gap();
        ctx.compute_trend(previous_gap);
        
        ctx
    }
}
```

### Helper for Tracking Previous Bounds

```rust
/// Tracker for computing iteration trends.
#[derive(Debug, Clone, Default)]
pub struct IterationTracker {
    previous_lower_bound: Option<f64>,
    previous_gap: Option<f64>,
    start_time: Option<Instant>,
}

impl IterationTracker {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start_time.map(|t| t.elapsed()).unwrap_or(Duration::ZERO)
    }
    
    pub fn update(&mut self, lower_bound: f64, gap: f64) {
        self.previous_lower_bound = Some(lower_bound);
        self.previous_gap = Some(gap);
    }
    
    pub fn previous_lower_bound(&self) -> Option<f64> {
        self.previous_lower_bound
    }
    
    pub fn previous_gap(&self) -> Option<f64> {
        self.previous_gap
    }
}
```

## Acceptance Criteria

- [ ] `DisplayContext::from_iteration()` builds valid context
- [ ] All fields from `IterationResult` mapped correctly
- [ ] `forward_cost_stats` computed from `forward_costs`
- [ ] Gap percentage computed correctly
- [ ] Gap trend computed from previous iteration
- [ ] `IterationTracker` tracks previous bounds across iterations
- [ ] `elapsed_total` reflects cumulative time
- [ ] `should_print` defaults to `true`

## Implementation Guide

### Step 1: Implement from_iteration

Add the builder method to DisplayContext.

### Step 2: Implement IterationTracker

Helper struct for tracking state across iterations.

### Step 3: Unit test gap computation

Verify gap and trend logic.

### Step 4: Test with mock data

Create test helpers for IterationResult.

## Pitfalls to Avoid

- ⚠️ Don't clone large vectors if avoidable (forward_costs is typically small)
- ⚠️ Handle first iteration specially (no previous bound/gap)
- ⚠️ `first_stage_bound` equals `lower_bound` initially; T-011 adds detail

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_from_iteration_basic() {
    let result = create_mock_iteration_result(1, 100000.0, vec![110000.0, 115000.0]);
    
    let ctx = DisplayContext::from_iteration(
        1,
        10,
        &result,
        None,
        Duration::from_secs(1),
        None,
    );
    
    assert_eq!(ctx.iteration, 1);
    assert_eq!(ctx.lower_bound, 100000.0);
    assert_eq!(ctx.forward_cost_stats.count, 2);
    assert!(ctx.gap_percent > 0.0);
    assert_eq!(ctx.gap_trend, GapTrend::Unknown); // First iteration
}

#[test]
fn test_from_iteration_with_previous() {
    let result1 = create_mock_iteration_result(1, 100000.0, vec![150000.0]);
    let result2 = create_mock_iteration_result(2, 110000.0, vec![130000.0]);
    
    let ctx2 = DisplayContext::from_iteration(
        2,
        10,
        &result2,
        Some(100000.0),
        Duration::from_secs(2),
        None,
    );
    
    // Gap improved (lower percentage)
    assert!(ctx2.gap_percent < 50.0); // ~18% vs ~50%
    assert_eq!(ctx2.gap_trend, GapTrend::Improving);
}

#[test]
fn test_iteration_tracker() {
    let mut tracker = IterationTracker::new();
    tracker.start();
    
    assert!(tracker.previous_lower_bound().is_none());
    
    tracker.update(100000.0, 20.0);
    
    assert_eq!(tracker.previous_lower_bound(), Some(100000.0));
    assert_eq!(tracker.previous_gap(), Some(20.0));
}
```

## Documentation Requirements

- [ ] Doc comments on `from_iteration()` explaining each parameter
- [ ] Doc comments on `IterationTracker`
- [ ] Example usage in module docs

## Effort Estimate

**Points**: 4
**Confidence**: Medium
**Rationale**: Requires careful mapping of all fields. Gap trend logic needs testing.

## Definition of Done

- [ ] Builder implemented and tested
- [ ] IterationTracker implemented
- [ ] All IterationResult fields mapped
- [ ] Gap computation tested
- [ ] Trend computation tested
- [ ] PR reviewed and merged
