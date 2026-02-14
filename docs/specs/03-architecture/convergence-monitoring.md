---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §16 (16.1-16.3)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Convergence Monitoring

## Purpose

This spec defines the POWE.RS SDDP convergence monitoring architecture: the convergence criteria and stopping rules, the convergence monitor implementation with bound tracking and stability detection, bound computation details including cross-rank aggregation, and the training log format for progress reporting.

## 1. Convergence Criteria

SDDP convergence is determined by the gap between lower and upper bounds on the optimal objective:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SDDP Convergence Monitoring                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Lower Bound (LB):                                                               │
│  ─────────────────                                                               │
│  - Computed from stage-1 LP objective (includes θ₂)                             │
│  - Deterministic: same value regardless of scenario                              │
│  - Monotonically non-decreasing as cuts are added                               │
│  - LB = min_x { c₁ᵀx₁ + θ₂ : constraints }                                      │
│                                                                                  │
│  Upper Bound (UB):                                                               │
│  ─────────────────                                                               │
│  - Statistical estimate from forward simulation costs                            │
│  - UB_k = (1/N) Σᵢ Σₜ cost(scenario i, stage t)                                 │
│  - Includes confidence interval: UB ± z_α × σ/√N                                │
│  - Not monotonic (depends on sampled scenarios)                                  │
│                                                                                  │
│  Convergence Gap:                                                                │
│  ────────────────                                                                │
│  gap = (UB - LB) / |UB|                                                         │
│                                                                                  │
│  Stopping Rules:                                                                 │
│  ───────────────                                                                 │
│  1. Gap tolerance: gap < ε (e.g., 1%)                                           │
│  2. Stable bound: LB unchanged for K iterations                                  │
│  3. Iteration limit: k ≥ max_iterations                                          │
│  4. Time limit: elapsed ≥ time_limit                                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Convergence Monitor Implementation

```rust
/// Tracks convergence statistics across iterations
pub struct ConvergenceMonitor {
    // Bound histories
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    upper_bound_stds: Vec<f64>,

    // Gap history
    gaps: Vec<f64>,

    // Iteration timing
    iteration_times: Vec<Duration>,

    // Stability tracking
    stable_lb_count: usize,
    last_lb_change_iteration: usize,
}

impl ConvergenceMonitor {
    pub fn new() -> Self {
        Self {
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            upper_bound_stds: Vec::new(),
            gaps: Vec::new(),
            iteration_times: Vec::new(),
            stable_lb_count: 0,
            last_lb_change_iteration: 0,
        }
    }

    /// Update with results from current iteration
    pub fn update(&mut self, forward: &GlobalForwardResult, fcf: &FutureCostFunction) {
        let iteration = self.lower_bounds.len();

        // Lower bound: stage-1 objective (deterministic)
        let lb = forward.lower_bound;

        // Upper bound: mean of scenario costs with std
        let ub = forward.mean_cost;
        let ub_std = forward.cost_std;

        // Check LB stability
        if let Some(&prev_lb) = self.lower_bounds.last() {
            if (lb - prev_lb).abs() < 1e-6 * prev_lb.abs().max(1.0) {
                self.stable_lb_count += 1;
            } else {
                self.stable_lb_count = 0;
                self.last_lb_change_iteration = iteration;
            }
        }

        // Compute gap
        let gap = if ub.abs() > 1e-10 {
            (ub - lb) / ub.abs()
        } else {
            0.0
        };

        // Record
        self.lower_bounds.push(lb);
        self.upper_bounds.push(ub);
        self.upper_bound_stds.push(ub_std);
        self.gaps.push(gap);
    }

    /// Check if converged based on configuration
    pub fn is_converged(&self, config: &TrainingConfig) -> bool {
        let iteration = self.lower_bounds.len();

        // Check gap tolerance
        if let Some(&gap) = self.gaps.last() {
            if gap < config.gap_tolerance {
                return true;
            }
        }

        // Check stable lower bound
        if self.stable_lb_count >= config.stable_iterations {
            return true;
        }

        false
    }

    /// Get current statistics for logging
    pub fn current_stats(&self) -> ConvergenceStats {
        ConvergenceStats {
            iteration: self.lower_bounds.len(),
            lower_bound: self.lower_bounds.last().copied().unwrap_or(0.0),
            upper_bound: self.upper_bounds.last().copied().unwrap_or(0.0),
            upper_bound_std: self.upper_bound_stds.last().copied().unwrap_or(0.0),
            gap: self.gaps.last().copied().unwrap_or(1.0),
            stable_iterations: self.stable_lb_count,
        }
    }
}

/// Statistics for logging/reporting
pub struct ConvergenceStats {
    pub iteration: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub upper_bound_std: f64,
    pub gap: f64,
    pub stable_iterations: usize,
}

impl std::fmt::Display for ConvergenceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Iter {:4} | LB: {:12.2} | UB: {:12.2} ± {:8.2} | Gap: {:6.2}%",
            self.iteration,
            self.lower_bound,
            self.upper_bound,
            self.upper_bound_std * 1.96,  // 95% CI
            self.gap * 100.0,
        )
    }
}
```

## 3. Bound Computation Details

```rust
/// Global forward result aggregated across all ranks
pub struct GlobalForwardResult {
    /// Lower bound (stage-1 objective, deterministic)
    pub lower_bound: f64,

    /// Mean cost across all scenarios
    pub mean_cost: f64,

    /// Standard deviation of scenario costs
    pub cost_std: f64,

    /// Number of scenarios
    pub n_scenarios: usize,

    /// 95% confidence interval half-width
    pub ci_95: f64,
}

impl<R: RiskMeasure, C: CutFormulation, H: HorizonMode> TrainingLoop<R, C, H> {
    /// Aggregate forward results from all ranks
    fn sync_forward_results(&self, local: &ForwardResult) -> GlobalForwardResult {
        // Gather local statistics
        let local_n = local.trajectories.len() as f64;
        let local_sum: f64 = local.trajectories.iter()
            .map(|t| t.total_cost)
            .sum();
        let local_sum_sq: f64 = local.trajectories.iter()
            .map(|t| t.total_cost.powi(2))
            .sum();

        // Reduce across ranks
        let mut global_n = 0.0;
        let mut global_sum = 0.0;
        let mut global_sum_sq = 0.0;

        self.comm.all_reduce(&local_n, &mut global_n, MpiOp::Sum);
        self.comm.all_reduce(&local_sum, &mut global_sum, MpiOp::Sum);
        self.comm.all_reduce(&local_sum_sq, &mut global_sum_sq, MpiOp::Sum);

        // Compute statistics
        let mean = global_sum / global_n;
        let variance = (global_sum_sq / global_n) - mean.powi(2);
        let std = variance.sqrt();
        let ci_95 = 1.96 * std / global_n.sqrt();

        // Lower bound: stage-1 objective from any scenario (deterministic)
        // All scenarios have same stage-1 LP, so use first
        let lb = if self.comm.rank() == 0 {
            local.trajectories.first()
                .map(|t| t.stage_costs[0])
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let mut global_lb = 0.0;
        self.comm.broadcast(&lb, &mut global_lb, 0);

        GlobalForwardResult {
            lower_bound: global_lb,
            mean_cost: mean,
            cost_std: std,
            n_scenarios: global_n as usize,
            ci_95,
        }
    }
}
```

## 4. Convergence Logging

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SDDP Training Log Format                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  POWE.RS SDDP Training                                                          │
│  Case: Brazilian_Interconnected_System                                           │
│  Started: 2026-01-31 10:30:00                                                   │
│  Ranks: 8 | Threads/rank: 24 | Stages: 120 | Hydros: 156                        │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Iter    1 | LB:  1.23456e+09 | UB:  2.34567e+09 ± 1.23e+08 | Gap: 47.38%      │
│  Iter    2 | LB:  1.45678e+09 | UB:  2.12345e+09 ± 9.87e+07 | Gap: 31.44%      │
│  Iter    3 | LB:  1.56789e+09 | UB:  1.98765e+09 ± 8.76e+07 | Gap: 21.11%      │
│  ...                                                                             │
│  Iter   47 | LB:  1.87654e+09 | UB:  1.89012e+09 ± 2.34e+07 | Gap:  0.72%      │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  CONVERGED after 47 iterations (gap < 1.00%)                                    │
│  Total time: 23m 45s | Avg iteration: 30.3s                                     │
│  Final LB: 1.87654e+09 | Final UB: 1.89012e+09 ± 2.34e+07                       │
│  Total cuts: 5,640 | Cuts/stage: ~47                                            │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Cross-References

- [Stopping Rules](../01-math/stopping-rules.md) — Mathematical definitions of the stopping rules implemented by the convergence monitor
- [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md) — Statistical upper bound theory, confidence interval construction, and bias corrections
- [Training Loop](./training-loop.md) — The SDDP training loop that invokes this convergence monitor each iteration
