//! Display context and statistics types.
//!
//! Contains the complete context for rendering iteration display,
//! including all metrics and computed statistics.

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

// TODO: Implement DisplayContext struct (T-004)
// TODO: Implement GapTrend enum (T-004)

use crate::timing::{BackwardTimingOutput, ForwardTimingOutput};
use std::time::{Duration, Instant};

/// Tracker for computing iteration trends.
///
/// Maintains state across iterations to compute gap trends and timing.
#[derive(Debug, Clone, Default)]
pub struct IterationTracker {
    previous_lower_bound: Option<f64>,
    previous_gap: Option<f64>,
    start_time: Option<Instant>,
}

impl IterationTracker {
    /// Create a new iteration tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start tracking (should be called before first iteration).
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Get elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Update tracker with current iteration's values.
    pub fn update(&mut self, lower_bound: f64, gap: f64) {
        self.previous_lower_bound = Some(lower_bound);
        self.previous_gap = Some(gap);
    }

    /// Get previous iteration's lower bound.
    pub fn previous_lower_bound(&self) -> Option<f64> {
        self.previous_lower_bound
    }

    /// Get previous iteration's gap.
    pub fn previous_gap(&self) -> Option<f64> {
        self.previous_gap
    }
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
            GapTrend::Improving => "\x1b[32m", // Green
            GapTrend::Worsening => "\x1b[31m", // Red
            GapTrend::Stable => "\x1b[33m",    // Yellow
            GapTrend::Unknown => "",
        }
    }
}

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

    /// Initial gap percentage from first iteration.
    /// Used to calculate progress toward target gap.
    pub initial_gap: Option<f64>,

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

impl DisplayContext {
    /// Create a new DisplayContext with required fields.
    pub fn new(iteration: usize, total_iterations: usize) -> Self {
        Self {
            iteration,
            total_iterations,
            should_print: true,
            lower_bound: 0.0,
            previous_lower_bound: None,
            initial_gap: None,
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
        iteration_result: &crate::sddp::IterationResult,
        previous_lower_bound: Option<f64>,
        elapsed_total: Duration,
        target_gap: Option<f64>,
    ) -> Self {
        // Compute forward cost statistics
        let forward_cost_stats =
            CostStatistics::from_costs(&iteration_result.forward_costs);

        // Compute first-stage statistics
        let first_stage_stats = CostStatistics::from_costs(
            &iteration_result.first_stage_branching_costs,
        );

        // Compute previous gap for trend (using approximation for now)
        let previous_gap = previous_lower_bound.and_then(|prev| {
            if prev.abs() > 1e-10 {
                Some(((forward_cost_stats.mean - prev) / prev.abs()) * 100.0)
            } else {
                None
            }
        });

        let mut ctx = Self {
            iteration,
            total_iterations,
            should_print: true, // Always true for now

            lower_bound: iteration_result.lower_bound,
            previous_lower_bound,
            initial_gap: None, // Set by caller if needed
            target_gap,
            gap_percent: 0.0, // Computed below
            gap_trend: GapTrend::Unknown,

            forward_costs: iteration_result.forward_costs.clone(),
            forward_cost_stats,
            forward_timing: iteration_result.timing.forward.clone(),

            backward_timing: iteration_result.timing.backward.clone(),
            first_stage_bound: iteration_result.lower_bound,
            first_stage_branching_costs: iteration_result
                .first_stage_branching_costs
                .clone(),
            first_stage_stats,

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

    /// Compute gap percentage from current bounds.
    pub fn compute_gap(&mut self) {
        let sim_cost = self.forward_cost_stats.mean;
        if self.lower_bound.abs() > 1e-10 {
            self.gap_percent = ((sim_cost - self.lower_bound)
                / self.lower_bound.abs())
                * 100.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_costs_typical() {
        let costs = vec![100.0, 110.0, 105.0, 115.0];
        let stats = CostStatistics::from_costs(&costs);

        assert_eq!(stats.count, 4);
        assert!((stats.mean - 107.5).abs() < 1e-10);
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 115.0);
        assert!((stats.range() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_costs_empty() {
        let stats = CostStatistics::from_costs(&[]);
        assert_eq!(stats.count, 0);
        assert!(stats.is_empty());
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.range(), 0.0);
    }

    #[test]
    fn test_from_costs_single() {
        let stats = CostStatistics::from_costs(&[42.0]);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
        assert_eq!(stats.std_dev, 0.0);
    }

    #[test]
    fn test_from_costs_identical() {
        let stats = CostStatistics::from_costs(&[5.0, 5.0, 5.0]);
        assert_eq!(stats.mean, 5.0);
        assert_eq!(stats.std_dev, 0.0);
    }

    #[test]
    fn test_from_costs_negative() {
        let stats = CostStatistics::from_costs(&[-10.0, -5.0, -15.0]);
        assert!((stats.mean - (-10.0)).abs() < 1e-10);
        assert_eq!(stats.min, -15.0);
        assert_eq!(stats.max, -5.0);
    }

    #[test]
    fn test_cv() {
        let stats = CostStatistics::from_costs(&[100.0, 110.0, 90.0]);
        let cv = stats.cv();
        assert!(cv > 0.0);
        assert!(cv < 1.0);
    }

    #[test]
    fn test_cv_zero_mean() {
        let stats = CostStatistics::from_costs(&[-1.0, 0.0, 1.0]);
        let cv = stats.cv();
        assert_eq!(cv, 0.0); // Should handle near-zero mean gracefully
    }

    #[test]
    fn test_format_compact() {
        let stats = CostStatistics::from_costs(&[100.0, 110.0, 105.0, 115.0]);
        let formatted = stats.format_compact();
        assert!(formatted.contains("μ="));
        assert!(formatted.contains("σ="));
    }

    #[test]
    fn test_format_detailed() {
        let stats = CostStatistics::from_costs(&[100.0, 110.0, 105.0, 115.0]);
        let formatted = stats.format_detailed();
        assert!(formatted.contains("n=4"));
    }

    #[test]
    fn test_format_empty() {
        let stats = CostStatistics::from_costs(&[]);
        assert_eq!(stats.format_compact(), "n/a");
        assert_eq!(stats.format_detailed(), "n/a");
    }

    #[test]
    fn test_large_values() {
        let stats = CostStatistics::from_costs(&[1e15, 2e15, 3e15]);
        assert!(stats.mean.is_finite());
        assert!(stats.std_dev.is_finite());
    }

    #[test]
    fn test_small_values() {
        let stats = CostStatistics::from_costs(&[1e-15, 2e-15, 3e-15]);
        assert!(stats.mean.is_finite());
        assert!(stats.std_dev.is_finite());
    }

    // DisplayContext tests
    #[test]
    fn test_display_context_new() {
        let ctx = DisplayContext::new(1, 10);
        assert_eq!(ctx.iteration, 1);
        assert_eq!(ctx.total_iterations, 10);
        assert!(ctx.should_print);
        assert_eq!(ctx.gap_trend, GapTrend::Unknown);
    }

    #[test]
    fn test_compute_gap() {
        let mut ctx = DisplayContext::new(1, 10);
        ctx.lower_bound = 100.0;
        ctx.forward_cost_stats = CostStatistics::from_costs(&[110.0]);
        ctx.compute_gap();

        assert!((ctx.gap_percent - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_gap_zero_bound() {
        let mut ctx = DisplayContext::new(1, 10);
        ctx.lower_bound = 0.0;
        ctx.forward_cost_stats = CostStatistics::from_costs(&[100.0]);
        ctx.compute_gap();

        assert!(ctx.gap_percent.is_infinite());
    }

    #[test]
    fn test_compute_trend_first_iteration() {
        let mut ctx = DisplayContext::new(1, 10);
        ctx.compute_trend(None);
        assert_eq!(ctx.gap_trend, GapTrend::Unknown);
    }

    #[test]
    fn test_compute_trend_improving() {
        let mut ctx = DisplayContext::new(2, 10);
        ctx.gap_percent = 5.0;
        ctx.compute_trend(Some(10.0)); // Gap decreased from 10% to 5%
        assert_eq!(ctx.gap_trend, GapTrend::Improving);
    }

    #[test]
    fn test_compute_trend_worsening() {
        let mut ctx = DisplayContext::new(2, 10);
        ctx.gap_percent = 10.0;
        ctx.compute_trend(Some(5.0)); // Gap increased from 5% to 10%
        assert_eq!(ctx.gap_trend, GapTrend::Worsening);
    }

    #[test]
    fn test_compute_trend_stable() {
        let mut ctx = DisplayContext::new(2, 10);
        ctx.gap_percent = 5.05;
        ctx.compute_trend(Some(5.0)); // Gap changed by 0.05% (within tolerance)
        assert_eq!(ctx.gap_trend, GapTrend::Stable);
    }

    #[test]
    fn test_target_progress_none() {
        let ctx = DisplayContext::new(1, 10);
        assert!(ctx.target_progress().is_none());
    }

    #[test]
    fn test_target_progress() {
        let mut ctx = DisplayContext::new(1, 10);
        ctx.target_gap = Some(10.0);
        ctx.gap_percent = 5.0;
        let progress = ctx.target_progress().unwrap();
        assert!((progress - 0.5).abs() < 1e-10); // Halfway to target
    }

    #[test]
    fn test_target_progress_met() {
        let mut ctx = DisplayContext::new(1, 10);
        ctx.target_gap = Some(10.0);
        ctx.gap_percent = 1.0;
        let progress = ctx.target_progress().unwrap();
        assert!((progress - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_target_progress_exceeded() {
        let mut ctx = DisplayContext::new(1, 10);
        ctx.target_gap = Some(10.0);
        ctx.gap_percent = 15.0; // Worse than target
        let progress = ctx.target_progress().unwrap();
        assert_eq!(progress, 0.0); // Clamped to 0
    }

    #[test]
    fn test_gap_trend_symbol() {
        assert_eq!(GapTrend::Improving.symbol(), "↓");
        assert_eq!(GapTrend::Worsening.symbol(), "↑");
        assert_eq!(GapTrend::Stable.symbol(), "→");
        assert_eq!(GapTrend::Unknown.symbol(), " ");
    }

    // IterationTracker tests
    #[test]
    fn test_iteration_tracker_new() {
        let tracker = IterationTracker::new();
        assert!(tracker.previous_lower_bound().is_none());
        assert!(tracker.previous_gap().is_none());
    }

    #[test]
    fn test_iteration_tracker_start() {
        let mut tracker = IterationTracker::new();
        tracker.start();
        assert!(tracker.elapsed() >= Duration::ZERO);
    }

    #[test]
    fn test_iteration_tracker_update() {
        let mut tracker = IterationTracker::new();
        tracker.start();

        assert!(tracker.previous_lower_bound().is_none());

        tracker.update(100000.0, 20.0);

        assert_eq!(tracker.previous_lower_bound(), Some(100000.0));
        assert_eq!(tracker.previous_gap(), Some(20.0));
    }

    #[test]
    fn test_iteration_tracker_multiple_updates() {
        let mut tracker = IterationTracker::new();
        tracker.start();

        tracker.update(100000.0, 20.0);
        tracker.update(110000.0, 15.0);

        assert_eq!(tracker.previous_lower_bound(), Some(110000.0));
        assert_eq!(tracker.previous_gap(), Some(15.0));
    }

    // DisplayContext::from_iteration tests
    fn create_mock_iteration_result(
        iteration: usize,
        lower_bound: f64,
        forward_costs: Vec<f64>,
    ) -> crate::sddp::IterationResult {
        use crate::timing::*;

        let num_forward = forward_costs.len();

        crate::sddp::IterationResult {
            iteration,
            lower_bound,
            forward_costs,
            timing: IterationTimingOutput {
                model_allocation: Duration::from_millis(10),
                forward: ForwardTimingOutput {
                    saa_sampling: Duration::from_millis(5),
                    model_preprocessing: Duration::from_millis(10),
                    solver: Duration::from_millis(100),
                    model_postprocessing: Duration::from_millis(5),
                    postprocessing: Duration::from_millis(5),
                    total: Duration::from_millis(125),
                    parallel_wall: Duration::from_millis(125),
                    parallel_overhead: Duration::ZERO,
                    solver_max: Duration::from_millis(100),
                    solver_calls: num_forward,
                },
                backward: BackwardTimingOutput {
                    model_preprocessing: Duration::from_millis(10),
                    solver: Duration::from_millis(200),
                    model_postprocessing: Duration::from_millis(5),
                    cut_selection: Duration::from_millis(10),
                    problem_update: Duration::from_millis(5),
                    total: Duration::from_millis(230),
                    solver_calls: 10,
                },
                model_cleanup: Duration::from_millis(5),
                total: Duration::from_millis(370),
                solver_calls: num_forward + 10,
            },
            num_cuts_added: 10,
            num_cuts_removed: 2,
            num_cuts_returned: 1,
            num_active_cuts: 100,
            first_stage_branching_costs: Vec::new(),
        }
    }

    #[test]
    fn test_from_iteration_basic() {
        let result =
            create_mock_iteration_result(1, 100000.0, vec![110000.0, 115000.0]);

        let ctx = DisplayContext::from_iteration(
            1,
            10,
            &result,
            None,
            Duration::from_secs(1),
            None,
        );

        assert_eq!(ctx.iteration, 1);
        assert_eq!(ctx.total_iterations, 10);
        assert_eq!(ctx.lower_bound, 100000.0);
        assert_eq!(ctx.forward_cost_stats.count, 2);
        assert!(ctx.gap_percent > 0.0);
        assert_eq!(ctx.gap_trend, GapTrend::Unknown); // First iteration
        assert_eq!(ctx.cuts_added, 10);
        assert_eq!(ctx.cuts_removed, 2);
    }

    #[test]
    fn test_from_iteration_with_previous() {
        let _result1 =
            create_mock_iteration_result(1, 100000.0, vec![150000.0]);
        let result2 = create_mock_iteration_result(2, 110000.0, vec![130000.0]);

        let ctx2 = DisplayContext::from_iteration(
            2,
            10,
            &result2,
            Some(100000.0),
            Duration::from_secs(2),
            None,
        );

        // Gap improved (from ~50% to ~18%)
        assert!(ctx2.gap_percent < 50.0);
        assert_eq!(ctx2.gap_trend, GapTrend::Improving);
    }

    #[test]
    fn test_from_iteration_with_target_gap() {
        let result = create_mock_iteration_result(1, 100000.0, vec![110000.0]);

        let ctx = DisplayContext::from_iteration(
            1,
            10,
            &result,
            None,
            Duration::from_secs(1),
            Some(20.0),
        );

        assert_eq!(ctx.target_gap, Some(20.0));
        let progress = ctx.target_progress().unwrap();
        assert!(progress > 0.0);
        assert!(progress < 1.0);
    }
}
