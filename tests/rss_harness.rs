//! RSS (Resident Set Size) Measurement Harness for Memory Stability Testing
//!
//! This module provides infrastructure for measuring and analyzing RSS during
//! SDDP training to verify that memory is stable between iterations.
//!
//! # Background
//!
//! Sprint 8 revealed that per-iteration Model architecture is correctly
//! implemented, but RSS continues to grow with glibc malloc because it
//! doesn't return freed memory to the OS. This harness enables comparison
//! of different allocators (mimalloc, jemalloc, malloc_trim).
//!
//! # Usage
//!
//! ```ignore
//! use tests::rss_harness::{RssCollector, RssAnalysis};
//!
//! let mut collector = RssCollector::new(2); // 2 warmup iterations
//! collector.record_iteration_start(1);
//! // ... iteration 1 work ...
//! collector.record_iteration_end(1, 100); // 100 active cuts
//!
//! let analysis = collector.analyze();
//! assert!(analysis.is_stable(0.05)); // 5% tolerance
//! ```

use std::fs;
use std::time::Instant;

/// Phase within an iteration for RSS measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RssPhase {
    /// RSS measured at the start of an iteration (after previous finalize).
    IterationStart,
    /// RSS measured at the end of an iteration (after finalize_iteration).
    IterationEnd,
}

impl std::fmt::Display for RssPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RssPhase::IterationStart => write!(f, "start"),
            RssPhase::IterationEnd => write!(f, "end"),
        }
    }
}

/// A single RSS measurement snapshot.
#[derive(Debug, Clone)]
pub struct RssSnapshot {
    /// Iteration number (1-based).
    pub iteration: usize,
    /// Phase within the iteration.
    pub phase: RssPhase,
    /// RSS in kilobytes.
    pub rss_kb: u64,
    /// Number of active cuts at this point.
    pub active_cuts: usize,
    /// Timestamp for debugging.
    pub timestamp: Instant,
}

impl RssSnapshot {
    /// Create a new RSS snapshot.
    pub fn new(
        iteration: usize,
        phase: RssPhase,
        rss_kb: u64,
        active_cuts: usize,
    ) -> Self {
        Self {
            iteration,
            phase,
            rss_kb,
            active_cuts,
            timestamp: Instant::now(),
        }
    }
}

/// Collector for RSS measurements during training.
#[derive(Debug)]
pub struct RssCollector {
    /// All collected snapshots.
    snapshots: Vec<RssSnapshot>,
    /// Number of warmup iterations to ignore when checking stability.
    warmup_iterations: usize,
    /// Start time for relative timestamps.
    start_time: Instant,
}

impl RssCollector {
    /// Create a new RSS collector.
    ///
    /// # Arguments
    ///
    /// * `warmup_iterations` - Number of initial iterations to ignore when
    ///   checking stability (typically 2 for HiGHS initialization).
    pub fn new(warmup_iterations: usize) -> Self {
        Self {
            snapshots: Vec::with_capacity(100),
            warmup_iterations,
            start_time: Instant::now(),
        }
    }

    /// Record RSS at the start of an iteration.
    pub fn record_iteration_start(&mut self, iteration: usize) {
        if let Some(rss_kb) = measure_rss_kb() {
            self.snapshots.push(RssSnapshot::new(
                iteration,
                RssPhase::IterationStart,
                rss_kb,
                0, // Active cuts not known at start
            ));
        }
    }

    /// Record RSS at the end of an iteration.
    pub fn record_iteration_end(
        &mut self,
        iteration: usize,
        active_cuts: usize,
    ) {
        if let Some(rss_kb) = measure_rss_kb() {
            self.snapshots.push(RssSnapshot::new(
                iteration,
                RssPhase::IterationEnd,
                rss_kb,
                active_cuts,
            ));
        }
    }

    /// Get all collected snapshots.
    pub fn snapshots(&self) -> &[RssSnapshot] {
        &self.snapshots
    }

    /// Analyze the collected RSS data.
    pub fn analyze(&self) -> RssAnalysis {
        RssAnalysis::from_snapshots(&self.snapshots, self.warmup_iterations)
    }

    /// Get the number of warmup iterations.
    pub fn warmup_iterations(&self) -> usize {
        self.warmup_iterations
    }
}

/// Analysis of RSS measurements.
#[derive(Debug)]
pub struct RssAnalysis {
    /// Snapshots used for analysis.
    snapshots: Vec<RssSnapshot>,
    /// Number of warmup iterations.
    warmup_iterations: usize,
    /// RSS delta within each iteration (end - start).
    pub iteration_deltas: Vec<i64>,
    /// RSS delta between iterations (start[n+1] - end[n]).
    pub inter_iteration_deltas: Vec<i64>,
    /// RSS at iteration end for post-warmup iterations.
    pub post_warmup_rss: Vec<u64>,
}

impl RssAnalysis {
    /// Create analysis from snapshots.
    pub fn from_snapshots(
        snapshots: &[RssSnapshot],
        warmup_iterations: usize,
    ) -> Self {
        let mut iteration_deltas = Vec::new();
        let mut inter_iteration_deltas = Vec::new();
        let mut post_warmup_rss = Vec::new();

        // Compute iteration deltas (end - start for same iteration)
        let mut iter_starts: std::collections::HashMap<usize, u64> =
            std::collections::HashMap::new();

        for snapshot in snapshots {
            match snapshot.phase {
                RssPhase::IterationStart => {
                    iter_starts.insert(snapshot.iteration, snapshot.rss_kb);
                }
                RssPhase::IterationEnd => {
                    if let Some(&start_rss) =
                        iter_starts.get(&snapshot.iteration)
                    {
                        iteration_deltas
                            .push(snapshot.rss_kb as i64 - start_rss as i64);
                    }
                    if snapshot.iteration > warmup_iterations {
                        post_warmup_rss.push(snapshot.rss_kb);
                    }
                }
            }
        }

        // Compute inter-iteration deltas (start[n+1] - end[n])
        let ends: Vec<_> = snapshots
            .iter()
            .filter(|s| matches!(s.phase, RssPhase::IterationEnd))
            .collect();
        let starts: Vec<_> = snapshots
            .iter()
            .filter(|s| matches!(s.phase, RssPhase::IterationStart))
            .collect();

        for end in &ends {
            if let Some(next_start) =
                starts.iter().find(|s| s.iteration == end.iteration + 1)
            {
                inter_iteration_deltas
                    .push(next_start.rss_kb as i64 - end.rss_kb as i64);
            }
        }

        Self {
            snapshots: snapshots.to_vec(),
            warmup_iterations,
            iteration_deltas,
            inter_iteration_deltas,
            post_warmup_rss,
        }
    }

    /// Check if RSS is stable after warmup.
    ///
    /// Stability is defined as:
    /// 1. No monotonic growth after warmup
    /// 2. All post-warmup RSS values within tolerance of baseline
    ///
    /// # Arguments
    ///
    /// * `tolerance_pct` - Tolerance as a fraction (e.g., 0.05 for 5%)
    pub fn is_stable(&self, tolerance_pct: f64) -> bool {
        if self.post_warmup_rss.len() < 2 {
            return true; // Not enough data to determine
        }

        let baseline = self.post_warmup_rss[0] as f64;
        let tolerance = baseline * tolerance_pct;

        // Check that no RSS exceeds baseline + tolerance
        let all_within_tolerance = self
            .post_warmup_rss
            .iter()
            .all(|&rss| (rss as f64 - baseline).abs() <= tolerance);

        // Check for monotonic growth (more than 3 consecutive increases)
        let no_monotonic_growth = !self.has_monotonic_growth(3);

        all_within_tolerance && no_monotonic_growth
    }

    /// Check if there's monotonic RSS growth for `consecutive` iterations.
    fn has_monotonic_growth(&self, consecutive: usize) -> bool {
        if self.post_warmup_rss.len() < consecutive {
            return false;
        }

        let mut consecutive_increases = 0;
        for window in self.post_warmup_rss.windows(2) {
            if window[1] > window[0] {
                consecutive_increases += 1;
                if consecutive_increases >= consecutive - 1 {
                    return true;
                }
            } else {
                consecutive_increases = 0;
            }
        }
        false
    }

    /// Get the final RSS value (last iteration end).
    pub fn final_rss_kb(&self) -> Option<u64> {
        self.snapshots
            .iter()
            .filter(|s| matches!(s.phase, RssPhase::IterationEnd))
            .last()
            .map(|s| s.rss_kb)
    }

    /// Get baseline RSS (first post-warmup iteration end).
    pub fn baseline_rss_kb(&self) -> Option<u64> {
        self.post_warmup_rss.first().copied()
    }

    /// Get average RSS growth per iteration after warmup.
    pub fn avg_growth_per_iteration(&self) -> f64 {
        if self.post_warmup_rss.len() < 2 {
            return 0.0;
        }

        let first = self.post_warmup_rss[0] as f64;
        let last = *self.post_warmup_rss.last().unwrap() as f64;
        let num_iterations = (self.post_warmup_rss.len() - 1) as f64;

        (last - first) / num_iterations
    }

    /// Get maximum RSS observed.
    pub fn max_rss_kb(&self) -> Option<u64> {
        self.snapshots.iter().map(|s| s.rss_kb).max()
    }

    /// Generate a summary string for debugging/reporting.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== RSS Analysis Summary ===\n");
        s.push_str(&format!("Warmup iterations: {}\n", self.warmup_iterations));
        s.push_str(&format!("Total snapshots: {}\n", self.snapshots.len()));

        if let Some(baseline) = self.baseline_rss_kb() {
            s.push_str(&format!(
                "Baseline RSS (iter {}): {} KB ({:.1} MB)\n",
                self.warmup_iterations + 1,
                baseline,
                baseline as f64 / 1024.0
            ));
        }

        if let Some(final_rss) = self.final_rss_kb() {
            s.push_str(&format!(
                "Final RSS: {} KB ({:.1} MB)\n",
                final_rss,
                final_rss as f64 / 1024.0
            ));
        }

        if let Some(max_rss) = self.max_rss_kb() {
            s.push_str(&format!(
                "Max RSS: {} KB ({:.1} MB)\n",
                max_rss,
                max_rss as f64 / 1024.0
            ));
        }

        let avg_growth = self.avg_growth_per_iteration();
        s.push_str(&format!(
            "Avg growth per iteration: {:.1} KB ({:.2} MB)\n",
            avg_growth,
            avg_growth / 1024.0
        ));

        s.push_str(&format!(
            "Is stable (5% tolerance): {}\n",
            self.is_stable(0.05)
        ));

        s.push_str("\n--- Per-Iteration RSS ---\n");
        for snapshot in &self.snapshots {
            if matches!(snapshot.phase, RssPhase::IterationEnd) {
                s.push_str(&format!(
                    "  Iter {:2} end: {:7} KB ({:6.1} MB), cuts: {}\n",
                    snapshot.iteration,
                    snapshot.rss_kb,
                    snapshot.rss_kb as f64 / 1024.0,
                    snapshot.active_cuts
                ));
            }
        }

        s
    }

    /// Generate a markdown table for reporting.
    pub fn to_markdown_table(&self) -> String {
        let mut s = String::new();
        s.push_str("| Iteration | RSS Start (KB) | RSS End (KB) | Delta (KB) | Active Cuts |\n");
        s.push_str("|-----------|----------------|--------------|------------|-------------|\n");

        let starts: std::collections::HashMap<_, _> = self
            .snapshots
            .iter()
            .filter(|s| matches!(s.phase, RssPhase::IterationStart))
            .map(|s| (s.iteration, s.rss_kb))
            .collect();

        for snapshot in &self.snapshots {
            if matches!(snapshot.phase, RssPhase::IterationEnd) {
                let start_rss =
                    starts.get(&snapshot.iteration).copied().unwrap_or(0);
                let delta = snapshot.rss_kb as i64 - start_rss as i64;
                s.push_str(&format!(
                    "| {} | {} | {} | {:+} | {} |\n",
                    snapshot.iteration,
                    start_rss,
                    snapshot.rss_kb,
                    delta,
                    snapshot.active_cuts
                ));
            }
        }

        s
    }
}

/// Measure current RSS in kilobytes.
///
/// Returns `None` on non-Linux platforms or if measurement fails.
#[cfg(target_os = "linux")]
pub fn measure_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse().ok();
            }
        }
    }
    None
}

/// Measure current RSS in kilobytes (non-Linux stub).
#[cfg(not(target_os = "linux"))]
pub fn measure_rss_kb() -> Option<u64> {
    None
}

/// Check if RSS measurement is available on this platform.
pub fn is_rss_available() -> bool {
    measure_rss_kb().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_rss_returns_value_on_linux() {
        #[cfg(target_os = "linux")]
        {
            let rss = measure_rss_kb();
            assert!(rss.is_some(), "RSS measurement should work on Linux");
            assert!(rss.unwrap() > 0, "RSS should be positive");
        }

        #[cfg(not(target_os = "linux"))]
        {
            let rss = measure_rss_kb();
            assert!(
                rss.is_none(),
                "RSS measurement should be None on non-Linux"
            );
        }
    }

    #[test]
    fn test_rss_analysis_is_stable_for_flat_data() {
        let snapshots = vec![
            RssSnapshot::new(1, RssPhase::IterationEnd, 100_000, 10),
            RssSnapshot::new(2, RssPhase::IterationEnd, 102_000, 20),
            RssSnapshot::new(3, RssPhase::IterationEnd, 101_000, 30), // baseline
            RssSnapshot::new(4, RssPhase::IterationEnd, 102_000, 40),
            RssSnapshot::new(5, RssPhase::IterationEnd, 100_500, 50),
            RssSnapshot::new(6, RssPhase::IterationEnd, 101_500, 60),
        ];

        let analysis = RssAnalysis::from_snapshots(&snapshots, 2);
        assert!(
            analysis.is_stable(0.05),
            "Flat data should be considered stable"
        );
    }

    #[test]
    fn test_rss_analysis_detects_monotonic_growth() {
        let snapshots = vec![
            RssSnapshot::new(1, RssPhase::IterationEnd, 100_000, 10),
            RssSnapshot::new(2, RssPhase::IterationEnd, 110_000, 20),
            RssSnapshot::new(3, RssPhase::IterationEnd, 120_000, 30), // baseline
            RssSnapshot::new(4, RssPhase::IterationEnd, 130_000, 40),
            RssSnapshot::new(5, RssPhase::IterationEnd, 140_000, 50),
            RssSnapshot::new(6, RssPhase::IterationEnd, 150_000, 60),
        ];

        let analysis = RssAnalysis::from_snapshots(&snapshots, 2);
        assert!(
            !analysis.is_stable(0.05),
            "Monotonic growth should not be considered stable"
        );
    }

    #[test]
    fn test_rss_collector_records_snapshots() {
        let mut collector = RssCollector::new(2);

        // These calls may or may not record depending on platform
        collector.record_iteration_start(1);
        collector.record_iteration_end(1, 10);
        collector.record_iteration_start(2);
        collector.record_iteration_end(2, 20);

        #[cfg(target_os = "linux")]
        {
            assert!(
                collector.snapshots().len() >= 4,
                "Should have recorded snapshots on Linux"
            );
        }
    }

    #[test]
    fn test_rss_analysis_summary_format() {
        let snapshots = vec![
            RssSnapshot::new(1, RssPhase::IterationStart, 50_000, 0),
            RssSnapshot::new(1, RssPhase::IterationEnd, 100_000, 10),
            RssSnapshot::new(2, RssPhase::IterationStart, 100_000, 0),
            RssSnapshot::new(2, RssPhase::IterationEnd, 102_000, 20),
            RssSnapshot::new(3, RssPhase::IterationStart, 102_000, 0),
            RssSnapshot::new(3, RssPhase::IterationEnd, 101_000, 30),
        ];

        let analysis = RssAnalysis::from_snapshots(&snapshots, 2);
        let summary = analysis.summary();

        assert!(summary.contains("RSS Analysis Summary"));
        assert!(summary.contains("Warmup iterations: 2"));
        assert!(summary.contains("Is stable"));
    }

    #[test]
    fn test_rss_analysis_markdown_table() {
        let snapshots = vec![
            RssSnapshot::new(1, RssPhase::IterationStart, 50_000, 0),
            RssSnapshot::new(1, RssPhase::IterationEnd, 100_000, 10),
            RssSnapshot::new(2, RssPhase::IterationStart, 100_000, 0),
            RssSnapshot::new(2, RssPhase::IterationEnd, 102_000, 20),
        ];

        let analysis = RssAnalysis::from_snapshots(&snapshots, 1);
        let table = analysis.to_markdown_table();

        assert!(table.contains("| Iteration |"));
        assert!(table.contains("| 1 |"));
        assert!(table.contains("| 2 |"));
    }
}
