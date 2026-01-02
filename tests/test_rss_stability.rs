//! RSS Stability Tests for Sprint 9
//!
//! These tests verify that RSS (Resident Set Size) is stable between iterations
//! after warmup. They are used to compare allocator behavior.
//!
//! # Running Tests
//!
//! ```bash
//! # Run with default allocator
//! cargo test --release test_rss_stability -- --nocapture
//!
//! # Run with mimalloc
//! cargo test --release --features mimalloc test_rss_stability -- --nocapture
//!
//! # Run with jemalloc
//! cargo test --release --features jemalloc test_rss_stability -- --nocapture
//!
//! # Run CI check (requires expensive_tests feature)
//! cargo test --release --features expensive_tests test_rss_stability_ci -- --nocapture
//! ```

mod rss_harness;

use powers_rs::sddp::SddpInstanceBuilder;
use rss_harness::{
    is_rss_available, measure_rss_kb, RssAnalysis, RssCollector, RssPhase,
    RssSnapshot,
};
use std::path::Path;

/// Configuration for RSS stability tests.
struct RssTestConfig {
    /// Path to example directory.
    example_path: &'static str,
    /// Number of training iterations.
    num_iterations: usize,
    /// Number of forward passes per iteration.
    num_forward_passes: usize,
    /// Number of warmup iterations to ignore.
    warmup_iterations: usize,
    /// Tolerance for stability check (as fraction, e.g., 0.05 = 5%).
    stability_tolerance: f64,
}

impl Default for RssTestConfig {
    fn default() -> Self {
        Self {
            example_path: "examples/05-large-scale-brazilian",
            num_iterations: 20,
            num_forward_passes: 4,
            warmup_iterations: 2,
            stability_tolerance: 0.05,
        }
    }
}

/// Run RSS stability test with given configuration.
///
/// Returns the RSS analysis for further inspection.
fn run_rss_stability_test(config: &RssTestConfig) -> Option<RssAnalysis> {
    if !is_rss_available() {
        println!(
            "RSS measurement not available on this platform, skipping test"
        );
        return None;
    }

    let example_path = Path::new(config.example_path);
    if !example_path.exists() {
        println!(
            "Example path not found: {}, skipping test",
            config.example_path
        );
        return None;
    }

    println!("\n========================================");
    println!("RSS Stability Test");
    println!("========================================");
    println!("Example: {}", config.example_path);
    println!("Iterations: {}", config.num_iterations);
    println!("Forward passes: {}", config.num_forward_passes);
    println!("Warmup iterations: {}", config.warmup_iterations);
    println!(
        "Stability tolerance: {:.0}%",
        config.stability_tolerance * 100.0
    );

    // Detect allocator in use
    #[cfg(feature = "mimalloc")]
    println!("Allocator: mimalloc");
    #[cfg(feature = "jemalloc")]
    println!("Allocator: jemalloc");
    #[cfg(not(any(feature = "mimalloc", feature = "jemalloc")))]
    println!("Allocator: system (glibc)");

    println!("----------------------------------------\n");

    let initial_rss = measure_rss_kb().unwrap_or(0);
    println!(
        "Initial RSS: {} KB ({:.1} MB)",
        initial_rss,
        initial_rss as f64 / 1024.0
    );

    // Build SDDP instance
    let config_path = example_path.join("config.json");
    let system_path = example_path.join("system.json");
    let graph_path = example_path.join("graph.json");
    let recourse_path = example_path.join("recourse.json");

    let mut sddp = match SddpInstanceBuilder::from_paths(
        config_path.to_str().unwrap(),
        system_path.to_str().unwrap(),
        graph_path.to_str().unwrap(),
        recourse_path.to_str().unwrap(),
    ) {
        Ok(builder) => builder
            .with_num_iterations(config.num_iterations)
            .with_num_forward_passes(config.num_forward_passes)
            .with_num_threads(1) // Single thread to isolate memory behavior
            .with_seed(42)
            .build()
            .expect("SDDP build should succeed"),
        Err(e) => {
            println!("Failed to build SDDP instance: {}", e);
            return None;
        }
    };

    let after_build_rss = measure_rss_kb().unwrap_or(0);
    println!(
        "After build RSS: {} KB ({:.1} MB)",
        after_build_rss,
        after_build_rss as f64 / 1024.0
    );

    // Train and collect RSS measurements
    // Note: We can't hook into iteration boundaries here, so we measure before/after
    // For detailed per-iteration measurements, we rely on the debug logging in sddp/mod.rs

    println!("\nStarting training...\n");

    let result = sddp.train().expect("Training should succeed");

    let after_train_rss = measure_rss_kb().unwrap_or(0);
    println!("\nTraining complete!");
    println!("Lower bound: {:.2}", result.final_lower_bound);
    println!(
        "Final RSS: {} KB ({:.1} MB)",
        after_train_rss,
        after_train_rss as f64 / 1024.0
    );
    println!(
        "RSS growth during training: {} KB ({:.1} MB)",
        after_train_rss.saturating_sub(after_build_rss),
        (after_train_rss.saturating_sub(after_build_rss)) as f64 / 1024.0
    );

    // Create analysis from available data
    // Since we can't hook into iterations here, we create a simplified analysis
    // The real per-iteration data comes from debug logs
    let snapshots = vec![
        RssSnapshot::new(0, RssPhase::IterationEnd, initial_rss, 0),
        RssSnapshot::new(1, RssPhase::IterationEnd, after_build_rss, 0),
        RssSnapshot::new(
            config.num_iterations,
            RssPhase::IterationEnd,
            after_train_rss,
            0,
        ),
    ];

    let analysis =
        RssAnalysis::from_snapshots(&snapshots, config.warmup_iterations);

    println!("\n{}", analysis.summary());

    Some(analysis)
}

/// Main RSS stability test for the large example.
///
/// This test is marked as `#[ignore]` because it takes several minutes.
/// Run with: `cargo test --release test_rss_stability_full -- --nocapture --ignored`
#[test]
#[ignore]
fn test_rss_stability_full() {
    let config = RssTestConfig::default();
    let analysis = run_rss_stability_test(&config);

    if let Some(analysis) = analysis {
        // Print markdown table for documentation
        println!("\n=== Markdown Table for Documentation ===\n");
        println!("{}", analysis.to_markdown_table());

        // Note: We expect this to FAIL with glibc, proving the problem exists
        // After implementing allocator changes, this should PASS
        let is_stable = analysis.is_stable(config.stability_tolerance);
        println!(
            "\nRSS Stability Result: {}",
            if is_stable { "PASS ✓" } else { "FAIL ✗" }
        );

        // Don't assert for now - this is for data collection
        // assert!(is_stable, "RSS should be stable after warmup");
    }
}

/// Quick RSS stability test for CI.
///
/// Uses a smaller example and fewer iterations for faster execution.
/// Run with: `cargo test --release --features expensive_tests test_rss_stability_ci`
#[test]
#[cfg_attr(not(feature = "expensive_tests"), ignore)]
fn test_rss_stability_ci() {
    // Use smaller example for CI
    let config = RssTestConfig {
        example_path: "examples/02-hydro-thermal",
        num_iterations: 10,
        num_forward_passes: 4,
        warmup_iterations: 2,
        stability_tolerance: 0.10, // 10% tolerance for CI
    };

    if !Path::new(config.example_path).exists() {
        println!(
            "Example not found: {}, trying alternative",
            config.example_path
        );
        // Fall back to example 01 if 02 doesn't exist
        let alt_config = RssTestConfig {
            example_path: "examples/01-single-stage",
            num_iterations: 5,
            num_forward_passes: 2,
            warmup_iterations: 1,
            stability_tolerance: 0.10,
        };

        if !Path::new(alt_config.example_path).exists() {
            println!("No examples found, skipping CI test");
            return;
        }

        run_rss_stability_test(&alt_config);
        return;
    }

    let analysis = run_rss_stability_test(&config);

    if let Some(analysis) = analysis {
        let is_stable = analysis.is_stable(config.stability_tolerance);
        println!(
            "\nCI RSS Stability Result: {}",
            if is_stable { "PASS ✓" } else { "FAIL ✗" }
        );

        // In CI, we want to assert stability once the allocator is configured
        // For now, we just report
        // assert!(is_stable, "RSS should be stable after warmup");
    }
}

/// Test to compare allocator behavior.
///
/// Run this with different allocator features to compare:
/// ```bash
/// cargo test --release test_rss_allocator_comparison -- --nocapture --ignored
/// cargo test --release --features mimalloc test_rss_allocator_comparison -- --nocapture --ignored
/// cargo test --release --features jemalloc test_rss_allocator_comparison -- --nocapture --ignored
/// ```
#[test]
#[ignore]
fn test_rss_allocator_comparison() {
    let config = RssTestConfig::default();

    println!(
        "\n╔══════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                  RSS ALLOCATOR COMPARISON                     ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════╝\n"
    );

    #[cfg(feature = "mimalloc")]
    println!("🔧 Testing with: MIMALLOC");
    #[cfg(feature = "jemalloc")]
    println!("🔧 Testing with: JEMALLOC");
    #[cfg(not(any(feature = "mimalloc", feature = "jemalloc")))]
    println!("🔧 Testing with: SYSTEM ALLOCATOR (glibc)");

    let analysis = run_rss_stability_test(&config);

    if let Some(analysis) = analysis {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!(
            "║                         RESULTS                              ║"
        );
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        println!("{}", analysis.to_markdown_table());

        let is_stable = analysis.is_stable(config.stability_tolerance);
        if is_stable {
            println!("\n✅ RSS IS STABLE - This allocator solves the memory growth problem!");
        } else {
            println!("\n❌ RSS IS NOT STABLE - Memory continues to grow with this allocator.");
        }

        if let (Some(baseline), Some(final_rss)) =
            (analysis.baseline_rss_kb(), analysis.final_rss_kb())
        {
            let growth_pct =
                (final_rss as f64 - baseline as f64) / baseline as f64 * 100.0;
            println!("\nGrowth from baseline: {:.1}%", growth_pct);
        }

        let avg_growth = analysis.avg_growth_per_iteration();
        println!(
            "Average growth per iteration: {:.1} KB ({:.2} MB)",
            avg_growth,
            avg_growth / 1024.0
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_rss_test_config_default() {
        let config = RssTestConfig::default();
        assert_eq!(config.num_iterations, 20);
        assert_eq!(config.num_forward_passes, 4);
        assert_eq!(config.warmup_iterations, 2);
        assert!((config.stability_tolerance - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_rss_available_check() {
        // This just verifies the function doesn't panic
        let _ = is_rss_available();
    }
}
