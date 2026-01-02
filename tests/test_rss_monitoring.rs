// Temporary test to monitor RSS during training
// Run with: cargo test --release test_rss_monitoring_example_05 -- --nocapture

use powers_rs::sddp::SddpInstanceBuilder;
use std::fs;

fn get_rss_kb() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse().unwrap_or(0);
                    }
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    0
}

#[test]
#[ignore]
fn test_rss_monitoring_example_05() {
    // Enable debug logging to see RSS at iteration boundaries
    std::env::set_var("RUST_LOG", "debug");
    let _ = env_logger::try_init();

    println!("\n=== RSS Monitoring Test for Example 05 ===\n");

    let rss_start = get_rss_kb();
    println!(
        "Initial RSS: {} KB ({:.1} MB)",
        rss_start,
        rss_start as f64 / 1024.0
    );

    // Build the SDDP instance
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/05-large-scale-brazilian/config.json",
        "examples/05-large-scale-brazilian/system.json",
        "examples/05-large-scale-brazilian/graph.json",
        "examples/05-large-scale-brazilian/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_num_iterations(20)
    .with_num_forward_passes(4)
    .with_num_threads(1) // Single thread to reduce memory
    .with_seed(42)
    .build()
    .expect("build should succeed");

    let rss_after_build = get_rss_kb();
    println!(
        "After build RSS: {} KB ({:.1} MB)",
        rss_after_build,
        rss_after_build as f64 / 1024.0
    );

    // Train and observe RSS growth
    let result = sddp.train().expect("Training should succeed");

    let rss_after_train = get_rss_kb();
    println!(
        "\nAfter training RSS: {} KB ({:.1} MB)",
        rss_after_train,
        rss_after_train as f64 / 1024.0
    );
    println!(
        "RSS growth during training: {} KB ({:.1} MB)",
        rss_after_train - rss_after_build,
        (rss_after_train - rss_after_build) as f64 / 1024.0
    );

    println!("\nTraining completed:");
    println!("  Lower bound: {:.2}", result.final_lower_bound);
    println!("  Upper bound: {:.2}", result.statistical_upper_bound);
}
