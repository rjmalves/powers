use std::time::Duration;

pub fn show_greeting() {
    println!(
        "\nPOWE.RS - Power Optimization for the World of Energy - in pure RuSt"
    );
    println!(
        "--------------------------------------------------------------------"
    );
}

pub fn input_reading_line(input_path: &str) {
    println!("\nReading input files from '{}'", input_path);
}

pub fn output_generation_line(input_path: &str) {
    println!("\nWriting outputs to '{}'", input_path);
}

pub fn show_farewell(time: Duration) {
    println!(
        "\nTotal running time: {:.2} s",
        time.as_millis() as f64 / 1000.0
    )
}

/// Helper function for displaying the greeting data for the training
pub fn training_greeting(num_iterations: usize, num_forward_passes: usize) {
    println!("\n# Training");
    println!("- Iterations: {num_iterations}");
    println!("- Forward passes: {num_forward_passes}\n");
}

/// Helper function for displaying the training table header
///
/// Enhanced table shows:
/// - Iteration number
/// - Lower bound (monotonically increasing)
/// - Current upper bound (simulation value)
/// - Relative gap percentage
/// - Forward pass time
/// - Backward pass time
/// - Total iteration time
pub fn training_table_header() {
    println!(
        "{0: >4} | {1: >11} | {2: >11} | {3: >8} | {4: >8} | {5: >8} | {6: >8}",
        "iter",
        "lower ($)",
        "simul ($)",
        "gap (%)",
        "fwd (s)",
        "bwd (s)",
        "total (s)"
    );
}

/// Helper function for displaying a divider for the training table
pub fn training_table_divider() {
    println!("{}", "-".repeat(80))
}

pub fn training_duration(time: Duration) {
    println!("\nTraining time: {:.2} s", time.as_millis() as f64 / 1000.0)
}
pub fn policy_size(num_cuts: usize) {
    println!("\nNumber of constructed cuts by node: {}", num_cuts)
}

/// Display enhanced iteration results in training table.
///
/// Shows comprehensive convergence metrics:
/// - Lower bound (monotonically increasing)
/// - Current upper bound (simulation value)
/// - Relative gap
/// - Timing breakdown (forward/backward/total)
///
/// # Arguments
///
/// * `iteration` - Iteration number
/// * `lower_bound` - Estimated lower bound (monotonically increasing)
/// * `simulation` - Current simulation value
/// * `relative_gap` - Relative gap percentage (gap / |lower|)
/// * `forward_time` - Forward pass time
/// * `backward_time` - Backward pass time
/// * `total_time` - Total iteration time
#[allow(clippy::too_many_arguments)]
pub fn training_table_row(
    iteration: usize,
    lower_bound: f64,
    upper_bound: f64,
    relative_gap: f64,
    forward_time: Duration,
    backward_time: Duration,
    total_time: Duration,
) {
    // Format relative gap as percentage with finite check
    let gap_pct_str = if relative_gap.is_finite() {
        format!("{:>8.2}", relative_gap * 100.0)
    } else {
        "      --".to_string()
    };

    println!(
        "{0: >4} | {1: >11.2} | {2: >11.2} | {3} | {4: >8.3} | {5: >8.3} | {6: >8.3}",
        iteration,
        lower_bound,
        upper_bound,
        gap_pct_str,
        forward_time.as_secs_f64(),
        backward_time.as_secs_f64(),
        total_time.as_secs_f64()
    );
}

/// Display detailed timing breakdown for an iteration.
///
/// Shows absolute timing values (in seconds with 3 decimal places) for each
/// phase of the forward and backward passes. This enables precise performance
/// diagnosis and regression detection.
///
/// **Flow-based timing categories** (T4.1 Phase 3.5 Refactoring):
/// - Forward: SAA sampling → Model prep → Solver → Model post → Aggregation
/// - Backward: Backward prep → Model prep → Solver → Model post → Cut selection → FCF update
///
/// # Arguments
///
/// * `forward_time` - Total forward pass time
/// * `forward_saa_time` - SAA sampling time (single-threaded)
/// * `forward_model_pre_time` - Model preprocessing time (multi-threaded average)
/// * `forward_solver_time` - Solver time (multi-threaded average)
/// * `forward_model_post_time` - Model postprocessing time (multi-threaded average)
/// * `forward_post_time` - Forward aggregation time (single-threaded)
/// * `backward_time` - Total backward pass time
/// * `backward_pre_time` - Backward preprocessing time (single-threaded)
/// * `backward_model_pre_time` - Model preprocessing time (multi-threaded average)
/// * `backward_solver_time` - Solver time (multi-threaded average)
/// * `backward_model_post_time` - Model postprocessing time (multi-threaded average)
/// * `backward_cutsel_time` - Cut selection time (single-threaded)
/// * `backward_fcf_time` - FCF update time (single-threaded)
/// * `solver_calls` - Total number of solver calls
/// * `cuts_added` - Number of cuts added this iteration
/// * `cuts_removed` - Number of dominated cuts removed this iteration
/// * `cuts_returned` - Number of inactive cuts returned this iteration
/// * `active_cuts` - Total number of active cuts after this iteration
///
/// # Example Output
///
/// ```text
///   ┌─ Forward Pass (2.345s) ──────────────────────────────────────────────┐
///   │  SAA Sampling:    0.001s  │  Model Prep:  0.023s  │  Solver:  1.780s │
///   │  Model Post:      0.012s  │  Aggregation: 0.001s  │                  │
///   └──────────────────────────────────────────────────────────────────────┘
///   ┌─ Backward Pass (1.234s) ─────────────────────────────────────────────┐
///   │  Backward Prep:   0.003s  │  Model Prep:  0.010s  │  Solver:  0.540s │
///   │  Model Post:      0.012s  │  Cut Select:  0.002s  │  FCF Upd: 0.247s │
///   └──────────────────────────────────────────────────────────────────────┘
///   Solvers: 488 calls | Cuts: +44 new, -12 dominated, +3 returned, 156 active
/// ```
#[allow(clippy::too_many_arguments)]
pub fn training_iteration_timing(
    forward_time: Duration,
    forward_saa_time: Duration,
    forward_model_pre_time: Duration,
    forward_solver_time: Duration,
    forward_model_post_time: Duration,
    forward_post_time: Duration,
    backward_time: Duration,
    backward_pre_time: Duration,
    backward_model_pre_time: Duration,
    backward_solver_time: Duration,
    backward_model_post_time: Duration,
    backward_cutsel_time: Duration,
    backward_fcf_time: Duration,
    solver_calls: usize,
    cuts_added: usize,
    cuts_removed: usize,
    cuts_returned: usize,
    active_cuts: usize,
) {
    // Professional box-drawing characters for visual hierarchy
    println!(
        "  ┌─ Forward Pass ({:.3}s) {}┐",
        forward_time.as_secs_f64(),
        "─".repeat(48)
    );
    println!(
        "  │  SAA Sampling: {:>7.3}s  │  Model Prep: {:>7.3}s  │  Solver: {:>7.3}s  │",
        forward_saa_time.as_secs_f64(),
        forward_model_pre_time.as_secs_f64(),
        forward_solver_time.as_secs_f64()
    );
    println!(
        "  │  Model Post:   {:>7.3}s  │  Aggregation:{:>7.3}s  │                    │",
        forward_model_post_time.as_secs_f64(),
        forward_post_time.as_secs_f64()
    );
    println!("  └{}┘", "─".repeat(72));

    println!(
        "  ┌─ Backward Pass ({:.3}s) {}┐",
        backward_time.as_secs_f64(),
        "─".repeat(47)
    );
    println!(
        "  │  Backward Prep:{:>7.3}s  │  Model Prep: {:>7.3}s  │  Solver: {:>7.3}s  │",
        backward_pre_time.as_secs_f64(),
        backward_model_pre_time.as_secs_f64(),
        backward_solver_time.as_secs_f64()
    );
    println!(
        "  │  Model Post:   {:>7.3}s  │  Cut Select: {:>7.3}s  │  FCF Upd:{:>7.3}s  │",
        backward_model_post_time.as_secs_f64(),
        backward_cutsel_time.as_secs_f64(),
        backward_fcf_time.as_secs_f64()
    );
    println!("  └{}┘", "─".repeat(72));

    // Cut selection statistics
    println!(
        "  Solver: {} calls | Cuts: +{} new, -{} dominated, +{} returned, {} active\n",
        solver_calls, cuts_added, cuts_removed, cuts_returned, active_cuts
    );
}

/// Helper function for displaying the greeting data for the simulation
pub fn simulation_greeting(num_simulation_scenarios: usize) {
    println!("\n# Simulating");
    println!("- Scenarios: {num_simulation_scenarios}\n");
}

pub fn simulation_stats(mean: f64, std: f64) {
    println!("Expected cost ($): {:.2} +- {:.2}", mean, std);
}

pub fn simulation_duration(time: Duration) {
    println!(
        "\nSimulation time: {:.2} s",
        time.as_millis() as f64 / 1000.0
    )
}
