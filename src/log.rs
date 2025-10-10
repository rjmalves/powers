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
    backward_fcf_state_update_time: Duration,
    backward_cut_cloning_time: Duration,
    backward_handler_application_time: Duration,
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
        "─".repeat(48)
    );
    println!(
        "  │  Backward Prep:{:>7.3}s  │  Model Prep: {:>7.3}s  │  Solver:   {:>7.3}s │",
        backward_pre_time.as_secs_f64(),
        backward_model_pre_time.as_secs_f64(),
        backward_solver_time.as_secs_f64()
    );
    println!(
        "  │  Model Post:   {:>7.3}s  │  Cut Select: {:>7.3}s  │                     │",
        backward_model_post_time.as_secs_f64(),
        backward_cutsel_time.as_secs_f64()
    );
    println!(
        "  │  FCF State:    {:>7.3}s  │  Cut Clone:  {:>7.3}s  │  Model Upd:{:>7.3}s │",
        backward_fcf_state_update_time.as_secs_f64(),
        backward_cut_cloning_time.as_secs_f64(),
        backward_handler_application_time.as_secs_f64()
    );
    println!("  └{}┘", "─".repeat(73));

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
