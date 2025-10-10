use std::time::Duration;

/// Format duration as HH:MM:SS.SSS
#[inline]
fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = duration.subsec_millis();

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Format cost in scientific notation with appropriate precision
#[inline]
fn format_cost(cost: f64) -> String {
    format!("{:.6e}", cost)
}

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
    println!("\nTotal running time: {}", format_duration(time))
}

/// Helper function for displaying the greeting data for the training
pub fn training_greeting(num_iterations: usize, num_forward_passes: usize) {
    println!("\n# Training");
    println!("- Iterations: {num_iterations}");
    println!("- Forward passes: {num_forward_passes}\n");
}

/// Helper function for displaying the training table header
pub fn training_table_header() {
    println!(
        "{0: >4} | {1: >14} | {2: >14} | {3: >12} | {4: >12} | {5: >12}",
        "iter", "lower ($)", "simul ($)", "fwd", "bwd", "total"
    );
}

/// Helper function for displaying a divider for the training table
pub fn training_table_divider() {
    println!("{}", "-".repeat(88))
}

pub fn training_duration(time: Duration) {
    println!("\nTraining time: {}", format_duration(time))
}
pub fn policy_size(num_cuts: usize) {
    println!("\nNumber of constructed cuts by node: {}", num_cuts)
}

/// Display enhanced iteration results in training table.
pub fn training_table_row(
    iteration: usize,
    lower_bound: f64,
    simulation_cost: f64,
    forward_time: Duration,
    backward_time: Duration,
    total_time: Duration,
) {
    println!(
        "{0: >4} | {1: >14} | {2: >14} | {3: >12} | {4: >12} | {5: >12}",
        iteration,
        format_cost(lower_bound),
        format_cost(simulation_cost),
        format_duration(forward_time),
        format_duration(backward_time),
        format_duration(total_time)
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
        "  ┌─ Forward Pass ({}) {}┐",
        format_duration(forward_time),
        "─".repeat(48)
    );
    println!(
        "  │  SAA Sampling: {:>12}  │  Model Prep: {:>12}  │  Solver: {:>12}  │",
        format_duration(forward_saa_time),
        format_duration(forward_model_pre_time),
        format_duration(forward_solver_time)
    );
    println!(
        "  │  Model Post:   {:>12}  │  Aggregation:{:>12}  │                    │",
        format_duration(forward_model_post_time),
        format_duration(forward_post_time)
    );
    println!("  └{}┘", "─".repeat(72));

    println!(
        "  ┌─ Backward Pass ({}) {}┐",
        format_duration(backward_time),
        "─".repeat(48)
    );
    println!(
        "  │  Backward Prep:{:>12}  │  Model Prep: {:>12}  │  Solver:   {:>12} │",
        format_duration(backward_pre_time),
        format_duration(backward_model_pre_time),
        format_duration(backward_solver_time)
    );
    println!(
        "  │  Model Post:   {:>12}  │  Cut Select: {:>12}  │                     │",
        format_duration(backward_model_post_time),
        format_duration(backward_cutsel_time)
    );
    println!(
        "  │  FCF State:    {:>12}  │  Cut Clone:  {:>12}  │  Model Upd:{:>12} │",
        format_duration(backward_fcf_state_update_time),
        format_duration(backward_cut_cloning_time),
        format_duration(backward_handler_application_time)
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
    println!(
        "Expected cost ($): {} ± {}",
        format_cost(mean),
        format_cost(std)
    );
}

pub fn simulation_duration(time: Duration) {
    println!("\nSimulation time: {}", format_duration(time))
}

/// Helper function for displaying greeting for final simulation
pub fn final_simulation_greeting(num_scenarios: usize) {
    println!("\n# Final Simulation (evaluating trained policy)");
    println!("- Scenarios: {}\n", num_scenarios);
}

/// Helper function for displaying final simulation statistics
pub fn final_simulation_stats(mean: f64, std: f64) {
    println!(
        "Final policy cost: {} ± {}",
        format_cost(mean),
        format_cost(std)
    );
}
