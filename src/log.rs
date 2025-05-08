use std::time::Duration;

/// Helper function for displaying the greeting data for the training
pub fn training_greeting(num_iterations: usize, num_stages: usize) {
    println!("\n# Training");
    println!("- Iterations: {num_iterations}");
    println!("- Stages: {num_stages}");
}

/// Helper function for displaying the training table header
pub fn training_table_header() {
    println!(
        "{0: ^10} | {1: ^15} | {2: ^14} | {3: ^12}",
        "iteration", "lower bound ($)", "simulation ($)", "time (s)"
    )
}

/// Helper function for displaying a divider for the training table
pub fn training_table_divider() {
    println!("------------------------------------------------------------")
}

pub fn training_duration(time: Duration) {
    println!("\nTraining time: {:.2} s", time.as_millis() as f64 / 1000.0)
}

/// Helper function for displaying a row of iteration results for
/// the training table
pub fn training_table_row(
    iteration: usize,
    lower_bound: f64,
    simulation: f64,
    time: Duration,
) {
    println!(
        "{0: >10} | {1: >15.4} | {2: >14.4} | {3: >12.2}",
        iteration,
        lower_bound,
        simulation,
        time.as_millis() as f64 / 1000.0
    )
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
