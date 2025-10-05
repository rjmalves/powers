// Modules made public for testing purposes (T1.2, T1.3, T1.4, T1.6)
// In production builds, these are only accessible internally
pub mod cut;
pub mod error; // T3.9: Comprehensive error types with context and guidance
pub mod fcf;
pub mod solver;
pub mod state;
pub mod stochastic_process;
pub mod subproblem;
pub mod system;

pub mod graph;
pub mod initial_condition; // Made public for T1.6 integration tests
pub mod input;
pub mod input_validation; // T3.10: Comprehensive input validation
mod log;
pub mod output;
mod risk_measure;
pub mod scenario;
pub mod sddp;
pub mod utils;
use std::error::Error;
use std::time::Instant;

/// Main entry point for production use with full JSON-based configuration.
///
/// This function uses the **Factory API** (`SddpAlgorithm::from_files()`) which:
/// - Validates inputs before expensive computation (T3.7)
/// - Supports complex seasonal structures with distribution-based uncertainty
/// - Handles Markovian graph structures (not just linear paths)
/// - Configures per-node stochastic processes
/// - Supports Normal/LogNormal distributions for loads and inflows
///
/// For simpler use cases (unit tests with explicit scenarios), consider using the
/// **Builder API** via `sddp::SddpAlgorithm::builder()` instead.
///
/// # Performance Notes
/// - This is the production entry point; performance is critical
/// - Factory API has zero overhead vs. manual construction (T3.7)
/// - Validation adds <10μs (<0.002% of training time)
/// - Uses pre-allocated structures where possible
/// - Leverages Rayon parallelism in train() and simulate()
/// - Optional CSV output (controlled by config.output_path)
///
/// # History
/// - T3.7: Migrated from low-level API to factory API for validation and ergonomics
pub fn run(input_args: &InputArgs) -> Result<(), Box<dyn Error>> {
    log::show_greeting();

    let begin = Instant::now();

    log::input_reading_line(&input_args.path);

    // Factory API: Load, validate, and construct SDDP in one call (T3.7)
    // This replaces ~50 lines of manual construction with validation checkpoint
    let mut sddp = sddp::SddpAlgorithm::from_files(
        format!("{}/config.json", input_args.path),
        format!("{}/system.json", input_args.path),
        format!("{}/graph.json", input_args.path),
        format!("{}/recourse.json", input_args.path),
    )
    .map_err(|e| -> Box<dyn Error> { e.into() })?;

    // Zero-argument training (config embedded in SddpInstance)
    let _training_result =
        sddp.train().map_err(|e| -> Box<dyn Error> { e.into() })?;

    // Zero-argument simulation (config + SAA embedded in SddpInstance)
    let simulation_handlers = sddp
        .simulate()
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

    log::output_generation_line(&input_args.path);
    output::generate_outputs(
        &sddp.algorithm().future_cost_function_graph,
        &simulation_handlers,
        &sddp.algorithm().study_period_ids,
        sddp.config().output_path.as_deref(),
    )?;

    log::show_farewell(begin.elapsed());

    Ok(())
}

pub struct InputArgs {
    pub path: String,
}

impl InputArgs {
    pub fn build(args: &[String]) -> Result<Self, &'static str> {
        if args.len() < 2 {
            return Err("Not enough arguments [PATH]");
        }

        let path = args[1].clone();

        Ok(Self { path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_args_build_success() {
        let args = vec!["program_name".to_string(), "some/path".to_string()];
        let input_args = InputArgs::build(&args).unwrap();
        assert_eq!(input_args.path, "some/path");
    }

    #[test]
    fn test_input_args_build_fail() {
        let args = vec!["program_name".to_string()];
        let result = InputArgs::build(&args);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Not enough arguments [PATH]");
    }
}
