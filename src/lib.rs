pub mod base_noise;
pub mod cli;
pub mod correlation;
pub mod correlation_applicator;
pub mod cut;
pub mod error;
pub mod estimation;
pub mod fcf;
pub mod lognormal3;
pub mod marginal_transformer;
pub mod solver;
pub mod state;
pub mod stochastic_process;
pub mod subproblem;
pub mod system;

pub mod graph;
pub mod initial_condition;
pub mod input;
pub mod input_validation;
mod log;
pub mod noise_model_cache;
pub mod output;
pub mod par_generator;
mod risk_measure;
pub mod scenario;
pub mod sddp;
pub mod seasonal_params;
pub mod unified_noise_spec;
pub mod utils;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

/// Main entry point for SDDP algorithm execution (run subcommand).
///
/// This function uses the **Factory API** (`SddpAlgorithm::from_files()`)
///
/// For simpler use cases (unit tests with explicit scenarios), consider using the
/// **Builder API** via `sddp::SddpAlgorithm::builder()` instead.
///
pub fn run(input_path: &Path) -> Result<(), Box<dyn Error>> {
    log::show_greeting();

    let begin = Instant::now();

    let path_str = input_path.display().to_string();
    log::input_reading_line(&path_str);

    let mut sddp = sddp::SddpAlgorithm::from_files(
        input_path.join("config.json"),
        input_path.join("system.json"),
        input_path.join("graph.json"),
        input_path.join("recourse.json"),
    )
    .map_err(|e| -> Box<dyn Error> { e.into() })?;

    let _training_result =
        sddp.train().map_err(|e| -> Box<dyn Error> { e.into() })?;

    if sddp.config().num_simulation_scenarios.is_some() {
        let simulation_trajectories = sddp
            .simulate()
            .map_err(|e| -> Box<dyn Error> { e.into() })?;

        log::output_generation_line(&path_str);
        output::generate_outputs(
            &sddp.algorithm().future_cost_function_graph,
            &simulation_trajectories,
            sddp.config().output_path.as_deref(),
        )?;
    } else {
        log::simulation_skipped();
    }

    log::show_farewell(begin.elapsed());

    Ok(())
}

/// Entry point for PAR parameter estimation from historical data (estimate-par subcommand).
///
/// Reads CSV with historical time series, estimates PAR model parameters using
/// Yule-Walker method, and outputs JSON compatible with recourse.json format.
///
/// # Performance Notes
///
/// - Time complexity: O(T + n_periods·p³) where T = data length, p = AR order
/// - Space complexity: O(T + n_periods·p²)
/// - CSV reading is I/O bound; estimation is CPU bound
/// - Typical runtime: <10ms for 10 years monthly data with PAR(1)
/// - Uses nalgebra for efficient matrix operations (Cholesky decomposition)
///
/// # Arguments
///
/// - `input_path`: Path to CSV file with historical data
/// - `n_periods`: Number of seasonal periods (e.g., 12 for monthly)
/// - `ar_order`: Autoregressive order (e.g., 1 for PAR(1))
/// - `min_samples`: Minimum samples per period for reliable estimation
/// - `output_path`: Optional output JSON file (None = stdout)
/// - `has_header`: Whether CSV has header row
///
pub fn estimate_par(
    input_path: &Path,
    n_periods: usize,
    ar_order: usize,
    min_samples: usize,
    output_path: Option<&Path>,
    has_header: bool,
) -> Result<(), Box<dyn Error>> {
    use crate::estimation::{EstimationConfig, YuleWalkerEstimator};
    use std::fs::File;
    use std::io::Write;

    // Read CSV data
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_path(input_path)?;

    // Parse all records into a Vec<Vec<f64>>
    // Each inner Vec is one time step, outer Vec is the time series
    let mut all_data: Vec<Vec<f64>> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let values: Result<Vec<f64>, _> =
            record.iter().map(|s| s.parse::<f64>()).collect();
        all_data.push(
            values.map_err(|e| format!("Failed to parse CSV value: {}", e))?,
        );
    }

    if all_data.is_empty() {
        return Err("CSV file is empty or contains no valid data".into());
    }

    let n_entities = all_data[0].len();
    let n_samples = all_data.len();

    eprintln!("📊 Read {} samples for {} entities", n_samples, n_entities);
    eprintln!("   Estimating PAR({}) with {} periods", ar_order, n_periods);

    // Estimate parameters for each entity
    let config = EstimationConfig {
        n_periods,
        ar_order,
        min_samples_per_period: min_samples,
    };

    let estimator = YuleWalkerEstimator::new(config)
        .map_err(|e| format!("Invalid estimation configuration: {}", e))?;

    let mut estimated_params = Vec::with_capacity(n_entities);

    for entity_idx in 0..n_entities {
        // Extract time series for this entity
        let entity_data: Vec<f64> =
            all_data.iter().map(|row| row[entity_idx]).collect();

        eprintln!("   Entity {}: estimating...", entity_idx + 1);

        let params = estimator.estimate(&entity_data).map_err(|e| {
            format!("Estimation failed for entity {}: {}", entity_idx + 1, e)
        })?;

        estimated_params.push(params.to_json_fragment());
    }

    // Build output JSON
    let output_json = serde_json::json!({
        "noise_models": estimated_params,
        "metadata": {
            "estimation_method": "Yule-Walker",
            "n_entities": n_entities,
            "n_samples": n_samples,
            "n_periods": n_periods,
            "ar_order": ar_order,
            "min_samples_per_period": min_samples,
        }
    });

    // Write output
    match output_path {
        Some(path) => {
            let mut file = File::create(path)?;
            serde_json::to_writer_pretty(&mut file, &output_json)?;
            file.write_all(b"\n")?;
            eprintln!("✅ Parameters written to {}", path.display());
        }
        None => {
            // Print to stdout
            println!("{}", serde_json::to_string_pretty(&output_json)?);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_estimate_par_basic() {
        // Create temporary CSV file with synthetic data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "hydro_1,hydro_2").unwrap();
        for i in 0..48 {
            // 4 years of monthly data
            let month = i % 12;
            let seasonal_mean = 40.0 + (month as f64) * 2.0;
            writeln!(
                temp_file,
                "{},{}",
                seasonal_mean + 1.0,
                seasonal_mean - 1.0
            )
            .unwrap();
        }
        temp_file.flush().unwrap();

        // Estimate parameters
        let result = estimate_par(
            temp_file.path(),
            12,   // monthly
            1,    // PAR(1)
            3,    // min 3 samples per period
            None, // no output file
            true, // has header (default behavior)
        );

        assert!(result.is_ok(), "Estimation failed: {:?}", result.err());
    }
}
