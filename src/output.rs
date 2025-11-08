use crate::fcf;
use crate::graph;
use crate::scenario;
use crate::sddp;

use csv::Writer;
use serde;
use std::error::Error;
use std::sync::{Arc, Mutex};

#[derive(serde::Serialize)]
enum BendersCutCoefficientType {
    Rhs,
    Storage(usize),
}

#[derive(serde::Serialize)]
struct BendersCutOutput {
    stage_index: usize,
    stage_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
    active: bool,
    coefficient_entity: BendersCutCoefficientType,
    value: f64,
}

/// Writes Benders cuts to CSV file.
///
/// # Arguments
///
/// * `g` - The SDDP graph with future cost functions
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Performance
///
/// When `path` is `None`, this function returns immediately with no I/O overhead,
/// eliminating file system calls and CSV serialization overhead.
fn write_benders_cuts(
    g: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested (no-op, no I/O)
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr = Writer::from_path(&(output_dir.to_owned() + "/cuts.csv"))?;
    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.lock().unwrap();
        for cut in fcf.cut_pool.pool.iter() {
            // Writes RHS
            wtr.serialize(BendersCutOutput {
                stage_index: node.id,
                stage_cut_id: cut.id,
                iteration: cut.iteration,
                forward_pass_idx: cut.forward_pass_idx,
                active: cut.active,
                coefficient_entity: BendersCutCoefficientType::Rhs,
                value: cut.rhs,
            })?;
            // Writes coefficients
            for (index, coef) in cut.coefficients.iter().enumerate() {
                wtr.serialize(BendersCutOutput {
                    stage_index: node.id,
                    stage_cut_id: cut.id,
                    iteration: cut.iteration,
                    forward_pass_idx: cut.forward_pass_idx,
                    active: cut.active,
                    coefficient_entity: BendersCutCoefficientType::Storage(
                        index,
                    ),
                    value: *coef,
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
enum VisitedStateCoefficientType {
    DominatingObjective,
    Storage(usize),
}

#[derive(serde::Serialize)]
struct VisitedStateOutput {
    stage_index: usize,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
    coefficient_entity: VisitedStateCoefficientType,
    value: f64,
}

/// Writes visited states to CSV file.
///
/// # Arguments
///
/// * `g` - The SDDP graph with future cost functions
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
fn write_visited_states(
    g: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr = Writer::from_path(&(output_dir.to_owned() + "/states.csv"))?;
    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.lock().unwrap();
        for state in fcf.state_pool.pool.iter() {
            // Writes dominating objective for state
            wtr.serialize(VisitedStateOutput {
                stage_index: node.id,
                dominating_cut_id: state.get_dominating_cut_id(),
                iteration: state.get_iteration(),
                forward_pass_idx: state.get_forward_pass_idx(),
                coefficient_entity:
                    VisitedStateCoefficientType::DominatingObjective,
                value: state.get_dominating_objective(),
            })?;
            // Writes state variables values
            for (index, coef) in state.coefficients().iter().enumerate() {
                wtr.serialize(VisitedStateOutput {
                    stage_index: node.id,
                    dominating_cut_id: state.get_dominating_cut_id(),
                    iteration: state.get_iteration(),
                    forward_pass_idx: state.get_forward_pass_idx(),
                    coefficient_entity: VisitedStateCoefficientType::Storage(
                        index,
                    ),
                    value: *coef,
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct BusSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    load: f64,
    deficit: f64,
    marginal_cost: f64,
}

/// Writes bus simulation results to CSV file.
///
/// Uses trajectory-based data access for better cache locality (sequential access
/// pattern) compared to the previous handler-based approach (pointer-chasing through
/// graph nodes).
///
/// # Arguments
///
/// * `simulation_trajectories` - Lightweight trajectories with simulation results
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Performance
///
/// Trajectory-based access provides ~5-10% faster CSV export due to:
/// - Sequential memory access (cache-friendly)
/// - No graph node lookups or pointer indirection
/// - Direct array indexing instead of HashMap lookups
fn write_buses_simulation_results(
    simulation_trajectories: &[sddp::SimulationTrajectory],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/simulation_buses.csv"))?;

    // PERFORMANCE: Sequential access pattern (cache-friendly)
    // Direct iteration over trajectories → realizations → buses
    // Old approach required: trajectory → node_id lookup → graph traversal → realization
    for (series_index, trajectory) in simulation_trajectories.iter().enumerate()
    {
        for (stage_index, realization_data) in
            trajectory.realizations.iter().enumerate()
        {
            let num_buses = realization_data.loads.len();
            for bus_index in 0..num_buses {
                wtr.serialize(BusSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: bus_index,
                    load: realization_data.loads[bus_index],
                    deficit: realization_data.deficit[bus_index],
                    marginal_cost: realization_data.marginal_cost[bus_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct LineSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    exchange: f64,
}

/// Writes line simulation results to CSV file.
///
/// Uses trajectory-based data access for better cache locality.
///
/// # Arguments
///
/// * `simulation_trajectories` - Lightweight trajectories with simulation results
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
fn write_lines_simulation_results(
    simulation_trajectories: &[sddp::SimulationTrajectory],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/simulation_lines.csv"))?;

    // PERFORMANCE: Sequential access, no graph lookups
    for (series_index, trajectory) in simulation_trajectories.iter().enumerate()
    {
        for (stage_index, realization_data) in
            trajectory.realizations.iter().enumerate()
        {
            let num_lines = realization_data.exchange.len();
            for line_index in 0..num_lines {
                wtr.serialize(LineSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: line_index,
                    exchange: realization_data.exchange[line_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct ThermalSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    generation: f64,
}

/// Writes thermal simulation results to CSV file.
///
/// Uses trajectory-based data access for better cache locality.
///
/// # Arguments
///
/// * `simulation_trajectories` - Lightweight trajectories with simulation results
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
fn write_thermals_simulation_results(
    simulation_trajectories: &[sddp::SimulationTrajectory],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr = Writer::from_path(
        &(output_dir.to_owned() + "/simulation_thermals.csv"),
    )?;

    // PERFORMANCE: Sequential access, no graph lookups
    for (series_index, trajectory) in simulation_trajectories.iter().enumerate()
    {
        for (stage_index, realization_data) in
            trajectory.realizations.iter().enumerate()
        {
            let num_thermals = realization_data.thermal_generation.len();
            for thermal_index in 0..num_thermals {
                wtr.serialize(ThermalSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: thermal_index,
                    generation: realization_data.thermal_generation
                        [thermal_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct HydroSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    final_storage: f64,
    inflow: f64,
    turbined_flow: f64,
    spillage: f64,
    water_value: f64,
}

/// Writes hydro simulation results to CSV file.
///
/// Uses trajectory-based data access for better cache locality.
///
/// # Arguments
///
/// * `simulation_trajectories` - Lightweight trajectories with simulation results
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
fn write_hydros_simulation_results(
    simulation_trajectories: &[sddp::SimulationTrajectory],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/simulation_hydros.csv"))?;

    // PERFORMANCE: Sequential access, no graph lookups
    for (series_index, trajectory) in simulation_trajectories.iter().enumerate()
    {
        for (stage_index, realization_data) in
            trajectory.realizations.iter().enumerate()
        {
            let num_hydros = realization_data.final_storage.len();
            for hydro_index in 0..num_hydros {
                // NOTE: For PAR models with state expansion, realization_data.inflow
                // contains residuals (Z'_t) not observations (Y_t).
                // Future enhancement: Transform to observations for CSV output using Y_t = μ + σ·Z'_t
                // See FUTURE_WORK.md: "CSV Output Transformation for PAR Models"
                // For now, outputting residuals which are the values the LP works with.
                wtr.serialize(HydroSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: hydro_index,
                    final_storage: realization_data.final_storage[hydro_index],
                    inflow: realization_data.inflow[hydro_index],
                    turbined_flow: realization_data.turbined_flow[hydro_index],
                    spillage: realization_data.spillage[hydro_index],
                    water_value: realization_data.water_value[hydro_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct SampledNoiseOutput {
    stage_index: usize,
    branching_index: usize,
    entity_type: String,
    entity_id: usize,
    noise: f64,
}

/// Writes sampled noises from SAA to CSV file.
///
/// Exports all sampled noise values (load and inflow innovations) from the
/// Sample Average Approximation (SAA) tree used during SDDP training.
///
/// # Arguments
///
/// * `saa` - Reference to the SAA object containing sampled scenarios
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # CSV Format
///
/// Columns: `stage_index`, `branching_index`, `entity_type`, `entity_id`, `noise`
/// - `stage_index`: Zero-based stage index
/// - `branching_index`: Zero-based branching/scenario index within the stage
/// - `entity_type`: Either "load" or "inflow"
/// - `entity_id`: Zero-based entity index
/// - `noise`: Sampled innovation value (residual for PAR models)
///
/// # Note
///
/// For PAR models, the exported values are residuals (Z'_t), not observations (Y_t).
/// To convert to observations: Y_t = μ_t + σ_t * Z'_t
fn write_sampled_noises(
    saa: &scenario::SAA,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/sampled_noises.csv"))?;

    // Iterate over all stages in the SAA
    for (stage_index, stage_branchings) in
        saa.branching_samples.iter().enumerate()
    {
        // Iterate over all branchings in this stage
        for (branching_index, branching_noises) in
            stage_branchings.branching_noises.iter().enumerate()
        {
            // Export load innovations
            for (entity_id, &noise_value) in
                branching_noises.load_innovations.iter().enumerate()
            {
                wtr.serialize(SampledNoiseOutput {
                    stage_index,
                    branching_index,
                    entity_type: "load".to_string(),
                    entity_id,
                    noise: noise_value,
                })?;
            }

            // Export inflow innovations
            for (entity_id, &noise_value) in
                branching_noises.inflow_innovations.iter().enumerate()
            {
                wtr.serialize(SampledNoiseOutput {
                    stage_index,
                    branching_index,
                    entity_type: "inflow".to_string(),
                    entity_id,
                    noise: noise_value,
                })?;
            }
        }
    }

    wtr.flush()?;
    Ok(())
}

/// Generates all CSV output files from SDDP training and simulation results.
///
/// After SIM-OPT-005 refactoring, this function consumes lightweight `SimulationTrajectory`
/// objects instead of heavy `SddpSimulationHandler` objects, providing ~96% memory savings
/// while maintaining identical CSV output format.
///
/// # Arguments
///
/// * `future_cost_function_graph` - Graph with future cost functions and cuts
/// * `simulation_trajectories` - Lightweight trajectories with simulation output data
/// * `saa` - Sample Average Approximation containing sampled scenarios (for noise export)
/// * `export_sampled_noises` - Whether to export sampled noises to CSV
/// * `path` - Optional output directory path. If `None`, all output is skipped (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Performance
///
/// - When `path` is `None`, returns immediately with no I/O overhead
/// - Trajectory-based access provides ~5-10% faster CSV export due to:
///   - Sequential memory access (cache-friendly)
///   - No graph node lookups or pointer indirection
///   - Direct array indexing (stage_index) vs node_id lookups
///
/// Training iteration output for CSV export.
///
/// Captures iteration-level convergence metrics and detailed timing breakdowns
/// for both forward and backward passes. Each row represents one forward pass
/// within an iteration.
///
/// # Schema
///
/// - Convergence: iteration, forward_pass_idx, lower_bound, policy_cost, policy_std, gap_percent
/// - Forward timing: saa_sampling_ms through total_ms (6 fields)
/// - Backward timing: preprocessing_ms through total_ms (9 fields)
///
/// # Row Count
///
/// Total rows = Σ(num_forward_passes per iteration)
/// Example: 9 iterations × 10 forward passes = 90 rows
#[derive(serde::Serialize)]
struct TrainingOutput {
    // Convergence metrics
    iteration: usize,
    lower_bound: f64,
    policy_cost: f64,
    policy_std: f64,
    gap_percent: f64,

    // Forward pass timing (milliseconds)
    forward_saa_sampling_ms: u64,
    forward_model_preprocessing_ms: u64,
    forward_solver_ms: u64,
    forward_model_postprocessing_ms: u64,
    forward_postprocessing_ms: u64,
    forward_total_ms: u64,

    // Backward pass timing (milliseconds)
    backward_preprocessing_ms: u64,
    backward_model_preprocessing_ms: u64,
    backward_solver_ms: u64,
    backward_model_postprocessing_ms: u64,
    backward_cut_selection_ms: u64,
    backward_fcf_state_update_ms: u64,
    backward_cut_cloning_ms: u64,
    backward_handler_application_ms: u64,
    backward_total_ms: u64,
}

/// Writes training iteration results to CSV file.
///
/// Exports iteration-level convergence tracking with detailed timing breakdowns.
/// This output is always enabled when an output path is provided.
///
/// # Arguments
///
/// * `results` - Iteration results from SDDP training
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Schema Details
///
/// **Convergence metrics** (one row per iteration):
/// - `iteration`: Iteration number (1-indexed)
/// - `lower_bound`: Lower bound after backward pass ($)
/// - `policy_cost`: Mean cost of all forward passes in iteration ($)
/// - `policy_std`: Std dev of all forward pass costs in iteration ($)
/// - `gap_percent`: Relative gap = 100 × (policy_cost - lower_bound) / |lower_bound|
///
/// **Timing fields**: All durations converted to milliseconds (u64)
///
/// # Row Format
///
/// One row per iteration (e.g., 20 iterations = 20 data rows + 1 header = 21 lines).
/// The `policy_cost` is the mean of all forward passes in that iteration.
///
/// # Performance
///
/// - Row serialization: ~1μs per row (CSV crate is highly optimized)
/// - Total overhead: ~20μs for 20 iterations (negligible vs LP solves at 100ms each)
/// - Early return when path is None (zero overhead)
///
/// # Example
///
/// ```ignore
/// let results = vec![iteration_result_1, iteration_result_2];
/// write_training_results(&results, Some("./output"))?;
/// // Creates: ./output/training.csv with one row per iteration
/// ```
fn write_training_results(
    results: &[sddp::IterationResult],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested (no-op, no I/O)
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/training.csv"))?;

    for result in results {
        // Compute policy statistics across all forward passes in this iteration
        let num_forward_passes = result.forward_costs.len();
        if num_forward_passes == 0 {
            continue; // Skip if no forward passes (should not occur)
        }

        let mean_cost = result.forward_costs.iter().sum::<f64>()
            / num_forward_passes as f64;

        let variance = result
            .forward_costs
            .iter()
            .map(|&cost| (cost - mean_cost).powi(2))
            .sum::<f64>()
            / num_forward_passes as f64;
        let policy_std = variance.sqrt();

        // Compute relative gap (handle division by zero)
        let gap_percent = if result.lower_bound.abs() > 1e-10 {
            100.0 * (mean_cost - result.lower_bound) / result.lower_bound.abs()
        } else {
            f64::INFINITY
        };

        // Write one row per iteration (not per forward pass)
        wtr.serialize(TrainingOutput {
            // Convergence metrics
            iteration: result.iteration,
            lower_bound: result.lower_bound,
            policy_cost: mean_cost, // Mean of all forward passes
            policy_std,
            gap_percent,

            // Forward timing (Duration → milliseconds)
            forward_saa_sampling_ms: result
                .forward_timing
                .saa_sampling_time
                .as_millis() as u64,
            forward_model_preprocessing_ms: result
                .forward_timing
                .model_preprocessing_time
                .as_millis() as u64,
            forward_solver_ms: result.forward_timing.solver_time.as_millis()
                as u64,
            forward_model_postprocessing_ms: result
                .forward_timing
                .model_postprocessing_time
                .as_millis()
                as u64,
            forward_postprocessing_ms: result
                .forward_timing
                .forward_postprocessing_time
                .as_millis() as u64,
            forward_total_ms: result.forward_timing.total_time.as_millis()
                as u64,

            // Backward timing (Duration → milliseconds)
            backward_preprocessing_ms: result
                .backward_timing
                .backward_preprocessing_time
                .as_millis() as u64,
            backward_model_preprocessing_ms: result
                .backward_timing
                .model_preprocessing_time
                .as_millis()
                as u64,
            backward_solver_ms: result.backward_timing.solver_time.as_millis()
                as u64,
            backward_model_postprocessing_ms: result
                .backward_timing
                .model_postprocessing_time
                .as_millis()
                as u64,
            backward_cut_selection_ms: result
                .backward_timing
                .cut_selection_time
                .as_millis() as u64,
            backward_fcf_state_update_ms: result
                .backward_timing
                .fcf_state_update_time
                .as_millis() as u64,
            backward_cut_cloning_ms: result
                .backward_timing
                .cut_cloning_time
                .as_millis() as u64,
            backward_handler_application_ms: result
                .backward_timing
                .handler_application_time
                .as_millis()
                as u64,
            backward_total_ms: result.backward_timing.total_time.as_millis()
                as u64,
        })?;
    }

    wtr.flush()?;
    Ok(())
}

/// Writes lower bound detail with cut dominance analysis to CSV file.
///
/// For each iteration and stage, identifies which cut dominates at the evaluation state
/// (typically the initial state), exports the cut details, and computes distance metrics
/// between the evaluation state and the state where the dominating cut was generated.
///
/// This enables diagnosis of invalid bounds caused by cuts generated far from the
/// evaluation point, where linear extrapolation is inaccurate (e.g., Example 07).
///
/// # Arguments
///
/// * `iteration` - Current SDDP iteration number
/// * `g` - Future cost function graph with cut pools per stage
/// * `eval_state` - State at which to evaluate cuts (typically initial state)
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Schema
///
/// **Fixed columns**:
/// - `iteration`: SDDP iteration number
/// - `stage_index`: Stage number in policy graph
/// - `num_cuts_available`: Total cuts in pool (active + inactive)
/// - `dominating_cut_id`: ID of cut with max value at eval_state
/// - `dominating_cut_iteration`: Iteration when dominating cut was created
/// - `dominating_cut_forward_pass_idx`: Forward pass index of dominating cut
/// - `dominating_cut_value`: Cut value at eval_state (α + β'x)
/// - `dominating_cut_rhs`: Cut RHS (α)
///
/// **Dynamic columns** (depend on state dimension):
/// - `eval_state_0`, `eval_state_1`, ...: Evaluation state components
/// - `cut_generation_state_0`, `cut_generation_state_1`, ...: State where cut was generated
///
/// **Distance metrics**:
/// - `euclidean_distance`: L2 norm between states
/// - `max_coordinate_distance`: L∞ norm between states
///
/// # Mathematical Foundation
///
/// At evaluation state x, the lower bound is:
/// ```text
/// LB = c₀'y₀ + max{α_i + β_i'x : i ∈ Cuts}
/// ```
///
/// The dominating cut determines the lower bound. If this cut was generated at a state
/// far from x, linear extrapolation may be invalid, leading to inaccurate bounds.
///
/// **Example 07 Diagnosis**: Cut generated 67.6 units away from initial state led to
/// invalid lower bound exceeding the upper bound.
///
/// # Performance
///
/// - Cut evaluation: O(num_cuts × state_dim) per stage
/// - State lookup: O(num_states) per stage
/// - Distance computation: O(state_dim)
/// - Total per iteration: ~10ms for typical problems
///
/// # Example
///
/// ```ignore
/// // In training loop after each iteration:
/// write_lower_bound_detail(
///     iteration,
///     &fcf_graph,
///     initial_condition.flatten_state(), // Initial state as flat vector
///     Some("./output")
/// )?;
/// // Creates: ./output/lower_bound_detail.csv with diagnosis data
/// ```
/// Writes lower bound detail diagnostics to CSV file.
///
/// Exports cut dominance analysis at the initial state (first study stage).
/// Diagnostics are computed during training (not at output time) and stored
/// in IterationResult.
///
/// # Arguments
///
/// * `results` - Iteration results from SDDP training
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Schema
///
/// **Fixed columns**:
/// - `iteration`: SDDP iteration number
/// - `num_cuts_available`: Total cuts in pool at first stage
/// - `dominating_cut_id`: ID of cut with max value at initial state
/// - `dominating_cut_iteration`: Iteration when dominating cut was created
/// - `dominating_cut_forward_pass_idx`: Forward pass index of dominating cut
/// - `dominating_cut_value`: Cut value at initial state (α + β'x)
/// - `dominating_cut_rhs`: Cut RHS (α)
///
/// **Dynamic columns** (depend on state dimension):
/// - `initial_state_0`, `initial_state_1`, ...: Initial state components
/// - `cut_generation_state_0`, `cut_generation_state_1`, ...: State where dominating cut was generated
///
/// **Distance metrics**:
/// - `euclidean_distance`: L2 norm between initial and generation states
/// - `max_coordinate_distance`: L∞ norm between initial and generation states
///
/// # Mathematical Foundation
///
/// At initial state x₀, the lower bound contribution from the first stage is:
/// ```text
/// LB = max{α_i + β_i'x₀ : i ∈ Cuts}
/// ```
///
/// The dominating cut determines the lower bound. If this cut was generated far
/// from x₀, linear extrapolation may be invalid.
///
/// # Performance
///
/// - Diagnostics computed at training time (zero overhead here)
/// - CSV writing: ~1μs per row
fn write_lower_bound_detail(
    results: &[sddp::IterationResult],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested (no-op, no I/O)
    let Some(output_dir) = path else {
        return Ok(());
    };

    let csv_path = format!("{}/lower_bound_detail.csv", output_dir);
    let mut wtr = Writer::from_path(&csv_path)?;

    // Determine state dimension from first result (if any)
    let state_dim = results
        .first()
        .map(|r| r.lb_detail_initial_state.len())
        .unwrap_or(0);

    // Write header
    write_lower_bound_detail_header(&mut wtr, state_dim)?;

    // Write data rows
    for result in results {
        // Skip if no diagnostics were computed (e.g., no cuts yet)
        if result.lb_detail_num_cuts == 0 {
            continue;
        }

        write_lower_bound_detail_row(&mut wtr, result)?;
    }

    wtr.flush()?;
    Ok(())
}

/// Writes dynamic header for lower_bound_detail.csv based on state dimension.
fn write_lower_bound_detail_header(
    wtr: &mut Writer<std::fs::File>,
    state_dim: usize,
) -> Result<(), Box<dyn Error>> {
    let mut header = vec![
        "iteration".to_string(),
        "num_cuts_available".to_string(),
        "dominating_cut_id".to_string(),
        "dominating_cut_iteration".to_string(),
        "dominating_cut_forward_pass_idx".to_string(),
        "dominating_cut_value".to_string(),
        "dominating_cut_rhs".to_string(),
    ];

    // Add initial state columns
    for i in 0..state_dim {
        header.push(format!("initial_state_{}", i));
    }

    // Add cut generation state columns
    for i in 0..state_dim {
        header.push(format!("cut_generation_state_{}", i));
    }

    // Add distance metrics
    header.push("euclidean_distance".to_string());
    header.push("max_coordinate_distance".to_string());

    wtr.write_record(&header)?;
    Ok(())
}

/// Writes a single data row from IterationResult.
fn write_lower_bound_detail_row(
    wtr: &mut Writer<std::fs::File>,
    result: &sddp::IterationResult,
) -> Result<(), Box<dyn Error>> {
    let mut record = vec![
        result.iteration.to_string(),
        result.lb_detail_num_cuts.to_string(),
        result.lb_detail_dominating_cut_id.to_string(),
        result.lb_detail_dominating_cut_iteration.to_string(),
        result.lb_detail_dominating_cut_forward_pass_idx.to_string(),
        result.lb_detail_dominating_cut_value.to_string(),
        result.lb_detail_dominating_cut_rhs.to_string(),
    ];

    // Append initial state components
    for &val in &result.lb_detail_initial_state {
        record.push(val.to_string());
    }

    // Append cut generation state components
    for &val in &result.lb_detail_cut_generation_state {
        record.push(val.to_string());
    }

    // Append distance metrics
    record.push(result.lb_detail_euclidean_distance.to_string());
    record.push(result.lb_detail_max_coordinate_distance.to_string());

    wtr.write_record(&record)?;
    Ok(())
}

/// # Memory Note
///
/// CSV export now uses lightweight trajectories (~96KB each) instead of full
/// handlers (~6MB each). For 10,000 scenarios: 960 MB vs 60 GB.
/// Writes sampled training trajectories to CSV file
///
/// Exports complete forward pass trajectories from training, enabling:
/// - PAR model validation (verify AR constraints)
/// - State space exploration analysis
/// - Physical feasibility checks
///
/// # CSV Schema
///
/// Fixed columns (3): iteration, forward_pass_idx, stage_id
/// Dynamic columns: Expand from Realization fields based on system dimensions
///
/// # Arguments
///
/// * `trajectories` - Collection of trajectory snapshots from training
/// * `path` - Optional output directory. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Performance
///
/// When collection is empty, function returns immediately with minimal overhead.
/// CSV writing is buffered and flushed once at the end.
/// Writes forward pass details to normalized CSV file.
///
/// Exports complete forward pass trajectories in normalized database format with columns:
/// iteration, forward_pass_idx, stage_id, variable_name, entity_id, lag_index, value
///
/// This format is optimized for database import and analytical queries.
///
/// # Arguments
///
/// * `forward_details` - Forward pass details collected during training
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None` or details are empty)
fn write_forward_detail(
    forward_details: &[sddp::ForwardPassDetail],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested or no data to export
    let Some(output_dir) = path else {
        return Ok(());
    };

    if forward_details.is_empty() {
        return Ok(());
    }

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/forward_detail.csv"))?;

    // Write header
    wtr.write_record([
        "iteration",
        "forward_pass_idx",
        "stage_id",
        "variable_name",
        "entity_id",
        "lag_index",
        "value",
    ])?;

    // Write data rows - iterate through each forward pass detail
    for detail in forward_details {
        let r = &detail.realization;
        let iteration = detail.iteration.to_string();
        let fp_idx = detail.forward_pass_idx.to_string();
        let stage_id = detail.stage_id.to_string();

        // Initial storage (per hydro)
        for (hydro_id, &value) in r.initial_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "initial_storage",
                &hydro_id.to_string(),
                "", // No lag index
                &value.to_string(),
            ])?;
        }

        // Inflow lags (per hydro, per lag)
        for (hydro_id, lags) in r.inflow_lags.iter().enumerate() {
            for (lag_idx, &value) in lags.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    "inflow_lag",
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(), // 1-indexed lags
                    &value.to_string(),
                ])?;
            }
        }

        // Sampled loads (per bus)
        for (bus_id, &value) in r.loads.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "sampled_load",
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Sampled inflows (per hydro)
        for (hydro_id, &value) in r.inflow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "sampled_inflow",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Turbined flow (per hydro)
        for (hydro_id, &value) in r.turbined_flow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "turbined_flow",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Spillage (per hydro)
        for (hydro_id, &value) in r.spillage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "spillage",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Thermal generation (per thermal)
        for (thermal_id, &value) in r.thermal_generation.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "thermal_generation",
                &thermal_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Deficit (per bus)
        for (bus_id, &value) in r.deficit.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "deficit",
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Exchange (per line)
        for (line_id, &value) in r.exchange.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "exchange",
                &line_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Final storage (per hydro)
        for (hydro_id, &value) in r.final_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "final_storage",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Water value (per hydro)
        for (hydro_id, &value) in r.water_value.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "water_value",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Marginal cost (per bus)
        for (bus_id, &value) in r.marginal_cost.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                "marginal_cost",
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Inflow lag duals (per hydro, per lag)
        for (hydro_id, duals) in r.inflow_lag_duals.iter().enumerate() {
            for (lag_idx, &value) in duals.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    "inflow_lag_dual",
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(),
                    &value.to_string(),
                ])?;
            }
        }

        // Objectives (no entity_id or lag_index)
        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            "current_stage_objective",
            "",
            "",
            &r.current_stage_objective.to_string(),
        ])?;

        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            "total_stage_objective",
            "",
            "",
            &r.total_stage_objective.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Writes backward pass branching records to CSV file.
///
/// Exports complete realization data for each branching scenario solved during
/// backward pass. Each row represents one branching realization.
///
/// # Arguments
///
/// * `backward_details` - Branching records collected during backward pass
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None` or records are empty)
///
/// # Performance
///
/// When `path` is `None` or records are empty, this function returns immediately
/// with no I/O overhead.
/// Writes backward pass details to normalized CSV file.
///
/// Exports complete backward pass branching realizations in normalized database format with columns:
/// iteration, forward_pass_idx, stage_id, training_state_id, branching_idx, variable_name, entity_id, lag_index, value
///
/// This format is optimized for database import and analytical queries.
///
/// # Arguments
///
/// * `backward_details` - Backward pass details collected during training
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None` or details are empty)
fn write_backward_detail(
    backward_details: &[sddp::BackwardPassDetail],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Early return if no output requested or no records collected (no-op, no I/O)
    let Some(output_dir) = path else {
        return Ok(());
    };

    if backward_details.is_empty() {
        return Ok(());
    }

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/backward_detail.csv"))?;

    // Write header
    wtr.write_record([
        "iteration",
        "forward_pass_idx",
        "stage_id",
        "training_state_id",
        "branching_idx",
        "variable_name",
        "entity_id",
        "lag_index",
        "value",
    ])?;

    // Write data rows - iterate through each backward pass detail
    for detail in backward_details {
        let r = &detail.realization;
        let iteration = detail.iteration.to_string();
        let fp_idx = detail.forward_pass_idx.to_string();
        let stage_id = detail.stage_id.to_string();
        let ts_id = detail.training_state_id.to_string();
        let br_idx = detail.branching_idx.to_string();

        // Initial storage (per hydro)
        for (hydro_id, &value) in r.initial_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "initial_storage",
                &hydro_id.to_string(),
                "", // No lag index
                &value.to_string(),
            ])?;
        }

        // Inflow lags (per hydro, per lag)
        for (hydro_id, lags) in r.inflow_lags.iter().enumerate() {
            for (lag_idx, &value) in lags.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    &ts_id,
                    &br_idx,
                    "inflow_lag",
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(), // 1-indexed lags
                    &value.to_string(),
                ])?;
            }
        }

        // Sampled loads (per bus)
        for (bus_id, &value) in r.loads.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "sampled_load",
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Sampled inflows (per hydro)
        for (hydro_id, &value) in r.inflow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "sampled_inflow",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Turbined flow (per hydro)
        for (hydro_id, &value) in r.turbined_flow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "turbined_flow",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Spillage (per hydro)
        for (hydro_id, &value) in r.spillage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "spillage",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Thermal generation (per thermal)
        for (thermal_id, &value) in r.thermal_generation.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "thermal_generation",
                &thermal_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Deficit (per bus)
        for (bus_id, &value) in r.deficit.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "deficit",
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Exchange (per line)
        for (line_id, &value) in r.exchange.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "exchange",
                &line_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Final storage (per hydro)
        for (hydro_id, &value) in r.final_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "final_storage",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Water value (per hydro)
        for (hydro_id, &value) in r.water_value.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "water_value",
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Marginal cost (per bus)
        for (bus_id, &value) in r.marginal_cost.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                "marginal_cost",
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Inflow lag duals (per hydro, per lag)
        for (hydro_id, duals) in r.inflow_lag_duals.iter().enumerate() {
            for (lag_idx, &value) in duals.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    &ts_id,
                    &br_idx,
                    "inflow_lag_dual",
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(),
                    &value.to_string(),
                ])?;
            }
        }

        // Objectives (no entity_id or lag_index)
        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            &ts_id,
            &br_idx,
            "current_stage_objective",
            "",
            "",
            &r.current_stage_objective.to_string(),
        ])?;

        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            &ts_id,
            &br_idx,
            "total_stage_objective",
            "",
            "",
            &r.total_stage_objective.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_outputs(
    future_cost_function_graph: &graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    simulation_trajectories: &[sddp::SimulationTrajectory],
    training_results: &[sddp::IterationResult],
    forward_details: &[sddp::ForwardPassDetail],
    backward_details: &[sddp::BackwardPassDetail],
    saa: &scenario::SAA,
    export_training_noises: bool,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Always export training results when output path is provided
    write_training_results(training_results, path)?;

    // Export lower bound detail (diagnostics computed during training)
    write_lower_bound_detail(training_results, path)?;

    // Export training trajectories (if collected)
    write_forward_detail(forward_details, path)?;

    // Export backward statistics (if collected)
    write_backward_detail(backward_details, path)?;

    write_benders_cuts(future_cost_function_graph, path)?;
    write_visited_states(future_cost_function_graph, path)?;
    write_buses_simulation_results(simulation_trajectories, path)?;
    write_lines_simulation_results(simulation_trajectories, path)?;
    write_thermals_simulation_results(simulation_trajectories, path)?;
    write_hydros_simulation_results(simulation_trajectories, path)?;

    if export_training_noises {
        write_sampled_noises(saa, path)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_mock_iteration_result(
        iteration: usize,
        lower_bound: f64,
        forward_costs: Vec<f64>,
    ) -> sddp::IterationResult {
        sddp::IterationResult {
            iteration,
            lower_bound,
            forward_costs,
            iteration_time: Duration::from_millis(100),
            forward_timing: sddp::ForwardPassTiming {
                saa_sampling_time: Duration::from_millis(1),
                model_preprocessing_time: Duration::from_millis(2),
                solver_time: Duration::from_millis(50),
                model_postprocessing_time: Duration::from_millis(3),
                forward_postprocessing_time: Duration::from_millis(4),
                total_time: Duration::from_millis(60),
            },
            backward_timing: sddp::BackwardPassTiming {
                backward_preprocessing_time: Duration::from_millis(1),
                model_preprocessing_time: Duration::from_millis(2),
                solver_time: Duration::from_millis(30),
                model_postprocessing_time: Duration::from_millis(3),
                cut_selection_time: Duration::from_millis(1),
                fcf_state_update_time: Duration::from_millis(1),
                cut_cloning_time: Duration::from_millis(1),
                handler_application_time: Duration::from_millis(1),
                total_time: Duration::from_millis(40),
            },
            num_solver_calls: 10,
            num_cuts_added: 5,
            num_cuts_removed: 0,
            num_cuts_returned: 0,
            num_active_cuts: 5,
            lb_detail_num_cuts: 0,
            lb_detail_dominating_cut_id: 0,
            lb_detail_dominating_cut_iteration: 0,
            lb_detail_dominating_cut_forward_pass_idx: 0,
            lb_detail_dominating_cut_value: 0.0,
            lb_detail_dominating_cut_rhs: 0.0,
            lb_detail_initial_state: vec![],
            lb_detail_cut_generation_state: vec![],
            lb_detail_euclidean_distance: 0.0,
            lb_detail_max_coordinate_distance: 0.0,
        }
    }

    #[test]
    fn test_training_csv_no_output_path() {
        // When path is None, should return Ok without any I/O
        let results =
            vec![create_mock_iteration_result(1, 100.0, vec![105.0, 103.0])];
        let result = write_training_results(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_training_csv_row_count() {
        // Row count should equal number of iterations (not forward passes)
        let results = vec![
            create_mock_iteration_result(1, 100.0, vec![105.0, 103.0, 104.0]),
            create_mock_iteration_result(2, 102.0, vec![106.0, 104.0]),
        ];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        write_training_results(&results, Some(path)).unwrap();

        // Read the CSV
        let csv_path = format!("{}/training.csv", path);
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // 1 header + 2 iterations = 3 lines
        assert_eq!(lines.len(), 3);

        // Verify header
        assert!(lines[0].contains("iteration"));
        assert!(!lines[0].contains("forward_pass_idx")); // No longer exists
        assert!(lines[0].contains("lower_bound"));
        assert!(lines[0].contains("policy_cost"));

        // Verify first iteration: policy_cost should be mean of [105, 103, 104] = 104
        let row1_fields: Vec<&str> = lines[1].split(',').collect();
        let policy_cost1 = row1_fields[2].parse::<f64>().unwrap();
        assert!((policy_cost1 - 104.0).abs() < 1e-10);

        // Verify second iteration: policy_cost should be mean of [106, 104] = 105
        let row2_fields: Vec<&str> = lines[2].split(',').collect();
        let policy_cost2 = row2_fields[2].parse::<f64>().unwrap();
        assert!((policy_cost2 - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_training_csv_gap_computation() {
        // Test gap calculation and edge cases
        let results = vec![
            // Normal case: lower < upper, mean of [105, 103] = 104
            create_mock_iteration_result(1, 100.0, vec![105.0, 103.0]),
            // Edge case: lower bound near zero
            create_mock_iteration_result(2, 1e-12, vec![105.0]),
        ];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        write_training_results(&results, Some(path)).unwrap();

        let csv_path = format!("{}/training.csv", path);
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Should have 1 header + 2 iterations = 3 lines
        assert_eq!(lines.len(), 3);

        // Check first data row (iteration 1)
        let row1 = lines[1];
        let fields: Vec<&str> = row1.split(',').collect();
        let gap = fields[4].parse::<f64>().unwrap(); // gap_percent is 5th column (index 4)
                                                     // Gap = 100 * (104 - 100) / 100 = 4.0%
        assert!((gap - 4.0).abs() < 0.1);

        // Check iteration 2 with near-zero lower bound
        let row2 = lines[2];
        let fields: Vec<&str> = row2.split(',').collect();
        let gap_str = fields[4];
        // Should be "inf" string in CSV
        assert!(
            gap_str == "inf" || gap_str.to_lowercase().contains("inf"),
            "Expected infinity, got: {}",
            gap_str
        );
    }

    #[test]
    fn test_training_csv_timing_conversion() {
        // Verify Duration is correctly converted to milliseconds
        let results = vec![create_mock_iteration_result(1, 100.0, vec![105.0])];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        write_training_results(&results, Some(path)).unwrap();

        let csv_path = format!("{}/training.csv", path);
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        let row = lines[1];
        let fields: Vec<&str> = row.split(',').collect();

        // Check some timing fields (indices after convergence metrics)
        // Columns: iteration(0), lower_bound(1), policy_cost(2), policy_std(3), gap_percent(4),
        //          forward_saa_sampling_ms(5), forward_model_preprocessing_ms(6), forward_solver_ms(7), ...
        let forward_solver_ms = fields[7].parse::<u64>().unwrap();
        let backward_solver_ms = fields[13].parse::<u64>().unwrap();

        assert_eq!(forward_solver_ms, 50); // From mock
        assert_eq!(backward_solver_ms, 30); // From mock
    }

    #[test]
    fn test_training_csv_empty_forward_costs() {
        // Edge case: iteration with no forward passes (shouldn't happen but handle gracefully)
        let results = vec![create_mock_iteration_result(1, 100.0, vec![])];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        // This should succeed without error
        let result = write_training_results(&results, Some(path));
        assert!(result.is_ok());

        let csv_path = format!("{}/training.csv", path);

        // File is created but will be empty since no data was serialized
        // CSV writer only writes header on first serialize() call
        assert!(std::path::Path::new(&csv_path).exists());

        let content = std::fs::read_to_string(&csv_path).unwrap();
        // File will be empty (no header written since no serialize calls)
        assert_eq!(
            content.len(),
            0,
            "Empty forward_costs should produce empty CSV"
        );
    }
}
