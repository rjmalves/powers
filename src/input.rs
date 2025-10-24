use crate::graph;
use crate::initial_condition;
use crate::input_validation::InputValidator;
use crate::scenario;
use crate::sddp;
use crate::subproblem;
use crate::system;
use crate::unified_noise_spec::{
    SeasonalNoiseParams, TemporalModelSpec, UnifiedNoiseSpec,
};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;

#[derive(Deserialize)]
pub struct Config {
    pub num_iterations: usize,
    pub num_forward_passes: usize,

    /// Number of scenarios for out-of-sample simulation.
    ///
    /// - `None`: Skip simulation (training-only mode)
    /// - `Some(n)`: Run simulation with n scenarios (must be > 0)
    ///
    /// Setting this to `None` or omitting it from config.json will skip the
    /// simulation phase entirely
    #[serde(default)]
    pub num_simulation_scenarios: Option<usize>,
    pub seed: u64,

    /// Number of threads for parallel execution.
    ///
    /// - `None`: Auto-detect available cores (uses `num_cpus`)
    /// - `Some(n)`: Use exactly `n` threads (must be > 0)
    ///
    /// Thread pool is configured before training and simulation.
    /// Rayon's global thread pool is used for both forward and backward passes.
    ///
    #[serde(default)]
    pub num_threads: Option<usize>,

    /// Optional path for CSV output files.
    ///
    /// If `None`, no CSV files will be written (useful for tests and benchmarks).
    /// This eliminates I/O overhead and prevents test directory clutter.
    ///
    /// Default: `None`
    #[serde(default)]
    pub output_path: Option<String>,
}

pub fn read_config_input(filepath: &str) -> Config {
    let contents =
        fs::read_to_string(filepath).expect("Error while reading config file");
    let parsed: Config = serde_json::from_str(&contents).unwrap();
    parsed
}

#[derive(Deserialize)]
pub struct BusInput {
    pub id: usize,
    pub deficit_cost: f64,
}

#[derive(Deserialize)]
pub struct LineInput {
    pub id: usize,
    pub source_bus_id: usize,
    pub target_bus_id: usize,
    pub direct_capacity: f64,
    pub reverse_capacity: f64,
    pub exchange_penalty: f64,
}

#[derive(Deserialize)]
pub struct ThermalInput {
    pub id: usize,
    pub bus_id: usize,
    pub cost: f64,
    pub min_generation: f64,
    pub max_generation: f64,
}

#[derive(Deserialize)]
pub struct HydroInput {
    pub id: usize,
    pub downstream_hydro_id: Option<usize>,
    pub bus_id: usize,
    pub productivity: f64,
    pub min_storage: f64,
    pub max_storage: f64,
    pub min_turbined_flow: f64,
    pub max_turbined_flow: f64,
    pub spillage_penalty: f64,
}

#[derive(Deserialize)]
pub struct SystemInput {
    pub buses: Vec<BusInput>,
    pub lines: Vec<LineInput>,
    pub thermals: Vec<ThermalInput>,
    pub hydros: Vec<HydroInput>,
}

pub fn read_system_input(filepath: &str) -> SystemInput {
    let contents =
        fs::read_to_string(filepath).expect("Error while reading config file");
    let parsed: SystemInput = serde_json::from_str(&contents).unwrap();
    parsed
}

fn validate_id_range(ids: &[usize], elem_name: &str) {
    let num_elements = ids.len();
    for elem_id in 0..num_elements {
        if !ids.contains(&elem_id) {
            panic!("ID {} not found for {}", elem_id, elem_name);
        }
    }
}

#[allow(dead_code)]
fn validate_entity_count(ids: &[usize], count: usize, elem_name: &str) {
    let entity_count = ids.len();
    if entity_count != count {
        panic!(
            "Error matching recourse for {}: {} != {}",
            elem_name, entity_count, count
        );
    }
}

impl SystemInput {
    pub fn build_sddp_system(&self) -> system::System {
        // ensure valid id ranges (0..)
        let buses_ids: Vec<usize> = self.buses.iter().map(|b| b.id).collect();
        let lines_ids: Vec<usize> = self.lines.iter().map(|b| b.id).collect();
        let thermals_ids: Vec<usize> =
            self.thermals.iter().map(|b| b.id).collect();
        let hydros_ids: Vec<usize> = self.hydros.iter().map(|b| b.id).collect();
        validate_id_range(&buses_ids, "buses");
        validate_id_range(&lines_ids, "lines");
        validate_id_range(&thermals_ids, "thermals");
        validate_id_range(&hydros_ids, "hydros");

        let num_buses = buses_ids.len();
        let mut buses = Vec::<system::Bus>::with_capacity(num_buses);
        for id in 0..num_buses {
            let bus = self.buses.iter().find(|b| b.id == id).unwrap();
            buses.push(system::Bus::new(id, bus.deficit_cost));
        }

        let num_lines = lines_ids.len();
        let mut lines = Vec::<system::Line>::with_capacity(num_lines);
        for id in 0..num_lines {
            let line = self.lines.iter().find(|l| l.id == id).unwrap();
            lines.push(system::Line::new(
                id,
                line.source_bus_id,
                line.target_bus_id,
                line.direct_capacity,
                line.reverse_capacity,
                line.exchange_penalty,
            ));
        }

        let num_thermals = thermals_ids.len();
        let mut thermals = Vec::<system::Thermal>::with_capacity(num_thermals);
        for id in 0..num_thermals {
            let thermal = self.thermals.iter().find(|t| t.id == id).unwrap();
            thermals.push(system::Thermal::new(
                id,
                thermal.bus_id,
                thermal.cost,
                thermal.min_generation,
                thermal.max_generation,
            ));
        }

        let num_hydros = hydros_ids.len();
        let mut hydros = Vec::<system::Hydro>::with_capacity(num_hydros);
        for id in 0..num_hydros {
            let hydro = self.hydros.iter().find(|h| h.id == id).unwrap();
            hydros.push(system::Hydro::new(
                id,
                hydro.downstream_hydro_id,
                hydro.bus_id,
                hydro.productivity,
                hydro.min_storage,
                hydro.max_storage,
                hydro.min_turbined_flow,
                hydro.max_turbined_flow,
                hydro.spillage_penalty,
            ));
        }

        system::System::new(buses, lines, thermals, hydros)
    }
}

#[derive(Deserialize)]
pub struct GraphNodeInput {
    pub id: usize,
    pub stage_id: usize,
    pub season_id: usize,
    pub start_date: String,
    pub end_date: String,
    pub risk_measure: String,
    pub load_stochastic_process: String,
    pub inflow_stochastic_process: String,
    pub state_variables: String,
    pub num_scenarios: usize,
}

#[derive(Deserialize)]
pub struct GraphEdgeInput {
    pub source_id: usize,
    pub target_id: usize,
    pub probability: f64,
    pub discount_rate: f64,
}

#[derive(Deserialize)]
pub struct GraphInput {
    pub nodes: Vec<GraphNodeInput>,
    pub edges: Vec<GraphEdgeInput>,
}

pub fn read_graph_input(filepath: &str) -> GraphInput {
    let contents =
        fs::read_to_string(filepath).expect("Error while reading graph file");
    let parsed: GraphInput = serde_json::from_str(&contents).unwrap();
    parsed
}

impl GraphInput {
    fn add_sddp_study_period_to_graph(
        &self,
        graph: &mut graph::DirectedGraph<sddp::NodeData>,
        system_input: &SystemInput,
        recourse: &Recourse,
    ) -> Result<(), String> {
        // Get unified specs (internal representation)
        let unified_specs = recourse
            .get_unified_specs()
            .map_err(|e| format!("Failed to get unified specs: {}", e))?;

        // Build study graph
        for node_input in self.nodes.iter() {
            let r = graph.add_node(sddp::NodeData::new(
                node_input.id as isize,
                node_input.stage_id,
                node_input.season_id,
                &node_input.start_date,
                &node_input.end_date,
                subproblem::StudyPeriodKind::Study,
                system_input.build_sddp_system(),
                &node_input.risk_measure,
                &node_input.load_stochastic_process,
                &unified_specs,
                &node_input.state_variables,
                node_input.num_scenarios,
            )?);
            if r.is_err() {
                panic!("Error while building graph in node {}", node_input.id);
            }
        }
        for edge_input in self.edges.iter() {
            let source_id = graph
                .get_node_id_with(|node_data| {
                    node_data.id == edge_input.source_id as isize
                })
                .unwrap_or_else(|| {
                    panic!(
                        "Error adding edge {} -> {}",
                        edge_input.source_id, edge_input.target_id
                    )
                });
            let target_id = graph
                .get_node_id_with(|node_data| {
                    node_data.id == edge_input.target_id as isize
                })
                .unwrap();
            let r = graph.add_edge(source_id, target_id);
            if r.is_err() {
                panic!(
                    "Error while building graph in edge {} -> {}",
                    edge_input.source_id, edge_input.target_id
                );
            }
        }
        Ok(())
    }

    fn add_sddp_pre_study_period_to_graph(
        &self,
        graph: &mut graph::DirectedGraph<sddp::NodeData>,
        system_input: &SystemInput,
        recourse: &Recourse,
    ) -> Result<(), String> {
        // Get unified specs (internal representation)
        let unified_specs = recourse
            .get_unified_specs()
            .map_err(|e| format!("Failed to get unified specs: {}", e))?;

        // Get state configuration from the first study node
        let first_node = self.nodes.first().ok_or("Graph has no nodes")?;
        let state_choice = &first_node.state_variables;
        let inflow_process_type = &first_node.inflow_stochastic_process;

        // Determine lag_order based on state choice
        let lag_order = match state_choice.as_str() {
            "storage" => 0,
            "storage_and_inflow" => {
                let inflow_process =
                    crate::stochastic_process::factory(inflow_process_type);
                inflow_process.lag_order()
            }
            _ => {
                return Err(format!(
                    "Unknown state_variables: '{}'",
                    state_choice
                ))
            }
        };

        // Create 1+p pre-study nodes with IDs: -p, -(p-1), ..., -1, 0
        let num_pre_study_nodes = 1 + lag_order;
        let mut pre_study_node_ids = Vec::with_capacity(num_pre_study_nodes);

        for pre_idx in 0..num_pre_study_nodes {
            let node_id_value = -(lag_order as isize - pre_idx as isize);
            let graph_node_id = graph
                .add_node(sddp::NodeData::new(
                    node_id_value,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    system_input.build_sddp_system(),
                    "expectation",
                    "naive",
                    &unified_specs,
                    state_choice,
                    1, // PreStudy always has 1 scenario
                )?)
                .map_err(|_| {
                    format!("Failed to add pre-study node {}", node_id_value)
                })?;
            pre_study_node_ids.push(graph_node_id);
        }

        // Connect pre-study nodes sequentially: PreStudy(-p) -> ... -> PreStudy(0)
        for i in 0..(num_pre_study_nodes - 1) {
            graph
                .add_edge(pre_study_node_ids[i], pre_study_node_ids[i + 1])
                .map_err(|_| {
                    format!(
                        "Failed to connect pre-study nodes {} -> {}",
                        i,
                        i + 1
                    )
                })?;
        }

        // Connect last pre-study node (ID=0) to first study node
        let first_study_node_id = graph
            .get_node_id_with(|node_data| {
                node_data.id == first_node.id as isize
            })
            .ok_or_else(|| {
                format!("Could not find study node with ID {}", first_node.id)
            })?;

        graph
            .add_edge(*pre_study_node_ids.last().unwrap(), first_study_node_id)
            .map_err(|_| {
                "Failed to connect pre-study to study period".to_string()
            })?;

        Ok(())
    }

    pub fn build_sddp_graph(
        &self,
        system_input: &SystemInput,
        recourse: &Recourse,
    ) -> Result<graph::DirectedGraph<sddp::NodeData>, String> {
        let mut g = graph::DirectedGraph::<sddp::NodeData>::new();

        self.add_sddp_study_period_to_graph(&mut g, system_input, recourse)?;
        self.add_sddp_pre_study_period_to_graph(
            &mut g,
            system_input,
            recourse,
        )?;
        Ok(g)
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct InitialStorage {
    pub hydro_id: usize,
    pub value: f64,
}

/// Historical inflow value for AR model initialization
///
/// Each entry specifies one lag value for one hydro plant.
/// For AR(p) models, each hydro must have p entries with lag values 1..p.
///
/// # Example (AR(2) for hydro 0)
/// ```json
/// [
///   {"hydro_id": 0, "lag": 1, "value": 120.0},
///   {"hydro_id": 0, "lag": 2, "value": 115.0}
/// ]
/// ```
#[derive(Deserialize, Serialize, Clone)]
pub struct PastInflow {
    pub hydro_id: usize,
    /// Lag index (1 = t-1, 2 = t-2, ..., p = t-p)
    pub lag: usize,
    /// Historical inflow value (must be non-negative)
    pub value: f64,
}

/// Initial condition for SDDP algorithm
///
/// Specifies starting reservoir storage and historical inflow lags for AR models.
#[derive(Deserialize, Serialize, Clone)]
pub struct InitialConditionInput {
    pub storage: Vec<InitialStorage>,
    /// Historical inflow lags for AR model initialization.
    /// For AR(p) models, each hydro must have p entries with lag=1..p.
    /// Independent noise models can leave this empty.
    pub inflow: Vec<PastInflow>,
}

#[derive(Deserialize)]
pub struct NormalParams {
    pub mu: f64,
    pub sigma: f64,
}

#[derive(Deserialize)]
pub struct LoadDistribution {
    pub bus_id: usize,
    pub normal: NormalParams,
}

/// Type of uncertainty in the stochastic process
///
/// Directly specifies what aspect of the power system is uncertain.
/// This replaces the indirect EntityType mapping for clearer semantics.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum UncertaintyType {
    /// Inflow uncertainty (hydro plant water inflows)
    Inflow,
    /// Load uncertainty (electrical demand at buses)
    Load,
}

/// Temporal model for stochastic processes
///
/// Specifies the temporal correlation structure of the stochastic process.
///
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TemporalModel {
    /// Independent process (no temporal correlation)
    ///
    /// Realizations are independent across time:
    /// Xₜ ~ F (marginal distribution)
    Independent,

    /// Periodic Autoregressive PAR(p) model
    ///
    /// AR parameters vary by season. Each season can have different:
    /// - μₘ: seasonal mean
    /// - σₘ: seasonal standard deviation
    /// - φₖₘ: AR coefficients (k = 1..pₘ)
    ///
    /// Implements the PAR(p) equation:
    /// ```text
    /// Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + ... + φₚₘ·aₜ₋ₚ + aₜ]
    /// ```
    ///
    /// where:
    /// - m = t mod num_seasons (seasonal index, maps to season_id in graph nodes)
    /// - aₜ = transformed residual (e.g., from LogNormal3)
    /// - pₘ = AR order for season m (can vary!)
    ///
    ///
    #[serde(rename = "periodic_ar")]
    PeriodicAutoregressive {
        /// Seasonal cycle length (e.g., 12 for monthly, 4 for quarterly)
        ///
        /// Maps to season_id values in graph nodes. Must be > 0.
        /// All seasonal arrays must have length equal to this value.
        num_seasons: usize,

        /// AR order for each season [p₀, p₁, ..., p_{num_seasons-1}]
        ///
        /// Each element specifies the AR order for that season.
        /// Orders can vary by season. Length must equal `num_seasons`.
        ar_orders: Vec<usize>,

        /// AR coefficients for each season
        ///
        /// `ar_coefficients[m]` contains the AR coefficients [φ₁ₘ, φ₂ₘ, ..., φₚₘ]
        /// for season m. The length of `ar_coefficients[m]` must equal `ar_orders[m]`.
        ///
        /// Outer vec length = `num_seasons`, inner vec[m] length = `ar_orders[m]`.
        ar_coefficients: Vec<Vec<f64>>,

        /// Seasonal mean for each season [μ₀, μ₁, ..., μ_{num_seasons-1}]
        ///
        /// Each element specifies the mean value for that season (μₘ).
        /// Length must equal `num_seasons`.
        seasonal_means: Vec<f64>,

        /// Seasonal standard deviation for each season [σ₀, σ₁, ..., σ_{num_seasons-1}]
        ///
        /// Each element specifies the standard deviation for that season (σₘ).
        /// All values must be > 0. Length must equal `num_seasons`.
        seasonal_stds: Vec<f64>,
    },
}

/// Seasonal statistics for a single period in PAR(p) model
///
/// Contains all statistical parameters for one period (season) in a Periodic
/// Autoregressive model. Each period in the seasonal cycle (identified by
/// `season_id` in graph nodes) has its own set of parameters.
///
/// - μₘ: seasonal mean (`mean`)
/// - σₘ: seasonal standard deviation (`std_dev`)
/// - γₘ: seasonal skewness (`skewness`, optional)
/// - pₘ: AR order for this period (`ar_order`)
///
/// where m is the period index (0..num_seasons-1)
///
/// # Usage
///
/// These statistics are used for:
/// 1. PAR(p) scenario generation (PAR-006)
/// 2. Parameter estimation from historical data (PAR-013)
/// 3. Validation of seasonal parameter consistency (PAR-005)
///
/// # Example
///
/// ```rust
/// use powers_rs::input::SeasonalStats;
///
/// // Wet season period with AR(2)
/// let wet_season = SeasonalStats {
///     period_index: 2,
///     mean: 150.0,        // μ₂ = 150.0
///     std_dev: 30.0,      // σ₂ = 30.0
///     skewness: Some(0.5), // γ₂ = 0.5 (right-skewed)
///     ar_order: 2,         // p₂ = 2 (AR(2) for this period)
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalStats {
    /// Period index in seasonal cycle (0..num_seasons-1)
    ///
    /// For monthly data: 0=Jan, 1=Feb, ..., 11=Dec
    /// For quarterly data: 0=Q1, 1=Q2, 2=Q3, 3=Q4
    pub period_index: usize,

    /// Seasonal mean μₘ for this period
    ///
    /// This is the expected value of the process at this period,
    /// before AR dynamics are applied.
    pub mean: f64,

    /// Seasonal standard deviation σₘ for this period
    ///
    /// Must be > 0. Represents the variability of the process
    /// at this period.
    pub std_dev: f64,

    /// Optional seasonal skewness γₘ for this period
    ///
    /// Used for distribution fitting (e.g., LogNormal3 parameters).
    /// - `None`: No skewness information available
    /// - `Some(γ)`: Skewness coefficient (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
    pub skewness: Option<f64>,

    /// AR order pₘ for this period
    ///
    /// Specifies how many past values are used in the AR equation
    /// for this period. Can vary by period (e.g., AR(1) in dry season,
    /// AR(2) in wet season).
    pub ar_order: usize,
}

/// Complete parameter set for Periodic Autoregressive PAR(p) model
///
/// Aggregates seasonal statistics with AR coefficients for all periods
/// in the seasonal cycle. Provides validation and convenient access methods.
///
/// This structure encapsulates all parameters needed for PAR(p) equation:
///
/// ```text
/// Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + ... + φₚₘ·aₜ₋ₚ + aₜ]
/// ```
///
/// where:
/// - m = t mod period (seasonal index)
/// - μₘ, σₘ: from `seasonal_stats[m]`
/// - φₖₘ: from `ar_coefficients[m][k-1]`
/// - aₜ: transformed residual
///
/// # Validation
///
/// Use `validate_consistency()` to check:
/// - All arrays have length = `period`
/// - Each `ar_coefficients[m]` has length = `seasonal_stats[m].ar_order`
/// - All standard deviations are positive
///
/// Use `validate_stationarity()` to verify AR coefficients satisfy
/// stability conditions for each period.
///
/// # Example
///
/// ```rust
/// use powers_rs::input::{PeriodicARParams, SeasonalStats};
///
/// // Create quarterly PAR(1) model
/// let params = PeriodicARParams {
///     num_seasons: 4,
///     seasonal_stats: vec![
///         SeasonalStats { period_index: 0, mean: 100.0, std_dev: 20.0, skewness: None, ar_order: 1 },
///         SeasonalStats { period_index: 1, mean: 150.0, std_dev: 30.0, skewness: None, ar_order: 1 },
///         SeasonalStats { period_index: 2, mean: 180.0, std_dev: 35.0, skewness: None, ar_order: 1 },
///         SeasonalStats { period_index: 3, mean: 120.0, std_dev: 25.0, skewness: None, ar_order: 1 },
///     ],
///     ar_coefficients: vec![
///         vec![0.7],
///         vec![0.75],
///         vec![0.8],
///         vec![0.7],
///     ],
/// };
///
/// // Access seasonal parameters
/// assert_eq!(params.num_seasons, 4);
/// assert_eq!(params.get_params_for_season(0).mean, 100.0);
/// assert_eq!(params.get_ar_coeffs_for_season(0), &[0.7]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicARParams {
    /// Seasonal cycle length (e.g., 12 for monthly, 4 for quarterly)
    ///
    /// Must match the `num_seasons` field in `TemporalModel::PeriodicAutoregressive`.
    pub num_seasons: usize,

    /// Statistical parameters for each season in the cycle
    ///
    /// Length must equal `num_seasons`. Each entry contains the seasonal
    /// statistics (μₘ, σₘ, γₘ, pₘ) for that season.
    pub seasonal_stats: Vec<SeasonalStats>,

    /// AR coefficients for each season
    ///
    /// `ar_coefficients[m]` contains [φ₁ₘ, φ₂ₘ, ..., φₚₘ] for season m.
    /// The length of `ar_coefficients[m]` must equal `seasonal_stats[m].ar_order`.
    ///
    /// Outer vec length = `num_seasons`, inner vec[m] length = `seasonal_stats[m].ar_order`.
    pub ar_coefficients: Vec<Vec<f64>>,
}

impl PeriodicARParams {
    /// Get seasonal statistics for a specific season
    ///
    /// # Arguments
    ///
    /// * `season_idx` - Season index (0-based, wraps around if >= num_seasons)
    ///
    /// # Returns
    ///
    /// Reference to the `SeasonalStats` for the requested season.
    /// Uses modulo arithmetic to handle wraparound.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::input::{PeriodicARParams, SeasonalStats};
    /// # let params = PeriodicARParams {
    /// #     num_seasons: 4,
    /// #     seasonal_stats: vec![
    /// #         SeasonalStats { period_index: 0, mean: 100.0, std_dev: 20.0, skewness: None, ar_order: 1 },
    /// #         SeasonalStats { period_index: 1, mean: 150.0, std_dev: 30.0, skewness: None, ar_order: 1 },
    /// #         SeasonalStats { period_index: 2, mean: 180.0, std_dev: 35.0, skewness: None, ar_order: 1 },
    /// #         SeasonalStats { period_index: 3, mean: 120.0, std_dev: 25.0, skewness: None, ar_order: 1 },
    /// #     ],
    /// #     ar_coefficients: vec![vec![0.7], vec![0.75], vec![0.8], vec![0.7]],
    /// # };
    /// let stats_q2 = params.get_params_for_season(1);
    /// assert_eq!(stats_q2.mean, 150.0);
    ///
    /// // Wraparound: season 4 wraps to season 0
    /// let stats_wrap = params.get_params_for_season(4);
    /// assert_eq!(stats_wrap.mean, 100.0);
    /// ```
    #[inline]
    pub fn get_params_for_season(&self, season_idx: usize) -> &SeasonalStats {
        &self.seasonal_stats[season_idx % self.num_seasons]
    }

    /// Get AR coefficients for a specific season
    ///
    /// # Arguments
    ///
    /// * `season_idx` - Season index (0-based, wraps around if >= num_seasons)
    ///
    /// # Returns
    ///
    /// Slice containing AR coefficients [φ₁ₘ, φ₂ₘ, ..., φₚₘ] for the requested season.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::input::{PeriodicARParams, SeasonalStats};
    /// # let params = PeriodicARParams {
    /// #     num_seasons: 2,
    /// #     seasonal_stats: vec![
    /// #         SeasonalStats { period_index: 0, mean: 100.0, std_dev: 20.0, skewness: None, ar_order: 1 },
    /// #         SeasonalStats { period_index: 1, mean: 150.0, std_dev: 30.0, skewness: None, ar_order: 2 },
    /// #     ],
    /// #     ar_coefficients: vec![vec![0.7], vec![0.5, 0.3]],
    /// # };
    /// let coeffs_p0 = params.get_ar_coeffs_for_season(0);
    /// assert_eq!(coeffs_p0, &[0.7]);
    ///
    /// let coeffs_p1 = params.get_ar_coeffs_for_season(1);
    /// assert_eq!(coeffs_p1, &[0.5, 0.3]);
    /// ```
    #[inline]
    pub fn get_ar_coeffs_for_season(&self, season_idx: usize) -> &[f64] {
        &self.ar_coefficients[season_idx % self.num_seasons]
    }

    /// Validate parameter consistency
    ///
    /// Checks:
    /// 1. All vectors have length = `period`
    /// 2. `ar_coefficients[m].len()` = `seasonal_stats[m].ar_order` for all m
    /// 3. All standard deviations are positive
    ///
    /// # Errors
    ///
    /// Returns `Err` with descriptive message if any validation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::input::{PeriodicARParams, SeasonalStats};
    /// // Valid params
    /// let valid = PeriodicARParams {
    ///     num_seasons: 2,
    ///     seasonal_stats: vec![
    ///         SeasonalStats { period_index: 0, mean: 100.0, std_dev: 20.0, skewness: None, ar_order: 1 },
    ///         SeasonalStats { period_index: 1, mean: 150.0, std_dev: 30.0, skewness: None, ar_order: 2 },
    ///     ],
    ///     ar_coefficients: vec![vec![0.7], vec![0.5, 0.3]],
    /// };
    /// assert!(valid.validate_consistency().is_ok());
    ///
    /// // Invalid: mismatched AR coefficient count
    /// let invalid = PeriodicARParams {
    ///     num_seasons: 2,
    ///     seasonal_stats: vec![
    ///         SeasonalStats { period_index: 0, mean: 100.0, std_dev: 20.0, skewness: None, ar_order: 2 },
    ///         SeasonalStats { period_index: 1, mean: 150.0, std_dev: 30.0, skewness: None, ar_order: 1 },
    ///     ],
    ///     ar_coefficients: vec![vec![0.7], vec![0.5]],  // Period 0 expects 2 coeffs!
    /// };
    /// assert!(invalid.validate_consistency().is_err());
    /// ```
    pub fn validate_consistency(&self) -> Result<(), String> {
        // Check seasonal_stats length
        if self.seasonal_stats.len() != self.num_seasons {
            return Err(format!(
                "seasonal_stats length {} != num_seasons {}",
                self.seasonal_stats.len(),
                self.num_seasons
            ));
        }

        // Check ar_coefficients length
        if self.ar_coefficients.len() != self.num_seasons {
            return Err(format!(
                "ar_coefficients length {} != num_seasons {}",
                self.ar_coefficients.len(),
                self.num_seasons
            ));
        }

        // Check each season's parameters
        for (m, stats) in self.seasonal_stats.iter().enumerate() {
            // Validate positive standard deviation
            if stats.std_dev <= 0.0 {
                return Err(format!(
                    "Season {} std_dev {} must be > 0",
                    m, stats.std_dev
                ));
            }

            // Validate AR coefficient count matches AR order
            if self.ar_coefficients[m].len() != stats.ar_order {
                return Err(format!(
                    "Season {} ar_coefficients length {} != ar_order {}",
                    m,
                    self.ar_coefficients[m].len(),
                    stats.ar_order
                ));
            }
        }

        Ok(())
    }
}

impl TryFrom<&TemporalModel> for PeriodicARParams {
    type Error = String;

    /// Convert from TemporalModel::PeriodicAutoregressive to PeriodicARParams
    ///
    /// # Arguments
    ///
    /// * `temporal` - Reference to a TemporalModel
    ///
    /// # Returns
    ///
    /// - `Ok(params)` if the model is PeriodicAutoregressive
    /// - `Err(msg)` if the model is not PeriodicAutoregressive
    ///
    /// # Example
    ///
    /// ```rust
    /// use powers_rs::input::{TemporalModel, PeriodicARParams};
    /// use std::convert::TryFrom;
    ///
    /// let temporal = TemporalModel::PeriodicAutoregressive {
    ///     num_seasons: 2,
    ///     ar_orders: vec![1, 2],
    ///     ar_coefficients: vec![vec![0.7], vec![0.5, 0.3]],
    ///     seasonal_means: vec![100.0, 150.0],
    ///     seasonal_stds: vec![20.0, 30.0],
    /// };
    ///
    /// let params = PeriodicARParams::try_from(&temporal).unwrap();
    /// assert_eq!(params.num_seasons, 2);
    /// assert_eq!(params.seasonal_stats[0].mean, 100.0);
    /// ```
    fn try_from(temporal: &TemporalModel) -> Result<Self, Self::Error> {
        match temporal {
            TemporalModel::PeriodicAutoregressive {
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => {
                let seasonal_stats: Vec<SeasonalStats> = (0..*num_seasons)
                    .map(|m| SeasonalStats {
                        period_index: m,
                        mean: seasonal_means[m],
                        std_dev: seasonal_stds[m],
                        skewness: None, // Will be estimated from data in PAR-013
                        ar_order: ar_orders[m],
                    })
                    .collect();

                Ok(PeriodicARParams {
                    num_seasons: *num_seasons,
                    seasonal_stats,
                    ar_coefficients: ar_coefficients.clone(),
                })
            }
            _ => Err("Cannot convert non-PeriodicAutoregressive model to PeriodicARParams".to_string()),
        }
    }
}

/// Marginal distribution for stochastic processes (Schema v2)
///
/// Specifies the target marginal distribution of realizations Xₜ.
/// This replaces the ambiguous `Distribution` enum from schema v1.
///
/// # Key Concept
///
/// In the 4-stage pipeline:
/// 1. Generate Z ~ N(0,1) (base noise)
/// 2. Apply correlation (if specified)
/// 3. **Transform to marginal**: Z → X ~ F
/// 4. Apply AR dynamics (if specified)
///
/// The marginal distribution is the **target distribution** after stage 3.
///
/// # Variants
///
/// - `Normal`: Symmetric, can be negative
/// - `LogNormal3`: Non-negative, right-skewed (recommended for inflows/loads)
///
/// # Example (Normal Marginal)
/// ```json
/// {
///   "marginal_distribution": {
///     "type": "normal",
///     "mean": 100.0,
///     "std_dev": 20.0
///   }
/// }
/// ```
///
/// # Example (LogNormal3 Marginal)
/// ```json
/// {
///   "marginal_distribution": {
///     "type": "lognormal3",
///     "gamma": 1.0,
///     "mu": 4.5,
///     "sigma": 0.3
///   }
/// }
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MarginalDistribution {
    /// Normal (Gaussian) marginal distribution
    ///
    /// X ~ N(μ, σ²)
    ///
    /// **Properties**:
    /// - Symmetric around mean
    /// - Can generate negative values
    /// - Suitable for loads with symmetric uncertainty
    Normal {
        /// Mean μ
        mean: f64,
        /// Standard deviation σ (must be > 0)
        #[serde(rename = "std_dev")]
        std_dev: f64,
    },

    /// 3-parameter log-normal distribution
    ///
    /// X = γ + exp(μ + σW) where W ~ N(0,1)
    ///
    /// **Properties**:
    /// - Always non-negative (X ≥ γ ≥ 0)
    /// - Right-skewed (models rare high inflows)
    /// - Zero LP overhead (enforced in scenario generation)
    ///
    /// **Parameters**:
    /// - γ (gamma): Minimum value (typically 1-5% of typical minimum)
    /// - μ (mu): Log of typical value after shift (2.0 to 6.0)
    /// - σ (sigma): Variability (0.2 to 1.0, larger = more skewed)
    ///
    /// **Use for**: Hydro inflows, loads (physical quantities)
    #[serde(rename = "lognormal3")]
    LogNormal3 {
        /// Location parameter γ (minimum value)
        ///
        /// Must be γ ≥ 0. Typically 1-5% of historical minimum.
        gamma: f64,

        /// Log-space mean μ
        ///
        /// Controls typical value: E[X] ≈ γ + exp(μ + σ²/2)
        mu: f64,

        /// Log-space standard deviation σ
        ///
        /// Must be σ > 0. Controls skewness: larger = more right-skewed.
        sigma: f64,
    },
}

/// Innovation distribution for AR models (Schema v2)
///
/// **DEPRECATED**: Use `PeriodicAutoregressive` models with `residual_distribution` instead.
/// This struct will be removed in version 0.3.0.
///
/// Specifies the distribution of white noise innovations εₜ in AR models:
/// Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
///
/// **Important**: Innovations are always Normal in standard AR theory.
/// Non-normality is introduced through marginal transformation, not innovations.
///
/// # Example
/// ```json
/// {
///   "innovation_distribution": {
///     "mean": 0.0,
///     "std_dev": 1.0
///   }
/// }
/// ```
///
/// # Typical Values
/// - `mean`: 0.0 (zero-mean innovations)
/// - `std_dev`: 1.0 (standard normal) or calibrated value
///
/// See `docs/guides/MIGRATION-TO-PAR.md` for conversion to PAR models.
#[deprecated(
    since = "0.2.1",
    note = "Use PeriodicAutoregressive models with residual_distribution instead. \
            This struct will be removed in version 0.3.0. \
            See docs/guides/MIGRATION-TO-PAR.md for conversion guide."
)]
/// Non-negativity enforcement method for AR models
///
/// Physical quantities (inflows, loads) must be non-negative, but standard
/// AR models can generate negative values. This enum specifies how to
/// enforce non-negativity while preserving temporal correlation.
///
/// # Methods
///
/// - **None**: Allow negative values (not recommended for production)
/// - **Shadow**: Log-space transformation (recommended)
///
/// # Shadow AR Process
///
/// The shadow method applies AR dynamics in log-space:
/// 1. Transform: Y = log(X + ε)
/// 2. AR equation: Yₜ = μ + Σφᵢ Yₜ₋ᵢ + εₜ
/// 3. Recover: X = exp(Y) - ε
///
/// Where ε (shift_epsilon) ensures numerical stability near zero.
///
/// **Advantages**:
/// - Guarantees non-negativity (exp always positive)
/// - Preserves AR correlation structure
/// - Computationally efficient (O(p) per realization)
///
/// # Example (Shadow AR)
/// ```json
/// {
///   "noise_type": "autoregressive",
///   "coefficients": [0.7],
///   "distribution": {"type": "normal", "mean": 0.0, "std_dev": 15.0},
///   "non_negativity_method": {
///     "type": "shadow",
///     "shift_epsilon": 0.01
///   }
/// }
/// ```
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum NonNegativityMethod {
    /// No enforcement (allow negative values)
    ///
    /// Only use for debugging or when negative values are acceptable.
    None,

    /// **DEPRECATED**: Shadow AR process (log-space transformation with LP constraints)
    ///
    /// **⚠️ DO NOT USE**: This method is deprecated and will be removed in a future version.
    /// It adds 5-7 LP constraints per variable per stage, causing 30-50% LP solve time overhead.
    ///
    /// **Use `lognormal3` instead** for zero LP overhead and 30-50% faster performance.
    ///
    /// The shadow AR approach was based on a misunderstanding: scenarios are RHS parameters,
    /// not LP variables. Non-negativity should be enforced during scenario generation,
    /// not in the LP formulation.
    #[deprecated(
        since = "0.2.0",
        note = "Use NonNegativityMethod::LogNormal3 instead. Shadow AR adds unnecessary LP constraints."
    )]
    Shadow {
        /// Shift parameter for log transformation stability
        shift_epsilon: f64,
    },

    /// 3-parameter log-normal transformation
    ///
    /// Generates non-negative scenarios via X = γ + exp(μ + σZ) where Z ~ N(0,1).
    ///
    /// **Parameters**:
    /// - If all three (`gamma`, `mu`, `sigma`) are provided: Use explicit parameters
    /// - If all three are `None`: Future feature - will estimate from historical data (not yet implemented)
    ///
    /// **Recommended parameter ranges**:
    /// - `gamma`: 0.0 to 10.0 (minimum inflow, typically 1-5% of typical minimum)
    /// - `mu`: 2.0 to 6.0 (log of typical inflow after shift)
    /// - `sigma`: 0.2 to 1.0 (variability, larger = more right-skewed)
    ///
    /// # Example
    ///
    /// ```json
    /// {
    ///   "non_negativity_method": {
    ///     "type": "lognormal3",
    ///     "gamma": 1.0,
    ///     "mu": 4.5,
    ///     "sigma": 0.3
    ///   }
    /// }
    /// ```
    ///
    /// See [`crate::lognormal3`] module for detailed mathematical background and references.
    #[serde(rename = "lognormal3")]
    LogNormal3 {
        /// Location parameter γ (minimum value), must be >= 0
        ///
        /// If `None`, will be estimated from historical data (future feature).
        /// For now, must be explicitly provided.
        gamma: Option<f64>,

        /// Mean of log-transformed variable μ
        ///
        /// If `None`, will be estimated from historical data (future feature).
        /// For now, must be explicitly provided.
        mu: Option<f64>,

        /// Standard deviation of log-transformed variable σ, must be > 0
        ///
        /// If `None`, will be estimated from historical data (future feature).
        /// For now, must be explicitly provided.
        sigma: Option<f64>,
    },
}

/// Distribution parameters for noise models
///
/// This enum represents the statistical distribution used for generating
/// noise realizations (either directly for independent models, or as
/// innovations for AR models).
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Distribution {
    /// Normal (Gaussian) distribution
    ///
    /// Parameterized by mean (μ) and standard deviation (σ).
    /// For AR models, this is typically zero-mean (μ = 0).
    Normal {
        mean: f64,
        #[serde(rename = "std_dev")]
        std_dev: f64,
    },

    /// Log-normal distribution
    ///
    /// Parameterized by μ and σ of the underlying normal distribution.
    /// Not recommended for AR innovations (asymmetric).
    Lognormal { mu: f64, sigma: f64 },
}

/// Noise model for a single entity (hydro inflow or bus load)
///
/// Represents either independent noise or an autoregressive process
/// for a specific entity within a season.
///
/// # Performance Note
/// Using season_id instead of a Vec<usize> of nodes provides:
/// - Better memory efficiency (~16+ bytes saved per model)
/// - O(1) lookup instead of O(n) iteration
/// - Improved cache locality (no pointer chasing)
///
/// # Example (Independent Inflow)
/// ```json
/// {
///   "noise_type": "independent",
///   "uncertainty_type": "inflow",
///   "entity_id": 0,
///   "season_id": 1,
///   "distribution": {"type": "normal", "mean": 100.0, "std_dev": 20.0}
/// }
/// ```
///
/// # Example (AR(1) Inflow)
/// ```json
/// {
///   "noise_type": "autoregressive",
///   "uncertainty_type": "inflow",
///   "entity_id": 0,
///   "season_id": 1,
///   "distribution": {"type": "normal", "mean": 0.0, "std_dev": 15.0},
///   "lag_order": 1,
///   "coefficients": [0.7]
/// }
/// ```
/// Noise model for a single entity
///
/// Represents either independent noise or an autoregressive process
/// for a specific entity within a season.
///
/// # Features
/// - Separated marginal distribution from innovation distribution
/// - Explicit temporal model (independent vs AR)
/// - LogNormal3 integrated into marginal (no separate non_negativity_method)
/// - Clear semantics for 4-stage pipeline
///
/// # Example (Independent Inflow with Normal)
/// ```json
/// {
///   "uncertainty_type": "inflow",
///   "entity_id": 0,
///   "season_id": 1,
///   "marginal_distribution": {
///     "type": "normal",
///     "mean": 100.0,
///     "std_dev": 20.0
///   },
///   "temporal_model": {
///     "type": "independent"
///   }
/// }
/// ```
///
/// # Example (Independent Inflow with LogNormal3)
/// ```json
/// {
///   "uncertainty_type": "inflow",
///   "entity_id": 0,
///   "season_id": 1,
///   "marginal_distribution": {
///     "type": "lognormal3",
///     "gamma": 1.0,
///     "mu": 4.5,
///     "sigma": 0.3
///   },
///   "temporal_model": {
///     "type": "independent"
///   }
/// }
/// ```
///
/// # Example (AR(1) Inflow with LogNormal3)
/// ```json
/// {
///   "uncertainty_type": "inflow",
///   "entity_id": 0,
///   "season_id": 1,
///   "marginal_distribution": {
///     "type": "lognormal3",
///     "gamma": 1.0,
///     "mu": 4.5,
///     "sigma": 0.3
///   },
///   "innovation_distribution": {
///     "mean": 0.0,
///     "std_dev": 1.0
///   },
///   "temporal_model": {
///     "type": "autoregressive",
///     "lag_order": 1,
///     "coefficients": [0.7]
///   }
/// }
/// ```
///
/// # Distribution Semantics (CRITICAL!)
///
/// The meaning of `marginal_distribution` **changes** based on `temporal_model`:
///
/// | Temporal Model          | Distribution Applied To           | Pipeline Stage  |
/// |-------------------------|-----------------------------------|-----------------|
/// | `Independent`           | Final series Xₜ                   | Direct          |
/// | `Autoregressive`        | Innovations εₜ                    | Before AR       |
/// | `PeriodicAutoregressive`| **IGNORED** (use residual_dist)   | N/A             |
///
/// For PAR models, use `residual_distribution` instead, which is applied to
/// de-seasonalized residuals aₜ before re-seasonalization via the PAR equation.
///
/// # Pipeline for PAR
///
/// The 4-stage pipeline for Periodic AR models:
///
/// 1. **Base Noise**: Generate w ~ N(0,1)
/// 2. **Correlation**: Apply Cholesky transformation b = D·w
/// 3. **Residual Transform**: a = residual_distribution.transform(b)  ← Uses residual_dist!
/// 4. **Re-seasonalize**: Z = μₘ + σₘ·[AR_term + a]
///
/// This separation is critical: LogNormal3 is applied to **residuals** (aₜ), not final
/// values (Zₜ), ensuring non-negativity while preserving seasonal structure.
///
#[allow(deprecated)] // Struct retained for backward compatibility during soft deprecation (PAR-018)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NoiseModel {
    /// Type of uncertainty (inflow or load)
    pub uncertainty_type: UncertaintyType,

    /// Entity ID (must exist in system.json)
    ///
    /// For inflow: matches hydro_id in system.json
    /// For load: matches bus_id in system.json
    pub entity_id: usize,

    /// Season ID where this noise model applies
    ///
    /// Must match a season_id present in graph nodes.
    /// The noise model will apply to all nodes in this season.
    pub season_id: usize,

    /// Probability distribution (unified field, v0.3.0+)
    ///
    /// **Required field for all temporal models.**
    ///
    /// # Semantics by Temporal Model
    ///
    /// - **Independent**: Marginal distribution of final series Xₜ
    ///   - Directly sampled: Xₜ ~ distribution
    ///
    /// - **PeriodicAutoregressive**: Residual distribution of innovations aₜ
    ///   - After de-seasonalization: aₜ ~ distribution
    ///   - Final series: Zₜ = μₘ + σₘ·[∑φₖₘ·aₜ₋ₖ + aₜ]
    ///
    /// # Example (Independent)
    ///
    /// ```json
    /// {
    ///   "distribution": { "type": "normal", "mean": 100.0, "std_dev": 20.0 },
    ///   "temporal_model": { "type": "independent" }
    /// }
    /// ```
    ///
    /// # Example (PAR)
    ///
    /// ```json
    /// {
    ///   "distribution": { "type": "lognormal3", "gamma": 1.0, "mu": 0.0, "sigma": 0.6 },
    ///   "temporal_model": { "type": "periodic_ar", "num_seasons": 12, ... }
    /// }
    /// ```
    pub distribution: MarginalDistribution,

    /// Temporal model (independent or periodic AR)
    ///
    /// Specifies the temporal correlation structure:
    /// - `Independent`: No temporal correlation
    /// - `PeriodicAutoregressive`: PAR(p) with seasonal parameters
    pub temporal_model: TemporalModel,
}

impl NoiseModel {
    /// Validate NoiseModel semantic constraints
    ///
    /// Ensures the distribution field is present.
    ///
    /// # Errors
    ///
    /// Returns descriptive error string if validation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = NoiseModel { /* ... */ };
    /// model.validate()?;
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        // Validation is now minimal since only Independent and PAR are supported
        // and distribution is a required field
        Ok(())
    }

    /// Get the distribution target for scenario generation pipeline
    ///
    /// Returns the appropriate distribution based on temporal model semantics.
    /// This method routes to the correct distribution for each pipeline stage.
    ///
    /// # Distribution Routing (v0.3.0+)
    ///
    /// Uses the unified `distribution` field. Semantics depend on temporal model:
    ///
    /// - **Independent**: `distribution` is marginal of final series Xₜ
    /// - **PeriodicAutoregressive**: `distribution` is residuals aₜ
    ///
    /// # Performance
    ///
    /// O(1) - simple enum match with reference return
    ///
    /// # Example
    ///
    /// ```ignore
    /// let noise_model = NoiseModel { /* PAR config */ };
    /// match noise_model.get_distribution_target() {
    ///     DistributionTarget::Residuals(dist) => {
    ///         // Apply to de-seasonalized residuals
    ///     }
    ///     _ => { /* ... */ }
    /// }
    /// ```
    pub fn get_distribution_target(&self) -> DistributionTarget<'_> {
        match &self.temporal_model {
            TemporalModel::Independent => {
                DistributionTarget::FinalSeries(&self.distribution)
            }
            TemporalModel::PeriodicAutoregressive { .. } => {
                DistributionTarget::Residuals(&self.distribution)
            }
        }
    }
}

/// Target for distribution application in scenario generation pipeline
///
/// Routes to the correct distribution based on temporal model semantics.
/// This is an internal helper for the scenario generation pipeline.
///
/// # Semantics
///
/// - **FinalSeries**: Distribution applied directly to output (Independent models)
/// - **Residuals**: Distribution applied to PAR residuals aₜ (PeriodicAutoregressive models)
///
/// # Performance
///
/// Uses references to avoid cloning distributions. The lifetime 'a ties the
/// reference to the source NoiseModel, ensuring zero-cost abstraction.
#[derive(Debug)]
pub enum DistributionTarget<'a> {
    /// Marginal distribution for final series (Independent models)
    FinalSeries(&'a MarginalDistribution),

    /// Distribution for residuals in PAR models
    Residuals(&'a MarginalDistribution),
}

// ============================================================================
// New Format: UncertaintySpecification (v0.5.0+)
// ============================================================================

/// New uncertainty specification format (v0.5.0+)
///
/// This is the **recommended format** for specifying uncertainties. It provides:
/// - One entity = one specification (no scattered multi-season entries)
/// - Clear separation: temporal model vs marginal distribution
/// - Explicit seasonal_distributions for independent models
/// - No misleading season_id at root level for PAR models
///
/// # Format Comparison
///
/// **Old format (noise_models)**: PAR model scattered across seasons
/// **New format (uncertainty_specifications)**: One clear entity-level specification
///
/// See `docs/migration/PAR_INPUT_FORMAT.md` for detailed comparison and migration guide.
///
/// # Example (PAR Model)
///
/// ```json
/// {
///   "uncertainty_type": "inflow",
///   "entity_id": 0,
///   "temporal_model": {
///     "type": "periodic_ar",
///     "num_seasons": 12,
///     "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
///     "ar_coefficients": [[0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]],
///     "seasonal_means": [90, 100, 120, 150, 180, 200, 180, 150, 120, 100, 85, 90],
///     "seasonal_stds": [20, 22, 25, 30, 35, 40, 35, 30, 25, 22, 18, 20]
///   },
///   "marginal_distribution": {
///     "type": "lognormal3",
///     "gamma": 1.0,
///     "mu": 4.5,
///     "sigma": 0.3
///   }
/// }
/// ```
///
/// # Example (Independent Model)
///
/// ```json
/// {
///   "uncertainty_type": "load",
///   "entity_id": 0,
///   "temporal_model": { "type": "independent" },
///   "seasonal_distributions": [
///     { "season_id": 0, "mean": 100.0, "std_dev": 20.0 },
///     { "season_id": 1, "mean": 110.0, "std_dev": 22.0 }
///   ]
/// }
/// ```
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct UncertaintySpecification {
    /// Type of uncertainty (inflow or load)
    pub uncertainty_type: UncertaintyType,

    /// Entity ID (zero-based index)
    ///
    /// For inflow: matches hydro_id in system.json
    /// For load: matches bus_id in system.json
    pub entity_id: usize,

    /// Temporal model (independent or periodic AR)
    pub temporal_model: TemporalModelInput,

    /// Marginal distribution for PAR models (entity-level)
    ///
    /// **Required for PAR models**, omit for independent models.
    ///
    /// Applied to residuals after de-seasonalization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marginal_distribution: Option<MarginalDistribution>,

    /// Seasonal distributions for independent models (per-season)
    ///
    /// **Required for independent models**, omit for PAR models.
    ///
    /// Each season must be specified with its mean and std_dev.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seasonal_distributions: Option<Vec<SeasonalDistribution>>,
}

/// Seasonal distribution parameters for independent models
///
/// Used in the new format to explicitly specify per-season parameters
/// for independent (non-correlated) temporal models. Supports both Normal
/// and LogNormal3 distributions.
///
/// # Example (Normal Distribution)
///
/// ```json
/// {
///   "season_id": 0,
///   "distribution": {
///     "type": "normal",
///     "mean": 100.0,
///     "std_dev": 20.0
///   }
/// }
/// ```
///
/// # Example (LogNormal3 Distribution)
///
/// ```json
/// {
///   "season_id": 1,
///   "distribution": {
///     "type": "lognormal3",
///     "gamma": 1.0,
///     "mu": 4.5,
///     "sigma": 0.3
///   }
/// }
/// ```
///
/// # Distribution Selection
///
/// - **Normal**: Use for symmetric uncertainties that can be negative (e.g., loads with small variance)
/// - **LogNormal3**: Use for non-negative uncertainties with right skew (e.g., inflows)
///
/// See `schemas/recourse.schema.json` for full specification.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct SeasonalDistribution {
    /// Season ID (must match season_id in graph nodes)
    pub season_id: usize,

    /// Distribution for this season (Normal or LogNormal3)
    #[serde(flatten)]
    pub distribution: MarginalDistribution,
}

impl SeasonalDistribution {
    /// Convert to internal `SeasonalNoiseParams` representation
    ///
    /// For Normal distributions, stores mean/std_dev directly with no override.
    /// For LogNormal3, computes mean/std_dev from lognormal parameters and stores
    /// the original distribution in `marginal_override` for the transformation pipeline.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Normal distribution
    /// let normal = SeasonalDistribution {
    ///     season_id: 0,
    ///     distribution: MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 }
    /// };
    /// let params = normal.to_seasonal_params();
    /// assert_eq!(params.mean, 100.0);
    /// assert_eq!(params.std_dev, 20.0);
    /// assert!(params.marginal_override.is_none());
    ///
    /// // LogNormal3 distribution
    /// let lognormal = SeasonalDistribution {
    ///     season_id: 1,
    ///     distribution: MarginalDistribution::LogNormal3 { gamma: 1.0, mu: 4.5, sigma: 0.3 }
    /// };
    /// let params = lognormal.to_seasonal_params();
    /// assert!(params.marginal_override.is_some());
    /// ```
    pub fn to_seasonal_params(
        &self,
    ) -> crate::unified_noise_spec::SeasonalNoiseParams {
        use crate::unified_noise_spec::SeasonalNoiseParams;

        match &self.distribution {
            MarginalDistribution::Normal { mean, std_dev } => {
                // Normal: use mean/std_dev directly, no override needed
                SeasonalNoiseParams {
                    mean: *mean,
                    std_dev: *std_dev,
                    marginal_override: None,
                }
            }
            MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                // LogNormal3: compute mean/std_dev from lognormal parameters
                // and store the distribution for marginal transformation

                // PERFORMANCE: These computations are done once at initialization,
                // not in the hot path. The cost (~50ns) is negligible.

                // Mean: E[X] = γ + exp(μ + σ²/2)
                let mean = gamma + (mu + sigma.powi(2) / 2.0).exp();

                // Variance: Var(X) = exp(2μ + σ²) × (exp(σ²) - 1)
                let variance = (2.0 * mu + sigma.powi(2)).exp()
                    * (sigma.powi(2).exp() - 1.0);
                let std_dev = variance.sqrt();

                SeasonalNoiseParams {
                    mean,
                    std_dev,
                    marginal_override: Some(self.distribution.clone()),
                }
            }
        }
    }
}

/// Temporal model input format (public-facing)
///
/// This enum is used in the new `UncertaintySpecification` format.
/// It has the same structure as `TemporalModel` but is separate to allow
/// for future extensions to the public API without breaking internal code.
///
/// # Example (Independent)
///
/// ```json
/// { "type": "independent" }
/// ```
///
/// # Example (Periodic AR)
///
/// ```json
/// {
///   "type": "periodic_ar",
///   "num_seasons": 12,
///   "ar_orders": [1, 1, 1, ...],
///   "ar_coefficients": [[0.7], [0.7], ...],
///   "seasonal_means": [90, 100, 120, ...],
///   "seasonal_stds": [20, 22, 25, ...]
/// }
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TemporalModelInput {
    /// Independent process (no temporal correlation)
    Independent,

    /// Periodic Autoregressive PAR(p) model
    #[serde(rename = "periodic_ar")]
    PeriodicAr {
        /// Seasonal cycle length (e.g., 12 for monthly, 4 for quarterly)
        num_seasons: usize,

        /// AR order for each season
        ar_orders: Vec<usize>,

        /// AR coefficients for each season
        ar_coefficients: Vec<Vec<f64>>,

        /// Seasonal mean for each season
        seasonal_means: Vec<f64>,

        /// Seasonal standard deviation for each season
        seasonal_stds: Vec<f64>,
    },
}

// ============================================================================
// Correlation Infrastructure
// ============================================================================

/// Correlation specification for multi-variate scenario generation
///
/// Enables realistic spatial and physical correlations (e.g., upstream/downstream
/// hydro correlation, regional load correlation during weather events).
///
/// Uses Gaussian copula with Cholesky decomposition to preserve marginal distributions
/// while introducing specified correlation structure.
///
/// # Example
///
/// ```json
/// {
///   "method": "cholesky",
///   "blocks": [
///     {
///       "name": "cascade_correlation",
///       "entities": [
///         { "entity_type": "inflow", "entity_id": 0 },
///         { "entity_type": "inflow", "entity_id": 1 }
///       ],
///       "correlation_matrix": [
///         [1.0, 0.8],
///         [0.8, 1.0]
///       ]
///     }
///   ]
/// }
/// ```
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct CorrelationSpecification {
    /// Correlation method to use
    pub method: CorrelationMethod,

    /// Correlation blocks (groups of correlated entities)
    ///
    /// Each block defines correlation among a subset of uncertainties.
    /// Entities not in any block are assumed independent.
    pub blocks: Vec<CorrelationBlock>,
}

/// Correlation method for multi-variate sampling
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CorrelationMethod {
    /// Independent sampling (no correlation)
    ///
    /// Backward compatible default. Each uncertainty sampled independently.
    None,

    /// Gaussian copula with Cholesky decomposition (recommended)
    ///
    /// Uses Cholesky factorization for O(n²) sampling per scenario.
    /// Preserves marginal distributions while introducing correlation.
    ///
    /// Production-proven approach used in SDDP.jl, SPTcpp, PSR SDDP.
    Cholesky,
}

/// Correlation block defining correlation among a group of entities
///
/// # Validation Rules
///
/// - `correlation_matrix` must be symmetric
/// - `correlation_matrix` must be positive semi-definite
/// - Diagonal elements must be 1.0
/// - Off-diagonal elements must be in [-1, 1]
/// - Matrix dimension must match number of entities
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct CorrelationBlock {
    /// Block name (for documentation and error messages)
    pub name: String,

    /// Entities in this correlation block
    ///
    /// Order must match row/column order in correlation_matrix
    pub entities: Vec<EntityReference>,

    /// Correlation matrix (symmetric, PSD, diagonal=1)
    ///
    /// Must be n×n where n = entities.len()
    pub correlation_matrix: Vec<Vec<f64>>,
}

/// Reference to an uncertain entity (hydro inflow, bus load, etc.)
///
/// Used to specify which uncertainties are correlated in a CorrelationBlock
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct EntityReference {
    /// Type of uncertainty (inflow, load, etc.)
    pub uncertainty_type: UncertaintyType,

    /// Entity ID (zero-based index)
    pub entity_id: usize,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Recourse {
    pub initial_condition: InitialConditionInput,

    /// Uncertainty specifications format (v0.3.0+)
    ///
    /// This is the format for specifying uncertainties. It provides:
    /// - One entity = one specification
    /// - Clear separation: temporal model vs marginal distribution
    /// - Explicit seasonal_distributions for independent models
    pub uncertainty_specifications: Vec<UncertaintySpecification>,

    /// Correlation specification for multi-variate scenario generation
    ///
    /// Optional. If omitted, all uncertainties are sampled independently.
    ///
    /// When specified, defines correlation structure across entities
    /// (e.g., upstream/downstream hydro correlation, regional loads).
    ///
    /// Uses Gaussian copula with Cholesky decomposition to preserve
    /// marginal distributions while introducing correlation.
    #[serde(default)]
    pub correlation: Option<CorrelationSpecification>,
}

pub fn read_recourse_input(filepath: &str) -> Recourse {
    let contents = fs::read_to_string(filepath)
        .expect("Error while reading recourse file");
    let parsed: Recourse = serde_json::from_str(&contents).unwrap();
    parsed
}

impl Recourse {
    pub fn build_sddp_initial_condition(
        &self,
    ) -> initial_condition::InitialCondition {
        let initial_condition_hydro_ids: Vec<usize> = self
            .initial_condition
            .storage
            .iter()
            .map(|s| s.hydro_id)
            .collect();
        validate_id_range(&initial_condition_hydro_ids, "initial storages");
        let num_hydros = initial_condition_hydro_ids.len();
        let mut storage = Vec::<f64>::with_capacity(num_hydros);
        for id in 0..num_hydros {
            let s = self
                .initial_condition
                .storage
                .iter()
                .find(|s| s.hydro_id == id)
                .unwrap();
            storage.push(s.value);
        }

        // Build lag inflows Vec<Vec<f64>> from Vec<PastInflow>
        // inflow[hydro_id][lag-1] = lag value (lag indices are 1-based in input)
        let inflow = if self.initial_condition.inflow.is_empty() {
            vec![]
        } else {
            // Group PastInflow by hydro_id and order by lag
            let mut inflow = vec![vec![]; num_hydros];
            for past_inflow in &self.initial_condition.inflow {
                let hydro_id = past_inflow.hydro_id;
                if hydro_id < num_hydros {
                    // Ensure capacity for lag index (lag-1 for 0-based indexing)
                    let lag_idx = past_inflow.lag.saturating_sub(1);
                    if inflow[hydro_id].len() <= lag_idx {
                        inflow[hydro_id].resize(lag_idx + 1, 0.0);
                    }
                    inflow[hydro_id][lag_idx] = past_inflow.value;
                }
            }
            inflow
        };

        initial_condition::InitialCondition::new(storage, inflow)
    }

    /// Validate that uncertainty_specifications is specified
    ///
    /// Returns `Ok(())` if validation passes, `Err(msg)` otherwise.
    ///
    /// # Errors
    ///
    /// - Missing `uncertainty_specifications`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let recourse = read_recourse_input("recourse.json");
    /// recourse.validate_format()?;
    /// ```
    pub fn validate_format(&self) -> Result<(), String> {
        if self.uncertainty_specifications.is_empty() {
            return Err(
                "Field 'uncertainty_specifications' in recourse.json is empty."
                    .into(),
            );
        }
        Ok(())
    }

    /// Get unified noise specs from uncertainty_specifications
    ///
    /// Converts from new format to internal `UnifiedNoiseSpec`.
    ///
    /// # Returns
    ///
    /// Vector of `UnifiedNoiseSpec` for internal use
    ///
    /// # Errors
    ///
    /// - Format validation fails
    /// - Conversion fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let recourse = read_recourse_input("recourse.json");
    /// let unified_specs = recourse.get_unified_specs()?;
    /// ```
    pub fn get_unified_specs(
        &self,
    ) -> Result<Vec<crate::unified_noise_spec::UnifiedNoiseSpec>, String> {
        self.validate_format()?;

        Self::convert_uncertainty_specifications(
            &self.uncertainty_specifications,
        )
    }

    /// Convert new format to internal UnifiedNoiseSpec
    ///
    /// # Arguments
    ///
    /// * `specs` - New format uncertainty specifications
    ///
    /// # Returns
    ///
    /// Vector of `UnifiedNoiseSpec` for internal use
    ///
    /// # Errors
    ///
    /// - Invalid specification (e.g., independent model missing seasonal_distributions)
    /// - PAR model missing marginal_distribution
    /// - Inconsistent data
    fn convert_uncertainty_specifications(
        specs: &[UncertaintySpecification],
    ) -> Result<Vec<crate::unified_noise_spec::UnifiedNoiseSpec>, String> {
        let mut unified_specs = Vec::new();

        for spec in specs {
            let (temporal_model, seasonal_params, marginal_distribution) =
                match &spec.temporal_model {
                    TemporalModelInput::Independent => {
                        // Independent model: requires seasonal_distributions
                        let seasonal_dists = spec.seasonal_distributions.as_ref()
                            .ok_or_else(|| {
                                format!(
                                    "Independent model for entity {} requires 'seasonal_distributions' field",
                                    spec.entity_id
                                )
                            })?;

                        let mut seasonal_params = HashMap::new();
                        for dist in seasonal_dists {
                            seasonal_params.insert(
                                dist.season_id,
                                dist.to_seasonal_params(),
                            );
                        }

                        (
                            TemporalModelSpec::Independent,
                            seasonal_params,
                            None, // Independent models don't have entity-level marginal
                        )
                    }
                    TemporalModelInput::PeriodicAr {
                        num_seasons,
                        ar_orders,
                        ar_coefficients,
                        seasonal_means,
                        seasonal_stds,
                    } => {
                        // PAR model: requires marginal_distribution
                        let marginal = spec.marginal_distribution.clone()
                            .ok_or_else(|| {
                                format!(
                                    "PAR model for entity {} requires 'marginal_distribution' field",
                                    spec.entity_id
                                )
                            })?;

                        // Build seasonal_params HashMap
                        let mut seasonal_params =
                            HashMap::with_capacity(*num_seasons);
                        let mut seasonal_ar_params =
                            HashMap::with_capacity(*num_seasons);

                        for season_id in 0..*num_seasons {
                            seasonal_params.insert(
                                season_id,
                                SeasonalNoiseParams {
                                    mean: seasonal_means[season_id],
                                    std_dev: seasonal_stds[season_id],
                                    marginal_override: None, // Use entity-level marginal
                                },
                            );

                            seasonal_ar_params.insert(
                                season_id,
                                crate::unified_noise_spec::SeasonalPARParams {
                                    ar_order: ar_orders[season_id],
                                    ar_coefficients: ar_coefficients[season_id]
                                        .clone(),
                                },
                            );
                        }

                        (
                            TemporalModelSpec::PeriodicAutoregressive {
                                num_seasons: *num_seasons,
                                seasonal_ar_params,
                            },
                            seasonal_params,
                            Some(marginal),
                        )
                    }
                };

            unified_specs.push(UnifiedNoiseSpec {
                uncertainty_type: spec.uncertainty_type.clone(),
                entity_id: spec.entity_id,
                temporal_model,
                seasonal_params,
                marginal_distribution,
            });
        }

        Ok(unified_specs)
    }

    /// Convert old format to new format (migration helper)
    ///
    /// Takes ownership of `noise_models` and populates `uncertainty_specifications`.
    /// This is a one-way conversion for migration purposes.
    ///
    /// # Algorithm
    ///
    /// 1. Take ownership of `noise_models` (leaving it as None)
    /// 2. Convert old format → `UnifiedNoiseSpec` (internal representation)
    /// 3. Convert `UnifiedNoiseSpec` → `UncertaintySpecification` (new format)
    /// 4. Store result in `uncertainty_specifications`
    ///
    /// # Returns
    ///
    /// - `Ok(())`: Migration successful, `uncertainty_specifications` populated
    ///
    /// Generate SAA scenarios using the new 4-stage ScenarioGenerator pipeline.
    ///
    /// This method replaces the old NodeNoiseGenerator approach with the new
    /// pipeline that supports AR temporal models, correlation,
    /// and proper marginal transformations.
    ///
    /// Scenarios are generated per-season: noise_models are filtered by each
    /// stage's season_id to handle season-specific uncertainty distributions.
    ///
    /// # Arguments
    ///
    /// * `g` - The scenario tree graph with NodeData containing num_scenarios per node
    /// * `initial_condition` - Initial condition (domain model, already converted from input)
    /// * `seed` - RNG seed for deterministic scenario generation
    ///
    /// # Returns
    ///
    /// SAA structure with scenarios for all stages and nodes
    ///
    /// # Panics
    ///
    /// Panics if no uncertainty specifications are found or if ScenarioGenerator construction fails
    pub fn generate_sddp_noises(
        &self,
        g: &graph::DirectedGraph<sddp::NodeData>,
        initial_condition: &initial_condition::InitialCondition,
        seed: u64,
    ) -> scenario::SAA {
        // Get unified specs (new format only)
        let unified_specs = self
            .get_unified_specs()
            .expect("uncertainty_specifications is required");

        // Count entities from graph for cache sizing
        let num_hydros = unified_specs
            .iter()
            .filter(|s| s.uncertainty_type == UncertaintyType::Inflow)
            .map(|s| s.entity_id)
            .max()
            .map(|id| id + 1)
            .unwrap_or(0);
        let num_loads = unified_specs
            .iter()
            .filter(|s| s.uncertainty_type == UncertaintyType::Load)
            .map(|s| s.entity_id)
            .max()
            .map(|id| id + 1)
            .unwrap_or(0);

        // Count seasons from graph
        let num_seasons = g
            .iter_nodes()
            .map(|node| node.data.season_id)
            .max()
            .map(|max_season| max_season + 1)
            .unwrap_or(1);

        // Build cache with pre-initialized PAR generators
        let cache =
            crate::noise_model_cache::NoiseModelCache::from_unified_specs(
                &unified_specs,
                initial_condition,
                num_hydros,
                num_loads,
                num_seasons,
            )
            .expect("Failed to build noise model cache");

        // Generate scenarios using the cache
        self.generate_sddp_noises_with_cache(&cache, g, seed)
    }

    /// Generate SDDP scenarios using pre-initialized NoiseModelCache (TICKET-13 fast path)
    ///
    /// # Arguments
    /// * `cache` - Pre-initialized cache with PAR generators and distributions
    /// * `g` - Graph with stage/season/scenario information
    /// * `seed` - Base RNG seed (varied per stage)
    ///
    /// # Performance
    /// This is 5-10% faster than the legacy path because PAR generators are pre-initialized
    /// and distributions are cached, eliminating repeated validation/allocation.
    fn generate_sddp_noises_with_cache(
        &self,
        cache: &crate::noise_model_cache::NoiseModelCache,
        g: &graph::DirectedGraph<sddp::NodeData>,
        seed: u64,
    ) -> scenario::SAA {
        // Determine num_stages from graph nodes
        let num_stages = g
            .iter_nodes()
            .map(|node| node.data.stage_id)
            .max()
            .map(|max_stage| max_stage + 1)
            .unwrap_or(1);

        // Build stage info: (stage_id, season_id, num_scenarios)
        let mut stage_info: Vec<(usize, usize, usize)> = Vec::new();
        for node in g.iter_nodes() {
            let stage = node.data.stage_id;
            if stage < num_stages
                && !stage_info.iter().any(|(s, _, _)| *s == stage)
            {
                stage_info.push((
                    stage,
                    node.data.season_id,
                    node.data.num_scenarios,
                ));
            }
        }
        stage_info.sort_by_key(|(stage, _, _)| *stage);

        // Initialize empty SAA
        let mut saa = scenario::SAA::new_empty();

        // Generate scenarios stage-by-stage using cache
        for (stage_id, season_id, num_scenarios) in stage_info {
            // PERFORMANCE: O(1) lookup in pre-built cache
            // Vary RNG seed per stage for independent samples
            use rand::SeedableRng;
            let mut rng =
                rand::rngs::StdRng::seed_from_u64(seed + stage_id as u64);
            let stage_scenarios = cache.generate_stage_scenarios(
                stage_id,
                season_id,
                num_scenarios,
                &mut rng,
            );

            // Convert to SAA format
            let mut branching_noises = Vec::with_capacity(num_scenarios);
            for scenario_id in 0..num_scenarios {
                branching_noises.push(scenario::SampledBranchingNoises {
                    inflow_noises: stage_scenarios.inflows[scenario_id].clone(),
                    load_noises: stage_scenarios.loads[scenario_id].clone(),
                    num_inflow_entities: stage_scenarios.inflows[scenario_id]
                        .len(),
                    num_load_entities: stage_scenarios.loads[scenario_id].len(),
                });
            }

            // Add to SAA
            while saa.branching_samples.len() <= stage_id {
                saa.branching_samples.push(scenario::SampledNodeBranchings {
                    num_branchings: 0,
                    branching_noises: vec![],
                });
            }
            saa.branching_samples[stage_id] = scenario::SampledNodeBranchings {
                num_branchings: num_scenarios,
                branching_noises,
            };
        }

        // Build index samplers for simulation
        saa.index_samplers = saa
            .branching_samples
            .iter()
            .map(|sample| {
                rand_distr::Uniform::<usize>::try_from(0..sample.num_branchings)
                    .unwrap()
            })
            .collect();

        saa
    }
}

pub struct Input {
    pub config: Config,
    pub system: SystemInput,
    pub graph: GraphInput,
    pub recourse: Recourse,
}

impl Input {
    pub fn build(path: &str) -> Self {
        let config = read_config_input(&(path.to_owned() + "/config.json"));
        let system = read_system_input(&(path.to_owned() + "/system.json"));
        let graph = read_graph_input(&(path.to_owned() + "/graph.json"));
        let recourse =
            read_recourse_input(&(path.to_owned() + "/recourse.json"));

        // Validate all inputs before expensive computation (fail-fast on first error)
        use crate::input_validation::InputValidator;
        if let Err(e) =
            InputValidator::validate_all(&config, &system, &graph, &recourse)
        {
            eprintln!("Input validation failed:\n{}", e);
            std::process::exit(1);
        }

        Self {
            config,
            system,
            graph,
            recourse,
        }
    }

    /// Load inputs from individual file paths with validation
    ///
    /// This method loads and validates all input files before returning.
    /// If validation fails, returns a descriptive error.
    ///
    /// # Errors
    ///
    /// Returns `PowersError` if:
    /// - Any file cannot be read or parsed
    /// - Validation fails (missing references, invalid constraints, etc.)
    pub fn from_paths(
        config_path: &std::path::Path,
        system_path: &std::path::Path,
        graph_path: &std::path::Path,
        recourse_path: &std::path::Path,
    ) -> Result<Self, crate::error::PowersError> {
        use crate::error::IoError;

        // Read config with error handling
        let config_str = config_path.to_str().ok_or_else(|| {
            Box::new(IoError::GenericIoError {
                path: format!("{:?}", config_path),
                error: "Invalid UTF-8 in path".to_string(),
                suggestion: "Ensure file paths use valid UTF-8 characters"
                    .to_string(),
            })
        })?;
        let config_contents = fs::read_to_string(config_str).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Box::new(IoError::FileNotFound {
                    path: config_str.to_string(),
                    current_dir: std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                })
            } else {
                Box::new(IoError::GenericIoError {
                    path: config_str.to_string(),
                    error: e.to_string(),
                    suggestion: "Check file permissions and path".to_string(),
                })
            }
        })?;
        let config: Config = serde_json::from_str(&config_contents).map_err(|e| {
            Box::new(IoError::GenericIoError {
                path: config_str.to_string(),
                error: format!("JSON parse error: {}", e),
                suggestion: "Check JSON syntax - missing commas, brackets, or quotes".to_string(),
            })
        })?;

        // Read system with error handling
        let system_str = system_path.to_str().ok_or_else(|| {
            Box::new(IoError::GenericIoError {
                path: format!("{:?}", system_path),
                error: "Invalid UTF-8 in path".to_string(),
                suggestion: "Ensure file paths use valid UTF-8 characters"
                    .to_string(),
            })
        })?;
        let system_contents = fs::read_to_string(system_str).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Box::new(IoError::FileNotFound {
                    path: system_str.to_string(),
                    current_dir: std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                })
            } else {
                Box::new(IoError::GenericIoError {
                    path: system_str.to_string(),
                    error: e.to_string(),
                    suggestion: "Check file permissions and path".to_string(),
                })
            }
        })?;
        let system: SystemInput = serde_json::from_str(&system_contents).map_err(|e| {
            Box::new(IoError::GenericIoError {
                path: system_str.to_string(),
                error: format!("JSON parse error: {}", e),
                suggestion: "Check JSON syntax - missing commas, brackets, or quotes".to_string(),
            })
        })?;

        // Read graph with error handling
        let graph_str = graph_path.to_str().ok_or_else(|| {
            Box::new(IoError::GenericIoError {
                path: format!("{:?}", graph_path),
                error: "Invalid UTF-8 in path".to_string(),
                suggestion: "Ensure file paths use valid UTF-8 characters"
                    .to_string(),
            })
        })?;
        let graph_contents = fs::read_to_string(graph_str).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Box::new(IoError::FileNotFound {
                    path: graph_str.to_string(),
                    current_dir: std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                })
            } else {
                Box::new(IoError::GenericIoError {
                    path: graph_str.to_string(),
                    error: e.to_string(),
                    suggestion: "Check file permissions and path".to_string(),
                })
            }
        })?;
        let graph: GraphInput = serde_json::from_str(&graph_contents).map_err(|e| {
            Box::new(IoError::GenericIoError {
                path: graph_str.to_string(),
                error: format!("JSON parse error: {}", e),
                suggestion: "Check JSON syntax - missing commas, brackets, or quotes".to_string(),
            })
        })?;

        // Read recourse with error handling
        let recourse_str = recourse_path.to_str().ok_or_else(|| {
            Box::new(IoError::GenericIoError {
                path: format!("{:?}", recourse_path),
                error: "Invalid UTF-8 in path".to_string(),
                suggestion: "Ensure file paths use valid UTF-8 characters"
                    .to_string(),
            })
        })?;
        let recourse_contents =
            fs::read_to_string(recourse_str).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Box::new(IoError::FileNotFound {
                        path: recourse_str.to_string(),
                        current_dir: std::env::current_dir()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| "<unknown>".to_string()),
                    })
                } else {
                    Box::new(IoError::GenericIoError {
                        path: recourse_str.to_string(),
                        error: e.to_string(),
                        suggestion: "Check file permissions and path"
                            .to_string(),
                    })
                }
            })?;
        let recourse: Recourse = serde_json::from_str(&recourse_contents).map_err(|e| {
            Box::new(IoError::GenericIoError {
                path: recourse_str.to_string(),
                error: format!("JSON parse error: {}", e),
                suggestion: "Check JSON syntax - missing commas, brackets, or quotes".to_string(),
            })
        })?;

        InputValidator::validate_all(&config, &system, &graph, &recourse)?;

        Ok(Self {
            config,
            system,
            graph,
            recourse,
        })
    }
}

// TODO: Re-enable and update tests after removing deprecated AR functionality
/*
#[cfg(test)]
#[allow(deprecated)] // Tests use deprecated fields for legacy format testing
mod tests {
    use super::*;

    #[test]
    fn test_read_config() {
        let filepath = "examples/01-deterministic/config.json";
        let config = read_config_input(filepath);
        assert_eq!(config.num_iterations, 10);
        assert_eq!(config.num_simulation_scenarios, Some(1));
    }

    #[test]
    fn test_config_deserialize_without_output_path() {
        // Test that missing output_path defaults to None
        let json = r#"{
            "num_iterations": 100,
            "num_forward_passes": 20,
            "num_simulation_scenarios": 1000,
            "seed": 42
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_iterations, 100);
        assert_eq!(config.seed, 42);
        assert!(config.output_path.is_none());
    }

    #[test]
    fn test_config_deserialize_with_output_path() {
        // Test that output_path is properly deserialized
        let json = r#"{
            "num_iterations": 100,
            "num_forward_passes": 20,
            "num_simulation_scenarios": 1000,
            "seed": 42,
            "output_path": "./test_output"
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.output_path, Some("./test_output".to_string()));
    }

    #[test]
    fn test_config_deserialize_with_null_output_path() {
        // Test that explicit null output_path becomes None
        let json = r#"{
            "num_iterations": 100,
            "num_forward_passes": 20,
            "num_simulation_scenarios": 1000,
            "seed": 42,
            "output_path": null
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(config.output_path.is_none());
    }

    #[test]
    fn test_config_with_null_simulation() {
        // Test that explicit null num_simulation_scenarios becomes None
        let json = r#"{
            "num_iterations": 100,
            "num_forward_passes": 20,
            "num_simulation_scenarios": null,
            "seed": 42
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_iterations, 100);
        assert_eq!(config.num_forward_passes, 20);
        assert_eq!(config.num_simulation_scenarios, None);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_config_without_simulation_field() {
        // Test that missing num_simulation_scenarios defaults to None
        let json = r#"{
            "num_iterations": 100,
            "num_forward_passes": 20,
            "seed": 42
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_iterations, 100);
        assert_eq!(config.num_forward_passes, 20);
        assert_eq!(config.num_simulation_scenarios, None);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_config_with_simulation_value() {
        // Test that integer num_simulation_scenarios becomes Some(n)
        let json = r#"{
            "num_iterations": 100,
            "num_forward_passes": 20,
            "num_simulation_scenarios": 500,
            "seed": 42
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_iterations, 100);
        assert_eq!(config.num_forward_passes, 20);
        assert_eq!(config.num_simulation_scenarios, Some(500));
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_read_system() {
        let filepath = "examples/01-deterministic/system.json";
        let system = read_system_input(filepath);
        assert_eq!(system.buses.len(), 1);
        assert_eq!(system.lines.len(), 0);
        assert_eq!(system.thermals.len(), 1);
        assert_eq!(system.hydros.len(), 1);
    }

    #[test]
    fn test_build_sddp_system() {
        let filepath = "examples/01-deterministic/system.json";
        let system = read_system_input(filepath);
        system.build_sddp_system();
    }

    #[test]
    fn test_read_recourse() {
        let filepath = "examples/01-deterministic/recourse.json";
        let recourse = read_recourse_input(filepath);
        assert_eq!(recourse.initial_condition.storage.len(), 1);
        // Check that noise_models format is used
        assert_eq!(recourse.noise_models.len(), 4); // 2 load + 2 inflow
    }

    #[test]
    fn test_read_input() {
        let path = "examples/01-deterministic";
        let input = Input::build(path);
        assert_eq!(input.config.num_iterations, 10);
    }

    #[test]
    fn test_validate_id_range_valid() {
        // Test with valid sequential IDs starting from 0
        let ids = vec![0, 1, 2, 3];
        validate_id_range(&ids, "test_elements");
        // Should not panic
    }

    #[test]
    fn test_validate_id_range_empty() {
        // Test with empty slice (valid edge case)
        let ids: Vec<usize> = vec![];
        validate_id_range(&ids, "test_elements");
        // Should not panic
    }

    #[test]
    fn test_validate_id_range_single() {
        // Test with single element
        let ids = vec![0];
        validate_id_range(&ids, "test_elements");
        // Should not panic
    }

    #[test]
    #[should_panic(expected = "ID 0 not found for test_elements")]
    fn test_validate_id_range_invalid_start() {
        // Test with IDs not starting from 0
        let ids = vec![1, 2, 3];
        validate_id_range(&ids, "test_elements");
    }

    #[test]
    #[should_panic(expected = "ID 2 not found for test_elements")]
    fn test_validate_id_range_gap() {
        // Test with gap in ID sequence
        let ids = vec![0, 1, 3]; // Missing 2
        validate_id_range(&ids, "test_elements");
    }

    #[test]
    #[should_panic(expected = "ID 1 not found for test_elements")]
    fn test_validate_id_range_duplicate() {
        // Test with duplicate IDs (creates gap)
        let ids = vec![0, 0, 2];
        validate_id_range(&ids, "test_elements");
    }

    #[test]
    fn test_validate_entity_count_match() {
        // Test when counts match
        let ids = vec![0, 1, 2];
        validate_entity_count(&ids, 3, "test_entities");
        // Should not panic
    }

    #[test]
    #[should_panic(
        expected = "Error matching recourse for test_entities: 2 != 3"
    )]
    fn test_validate_entity_count_mismatch_less() {
        // Test when entity count is less than expected
        let ids = vec![0, 1];
        validate_entity_count(&ids, 3, "test_entities");
    }

    #[test]
    #[should_panic(
        expected = "Error matching recourse for test_entities: 4 != 3"
    )]
    fn test_validate_entity_count_mismatch_more() {
        // Test when entity count is more than expected
        let ids = vec![0, 1, 2, 3];
        validate_entity_count(&ids, 3, "test_entities");
    }

    #[test]
    fn test_validate_entity_count_empty() {
        // Test with empty counts (edge case)
        let ids: Vec<usize> = vec![];
        validate_entity_count(&ids, 0, "test_entities");
        // Should not panic
    }

    #[test]
    fn test_from_paths_invalid_config_json() {
        use std::io::Write;
        use std::path::Path;

        // Create temporary file with invalid JSON
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("test_invalid_config.json");
        let mut file = std::fs::File::create(&config_path).unwrap();
        write!(file, "{{invalid json syntax").unwrap();
        drop(file);

        let result = Input::from_paths(
            &config_path,
            Path::new("example/system.json"),
            Path::new("example/graph.json"),
            Path::new("example/recourse.json"),
        );

        // Cleanup
        let _ = std::fs::remove_file(&config_path);

        // Verify error
        assert!(result.is_err(), "Should return error for invalid JSON");
        if let Err(error) = result {
            let error_msg = format!("{}", error);
            assert!(
                error_msg.contains("JSON parse error")
                    || error_msg.contains("parse"),
                "Error should mention JSON parsing issue: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_from_paths_invalid_system_json() {
        use std::io::Write;
        use std::path::Path;

        // Create temporary file with invalid JSON
        let temp_dir = std::env::temp_dir();
        let system_path = temp_dir.join("test_invalid_system.json");
        let mut file = std::fs::File::create(&system_path).unwrap();
        write!(file, "{{\"buses\": [{{missing_bracket").unwrap();
        drop(file);

        let result = Input::from_paths(
            Path::new("examples/01-deterministic/config.json"),
            &system_path,
            Path::new("examples/01-deterministic/graph.json"),
            Path::new("examples/01-deterministic/recourse.json"),
        );

        // Cleanup
        let _ = std::fs::remove_file(&system_path);

        // Verify error
        assert!(result.is_err(), "Should return error for invalid JSON");
        if let Err(error) = result {
            let error_msg = format!("{}", error);
            assert!(
                error_msg.contains("JSON parse error")
                    || error_msg.contains("parse"),
                "Error should mention JSON parsing issue: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_from_paths_invalid_graph_json() {
        use std::io::Write;
        use std::path::Path;

        // Create temporary file with invalid JSON
        let temp_dir = std::env::temp_dir();
        let graph_path = temp_dir.join("test_invalid_graph.json");
        let mut file = std::fs::File::create(&graph_path).unwrap();
        write!(file, "[1, 2, 3, unquoted_string]").unwrap();
        drop(file);

        let result = Input::from_paths(
            Path::new("examples/01-deterministic/config.json"),
            Path::new("examples/01-deterministic/system.json"),
            &graph_path,
            Path::new("examples/01-deterministic/recourse.json"),
        );

        // Cleanup
        let _ = std::fs::remove_file(&graph_path);

        // Verify error
        assert!(result.is_err(), "Should return error for invalid JSON");
        if let Err(error) = result {
            let error_msg = format!("{}", error);
            assert!(
                error_msg.contains("JSON parse error")
                    || error_msg.contains("parse"),
                "Error should mention JSON parsing issue: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_from_paths_invalid_recourse_json() {
        use std::io::Write;
        use std::path::Path;

        // Create temporary file with invalid JSON
        let temp_dir = std::env::temp_dir();
        let recourse_path = temp_dir.join("test_invalid_recourse.json");
        let mut file = std::fs::File::create(&recourse_path).unwrap();
        write!(file, "{{\"initial_condition\":").unwrap();
        drop(file);

        let result = Input::from_paths(
            Path::new("examples/01-deterministic/config.json"),
            Path::new("examples/01-deterministic/system.json"),
            Path::new("examples/01-deterministic/graph.json"),
            &recourse_path,
        );

        // Cleanup
        let _ = std::fs::remove_file(&recourse_path);

        // Verify error
        assert!(result.is_err(), "Should return error for invalid JSON");
        if let Err(error) = result {
            let error_msg = format!("{}", error);
            assert!(
                error_msg.contains("JSON parse error")
                    || error_msg.contains("parse"),
                "Error should mention JSON parsing issue: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_read_recourse_new_format_independent() {
        // Test noise_models format with independent noise
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": []
            },
            "noise_models": [
                {
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "marginal_distribution": {
                        "type": "normal",
                        "mean": 100.0,
                        "std_dev": 20.0
                    },
                    "temporal_model": {
                        "type": "independent"
                    }
                }
            ]
        }"#;
        let recourse: Recourse = serde_json::from_str(json).unwrap();

        // Should have noise_models
        assert_eq!(recourse.noise_models.len(), 1);

        let model = &recourse.noise_models[0];
        assert!(matches!(model.uncertainty_type, UncertaintyType::Inflow));
        assert_eq!(model.entity_id, 0);
        assert_eq!(model.season_id, 1);
        assert!(matches!(
            &model.marginal_distribution,
            Some(MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0
            })
        ));
        assert!(model.innovation_distribution.is_none());
        assert!(matches!(model.temporal_model, TemporalModel::Independent));
    }

    #[test]
    #[allow(deprecated)] // Test for deprecated AR model during soft deprecation
    fn test_read_recourse_new_format_ar1_structure() {
        // Test noise_models format with AR(1) structure
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": [{"hydro_id": 0, "lag": 1, "value": 120.0}]
            },
            "noise_models": [
                {
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "marginal_distribution": {
                        "type": "normal",
                        "mean": 100.0,
                        "std_dev": 30.0
                    },
                    "innovation_distribution": {
                        "mean": 0.0,
                        "std_dev": 15.0
                    },
                    "temporal_model": {
                        "type": "autoregressive",
                        "lag_order": 1,
                        "coefficients": [0.7]
                    }
                }
            ]
        }"#;
        let recourse: Recourse = serde_json::from_str(json).unwrap();

        // Should have noise_models
        assert_eq!(recourse.noise_models.len(), 1);

        let model = &recourse.noise_models[0];
        assert!(matches!(model.uncertainty_type, UncertaintyType::Inflow));
        assert_eq!(model.entity_id, 0);
        assert_eq!(model.season_id, 1);

        // Verify marginal distribution
        if let Some(MarginalDistribution::Normal { mean, std_dev }) =
            &model.marginal_distribution
        {
            assert_eq!(*mean, 100.0);
            assert_eq!(*std_dev, 30.0);
        } else {
            panic!("Expected Normal marginal distribution");
        }

        // Verify innovation distribution
        let innovation = model.innovation_distribution.as_ref().unwrap();
        assert_eq!(innovation.mean, 0.0); // Innovations should be zero-mean
        assert_eq!(innovation.std_dev, 15.0);

        // Verify temporal model
        if let TemporalModel::Autoregressive {
            lag_order,
            coefficients,
        } = &model.temporal_model
        {
            assert_eq!(*lag_order, 1);
            assert_eq!(coefficients, &[0.7]);
        } else {
            panic!("Expected Autoregressive temporal model");
        }
    }

    #[test]
    fn test_recourse_new_format() {
        // Ensure noise_models format works
        let recourse =
            read_recourse_input("examples/02-stochastic/recourse.json");

        // Should have valid data
        assert!(!recourse.noise_models.is_empty());
    }

    // ========================================================================
    // Schema Tests
    // ========================================================================

    #[test]
    fn test_parse_v2_independent_normal() {
        // Test: Parse format with independent Normal distribution
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": []
            },
            "noise_models": [
                {
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "marginal_distribution": {
                        "type": "normal",
                        "mean": 100.0,
                        "std_dev": 20.0
                    },
                    "temporal_model": {
                        "type": "independent"
                    }
                }
            ]
        }"#;

        let recourse: Recourse = serde_json::from_str(json).unwrap();
        assert_eq!(recourse.noise_models.len(), 1);

        let model = &recourse.noise_models[0];
        assert_eq!(model.entity_id, 0);
        assert_eq!(model.season_id, 1);
        assert!(matches!(
            &model.marginal_distribution,
            Some(MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0
            })
        ));
        assert!(model.innovation_distribution.is_none());
        assert!(matches!(model.temporal_model, TemporalModel::Independent));

        // Validate semantics
        model.validate().unwrap();
    }

    #[test]
    fn test_parse_v2_independent_lognormal3() {
        // Test: Parse format with independent LogNormal3 distribution
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": []
            },
            "noise_models": [
                {
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "marginal_distribution": {
                        "type": "lognormal3",
                        "gamma": 1.0,
                        "mu": 4.5,
                        "sigma": 0.3
                    },
                    "temporal_model": {
                        "type": "independent"
                    }
                }
            ]
        }"#;

        let recourse: Recourse = serde_json::from_str(json).unwrap();
        assert_eq!(recourse.noise_models.len(), 1);

        let model = &recourse.noise_models[0];
        assert!(matches!(
            &model.marginal_distribution,
            Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3
            })
        ));
        assert!(model.innovation_distribution.is_none());
        assert!(matches!(model.temporal_model, TemporalModel::Independent));

        // Validate semantics
        model.validate().unwrap();
    }

    #[test]
    #[allow(deprecated)] // Test for deprecated AR model during soft deprecation
    fn test_parse_v2_ar_with_lognormal3() {
        // Test: Parse AR model with LogNormal3 marginal and Normal innovation
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": [{"hydro_id": 0, "lag": 1, "value": 120.0}]
            },
            "noise_models": [
                {
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "marginal_distribution": {
                        "type": "lognormal3",
                        "gamma": 1.0,
                        "mu": 4.5,
                        "sigma": 0.3
                    },
                    "innovation_distribution": {
                        "mean": 0.0,
                        "std_dev": 15.0
                    },
                    "temporal_model": {
                        "type": "autoregressive",
                        "lag_order": 1,
                        "coefficients": [0.7]
                    }
                }
            ]
        }"#;

        let recourse: Recourse = serde_json::from_str(json).unwrap();
        assert_eq!(recourse.noise_models.len(), 1);

        let model = &recourse.noise_models[0];
        assert!(matches!(
            &model.marginal_distribution,
            Some(MarginalDistribution::LogNormal3 { .. })
        ));

        let innovation = model.innovation_distribution.as_ref().unwrap();
        assert_eq!(innovation.mean, 0.0);
        assert_eq!(innovation.std_dev, 15.0);

        if let TemporalModel::Autoregressive {
            lag_order,
            coefficients,
        } = &model.temporal_model
        {
            assert_eq!(*lag_order, 1);
            assert_eq!(coefficients, &[0.7]);
        } else {
            panic!("Expected Autoregressive temporal model");
        }

        // Validate semantics
        model.validate().unwrap();
    }

    // ========================================================================
    // PAR-001: Tests for PeriodicAutoregressive variant
    // ========================================================================

    #[test]
    fn test_par_deserialize_valid_12_period() {
        // Test: Deserialize valid 12-period PAR config
        let json = r#"{
            "type": "periodic_ar",
            "num_seasons": 12,
            "ar_orders": [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "ar_coefficients": [
                [0.7], [0.75], [0.6, 0.2], [0.7], [0.65], [0.6],
                [0.6], [0.65], [0.7], [0.75], [0.8], [0.75]
            ],
            "seasonal_means": [
                100.0, 120.0, 150.0, 180.0, 200.0, 180.0,
                150.0, 120.0, 100.0, 90.0, 80.0, 90.0
            ],
            "seasonal_stds": [
                20.0, 25.0, 30.0, 35.0, 40.0, 35.0,
                30.0, 25.0, 20.0, 18.0, 15.0, 18.0
            ]
        }"#;

        let model: TemporalModel = serde_json::from_str(json).unwrap();

        match model {
            TemporalModel::PeriodicAutoregressive {
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => {
                assert_eq!(num_seasons, 12);
                assert_eq!(ar_orders.len(), 12);
                assert_eq!(ar_orders[0], 1);
                assert_eq!(ar_orders[2], 2); // Third season has AR(2)
                assert_eq!(ar_coefficients.len(), 12);
                assert_eq!(ar_coefficients[0], vec![0.7]);
                assert_eq!(ar_coefficients[2], vec![0.6, 0.2]);
                assert_eq!(seasonal_means.len(), 12);
                assert_eq!(seasonal_means[0], 100.0);
                assert_eq!(seasonal_stds.len(), 12);
                assert_eq!(seasonal_stds[0], 20.0);
            }
            _ => panic!("Expected PeriodicAutoregressive variant"),
        }
    }

    #[test]
    fn test_par_serialize_roundtrip() {
        // Test: Serialize PeriodicAutoregressive back to JSON and deserialize
        let original = TemporalModel::PeriodicAutoregressive {
            num_seasons: 4,
            ar_orders: vec![1, 2, 1, 1],
            ar_coefficients: vec![
                vec![0.7],
                vec![0.5, 0.3],
                vec![0.6],
                vec![0.65],
            ],
            seasonal_means: vec![100.0, 150.0, 180.0, 120.0],
            seasonal_stds: vec![20.0, 30.0, 35.0, 25.0],
        };

        // Serialize to JSON
        let json = serde_json::to_string(&original).unwrap();

        // Deserialize back
        let deserialized: TemporalModel = serde_json::from_str(&json).unwrap();

        // Verify equality
        assert_eq!(original, deserialized);
    }

    #[test]
    #[allow(deprecated)] // Test for deprecated AR model during soft deprecation
    fn test_par_coexists_with_other_variants() {
        // Test: All three variants (Independent, Autoregressive, PeriodicAutoregressive) coexist
        let independent = TemporalModel::Independent;
        let ar = TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.7],
        };
        let par = TemporalModel::PeriodicAutoregressive {
            num_seasons: 12,
            ar_orders: vec![1; 12],
            ar_coefficients: vec![vec![0.7]; 12],
            seasonal_means: vec![100.0; 12],
            seasonal_stds: vec![20.0; 12],
        };

        // Store in a Vec to verify they can coexist
        let models = vec![independent, ar, par];
        assert_eq!(models.len(), 3);

        // Verify each variant
        assert!(matches!(models[0], TemporalModel::Independent));
        assert!(matches!(models[1], TemporalModel::Autoregressive { .. }));
        assert!(matches!(
            models[2],
            TemporalModel::PeriodicAutoregressive { .. }
        ));
    }

    #[test]
    #[allow(deprecated)] // Test for deprecated AR model during soft deprecation
    fn test_par_backward_compatible_ar_configs() {
        // Test: Existing AR configs still deserialize correctly after adding PAR variant
        let json = r#"{
            "type": "autoregressive",
            "lag_order": 2,
            "coefficients": [0.6, 0.3]
        }"#;

        let model: TemporalModel = serde_json::from_str(json).unwrap();

        match model {
            TemporalModel::Autoregressive {
                lag_order,
                coefficients,
            } => {
                assert_eq!(lag_order, 2);
                assert_eq!(coefficients, vec![0.6, 0.3]);
            }
            _ => panic!("Expected Autoregressive variant"),
        }
    }

    #[test]
    fn test_par_quarterly_period() {
        // Test: PAR with quarterly (4-period) configuration
        let json = r#"{
            "type": "periodic_ar",
            "num_seasons": 4,
            "ar_orders": [1, 2, 1, 1],
            "ar_coefficients": [
                [0.7], [0.5, 0.3], [0.6], [0.65]
            ],
            "seasonal_means": [100.0, 150.0, 180.0, 120.0],
            "seasonal_stds": [20.0, 30.0, 35.0, 25.0]
        }"#;

        let model: TemporalModel = serde_json::from_str(json).unwrap();

        match model {
            TemporalModel::PeriodicAutoregressive {
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => {
                assert_eq!(num_seasons, 4);
                assert_eq!(ar_orders, vec![1, 2, 1, 1]);
                assert_eq!(ar_coefficients[1], vec![0.5, 0.3]); // Q2 has AR(2)
                assert_eq!(seasonal_means, vec![100.0, 150.0, 180.0, 120.0]);
                assert_eq!(seasonal_stds, vec![20.0, 30.0, 35.0, 25.0]);
            }
            _ => panic!("Expected PeriodicAutoregressive variant"),
        }
    }

    #[test]
    fn test_par_varying_ar_orders() {
        // Test: PAR with varying AR orders across periods
        let model = TemporalModel::PeriodicAutoregressive {
            num_seasons: 3,
            ar_orders: vec![0, 1, 2], // AR(0), AR(1), AR(2)
            ar_coefficients: vec![vec![], vec![0.7], vec![0.5, 0.3]],
            seasonal_means: vec![100.0, 120.0, 150.0],
            seasonal_stds: vec![20.0, 25.0, 30.0],
        };

        match model {
            TemporalModel::PeriodicAutoregressive {
                ar_orders,
                ar_coefficients,
                ..
            } => {
                assert_eq!(ar_orders[0], 0); // Period 0: AR(0) (white noise)
                assert_eq!(ar_orders[1], 1); // Period 1: AR(1)
                assert_eq!(ar_orders[2], 2); // Period 2: AR(2)
                assert_eq!(ar_coefficients[0].len(), 0);
                assert_eq!(ar_coefficients[1].len(), 1);
                assert_eq!(ar_coefficients[2].len(), 2);
            }
            _ => panic!("Expected PeriodicAutoregressive variant"),
        }
    }

    // ========================================================================
    // PAR-002: Tests for SeasonalStats and PeriodicARParams
    // ========================================================================

    #[test]
    fn test_par002_create_seasonal_stats() {
        // Test: Create SeasonalStats and verify all fields
        let stats = SeasonalStats {
            period_index: 2,
            mean: 150.0,
            std_dev: 30.0,
            skewness: Some(0.5),
            ar_order: 2,
        };

        assert_eq!(stats.period_index, 2);
        assert_eq!(stats.mean, 150.0);
        assert_eq!(stats.std_dev, 30.0);
        assert_eq!(stats.skewness, Some(0.5));
        assert_eq!(stats.ar_order, 2);

        // Test with None skewness
        let stats_no_skew = SeasonalStats {
            period_index: 0,
            mean: 100.0,
            std_dev: 20.0,
            skewness: None,
            ar_order: 1,
        };
        assert_eq!(stats_no_skew.skewness, None);
    }

    #[test]
    fn test_par002_convert_from_temporal_model() {
        // Test: Convert TemporalModel::PeriodicAutoregressive to PeriodicARParams
        let temporal = TemporalModel::PeriodicAutoregressive {
            num_seasons: 2,
            ar_orders: vec![1, 2],
            ar_coefficients: vec![vec![0.7], vec![0.5, 0.3]],
            seasonal_means: vec![100.0, 150.0],
            seasonal_stds: vec![20.0, 30.0],
        };

        let params = PeriodicARParams::try_from(&temporal).unwrap();

        assert_eq!(params.num_seasons, 2);
        assert_eq!(params.seasonal_stats.len(), 2);
        assert_eq!(params.ar_coefficients.len(), 2);

        // Check first period
        assert_eq!(params.seasonal_stats[0].period_index, 0);
        assert_eq!(params.seasonal_stats[0].mean, 100.0);
        assert_eq!(params.seasonal_stats[0].std_dev, 20.0);
        assert_eq!(params.seasonal_stats[0].ar_order, 1);
        assert_eq!(params.ar_coefficients[0], vec![0.7]);

        // Check second period
        assert_eq!(params.seasonal_stats[1].period_index, 1);
        assert_eq!(params.seasonal_stats[1].mean, 150.0);
        assert_eq!(params.seasonal_stats[1].std_dev, 30.0);
        assert_eq!(params.seasonal_stats[1].ar_order, 2);
        assert_eq!(params.ar_coefficients[1], vec![0.5, 0.3]);
    }

    #[test]
    #[allow(deprecated)] // Test for deprecated AR model during soft deprecation
    fn test_par002_convert_from_non_par_temporal_model() {
        // Test: Converting non-PAR TemporalModel should fail
        let temporal = TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.7],
        };

        let result = PeriodicARParams::try_from(&temporal);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Cannot convert non-PeriodicAutoregressive"));
    }

    #[test]
    fn test_par002_get_params_for_season_wraparound() {
        // Test: get_params_for_season with wraparound
        let params = PeriodicARParams {
            num_seasons: 4,
            seasonal_stats: vec![
                SeasonalStats {
                    period_index: 0,
                    mean: 100.0,
                    std_dev: 20.0,
                    skewness: None,
                    ar_order: 1,
                },
                SeasonalStats {
                    period_index: 1,
                    mean: 150.0,
                    std_dev: 30.0,
                    skewness: None,
                    ar_order: 1,
                },
                SeasonalStats {
                    period_index: 2,
                    mean: 180.0,
                    std_dev: 35.0,
                    skewness: None,
                    ar_order: 1,
                },
                SeasonalStats {
                    period_index: 3,
                    mean: 120.0,
                    std_dev: 25.0,
                    skewness: None,
                    ar_order: 1,
                },
            ],
            ar_coefficients: vec![vec![0.7], vec![0.75], vec![0.8], vec![0.7]],
        };

        // Normal access
        assert_eq!(params.get_params_for_season(0).mean, 100.0);
        assert_eq!(params.get_params_for_season(1).mean, 150.0);
        assert_eq!(params.get_params_for_season(3).mean, 120.0);

        // Wraparound: period 4 -> 0, period 5 -> 1
        assert_eq!(params.get_params_for_season(4).mean, 100.0);
        assert_eq!(params.get_params_for_season(5).mean, 150.0);
        assert_eq!(params.get_params_for_season(7).mean, 120.0); // 7 % 4 = 3

        // Test get_ar_coeffs_for_season
        assert_eq!(params.get_ar_coeffs_for_season(0), &[0.7]);
        assert_eq!(params.get_ar_coeffs_for_season(4), &[0.7]); // Wraparound
    }

    #[test]
    fn test_par002_validate_consistency_catches_length_mismatch() {
        // Test: validate_consistency catches seasonal_stats length mismatch
        let invalid_stats = PeriodicARParams {
            num_seasons: 3,
            seasonal_stats: vec![
                SeasonalStats {
                    period_index: 0,
                    mean: 100.0,
                    std_dev: 20.0,
                    skewness: None,
                    ar_order: 1,
                },
                SeasonalStats {
                    period_index: 1,
                    mean: 150.0,
                    std_dev: 30.0,
                    skewness: None,
                    ar_order: 1,
                },
                // Missing third period!
            ],
            ar_coefficients: vec![vec![0.7], vec![0.75], vec![0.8]],
        };

        let result = invalid_stats.validate_consistency();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("seasonal_stats length 2 != num_seasons 3"));

        // Test: validate_consistency catches ar_coefficients length mismatch
        let invalid_coeffs = PeriodicARParams {
            num_seasons: 2,
            seasonal_stats: vec![
                SeasonalStats {
                    period_index: 0,
                    mean: 100.0,
                    std_dev: 20.0,
                    skewness: None,
                    ar_order: 1,
                },
                SeasonalStats {
                    period_index: 1,
                    mean: 150.0,
                    std_dev: 30.0,
                    skewness: None,
                    ar_order: 1,
                },
            ],
            ar_coefficients: vec![vec![0.7]], // Only one period!
        };

        let result = invalid_coeffs.validate_consistency();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("ar_coefficients length 1 != num_seasons 2"));

        // Test: validate_consistency catches AR coefficient count mismatch
        let invalid_ar_count = PeriodicARParams {
            num_seasons: 2,
            seasonal_stats: vec![
                SeasonalStats {
                    period_index: 0,
                    mean: 100.0,
                    std_dev: 20.0,
                    skewness: None,
                    ar_order: 2, // Expects 2 coefficients
                },
                SeasonalStats {
                    period_index: 1,
                    mean: 150.0,
                    std_dev: 30.0,
                    skewness: None,
                    ar_order: 1,
                },
            ],
            ar_coefficients: vec![vec![0.7], vec![0.75]], // Period 0 has only 1 coeff, needs 2!
        };

        let result = invalid_ar_count.validate_consistency();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Season 0 ar_coefficients length 1 != ar_order 2"));

        // Test: validate_consistency catches negative std_dev
        let invalid_std = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: -5.0, // Invalid!
                skewness: None,
                ar_order: 1,
            }],
            ar_coefficients: vec![vec![0.7]],
        };

        let result = invalid_std.validate_consistency();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("std_dev -5 must be > 0"));
    }

    #[test]
    fn test_par002_validate_stationarity_ar1() {
        // Test: validate_stationarity detects unstable AR(1): |φ| > 1
        let unstable = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: 20.0,
                skewness: None,
                ar_order: 1,
            }],
            ar_coefficients: vec![vec![1.2]], // |φ₁| = 1.2 > 1
        };

        let result = unstable.validate_stationarity();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("AR(1)"));
        assert!(err_msg.contains("unstable"));

        // Test: Stable AR(1)
        let stable = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: 20.0,
                skewness: None,
                ar_order: 1,
            }],
            ar_coefficients: vec![vec![0.7]], // |φ₁| = 0.7 < 1
        };

        assert!(stable.validate_stationarity().is_ok());

        // Test: Boundary case |φ| = -0.99 (stable)
        let boundary = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: 20.0,
                skewness: None,
                ar_order: 1,
            }],
            ar_coefficients: vec![vec![-0.99]],
        };

        assert!(boundary.validate_stationarity().is_ok());
    }

    #[test]
    fn test_par002_validate_stationarity_ar2() {
        // Test: validate_stationarity accepts stable AR(2)
        let stable = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: 20.0,
                skewness: None,
                ar_order: 2,
            }],
            ar_coefficients: vec![vec![0.5, 0.3]], // Stable AR(2)
        };

        assert!(stable.validate_stationarity().is_ok());

        // Test: Unstable AR(2) - violates φ₁ + φ₂ < 1
        let unstable1 = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: 20.0,
                skewness: None,
                ar_order: 2,
            }],
            ar_coefficients: vec![vec![0.7, 0.4]], // 0.7 + 0.4 = 1.1 >= 1
        };

        let result = unstable1.validate_stationarity();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("φ₁+φ₂"));

        // Test: Unstable AR(2) - violates φ₂ - φ₁ < 1
        let unstable2 = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: 20.0,
                skewness: None,
                ar_order: 2,
            }],
            ar_coefficients: vec![vec![-0.5, 0.6]], // 0.6 - (-0.5) = 1.1 >= 1
        };

        let result = unstable2.validate_stationarity();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("φ₂-φ₁"));

        // Test: Unstable AR(2) - violates |φ₂| < 1
        let unstable3 = PeriodicARParams {
            num_seasons: 1,
            seasonal_stats: vec![SeasonalStats {
                period_index: 0,
                mean: 100.0,
                std_dev: 20.0,
                skewness: None,
                ar_order: 2,
            }],
            ar_coefficients: vec![vec![0.2, -1.1]], // |φ₂| = 1.1 >= 1
        };

        let result = unstable3.validate_stationarity();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("|φ₂|"));
    }

    #[test]
    fn test_par002_serialize_deserialize_roundtrip() {
        // Test: Serialize and deserialize SeasonalStats
        let stats = SeasonalStats {
            period_index: 3,
            mean: 180.0,
            std_dev: 35.0,
            skewness: Some(0.5),
            ar_order: 2,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: SeasonalStats = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.period_index, 3);
        assert_eq!(deserialized.mean, 180.0);
        assert_eq!(deserialized.std_dev, 35.0);
        assert_eq!(deserialized.skewness, Some(0.5));
        assert_eq!(deserialized.ar_order, 2);

        // Test: Serialize and deserialize PeriodicARParams
        let params = PeriodicARParams {
            num_seasons: 2,
            seasonal_stats: vec![
                SeasonalStats {
                    period_index: 0,
                    mean: 100.0,
                    std_dev: 20.0,
                    skewness: None,
                    ar_order: 1,
                },
                SeasonalStats {
                    period_index: 1,
                    mean: 150.0,
                    std_dev: 30.0,
                    skewness: Some(0.3),
                    ar_order: 2,
                },
            ],
            ar_coefficients: vec![vec![0.7], vec![0.5, 0.3]],
        };

        let json = serde_json::to_string(&params).unwrap();
        let deserialized: PeriodicARParams =
            serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.num_seasons, 2);
        assert_eq!(deserialized.seasonal_stats.len(), 2);
        assert_eq!(deserialized.ar_coefficients[1], vec![0.5, 0.3]);
    }

    // ========================================================================
    // PAR-003: Tests for NoiseModel with Periodic AR and residual_distribution
    // ========================================================================

    #[test]
    fn test_par003_independent_uses_marginal_for_final_series() {
        // Test: Independent model uses marginal_distribution directly for final series
        let mut model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: None, // Will be populated by migration
            marginal_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            }),
            innovation_distribution: None,
            temporal_model: TemporalModel::Independent,
            residual_distribution: None,
        };

        // Migrate distribution fields
        model.migrate_distribution_fields().unwrap();

        // Validate model
        assert!(model.validate().is_ok());

        // Check distribution target
        match model.get_distribution_target() {
            DistributionTarget::FinalSeries(dist) => {
                // Should route to marginal_distribution
                match dist {
                    MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                        assert_eq!(*gamma, 1.0);
                        assert_eq!(*mu, 4.5);
                        assert_eq!(*sigma, 0.3);
                    }
                    _ => panic!("Expected LogNormal3"),
                }
            }
            _ => panic!("Expected FinalSeries target for Independent model"),
        }
    }

    #[test]
    #[allow(deprecated)] // Test using deprecated AR model during soft deprecation
    fn test_par003_ar_uses_marginal_for_innovations() {
        // Test: AR model uses marginal_distribution for innovations
        let mut model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: None, // Will be populated by migration
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            }),
            innovation_distribution: Some(InnovationDistribution {
                mean: 0.0,
                std_dev: 15.0,
            }),
            temporal_model: TemporalModel::Autoregressive {
                lag_order: 1,
                coefficients: vec![0.7],
            },
            residual_distribution: None,
        };

        // Migrate AR to PAR first, then migrate distribution fields
        model.migrate_ar_to_par().unwrap();
        model.migrate_distribution_fields().unwrap();

        // Validate model
        assert!(model.validate().is_ok());

        // Check distribution target (after migration, AR becomes PAR)
        match model.get_distribution_target() {
            DistributionTarget::Residuals(dist) => {
                // After AR->PAR migration, uses residual distribution (standard normal)
                match dist {
                    MarginalDistribution::Normal { mean, std_dev } => {
                        assert_eq!(*mean, 0.0);
                        assert_eq!(*std_dev, 1.0); // Standard normal for residuals
                    }
                    _ => panic!(
                        "Expected Normal residual distribution after migration"
                    ),
                }
            }
            _ => panic!("Expected Residuals target after AR->PAR migration"),
        }
    }

    #[test]
    #[allow(deprecated)] // Test using deprecated residual_distribution during migration
    fn test_par003_par_uses_residual_distribution() {
        // Test: PAR model uses residual_distribution for residuals
        let mut model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: None, // Will be populated by migration
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }), // Ignored for PAR
            innovation_distribution: None,
            temporal_model: TemporalModel::PeriodicAutoregressive {
                num_seasons: 12,
                ar_orders: vec![1; 12],
                ar_coefficients: vec![vec![0.7]; 12],
                seasonal_means: vec![100.0; 12],
                seasonal_stds: vec![20.0; 12],
            },
            residual_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 1.0,
            }),
        };

        // Migrate distribution fields
        model.migrate_distribution_fields().unwrap();

        // Validate model
        assert!(model.validate().is_ok());

        // Check distribution target
        match model.get_distribution_target() {
            DistributionTarget::Residuals(dist) => {
                // Should route to residual_distribution
                match dist {
                    MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                        assert_eq!(*gamma, 1.0);
                        assert_eq!(*mu, 0.0);
                        assert_eq!(*sigma, 1.0);
                    }
                    _ => panic!("Expected LogNormal3"),
                }
            }
            _ => panic!("Expected Residuals target for PAR model"),
        }
    }

    #[test]
    fn test_par003_validation_par_requires_residual_distribution() {
        // Test: PAR model without residual_distribution fails validation
        let model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: None, // Will be populated by migration
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }),
            innovation_distribution: None,
            temporal_model: TemporalModel::PeriodicAutoregressive {
                num_seasons: 12,
                ar_orders: vec![1; 12],
                ar_coefficients: vec![vec![0.7]; 12],
                seasonal_means: vec![100.0; 12],
                seasonal_stds: vec![20.0; 12],
            },
            residual_distribution: None, // Missing!
        };

        let result = model.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("missing distribution"),
            "Error should mention missing distribution: {}",
            err
        );
    }

    #[test]
    fn test_par003_validation_par_rejects_innovation_distribution() {
        // Test: PAR model with innovation_distribution fails validation
        let model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: None, // Will be populated by migration
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }),
            innovation_distribution: Some(InnovationDistribution {
                mean: 0.0,
                std_dev: 15.0,
            }), // Invalid for PAR!
            temporal_model: TemporalModel::PeriodicAutoregressive {
                num_seasons: 12,
                ar_orders: vec![1; 12],
                ar_coefficients: vec![vec![0.7]; 12],
                seasonal_means: vec![100.0; 12],
                seasonal_stds: vec![20.0; 12],
            },
            residual_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 1.0,
            }),
        };

        let result = model.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("has innovation_distribution"),
            "Error should mention invalid innovation_distribution for PAR: {}",
            err
        );
    }

    #[test]
    #[allow(deprecated)] // Test using deprecated residual_distribution field
    fn test_par003_validation_independent_rejects_residual_distribution() {
        // Test: Independent model with residual_distribution fails validation
        let model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: None, // Will be populated by migration
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            }),
            innovation_distribution: None,
            temporal_model: TemporalModel::Independent,
            residual_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 1.0,
            }), // Invalid for Independent!
        };

        let result = model.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("must not have residual_distribution"),
            "Error should mention invalid residual_distribution for Independent: {}",
            err
        );
    }

    #[test]
    #[allow(deprecated)] // Test using deprecated AR model and fields
    fn test_par003_validation_ar_rejects_residual_distribution() {
        // Test: AR model with residual_distribution fails validation
        let model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: None, // Will be populated by migration
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            }),
            innovation_distribution: Some(InnovationDistribution {
                mean: 0.0,
                std_dev: 15.0,
            }),
            temporal_model: TemporalModel::Autoregressive {
                lag_order: 1,
                coefficients: vec![0.7],
            },
            residual_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 1.0,
            }), // Invalid for AR!
        };

        let result = model.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("must not have residual_distribution"),
            "Error should mention invalid residual_distribution for AR: {}",
            err
        );
    }

    #[test]
    #[allow(deprecated)] // Test uses deprecated residual_distribution field
    fn test_par003_backward_compatibility_old_json_deserializes() {
        // Test: Old JSON configs without residual_distribution still work
        let json = r#"{
            "uncertainty_type": "inflow",
            "entity_id": 0,
            "season_id": 1,
            "marginal_distribution": {
                "type": "lognormal3",
                "gamma": 1.0,
                "mu": 4.5,
                "sigma": 0.3
            },
            "temporal_model": {
                "type": "independent"
            }
        }"#;

        let model: NoiseModel = serde_json::from_str(json).unwrap();

        // Should deserialize successfully with residual_distribution = None
        assert!(model.residual_distribution.is_none());
        assert!(model.validate().is_ok());
    }

    #[test]
    #[allow(deprecated)] // Test uses deprecated residual_distribution field
    fn test_par003_par_json_with_residual_distribution() {
        // Test: PAR config with residual_distribution deserializes correctly
        let json = r#"{
            "uncertainty_type": "inflow",
            "entity_id": 0,
            "season_id": 1,
            "marginal_distribution": {
                "type": "normal",
                "mean": 0.0,
                "std_dev": 1.0
            },
            "temporal_model": {
                "type": "periodic_ar",
                "num_seasons": 4,
                "ar_orders": [1, 1, 2, 1],
                "ar_coefficients": [[0.7], [0.75], [0.6, 0.2], [0.7]],
                "seasonal_means": [100.0, 120.0, 150.0, 180.0],
                "seasonal_stds": [20.0, 25.0, 30.0, 35.0]
            },
            "residual_distribution": {
                "type": "lognormal3",
                "gamma": 1.0,
                "mu": 0.0,
                "sigma": 1.0
            }
        }"#;

        let model: NoiseModel = serde_json::from_str(json).unwrap();

        // Verify deserialization
        assert!(model.residual_distribution.is_some());
        match &model.residual_distribution {
            Some(MarginalDistribution::LogNormal3 { gamma, mu, sigma }) => {
                assert_eq!(*gamma, 1.0);
                assert_eq!(*mu, 0.0);
                assert_eq!(*sigma, 1.0);
            }
            _ => panic!("Expected Some(LogNormal3)"),
        }

        // Validate model
        assert!(model.validate().is_ok());
    }
}
*/

#[cfg(test)]
mod seasonal_distribution_tests {
    use super::*;

    #[test]
    fn test_parse_normal_seasonal_distribution() {
        let json = r#"{
            "season_id": 0,
            "type": "normal",
            "mean": 100.0,
            "std_dev": 20.0
        }"#;

        let dist: SeasonalDistribution = serde_json::from_str(json).unwrap();

        assert_eq!(dist.season_id, 0);
        match &dist.distribution {
            MarginalDistribution::Normal { mean, std_dev } => {
                assert_eq!(*mean, 100.0);
                assert_eq!(*std_dev, 20.0);
            }
            _ => panic!("Expected Normal distribution"),
        }
    }

    #[test]
    fn test_parse_lognormal3_seasonal_distribution() {
        let json = r#"{
            "season_id": 1,
            "type": "lognormal3",
            "gamma": 1.0,
            "mu": 4.5,
            "sigma": 0.3
        }"#;

        let dist: SeasonalDistribution = serde_json::from_str(json).unwrap();

        assert_eq!(dist.season_id, 1);
        match &dist.distribution {
            MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                assert_eq!(*gamma, 1.0);
                assert_eq!(*mu, 4.5);
                assert_eq!(*sigma, 0.3);
            }
            _ => panic!("Expected LogNormal3 distribution"),
        }
    }

    #[test]
    fn test_normal_to_seasonal_params() {
        let dist = SeasonalDistribution {
            season_id: 0,
            distribution: MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
        };

        let params = dist.to_seasonal_params();

        assert_eq!(params.mean, 100.0);
        assert_eq!(params.std_dev, 20.0);
        assert!(
            params.marginal_override.is_none(),
            "Normal distribution should not have marginal_override"
        );
    }

    #[test]
    fn test_lognormal3_to_seasonal_params() {
        let dist = SeasonalDistribution {
            season_id: 1,
            distribution: MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            },
        };

        let params = dist.to_seasonal_params();

        // Verify marginal_override is set
        assert!(
            params.marginal_override.is_some(),
            "LogNormal3 should have marginal_override"
        );

        match &params.marginal_override {
            Some(MarginalDistribution::LogNormal3 { gamma, mu, sigma }) => {
                assert_eq!(*gamma, 1.0);
                assert_eq!(*mu, 4.5);
                assert_eq!(*sigma, 0.3);
            }
            _ => panic!("Expected LogNormal3 in marginal_override"),
        }

        // Verify computed mean and std_dev
        // E[X] = γ + exp(μ + σ²/2)
        let expected_mean = 1.0 + (4.5 + 0.3_f64.powi(2) / 2.0).exp();
        assert!(
            (params.mean - expected_mean).abs() < 1e-6,
            "Mean should be computed from LogNormal3 parameters"
        );

        // Var(X) = exp(2μ + σ²) × (exp(σ²) - 1)
        let variance =
            (2.0 * 4.5 + 0.3_f64.powi(2)).exp() * (0.3_f64.powi(2).exp() - 1.0);
        let expected_std_dev = variance.sqrt();
        assert!(
            (params.std_dev - expected_std_dev).abs() < 1e-6,
            "Std dev should be computed from LogNormal3 parameters"
        );
    }

    #[test]
    fn test_lognormal3_with_zero_gamma() {
        // Test gamma=0 (reduces to 2-parameter lognormal)
        let dist = SeasonalDistribution {
            season_id: 0,
            distribution: MarginalDistribution::LogNormal3 {
                gamma: 0.0,
                mu: 3.0,
                sigma: 0.5,
            },
        };

        let params = dist.to_seasonal_params();

        // Mean should still be computed correctly
        let expected_mean = 0.0 + (3.0 + 0.5_f64.powi(2) / 2.0).exp();
        assert!((params.mean - expected_mean).abs() < 1e-6);

        assert!(params.marginal_override.is_some());
    }

    #[test]
    fn test_uncertainty_specification_with_lognormal3() {
        let json = r#"{
            "uncertainty_type": "inflow",
            "entity_id": 0,
            "temporal_model": {
                "type": "independent"
            },
            "seasonal_distributions": [
                {
                    "season_id": 0,
                    "type": "lognormal3",
                    "gamma": 1.0,
                    "mu": 4.5,
                    "sigma": 0.3
                },
                {
                    "season_id": 1,
                    "type": "lognormal3",
                    "gamma": 1.0,
                    "mu": 4.6,
                    "sigma": 0.35
                }
            ]
        }"#;

        let spec: UncertaintySpecification =
            serde_json::from_str(json).unwrap();

        assert_eq!(spec.uncertainty_type, UncertaintyType::Inflow);
        assert_eq!(spec.entity_id, 0);

        let seasonal_dists = spec.seasonal_distributions.as_ref().unwrap();
        assert_eq!(seasonal_dists.len(), 2);

        // Check first season
        match &seasonal_dists[0].distribution {
            MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                assert_eq!(*gamma, 1.0);
                assert_eq!(*mu, 4.5);
                assert_eq!(*sigma, 0.3);
            }
            _ => panic!("Expected LogNormal3 for season 0"),
        }

        // Check second season
        match &seasonal_dists[1].distribution {
            MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                assert_eq!(*gamma, 1.0);
                assert_eq!(*mu, 4.6);
                assert_eq!(*sigma, 0.35);
            }
            _ => panic!("Expected LogNormal3 for season 1"),
        }
    }

    #[test]
    fn test_mixed_distributions_in_specification() {
        // Test mixed Normal and LogNormal3 in same specification
        let json = r#"{
            "uncertainty_type": "load",
            "entity_id": 0,
            "temporal_model": {
                "type": "independent"
            },
            "seasonal_distributions": [
                {
                    "season_id": 0,
                    "type": "normal",
                    "mean": 100.0,
                    "std_dev": 20.0
                },
                {
                    "season_id": 1,
                    "type": "lognormal3",
                    "gamma": 1.0,
                    "mu": 4.5,
                    "sigma": 0.3
                }
            ]
        }"#;

        let spec: UncertaintySpecification =
            serde_json::from_str(json).unwrap();

        let seasonal_dists = spec.seasonal_distributions.as_ref().unwrap();
        assert_eq!(seasonal_dists.len(), 2);

        // Season 0: Normal
        match &seasonal_dists[0].distribution {
            MarginalDistribution::Normal { mean, std_dev } => {
                assert_eq!(*mean, 100.0);
                assert_eq!(*std_dev, 20.0);
            }
            _ => panic!("Expected Normal for season 0"),
        }

        // Season 1: LogNormal3
        match &seasonal_dists[1].distribution {
            MarginalDistribution::LogNormal3 { .. } => {
                // OK
            }
            _ => panic!("Expected LogNormal3 for season 1"),
        }
    }

    #[test]
    fn test_serialize_seasonal_distribution_normal() {
        let dist = SeasonalDistribution {
            season_id: 0,
            distribution: MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
        };

        let json = serde_json::to_string(&dist).unwrap();

        // Should contain the flattened distribution fields
        assert!(json.contains("\"type\":\"normal\""));
        assert!(json.contains("\"mean\":100"));
        assert!(json.contains("\"std_dev\":20"));
        assert!(json.contains("\"season_id\":0"));
    }

    #[test]
    fn test_serialize_seasonal_distribution_lognormal3() {
        let dist = SeasonalDistribution {
            season_id: 1,
            distribution: MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            },
        };

        let json = serde_json::to_string(&dist).unwrap();

        // Should contain the flattened distribution fields
        assert!(json.contains("\"type\":\"lognormal3\""));
        assert!(json.contains("\"gamma\":1"));
        assert!(json.contains("\"mu\":4.5"));
        assert!(json.contains("\"sigma\":0.3"));
        assert!(json.contains("\"season_id\":1"));
    }
}
