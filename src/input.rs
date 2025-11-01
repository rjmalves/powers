use crate::graph;
use crate::initial_condition;
use crate::input_validation::InputValidator;
use crate::scenario;
use crate::sddp;
use crate::subproblem;
use crate::system;
use crate::uncertainty_model::UncertaintyModel;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;

#[derive(Deserialize)]
pub struct Config {
    pub num_iterations: usize,
    pub num_forward_passes: usize,

    #[serde(default)]
    pub num_simulation_scenarios: Option<usize>,
    pub seed: u64,

    #[serde(default)]
    pub num_threads: Option<usize>,

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
        uncertainty_models: &std::sync::Arc<Vec<UncertaintyModel>>,
    ) -> Result<(), String> {
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
                uncertainty_models.clone(), // Arc::clone is cheap (just pointer increment)
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
        uncertainty_models: &std::sync::Arc<Vec<UncertaintyModel>>,
    ) -> Result<(), String> {
        let first_node = self.nodes.first().ok_or("Graph has no nodes")?;
        let state_choice = &first_node.state_variables;

        // Compute lag_order from uncertainty_models (not from deprecated inflow_process)
        // For storage_and_inflow state, we need the maximum AR order across all PAR models
        let lag_order = match state_choice.as_str() {
            "storage" => 0,
            "storage_and_inflow" => {
                // Find max AR order from all inflow PAR models
                let max_lag = uncertainty_models
                    .iter()
                    .filter_map(|model| match model {
                        UncertaintyModel::PeriodicAR {
                            entity_type,
                            par_params,
                            ..
                        } if *entity_type == UncertaintyType::Inflow => {
                            // Get max AR order across all seasons for this hydro
                            Some(par_params.max_ar_order)
                        }
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0); // Default to 0 if no inflow specs

                max_lag
            }
            _ => {
                return Err(format!(
                    "Unknown state_variables: '{}'",
                    state_choice
                ))
            }
        };

        let first_study_season = first_node.season_id;

        // Get num_seasons from uncertainty_models (if PAR model exists)
        // Default to 12 if no PAR model (for Independent or single-season models)
        let num_seasons = uncertainty_models
            .iter()
            .find_map(|model| match model {
                crate::uncertainty_model::UncertaintyModel::PeriodicAR {
                    par_params,
                    ..
                } => Some(par_params.num_seasons),
                _ => None,
            })
            .unwrap_or(12);

        let prestudy_season_ids: Vec<usize> = (0..=lag_order)
            .map(|offset| {
                first_study_season
                    .wrapping_sub(offset)
                    .wrapping_add(num_seasons)
                    % num_seasons
            })
            .collect();

        let num_pre_study_nodes = 1 + lag_order;
        let mut pre_study_node_ids = Vec::with_capacity(num_pre_study_nodes);

        for pre_idx in 0..num_pre_study_nodes {
            let node_id_value = -(lag_order as isize - pre_idx as isize);

            // INDEXING: prestudy_season_ids are [newest, ..., oldest]
            // but PreStudy nodes are created [oldest, ..., newest] (by node_id)
            // So we need to reverse the indexing: oldest node uses last season_id
            let season_id_idx = num_pre_study_nodes - 1 - pre_idx;
            let season_id = prestudy_season_ids[season_id_idx];

            let graph_node_id = graph
                .add_node(sddp::NodeData::new(
                    node_id_value,
                    0,
                    season_id,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    system_input.build_sddp_system(),
                    "expectation",
                    uncertainty_models.clone(), // Arc::clone is cheap (just pointer increment)
                    state_choice,
                    1,
                )?)
                .map_err(|_| {
                    format!("Failed to add pre-study node {}", node_id_value)
                })?;
            pre_study_node_ids.push(graph_node_id);
        }

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

        // Build uncertainty models ONCE for the entire graph
        // Wrap in Arc to share across all nodes without duplication
        let uncertainty_models =
            std::sync::Arc::new(recourse.build_uncertainty_models().map_err(
                |e| format!("Failed to build uncertainty models: {}", e),
            )?);

        self.add_sddp_study_period_to_graph(
            &mut g,
            system_input,
            &uncertainty_models,
        )?;
        self.add_sddp_pre_study_period_to_graph(
            &mut g,
            system_input,
            &uncertainty_models,
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
#[derive(Deserialize, Serialize, Clone)]
pub struct PastInflow {
    pub hydro_id: usize,
    pub lag: usize,
    pub value: f64,
}

/// Initial condition for SDDP algorithm
#[derive(Deserialize, Serialize, Clone)]
pub struct InitialConditionInput {
    pub storage: Vec<InitialStorage>,
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
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum UncertaintyType {
    Inflow,
    Load,
}

/// Temporal model for stochastic processes
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TemporalModel {
    /// Independent process (no temporal correlation)
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
    #[serde(rename = "periodic_ar")]
    PeriodicAutoregressive {
        num_seasons: usize,
        ar_orders: Vec<usize>,
        ar_coefficients: Vec<Vec<f64>>,
        seasonal_means: Vec<f64>,
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
/// 1. PAR(p) scenario generation
/// 2. Parameter estimation from historical data
/// 3. Validation of seasonal parameter consistency
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
    pub period_index: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: Option<f64>,
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
    pub num_seasons: usize,
    pub seasonal_stats: Vec<SeasonalStats>,
    pub ar_coefficients: Vec<Vec<f64>>,
}

impl PeriodicARParams {
    /// Get seasonal statistics for a specific season
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
        if self.seasonal_stats.len() != self.num_seasons {
            return Err(format!(
                "seasonal_stats length {} != num_seasons {}",
                self.seasonal_stats.len(),
                self.num_seasons
            ));
        }

        if self.ar_coefficients.len() != self.num_seasons {
            return Err(format!(
                "ar_coefficients length {} != num_seasons {}",
                self.ar_coefficients.len(),
                self.num_seasons
            ));
        }

        for (m, stats) in self.seasonal_stats.iter().enumerate() {
            if stats.std_dev <= 0.0 {
                return Err(format!(
                    "Season {} std_dev {} must be > 0",
                    m, stats.std_dev
                ));
            }

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
                        skewness: None, // Future enhancement: compute during estimation
                                        // See FUTURE_WORK.md: "Skewness Parameter in Seasonal Statistics"
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

/// Marginal distribution for stochastic processes
///
/// Specifies the target marginal distribution of realizations Xₜ.
///
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
    Normal {
        mean: f64,
        #[serde(rename = "std_dev")]
        std_dev: f64,
    },

    #[serde(rename = "lognormal3")]
    LogNormal3 { gamma: f64, mu: f64, sigma: f64 },
}

/// Distribution parameters for noise models
///
/// This enum represents the statistical distribution used for generating
/// noise realizations (either directly for independent models, or as
/// innovations for AR models).
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Distribution {
    Normal {
        mean: f64,
        #[serde(rename = "std_dev")]
        std_dev: f64,
    },
    Lognormal {
        mu: f64,
        sigma: f64,
    },
}

/// This is the **recommended format** for specifying uncertainties. It provides:
/// - One entity = one specification (no scattered multi-season entries)
/// - Clear separation: temporal model vs marginal distribution
/// - Explicit seasonal_distributions for independent models
/// - No misleading season_id at root level for PAR models
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
    pub uncertainty_type: UncertaintyType,
    pub entity_id: usize,
    pub temporal_model: TemporalModelInput,
    pub seasonal_distributions: Option<Vec<SeasonalDistribution>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct SeasonalDistribution {
    pub season_id: usize,
    #[serde(flatten)]
    pub distribution: MarginalDistribution,
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
    Independent,
    #[serde(rename = "periodic_ar")]
    PeriodicAr {
        num_seasons: usize,
        ar_orders: Vec<usize>,
        ar_coefficients: Vec<Vec<f64>>,
        seasonal_means: Vec<f64>,
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
    None,

    /// Gaussian copula with Cholesky decomposition (recommended)
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
#[derive(Deserialize, Serialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityReference {
    /// Type of uncertainty (inflow, load, etc.)
    pub uncertainty_type: UncertaintyType,

    /// Entity ID (zero-based index)
    pub entity_id: usize,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Recourse {
    pub initial_condition: InitialConditionInput,
    pub uncertainty_specifications: Vec<UncertaintySpecification>,
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

    /// Build uncertainty models from specifications
    ///
    /// Converts JSON specifications to validated `UncertaintyModel` instances.
    ///
    /// # Returns
    ///
    /// Vector of validated `UncertaintyModel` for direct use
    ///
    /// # Errors
    ///
    /// - Format validation fails
    /// - Model construction fails (array length, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let recourse = read_recourse_input("recourse.json");
    /// let models = recourse.build_uncertainty_models()?;
    /// ```
    pub fn build_uncertainty_models(
        &self,
    ) -> Result<Vec<crate::uncertainty_model::UncertaintyModel>, String> {
        self.validate_format()?;

        let models: Result<Vec<_>, _> = self
            .uncertainty_specifications
            .iter()
            .map(|spec| {
                UncertaintyModel::from_specification(spec)
                    .map_err(|e| e.to_string())
            })
            .collect();

        println!("Built {:?} uncertainty models", models);

        models
    }

    /// Generate SDDP scenarios using new scenario_generator module
    ///
    /// Creates Sample Average Approximation (SAA) scenarios for each node in the graph
    /// using the configured uncertainty specifications and correlation structure.
    ///
    /// # Arguments
    ///
    /// - `g`: SDDP graph with node data (seasons, branchings)
    /// - `initial_condition`: Initial storage and inflow lags
    /// - `seed`: Random seed for reproducible scenario generation
    ///
    /// # Returns
    ///
    /// SAA structure with scenarios for all stages, compatible with SDDP train/simulate
    pub fn generate_sddp_noises(
        &self,
        g: &graph::DirectedGraph<sddp::NodeData>,
        initial_condition: &initial_condition::InitialCondition,
        seed: u64,
    ) -> scenario::SAA {
        use crate::scenario_generator::ScenarioGenerator;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256Plus;

        // Build uncertainty models
        let uncertainty_models = self
            .build_uncertainty_models()
            .expect("Failed to build uncertainty models for SAA generation");

        // Create scenario generator
        let mut generator = ScenarioGenerator::new(
            uncertainty_models.clone(),
            initial_condition,
            self.correlation.as_ref(),
        )
        .expect("Failed to create scenario generator");

        // Initialize RNG
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);

        // Create SAA structure
        let mut saa = scenario::SAA::new_empty();

        // Generate scenarios for each Study node
        for node in g.iter_nodes() {
            // Skip PreStudy nodes
            if matches!(
                node.data.kind,
                crate::subproblem::StudyPeriodKind::PreStudy
            ) {
                continue;
            }

            let stage_id = node.data.stage_id;
            let season_id = node.data.season_id;
            let num_branchings = node.data.num_scenarios;

            // Generate scenarios for this stage
            let stage_scenarios = generator.generate_stage_scenarios(
                season_id,
                num_branchings,
                &mut rng,
            );

            // Separate load and inflow entities
            let mut load_noises: Vec<Vec<f64>> = vec![];
            let mut inflow_noises: Vec<Vec<f64>> = vec![];

            // Count entities by type
            let num_load_entities = uncertainty_models
                .iter()
                .filter(|m| matches!(m.entity_type(), UncertaintyType::Load))
                .count();
            let num_inflow_entities = uncertainty_models
                .iter()
                .filter(|m| matches!(m.entity_type(), UncertaintyType::Inflow))
                .count();

            // Pre-allocate entity vectors
            for _ in 0..num_load_entities {
                load_noises.push(Vec::with_capacity(num_branchings));
            }
            for _ in 0..num_inflow_entities {
                inflow_noises.push(Vec::with_capacity(num_branchings));
            }

            // Extract scenarios by entity type
            for scenario in &stage_scenarios.scenarios {
                let mut load_idx = 0;
                let mut inflow_idx = 0;

                for (model_idx, model) in uncertainty_models.iter().enumerate()
                {
                    match model.entity_type() {
                        UncertaintyType::Load => {
                            load_noises[load_idx]
                                .push(scenario.values[model_idx]);
                            load_idx += 1;
                        }
                        UncertaintyType::Inflow => {
                            inflow_noises[inflow_idx]
                                .push(scenario.values[model_idx]);
                            inflow_idx += 1;
                        }
                    }
                }
            }

            // Set scenarios for this stage
            saa.set_noises_by_stage(
                stage_id,
                num_branchings,
                num_load_entities,
                num_inflow_entities,
                load_noises,
                inflow_noises,
            );

            // Also need to add the uniform sampler for this stage
            // (needed for sample_scenario to work)
            while saa.index_samplers.len() <= stage_id {
                // Add dummy sampler for missing stages
                saa.index_samplers.push(
                    rand_distr::Uniform::<usize>::try_from(0..1).unwrap(),
                );
            }
            saa.index_samplers[stage_id] =
                rand_distr::Uniform::<usize>::try_from(0..num_branchings)
                    .unwrap();
        }

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
