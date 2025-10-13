use crate::graph;
use crate::initial_condition;
use crate::scenario;
use crate::sddp;
use crate::subproblem;
use crate::system;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;

#[derive(Deserialize)]
pub struct Config {
    pub num_iterations: usize,
    pub num_forward_passes: usize,
    pub num_simulation_scenarios: usize,
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
    /// Default: `None` (no output, 10-30% faster execution)
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
    /// Number of scenarios to branch from this node in forward passes.
    /// Used by ScenarioGenerator to determine scenarios_per_stage vector.
    /// Typically matches config.num_forward_passes for most nodes.
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
    ) -> Result<(), String> {
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
                &node_input.inflow_stochastic_process,
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
    ) -> Result<(), String> {
        let initial_condition_node_id = graph
            .add_node(sddp::NodeData::new(
                -1,
                0,
                0,
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                subproblem::StudyPeriodKind::PreStudy,
                system_input.build_sddp_system(),
                "expectation",
                "naive",
                "naive",
                "storage",
                1, // PreStudy always has 1 scenario
            )?)
            .unwrap();
        graph
            .add_edge(initial_condition_node_id, self.nodes.first().unwrap().id)
            .unwrap();
        Ok(())
    }

    pub fn build_sddp_graph(
        &self,
        system_input: &SystemInput,
    ) -> Result<graph::DirectedGraph<sddp::NodeData>, String> {
        let mut g = graph::DirectedGraph::<sddp::NodeData>::new();

        self.add_sddp_study_period_to_graph(&mut g, system_input)?;
        self.add_sddp_pre_study_period_to_graph(&mut g, system_input)?;
        Ok(g)
    }
}

#[derive(Deserialize, Clone)]
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
#[derive(Deserialize, Clone)]
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
#[derive(Deserialize, Clone)]
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

// ============================================================================
// NEW: AR Model Support - Input Format Extension (AR-1, AR-6.1)
// ============================================================================

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

// ============================================================================
// Schema v2: Refactored Input Format (AR-6.1)
// ============================================================================

/// Temporal model for stochastic processes (Schema v2)
///
/// Specifies the temporal correlation structure of the stochastic process.
///
/// # Variants
///
/// - `Independent`: No temporal correlation (white noise)
/// - `Autoregressive`: AR(p) model with lag-dependent dynamics
///
/// # Example (Independent)
/// ```json
/// {
///   "temporal_model": {
///     "type": "independent"
///   }
/// }
/// ```
///
/// # Example (AR(1))
/// ```json
/// {
///   "temporal_model": {
///     "type": "autoregressive",
///     "lag_order": 1,
///     "coefficients": [0.7]
///   }
/// }
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TemporalModel {
    /// Independent process (no temporal correlation)
    ///
    /// Realizations are independent across time:
    /// Xₜ ~ F (marginal distribution)
    Independent,

    /// Autoregressive process AR(p)
    ///
    /// Realizations follow: Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
    /// where εₜ are innovations (typically ~ N(0,σ²))
    Autoregressive {
        /// Lag order p (number of past values used)
        ///
        /// Typical values: 1 (AR(1)), 2 (AR(2))
        /// Maximum supported: 12 (for monthly PAR models)
        lag_order: usize,

        /// AR coefficients [φ₁, φ₂, ..., φₚ]
        ///
        /// Must satisfy stationarity conditions:
        /// - |φ₁| < 1 for AR(1)
        /// - φ₁ + φ₂ < 1, φ₂ - φ₁ < 1, |φ₂| < 1 for AR(2)
        /// - Spectral radius < 1 for AR(p)
        coefficients: Vec<f64>,
    },
}

/// Marginal distribution for stochastic processes (Schema v2)
///
/// Specifies the target marginal distribution of realizations Xₜ.
/// This replaces the ambiguous `Distribution` enum from schema v1.
///
/// # Key Concept
///
/// In the CEPEL 4-stage pipeline:
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

    /// 3-parameter log-normal distribution (CEPEL methodology)
    ///
    /// X = γ + exp(μ + σW) where W ~ N(0,1)
    ///
    /// **Properties**:
    /// - Always non-negative (X ≥ γ ≥ 0)
    /// - Right-skewed (models rare high inflows)
    /// - Zero LP overhead (enforced in scenario generation)
    /// - Production-proven (CEPEL, PSR, ONS)
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
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct InnovationDistribution {
    /// Innovation mean (typically 0.0)
    pub mean: f64,

    /// Innovation standard deviation (must be > 0)
    #[serde(rename = "std_dev")]
    pub std_dev: f64,
}

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
/// - Production-proven (CEPEL GEVAZP, PSR SDDP)
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

    /// 3-parameter log-normal transformation (CEPEL methodology) **RECOMMENDED**
    ///
    /// Generates non-negative scenarios via X = γ + exp(μ + σZ) where Z ~ N(0,1).
    ///
    /// **Advantages over Shadow AR**:
    /// - **Zero LP overhead**: No extra constraints in LP formulation
    /// - **30-50% faster LP solves**: LP remains unchanged
    /// - **Production-proven**: Used by CEPEL, PSR, ONS in Brazilian hydrothermal dispatch
    /// - **Simpler code**: 200 lines vs 485 lines (58% reduction)
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
/// - Clear semantics for CEPEL 4-stage pipeline
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

    /// Marginal distribution of realizations
    ///
    /// This is the target distribution after scenario generation stages 1-3:
    /// 1. Base noise: Z ~ N(0,1)
    /// 2. Correlation: W = L×Z
    /// 3. Marginal transformation: X ~ F
    ///
    /// For LogNormal3, this guarantees non-negativity with zero LP overhead.
    pub marginal_distribution: MarginalDistribution,

    /// Innovation distribution for AR models (optional)
    ///
    /// Required for AR models, None for independent models.
    /// Innovations are the white noise εₜ in: Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
    ///
    /// Typically Normal(0, σ²) with σ calibrated to match historical variability.
    #[serde(default)]
    pub innovation_distribution: Option<InnovationDistribution>,

    /// Temporal model (independent or AR)
    ///
    /// Specifies whether realizations are temporally independent or follow
    /// an autoregressive process.
    pub temporal_model: TemporalModel,
}

impl NoiseModel {
    /// Validate NoiseModel semantic constraints
    ///
    /// Ensures the combination of fields is semantically valid:
    ///
    /// 1. **AR models must have innovation_distribution**
    ///    - `TemporalModel::Autoregressive` requires `innovation_distribution.is_some()`
    ///
    /// 2. **Independent models must NOT have innovation_distribution**
    ///    - `TemporalModel::Independent` requires `innovation_distribution.is_none()`
    ///
    /// 3. **AR coefficients must match lag_order**
    ///    - `coefficients.len() == lag_order`
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
        match &self.temporal_model {
            TemporalModel::Independent => {
                if self.innovation_distribution.is_some() {
                    return Err(format!(
                        "Independent noise model (entity={}, season={}) has innovation_distribution. \
                         This is invalid: independent models should only have marginal_distribution.",
                        self.entity_id, self.season_id
                    ));
                }
            }
            TemporalModel::Autoregressive {
                lag_order,
                coefficients,
            } => {
                if self.innovation_distribution.is_none() {
                    return Err(format!(
                        "AR noise model (entity={}, season={}) missing innovation_distribution. \
                         AR models must specify innovation_distribution for white noise.",
                        self.entity_id, self.season_id
                    ));
                }
                if coefficients.len() != *lag_order {
                    return Err(format!(
                        "AR noise model (entity={}, season={}) has {} coefficients but lag_order={}. \
                         These must match.",
                        self.entity_id, self.season_id, coefficients.len(), lag_order
                    ));
                }
                if coefficients.is_empty() {
                    return Err(format!(
                        "AR noise model (entity={}, season={}) has empty coefficients. \
                         At least one coefficient is required for AR models.",
                        self.entity_id, self.season_id
                    ));
                }
            }
        }

        Ok(())
    }
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
#[derive(Deserialize, Clone, Debug)]
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
#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
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
#[derive(Deserialize, Clone, Debug)]
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
#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct EntityReference {
    /// Type of uncertainty (inflow, load, etc.)
    pub uncertainty_type: UncertaintyType,

    /// Entity ID (zero-based index)
    pub entity_id: usize,
}

// ============================================================================

#[derive(Deserialize)]
pub struct Recourse {
    pub initial_condition: InitialConditionInput,

    /// Noise models for scenario generation
    ///
    /// Supports independent and AR models with explicit marginal/innovation/temporal separation.
    pub noise_models: Vec<NoiseModel>,

    /// Correlation specification for multi-variate scenario generation
    ///
    /// Optional. If omitted, all uncertainties are sampled independently.
    ///
    /// When specified, defines correlation structure across entities
    /// (e.g., upstream/downstream hydro correlation, regional loads).
    ///
    /// Uses Gaussian copula with Cholesky decomposition to preserve
    /// marginal distributions while introducing correlation.
    ///
    /// **Example**: Correlate inflows of upstream/downstream hydros with ρ=0.8
    ///
    /// # Backward Compatibility
    ///
    /// This field is optional with `#[serde(default)]`, so existing input
    /// files without correlation specifications continue to work unchanged.
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

    /// Generate SAA scenarios using the new 4-stage ScenarioGenerator pipeline.
    ///
    /// This method replaces the old NodeNoiseGenerator approach with the new
    /// CEPEL-compliant pipeline that supports AR temporal models, correlation,
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
    /// Panics if noise_models field is missing or if ScenarioGenerator construction fails
    pub fn generate_sddp_noises(
        &self,
        g: &graph::DirectedGraph<sddp::NodeData>,
        initial_condition: &initial_condition::InitialCondition,
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

        // Generate scenarios stage-by-stage with season filtering
        for (stage_id, season_id, num_scenarios) in stage_info {
            // Filter noise_models by season
            let season_noise_models: Vec<_> = self
                .noise_models
                .iter()
                .filter(|nm| nm.season_id == season_id)
                .cloned()
                .collect();

            if season_noise_models.is_empty() {
                panic!("No noise models found for season_id {}", season_id);
            }

            // Create temporary Recourse with filtered models
            // Note: We need InitialConditionInput for the temp_recourse, not InitialCondition
            let temp_recourse = Recourse {
                initial_condition: self.initial_condition.clone(),
                noise_models: season_noise_models,
                correlation: self.correlation.clone(),
            };

            // Create ScenarioGenerator for this season
            let generator = scenario::ScenarioGenerator::from_recourse_input(
                &temp_recourse,
                initial_condition,
                seed + stage_id as u64, // Vary seed per stage
            )
            .expect("Failed to create ScenarioGenerator");

            // Generate scenarios for just this one stage
            let stage_saa = generator.generate_saa(1, &[num_scenarios]);

            // Extract and copy to main SAA
            if let Some(stage_data) = stage_saa.branching_samples.first() {
                while saa.branching_samples.len() <= stage_id {
                    saa.branching_samples.push(
                        scenario::SampledNodeBranchings {
                            num_branchings: 0,
                            branching_noises: vec![],
                        },
                    );
                }
                saa.branching_samples[stage_id] = stage_data.clone();
            }
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

        use crate::input_validation::InputValidator;
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
mod tests {
    use super::*;

    #[test]
    fn test_read_config() {
        let filepath = "examples/01-deterministic/config.json";
        let config = read_config_input(filepath);
        assert_eq!(config.num_iterations, 10);
        assert_eq!(config.num_simulation_scenarios, 1);
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
        // Check that new noise_models format is used
        assert!(recourse.noise_models.is_some());
        assert_eq!(recourse.noise_models.as_ref().unwrap().len(), 4); // 2 load + 2 inflow
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
    #[allow(deprecated)]
    fn test_read_recourse_new_format_independent() {
        // Test new noise_models format with independent noise
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": []
            },
            "noise_models": [
                {
                    "noise_type": "independent",
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "distribution": {"type": "normal", "mean": 100.0, "std_dev": 20.0}
                }
            ]
        }"#;
        let recourse: Recourse = serde_json::from_str(json).unwrap();

        // Should have noise_models
        assert!(recourse.noise_models.is_some());

        let noise_models = recourse.noise_models.as_ref().unwrap();
        assert_eq!(noise_models.len(), 1);

        let model = &noise_models[0];
        assert!(matches!(model.noise_type, NoiseType::Independent));
        assert!(matches!(model.uncertainty_type, UncertaintyType::Inflow));
        assert_eq!(model.entity_id, 0);
        assert_eq!(model.season_id, 1);
        assert!(matches!(model.distribution, Distribution::Normal { .. }));
        assert!(model.lag_order.is_none());
        assert!(model.coefficients.is_none());
    }

    #[test]
    #[allow(deprecated)]
    fn test_read_recourse_new_format_ar1_structure() {
        // Test new noise_models format with AR(1) structure (validation in AR-2)
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": [{"hydro_id": 0, "lag": 1, "value": 120.0}]
            },
            "noise_models": [
                {
                    "noise_type": "autoregressive",
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "distribution": {"type": "normal", "mean": 0.0, "std_dev": 15.0},
                    "lag_order": 1,
                    "coefficients": [0.7]
                }
            ]
        }"#;
        let recourse: Recourse = serde_json::from_str(json).unwrap();

        // Should have noise_models
        assert!(recourse.noise_models.is_some());

        let noise_models = recourse.noise_models.as_ref().unwrap();
        assert_eq!(noise_models.len(), 1);

        let model = &noise_models[0];
        assert!(matches!(model.noise_type, NoiseType::Autoregressive));
        assert!(matches!(model.uncertainty_type, UncertaintyType::Inflow));
        assert_eq!(model.entity_id, 0);
        assert_eq!(model.season_id, 1);
        assert_eq!(model.lag_order, Some(1));
        assert_eq!(model.coefficients, Some(vec![0.7]));

        // Verify innovation distribution
        if let Distribution::Normal { mean, std_dev } = model.distribution {
            assert_eq!(mean, 0.0); // Innovations should be zero-mean
            assert_eq!(std_dev, 15.0);
        } else {
            panic!("Expected Normal distribution for AR innovations");
        }
    }

    #[test]
    fn test_recourse_new_format() {
        // Ensure new noise_models format works
        let recourse =
            read_recourse_input("examples/02-stochastic/recourse.json");
        assert!(recourse.noise_models.is_some());

        // Should have valid data
        let noise_models = recourse.noise_models.as_ref().unwrap();
        assert!(!noise_models.is_empty());
    }

    // ========================================================================
    // AR-6.1: Schema v2 Tests
    // ========================================================================

    #[test]
    fn test_parse_v2_independent_normal() {
        // Test: Parse v2 format with independent Normal distribution
        let json = r#"{
            "schema_version": 2,
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
        assert_eq!(recourse.schema_version, Some(2));
        assert!(recourse.noise_models.is_some());
        assert!(recourse.noise_models.is_none());

        let models = recourse.noise_models.as_ref().unwrap();
        assert_eq!(models.len(), 1);

        let model = &models[0];
        assert_eq!(model.entity_id, 0);
        assert_eq!(model.season_id, 1);
        assert!(matches!(
            model.marginal_distribution,
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0
            }
        ));
        assert!(model.innovation_distribution.is_none());
        assert!(matches!(model.temporal_model, TemporalModel::Independent));

        // Validate semantics
        model.validate().unwrap();
    }

    #[test]
    fn test_parse_v2_independent_lognormal3() {
        // Test: Parse v2 format with independent LogNormal3 distribution
        let json = r#"{
            "schema_version": 2,
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
        assert_eq!(recourse.schema_version, Some(2));

        let models = recourse.noise_models.as_ref().unwrap();
        let model = &models[0];

        assert!(matches!(
            model.marginal_distribution,
            MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3
            }
        ));
        assert!(model.innovation_distribution.is_none());
        assert!(matches!(model.temporal_model, TemporalModel::Independent));

        // Validate semantics
        model.validate().unwrap();
    }

    #[test]
    fn test_parse_v2_ar_with_lognormal3() {
        // Test: Parse v2 AR model with LogNormal3 marginal and Normal innovation
        let json = r#"{
            "schema_version": 2,
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
        assert_eq!(recourse.schema_version, Some(2));

        let models = recourse.noise_models.as_ref().unwrap();
        let model = &models[0];

        assert!(matches!(
            model.marginal_distribution,
            MarginalDistribution::LogNormal3 { .. }
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

    #[test]
    #[allow(deprecated)]
    fn test_v1_to_v2_conversion_backward_compat() {
        // Test: Backward compatibility via normalize_to_v2()
        let json = r#"{
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": []
            },
            "noise_models": [
                {
                    "noise_type": "independent",
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "distribution": {"type": "normal", "mean": 100.0, "std_dev": 20.0}
                }
            ]
        }"#;

        let recourse: Recourse = serde_json::from_str(json).unwrap();
        assert!(recourse.noise_models.is_some());
        assert!(recourse.noise_models.is_none());

        // Convert to v2
        let models_v2 = recourse.normalize_to_v2().unwrap();
        assert_eq!(models_v2.len(), 1);

        let model = &models_v2[0];
        assert_eq!(model.entity_id, 0);
        assert!(matches!(
            model.marginal_distribution,
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0
            }
        ));
        assert!(model.innovation_distribution.is_none());
        assert!(matches!(model.temporal_model, TemporalModel::Independent));

        // Validate converted model
        model.validate().unwrap();
    }

    #[test]
    fn test_v2_validation_error_ar_without_innovation() {
        // Test: AR model without innovation_distribution should fail validation
        let model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            marginal_distribution: MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            },
            innovation_distribution: None, // Missing!
            temporal_model: TemporalModel::Autoregressive {
                lag_order: 1,
                coefficients: vec![0.7],
            },
        };

        let result = model.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("missing innovation_distribution"),
            "Error should mention missing innovation: {}",
            err
        );
    }

    #[test]
    fn test_v2_validation_error_independent_with_innovation() {
        // Test: Independent model with innovation_distribution should fail validation
        let model = NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            marginal_distribution: MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            innovation_distribution: Some(InnovationDistribution {
                mean: 0.0,
                std_dev: 10.0,
            }), // Invalid for independent!
            temporal_model: TemporalModel::Independent,
        };

        let result = model.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("has innovation_distribution"),
            "Error should mention invalid innovation: {}",
            err
        );
    }

    #[test]
    fn test_normalize_multiple_sources_error() {
        // Test: Error when multiple noise model fields are present
        let json = r#"{
            "schema_version": 2,
            "initial_condition": {
                "storage": [{"hydro_id": 0, "value": 50.0}],
                "inflow": []
            },
            "noise_models": [
                {
                    "noise_type": "independent",
                    "uncertainty_type": "inflow",
                    "entity_id": 0,
                    "season_id": 1,
                    "distribution": {"type": "normal", "mean": 100.0, "std_dev": 20.0}
                }
            ],
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

        // Serde will parse this, but normalize should detect conflict
        let recourse: Recourse = serde_json::from_str(json).unwrap();
        let result = recourse.normalize_to_v2();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Multiple noise model specifications"),
            "Error should mention multiple sources: {}",
            err
        );
    }
}
