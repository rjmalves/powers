//! Buffer sizing computation for memory pre-allocation.
//!
//! Computes all buffer dimensions from input configuration at startup,
//! enabling pre-allocation of buffers used in hot paths.

use crate::graph::DirectedGraph;
use crate::input::Config;
use crate::sddp::NodeData;
use crate::system::System;
use crate::temporal_model::TemporalModel;

/// Per-node sizing information for heterogeneous SDDP graphs.
///
/// POWE.RS allows different nodes to have:
/// - Different state implementations (StorageState vs StorageAndInflowState)
/// - Different AR orders per hydro
/// - Different scenario counts
///
/// This struct captures these heterogeneous dimensions for accurate buffer pre-allocation.
#[derive(Debug, Clone)]
pub struct NodeSizing {
    /// Node ID in the graph
    pub node_id: usize,

    /// State vector dimension for this node
    /// - StorageState: num_hydros
    /// - StorageAndInflowState: num_hydros + sum(AR_orders)
    pub state_dimension: usize,

    /// Number of scenarios branching from this node
    pub num_scenarios: usize,

    /// Number of decision variables in subproblem
    pub subproblem_var_count: usize,

    /// Number of constraints (excluding cuts)
    pub subproblem_constraint_count: usize,

    /// State choice: "storage" or "storage_and_inflow"
    pub state_choice: String,
}

/// Computes all buffer dimensions from input configuration.
///
/// This struct centralizes sizing logic, capturing per-node heterogeneous
/// dimensions and computing aggregate statistics. All sizes are deterministic
/// and immutable after construction.
///
/// # Heterogeneous Node Support
///
/// POWE.RS nodes can have different:
/// - State implementations (StorageState vs StorageAndInflowState)
/// - State dimensions (varies 2-3x depending on AR orders)
/// - Scenario counts (varies by stage)
///
/// This struct captures these differences for accurate buffer pre-allocation.
///
/// # Fields Organization
///
/// ## Per-Node Dimensions (Heterogeneous)
/// - `node_sizing`: Vec of sizing info for each node
///
/// ## Aggregate Statistics (Derived)
/// - `max_state_dimension`: Largest state across all nodes
/// - `min_state_dimension`: Smallest state across all nodes
/// - `avg_state_dimension`: Average state dimension
/// - `max_scenarios_per_node`: Maximum branching factor
/// - `max_subproblem_vars`: Maximum variables across all nodes
///
/// ## System-Wide Dimensions (Uniform)
/// - `num_hydros`, `num_thermals`, `num_buses`, `num_lines`
/// - `num_stages`, `num_nodes`
/// - `max_iterations`, `num_forward_passes`, `num_simulations`
/// - `num_threads`
///
/// # Example
///
/// ```rust,ignore
/// use powers_rs::memory::SizingInfo;
///
/// let sizing = SizingInfo::from_input(&system, &graph, &config);
///
/// // Access per-node info
/// let node0 = sizing.node(0).unwrap();
/// println!("Node 0 state dim: {}", node0.state_dimension);
///
/// // Access aggregates
/// println!("Max state dim: {}", sizing.max_state_dimension);
/// println!("Estimated memory: {} MB", sizing.estimate_memory_bytes() / 1_000_000);
/// ```
#[derive(Debug, Clone)]
pub struct SizingInfo {
    // ========================================
    // PER-NODE DIMENSIONS (Heterogeneous)
    // ========================================
    /// Sizing information for each node in the graph.
    /// Index matches node_id.
    pub node_sizing: Vec<NodeSizing>,

    // ========================================
    // AGGREGATE STATISTICS (Derived)
    // ========================================
    /// Maximum state dimension across all nodes
    pub max_state_dimension: usize,

    /// Minimum state dimension across all nodes
    pub min_state_dimension: usize,

    /// Average state dimension across all nodes
    pub avg_state_dimension: f64,

    /// Maximum scenarios branching from any node
    pub max_scenarios_per_node: usize,

    /// Maximum subproblem variables across all nodes
    pub max_subproblem_vars: usize,

    // ========================================
    // SYSTEM-WIDE DIMENSIONS (Uniform)
    // ========================================
    /// Number of hydroelectric plants
    pub num_hydros: usize,

    /// Number of thermal plants
    pub num_thermals: usize,

    /// Number of electrical buses
    pub num_buses: usize,

    /// Number of transmission lines
    pub num_lines: usize,

    /// Number of decision stages
    pub num_stages: usize,

    /// Total nodes in scenario tree
    pub num_nodes: usize,

    // ========================================
    // TRAINING/SIMULATION DIMENSIONS
    // ========================================
    /// Maximum SDDP iterations
    pub max_iterations: usize,

    /// Forward passes per iteration
    pub num_forward_passes: usize,

    /// Out-of-sample simulation scenarios
    pub num_simulations: usize,

    /// Worker thread count
    pub num_threads: usize,
}

impl SizingInfo {
    /// Computes sizing information from input configuration.
    ///
    /// Extracts per-node dimensions from the SDDP graph and computes
    /// aggregate statistics for buffer pre-allocation.
    ///
    /// # Arguments
    ///
    /// * `system` - Power system configuration
    /// * `graph` - Scenario tree with NodeData
    /// * `config` - SDDP algorithm configuration
    ///
    /// # Performance
    ///
    /// O(n) where n is number of nodes. Completes in <50ms for large systems.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sizing = SizingInfo::from_input(&system, &graph, &config);
    /// assert!(sizing.max_state_dimension > 0);
    /// assert_eq!(sizing.node_sizing.len(), graph.node_count());
    /// ```
    pub fn from_input(
        system: &System,
        graph: &DirectedGraph<NodeData>,
        config: &Config,
    ) -> Self {
        // System dimensions
        let num_hydros = system.meta.hydros_count;
        let num_thermals = system.meta.thermals_count;
        let num_buses = system.meta.buses_count;
        let num_lines = system.meta.lines_count;

        // Compute per-node sizing
        let node_sizing: Vec<NodeSizing> = graph
            .iter_nodes()
            .map(|node| {
                let state_dim = compute_state_dimension_for_node(
                    num_hydros,
                    &node.data.state_choice,
                    &node.data.uncertainty_models,
                );

                let subproblem_var_count =
                    compute_variable_count(num_hydros, num_thermals, num_buses);

                let subproblem_constraint_count =
                    compute_constraint_count(num_hydros, num_buses, num_lines);

                NodeSizing {
                    node_id: node.id,
                    state_dimension: state_dim,
                    num_scenarios: node.data.num_scenarios,
                    subproblem_var_count,
                    subproblem_constraint_count,
                    state_choice: node.data.state_choice.clone(),
                }
            })
            .collect();

        // Compute aggregate statistics
        let max_state_dimension = node_sizing
            .iter()
            .map(|ns| ns.state_dimension)
            .max()
            .unwrap_or(0);

        let min_state_dimension = node_sizing
            .iter()
            .map(|ns| ns.state_dimension)
            .min()
            .unwrap_or(0);

        let avg_state_dimension = if !node_sizing.is_empty() {
            node_sizing
                .iter()
                .map(|ns| ns.state_dimension as f64)
                .sum::<f64>()
                / node_sizing.len() as f64
        } else {
            0.0
        };

        let max_scenarios_per_node = node_sizing
            .iter()
            .map(|ns| ns.num_scenarios)
            .max()
            .unwrap_or(0);

        let max_subproblem_vars = node_sizing
            .iter()
            .map(|ns| ns.subproblem_var_count)
            .max()
            .unwrap_or(0);

        // Graph dimensions
        let num_stages = graph.node_count();
        let num_nodes = num_stages;

        // Training dimensions
        let max_iterations = config.training.num_iterations;
        let num_forward_passes = config.training.num_forward_passes;

        // Simulation dimensions
        let num_simulations = config.simulation.num_scenarios.unwrap_or(0);

        // Parallelism
        let num_threads = config
            .general
            .num_threads
            .unwrap_or_else(rayon::current_num_threads);

        Self {
            node_sizing,
            max_state_dimension,
            min_state_dimension,
            avg_state_dimension,
            max_scenarios_per_node,
            max_subproblem_vars,
            num_hydros,
            num_thermals,
            num_buses,
            num_lines,
            num_stages,
            num_nodes,
            max_iterations,
            num_forward_passes,
            num_simulations,
            num_threads,
        }
    }

    /// Estimates total memory usage in bytes.
    ///
    /// Uses per-node state dimensions for accurate cut coefficient sizing.
    /// Accounts for major data structures:
    /// - Cuts (objective + coefficients per cut, per node)
    /// - Forward pass trajectories
    /// - Thread-local buffers
    ///
    /// # Returns
    ///
    /// Estimated peak memory usage in bytes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sizing = SizingInfo::from_input(&system, &graph, &config);
    /// let mb = sizing.estimate_memory_bytes() / 1_000_000;
    /// println!("Estimated memory: {} MB", mb);
    /// ```
    pub fn estimate_memory_bytes(&self) -> usize {
        // Cut storage: use per-node state dimensions
        let avg_cuts_per_node = 100;
        let total_cuts_memory: usize = self
            .node_sizing
            .iter()
            .map(|ns| {
                let cut_coefficients_size =
                    ns.state_dimension * std::mem::size_of::<f64>();
                let cut_size =
                    std::mem::size_of::<f64>() + cut_coefficients_size;
                avg_cuts_per_node * cut_size
            })
            .sum();

        // Forward pass trajectories: use max subproblem vars
        let realization_size =
            self.max_subproblem_vars * std::mem::size_of::<f64>();
        let trajectory_size = self.num_stages * realization_size;
        let forward_pass_memory =
            self.num_forward_passes * trajectory_size * self.max_iterations;

        // Thread-local buffers: use max state dimension
        let thread_buffer_size =
            self.max_state_dimension * std::mem::size_of::<f64>() * 10; // ~10 buffers per thread
        let thread_memory = self.num_threads * thread_buffer_size;

        total_cuts_memory + forward_pass_memory + thread_memory
    }

    /// Logs comprehensive sizing summary at INFO level.
    ///
    /// Outputs formatted sizing information with per-node statistics
    /// for diagnostics and debugging. Use this at startup to verify
    /// sizing computation.
    ///
    /// # Example Output
    ///
    /// ```text
    /// [INFO] Buffer Sizing Information:
    /// [INFO]   System: 156 hydros, 48 thermals, 32 buses, 64 lines
    /// [INFO]   State dimensions: min=156, max=390, avg=273.0
    /// [INFO]   State choices: 1 storage, 7 storage_and_inflow
    /// [INFO]   Graph: 8 stages, 8 nodes, max_scenarios=4
    /// [INFO]   Training: 32 iterations, 4 forward_passes
    /// [INFO]   Simulation: 128 scenarios
    /// [INFO]   Parallelism: 8 threads
    /// [INFO]   Estimated memory: 2400 MB
    /// ```
    pub fn log_summary(&self) {
        log::info!("Buffer Sizing Information:");
        log::info!(
            "  System: {} hydros, {} thermals, {} buses, {} lines",
            self.num_hydros,
            self.num_thermals,
            self.num_buses,
            self.num_lines
        );

        log::info!(
            "  State dimensions: min={}, max={}, avg={:.1}",
            self.min_state_dimension,
            self.max_state_dimension,
            self.avg_state_dimension
        );

        // Show state choice distribution
        let storage_count = self
            .node_sizing
            .iter()
            .filter(|ns| ns.state_choice == "storage")
            .count();
        let storage_and_inflow_count = self
            .node_sizing
            .iter()
            .filter(|ns| ns.state_choice == "storage_and_inflow")
            .count();

        log::info!(
            "  State choices: {} storage, {} storage_and_inflow",
            storage_count,
            storage_and_inflow_count
        );

        log::info!(
            "  Graph: {} stages, {} nodes, max_scenarios={}",
            self.num_stages,
            self.num_nodes,
            self.max_scenarios_per_node
        );

        log::info!(
            "  Training: {} iterations, {} forward_passes",
            self.max_iterations,
            self.num_forward_passes
        );

        log::info!("  Simulation: {} scenarios", self.num_simulations);
        log::info!("  Parallelism: {} threads", self.num_threads);

        log::info!(
            "  Estimated memory: {} MB",
            self.estimate_memory_bytes() / 1_000_000
        );
    }

    /// Gets sizing for a specific node.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node ID in the graph
    ///
    /// # Returns
    ///
    /// Reference to NodeSizing if node exists, None otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let node0 = sizing.node(0).unwrap();
    /// println!("State dim: {}", node0.state_dimension);
    /// ```
    pub fn node(&self, node_id: usize) -> Option<&NodeSizing> {
        self.node_sizing.get(node_id)
    }

    /// Gets state dimension for a specific node.
    ///
    /// Convenience method for accessing just the state dimension.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let state_dim = sizing.state_dimension_for_node(5).unwrap_or(0);
    /// ```
    pub fn state_dimension_for_node(&self, node_id: usize) -> Option<usize> {
        self.node(node_id).map(|ns| ns.state_dimension)
    }

    /// Checks if all nodes have uniform state dimensions.
    ///
    /// Returns true if min == max, meaning no heterogeneity.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if sizing.has_uniform_state_dimensions() {
    ///     println!("Can use uniform buffer allocation");
    /// }
    /// ```
    pub fn has_uniform_state_dimensions(&self) -> bool {
        self.max_state_dimension == self.min_state_dimension
    }

    /// Gets all nodes with a specific state choice.
    ///
    /// # Arguments
    ///
    /// * `choice` - "storage" or "storage_and_inflow"
    ///
    /// # Returns
    ///
    /// Vector of node IDs with the specified state choice.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let storage_nodes = sizing.nodes_with_state_choice("storage");
    /// println!("Found {} storage-only nodes", storage_nodes.len());
    /// ```
    pub fn nodes_with_state_choice(&self, choice: &str) -> Vec<usize> {
        self.node_sizing
            .iter()
            .filter(|ns| ns.state_choice == choice)
            .map(|ns| ns.node_id)
            .collect()
    }
}

/// Computes state dimension for a specific node.
///
/// Handles both StorageState and StorageAndInflowState by checking
/// the node's state_choice and uncertainty models.
///
/// # Arguments
///
/// * `num_hydros` - Number of hydroelectric plants
/// * `state_choice` - "storage" or "storage_and_inflow"
/// * `uncertainty_models` - Uncertainty models with AR orders
///
/// # Returns
///
/// State vector dimension for this node.
///
/// # Examples
///
/// ```rust,ignore
/// // StorageState: just storage
/// let dim = compute_state_dimension_for_node(156, "storage", &models);
/// assert_eq!(dim, 156);
///
/// // StorageAndInflowState: storage + inflow lags
/// let dim = compute_state_dimension_for_node(156, "storage_and_inflow", &models);
/// assert!(dim > 156);  // Includes AR lag states
/// ```
fn compute_state_dimension_for_node(
    num_hydros: usize,
    state_choice: &str,
    uncertainty_models: &[TemporalModel],
) -> usize {
    match state_choice {
        "storage" => num_hydros,
        "storage_and_inflow" => {
            // Only inflow lags contribute to state (not load lags)
            let inflow_lags: usize = uncertainty_models
                .iter()
                .filter(|tm| {
                    matches!(
                        tm.entity_type,
                        crate::input::UncertaintyType::Inflow
                    )
                })
                .map(|tm| tm.max_ar_order)
                .sum();

            num_hydros + inflow_lags
        }
        _ => {
            log::warn!(
                "Unknown state_choice '{}', defaulting to storage",
                state_choice
            );
            num_hydros
        }
    }
}

/// Computes number of variables in each subproblem.
///
/// Variables include:
/// - Hydro: 3 per plant (generation, spillage, end-storage)
/// - Thermal: 1 per plant (generation)
/// - Deficit: 1 per bus
/// - Future cost: 1 (alpha variable)
///
/// # Formula
///
/// ```text
/// var_count = 3*num_hydros + num_thermals + num_buses + 1
/// ```
///
/// # Arguments
///
/// * `num_hydros` - Number of hydroelectric plants
/// * `num_thermals` - Number of thermal plants
/// * `num_buses` - Number of electrical buses
///
/// # Returns
///
/// Total number of LP variables per subproblem.
fn compute_variable_count(
    num_hydros: usize,
    num_thermals: usize,
    num_buses: usize,
) -> usize {
    let hydro_vars = num_hydros * 3; // generation, spillage, storage
    let thermal_vars = num_thermals;
    let deficit_vars = num_buses;
    let future_cost_vars = 1; // alpha

    hydro_vars + thermal_vars + deficit_vars + future_cost_vars
}

/// Computes number of constraints in each subproblem (excluding cuts).
///
/// Constraints include:
/// - Hydro balance: 1 per plant
/// - Bus balance: 1 per bus
/// - Line limits: 2 per line (forward/reverse)
///
/// Note: Future cost cuts are dynamic and not counted here.
///
/// # Formula
///
/// ```text
/// constraint_count = num_hydros + num_buses + 2*num_lines
/// ```
///
/// # Arguments
///
/// * `num_hydros` - Number of hydroelectric plants
/// * `num_buses` - Number of electrical buses
/// * `num_lines` - Number of transmission lines
///
/// # Returns
///
/// Number of LP constraints per subproblem (excluding dynamic cuts).
fn compute_constraint_count(
    num_hydros: usize,
    num_buses: usize,
    num_lines: usize,
) -> usize {
    let hydro_balance = num_hydros;
    let bus_balance = num_buses;
    let line_limits = num_lines * 2; // forward and reverse

    hydro_balance + bus_balance + line_limits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{
        GeneralConfig, SimulationConfig, TrainingConfig, UncertaintyType,
    };
    use std::sync::Arc;

    fn make_test_system(
        num_hydros: usize,
        num_thermals: usize,
        num_buses: usize,
        num_lines: usize,
    ) -> System {
        let buses = (0..num_buses)
            .map(|id| crate::system::Bus::new(id, 1000.0))
            .collect();
        let lines = (0..num_lines)
            .map(|id| crate::system::Line::new(id, 0, 0, 100.0, 100.0, 1.0))
            .collect();
        let thermals = (0..num_thermals)
            .map(|id| crate::system::Thermal::new(id, 0, 50.0, 0.0, 100.0))
            .collect();
        let hydros = (0..num_hydros)
            .map(|id| {
                crate::system::Hydro::new(
                    id, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0,
                )
            })
            .collect();

        System::new(buses, lines, thermals, hydros)
    }

    fn make_test_node_data(
        node_id: usize,
        state_choice: &str,
        temporal_models: Vec<TemporalModel>,
        num_scenarios: usize,
    ) -> NodeData {
        NodeData {
            id: node_id as isize,
            stage_id: node_id,
            season_id: 0,
            start_date: chrono::Utc::now(),
            end_date: chrono::Utc::now(),
            kind: crate::subproblem::StudyPeriodKind::PreStudy,
            system: make_test_system(3, 2, 4, 5),
            risk_measure: Box::new(crate::risk_measure::Expectation::new()),
            uncertainty_models: Arc::new(temporal_models),
            state_choice: state_choice.to_string(),
            num_scenarios,
        }
    }

    fn make_test_graph(
        num_stages: usize,
        state_choice: &str,
    ) -> DirectedGraph<NodeData> {
        let mut graph = DirectedGraph::new();
        for i in 0..num_stages {
            let node = make_test_node_data(i, state_choice, vec![], 1);
            graph.add_node(node).unwrap();
        }
        graph
    }

    fn make_test_config(
        num_iterations: usize,
        num_forward_passes: usize,
        num_simulations: Option<usize>,
        num_threads: Option<usize>,
    ) -> Config {
        Config {
            general: GeneralConfig {
                seed: 42,
                num_threads,
            },
            training: TrainingConfig {
                num_iterations,
                num_forward_passes,
                enable_cut_selection: true,
            },
            simulation: SimulationConfig {
                num_scenarios: num_simulations,
            },
            output: crate::input::OutputConfig::default(),
            logging: crate::logging::LoggingConfig::default(),
        }
    }

    fn make_inflow_model(ar_order: usize) -> TemporalModel {
        if ar_order == 0 {
            TemporalModel::from_independent(
                UncertaintyType::Inflow,
                0,
                vec![100.0],
                vec![10.0],
                vec![crate::input::MarginalDistribution::Normal {
                    mean: 100.0,
                    std_dev: 10.0,
                }],
            )
            .unwrap()
        } else {
            TemporalModel::from_par(
                UncertaintyType::Inflow,
                0,
                1,
                vec![100.0],
                vec![10.0],
                vec![crate::input::MarginalDistribution::Normal {
                    mean: 100.0,
                    std_dev: 10.0,
                }],
                vec![ar_order],
                vec![vec![0.7; ar_order]],
            )
            .unwrap()
        }
    }

    #[test]
    fn test_compute_state_dimension_for_node_storage_only() {
        let state_dim = compute_state_dimension_for_node(3, "storage", &[]);
        assert_eq!(state_dim, 3, "Storage-only: dimension equals num_hydros");
    }

    #[test]
    fn test_compute_state_dimension_for_node_with_ar_lags() {
        let models = vec![
            make_inflow_model(1),
            make_inflow_model(0),
            make_inflow_model(2),
        ];
        let state_dim =
            compute_state_dimension_for_node(3, "storage_and_inflow", &models);
        assert_eq!(state_dim, 6, "3 hydros + (1+0+2) AR lags = 6");
    }

    #[test]
    fn test_compute_variable_count() {
        let var_count = compute_variable_count(3, 2, 4);
        // 3*3 (hydros) + 2 (thermals) + 4 (buses) + 1 (alpha) = 16
        assert_eq!(var_count, 16);
    }

    #[test]
    fn test_compute_constraint_count() {
        let constraint_count = compute_constraint_count(3, 4, 5);
        // 3 (hydro balance) + 4 (bus balance) + 2*5 (line limits) = 17
        assert_eq!(constraint_count, 17);
    }

    #[test]
    fn test_sizing_info_from_input_small() {
        let system = make_test_system(3, 2, 4, 5);
        let graph = make_test_graph(8, "storage");
        let config = make_test_config(10, 4, Some(100), Some(4));

        let sizing = SizingInfo::from_input(&system, &graph, &config);

        assert_eq!(sizing.num_hydros, 3);
        assert_eq!(sizing.num_thermals, 2);
        assert_eq!(sizing.num_buses, 4);
        assert_eq!(sizing.num_lines, 5);
        assert_eq!(sizing.max_state_dimension, 3);
        assert_eq!(sizing.min_state_dimension, 3);
        assert_eq!(sizing.num_stages, 8);
        assert_eq!(sizing.max_iterations, 10);
        assert_eq!(sizing.num_forward_passes, 4);
        assert_eq!(sizing.num_simulations, 100);
        assert_eq!(sizing.num_threads, 4);
        assert_eq!(sizing.max_subproblem_vars, 16);
    }

    #[test]
    fn test_sizing_info_with_ar_models() {
        let system = make_test_system(3, 0, 2, 1);

        // Create graph with storage_and_inflow nodes
        let mut graph = DirectedGraph::new();
        let temporal_models = vec![
            make_inflow_model(1),
            make_inflow_model(2),
            make_inflow_model(0),
        ];

        for i in 0..5 {
            let node = make_test_node_data(
                i,
                "storage_and_inflow",
                temporal_models.clone(),
                1,
            );
            graph.add_node(node).unwrap();
        }

        let config = make_test_config(20, 8, None, None);

        let sizing = SizingInfo::from_input(&system, &graph, &config);

        assert_eq!(sizing.max_state_dimension, 6); // 3 hydros + 3 lags
        assert_eq!(sizing.min_state_dimension, 6);
        assert!(sizing.num_threads > 0); // Should default to rayon thread count
    }

    #[test]
    fn test_estimate_memory_bytes() {
        let system = make_test_system(100, 50, 30, 40);
        let graph = make_test_graph(12, "storage");
        let config = make_test_config(32, 4, Some(128), Some(8));

        let sizing = SizingInfo::from_input(&system, &graph, &config);

        let memory = sizing.estimate_memory_bytes();
        assert!(
            memory > 1_000_000,
            "Memory estimate should be > 1MB for realistic system"
        );
        assert!(
            memory < 10_000_000_000,
            "Memory estimate should be < 10GB for realistic system"
        );
    }
}
