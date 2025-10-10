use crate::graph;
use crate::initial_condition;
use crate::scenario;
use crate::sddp;
use crate::subproblem;
use crate::system;
use rand_distr::{LogNormal, Normal};
use serde::Deserialize;
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

#[derive(Deserialize)]
pub struct InitialStorage {
    pub hydro_id: usize,
    pub value: f64,
}

#[derive(Deserialize)]
pub struct PastInflow {
    pub hydro_id: usize,
    pub lag: usize,
    pub value: f64,
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
pub struct LognormalParams {
    pub mu: f64,
    pub sigma: f64,
}

#[derive(Deserialize)]
pub struct InflowDistribution {
    pub hydro_id: usize,
    pub lognormal: LognormalParams,
}

#[derive(Deserialize)]
pub struct UncertaintyDistributions {
    pub load: Vec<LoadDistribution>,
    pub inflow: Vec<InflowDistribution>,
}

#[derive(Deserialize)]
pub struct SeasonalUncertaintyInput {
    pub season_id: usize,
    pub num_branchings: usize,
    pub distributions: UncertaintyDistributions,
}

#[derive(Deserialize)]
pub struct Recourse {
    pub initial_condition: InitialConditionInput,
    pub uncertainties: Vec<SeasonalUncertaintyInput>,
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
        initial_condition::InitialCondition::new(storage, vec![])
    }

    pub fn generate_sddp_noises(
        &self,
        g: &graph::DirectedGraph<sddp::NodeData>,
        seed: u64,
    ) -> scenario::SAA {
        let mut scenario_generator = scenario::NoiseGenerator::new();

        for node_id in 0..g.node_count() {
            let node = g.get_node(node_id).unwrap();
            let num_buses = node.data.system.meta.buses_count;
            let num_hydros = node.data.system.meta.hydros_count;
            let node_uncertainties = self
                .uncertainties
                .iter()
                .find(|s| s.season_id == node.data.season_id);
            match node_uncertainties {
                Some(node_uncertainties) => {
                    let scenario_bus_ids: Vec<usize> = node_uncertainties
                        .distributions
                        .load
                        .iter()
                        .map(|s| s.bus_id)
                        .collect();
                    validate_id_range(&scenario_bus_ids, "load distributions");
                    validate_entity_count(
                        scenario_bus_ids.as_slice(),
                        num_buses,
                        "bus loads",
                    );
                    let scenario_hydro_ids: Vec<usize> = node_uncertainties
                        .distributions
                        .inflow
                        .iter()
                        .map(|s| s.hydro_id)
                        .collect();
                    validate_id_range(
                        &scenario_hydro_ids,
                        "inflow distributions",
                    );
                    validate_entity_count(
                        scenario_hydro_ids.as_slice(),
                        num_hydros,
                        "hydro inflows",
                    );
                    let mut load_distributions =
                        Vec::<Normal<f64>>::with_capacity(num_buses);
                    let mut inflow_distributions =
                        Vec::<LogNormal<f64>>::with_capacity(num_hydros);
                    for id in 0..num_buses {
                        let load_distribution = node_uncertainties
                            .distributions
                            .load
                            .iter()
                            .find(|s| s.bus_id == id)
                            .unwrap();
                        load_distributions.push(
                            Normal::new(
                                load_distribution.normal.mu,
                                load_distribution.normal.sigma,
                            )
                            .unwrap(),
                        );
                    }
                    for id in 0..num_hydros {
                        let inflow_distribution = node_uncertainties
                            .distributions
                            .inflow
                            .iter()
                            .find(|s| s.hydro_id == id)
                            .unwrap();
                        inflow_distributions.push(
                            LogNormal::new(
                                inflow_distribution.lognormal.mu,
                                inflow_distribution.lognormal.sigma,
                            )
                            .unwrap(),
                        );
                    }
                    scenario_generator.add_node_generator(
                        load_distributions,
                        inflow_distributions,
                        node_uncertainties.num_branchings,
                    );
                }
                None => panic!(
                    "Could not find load distributions for node {}",
                    node.id
                ),
            }
        }
        scenario_generator.generate(seed)
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
        assert_eq!(recourse.uncertainties.len(), 2);
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
}
