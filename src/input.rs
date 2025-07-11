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
        if ids.iter().find(|id| **id == elem_id).is_none() {
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
                .expect(&format!(
                    "Error adding edge {} -> {}",
                    edge_input.source_id, edge_input.target_id
                ));
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
            .add_edge(initial_condition_node_id, self.nodes.get(0).unwrap().id)
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
        return Self {
            config,
            system,
            graph,
            recourse,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_config() {
        let filepath = "example/config.json";
        let config = read_config_input(filepath);
        assert_eq!(config.num_iterations, 32);
        assert_eq!(config.num_simulation_scenarios, 128);
    }

    #[test]
    fn test_read_system() {
        let filepath = "example/system.json";
        let system = read_system_input(filepath);
        assert_eq!(system.buses.len(), 1);
        assert_eq!(system.lines.len(), 0);
        assert_eq!(system.thermals.len(), 2);
        assert_eq!(system.hydros.len(), 1);
    }

    #[test]
    fn test_build_sddp_system() {
        let filepath = "example/system.json";
        let system = read_system_input(filepath);
        system.build_sddp_system();
    }

    #[test]
    fn test_read_recourse() {
        let filepath = "example/recourse.json";
        let recourse = read_recourse_input(filepath);
        assert_eq!(recourse.initial_condition.storage.len(), 1);
        assert_eq!(recourse.uncertainties.len(), 12);
    }

    #[test]
    fn test_read_input() {
        let path = "example";
        let input = Input::build(path);
        assert_eq!(input.config.num_iterations, 32);
    }
}
