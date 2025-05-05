use crate::scenario;
use crate::sddp;
use rand_distr::LogNormal;
use serde::Deserialize;
use serde_json;
use std::fs;

#[derive(Deserialize)]
pub struct Config {
    pub num_iterations: usize,
    pub num_stages: usize,
    pub num_branchings: usize,
    pub num_simulation_scenarios: usize,
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
    pub fn build_sddp_system(&self) -> sddp::System {
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
        let mut buses = Vec::<sddp::Bus>::with_capacity(num_buses);
        for id in 0..num_buses {
            let bus = self.buses.iter().find(|b| b.id == id).unwrap();
            buses.push(sddp::Bus::new(id, bus.deficit_cost));
        }

        let num_lines = lines_ids.len();
        let mut lines = Vec::<sddp::Line>::with_capacity(num_lines);
        for id in 0..num_lines {
            let line = self.lines.iter().find(|l| l.id == id).unwrap();
            lines.push(sddp::Line::new(
                id,
                line.source_bus_id,
                line.target_bus_id,
                line.direct_capacity,
                line.reverse_capacity,
                line.exchange_penalty,
            ));
        }

        let num_thermals = thermals_ids.len();
        let mut thermals = Vec::<sddp::Thermal>::with_capacity(num_thermals);
        for id in 0..num_thermals {
            let thermal = self.thermals.iter().find(|t| t.id == id).unwrap();
            thermals.push(sddp::Thermal::new(
                id,
                thermal.bus_id,
                thermal.cost,
                thermal.min_generation,
                thermal.max_generation,
            ));
        }

        let num_hydros = hydros_ids.len();
        let mut hydros = Vec::<sddp::Hydro>::with_capacity(num_hydros);
        for id in 0..num_hydros {
            let hydro = self.hydros.iter().find(|h| h.id == id).unwrap();
            hydros.push(sddp::Hydro::new(
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

        sddp::System::new(buses, lines, thermals, hydros)
    }
}

#[derive(Deserialize)]
pub struct InitialState {
    pub hydro_id: usize,
    pub initial_storage: f64,
}

#[derive(Deserialize)]
pub struct Load {
    pub bus_id: usize,
    pub value: f64,
}

#[derive(Deserialize)]
pub struct StageLoads {
    pub stage_id: usize,
    pub values: Vec<Load>,
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
pub struct StageInflowDistributions {
    pub stage_id: usize,
    pub num_branchings: usize,
    pub distributions: Vec<InflowDistribution>,
}

#[derive(Deserialize)]
pub struct Recourse {
    pub initial_states: Vec<InitialState>,
    pub loads: Vec<StageLoads>,
    pub inflow_distributions: Vec<StageInflowDistributions>,
}

pub fn read_recourse_input(filepath: &str) -> Recourse {
    let contents =
        fs::read_to_string(filepath).expect("Error while reading config file");
    let parsed: Recourse = serde_json::from_str(&contents).unwrap();
    parsed
}

impl Recourse {
    pub fn build_sddp_initial_storages(&self) -> Vec<f64> {
        let initial_state_hydro_ids: Vec<usize> =
            self.initial_states.iter().map(|s| s.hydro_id).collect();
        validate_id_range(&initial_state_hydro_ids, "initial storages");
        let num_hydros = initial_state_hydro_ids.len();
        let mut initial_storages = Vec::<f64>::with_capacity(num_hydros);
        for id in 0..num_hydros {
            let s = self
                .initial_states
                .iter()
                .find(|s| s.hydro_id == id)
                .unwrap();
            initial_storages.push(s.initial_storage);
        }
        initial_storages
    }

    pub fn build_sddp_loads(
        &self,
        num_stages: usize,
        num_buses: usize,
    ) -> Vec<Vec<f64>> {
        let mut loads = Vec::<Vec<f64>>::with_capacity(num_buses);
        for stage in 0..num_stages {
            let stage_loads = self.loads.iter().find(|s| s.stage_id == stage);
            match stage_loads {
                Some(stage_loads) => {
                    let load_buses_ids: Vec<usize> =
                        stage_loads.values.iter().map(|s| s.bus_id).collect();
                    validate_id_range(&load_buses_ids, "loads");
                    validate_entity_count(
                        load_buses_ids.as_slice(),
                        num_buses,
                        "bus loads",
                    );
                    loads.push(vec![]);
                    for id in 0..num_buses {
                        let s = stage_loads
                            .values
                            .iter()
                            .find(|s| s.bus_id == id)
                            .unwrap();
                        loads[stage].push(s.value);
                    }
                }
                None => panic!("Could not find loads for stage {}", stage),
            }
        }
        loads
    }

    pub fn build_sddp_scenario_generator(
        &self,
        num_stages: usize,
        num_hydros: usize,
    ) -> scenario::ScenarioGenerator {
        let mut scenario_generator = scenario::ScenarioGenerator::new();

        for stage in 0..num_stages {
            let stage_inflows = self
                .inflow_distributions
                .iter()
                .find(|s| s.stage_id == stage);
            match stage_inflows {
                Some(stage_inflows) => {
                    let scenario_hydro_ids: Vec<usize> = stage_inflows
                        .distributions
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
                    let mut distributions =
                        Vec::<LogNormal<f64>>::with_capacity(num_hydros);
                    for id in 0..num_hydros {
                        let s = stage_inflows
                            .distributions
                            .iter()
                            .find(|s| s.hydro_id == id)
                            .unwrap();
                        distributions.push(
                            LogNormal::new(s.lognormal.mu, s.lognormal.sigma)
                                .unwrap(),
                        );
                    }
                    scenario_generator.add_stage_generator(
                        distributions,
                        stage_inflows.num_branchings,
                    );
                }
                None => panic!(
                    "Could not find inflow distributions for stage {}",
                    stage
                ),
            }
        }
        scenario_generator
    }
}

pub struct Input {
    pub config: Config,
    pub system: SystemInput,
    pub recourse: Recourse,
}

impl Input {
    pub fn build(path: &str) -> Self {
        let config = read_config_input(&(path.to_owned() + "/config.json"));
        let system = read_system_input(&(path.to_owned() + "/system.json"));
        let recourse =
            read_recourse_input(&(path.to_owned() + "/recourse.json"));
        return Self {
            config,
            system,
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
        assert_eq!(config.num_iterations, 1024);
        assert_eq!(config.num_stages, 12);
        assert_eq!(config.num_branchings, 10);
        assert_eq!(config.num_simulation_scenarios, 1000);
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
        assert_eq!(recourse.initial_states.len(), 1);
        assert_eq!(recourse.loads.len(), 12);
        assert_eq!(recourse.inflow_distributions.len(), 12);
    }

    #[test]
    fn test_read_input() {
        let path = "example";
        let input = Input::build(path);
        assert_eq!(input.config.num_branchings, 10);
    }
}
