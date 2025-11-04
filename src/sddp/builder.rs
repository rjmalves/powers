use crate::error::PowersError;
use crate::graph::DirectedGraph;
use crate::initial_condition::InitialCondition;
use crate::input::{Config, GraphInput, Input, Recourse, SystemInput};
use crate::scenario::{NoiseGenerator, SAA};
use crate::sddp::{NodeData, SddpAlgorithm, SddpInstance};
use crate::subproblem::StudyPeriodKind;
use crate::system::System;

use rand_distr::Normal;

#[derive(Debug, Clone)]
enum InflowSpec {
    NotSet,
    Deterministic(Vec<Vec<f64>>),
    Stochastic {
        scenarios: Vec<Vec<Vec<f64>>>,
        probabilities: Vec<Vec<f64>>,
    },
}

#[derive(Debug, Clone)]
enum LoadSpec {
    NotSet,
    Deterministic(Vec<Vec<f64>>),
    Stochastic(Vec<Vec<Vec<f64>>>),
}

/// High-level builder for SDDP algorithm instances (simplified API).
///
/// **Limitations**: This builder only supports Independent noise models (no AR/PAR dynamics).
/// For production use with AR or PAR models, use `SddpInstanceBuilder::from_paths()` instead,
/// which reads `unified_specs` from JSON and properly handles temporal dependencies.
///
pub struct SddpBuilder {
    system_factory: Option<Box<dyn Fn() -> System>>,
    initial_storage: Option<Vec<f64>>,
    num_stages: Option<usize>,
    inflows: InflowSpec,
    loads: LoadSpec,
    seed: u64,
}

impl SddpBuilder {
    pub fn new() -> Self {
        Self {
            system_factory: None,
            initial_storage: None,
            num_stages: None,
            inflows: InflowSpec::NotSet,
            loads: LoadSpec::NotSet,
            seed: 42,
        }
    }

    /// Set the power system factory.
    pub fn system_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> System + 'static,
    {
        self.system_factory = Some(Box::new(factory));
        self
    }

    /// Set initial storage levels for all hydros.
    pub fn initial_storage(mut self, storage: Vec<f64>) -> Self {
        self.initial_storage = Some(storage);
        self
    }

    /// Set the number of decision stages.
    pub fn num_stages(mut self, num_stages: usize) -> Self {
        self.num_stages = Some(num_stages);
        self
    }

    /// Set random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set deterministic inflows (single scenario per stage).
    pub fn deterministic_inflows(mut self, inflows: Vec<Vec<f64>>) -> Self {
        self.inflows = InflowSpec::Deterministic(inflows);
        self
    }

    /// Set stochastic inflows (multiple scenarios per stage).
    pub fn stochastic_inflows(mut self, scenarios: Vec<Vec<Vec<f64>>>) -> Self {
        self.inflows = InflowSpec::Stochastic {
            scenarios,
            probabilities: vec![],
        };
        self
    }

    /// Set scenario probabilities for stochastic inflows.
    pub fn scenario_probabilities(
        mut self,
        probabilities: Vec<Vec<f64>>,
    ) -> Self {
        // Combine with existing scenarios
        if let InflowSpec::Stochastic { scenarios, .. } = &self.inflows {
            self.inflows = InflowSpec::Stochastic {
                scenarios: scenarios.clone(),
                probabilities,
            };
        } else {
            panic!("scenario_probabilities() must be called after stochastic_inflows()");
        }
        self
    }

    /// Set deterministic loads (single value per stage).
    pub fn deterministic_loads(mut self, loads: Vec<Vec<f64>>) -> Self {
        self.loads = LoadSpec::Deterministic(loads);
        self
    }

    /// Set stochastic loads (multiple scenarios per stage).
    pub fn stochastic_loads(mut self, scenarios: Vec<Vec<Vec<f64>>>) -> Self {
        self.loads = LoadSpec::Stochastic(scenarios);
        self
    }

    /// Build the SDDP algorithm instance.
    ///
    /// Validates all required fields, constructs the graph and SAA, and creates
    /// the final `SddpAlgorithm` instance.
    ///
    pub fn build(self) -> Result<SddpAlgorithm, String> {
        let system_factory = self
            .system_factory
            .ok_or_else(|| "system_factory is required".to_string())?;
        let initial_storage = self
            .initial_storage
            .ok_or_else(|| "initial_storage is required".to_string())?;
        let num_stages = self
            .num_stages
            .ok_or_else(|| "num_stages is required".to_string())?;
        let seed = self.seed;
        let inflows = self.inflows;

        let system_for_validation = system_factory();
        if num_stages == 0 {
            return Err("num_stages must be greater than 0".to_string());
        }
        if matches!(inflows, InflowSpec::NotSet) {
            return Err("inflows are required (use deterministic_inflows() or stochastic_inflows())".to_string());
        }
        if initial_storage.len() != system_for_validation.meta.hydros_count {
            return Err(format!(
                "initial_storage length ({}) must match number of hydros ({})",
                initial_storage.len(),
                system_for_validation.meta.hydros_count
            ));
        }

        let graph = build_graph(&system_factory, num_stages, "storage")?;
        let initial_condition = InitialCondition::new(initial_storage, vec![]);

        SddpAlgorithm::new(graph, initial_condition, seed)
    }

    /// Build the SDDP algorithm along with its SAA (for training).
    pub fn build_with_saa(self) -> Result<(SddpAlgorithm, SAA), String> {
        let system_factory = self
            .system_factory
            .ok_or_else(|| "system_factory is required".to_string())?;
        let initial_storage = self
            .initial_storage
            .ok_or_else(|| "initial_storage is required".to_string())?;
        let num_stages = self
            .num_stages
            .ok_or_else(|| "num_stages is required".to_string())?;
        let seed = self.seed;
        let inflows = self.inflows;

        let system_for_validation = system_factory();
        if num_stages == 0 {
            return Err("num_stages must be greater than 0".to_string());
        }
        if matches!(inflows, InflowSpec::NotSet) {
            return Err("inflows are required (use deterministic_inflows() or stochastic_inflows())".to_string());
        }
        if initial_storage.len() != system_for_validation.meta.hydros_count {
            return Err(format!(
                "initial_storage length ({}) must match number of hydros ({})",
                initial_storage.len(),
                system_for_validation.meta.hydros_count
            ));
        }

        let graph = build_graph(&system_factory, num_stages, "storage")?;
        let initial_condition = InitialCondition::new(initial_storage, vec![]);
        let saa = build_saa(
            &system_for_validation,
            num_stages,
            &inflows,
            &self.loads,
            seed,
        )?;
        let sddp = SddpAlgorithm::new(graph, initial_condition, seed)?;

        Ok((sddp, saa))
    }
}

/// Helper function to create empty unified specs for builder test utilities
/// Create default Independent uncertainty models for programmatic builder.
///
/// These models enable inflow variables to be created in the LP, allowing
/// inflows from SAA to be properly incorporated into water balance.
///
/// Uses standard normal parameters (μ=0, σ=1) so that:
/// - Transform: Y_t = 0 + 1*Z'_t = Z'_t
/// - AR (independent): Z'_t = ε_t  
/// - Result: Physical inflow Y_t = innovation ε_t from SAA
fn create_default_uncertainty_models(
    system: &System,
) -> std::sync::Arc<Vec<crate::temporal_model::TemporalModel>> {
    use crate::input::{MarginalDistribution, UncertaintyType};

    let num_seasons = 12; // Default monthly seasons
    let mut models = Vec::new();

    // Create Independent model (PAR(0)) for each hydro
    for hydro_id in 0..system.meta.hydros_count {
        // Standard normal seasonal distributions (μ=0, σ=1)
        let seasonal_distributions: Vec<MarginalDistribution> = (0
            ..num_seasons)
            .map(|_season_id| MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            })
            .collect();

        let means = vec![0.0; num_seasons];
        let stds = vec![1.0; num_seasons];
        let ar_orders = vec![0; num_seasons]; // Independent = PAR(0)
        let ar_coefficients = vec![vec![]; num_seasons]; // No AR coefficients

        models.push(
            crate::temporal_model::TemporalModel::from_par(
                UncertaintyType::Inflow,
                hydro_id,
                num_seasons,
                means,
                stds,
                seasonal_distributions,
                ar_orders,
                ar_coefficients,
            )
            .expect("Failed to create default TemporalModel"),
        );
    }

    std::sync::Arc::new(models)
}

/// Compute season IDs for PreStudy nodes via cycle-back from first Study node
///
/// PreStudy nodes represent historical time periods leading up to the study start.
/// Their season IDs should cycle backward from the first Study node's season to
/// ensure correct seasonal parameters are used for observation→residual transforms.
///
fn compute_prestudy_season_ids(
    first_study_season: usize,
    lag_order: usize,
    num_seasons: usize,
) -> Vec<usize> {
    assert!(
        num_seasons > 0,
        "num_seasons must be > 0 for season computation"
    );

    let num_pre_study_nodes = 1 + lag_order;
    let mut season_ids = Vec::with_capacity(num_pre_study_nodes);

    // Cycle backward from first_study_season, ordered newest to oldest
    for offset in 0..num_pre_study_nodes {
        let season_id = first_study_season
            .wrapping_sub(offset)
            .wrapping_add(num_seasons)
            % num_seasons;

        season_ids.push(season_id);
    }

    season_ids
}

/// Build a simple graph for SddpBuilder (simplified builder for tests/simple cases).
///
/// This function is used by `SddpBuilder`, which provides a simplified API
/// for deterministic/stochastic scenarios without AR dynamics. It always creates
/// graphs with a single PreStudy node (lag_order=0).
///
fn build_graph(
    system_factory: &dyn Fn() -> System,
    num_stages: usize,
    state_choice: &str,
) -> Result<DirectedGraph<NodeData>, String> {
    let mut graph = DirectedGraph::<NodeData>::new();

    let lag_order = match state_choice {
        "storage" => 0,
        "storage_and_inflow" => 0,
        _ => {
            return Err(format!(
                "Unknown state_choice: '{}'. Valid options: 'storage', 'storage_and_inflow'",
                state_choice
            ));
        }
    };

    // Create pre-study nodes: 1 + lag_order total
    let num_pre_study_nodes = 1 + lag_order;
    let mut pre_study_ids = Vec::with_capacity(num_pre_study_nodes);

    let first_study_season = 1;
    let num_seasons = 12;
    let prestudy_season_ids =
        compute_prestudy_season_ids(first_study_season, lag_order, num_seasons);

    // Create default uncertainty models for the builder
    let system = system_factory();
    let uncertainty_models = create_default_uncertainty_models(&system);

    for pre_idx in 0..num_pre_study_nodes {
        let node_id = -(lag_order as isize - pre_idx as isize);
        let season_id_idx = num_pre_study_nodes - 1 - pre_idx;
        let season_id = prestudy_season_ids[season_id_idx];

        let pre_study_id = graph
            .add_node(NodeData::new(
                node_id,
                0,
                season_id,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                StudyPeriodKind::PreStudy,
                system_factory(),
                "expectation",
                uncertainty_models.clone(),
                state_choice,
                1,
            )?)
            .map_err(|e| {
                format!("Failed to add PreStudy node {}: {:?}", node_id, e)
            })?;

        pre_study_ids.push(pre_study_id);
    }

    for i in 0..num_pre_study_nodes.saturating_sub(1) {
        graph
            .add_edge(pre_study_ids[i], pre_study_ids[i + 1])
            .map_err(|e| {
                format!("Failed to connect PreStudy nodes: {:?}", e)
            })?;
    }

    let last_pre_study_id = *pre_study_ids.last().unwrap();
    let mut previous_node_id = last_pre_study_id;

    for stage in 1..=num_stages {
        let stage_id = graph
            .add_node(NodeData::new(
                stage as isize,
                stage,
                stage,
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                StudyPeriodKind::Study,
                system_factory(),
                "expectation",
                uncertainty_models.clone(),
                state_choice,
                1,
            )?)
            .map_err(|e| {
                format!("Failed to add Study node for stage {}: {:?}", stage, e)
            })?;

        graph.add_edge(previous_node_id, stage_id).map_err(|e| {
            format!("Failed to add edge for stage {}: {:?}", stage, e)
        })?;

        previous_node_id = stage_id;
    }

    Ok(graph)
}

fn build_saa(
    system: &System,
    num_stages: usize,
    inflows: &InflowSpec,
    loads: &LoadSpec,
    seed: u64,
) -> Result<SAA, String> {
    match inflows {
        InflowSpec::NotSet => {
            Err("Inflows not set (should be caught earlier)".to_string())
        }
        InflowSpec::Deterministic(inflows) => {
            build_deterministic_saa(system, num_stages, inflows, loads, seed)
        }
        InflowSpec::Stochastic {
            scenarios,
            probabilities,
        } => build_stochastic_saa(
            system,
            num_stages,
            scenarios,
            probabilities,
            loads,
            seed,
        ),
    }
}

/// Build deterministic SAA (single scenario per stage).
fn build_deterministic_saa(
    system: &System,
    num_stages: usize,
    inflows: &[Vec<f64>],
    loads: &LoadSpec,
    seed: u64,
) -> Result<SAA, String> {
    if inflows.len() != num_stages {
        return Err(format!(
            "deterministic_inflows length ({}) must match num_stages ({})",
            inflows.len(),
            num_stages
        ));
    }

    for (stage, stage_inflows) in inflows.iter().enumerate() {
        if stage_inflows.len() != system.meta.hydros_count {
            return Err(format!(
                "Stage {} inflows length ({}) must match number of hydros ({})",
                stage + 1,
                stage_inflows.len(),
                system.meta.hydros_count
            ));
        }
    }

    match loads {
        LoadSpec::NotSet => {}
        LoadSpec::Deterministic(load_values) => {
            if load_values.len() != num_stages {
                return Err(format!(
                    "deterministic_loads length ({}) must match num_stages ({})",
                    load_values.len(),
                    num_stages
                ));
            }

            for (stage, stage_loads) in load_values.iter().enumerate() {
                if stage_loads.len() != system.meta.buses_count {
                    return Err(format!(
                        "Stage {} load must match number of buses (expected {}, got {})",
                        stage + 1,
                        system.meta.buses_count,
                        stage_loads.len()
                    ));
                }
            }
        }
        LoadSpec::Stochastic(_) => {
            return Err(
                "Cannot use stochastic loads with deterministic inflows"
                    .to_string(),
            );
        }
    }

    // Create NoiseGenerator with deterministic distributions
    let mut generator = NoiseGenerator::new();

    let prestudy_load_value = match loads {
        LoadSpec::NotSet => 0.0,
        LoadSpec::Deterministic(load_values) => load_values[0][0],
        LoadSpec::Stochastic(_) => unreachable!(),
    };
    let prestudy_load = vec![Normal::new(prestudy_load_value, 0.0).unwrap()];
    let prestudy_inflow =
        vec![Normal::new(0.0, 0.0).unwrap(); system.meta.hydros_count];
    generator.add_node_generator(prestudy_load, prestudy_inflow, 1);

    // Add deterministic generators for each stage
    for (stage_idx, stage_inflows) in inflows.iter().enumerate() {
        let load_values = match loads {
            LoadSpec::NotSet => vec![0.0; system.meta.buses_count],
            LoadSpec::Deterministic(load_values) => {
                load_values[stage_idx].clone()
            }
            LoadSpec::Stochastic(_) => unreachable!(),
        };
        let load_dists = load_values
            .iter()
            .map(|&load| Normal::new(load, 0.0).unwrap())
            .collect::<Vec<Normal<f64>>>();
        let inflow_dists: Vec<Normal<f64>> = stage_inflows
            .iter()
            .map(|&inflow| Normal::new(inflow, 0.0).unwrap())
            .collect();

        generator.add_node_generator(load_dists, inflow_dists, 1); // 1 branching (deterministic)
    }

    Ok(generator.generate(seed))
}

/// Build stochastic SAA (multiple scenarios per stage).
fn build_stochastic_saa(
    system: &System,
    num_stages: usize,
    scenarios: &[Vec<Vec<f64>>],
    probabilities: &[Vec<f64>],
    loads: &LoadSpec,
    seed: u64,
) -> Result<SAA, String> {
    if scenarios.len() != num_stages {
        return Err(format!(
            "stochastic_inflows length ({}) must match num_stages ({})",
            scenarios.len(),
            num_stages
        ));
    }
    if probabilities.len() != num_stages {
        return Err(format!(
            "scenario_probabilities length ({}) must match num_stages ({})",
            probabilities.len(),
            num_stages
        ));
    }

    for stage in 0..num_stages {
        let stage_scenarios = &scenarios[stage];
        let stage_probs = &probabilities[stage];

        if stage_scenarios.len() != stage_probs.len() {
            return Err(format!(
                "Stage {}: number of scenarios ({}) must match number of probabilities ({})",
                stage + 1,
                stage_scenarios.len(),
                stage_probs.len()
            ));
        }

        let prob_sum: f64 = stage_probs.iter().sum();
        if (prob_sum - 1.0).abs() > 1e-6 {
            return Err(format!(
                "Stage {}: probabilities must sum to 1.0 (got {:.6})",
                stage + 1,
                prob_sum
            ));
        }

        for (scenario_idx, scenario) in stage_scenarios.iter().enumerate() {
            if scenario.len() != system.meta.hydros_count {
                return Err(format!(
                    "Stage {}, scenario {}: inflows length ({}) must match number of hydros ({})",
                    stage + 1,
                    scenario_idx,
                    scenario.len(),
                    system.meta.hydros_count
                ));
            }
        }
    }

    match loads {
        LoadSpec::NotSet => {}
        LoadSpec::Deterministic(load_values) => {
            if load_values.len() != num_stages {
                return Err(format!(
                    "deterministic_loads length ({}) must match num_stages ({})",
                    load_values.len(),
                    num_stages
                ));
            }

            for (stage, loads) in load_values.iter().enumerate() {
                if loads.len() != system.meta.buses_count {
                    return Err(format!(
                        "Stage {} load must match number of buses (expected {}, got {})",
                        stage + 1,
                        system.meta.buses_count,
                        loads.len()
                    ));
                }
            }
        }
        LoadSpec::Stochastic(load_scenarios) => {
            if load_scenarios.len() != num_stages {
                return Err(format!(
                    "stochastic_loads length ({}) must match num_stages ({})",
                    load_scenarios.len(),
                    num_stages
                ));
            }
            for stage in 0..num_stages {
                let stage_load_scenarios = &load_scenarios[stage];
                let stage_inflow_scenarios = &scenarios[stage];

                if stage_load_scenarios.len() != stage_inflow_scenarios.len() {
                    return Err(format!(
                        "Stage {}: number of load scenarios ({}) must match number of inflow scenarios ({})",
                        stage + 1,
                        stage_load_scenarios.len(),
                        stage_inflow_scenarios.len()
                    ));
                }

                for (scenario_idx, loads) in
                    stage_load_scenarios.iter().enumerate()
                {
                    if loads.len() != system.meta.buses_count {
                        return Err(format!(
                            "Stage {}, scenario {}: load must match number of buses (expected {}, got {})",
                            stage + 1,
                            scenario_idx,
                            system.meta.buses_count,
                            loads.len()
                        ));
                    }
                }
            }
        }
    }

    // Build SAA manually with set_noises_by_stage for discrete scenarios
    let mut generator = NoiseGenerator::new();

    let prestudy_load_value = match loads {
        LoadSpec::NotSet => 0.0,
        LoadSpec::Deterministic(load_values) => load_values[0][0],
        LoadSpec::Stochastic(load_scenarios) => load_scenarios[0][0][0],
    };
    let prestudy_load = vec![Normal::new(prestudy_load_value, 0.0).unwrap()];
    let prestudy_inflow =
        vec![Normal::new(0.0, 0.0).unwrap(); system.meta.hydros_count];
    generator.add_node_generator(prestudy_load, prestudy_inflow, 1);

    for stage_scenarios in scenarios.iter() {
        let num_scenarios = stage_scenarios.len();

        let load_dist = vec![Normal::new(0.0, 0.0).unwrap()];
        let inflow_dists: Vec<Normal<f64>> = (0..system.meta.hydros_count)
            .map(|_| Normal::new(0.0, 0.0).unwrap())
            .collect();

        generator.add_node_generator(load_dist, inflow_dists, num_scenarios);
    }

    let mut saa = generator.generate(seed);

    for (stage_idx, stage_scenarios) in scenarios.iter().enumerate() {
        let node_idx = stage_idx + 1;
        let num_scenarios = stage_scenarios.len();
        let load_noises: Vec<Vec<f64>> = match loads {
            LoadSpec::NotSet => {
                vec![vec![0.0; num_scenarios]; system.meta.buses_count]
            }
            LoadSpec::Deterministic(load_values) => {
                let stage_load_values = &load_values[stage_idx];
                stage_load_values
                    .iter()
                    .map(|&load| vec![load; num_scenarios])
                    .collect()
            }
            LoadSpec::Stochastic(load_scenarios) => {
                let stage_load_scenarios = &load_scenarios[stage_idx];
                let num_buses = system.meta.buses_count;
                let mut transposed = vec![vec![0.0; num_scenarios]; num_buses];
                for (scenario_idx, scenario_loads) in
                    stage_load_scenarios.iter().enumerate()
                {
                    for (bus_idx, &load) in scenario_loads.iter().enumerate() {
                        transposed[bus_idx][scenario_idx] = load;
                    }
                }
                transposed
            }
        };

        let mut inflow_noises: Vec<Vec<f64>> =
            vec![vec![0.0; num_scenarios]; system.meta.hydros_count];

        for (scenario_idx, scenario_inflows) in
            stage_scenarios.iter().enumerate()
        {
            for (hydro_idx, &inflow) in scenario_inflows.iter().enumerate() {
                inflow_noises[hydro_idx][scenario_idx] = inflow;
            }
        }

        saa.set_noises_by_stage(
            node_idx,
            num_scenarios,
            system.meta.buses_count,
            system.meta.hydros_count,
            load_noises,
            inflow_noises,
        );
    }

    Ok(saa)
}

impl Default for SddpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helper: Create a minimal system
    fn create_test_system() -> System {
        let json = r#"{
            "buses": [{"id": 0, "deficit_cost": 50.0}],
            "lines": [],
            "thermals": [{
                "id": 0,
                "bus_id": 0,
                "cost": 10.0,
                "min_generation": 0.0,
                "max_generation": 30.0
            }],
            "hydros": [{
                "id": 0,
                "downstream_hydro_id": null,
                "bus_id": 0,
                "productivity": 1.0,
                "min_storage": 0.0,
                "max_storage": 100.0,
                "min_turbined_flow": 0.0,
                "max_turbined_flow": 50.0,
                "spillage_penalty": 0.01
            }]
        }"#;
        let input: crate::input::SystemInput =
            serde_json::from_str(json).expect("Failed to parse test system");
        input.build_sddp_system()
    }

    #[test]
    fn test_builder_requires_system() {
        let result = SddpBuilder::new()
            .initial_storage(vec![50.0])
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("system"));
        }
    }

    #[test]
    fn test_builder_requires_storage() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("storage"));
        }
    }

    #[test]
    fn test_builder_requires_stages() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("num_stages"));
        }
    }

    #[test]
    fn test_builder_requires_inflows() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("inflows"));
        }
    }

    #[test]
    fn test_builder_validates_positive_stages() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(0)
            .deterministic_inflows(vec![])
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("greater than 0"));
        }
    }

    #[test]
    fn test_builder_deterministic_scenario() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .seed(42)
            .build();

        assert!(result.is_ok(), "Builder failed: {:?}", result.err());
    }

    #[test]
    fn test_builder_deterministic_loads() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .deterministic_loads(vec![vec![40.0], vec![40.0]])
            .seed(42)
            .build();

        assert!(
            result.is_ok(),
            "Builder with loads failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_builder_loads_defaults_to_zero() {
        // Build without specifying loads - should default to 0.0 MW
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .seed(42)
            .build();

        assert!(
            result.is_ok(),
            "Builder without loads failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_builder_validates_load_stage_count() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .deterministic_loads(vec![vec![40.0]]) // Wrong: 1 load for 2 stages
            .seed(42)
            .build_with_saa();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("must match num_stages"));
        }
    }

    #[test]
    fn test_builder_rejects_stochastic_loads_with_deterministic_inflows() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .stochastic_loads(vec![
                vec![vec![40.0]],                         // Stage 1: 1 scenario
                vec![vec![35.0], vec![40.0], vec![45.0]], // Stage 2: 3 scenarios
            ])
            .seed(42)
            .build_with_saa();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains(
                "Cannot use stochastic loads with deterministic inflows"
            ));
        }
    }

    #[test]
    fn test_builder_stochastic_loads_with_stochastic_inflows() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .stochastic_inflows(vec![
                vec![vec![30.0]], // Stage 1: 1 scenario
                vec![
                    vec![20.0], // Stage 2: dry
                    vec![40.0], // Stage 2: average
                    vec![60.0], // Stage 2: wet
                ],
            ])
            .scenario_probabilities(vec![
                vec![1.0],              // Stage 1: 100%
                vec![0.25, 0.50, 0.25], // Stage 2: dry/avg/wet
            ])
            .stochastic_loads(vec![
                vec![vec![40.0]], // Stage 1: 1 scenario (40 MW)
                vec![vec![35.0], vec![40.0], vec![45.0]], // Stage 2: 3 scenarios (low/med/high demand)
            ])
            .seed(42)
            .build();

        assert!(
            result.is_ok(),
            "Builder with stochastic loads failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_builder_validates_stochastic_loads_scenario_count() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .stochastic_inflows(vec![
                vec![vec![30.0]], // Stage 1: 1 scenario
                vec![
                    vec![20.0], // Stage 2: dry
                    vec![40.0], // Stage 2: average
                    vec![60.0], // Stage 2: wet
                ],
            ])
            .scenario_probabilities(vec![
                vec![1.0],              // Stage 1: 100%
                vec![0.25, 0.50, 0.25], // Stage 2: dry/avg/wet
            ])
            .stochastic_loads(vec![
                vec![vec![40.0]],             // Stage 1: 1 scenario ✓
                vec![vec![35.0], vec![40.0]], // Stage 2: 2 scenarios ✗ (should be 3)
            ])
            .seed(42)
            .build_with_saa();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("must match number of inflow scenarios"));
        }
    }

    // Note: Stochastic and validation tests will be added in next phase

    // ========== PAR-007: Multi-Node Pre-Study Tests ==========

    #[test]
    fn test_build_graph_storage_single_prestudy() {
        // Test that "storage" state creates 1 pre-study node
        let system_factory = || create_test_system();
        let graph = build_graph(&system_factory, 3, "storage", false)
            .expect("Failed to build graph");

        // Should have 4 nodes total: 1 pre-study + 3 study
        assert_eq!(graph.node_count(), 4);

        // Check pre-study node ID
        let pre_study_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|node| matches!(node.data.kind, StudyPeriodKind::PreStudy))
            .map(|node| node.id)
            .collect();

        assert_eq!(
            pre_study_nodes.len(),
            1,
            "Should have exactly 1 pre-study node"
        );

        let pre_study_id = pre_study_nodes[0];
        let pre_study_node = graph.get_node(pre_study_id).unwrap();
        assert_eq!(
            pre_study_node.data.id, 0,
            "Pre-study node should have ID 0 (lag_order=0)"
        );
        assert_eq!(pre_study_node.data.state_choice, "storage");
    }

    #[test]
    fn test_build_graph_storage_and_inflow_multiple_prestudy() {
        // Test that "storage_and_inflow" with lag_order=0 (naive) creates 1 pre-study node
        let system_factory = || create_test_system();
        let graph =
            build_graph(&system_factory, 3, "storage_and_inflow", false)
                .expect("Failed to build graph");

        // Naive process has lag_order=0, so should still be 1 pre-study node
        assert_eq!(graph.node_count(), 4); // 1 pre-study + 3 study

        let pre_study_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|node| matches!(node.data.kind, StudyPeriodKind::PreStudy))
            .map(|node| node.id)
            .collect();

        assert_eq!(pre_study_nodes.len(), 1);

        let pre_study_node = graph.get_node(pre_study_nodes[0]).unwrap();
        assert_eq!(pre_study_node.data.id, 0);
        assert_eq!(pre_study_node.data.state_choice, "storage_and_inflow");
    }

    #[test]
    fn test_build_graph_sequential_prestudy_connections() {
        // Test that pre-study nodes are connected sequentially
        let system_factory = || create_test_system();
        let graph = build_graph(&system_factory, 2, "storage", false)
            .expect("Failed to build graph");

        // Get pre-study node
        let pre_study_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|node| matches!(node.data.kind, StudyPeriodKind::PreStudy))
            .map(|node| node.id)
            .collect();

        let pre_study_id = pre_study_nodes[0];

        // Get first study node
        let study_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|node| matches!(node.data.kind, StudyPeriodKind::Study))
            .map(|node| node.id)
            .collect();

        // Pre-study should connect to first study node
        let children = graph
            .get_children(pre_study_id)
            .expect("Pre-study should have children");
        assert_eq!(children.len(), 1, "Pre-study should have 1 successor");
        assert!(
            study_nodes.contains(&children[0]),
            "Pre-study should connect to study node"
        );
    }

    #[test]
    fn test_build_graph_study_nodes_sequential() {
        // Test that study nodes are numbered 1..=num_stages
        let system_factory = || create_test_system();
        let num_stages = 5;
        let graph = build_graph(&system_factory, num_stages, "storage", false)
            .expect("Failed to build graph");

        let mut study_node_ids: Vec<_> = graph
            .iter_nodes()
            .filter(|node| matches!(node.data.kind, StudyPeriodKind::Study))
            .map(|node| node.data.id)
            .collect();

        study_node_ids.sort();

        let expected: Vec<_> = (1..=num_stages as isize).collect();
        assert_eq!(
            study_node_ids, expected,
            "Study nodes should be numbered 1..=num_stages"
        );
    }

    #[test]
    fn test_build_graph_invalid_state_choice() {
        // Test that invalid state_choice returns error
        let system_factory = || create_test_system();
        let result = build_graph(&system_factory, 2, "invalid_choice", false);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("Unknown state_choice"));
            assert!(e.contains("invalid_choice"));
            assert!(e.contains("storage"));
            assert!(e.contains("storage_and_inflow"));
        }
    }

    #[test]
    fn test_build_graph_total_node_count() {
        // Test total node count = num_pre_study + num_stages
        let system_factory = || create_test_system();

        // storage: 1 pre-study + N study = N+1 total
        let graph_storage = build_graph(&system_factory, 10, "storage", false)
            .expect("Failed to build graph");
        assert_eq!(graph_storage.node_count(), 11); // 1 + 10

        // storage_and_inflow with naive (lag_order=0): same as storage
        let graph_inflow =
            build_graph(&system_factory, 10, "storage_and_inflow", false)
                .expect("Failed to build graph");
        assert_eq!(graph_inflow.node_count(), 11); // 1 + 10
    }

    #[test]
    fn test_build_graph_all_nodes_have_system() {
        // Test that all nodes have valid system instances
        let system_factory = || create_test_system();
        let graph = build_graph(&system_factory, 3, "storage", false)
            .expect("Failed to build graph");

        for node in graph.iter_nodes() {
            assert_eq!(
                node.data.system.meta.hydros_count, 1,
                "Each node should have valid system"
            );
        }
    }
}

/// Builder for flexible SDDP instance construction with parameter modification.
///
/// **This is the production builder** that supports all features including AR/PAR models.
/// It reads `unified_specs` from JSON files and properly constructs graphs with the
/// correct number of PreStudy nodes based on the AR order.
///
pub struct SddpInstanceBuilder {
    system: SystemInput,
    graph: GraphInput,
    recourse: Recourse,
    config: Config,
}

impl SddpInstanceBuilder {
    /// Load inputs from individual file paths with validation.
    pub fn from_paths(
        config_path: impl AsRef<std::path::Path>,
        system_path: impl AsRef<std::path::Path>,
        graph_path: impl AsRef<std::path::Path>,
        recourse_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, PowersError> {
        let input = Input::from_paths(
            config_path.as_ref(),
            system_path.as_ref(),
            graph_path.as_ref(),
            recourse_path.as_ref(),
        )?;

        Ok(Self {
            system: input.system,
            graph: input.graph,
            recourse: input.recourse,
            config: input.config,
        })
    }

    /// Modify the number of forward passes per iteration.
    #[inline]
    pub fn with_num_forward_passes(
        mut self,
        num_forward_passes: usize,
    ) -> Self {
        self.config.num_forward_passes = num_forward_passes;
        self
    }

    /// Modify the number of SDDP iterations.
    #[inline]
    pub fn with_num_iterations(mut self, num_iterations: usize) -> Self {
        self.config.num_iterations = num_iterations;
        self
    }

    /// Modify the random seed for deterministic sampling.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Modify the number of threads for parallel execution.
    #[inline]
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = Some(num_threads);
        self
    }

    /// Build the SDDP algorithm instance with the configured parameters.
    ///
    /// This method:
    /// 1. Builds the SDDP graph from graph input
    /// 2. Creates initial condition from recourse data
    /// 3. Generates SAA scenarios using the configured seed
    /// 4. Creates the SDDP algorithm
    /// 5. Returns `SddpInstance` ready for training/simulation
    ///
    pub fn build(self) -> Result<SddpInstance, PowersError> {
        let seed = self.config.seed;

        let node_data_graph = self
            .graph
            .build_sddp_graph(&self.system, &self.recourse)
            .map_err(|e| {
                PowersError::Other(format!("Failed to build SDDP graph: {}", e))
            })?;

        let initial_condition = self.recourse.build_sddp_initial_condition();
        let saa = self.recourse.generate_sddp_noises(
            &node_data_graph,
            &initial_condition,
            seed,
        );

        let algorithm =
            SddpAlgorithm::new(node_data_graph, initial_condition, seed)
                .map_err(PowersError::Other)?;

        Ok(SddpInstance::new(algorithm, self.config, saa))
    }
}

#[cfg(test)]
mod instance_builder_tests {
    use super::*;

    #[test]
    fn test_builder_from_paths_loads_inputs() {
        let builder = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        );

        assert!(builder.is_ok(), "Builder should load valid inputs");
        let builder = builder.unwrap();

        // Verify config was loaded
        assert!(builder.config.num_iterations > 0);
        assert!(builder.config.num_forward_passes > 0);

        // Verify system was loaded (spot check)
        assert!(!builder.system.buses.is_empty());
        assert!(!builder.system.hydros.is_empty());
    }

    #[test]
    fn test_builder_with_num_forward_passes() {
        let builder = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        )
        .unwrap();

        let original_num_fwd = builder.config.num_forward_passes;
        let builder = builder.with_num_forward_passes(99);

        assert_eq!(builder.config.num_forward_passes, 99);
        assert_ne!(builder.config.num_forward_passes, original_num_fwd);
    }

    #[test]
    fn test_builder_with_num_iterations() {
        let builder = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        )
        .unwrap();

        let original_num_iters = builder.config.num_iterations;
        let builder = builder.with_num_iterations(50);

        assert_eq!(builder.config.num_iterations, 50);
        assert_ne!(builder.config.num_iterations, original_num_iters);
    }

    #[test]
    fn test_builder_with_seed() {
        let builder = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        )
        .unwrap();

        let original_seed = builder.config.seed;
        let builder = builder.with_seed(999);

        assert_eq!(builder.config.seed, 999);
        assert_ne!(builder.config.seed, original_seed);
    }

    #[test]
    fn test_builder_chain_multiple_modifiers() {
        let builder = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        )
        .unwrap()
        .with_num_iterations(10)
        .with_num_forward_passes(32)
        .with_seed(42);

        assert_eq!(builder.config.num_iterations, 10);
        assert_eq!(builder.config.num_forward_passes, 32);
        assert_eq!(builder.config.seed, 42);
    }

    #[test]
    fn test_builder_build_creates_valid_instance() {
        let sddp = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        )
        .unwrap()
        .build();

        assert!(
            sddp.is_ok(),
            "Builder should successfully build SddpInstance"
        );
    }

    #[test]
    fn test_builder_with_modified_config() {
        let mut sddp = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        )
        .unwrap()
        .with_num_iterations(5)
        .with_num_forward_passes(2)
        .with_seed(999)
        .build()
        .unwrap();

        let result = sddp.train();
        assert!(
            result.is_ok(),
            "Training should succeed with modified config"
        );
    }

    #[test]
    fn test_compute_prestudy_season_ids_no_wrap() {
        // Case: first_study=5, lag_order=2, num_seasons=12
        // Expected: [5, 4, 3] (newest to oldest: May, April, March)
        let season_ids = compute_prestudy_season_ids(5, 2, 12);
        assert_eq!(season_ids.len(), 3);
        assert_eq!(season_ids, vec![5, 4, 3]);
    }

    #[test]
    fn test_compute_prestudy_season_ids_with_wrap() {
        // Case: first_study=1, lag_order=3, num_seasons=12
        // Expected: [1, 0, 11, 10] (newest to oldest: Jan, Dec, Nov, Oct)
        let season_ids = compute_prestudy_season_ids(1, 3, 12);
        assert_eq!(season_ids.len(), 4);
        assert_eq!(season_ids, vec![1, 0, 11, 10]);
    }

    #[test]
    fn test_compute_prestudy_season_ids_wrap_from_zero() {
        // Case: first_study=0, lag_order=1, num_seasons=12
        // Expected: [0, 11] (newest to oldest: Dec, Nov)
        let season_ids = compute_prestudy_season_ids(0, 1, 12);
        assert_eq!(season_ids.len(), 2);
        assert_eq!(season_ids, vec![0, 11]);
    }

    #[test]
    fn test_compute_prestudy_season_ids_last_season() {
        // Case: first_study=11, lag_order=2, num_seasons=12
        // Expected: [11, 10, 9] (newest to oldest: Nov, Oct, Sep)
        let season_ids = compute_prestudy_season_ids(11, 2, 12);
        assert_eq!(season_ids.len(), 3);
        assert_eq!(season_ids, vec![11, 10, 9]);
    }

    #[test]
    fn test_compute_prestudy_season_ids_lag_order_zero() {
        // Case: lag_order=0 → single PreStudy node
        // Expected: [5] (same as first Study season)
        let season_ids = compute_prestudy_season_ids(5, 0, 12);
        assert_eq!(season_ids.len(), 1);
        assert_eq!(season_ids, vec![5]);
    }

    #[test]
    fn test_compute_prestudy_season_ids_non_standard_seasons() {
        // Case: 4-season model (quarterly)
        // first_study=2, lag_order=1, num_seasons=4
        // Expected: [2, 1] (newest to oldest: Q3, Q2)
        let season_ids = compute_prestudy_season_ids(2, 1, 4);
        assert_eq!(season_ids.len(), 2);
        assert_eq!(season_ids, vec![2, 1]);
    }

    #[test]
    fn test_compute_prestudy_season_ids_wrap_quarterly() {
        // Case: 4-season model with wraparound
        // first_study=0, lag_order=2, num_seasons=4
        // Expected: [0, 3, 2] (newest to oldest: Q1, Q4, Q3)
        let season_ids = compute_prestudy_season_ids(0, 2, 4);
        assert_eq!(season_ids.len(), 3);
        assert_eq!(season_ids, vec![0, 3, 2]);
    }

    #[test]
    #[should_panic(expected = "num_seasons must be > 0")]
    fn test_compute_prestudy_season_ids_panics_on_zero_seasons() {
        // Should panic with defensive assertion
        compute_prestudy_season_ids(5, 2, 0);
    }
}
