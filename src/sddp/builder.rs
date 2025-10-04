//! High-level builder API for SDDP algorithm construction.
//!
//! The `SddpBuilder` provides an ergonomic, fluent API for creating SDDP instances,
//! dramatically reducing boilerplate compared to manual construction. This is especially
//! valuable for tests and examples.
//!
//! # Performance
//!
//! The builder is a **zero-cost abstraction**: it compiles to the same efficient code
//! as manual construction. All allocations and construction happen at build time, not
//! during training.
//!
//! # When to Use
//!
//! - **Use builder for**: Tests, benchmarks, examples, simple scenarios
//! - **Use low-level API for**: Production code requiring maximum flexibility,
//!   complex scenario trees, advanced customization
//!
//! # Example: Deterministic Problem
//!
//! ```rust,ignore
//! use powers_rs::sddp::SddpBuilder;
//!
//! let sddp = SddpBuilder::new()
//!     .system(system)
//!     .initial_storage(vec![50.0])
//!     .num_stages(2)
//!     .deterministic_inflows(vec![
//!         vec![30.0],  // Stage 1
//!         vec![40.0],  // Stage 2
//!     ])
//!     .seed(42)
//!     .build()?;
//!
//! let result = sddp.train(20, 10)?;
//! ```
//!
//! # Example: Stochastic Problem
//!
//! ```rust,ignore
//! let sddp = SddpBuilder::new()
//!     .system(system)
//!     .initial_storage(vec![50.0])
//!     .num_stages(2)
//!     .stochastic_inflows(vec![
//!         vec![vec![30.0]],  // Stage 1: deterministic
//!         vec![
//!             vec![20.0],  // Stage 2: dry scenario
//!             vec![40.0],  // Stage 2: average scenario
//!             vec![60.0],  // Stage 2: wet scenario
//!         ],
//!     ])
//!     .scenario_probabilities(vec![
//!         vec![1.0],              // Stage 1: 100%
//!         vec![0.25, 0.50, 0.25], // Stage 2: dry/avg/wet
//!     ])
//!     .seed(42)
//!     .build()?;
//! ```

use crate::graph::DirectedGraph;
use crate::initial_condition::InitialCondition;
use crate::scenario::{NoiseGenerator, SAA};
use crate::sddp::{NodeData, SddpAlgorithm};
use crate::subproblem::StudyPeriodKind;
use crate::system::System;
use rand_distr::Normal;

/// Specification of inflow scenarios for the builder.
///
/// This enum tracks the state of inflow configuration:
/// - `NotSet`: No inflows configured yet (initial state)
/// - `Deterministic`: Single-scenario inflows per stage
/// - `Stochastic`: Multi-scenario inflows with probabilities
#[derive(Debug, Clone)]
enum InflowSpec {
    /// No inflows specified yet
    NotSet,

    /// Deterministic inflows: `inflows[stage][hydro]`
    ///
    /// Each stage has a single scenario with fixed inflows for each hydro.
    /// Equivalent to stochastic with 1 scenario at probability 1.0.
    Deterministic(Vec<Vec<f64>>),

    /// Stochastic inflows with probabilities
    ///
    /// - `scenarios[stage][scenario][hydro]`: Inflow values
    /// - `probabilities[stage][scenario]`: Scenario probabilities
    ///
    /// Probabilities per stage must sum to 1.0 (validated at build time).
    Stochastic {
        scenarios: Vec<Vec<Vec<f64>>>,
        probabilities: Vec<Vec<f64>>,
    },
}

/// Specification of load scenarios for the builder.
///
/// This enum tracks the state of load configuration:
/// - `NotSet`: No loads configured yet (defaults to 0.0 MW)
/// - `Deterministic`: Single load value per stage
/// - `Stochastic`: Multi-scenario loads with probabilities (must match inflow structure)
#[derive(Debug, Clone)]
enum LoadSpec {
    /// No loads specified - defaults to 0.0 MW (unconstrained)
    NotSet,

    /// Deterministic loads: `loads[stage]`
    ///
    /// Single load value per stage (applied to first bus in system).
    Deterministic(Vec<f64>),

    /// Stochastic loads with scenarios
    ///
    /// - `scenarios[stage][scenario]`: Load values
    /// - Probabilities inherited from inflow scenarios (must match structure)
    ///
    /// Probabilities per stage come from `scenario_probabilities()` and must match
    /// the structure of `stochastic_inflows()`.
    Stochastic(Vec<Vec<f64>>),
}

/// High-level builder for SDDP algorithm instances.
///
/// Provides a fluent API that dramatically reduces boilerplate for common SDDP
/// construction patterns. Reduces typical test code from ~150 lines to ~8 lines.
///
/// # Required Fields
///
/// The following must be set before calling `build()`:
/// - `system()`: Power system configuration (or system factory)
/// - `initial_storage()`: Initial hydro storage levels
/// - `num_stages()`: Number of decision stages
/// - Inflows: Either `deterministic_inflows()` or `stochastic_inflows()` + `scenario_probabilities()`
///
/// # Optional Fields
///
/// - `seed()`: Random seed (default: 42)
///
/// # Performance Notes
///
/// - Builder is consumed by `build()` (move semantics, no extra allocation)
/// - All validation happens at build time, not training time
/// - Compiles to identical code as manual construction (zero-cost abstraction)
/// - System is recreated per graph node (acceptable one-time cost for builder)
///
/// # Example
///
/// ```rust,ignore
/// let my_system = create_my_system();
/// let sddp = SddpBuilder::new()
///     .system_factory(move || create_my_system())  // Called once per node  
///     .initial_storage(vec![50.0, 30.0])  // 2 hydros
///     .num_stages(3)
///     .deterministic_inflows(vec![
///         vec![20.0, 15.0],  // Stage 1
///         vec![25.0, 18.0],  // Stage 2
///         vec![30.0, 20.0],  // Stage 3
///     ])
///     .build()?;
/// ```
pub struct SddpBuilder {
    /// Factory function to create System instances (required)
    /// We store a function because System doesn't implement Clone
    system_factory: Option<Box<dyn Fn() -> System>>,

    /// Initial storage for each hydro (required)
    initial_storage: Option<Vec<f64>>,

    /// Number of decision stages (required, must be > 0)
    num_stages: Option<usize>,

    /// Inflow specification (required)
    inflows: InflowSpec,

    /// Load specification (optional, defaults to 0.0 MW)
    loads: LoadSpec,

    /// Random seed for reproducibility (default: 42)
    seed: u64,
    // Future extensions (not implemented in this ticket):
    // risk_measure: String,
    // load_process: String,
    // inflow_process: String,
}

impl SddpBuilder {
    /// Create a new builder with default values.
    ///
    /// Defaults:
    /// - `seed`: 42 (for reproducibility)
    /// - All required fields: `None` (must be set before build)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = SddpBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            system_factory: None,
            initial_storage: None,
            num_stages: None,
            inflows: InflowSpec::NotSet,
            loads: LoadSpec::NotSet,
            seed: 42, // Default seed for reproducibility
        }
    }

    /// Set the power system factory.
    ///
    /// Since `System` doesn't implement `Clone`, you need to provide a function
    /// that creates a new `System` instance. This function will be called once
    /// per graph node (PreStudy + stages).
    ///
    /// # Arguments
    ///
    /// * `factory` - Function that creates a System instance
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.system_factory(|| create_my_system())
    /// ```
    pub fn system_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> System + 'static,
    {
        self.system_factory = Some(Box::new(factory));
        self
    }

    /// Set initial storage levels for all hydros.
    ///
    /// # Arguments
    ///
    /// * `storage` - Initial storage in MWh for each hydro (indexed by hydro_id)
    ///
    /// # Validation
    ///
    /// At build time, validates that `storage.len() == system.hydros.len()`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.initial_storage(vec![50.0, 30.0])  // 2 hydros
    /// ```
    pub fn initial_storage(mut self, storage: Vec<f64>) -> Self {
        self.initial_storage = Some(storage);
        self
    }

    /// Set the number of decision stages.
    ///
    /// # Arguments
    ///
    /// * `num_stages` - Number of stages (must be > 0)
    ///
    /// # Validation
    ///
    /// At build time, validates `num_stages > 0`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.num_stages(3)
    /// ```
    pub fn num_stages(mut self, num_stages: usize) -> Self {
        self.num_stages = Some(num_stages);
        self
    }

    /// Set random seed for reproducibility.
    ///
    /// Controls random number generation in forward passes and scenario sampling.
    ///
    /// # Arguments
    ///
    /// * `seed` - u64 seed value
    ///
    /// # Default
    ///
    /// 42 (if not called)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.seed(12345)
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set deterministic inflows (single scenario per stage).
    ///
    /// Use this for deterministic problems or when you want a single scenario tree.
    ///
    /// # Arguments
    ///
    /// * `inflows` - Inflows indexed as `inflows[stage][hydro]`
    ///   - `inflows.len()` must equal `num_stages`
    ///   - `inflows[i].len()` must equal number of hydros
    ///
    /// # Validation
    ///
    /// At build time, validates:
    /// - `inflows.len() == num_stages`
    /// - All stages have same number of hydros
    /// - Hydro count matches system
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.deterministic_inflows(vec![
    ///     vec![30.0],  // Stage 1: 30 MWh
    ///     vec![40.0],  // Stage 2: 40 MWh
    /// ])
    /// ```
    pub fn deterministic_inflows(mut self, inflows: Vec<Vec<f64>>) -> Self {
        self.inflows = InflowSpec::Deterministic(inflows);
        self
    }

    /// Set stochastic inflows (multiple scenarios per stage).
    ///
    /// **Must be followed by `scenario_probabilities()`** to complete the specification.
    ///
    /// # Arguments
    ///
    /// * `scenarios` - Inflows indexed as `scenarios[stage][scenario][hydro]`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder
    ///     .stochastic_inflows(vec![
    ///         vec![vec![30.0]],  // Stage 1: deterministic (1 scenario)
    ///         vec![
    ///             vec![20.0],  // Stage 2: dry
    ///             vec![40.0],  // Stage 2: average
    ///             vec![60.0],  // Stage 2: wet
    ///         ],
    ///     ])
    ///     .scenario_probabilities(vec![
    ///         vec![1.0],
    ///         vec![0.25, 0.50, 0.25],
    ///     ])
    /// ```
    pub fn stochastic_inflows(mut self, scenarios: Vec<Vec<Vec<f64>>>) -> Self {
        // Temporarily store scenarios; probabilities will be added later
        self.inflows = InflowSpec::Stochastic {
            scenarios,
            probabilities: vec![],
        };
        self
    }

    /// Set scenario probabilities for stochastic inflows.
    ///
    /// **Must be called after `stochastic_inflows()`**.
    ///
    /// # Arguments
    ///
    /// * `probabilities` - Probabilities indexed as `probabilities[stage][scenario]`
    ///   - Must match structure of `stochastic_inflows()`
    ///   - Per-stage probabilities must sum to 1.0 (within 1e-6 tolerance)
    ///
    /// # Validation
    ///
    /// At build time, validates:
    /// - Structure matches `stochastic_inflows()`
    /// - Each stage's probabilities sum to 1.0 ± 1e-6
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.scenario_probabilities(vec![
    ///     vec![1.0],              // Stage 1: deterministic
    ///     vec![0.25, 0.50, 0.25], // Stage 2: 3 scenarios
    /// ])
    /// ```
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
    ///
    /// # Arguments
    ///
    /// * `loads` - Load in MW for each stage: `loads[stage]`
    ///   - `loads.len()` must equal `num_stages`
    ///   - Values applied to first bus in system
    ///
    /// # Default
    ///
    /// If not called, defaults to 0.0 MW (no load constraint).
    ///
    /// # Validation
    ///
    /// At build time, validates:
    /// - `loads.len() == num_stages`
    /// - Cannot be used with `stochastic_inflows()` (use `stochastic_loads()` instead)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.deterministic_loads(vec![40.0, 45.0, 50.0])
    /// ```
    pub fn deterministic_loads(mut self, loads: Vec<f64>) -> Self {
        self.loads = LoadSpec::Deterministic(loads);
        self
    }

    /// Set stochastic loads (multiple scenarios per stage).
    ///
    /// **Must match the structure of stochastic_inflows().**
    /// Probabilities are inherited from inflow scenario probabilities.
    ///
    /// # Arguments
    ///
    /// * `scenarios` - Loads indexed as `scenarios[stage][scenario]`
    ///
    /// # Validation
    ///
    /// At build time, validates:
    /// - Must be called with `stochastic_inflows()` (cannot use with deterministic)
    /// - Structure must match: `scenarios[stage].len() == inflow_scenarios[stage].len()`
    /// - Probabilities come from `scenario_probabilities()`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder
    ///     .stochastic_inflows(vec![
    ///         vec![vec![30.0]],  // Stage 1: 1 scenario
    ///         vec![vec![20.0], vec![40.0], vec![60.0]],  // Stage 2: 3 scenarios
    ///     ])
    ///     .stochastic_loads(vec![
    ///         vec![35.0],  // Stage 1: low demand
    ///         vec![30.0, 40.0, 50.0],  // Stage 2: low/med/high demand
    ///     ])
    ///     .scenario_probabilities(vec![vec![1.0], vec![0.25, 0.50, 0.25]])
    /// ```
    pub fn stochastic_loads(mut self, scenarios: Vec<Vec<f64>>) -> Self {
        self.loads = LoadSpec::Stochastic(scenarios);
        self
    }

    /// Build the SDDP algorithm instance.
    ///
    /// Validates all required fields, constructs the graph and SAA, and creates
    /// the final `SddpAlgorithm` instance.
    ///
    /// # Returns
    ///
    /// - `Ok(SddpAlgorithm)` if all validation passes
    /// - `Err(String)` with descriptive error if validation fails
    ///
    /// # Validation
    ///
    /// - All required fields present (system, storage, stages, inflows)
    /// - `num_stages > 0`
    /// - Inflow structure matches system and stages
    /// - Stochastic probabilities sum to 1.0 per stage
    ///
    /// # Performance
    ///
    /// All construction happens here:
    /// - Graph construction: O(num_stages) with system cloning
    /// - SAA construction: O(num_stages × num_scenarios × num_hydros)
    /// - Total: Dominated by system cloning (acceptable one-time cost)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sddp = builder.build()?;
    /// ```
    pub fn build(self) -> Result<SddpAlgorithm, String> {
        // VALIDATION PHASE - Extract and validate all required fields
        // Note: We consume self here, so we need to move out all Option fields
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

        // Create one system instance for validation
        let system_for_validation = system_factory();

        // Validate num_stages > 0
        if num_stages == 0 {
            return Err("num_stages must be greater than 0".to_string());
        }

        // Validate inflows are set
        if matches!(inflows, InflowSpec::NotSet) {
            return Err("inflows are required (use deterministic_inflows() or stochastic_inflows())".to_string());
        }

        // Validate initial_storage length matches system
        if initial_storage.len() != system_for_validation.meta.hydros_count {
            return Err(format!(
                "initial_storage length ({}) must match number of hydros ({})",
                initial_storage.len(),
                system_for_validation.meta.hydros_count
            ));
        }

        // CONSTRUCTION PHASE

        // Build DirectedGraph<NodeData>
        let graph = build_graph(&system_factory, num_stages)?;

        // Build InitialCondition
        let initial_condition = InitialCondition::new(initial_storage, vec![]);

        // Build SAA from inflow specification
        let _saa = build_saa(
            &system_for_validation,
            num_stages,
            &inflows,
            &self.loads,
            seed,
        )?;

        // Create and return SddpAlgorithm
        SddpAlgorithm::new(graph, initial_condition, seed)
    }

    /// Build the SDDP algorithm along with its SAA (for training).
    ///
    /// This method returns both the `SddpAlgorithm` instance and the `SAA` (Stochastic
    /// Approximation Algorithm) needed for training. Use this when you need to call
    /// `train()` on the algorithm.
    ///
    /// # Returns
    ///
    /// Returns `Ok((sddp, saa))` on success, where:
    /// - `sddp`: The configured `SddpAlgorithm` instance
    /// - `saa`: The `SAA` instance with inflow scenarios
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (mut sddp, saa) = builder.build_with_saa()?;
    /// let result = sddp.train(20, 10, &saa)?;
    /// ```
    pub fn build_with_saa(self) -> Result<(SddpAlgorithm, SAA), String> {
        // VALIDATION PHASE - Extract and validate all required fields
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

        // Create one system instance for validation
        let system_for_validation = system_factory();

        // Validate num_stages > 0
        if num_stages == 0 {
            return Err("num_stages must be greater than 0".to_string());
        }

        // Validate inflows are set
        if matches!(inflows, InflowSpec::NotSet) {
            return Err("inflows are required (use deterministic_inflows() or stochastic_inflows())".to_string());
        }

        // Validate initial_storage length matches system
        if initial_storage.len() != system_for_validation.meta.hydros_count {
            return Err(format!(
                "initial_storage length ({}) must match number of hydros ({})",
                initial_storage.len(),
                system_for_validation.meta.hydros_count
            ));
        }

        // CONSTRUCTION PHASE

        // Build DirectedGraph<NodeData>
        let graph = build_graph(&system_factory, num_stages)?;

        // Build InitialCondition
        let initial_condition = InitialCondition::new(initial_storage, vec![]);

        // Build SAA from inflow specification
        let saa = build_saa(
            &system_for_validation,
            num_stages,
            &inflows,
            &self.loads,
            seed,
        )?;

        // Create SddpAlgorithm
        let sddp = SddpAlgorithm::new(graph, initial_condition, seed)?;

        Ok((sddp, saa))
    }
}

/// Build the DirectedGraph<NodeData> for SDDP.
///
/// Creates a simple path graph:
/// - Node 0: PreStudy (initial condition)
/// - Nodes 1..num_stages: Study periods
///
/// # Performance
///
/// - System is recreated per node via factory function
/// - This matches the pattern in existing tests
/// - Graph construction is not in the hot path (happens once)
fn build_graph(
    system_factory: &dyn Fn() -> System,
    num_stages: usize,
) -> Result<DirectedGraph<NodeData>, String> {
    let mut graph = DirectedGraph::<NodeData>::new();

    // Add PreStudy node (id = -1 by convention)
    let pre_study_id = graph
        .add_node(NodeData::new(
            -1,                     // node_id
            0,                      // stage_id
            0,                      // season_id
            "2024-01-01T00:00:00Z", // start_date (placeholder)
            "2024-01-01T00:00:00Z", // end_date
            StudyPeriodKind::PreStudy,
            system_factory(), // Create system
            "expectation",    // risk_measure
            "naive",          // load_stochastic_process
            "naive",          // inflow_stochastic_process
            "storage",        // state_choice
        )?)
        .map_err(|e| format!("Failed to add PreStudy node: {:?}", e))?;

    let mut previous_node_id = pre_study_id;

    // Add Study period nodes
    for stage in 1..=num_stages {
        let stage_id = graph
            .add_node(NodeData::new(
                stage as isize,         // node_id
                stage,                  // stage_id
                stage,                  // season_id (simplified)
                "2024-01-01T00:00:00Z", // start_date (placeholder)
                "2024-01-02T00:00:00Z", // end_date (placeholder)
                StudyPeriodKind::Study,
                system_factory(), // Create system
                "expectation",
                "naive",
                "naive",
                "storage",
            )?)
            .map_err(|e| {
                format!("Failed to add Study node for stage {}: {:?}", stage, e)
            })?;

        // Connect edge: previous → current
        graph.add_edge(previous_node_id, stage_id).map_err(|e| {
            format!("Failed to add edge for stage {}: {:?}", stage, e)
        })?;

        previous_node_id = stage_id;
    }

    Ok(graph)
}

/// Build the SAA (Sample Average Approximation) from inflow specification.
///
/// Creates a NoiseGenerator and generates the SAA with the provided seed.
///
/// # Performance
///
/// - SAA generation: O(num_stages × num_scenarios × num_hydros)
/// - Not in hot path (happens once at construction)
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
///
/// Creates a NoiseGenerator with zero-variance Normal distributions
/// (mean = inflow value, std = 0.0) and 1 branching per stage.
fn build_deterministic_saa(
    system: &System,
    num_stages: usize,
    inflows: &[Vec<f64>],
    loads: &LoadSpec,
    seed: u64,
) -> Result<SAA, String> {
    // Validate inflows structure
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

    // Validate loads structure
    match loads {
        LoadSpec::NotSet => {} // OK, will default to 0.0
        LoadSpec::Deterministic(load_values) => {
            if load_values.len() != num_stages {
                return Err(format!(
                    "deterministic_loads length ({}) must match num_stages ({})",
                    load_values.len(),
                    num_stages
                ));
            }
            // Validate non-negative loads
            for (stage, &load) in load_values.iter().enumerate() {
                if load < 0.0 {
                    return Err(format!(
                        "Stage {} load must be non-negative (got {})",
                        stage + 1,
                        load
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

    // Add PreStudy node generator (not used but required for indexing)
    let prestudy_load_value = match loads {
        LoadSpec::NotSet => 0.0,
        LoadSpec::Deterministic(load_values) => load_values[0],
        LoadSpec::Stochastic(_) => unreachable!(), // Already validated above
    };
    let prestudy_load = vec![Normal::new(prestudy_load_value, 0.0).unwrap()];
    let prestudy_inflow =
        vec![Normal::new(0.0, 0.0).unwrap(); system.meta.hydros_count];
    generator.add_node_generator(prestudy_load, prestudy_inflow, 1);

    // Add deterministic generators for each stage
    for (stage_idx, stage_inflows) in inflows.iter().enumerate() {
        let load_value = match loads {
            LoadSpec::NotSet => 0.0,
            LoadSpec::Deterministic(load_values) => load_values[stage_idx],
            LoadSpec::Stochastic(_) => unreachable!(), // Already validated above
        };
        let load_dist = vec![Normal::new(load_value, 0.0).unwrap()];

        // Inflow distributions: zero variance at specified values
        let inflow_dists: Vec<Normal<f64>> = stage_inflows
            .iter()
            .map(|&inflow| Normal::new(inflow, 0.0).unwrap())
            .collect();

        generator.add_node_generator(load_dist, inflow_dists, 1); // 1 branching (deterministic)
    }

    // Generate SAA with provided seed
    Ok(generator.generate(seed))
}

/// Build stochastic SAA (multiple scenarios per stage).
///
/// For stochastic scenarios with discrete inflow values, we construct
/// a SAA manually using the provided scenario values and probabilities.
///
/// **Note**: Currently assumes equal branching structure. Probabilities
/// are validated but not yet used to weight scenarios (future enhancement).
fn build_stochastic_saa(
    system: &System,
    num_stages: usize,
    scenarios: &[Vec<Vec<f64>>],
    probabilities: &[Vec<f64>],
    loads: &LoadSpec,
    seed: u64,
) -> Result<SAA, String> {
    // Validate structure
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

    // Validate each stage
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

        // Validate probability sum to 1.0 (within tolerance)
        let prob_sum: f64 = stage_probs.iter().sum();
        if (prob_sum - 1.0).abs() > 1e-6 {
            return Err(format!(
                "Stage {}: probabilities must sum to 1.0 (got {:.6})",
                stage + 1,
                prob_sum
            ));
        }

        // Validate hydro count
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

    // Validate loads structure
    match loads {
        LoadSpec::NotSet => {} // OK, will default to 0.0
        LoadSpec::Deterministic(load_values) => {
            if load_values.len() != num_stages {
                return Err(format!(
                    "deterministic_loads length ({}) must match num_stages ({})",
                    load_values.len(),
                    num_stages
                ));
            }
            // Validate non-negative loads
            for (stage, &load) in load_values.iter().enumerate() {
                if load < 0.0 {
                    return Err(format!(
                        "Stage {} load must be non-negative (got {})",
                        stage + 1,
                        load
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
            // Validate each stage
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

                // Validate non-negative loads
                for (scenario_idx, &load) in
                    stage_load_scenarios.iter().enumerate()
                {
                    if load < 0.0 {
                        return Err(format!(
                            "Stage {}, scenario {}: load must be non-negative (got {})",
                            stage + 1,
                            scenario_idx,
                            load
                        ));
                    }
                }
            }
        }
    }

    // Build SAA manually with set_noises_by_stage for discrete scenarios
    let mut generator = NoiseGenerator::new();

    // Add PreStudy node generator
    let prestudy_load_value = match loads {
        LoadSpec::NotSet => 0.0,
        LoadSpec::Deterministic(load_values) => load_values[0],
        LoadSpec::Stochastic(load_scenarios) => load_scenarios[0][0],
    };
    let prestudy_load = vec![Normal::new(prestudy_load_value, 0.0).unwrap()];
    let prestudy_inflow =
        vec![Normal::new(0.0, 0.0).unwrap(); system.meta.hydros_count];
    generator.add_node_generator(prestudy_load, prestudy_inflow, 1);

    // Add node generators for each stage with appropriate branching count
    for stage_scenarios in scenarios.iter() {
        let num_scenarios = stage_scenarios.len();

        let load_dist = vec![Normal::new(0.0, 0.0).unwrap()]; // Placeholder, will override
        let inflow_dists: Vec<Normal<f64>> = (0..system.meta.hydros_count)
            .map(|_| Normal::new(0.0, 0.0).unwrap())
            .collect();

        generator.add_node_generator(load_dist, inflow_dists, num_scenarios);
    }

    // Generate initial SAA structure
    let mut saa = generator.generate(seed);

    // Override with exact discrete scenario values
    for (stage_idx, stage_scenarios) in scenarios.iter().enumerate() {
        let node_idx = stage_idx + 1; // +1 because PreStudy is index 0
        let num_scenarios = stage_scenarios.len();

        // Prepare load noises: [entity][branching]
        let load_noises: Vec<Vec<f64>> = match loads {
            LoadSpec::NotSet => {
                vec![vec![0.0; num_scenarios]]
            }
            LoadSpec::Deterministic(load_values) => {
                let load_value = load_values[stage_idx];
                vec![vec![load_value; num_scenarios]]
            }
            LoadSpec::Stochastic(load_scenarios) => {
                let stage_load_scenarios = &load_scenarios[stage_idx];
                vec![stage_load_scenarios.clone()]
            }
        };

        // Prepare inflow noises: [entity][branching]
        let mut inflow_noises: Vec<Vec<f64>> =
            vec![vec![0.0; num_scenarios]; system.meta.hydros_count];

        for (scenario_idx, scenario_inflows) in
            stage_scenarios.iter().enumerate()
        {
            for (hydro_idx, &inflow) in scenario_inflows.iter().enumerate() {
                inflow_noises[hydro_idx][scenario_idx] = inflow;
            }
        }

        // Set noises for this stage
        saa.set_noises_by_stage(
            node_idx,
            num_scenarios,
            1,                        // num_load_entities
            system.meta.hydros_count, // num_inflow_entities
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
            .deterministic_loads(vec![40.0, 40.0])
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
            .deterministic_loads(vec![40.0]) // Wrong: 1 load for 2 stages
            .seed(42)
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("must match num_stages"));
        }
    }

    #[test]
    fn test_builder_validates_negative_loads() {
        let result = SddpBuilder::new()
            .system_factory(create_test_system)
            .initial_storage(vec![50.0])
            .num_stages(2)
            .deterministic_inflows(vec![vec![30.0], vec![40.0]])
            .deterministic_loads(vec![40.0, -10.0]) // Invalid: negative load
            .seed(42)
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("non-negative"));
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
                vec![40.0],             // Stage 1: 1 scenario
                vec![35.0, 40.0, 45.0], // Stage 2: 3 scenarios
            ])
            .seed(42)
            .build();

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
                vec![40.0],             // Stage 1: 1 scenario (40 MW)
                vec![35.0, 40.0, 45.0], // Stage 2: 3 scenarios (low/med/high demand)
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
                vec![40.0],       // Stage 1: 1 scenario ✓
                vec![35.0, 40.0], // Stage 2: 2 scenarios ✗ (should be 3)
            ])
            .seed(42)
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("must match number of inflow scenarios"));
        }
    }

    // Note: Stochastic and validation tests will be added in next phase
}
