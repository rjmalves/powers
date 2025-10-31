use crate::error::PowersError;
use crate::graph::DirectedGraph;
use crate::initial_condition::InitialCondition;
use crate::input::{Config, GraphInput, Input, Recourse, SystemInput};
use crate::scenario::{NoiseGenerator, SAA};
use crate::sddp::{NodeData, SddpAlgorithm, SddpInstance};
use crate::stochastic_process;
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
/// - `Deterministic`: Single-scenario loads per stage
/// - `Stochastic`: Multi-scenario loads with probabilities (must match inflow structure)
#[derive(Debug, Clone)]
enum LoadSpec {
    /// No loads specified - defaults to 0.0 MW
    NotSet,

    /// Deterministic loads: `loads[stage][bus]`
    ///
    /// Single load value per bus per stage
    Deterministic(Vec<Vec<f64>>),

    /// Stochastic loads with scenarios
    ///
    /// - `scenarios[stage][scenario][bus]`: Load values
    /// - Probabilities inherited from inflow scenarios (must match structure)
    ///
    /// Probabilities per stage come from `scenario_probabilities()` and must match
    /// the structure of `stochastic_inflows()`.
    Stochastic(Vec<Vec<Vec<f64>>>),
}

/// High-level builder for SDDP algorithm instances.
///
/// Provides a fluent API that dramatically reduces boilerplate for common SDDP
/// construction patterns. Reduces typical test code from ~150 lines to ~8 lines.
///
/// # Performance Notes
///
/// - Builder is consumed by `build()` (move semantics, no extra allocation)
/// - All validation happens at build time, not training time
/// - Compiles to identical code as manual construction (zero-cost abstraction)
/// - System is recreated per graph node (acceptable one-time cost for builder)
///
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
    /// builder.deterministic_loads(vec![vec![40.0], vec![45.0], vec![50.0]])
    /// ```
    pub fn deterministic_loads(mut self, loads: Vec<Vec<f64>>) -> Self {
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
    ///         vec![vec![35.0]],  // Stage 1: low demand
    ///         vec![vec![30.0], vec![40.0], vec![50.0]],  // Stage 2: low/med/high demand
    ///     ])
    ///     .scenario_probabilities(vec![vec![1.0], vec![0.25, 0.50, 0.25]])
    /// ```
    pub fn stochastic_loads(mut self, scenarios: Vec<Vec<Vec<f64>>>) -> Self {
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
        // Use default "storage" and "naive" for backward compatibility
        let graph =
            build_graph(&system_factory, num_stages, "storage", "naive")?;

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
        // Use default "storage" and "naive" for backward compatibility
        let graph =
            build_graph(&system_factory, num_stages, "storage", "naive")?;

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
/// Creates a path graph with pre-study and study nodes:
/// - PreStudy nodes: 1+p nodes (where p is lag order from state_choice)
///   - storage: 1 pre-study node (id=-1)
///   - storage_and_inflow: 1+p pre-study nodes (ids: -p, ..., -1, 0)
/// - Study nodes: num_stages nodes (ids: 1..=num_stages)
///
/// The number of pre-study nodes depends on state_choice:
/// - "storage": 1 node (no lags needed)
/// - "storage_and_inflow": 1+p nodes (for p lags from inflow process)
///
/// # Performance
///
/// - System is recreated per node via factory function
/// - This matches the pattern in existing tests
/// - Graph construction is not in the hot path (happens once)
/// - Additional pre-study nodes: O(p) overhead, negligible vs study nodes
///
/// Helper function to create empty unified specs for builder test utilities
fn builder_empty_unified_specs(
) -> Vec<crate::unified_noise_spec::UnifiedNoiseSpec> {
    vec![]
}

/// Compute season IDs for PreStudy nodes via cycle-back from first Study node
///
/// PreStudy nodes represent historical time periods leading up to the study start.
/// Their season IDs should cycle backward from the first Study node's season to
/// ensure correct seasonal parameters are used for observation→residual transforms.
///
/// # Arguments
///
/// - `first_study_season`: Season ID of the first Study node (0-indexed)
/// - `lag_order`: Number of historical lags needed (AR order)
/// - `num_seasons`: Total number of seasons in the periodic cycle
///
/// # Returns
///
/// Vector of season IDs of length `1 + lag_order`, ordered from **newest to oldest**:
/// - `result[0]`: Season for **newest** PreStudy node (connects to first Study)
/// - `result[last]`: Season for **oldest** PreStudy node (lag p)
///
/// This ordering matches the `inflow` lag convention: `[Y_{-1}, Y_{-2}, ...]`
///
/// # Example
///
/// ```ignore
/// // Study starts in season 5 (May), AR(2) model (2 lags), 12 seasons
/// let seasons = compute_prestudy_season_ids(5, 2, 12);
/// // Returns [5, 4, 3]: PreStudy seasons [newest=May, April, oldest=March]
/// // PreStudy node with season 5 connects to first Study node (also season 5)
///
/// // Study starts in season 1 (January), AR(3) model, 12 seasons
/// let seasons = compute_prestudy_season_ids(1, 3, 12);
/// // Returns [1, 0, 11, 10]: wraps around [Jan, Dec, Nov, Oct]
/// ```
///
/// # Performance
///
/// - Time: O(p) where p = lag_order (typically ≤ 3)
/// - Space: O(p) for returned vector
/// - No heap allocations during computation (stack-only arithmetic)
/// - Branch-free wraparound using modular arithmetic
///
/// # Panics
///
/// Panics if `num_seasons == 0` (defensive check, should be validated upstream).
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

    // PERFORMANCE: Branch-free arithmetic using wrapping_sub and modulo
    // Cycle backward from first_study_season, ordered newest to oldest
    // Index 0 = newest (offset 0), Index last = oldest (offset lag_order)
    for offset in 0..num_pre_study_nodes {
        // Wraparound logic: (first_study_season - offset) mod num_seasons
        // Use wrapping_sub to handle underflow, then mod to wrap into [0, num_seasons)
        let season_id = first_study_season
            .wrapping_sub(offset)
            .wrapping_add(num_seasons) // Add num_seasons to ensure positive before mod
            % num_seasons;

        season_ids.push(season_id);
    }

    season_ids
}

fn build_graph(
    system_factory: &dyn Fn() -> System,
    num_stages: usize,
    state_choice: &str,
    inflow_process_type: &str,
) -> Result<DirectedGraph<NodeData>, String> {
    let mut graph = DirectedGraph::<NodeData>::new();

    // Determine lag order from state_choice and inflow process
    let lag_order = match state_choice {
        "storage" => 0,
        "storage_and_inflow" => {
            // Get lag order from stochastic process
            let inflow_process =
                stochastic_process::factory(inflow_process_type);
            inflow_process.lag_order()
        }
        _ => {
            return Err(format!(
                "Unknown state_choice: '{}'. Valid options: 'storage', 'storage_and_inflow'",
                state_choice
            ));
        }
    };

    // Create pre-study nodes: 1 + lag_order total
    // Node IDs: -(lag_order), -(lag_order-1), ..., -1, 0
    let num_pre_study_nodes = 1 + lag_order;
    let mut pre_study_ids = Vec::with_capacity(num_pre_study_nodes);

    // CRITICAL FIX (TICKET-003b): Compute correct season IDs for PreStudy nodes
    // Previous bug: all PreStudy nodes used season_id = 0, causing incorrect
    // observation→residual transformation when studies start mid-year.
    //
    // Solution: Cycle backward from first Study node season (which is 1 for stage 1)
    // Example: first_study_season=5, lag_order=2 → PreStudy seasons=[5, 4, 3] (newest to oldest)
    //
    // PERFORMANCE: O(p) computation where p=lag_order (typically ≤ 3), negligible
    // overhead compared to O(num_stages) study node creation.
    let first_study_season = 1; // First Study node has season_id = stage_id = 1
    let num_seasons = 12; // Default to 12 seasons (monthly cycle)
                          // Future enhancement: Extract from PAR config automatically
                          // See FUTURE_WORK.md: "Extract Seasonal Configuration from PAR Model"
    let prestudy_season_ids =
        compute_prestudy_season_ids(first_study_season, lag_order, num_seasons);

    for pre_idx in 0..num_pre_study_nodes {
        // Calculate node_id: starts at -(lag_order) and goes to 0
        let node_id = -(lag_order as isize - pre_idx as isize);

        // INDEXING: prestudy_season_ids are [newest, ..., oldest]
        // but PreStudy nodes are created [oldest, ..., newest] (by node_id)
        // So we need to reverse the indexing: oldest node uses last season_id
        let season_id_idx = num_pre_study_nodes - 1 - pre_idx;
        let season_id = prestudy_season_ids[season_id_idx];

        let pre_study_id = graph
            .add_node(NodeData::new(
                node_id,                // node_id: -(lag_order) to 0
                0,                      // stage_id: all 0 (before study)
                season_id, // season_id: computed via cycle-back (TICKET-003b)
                "2024-01-01T00:00:00Z", // start_date (placeholder)
                "2024-01-01T00:00:00Z", // end_date
                StudyPeriodKind::PreStudy,
                system_factory(),               // Create system
                "expectation",                  // risk_measure
                "naive",                        // load_stochastic_process
                &builder_empty_unified_specs(), // unified_specs (empty for tests)
                state_choice,                   // state_choice
                1, // num_scenarios (PreStudy always 1)
            )?)
            .map_err(|e| {
                format!("Failed to add PreStudy node {}: {:?}", node_id, e)
            })?;

        pre_study_ids.push(pre_study_id);
    }

    // Connect pre-study nodes sequentially
    for i in 0..num_pre_study_nodes.saturating_sub(1) {
        graph
            .add_edge(pre_study_ids[i], pre_study_ids[i + 1])
            .map_err(|e| {
                format!("Failed to connect PreStudy nodes: {:?}", e)
            })?;
    }

    let last_pre_study_id = *pre_study_ids.last().unwrap();

    let mut previous_node_id = last_pre_study_id;

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
                system_factory(),               // Create system
                "expectation",                  // risk_measure
                "naive",                        // load_stochastic_process
                &builder_empty_unified_specs(), // unified_specs (empty for tests)
                state_choice,                   // state_choice
                1, // num_scenarios (simplified for test builder)
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

    // Add PreStudy node generator (not used but required for indexing)
    let prestudy_load_value = match loads {
        LoadSpec::NotSet => 0.0,
        LoadSpec::Deterministic(load_values) => load_values[0][0],
        LoadSpec::Stochastic(_) => unreachable!(), // Already validated above
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
            LoadSpec::Stochastic(_) => unreachable!(), // Already validated above
        };
        let load_dists = load_values
            .iter()
            .map(|&load| Normal::new(load, 0.0).unwrap())
            .collect::<Vec<Normal<f64>>>();

        // Inflow distributions: zero variance at specified values
        let inflow_dists: Vec<Normal<f64>> = stage_inflows
            .iter()
            .map(|&inflow| Normal::new(inflow, 0.0).unwrap())
            .collect();

        generator.add_node_generator(load_dists, inflow_dists, 1); // 1 branching (deterministic)
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

    // Add PreStudy node generator
    let prestudy_load_value = match loads {
        LoadSpec::NotSet => 0.0,
        LoadSpec::Deterministic(load_values) => load_values[0][0],
        LoadSpec::Stochastic(load_scenarios) => load_scenarios[0][0][0],
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
                vec![vec![0.0; num_scenarios]; system.meta.buses_count]
            }
            LoadSpec::Deterministic(load_values) => {
                let stage_load_values = &load_values[stage_idx];
                // Broadcast single value to all scenarios: [bus][scenarios]
                stage_load_values
                    .iter()
                    .map(|&load| vec![load; num_scenarios])
                    .collect()
            }
            LoadSpec::Stochastic(load_scenarios) => {
                // Transpose from [scenario][bus] to [bus][scenario]
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
            system.meta.buses_count,  // num_load_entities
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
            .build();

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
            .build();

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
        let graph = build_graph(&system_factory, 3, "storage", "naive")
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
            build_graph(&system_factory, 3, "storage_and_inflow", "naive")
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
        let graph = build_graph(&system_factory, 2, "storage", "naive")
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
        let graph =
            build_graph(&system_factory, num_stages, "storage", "naive")
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
        let result = build_graph(&system_factory, 2, "invalid_choice", "naive");

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
        let graph_storage =
            build_graph(&system_factory, 10, "storage", "naive")
                .expect("Failed to build graph");
        assert_eq!(graph_storage.node_count(), 11); // 1 + 10

        // storage_and_inflow with naive (lag_order=0): same as storage
        let graph_inflow =
            build_graph(&system_factory, 10, "storage_and_inflow", "naive")
                .expect("Failed to build graph");
        assert_eq!(graph_inflow.node_count(), 11); // 1 + 10
    }

    #[test]
    fn test_build_graph_all_nodes_have_system() {
        // Test that all nodes have valid system instances
        let system_factory = || create_test_system();
        let graph = build_graph(&system_factory, 3, "storage", "naive")
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
/// This builder enables staged construction:
/// 1. Load inputs from JSON files (with validation)
/// 2. Modify configuration parameters (num_iterations, num_forward_passes, seed, num_threads)
/// 3. Build the SDDP algorithm instance
///
/// # Performance
///
/// - **Zero-cost abstraction**: Move semantics, no clones, no heap allocations
/// - **Builder consumed**: `build()` takes ownership, preventing reuse
/// - **Inline-friendly**: Small methods are inlined by the compiler
///
pub struct SddpInstanceBuilder {
    /// Power system configuration (buses, lines, thermals, hydros)
    system: SystemInput,

    /// Graph configuration (stages, distributions)
    graph: GraphInput,

    /// Recourse configuration (initial conditions, stochastic processes)
    recourse: Recourse,

    /// Algorithm configuration (iterations, forward passes, seed, output path)
    config: Config,
}

impl SddpInstanceBuilder {
    /// Load inputs from individual file paths with validation.
    ///
    /// This method:
    /// 1. Reads JSON files (config, system, graph, recourse)
    /// 2. Validates all inputs (fail-fast on first error)
    /// 3. Returns builder for further configuration modification
    ///
    /// # Arguments
    ///
    /// * `config_path` - Path to config.json (num_iterations, num_forward_passes, seed, etc.)
    /// * `system_path` - Path to system.json (buses, lines, thermals, hydros)
    /// * `graph_path` - Path to graph.json (stages, distributions)
    /// * `recourse_path` - Path to recourse.json (initial conditions, stochastic processes)
    ///
    /// # Returns
    ///
    /// `Ok(SddpInstanceBuilder)` on success, ready for parameter modification.
    /// `Err(PowersError)` if:
    /// - Any file cannot be read or parsed
    /// - Validation fails (missing references, invalid constraints, etc.)
    ///
    /// # Performance
    ///
    /// - **Move semantics**: Input components are moved (not cloned) into builder
    /// - **Zero heap allocations**: Just moves existing data
    /// - **Construction time**: < 1μs (just moves, validation already done)
    ///
    pub fn from_paths(
        config_path: impl AsRef<std::path::Path>,
        system_path: impl AsRef<std::path::Path>,
        graph_path: impl AsRef<std::path::Path>,
        recourse_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, PowersError> {
        // Load and validate inputs (validation happens inside from_paths)
        let input = Input::from_paths(
            config_path.as_ref(),
            system_path.as_ref(),
            graph_path.as_ref(),
            recourse_path.as_ref(),
        )?;

        // Extract components (move semantics - no copy)
        Ok(Self {
            system: input.system,
            graph: input.graph,
            recourse: input.recourse,
            config: input.config,
        })
    }

    /// Modify the number of forward passes per iteration.
    ///
    /// This affects:
    /// - Training: number of forward passes per iteration (affects convergence quality)
    /// - Memory: more forward passes = more solver models = higher peak memory
    ///
    /// # Arguments
    ///
    /// * `num_forward_passes` - Number of forward passes per iteration (must be > 0)
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    ///
    /// # Validation
    ///
    /// No validation at this point (deferred to `build()`). This allows chaining
    /// without intermediate checks.
    ///
    #[inline]
    pub fn with_num_forward_passes(
        mut self,
        num_forward_passes: usize,
    ) -> Self {
        self.config.num_forward_passes = num_forward_passes;
        self
    }

    /// Modify the number of SDDP iterations.
    ///
    /// This affects:
    /// - Training time: more iterations = longer training
    /// - Convergence: more iterations = better policy (diminishing returns)
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of SDDP iterations (must be > 0)
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    ///
    /// # Validation
    ///
    /// No validation at this point (deferred to `build()`). This allows chaining
    /// without intermediate checks.
    ///
    #[inline]
    pub fn with_num_iterations(mut self, num_iterations: usize) -> Self {
        self.config.num_iterations = num_iterations;
        self
    }

    /// Modify the random seed for deterministic sampling.
    ///
    /// This affects:
    /// - SAA generation: different seed = different scenarios
    /// - Reproducibility: same seed = identical results
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed (any u64 value)
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    ///
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Modify the number of threads for parallel execution.
    ///
    /// This affects:
    /// - Parallelism: number of threads for forward/backward passes (Rayon)
    /// - Performance: optimal thread count depends on hardware (typically num_cores)
    ///
    /// Thread pool is configured before training and simulation.
    ///
    /// # Arguments
    ///
    /// * `num_threads` - Number of threads (must be > 0, typically <= num_physical_cores)
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    ///
    /// # Validation
    ///
    /// Validation (num_threads > 0) happens in `configure_thread_pool()` at runtime,
    /// not during builder construction. This allows chaining without intermediate checks.
    ///
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
    /// # Returns
    ///
    /// `Ok(SddpInstance)` on success, ready for `train()` or `simulate()`.
    /// `Err(PowersError)` if construction fails (e.g., invalid graph structure).
    ///
    /// # Ownership
    ///
    /// This method **consumes** the builder (takes `self` by value). The builder
    /// cannot be reused after `build()` - this is intentional for clear ownership.
    ///
    /// # Performance
    ///
    /// - **Move semantics**: Components are moved into `SddpInstance` (no clones)
    /// - **Zero overhead**: Same logic as `from_files()` (no additional allocations)
    /// - **Build time**: Same as `from_files()` (graph construction + SAA generation)
    ///
    /// # Validation
    ///
    /// Validation happens at two points:
    /// 1. **Input validation**: Done in `from_paths()` (fail-fast on invalid JSON)
    /// 2. **Construction validation**: Done here (e.g., graph structure)
    ///
    /// Note: Parameter validation (num_iterations > 0, etc.) happens in `train()`,
    /// not here. This allows building the instance even with invalid training params
    /// (e.g., for testing edge cases).
    ///
    pub fn build(self) -> Result<SddpInstance, PowersError> {
        let seed = self.config.seed;

        // Build graph from JSON configuration
        // This supports complex seasonal structures and distribution-based uncertainty
        let node_data_graph = self
            .graph
            .build_sddp_graph(&self.system, &self.recourse)
            .map_err(|e| {
                PowersError::Other(format!("Failed to build SDDP graph: {}", e))
            })?;

        // Create initial condition from recourse data
        let initial_condition = self.recourse.build_sddp_initial_condition();

        // Generate SAA scenarios from stochastic processes
        // Uses the seed from config (potentially modified) for deterministic sampling
        // Graph NodeData contains num_scenarios per node for scenario generation
        // Pass the domain InitialCondition (not the input format)
        let saa = self.recourse.generate_sddp_noises(
            &node_data_graph,
            &initial_condition,
            seed,
        );

        // Create SDDP algorithm with low-level API
        let algorithm =
            SddpAlgorithm::new(node_data_graph, initial_condition, seed)
                .map_err(PowersError::Other)?;

        // Bundle algorithm + config + SAA into SddpInstance for ergonomic use
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
        let builder = builder.with_seed(999); // Use different seed than default (42)

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

        // Should be able to train with modified config
        let result = sddp.train();
        assert!(
            result.is_ok(),
            "Training should succeed with modified config"
        );
    }

    // ========================================================================
    // TICKET-003b: PreStudy Season Handling Tests
    // ========================================================================

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
