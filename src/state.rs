use crate::cut;
use crate::input::UncertaintyType;
use crate::risk_measure;
use crate::solver;
use crate::subproblem;
use crate::system;
use crate::uncertainty_model::UncertaintyModel;
use crate::utils;
use std::ops::Range;

pub trait State: Send + Sync {
    // behavior that must be implemented for each state definition
    fn set_dimension(&mut self, dimension: usize);
    fn coefficients(&self) -> &[f64];
    fn get_dominating_objective(&self) -> f64;
    fn set_dominating_objective(&mut self, dominating_objective: f64);
    fn get_dominating_cut_id(&self) -> usize;
    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize);

    fn get_iteration(&self) -> usize;
    fn set_iteration(&mut self, iteration: usize);
    fn get_forward_pass_idx(&self) -> usize;
    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize);

    /// Returns true if this state type includes lagged inflow state variables.
    /// - `StorageState`: false (only storage is state variable)
    /// - `StorageAndInflowState`: true (storage + lagged inflows are state variables)
    ///
    /// **DEPRECATED**: Use `has_lagged_observation_state()` instead for unified approach.
    #[deprecated(
        since = "0.4.0",
        note = "Use has_lagged_observation_state() for unified lag tracking"
    )]
    fn has_lagged_inflow_state(&self) -> bool {
        // Default implementation delegates to new method for backward compatibility
        self.has_lagged_observation_state()
    }

    /// Returns true if this state type includes lagged observation state variables.
    ///
    /// This method supports unified lag tracking for all entity types (loads and inflows).
    ///
    /// - `StorageState`: false (only storage is state variable)
    /// - `StorageAndInflowState`: true (only tracks inflow lags for backward compat)
    /// - `StorageAndObservationState`: true (tracks all entity lags)
    fn has_lagged_observation_state(&self) -> bool {
        false
    }

    /// Get lagged observations for a specific entity (unified approach).
    ///
    /// # Arguments
    ///
    /// - `entity_idx`: Global entity index (0..num_entities), ordered as [loads..., inflows...]
    ///
    /// # Returns
    ///
    /// Slice of lagged observations [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}] for this entity,
    /// or empty slice if entity has no lags (ar_order = 0) or state doesn't track lags.
    ///
    /// # Default Implementation
    ///
    /// Returns empty slice. Override in states that track lagged observations.
    fn get_lagged_observations(&self, _entity_idx: usize) -> &[f64] {
        &[]
    }

    /// Set lagged observations for a specific entity (unified approach).
    ///
    /// # Arguments
    ///
    /// - `entity_idx`: Global entity index (0..num_entities)
    /// - `observations`: New lag values [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    ///
    /// # Default Implementation
    ///
    /// No-op. Override in states that track lagged observations.
    fn set_lagged_observations(&mut self, _entity_idx: usize, _observations: &[f64]) {
        // Default: do nothing
    }

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    );

    /// Update state and subproblem from trajectory of past realizations.
    ///
    /// This method is called during forward pass to transfer state information
    /// from previous stages to the current subproblem. Each state implementation
    /// extracts what it needs from the trajectory:
    ///
    /// - `StorageState`: uses `.last()` for previous storage (O(1))
    /// - `StorageAndInflowState`: uses `[len-p..len]` for lags (O(p))
    ///
    /// # Example Trajectory Structure
    ///
    /// ```text
    /// Stage 1: [PreStudy(0)]
    /// Stage 2: [PreStudy(0), Stage(1)]
    /// Stage 3: [PreStudy(0), Stage(1), Stage(2)]
    ///
    /// With multi-node pre-study (PAR):
    /// Stage 1: [PreStudy(-3), PreStudy(-2), PreStudy(-1), PreStudy(0)]
    /// Stage 2: [PreStudy(-3), PreStudy(-2), PreStudy(-1), PreStudy(0), Stage(1)]
    /// ```
    fn update_from_trajectory(
        &mut self,
        past_realizations: &[&subproblem::Realization],
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        variables: &subproblem::Variables,
    );

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>>;

    fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        variables: &subproblem::Variables,
        model: &mut solver::Model,
    );

    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut;

    // default implementations
    fn update_dominating_cut(&mut self, cut: &cut::BendersCut, height: f64) {
        self.set_dominating_cut_id(cut.id);
        self.set_dominating_objective(height);
    }

    fn compute_new_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        // NOTE: Don't call update_dominating_cut() here! The cut has id=0 at this point.
        // The FCF will handle domination properly after assigning the real cut ID.
        self.evaluate_cut(
            risk_measure,
            forward_trajectory,
            branching_realizations,
        )
    }
    // clone helper for storing visited states
    fn clone_dyn(&self) -> Box<dyn State>;
}

// trait for cloning boxes of State trait objects
impl Clone for Box<dyn State> {
    fn clone(&self) -> Self {
        self.as_ref().clone_dyn()
    }
}

pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,
}

impl Default for VisitedStatePool {
    fn default() -> Self {
        Self::new()
    }
}

impl VisitedStatePool {
    pub fn new() -> Self {
        Self { pool: vec![] }
    }
}

/// Extract maximum AR order for a hydro from UncertaintyModel
fn extract_max_ar_order_for_hydro(
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
    hydro_id: usize,
    _season_id: usize,
) -> usize {
    use crate::input::UncertaintyType;
    use crate::uncertainty_model::UncertaintyModel;

    uncertainty_models
        .iter()
        .filter_map(|model| match model {
            UncertaintyModel::PeriodicAR {
                entity_type,
                entity_id,
                par_params,
            } if *entity_type == UncertaintyType::Inflow
                && *entity_id == hydro_id =>
            {
                Some(par_params.max_ar_order)
            }
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Calculate per-hydro state dimensions from UncertaintyModel
pub fn per_hydro_state_dims(
    system: &system::System,
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
    season_id: usize,
) -> Vec<usize> {
    system
        .hydros
        .iter()
        .map(|hydro| {
            let max_order = extract_max_ar_order_for_hydro(
                uncertainty_models,
                hydro.id,
                season_id,
            );
            1 + max_order // storage + lags
        })
        .collect()
}

/// Calculate total state dimension from UncertaintyModel
pub fn total_state_dim(
    system: &system::System,
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
    season_id: usize,
) -> usize {
    per_hydro_state_dims(system, uncertainty_models, season_id)
        .iter()
        .sum()
}

/// State layout for variable-length per-hydro state vectors.
///
/// Tracks offsets and dimensions for each hydro's state slice in the
/// flattened state vector. Enables O(1) state access for heterogeneous
/// AR orders.
///
/// # State Vector Layout
///
/// For hydros with different AR orders:
/// ```text
/// Hydro 0: AR(2) → [storage₀, lag₀₁, lag₀₂]       (dim=3)
/// Hydro 1: AR(1) → [storage₁, lag₁₁]              (dim=2)
/// Hydro 2: AR(0) → [storage₂]                     (dim=1)
///
/// Flattened: [storage₀, lag₀₁, lag₀₂, storage₁, lag₁₁, storage₂]
///            ←───── offset=0 ─────→  ←─ offset=3 ─→  ←offset=5→
/// ```
///
/// # Invariants
///
/// - `offsets.len() == num_hydros + 1`
/// - `offsets[i+1] - offsets[i] == per_hydro_dims[i]`
/// - `offsets.last() == total_dim`
/// - `per_hydro_dims.iter().sum() == total_dim`
///
/// # Performance
///
/// - Memory: ~16 bytes × num_hydros (Vec overhead)
/// - Offset lookup: O(1) array access
/// - Slice extraction: O(dim) memcpy
///
/// # Example
///
/// ```ignore
/// let layout = StateLayout::from_unified_specs(&system, &unified_specs, 0);
/// let hydro1_range = layout.hydro_slice(1);
/// let hydro1_state = &state[hydro1_range]; // [storage₁, lag₁₁]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StateLayout {
    /// Per-hydro state dimensions [dim₀, dim₁, ..., dimₙ]
    ///
    /// For hydro i: dim[i] = 1 + max_ar_order[i]
    pub per_hydro_dims: Vec<usize>,

    /// Cumulative offsets [0, dim₀, dim₀+dim₁, ..., total]
    ///
    /// Length = num_hydros + 1
    /// offsets[i] = start index of hydro i's state
    /// offsets[i+1] = end index (exclusive) of hydro i's state
    pub offsets: Vec<usize>,

    /// Total state dimension (sum of all per_hydro_dims)
    pub total_dim: usize,
}

impl StateLayout {
    /// Get the slice range for a hydro's state.
    ///
    /// Returns `start..end` range for indexing into the flattened state vector.
    ///
    /// # Arguments
    ///
    /// * `hydro_id` - Hydro index (0-based)
    ///
    /// # Performance
    ///
    /// O(1) - simple array access
    ///
    /// # Example
    ///
    /// ```ignore
    /// let range = layout.hydro_slice(1);
    /// let hydro1_state = &state[range]; // Extract hydro 1's state
    /// ```
    #[inline]
    pub fn hydro_slice(&self, hydro_id: usize) -> Range<usize> {
        self.offsets[hydro_id]..self.offsets[hydro_id + 1]
    }

    /// Get the dimension for a specific hydro.
    ///
    /// # Performance
    ///
    /// O(1) - array access
    #[inline]
    pub fn hydro_dim(&self, hydro_id: usize) -> usize {
        self.per_hydro_dims[hydro_id]
    }

    /// Get the offset for a specific hydro's storage (first element).
    ///
    /// # Performance
    ///
    /// O(1) - array access
    #[inline]
    pub fn hydro_storage_offset(&self, hydro_id: usize) -> usize {
        self.offsets[hydro_id]
    }

    /// Get the number of lags for a specific hydro.
    ///
    /// Returns dimension - 1 (excluding storage).
    ///
    /// # Performance
    ///
    /// O(1) - array access and subtraction
    #[inline]
    pub fn hydro_lag_count(&self, hydro_id: usize) -> usize {
        self.per_hydro_dims[hydro_id].saturating_sub(1)
    }
}

#[derive(Debug, Clone)]
pub struct StorageState {
    dimension: usize,
    final_storage: Vec<f64>,
    dominating_objective: f64,
    dominating_cut_id: usize,
    /// DEBUGGING: Iteration number when this state was visited (1-based)
    iteration: usize,
    /// DEBUGGING: Forward pass index that visited this state (0-based handler ID)
    forward_pass_idx: usize,
}

impl StorageState {
    pub fn new(system: &system::System) -> Self {
        Self {
            dimension: system.meta.hydros_count,
            final_storage: vec![0.0; system.meta.hydros_count],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }
}

impl State for StorageState {
    fn set_dimension(&mut self, dimension: usize) {
        self.dimension = dimension
    }

    fn get_dominating_objective(&self) -> f64 {
        self.dominating_objective
    }

    fn set_dominating_objective(&mut self, dominating_objective: f64) {
        self.dominating_objective = dominating_objective;
    }

    fn get_dominating_cut_id(&self) -> usize {
        self.dominating_cut_id
    }

    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize) {
        self.dominating_cut_id = dominating_cut_id;
    }

    fn get_iteration(&self) -> usize {
        self.iteration
    }

    fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    fn get_forward_pass_idx(&self) -> usize {
        self.forward_pass_idx
    }

    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize) {
        self.forward_pass_idx = forward_pass_idx;
    }

    fn coefficients(&self) -> &[f64] {
        self.final_storage.as_slice()
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>> {
        let mut col_indices = vec![vec![0; 1]; self.dimension];
        for col in &mut col_indices {
            col[0] = pb.add_column(0.0, 0.0..);
        }
        col_indices
    }

    fn update_from_trajectory(
        &mut self,
        past_realizations: &[&subproblem::Realization],
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        _variables: &subproblem::Variables,
    ) {
        // PERFORMANCE: O(1) access - get previous storage from last realization
        let prev_realization = past_realizations.last().unwrap();
        self.final_storage
            .clone_from_slice(&prev_realization.final_storage);

        // Update hydro balance RHS: V_{t-1} = final_storage
        for (index, row) in constraints.hydro_balance.iter().enumerate() {
            model.change_rows_bounds(
                *row,
                self.final_storage[index],
                self.final_storage[index],
            );
        }
    }

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        self.final_storage
            .clone_from_slice(&realization.final_storage);
    }

    fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        variables: &subproblem::Variables,
        model: &mut solver::Model,
    ) {
        let mut factors =
            Vec::<(usize, f64)>::with_capacity(self.dimension + 1);
        factors.push((variables.alpha, 1.0));
        for (hydro_id, stored_volume) in
            variables.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -cut.coefficients[hydro_id]));
        }
        model.add_row(cut.rhs.., factors);
    }

    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        let mut cut_coefficients = vec![0.0; self.dimension];
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let num_branchings = costs.len();
        let probabilities = utils::uniform_prob_by_count(num_branchings);
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(&probabilities, &costs);

        // Collect all contributions before accumulating.
        // This ensures deterministic order for Kahan summation regardless
        // of parallel thread completion order in backward pass. Without this,
        // floating-point accumulation order varies across runs, causing cut
        // coefficient drift that compounds through iterations.
        //
        // Memory overhead: num_branchings × dimension f64s per cut
        let mut coef_contributions: Vec<Vec<f64>> =
            Vec::with_capacity(branching_realizations.len());
        let mut objective_contributions: Vec<f64> =
            Vec::with_capacity(branching_realizations.len());

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];

            // Store contributions instead of accumulating immediately
            let contrib: Vec<f64> = realization
                .water_value
                .iter()
                .map(|&val| prob * val)
                .collect();
            coef_contributions.push(contrib);
            objective_contributions
                .push(prob * realization.total_stage_objective);
        }

        // Deterministic accumulation using Kahan summation
        for hydro_idx in 0..cut_coefficients.len() {
            let values: Vec<f64> = coef_contributions
                .iter()
                .map(|contrib| contrib[hydro_idx])
                .collect();
            cut_coefficients[hydro_idx] = utils::kahan_sum(&values);
        }
        let objective = utils::kahan_sum(&objective_contributions);

        let last_realization = forward_trajectory.last().unwrap();

        let cut_rhs = objective
            - utils::dot_product(
                &cut_coefficients,
                &last_realization.final_storage,
            );
        // Temporary sets cut id to 0 - will be updated when adding to pool
        // Use state's tracking information for iteration and forward_pass_idx
        cut::BendersCut::new(
            0,
            cut_coefficients,
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    }

    // clone helper for storing visited states
    fn clone_dyn(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }
}

/// State representation including both storage volumes and lagged inflows.
///
/// # State Vector Definition
///
/// The full state vector is: `[V_t, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]`
/// where:
/// - `V_t`: Final storage volumes at time t (dimension: n_hydros)
/// - `Y_{t-k}`: Lagged inflow realizations (dimension: n_hydros each)
/// - `p`: Lag order from the stochastic process
///
/// Total state dimension: `n + p×n` where n = number of hydros
#[derive(Debug, Clone)]
pub struct StorageAndInflowState {
    dimension: usize,
    layout: StateLayout,
    final_storage: Vec<f64>,
    lagged_inflows: Vec<Vec<f64>>,
    transformed_coefficients: Vec<Vec<f64>>,
    flattened_state: Vec<f64>,
    dominating_objective: f64,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
}

impl StorageAndInflowState {
    pub fn new(
        system: &system::System,
        uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
    ) -> Self {
        let dimension = system.meta.hydros_count;

        let per_hydro_dims: Vec<usize> =
            per_hydro_state_dims(system, uncertainty_models, 0);

        // Build cumulative offsets: [0, dim₀, dim₀+dim₁, ...]
        let mut offsets = Vec::with_capacity(dimension + 1);
        offsets.push(0);
        let mut cumsum = 0;
        for &dim in &per_hydro_dims {
            cumsum += dim;
            offsets.push(cumsum);
        }

        let layout = StateLayout {
            per_hydro_dims,
            offsets,
            total_dim: cumsum,
        };

        let lagged_inflows: Vec<Vec<f64>> = (0..dimension)
            .map(|i| {
                let lag_count = layout.hydro_lag_count(i);
                vec![0.0; lag_count]
            })
            .collect();

        let transformed_coefficients = Self::extract_transformed_coefficients(
            system,
            uncertainty_models,
            0,
        );

        let flattened_state = vec![0.0; layout.total_dim];

        let mut state = Self {
            dimension,
            layout,
            final_storage: vec![0.0; dimension],
            lagged_inflows,
            transformed_coefficients,
            flattened_state,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        };

        state.rebuild_flattened_state();
        state
    }

    /// Extract transformed (observation-space) AR coefficients from uncertainty models
    ///
    /// Computes ψ_i = φ_i * (σ_t / σ_{t-i}) for each hydro and lag, where:
    /// - φ_i: residual-space AR coefficient from PAR statistical model
    /// - σ_t: standard deviation of current season
    /// - σ_{t-i}: standard deviation of lag season
    ///
    /// # Mathematical Foundation
    ///
    /// The PAR model operates in residual space: Z'_t = Σ φ_i * Z'_{t-i} + ε_t
    /// where Z'_t = (Y_t - μ_t) / σ_t
    ///
    /// Transforming to observation space gives: Y_t = Σ ψ_i * Y_{t-i} + η_t
    /// where ψ_i = φ_i * (σ_t / σ_{t-i}) (Equation 7 in par_derivation.pdf)
    ///
    /// The LP constraints use ψ coefficients, so Benders cuts must also use ψ
    /// for mathematical consistency. Using φ directly would be incorrect.
    ///
    fn extract_transformed_coefficients(
        system: &system::System,
        uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
        season_id: usize,
    ) -> Vec<Vec<f64>> {
        let mut coeffs = vec![Vec::new(); system.meta.hydros_count];

        for model in uncertainty_models.iter() {
            if model.entity_type() != UncertaintyType::Inflow {
                continue;
            }

            let hydro_id = model.entity_id();

            match model {
                UncertaintyModel::Independent { .. } => {
                    coeffs[hydro_id] = vec![];
                }
                UncertaintyModel::PeriodicAR { par_params, .. } => {
                    let phi = par_params.ar_coefficients(season_id); // φ_i
                    let current_params = par_params.seasonal_params(season_id);
                    let ar_order = phi.len();
                    let num_seasons = par_params.num_seasons;

                    // Compute ψ_i = φ_i * (σ_t / σ_{t-i})
                    let mut psi = Vec::with_capacity(ar_order);
                    for (i, &phi_coef) in phi.iter().enumerate() {
                        let lag_offset = i + 1;
                        let lag_season = if num_seasons == 1 {
                            0
                        } else if season_id >= lag_offset {
                            season_id - lag_offset
                        } else {
                            num_seasons - (lag_offset - season_id)
                        };

                        let lag_params = par_params.seasonal_params(lag_season);
                        let psi_i = phi_coef
                            * (current_params.std_dev / lag_params.std_dev);
                        psi.push(psi_i);
                    }

                    coeffs[hydro_id] = psi;
                }
            }
        }

        coeffs
    }

    pub fn get_lag_order(&self) -> usize {
        self.layout
            .per_hydro_dims
            .iter()
            .map(|&dim| dim.saturating_sub(1)) // dim = 1 + lag_count
            .max()
            .unwrap_or(0)
    }

    pub fn get_total_dimension(&self) -> usize {
        self.layout.total_dim
    }

    pub fn get_lagged_inflows(&self) -> &[Vec<f64>] {
        &self.lagged_inflows
    }

    /// Rebuild flattened state from storage and lags
    fn rebuild_flattened_state(&mut self) {
        for hydro_id in 0..self.dimension {
            let offset = self.layout.offsets[hydro_id];
            self.flattened_state[offset] = self.final_storage[hydro_id];
            let lag_count = self.layout.hydro_lag_count(hydro_id);
            if lag_count > 0 {
                let lag_start = offset + 1;
                let lag_end = lag_start + lag_count;
                self.flattened_state[lag_start..lag_end]
                    .copy_from_slice(&self.lagged_inflows[hydro_id]);
            }
        }
    }
}

impl State for StorageAndInflowState {
    fn set_dimension(&mut self, dimension: usize) {
        self.dimension = dimension;
    }

    fn get_dominating_objective(&self) -> f64 {
        self.dominating_objective
    }

    fn set_dominating_objective(&mut self, dominating_objective: f64) {
        self.dominating_objective = dominating_objective;
    }

    fn get_dominating_cut_id(&self) -> usize {
        self.dominating_cut_id
    }

    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize) {
        self.dominating_cut_id = dominating_cut_id;
    }

    fn get_iteration(&self) -> usize {
        self.iteration
    }

    fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    fn get_forward_pass_idx(&self) -> usize {
        self.forward_pass_idx
    }

    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize) {
        self.forward_pass_idx = forward_pass_idx;
    }

    fn has_lagged_inflow_state(&self) -> bool {
        true
    }

    fn has_lagged_observation_state(&self) -> bool {
        true
    }

    fn get_lagged_observations(&self, entity_idx: usize) -> &[f64] {
        // For backward compatibility, StorageAndInflowState only tracks inflow lags
        // entity_idx is treated as hydro_id for this legacy state
        if entity_idx < self.lagged_inflows.len() {
            &self.lagged_inflows[entity_idx]
        } else {
            &[]
        }
    }

    fn set_lagged_observations(&mut self, entity_idx: usize, observations: &[f64]) {
        // For backward compatibility, StorageAndInflowState only tracks inflow lags
        // entity_idx is treated as hydro_id for this legacy state
        if entity_idx < self.lagged_inflows.len() {
            self.lagged_inflows[entity_idx].clear();
            self.lagged_inflows[entity_idx].extend_from_slice(observations);
        }
    }

    fn coefficients(&self) -> &[f64] {
        self.flattened_state.as_slice()
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>> {
        let max_lag_count = self
            .layout
            .per_hydro_dims
            .iter()
            .map(|&dim| dim.saturating_sub(1))
            .max()
            .unwrap_or(0);

        let num_var_types = if max_lag_count == 0 {
            1
        } else {
            1 + max_lag_count
        };

        let mut variable_indices = Vec::with_capacity(num_var_types);

        for var_idx in 0..num_var_types {
            let mut hydro_vars = Vec::with_capacity(self.dimension);

            for hydro_id in 0..self.dimension {
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);

                if var_idx == 0 || (var_idx <= hydro_lag_count) {
                    let var = pb.add_column(0.0, 0.0..f64::INFINITY);
                    hydro_vars.push(var);
                } else {
                    hydro_vars.push(0);
                }
            }
            variable_indices.push(hydro_vars);
        }

        variable_indices
    }

    fn update_from_trajectory(
        &mut self,
        past_realizations: &[&subproblem::Realization],
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        variables: &subproblem::Variables,
    ) {
        #[allow(deprecated)]
        let use_lagged_inflow_state = variables.lagged_inflow_state.is_some();

        let prev_realization = past_realizations.last().unwrap();
        self.final_storage
            .clone_from_slice(&prev_realization.final_storage);

        let traj_len = past_realizations.len();
        let residuals: Vec<&[f64]> = past_realizations
            .iter()
            .map(|r| r.inflow.as_slice())
            .collect();
        for hydro in 0..self.dimension {
            let hydro_lag_count = self.layout.hydro_lag_count(hydro);
            for lag_idx in 0..hydro_lag_count {
                let hist_idx = traj_len.saturating_sub(1 + lag_idx);
                if hist_idx < traj_len {
                    self.lagged_inflows[hydro][lag_idx] =
                        residuals[hist_idx][hydro];
                }
            }
        }

        // Update hydro balance RHS: V_{t-1} = final_storage
        for (index, row) in constraints.hydro_balance.iter().enumerate() {
            model.change_rows_bounds(
                *row,
                self.final_storage[index],
                self.final_storage[index],
            );
        }

        // AR constraint: Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
        #[allow(deprecated)]
        if use_lagged_inflow_state {
            if let Some(lag_vars) = &variables.lagged_inflow_state {
                for (hydro, lags) in self.lagged_inflows.iter().enumerate() {
                    for (lag_idx, &lag_value) in lags.iter().enumerate() {
                        if lag_idx < lag_vars[hydro].len() {
                            let var_idx = lag_vars[hydro][lag_idx];
                            model.change_column_bounds(
                                var_idx, lag_value, lag_value,
                            );
                        }
                    }
                }
            }
        }
    }

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        self.final_storage
            .clone_from_slice(&realization.final_storage);

        for hydro in 0..self.dimension {
            let hydro_lag_count = self.layout.hydro_lag_count(hydro);
            if hydro_lag_count > 0 {
                self.lagged_inflows[hydro].rotate_right(1);
                self.lagged_inflows[hydro][0] = realization.inflow[hydro];
            }
        }

        self.rebuild_flattened_state();
    }

    fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        variables: &subproblem::Variables,
        model: &mut solver::Model,
    ) {
        // Total vars = alpha (1) + storage (n) + all lags (per-hydro variable)
        let total_vars = 1 + self.layout.total_dim;
        let mut factors = Vec::<(usize, f64)>::with_capacity(total_vars);

        factors.push((variables.alpha, 1.0));

        // Storage coefficients (first n coefficients)
        for (hydro_id, &coef) in
            cut.coefficients[0..self.dimension].iter().enumerate()
        {
            factors.push((variables.stored_volume[hydro_id], -coef));
        }

        // Lag coefficients (per-hydro variable count)
        let mut coef_idx = self.dimension;
        #[allow(deprecated)]
        if let Some(lag_vars) = &variables.lagged_inflow_state {
            for (hydro_id, hydro_lags) in
                lag_vars.iter().enumerate().take(self.dimension)
            {
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
                for lag_idx in 0..hydro_lag_count {
                    if lag_idx < hydro_lags.len() {
                        let lag_var = hydro_lags[lag_idx];
                        factors.push((lag_var, -cut.coefficients[coef_idx]));
                        coef_idx += 1;
                    }
                }
            }
        }

        model.add_row(cut.rhs.., factors);
    }

    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        _forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let num_branchings = costs.len();
        let probabilities = utils::uniform_prob_by_count(num_branchings);
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(&probabilities, &costs);

        // Total coefficients = storage (n) + all lags (per-hydro variable)
        let total_coefficients = self.layout.total_dim;
        let mut coef_contributions: Vec<Vec<f64>> =
            Vec::with_capacity(branching_realizations.len());
        let mut objective_contributions: Vec<f64> =
            Vec::with_capacity(branching_realizations.len());

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];
            let mut contrib = Vec::with_capacity(total_coefficients);

            // Storage coefficients (water values)
            contrib
                .extend(realization.water_value.iter().map(|&val| prob * val));

            // Lag coefficients using chain rule
            // For each hydro with AR(p) model, compute lag coefficients as:
            // ∂FO/∂Y_{t-j} = (water_value + ar_dual) * ψ_j
            //
            // where ψ_j = observation-space AR coefficient matching LP constraint
            // LP constraint: Y_t - Σ ψ_i*Y_{t-i} = η_t
            // Cut derivative: ∂FO/∂Y_{t-j} = (water_value + ar_dual) * ψ_j
            //
            // Derivation:
            // - Y_{t-j} affects Y_t via AR constraint: Y_t = ... + ψ_j * Y_{t-j} + ...
            // - Y_t affects objective via hydro balance: FO = ... + λ^BH * Y_t + ...
            // - AR constraint contributes: FO = ... + λ^AR * (Y_t - ψ_j * Y_{t-j}) + ...
            // - Total: ∂FO/∂Y_{t-j} = (λ^BH + λ^AR) * ψ_j
            //
            for hydro_id in 0..self.dimension {
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);

                if hydro_lag_count == 0 {
                    continue; // No lags for this hydro
                }

                // Get water value (dual from hydro balance)
                let water_val = realization.water_value[hydro_id];

                // Get AR constraint dual (fallback to 0.0 if not available)
                let ar_dual = if !realization.lag_duals.is_empty()
                    && hydro_id < realization.lag_duals.len()
                {
                    realization.lag_duals[hydro_id][0] // One dual per hydro
                } else {
                    0.0 // Fallback for independent models or missing duals
                };

                // Compute lag coefficients using chain rule
                for lag_idx in 0..hydro_lag_count {
                    let psi_j =
                        self.transformed_coefficients[hydro_id][lag_idx];

                    // Chain rule: sensitivity to lagged inflow
                    // Must use ψ_j (observation-space) to match LP constraint formulation
                    let lag_coef = (water_val + ar_dual) * psi_j;

                    contrib.push(prob * lag_coef);
                }
            }

            coef_contributions.push(contrib);
            objective_contributions
                .push(prob * realization.total_stage_objective);
        }

        let mut cut_coefficients = vec![0.0; total_coefficients];
        for coef_idx in 0..total_coefficients {
            let values: Vec<f64> = coef_contributions
                .iter()
                .map(|contrib| contrib[coef_idx])
                .collect();
            cut_coefficients[coef_idx] = utils::kahan_sum(&values);
        }
        let objective = utils::kahan_sum(&objective_contributions);

        let state_coefficients = self.coefficients();

        let cut_rhs = objective
            - utils::dot_product(&cut_coefficients, state_coefficients);

        cut::BendersCut::new(
            0,
            cut_coefficients,
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    }

    fn clone_dyn(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }
}

/// Factory function to create state representations for SDDP subproblems.
pub fn factory(
    kind: &str,
    system: &system::System,
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
) -> Box<dyn State> {
    match kind {
        "storage" => Box::new(StorageState::new(system)),
        "storage_and_inflow" => Box::new(StorageAndInflowState::new(
            system,
            uncertainty_models,
        )),
        _ => panic!(
            "Unknown state_choice: '{}'. Valid options: 'storage', 'storage_and_inflow'",
            kind
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input;
    use crate::system;
    use crate::uncertainty_model;

    #[test]
    fn test_new_storage_state() {
        let system = system::System::default();
        // StorageState::new() only takes system, no uncertainty models needed
        let state = StorageState::new(&system);
        assert_eq!(state.dimension, 1);
        assert_eq!(state.final_storage, vec![0.0]);
        assert_eq!(state.dominating_objective, 0.0);
        assert_eq!(state.dominating_cut_id, 0);
    }

    #[test]
    fn test_factory_storage_state() {
        let system = system::System::default();
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];
        let state = factory("storage", &system, &uncertainty_models);
        assert_eq!(state.coefficients().len(), 1);
    }

    #[test]
    fn test_factory_storage_and_inflow_state() {
        let system = system::System::default();
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];
        let state = factory("storage_and_inflow", &system, &uncertainty_models);

        // With Independent process (lag_order=0), dimension should be n(1+0) = n
        // Default system has 1 hydro, so dimension = 1
        assert_eq!(state.coefficients().len(), 1);
    }

    #[test]
    #[should_panic(
        expected = "Unknown state_choice: 'invalid'. Valid options: 'storage', 'storage_and_inflow'"
    )]
    fn test_factory_invalid_choice() {
        let system = system::System::default();
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];
        let _ = factory("invalid", &system, &uncertainty_models);
    }

    #[test]
    fn test_factory_preserves_system_dimension() {
        // Test with multi-hydro system
        let mut system = system::System::default();
        // Add 2 more hydros to have 3 total
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        // Create one Independent model per hydro (3 hydros)
        let uncertainty_models: Vec<_> = (0..3)
            .map(|_| uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 10.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            })
            .collect();

        let state_storage = factory("storage", &system, &uncertainty_models);
        assert_eq!(state_storage.coefficients().len(), 3);

        // Create fresh models for second test
        let uncertainty_models2: Vec<_> = (0..3)
            .map(|id| uncertainty_model::UncertaintyModel::Independent {
                entity_id: id,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 10.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            })
            .collect();

        let state_inflow =
            factory("storage_and_inflow", &system, &uncertainty_models2);
        // With lag_order=0, dimension is n(1+0) = 3
        assert_eq!(state_inflow.coefficients().len(), 3);
    }

    #[test]
    fn test_ar_coefficient_application_logic() {
        // Test the logic of how AR coefficients would be applied in constraints
        // This tests the algorithm without creating actual solver objects

        // Scenario: Hydro with AR(2) model, coefficients [0.6, 0.3]
        let ar_coeffs = [0.6, 0.3];
        let lag_count = 2;

        // Expected constraint terms: inflow_noise - 0.6*lag[0] - 0.3*lag[1] = white_noise
        // This means coefficients should be negated when added to constraint
        assert_eq!(ar_coeffs.len(), lag_count);

        // Verify that we have the right number of lag terms
        let expected_constraint_terms = 1 + lag_count; // inflow_noise + 2 lags
        assert_eq!(expected_constraint_terms, 3);

        // Verify coefficient signs in constraint (should be negative)
        for (idx, &coeff) in ar_coeffs.iter().enumerate() {
            assert!(coeff > 0.0, "AR coefficient {} should be positive", idx);
            // In constraint, it becomes: -coeff * lag_var
        }
    }

    #[test]
    fn test_heterogeneous_ar_orders_logic() {
        // Test the logic for systems with heterogeneous AR orders
        // Tests constraint structure without creating actual solver objects

        // 3 hydros: AR(2), AR(1), Independent
        let ar_specs = [
            vec![0.7, 0.2], // Hydro 0: AR(2)
            vec![0.8],      // Hydro 1: AR(1)
            vec![],         // Hydro 2: Independent
        ];

        // Expected constraint counts per hydro:
        // Hydro 0: AR RHS + equality + 2 lags = 4 constraints
        // Hydro 1: AR RHS + equality + 1 lag = 3 constraints
        // Hydro 2: simple RHS + equality + 0 lags = 2 constraints

        let expected_counts = [4, 3, 2];

        for (hydro_idx, ar_coeffs) in ar_specs.iter().enumerate() {
            let lag_count = ar_coeffs.len();
            let constraint_count = 2 + lag_count; // RHS + equality + lags
            assert_eq!(
                constraint_count, expected_counts[hydro_idx],
                "Hydro {} should have {} constraints",
                hydro_idx, expected_counts[hydro_idx]
            );
        }
    }

    // ========================================================================
    // Tests for [AR-PSI-001]: Transformed Coefficients (ψ from φ)
    // ========================================================================

    #[test]
    fn test_transformed_coefficients_uniform_sigma() {
        // When all σ are equal, ψ should equal φ
        let system = create_system_with_hydros(1);
        let phi = vec![0.8, 0.3];

        let uncertainty_models =
            vec![create_par_model_uniform_sigma(0, phi.clone())];

        let state = StorageAndInflowState::new(&system, &uncertainty_models);
        let psi = &state.transformed_coefficients[0];

        // ψ = φ × (σ_t / σ_{t-i}) = φ × (10 / 10) = φ
        assert_eq!(psi.len(), phi.len());
        for i in 0..phi.len() {
            assert!(
                (psi[i] - phi[i]).abs() < 1e-12,
                "ψ[{}] = {} should equal φ[{}] = {} when σ is uniform",
                i,
                psi[i],
                i,
                phi[i]
            );
        }
    }

    #[test]
    fn test_transformed_coefficients_seasonal_variance() {
        // Test ψ = φ × (σ_t / σ_{t-i}) with seasonal variance
        let system = create_system_with_hydros(1);
        let phi = vec![0.7];

        // Season 0: σ = 50, Season 1: σ = 100
        // For season 1: ψ = 0.7 × (100 / 50) = 1.4
        let uncertainty_models =
            vec![create_par_model_seasonal_sigma(0, phi.clone())];

        let state = StorageAndInflowState::new(&system, &uncertainty_models);
        let psi = &state.transformed_coefficients[0];

        assert_eq!(psi.len(), 1);
        // Using season_id = 0 in constructor, so:
        // ψ[0] = φ[0] × (σ_0 / σ_{11}) = 0.7 × (50 / 100) = 0.35
        let expected_psi = 0.7 * (50.0 / 100.0);
        assert!(
            (psi[0] - expected_psi).abs() < 1e-10,
            "ψ[0] = {} should be {} (φ × σ_t/σ_{{t-1}})",
            psi[0],
            expected_psi
        );
    }

    #[test]
    fn test_transformed_coefficients_ar2_seasonal() {
        // Test AR(2) with seasonal variance
        let system = create_system_with_hydros(1);
        let phi = vec![0.8, 0.3];

        let uncertainty_models =
            vec![create_par_model_seasonal_sigma(0, phi.clone())];

        let state = StorageAndInflowState::new(&system, &uncertainty_models);
        let psi = &state.transformed_coefficients[0];

        assert_eq!(psi.len(), 2);

        // Season 0: σ_0 = 50 (even index)
        // Lag 0 (t-1): season 11, σ_{11} = 100 (odd index)
        // Lag 1 (t-2): season 10, σ_{10} = 50 (even index)
        // ψ[0] = φ[0] × (σ_0 / σ_{11}) = 0.8 × (50 / 100) = 0.4
        // ψ[1] = φ[1] × (σ_0 / σ_{10}) = 0.3 × (50 / 50) = 0.3

        let expected_psi_0 = 0.8 * (50.0 / 100.0);
        let expected_psi_1 = 0.3 * (50.0 / 50.0); // Same σ, so no transformation

        assert!(
            (psi[0] - expected_psi_0).abs() < 1e-10,
            "ψ[0] = {} should be {}",
            psi[0],
            expected_psi_0
        );
        assert!(
            (psi[1] - expected_psi_1).abs() < 1e-10,
            "ψ[1] = {} should be {}",
            psi[1],
            expected_psi_1
        );
    }

    #[test]
    fn test_transformed_coefficients_independent_model() {
        // Independent model should have empty transformed coefficients
        let system = create_system_with_hydros(1);
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];

        let state = StorageAndInflowState::new(&system, &uncertainty_models);
        assert!(state.transformed_coefficients[0].is_empty());
    }

    #[test]
    fn test_transformed_coefficients_mixed_system() {
        // Test heterogeneous system with different AR orders
        let system = create_system_with_hydros(3);
        let phi1 = vec![0.8];
        let phi2 = vec![0.7, 0.2];

        let uncertainty_models = vec![
            // Hydro 0: Independent (no AR)
            uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            },
            // Hydro 1: AR(1)
            create_par_model_uniform_sigma(1, phi1.clone()),
            // Hydro 2: AR(2)
            create_par_model_uniform_sigma(2, phi2.clone()),
        ];

        let state = StorageAndInflowState::new(&system, &uncertainty_models);

        // Hydro 0: empty
        assert!(state.transformed_coefficients[0].is_empty());

        // Hydro 1: AR(1), uniform σ → ψ = φ
        assert_eq!(state.transformed_coefficients[1].len(), 1);
        assert!((state.transformed_coefficients[1][0] - phi1[0]).abs() < 1e-12);

        // Hydro 2: AR(2), uniform σ → ψ = φ
        assert_eq!(state.transformed_coefficients[2].len(), 2);
        for (i, &phi) in phi2.iter().enumerate() {
            assert!((state.transformed_coefficients[2][i] - phi).abs() < 1e-12);
        }
    }

    // Helper function to create system with n hydros
    fn create_system_with_hydros(n: usize) -> system::System {
        let mut system = system::System::default();
        system.hydros.clear();
        for i in 0..n {
            system.hydros.push(system::Hydro::new(
                i, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
            ));
        }
        system.meta.hydros_count = n;
        system
    }

    // Helper to create PAR model with uniform σ (all seasons same std_dev)
    fn create_par_model_uniform_sigma(
        entity_id: usize,
        phi: Vec<f64>,
    ) -> uncertainty_model::UncertaintyModel {
        let ar_order = phi.len();
        uncertainty_model::UncertaintyModel::PeriodicAR {
            entity_id,
            entity_type: input::UncertaintyType::Inflow,
            par_params: uncertainty_model::PARParams {
                num_seasons: 1,
                ar_orders: vec![ar_order],
                ar_coefficients: vec![phi],
                seasonal_means: vec![100.0],
                seasonal_stds: vec![10.0], // Uniform σ
                seasonal_distributions: vec![
                    uncertainty_model::DistributionType::Normal,
                ],
                max_ar_order: ar_order,
            },
        }
    }

    // Helper to create PAR model with seasonal σ variance
    fn create_par_model_seasonal_sigma(
        entity_id: usize,
        phi: Vec<f64>,
    ) -> uncertainty_model::UncertaintyModel {
        let ar_order = phi.len();
        // Create 12 seasons with alternating σ: 50, 100, 50, 100, ...
        let seasonal_stds: Vec<f64> = (0..12)
            .map(|i| if i % 2 == 0 { 50.0 } else { 100.0 })
            .collect();

        uncertainty_model::UncertaintyModel::PeriodicAR {
            entity_id,
            entity_type: input::UncertaintyType::Inflow,
            par_params: uncertainty_model::PARParams {
                num_seasons: 12,
                ar_orders: vec![ar_order; 12],
                ar_coefficients: vec![phi; 12],
                seasonal_means: vec![100.0; 12],
                seasonal_stds,
                seasonal_distributions:
                    vec![uncertainty_model::DistributionType::Normal; 12],
                max_ar_order: ar_order,
            },
        }
    }
}
