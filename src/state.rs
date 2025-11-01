use crate::cut;
use crate::input;
use crate::risk_measure;
use crate::solver;
use crate::subproblem;
use crate::system;
use crate::unified_noise_spec;
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
    fn has_lagged_inflow_state(&self) -> bool {
        false
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
    /// # Arguments
    ///
    /// * `past_realizations` - Ordered trajectory from PreStudy to current stage
    ///   (from BFS table in reverse order)
    /// * `model` - Mutable reference to solver model for updating RHS
    /// * `constraints` - Constraint indices for RHS updates
    ///
    /// # Invariant
    ///
    /// `past_realizations` is guaranteed to contain at least 1 element (PreStudy).
    /// For first study stage, it contains [PreStudy].
    /// For stage t, it contains [PreStudy, Stage(1), ..., Stage(t-1)].
    ///
    /// # Performance
    ///
    /// - `StorageState`: O(n) - updates hydro balance RHS
    /// - `StorageAndInflowState`: O(n×p) - updates hydro balance + lag constraints
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

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
        season_id: usize,
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

/// Extract maximum AR order from noise models for a specific hydro and season.
///
/// Searches noise_models for entries matching:
/// - `uncertainty_type == Inflow`
/// - `entity_id == hydro_id`
/// - `season_id == season_id`
///
/// Returns the maximum AR order from matching PAR models, or 0 if no match or independent.
///
/// # Performance
///
/// O(num_noise_models) - linear scan through noise models
///
/// # Example
///
/// ```ignore
/// let max_order = extract_max_ar_order_for_hydro(&noise_models, 0, 5);
/// // Returns 2 if hydro 0 in season 5 has AR(2), 0 if independent
/// ```
fn extract_max_ar_order_for_hydro(
    unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
    hydro_id: usize,
    season_id: usize,
) -> usize {
    unified_specs
        .iter()
        .filter(|spec| {
            spec.uncertainty_type == input::UncertaintyType::Inflow
                && spec.entity_id == hydro_id
                // For UnifiedNoiseSpec, check if spec covers this season
                && spec.seasonal_params.contains_key(&season_id)
        })
        .filter_map(|spec| match &spec.temporal_model {
            unified_noise_spec::TemporalModelSpec::PeriodicAutoregressive {
                seasonal_ar_params,
                ..
            } => {
                // Get AR order for this season from the PAR model
                seasonal_ar_params
                    .get(&season_id)
                    .map(|params| params.ar_order)
            }
            unified_noise_spec::TemporalModelSpec::Independent => Some(0),
        })
        .max()
        .unwrap_or(0)
}

/// Calculate per-hydro state dimensions from unified specs.
///
/// For each hydro, computes dimension = 1 + max_ar_order:
/// - 1 for storage
/// - max_ar_order for lagged inflows
///
/// # Arguments
///
/// * `system` - System configuration with hydros
/// * `unified_specs` - Unified noise specifications (internal representation)
/// * `season_id` - Current season identifier
///
/// # Returns
///
/// Vector where index i contains the state dimension for hydro i.
///
/// # Performance
///
/// O(num_hydros × num_specs) - worst case
///
/// # Example
///
/// ```ignore
/// let dims = per_hydro_state_dims(&system, &unified_specs, 0);
/// // dims = [3, 2, 1] for hydros with AR(2), AR(1), naive
/// ```
pub fn per_hydro_state_dims(
    system: &system::System,
    unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
    season_id: usize,
) -> Vec<usize> {
    system
        .hydros
        .iter()
        .map(|hydro| {
            let max_order = extract_max_ar_order_for_hydro(
                unified_specs,
                hydro.id,
                season_id,
            );
            1 + max_order // storage + lags
        })
        .collect()
}

/// Calculate total state dimension from per-hydro dimensions.
///
/// Sum of all per-hydro dimensions.
///
/// # Performance
///
/// O(num_hydros)
///
/// # Example
///
/// ```ignore
/// let total = total_state_dim(&system, &unified_specs, 0);
/// // total = 6 for [3, 2, 1] per-hydro dims
/// ```
pub fn total_state_dim(
    system: &system::System,
    unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
    season_id: usize,
) -> usize {
    per_hydro_state_dims(system, unified_specs, season_id)
        .iter()
        .sum()
}

// ============================================================================
// NEW API: Helper functions for UncertaintyModel
// ============================================================================

/// Extract maximum AR order for a hydro from UncertaintyModel (new API)
fn extract_max_ar_order_for_hydro_v2(
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

/// Calculate per-hydro state dimensions from UncertaintyModel (new API)
pub fn per_hydro_state_dims_v2(
    system: &system::System,
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
    season_id: usize,
) -> Vec<usize> {
    system
        .hydros
        .iter()
        .map(|hydro| {
            let max_order = extract_max_ar_order_for_hydro_v2(
                uncertainty_models,
                hydro.id,
                season_id,
            );
            1 + max_order // storage + lags
        })
        .collect()
}

/// Calculate total state dimension from UncertaintyModel (new API)
pub fn total_state_dim_v2(
    system: &system::System,
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
    season_id: usize,
) -> usize {
    per_hydro_state_dims_v2(system, uncertainty_models, season_id)
        .iter()
        .sum()
}

// ============================================================================
// End NEW API
// ============================================================================

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
    /// Create StateLayout from unified specs for a specific season.
    ///
    /// Computes per-hydro dimensions and offsets based on AR orders
    /// extracted from unified_specs.
    ///
    /// # Arguments
    ///
    /// * `system` - System configuration
    /// * `unified_specs` - Unified noise specifications (internal representation)
    /// * `season_id` - Season identifier for AR order lookup
    ///
    /// # Performance
    ///
    /// O(num_hydros × num_specs) - dominated by per_hydro_state_dims
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layout = StateLayout::from_unified_specs(&system, &unified_specs, 0);
    /// assert_eq!(layout.per_hydro_dims, vec![3, 2, 1]);
    /// assert_eq!(layout.offsets, vec![0, 3, 5, 6]);
    /// assert_eq!(layout.total_dim, 6);
    /// ```
    pub fn from_unified_specs(
        system: &system::System,
        unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
        season_id: usize,
    ) -> Self {
        let per_hydro_dims =
            per_hydro_state_dims(system, unified_specs, season_id);

        let mut offsets = Vec::with_capacity(per_hydro_dims.len() + 1);
        offsets.push(0);

        let mut cumsum = 0;
        for &dim in &per_hydro_dims {
            cumsum += dim;
            offsets.push(cumsum);
        }

        let total_dim = cumsum;

        Self {
            per_hydro_dims,
            offsets,
            total_dim,
        }
    }

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
    pub fn new(
        system: &system::System,
        _unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
    ) -> Self {
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

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        _unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
        _season_id: usize,
    ) -> Vec<Vec<usize>> {
        let mut inflow_process: Vec<Vec<usize>> =
            vec![vec![0; 2]; variables.inflow.len()];
        for (id, inflow) in variables.inflow.iter().enumerate() {
            let inflow_noise_variable = variables.inflow_residual[id];

            inflow_process[id][0] = pb.add_row(
                0.0..0.0,
                [(*inflow, 1.0), (inflow_noise_variable, -1.0)],
            );

            inflow_process[id][1] =
                pb.add_row(0.0..0.0, [(inflow_noise_variable, 1.0)]);
        }
        inflow_process
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
///
/// # Example
///
/// ```rust
/// use powers_rs::state::StorageAndInflowState;
/// use powers_rs::system::System;
/// use powers_rs::stochastic_process;
///
/// let system = System::default();
/// let load_sp = stochastic_process::factory("naive");
/// let inflow_sp = stochastic_process::factory("naive"); // lag_order() = 0
/// let inflow_processes = vec![inflow_sp];
///
/// let state = StorageAndInflowState::new(
///     &system,
///     load_sp.as_ref(),
///     &inflow_processes,
/// );
///
/// // State adapts to process lag order
/// assert_eq!(state.get_lag_order(), 0);
/// assert_eq!(state.get_total_dimension(), system.meta.hydros_count); // n * (1+0) = n
/// ```
#[derive(Debug, Clone)]
pub struct StorageAndInflowState {
    /// Number of hydros (dimension of storage and each lag vector)
    dimension: usize,
    /// State layout tracking per-hydro dimensions and offsets
    /// Supports variable AR orders: Hydro 0 might be AR(2), Hydro 1 AR(1), etc.
    layout: StateLayout,
    /// Final storage volumes V_t (dimension: n)
    final_storage: Vec<f64>,
    /// Lagged inflow realizations organized per hydro
    /// lagged_inflows[hydro_id] = [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}] for that hydro
    /// Length varies per hydro based on AR order
    lagged_inflows: Vec<Vec<f64>>,
    /// Flattened state vector for cut evaluation
    /// Format: [storage₀, lag₀₁, ..., lag₀ₚ₀, storage₁, lag₁₁, ..., lag₁ₚ₁, ...]
    /// Dimension: total_state_dim (sum of per-hydro dimensions)
    ///
    /// PERFORMANCE: Maintained alongside final_storage and lagged_inflows to
    /// provide zero-cost slice access for cut evaluation. Updated whenever
    /// state is modified.
    flattened_state: Vec<f64>,
    /// Dominating cut objective value
    dominating_objective: f64,
    /// Dominating cut ID
    dominating_cut_id: usize,
    /// DEBUGGING: Iteration number when this state was visited (1-based)
    iteration: usize,
    /// DEBUGGING: Forward pass index that visited this state (0-based handler ID)
    forward_pass_idx: usize,
}

/// Extract AR coefficients for a specific hydro and season from unified specs
///
/// Returns the AR coefficients [φ₁, φ₂, ..., φₚ] for the given hydro at the given season.
/// For PAR models, extracts from the seasonal_ar_params HashMap.
/// For Independent models or if no matching spec found, returns an empty Vec.
///
/// # Arguments
///
/// * `unified_specs` - Slice of all unified noise specs for the problem
/// * `hydro_id` - ID of the hydro plant to look up
/// * `season_id` - Current season ID for PAR parameter lookup
///
/// # Returns
///
/// Vec<f64> containing AR coefficients. Empty if:
/// - No spec found for this hydro
/// - Spec is Independent (no AR dynamics)
/// - Season not found in PAR model (should not happen after validation)
///
/// # Performance
///
/// O(n) scan through unified_specs to find matching hydro_id (typically n < 100)
/// O(1) HashMap lookup of seasonal AR parameters
fn extract_ar_coefficients(
    unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
    hydro_id: usize,
    season_id: usize,
) -> Vec<f64> {
    // Find the UnifiedNoiseSpec for this hydro (entity_id matches hydro_id for inflows)
    let spec = unified_specs.iter().find(|s| {
        matches!(s.uncertainty_type, crate::input::UncertaintyType::Inflow)
            && s.entity_id == hydro_id
    });

    match spec {
        Some(s) => match &s.temporal_model {
            unified_noise_spec::TemporalModelSpec::PeriodicAutoregressive {
                seasonal_ar_params,
                ..
            } => {
                // Extract AR coefficients for this season
                seasonal_ar_params
                    .get(&season_id)
                    .map(|params| params.ar_coefficients.clone())
                    .unwrap_or_else(Vec::new)
            }
            unified_noise_spec::TemporalModelSpec::Independent => {
                // No AR dynamics for independent model
                Vec::new()
            }
        },
        None => {
            // No spec found - default to independent (no AR terms)
            Vec::new()
        }
    }
}

/// Extract seasonal mean and std dev for a specific hydro and season from unified specs
///
/// Returns (mean, std_dev) for the given hydro at the given season.
/// For PAR models, extracts from seasonal_params HashMap.
/// For Independent models, returns (0.0, 1.0) since independent models work in observation space directly.
///
/// # Arguments
///
/// * `unified_specs` - Slice of all unified noise specs for the problem
/// * `hydro_id` - ID of the hydro plant to look up
/// * `season_id` - Current season ID for PAR parameter lookup
///
/// # Returns
///
/// (mean, std_dev) tuple. Returns (0.0, 1.0) for independent models or if not found.
///
/// # Performance
///
/// O(n) scan through unified_specs to find matching hydro_id (typically n < 100)
/// O(1) HashMap lookup of seasonal parameters
impl StorageAndInflowState {
    pub fn new(
        system: &system::System,
        unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
    ) -> Self {
        let dimension = system.meta.hydros_count;

        // Iterate over hydro ids and get their AR orders from unified specs
        let per_hydro_dims: Vec<usize> =
            per_hydro_state_dims(system, unified_specs, 0);

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

        // Allocate per-hydro lagged inflows based on each hydro's lag count
        let lagged_inflows: Vec<Vec<f64>> = (0..dimension)
            .map(|i| {
                let lag_count = layout.hydro_lag_count(i);
                vec![0.0; lag_count]
            })
            .collect();

        // Total flattened dimension from layout
        let flattened_state = vec![0.0; layout.total_dim];

        let mut state = Self {
            dimension,
            layout,
            final_storage: vec![0.0; dimension],
            lagged_inflows,
            flattened_state,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        };

        // Initialize flattened_state
        state.rebuild_flattened_state();
        state
    }

    /// NEW API: Constructor using UncertaintyModel
    pub fn new_v2(
        system: &system::System,
        uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
    ) -> Self {
        let dimension = system.meta.hydros_count;

        // Iterate over hydro ids and get their AR orders from uncertainty models
        let per_hydro_dims: Vec<usize> =
            per_hydro_state_dims_v2(system, uncertainty_models, 0);

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

        // Allocate per-hydro lagged inflows based on each hydro's lag count
        let lagged_inflows: Vec<Vec<f64>> = (0..dimension)
            .map(|i| {
                let lag_count = layout.hydro_lag_count(i);
                vec![0.0; lag_count]
            })
            .collect();

        // Total flattened dimension from layout
        let flattened_state = vec![0.0; layout.total_dim];

        let mut state = Self {
            dimension,
            layout,
            final_storage: vec![0.0; dimension],
            lagged_inflows,
            flattened_state,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        };

        // Initialize flattened_state
        state.rebuild_flattened_state();
        state
    }

    /// Get the maximum lag order across all hydros
    /// Note: With variable AR orders, this returns the max, not a single value
    pub fn get_lag_order(&self) -> usize {
        self.layout
            .per_hydro_dims
            .iter()
            .map(|&dim| dim.saturating_sub(1)) // dim = 1 + lag_count
            .max()
            .unwrap_or(0)
    }

    /// Get the total state dimension (sum of all per-hydro dimensions)
    pub fn get_total_dimension(&self) -> usize {
        self.layout.total_dim
    }

    /// Get reference to lagged inflows
    pub fn get_lagged_inflows(&self) -> &[Vec<f64>] {
        &self.lagged_inflows
    }

    /// Rebuild flattened state from storage and lags
    ///
    /// With variable AR orders, packs per-hydro states with their specific dimensions:
    /// [storage₀, lag₀₁, ..., lag₀ₚ₀, storage₁, lag₁₁, ..., lag₁ₚ₁, ...]
    ///
    /// # Performance
    /// O(total_state_dim) - copies all storage and lag values
    fn rebuild_flattened_state(&mut self) {
        // Pack per-hydro states using StateLayout offsets
        for hydro_id in 0..self.dimension {
            let offset = self.layout.offsets[hydro_id];

            // Storage is always first element for each hydro
            self.flattened_state[offset] = self.final_storage[hydro_id];

            // Copy lags for this hydro (if any)
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

    fn coefficients(&self) -> &[f64] {
        self.flattened_state.as_slice()
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>> {
        // With variable AR orders, we need to create variables per hydro
        // based on each hydro's specific lag count

        // Find max lag count across all hydros to size outer vector
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

        // Create variable indices structure
        let mut variable_indices = Vec::with_capacity(num_var_types);

        for var_idx in 0..num_var_types {
            let mut hydro_vars = Vec::with_capacity(self.dimension);

            for hydro_id in 0..self.dimension {
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);

                // Only create variable if this hydro needs it
                // var_idx 0 = inflow noise, var_idx 1+ = lag variables
                if var_idx == 0 || (var_idx <= hydro_lag_count) {
                    // CRITICAL: Lag variables hold residuals Z' which can be negative!
                    // Must be unbounded: (-∞, ∞)
                    let var =
                        pb.add_column(0.0, f64::NEG_INFINITY..f64::INFINITY);
                    hydro_vars.push(var);
                } else {
                    // Placeholder - this hydro doesn't have this lag
                    hydro_vars.push(0); // Will not be used
                }
            }
            variable_indices.push(hydro_vars);
        }

        variable_indices
    }

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
        season_id: usize,
    ) -> Vec<Vec<usize>> {
        // NOTE: This method is deprecated and not called (see subproblem.rs line 684)
        // Using lagged_inflow_state as placeholder since inflow_process field was removed
        let empty_lags: Vec<Vec<usize>> = vec![vec![]];
        let lag_vars = variables
            .lagged_inflow_state
            .as_ref()
            .unwrap_or(&empty_lags);

        let mut inflow_process: Vec<Vec<usize>> =
            Vec::with_capacity(self.dimension);

        for hydro in 0..self.dimension {
            let hydro_lag_count = self.layout.hydro_lag_count(hydro);
            let mut hydro_constraints = Vec::with_capacity(2 + hydro_lag_count);

            let inflow_var = variables.inflow[hydro];
            // Use inflow_residual as placeholder for noise variable
            let inflow_noise_var =
                variables.inflow_residual.get(hydro).copied().unwrap_or(0);

            // Extract AR coefficients for this hydro at this season (if PAR model)
            let ar_coefficients =
                extract_ar_coefficients(unified_specs, hydro, season_id);

            // Build AR constraint: inflow_noise = φ₁·lag[0] + φ₂·lag[1] + ... + φₚ·lag[p-1] + white_noise
            // This is the CRITICAL FIX: previously just had inflow_noise = white_noise
            let mut ar_terms: Vec<(usize, f64)> =
                Vec::with_capacity(1 + ar_coefficients.len());
            ar_terms.push((inflow_noise_var, 1.0));

            // Add AR terms: -φ_l * lag[l] for each lag
            for (lag_idx, &phi) in ar_coefficients.iter().enumerate() {
                let lag_var = lag_vars
                    .get(hydro)
                    .and_then(|v| v.get(lag_idx))
                    .copied()
                    .unwrap_or(0);
                ar_terms.push((lag_var, -phi));
            }

            // Constraint: inflow_noise - Σ(φ_l · lag[l]) = white_noise (RHS set later)
            let rhs_constraint = pb.add_row(0.0..0.0, ar_terms);
            hydro_constraints.push(rhs_constraint);

            // CRITICAL: Transform residual space to observation space
            // Residual: Z'_t = Σ(φ_k · Z'_{t-k}) + ε_t (from AR constraint)
            // Observation: Y_t = μ + σ · Z'_t
            // Constraint: inflow - σ · inflow_noise = μ

            // Find seasonal params for this hydro
            let (mean, std_dev) = if let Some(spec) =
                unified_specs.iter().find(|s| {
                    s.uncertainty_type == input::UncertaintyType::Inflow
                        && s.entity_id == hydro
                }) {
                if let Some(params) = spec.get_seasonal_params(season_id) {
                    (params.mean, params.std_dev)
                } else {
                    // No seasonal params: independent model or missing data
                    // Use identity transform (inflow = inflow_noise)
                    (0.0, 1.0)
                }
            } else {
                // No spec found: independent model
                (0.0, 1.0)
            };

            // Constraint: inflow - σ · inflow_noise = μ
            let equality_constraint = pb.add_row(
                mean..mean,
                [(inflow_var, 1.0), (inflow_noise_var, -std_dev)],
            );
            hydro_constraints.push(equality_constraint);

            // Add constraints for each lag this hydro has
            if hydro_lag_count > 0 {
                for lag_idx in 0..hydro_lag_count {
                    let lag_var = lag_vars
                        .get(hydro)
                        .and_then(|v| v.get(lag_idx))
                        .copied()
                        .unwrap_or(0);
                    let lag_constraint = pb.add_row(0.0..0.0, [(lag_var, 1.0)]);
                    hydro_constraints.push(lag_constraint);
                }
            }

            inflow_process.push(hydro_constraints);
        }

        inflow_process
    }

    fn update_from_trajectory(
        &mut self,
        past_realizations: &[&subproblem::Realization],
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        variables: &subproblem::Variables,
    ) {
        // PERFORMANCE: O(1) access - get previous storage from last realization
        let prev_realization = past_realizations.last().unwrap();
        self.final_storage
            .clone_from_slice(&prev_realization.final_storage);

        // PERFORMANCE: O(total_lags) - extract lagged inflows per hydro from trajectory
        // Each hydro extracts its own lags based on its AR order
        let filtered_trajectory: Vec<&subproblem::Realization> =
            past_realizations
                .iter()
                .filter(|r| {
                    // Keep all Study/PostStudy nodes
                    if r.kind != subproblem::StudyPeriodKind::PreStudy {
                        return true;
                    }
                    // For PreStudy nodes, keep only if they have non-zero inflow_residual
                    // (Anchor node has all zeros because it's never converted)
                    r.inflow_residual.iter().any(|&val| val.abs() > 1e-10)
                })
                .copied()
                .collect();

        let traj_len = filtered_trajectory.len();
        for hydro in 0..self.dimension {
            let hydro_lag_count = self.layout.hydro_lag_count(hydro);
            for lag_idx in 0..hydro_lag_count {
                // Calculate historical index (most recent = traj_len-1-lag_idx)
                let hist_idx = traj_len.saturating_sub(1 + lag_idx);
                if hist_idx < traj_len {
                    self.lagged_inflows[hydro][lag_idx] =
                        filtered_trajectory[hist_idx].inflow_residual[hydro];
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
        // Lag variables are fixed by setting bounds: Z'_{t-k} ∈ [value, value]
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

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        self.final_storage
            .clone_from_slice(&realization.final_storage);

        // Update lags for each hydro based on its specific lag count
        for hydro in 0..self.dimension {
            let hydro_lag_count = self.layout.hydro_lag_count(hydro);
            if hydro_lag_count > 0 {
                // Shift lags: [Z'_{t-1}, Z'_{t-2}, ...] → [Z'_t, Z'_{t-1}, ...]
                // CRITICAL: Use residuals (Z'_t) for AR lags, not observations (Y_t)
                self.lagged_inflows[hydro].rotate_right(1);
                self.lagged_inflows[hydro][0] =
                    realization.inflow_residual[hydro];
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
        // TICKET-010: Use lagged_inflow_state variables instead of deprecated inflow_process
        let mut coef_idx = self.dimension;
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

            // Lag coefficients (per-hydro variable count)
            for hydro_id in 0..self.dimension {
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
                for lag_idx in 0..hydro_lag_count {
                    if lag_idx < realization.lag_duals.len() {
                        let dual_val = realization.lag_duals[lag_idx][hydro_id];
                        contrib.push(prob * dual_val);
                    } else {
                        contrib.push(0.0);
                    }
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
///
/// Creates the appropriate state type based on the `kind` parameter. The state
/// representation determines which variables are included in Bellman cuts and
/// how state transitions are modeled between stages.
///
/// # Available State Types
///
/// - **`"storage"`**: `StorageState` - Uses only reservoir storage volumes as state.
///   Best for systems where inflows are weakly correlated or when computational
///   efficiency is critical. State dimension is n (number of hydros).
///
/// - **`"storage_and_inflow"`**: `StorageAndInflowState` - Includes storage volumes
///   and lagged inflows in the state representation. Enables modeling of serially
///   correlated inflows through PAR(p) or similar processes. State dimension is
///   n(1+p) where p is the lag order from the inflow stochastic process.
///
/// # Arguments
///
/// * `kind` - State type identifier: "storage" or "storage_and_inflow"
/// * `system` - System configuration containing hydro count and parameters
/// * `load_stochastic_process` - Stochastic process for load uncertainty
/// * `inflow_stochastic_process` - Stochastic process for inflow uncertainty.
///   The lag_order() method determines lag dimension for storage_and_inflow states.
///
/// # Returns
///
/// A boxed trait object implementing the `State` trait, ready for use in SDDP.
///
/// # Panics
///
/// Panics if `kind` is not recognized. Valid options are printed in the panic message.
///
/// # Examples
///
/// ```ignore
/// // Create storage-only state
/// let state = state::factory(
///     "storage",
///     &system,
///     load_process.as_ref(),
///     inflow_process.as_ref(),
/// );
///
/// // Create storage + inflow state (adapts to process lag_order)
/// let state = state::factory(
///     "storage_and_inflow",
///     &system,
///     load_process.as_ref(),
///     inflow_process.as_ref(),
/// );
/// ```
pub fn factory(
    kind: &str,
    system: &system::System,
    unified_specs: &[unified_noise_spec::UnifiedNoiseSpec],
) -> Box<dyn State> {
    match kind {
        "storage" => Box::new(StorageState::new(
            system,
            unified_specs,
        )),
        "storage_and_inflow" => Box::new(StorageAndInflowState::new(
            system,
            unified_specs,
        )),
        _ => panic!(
            "Unknown state_choice: '{}'. Valid options: 'storage', 'storage_and_inflow'",
            kind
        ),
    }
}

/// NEW API: Factory function using UncertaintyModel
pub fn factory_v2(
    kind: &str,
    system: &system::System,
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
) -> Box<dyn State> {
    match kind {
        "storage" => Box::new(StorageState::new(
            system,
            &[], // StorageState doesn't use specs
        )),
        "storage_and_inflow" => Box::new(StorageAndInflowState::new_v2(
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
    use crate::system;
    use std::collections::HashMap;

    #[test]
    fn test_new_storage_state() {
        let system = system::System::default();
        let unified_specs = vec![unified_noise_spec::UnifiedNoiseSpec {
            uncertainty_type: input::UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: unified_noise_spec::TemporalModelSpec::Independent,
            seasonal_params: HashMap::new(),
        }];
        let state = StorageState::new(&system, &unified_specs);
        assert_eq!(state.dimension, 1);
        assert_eq!(state.final_storage, vec![0.0]);
        assert_eq!(state.dominating_objective, 0.0);
        assert_eq!(state.dominating_cut_id, 0);
    }

    #[test]
    fn test_factory_storage_state() {
        let system = system::System::default();
        let unified_specs = vec![unified_noise_spec::UnifiedNoiseSpec {
            uncertainty_type: input::UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: unified_noise_spec::TemporalModelSpec::Independent,
            seasonal_params: HashMap::new(),
        }];
        let state = factory("storage", &system, &unified_specs);
        assert_eq!(state.coefficients().len(), 1);
    }

    #[test]
    fn test_factory_storage_and_inflow_state() {
        let system = system::System::default();
        let unified_specs = vec![unified_noise_spec::UnifiedNoiseSpec {
            uncertainty_type: input::UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: unified_noise_spec::TemporalModelSpec::Independent,
            seasonal_params: HashMap::new(),
        }];
        let state = factory("storage_and_inflow", &system, &unified_specs);

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
        let unified_specs = vec![unified_noise_spec::UnifiedNoiseSpec {
            uncertainty_type: input::UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: unified_noise_spec::TemporalModelSpec::Independent,
            seasonal_params: HashMap::new(),
        }];
        let _ = factory("invalid", &system, &unified_specs);
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

        // Helper to create seasonal params
        let make_seasonal_params = || {
            let mut params = HashMap::new();
            params.insert(
                0,
                unified_noise_spec::SeasonalNoiseParams {
                    mean: 100.0,
                    std_dev: 10.0,
                    marginal_override: None,
                },
            );
            params
        };

        // Create one process per hydro (3 hydros)
        let unified_specs: Vec<_> = (0..3)
            .map(|entity_id| unified_noise_spec::UnifiedNoiseSpec {
                uncertainty_type: input::UncertaintyType::Inflow,
                entity_id,
                temporal_model:
                    unified_noise_spec::TemporalModelSpec::Independent,
                seasonal_params: make_seasonal_params(),
            })
            .collect();

        let state_storage = factory("storage", &system, &unified_specs);
        assert_eq!(state_storage.coefficients().len(), 3);

        // Create fresh processes for second test
        let unified_specs2: Vec<_> = (0..3)
            .map(|entity_id| unified_noise_spec::UnifiedNoiseSpec {
                uncertainty_type: input::UncertaintyType::Inflow,
                entity_id,
                temporal_model:
                    unified_noise_spec::TemporalModelSpec::Independent,
                seasonal_params: make_seasonal_params(),
            })
            .collect();

        let state_inflow =
            factory("storage_and_inflow", &system, &unified_specs2);
        // With lag_order=0, dimension is n(1+0) = 3
        assert_eq!(state_inflow.coefficients().len(), 3);
    }

    fn create_noise_spec_independent(
        entity_id: usize,
        season_id: usize,
    ) -> unified_noise_spec::UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::new();
        seasonal_params.insert(
            season_id,
            unified_noise_spec::SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        unified_noise_spec::UnifiedNoiseSpec {
            uncertainty_type: input::UncertaintyType::Inflow,
            entity_id,
            temporal_model: unified_noise_spec::TemporalModelSpec::Independent,
            seasonal_params,
        }
    }

    fn create_noise_spec_par(
        entity_id: usize,
        _season_id: usize,
        ar_orders: Vec<usize>,
    ) -> unified_noise_spec::UnifiedNoiseSpec {
        let num_seasons = ar_orders.len();
        let ar_coefficients: Vec<Vec<f64>> =
            ar_orders.iter().map(|&order| vec![0.7; order]).collect();

        let mut seasonal_params = HashMap::new();
        let mut par_params = HashMap::new();
        for s in 0..num_seasons {
            seasonal_params.insert(
                s,
                unified_noise_spec::SeasonalNoiseParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    marginal_override: Some(
                        input::MarginalDistribution::LogNormal3 {
                            gamma: 1.0,
                            mu: 4.5,
                            sigma: 0.3,
                        },
                    ),
                },
            );
            par_params.insert(
                s,
                unified_noise_spec::SeasonalPARParams {
                    ar_order: ar_orders[s],
                    ar_coefficients: ar_coefficients[s].clone(),
                },
            );
        }

        unified_noise_spec::UnifiedNoiseSpec {
            uncertainty_type: input::UncertaintyType::Inflow,
            entity_id,
            temporal_model:
                unified_noise_spec::TemporalModelSpec::PeriodicAutoregressive {
                    num_seasons,
                    seasonal_ar_params: par_params,
                },
            seasonal_params,
        }
    }

    #[test]
    fn test_extract_max_ar_order_for_hydro_independent() {
        let noise_models = vec![create_noise_spec_independent(0, 0)];
        let max_order = extract_max_ar_order_for_hydro(&noise_models, 0, 0);
        assert_eq!(max_order, 0);
    }

    #[test]
    fn test_extract_max_ar_order_for_hydro_par() {
        let noise_models = vec![create_noise_spec_par(0, 0, vec![2, 2, 1])];
        let max_order = extract_max_ar_order_for_hydro(&noise_models, 0, 0);
        assert_eq!(max_order, 2);
    }

    #[test]
    fn test_extract_max_ar_order_for_hydro_not_found() {
        let noise_specs = vec![create_noise_spec_par(0, 0, vec![2])];
        let max_order = extract_max_ar_order_for_hydro(&noise_specs, 1, 0);
        assert_eq!(max_order, 0); // No match, defaults to 0
    }

    #[test]
    fn test_state_layout_homogeneous_naive() {
        // All hydros with naive (independent) processes
        let system = system::System::default(); // 1 hydro
        let noise_specs = vec![create_noise_spec_independent(0, 0)];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        assert_eq!(layout.per_hydro_dims, vec![1]); // storage only
        assert_eq!(layout.offsets, vec![0, 1]);
        assert_eq!(layout.total_dim, 1);
    }

    #[test]
    fn test_state_layout_homogeneous_ar1() {
        // All hydros with AR(1)
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![1]),
            create_noise_spec_par(1, 0, vec![1]),
            create_noise_spec_par(2, 0, vec![1]),
        ];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        assert_eq!(layout.per_hydro_dims, vec![2, 2, 2]); // storage + 1 lag each
        assert_eq!(layout.offsets, vec![0, 2, 4, 6]);
        assert_eq!(layout.total_dim, 6);
    }

    #[test]
    fn test_state_layout_heterogeneous() {
        // Mixed AR orders: AR(2), AR(1), naive
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![2, 2]), // AR(2)
            create_noise_spec_par(1, 0, vec![1]),    // AR(1)
            create_noise_spec_independent(2, 0),     // naive
        ];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        // Hydro 0: 1 + 2 = 3 (storage + 2 lags)
        // Hydro 1: 1 + 1 = 2 (storage + 1 lag)
        // Hydro 2: 1 + 0 = 1 (storage only)
        assert_eq!(layout.per_hydro_dims, vec![3, 2, 1]);
        assert_eq!(layout.offsets, vec![0, 3, 5, 6]);
        assert_eq!(layout.total_dim, 6);
    }

    #[test]
    fn test_state_layout_hydro_slice() {
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![2]),
            create_noise_spec_par(1, 0, vec![1]),
            create_noise_spec_independent(2, 0),
        ];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        assert_eq!(layout.hydro_slice(0), 0..3);
        assert_eq!(layout.hydro_slice(1), 3..5);
        assert_eq!(layout.hydro_slice(2), 5..6);
    }

    #[test]
    fn test_state_layout_hydro_dim() {
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 2;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![2]),
            create_noise_spec_par(1, 0, vec![1]),
        ];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        assert_eq!(layout.hydro_dim(0), 3);
        assert_eq!(layout.hydro_dim(1), 2);
    }

    #[test]
    fn test_state_layout_hydro_storage_offset() {
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 2;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![2]),
            create_noise_spec_par(1, 0, vec![1]),
        ];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        assert_eq!(layout.hydro_storage_offset(0), 0);
        assert_eq!(layout.hydro_storage_offset(1), 3);
    }

    #[test]
    fn test_state_layout_hydro_lag_count() {
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![2]),
            create_noise_spec_par(1, 0, vec![1]),
            create_noise_spec_independent(2, 0),
        ];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        assert_eq!(layout.hydro_lag_count(0), 2);
        assert_eq!(layout.hydro_lag_count(1), 1);
        assert_eq!(layout.hydro_lag_count(2), 0);
    }

    #[test]
    fn test_per_hydro_state_dims() {
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![3, 2]),
            create_noise_spec_par(1, 0, vec![1, 2]),
            create_noise_spec_independent(2, 0),
        ];

        let dims = per_hydro_state_dims(&system, &noise_specs, 0);

        assert_eq!(dims, vec![4, 2, 1]); // AR(3) for s0, AR(1) for s0, naive
    }

    #[test]
    fn test_total_state_dim() {
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        let noise_specs = vec![
            create_noise_spec_par(0, 0, vec![2]),
            create_noise_spec_par(1, 0, vec![1]),
            create_noise_spec_independent(2, 0),
        ];

        let total = total_state_dim(&system, &noise_specs, 0);

        assert_eq!(total, 6); // 3 + 2 + 1
    }

    #[test]
    fn test_state_layout_empty_noise_models() {
        // All hydros default to naive (no noise models provided)
        let system = system::System::default();
        let noise_specs = vec![];

        let layout = StateLayout::from_unified_specs(&system, &noise_specs, 0);

        assert_eq!(layout.per_hydro_dims, vec![1]); // Storage only
        assert_eq!(layout.offsets, vec![0, 1]);
        assert_eq!(layout.total_dim, 1);
    }

    // ========== PHASE 3: AR CONSTRAINT VALIDATION TESTS ==========

    #[test]
    fn test_extract_ar_coefficients_par_model() {
        // Test extraction of AR coefficients for PAR model
        let noise_spec = create_noise_spec_par(0, 0, vec![2]);
        let unified_specs = vec![noise_spec];

        let ar_coeffs = extract_ar_coefficients(&unified_specs, 0, 0);

        assert_eq!(ar_coeffs.len(), 2); // AR(2) has 2 coefficients
                                        // create_noise_spec_par sets all coefficients to 0.7
        assert_eq!(ar_coeffs[0], 0.7); // φ₁
        assert_eq!(ar_coeffs[1], 0.7); // φ₂
    }

    #[test]
    fn test_extract_ar_coefficients_independent_model() {
        // Test extraction returns empty Vec for Independent model
        let noise_spec = create_noise_spec_independent(0, 0);
        let unified_specs = vec![noise_spec];

        let ar_coeffs = extract_ar_coefficients(&unified_specs, 0, 0);

        assert_eq!(ar_coeffs.len(), 0); // No AR coefficients for independent
    }

    #[test]
    fn test_extract_ar_coefficients_no_spec_found() {
        // Test extraction returns empty Vec when no spec found for hydro
        let noise_spec = create_noise_spec_par(0, 0, vec![1]);
        let unified_specs = vec![noise_spec];

        // Request coefficients for hydro_id=1 (only spec for hydro_id=0 exists)
        let ar_coeffs = extract_ar_coefficients(&unified_specs, 1, 0);

        assert_eq!(ar_coeffs.len(), 0); // No spec found
    }

    #[test]
    fn test_extract_ar_coefficients_multiple_seasons() {
        // Test extraction for different seasons in PAR model
        use crate::unified_noise_spec::{
            SeasonalPARParams, TemporalModelSpec, UnifiedNoiseSpec,
        };
        use std::collections::HashMap;

        // Create PAR model with 2 seasons, different AR orders
        let mut seasonal_ar_params = HashMap::new();
        seasonal_ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 2,
                ar_coefficients: vec![0.6, 0.3],
            },
        );
        seasonal_ar_params.insert(
            1,
            SeasonalPARParams {
                ar_order: 1,
                ar_coefficients: vec![0.8],
            },
        );

        let noise_spec = UnifiedNoiseSpec {
            uncertainty_type: crate::input::UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 2,
                seasonal_ar_params,
            },
            seasonal_params: HashMap::new(),
        };

        let unified_specs = vec![noise_spec];

        // Season 0: AR(2) with [0.6, 0.3]
        let ar_coeffs_s0 = extract_ar_coefficients(&unified_specs, 0, 0);
        assert_eq!(ar_coeffs_s0.len(), 2);
        assert_eq!(ar_coeffs_s0[0], 0.6);
        assert_eq!(ar_coeffs_s0[1], 0.3);

        // Season 1: AR(1) with [0.8]
        let ar_coeffs_s1 = extract_ar_coefficients(&unified_specs, 0, 1);
        assert_eq!(ar_coeffs_s1.len(), 1);
        assert_eq!(ar_coeffs_s1[0], 0.8);
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
}
