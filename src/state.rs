use crate::cut;
use crate::risk_measure;
use crate::solver;
use crate::stochastic_process;
use crate::subproblem;
use crate::system;
use crate::utils;

pub trait State: Send + Sync {
    // behavior that must be implemented for each state definition
    fn set_dimension(&mut self, dimension: usize);
    fn coefficients(&self) -> &[f64];
    fn get_dominating_objective(&self) -> f64;
    fn set_dominating_objective(&mut self, dominating_objective: f64);
    fn get_dominating_cut_id(&self) -> usize;
    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize);

    /// DEBUGGING: Get iteration number when this state was visited (1-based)
    fn get_iteration(&self) -> usize;
    /// DEBUGGING: Set iteration number when this state was visited
    fn set_iteration(&mut self, iteration: usize);
    /// DEBUGGING: Get forward pass index that visited this state (0-based handler ID)
    fn get_forward_pass_idx(&self) -> usize;
    /// DEBUGGING: Set forward pass index that visited this state
    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize);

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    );

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>>;

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>>;

    fn set_inflows_in_subproblem(
        &self,
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        inflows: &[f64],
    );

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
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
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
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
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
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        let mut inflow_process: Vec<Vec<usize>> =
            vec![vec![0; 2]; variables.inflow.len()];
        // inflow process contraints are, for each hydro:
        // inflow - inflow_noise = 0
        // inflow_noise = (value to be set in runtime)
        for (id, inflow) in variables.inflow.iter().enumerate() {
            let inflow_noise_variable =
                *variables.inflow_process.get(id).unwrap().first().unwrap();
            inflow_process[id][0] = pb.add_row(
                0.0..0.0,
                [(*inflow, 1.0), (inflow_noise_variable, -1.0)],
            );
            inflow_process[id][1] =
                pb.add_row(0.0..0.0, [(inflow_noise_variable, 1.0)]);
        }
        inflow_process
    }

    fn set_inflows_in_subproblem(
        &self,
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        inflows: &[f64],
    ) {
        for (index, row) in constraints.inflow_process.iter().enumerate() {
            model.change_rows_bounds(
                *row.get(1).unwrap(),
                inflows[index],
                inflows[index],
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
///
/// let state = StorageAndInflowState::new(
///     &system,
///     load_sp.as_ref(),
///     inflow_sp.as_ref(),
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
    /// Lag order from the stochastic process (p)
    lag_order: usize,
    /// Final storage volumes V_t (dimension: n)
    final_storage: Vec<f64>,
    /// Lagged inflow realizations [Y_{t-1}, ..., Y_{t-p}] (dimension: p × n)
    ///
    /// Layout: lagged_inflows[k-1] = Y_{t-k} for k=1..p
    /// - lagged_inflows[0] = Y_{t-1} (most recent lag)
    /// - lagged_inflows[1] = Y_{t-2}
    /// - lagged_inflows[p-1] = Y_{t-p} (oldest lag)
    lagged_inflows: Vec<Vec<f64>>,
    /// Flattened state vector for cut evaluation
    /// Format: [V_1, ..., V_n, Y_{t-1,1}, ..., Y_{t-1,n}, ..., Y_{t-p,n}]
    /// Dimension: n(1+p)
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

impl StorageAndInflowState {
    pub fn new(
        system: &system::System,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Self {
        let dimension = system.meta.hydros_count;
        let lag_order = inflow_stochastic_process.lag_order();

        // Pre-allocate lagged inflows: p vectors of n elements each
        let lagged_inflows = vec![vec![0.0; dimension]; lag_order];

        // Total flattened dimension: n + p×n = n(1+p)
        let total_dimension = dimension * (1 + lag_order);
        let flattened_state = vec![0.0; total_dimension];

        let mut state = Self {
            dimension,
            lag_order,
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

    /// Get the lag order (p) from the stochastic process
    pub fn get_lag_order(&self) -> usize {
        self.lag_order
    }

    /// Get the total state dimension: n(1+p)
    pub fn get_total_dimension(&self) -> usize {
        self.dimension * (1 + self.lag_order)
    }

    /// Get reference to lagged inflows
    pub fn get_lagged_inflows(&self) -> &[Vec<f64>] {
        &self.lagged_inflows
    }

    /// Rebuild flattened state from storage and lags
    ///
    /// Copies data to maintain invariant: flattened_state = [V, Y_{t-1}, ..., Y_{t-p}]
    ///
    /// # Performance
    /// O(n(1+p)) - copies all storage and lag values
    fn rebuild_flattened_state(&mut self) {
        let n = self.dimension;

        // Copy storage: indices [0..n]
        self.flattened_state[0..n].copy_from_slice(&self.final_storage);

        // Copy lags: indices [n..n(1+p)]
        for (lag_idx, lag_values) in self.lagged_inflows.iter().enumerate() {
            let start = n * (1 + lag_idx);
            let end = start + n;
            self.flattened_state[start..end].copy_from_slice(lag_values);
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

    fn coefficients(&self) -> &[f64] {
        self.flattened_state.as_slice()
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        let num_vars = if self.lag_order == 0 {
            1
        } else {
            1 + self.lag_order
        };

        let mut variable_indices = Vec::with_capacity(num_vars);

        for _var_idx in 0..num_vars {
            let mut hydro_vars = Vec::with_capacity(self.dimension);
            for _hydro in 0..self.dimension {
                let var = pb.add_column(0.0, 0.0..);
                hydro_vars.push(var);
            }
            variable_indices.push(hydro_vars);
        }

        variable_indices
    }

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        let lag_vars = &variables.inflow_process;

        let mut inflow_process: Vec<Vec<usize>> =
            Vec::with_capacity(self.dimension);

        for hydro in 0..self.dimension {
            let mut hydro_constraints = Vec::with_capacity(2 + self.lag_order);

            let inflow_var = variables.inflow[hydro];
            let inflow_noise_var = lag_vars[0][hydro];

            let equality_constraint = pb.add_row(
                0.0..0.0,
                [(inflow_var, 1.0), (inflow_noise_var, -1.0)],
            );
            hydro_constraints.push(equality_constraint);

            let rhs_constraint =
                pb.add_row(0.0..0.0, [(inflow_noise_var, 1.0)]);
            hydro_constraints.push(rhs_constraint);

            if self.lag_order > 0 {
                for lag_vars_for_lag in lag_vars.iter().skip(1) {
                    let lag_var = lag_vars_for_lag[hydro];
                    let lag_constraint = pb.add_row(0.0..0.0, [(lag_var, 1.0)]);
                    hydro_constraints.push(lag_constraint);
                }
            }

            inflow_process.push(hydro_constraints);
        }

        inflow_process
    }

    fn set_inflows_in_subproblem(
        &self,
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        inflows: &[f64],
    ) {
        for (hydro, hydro_constraints) in
            constraints.inflow_process.iter().enumerate()
        {
            let inflow_rhs_constraint = hydro_constraints[1];
            model.change_rows_bounds(
                inflow_rhs_constraint,
                inflows[hydro],
                inflows[hydro],
            );

            for lag_idx in 0..self.lag_order {
                let lag_constraint = hydro_constraints[2 + lag_idx];
                let lag_value = self.lagged_inflows[lag_idx][hydro];
                model.change_rows_bounds(lag_constraint, lag_value, lag_value);
            }
        }
    }

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        self.final_storage
            .clone_from_slice(&realization.final_storage);

        if self.lag_order > 0 {
            self.lagged_inflows.rotate_right(1);
            self.lagged_inflows[0].clone_from_slice(&realization.inflow);
        }

        self.rebuild_flattened_state();
    }

    fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        variables: &subproblem::Variables,
        model: &mut solver::Model,
    ) {
        let total_vars = 1 + self.dimension * (1 + self.lag_order);
        let mut factors = Vec::<(usize, f64)>::with_capacity(total_vars);

        factors.push((variables.alpha, 1.0));

        for (hydro_id, &coef) in
            cut.coefficients[0..self.dimension].iter().enumerate()
        {
            factors.push((variables.stored_volume[hydro_id], -coef));
        }

        let mut coef_idx = self.dimension;
        for lag_idx in 0..self.lag_order {
            let lag_var_idx = 1 + lag_idx;
            for hydro_id in 0..self.dimension {
                let lag_var = variables.inflow_process[lag_var_idx][hydro_id];
                factors.push((lag_var, -cut.coefficients[coef_idx]));
                coef_idx += 1;
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

        let total_coefficients = self.dimension * (1 + self.lag_order);
        let mut coef_contributions: Vec<Vec<f64>> =
            Vec::with_capacity(branching_realizations.len());
        let mut objective_contributions: Vec<f64> =
            Vec::with_capacity(branching_realizations.len());

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];
            let mut contrib = Vec::with_capacity(total_coefficients);

            contrib
                .extend(realization.water_value.iter().map(|&val| prob * val));

            for lag_idx in 0..self.lag_order {
                if lag_idx < realization.lag_duals.len() {
                    contrib.extend(
                        realization.lag_duals[lag_idx]
                            .iter()
                            .map(|&val| prob * val),
                    );
                } else {
                    contrib.extend(vec![0.0; self.dimension]);
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

pub fn factory(
    kind: &str,
    system: &system::System,
    load_stochastic_process: &dyn stochastic_process::StochasticProcess,
    inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
) -> Box<dyn State> {
    match kind {
        "storage" => Box::new(StorageState::new(
            system,
            load_stochastic_process,
            inflow_stochastic_process,
        )),
        "storage_and_inflow" => Box::new(StorageAndInflowState::new(
            system,
            load_stochastic_process,
            inflow_stochastic_process,
        )),
        _ => panic!("state kind {} not supported", kind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system;

    #[test]
    fn test_new_storage_state() {
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let state =
            StorageState::new(&system, load_sp.as_ref(), inflow_sp.as_ref());
        assert_eq!(state.dimension, 1);
        assert_eq!(state.final_storage, vec![0.0]);
        assert_eq!(state.dominating_objective, 0.0);
        assert_eq!(state.dominating_cut_id, 0);
    }

    #[test]
    fn test_factory_storage_state() {
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let state =
            factory("storage", &system, load_sp.as_ref(), inflow_sp.as_ref());
        assert_eq!(state.coefficients().len(), 1);
    }
}
