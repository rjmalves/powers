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

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    );

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Vec<Vec<usize>>;

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
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
        risk_measure: &Box<dyn risk_measure::RiskMeasure>,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &Vec<subproblem::Realization>,
    ) -> cut::BendersCut;

    // default implementations
    fn update_dominating_cut(&mut self, cut: &cut::BendersCut, height: f64) {
        self.set_dominating_cut_id(cut.id);
        self.set_dominating_objective(height);
    }

    fn compute_new_cut(
        &mut self,
        risk_measure: &Box<dyn risk_measure::RiskMeasure>,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &Vec<subproblem::Realization>,
    ) -> cut::BendersCut {
        let cut = self.evaluate_cut(
            risk_measure,
            forward_trajectory,
            branching_realizations,
        );
        // side effects: when an state is used to compute a cut, the cut immediately dominates it
        self.update_dominating_cut(
            &cut,
            cut.eval_height_at_state(self.coefficients()),
        );

        cut
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
}

impl StorageState {
    pub fn new(
        system: &system::System,
        _load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        _inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Self {
        Self {
            dimension: system.meta.hydros_count,
            final_storage: vec![0.0; system.meta.hydros_count],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
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

    fn coefficients(&self) -> &[f64] {
        &self.final_storage.as_slice()
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        _load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        _inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Vec<Vec<usize>> {
        let mut col_indices = vec![vec![0; 1]; self.dimension];
        for id in (0..self.dimension).into_iter() {
            col_indices[id][0] = pb.add_column(0.0, 0.0..);
        }
        col_indices
    }

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        _load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        _inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Vec<Vec<usize>> {
        let mut inflow_process: Vec<Vec<usize>> =
            vec![vec![0; 2]; variables.inflow.len()];
        // inflow process contraints are, for each hydro:
        // inflow - inflow_noise = 0
        // inflow_noise = (value to be set in runtime)
        for (id, inflow) in variables.inflow.iter().enumerate() {
            let inflow_noise_variable =
                *variables.inflow_process.get(id).unwrap().get(0).unwrap();
            inflow_process[id][0] = pb.add_row(
                0.0..0.0,
                &[(*inflow, 1.0), (inflow_noise_variable, -1.0)],
            );
            inflow_process[id][1] =
                pb.add_row(0.0..0.0, &[(inflow_noise_variable, 1.0)]);
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
            factors.push((*stored_volume, -1.0 * cut.coefficients[hydro_id]));
        }
        model.add_row(cut.rhs.., factors);
    }

    fn evaluate_cut(
        &mut self,
        risk_measure: &Box<dyn risk_measure::RiskMeasure>,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &Vec<subproblem::Realization>,
    ) -> cut::BendersCut {
        let mut cut_coefficients = vec![0.0; self.dimension];
        let mut objective = 0.0;
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let num_branchings = costs.len();
        let probabilities = utils::uniform_prob_by_count(num_branchings);
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(&probabilities, &costs);
        for (index, realization) in branching_realizations.iter().enumerate() {
            for hydro_id in 0..self.dimension {
                cut_coefficients[hydro_id] += adjusted_probabilities[index]
                    * realization.water_value[hydro_id]
            }
            objective += adjusted_probabilities[index]
                * realization.total_stage_objective;
        }

        let last_realization = forward_trajectory.last().unwrap();

        let cut_rhs = objective
            - utils::dot_product(
                &cut_coefficients,
                &last_realization.final_storage,
            );
        // temporary sets cut id to 0 - will be updated when adding to pool
        cut::BendersCut::new(0, cut_coefficients, cut_rhs)
    }

    // clone helper for storing visited states
    fn clone_dyn(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }
}

pub fn factory(
    kind: &str,
    system: &system::System,
    load_stochastic_process: &Box<dyn stochastic_process::StochasticProcess>,
    inflow_stochastic_process: &Box<dyn stochastic_process::StochasticProcess>,
) -> Box<dyn State> {
    match kind {
        "storage" => Box::new(StorageState::new(
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
        let state = StorageState::new(&system, &load_sp, &inflow_sp);
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
        let state = factory("storage", &system, &load_sp, &inflow_sp);
        assert_eq!(state.coefficients().len(), 1);
    }
}