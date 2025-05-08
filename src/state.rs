use crate::cut;
use crate::risk_measure;
use crate::subproblem;
use crate::utils;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct State {
    dimension: usize,
    initial_storage: Arc<Vec<f64>>,
    final_storage: Arc<Vec<f64>>,
    dominating_objective: f64,
    dominating_cut_id: usize,
}

impl State {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            initial_storage: Arc::new(vec![]),
            final_storage: Arc::new(vec![]),
            dominating_objective: 0.0,
            dominating_cut_id: 0,
        }
    }

    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    pub fn get_initial_storage(&self) -> &[f64] {
        self.initial_storage.as_slice()
    }

    pub fn get_final_storage(&self) -> &[f64] {
        self.initial_storage.as_slice()
    }

    pub fn get_dominating_objective(&self) -> f64 {
        self.dominating_objective
    }

    pub fn get_dominating_cut_id(&self) -> usize {
        self.dominating_cut_id
    }

    pub fn set_initial_storage(&mut self, storage: Vec<f64>) {
        self.initial_storage = Arc::new(storage);
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.final_storage.as_slice()
    }

    fn evaluate_cut(
        &mut self,
        cut_id: usize,
        risk_measure: &risk_measure::RiskMeasure,
        forward_realization: &subproblem::Realization,
        branching_realizations: &Vec<subproblem::Realization>,
    ) -> cut::BendersCut {
        let mut cut_coefficients = vec![0.0; self.dimension];
        let mut objective = 0.0;
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let num_branchings = costs.len();
        let p = 1.0 / num_branchings as f64;
        let probabilities = vec![p; num_branchings];
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

        let cut_rhs = objective
            - utils::dot_product(
                &cut_coefficients,
                &forward_realization.initial_storage,
            );
        cut::BendersCut::new(cut_id, cut_coefficients, cut_rhs)
    }

    pub fn compute_new_cut(
        &mut self,
        cut_id: usize,
        risk_measure: &risk_measure::RiskMeasure,
        forward_realization: &subproblem::Realization,
        branching_realizations: &Vec<subproblem::Realization>,
    ) -> cut::BendersCut {
        let cut = self.evaluate_cut(
            cut_id,
            risk_measure,
            forward_realization,
            branching_realizations,
        );

        // side effects: when an state is used to compute a cut, the cut immediately dominates it
        self.update_dominating_cut(
            &cut,
            cut.eval_height_at_state(self.coefficients()),
        );

        cut
    }

    pub fn update_with_parent_node_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        self.initial_storage = Arc::clone(&realization.final_storage);
    }

    pub fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        self.final_storage = Arc::clone(&realization.final_storage);
    }

    pub fn update_dominating_cut(
        &mut self,
        cut: &cut::BendersCut,
        height: f64,
    ) {
        self.dominating_cut_id = cut.id;
        self.dominating_objective = height;
    }

    pub fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        subproblem: &mut subproblem::Subproblem,
    ) {
        let mut factors =
            Vec::<(usize, f64)>::with_capacity(self.get_dimension() + 1);
        factors.push((subproblem.accessors.alpha, 1.0));
        for (hydro_id, stored_volume) in
            subproblem.accessors.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -1.0 * cut.coefficients[hydro_id]));
        }
        subproblem.model.add_row(cut.rhs.., factors);
    }
}
