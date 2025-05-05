use crate::risk_measure;
use crate::sddp;
use crate::utils;

pub struct State {
    hydro_storages: Vec<f64>,
    hydro_storage_duals: Vec<f64>,
}

impl State {
    pub fn new(num_hydros: usize) -> Self {
        Self {
            hydro_storages: Vec::<f64>::with_capacity(num_hydros),
            hydro_storage_duals: Vec::<f64>::with_capacity(num_hydros),
        }
    }

    pub fn get_hydro_storages(&self) -> &[f64] {
        self.hydro_storages.as_slice()
    }

    pub fn get_hydro_storage_duals(&self) -> &[f64] {
        self.hydro_storage_duals.as_slice()
    }

    pub fn set_hydro_storages(&mut self, hydro_storages: &[f64]) {
        self.hydro_storages.extend_from_slice(hydro_storages);
    }

    pub fn set_hydro_storage_duals(&mut self, hydro_storage_duals: &[f64]) {
        self.hydro_storage_duals
            .extend_from_slice(hydro_storage_duals);
    }

    pub fn compute_cut(
        &self,
        cut_id: usize,
        risk_measure: &risk_measure::RiskMeasure,
        forward_realization: &sddp::Realization,
        branching_realizations: &Vec<sddp::Realization>,
    ) -> sddp::BendersCut {
        let num_hydros =
            forward_realization.initial_state.get_hydro_storages().len();
        let mut coefficients = vec![0.0; num_hydros];
        let mut objective = 0.0;
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let probabilities = risk_measure.adjust_probabilities(&[0.0], &costs);
        for (index, realization) in branching_realizations.iter().enumerate() {
            for hydro_id in 0..num_hydros {
                coefficients[hydro_id] += probabilities[index]
                    * realization.final_state.get_hydro_storage_duals()
                        [hydro_id]
            }
            objective +=
                probabilities[index] * realization.total_stage_objective;
        }

        let cut_rhs = objective
            - utils::dot_product(
                &coefficients,
                &forward_realization.initial_state.get_hydro_storages(),
            );
        sddp::BendersCut::new(cut_id, coefficients, cut_rhs)
    }
}
