use crate::solver;
use crate::system;
use std::sync::Arc;

/// Helper function for removing the future cost term from the stage objective,
/// a.k.a the `alpha` term, or the epigraphical variable, assuming the objective
/// function is:
///
/// c^T x + `alpha`
fn get_current_stage_objective(
    total_stage_objective: f64,
    solution: &solver::Solution,
) -> f64 {
    let future_objective = solution.colvalue.last().unwrap();
    total_stage_objective - future_objective
}

/// Helper function for setting the same default solver options on
/// every solved problem.
fn set_default_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "off");
    model.set_option("solver", "simplex");
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("primal_feasibility_tolerance", 1e-7);
    model.set_option("dual_feasibility_tolerance", 1e-7);
    model.set_option("time_limit", 300);
}

/// Helper function for setting the solver options when retrying a solve
fn set_first_retry_solver_options(model: &mut solver::Model) {
    model.set_option("primal_feasibility_tolerance", 1e-6);
    model.set_option("dual_feasibility_tolerance", 1e-6);
}

/// Helper function for setting the solver options when retrying a solve
fn set_second_retry_solver_options(model: &mut solver::Model) {
    model.set_option("primal_feasibility_tolerance", 1e-5);
    model.set_option("dual_feasibility_tolerance", 1e-5);
}

/// Helper function for setting the solver options when retrying a solve
fn set_third_retry_solver_options(model: &mut solver::Model) {
    model.set_option("simplex_strategy", 4);
}

/// Helper function for setting the solver options when retrying a solve
fn set_final_retry_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "on");
    model.set_option("solver", "ipm");
    model.set_option("primal_feasibility_tolerance", 1e-7);
    model.set_option("dual_feasibility_tolerance", 1e-7);
}

/// Helper function for setting the solver options when retrying a solve
fn set_retry_solver_options(model: &mut solver::Model, retry: usize) {
    match retry {
        1 => set_first_retry_solver_options(model),
        2 => set_second_retry_solver_options(model),
        3 => set_third_retry_solver_options(model),
        4 => set_final_retry_solver_options(model),
        _ => set_default_solver_options(model),
    }
}

/// Helper accessor for indexing desired variables and constraints
/// in each subproblem
#[derive(Debug)]
pub struct Accessors {
    pub deficit: Vec<usize>,
    pub direct_exchange: Vec<usize>,
    pub reverse_exchange: Vec<usize>,
    pub thermal_gen: Vec<usize>,
    pub turbined_flow: Vec<usize>,
    pub spillage: Vec<usize>,
    pub stored_volume: Vec<usize>,
    pub alpha: usize,
    pub load_balance: Vec<usize>,
    pub hydro_balance: Vec<usize>,
}

/// A subproblem that contains a solver model and is associated to a single
/// node in the computing graph
#[derive(Debug)]
pub struct Subproblem {
    pub model: solver::Model,
    pub accessors: Accessors,
}

impl Subproblem {
    pub fn new(system: &system::System) -> Self {
        let mut pb = solver::Problem::new();

        // VARIABLES
        let deficit: Vec<usize> = system
            .buses
            .iter()
            .map(|bus| pb.add_column(bus.deficit_cost, 0.0..))
            .collect();
        let direct_exchange: Vec<usize> = system
            .lines
            .iter()
            .map(|line| {
                pb.add_column(line.exchange_penalty, 0.0..line.direct_capacity)
            })
            .collect();
        let reverse_exchange: Vec<usize> = system
            .lines
            .iter()
            .map(|line| {
                pb.add_column(line.exchange_penalty, 0.0..line.reverse_capacity)
            })
            .collect();
        let thermal_gen: Vec<usize> = system
            .thermals
            .iter()
            .map(|thermal| {
                pb.add_column(
                    thermal.cost,
                    0.0..(thermal.max_generation - thermal.min_generation),
                )
            })
            .collect();
        let turbined_flow: Vec<usize> = system
            .hydros
            .iter()
            .map(|hydro| {
                pb.add_column(
                    0.0,
                    hydro.min_turbined_flow..hydro.max_turbined_flow,
                )
            })
            .collect();
        let spillage: Vec<usize> = system
            .hydros
            .iter()
            .map(|hydro| pb.add_column(hydro.spillage_penalty, 0.0..))
            .collect();
        let stored_volume: Vec<usize> = system
            .hydros
            .iter()
            .map(|hydro| {
                pb.add_column(0.0, hydro.min_storage..hydro.max_storage)
            })
            .collect();

        let alpha = pb.add_column(1.0, 0.0..);

        // Adds load balance with 0.0 as RHS
        let mut load_balance: Vec<usize> = vec![0; system.meta.buses_count];
        for bus in system.buses.iter() {
            let mut factors = vec![(deficit[bus.id], 1.0)];
            for thermal_id in bus.thermal_ids.iter() {
                factors.push((thermal_gen[*thermal_id], 1.0));
            }
            for hydro_id in bus.hydro_ids.iter() {
                factors.push((
                    turbined_flow[*hydro_id],
                    system.hydros.get(*hydro_id).unwrap().productivity,
                ));
            }
            for line_id in bus.source_line_ids.iter() {
                factors.push((reverse_exchange[*line_id], 1.0));
                factors.push((direct_exchange[*line_id], -1.0));
            }
            for line_id in bus.target_line_ids.iter() {
                factors.push((direct_exchange[*line_id], 1.0));
                factors.push((reverse_exchange[*line_id], -1.0));
            }
            load_balance[bus.id] = pb.add_row(0.0..0.0, &factors);
        }

        // Adds hydro balance with 0.0 as RHS
        let mut hydro_balance: Vec<usize> = vec![0; system.meta.hydros_count];
        for hydro in system.hydros.iter() {
            let mut factors: Vec<(usize, f64)> = vec![
                (stored_volume[hydro.id], 1.0),
                (turbined_flow[hydro.id], 1.0),
                (spillage[hydro.id], 1.0),
            ];
            for upstream_hydro_id in hydro.upstream_hydro_ids.iter() {
                factors.push((turbined_flow[*upstream_hydro_id], -1.0));
                factors.push((spillage[*upstream_hydro_id], -1.0));
            }
            hydro_balance[hydro.id] = pb.add_row(0.0..0.0, &factors);
        }

        // evaluates problem offset from minimal thermal generation
        let mut offset = 0.0;
        for thermal in system.thermals.iter() {
            offset += thermal.cost * thermal.min_generation;
        }
        pb.offset = offset;

        let mut model = pb.optimise(solver::Sense::Minimise);
        set_retry_solver_options(&mut model, 0);

        // for making better allocation
        let accessors = Accessors {
            deficit,
            direct_exchange,
            reverse_exchange,
            thermal_gen,
            turbined_flow,
            spillage,
            stored_volume,
            alpha,
            load_balance,
            hydro_balance,
        };

        Subproblem { model, accessors }
    }

    fn set_load_balance_rhs(&mut self, loads: &[f64]) {
        for (index, row) in self.accessors.load_balance.iter().enumerate() {
            self.model
                .change_rows_bounds(*row, loads[index], loads[index]);
        }
    }

    fn set_hydro_balance_rhs(
        &mut self,
        inflows: &[f64],
        initial_storages: &[f64],
    ) {
        let mut rhs: Vec<f64> = vec![0.0; inflows.len()];
        for i in 0..rhs.len() {
            rhs[i] = inflows[i] + initial_storages[i];
        }
        for (index, row) in self.accessors.hydro_balance.iter().enumerate() {
            self.model.change_rows_bounds(*row, rhs[index], rhs[index]);
        }
    }

    fn set_uncertainties<'a>(
        &mut self,
        initial_storage: &[f64],
        bus_loads: &[f64],
        hydros_inflow: &[f64],
    ) {
        self.set_load_balance_rhs(bus_loads);
        self.set_hydro_balance_rhs(hydros_inflow, initial_storage);
    }

    fn retry_solve(&mut self) {
        let mut retry: usize = 0;
        loop {
            if retry > 4 {
                panic!("Error while solving model");
            }
            self.model.solve();

            match self.model.status() {
                solver::HighsModelStatus::Optimal => {
                    if retry != 0 {
                        set_default_solver_options(&mut self.model);
                    }
                    return;
                }
                solver::HighsModelStatus::Infeasible => {
                    retry += 1;
                    set_retry_solver_options(&mut self.model, retry);
                }
                _ => panic!("Error while solving model"),
            }
        }
    }

    pub fn first_cut_row_index(&self) -> usize {
        self.accessors.hydro_balance.last().unwrap() + 1
    }

    pub fn realize_uncertainties(
        &mut self,
        initial_storage: &[f64],
        load: &[f64],
        inflow: &[f64],
    ) -> Realization {
        self.set_uncertainties(initial_storage, load, inflow);

        self.retry_solve();

        match self.model.status() {
            solver::HighsModelStatus::Optimal => {
                let mut solution = self.model.get_solution();
                self.slice_solution_rows_to_problem_constraints(&mut solution);
                let basis = self.model.get_basis();
                let total_stage_objective = self.model.get_objective_value();
                let current_stage_objective = get_current_stage_objective(
                    total_stage_objective,
                    &solution,
                );
                let deficit = self.get_deficit_from_solution(&solution);
                let direct_exchange =
                    self.get_direct_exchange_from_solution(&solution);
                let reverse_exchange =
                    self.get_reverse_exchange_from_solution(&solution);
                // evals net exchange
                let exchange = direct_exchange
                    .iter()
                    .enumerate()
                    .map(|(i, e)| e - reverse_exchange[i])
                    .collect();
                let thermal_generation =
                    self.get_thermal_gen_from_solution(&solution);
                let final_storage =
                    self.get_final_storage_from_solution(&solution);
                let turbined_flow =
                    self.get_turbined_flow_from_solution(&solution);
                let spillage = self.get_spillage_from_solution(&solution);
                let water_value =
                    self.get_water_values_from_solution(&solution);
                let marginal_cost: Vec<f64> =
                    self.get_marginal_cost_from_solution(&solution);

                self.model.clear_solver();

                return Realization::new(
                    load.to_vec(),
                    deficit,
                    exchange,
                    inflow.to_vec(),
                    turbined_flow,
                    spillage,
                    thermal_generation,
                    Arc::new(water_value),
                    marginal_cost,
                    current_stage_objective,
                    total_stage_objective,
                    Arc::new(initial_storage.to_vec()),
                    Arc::new(final_storage),
                    basis,
                );
            }
            _ => panic!("Error while solving subproblem"),
        }
    }

    pub fn get_deficit_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.deficit.first().unwrap();
        let last = *self.accessors.deficit.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    pub fn get_direct_exchange_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        match self.accessors.direct_exchange.is_empty() {
            true => vec![],
            false => {
                let first = *self.accessors.direct_exchange.first().unwrap();
                let last = *self.accessors.direct_exchange.last().unwrap() + 1;
                solution.colvalue[first..last].to_vec()
            }
        }
    }

    pub fn get_reverse_exchange_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        match self.accessors.reverse_exchange.is_empty() {
            true => vec![],
            false => {
                let first = *self.accessors.reverse_exchange.first().unwrap();
                let last = *self.accessors.reverse_exchange.last().unwrap() + 1;
                solution.colvalue[first..last].to_vec()
            }
        }
    }

    pub fn get_thermal_gen_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        match self.accessors.thermal_gen.is_empty() {
            true => vec![],
            false => {
                let first = *self.accessors.thermal_gen.first().unwrap();
                let last = *self.accessors.thermal_gen.last().unwrap() + 1;
                solution.colvalue[first..last].to_vec()
            }
        }
    }

    pub fn get_spillage_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.spillage.first().unwrap();
        let last = *self.accessors.spillage.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    pub fn get_turbined_flow_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.turbined_flow.first().unwrap();
        let last = *self.accessors.turbined_flow.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    pub fn get_final_storage_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.stored_volume.first().unwrap();
        let last = *self.accessors.stored_volume.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    pub fn get_water_values_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.hydro_balance.first().unwrap();
        let last = *self.accessors.hydro_balance.last().unwrap() + 1;
        solution.rowdual[first..last].to_vec()
    }

    pub fn get_marginal_cost_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.load_balance.first().unwrap();
        let last = *self.accessors.load_balance.last().unwrap() + 1;
        solution.rowdual[first..last].to_vec()
    }

    pub fn slice_solution_rows_to_problem_constraints(
        &self,
        solution: &mut solver::Solution,
    ) {
        let end = *self.accessors.hydro_balance.last().unwrap() + 1;
        solution.rowvalue.truncate(end);
        solution.rowdual.truncate(end);
    }
}

pub struct Realization {
    pub loads: Vec<f64>,
    pub deficit: Vec<f64>,
    pub exchange: Vec<f64>,
    pub inflow: Vec<f64>,
    pub turbined_flow: Vec<f64>,
    pub spillage: Vec<f64>,
    pub thermal_generation: Vec<f64>,
    pub water_value: Arc<Vec<f64>>,
    pub marginal_cost: Vec<f64>,
    pub current_stage_objective: f64,
    pub total_stage_objective: f64,
    pub initial_storage: Arc<Vec<f64>>,
    pub final_storage: Arc<Vec<f64>>,
    pub basis: solver::Basis,
}

impl Realization {
    pub fn new(
        loads: Vec<f64>,
        deficit: Vec<f64>,
        exchange: Vec<f64>,
        inflow: Vec<f64>,
        turbined_flow: Vec<f64>,
        spillage: Vec<f64>,
        thermal_generation: Vec<f64>,
        water_value: Arc<Vec<f64>>,
        marginal_cost: Vec<f64>,
        current_stage_objective: f64,
        total_stage_objective: f64,
        initial_storage: Arc<Vec<f64>>,
        final_storage: Arc<Vec<f64>>,
        basis: solver::Basis,
    ) -> Self {
        Self {
            loads,
            deficit,
            exchange,
            inflow,
            turbined_flow,
            spillage,
            thermal_generation,
            water_value,
            marginal_cost,
            current_stage_objective,
            total_stage_objective,
            initial_storage,
            final_storage,
            basis,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_subproblem_with_default_system() {
        let system = system::System::default();
        let subproblem = Subproblem::new(&system);
        assert_eq!(subproblem.accessors.deficit.len(), 1);
        assert_eq!(subproblem.accessors.direct_exchange.len(), 0);
        assert_eq!(subproblem.accessors.reverse_exchange.len(), 0);
        assert_eq!(subproblem.accessors.thermal_gen.len(), 2);
        assert_eq!(subproblem.accessors.turbined_flow.len(), 1);
        assert_eq!(subproblem.accessors.spillage.len(), 1);
        assert_eq!(subproblem.accessors.stored_volume.len(), 1);
    }

    #[test]
    fn test_solve_subproblem_with_default_system() {
        let system = system::System::default();
        let mut subproblem = Subproblem::new(&system);
        let inflow = [0.0];
        let initial_storage = [83.333];
        let load = [50.0];
        subproblem.set_load_balance_rhs(&load);
        subproblem.set_hydro_balance_rhs(&inflow, &initial_storage);

        subproblem.model.solve();
        assert_eq!(
            subproblem.model.status(),
            solver::HighsModelStatus::Optimal
        );
    }

    #[test]
    fn test_get_solution_cost_with_default_system() {
        let system = system::System::default();
        let mut subproblem = Subproblem::new(&system);
        let inflow = [0.0];
        let initial_storage = [23.333];
        let load = [50.0];
        subproblem.set_load_balance_rhs(&load);
        subproblem.set_hydro_balance_rhs(&inflow, &initial_storage);

        subproblem.model.solve();
        assert_eq!(subproblem.model.get_objective_value(), 191.67000000000002);
    }
}
