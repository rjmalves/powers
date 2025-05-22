use crate::fcf;
use crate::risk_measure;
use crate::scenario;
use crate::solver;
use crate::state;
use crate::stochastic_process;
use crate::system;
use std::sync::{Arc, Mutex};

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

/// Helper accessor for indexing desired variables in each subproblem
pub struct Variables {
    pub deficit: Vec<usize>,
    pub direct_exchange: Vec<usize>,
    pub reverse_exchange: Vec<usize>,
    pub thermal_gen: Vec<usize>,
    pub turbined_flow: Vec<usize>,
    pub spillage: Vec<usize>,
    pub stored_volume: Vec<usize>,
    pub inflow: Vec<usize>,
    pub inflow_process: Vec<Vec<usize>>,
    pub alpha: usize,
}

impl Variables {
    pub fn with_capacity(system: &system::System) -> Self {
        Self {
            deficit: Vec::<usize>::with_capacity(system.meta.buses_count),
            direct_exchange: Vec::<usize>::with_capacity(
                system.meta.lines_count,
            ),
            reverse_exchange: Vec::<usize>::with_capacity(
                system.meta.lines_count,
            ),
            thermal_gen: Vec::<usize>::with_capacity(
                system.meta.thermals_count,
            ),
            turbined_flow: Vec::<usize>::with_capacity(system.meta.buses_count),
            spillage: Vec::<usize>::with_capacity(system.meta.hydros_count),
            stored_volume: Vec::<usize>::with_capacity(
                system.meta.hydros_count,
            ),
            inflow: Vec::<usize>::with_capacity(system.meta.hydros_count),
            inflow_process: Vec::<Vec<usize>>::with_capacity(
                system.meta.hydros_count,
            ),
            alpha: 0,
        }
    }
}

/// Helper accessor for indexing desired variables in each subproblem
pub struct Constraints {
    pub load_balance: Vec<usize>,
    pub hydro_balance: Vec<usize>,
    pub inflow_process: Vec<Vec<usize>>,
}

impl Constraints {
    pub fn with_capacity(system: &system::System) -> Self {
        Self {
            load_balance: Vec::<usize>::with_capacity(system.meta.buses_count),
            hydro_balance: Vec::<usize>::with_capacity(
                system.meta.hydros_count,
            ),
            inflow_process: Vec::<Vec<usize>>::with_capacity(
                system.meta.hydros_count,
            ),
        }
    }
}

/// A subproblem that contains a solver model and is associated to a single
/// node in the computing graph
pub struct Subproblem {
    pub model: Option<solver::Model>,
    pub state: Box<dyn state::State>,
    pub variables: Variables,
    pub constraints: Constraints,
}

impl Subproblem {
    pub fn new(
        system: &system::System,
        state_choice: &str,
        load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Self {
        let state = state::factory(
            state_choice,
            system,
            load_stochastic_process,
            inflow_stochastic_process,
        );
        let mut pb = solver::Problem::new();
        let variables = Subproblem::add_variables_to_subproblem(
            &mut pb,
            system,
            &state,
            load_stochastic_process,
            inflow_stochastic_process,
        );
        let constraints = Subproblem::add_constraints_to_subproblem(
            &mut pb,
            &variables,
            system,
            &state,
            load_stochastic_process,
            inflow_stochastic_process,
        );
        Self::add_offset_to_subproblem(&mut pb, system);

        let mut model = pb.optimise(solver::Sense::Minimise);
        set_retry_solver_options(&mut model, 0);

        Self {
            model: Some(model),
            state,
            variables,
            constraints,
        }
    }

    pub fn with_capacity(
        system: &system::System,
        state_choice: &str,
        load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Self {
        let state = state::factory(
            state_choice,
            system,
            load_stochastic_process,
            inflow_stochastic_process,
        );

        Self {
            model: None,
            state,
            variables: Variables::with_capacity(system),
            constraints: Constraints::with_capacity(system),
        }
    }

    fn add_variables_to_subproblem(
        pb: &mut solver::Problem,
        system: &system::System,
        state: &Box<dyn state::State>,
        load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Variables {
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
        let inflow: Vec<usize> = system
            .hydros
            .iter()
            .map(|_hydro| pb.add_column(0.0, 0.0..))
            .collect();

        // Adds inflow as variables, bounded at 0, which will be fixed in runtime
        let inflow_process = state.add_variables_to_subproblem(
            pb,
            &load_stochastic_process,
            &inflow_stochastic_process,
        );

        let alpha = pb.add_column(1.0, 0.0..);

        Variables {
            deficit,
            direct_exchange,
            reverse_exchange,
            thermal_gen,
            turbined_flow,
            spillage,
            stored_volume,
            inflow,
            inflow_process,
            alpha,
        }
    }

    fn add_constraints_to_subproblem(
        pb: &mut solver::Problem,
        variables: &Variables,
        system: &system::System,
        state: &Box<dyn state::State>,
        load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
    ) -> Constraints {
        // Adds load balance with 0.0 as RHS
        let mut load_balance: Vec<usize> = vec![0; system.meta.buses_count];
        for bus in system.buses.iter() {
            let mut factors = vec![(variables.deficit[bus.id], 1.0)];
            for thermal_id in bus.thermal_ids.iter() {
                factors.push((variables.thermal_gen[*thermal_id], 1.0));
            }
            for hydro_id in bus.hydro_ids.iter() {
                factors.push((
                    variables.turbined_flow[*hydro_id],
                    system.hydros.get(*hydro_id).unwrap().productivity,
                ));
            }
            for line_id in bus.source_line_ids.iter() {
                factors.push((variables.reverse_exchange[*line_id], 1.0));
                factors.push((variables.direct_exchange[*line_id], -1.0));
            }
            for line_id in bus.target_line_ids.iter() {
                factors.push((variables.direct_exchange[*line_id], 1.0));
                factors.push((variables.reverse_exchange[*line_id], -1.0));
            }
            load_balance[bus.id] = pb.add_row(0.0..0.0, &factors);
        }

        // Adds hydro balance with 0.0 as RHS
        let mut hydro_balance: Vec<usize> = vec![0; system.meta.hydros_count];
        for hydro in system.hydros.iter() {
            let mut factors: Vec<(usize, f64)> = vec![
                (variables.stored_volume[hydro.id], 1.0),
                (variables.turbined_flow[hydro.id], 1.0),
                (variables.spillage[hydro.id], 1.0),
                (variables.inflow[hydro.id], -1.0),
            ];
            for upstream_hydro_id in hydro.upstream_hydro_ids.iter() {
                factors
                    .push((variables.turbined_flow[*upstream_hydro_id], -1.0));
                factors.push((variables.spillage[*upstream_hydro_id], -1.0));
            }
            hydro_balance[hydro.id] = pb.add_row(0.0..0.0, &factors);
        }

        // Adds inflow process as variables, bounded at 0, which will be fixed in runtime
        let inflow_process = state.add_constraints_to_subproblem(
            pb,
            variables,
            load_stochastic_process,
            inflow_stochastic_process,
        );

        Constraints {
            load_balance,
            hydro_balance,
            inflow_process,
        }
    }

    fn add_offset_to_subproblem(
        pb: &mut solver::Problem,
        system: &system::System,
    ) {
        let mut offset = 0.0;
        for thermal in system.thermals.iter() {
            offset += thermal.cost * thermal.min_generation;
        }
        pb.offset = offset;
    }

    fn set_load_balance_rhs(&mut self, loads: &[f64]) {
        if let Some(model) = self.model.as_mut() {
            for (index, row) in self.constraints.load_balance.iter().enumerate()
            {
                model.change_rows_bounds(*row, loads[index], loads[index]);
            }
        }
    }

    fn set_hydro_balance_rhs(&mut self, initial_storages: &[f64]) {
        if let Some(model) = self.model.as_mut() {
            for (index, row) in
                self.constraints.hydro_balance.iter().enumerate()
            {
                model.change_rows_bounds(
                    *row,
                    initial_storages[index],
                    initial_storages[index],
                );
            }
        }
    }

    pub fn update_with_current_trajectory(
        &mut self,
        realizations: Vec<&Realization>,
    ) {
        let realization = realizations.last().unwrap();
        self.set_hydro_balance_rhs(&realization.final_storage);
    }

    pub fn update_with_current_realization(
        &mut self,
        realization: &Realization,
    ) {
        self.state.update_with_current_realization(realization);
    }

    pub fn compute_new_cut(
        &self,
        cut_id: usize,
        forward_trajectory: &[&Realization],
        branching_realizations: &Vec<Realization>,
        risk_measure: &Box<dyn risk_measure::RiskMeasure>,
    ) -> fcf::CutStatePair {
        // this only works when all nodes have the same state definition??
        let mut visited_state = self.state.clone();
        let cut = visited_state.compute_new_cut(
            cut_id,
            risk_measure,
            forward_trajectory,
            branching_realizations,
        );
        fcf::CutStatePair::new(cut, visited_state)
    }

    pub fn add_cut_and_evaluate_cut_selection(
        &mut self,
        cut_state_pair: fcf::CutStatePair,
        future_cost_function: Arc<Mutex<fcf::FutureCostFunction>>,
    ) {
        let mut cut = cut_state_pair.cut;
        let mut visited_state = cut_state_pair.state;

        if let Some(model) = self.model.as_mut() {
            self.state.add_cut_constraint_to_model(
                &mut cut,
                &self.variables,
                model,
            );
        }
        let mut fcf = future_cost_function.lock().unwrap();
        fcf.update_cut_pool_on_add(cut.id);
        fcf.eval_new_cut_domination(&mut cut);

        fcf.add_cut(cut);

        // Obtains returning cut ids, based on cut selection
        let returning_cut_ids =
            fcf.update_old_cuts_domination(&mut visited_state);

        fcf.add_state(visited_state);

        // Obtains removing cut ids, based on cut selection
        let mut removing_cut_ids = Vec::<usize>::new();
        for cut in fcf.cut_pool.pool.iter_mut() {
            if (cut.non_dominated_state_count <= 0) && cut.active {
                removing_cut_ids.push(cut.id);
            }
        }

        // Returns cuts to model
        for cut_id in returning_cut_ids.iter() {
            let cut = fcf.cut_pool.pool.get_mut(*cut_id).unwrap();
            if let Some(model) = self.model.as_mut() {
                self.state.add_cut_constraint_to_model(
                    cut,
                    &self.variables,
                    model,
                );
            }
            fcf.update_cut_pool_on_return(*cut_id);
        }

        // Removes cuts from model
        for cut_id in removing_cut_ids.iter() {
            let cut_index = fcf.get_active_cut_index_by_id(*cut_id);
            let row_index = self.first_cut_row_index() + cut_index;
            if let Some(model) = self.model.as_mut() {
                model.delete_row(row_index).unwrap();
            }
            fcf.update_cut_pool_on_remove(*cut_id, cut_index);
        }
    }

    fn set_uncertainties<'a>(
        &mut self,
        bus_loads: &[f64],
        hydros_inflow: &[f64],
    ) {
        self.set_load_balance_rhs(bus_loads);
        if let Some(model) = self.model.as_mut() {
            self.state.set_inflows_in_subproblem(
                model,
                &self.constraints,
                hydros_inflow,
            );
        }
    }

    fn retry_solve(&mut self) {
        let mut retry: usize = 0;
        if let Some(model) = self.model.as_mut() {
            loop {
                if retry > 4 {
                    panic!("Error while solving model");
                }
                model.solve();

                match model.status() {
                    solver::HighsModelStatus::Optimal => {
                        if retry != 0 {
                            set_default_solver_options(model);
                        }
                        return;
                    }
                    solver::HighsModelStatus::Infeasible => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    _ => panic!("Error while solving model"),
                }
            }
        }
    }

    fn first_cut_row_index(&self) -> usize {
        self.constraints
            .inflow_process
            .last()
            .unwrap()
            .last()
            .unwrap()
            + 1
    }

    pub fn realize_uncertainties(
        &mut self,
        noises: &scenario::SampledBranchingNoises,
        load_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        inflow_stochastic_process: &Box<
            dyn stochastic_process::StochasticProcess,
        >,
        realization_container: &mut Realization,
    ) {
        let load = load_stochastic_process.realize(noises.get_load_noises());
        let inflow_noises =
            inflow_stochastic_process.realize(noises.get_inflow_noises());

        self.set_uncertainties(load, inflow_noises);

        self.retry_solve();

        match &self.model {
            Some(model) => match model.status() {
                solver::HighsModelStatus::Optimal => {
                    let mut solution = model.get_solution();
                    self.slice_solution_rows_to_problem_constraints(
                        &mut solution,
                    );

                    // basis
                    realization_container.basis.clone_from(&model.get_basis());

                    // costs
                    realization_container.total_stage_objective =
                        model.get_objective_value();
                    realization_container.current_stage_objective =
                        get_current_stage_objective(
                            realization_container.total_stage_objective,
                            &solution,
                        );

                    // bus results
                    self.get_deficit_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_marginal_cost_from_solution(
                        &solution,
                        realization_container,
                    );
                    // line results
                    self.get_net_exchange_from_solution(
                        &solution,
                        realization_container,
                    );
                    // thermal results
                    self.get_thermal_gen_from_solution(
                        &solution,
                        realization_container,
                    );
                    // hydro results
                    self.get_inflow_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_final_storage_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_turbined_flow_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_spillage_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_water_values_from_solution(
                        &solution,
                        realization_container,
                    );

                    model.clear_solver();
                }
                _ => panic!("Error while solving subproblem"),
            },
            None => panic!("Error while solving subproblem"),
        }
    }

    fn get_deficit_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.deficit.first().unwrap();
        let last = *self.variables.deficit.last().unwrap() + 1;
        realization_container
            .deficit
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_net_exchange_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        if !self.variables.direct_exchange.is_empty() {
            let direct_first = *self.variables.direct_exchange.first().unwrap();
            let direct_last =
                *self.variables.direct_exchange.last().unwrap() + 1;
            let reverse_first =
                *self.variables.reverse_exchange.first().unwrap();
            let reverse_last =
                *self.variables.reverse_exchange.last().unwrap() + 1;
            realization_container.exchange.clone_from_slice(
                &solution.colvalue[direct_first..direct_last],
            );
            realization_container
                .exchange
                .iter_mut()
                .zip(&solution.colvalue[reverse_first..reverse_last])
                .for_each(|(direct, reverse)| *direct -= *reverse);
        }
    }

    fn get_thermal_gen_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        if !self.variables.thermal_gen.is_empty() {
            let first = *self.variables.thermal_gen.first().unwrap();
            let last = *self.variables.thermal_gen.last().unwrap() + 1;
            realization_container
                .thermal_generation
                .clone_from_slice(&solution.colvalue[first..last]);
        }
    }

    fn get_spillage_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.spillage.first().unwrap();
        let last = *self.variables.spillage.last().unwrap() + 1;
        realization_container
            .spillage
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_turbined_flow_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.turbined_flow.first().unwrap();
        let last = *self.variables.turbined_flow.last().unwrap() + 1;
        realization_container
            .turbined_flow
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_final_storage_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.stored_volume.first().unwrap();
        let last = *self.variables.stored_volume.last().unwrap() + 1;
        realization_container
            .final_storage
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_inflow_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.inflow.first().unwrap();
        let last = *self.variables.inflow.last().unwrap() + 1;
        realization_container
            .inflow
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_water_values_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.constraints.hydro_balance.first().unwrap();
        let last = *self.constraints.hydro_balance.last().unwrap() + 1;
        realization_container
            .water_value
            .clone_from_slice(&solution.rowdual[first..last]);
    }

    fn get_marginal_cost_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.constraints.load_balance.first().unwrap();
        let last = *self.constraints.load_balance.last().unwrap() + 1;
        realization_container
            .marginal_cost
            .clone_from_slice(&solution.rowdual[first..last]);
    }

    fn slice_solution_rows_to_problem_constraints(
        &self,
        solution: &mut solver::Solution,
    ) {
        let end = *self
            .constraints
            .inflow_process
            .last()
            .unwrap()
            .last()
            .unwrap()
            + 1;
        solution.rowvalue.truncate(end);
        solution.rowdual.truncate(end);
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum StudyPeriodKind {
    PreStudy,
    Study,
    PostStudy,
}

#[derive(Debug, Clone)]
pub struct Realization {
    pub kind: StudyPeriodKind,
    pub loads: Vec<f64>,
    pub deficit: Vec<f64>,
    pub exchange: Vec<f64>,
    pub inflow: Vec<f64>,
    pub turbined_flow: Vec<f64>,
    pub spillage: Vec<f64>,
    pub thermal_generation: Vec<f64>,
    pub water_value: Vec<f64>,
    pub marginal_cost: Vec<f64>,
    pub current_stage_objective: f64,
    pub total_stage_objective: f64,
    pub final_storage: Vec<f64>,
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
        water_value: Vec<f64>,
        marginal_cost: Vec<f64>,
        current_stage_objective: f64,
        total_stage_objective: f64,
        final_storage: Vec<f64>,
        basis: solver::Basis,
    ) -> Self {
        Self {
            kind: StudyPeriodKind::Study,
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
            final_storage,
            basis,
        }
    }

    pub fn with_capacity(
        kind: &StudyPeriodKind,
        system: &system::System,
    ) -> Self {
        Self {
            kind: kind.clone(),
            loads: vec![0.0; system.meta.buses_count],
            deficit: vec![0.0; system.meta.buses_count],
            exchange: vec![0.0; system.meta.lines_count],
            inflow: vec![0.0; system.meta.hydros_count],
            turbined_flow: vec![0.0; system.meta.hydros_count],
            spillage: vec![0.0; system.meta.hydros_count],
            thermal_generation: vec![0.0; system.meta.thermals_count],
            water_value: vec![0.0; system.meta.hydros_count],
            marginal_cost: vec![0.0; system.meta.buses_count],
            current_stage_objective: 0.0,
            total_stage_objective: 0.0,
            final_storage: vec![0.0; system.meta.hydros_count],
            basis: solver::Basis::new(),
        }
    }
}

impl Default for Realization {
    fn default() -> Self {
        Self {
            kind: StudyPeriodKind::Study,
            loads: vec![],
            deficit: vec![],
            exchange: vec![],
            inflow: vec![],
            turbined_flow: vec![],
            spillage: vec![],
            thermal_generation: vec![],
            water_value: vec![],
            marginal_cost: vec![],
            current_stage_objective: 0.0,
            total_stage_objective: 0.0,
            final_storage: vec![],
            basis: solver::Basis::new(),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_subproblem_with_default_system() {
        let system = system::System::default();
        let load_stochastic_process = stochastic_process::factory("naive");
        let inflow_stochastic_process = stochastic_process::factory("naive");
        let subproblem = Subproblem::new(
            &system,
            "storage",
            &load_stochastic_process,
            &inflow_stochastic_process,
        );
        assert_eq!(subproblem.variables.deficit.len(), 1);
        assert_eq!(subproblem.variables.direct_exchange.len(), 0);
        assert_eq!(subproblem.variables.reverse_exchange.len(), 0);
        assert_eq!(subproblem.variables.thermal_gen.len(), 2);
        assert_eq!(subproblem.variables.turbined_flow.len(), 1);
        assert_eq!(subproblem.variables.spillage.len(), 1);
        assert_eq!(subproblem.variables.stored_volume.len(), 1);
        assert_eq!(subproblem.variables.inflow.len(), 1);
    }

    #[test]
    fn test_solve_subproblem_with_default_system() {
        let system = system::System::default();
        let load_stochastic_process = stochastic_process::factory("naive");
        let inflow_stochastic_process = stochastic_process::factory("naive");
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            &load_stochastic_process,
            &inflow_stochastic_process,
        );
        let inflow = [0.0];
        let initial_storage = [83.333];
        let load = [50.0];

        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model {
            model.solve();
            assert_eq!(model.status(), solver::HighsModelStatus::Optimal);
        }
    }

    #[test]
    fn test_get_solution_cost_with_default_system() {
        let system = system::System::default();
        let load_stochastic_process = stochastic_process::factory("naive");
        let inflow_stochastic_process = stochastic_process::factory("naive");
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            &load_stochastic_process,
            &inflow_stochastic_process,
        );
        let inflow = [0.0];
        let initial_storage = [23.333];
        let load = [50.0];

        subproblem.set_hydro_balance_rhs(&initial_storage);

        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model {
            model.solve();
            assert_eq!(model.get_objective_value(), 191.67000000000002);
        }
    }
}
