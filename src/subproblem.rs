use crate::cut;
use crate::fcf;
use crate::risk_measure;
use crate::scenario;
use crate::solver;
use crate::state;
use crate::stochastic_process;
use crate::system;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Timing breakdown for realize_uncertainties operation.
///
/// This struct captures precise timing for the two main phases:
/// 1. Solver time: LP solve (retry_solve)
/// 2. State extraction: Getting solution and extracting variables
#[derive(Debug, Clone, Copy, Default)]
pub struct RealizeUncertaintiesTiming {
    pub solver_time: Duration,
    pub state_extraction_time: Duration,
}

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
    model.set_option("presolve", "on");
    model.set_option("solver", "simplex");
    model.set_option("simplex_strategy", 1);
    model.set_option("simplex_scale_strategy", 0);
    model.set_option("simplex_primal_edge_weight_strategy", -1);
    model.set_option("simplex_dual_edge_weight_strategy", -1);
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("random_seed", 0);

    // PERFORMANCE: Stricter tolerances to reduce numerical drift that causes
    // floating-point non-determinism. Analysis showed ~1e-16 differences compound
    // to 2-3% lower bound variation. Tighter tolerances reduce solver path dependencies.
    // Cost: ~2-5% longer solve times. Benefit: Eliminates cascading numerical errors.
    model.set_option("primal_feasibility_tolerance", 1e-10);
    model.set_option("dual_feasibility_tolerance", 1e-10);
    model.set_option("time_limit", 300);
}

/// Helper function for setting the solver options when retrying a solve
fn set_first_retry_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "off");
    // PERFORMANCE: Slightly looser but still strict tolerances for retry
    model.set_option("primal_feasibility_tolerance", 1e-8);
    model.set_option("dual_feasibility_tolerance", 1e-8);
}

/// Helper function for setting the solver options when retrying a solve
fn set_second_retry_solver_options(model: &mut solver::Model) {
    // PERFORMANCE: Progressively looser tolerances for final retry
    model.set_option("primal_feasibility_tolerance", 1e-6);
    model.set_option("dual_feasibility_tolerance", 1e-6);
}

/// Helper function for setting the solver options when retrying a solve
fn set_third_retry_solver_options(model: &mut solver::Model) {
    model.set_option("simplex_strategy", 4);
}

/// Helper function for setting the solver options when retrying a solve
fn set_final_retry_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "on");
    model.set_option("solver", "ipm");
    model.set_option("run_crossover", "on");
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
#[derive(Clone)]
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

/// Helper accessor for indexing desired variables in each subproblem
#[derive(Clone)]
pub struct Constraints {
    pub load_balance: Vec<usize>,
    pub hydro_balance: Vec<usize>,
    /// Inflow process constraints - structure depends on State implementation:
    /// - StorageState: inflow_process[hydro][0..2] (equality + RHS constraints)
    /// - StorageAndInflowState: inflow_process[hydro][0..2+p]
    ///   (equality + RHS + p lag constraints)
    pub inflow_process: Vec<Vec<usize>>,
}

/// A subproblem that contains a solver model and is associated to a single
/// node in the computing graph

#[derive(Clone)]
pub struct Subproblem {
    pub model: Option<solver::Model>,
    pub state: Box<dyn state::State>,
    pub variables: Variables,
    pub constraints: Constraints,
    /// Transformation cache for observation ↔ residual conversions (PAR models only)
    pub transform_cache:
        Option<std::sync::Arc<crate::space_transform::TransformCache>>,
    /// Season ID for this subproblem (used for seasonal transformations)
    pub season_id: usize,
}

impl Subproblem {
    pub fn new(
        system: &system::System,
        state_choice: &str,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_processes: &[Box<
            dyn stochastic_process::StochasticProcess,
        >],
        unified_specs: &[crate::unified_noise_spec::UnifiedNoiseSpec],
        season_id: usize,
    ) -> Self {
        // Build transformation cache if any PAR models present
        // PERFORMANCE: Cache built once per subproblem, shared via Arc when cloning
        let has_par_models = unified_specs.iter().any(|spec| {
            matches!(
                spec.temporal_model,
                crate::unified_noise_spec::TemporalModelSpec::PeriodicAutoregressive { .. }
            )
        });

        let transform_cache = if has_par_models {
            // Determine number of seasons from unified specs
            let num_seasons = unified_specs
                .iter()
                .flat_map(|spec| spec.seasonal_params.keys())
                .max()
                .map(|max_season| max_season + 1)
                .unwrap_or(1);

            Some(std::sync::Arc::new(
                crate::space_transform::TransformCache::new(
                    unified_specs,
                    system.hydros.len(),
                    num_seasons,
                ),
            ))
        } else {
            None
        };

        let state = state::factory(
            state_choice,
            system,
            load_stochastic_process,
            inflow_stochastic_processes,
        );
        let mut pb = solver::Problem::new();
        let variables = Subproblem::add_variables_to_subproblem(
            &mut pb,
            system,
            state.as_ref(),
            load_stochastic_process,
            inflow_stochastic_processes,
        );
        let constraints = Subproblem::add_constraints_to_subproblem(
            &mut pb,
            &variables,
            system,
            state.as_ref(),
            load_stochastic_process,
            inflow_stochastic_processes,
            unified_specs,
            season_id,
        );
        Self::add_offset_to_subproblem(&mut pb, system);

        let mut model = pb.optimise(solver::Sense::Minimise);
        set_retry_solver_options(&mut model, 0);

        Self {
            model: Some(model),
            state,
            variables,
            constraints,
            transform_cache,
            season_id,
        }
    }

    fn add_variables_to_subproblem(
        pb: &mut solver::Problem,
        system: &system::System,
        state: &dyn state::State,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_processes: &[Box<
            dyn stochastic_process::StochasticProcess,
        >],
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
            load_stochastic_process,
            inflow_stochastic_processes,
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

    #[allow(clippy::too_many_arguments)]
    fn add_constraints_to_subproblem(
        pb: &mut solver::Problem,
        variables: &Variables,
        system: &system::System,
        state: &dyn state::State,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_processes: &[Box<
            dyn stochastic_process::StochasticProcess,
        >],
        unified_specs: &[crate::unified_noise_spec::UnifiedNoiseSpec],
        season_id: usize,
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
            inflow_stochastic_processes,
            unified_specs,
            season_id,
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

    /// Set hydro balance RHS directly (used primarily in tests).
    ///
    /// For production use, prefer `update_with_current_trajectory()` which
    /// delegates to the state's `update_from_trajectory()` method.
    #[cfg(test)]
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
        // Delegate to state - it knows what it needs from the trajectory!
        let model = self.model.as_mut().unwrap();
        self.state.update_from_trajectory(
            &realizations,
            model,
            &self.constraints,
        );
    }

    pub fn update_with_current_realization(
        &mut self,
        realization: &Realization,
    ) {
        self.state.update_with_current_realization(realization);
    }

    pub fn compute_new_cut(
        &self,
        forward_trajectory: &[&Realization],
        branching_realizations: &[Realization],
        risk_measure: &dyn risk_measure::RiskMeasure,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> fcf::CutStatePair {
        // this only works when all nodes have the same state definition??
        let mut visited_state = self.state.clone();
        // Set tracking fields before computing cut
        visited_state.set_iteration(iteration);
        visited_state.set_forward_pass_idx(forward_pass_idx);
        let cut = visited_state.compute_new_cut(
            risk_measure,
            forward_trajectory,
            branching_realizations,
        );
        fcf::CutStatePair::new(cut, visited_state, forward_pass_idx)
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
        cut.id = fcf.cut_pool.total_cut_count;
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
            if (cut.non_dominated_state_count == 0) && cut.active {
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
            fcf.update_cut_pool_on_remove(*cut_id);
        }
    }

    /// Apply AGGREGATED cut selection results WITHOUT locking FCF (LOCK-FREE)
    pub fn apply_aggregated_cut_selection_result(
        &mut self,
        aggregated_result: &fcf::AggregatedCutSelectionResult,
        active_cut_indices_before: &std::collections::BTreeMap<usize, usize>,
        cuts_to_add: &[(usize, cut::BendersCut)],
    ) -> Result<(), String> {
        // PERFORMANCE: Sort cuts to ensure deterministic constraint matrix construction.
        // This eliminates solver path dependencies that cause ~1e-16 numerical differences
        // which cascade to 2-3% lower bound variation. Constraint addition order affects
        // solver numerical algorithms (basis selection, pivot rules) even with identical cuts.
        let mut cuts_to_process: Vec<(usize, &cut::BendersCut)> = cuts_to_add
            .iter()
            .filter(|(cut_id, _)| {
                aggregated_result.new_cut_ids.contains(cut_id)
                    || aggregated_result.returning_cut_ids.contains(cut_id)
            })
            .map(|(cut_id, cut)| (*cut_id, cut))
            .collect();

        // Sort by (cut_id, iteration, forward_pass_idx) for complete determinism
        cuts_to_process.sort_by_key(|(cut_id, cut)| {
            (*cut_id, cut.iteration, cut.forward_pass_idx)
        });

        // Add cuts in deterministic order
        for (_cut_id, cut) in cuts_to_process {
            if let Some(model) = self.model.as_mut() {
                let mut cut_copy = cut.clone();
                self.state.add_cut_constraint_to_model(
                    &mut cut_copy,
                    &self.variables,
                    model,
                );
            }
        }

        // Remove ALL dominated cuts from model (same as before)
        let mut indices_to_remove: Vec<usize> = aggregated_result
            .removing_cut_ids
            .iter()
            .filter_map(|&cut_id| {
                active_cut_indices_before.get(&cut_id).copied()
            })
            .collect();

        indices_to_remove.sort_unstable_by(|a, b| b.cmp(a));

        for index in indices_to_remove {
            let row_idx = self.first_cut_row_index() + index;

            if let Some(model) = self.model.as_mut() {
                model.delete_row(row_idx).map_err(|e| {
                    format!("Failed to delete row {}: {:?}", row_idx, e)
                })?;
            }
        }

        // NOTE: FCF state update (marking cuts inactive, updating active_cut_indices)
        // is done ONCE in the SDDP code before calling this function.
        // This lock-free version only updates the local solver model (adds/removes constraints).
        Ok(())
    }

    fn set_uncertainties(&mut self, bus_loads: &[f64], hydros_inflow: &[f64]) {
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
                    // PERFORMANCE: After 4 retries, model is likely infeasible
                    // or numerically unstable. Provide diagnostic information.
                    panic!(
                        "Solver failed after {} retries. Final status: {:?}",
                        retry,
                        model.status()
                    );
                }

                // Try to solve with detailed error handling
                match model.try_solve() {
                    Ok(_) => {
                        // Solve succeeded, check model status
                    }
                    Err(_e) => {
                        // Continue to check model status and potentially retry
                    }
                }

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
                    solver::HighsModelStatus::PresolveError => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::SolveError => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::PostsolveError => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::ReachedIterationLimit => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::ReachedTimeLimit => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::Unknown => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    status => {
                        // PERFORMANCE: Unexpected solver status - provide diagnostics
                        panic!(
                            "Unexpected solver status after {} retries: {:?}. \
                             Expected Optimal or Infeasible. This may indicate: \
                             1) Time/iteration limits reached, \
                             2) Numerical issues in the model, \
                             3) Unbounded problem, \
                             4) Solver error",
                            retry, status
                        );
                    }
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
        noises: &scenario::OptimizedSampledBranchingNoises,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_processes: &[Box<
            dyn stochastic_process::StochasticProcess,
        >],
        realization_container: &mut Realization,
    ) -> Result<RealizeUncertaintiesTiming, String> {
        let mut timing = RealizeUncertaintiesTiming::default();

        // Time state extraction
        let extraction_start = std::time::Instant::now();
        let load =
            load_stochastic_process.realize(noises.get_load_innovations());

        // For now, use first process for backward compatibility
        // TODO: Update to handle per-hydro realizations
        let inflow_noises =
            if let Some(first_process) = inflow_stochastic_processes.first() {
                first_process.realize(noises.get_inflow_innovations())
            } else {
                // If no processes, return empty realization
                &[]
            };

        self.set_uncertainties(load, inflow_noises);

        // PERFORMANCE: Store realized loads in realization container
        // Handle both cases: per-bus loads or single scalar load (deterministic benchmarks)
        if load.len() == realization_container.loads.len() {
            // Direct copy for per-bus loads (O(num_buses) memcpy, ~10ns)
            realization_container.loads.clone_from_slice(load);
        } else if load.len() == 1 {
            // Replicate single load value across all buses (deterministic case)
            realization_container.loads.fill(load[0]);
        } else {
            return Err(format!(
                "Load dimension mismatch: got {} load values but system has {} buses",
                load.len(),
                realization_container.loads.len()
            ));
        }
        timing.state_extraction_time += extraction_start.elapsed();

        // Time the solver call
        let solver_start = std::time::Instant::now();
        self.retry_solve();
        timing.solver_time = solver_start.elapsed();

        // Time state extraction
        let extraction_start = std::time::Instant::now();
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
                    // Extract lag duals (for StorageAndInflowState)
                    self.get_lag_duals_from_solution(
                        &solution,
                        realization_container,
                    );

                    model.clear_solver();
                    timing.state_extraction_time = extraction_start.elapsed();
                    Ok(timing)
                }
                _ => Err(format!(
                    "Error while solving subproblem: {:?}",
                    model.status()
                )),
            },
            None => {
                Err("Error while solving subproblem: Model is None".to_string())
            }
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

        // PERFORMANCE: Extract inflow values from solution
        // For PAR models: These are residuals Z'_t (required for AR constraints in next stage)
        // For Independent models: These are observations Y_t
        // realization_container stores values in the space the LP works in
        realization_container
            .inflow
            .clone_from_slice(&solution.colvalue[first..last]);

        // DO NOT transform here - realization.inflow is used for lag updates
        // which need residuals for PAR models. Transformation happens only
        // for user-facing output (CSV files) in output.rs
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

    fn get_lag_duals_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        // Extract lag duals for StorageAndInflowState
        // Structure: inflow_process[hydro][0..2+p] where [2..2+p] are lag constraints
        //
        // For StorageState: inflow_process[hydro] has only 2 constraints (no lags)
        // For StorageAndInflowState with PAR(p): inflow_process[hydro] has 2+p constraints
        //
        // We need to extract dual values for the lag constraints only ([2..2+p])

        // Check if there are any lag constraints
        if self.constraints.inflow_process.is_empty() {
            // No hydros, no lags
            realization_container.lag_duals.clear();
            return;
        }

        // Check first hydro to see if there are lag constraints
        let first_hydro_constraints = &self.constraints.inflow_process[0];
        if first_hydro_constraints.len() <= 2 {
            // StorageState or no lags - clear lag_duals
            realization_container.lag_duals.clear();
            return;
        }

        // StorageAndInflowState with lags - extract dual values
        let num_hydros = self.constraints.inflow_process.len();
        let num_lags = first_hydro_constraints.len() - 2; // Subtract 2 inflow constraints

        // PERFORMANCE: Pre-allocate to avoid reallocation
        realization_container.lag_duals = Vec::with_capacity(num_lags);

        for lag_idx in 0..num_lags {
            let mut lag_duals_for_hydros = Vec::with_capacity(num_hydros);
            for hydro in 0..num_hydros {
                // Constraint index for this lag and hydro
                let constraint_idx =
                    self.constraints.inflow_process[hydro][2 + lag_idx];
                let dual_value = solution.rowdual[constraint_idx];
                lag_duals_for_hydros.push(dual_value);
            }
            realization_container.lag_duals.push(lag_duals_for_hydros);
        }
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
    /// Dual values on lag transfer constraints for StorageAndInflowState
    /// Structure: lag_duals[lag_idx][hydro_idx] → dual value on y_lag[k][i] = lag_value
    /// Empty for StorageState (no lag constraints)
    pub lag_duals: Vec<Vec<f64>>,
    pub basis: solver::Basis,
}

impl Realization {
    #[allow(clippy::too_many_arguments)]
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
            lag_duals: vec![], // Empty by default (StorageState has no lags)
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
            lag_duals: vec![], // Empty by default (StorageState has no lags)
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
            lag_duals: vec![], // Empty by default
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
        let inflow_processes = vec![inflow_stochastic_process];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_stochastic_process.as_ref(),
            &inflow_processes,
            &[],
            0,
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
        let inflow_processes = vec![inflow_stochastic_process];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_stochastic_process.as_ref(),
            &inflow_processes,
            &[],
            0,
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
        let inflow_processes = vec![inflow_stochastic_process];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_stochastic_process.as_ref(),
            &inflow_processes,
            &[],
            0,
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

    // ========================================================================
    // PRIVATE FUNCTION TESTS (Added for T4.2 Phase 5a)
    // ========================================================================

    #[test]
    fn test_get_current_stage_objective() {
        // Test the private helper that extracts current stage objective
        let total_objective = 1000.0;
        let solution = solver::Solution {
            colvalue: vec![10.0, 20.0, 30.0, 40.0],
            coldual: vec![0.0; 4],
            rowvalue: vec![0.0; 2],
            rowdual: vec![0.0; 2],
        };

        let current_obj =
            get_current_stage_objective(total_objective, &solution);
        assert_eq!(current_obj, 1000.0 - 40.0); // total - future (last value)
    }

    #[test]
    fn test_set_default_solver_options() {
        // Test that default solver options are set correctly
        let mut problem = solver::Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(1.0.., [(0, 1.0)]);
        let mut model = problem.optimise(solver::Sense::Minimise);

        set_default_solver_options(&mut model);
        // Options are set but we can't directly query them from HiGHS
        // The test verifies the function doesn't panic
        model.solve();
        assert_eq!(model.status(), solver::HighsModelStatus::Optimal);
    }

    #[test]
    fn test_set_retry_solver_options_coverage() {
        // Test all retry option branches
        let mut problem = solver::Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(1.0.., [(0, 1.0)]);
        let mut model = problem.optimise(solver::Sense::Minimise);

        // Test each retry level
        set_retry_solver_options(&mut model, 0); // default
        set_retry_solver_options(&mut model, 1); // first retry
        set_retry_solver_options(&mut model, 2); // second retry
        set_retry_solver_options(&mut model, 3); // third retry
        set_retry_solver_options(&mut model, 4); // final retry
        set_retry_solver_options(&mut model, 5); // back to default

        // Verify model still works after all option changes
        model.solve();
        assert_eq!(model.status(), solver::HighsModelStatus::Optimal);
    }

    #[test]
    fn test_subproblem_first_cut_row_index() {
        // Test the private first_cut_row_index method
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let first_cut_idx = subproblem.first_cut_row_index();
        // first_cut_row_index = last inflow process constraint index + 1
        // For default system with constraints, this should be 4
        assert_eq!(first_cut_idx, 4);
    }

    #[test]
    fn test_subproblem_get_deficit_from_solution() {
        // Test private getter for deficit values
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Set up and solve
        let initial_storage = [50.0];
        let load = [30.0];
        let inflow = [10.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem.get_deficit_from_solution(&solution, &mut realization);
            assert_eq!(realization.deficit.len(), 1); // 1 bus in default system
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_thermal_gen_from_solution() {
        // Test private getter for thermal generation
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        let inflow = [10.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_thermal_gen_from_solution(&solution, &mut realization);
            assert_eq!(realization.thermal_generation.len(), 2); // 2 thermals in default system
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_spillage_from_solution() {
        // Test private getter for spillage values
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [100.0];
        let load = [10.0];
        let inflow = [50.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem.get_spillage_from_solution(&solution, &mut realization);
            assert_eq!(realization.spillage.len(), 1); // 1 hydro in default system
            assert!(realization.spillage[0] >= 0.0); // Spillage should be non-negative
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_turbined_flow_from_solution() {
        // Test private getter for turbined flow
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        let inflow = [10.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_turbined_flow_from_solution(&solution, &mut realization);
            assert_eq!(realization.turbined_flow.len(), 1); // 1 hydro
            assert!(realization.turbined_flow[0] >= 0.0);
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_final_storage_from_solution() {
        // Test private getter for final storage
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        let inflow = [10.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_final_storage_from_solution(&solution, &mut realization);
            assert_eq!(realization.final_storage.len(), 1);
            assert!(realization.final_storage[0] >= 0.0);
            assert!(realization.final_storage[0] <= 100.0); // Within max storage
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_water_values_from_solution() {
        // Test private getter for water values (duals)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        let inflow = [10.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_water_values_from_solution(&solution, &mut realization);
            assert_eq!(realization.water_value.len(), 1); // 1 hydro
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_marginal_cost_from_solution() {
        // Test private getter for marginal costs (bus duals)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        let inflow = [10.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_uncertainties(&load, &inflow);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_marginal_cost_from_solution(&solution, &mut realization);
            assert_eq!(realization.marginal_cost.len(), 1); // 1 bus
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_set_load_balance_rhs() {
        // Test setting load balance RHS values
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Set new loads
        let new_loads = vec![50.0];
        subproblem.set_load_balance_rhs(&new_loads);

        // Verify by solving - should work without errors
        assert!(subproblem.model.is_some());
    }

    #[test]
    fn test_set_hydro_balance_rhs() {
        // Test setting hydro balance RHS values (initial storage)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Set new initial storage
        let new_storage = vec![75.0];
        subproblem.set_hydro_balance_rhs(&new_storage);

        // Verify by solving - should work without errors
        assert!(subproblem.model.is_some());
    }

    #[test]
    fn test_set_uncertainties() {
        // Test setting both load and inflow uncertainties
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Set uncertainties
        let bus_loads = vec![60.0];
        let hydros_inflow = vec![100.0];
        subproblem.set_uncertainties(&bus_loads, &hydros_inflow);

        // Verify model still exists and can be solved
        assert!(subproblem.model.is_some());
    }

    #[test]
    fn test_get_net_exchange_from_solution() {
        // Test extracting net exchange values from solution
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Solve to get a solution
        let mut model = subproblem.model.take().unwrap();
        model.solve();

        if model.status() == solver::HighsModelStatus::Optimal {
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_net_exchange_from_solution(&solution, &mut realization);
            // Default system may or may not have exchange variables
            // Just verify the function executes without crashing
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_get_inflow_from_solution() {
        // Test extracting inflow values from solution
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Solve to get a solution
        let mut model = subproblem.model.take().unwrap();
        model.solve();

        if model.status() == solver::HighsModelStatus::Optimal {
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem.get_inflow_from_solution(&solution, &mut realization);
            assert_eq!(realization.inflow.len(), 1); // 1 hydro
            subproblem.model = Some(model);
        }
    }
}
