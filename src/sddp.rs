//! Implementation of the Stochastic Dual Dynamic Programming (SDDP)
//! algorithm for the hydrothermal dispatch problem. In exchange for
//! the simplified power system and state definition, some "smart"
//! optimizations and features are already considered in this code.
//!
//! The underlying power system is modeled with only four entities:
//! - Buses
//! - Lines
//! - Thermals
//! - Hydros
//!
//! Some considerations about the implementation:
//!
//! 1. Only hydro storages are considered as state variables.
//! 2. No memory management was made ready for parallelism (no locks and mutexes)
//! 3. Only risk-neutral policy evaluation is supported (no risk-aversion)
//! 4. An exact cut selection strategy (inspired in SDDP.jl) is implemented
//! 5. Only the "single-cut" (average cut) variant of the algorithm is supported.
//!
//! The only external dependencies are:
//!
//! 1. Random number generation and distribution sampling from rand* crates
//! 2. Low-level C-bindings from the highs-sys crate
//! 3. JSON and CSV serializers from the serde, serde_json and csv crates

use crate::graph;
use crate::solver;
use rand::prelude::*;
use rand_distr::{LogNormal, Uniform};
use rand_xoshiro::Xoshiro256Plus;
use std::f64;
use std::sync::Arc;
use std::time::{Duration, Instant};

// TODO - general optimizations
// 1. Pre-allocate everywhere when the total size of the containers
// is known, in repacement to calling push! (or init vectors with allocated capacity)
// 2. Better handle cut and state storage:
//     - currently allocating twice the memory for cuts (BendersCut and Model row)
//     - currently allocating twice the memory for states of the same iteration (VisitedState and Realization)
// Expected memory cost for allocating 2200 state variables as f64 for 120 stages: 2MB

/// Helper function for evaluating the dot product between two vectors.
/// This implementation expect f64 slices and does not use any kind
/// of SSE operations. The slices are expected to have the same length.
///
/// ## Example
///
/// ```
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0, 1.0, 1.0];
///
/// let dot = powers_rs::sddp::dot_product(&a, &b);
/// assert_eq!(dot, 6.0);
/// ```
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut product = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

/// Helper function for removing the future cost term from the stage objective,
/// a.k.a the `alpha` term, or the epigraphical variable, assuming the objective
/// function is:
///
/// c^T x + `alpha`
pub fn get_current_stage_objective(
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
        _ => set_final_retry_solver_options(model),
    }
}

pub struct VisitedState {
    pub state: Arc<Vec<f64>>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
}

impl VisitedState {
    pub fn new(
        state: Arc<Vec<f64>>,
        dominating_objective: f64,
        dominating_cut_id: usize,
    ) -> Self {
        Self {
            state,
            dominating_objective,
            dominating_cut_id,
        }
    }
}

pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: bool,
    pub non_dominated_state_count: isize,
}

impl BendersCut {
    pub fn new(
        id: usize,
        average_water_value: Vec<f64>,
        average_cost: f64,
    ) -> Self {
        Self {
            id,
            coefficients: average_water_value,
            rhs: average_cost,
            active: true,
            non_dominated_state_count: 1,
        }
    }

    pub fn eval_height_at_state(&self, state: &[f64]) -> f64 {
        self.rhs + dot_product(&self.coefficients, state)
    }
}

pub struct NodeData {
    pub system: System,
    pub subproblem: Subproblem,
    pub initial_storage: Vec<f64>,
}

impl NodeData {
    pub fn new(system: System) -> Self {
        let subproblem = Subproblem::new(&system);
        let num_hydros = system.hydros.len();
        Self {
            system,
            subproblem,
            initial_storage: Vec::<f64>::with_capacity(num_hydros),
        }
    }
}

pub struct Bus {
    id: usize,
    deficit_cost: f64,
    hydro_ids: Vec<usize>,
    thermal_ids: Vec<usize>,
    source_line_ids: Vec<usize>,
    target_line_ids: Vec<usize>,
}

impl Bus {
    pub fn new(id: usize, deficit_cost: f64) -> Self {
        Self {
            id,
            deficit_cost,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        }
    }

    pub fn add_hydro(&mut self, hydro_id: usize) {
        self.hydro_ids.push(hydro_id);
    }

    pub fn add_thermal(&mut self, thermal_id: usize) {
        self.thermal_ids.push(thermal_id);
    }

    pub fn add_source_line(&mut self, line_id: usize) {
        self.source_line_ids.push(line_id);
    }

    pub fn add_target_line(&mut self, line_id: usize) {
        self.target_line_ids.push(line_id);
    }
}

pub struct Line {
    pub id: usize,
    pub source_bus_id: usize,
    pub target_bus_id: usize,
    pub direct_capacity: f64,
    pub reverse_capacity: f64,
    pub exchange_penalty: f64,
}

impl Line {
    pub fn new(
        id: usize,
        source_bus_id: usize,
        target_bus_id: usize,
        direct_capacity: f64,
        reverse_capacity: f64,
        exchange_penalty: f64,
    ) -> Self {
        Self {
            id,
            source_bus_id,
            target_bus_id,
            direct_capacity,
            reverse_capacity,
            exchange_penalty,
        }
    }
}

pub struct Thermal {
    pub id: usize,
    pub bus_id: usize,
    pub cost: f64,
    pub min_generation: f64,
    pub max_generation: f64,
}

impl Thermal {
    pub fn new(
        id: usize,
        bus_id: usize,
        cost: f64,
        min_generation: f64,
        max_generation: f64,
    ) -> Self {
        Self {
            id,
            bus_id,
            cost,
            min_generation,
            max_generation,
        }
    }
}

pub struct Hydro {
    pub id: usize,
    pub downstream_hydro_id: Option<usize>,
    pub bus_id: usize,
    pub productivity: f64,
    pub min_storage: f64,
    pub max_storage: f64,
    pub min_turbined_flow: f64,
    pub max_turbined_flow: f64,
    pub spillage_penalty: f64,
    pub upstream_hydro_ids: Vec<usize>,
}

impl Hydro {
    pub fn new(
        id: usize,
        downstream_hydro_id: Option<usize>,
        bus_id: usize,
        productivity: f64,
        min_storage: f64,
        max_storage: f64,
        min_turbined_flow: f64,
        max_turbined_flow: f64,
        spillage_penalty: f64,
    ) -> Self {
        Self {
            id,
            downstream_hydro_id,
            bus_id,
            productivity,
            min_storage,
            max_storage,
            min_turbined_flow,
            max_turbined_flow,
            spillage_penalty,
            upstream_hydro_ids: vec![],
        }
    }

    pub fn add_upstream_hydro(&mut self, hydro_id: usize) {
        self.upstream_hydro_ids.push(hydro_id);
    }
}

#[allow(dead_code)]
pub struct SystemMetadata {
    buses_count: usize,
    lines_count: usize,
    thermals_count: usize,
    hydros_count: usize,
}

pub struct System {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    thermals: Vec<Thermal>,
    hydros: Vec<Hydro>,
    meta: SystemMetadata,
}

impl System {
    pub fn new(
        mut buses: Vec<Bus>,
        lines: Vec<Line>,
        thermals: Vec<Thermal>,
        hydros: Vec<Hydro>,
    ) -> Self {
        for l in lines.iter() {
            buses[l.source_bus_id].add_source_line(l.id);
            buses[l.target_bus_id].add_target_line(l.id);
        }
        for t in thermals.iter() {
            buses[t.bus_id].add_thermal(t.id);
        }
        for h in hydros.iter() {
            buses[h.bus_id].add_hydro(h.id);
        }

        let buses_count = buses.len();
        let lines_count = lines.len();
        let thermals_count = thermals.len();
        let hydros_count = hydros.len();

        Self {
            buses,
            lines,
            thermals,
            hydros,
            meta: SystemMetadata {
                buses_count,
                lines_count,
                thermals_count,
                hydros_count,
            },
        }
    }

    pub fn default() -> Self {
        let buses = vec![Bus::new(0, 50.0)];
        let lines: Vec<Line> = vec![];
        let thermals = vec![
            Thermal::new(0, 0, 5.0, 0.0, 15.0),
            Thermal::new(1, 0, 10.0, 0.0, 15.0),
        ];
        let hydros =
            vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

        Self::new(buses, lines, thermals, hydros)
    }
}

struct Accessors {
    deficit: Vec<usize>,
    direct_exchange: Vec<usize>,
    reverse_exchange: Vec<usize>,
    thermal_gen: Vec<usize>,
    turbined_flow: Vec<usize>,
    spillage: Vec<usize>,
    stored_volume: Vec<usize>,
    alpha: usize,
    load_balance: Vec<usize>,
    hydro_balance: Vec<usize>,
}

pub struct Subproblem {
    model: solver::Model,
    accessors: Accessors,
    num_state_variables: usize,
    num_cuts: usize,
    active_cut_ids: Vec<usize>,
    pub states: Vec<VisitedState>,
    pub cuts: Vec<BendersCut>,
}

impl Subproblem {
    pub fn new(system: &System) -> Self {
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
        set_default_solver_options(&mut model);

        // for making better allocation
        let num_state_variables = stored_volume.len();
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

        Subproblem {
            model,
            accessors,
            num_state_variables,
            num_cuts: 0,
            active_cut_ids: Vec::<usize>::new(),
            states: Vec::<VisitedState>::new(),
            cuts: Vec::<BendersCut>::new(),
        }
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

    pub fn set_uncertainties<'a>(
        &mut self,
        bus_loads: &'a Vec<f64>,
        initial_storage: &Vec<f64>,
        hydros_inflow: &Vec<f64>,
    ) {
        self.set_load_balance_rhs(bus_loads);
        self.set_hydro_balance_rhs(hydros_inflow, initial_storage);
    }

    fn get_deficit_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.deficit.first().unwrap();
        let last = *self.accessors.deficit.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    fn get_direct_exchange_from_solution(
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

    fn get_reverse_exchange_from_solution(
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

    fn get_thermal_gen_from_solution(
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

    fn get_spillage_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.spillage.first().unwrap();
        let last = *self.accessors.spillage.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    fn get_turbined_flow_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.turbined_flow.first().unwrap();
        let last = *self.accessors.turbined_flow.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    fn get_final_storage_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.stored_volume.first().unwrap();
        let last = *self.accessors.stored_volume.last().unwrap() + 1;
        solution.colvalue[first..last].to_vec()
    }

    fn get_water_values_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.hydro_balance.first().unwrap();
        let last = *self.accessors.hydro_balance.last().unwrap() + 1;
        solution.rowdual[first..last].to_vec()
    }

    fn get_marginal_cost_from_solution(
        &self,
        solution: &solver::Solution,
    ) -> Vec<f64> {
        let first = *self.accessors.load_balance.first().unwrap();
        let last = *self.accessors.load_balance.last().unwrap() + 1;
        solution.rowdual[first..last].to_vec()
    }

    fn slice_solution_rows_to_problem_constraints(
        &self,
        solution: &mut solver::Solution,
    ) {
        let end = *self.accessors.hydro_balance.last().unwrap() + 1;
        solution.rowvalue.truncate(end);
        solution.rowdual.truncate(end);
    }

    fn eval_new_cut_domination(&mut self, cut: &mut BendersCut) {
        // Tests the new cut on every previously visited state. If this cut dominates,
        // decrements the previous dominating cut counter and updates this.
        for state in self.states.iter_mut() {
            let height = cut.eval_height_at_state(&state.state);
            if height > state.dominating_objective {
                // println!(
                //     "State {:?} is dominated by new cut! ({})",
                //     state, height
                // );
                self.cuts[state.dominating_cut_id].non_dominated_state_count -=
                    1;
                cut.non_dominated_state_count += 1;
                state.dominating_cut_id = cut.id;
                state.dominating_objective = height;
            }
        }
    }

    fn update_old_cuts_domination(
        &mut self,
        current_state: &mut VisitedState,
    ) -> Vec<usize> {
        // Tests the cuts that are not in the model for the new state. If any of these cuts
        // dominate the new state, increment their counter and puts them back inside the model
        let mut cut_non_dominated_decrement_ids = Vec::<usize>::new();
        let mut cut_ids_to_return_to_model = Vec::<usize>::new();
        for old_cut in self.cuts.iter_mut() {
            match old_cut.active {
                true => continue,
                false => {
                    let height =
                        old_cut.eval_height_at_state(&current_state.state);
                    if height > current_state.dominating_objective {
                        // println!(
                        //     "Old cut {:?} dominates new state! ({})",
                        //     old_cut, height
                        // );
                        cut_non_dominated_decrement_ids
                            .push(current_state.dominating_cut_id);

                        old_cut.non_dominated_state_count += 1;
                        current_state.dominating_cut_id = old_cut.id;
                        current_state.dominating_objective = height;
                        cut_ids_to_return_to_model.push(old_cut.id);
                    }
                    continue;
                }
            }
        }

        // println!("{:?}", self.active_cut_ids);

        // Decrements the non-dominating counts
        for cut_id in cut_non_dominated_decrement_ids.iter() {
            self.cuts[*cut_id].non_dominated_state_count -= 1;
        }

        cut_ids_to_return_to_model
    }

    fn return_and_remove_cuts_from_model(
        &mut self,
        cut_ids_to_return_to_model: &[usize],
    ) {
        // Add cuts back to model
        for cut_id in cut_ids_to_return_to_model.iter() {
            self.return_cut_to_model(*cut_id);
        }

        // Iterate over all the cuts, deleting from the model the cuts that should be deleted
        let mut cut_ids_to_remove_from_model = Vec::<usize>::new();
        for cut in self.cuts.iter_mut() {
            if (cut.non_dominated_state_count <= 0) && cut.active {
                cut_ids_to_remove_from_model.push(cut.id);
            }
        }
        // println!("Cut IDs to remove: {:?}", cut_ids_to_remove_from_model);

        for cut_id in cut_ids_to_remove_from_model.iter() {
            self.remove_cut_from_model(*cut_id);
        }
    }

    pub fn add_cut_to_model(&mut self, cut: &mut BendersCut) {
        // println!("Adding cut with ID {} to model", cut.id);
        let mut factors =
            Vec::<(usize, f64)>::with_capacity(self.num_state_variables + 1);
        factors.push((self.accessors.alpha, 1.0));
        for (hydro_id, stored_volume) in
            self.accessors.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -1.0 * cut.coefficients[hydro_id]));
        }
        self.model.add_row(cut.rhs.., factors);
        self.active_cut_ids.push(cut.id);
        self.num_cuts += 1;
    }

    pub fn return_cut_to_model(&mut self, cut_id: usize) {
        // println!("Returning cut with ID {} to model", cut_id);
        let mut factors =
            Vec::<(usize, f64)>::with_capacity(self.num_state_variables + 1);
        let cut = self.cuts.get_mut(cut_id).unwrap();
        factors.push((self.accessors.alpha, 1.0));
        for (hydro_id, stored_volume) in
            self.accessors.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -1.0 * cut.coefficients[hydro_id]));
        }
        self.model.add_row(cut.rhs.., factors);
        self.active_cut_ids.push(cut_id);
        self.cuts[cut_id].active = true;
    }

    pub fn remove_cut_from_model(&mut self, cut_id: usize) {
        // println!("Removing cut with ID {} from model", cut_id);
        let cut_index = self
            .active_cut_ids
            .iter()
            .position(|&x| x == cut_id)
            .unwrap();
        let row_index =
            *self.accessors.hydro_balance.last().unwrap() + 1 + cut_index;
        // println!("Model row index: {}", row_index);
        self.model.delete_row(row_index).unwrap();
        self.active_cut_ids.remove(cut_index);
        self.cuts[cut_id].active = false;
    }
}

pub struct Realization<'a> {
    pub bus_loads: &'a Vec<f64>,
    pub deficit: Vec<f64>,
    pub exchange: Vec<f64>,
    pub hydros_initial_storage: Arc<Vec<f64>>,
    pub hydros_final_storage: Arc<Vec<f64>>,
    pub inflow: Vec<f64>,
    pub turbined_flow: Vec<f64>,
    pub spillage: Vec<f64>,
    pub thermal_generation: Vec<f64>,
    pub water_values: Vec<f64>,
    pub marginal_cost: Vec<f64>,
    pub current_stage_objective: f64,
    pub total_stage_objective: f64,
    pub basis: solver::Basis,
}

impl<'a> Realization<'a> {
    pub fn new(
        bus_loads: &'a Vec<f64>,
        deficit: Vec<f64>,
        exchange: Vec<f64>,
        hydros_initial_storage: Arc<Vec<f64>>,
        hydros_final_storage: Arc<Vec<f64>>,
        inflow: Vec<f64>,
        turbined_flow: Vec<f64>,
        spillage: Vec<f64>,
        thermal_generation: Vec<f64>,
        water_values: Vec<f64>,
        marginal_cost: Vec<f64>,
        current_stage_objective: f64,
        total_stage_objective: f64,
        basis: solver::Basis,
    ) -> Self {
        Self {
            bus_loads,
            deficit,
            exchange,
            hydros_initial_storage,
            hydros_final_storage,
            inflow,
            turbined_flow,
            spillage,
            thermal_generation,
            water_values,
            marginal_cost,
            current_stage_objective,
            total_stage_objective,
            basis,
        }
    }
}

pub struct Trajectory<'a> {
    pub realizations: Vec<Realization<'a>>,
    pub cost: f64,
}

impl<'a> Trajectory<'a> {
    pub fn new(realizations: Vec<Realization<'a>>, cost: f64) -> Self {
        Self { realizations, cost }
    }
}

/// Runs a single step of the forward pass / backward branching,
/// solving a node's subproblem for some sampled uncertainty realization.
///
/// Returns the realization with relevant data.
fn realize_uncertainties<'a>(
    node: &mut graph::Node<NodeData>,
    bus_loads: &'a Vec<f64>, // loads for stage 'index' ordered by id
    hydros_initial_storage: Arc<Vec<f64>>, // inflows for stage 'index' ordered by id
    hydros_inflow: &Vec<f64>, // inflows for stage 'index' ordered by id
) -> Realization<'a> {
    node.data.subproblem.set_uncertainties(
        bus_loads,
        hydros_initial_storage.as_ref(),
        hydros_inflow,
    );
    let mut retry: usize = 0;
    loop {
        if retry > 3 {
            panic!("Error while solving model");
        }
        node.data.subproblem.model.solve();

        match node.data.subproblem.model.status() {
            solver::HighsModelStatus::Optimal => {
                let mut solution = node.data.subproblem.model.get_solution();
                node.data
                    .subproblem
                    .slice_solution_rows_to_problem_constraints(&mut solution);
                let basis = node.data.subproblem.model.get_basis();
                let total_stage_objective =
                    node.data.subproblem.model.get_objective_value();
                let current_stage_objective = get_current_stage_objective(
                    total_stage_objective,
                    &solution,
                );
                let deficit =
                    node.data.subproblem.get_deficit_from_solution(&solution);
                let direct_exchange = node
                    .data
                    .subproblem
                    .get_direct_exchange_from_solution(&solution);
                let reverse_exchange = node
                    .data
                    .subproblem
                    .get_reverse_exchange_from_solution(&solution);
                // evals net exchange
                let exchange = direct_exchange
                    .iter()
                    .enumerate()
                    .map(|(i, e)| e - reverse_exchange[i])
                    .collect();
                let thermal_generation = node
                    .data
                    .subproblem
                    .get_thermal_gen_from_solution(&solution);
                let hydros_final_storage = node
                    .data
                    .subproblem
                    .get_final_storage_from_solution(&solution);
                let turbined_flow = node
                    .data
                    .subproblem
                    .get_turbined_flow_from_solution(&solution);
                let spillage =
                    node.data.subproblem.get_spillage_from_solution(&solution);
                let water_values = node
                    .data
                    .subproblem
                    .get_water_values_from_solution(&solution);
                let marginal_cost = node
                    .data
                    .subproblem
                    .get_marginal_cost_from_solution(&solution);
                node.data.subproblem.model.clear_solver();
                if retry != 0 {
                    set_default_solver_options(&mut node.data.subproblem.model);
                }
                return Realization::new(
                    bus_loads,
                    deficit,
                    exchange,
                    hydros_initial_storage,
                    Arc::new(hydros_final_storage),
                    hydros_inflow.clone(),
                    turbined_flow,
                    spillage,
                    thermal_generation,
                    water_values,
                    marginal_cost,
                    current_stage_objective,
                    total_stage_objective,
                    basis,
                );
            }
            solver::HighsModelStatus::Infeasible => {
                retry += 1;
                set_retry_solver_options(
                    &mut node.data.subproblem.model,
                    retry,
                );
            }
            _ => panic!("Error while solving model"),
        }
    }
}

/// Runs a forward pass of the SDDP algorithm, obtaining a viable
/// trajectory of states to be used in the backward pass.
///
/// Returns the sampled trajectory.
fn forward<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: Arc<Vec<f64>>,
    hydros_inflow: &Vec<&Vec<f64>>, // indexed by stage | hydro
) -> Trajectory<'a> {
    let mut realizations = Vec::<Realization>::with_capacity(g.node_count());
    let mut cost = 0.0;

    for id in 0..g.node_count() {
        let initial_storage = if g.is_root(id) {
            Arc::clone(&hydros_initial_storage)
        } else {
            Arc::clone(&realizations.last().unwrap().hydros_final_storage)
        };
        let node = g.get_node_mut(id).unwrap();
        let realization = realize_uncertainties(
            node,
            &bus_loads[id], // loads for stage 'index' ordered by id
            initial_storage,
            hydros_inflow[id], // inflows for stage 'index' ordered by id
        );
        cost += realization.current_stage_objective;
        realizations.push(realization);
    }
    Trajectory::new(realizations, cost)
}

// fn hot_start_with_forward_solution<'a>(
//     node: &mut Node,
//     node_forward_realization: &'a Realization,
// ) {
//     let num_model_rows = node.subproblem.model.num_rows();
//     let mut forward_rows = node_forward_realization.solution.rowvalue.to_vec();
//     let mut forward_dual_rows =
//         node_forward_realization.solution.rowdual.to_vec();
//     let num_forward_rows = forward_rows.len();

//     // checks if should add zeros to the rows (new cuts added)
//     if num_forward_rows < num_model_rows {
//         let row_diff = num_model_rows - num_forward_rows;
//         forward_rows.append(&mut vec![0.0; row_diff]);
//         forward_dual_rows.append(&mut vec![0.0; row_diff]);
//     }

//     node.subproblem.model.set_solution(
//         Some(&node_forward_realization.solution.colvalue),
//         Some(&forward_rows),
//         Some(&node_forward_realization.solution.coldual),
//         Some(&forward_dual_rows),
//     );
// }

fn reuse_forward_basis<'a>(
    node: &mut graph::Node<NodeData>,
    node_forward_realization: &'a Realization,
) {
    let num_model_rows = node.data.subproblem.model.num_rows();
    let mut forward_rows = node_forward_realization.basis.rows().to_vec();
    let num_forward_rows = forward_rows.len();

    // checks if should add zeros to the rows (new cuts added)
    if num_forward_rows < num_model_rows {
        let row_diff = num_model_rows - num_forward_rows;
        forward_rows.append(&mut vec![0; row_diff]);
    } else if num_forward_rows > num_model_rows {
        forward_rows.truncate(num_model_rows);
    }

    node.data.subproblem.model.set_basis(
        Some(node_forward_realization.basis.columns()),
        Some(&forward_rows),
    );
}

/// Solves a node's subproblem for all it's branchings and
/// returns the solutions.
fn solve_all_branchings<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    node_id: usize,
    num_branchings: usize,
    node_forward_realization: &'a Realization,
    node_saa: &'a Vec<Vec<f64>>, // indexed by stage | branching | hydro
) -> Vec<Realization<'a>> {
    let mut realizations = Vec::<Realization>::with_capacity(num_branchings);
    let node = g.get_node_mut(node_id).unwrap();
    for hydros_inflow in node_saa.iter() {
        reuse_forward_basis(node, node_forward_realization);
        // hot_start_with_forward_solution(node, node_forward_realization);
        let realization = realize_uncertainties(
            node,
            node_forward_realization.bus_loads,
            Arc::clone(&node_forward_realization.hydros_initial_storage),
            hydros_inflow,
        );
        realizations.push(realization);
    }

    realizations
}

/// Evaluates and returns the new cut to be added to a node from the
/// solutions of the node's subproblem for all branchings.
fn eval_average_cut(
    node: &graph::Node<NodeData>,
    cut_id: usize,
    num_branchings: usize,
    branchings_realizations: &Vec<Realization>,
    node_forward_realization: &Realization,
) -> BendersCut {
    let num_hydros = node.data.system.meta.hydros_count;
    let mut average_water_values = vec![0.0; num_hydros];
    let mut average_solution_cost = 0.0;
    for realization in branchings_realizations.iter() {
        for hydro_id in 0..num_hydros {
            average_water_values[hydro_id] += realization.water_values[hydro_id]
        }
        average_solution_cost += realization.total_stage_objective;
    }

    // evaluate average cut
    average_solution_cost = average_solution_cost / (num_branchings as f64);
    for hydro_id in 0..num_hydros {
        average_water_values[hydro_id] =
            average_water_values[hydro_id] / (num_branchings as f64);
    }

    let cut_rhs = average_solution_cost
        - dot_product(
            &average_water_values,
            &node_forward_realization.hydros_initial_storage,
        );
    BendersCut::new(cut_id, average_water_values, cut_rhs)
}

fn update_future_cost_function(
    g: &mut graph::DirectedGraph<NodeData>,
    parent_id: usize,
    child_id: usize,
    num_branchings: usize,
    forward_realization: &Realization,
    branchings_realizations: &Vec<Realization>,
) {
    let child_node = g.get_node(child_id).unwrap();
    let new_cut_id = g.get_node(parent_id).unwrap().data.subproblem.num_cuts;
    let mut cut = eval_average_cut(
        &child_node,
        new_cut_id,
        num_branchings,
        branchings_realizations,
        forward_realization,
    );
    let mut state = VisitedState::new(
        Arc::clone(&forward_realization.hydros_final_storage),
        cut.eval_height_at_state(&forward_realization.hydros_final_storage),
        cut.id,
    );

    let parent_node: &mut graph::Node<NodeData> =
        g.get_node_mut(parent_id).unwrap();
    // Adds cuts to model and applies exact cut selection
    parent_node.data.subproblem.add_cut_to_model(&mut cut);
    parent_node
        .data
        .subproblem
        .eval_new_cut_domination(&mut cut);
    parent_node.data.subproblem.cuts.push(cut);
    let cut_ids_to_return_to_model = parent_node
        .data
        .subproblem
        .update_old_cuts_domination(&mut state);
    parent_node
        .data
        .subproblem
        .return_and_remove_cuts_from_model(&cut_ids_to_return_to_model);
    parent_node.data.subproblem.states.push(state);
}

/// Evaluates and returns the lower bound from the solutions
/// of the first stage problem for all branchings.
fn eval_first_stage_bound(
    num_branchings: usize,
    branchings_realizations: &Vec<Realization>,
) -> f64 {
    let mut average_solution_cost = 0.0;
    for realization in branchings_realizations.iter() {
        average_solution_cost += realization.total_stage_objective
    }
    return average_solution_cost / (num_branchings as f64);
}

/// Runs a backward pass of the SDDP algorithm, adding a new cut for
/// each node in the graph, except the first stage node, which is used
/// on estimating the lower bound of the current iteration.
///
/// Returns the current estimated lower bound.
fn backward(
    g: &mut graph::DirectedGraph<NodeData>,
    trajectory: &Trajectory,
    saa: &Vec<Vec<Vec<f64>>>, // indexed by stage | branching | hydro
    num_branchings: usize,
) -> f64 {
    for id in (0..g.node_count()).rev() {
        let node_forward_realization = trajectory.realizations.get(id).unwrap();
        let node_saa = saa.get(id).unwrap();
        let realizations = solve_all_branchings(
            g,
            id,
            num_branchings,
            node_forward_realization,
            node_saa,
        );
        if !g.is_root(id) {
            let parent_id = g.get_parents(id).unwrap()[0];
            update_future_cost_function(
                g,
                parent_id,
                id,
                num_branchings,
                node_forward_realization,
                &realizations,
            );
        } else {
            return eval_first_stage_bound(num_branchings, &realizations);
        }
    }
    // TODO - better handle this edge case by returning a Result<>
    return 0.0;
}

/// Runs a single iteration, comprised of forward and backward passes,
/// of the SDDP algorithm.
fn iterate<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: Arc<Vec<f64>>,
    hydros_inflow: &'a Vec<&'a Vec<f64>>,
    saa: &'a Vec<Vec<Vec<f64>>>,
    num_branchings: usize,
) -> (f64, f64, Duration) {
    let begin = Instant::now();

    let trajectory =
        forward(g, bus_loads, hydros_initial_storage, hydros_inflow);

    let trajectory_cost = trajectory.cost;
    let first_stage_bound = backward(g, &trajectory, saa, num_branchings);

    let iteration_time = begin.elapsed();
    return (trajectory_cost, first_stage_bound, iteration_time);
}

/// Generates an inflow SAA indexed by stage | branching | hydro.
///
/// `scenario_generator` must be indexed by stage | hydro
///
/// `num_stages` is the number of nodes in the graph
///
/// `num_branchings` is the number of branchings (same for all nodes)
///
/// ## Example
///
/// ```
/// let mu = 3.6;
/// let sigma = 0.6928;
/// let num_hydros = 2;
/// let scenario_generator: Vec<Vec<rand_distr::LogNormal<f64>>> =
///     vec![vec![rand_distr::LogNormal::new(mu, sigma).unwrap(); num_hydros]];
/// let num_stages = 1;
/// let num_branchings = 10;
///
/// let saa = powers_rs::sddp::generate_saa(&scenario_generator, num_hydros, num_stages, num_branchings);
/// assert_eq!(saa.len(), num_stages);
/// assert_eq!(saa[0].len(), num_branchings);
/// assert_eq!(saa[0][0].len(), num_hydros);
///
/// ```
pub fn generate_saa<'a>(
    scenario_generator: &'a Vec<Vec<LogNormal<f64>>>,
    num_hydros: usize,
    num_stages: usize,
    num_branchings: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut rng = Xoshiro256Plus::seed_from_u64(0);

    let mut saa: Vec<Vec<Vec<f64>>> =
        vec![
            vec![Vec::<f64>::with_capacity(num_hydros); num_branchings];
            num_stages
        ];
    for (stage_index, stage_generator) in scenario_generator.iter().enumerate()
    {
        let hydro_inflows: Vec<Vec<f64>> = stage_generator
            .iter()
            .map(|hydro_generator| {
                hydro_generator
                    .sample_iter(&mut rng)
                    .take(num_branchings)
                    .collect()
            })
            .collect();
        for branching_index in 0..num_branchings {
            for inflows in hydro_inflows.iter() {
                saa[stage_index][branching_index]
                    .push(inflows[branching_index]);
            }
        }
    }
    saa
}

/// Helper function for displaying the greeting data for the training
fn training_greeting(
    num_iterations: usize,
    num_stages: usize,
    num_branchings: usize,
) {
    println!("\n# Training");
    println!("- Iterations: {num_iterations}");
    println!("- Stages: {num_stages}");
    println!("- Branchings: {num_branchings}\n");
}

/// Helper function for displaying the training table header
fn training_table_header() {
    println!(
        "{0: ^10} | {1: ^15} | {2: ^14} | {3: ^12}",
        "iteration", "lower bound ($)", "simulation ($)", "time (s)"
    )
}

/// Helper function for displaying a divider for the training table
fn training_table_divider() {
    println!("------------------------------------------------------------")
}

fn training_duration(time: Duration) {
    println!("\nTraining time: {:.2} s", time.as_millis() as f64 / 1000.0)
}

/// Helper function for displaying a row of iteration results for
/// the training table
fn training_table_row(
    iteration: usize,
    lower_bound: f64,
    simulation: f64,
    time: Duration,
) {
    println!(
        "{0: >10} | {1: >15.4} | {2: >14.4} | {3: >12.2}",
        iteration,
        lower_bound,
        simulation,
        time.as_millis() as f64 / 1000.0
    )
}

/// Runs a training step of the SDDP algorithm over a graph.
pub fn train<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    num_iterations: usize,
    num_branchings: usize,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: Arc<Vec<f64>>,
    scenario_generator: &'a Vec<Vec<LogNormal<f64>>>,
) {
    let begin = Instant::now();

    let mut rng = Xoshiro256Plus::seed_from_u64(0);
    let forward_indices_dist =
        Uniform::<usize>::try_from(0..num_branchings).unwrap();

    let num_stages = g.node_count();
    let num_hydros = hydros_initial_storage.len();
    let saa = generate_saa(
        scenario_generator,
        num_hydros,
        num_stages,
        num_branchings,
    );

    training_greeting(num_iterations, g.node_count(), num_branchings);
    training_table_divider();
    training_table_header();
    training_table_divider();

    for index in 0..num_iterations {
        // Generates indices for sampling inflows, indexed by stage
        let forward_branching_indices: Vec<usize> = forward_indices_dist
            .sample_iter(&mut rng)
            .take(num_stages)
            .collect();
        // Samples the SAA at the previously generated indices
        let hydros_inflow = saa
            .iter()
            .enumerate()
            .map(|(index, stage_inflows)| {
                &stage_inflows[forward_branching_indices[index]]
            })
            .collect();

        let (simulation, lower_bound, time) = iterate(
            g,
            bus_loads,
            Arc::clone(&hydros_initial_storage),
            &hydros_inflow,
            &saa,
            num_branchings,
        );

        training_table_row(index + 1, lower_bound, simulation, time);
    }

    training_table_divider();
    let duration = begin.elapsed();
    training_duration(duration);
}

/// Helper function for displaying the greeting data for the simulation
fn simulation_greeting(num_simulation_scenarios: usize) {
    println!("\n# Simulating");
    println!("- Scenarios: {num_simulation_scenarios}\n");
}

fn simulation_stats(mean: f64, std: f64) {
    println!("Expected cost ($): {:.2} +- {:.2}", mean, std);
}

fn simulation_duration(time: Duration) {
    println!(
        "\nSimulation time: {:.2} s",
        time.as_millis() as f64 / 1000.0
    )
}

/// Runs a simulation using the policy obtained by the SDDP algorithm.
pub fn simulate<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    num_simulation_scenarios: usize,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: Arc<Vec<f64>>,
    scenario_generator: &'a Vec<Vec<LogNormal<f64>>>,
) -> Vec<Trajectory<'a>> {
    let begin = Instant::now();

    let num_stages = g.node_count();
    let num_hydros = hydros_initial_storage.len();

    simulation_greeting(num_simulation_scenarios);

    let mut trajectories =
        Vec::<Trajectory>::with_capacity(num_simulation_scenarios);

    for _ in 0..num_simulation_scenarios {
        // Generates an SAA and samples the single scenario
        let saa = generate_saa(scenario_generator, num_hydros, num_stages, 1);
        let hydros_inflow =
            saa.iter().map(|stage_inflows| &stage_inflows[0]).collect();

        let trajectory = forward(
            g,
            bus_loads,
            Arc::clone(&hydros_initial_storage),
            &hydros_inflow,
        );
        trajectories.push(trajectory);
    }

    let simulation_costs: Vec<f64> =
        trajectories.iter().map(|t| t.cost).collect();

    let total_cost: f64 = simulation_costs.iter().sum();
    let mean_cost = total_cost / (num_simulation_scenarios as f64);
    let cost_deviations: Vec<f64> = simulation_costs
        .iter()
        .map(|c| (c - mean_cost) * (c - mean_cost))
        .collect();
    let cost_total_deviation: f64 = cost_deviations.iter().sum();
    let std_cost =
        f64::sqrt(cost_total_deviation / (num_simulation_scenarios as f64));

    simulation_stats(mean_cost, std_cost);
    let duration = begin.elapsed();
    simulation_duration(duration);

    trajectories
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_default_system() {
        let system = System::default();
        assert_eq!(system.buses.len(), 1);
        assert_eq!(system.lines.len(), 0);
        assert_eq!(system.thermals.len(), 2);
        assert_eq!(system.hydros.len(), 1);
    }

    #[test]
    fn test_create_subproblem_with_default_system() {
        let system = System::default();
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
        let system = System::default();
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
        let system = System::default();
        let mut subproblem = Subproblem::new(&system);
        let inflow = [0.0];
        let initial_storage = [23.333];
        let load = [50.0];
        subproblem.set_load_balance_rhs(&load);
        subproblem.set_hydro_balance_rhs(&inflow, &initial_storage);

        subproblem.model.solve();
        assert_eq!(subproblem.model.get_objective_value(), 191.67000000000002);
    }

    #[test]
    fn test_forward_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let id0 = g.add_node(NodeData::new(System::default()));
        let id1 = g.add_node(NodeData::new(System::default()));
        let id2 = g.add_node(NodeData::new(System::default()));
        g.add_edge(id0, id1).unwrap();
        g.add_edge(id1, id2).unwrap();
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = Arc::new(vec![83.222]);
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        forward(&mut g, &bus_loads, hydros_initial_storage, &hydros_inflow);
    }

    #[test]
    fn test_backward_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let id0 = g.add_node(NodeData::new(System::default()));
        let id1 = g.add_node(NodeData::new(System::default()));
        let id2 = g.add_node(NodeData::new(System::default()));
        g.add_edge(id0, id1).unwrap();
        g.add_edge(id1, id2).unwrap();
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = Arc::new(vec![83.222]);
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let trajectory =
            forward(&mut g, &bus_loads, hydros_initial_storage, &hydros_inflow);
        let branchings = vec![
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
        ];
        backward(&mut g, &trajectory, &branchings, 3);
    }

    #[test]
    fn test_iterate_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let id0 = g.add_node(NodeData::new(System::default()));
        let id1 = g.add_node(NodeData::new(System::default()));
        let id2 = g.add_node(NodeData::new(System::default()));
        g.add_edge(id0, id1).unwrap();
        g.add_edge(id1, id2).unwrap();
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = Arc::new(vec![83.222]);
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let branchings = vec![
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
        ];
        iterate(
            &mut g,
            &bus_loads,
            hydros_initial_storage,
            &hydros_inflow,
            &branchings,
            3,
        );
    }

    #[test]
    fn test_train_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = g.add_node(NodeData::new(System::default()));
        let mut scenario_generator: Vec<Vec<LogNormal<f64>>> =
            vec![vec![LogNormal::new(3.6, 0.6928).unwrap()]];
        let mut bus_loads = vec![vec![75.0]];
        for _ in 1..4 {
            let new_id = g.add_node(NodeData::new(System::default()));
            g.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            scenario_generator.push(vec![LogNormal::new(3.6, 0.6928).unwrap()]);
            bus_loads.push(vec![75.0]);
        }
        let hydros_initial_storage = Arc::new(vec![83.222]);
        train(
            &mut g,
            24,
            3,
            &bus_loads,
            hydros_initial_storage,
            &scenario_generator,
        );
    }

    #[test]
    fn test_simulate_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = g.add_node(NodeData::new(System::default()));
        let mut scenario_generator: Vec<Vec<LogNormal<f64>>> =
            vec![vec![LogNormal::new(3.6, 0.6928).unwrap()]];
        let mut bus_loads = vec![vec![75.0]];
        for _ in 1..4 {
            let new_id = g.add_node(NodeData::new(System::default()));
            g.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            scenario_generator.push(vec![LogNormal::new(3.6, 0.6928).unwrap()]);
            bus_loads.push(vec![75.0]);
        }
        let hydros_initial_storage = Arc::new(vec![83.222]);
        train(
            &mut g,
            24,
            3,
            &bus_loads,
            Arc::clone(&hydros_initial_storage),
            &scenario_generator,
        );
        simulate(
            &mut g,
            100,
            &bus_loads,
            hydros_initial_storage,
            &scenario_generator,
        );
    }
}
