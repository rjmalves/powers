use crate::myhighs;
use rand::prelude::*;
use rand_distr::{LogNormal, Uniform};
use rand_xoshiro::Xoshiro256Plus;
use std::ops::{Index, Range};
use std::time::{Duration, Instant};

// TODO - general optimizations
// 3. Pre-allocate everywhere when the total size of the containers
// is known, in repacement to calling push! (or init vectors with allocated capacity)
// 4. Use the model "offset" field for the objective, replacing
// minimal thermal generation costs (maybe implement a better way to build a system?).
// 5. Implement solver cascading fallbacks
// 6. Better handle cut and state storage:
//     - currently allocating twice the memory for cuts (BendersCut and Model row)
//     - currently allocating twice the memory for states of the same iteration (VisitedState and Realization)

/// Helper function for evaluating the dot product between two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut product = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

/// Helper function for removing the future cost term from the stage objective
pub fn get_current_stage_objective(
    total_stage_objective: f64,
    solution: &myhighs::Solution,
) -> f64 {
    let future_objective = solution.colvalue.last().unwrap();
    total_stage_objective - future_objective
}

/// Helper function for setting the same solver options on
/// every solved problem.
fn set_solver_options(model: &mut myhighs::Model) {
    model.set_option("presolve", "off");
    model.set_option("solver", "simplex");
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("primal_feasibility_tolerance", 1e-7);
    model.set_option("dual_feasibility_tolerance", 1e-7);
    model.set_option("time_limit", 300);
}

// TODO - implement solving process with fallbacks

#[derive(Debug)]
struct VisitedState {
    pub state: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
}

impl VisitedState {
    pub fn new(
        state: Vec<f64>,
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

#[derive(Debug, Clone)]
struct BendersCut {
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

pub struct Node {
    pub index: usize,
    pub system: System,
    subproblem: Subproblem,
}

impl Node {
    pub fn new(index: usize, system: System) -> Self {
        let subproblem = Subproblem::new(&system);
        Self {
            index,
            system,
            subproblem,
        }
    }
}

pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new(node: Node) -> Self {
        Self { nodes: vec![node] }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn append(&mut self, node: Node) {
        self.nodes.push(node);
    }
}

#[derive(Clone)]
struct Bus {
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

#[derive(Clone)]
struct Line {
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

#[derive(Clone)]
struct Thermal {
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

#[derive(Clone)]
struct Hydro {
    pub id: usize,
    pub downstream_hydro_id: Option<usize>,
    pub bus_id: usize,
    pub productivity: f64,
    pub min_storage: f64,
    pub max_storage: f64,
    pub min_generation: f64,
    pub max_generation: f64,
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
        min_generation: f64,
        max_generation: f64,
        spillage_penalty: f64,
    ) -> Self {
        Self {
            id,
            downstream_hydro_id,
            bus_id,
            productivity,
            min_storage,
            max_storage,
            min_generation,
            max_generation,
            spillage_penalty,
            upstream_hydro_ids: vec![],
        }
    }

    pub fn add_upstream_hydro(&mut self, hydro_id: usize) {
        self.upstream_hydro_ids.push(hydro_id);
    }
}

#[derive(Clone)]
pub struct SystemMetadata {
    buses_count: usize,
    lines_count: usize,
    thermals_count: usize,
    hydros_count: usize,
}

#[derive(Clone)]
pub struct System {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    thermals: Vec<Thermal>,
    hydros: Vec<Hydro>,
    meta: SystemMetadata,
}

impl System {
    pub fn default() -> Self {
        let mut buses = vec![Bus::new(0, 50.0)];
        let lines: Vec<Line> = vec![];

        let thermals = vec![
            Thermal::new(0, 0, 5.0, 0.0, 15.0),
            Thermal::new(1, 0, 10.0, 0.0, 15.0),
        ];
        for t in thermals.iter() {
            buses.get_mut(0).unwrap().thermal_ids.push(t.id);
        }

        let hydros =
            vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];
        for h in hydros.iter() {
            buses.get_mut(0).unwrap().hydro_ids.push(h.id);
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
}

#[derive(Clone, Debug)]
struct Accessors {
    deficit: Vec<usize>,
    direct_exchange: Vec<usize>,
    reverse_exchange: Vec<usize>,
    thermal_gen: Vec<usize>,
    turbined_flow: Vec<usize>,
    spillage: Vec<usize>,
    stored_volume: Vec<usize>,
    stored_volume_range: Range<usize>,
    min_generation_slack: Vec<usize>,
    alpha: usize,
    load_balance: Vec<usize>,
    load_balance_range: Range<usize>,
    hydro_balance: Vec<usize>,
    hydro_balance_range: Range<usize>,
}

struct Subproblem {
    model: myhighs::Model,
    accessors: Accessors,
    num_state_variables: usize,
    num_decision_variables: usize,
    num_problem_constraints: usize,
    num_cuts: usize,
    active_cut_ids: Vec<usize>,
    states: Vec<VisitedState>,
    cuts: Vec<BendersCut>,
}

impl Subproblem {
    pub fn new(system: &System) -> Self {
        let mut pb = myhighs::Problem::new();

        const MIN_GENERATION_PENALTY: f64 = 50.050;

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
                    thermal.min_generation..thermal.max_generation,
                )
            })
            .collect();
        let turbined_flow: Vec<usize> = system
            .hydros
            .iter()
            .map(|hydro| {
                pb.add_column(
                    0.0,
                    (hydro.min_generation / hydro.productivity)
                        ..(hydro.max_generation / hydro.productivity),
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
        let min_generation_slack: Vec<usize> = system
            .hydros
            .iter()
            .map(|_hydro| pb.add_column(MIN_GENERATION_PENALTY, 0.0..))
            .collect();

        let stored_volume_range = (*stored_volume.first().unwrap())
            ..(*stored_volume.last().unwrap() + 1);

        let alpha = pb.add_column(1.0, 0.0..);

        for hydro in system.hydros.iter() {
            pb.add_row(
                hydro.min_generation..,
                &[
                    (turbined_flow[hydro.id], 1.0),
                    (min_generation_slack[hydro.id], 1.0),
                ],
            );
        }

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
        let load_balance_range = (*load_balance.first().unwrap())
            ..(*load_balance.last().unwrap() + 1);

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
        let hydro_balance_range = (*hydro_balance.first().unwrap())
            ..(*hydro_balance.last().unwrap() + 1);

        let mut model = pb.optimise(myhighs::Sense::Minimise);
        set_solver_options(&mut model);

        // for making better allocation
        let num_state_variables = stored_volume.len();
        let num_decision_variables = alpha + 1;
        let num_problem_constraints = hydro_balance.last().unwrap() + 1;

        let accessors = Accessors {
            deficit,
            direct_exchange,
            reverse_exchange,
            thermal_gen,
            turbined_flow,
            spillage,
            stored_volume,
            stored_volume_range,
            min_generation_slack,
            alpha,
            load_balance,
            load_balance_range,
            hydro_balance,
            hydro_balance_range,
        };

        Subproblem {
            model,
            accessors,
            num_state_variables,
            num_decision_variables,
            num_problem_constraints,
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
        hydros_inflow: &'a Vec<f64>,
    ) {
        self.set_load_balance_rhs(bus_loads);
        self.set_hydro_balance_rhs(hydros_inflow, initial_storage);
    }

    fn get_final_storage_from_solution(
        &self,
        solution: &myhighs::Solution,
    ) -> Vec<f64> {
        let range = &self.accessors.stored_volume_range;
        solution.colvalue[range.start..range.end].to_vec()
    }

    fn get_water_values_from_solution(
        &self,
        solution: &myhighs::Solution,
    ) -> Vec<f64> {
        let range = &self.accessors.hydro_balance_range;
        solution.rowdual[range.start..range.end].to_vec()
    }

    fn slice_solution_rows_to_problem_constraints(
        &self,
        solution: &mut myhighs::Solution,
    ) {
        let end = &self.accessors.hydro_balance_range.end;
        solution.rowvalue.truncate(*end);
        solution.rowdual.truncate(*end);
    }

    pub fn cut_selection(
        &mut self,
        cut: &mut BendersCut,
        forward_realization: &Realization,
    ) -> VisitedState {
        // Updates cut ID
        cut.id = self.num_cuts;

        let mut current_state = VisitedState::new(
            forward_realization.hydros_final_storage.clone(),
            cut.eval_height_at_state(&forward_realization.hydros_final_storage),
            cut.id,
        );
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

        // Updates cuts set
        // TODO - handle the ownership in a better way, eliminating this clone()
        // maybe break this huge cut selection function into smaller ones?
        self.add_cut_to_model(cut);
        self.cuts.push(cut.clone());

        // println!("{:?}", self.active_cut_ids);

        // Decrements the non-dominating counts
        for cut_id in cut_non_dominated_decrement_ids.iter() {
            self.cuts[*cut_id].non_dominated_state_count -= 1;
        }

        // Add cuts back to model
        for cut_id in cut_ids_to_return_to_model.iter() {
            self.return_cut_to_model(*cut_id);
        }

        // println!("Cut IDs to return: {:?}", cut_ids_to_return_to_model);

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

        current_state
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
        let row_index = self.accessors.hydro_balance_range.end + cut_index;
        // println!("Model row index: {}", row_index);
        self.model.delete_benders_cut_row(row_index).unwrap();
        self.active_cut_ids.remove(cut_index);
        self.cuts[cut_id].active = false;
    }
}

#[derive(Clone, Debug)]
struct Realization<'a> {
    pub bus_loads: &'a Vec<f64>,
    pub hydros_initial_storage: Vec<f64>,
    pub hydros_final_storage: Vec<f64>,
    pub water_values: Vec<f64>,
    pub current_stage_objective: f64,
    pub total_stage_objective: f64,
    pub basis: myhighs::Basis,
}

impl<'a> Realization<'a> {
    pub fn new(
        bus_loads: &'a Vec<f64>,
        hydros_initial_storage: Vec<f64>,
        hydros_final_storage: Vec<f64>,
        water_values: Vec<f64>,
        current_stage_objective: f64,
        total_stage_objective: f64,
        basis: myhighs::Basis,
    ) -> Self {
        Self {
            bus_loads,
            hydros_initial_storage,
            hydros_final_storage,
            water_values,
            current_stage_objective,
            total_stage_objective,
            basis,
        }
    }
}

#[derive(Clone, Debug)]
struct Trajectory<'a> {
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
    node: &mut Node,
    bus_loads: &'a Vec<f64>, // loads for stage 'index' ordered by id
    initial_storage: &Vec<f64>,
    hydros_inflow: &'a Vec<f64>, // inflows for stage 'index' ordered by id
) -> Realization<'a> {
    node.subproblem.set_uncertainties(
        bus_loads,
        initial_storage,
        hydros_inflow,
    );
    node.subproblem.model.solve();
    match node.subproblem.model.status() {
        myhighs::HighsModelStatus::Optimal => {
            let mut solution = node.subproblem.model.get_solution();
            node.subproblem
                .slice_solution_rows_to_problem_constraints(&mut solution);
            let basis = node.subproblem.model.get_basis();
            let total_stage_objective =
                node.subproblem.model.get_objective_value();
            let current_stage_objective =
                get_current_stage_objective(total_stage_objective, &solution);
            let hydros_final_storage =
                node.subproblem.get_final_storage_from_solution(&solution);
            let water_values =
                node.subproblem.get_water_values_from_solution(&solution);
            node.subproblem.model.clear_solver();
            Realization::new(
                bus_loads,
                initial_storage.clone(),
                hydros_final_storage,
                water_values,
                current_stage_objective,
                total_stage_objective,
                basis,
            )
        }
        _ => panic!("Error while solving model"),
    }
}

/// Runs a forward pass of the SDDP algorithm, obtaining a viable
/// trajectory of states to be used in the backward pass.
///
/// Returns the sampled trajectory.
fn forward<'a>(
    nodes: &mut Vec<Node>,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    hydros_inflow: &'a Vec<&'a Vec<f64>>, // indexed by stage | hydro
) -> Trajectory<'a> {
    let mut realizations = Vec::<Realization>::with_capacity(nodes.len());
    let mut cost = 0.0;

    for (index, node) in nodes.iter_mut().enumerate() {
        let node_initial_storage = if node.index == 0 {
            hydros_initial_storage.clone()
        } else {
            realizations.last().unwrap().hydros_final_storage.clone()
        };
        let realization = realize_uncertainties(
            node,
            &bus_loads[index], // loads for stage 'index' ordered by id
            &node_initial_storage,
            &hydros_inflow[index], // inflows for stage 'index' ordered by id
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
    node: &mut Node,
    node_forward_realization: &'a Realization,
) {
    let num_model_rows = node.subproblem.model.num_rows();
    let mut forward_rows = node_forward_realization.basis.rows().to_vec();
    let num_forward_rows = forward_rows.len();

    // checks if should add zeros to the rows (new cuts added)
    if num_forward_rows < num_model_rows {
        let row_diff = num_model_rows - num_forward_rows;
        forward_rows.append(&mut vec![0; row_diff]);
    } else if num_forward_rows > num_model_rows {
        forward_rows.truncate(num_model_rows);
    }

    node.subproblem.model.set_basis(
        Some(node_forward_realization.basis.columns()),
        Some(&forward_rows),
    );
}

/// Solves a node's subproblem for all it's branchings and
/// returns the solutions.
fn solve_all_branchings<'a>(
    node: &mut Node,
    num_branchings: usize,
    node_forward_realization: &'a Realization,
    node_saa: &'a Vec<Vec<f64>>, // indexed by stage | branching | hydro
) -> Vec<Realization<'a>> {
    let mut realizations = Vec::<Realization>::with_capacity(num_branchings);
    for hydros_inflow in node_saa.iter() {
        reuse_forward_basis(node, node_forward_realization);
        // hot_start_with_forward_solution(node, node_forward_realization);
        let realization = realize_uncertainties(
            node,
            node_forward_realization.bus_loads,
            &node_forward_realization.hydros_initial_storage,
            hydros_inflow,
        );
        realizations.push(realization);
    }

    realizations
}

/// Evaluates and returns the new cut to be added to a node from the
/// solutions of the node's subproblem for all branchings.
fn eval_average_cut(
    node: &Node,
    num_branchings: usize,
    branchings_realizations: &Vec<Realization>,
    node_forward_realization: &Realization,
) -> BendersCut {
    let num_hydros = node.system.meta.hydros_count;
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
    BendersCut::new(0, average_water_values.clone(), cut_rhs)
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
    nodes: &mut Vec<Node>,
    trajectory: &Trajectory,
    saa: &Vec<Vec<Vec<f64>>>, // indexed by stage | branching | hydro
    num_branchings: usize,
) -> f64 {
    let mut node_iter = nodes.iter_mut().rev().peekable();
    loop {
        let node = node_iter.next().unwrap();
        let index = node.index;
        let node_forward_realization =
            trajectory.realizations.get(index).unwrap();
        let node_saa = saa.get(index).unwrap();

        match node_iter.peek() {
            Some(_) => {
                let realizations = solve_all_branchings(
                    node,
                    num_branchings,
                    node_forward_realization,
                    node_saa,
                );

                let mut cut = eval_average_cut(
                    &node,
                    num_branchings,
                    &realizations,
                    node_forward_realization,
                );

                // println!("Node: {}", node.index);

                let state = node_iter
                    .peek_mut()
                    .unwrap()
                    .subproblem
                    .cut_selection(&mut cut, &node_forward_realization);

                // Updates visited states set
                node_iter.peek_mut().unwrap().subproblem.states.push(state);
            }
            None => {
                let realizations = solve_all_branchings(
                    node,
                    num_branchings,
                    node_forward_realization,
                    node_saa,
                );
                return eval_first_stage_bound(num_branchings, &realizations);
            }
        }
    }
}

/// Runs a single iteration, comprised of forward and backward passes,
/// of the SDDP algorithm.
fn iterate<'a>(
    graph: &mut Graph,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    hydros_inflow: &'a Vec<&'a Vec<f64>>,
    saa: &Vec<Vec<Vec<f64>>>,
    num_branchings: usize,
) -> (f64, f64, Duration) {
    let begin = Instant::now();

    let trajectory = forward(
        &mut graph.nodes,
        bus_loads,
        hydros_initial_storage,
        hydros_inflow,
    );

    let trajectory_cost = trajectory.cost;
    let first_stage_bound =
        backward(&mut graph.nodes, &trajectory, saa, num_branchings);

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
/// let saa = powers::sddp::generate_saa(&scenario_generator, num_hydros, num_stages, num_branchings);
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
    graph: &mut Graph,
    num_iterations: usize,
    num_branchings: usize,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    scenario_generator: &'a Vec<Vec<LogNormal<f64>>>,
) {
    let mut rng = Xoshiro256Plus::seed_from_u64(0);
    let forward_indices_dist =
        Uniform::<usize>::try_from(0..num_branchings).unwrap();

    let num_stages = graph.len();
    let num_hydros = hydros_initial_storage.len();
    let saa = generate_saa(
        scenario_generator,
        num_hydros,
        num_stages,
        num_branchings,
    );

    training_greeting(num_iterations, graph.len(), num_branchings);
    training_table_divider();
    training_table_header();
    training_table_divider();

    for index in 0..num_iterations {
        // Generates indices for sampling inflows, indexed by stage
        let forward_branching_indices: Vec<usize> = forward_indices_dist
            .sample_iter(&mut rng)
            .take(num_stages)
            .collect();

        let hydros_inflow = saa
            .iter()
            .enumerate()
            .map(|(index, stage_inflows)| {
                &stage_inflows[forward_branching_indices[index]]
            })
            .collect();

        let (simulation, lower_bound, time) = iterate(
            graph,
            bus_loads,
            hydros_initial_storage,
            &hydros_inflow,
            &saa,
            num_branchings,
        );

        training_table_row(index + 1, lower_bound, simulation, time);
    }

    training_table_divider();
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
        assert_eq!(subproblem.accessors.min_generation_slack.len(), 1);
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
            myhighs::HighsModelStatus::Optimal
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
    fn test_create_node() {
        let node = Node::new(0, System::default());
        assert_eq!(node.index, 0);
    }

    #[test]
    fn test_create_graph() {
        let node0 = Node::new(0, System::default());
        let graph = Graph::new(node0);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_append_node_to_graph() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_forward_with_default_system() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let node2 = Node::new(2, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        graph.append(node2);
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = vec![83.222];
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        forward(
            &mut graph.nodes,
            &bus_loads,
            &hydros_initial_storage,
            &hydros_inflow,
        );
    }

    #[test]
    fn test_backward_with_default_system() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let node2 = Node::new(2, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        graph.append(node2);
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = vec![83.222];
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let trajectory = forward(
            &mut graph.nodes,
            &bus_loads,
            &hydros_initial_storage,
            &hydros_inflow,
        );
        let branchings = vec![
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
        ];
        backward(&mut graph.nodes, &trajectory, &branchings, 3);
    }

    #[test]
    fn test_iterate_with_default_system() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let node2 = Node::new(2, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        graph.append(node2);
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = vec![83.222];
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let branchings = vec![
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
        ];
        iterate(
            &mut graph,
            &bus_loads,
            &hydros_initial_storage,
            &hydros_inflow,
            &branchings,
            3,
        );
    }

    #[test]
    fn test_train_with_default_system() {
        let node0 = Node::new(0, System::default());
        let mut graph = Graph::new(node0);
        let mut scenario_generator: Vec<Vec<LogNormal<f64>>> =
            vec![vec![LogNormal::new(3.6, 0.6928).unwrap()]];
        let mut bus_loads = vec![vec![75.0]];
        for n in 1..4 {
            let node = Node::new(n, System::default());
            graph.append(node);
            scenario_generator.push(vec![LogNormal::new(3.6, 0.6928).unwrap()]);
            bus_loads.push(vec![75.0]);
        }
        let hydros_initial_storage = vec![83.222];
        train(
            &mut graph,
            24,
            3,
            &bus_loads,
            &hydros_initial_storage,
            &scenario_generator,
        );
    }
}
