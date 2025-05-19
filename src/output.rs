use crate::graph;
use crate::sddp;
use crate::subproblem;
use csv::Writer;
use serde;
use std::error::Error;

#[derive(serde::Serialize)]
enum BendersCutCoefficientType {
    RHS,
    Storage(usize),
}

#[derive(serde::Serialize)]
struct BendersCutOutput {
    stage_index: usize,
    stage_cut_id: usize,
    active: bool,
    coefficient_entity: BendersCutCoefficientType,
    value: f64,
}

fn write_benders_cuts(
    g: &graph::DirectedGraph<sddp::NodeData>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(&(path.to_owned() + "/cuts.csv"))?;
    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.future_cost_function.lock().unwrap();
        for cut in fcf.cut_pool.pool.iter() {
            // Writes RHS
            wtr.serialize(BendersCutOutput {
                stage_index: node.id,
                stage_cut_id: cut.id,
                active: cut.active,
                coefficient_entity: BendersCutCoefficientType::RHS,
                value: cut.rhs,
            })?;
            // Writes coefficients
            for (index, coef) in cut.coefficients.iter().enumerate() {
                wtr.serialize(BendersCutOutput {
                    stage_index: node.id,
                    stage_cut_id: cut.id,
                    active: cut.active,
                    coefficient_entity: BendersCutCoefficientType::Storage(
                        index,
                    ),
                    value: *coef,
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
enum VisitedStateCoefficientType {
    DominatingObjective,
    Storage(usize),
}

#[derive(serde::Serialize)]
struct VisitedStateOutput {
    stage_index: usize,
    dominating_cut_id: usize,
    coefficient_entity: VisitedStateCoefficientType,
    value: f64,
}

fn write_visited_states(
    g: &graph::DirectedGraph<sddp::NodeData>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(&(path.to_owned() + "/states.csv"))?;
    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.future_cost_function.lock().unwrap();
        for state in fcf.state_pool.pool.iter() {
            // Writes dominating objective for state
            wtr.serialize(VisitedStateOutput {
                stage_index: node.id,
                dominating_cut_id: state.get_dominating_cut_id(),
                coefficient_entity:
                    VisitedStateCoefficientType::DominatingObjective,
                value: state.get_dominating_objective(),
            })?;
            // Writes state variables values
            for (index, coef) in state.coefficients().iter().enumerate() {
                wtr.serialize(VisitedStateOutput {
                    stage_index: node.id,
                    dominating_cut_id: state.get_dominating_cut_id(),
                    coefficient_entity: VisitedStateCoefficientType::Storage(
                        index,
                    ),
                    value: *coef,
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct BusSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    load: f64,
    deficit: f64,
    marginal_cost: f64,
}

fn write_buses_simulation_results(
    trajectories: &Vec<subproblem::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_buses.csv"))?;
    for (trajectory_index, trajectory) in trajectories.iter().enumerate() {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            let num_buses = realization.loads.len();
            for bus_index in 0..num_buses {
                wtr.serialize(BusSimulationOutput {
                    stage_index,
                    series_index: trajectory_index,
                    entity_index: bus_index,
                    load: realization.loads[bus_index],
                    deficit: realization.deficit[bus_index],
                    marginal_cost: realization.marginal_cost[bus_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct LineSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    exchange: f64,
}

fn write_lines_simulation_results(
    trajectories: &Vec<subproblem::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_lines.csv"))?;
    for (trajectory_index, trajectory) in trajectories.iter().enumerate() {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            let num_lines = realization.exchange.len();
            for line_index in 0..num_lines {
                wtr.serialize(LineSimulationOutput {
                    stage_index,
                    series_index: trajectory_index,
                    entity_index: line_index,
                    exchange: realization.exchange[line_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct ThermalSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    generation: f64,
}

fn write_thermals_simulation_results(
    trajectories: &Vec<subproblem::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_thermals.csv"))?;
    for (trajectory_index, trajectory) in trajectories.iter().enumerate() {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            let num_thermals = realization.thermal_generation.len();
            for thermal_index in 0..num_thermals {
                wtr.serialize(ThermalSimulationOutput {
                    stage_index,
                    series_index: trajectory_index,
                    entity_index: thermal_index,
                    generation: realization.thermal_generation[thermal_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}
#[derive(serde::Serialize)]
struct HydroSimulationOutput {
    stage_index: usize,
    series_index: usize,
    entity_index: usize,
    final_storage: f64,
    inflow: f64,
    turbined_flow: f64,
    spillage: f64,
    water_value: f64,
}

fn write_hydros_simulation_results(
    trajectories: &Vec<subproblem::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_hydros.csv"))?;
    for (trajectory_index, trajectory) in trajectories.iter().enumerate() {
        for (stage_index, realization) in
            trajectory.realizations[1..].iter().enumerate()
        {
            let num_hydros = realization.final_storage.len();
            for hydro_index in 0..num_hydros {
                wtr.serialize(HydroSimulationOutput {
                    stage_index,
                    series_index: trajectory_index,
                    entity_index: hydro_index,
                    final_storage: realization.final_storage[hydro_index],
                    inflow: realization.inflow[hydro_index],
                    turbined_flow: realization.turbined_flow[hydro_index],
                    spillage: realization.spillage[hydro_index],
                    water_value: realization.water_value[hydro_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

pub fn generate_outputs(
    g: &graph::DirectedGraph<sddp::NodeData>,
    trajectories: &Vec<subproblem::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    write_benders_cuts(g, path)?;
    write_visited_states(g, path)?;
    write_buses_simulation_results(trajectories, path)?;
    write_lines_simulation_results(trajectories, path)?;
    write_thermals_simulation_results(trajectories, path)?;
    write_hydros_simulation_results(trajectories, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_write_benders_cuts() {}
}
