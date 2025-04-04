use crate::sddp;
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
    graph: &sddp::Graph,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(&(path.to_owned() + "/cuts.csv"))?;
    for node in graph.nodes.iter() {
        for cut in node.subproblem.cuts.iter() {
            // Writes RHS
            wtr.serialize(BendersCutOutput {
                stage_index: node.index,
                stage_cut_id: cut.id,
                active: cut.active,
                coefficient_entity: BendersCutCoefficientType::RHS,
                value: cut.rhs,
            })?;
            // Writes coefficients
            for (index, coef) in cut.coefficients.iter().enumerate() {
                wtr.serialize(BendersCutOutput {
                    stage_index: node.index,
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
    graph: &sddp::Graph,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(&(path.to_owned() + "/states.csv"))?;
    for node in graph.nodes.iter() {
        for state in node.subproblem.states.iter() {
            // Writes dominating objective for state
            wtr.serialize(VisitedStateOutput {
                stage_index: node.index,
                dominating_cut_id: state.dominating_cut_id,
                coefficient_entity:
                    VisitedStateCoefficientType::DominatingObjective,
                value: state.dominating_objective,
            })?;
            // Writes state variables values
            for (index, coef) in state.state.iter().enumerate() {
                wtr.serialize(VisitedStateOutput {
                    stage_index: node.index,
                    dominating_cut_id: state.dominating_cut_id,
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
    entity_index: usize,
    load: f64,
    deficit: f64,
    marginal_cost: f64,
}

fn write_buses_simulation_results(
    trajectories: &Vec<sddp::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_buses.csv"))?;
    for trajectory in trajectories.iter() {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            let num_buses = realization.bus_loads.len();
            for bus_index in 0..num_buses {
                wtr.serialize(BusSimulationOutput {
                    stage_index,
                    entity_index: bus_index,
                    load: realization.bus_loads[bus_index],
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
    entity_index: usize,
    exchange: f64,
}

fn write_lines_simulation_results(
    trajectories: &Vec<sddp::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_lines.csv"))?;
    for trajectory in trajectories.iter() {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            let num_lines = realization.exchange.len();
            for line_index in 0..num_lines {
                wtr.serialize(LineSimulationOutput {
                    stage_index,
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
    entity_index: usize,
    generation: f64,
}

fn write_thermals_simulation_results(
    trajectories: &Vec<sddp::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_thermals.csv"))?;
    for trajectory in trajectories.iter() {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            let num_thermals = realization.thermal_generation.len();
            for thermal_index in 0..num_thermals {
                wtr.serialize(ThermalSimulationOutput {
                    stage_index,
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
    entity_index: usize,
    initial_storage: f64,
    final_storage: f64,
    inflow: f64,
    turbined_flow: f64,
    spillage: f64,
    water_value: f64,
}

fn write_hydros_simulation_results(
    trajectories: &Vec<sddp::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_hydros.csv"))?;
    for trajectory in trajectories.iter() {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            let num_hydros = realization.hydros_initial_storage.len();
            for hydro_index in 0..num_hydros {
                wtr.serialize(HydroSimulationOutput {
                    stage_index,
                    entity_index: hydro_index,
                    initial_storage: realization.hydros_initial_storage
                        [hydro_index],
                    final_storage: realization.hydros_final_storage
                        [hydro_index],
                    inflow: realization.inflow[hydro_index],
                    turbined_flow: realization.turbined_flow[hydro_index],
                    spillage: realization.spillage[hydro_index],
                    water_value: realization.water_values[hydro_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

pub fn generate_outputs(
    graph: &sddp::Graph,
    trajectories: &Vec<sddp::Trajectory>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    write_benders_cuts(graph, path)?;
    write_visited_states(graph, path)?;
    write_buses_simulation_results(trajectories, path)?;
    write_lines_simulation_results(trajectories, path)?;
    write_thermals_simulation_results(trajectories, path)?;
    write_hydros_simulation_results(trajectories, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_benders_cuts() {}
}
