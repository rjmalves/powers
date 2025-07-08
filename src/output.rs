use crate::fcf;
use crate::graph;
use crate::sddp;

use csv::Writer;
use serde;
use std::error::Error;
use std::sync::{Arc, Mutex};

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
    g: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(&(path.to_owned() + "/cuts.csv"))?;
    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.lock().unwrap();
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
    g: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(&(path.to_owned() + "/states.csv"))?;
    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.lock().unwrap();
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
    simulation_handlers: &Vec<sddp::SddpSimulationHandler>,
    study_period_ids: &Vec<usize>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_buses.csv"))?;
    for (series_index, handler) in simulation_handlers.iter().enumerate() {
        for (stage_index, realization_id) in study_period_ids.iter().enumerate()
        {
            let realization =
                handler.get_realization_at_node(*realization_id).unwrap();
            let num_buses = realization.data.loads.len();
            for bus_index in 0..num_buses {
                wtr.serialize(BusSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: bus_index,
                    load: realization.data.loads[bus_index],
                    deficit: realization.data.deficit[bus_index],
                    marginal_cost: realization.data.marginal_cost[bus_index],
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
    simulation_handlers: &Vec<sddp::SddpSimulationHandler>,
    study_period_ids: &Vec<usize>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_lines.csv"))?;
    for (series_index, handler) in simulation_handlers.iter().enumerate() {
        for (stage_index, realization_id) in study_period_ids.iter().enumerate()
        {
            let realization =
                handler.get_realization_at_node(*realization_id).unwrap();
            let num_lines = realization.data.exchange.len();
            for line_index in 0..num_lines {
                wtr.serialize(LineSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: line_index,
                    exchange: realization.data.exchange[line_index],
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
    simulation_handlers: &Vec<sddp::SddpSimulationHandler>,
    study_period_ids: &Vec<usize>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_thermals.csv"))?;
    for (series_index, handler) in simulation_handlers.iter().enumerate() {
        for (stage_index, realization_id) in study_period_ids.iter().enumerate()
        {
            let realization =
                handler.get_realization_at_node(*realization_id).unwrap();
            let num_thermals = realization.data.thermal_generation.len();
            for thermal_index in 0..num_thermals {
                wtr.serialize(ThermalSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: thermal_index,
                    generation: realization.data.thermal_generation
                        [thermal_index],
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
    simulation_handlers: &Vec<sddp::SddpSimulationHandler>,
    study_period_ids: &Vec<usize>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr =
        Writer::from_path(&(path.to_owned() + "/simulation_hydros.csv"))?;
    for (series_index, handler) in simulation_handlers.iter().enumerate() {
        for (stage_index, realization_id) in study_period_ids.iter().enumerate()
        {
            let realization =
                handler.get_realization_at_node(*realization_id).unwrap();
            let num_hydros = realization.data.final_storage.len();
            for hydro_index in 0..num_hydros {
                wtr.serialize(HydroSimulationOutput {
                    stage_index,
                    series_index,
                    entity_index: hydro_index,
                    final_storage: realization.data.final_storage[hydro_index],
                    inflow: realization.data.inflow[hydro_index],
                    turbined_flow: realization.data.turbined_flow[hydro_index],
                    spillage: realization.data.spillage[hydro_index],
                    water_value: realization.data.water_value[hydro_index],
                })?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

pub fn generate_outputs(
    future_cost_function_graph: &graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    simulation_handlers: &Vec<sddp::SddpSimulationHandler>,
    study_period_ids: &Vec<usize>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    write_benders_cuts(future_cost_function_graph, path)?;
    write_visited_states(future_cost_function_graph, path)?;
    write_buses_simulation_results(
        simulation_handlers,
        study_period_ids,
        path,
    )?;
    write_lines_simulation_results(
        simulation_handlers,
        study_period_ids,
        path,
    )?;
    write_thermals_simulation_results(
        simulation_handlers,
        study_period_ids,
        path,
    )?;
    write_hydros_simulation_results(
        simulation_handlers,
        study_period_ids,
        path,
    )?;
    Ok(())
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::cut;
//     use std::fs;

//     #[test]
//     fn test_write_benders_cuts() {
//         let mut graph = graph::DirectedGraph::new();
//         let mut fcf = fcf::FutureCostFunction::new();
//         fcf.add_cut(cut::BendersCut::new(0, vec![1.5], 10.0));
//         graph.add_node(Arc::new(Mutex::new(fcf))).unwrap();
//         let dir = tempfile::tempdir().unwrap();
//         let path = dir.path().to_str().unwrap();

//         write_benders_cuts(&graph, path).unwrap();

//         let contents = fs::read_to_string(path.to_owned() + "/cuts.csv").unwrap();
//         let expected = "stage_index,stage_cut_id,active,coefficient_entity,value\n0,0,true,RHS,10.0\n0,0,true,Storage(0),1.5\n";
//         assert_eq!(contents, expected);
//     }
// }
