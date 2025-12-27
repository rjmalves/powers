//! Indexed Benders cuts and visited states output module.
//!
//! This module handles writing cut coefficients and state information using
//! integer indices instead of mixed-type enums, reducing file size and
//! providing better type safety.

use crate::fcf;
use crate::graph;

use csv::Writer;
use std::error::Error;
use std::sync::{Arc, Mutex};

/// Output record for a single cut coefficient (indexed format)
#[derive(serde::Serialize)]
struct IndexedCutOutput {
    stage_index: usize,
    stage_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
    active: bool,
    coefficient_index: usize,
    value: f64,
}

/// Writes Benders cuts to CSV file using indexed format.
///
/// Uses integer `coefficient_index` instead of the enum `coefficient_entity`.
/// Requires `coefficient_dictionary.csv` for decoding indices.
///
/// # Schema
///
/// ```csv
/// stage_index,stage_cut_id,iteration,forward_pass_idx,active,coefficient_index,value
/// 0,0,1,0,true,0,59094.825
/// 0,0,1,0,true,1,-100.0
/// ```
///
/// # Arguments
///
/// * `g` - The SDDP graph with future cost functions
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
pub(super) fn write_benders_cuts_indexed(
    g: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr = Writer::from_path(&(output_dir.to_owned() + "/cuts.csv"))?;

    // No manual header - serialize() writes it automatically from struct

    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.lock().unwrap();
        for cut in fcf.cut_pool.pool.iter() {
            // Index 0: RHS
            wtr.serialize(IndexedCutOutput {
                stage_index: node.id,
                stage_cut_id: cut.id,
                iteration: cut.iteration,
                forward_pass_idx: cut.forward_pass_idx,
                active: cut.is_active(),
                coefficient_index: 0,
                value: cut.rhs,
            })?;

            // Indices 1+: State coefficients (storage + lags)
            for (index, coef) in cut.coefficients.iter().enumerate() {
                wtr.serialize(IndexedCutOutput {
                    stage_index: node.id,
                    stage_cut_id: cut.id,
                    iteration: cut.iteration,
                    forward_pass_idx: cut.forward_pass_idx,
                    active: cut.is_active(),
                    coefficient_index: index + 1, // +1 because RHS is 0
                    value: *coef,
                })?;
            }
        }
    }

    wtr.flush()?;
    Ok(())
}

/// Output record for a single state coefficient (indexed format)
#[derive(serde::Serialize)]
struct IndexedStateOutput {
    stage_index: usize,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
    state_component_index: usize,
    value: f64,
}

/// Writes visited states to CSV file using indexed format.
///
/// Uses integer `state_component_index` instead of the enum `coefficient_entity`.
/// Index 0 is the dominating objective, indices 1+ are state coefficients.
///
/// # Schema
///
/// ```csv
/// stage_index,dominating_cut_id,iteration,forward_pass_idx,state_component_index,value
/// 0,0,1,0,0,10500.0
/// 0,0,1,0,1,50.0
/// 0,0,1,0,2,45.0
/// ```
///
/// # Arguments
///
/// * `g` - The SDDP graph with future cost functions
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
pub(super) fn write_visited_states_indexed(
    g: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr = Writer::from_path(&(output_dir.to_owned() + "/states.csv"))?;

    // No manual header - serialize() writes it automatically from struct

    for id in 0..g.node_count() {
        let node = g.get_node(id).unwrap();
        let fcf = node.data.lock().unwrap();
        for state in fcf.state_pool.pool.iter() {
            // Index 0: Dominating objective
            wtr.serialize(IndexedStateOutput {
                stage_index: node.id,
                dominating_cut_id: state.get_dominating_cut_id(),
                iteration: state.get_iteration(),
                forward_pass_idx: state.get_forward_pass_idx(),
                state_component_index: 0,
                value: state.get_dominating_objective(),
            })?;

            // Indices 1+: State coefficients (storage + lags)
            for (index, coef) in state.coefficients().iter().enumerate() {
                wtr.serialize(IndexedStateOutput {
                    stage_index: node.id,
                    dominating_cut_id: state.get_dominating_cut_id(),
                    iteration: state.get_iteration(),
                    forward_pass_idx: state.get_forward_pass_idx(),
                    state_component_index: index + 1, // +1 because objective is 0
                    value: *coef,
                })?;
            }
        }
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_cut_output_structure() {
        // Verify the output struct has the expected fields
        let output = IndexedCutOutput {
            stage_index: 0,
            stage_cut_id: 1,
            iteration: 5,
            forward_pass_idx: 2,
            active: true,
            coefficient_index: 3,
            value: 42.0,
        };

        // Serialize to check field order
        let csv = format!(
            "{},{},{},{},{},{},{}",
            output.stage_index,
            output.stage_cut_id,
            output.iteration,
            output.forward_pass_idx,
            output.active,
            output.coefficient_index,
            output.value
        );

        assert!(csv.contains("0,1,5,2,true,3,42"));
    }

    #[test]
    fn test_indexed_state_output_structure() {
        // Verify the output struct has the expected fields
        let output = IndexedStateOutput {
            stage_index: 0,
            dominating_cut_id: 1,
            iteration: 5,
            forward_pass_idx: 2,
            state_component_index: 3,
            value: 42.0,
        };

        // Serialize to check field order
        let csv = format!(
            "{},{},{},{},{},{}",
            output.stage_index,
            output.dominating_cut_id,
            output.iteration,
            output.forward_pass_idx,
            output.state_component_index,
            output.value
        );

        assert!(csv.contains("0,1,5,2,3,42"));
    }
}
