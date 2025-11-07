use crate::error::{PowersError, ValidationError};
use crate::input::{Config, GraphInput, Recourse, SystemInput};
use std::collections::{HashMap, HashSet};

/// Input validation utilities for comprehensive error checking.
pub struct InputValidator;

impl InputValidator {
    /// Validate config with minimal checks
    pub fn validate_config_minimal(config: &Config) -> Result<(), PowersError> {
        if config.num_iterations == 0 {
            return Err(Box::new(ValidationError::InvalidFieldValue {
                file: "config.json".to_string(),
                field: "num_iterations".to_string(),
                value: config.num_iterations.to_string(),
                constraint: "must be positive (> 0)".to_string(),
                suggestion: "Set num_iterations to at least 1".to_string(),
            })
            .into());
        }

        if config.num_forward_passes == 0 {
            return Err(Box::new(ValidationError::InvalidFieldValue {
                file: "config.json".to_string(),
                field: "num_forward_passes".to_string(),
                value: config.num_forward_passes.to_string(),
                constraint: "must be positive (> 0)".to_string(),
                suggestion: "Set num_forward_passes to at least 1".to_string(),
            })
            .into());
        }

        // Validate num_simulation_scenarios if provided
        if let Some(num_sim) = config.num_simulation_scenarios {
            if num_sim == 0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                file: "config.json".to_string(),
                field: "num_simulation_scenarios".to_string(),
                value: num_sim.to_string(),
                constraint: "must be positive (> 0) when provided".to_string(),
                suggestion: "Set num_simulation_scenarios to at least 1, or omit/set to null to skip simulation"
                    .to_string(),
            })
            .into());
            }
        }

        Ok(())
    }

    pub fn validate_system(system: &SystemInput) -> Result<(), PowersError> {
        let bus_ids: Vec<usize> = system.buses.iter().map(|b| b.id).collect();
        Self::validate_id_range_comprehensive(
            &bus_ids,
            "buses",
            "system.json",
        )?;
        let bus_id_set: HashSet<usize> = bus_ids.iter().copied().collect();

        let line_ids: Vec<usize> = system.lines.iter().map(|l| l.id).collect();
        Self::validate_id_range_comprehensive(
            &line_ids,
            "lines",
            "system.json",
        )?;

        for line in &system.lines {
            if !bus_id_set.contains(&line.source_bus_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "system.json".to_string(),
                    context: format!("line {}", line.id),
                    ref_type: "source_bus_id".to_string(),
                    ref_id: line.source_bus_id.to_string(),
                    available: bus_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    suggestion:
                        "Check that source_bus_id matches an existing bus id"
                            .to_string(),
                })
                .into());
            }
            if !bus_id_set.contains(&line.target_bus_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "system.json".to_string(),
                    context: format!("line {}", line.id),
                    ref_type: "target_bus_id".to_string(),
                    ref_id: line.target_bus_id.to_string(),
                    available: bus_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    suggestion:
                        "Check that target_bus_id matches an existing bus id"
                            .to_string(),
                })
                .into());
            }

            if line.direct_capacity < 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "system.json".to_string(),
                    field: format!("lines[{}].direct_capacity", line.id),
                    value: line.direct_capacity.to_string(),
                    constraint: "must be non-negative (>= 0)".to_string(),
                    suggestion: "Set direct_capacity to 0 or a positive value"
                        .to_string(),
                })
                .into());
            }
            if line.reverse_capacity < 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "system.json".to_string(),
                    field: format!("lines[{}].reverse_capacity", line.id),
                    value: line.reverse_capacity.to_string(),
                    constraint: "must be non-negative (>= 0)".to_string(),
                    suggestion: "Set reverse_capacity to 0 or a positive value"
                        .to_string(),
                })
                .into());
            }
            if line.exchange_penalty < 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "system.json".to_string(),
                    field: format!("lines[{}].exchange_penalty", line.id),
                    value: line.exchange_penalty.to_string(),
                    constraint: "must be non-negative (>= 0)".to_string(),
                    suggestion: "Set exchange_penalty to 0 or a positive value"
                        .to_string(),
                })
                .into());
            }
        }

        let thermal_ids: Vec<usize> =
            system.thermals.iter().map(|t| t.id).collect();
        Self::validate_id_range_comprehensive(
            &thermal_ids,
            "thermals",
            "system.json",
        )?;

        for thermal in &system.thermals {
            if !bus_id_set.contains(&thermal.bus_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "system.json".to_string(),
                    context: format!("thermal {}", thermal.id),
                    ref_type: "bus_id".to_string(),
                    ref_id: thermal.bus_id.to_string(),
                    available: bus_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    suggestion: "Check that bus_id matches an existing bus id"
                        .to_string(),
                })
                .into());
            }

            if thermal.cost < 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "system.json".to_string(),
                    field: format!("thermals[{}].cost", thermal.id),
                    value: thermal.cost.to_string(),
                    constraint: "must be non-negative (>= 0)".to_string(),
                    suggestion: "Set cost to 0 or a positive value".to_string(),
                })
                .into());
            }

            if thermal.min_generation > thermal.max_generation {
                return Err(Box::new(ValidationError::ConstraintViolation {
                    file: "system.json".to_string(),
                    context: format!("thermal {}", thermal.id),
                    constraint: "min_generation must be <= max_generation"
                        .to_string(),
                    details: format!(
                        "min_generation={}, max_generation={}",
                        thermal.min_generation, thermal.max_generation
                    ),
                    suggestion: format!(
                        "Set min_generation <= {} or increase max_generation",
                        thermal.max_generation
                    ),
                })
                .into());
            }
        }

        let hydro_ids: Vec<usize> =
            system.hydros.iter().map(|h| h.id).collect();
        Self::validate_id_range_comprehensive(
            &hydro_ids,
            "hydros",
            "system.json",
        )?;
        let hydro_id_set: HashSet<usize> = hydro_ids.iter().copied().collect();

        for hydro in &system.hydros {
            if !bus_id_set.contains(&hydro.bus_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "system.json".to_string(),
                    context: format!("hydro {}", hydro.id),
                    ref_type: "bus_id".to_string(),
                    ref_id: hydro.bus_id.to_string(),
                    available: bus_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    suggestion: "Check that bus_id matches an existing bus id"
                        .to_string(),
                })
                .into());
            }

            if hydro.productivity <= 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "system.json".to_string(),
                    field: format!("hydros[{}].productivity", hydro.id),
                    value: hydro.productivity.to_string(),
                    constraint: "must be positive (> 0)".to_string(),
                    suggestion: "Set productivity to a positive value"
                        .to_string(),
                })
                .into());
            }

            if hydro.min_storage > hydro.max_storage {
                return Err(Box::new(ValidationError::ConstraintViolation {
                    file: "system.json".to_string(),
                    context: format!("hydro {}", hydro.id),
                    constraint: "min_storage must be <= max_storage"
                        .to_string(),
                    details: format!(
                        "min_storage={}, max_storage={}",
                        hydro.min_storage, hydro.max_storage
                    ),
                    suggestion: format!(
                        "Set min_storage <= {} or increase max_storage",
                        hydro.max_storage
                    ),
                })
                .into());
            }

            if hydro.min_turbined_flow > hydro.max_turbined_flow {
                return Err(Box::new(ValidationError::ConstraintViolation {
                    file: "system.json".to_string(),
                    context: format!("hydro {}", hydro.id),
                    constraint: "min_turbined_flow must be <= max_turbined_flow".to_string(),
                    details: format!("min_turbined_flow={}, max_turbined_flow={}", hydro.min_turbined_flow, hydro.max_turbined_flow),
                    suggestion: format!("Set min_turbined_flow <= {} or increase max_turbined_flow", hydro.max_turbined_flow),
                })
                .into());
            }

            if let Some(downstream_id) = hydro.downstream_hydro_id {
                if !hydro_id_set.contains(&downstream_id) {
                    return Err(Box::new(ValidationError::InvalidReference {
                        file: "system.json".to_string(),
                        context: format!("hydro {}", hydro.id),
                        ref_type: "downstream_hydro_id".to_string(),
                        ref_id: downstream_id.to_string(),
                        available: hydro_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "),
                        suggestion: "Check that downstream_hydro_id matches an existing hydro id".to_string(),
                    })
                    .into());
                }
            }
        }

        Ok(())
    }

    /// Helper: Validate ID range is sequential from 0 with no gaps or duplicates.
    fn validate_id_range_comprehensive(
        ids: &[usize],
        entity_name: &str,
        file: &str,
    ) -> Result<(), PowersError> {
        if ids.is_empty() {
            return Ok(());
        }

        let id_set: HashSet<usize> = ids.iter().copied().collect();
        if id_set.len() != ids.len() {
            return Err(Box::new(ValidationError::ConstraintViolation {
                file: file.to_string(),
                context: entity_name.to_string(),
                constraint: "IDs must be unique".to_string(),
                details: format!(
                    "Found {} IDs but only {} unique",
                    ids.len(),
                    id_set.len()
                ),
                suggestion: format!(
                    "Check for duplicate IDs in {} array",
                    entity_name
                ),
            })
            .into());
        }

        let max_id = *ids.iter().max().unwrap();
        if max_id != ids.len() - 1 {
            return Err(Box::new(ValidationError::InvalidFieldValue {
                file: file.to_string(),
                field: format!("{} IDs", entity_name),
                value: format!("max_id={}, count={}", max_id, ids.len()),
                constraint: "IDs must be sequential from 0 (0, 1, 2, ...)"
                    .to_string(),
                suggestion: format!(
                    "Ensure {} IDs are sequential from 0 to {} with no gaps",
                    entity_name,
                    ids.len() - 1
                ),
            })
            .into());
        }

        for expected_id in 0..=max_id {
            if !id_set.contains(&expected_id) {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: file.to_string(),
                    field: format!("{} IDs", entity_name),
                    value: format!("missing ID {}", expected_id),
                    constraint: "IDs must be sequential from 0 with no gaps"
                        .to_string(),
                    suggestion: format!(
                        "Add {} with id={} or renumber to remove gap",
                        entity_name, expected_id
                    ),
                })
                .into());
            }
        }

        Ok(())
    }

    /// Validate graph input for node uniqueness, edges, and probabilities.
    pub fn validate_graph(graph: &GraphInput) -> Result<(), PowersError> {
        if graph.nodes.is_empty() {
            return Ok(()); // Empty graph is valid
        }

        let node_ids: Vec<usize> = graph.nodes.iter().map(|n| n.id).collect();
        let node_id_set: HashSet<usize> = node_ids.iter().copied().collect();
        if node_id_set.len() != node_ids.len() {
            return Err(Box::new(ValidationError::ConstraintViolation {
                file: "graph.json".to_string(),
                context: "nodes".to_string(),
                constraint: "node IDs must be unique".to_string(),
                details: format!(
                    "Found {} nodes but only {} unique IDs",
                    node_ids.len(),
                    node_id_set.len()
                ),
                suggestion: "Check for duplicate node IDs in graph.json"
                    .to_string(),
            })
            .into());
        }

        let stage_ids: Vec<usize> =
            graph.nodes.iter().map(|n| n.stage_id).collect();
        let unique_stages: HashSet<usize> = stage_ids.iter().copied().collect();
        let max_stage = *unique_stages.iter().max().unwrap();
        if unique_stages.len() != max_stage + 1 {
            return Err(Box::new(ValidationError::InvalidFieldValue {
                file: "graph.json".to_string(),
                field: "stage_id".to_string(),
                value: format!(
                    "max_stage={}, unique_count={}",
                    max_stage,
                    unique_stages.len()
                ),
                constraint: "stage IDs must be sequential from 0".to_string(),
                suggestion: format!(
                    "Ensure stage IDs are sequential from 0 to {} with no gaps",
                    max_stage
                ),
            })
            .into());
        }

        let valid_risk_measures = ["expectation", "cvar", "worstcase"];
        for node in &graph.nodes {
            let risk_lower = node.risk_measure.to_lowercase();
            if !valid_risk_measures.contains(&risk_lower.as_str()) {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "graph.json".to_string(),
                    field: format!("nodes[{}].risk_measure", node.id),
                    value: node.risk_measure.clone(),
                    constraint: "must be one of: expectation, cvar, worstcase"
                        .to_string(),
                    suggestion: format!(
                        "Valid options: {}",
                        valid_risk_measures.join(", ")
                    ),
                })
                .into());
            }
        }

        for edge in &graph.edges {
            if !node_id_set.contains(&edge.source_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "graph.json".to_string(),
                    context: format!(
                        "edge from {} to {}",
                        edge.source_id, edge.target_id
                    ),
                    ref_type: "source_id".to_string(),
                    ref_id: edge.source_id.to_string(),
                    available: node_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    suggestion:
                        "Check that source_id matches an existing node id"
                            .to_string(),
                })
                .into());
            }

            if !node_id_set.contains(&edge.target_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "graph.json".to_string(),
                    context: format!(
                        "edge from {} to {}",
                        edge.source_id, edge.target_id
                    ),
                    ref_type: "target_id".to_string(),
                    ref_id: edge.target_id.to_string(),
                    available: node_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    suggestion:
                        "Check that target_id matches an existing node id"
                            .to_string(),
                })
                .into());
            }

            if edge.probability <= 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "graph.json".to_string(),
                    field: format!(
                        "edge from {} to {}: probability",
                        edge.source_id, edge.target_id
                    ),
                    value: edge.probability.to_string(),
                    constraint: "must be positive (> 0)".to_string(),
                    suggestion: "Set probability to a value between 0 and 1"
                        .to_string(),
                })
                .into());
            }
            if edge.probability > 1.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "graph.json".to_string(),
                    field: format!(
                        "edge from {} to {}: probability",
                        edge.source_id, edge.target_id
                    ),
                    value: edge.probability.to_string(),
                    constraint: "must be <= 1".to_string(),
                    suggestion: "Set probability to a value between 0 and 1"
                        .to_string(),
                })
                .into());
            }

            if edge.discount_rate < 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "graph.json".to_string(),
                    field: format!(
                        "edge from {} to {}: discount_rate",
                        edge.source_id, edge.target_id
                    ),
                    value: edge.discount_rate.to_string(),
                    constraint: "must be non-negative (>= 0)".to_string(),
                    suggestion: "Set discount_rate to 0 or a positive value"
                        .to_string(),
                })
                .into());
            }
        }

        let mut edges_by_source: HashMap<usize, Vec<&_>> = HashMap::new();
        for edge in &graph.edges {
            edges_by_source
                .entry(edge.source_id)
                .or_insert_with(Vec::new)
                .push(edge);
        }

        for (source_id, edges) in edges_by_source {
            let sum: f64 = edges.iter().map(|e| e.probability).sum();
            let tolerance = 1e-6;
            if (sum - 1.0).abs() > tolerance {
                let edge_details: Vec<String> = edges
                    .iter()
                    .map(|e| format!("→{} (p={})", e.target_id, e.probability))
                    .collect();
                return Err(Box::new(ValidationError::ConstraintViolation {
                    file: "graph.json".to_string(),
                    context: format!("node {} outgoing edges", source_id),
                    constraint: "outgoing edge probabilities must sum to 1.0 (±1e-6)".to_string(),
                    details: format!("sum={}, edges: {}", sum, edge_details.join(", ")),
                    suggestion: format!("Adjust probabilities so they sum to 1.0 (current sum: {})", sum),
                })
                .into());
            }
        }

        Ok(())
    }

    pub fn validate_recourse(
        recourse: &Recourse,
        system: &SystemInput,
    ) -> Result<(), PowersError> {
        // Validate uncertainty_specifications (new format)
        if recourse.uncertainty_specifications.is_empty() {
            return Err(PowersError::Validation(Box::new(ValidationError::MissingField {
                file: "recourse.json".to_string(),
                field: "uncertainty_specifications".to_string(),
                suggestion: "Add uncertainty_specifications array to define noise models".to_string(),
            })));
        }

        // Validate entity_id references exist in system
        // Build available entity sets
        let hydro_ids: HashSet<usize> =
            system.hydros.iter().map(|h| h.id).collect();
        let bus_ids: HashSet<usize> =
            system.buses.iter().map(|b| b.id).collect();
        let num_hydros = system.hydros.len();
        let num_buses = system.buses.len();

        for (idx, spec) in
            recourse.uncertainty_specifications.iter().enumerate()
        {
            match spec.uncertainty_type {
                crate::input::UncertaintyType::Inflow => {
                    // Inflow uncertainty must reference an existing hydro
                    if !hydro_ids.contains(&spec.entity_id) {
                        return Err(Box::new(ValidationError::InvalidReference {
                            file: "recourse.json".to_string(),
                            context: format!("uncertainty_specifications[{}] (inflow)", idx),
                            ref_type: "entity_id".to_string(),
                            ref_id: spec.entity_id.to_string(),
                            available: format!(
                                "hydros: 0..{} ({})",
                                num_hydros,
                                hydro_ids.iter()
                                    .map(|id| id.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                            suggestion: format!(
                                "Inflow entity_id must match an existing hydro_id in system.json (found {} hydros)",
                                num_hydros
                            ),
                        }).into());
                    }
                }
                crate::input::UncertaintyType::Load => {
                    // Load uncertainty must reference an existing bus
                    if !bus_ids.contains(&spec.entity_id) {
                        return Err(Box::new(ValidationError::InvalidReference {
                            file: "recourse.json".to_string(),
                            context: format!("uncertainty_specifications[{}] (load)", idx),
                            ref_type: "entity_id".to_string(),
                            ref_id: spec.entity_id.to_string(),
                            available: format!(
                                "buses: 0..{} ({})",
                                num_buses,
                                bus_ids.iter()
                                    .map(|id| id.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                            suggestion: format!(
                                "Load entity_id must match an existing bus_id in system.json (found {} buses)",
                                num_buses
                            ),
                        }).into());
                    }
                }
            }
        }

        // Validate initial_condition storage references
        for storage in &recourse.initial_condition.storage {
            if !hydro_ids.contains(&storage.hydro_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "recourse.json".to_string(),
                    context: "initial_condition.storage".to_string(),
                    ref_type: "hydro_id".to_string(),
                    ref_id: storage.hydro_id.to_string(),
                    available: format!(
                        "hydros: 0..{} ({})",
                        num_hydros,
                        hydro_ids.iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    suggestion: format!(
                        "Initial storage hydro_id must match an existing hydro in system.json (found {} hydros)",
                        num_hydros
                    ),
                }).into());
            }
        }

        // Validate initial_condition inflow references
        for inflow in &recourse.initial_condition.inflow {
            if !hydro_ids.contains(&inflow.hydro_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "recourse.json".to_string(),
                    context: "initial_condition.inflow".to_string(),
                    ref_type: "hydro_id".to_string(),
                    ref_id: inflow.hydro_id.to_string(),
                    available: format!(
                        "hydros: 0..{} ({})",
                        num_hydros,
                        hydro_ids.iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    suggestion: format!(
                        "Initial inflow hydro_id must match an existing hydro in system.json (found {} hydros)",
                        num_hydros
                    ),
                }).into());
            }
        }

        Ok(())
    }

    pub fn validate_consistency(
        _config: &Config,
        _system: &SystemInput,
        graph: &GraphInput,
        recourse: &Recourse,
    ) -> Result<(), PowersError> {
        // Collect all season_ids used in graph
        let graph_seasons: HashSet<usize> =
            graph.nodes.iter().map(|node| node.season_id).collect();

        // Validate season_id references in seasonal_distributions
        for (idx, spec) in
            recourse.uncertainty_specifications.iter().enumerate()
        {
            if let Some(seasonal_dists) = &spec.seasonal_distributions {
                for dist in seasonal_dists {
                    if !graph_seasons.contains(&dist.season_id) {
                        return Err(Box::new(ValidationError::InvalidReference {
                            file: "recourse.json".to_string(),
                            context: format!(
                                "uncertainty_specifications[{}].seasonal_distributions",
                                idx
                            ),
                            ref_type: "season_id".to_string(),
                            ref_id: dist.season_id.to_string(),
                            available: format!(
                                "seasons in graph: {}",
                                {
                                    let mut seasons: Vec<usize> = graph_seasons.iter().copied().collect();
                                    seasons.sort();
                                    seasons.iter()
                                        .map(|s| s.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                }
                            ),
                            suggestion: "season_id in seasonal_distributions must match a season_id used in graph.json nodes".to_string(),
                        }).into());
                    }
                }
            }
        }

        Ok(())
    }

    pub fn validate_all(
        config: &Config,
        system: &SystemInput,
        graph: &GraphInput,
        recourse: &Recourse,
    ) -> Result<(), PowersError> {
        Self::validate_config_minimal(config)?;
        Self::validate_system(system)?;
        Self::validate_graph(graph)?;
        Self::validate_recourse(recourse, system)?;
        Self::validate_consistency(config, system, graph, recourse)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::*;

    #[test]
    fn test_validate_id_range_comprehensive_valid_sequential() {
        // Valid sequential IDs from 0
        let ids = vec![0, 1, 2, 3, 4];
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_id_range_comprehensive_empty() {
        // Empty array is valid
        let ids: Vec<usize> = vec![];
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_id_range_comprehensive_single() {
        // Single element [0] is valid
        let ids = vec![0];
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_id_range_comprehensive_duplicates() {
        // Duplicate IDs should fail
        let ids = vec![0, 1, 2, 1, 3];
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unique"));
    }

    #[test]
    fn test_validate_id_range_comprehensive_gap() {
        // Gap in sequence should fail
        let ids = vec![0, 1, 3, 4]; // Missing 2
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("gap")
                || err.to_string().contains("missing")
        );
    }

    #[test]
    fn test_validate_id_range_comprehensive_not_starting_from_zero() {
        // IDs not starting from 0 should fail
        let ids = vec![1, 2, 3];
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_id_range_comprehensive_unordered_but_complete() {
        // Unordered but complete sequence should pass
        let ids = vec![2, 0, 3, 1, 4];
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_id_range_comprehensive_large_gap() {
        // Large number with gap should fail
        let ids = vec![0, 1, 2, 100]; // Huge gap
        let result = InputValidator::validate_id_range_comprehensive(
            &ids,
            "entities",
            "test.json",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_system_invalid_line_source_bus() {
        // Test InvalidReference for line source_bus_id
        let system = SystemInput {
            buses: vec![BusInput {
                id: 0,
                deficit_cost: 1000.0,
            }],
            lines: vec![LineInput {
                id: 0,
                source_bus_id: 99, // Invalid reference
                target_bus_id: 0,
                direct_capacity: 100.0,
                reverse_capacity: 100.0,
                exchange_penalty: 1.0,
            }],
            thermals: vec![],
            hydros: vec![],
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("source_bus_id") || err_str.contains("99"));
    }

    #[test]
    fn test_validate_system_invalid_line_target_bus() {
        // Test InvalidReference for line target_bus_id
        let system = SystemInput {
            buses: vec![BusInput {
                id: 0,
                deficit_cost: 1000.0,
            }],
            lines: vec![LineInput {
                id: 0,
                source_bus_id: 0,
                target_bus_id: 88, // Invalid reference
                direct_capacity: 100.0,
                reverse_capacity: 100.0,
                exchange_penalty: 1.0,
            }],
            thermals: vec![],
            hydros: vec![],
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("target_bus_id") || err_str.contains("88"));
    }

    #[test]
    fn test_validate_system_negative_direct_capacity() {
        // Test InvalidFieldValue for negative direct_capacity
        let system = SystemInput {
            buses: vec![
                BusInput {
                    id: 0,
                    deficit_cost: 1000.0,
                },
                BusInput {
                    id: 1,
                    deficit_cost: 1000.0,
                },
            ],
            lines: vec![LineInput {
                id: 0,
                source_bus_id: 0,
                target_bus_id: 1,
                direct_capacity: -50.0, // Negative capacity
                reverse_capacity: 100.0,
                exchange_penalty: 1.0,
            }],
            thermals: vec![],
            hydros: vec![],
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("direct_capacity") && err_str.contains("negative")
        );
    }

    #[test]
    fn test_validate_system_invalid_thermal_bus() {
        // Test InvalidReference for thermal bus_id
        let system = SystemInput {
            buses: vec![BusInput {
                id: 0,
                deficit_cost: 1000.0,
            }],
            lines: vec![],
            thermals: vec![ThermalInput {
                id: 0,
                bus_id: 77, // Invalid reference
                cost: 10.0,
                min_generation: 0.0,
                max_generation: 100.0,
            }],
            hydros: vec![],
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("bus_id") || err_str.contains("77"));
    }

    #[test]
    fn test_validate_system_negative_thermal_cost() {
        // Test InvalidFieldValue for negative thermal cost
        let system = SystemInput {
            buses: vec![BusInput {
                id: 0,
                deficit_cost: 1000.0,
            }],
            lines: vec![],
            thermals: vec![ThermalInput {
                id: 0,
                bus_id: 0,
                cost: -5.0, // Negative cost
                min_generation: 0.0,
                max_generation: 100.0,
            }],
            hydros: vec![],
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("cost") && err_str.contains("negative"));
    }

    #[test]
    fn test_validate_config_with_some_simulation() {
        // Valid config with Some(num_simulation_scenarios)
        let config = Config {
            num_iterations: 10,
            num_forward_passes: 4,
            num_simulation_scenarios: Some(100),
            seed: 42,
            num_threads: None,
            output_path: None,
            enable_cut_selection: true,
            export_sampled_noises_training: false,
        };
        let result = InputValidator::validate_config_minimal(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_config_with_none_simulation() {
        // Valid config with None (simulation skipped)
        let config = Config {
            num_iterations: 10,
            num_forward_passes: 4,
            num_simulation_scenarios: None,
            seed: 42,
            num_threads: None,
            output_path: None,
            enable_cut_selection: true,
            export_sampled_noises_training: false,
        };
        let result = InputValidator::validate_config_minimal(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_config_rejects_zero_simulation() {
        // Invalid: Some(0) should be rejected
        let config = Config {
            num_iterations: 10,
            num_forward_passes: 4,
            num_simulation_scenarios: Some(0),
            seed: 42,
            num_threads: None,
            output_path: None,
            enable_cut_selection: true,
            export_sampled_noises_training: false,
        };
        let result = InputValidator::validate_config_minimal(&config);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("num_simulation_scenarios"));
        assert!(err_str.contains("positive") || err_str.contains("> 0"));
    }
}
