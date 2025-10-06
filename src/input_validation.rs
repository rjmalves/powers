// T3.10: Comprehensive Input Validation
//
// This module contains validation logic for all input types.
// Performance target: <100μs total overhead for typical inputs.

use crate::error::{PowersError, ValidationError};
use crate::input::{Config, GraphInput, Recourse, SystemInput};
use std::collections::{HashMap, HashSet};

/// Input validation utilities for comprehensive error checking.
pub struct InputValidator;

impl InputValidator {
    /// Validate config with minimal checks (T3.7 Phase 1).
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

        if config.num_simulation_scenarios == 0 {
            return Err(Box::new(ValidationError::InvalidFieldValue {
                file: "config.json".to_string(),
                field: "num_simulation_scenarios".to_string(),
                value: config.num_simulation_scenarios.to_string(),
                constraint: "must be positive (> 0)".to_string(),
                suggestion: "Set num_simulation_scenarios to at least 1"
                    .to_string(),
            })
            .into());
        }

        Ok(())
    }

    /// Validate system input for ID consistency, references, and constraints.
    ///
    /// # Performance
    ///
    /// O(n) where n = total entities. Uses HashSet for O(1) existence checks.
    pub fn validate_system(system: &SystemInput) -> Result<(), PowersError> {
        // Validate bus IDs
        let bus_ids: Vec<usize> = system.buses.iter().map(|b| b.id).collect();
        Self::validate_id_range_comprehensive(
            &bus_ids,
            "buses",
            "system.json",
        )?;
        let bus_id_set: HashSet<usize> = bus_ids.iter().copied().collect();

        // Validate line IDs and references
        let line_ids: Vec<usize> = system.lines.iter().map(|l| l.id).collect();
        Self::validate_id_range_comprehensive(
            &line_ids,
            "lines",
            "system.json",
        )?;

        for line in &system.lines {
            // Validate bus references
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

            // Validate capacity constraints
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

        // Validate thermal IDs and references
        let thermal_ids: Vec<usize> =
            system.thermals.iter().map(|t| t.id).collect();
        Self::validate_id_range_comprehensive(
            &thermal_ids,
            "thermals",
            "system.json",
        )?;

        for thermal in &system.thermals {
            // Validate bus reference
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

            // Validate cost (non-negative)
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

            // Validate min <= max generation
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

        // Validate hydro IDs and references
        let hydro_ids: Vec<usize> =
            system.hydros.iter().map(|h| h.id).collect();
        Self::validate_id_range_comprehensive(
            &hydro_ids,
            "hydros",
            "system.json",
        )?;
        let hydro_id_set: HashSet<usize> = hydro_ids.iter().copied().collect();

        for hydro in &system.hydros {
            // Validate bus reference
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

            // Validate productivity (must be positive)
            if hydro.productivity <= 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "system.json".to_string(),
                    field: format!("hydros[{}].productivity", hydro.id),
                    value: hydro.productivity.to_string(),
                    constraint: "must be positive (> 0)".to_string(),
                    suggestion: "Set productivity to a positive value (typically 0.5 to 1.5)".to_string(),
                })
                .into());
            }

            // Validate min <= max storage
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

            // Validate min <= max turbined flow
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

            // Validate downstream reference if present
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
    ///
    /// # Performance
    ///
    /// O(n) using HashSet for duplicate detection.
    fn validate_id_range_comprehensive(
        ids: &[usize],
        entity_name: &str,
        file: &str,
    ) -> Result<(), PowersError> {
        if ids.is_empty() {
            return Ok(()); // Empty arrays are valid
        }

        // Check for duplicates
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

        // Check for sequential from 0
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

        // Check for gaps (all IDs from 0 to max_id should exist)
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
    ///
    /// # Performance
    ///
    /// O(n + m) where n = nodes, m = edges. Uses HashSet for O(1) lookups.
    pub fn validate_graph(graph: &GraphInput) -> Result<(), PowersError> {
        if graph.nodes.is_empty() {
            return Ok(()); // Empty graph is valid
        }

        // Validate node IDs unique
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

        // Validate stage IDs sequential from 0
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

        // Validate risk measures
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

        // Validate edges
        for edge in &graph.edges {
            // Validate source exists
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

            // Validate target exists
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

            // Validate probability bounds
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

            // Validate discount rate
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

        // Validate probability normalization per source node
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

    /// Validate recourse input for storage bounds and distribution parameters.
    ///
    /// # Performance
    ///
    /// O(n) where n = storages + distributions. Typical overhead: <20μs.
    pub fn validate_recourse(
        recourse: &Recourse,
        system: &SystemInput,
    ) -> Result<(), PowersError> {
        // Build hydro lookup map for bounds checking
        let hydro_map: HashMap<usize, &_> =
            system.hydros.iter().map(|h| (h.id, h)).collect();

        // Validate initial storage length
        if recourse.initial_condition.storage.len() != system.hydros.len() {
            return Err(Box::new(ValidationError::ConstraintViolation {
                file: "recourse.json".to_string(),
                context: "initial_condition.storage".to_string(),
                constraint: format!(
                    "length must match number of hydros ({})",
                    system.hydros.len()
                ),
                details: format!(
                    "storage array length: {}, system hydros: {}",
                    recourse.initial_condition.storage.len(),
                    system.hydros.len()
                ),
                suggestion: format!(
                    "Add {} more storage entries or adjust system.json",
                    system.hydros.len().saturating_sub(
                        recourse.initial_condition.storage.len()
                    )
                ),
            })
            .into());
        }

        // Validate duplicate storage hydro_ids (T3.10 validation)
        let mut seen_storage_hydro_ids = HashSet::new();
        for storage in &recourse.initial_condition.storage {
            if !seen_storage_hydro_ids.insert(storage.hydro_id) {
                return Err(Box::new(ValidationError::ConstraintViolation {
                    file: "recourse.json".to_string(),
                    context: "initial_condition.storage".to_string(),
                    constraint: "hydro_id must be unique".to_string(),
                    details: format!(
                        "duplicate hydro_id: {}",
                        storage.hydro_id
                    ),
                    suggestion:
                        "Remove duplicate storage entry or correct hydro_id"
                            .to_string(),
                })
                .into());
            }
        }

        // Validate each initial storage
        for storage in &recourse.initial_condition.storage {
            // Validate hydro_id exists
            let hydro = hydro_map.get(&storage.hydro_id).ok_or_else(|| {
                Box::new(ValidationError::InvalidReference {
                    file: "recourse.json".to_string(),
                    context: "initial_condition.storage".to_string(),
                    ref_type: "hydro_id".to_string(),
                    ref_id: storage.hydro_id.to_string(),
                    available: system.hydros.iter().map(|h| h.id.to_string()).collect::<Vec<_>>().join(", "),
                    suggestion: "Check that hydro_id matches an existing hydro in system.json".to_string(),
                })
            })?;

            // Validate storage within bounds
            if storage.value < hydro.min_storage
                || storage.value > hydro.max_storage
            {
                return Err(Box::new(ValidationError::ConstraintViolation {
                    file: "recourse.json".to_string(),
                    context: format!(
                        "initial_condition.storage[hydro_id={}]",
                        storage.hydro_id
                    ),
                    constraint: format!(
                        "value must be in [{}, {}]",
                        hydro.min_storage, hydro.max_storage
                    ),
                    details: format!("value: {}", storage.value),
                    suggestion: format!(
                        "Set value between {} and {}",
                        hydro.min_storage, hydro.max_storage
                    ),
                })
                .into());
            }
        }

        // Validate past inflow data (T3.10 validation)
        for inflow in &recourse.initial_condition.inflow {
            // Validate hydro_id exists
            if !hydro_map.contains_key(&inflow.hydro_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "recourse.json".to_string(),
                    context: "initial_condition.inflow".to_string(),
                    ref_type: "hydro_id".to_string(),
                    ref_id: inflow.hydro_id.to_string(),
                    available: system
                        .hydros
                        .iter()
                        .map(|h| h.id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    suggestion:
                        "Check that hydro_id matches an existing hydro in system.json"
                            .to_string(),
                })
                .into());
            }

            // Validate past inflow value is non-negative
            if inflow.value < 0.0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "recourse.json".to_string(),
                    field: format!(
                        "initial_condition.inflow[hydro_id={}].value",
                        inflow.hydro_id
                    ),
                    value: inflow.value.to_string(),
                    constraint: "must be non-negative (>= 0)".to_string(),
                    suggestion:
                        "Set past inflow value to a non-negative number"
                            .to_string(),
                })
                .into());
            }

            // Validate past inflow lag is positive
            if inflow.lag == 0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "recourse.json".to_string(),
                    field: format!(
                        "initial_condition.inflow[hydro_id={}].lag",
                        inflow.hydro_id
                    ),
                    value: "0".to_string(),
                    constraint: "must be positive (> 0)".to_string(),
                    suggestion: "Set lag to at least 1 (number of periods)"
                        .to_string(),
                })
                .into());
            }
        }

        // Validate duplicate season_ids in uncertainties (T3.10 validation)
        let mut seen_season_ids = HashSet::new();
        for uncertainty in &recourse.uncertainties {
            if !seen_season_ids.insert(uncertainty.season_id) {
                return Err(Box::new(ValidationError::ConstraintViolation {
                    file: "recourse.json".to_string(),
                    context: "uncertainties".to_string(),
                    constraint: "season_id must be unique".to_string(),
                    details: format!("duplicate season_id: {}", uncertainty.season_id),
                    suggestion: "Remove duplicate uncertainty entry or correct season_id"
                        .to_string(),
                })
                .into());
            }
        }

        // Validate seasonal uncertainties
        for (season_idx, uncertainty) in
            recourse.uncertainties.iter().enumerate()
        {
            // Validate num_branchings
            if uncertainty.num_branchings == 0 {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "recourse.json".to_string(),
                    field: format!(
                        "uncertainties[{}].num_branchings",
                        season_idx
                    ),
                    value: "0".to_string(),
                    constraint: "must be positive (> 0)".to_string(),
                    suggestion: "Set num_branchings to at least 1".to_string(),
                })
                .into());
            }

            // Validate load distributions
            for load_dist in &uncertainty.distributions.load {
                // Validate bus_id exists
                if !system.buses.iter().any(|b| b.id == load_dist.bus_id) {
                    return Err(Box::new(ValidationError::InvalidReference {
                        file: "recourse.json".to_string(),
                        context: format!("uncertainties[{}].distributions.load", season_idx),
                        ref_type: "bus_id".to_string(),
                        ref_id: load_dist.bus_id.to_string(),
                        available: system.buses.iter().map(|b| b.id.to_string()).collect::<Vec<_>>().join(", "),
                        suggestion: "Check that bus_id matches an existing bus in system.json".to_string(),
                    })
                    .into());
                }

                // Validate Normal parameters
                if load_dist.normal.sigma < 0.0 {
                    return Err(Box::new(ValidationError::InvalidFieldValue {
                        file: "recourse.json".to_string(),
                        field: format!("uncertainties[{}].distributions.load[bus_id={}].normal.sigma", season_idx, load_dist.bus_id),
                        value: load_dist.normal.sigma.to_string(),
                        constraint: "must be non-negative (>= 0)".to_string(),
                        suggestion: "Set sigma to a non-negative value (standard deviation)".to_string(),
                    })
                    .into());
                }
            }

            // Validate inflow distributions
            for inflow_dist in &uncertainty.distributions.inflow {
                // Validate hydro_id exists
                if !hydro_map.contains_key(&inflow_dist.hydro_id) {
                    return Err(Box::new(ValidationError::InvalidReference {
                        file: "recourse.json".to_string(),
                        context: format!("uncertainties[{}].distributions.inflow", season_idx),
                        ref_type: "hydro_id".to_string(),
                        ref_id: inflow_dist.hydro_id.to_string(),
                        available: system.hydros.iter().map(|h| h.id.to_string()).collect::<Vec<_>>().join(", "),
                        suggestion: "Check that hydro_id matches an existing hydro in system.json".to_string(),
                    })
                    .into());
                }

                // Validate LogNormal parameters
                if inflow_dist.lognormal.mu <= 0.0 {
                    return Err(Box::new(ValidationError::InvalidFieldValue {
                        file: "recourse.json".to_string(),
                        field: format!("uncertainties[{}].distributions.inflow[hydro_id={}].lognormal.mu", season_idx, inflow_dist.hydro_id),
                        value: inflow_dist.lognormal.mu.to_string(),
                        constraint: "must be positive (> 0)".to_string(),
                        suggestion: "Set mu to a positive value (LogNormal scale parameter)".to_string(),
                    })
                    .into());
                }
                if inflow_dist.lognormal.sigma < 0.0 {
                    return Err(Box::new(ValidationError::InvalidFieldValue {
                        file: "recourse.json".to_string(),
                        field: format!("uncertainties[{}].distributions.inflow[hydro_id={}].lognormal.sigma", season_idx, inflow_dist.hydro_id),
                        value: inflow_dist.lognormal.sigma.to_string(),
                        constraint: "must be non-negative (>= 0)".to_string(),
                        suggestion: "Set sigma to a non-negative value (LogNormal shape parameter)".to_string(),
                    })
                    .into());
                }
            }
        }

        Ok(())
    }

    /// Validate cross-file consistency between config, system, graph, and recourse.
    ///
    /// # Performance
    ///
    /// O(n + m) where n = nodes, m = uncertainties. Typical overhead: <10μs.
    pub fn validate_consistency(
        _config: &Config,
        system: &SystemInput,
        graph: &GraphInput,
        recourse: &Recourse,
    ) -> Result<(), PowersError> {
        // Build set of available seasons from recourse
        let available_seasons: HashSet<usize> =
            recourse.uncertainties.iter().map(|u| u.season_id).collect();

        // Validate graph nodes reference valid seasons
        for node in &graph.nodes {
            if !available_seasons.contains(&node.season_id) {
                return Err(Box::new(ValidationError::InvalidReference {
                    file: "graph.json".to_string(),
                    context: format!("node {}", node.id),
                    ref_type: "season_id".to_string(),
                    ref_id: node.season_id.to_string(),
                    available: available_seasons.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "),
                    suggestion: "Add corresponding season to recourse.json uncertainties array".to_string(),
                })
                .into());
            }
        }

        // Cross-validate recourse references (defensive programming)
        let bus_ids: HashSet<usize> =
            system.buses.iter().map(|b| b.id).collect();
        let hydro_ids: HashSet<usize> =
            system.hydros.iter().map(|h| h.id).collect();

        for uncertainty in &recourse.uncertainties {
            for load_dist in &uncertainty.distributions.load {
                if !bus_ids.contains(&load_dist.bus_id) {
                    return Err(Box::new(ValidationError::InvalidReference {
                        file: "recourse.json".to_string(),
                        context: format!("uncertainties[season_id={}].distributions.load", uncertainty.season_id),
                        ref_type: "bus_id".to_string(),
                        ref_id: load_dist.bus_id.to_string(),
                        available: bus_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "),
                        suggestion: "Check that bus_id matches an existing bus in system.json".to_string(),
                    })
                    .into());
                }
            }

            for inflow_dist in &uncertainty.distributions.inflow {
                if !hydro_ids.contains(&inflow_dist.hydro_id) {
                    return Err(Box::new(ValidationError::InvalidReference {
                        file: "recourse.json".to_string(),
                        context: format!("uncertainties[season_id={}].distributions.inflow", uncertainty.season_id),
                        ref_type: "hydro_id".to_string(),
                        ref_id: inflow_dist.hydro_id.to_string(),
                        available: hydro_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "),
                        suggestion: "Check that hydro_id matches an existing hydro in system.json".to_string(),
                    })
                    .into());
                }
            }
        }

        Ok(())
    }

    /// Validate all input components (config, system, graph, recourse, consistency).
    ///
    /// Runs validations in order: config → system → graph → recourse → consistency.
    /// Fails fast on first error.
    ///
    /// # Performance
    ///
    /// Total overhead typically <100μs for standard inputs (<0.02% of training time).
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
