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
        // Validate noise models (schema v2 format)
        // Note: noise_models field is now required (not Option)
        Self::validate_noise_models_v2(&recourse.noise_models, system)?;

        // Validate initial lag values for AR models
        Self::validate_ar_initial_lags_v2(
            &recourse.initial_condition,
            &recourse.noise_models,
        )?;

        Ok(())
    }

    /// Validate cross-file consistency between config, system, graph, and recourse.
    ///
    /// # Performance
    ///
    /// O(1). Typical overhead: <1μs.
    pub fn validate_consistency(
        _config: &Config,
        _system: &SystemInput,
        _graph: &GraphInput,
        _recourse: &Recourse,
    ) -> Result<(), PowersError> {
        // Legacy uncertainties format validation removed (AR-6.7)
        // New noise_models format validation is performed in validate_recourse()
        Ok(())
    }

    /// Validate noise_models format (AR-2)
    ///
    /// Validates AR model parameters including stationarity conditions,
    /// coefficient counts, distributions, and entity references.
    ///
    /// # Performance
    ///
    /// Validation is O(n) where n = number of noise models.
    /// Typical overhead: <10μs per noise model.
    #[allow(deprecated)]
    fn validate_noise_models_v2(
        noise_models: &[crate::input::NoiseModel],
        system: &SystemInput,
    ) -> Result<(), PowersError> {
        use crate::input::UncertaintyType;

        // Build lookup sets for O(1) entity validation
        let hydro_ids: HashSet<usize> =
            system.hydros.iter().map(|h| h.id).collect();
        let bus_ids: HashSet<usize> =
            system.buses.iter().map(|b| b.id).collect();

        for (idx, model) in noise_models.iter().enumerate() {
            let context = format!("noise_models[{}]", idx);

            // Validate entity exists
            match model.uncertainty_type {
                UncertaintyType::Inflow => {
                    if !hydro_ids.contains(&model.entity_id) {
                        return Err(Box::new(ValidationError::InvalidReference {
                            file: "recourse.json".to_string(),
                            context: context.clone(),
                            ref_type: "hydro entity_id".to_string(),
                            ref_id: model.entity_id.to_string(),
                            available: hydro_ids
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            suggestion: format!(
                                "Change entity_id to a valid hydro_id (available: {})",
                                hydro_ids
                                    .iter()
                                    .map(|id| id.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        })
                        .into());
                    }
                }
                UncertaintyType::Load => {
                    if !bus_ids.contains(&model.entity_id) {
                        return Err(Box::new(ValidationError::InvalidReference {
                            file: "recourse.json".to_string(),
                            context: context.clone(),
                            ref_type: "bus entity_id".to_string(),
                            ref_id: model.entity_id.to_string(),
                            available: bus_ids
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            suggestion: format!(
                                "Change entity_id to a valid bus_id (available: {})",
                                bus_ids
                                    .iter()
                                    .map(|id| id.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        })
                        .into());
                    }
                }
            }

            // Use the NoiseModel's built-in validate() method
            model.validate().map_err(|e| {
                Box::new(ValidationError::ConstraintViolation {
                    file: "recourse.json".to_string(),
                    context: context.clone(),
                    constraint: "noise model semantic validation".to_string(),
                    details: e,
                    suggestion: "Check marginal_distribution, innovation_distribution, and temporal_model fields".to_string(),
                })
            })?;

            // Additional validation for distribution parameters
            use crate::input::MarginalDistribution;
            match &model.distribution {
                MarginalDistribution::Normal { mean: _, std_dev } => {
                    if *std_dev <= 0.0 {
                        return Err(Box::new(
                            ValidationError::InvalidFieldValue {
                                file: "recourse.json".to_string(),
                                field: format!(
                                    "{}.distribution.std_dev",
                                    context
                                ),
                                value: std_dev.to_string(),
                                constraint: "must be positive (> 0)"
                                    .to_string(),
                                suggestion: "Set std_dev to a positive value"
                                    .to_string(),
                            },
                        )
                        .into());
                    }
                }
                MarginalDistribution::LogNormal3 {
                    gamma,
                    mu: _,
                    sigma,
                } => {
                    if *gamma < 0.0 {
                        return Err(Box::new(
                            ValidationError::InvalidFieldValue {
                                file: "recourse.json".to_string(),
                                field: format!(
                                    "{}.distribution.gamma",
                                    context
                                ),
                                value: gamma.to_string(),
                                constraint: "must be non-negative (≥ 0)"
                                    .to_string(),
                                suggestion: "Set gamma ≥ 0".to_string(),
                            },
                        )
                        .into());
                    }
                    if *sigma <= 0.0 {
                        return Err(Box::new(
                            ValidationError::InvalidFieldValue {
                                file: "recourse.json".to_string(),
                                field: format!(
                                    "{}.marginal_distribution.sigma",
                                    context
                                ),
                                value: sigma.to_string(),
                                constraint: "must be positive (> 0)"
                                    .to_string(),
                                suggestion: "Set sigma > 0".to_string(),
                            },
                        )
                        .into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate AR stationarity conditions (PAR-021: AR removed)
    ///
    /// Ensures AR coefficients satisfy stationarity requirements:
    /// - AR(1): |φ| < 1
    /// - AR(2): Triangle conditions (roots inside unit circle)
    /// - AR(3): Numerical root-finding (simplified check)
    ///
    /// AR-4 Enhancement: Adds spectral radius and ACF half-life checks
    /// with detailed warnings for borderline cases.
    ///
    /// # Performance
    ///
    /// O(1) for AR(1) and AR(2), O(p) for AR(3). <1μs per call (AR-2).
    /// AR-4 additions: +<10μs for spectral radius + ACF half-life.
    #[allow(dead_code)]
    #[allow(deprecated)] // Still validates deprecated AR models during soft deprecation (PAR-018)
    fn validate_ar_stationarity(
        coefficients: &[f64],
        lag_order: usize,
        context: &str,
    ) -> Result<(), PowersError> {
        // Compute spectral radius for all AR models (AR-4)
        let spectral_radius = Self::compute_spectral_radius(coefficients);

        match lag_order {
            1 => {
                // AR(1): |φ| < 1
                let phi = coefficients[0];
                if phi.abs() >= 1.0 {
                    return Err(Box::new(ValidationError::ConstraintViolation {
                        file: "recourse.json".to_string(),
                        context: context.to_string(),
                        constraint: "AR(1) coefficient must satisfy |φ| < 1 for stationarity".to_string(),
                        details: format!("φ = {} violates |φ| < 1 (spectral radius = {})", phi, spectral_radius),
                        suggestion: "Choose |φ| < 1 (e.g., 0.7 for positive correlation, -0.7 for oscillation)".to_string(),
                    })
                    .into());
                }
            }
            2 => {
                // AR(2): Triangle conditions for stationarity
                // φ₂ + φ₁ < 1
                // φ₂ - φ₁ < 1
                // |φ₂| < 1
                let phi1 = coefficients[0];
                let phi2 = coefficients[1];

                if phi2.abs() >= 1.0 {
                    return Err(Box::new(
                        ValidationError::ConstraintViolation {
                            file: "recourse.json".to_string(),
                            context: context.to_string(),
                            constraint:
                                "AR(2) requires |φ₂| < 1 for stationarity"
                                    .to_string(),
                            details: format!(
                                "|φ₂| = {} violates |φ₂| < 1",
                                phi2.abs()
                            ),
                            suggestion: "Choose |φ₂| < 1".to_string(),
                        },
                    )
                    .into());
                }

                if phi2 + phi1 >= 1.0 {
                    return Err(Box::new(ValidationError::ConstraintViolation {
                        file: "recourse.json".to_string(),
                        context: context.to_string(),
                        constraint: "AR(2) requires φ₂ + φ₁ < 1 for stationarity".to_string(),
                        details: format!("φ₂ + φ₁ = {} violates constraint", phi2 + phi1),
                        suggestion: format!("Reduce coefficients so φ₂ + φ₁ < 1 (current: {})", phi2 + phi1),
                    })
                    .into());
                }

                if phi2 - phi1 >= 1.0 {
                    return Err(Box::new(ValidationError::ConstraintViolation {
                        file: "recourse.json".to_string(),
                        context: context.to_string(),
                        constraint: "AR(2) requires φ₂ - φ₁ < 1 for stationarity".to_string(),
                        details: format!("φ₂ - φ₁ = {} violates constraint", phi2 - phi1),
                        suggestion: format!("Adjust coefficients so φ₂ - φ₁ < 1 (current: {})", phi2 - phi1),
                    })
                    .into());
                }
            }
            3 => {
                // AR(3): Simplified check (sum of absolute coefficients < 1)
                // Full check requires numerical root-finding
                let sum_abs: f64 = coefficients.iter().map(|c| c.abs()).sum();
                if sum_abs >= 1.0 {
                    eprintln!(
                        "Warning: {}: AR(3) coefficients have Σ|φᵢ| = {} ≥ 1. This is a necessary (but not sufficient) condition for non-stationarity. Consider reducing coefficient magnitudes.",
                        context,
                        sum_abs
                    );
                }
            }
            _ => {
                // Should never reach here (validated earlier)
                unreachable!("lag_order must be 1, 2, or 3");
            }
        }

        // AR-4: Enhanced stability warnings based on spectral radius
        // and autocorrelation function half-life
        //
        // References:
        // - Hamilton (1994), Section 3.5: Forecasting
        // - Box et al. (2015), Chapter 7: Model Building

        // Check spectral radius thresholds
        if spectral_radius > 0.99 {
            eprintln!(
                "⚠️  STABILITY WARNING: {}: Spectral radius ρ = {:.4} (very close to unit root)",
                context, spectral_radius
            );
            eprintln!(
                "    → Expect VERY slow convergence and poor mixing in SDDP"
            );
            eprintln!("    → Autocorrelation persists for many stages");
            eprintln!(
                "    → Suggestion: Reduce coefficient magnitudes by ~10% (multiply by 0.9)"
            );
        } else if spectral_radius > 0.95 {
            eprintln!(
                "⚠️  STABILITY WARNING: {}: Spectral radius ρ = {:.4} (borderline stability)",
                context, spectral_radius
            );
            eprintln!("    → May experience slow convergence");
            eprintln!(
                "    → Consider reducing coefficients if SDDP convergence is poor"
            );
        }

        // Compute and check ACF half-life
        if let Some(half_life) = Self::compute_acf_half_life(coefficients) {
            if half_life > 20 {
                eprintln!(
                    "⚠️  MIXING WARNING: {}: Autocorrelation half-life = {} stages (long)",
                    context, half_life
                );
                eprintln!(
                    "    → Requires many stages ({}) for autocorrelation to decay to 50%",
                    half_life
                );
                eprintln!("    → SDDP may need deeper scenario trees for proper sampling");
                eprintln!(
                    "    → Suggestion: Ensure planning horizon covers at least {} stages",
                    half_life * 2
                );
            } else if spectral_radius <= 0.95 {
                // Good case: log success (only if not already warned about spectral radius)
                eprintln!(
                    "✓  AR Stability: {}: ρ = {:.4}, ACF half-life = {} stages (good mixing)",
                    context, spectral_radius, half_life
                );
            }
        } else {
            eprintln!(
                "⚠️  MIXING WARNING: {}: Autocorrelation does not decay to 50% within 100 stages",
                context
            );
            eprintln!(
                "    → Extremely slow mixing (spectral radius ρ = {:.4})",
                spectral_radius
            );
            eprintln!("    → Model may be effectively non-stationary");
        }

        // Check for numerical precision issues
        for (i, &coef) in coefficients.iter().enumerate() {
            if coef.abs() < 1e-10 {
                eprintln!(
                    "⚠️  NUMERICAL WARNING: {}: Coefficient φ_{} = {:.2e} is effectively zero",
                    context,
                    i + 1,
                    coef
                );
                eprintln!("    → Consider removing this lag from the model");
            }
        }

        Ok(())
    }

    /// Compute spectral radius of AR(p) characteristic polynomial (AR-4)
    ///
    /// The spectral radius is the maximum absolute value of the roots of the
    /// characteristic polynomial: λᵖ - φ₁λᵖ⁻¹ - ... - φₚ = 0
    ///
    /// For stationarity, we need ρ < 1. Values close to 1 indicate slow mixing.
    ///
    /// # Mathematical Foundation
    ///
    /// **References:**
    /// - Hamilton, J. D. (1994). "Time Series Analysis", Princeton University Press.
    ///   Chapter 3: Stationary ARMA Processes. (Companion matrix method, pp. 53-59)
    /// - Brockwell, P. J., & Davis, R. A. (2016). "Introduction to Time Series and
    ///   Forecasting" (3rd ed.), Springer. Chapter 3.1: Stationarity conditions.
    /// - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
    ///   "Time Series Analysis: Forecasting and Control" (5th ed.), Wiley.
    ///   Chapter 3: Linear Stationary Models.
    ///
    /// # Performance
    ///
    /// - AR(1): O(1) - direct computation
    /// - AR(2): O(1) - quadratic formula (real or complex roots)
    /// - AR(3): O(1) - conservative approximation using sum of absolute values
    ///
    /// Typical: <1μs per call
    ///
    /// # Approximations
    ///
    /// AR(3): Uses conservative bound Σ|φᵢ| ≤ 1 (sufficient for stationarity).
    /// Exact computation would require numerical root-finding (cubic equation),
    /// but this is avoided for performance (hot path in validation).
    ///
    /// # Note
    /// Public visibility for testing purposes.
    pub fn compute_spectral_radius(coefficients: &[f64]) -> f64 {
        let p = coefficients.len();

        match p {
            1 => {
                // AR(1): ρ = |φ|
                // Characteristic equation: λ - φ = 0 → λ = φ
                coefficients[0].abs()
            }
            2 => {
                // AR(2): Solve λ² - φ₁λ - φ₂ = 0
                // Roots: λ = (φ₁ ± √(φ₁² + 4φ₂)) / 2
                //
                // Reference: Hamilton (1994), eq. (3.1.8)
                let phi1 = coefficients[0];
                let phi2 = coefficients[1];

                let discriminant = phi1 * phi1 + 4.0 * phi2;

                if discriminant >= 0.0 {
                    // Real roots
                    let sqrt_disc = discriminant.sqrt();
                    let root1 = (phi1 + sqrt_disc) / 2.0;
                    let root2 = (phi1 - sqrt_disc) / 2.0;
                    root1.abs().max(root2.abs())
                } else {
                    // Complex conjugate roots: λ = (φ₁ ± i√|Δ|) / 2
                    // Magnitude: |λ| = √((φ₁/2)² + |Δ|/4) = √(-φ₂)
                    //
                    // Derivation: For complex z = a ± bi, |z| = √(a² + b²)
                    // Here: a = φ₁/2, b = √|Δ|/2 = √(-φ₁² - 4φ₂)/2
                    // |λ|² = (φ₁/2)² + (-φ₁² - 4φ₂)/4 = φ₁²/4 - φ₁²/4 - φ₂ = -φ₂
                    //
                    // Reference: Brockwell & Davis (2016), Theorem 3.1.1
                    (-phi2).sqrt()
                }
            }
            3 => {
                // AR(3): Use conservative approximation
                //
                // Sufficient condition for stationarity: Σ|φᵢ| < 1
                // (Not necessary, but fast to compute and safe)
                //
                // Reference: Lutkepohl, H. (2005). "New Introduction to Multiple
                // Time Series Analysis", Springer. Proposition 2.1 (p. 20).
                //
                // PERFORMANCE: Exact spectral radius requires solving cubic equation
                // (Cardano's formula or numerical methods), which is expensive.
                // For validation, conservative bound is sufficient.
                let sum_abs: f64 = coefficients.iter().map(|c| c.abs()).sum();
                sum_abs
            }
            _ => {
                // Should never reach here (validated as 1..=3 earlier)
                eprintln!(
                    "Warning: Spectral radius for AR({}) not implemented, using conservative estimate",
                    p
                );
                0.99 // Conservative: assume borderline stationary
            }
        }
    }

    /// Compute autocorrelation function (ACF) half-life (AR-4)
    ///
    /// Returns the smallest lag k where |ρₖ| < 0.5, indicating how quickly
    /// autocorrelation decays. Long half-lives (>20) indicate slow mixing
    /// and may require many SDDP stages for proper decorrelation.
    ///
    /// # Mathematical Foundation
    ///
    /// **References:**
    /// - Box et al. (2015). "Time Series Analysis: Forecasting and Control",
    ///   Chapter 3.2: Autocorrelation function of AR processes.
    /// - Brockwell & Davis (2016), Section 3.2: The ACF and PACF.
    /// - Hamilton (1994), Section 3.3: Autocovariance-generating function.
    ///
    /// For AR(p): ρₖ satisfies Yule-Walker equations:
    ///   ρₖ = φ₁ρₖ₋₁ + φ₂ρₖ₋₂ + ... + φₚρₖ₋ₚ  (k ≥ p)
    ///
    /// Initial conditions (k < p) computed from system of equations.
    /// Reference: Hamilton (1994), eq. (3.3.8)-(3.3.10)
    ///
    /// # Performance
    ///
    /// - AR(1): O(1) - closed form ρₖ = φᵏ
    /// - AR(2), AR(3): O(k) where k is half-life (typically k < 100)
    ///
    /// Typical: <10μs per call (early termination when |ρₖ| < 0.5)
    ///
    /// # Note
    /// Public visibility for testing purposes.
    pub fn compute_acf_half_life(coefficients: &[f64]) -> Option<usize> {
        let p = coefficients.len();

        if p == 1 {
            // AR(1): ρₖ = φᵏ
            // Half-life: φʰ = 0.5 → h = log(0.5) / log(φ)
            //
            // Reference: Box et al. (2015), eq. (3.2.7)
            let phi = coefficients[0];

            if phi.abs() < 1e-10 {
                return Some(0); // White noise: immediate decay
            }

            let log_phi = phi.abs().ln();
            if log_phi.abs() < 1e-10 {
                return Some(100); // φ ≈ 1: very slow decay
            }

            let half_life = (0.5_f64.ln() / log_phi).ceil() as usize;
            Some(half_life.min(100)) // Cap at 100 for sanity
        } else {
            // AR(p): Iterative Yule-Walker recursion
            //
            // ρₖ = φ₁ρₖ₋₁ + φ₂ρₖ₋₂ + ... + φₚρₖ₋ₚ
            //
            // Initial conditions for k < p solved from Yule-Walker system:
            // [1    ρ₁   ρ₂  ... ρₚ₋₁] [1 ]   [1 ]
            // [ρ₁   1    ρ₁  ... ρₚ₋₂] [φ₁]   [ρ₁]
            // [ρ₂   ρ₁   1   ... ρₚ₋₃] [φ₂] = [ρ₂]
            // ...                       ...    ...
            // [ρₚ₋₁ ρₚ₋₂ ... 1       ] [φₚ]   [ρₚ]
            //
            // For simplicity, use approximate initial conditions and iterate.
            // Reference: Brockwell & Davis (2016), Algorithm 3.1
            Self::compute_acf_half_life_iterative(coefficients)
        }
    }

    /// Iterative ACF computation for AR(p) with p ≥ 2
    ///
    /// Uses Yule-Walker recursion with approximate initial conditions.
    /// Reference: Hamilton (1994), eq. (3.3.14)
    fn compute_acf_half_life_iterative(coefficients: &[f64]) -> Option<usize> {
        let p = coefficients.len();
        let mut acf = vec![1.0]; // ρ₀ = 1 (autocorrelation at lag 0)

        // Approximate initial conditions for ρ₁, ..., ρₚ₋₁
        // Use simplified approach: ρₖ ≈ φ₁ᵏ for small k
        // (This is exact for AR(1), good approximation for AR(2), AR(3))
        for k in 1..p {
            let mut rho_k = 0.0;
            for (j, &phi_j) in coefficients.iter().enumerate().take(k) {
                let lag = k - (j + 1);
                rho_k += phi_j * acf[lag];
            }
            // Add contribution from uninitialized lags (assume exponential decay)
            for j in k..p {
                rho_k += coefficients[j]
                    * coefficients[0].powi((k as i32) - (j as i32) - 1);
            }
            acf.push(rho_k);
        }

        // Iterative Yule-Walker for k ≥ p
        for k in p..=100 {
            let mut rho_k = 0.0;
            for (j, &phi_j) in coefficients.iter().enumerate() {
                rho_k += phi_j * acf[k - (j + 1)];
            }
            acf.push(rho_k);

            if rho_k.abs() < 0.5 {
                return Some(k);
            }
        }

        None // Didn't reach half-life in 100 lags
    }

    /// Validate initial lag values for AR models (AR-3)
    ///
    /// Ensures AR models have proper historical lag values for initialization:
    /// - AR(p) models require exactly p lag values with lag indices 1..p
    /// - Lag values must be non-negative (inflows)
    /// - Only AR inflow models need lag values (load models don't)
    ///
    /// PAR-021: AR validation removed (stub for backward compatibility)
    fn validate_ar_initial_lags_v2(
        _initial_condition: &crate::input::InitialConditionInput,
        _noise_models: &[crate::input::NoiseModel],
    ) -> Result<(), PowersError> {
        // PAR-021: AR models removed, no validation needed
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PRIVATE FUNCTION TESTS (Added for T4.2 Phase 5b)
    // ========================================================================

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
        use crate::input::*;
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
            par_config: None,
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("source_bus_id") || err_str.contains("99"));
    }

    #[test]
    fn test_validate_system_invalid_line_target_bus() {
        use crate::input::*;
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
            par_config: None,
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("target_bus_id") || err_str.contains("88"));
    }

    #[test]
    fn test_validate_system_negative_direct_capacity() {
        use crate::input::*;
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
            par_config: None,
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
        use crate::input::*;
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
            par_config: None,
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("bus_id") || err_str.contains("77"));
    }

    #[test]
    fn test_validate_system_negative_thermal_cost() {
        use crate::input::*;
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
            par_config: None,
        };
        let result = InputValidator::validate_system(&system);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("cost") && err_str.contains("negative"));
    }

    #[test]
    fn test_validate_config_with_some_simulation() {
        use crate::input::*;
        // Valid config with Some(num_simulation_scenarios)
        let config = Config {
            num_iterations: 10,
            num_forward_passes: 4,
            num_simulation_scenarios: Some(100),
            seed: 42,
            num_threads: None,
            output_path: None,
        };
        let result = InputValidator::validate_config_minimal(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_config_with_none_simulation() {
        use crate::input::*;
        // Valid config with None (simulation skipped)
        let config = Config {
            num_iterations: 10,
            num_forward_passes: 4,
            num_simulation_scenarios: None,
            seed: 42,
            num_threads: None,
            output_path: None,
        };
        let result = InputValidator::validate_config_minimal(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_config_rejects_zero_simulation() {
        use crate::input::*;
        // Invalid: Some(0) should be rejected
        let config = Config {
            num_iterations: 10,
            num_forward_passes: 4,
            num_simulation_scenarios: Some(0),
            seed: 42,
            num_threads: None,
            output_path: None,
        };
        let result = InputValidator::validate_config_minimal(&config);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("num_simulation_scenarios"));
        assert!(err_str.contains("positive") || err_str.contains("> 0"));
    }
}
