//! High-performance scenario generator for SDDP
//!
//! # Architecture
//!
//! ```text
//! ScenarioGenerator
//!   ├─ models: Vec<UncertaintyModel>          (validated at construction)
//!   ├─ correlation: Option<CorrelationMatrix> (if specified)
//!   └─ buffers: [base_noise, transformed]     (reused per generation)
//!
//! Generation Pipeline:
//!   1. Base Noise: Sample Z ~ N(0,1)
//!   2. Correlation: Apply L×Z if correlation specified
//!   3. Marginal Transform: Apply distribution transform
//!   4. For PAR models: Store innovation only (AR dynamics applied at solve time)
//! ```
//!
//! # What Gets Stored in SAA
//!
//! - **Load entities**: Observations (from `scenario.values`)
//! - **Inflow entities**: Innovations only (from `scenario.innovations`)
//!
//! For PAR models on inflows, this means:
//! - During generation: We only sample innovations ε_t
//! - In SAA storage: Only innovations ε_t are kept
//! - During execution: Full observations Y_t recomputed from innovations using LP constraints
//!
//! ## Lag Buffer for AR Dynamics
//!
//! The lag buffer for PAR (Periodic Autoregressive) models is maintained in
//! `Subproblem.inflow_manager`, NOT in this generator:
//!
//! - **Location**: `Subproblem.inflow_manager`
//! - **Purpose**: Track historical observations for AR constraint RHS computation
//! - **Space**: Observation space (Y_t values)
//! - **Usage**: During forward pass, compute: Y_t = deterministic_base + σ·ε_t + Σ[φ_i·Y_{t-i}]
//! - **Impact**: This is what **actually affects SDDP results**
//!
//! This design is optimal because:
//! 1. SAA storage is minimized (innovations only, not full trajectories)
//! 2. AR dynamics are computed with actual realized observations during execution
//! 3. No need to maintain parallel lag buffers during generation
//!

use crate::correlation_applicator::CorrelationApplicator;
use crate::error::PowersError;
use crate::initial_condition::InitialCondition;
use crate::input::{CorrelationSpecification, EntityReference};
use crate::uncertainty_model::{DistributionType, UncertaintyModel};
use rand::Rng;
use rand_distr::StandardNormal;
use std::collections::HashMap;

/// Single scenario output
#[derive(Debug, Clone)]
pub struct Scenario {
    pub values: Vec<f64>,
    pub innovations: Vec<f64>,
}

impl Scenario {
    fn with_capacity(n_entities: usize) -> Self {
        Self {
            values: Vec::with_capacity(n_entities),
            innovations: Vec::with_capacity(n_entities),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StageScenarios {
    pub scenarios: Vec<Scenario>,
    pub n_entities: usize,
}

impl StageScenarios {
    fn with_capacity(n_scenarios: usize, n_entities: usize) -> Self {
        Self {
            scenarios: Vec::with_capacity(n_scenarios),
            n_entities,
        }
    }

    fn push(&mut self, scenario: Scenario) {
        self.scenarios.push(scenario);
    }
}

/// High-performance scenario generator
///
/// # Note on PAR Models
///
/// For PAR (Periodic Autoregressive) models, this generator only samples innovations ε_t.
/// The full AR dynamics Y_t = deterministic_base + σ·ε_t + Σ[φ_i·Y_{t-i}] are computed
/// during SDDP execution in `Subproblem` using the active lag buffer (`inflow_manager`).
///
pub struct ScenarioGenerator {
    models: Vec<UncertaintyModel>,
    correlation: Option<CorrelationApplicator>,
    base_noise_buffer: Vec<f64>,
    transformed_buffer: Vec<f64>,
}

impl ScenarioGenerator {
    pub fn new(
        models: Vec<UncertaintyModel>,
        _initial_condition: &InitialCondition,
        correlation_spec: Option<&CorrelationSpecification>,
    ) -> Result<Self, PowersError> {
        let n_entities = models.len();

        // Build correlation applicator if specified
        let correlation = if let Some(corr_spec) = correlation_spec {
            // Build entity_to_global_index mapping
            let mut entity_to_global_index = HashMap::new();
            for (idx, model) in models.iter().enumerate() {
                entity_to_global_index.insert(
                    EntityReference {
                        uncertainty_type: model.entity_type(),
                        entity_id: model.entity_id(),
                    },
                    idx,
                );
            }

            // Build correlation blocks
            let mut correlation_blocks = Vec::new();
            for input_block in &corr_spec.blocks {
                use crate::correlation_applicator::CorrelationBlock;

                let n = input_block.correlation_matrix.len();
                let matrix_data: Vec<f64> = input_block
                    .correlation_matrix
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect();

                let correlation_matrix =
                    nalgebra::DMatrix::from_row_slice(n, n, &matrix_data);

                let block = CorrelationBlock::new(
                    input_block.entities.clone(),
                    correlation_matrix,
                )
                .map_err(|e| {
                    PowersError::Other(format!(
                        "Correlation block '{}': {}",
                        input_block.name, e
                    ))
                })?;

                correlation_blocks.push(block);
            }

            Some(CorrelationApplicator::new(
                correlation_blocks,
                entity_to_global_index,
            ))
        } else {
            None
        };

        Ok(Self {
            models,
            correlation,
            base_noise_buffer: Vec::with_capacity(n_entities),
            transformed_buffer: Vec::with_capacity(n_entities),
        })
    }

    pub fn generate_stage_scenarios(
        &mut self,
        season_id: usize,
        num_scenarios: usize,
        rng: &mut impl Rng,
    ) -> StageScenarios {
        let n_entities = self.models.len();

        let mut stage_scenarios =
            StageScenarios::with_capacity(num_scenarios, n_entities);

        self.base_noise_buffer.resize(n_entities, 0.0);
        self.transformed_buffer.resize(n_entities, 0.0);

        for _ in 0..num_scenarios {
            // Step 1: Generate base Gaussian noise Z ~ N(0,1)
            for noise in &mut self.base_noise_buffer {
                *noise = rng.sample(StandardNormal);
            }

            // Step 2: Apply correlation (if specified)
            let transformed_noise = if let Some(ref corr) = self.correlation {
                let base_wrapper = vec![self.base_noise_buffer.clone()];
                let correlated = corr.apply_correlation(&base_wrapper);
                correlated[0].clone()
            } else {
                self.base_noise_buffer.clone()
            };

            // Step 3: Apply marginal transforms + temporal models
            let mut scenario = Scenario::with_capacity(n_entities);

            for (entity_idx, model) in self.models.iter().enumerate() {
                let base_noise = transformed_noise[entity_idx];
                let params = model.seasonal_params(season_id);

                match model {
                    UncertaintyModel::Independent { .. } => {
                        // For observation-space formulation:
                        // - Normal: innovation = ε_t ~ N(0,1), observation = μ + σ*ε_t
                        // - LogNormal3: innovation = transformed LogNormal value (for positivity)
                        //   This breaks mathematical purity but ensures non-negative inflows

                        // Transform base noise to get the innovation
                        let innovation =
                            params.distribution.transform(base_noise, 0.0, 1.0);

                        // Calculate observation based on distribution type
                        let observation = match params.distribution {
                            DistributionType::Normal => {
                                // Linear: Y_t = μ + σ*ε_t
                                params.mean + params.std_dev * innovation
                            }
                            DistributionType::LogNormal3 { .. } => {
                                // LogNormal3: innovation is already transformed
                                // Y_t = μ + innovation (not μ + σ*innovation)
                                params.mean + innovation
                            }
                        };

                        scenario.values.push(observation);
                        scenario.innovations.push(innovation); // Transformed for LogNormal3, ε_t for Normal
                    }
                    UncertaintyModel::PeriodicAR {
                        entity_type: _,
                        entity_id: _,
                        par_params: _,
                    } => {
                        // PAR: Sample innovation only (what actually goes to SAA)
                        //
                        // For PAR models during SAA generation, we only need the innovation ε_t.
                        // The full observation Y_t will be computed during SDDP execution using
                        // Subproblem.inflow_manager, which combines:
                        //   Y_t = deterministic_base + σ·ε_t + Σ[φ_i·Y_{t-i}]
                        //
                        // This is the correct approach because:
                        // - SAA stores only innovations (not observations) for inflows
                        // - AR dynamics are applied at LP solve time, not during generation
                        // - The lag buffer in Subproblem.inflow_manager is the active system
                        //
                        // See input.rs lines 1232-1250: only innovations are extracted for inflows.
                        //
                        // Innovation types:
                        // - Normal: innovation = ε_t ~ N(0,1), use directly in η_t = μ + σ*ε_t
                        // - LogNormal3: innovation = sampled LogNormal value (for positivity)
                        //   This breaks mathematical purity but ensures non-negative inflows

                        let innovation =
                            params.distribution.transform(base_noise, 0.0, 1.0);

                        // Store innovation (what goes to SAA)
                        scenario.innovations.push(innovation);

                        // Placeholder values (unused for inflows, will be computed at solve time)
                        scenario.values.push(0.0);
                    }
                }
            }

            stage_scenarios.push(scenario);
        }

        stage_scenarios
    }

    pub fn num_entities(&self) -> usize {
        self.models.len()
    }
}

#[cfg(test)]
mod tests {}
