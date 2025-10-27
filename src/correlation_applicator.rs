//! Correlation Application via Cholesky Decomposition (Pipeline Stage 2)
//!
//! # Overview
//!
//! Applies correlation structure to independent standard normal samples Z ~ N(0,1)
//! using Cholesky decomposition: W = L × Z, where L is the Cholesky factor of the
//! correlation matrix R = LL^T.
//!
//! This is **Stage 2** of the 4-stage scenario generation pipeline:
//! 1. Base Noise: Generate Z ~ N(0,1) (independent) [`crate::base_noise`]
//! 2. **Correlation: Apply W = L×Z → W ~ N(0,R)** ← This module
//! 3. Marginal: Transform to target distributions
//! 4. Temporal: Apply AR dynamics
//!
//! # Algorithm
//!
//! **Input**: Z = [Z₁, Z₂, ..., Zₙ]^T where Zᵢ ~ N(0,1) independent
//!
//! **Correlation Matrix**: R with Rᵢⱼ = Corr(Xᵢ, Xⱼ)
//!
//! **Cholesky Decomposition**: R = LL^T where L is lower triangular
//!
//! **Transformation**: W = LZ
//!
//! **Properties**:
//! - E[W] = LE[Z] = 0 (mean preserved)
//! - Var(Wᵢ) = 1 (variance preserved)
//! - Cov(W) = L Cov(Z) L^T = LIL^T = R (correlation achieved)
use nalgebra::{Cholesky, DMatrix, DVector};
use std::collections::{HashMap, HashSet};

/// Reference to an entity in the stochastic system
///
/// Identifies which entity (hydro inflow, bus load, etc.)
/// participates in a correlation block.
///
/// # Examples
///
/// ```
/// use powers_rs::correlation_applicator::{EntityRef, UncertaintyType};
///
/// let hydro_1 = EntityRef {
///     uncertainty_type: UncertaintyType::HydroInflow,
///     entity_id: 1,
/// };
///
/// let load_5 = EntityRef {
///     uncertainty_type: UncertaintyType::Load,
///     entity_id: 5,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityRef {
    pub uncertainty_type: UncertaintyType,
    pub entity_id: usize,
}

/// Type of uncertain parameter in the stochastic system
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UncertaintyType {
    HydroInflow,
    Load,
}

/// Cholesky factor wrapper for efficient correlation application
///
/// Wraps the lower triangular Cholesky factor L where R = LL^T for
/// a correlation matrix R. Provides efficient matrix-vector multiply
/// for applying correlation: W = L×Z.
///
/// # Examples
///
/// ```
/// use powers_rs::correlation_applicator::CholeskyFactor;
/// use nalgebra::DMatrix;
///
/// let correlation = DMatrix::from_row_slice(2, 2, &[1.0, 0.7, 0.7, 1.0]);
/// let factor = CholeskyFactor::new(correlation).unwrap();
///
/// let z = vec![0.5, -0.3];
/// let w = factor.transform(&z);
/// ```
#[derive(Clone, Debug)]
pub struct CholeskyFactor {
    l: DMatrix<f64>,
}

impl CholeskyFactor {
    pub fn new(correlation_matrix: DMatrix<f64>) -> Result<Self, String> {
        let cholesky = Cholesky::new(correlation_matrix).ok_or_else(|| {
            "Cholesky decomposition failed - matrix not positive semi-definite"
                .to_string()
        })?;

        Ok(Self { l: cholesky.l() })
    }

    pub fn transform(&self, z: &[f64]) -> Vec<f64> {
        let n = self.l.nrows();
        assert_eq!(
            z.len(),
            n,
            "Input vector length must match matrix dimension"
        );

        let z_vec = DVector::from_row_slice(z);
        let w = &self.l * z_vec;
        w.as_slice().to_vec()
    }

    pub fn dimension(&self) -> usize {
        self.l.nrows()
    }

    pub fn as_matrix(&self) -> &DMatrix<f64> {
        &self.l
    }
}

/// A group of correlated entities with shared correlation structure
///
/// Represents a set of entities that are correlated with each other
/// according to a correlation matrix. Entities not in any block remain
/// independent.
///
/// # Examples
///
/// ```
/// use powers_rs::correlation_applicator::{CorrelationBlock, EntityRef, UncertaintyType};
/// use nalgebra::DMatrix;
///
/// // Correlate hydro inflows for reservoirs 0 and 1 with ρ=0.8
/// let correlation_matrix = DMatrix::from_row_slice(2, 2, &[
///     1.0, 0.8,
///     0.8, 1.0,
/// ]);
///
/// let entities = vec![
///     EntityRef { uncertainty_type: UncertaintyType::HydroInflow, entity_id: 0 },
///     EntityRef { uncertainty_type: UncertaintyType::HydroInflow, entity_id: 1 },
/// ];
///
/// let block = CorrelationBlock::new(entities, correlation_matrix).unwrap();
/// ```
pub struct CorrelationBlock {
    entities: Vec<EntityRef>,
    cholesky_factor: CholeskyFactor,
    entity_to_index: HashMap<EntityRef, usize>,
}

impl CorrelationBlock {
    pub fn new(
        entities: Vec<EntityRef>,
        correlation_matrix: DMatrix<f64>,
    ) -> Result<Self, String> {
        let n = entities.len();

        if correlation_matrix.nrows() != n || correlation_matrix.ncols() != n {
            return Err(format!(
                "Correlation matrix dimensions {}×{} don't match entity count {}",
                correlation_matrix.nrows(),
                correlation_matrix.ncols(),
                n
            ));
        }

        let unique_entities: HashSet<_> = entities.iter().copied().collect();
        if unique_entities.len() != n {
            return Err("Duplicate entities in correlation block".to_string());
        }

        let entity_to_index: HashMap<EntityRef, usize> = entities
            .iter()
            .enumerate()
            .map(|(idx, &entity)| (entity, idx))
            .collect();

        let cholesky_factor = CholeskyFactor::new(correlation_matrix)
            .map_err(|e| format!("Failed to compute Cholesky factor: {}", e))?;

        Ok(Self {
            entities,
            cholesky_factor,
            entity_to_index,
        })
    }

    pub fn entities(&self) -> &[EntityRef] {
        &self.entities
    }

    pub fn cholesky_factor(&self) -> &CholeskyFactor {
        &self.cholesky_factor
    }

    pub fn transform(&self, z_block: &[f64]) -> Vec<f64> {
        assert_eq!(
            z_block.len(),
            self.entities.len(),
            "z_block length must match entity count"
        );
        self.cholesky_factor.transform(z_block)
    }

    pub fn entity_index(&self, entity: &EntityRef) -> Option<usize> {
        self.entity_to_index.get(entity).copied()
    }
}

/// Correlation applicator for pipeline stage 2
///
/// Applies correlation structure to independent standard normal samples
/// via Cholesky decomposition. Supports multiple correlation blocks for
/// different entity groups.
///
/// # Examples
///
/// ```
/// use powers_rs::correlation_applicator::{
///     CorrelationApplicator, CorrelationBlock, EntityRef, UncertaintyType
/// };
/// use nalgebra::DMatrix;
///
/// // Define correlation block for 2 hydro inflows
/// let correlation_matrix = DMatrix::from_row_slice(2, 2, &[
///     1.0, 0.7,
///     0.7, 1.0,
/// ]);
///
/// let entities = vec![
///     EntityRef { uncertainty_type: UncertaintyType::HydroInflow, entity_id: 0 },
///     EntityRef { uncertainty_type: UncertaintyType::HydroInflow, entity_id: 1 },
/// ];
///
/// let block = CorrelationBlock::new(entities, correlation_matrix).unwrap();
///
/// // Create applicator with entity mapping
/// let entity_map: std::collections::HashMap<EntityRef, usize> = [
///     (EntityRef { uncertainty_type: UncertaintyType::HydroInflow, entity_id: 0 }, 0),
///     (EntityRef { uncertainty_type: UncertaintyType::HydroInflow, entity_id: 1 }, 1),
///     (EntityRef { uncertainty_type: UncertaintyType::Load, entity_id: 0 }, 2),
/// ].iter().copied().collect();
///
/// let applicator = CorrelationApplicator::new(vec![block], entity_map);
///
/// // Generate base noise (independent)
/// let base_samples = vec![
///     vec![0.5, -0.3, 1.2],  // scenario 0: [hydro_0, hydro_1, load_0]
///     vec![-1.0, 0.8, 0.2],  // scenario 1
/// ];
///
/// // Apply correlation
/// let correlated = applicator.apply_correlation(&base_samples);
///
/// // hydro_0 and hydro_1 now have ρ≈0.7, load_0 remains independent
/// assert_eq!(correlated.len(), 2);
/// assert_eq!(correlated[0].len(), 3);
/// ```
pub struct CorrelationApplicator {
    blocks: Vec<CorrelationBlock>,
    entity_to_global_index: HashMap<EntityRef, usize>,
}

impl CorrelationApplicator {
    pub fn new(
        blocks: Vec<CorrelationBlock>,
        entity_to_global_index: HashMap<EntityRef, usize>,
    ) -> Self {
        for block in &blocks {
            for entity in block.entities() {
                assert!(
                    entity_to_global_index.contains_key(entity),
                    "Entity {:?} in block not found in global mapping",
                    entity
                );
            }
        }

        let mut seen_entities = HashSet::new();
        for block in &blocks {
            for entity in block.entities() {
                if !seen_entities.insert(entity) {
                    panic!(
                        "Entity {:?} appears in multiple correlation blocks",
                        entity
                    );
                }
            }
        }

        Self {
            blocks,
            entity_to_global_index,
        }
    }

    pub fn apply_correlation(
        &self,
        base_samples: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        if base_samples.is_empty() {
            return Vec::new();
        }

        let num_scenarios = base_samples.len();

        let mut correlated_samples = base_samples.to_vec();

        for block in &self.blocks {
            let block_entities = block.entities();

            let global_indices: Vec<usize> = block_entities
                .iter()
                .map(|entity| {
                    *self
                        .entity_to_global_index
                        .get(entity)
                        .expect("Entity must be in global mapping")
                })
                .collect();

            for scenario_idx in 0..num_scenarios {
                let z_block: Vec<f64> = global_indices
                    .iter()
                    .map(|&idx| base_samples[scenario_idx][idx])
                    .collect();

                let w_block = block.transform(&z_block);

                for (i, &global_idx) in global_indices.iter().enumerate() {
                    correlated_samples[scenario_idx][global_idx] = w_block[i];
                }
            }
        }

        correlated_samples
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[CorrelationBlock] {
        &self.blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use statrs::distribution::{ContinuousCDF, Normal};

    fn create_entity_map(num_entities: usize) -> HashMap<EntityRef, usize> {
        (0..num_entities)
            .map(|i| {
                (
                    EntityRef {
                        uncertainty_type: UncertaintyType::HydroInflow,
                        entity_id: i,
                    },
                    i,
                )
            })
            .collect()
    }

    #[test]
    fn test_uncorrelated_case_identity_matrix() {
        let correlation_matrix = DMatrix::identity(2, 2);
        let entities = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];

        let block =
            CorrelationBlock::new(entities, correlation_matrix).unwrap();
        let entity_map = create_entity_map(2);
        let applicator = CorrelationApplicator::new(vec![block], entity_map);

        let base_samples = vec![vec![0.5, -0.3], vec![-1.0, 0.8]];
        let correlated = applicator.apply_correlation(&base_samples);

        assert_eq!(correlated.len(), 2);
        for (i, scenario) in correlated.iter().enumerate() {
            for (j, &value) in scenario.iter().enumerate() {
                assert_relative_eq!(value, base_samples[i][j], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_perfect_correlation() {
        // Test: High correlation (ρ=0.999) → samples become very similar
        // Note: Perfect correlation (ρ=1.0 for all pairs) is not a valid correlation matrix
        // for n>2 (not positive definite). We use ρ=0.999 instead.
        let correlation_matrix =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.999, 0.999, 1.0]);
        let entities = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];

        let block =
            CorrelationBlock::new(entities, correlation_matrix).unwrap();
        let entity_map = create_entity_map(2);
        let applicator = CorrelationApplicator::new(vec![block], entity_map);

        let base_samples = vec![vec![0.5, -0.3], vec![-1.0, 0.8]];

        let correlated = applicator.apply_correlation(&base_samples);

        // With near-perfect correlation, entities should be very similar within each scenario
        for scenario in &correlated {
            let diff = (scenario[0] - scenario[1]).abs();
            assert!(
                diff < 0.1,
                "High correlation should make values similar, diff={}",
                diff
            );
        }
    }

    #[test]
    fn test_partial_correlation() {
        // Test: ρ=0.7 → sample correlation should match specification
        let correlation_matrix =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.7, 0.7, 1.0]);
        let entities = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];

        let block =
            CorrelationBlock::new(entities, correlation_matrix).unwrap();
        let entity_map = create_entity_map(2);
        let applicator = CorrelationApplicator::new(vec![block], entity_map);

        // Generate many samples to compute sample correlation
        let num_scenarios = 10000;
        let standard_normal =
            Normal::new(0.0, 1.0).expect("Failed to create normal dist");

        let base_samples: Vec<Vec<f64>> = (0..num_scenarios)
            .map(|i| {
                // Deterministic samples for reproducibility, using inverse CDF
                let u1 = (i as f64 + 0.5) / num_scenarios as f64;
                let u2 = (((i * 7 + 13) % num_scenarios) as f64 + 0.5)
                    / num_scenarios as f64;
                vec![
                    standard_normal.inverse_cdf(u1),
                    standard_normal.inverse_cdf(u2),
                ]
            })
            .collect();

        let correlated = applicator.apply_correlation(&base_samples);

        // Compute sample correlation
        let samples_0: Vec<f64> = correlated.iter().map(|s| s[0]).collect();
        let samples_1: Vec<f64> = correlated.iter().map(|s| s[1]).collect();

        let mean_0 = samples_0.iter().sum::<f64>() / num_scenarios as f64;
        let mean_1 = samples_1.iter().sum::<f64>() / num_scenarios as f64;

        let cov: f64 = samples_0
            .iter()
            .zip(&samples_1)
            .map(|(x, y)| (x - mean_0) * (y - mean_1))
            .sum::<f64>()
            / num_scenarios as f64;

        let var_0 = samples_0.iter().map(|x| (x - mean_0).powi(2)).sum::<f64>()
            / num_scenarios as f64;
        let var_1 = samples_1.iter().map(|y| (y - mean_1).powi(2)).sum::<f64>()
            / num_scenarios as f64;

        let corr = cov / (var_0.sqrt() * var_1.sqrt());

        // With 10000 deterministic samples, expect correlation within ±0.08
        // (larger tolerance than true random samples due to quasi-random pattern)
        assert_relative_eq!(corr, 0.7, epsilon = 0.08);
    }

    #[test]
    fn test_multiple_blocks() {
        // Test: Two separate correlation blocks
        let correlation_matrix_1 =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.8, 0.8, 1.0]);
        let entities_1 = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];
        let block_1 =
            CorrelationBlock::new(entities_1, correlation_matrix_1).unwrap();

        let correlation_matrix_2 =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let entities_2 = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 1,
            },
        ];
        let block_2 =
            CorrelationBlock::new(entities_2, correlation_matrix_2).unwrap();

        let entity_map: HashMap<EntityRef, usize> = [
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::HydroInflow,
                    entity_id: 0,
                },
                0,
            ),
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::HydroInflow,
                    entity_id: 1,
                },
                1,
            ),
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::Load,
                    entity_id: 0,
                },
                2,
            ),
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::Load,
                    entity_id: 1,
                },
                3,
            ),
        ]
        .iter()
        .copied()
        .collect();

        let applicator =
            CorrelationApplicator::new(vec![block_1, block_2], entity_map);

        let base_samples =
            vec![vec![0.5, -0.3, 1.2, -0.8], vec![-1.0, 0.8, 0.2, 1.5]];

        let correlated = applicator.apply_correlation(&base_samples);

        // Check dimensions preserved
        assert_eq!(correlated.len(), 2);
        assert_eq!(correlated[0].len(), 4);

        // Check that blocks were applied (samples changed)
        assert_ne!(correlated, base_samples);
    }

    #[test]
    fn test_mixed_correlated_and_independent() {
        // Test: Some entities correlated, others remain independent
        let correlation_matrix =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.9, 0.9, 1.0]);
        let entities = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];

        let block =
            CorrelationBlock::new(entities, correlation_matrix).unwrap();

        let entity_map: HashMap<EntityRef, usize> = [
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::HydroInflow,
                    entity_id: 0,
                },
                0,
            ),
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::HydroInflow,
                    entity_id: 1,
                },
                1,
            ),
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::Load,
                    entity_id: 0,
                },
                2,
            ),
        ]
        .iter()
        .copied()
        .collect();

        let applicator = CorrelationApplicator::new(vec![block], entity_map);

        let base_samples = vec![vec![0.5, -0.3, 1.2], vec![-1.0, 0.8, 0.2]];

        let correlated = applicator.apply_correlation(&base_samples);

        // Entity 2 (Load 0) is not in any block, so should remain unchanged
        for i in 0..base_samples.len() {
            assert_relative_eq!(
                correlated[i][2],
                base_samples[i][2],
                epsilon = 1e-10
            );
        }

        // Entities 0 and 1 should be transformed (check significant difference)
        let diff_0 = (correlated[0][0] - base_samples[0][0]).abs();
        let diff_1 = (correlated[0][1] - base_samples[0][1]).abs();
        assert!(
            diff_0 > 1e-6 || diff_1 > 1e-6,
            "At least one entity should be transformed"
        );
    }

    #[test]
    fn test_statistical_properties_preserved() {
        // Test: Mean=0, Var=1 preserved after correlation
        use statrs::distribution::{ContinuousCDF, Normal};

        let correlation_matrix =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.6, 0.6, 1.0]);
        let entities = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];

        let block =
            CorrelationBlock::new(entities, correlation_matrix).unwrap();
        let entity_map = create_entity_map(2);
        let applicator = CorrelationApplicator::new(vec![block], entity_map);

        // Generate many samples with proper standard normal distribution
        let num_scenarios = 10000;
        let standard_normal =
            Normal::new(0.0, 1.0).expect("Failed to create normal dist");

        let base_samples: Vec<Vec<f64>> = (0..num_scenarios)
            .map(|i| {
                // Convert uniform [0,1] to standard normal via inverse CDF
                // Use (i+0.5)/n to avoid boundary issues at 0 and 1
                let u1 = (i as f64 + 0.5) / num_scenarios as f64;
                let u2 = (((i * 7 + 13) % num_scenarios) as f64 + 0.5)
                    / num_scenarios as f64;
                vec![
                    standard_normal.inverse_cdf(u1),
                    standard_normal.inverse_cdf(u2),
                ]
            })
            .collect();

        let correlated = applicator.apply_correlation(&base_samples);

        // Check mean ≈ 0, std ≈ 1 for both entities
        for entity_idx in 0..2 {
            let samples: Vec<f64> =
                correlated.iter().map(|s| s[entity_idx]).collect();
            let mean = samples.iter().sum::<f64>() / num_scenarios as f64;
            let variance =
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / num_scenarios as f64;

            // With deterministic quasi-random samples, variance can deviate more
            assert_relative_eq!(mean, 0.0, epsilon = 0.05);
            assert_relative_eq!(variance, 1.0, epsilon = 0.25);
        }
    }

    #[test]
    fn test_near_singular_matrix_handled() {
        // Test: Near-singular matrix (high correlation) handled gracefully
        let correlation_matrix =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.9999, 0.9999, 1.0]);
        let entities = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];

        // Should succeed with near-singular matrix
        let block = CorrelationBlock::new(entities, correlation_matrix);
        assert!(block.is_ok(), "Near-singular matrix should be handled");

        let entity_map = create_entity_map(2);
        let applicator =
            CorrelationApplicator::new(vec![block.unwrap()], entity_map);

        let base_samples = vec![vec![0.5, -0.3], vec![-1.0, 0.8]];
        let correlated = applicator.apply_correlation(&base_samples);

        // Should produce output with correct dimensions
        assert_eq!(correlated.len(), 2);
        assert_eq!(correlated[0].len(), 2);
    }

    #[test]
    fn test_duplicate_entities_in_block_error() {
        // Test: Duplicate entities in same block should return error
        let correlation_matrix =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.7, 0.7, 1.0]);
        let entities = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            }, // Duplicate
        ];

        let result = CorrelationBlock::new(entities, correlation_matrix);
        assert!(result.is_err());
        let err_msg = result.err().unwrap();
        assert!(
            err_msg.contains("Duplicate entities in correlation block"),
            "Expected duplicate error, got: {}",
            err_msg
        );
    }

    #[test]
    #[should_panic(expected = "appears in multiple correlation blocks")]
    fn test_duplicate_entities_across_blocks_panics() {
        // Test: Same entity in multiple blocks should panic
        let correlation_matrix_1 =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.8, 0.8, 1.0]);
        let entities_1 = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 0,
            },
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            },
        ];
        let block_1 =
            CorrelationBlock::new(entities_1, correlation_matrix_1).unwrap();

        let correlation_matrix_2 =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let entities_2 = vec![
            EntityRef {
                uncertainty_type: UncertaintyType::HydroInflow,
                entity_id: 1,
            }, // Duplicate from block_1
            EntityRef {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
            },
        ];
        let block_2 =
            CorrelationBlock::new(entities_2, correlation_matrix_2).unwrap();

        let entity_map: HashMap<EntityRef, usize> = [
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::HydroInflow,
                    entity_id: 0,
                },
                0,
            ),
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::HydroInflow,
                    entity_id: 1,
                },
                1,
            ),
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::Load,
                    entity_id: 0,
                },
                2,
            ),
        ]
        .iter()
        .copied()
        .collect();

        let _ = CorrelationApplicator::new(vec![block_1, block_2], entity_map);
    }

    #[test]
    fn test_empty_blocks() {
        // Test: No correlation blocks → all entities remain independent
        let entity_map = create_entity_map(3);
        let applicator = CorrelationApplicator::new(vec![], entity_map);

        let base_samples = vec![vec![0.5, -0.3, 1.2], vec![-1.0, 0.8, 0.2]];

        let correlated = applicator.apply_correlation(&base_samples);

        // With no blocks, output should equal input
        assert_eq!(correlated, base_samples);
    }
}
