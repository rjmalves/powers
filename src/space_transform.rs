//! Space transformation utilities for PAR models
//!
//! PAR models have two spaces:
//! - **Observation space** (Y): Actual inflow values with mean μ and std σ
//! - **Residual space** (Z'): Standardized values where AR equation operates
//!
//! The AR equation operates on residuals:
//!   Z'_t = φ₁·Z'_{t-1} + φ₂·Z'_{t-2} + ... + φₚ·Z'_{t-p} + a_t
//!
//! Relationship:
//!   Y_t = μ_m + σ_m · Z'_t
//!   Z'_t = (Y_t - μ_m) / σ_m

use crate::unified_noise_spec::{TemporalModelSpec, UnifiedNoiseSpec};

/// Seasonal transformation parameters for a single entity
#[derive(Debug, Clone, Copy)]
pub struct SeasonalTransform {
    pub mean: f64,
    pub std_dev: f64,
}

impl SeasonalTransform {
    /// Transform observation to residual: Z' = (Y - μ) / σ
    pub fn to_residual(&self, observation: f64) -> f64 {
        if self.std_dev > 1e-10 {
            (observation - self.mean) / self.std_dev
        } else {
            // Degenerate case: zero variance
            // Return 0.0 (residual at mean)
            0.0
        }
    }

    /// Transform residual to observation: Y = μ + σ·Z'
    pub fn to_observation(&self, residual: f64) -> f64 {
        self.mean + self.std_dev * residual
    }
}

/// Cache of transformation parameters per entity and season
///
/// Structure: transforms[entity_id][season_id] → SeasonalTransform
///
/// # Performance
/// - O(1) lookup after construction
/// - Small memory footprint (~16 bytes per entity-season pair)
pub struct TransformCache {
    /// Transforms indexed by entity_id, then season_id
    /// Outer Vec: entity_id
    /// Inner Vec: season_id (empty if entity has no PAR model)
    transforms: Vec<Vec<Option<SeasonalTransform>>>,
    #[allow(dead_code)] // Stored for potential future use in diagnostics
    num_seasons: usize,
}

impl TransformCache {
    /// Build transformation cache from unified specs
    ///
    /// # Arguments
    /// * `unified_specs` - Noise specifications (may contain PAR models)
    /// * `num_entities` - Total number of entities (hydros or loads)
    /// * `num_seasons` - Number of seasons in the model
    ///
    /// # Returns
    /// Cache with O(1) lookup for any (entity_id, season_id) pair
    ///
    /// # Performance
    /// - Construction: O(num_specs × num_seasons)
    /// - Memory: ~16 bytes × num_entities × num_seasons
    pub fn new(
        unified_specs: &[UnifiedNoiseSpec],
        num_entities: usize,
        num_seasons: usize,
    ) -> Self {
        let mut transforms = vec![vec![None; num_seasons]; num_entities];

        for spec in unified_specs {
            let entity_id = spec.entity_id;
            if entity_id >= num_entities {
                continue; // Skip out-of-range entities
            }

            // Only process PAR models (Independent models don't need transformation)
            if let TemporalModelSpec::PeriodicAutoregressive {
                seasonal_ar_params: _,
                ..
            } = &spec.temporal_model
            {
                // For each season, extract mean and std_dev
                for (season_id, params) in &spec.seasonal_params {
                    if *season_id < num_seasons {
                        transforms[entity_id][*season_id] =
                            Some(SeasonalTransform {
                                mean: params.mean,
                                std_dev: params.std_dev,
                            });
                    }
                }
            }
        }

        Self {
            transforms,
            num_seasons,
        }
    }

    /// Get transformation parameters for entity and season
    ///
    /// # Returns
    /// - `Some(SeasonalTransform)` if entity has PAR model for this season
    /// - `None` if entity uses Independent model or no spec found
    ///
    /// # Performance
    /// O(1) - direct indexing
    pub fn get(
        &self,
        entity_id: usize,
        season_id: usize,
    ) -> Option<SeasonalTransform> {
        self.transforms
            .get(entity_id)?
            .get(season_id)
            .copied()
            .flatten()
    }

    /// Check if entity requires transformation (has PAR model)
    pub fn needs_transform(&self, entity_id: usize) -> bool {
        self.transforms
            .get(entity_id)
            .map(|seasons| seasons.iter().any(|t| t.is_some()))
            .unwrap_or(false)
    }

    /// Transform observation vector to residuals (in-place)
    ///
    /// # Arguments
    /// * `observations` - Observation values Y_t (modified in-place)
    /// * `season_id` - Current season
    ///
    /// # Performance
    /// O(num_entities) - one pass through observations
    pub fn observations_to_residuals(
        &self,
        observations: &mut [f64],
        season_id: usize,
    ) {
        for (entity_id, obs) in observations.iter_mut().enumerate() {
            if let Some(transform) = self.get(entity_id, season_id) {
                *obs = transform.to_residual(*obs);
            }
            // Else: keep observation unchanged (Independent model)
        }
    }

    /// Transform residual vector to observations (in-place)
    ///
    /// # Performance
    /// O(num_entities) - one pass through residuals
    pub fn residuals_to_observations(
        &self,
        residuals: &mut [f64],
        season_id: usize,
    ) {
        for (entity_id, res) in residuals.iter_mut().enumerate() {
            if let Some(transform) = self.get(entity_id, season_id) {
                *res = transform.to_observation(*res);
            }
            // Else: keep residual unchanged (Independent model)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::UncertaintyType;
    use crate::unified_noise_spec::{SeasonalNoiseParams, SeasonalPARParams};
    use std::collections::HashMap;

    fn create_test_par_spec(
        entity_id: usize,
        num_seasons: usize,
    ) -> UnifiedNoiseSpec {
        let mut seasonal_ar_params = HashMap::new();
        let mut seasonal_params = HashMap::new();

        for season_id in 0..num_seasons {
            seasonal_ar_params.insert(
                season_id,
                SeasonalPARParams {
                    ar_order: 2,
                    ar_coefficients: vec![0.7, 0.2],
                },
            );
            seasonal_params.insert(
                season_id,
                SeasonalNoiseParams {
                    mean: 100.0 + (season_id as f64) * 10.0, // 100, 110, 120, ...
                    std_dev: 20.0,
                    marginal_override: None,
                },
            );
        }

        UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons,
                seasonal_ar_params,
            },
            seasonal_params,
            marginal_distribution: None,
        }
    }

    fn create_test_independent_spec(
        entity_id: usize,
        num_seasons: usize,
    ) -> UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::new();

        for season_id in 0..num_seasons {
            seasonal_params.insert(
                season_id,
                SeasonalNoiseParams {
                    mean: 50.0,
                    std_dev: 10.0,
                    marginal_override: None,
                },
            );
        }

        UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
            marginal_distribution: None,
        }
    }

    #[test]
    fn test_seasonal_transform_to_residual() {
        let transform = SeasonalTransform {
            mean: 100.0,
            std_dev: 20.0,
        };

        // Y = 100 → Z' = 0
        assert!((transform.to_residual(100.0) - 0.0).abs() < 1e-10);

        // Y = 120 → Z' = 1.0
        assert!((transform.to_residual(120.0) - 1.0).abs() < 1e-10);

        // Y = 80 → Z' = -1.0
        assert!((transform.to_residual(80.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_transform_to_observation() {
        let transform = SeasonalTransform {
            mean: 100.0,
            std_dev: 20.0,
        };

        // Z' = 0 → Y = 100
        assert!((transform.to_observation(0.0) - 100.0).abs() < 1e-10);

        // Z' = 1.0 → Y = 120
        assert!((transform.to_observation(1.0) - 120.0).abs() < 1e-10);

        // Z' = -1.0 → Y = 80
        assert!((transform.to_observation(-1.0) - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_transform_round_trip() {
        let transform = SeasonalTransform {
            mean: 100.0,
            std_dev: 20.0,
        };

        let observations = vec![80.0, 100.0, 120.0, 95.0, 105.0];

        for &obs in &observations {
            let residual = transform.to_residual(obs);
            let recovered = transform.to_observation(residual);
            assert!(
                (obs - recovered).abs() < 1e-10,
                "Round trip failed: {} → {} → {}",
                obs,
                residual,
                recovered
            );
        }
    }

    #[test]
    fn test_transform_cache_construction() {
        let specs = vec![
            create_test_par_spec(0, 3), // Entity 0: PAR with 3 seasons
            create_test_par_spec(2, 3), // Entity 2: PAR with 3 seasons
                                        // Entity 1: No spec (Independent)
        ];

        let cache = TransformCache::new(&specs, 4, 3);

        // Entity 0: Has transforms for all seasons
        assert!(cache.needs_transform(0));
        assert!(cache.get(0, 0).is_some());
        assert!(cache.get(0, 1).is_some());
        assert!(cache.get(0, 2).is_some());

        // Entity 1: No transforms (Independent)
        assert!(!cache.needs_transform(1));
        assert!(cache.get(1, 0).is_none());

        // Entity 2: Has transforms
        assert!(cache.needs_transform(2));
        assert!(cache.get(2, 0).is_some());
    }

    #[test]
    fn test_transform_cache_seasonal_params() {
        let specs = vec![create_test_par_spec(0, 3)];
        let cache = TransformCache::new(&specs, 1, 3);

        // Season 0: mean=100, std=20
        let t0 = cache.get(0, 0).unwrap();
        assert!((t0.mean - 100.0).abs() < 1e-10);
        assert!((t0.std_dev - 20.0).abs() < 1e-10);

        // Season 1: mean=110, std=20
        let t1 = cache.get(0, 1).unwrap();
        assert!((t1.mean - 110.0).abs() < 1e-10);
        assert!((t1.std_dev - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_observations_to_residuals() {
        let specs = vec![
            create_test_par_spec(0, 1), // Entity 0: PAR, mean=100, std=20
                                        // Entity 1: Independent (no spec)
        ];
        let cache = TransformCache::new(&specs, 2, 1);

        let mut values = vec![120.0, 50.0]; // [Entity 0, Entity 1]
        cache.observations_to_residuals(&mut values, 0);

        // Entity 0: Y=120 → Z'=1.0
        assert!((values[0] - 1.0).abs() < 1e-10);

        // Entity 1: No transform, keep original
        assert!((values[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_residuals_to_observations() {
        let specs = vec![create_test_par_spec(0, 1)];
        let cache = TransformCache::new(&specs, 2, 1);

        let mut values = vec![1.0, 50.0]; // [Residual for Entity 0, Value for Entity 1]
        cache.residuals_to_observations(&mut values, 0);

        // Entity 0: Z'=1.0 → Y=120
        assert!((values[0] - 120.0).abs() < 1e-10);

        // Entity 1: No transform
        assert!((values[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_std_dev_handling() {
        let transform = SeasonalTransform {
            mean: 100.0,
            std_dev: 0.0, // Degenerate case
        };

        // Should not panic, should return 0.0
        let residual = transform.to_residual(100.0);
        assert!((residual - 0.0).abs() < 1e-10);

        // Inverse should still work (returns mean)
        let obs = transform.to_observation(1.0);
        assert!((obs - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_par_and_independent() {
        let specs = vec![
            create_test_par_spec(0, 1),         // Entity 0: PAR
            create_test_independent_spec(1, 1), // Entity 1: Independent
        ];
        let cache = TransformCache::new(&specs, 2, 1);

        let mut values = vec![120.0, 50.0]; // [PAR entity, Independent entity]
        cache.observations_to_residuals(&mut values, 0);

        // Entity 0 transformed, Entity 1 unchanged
        assert!((values[0] - 1.0).abs() < 1e-10); // Z' = (120-100)/20 = 1.0
        assert!((values[1] - 50.0).abs() < 1e-10); // Unchanged
    }
}
