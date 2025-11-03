//! Unified temporal model for uncertainty representation
//!
//! This module provides a single, unified representation for all temporal
//! uncertainty models, eliminating the artificial distinction between
//! "Independent" and "PAR" models. Independent models are simply PAR(0)
//! models with ar_orders = [0, 0, ...].

use crate::error::PowersError;
use crate::input::{MarginalDistribution, UncertaintyType};

/// Unified temporal model for all entities
///
/// This struct replaces the old `UncertaintyModel` enum, which had separate
/// variants for Independent and PeriodicAR. By recognizing that Independent
/// is just PAR(0), we can use a single representation for everything.
///
/// # Fields
///
/// - `entity_type`: Load or Inflow
/// - `entity_id`: Index within that entity type
/// - `num_seasons`: Number of seasons in the cycle (e.g., 12 for monthly)
/// - `seasonal_means`: [μ₀, μ₁, ..., μₙ₋₁] - used in LP constraints
/// - `seasonal_stds`: [σ₀, σ₁, ..., σₙ₋₁] - used in LP constraints
/// - `seasonal_distributions`: Marginal distributions for inverse CDF transform
/// - `ar_orders`: [p₀, p₁, ..., pₙ₋₁] - AR order per season (0 for independent)
/// - `ar_coefficients`: AR coefficients per season (empty vec for independent)
///
/// # Precomputed Fields (for efficiency)
///
/// - `max_ar_order`: max(ar_orders) - determines lag buffer size
/// - `psi_coefficients`: Transformed AR coefficients for LP constraints
/// - `deterministic_bases`: μₛ - Σ(φᵢ·μₛ₋ᵢ) per season (precomputed)
///
/// # Example: Independent Model
///
/// ```ignore
/// TemporalModel {
///     ar_orders: vec![0, 0, 0, ..., 0],           // All zeros
///     ar_coefficients: vec![vec![], vec![], ...], // All empty
///     psi_coefficients: vec![vec![], vec![], ...],// All empty
///     deterministic_bases: seasonal_means.clone(),// No AR adjustment
///     max_ar_order: 0,
///     // ... other fields
/// }
/// ```
///
/// # Example: PAR(1) Model
///
/// ```ignore
/// TemporalModel {
///     ar_orders: vec![1, 1, 1, ..., 1],
///     ar_coefficients: vec![vec![0.7], vec![0.7], ...],
///     psi_coefficients: vec![vec![0.7], vec![0.7], ...], // After transformation
///     deterministic_bases: /* precomputed per season */,
///     max_ar_order: 1,
///     // ... other fields
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TemporalModel {
    // Entity identification
    pub entity_type: UncertaintyType,
    pub entity_id: usize,

    // Temporal structure
    pub num_seasons: usize,

    // Seasonal parameters (for LP constraints)
    pub seasonal_means: Vec<f64>,
    pub seasonal_stds: Vec<f64>,

    // Marginal distributions (for inverse CDF transformation)
    pub seasonal_distributions: Vec<MarginalDistribution>,

    // AR structure (ar_orders can be all zeros for independent models)
    pub ar_orders: Vec<usize>,
    pub ar_coefficients: Vec<Vec<f64>>,

    // Precomputed for efficiency
    pub max_ar_order: usize,
    pub psi_coefficients: Vec<Vec<f64>>,
    pub deterministic_bases: Vec<f64>,
}

/// Lightweight seasonal parameters (copied per use)
#[derive(Debug, Clone)]
pub struct SeasonalParams {
    pub mean: f64,
    pub std_dev: f64,
    pub distribution: MarginalDistribution,
}

impl TemporalModel {
    /// Create an Independent model (PAR(0))
    ///
    /// # Arguments
    ///
    /// * `entity_type` - Load or Inflow
    /// * `entity_id` - Entity index within type
    /// * `seasonal_means` - Mean for each season
    /// * `seasonal_stds` - Standard deviation for each season
    /// * `seasonal_distributions` - Marginal distribution for each season
    pub fn from_independent(
        entity_type: UncertaintyType,
        entity_id: usize,
        seasonal_means: Vec<f64>,
        seasonal_stds: Vec<f64>,
        seasonal_distributions: Vec<MarginalDistribution>,
    ) -> Result<Self, PowersError> {
        let num_seasons = seasonal_means.len();

        // Validate lengths
        if seasonal_stds.len() != num_seasons {
            return Err(PowersError::Other(
                "seasonal_stds length must match seasonal_means".to_string(),
            ));
        }
        if seasonal_distributions.len() != num_seasons {
            return Err(PowersError::Other(
                "seasonal_distributions length must match seasonal_means"
                    .to_string(),
            ));
        }

        // Independent model: all AR orders are 0
        let ar_orders = vec![0; num_seasons];
        let ar_coefficients = vec![vec![]; num_seasons];
        let psi_coefficients = vec![vec![]; num_seasons];

        // For independent model, deterministic base is just the mean
        let deterministic_bases = seasonal_means.clone();

        Ok(Self {
            entity_type,
            entity_id,
            num_seasons,
            seasonal_means,
            seasonal_stds,
            seasonal_distributions,
            ar_orders,
            ar_coefficients,
            max_ar_order: 0,
            psi_coefficients,
            deterministic_bases,
        })
    }

    /// Create a PAR model
    ///
    /// # Arguments
    ///
    /// * `entity_type` - Load or Inflow
    /// * `entity_id` - Entity index within type
    /// * `num_seasons` - Number of seasons
    /// * `seasonal_means` - Mean for each season
    /// * `seasonal_stds` - Standard deviation for each season
    /// * `seasonal_distributions` - Marginal distribution for each season
    /// * `ar_orders` - AR order for each season
    /// * `ar_coefficients` - AR coefficients for each season [φ₁, φ₂, ..., φₚ]
    #[allow(clippy::too_many_arguments)]
    pub fn from_par(
        entity_type: UncertaintyType,
        entity_id: usize,
        num_seasons: usize,
        seasonal_means: Vec<f64>,
        seasonal_stds: Vec<f64>,
        seasonal_distributions: Vec<MarginalDistribution>,
        ar_orders: Vec<usize>,
        ar_coefficients: Vec<Vec<f64>>,
    ) -> Result<Self, PowersError> {
        // Validate lengths
        if seasonal_means.len() != num_seasons {
            return Err(PowersError::Other(
                "seasonal_means length must match num_seasons".to_string(),
            ));
        }
        if seasonal_stds.len() != num_seasons {
            return Err(PowersError::Other(
                "seasonal_stds length must match num_seasons".to_string(),
            ));
        }
        if seasonal_distributions.len() != num_seasons {
            return Err(PowersError::Other(
                "seasonal_distributions length must match num_seasons"
                    .to_string(),
            ));
        }
        if ar_orders.len() != num_seasons {
            return Err(PowersError::Other(
                "ar_orders length must match num_seasons".to_string(),
            ));
        }
        if ar_coefficients.len() != num_seasons {
            return Err(PowersError::Other(
                "ar_coefficients length must match num_seasons".to_string(),
            ));
        }

        // Validate AR coefficients match AR orders
        for (season, (&order, coeffs)) in
            ar_orders.iter().zip(&ar_coefficients).enumerate()
        {
            if coeffs.len() != order {
                return Err(PowersError::Other(format!(
                    "Season {}: ar_coefficients length {} doesn't match ar_order {}",
                    season,
                    coeffs.len(),
                    order
                )));
            }
        }

        let max_ar_order = *ar_orders.iter().max().unwrap_or(&0);

        // Compute psi coefficients (for PAR, psi = phi initially, may be transformed)
        // For now, just copy the coefficients
        let psi_coefficients = ar_coefficients.clone();

        // Compute deterministic bases: μₛ - Σ(φᵢ·μₛ₋ᵢ)
        let deterministic_bases = Self::compute_deterministic_bases(
            &seasonal_means,
            &ar_coefficients,
            num_seasons,
        );

        Ok(Self {
            entity_type,
            entity_id,
            num_seasons,
            seasonal_means,
            seasonal_stds,
            seasonal_distributions,
            ar_orders,
            ar_coefficients,
            max_ar_order,
            psi_coefficients,
            deterministic_bases,
        })
    }

    /// Compute deterministic bases: μₛ - Σ(φᵢ·μₛ₋ᵢ)
    fn compute_deterministic_bases(
        seasonal_means: &[f64],
        ar_coefficients: &[Vec<f64>],
        num_seasons: usize,
    ) -> Vec<f64> {
        let mut bases = Vec::with_capacity(num_seasons);

        for season in 0..num_seasons {
            let mut base = seasonal_means[season];
            let coeffs = &ar_coefficients[season];

            // Subtract Σ(φᵢ·μₛ₋ᵢ)
            for (lag, &phi) in coeffs.iter().enumerate() {
                let lag_season = if season > lag {
                    season - lag - 1
                } else {
                    num_seasons + season - lag - 1
                };
                base -= phi * seasonal_means[lag_season];
            }

            bases.push(base);
        }

        bases
    }

    /// Create from UncertaintySpecification (JSON input)
    pub fn from_specification(
        spec: &crate::input::UncertaintySpecification,
    ) -> Result<Self, PowersError> {
        // Convert temporal model input to unified format
        let unified_input = spec.temporal_model.clone();

        // Extract distributions
        let seasonal_distributions: Vec<_> = spec
            .seasonal_distributions
            .as_ref()
            .ok_or_else(|| {
                PowersError::Other(
                    "seasonal_distributions required for TemporalModel"
                        .to_string(),
                )
            })?
            .iter()
            .map(|s| s.distribution.clone())
            .collect();

        Self::from_par(
            spec.uncertainty_type,
            spec.entity_id,
            unified_input.num_seasons,
            unified_input.seasonal_means,
            unified_input.seasonal_stds,
            seasonal_distributions,
            unified_input.ar_orders,
            unified_input.ar_coefficients,
        )
    }

    /// Get seasonal parameters for a given season
    pub fn seasonal_params(&self, season_id: usize) -> SeasonalParams {
        SeasonalParams {
            mean: self.seasonal_means[season_id],
            std_dev: self.seasonal_stds[season_id],
            distribution: self.seasonal_distributions[season_id].clone(),
        }
    }

    /// Get entity type
    pub fn entity_type(&self) -> UncertaintyType {
        self.entity_type
    }

    /// Get entity ID
    pub fn entity_id(&self) -> usize {
        self.entity_id
    }

    /// Check if this model has AR dynamics (max_ar_order > 0)
    pub fn is_autoregressive(&self) -> bool {
        self.max_ar_order > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_dist(mean: f64, std_dev: f64) -> MarginalDistribution {
        MarginalDistribution::Normal { mean, std_dev }
    }

    #[test]
    fn test_from_independent() {
        let means = vec![100.0, 110.0, 120.0];
        let stds = vec![10.0, 15.0, 20.0];
        let dists = vec![
            make_normal_dist(100.0, 10.0),
            make_normal_dist(110.0, 15.0),
            make_normal_dist(120.0, 20.0),
        ];

        let model = TemporalModel::from_independent(
            UncertaintyType::Load,
            0,
            means.clone(),
            stds.clone(),
            dists,
        )
        .unwrap();

        assert_eq!(model.entity_type, UncertaintyType::Load);
        assert_eq!(model.entity_id, 0);
        assert_eq!(model.num_seasons, 3);
        assert_eq!(model.seasonal_means, means);
        assert_eq!(model.seasonal_stds, stds);
        assert_eq!(model.ar_orders, vec![0, 0, 0]);
        assert_eq!(model.max_ar_order, 0);
        assert!(!model.is_autoregressive());

        // Deterministic bases should equal means for independent model
        assert_eq!(model.deterministic_bases, means);
    }

    #[test]
    fn test_from_par() {
        let num_seasons = 3;
        let means = vec![70.0, 65.0, 60.0];
        let stds = vec![20.0, 20.0, 20.0];
        let dists = vec![
            make_normal_dist(70.0, 20.0),
            make_normal_dist(65.0, 20.0),
            make_normal_dist(60.0, 20.0),
        ];
        let ar_orders = vec![1, 1, 1];
        let ar_coefficients = vec![vec![0.7], vec![0.7], vec![0.7]];

        let model = TemporalModel::from_par(
            UncertaintyType::Inflow,
            0,
            num_seasons,
            means.clone(),
            stds,
            dists,
            ar_orders.clone(),
            ar_coefficients.clone(),
        )
        .unwrap();

        assert_eq!(model.num_seasons, num_seasons);
        assert_eq!(model.ar_orders, ar_orders);
        assert_eq!(model.ar_coefficients, ar_coefficients);
        assert_eq!(model.max_ar_order, 1);
        assert!(model.is_autoregressive());

        // Check deterministic bases are computed
        assert_eq!(model.deterministic_bases.len(), num_seasons);

        // For season 0: base = 70.0 - 0.7 * 60.0 = 28.0
        assert!((model.deterministic_bases[0] - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_par_with_varying_orders() {
        let num_seasons = 3;
        let means = vec![50.0, 60.0, 70.0];
        let stds = vec![10.0, 10.0, 10.0];
        let dists = vec![
            make_normal_dist(50.0, 10.0),
            make_normal_dist(60.0, 10.0),
            make_normal_dist(70.0, 10.0),
        ];
        let ar_orders = vec![0, 1, 2]; // Varying orders
        let ar_coefficients = vec![vec![], vec![0.5], vec![0.4, 0.3]];

        let model = TemporalModel::from_par(
            UncertaintyType::Inflow,
            0,
            num_seasons,
            means,
            stds,
            dists,
            ar_orders,
            ar_coefficients,
        )
        .unwrap();

        assert_eq!(model.max_ar_order, 2);
        assert!(model.is_autoregressive());
    }

    #[test]
    fn test_seasonal_params() {
        let means = vec![100.0, 110.0, 120.0];
        let stds = vec![10.0, 15.0, 20.0];
        let dists = vec![
            make_normal_dist(100.0, 10.0),
            make_normal_dist(110.0, 15.0),
            make_normal_dist(120.0, 20.0),
        ];

        let model = TemporalModel::from_independent(
            UncertaintyType::Load,
            0,
            means,
            stds,
            dists,
        )
        .unwrap();

        let params = model.seasonal_params(1);
        assert_eq!(params.mean, 110.0);
        assert_eq!(params.std_dev, 15.0);
    }

    #[test]
    fn test_validation_mismatched_lengths() {
        let means = vec![100.0, 110.0];
        let stds = vec![10.0]; // Wrong length
        let dists =
            vec![make_normal_dist(100.0, 10.0), make_normal_dist(110.0, 15.0)];

        let result = TemporalModel::from_independent(
            UncertaintyType::Load,
            0,
            means,
            stds,
            dists,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_validation_ar_coefficients_length() {
        let num_seasons = 2;
        let means = vec![50.0, 60.0];
        let stds = vec![10.0, 10.0];
        let dists =
            vec![make_normal_dist(50.0, 10.0), make_normal_dist(60.0, 10.0)];
        let ar_orders = vec![2, 1];
        let ar_coefficients = vec![vec![0.5], vec![0.4]]; // First season has wrong length

        let result = TemporalModel::from_par(
            UncertaintyType::Inflow,
            0,
            num_seasons,
            means,
            stds,
            dists,
            ar_orders,
            ar_coefficients,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_compute_deterministic_bases_simple() {
        let means = vec![100.0];
        let ar_coeffs = vec![vec![]];

        let bases =
            TemporalModel::compute_deterministic_bases(&means, &ar_coeffs, 1);

        assert_eq!(bases, vec![100.0]);
    }

    #[test]
    fn test_compute_deterministic_bases_with_ar() {
        let means = vec![70.0, 65.0, 60.0];
        let ar_coeffs = vec![vec![0.7], vec![0.7], vec![0.7]];

        let bases =
            TemporalModel::compute_deterministic_bases(&means, &ar_coeffs, 3);

        // Season 0: 70.0 - 0.7 * 60.0 = 28.0
        assert!((bases[0] - 28.0).abs() < 1e-10);
        // Season 1: 65.0 - 0.7 * 70.0 = 16.0
        assert!((bases[1] - 16.0).abs() < 1e-10);
        // Season 2: 60.0 - 0.7 * 65.0 = 14.5
        assert!((bases[2] - 14.5).abs() < 1e-10);
    }
}
