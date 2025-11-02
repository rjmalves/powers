//! Pre-computed observation-space inflow scenarios for SDDP
//!
//! # Mathematical Foundation
//!
//! Given a PAR model with residual-space AR coefficients φ_i, we transform to
//! observation-space coefficients ψ_i and pre-compute the noise term η_t:
//!
//! ```text
//! Transformed coefficients: ψ_i = φ_i * (σ_t / σ_{t-i})
//! Noise term: η_t = -Σ[ψ_i * μ_{t-i}] + μ_t + σ_t * ε_t
//! Observation: Y_t = Σ ψ_i * Y_{t-i} + η_t
//! ```

use crate::uncertainty_model::{
    DistributionType, SeasonalParams, UncertaintyModel,
};

/// Pre-computed observation-space scenario for one inflow entity
#[derive(Debug, Clone)]
pub struct PrecomputedInflowScenario {
    /// Transformed AR coefficients ψ_i = φ_i * (σ_t / σ_{t-i})
    pub transformed_coefficients: Vec<f64>,
    /// Pre-computed noise term η_t = -Σ[ψ_i * μ_{t-i}] + μ_t + σ_t * ε_t
    pub noise_term: f64,
    /// Final observation Y_t = Σ ψ_i * Y_{t-i} + η_t
    pub observation: f64,
    pub hydro_id: usize,
}

impl PrecomputedInflowScenario {
    pub fn from_independent(
        hydro_id: usize,
        seasonal_params: SeasonalParams,
        innovation: f64,
    ) -> Self {
        // Handle distribution-specific noise term calculation
        let noise_term = match seasonal_params.distribution {
            DistributionType::Normal => {
                // η_t = μ_t + σ_t * ε_t
                seasonal_params.mean + seasonal_params.std_dev * innovation
            }
            DistributionType::LogNormal3 { .. } => {
                // η_t = μ_t + innovation
                seasonal_params.mean + innovation
            }
        };

        // Y_t = η_t (no lags)
        let observation = noise_term;

        Self {
            transformed_coefficients: Vec::new(),
            noise_term,
            observation,
            hydro_id,
        }
    }

    pub fn from_periodic_ar(
        hydro_id: usize,
        current_params: SeasonalParams,
        ar_coefficients: &[f64],
        lag_params: &[SeasonalParams],
        lag_observations: &[f64],
        innovation: f64,
    ) -> Self {
        let ar_order = ar_coefficients.len();

        // Step 1: Transform coefficients ψ_i = φ_i * (σ_t / σ_{t-i})
        let mut transformed_coefficients = Vec::with_capacity(ar_order);
        for i in 0..ar_order {
            let psi_i = ar_coefficients[i]
                * (current_params.std_dev / lag_params[i].std_dev);
            transformed_coefficients.push(psi_i);
        }

        // Step 2: Calculate deterministic component of noise term
        // det_component = -Σ[ψ_i * μ_{t-i}] + μ_t
        let mut deterministic_term = current_params.mean;
        for i in 0..ar_order {
            deterministic_term -=
                transformed_coefficients[i] * lag_params[i].mean;
        }

        // Step 3: Add stochastic component based on distribution type
        let noise_term = match current_params.distribution {
            DistributionType::Normal => {
                // For Normal: η_t = det_component + σ_t * ε_t
                deterministic_term + current_params.std_dev * innovation
            }
            DistributionType::LogNormal3 { .. } => {
                // For LogNormal3: innovation is already the transformed value
                // This breaks mathematical purity but ensures non-negative inflows.
                // This is an accepted mathematical approximation for log-normal models.
                // η_t = det_component + innovation (already includes the transformation)
                deterministic_term + innovation
            }
        };

        // Step 4: Calculate final observation
        // Y_t = Σ ψ_i * Y_{t-i} + η_t
        let mut observation = noise_term;
        for i in 0..ar_order {
            observation += transformed_coefficients[i] * lag_observations[i];
        }

        Self {
            transformed_coefficients,
            noise_term,
            observation,
            hydro_id,
        }
    }

    pub fn from_par_model(
        model: &UncertaintyModel,
        season_id: usize,
        innovation: f64,
        lag_observations: &[f64],
    ) -> Result<Self, String> {
        let hydro_id = model.entity_id();
        let current_params = model.seasonal_params(season_id);

        match model {
            UncertaintyModel::Independent { .. } => {
                Ok(Self::from_independent(hydro_id, current_params, innovation))
            }
            UncertaintyModel::PeriodicAR { par_params, .. } => {
                let ar_coefficients = par_params.ar_coefficients(season_id);
                let ar_order = ar_coefficients.len();

                // Validate lag observations length
                if lag_observations.len() < ar_order {
                    return Err(format!(
                        "Insufficient lag observations: got {}, need {} for season {}",
                        lag_observations.len(),
                        ar_order,
                        season_id
                    ));
                }

                // Get seasonal parameters for each lag
                let num_seasons = par_params.num_seasons;
                let mut lag_params = Vec::with_capacity(ar_order);
                for lag in 1..=ar_order {
                    // Handle seasonal wrapping correctly (prevent underflow)
                    let lag_season =
                        ((season_id + num_seasons * 2) - lag) % num_seasons;
                    lag_params.push(par_params.seasonal_params(lag_season));
                }

                Ok(Self::from_periodic_ar(
                    hydro_id,
                    current_params,
                    ar_coefficients,
                    &lag_params,
                    &lag_observations[..ar_order],
                    innovation,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncertainty_model::DistributionType;

    #[test]
    fn test_independent_model() {
        let params = SeasonalParams {
            mean: 100.0,
            std_dev: 20.0,
            distribution: DistributionType::Normal,
        };

        let innovation = 0.5;
        let scenario =
            PrecomputedInflowScenario::from_independent(0, params, innovation);

        // η_t = μ + σ * ε = 100 + 20 * 0.5 = 110
        assert_eq!(scenario.noise_term, 110.0);
        assert_eq!(scenario.observation, 110.0);
        assert!(scenario.transformed_coefficients.is_empty());
        assert_eq!(scenario.hydro_id, 0);
    }

    #[test]
    fn test_par1_model() {
        let current = SeasonalParams {
            mean: 150.0,
            std_dev: 30.0,
            distribution: DistributionType::Normal,
        };

        let lag = SeasonalParams {
            mean: 100.0,
            std_dev: 20.0,
            distribution: DistributionType::Normal,
        };

        let phi = vec![0.7];
        let lag_obs = vec![120.0];
        let innovation = 0.5;

        let scenario = PrecomputedInflowScenario::from_periodic_ar(
            0,
            current,
            &phi,
            &[lag],
            &lag_obs,
            innovation,
        );

        // ψ_1 = φ_1 * (σ_t / σ_{t-1}) = 0.7 * (30/20) = 1.05
        assert!((scenario.transformed_coefficients[0] - 1.05).abs() < 1e-10);

        // η_t = -ψ_1*μ_{t-1} + μ_t + σ_t*ε
        //     = -1.05*100 + 150 + 30*0.5
        //     = -105 + 150 + 15 = 60
        assert!((scenario.noise_term - 60.0).abs() < 1e-10);

        // Y_t = ψ_1*Y_{t-1} + η_t = 1.05*120 + 60 = 126 + 60 = 186
        assert!((scenario.observation - 186.0).abs() < 1e-10);
    }

    #[test]
    fn test_par2_model() {
        let current = SeasonalParams {
            mean: 150.0,
            std_dev: 30.0,
            distribution: DistributionType::Normal,
        };

        let lag1 = SeasonalParams {
            mean: 100.0,
            std_dev: 20.0,
            distribution: DistributionType::Normal,
        };

        let lag2 = SeasonalParams {
            mean: 80.0,
            std_dev: 15.0,
            distribution: DistributionType::Normal,
        };

        let phi = vec![0.7, 0.3];
        let lag_obs = vec![120.0, 90.0];
        let innovation = 0.5;

        let scenario = PrecomputedInflowScenario::from_periodic_ar(
            0,
            current,
            &phi,
            &[lag1, lag2],
            &lag_obs,
            innovation,
        );

        // ψ_1 = 0.7 * (30/20) = 1.05
        assert!((scenario.transformed_coefficients[0] - 1.05).abs() < 1e-10);

        // ψ_2 = 0.3 * (30/15) = 0.6
        assert!((scenario.transformed_coefficients[1] - 0.6).abs() < 1e-10);

        // η_t = -ψ_1*μ_{t-1} - ψ_2*μ_{t-2} + μ_t + σ_t*ε
        //     = -1.05*100 - 0.6*80 + 150 + 30*0.5
        //     = -105 - 48 + 150 + 15 = 12
        assert!((scenario.noise_term - 12.0).abs() < 1e-10);

        // Y_t = ψ_1*Y_{t-1} + ψ_2*Y_{t-2} + η_t
        //     = 1.05*120 + 0.6*90 + 12
        //     = 126 + 54 + 12 = 192
        assert!((scenario.observation - 192.0).abs() < 1e-10);
    }
}
