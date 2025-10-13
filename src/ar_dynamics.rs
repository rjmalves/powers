//! AR Temporal Dynamics Application (CEPEL Pipeline Stage 4)
//!
//! # Overview
//!
//! Applies autoregressive (AR) temporal dynamics to introduce temporal correlation
//! across stages. This is the **final stage** of the CEPEL 4-stage scenario generation
//! pipeline, completing the transformation from independent standard normals to
//! temporally correlated, non-negative stochastic processes.
//!
//! This is **Stage 4** of the CEPEL 4-stage scenario generation pipeline:
//! 1. Base Noise: Generate Z ~ N(0,1) (independent) [`crate::base_noise`]
//! 2. Correlation: Apply W = L×Z → W ~ N(0,R) [`crate::correlation_applicator`]
//! 3. Marginal: Transform to target distributions [`crate::marginal_transformer`]
//! 4. **Temporal: Apply AR dynamics** ← This module
//!
//! # Algorithm
//!
//! **AR(p) Recurrence Relation**:
//!
//! ```text
//! Xₜ = φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ
//!
//! where:
//! - Xₜ: Realization at time t (e.g., hydro inflow)
//! - φᵢ: AR coefficients (must satisfy stationarity conditions)
//! - Xₜ₋ᵢ: Past realizations (lag values)
//! - εₜ: Innovation (from Stage 3: marginal-transformed, correlated noise)
//! ```
//!
//! **For Independent Models**: Xₜ = εₜ (no temporal correlation)
//!
//! **For AR Models**: Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ (temporal correlation via lags)
//!
//! # Lag Buffer Management
//!
//! After computing Xₜ, the lag buffer is updated for the next stage:
//!
//! ```text
//! Before: [Xₜ₋₁, Xₜ₋₂, ..., Xₜ₋ₚ]
//! After:  [Xₜ,   Xₜ₋₁, ..., Xₜ₋ₚ₊₁]  (shift right, discard oldest)
//! ```
//!
//! This ensures that when applying AR dynamics to the next stage t+1, the buffer
//! contains the correct lags [Xₜ, Xₜ₋₁, ..., Xₜ₋ₚ₊₁].
//!
//! # Non-Negativity Enforcement
//!
//! For physical quantities (hydro inflows, loads), negative values are invalid.
//! After applying AR dynamics, values are clamped to zero:
//!
//! ```text
//! Xₜ = max(0, φ₁Xₜ₋₁ + ... + φₚXₜ₋ₚ + εₜ)
//! ```
//!
//! With proper marginal specification (LogNormal3 with appropriate parameters),
//! negative values should be extremely rare (<0.01% of scenarios).
//!
//! # Performance
//!
//! - **Time**: O(S × E × p) where S=scenarios, E=entities, p=lag order
//! - **Space**: O(S × E) for output + O(E × p) for lag buffers
//! - **Target**: <25ms for 1000 scenarios × 10 entities × AR(2)
//!
//! # Statistical Properties
//!
//! ## AR(1) Model: Xₜ = φ₁Xₜ₋₁ + εₜ
//!
//! ```text
//! Autocorrelation:
//!   ACF(1) = φ₁
//!   ACF(2) = φ₁²
//!   ACF(k) = φ₁ᵏ
//!
//! Variance:
//!   Var(Xₜ) = σ²_ε / (1 - φ₁²)
//! ```
//!
//! ## AR(2) Model: Xₜ = φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + εₜ
//!
//! ```text
//! Autocorrelation:
//!   ACF(1) = φ₁ / (1 - φ₂)
//!   ACF(2) = φ₁·ACF(1) + φ₂
//!   ACF(k) = φ₁·ACF(k-1) + φ₂·ACF(k-2)
//! ```
//!
//! # References
//!
//! - CEPEL Technical Reports: Scenario Generation for Hydrothermal Systems
//! - Box, G.E.P., Jenkins, G.M. (2015): "Time Series Analysis: Forecasting and Control"
//! - Brockwell, P.J., Davis, R.A. (2016): "Introduction to Time Series and Forecasting"
//! - PSR SDDP Technical Folder: Autoregressive Models for Hydrological Systems

use crate::input::TemporalModel;
use std::collections::HashMap;

/// AR dynamics applicator for CEPEL pipeline stage 4
///
/// Applies autoregressive temporal dynamics to innovations, introducing temporal
/// correlation across stages. Manages lag buffers for multi-stage application.
///
/// # Examples
///
/// ```
/// use powers_rs::ar_dynamics::ARDynamicsApplicator;
/// use powers_rs::input::TemporalModel;
/// use std::collections::HashMap;
///
/// // Define temporal models for each entity
/// let temporal_models = vec![
///     TemporalModel::Autoregressive {
///         lag_order: 1,
///         coefficients: vec![0.7],  // AR(1) with φ₁=0.7
///     },
///     TemporalModel::Independent,
/// ];
///
/// // Initial lag values for AR entities (from InitialConditionInput)
/// let mut initial_lags = HashMap::new();
/// initial_lags.insert(0, vec![120.0]);  // Entity 0: AR(1) with lag₁=120.0
///
/// let applicator = ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();
///
/// // Innovations from Stage 3 (marginal-transformed, correlated)
/// let innovations = vec![
///     vec![10.0, 5.0],   // scenario 0: [entity_0_innovation, entity_1_innovation]
///     vec![-5.0, 8.0],   // scenario 1
/// ];
///
/// // Apply AR dynamics: Xₜ = φ₁Xₜ₋₁ + εₜ for AR entities, Xₜ = εₜ for independent
/// let (realizations, updated_lags) = applicator.apply_ar_dynamics(&innovations);
///
/// // entity_0: Xₜ = 0.7*120 + 10 = 94.0, then 0.7*94 + (-5) = 60.8
/// // entity_1: Xₜ = εₜ (independent, no AR)
/// assert_eq!(realizations.len(), 2);
/// assert_eq!(realizations[0].len(), 2);
/// ```
#[derive(Debug)]
pub struct ARDynamicsApplicator {
    /// Temporal model specification for each entity
    entity_temporal_models: Vec<TemporalModel>,

    /// Current lag buffers for AR entities
    /// Key: entity index, Value: [Xₜ₋₁, Xₜ₋₂, ..., Xₜ₋ₚ]
    lag_buffers: HashMap<usize, Vec<f64>>,
}

impl ARDynamicsApplicator {
    /// Create a new AR dynamics applicator
    ///
    /// # Arguments
    ///
    /// * `entity_temporal_models` - Temporal model for each entity (length = num_entities)
    ///   - `TemporalModel::Independent`: No AR dynamics (Xₜ = εₜ)
    ///   - `TemporalModel::Autoregressive`: Apply AR(p) dynamics
    /// * `initial_lags` - Initial lag values for AR entities (from `InitialConditionInput`)
    ///   - Key: entity index (must match entities with AR models)
    ///   - Value: [Xₜ₋₁, Xₜ₋₂, ..., Xₜ₋ₚ] (length must match lag_order)
    ///
    /// # Returns
    ///
    /// Applicator ready to apply AR dynamics, or error if validation fails
    ///
    /// # Errors
    ///
    /// - Empty temporal models vector
    /// - AR entity missing initial lags
    /// - Lag count doesn't match lag_order
    /// - Coefficient count doesn't match lag_order
    /// - Invalid lag values (NaN, infinity)
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::ar_dynamics::ARDynamicsApplicator;
    /// use powers_rs::input::TemporalModel;
    /// use std::collections::HashMap;
    ///
    /// let temporal_models = vec![
    ///     TemporalModel::Autoregressive {
    ///         lag_order: 2,
    ///         coefficients: vec![0.6, 0.2],  // AR(2)
    ///     },
    ///     TemporalModel::Independent,
    /// ];
    ///
    /// let mut initial_lags = HashMap::new();
    /// initial_lags.insert(0, vec![120.0, 115.0]);  // Entity 0: [Xₜ₋₁, Xₜ₋₂]
    ///
    /// let applicator = ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();
    /// ```
    pub fn new(
        entity_temporal_models: Vec<TemporalModel>,
        initial_lags: HashMap<usize, Vec<f64>>,
    ) -> Result<Self, String> {
        // Validate non-empty
        if entity_temporal_models.is_empty() {
            return Err("Temporal models vector cannot be empty".to_string());
        }

        // Validate each temporal model and initial lags
        for (entity_idx, temporal_model) in
            entity_temporal_models.iter().enumerate()
        {
            Self::validate_temporal_model(
                temporal_model,
                entity_idx,
                &initial_lags,
            )?;
        }

        Ok(Self {
            entity_temporal_models,
            lag_buffers: initial_lags,
        })
    }

    /// Apply AR dynamics to innovations
    ///
    /// Transforms innovations εₜ (from Stage 3: marginal-transformed, correlated)
    /// to realizations Xₜ via autoregressive dynamics:
    ///
    /// - Independent: Xₜ = εₜ
    /// - AR(p): Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
    ///
    /// After computing Xₜ, lag buffers are updated for the next stage.
    ///
    /// # Arguments
    ///
    /// * `innovations` - Innovations from Stage 3 [scenario][entity]
    ///   - Shape: [num_scenarios][num_entities]
    ///   - Values: Marginal-transformed, correlated noise
    ///   - num_entities must match entity_temporal_models.len()
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `realizations`: Xₜ values [scenario][entity] after AR dynamics
    /// - `updated_lags`: Updated lag buffers for next stage (HashMap<entity_idx, lags>)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - innovations is empty
    /// - num_entities doesn't match entity_temporal_models.len()
    ///
    /// # Performance
    ///
    /// - Time: O(S × E × p) where S=scenarios, E=entities, p=max(lag_order)
    /// - Space: O(S × E) (allocates output)
    /// - Typical: <25ms for 1000 scenarios × 10 entities × AR(2)
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::ar_dynamics::ARDynamicsApplicator;
    /// use powers_rs::input::TemporalModel;
    /// use std::collections::HashMap;
    ///
    /// let temporal_models = vec![
    ///     TemporalModel::Autoregressive {
    ///         lag_order: 1,
    ///         coefficients: vec![0.7],
    ///     },
    /// ];
    ///
    /// let mut initial_lags = HashMap::new();
    /// initial_lags.insert(0, vec![100.0]);
    ///
    /// let applicator = ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();
    ///
    /// let innovations = vec![vec![10.0], vec![5.0]];
    /// let (realizations, updated_lags) = applicator.apply_ar_dynamics(&innovations);
    ///
    /// // Scenario 0: X = 0.7*100 + 10 = 80.0
    /// // Scenario 1: X = 0.7*80 + 5 = 61.0 (uses updated lag from scenario 0)
    /// assert_eq!(realizations[0][0], 80.0);
    /// ```
    pub fn apply_ar_dynamics(
        &self,
        innovations: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, HashMap<usize, Vec<f64>>) {
        assert!(!innovations.is_empty(), "Innovations cannot be empty");

        let num_scenarios = innovations.len();
        let num_entities = innovations[0].len();

        assert_eq!(
            num_entities,
            self.entity_temporal_models.len(),
            "Number of entities ({}) must match temporal models count ({})",
            num_entities,
            self.entity_temporal_models.len()
        );

        // PERFORMANCE: Pre-allocate output with exact capacity
        let mut realizations = Vec::with_capacity(num_scenarios);

        // Clone lag buffers for updating during scenario loop
        let mut current_lags = self.lag_buffers.clone();

        // Apply AR dynamics to each scenario sequentially
        // Note: Scenarios must be processed sequentially because each scenario
        // updates the lag buffer for the next scenario (temporal dependence)
        for scenario in innovations {
            let mut scenario_realizations = Vec::with_capacity(num_entities);

            // Apply AR dynamics to each entity independently
            for (entity_idx, &innovation) in scenario.iter().enumerate() {
                let realization = self.apply_ar_single_entity(
                    innovation,
                    entity_idx,
                    &mut current_lags,
                );

                scenario_realizations.push(realization);
            }

            realizations.push(scenario_realizations);
        }

        (realizations, current_lags)
    }

    /// Apply AR dynamics to a single entity
    ///
    /// # Arguments
    ///
    /// * `innovation` - Innovation εₜ for this entity
    /// * `entity_idx` - Index of entity (for temporal model lookup)
    /// * `lag_buffers` - Mutable reference to lag buffers (updated in-place)
    ///
    /// # Returns
    ///
    /// Realization Xₜ after AR dynamics (clamped to 0 if negative)
    #[inline]
    fn apply_ar_single_entity(
        &self,
        innovation: f64,
        entity_idx: usize,
        lag_buffers: &mut HashMap<usize, Vec<f64>>,
    ) -> f64 {
        match &self.entity_temporal_models[entity_idx] {
            TemporalModel::Independent => {
                // No AR dynamics: Xₜ = εₜ
                // PERFORMANCE: Fast path for independent entities
                innovation
            }
            TemporalModel::Autoregressive {
                coefficients,
                lag_order,
            } => {
                // AR(p) dynamics: Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
                let lags = lag_buffers
                    .get(&entity_idx)
                    .expect("AR entity missing lag buffer");

                // PERFORMANCE: Compute AR component with iterator fusion
                let ar_component: f64 = coefficients
                    .iter()
                    .zip(lags.iter())
                    .map(|(phi, lag)| phi * lag)
                    .sum();

                let x_t = ar_component + innovation;

                // Non-negativity enforcement: Clamp to 0 for physical quantities
                // PERFORMANCE: max(0.0, x) is a single instruction on modern CPUs
                let x_t_clamped = x_t.max(0.0);

                // Update lag buffer: [Xₜ, Xₜ₋₁, ..., Xₜ₋ₚ₊₁]
                // PERFORMANCE: Insert at front and truncate is efficient for small p (<12)
                let lags_mut = lag_buffers.get_mut(&entity_idx).unwrap();
                lags_mut.insert(0, x_t_clamped);
                lags_mut.truncate(*lag_order);

                x_t_clamped
            }
        }
    }

    /// Validate temporal model and initial lags
    ///
    /// # Arguments
    ///
    /// * `temporal_model` - Temporal model to validate
    /// * `entity_idx` - Entity index for error reporting
    /// * `initial_lags` - Initial lag values to validate
    ///
    /// # Returns
    ///
    /// Ok if valid, Err with descriptive message otherwise
    fn validate_temporal_model(
        temporal_model: &TemporalModel,
        entity_idx: usize,
        initial_lags: &HashMap<usize, Vec<f64>>,
    ) -> Result<(), String> {
        match temporal_model {
            TemporalModel::Independent => {
                // No validation needed for independent models
                Ok(())
            }
            TemporalModel::Autoregressive {
                lag_order,
                coefficients,
            } => {
                // Validate coefficient count matches lag order
                if coefficients.len() != *lag_order {
                    return Err(format!(
                        "AR entity[{}]: coefficient count ({}) must match lag_order ({})",
                        entity_idx,
                        coefficients.len(),
                        lag_order
                    ));
                }

                // Validate initial lags exist for AR entity
                let lags = initial_lags.get(&entity_idx).ok_or_else(|| {
                    format!(
                        "AR entity[{}]: missing initial lags (lag_order={})",
                        entity_idx, lag_order
                    )
                })?;

                // Validate lag count matches lag order
                if lags.len() != *lag_order {
                    return Err(format!(
                        "AR entity[{}]: lag count ({}) must match lag_order ({})",
                        entity_idx,
                        lags.len(),
                        lag_order
                    ));
                }

                // Validate lag values are finite
                for (lag_idx, &lag_value) in lags.iter().enumerate() {
                    if !lag_value.is_finite() {
                        return Err(format!(
                            "AR entity[{}]: lag[{}] is not finite (value={})",
                            entity_idx, lag_idx, lag_value
                        ));
                    }
                }

                // Validate lag values are non-negative (physical constraint)
                for (lag_idx, &lag_value) in lags.iter().enumerate() {
                    if lag_value < 0.0 {
                        return Err(format!(
                            "AR entity[{}]: lag[{}] is negative (value={})",
                            entity_idx, lag_idx, lag_value
                        ));
                    }
                }

                Ok(())
            }
        }
    }

    /// Get number of entities
    pub fn num_entities(&self) -> usize {
        self.entity_temporal_models.len()
    }

    /// Get reference to entity temporal models
    pub fn entity_temporal_models(&self) -> &[TemporalModel] {
        &self.entity_temporal_models
    }

    /// Get reference to lag buffers
    pub fn lag_buffers(&self) -> &HashMap<usize, Vec<f64>> {
        &self.lag_buffers
    }
}

/// Compute theoretical autocorrelation function (ACF) for AR(1) model
///
/// For AR(1) with coefficient φ₁: ACF(k) = φ₁ᵏ
///
/// # Arguments
///
/// * `phi1` - AR(1) coefficient
/// * `lag` - Lag k for ACF(k)
///
/// # Returns
///
/// Theoretical ACF(k) value
///
/// # Examples
///
/// ```
/// use powers_rs::ar_dynamics::theoretical_acf_ar1;
///
/// let phi1 = 0.7;
/// assert!((theoretical_acf_ar1(phi1, 0) - 1.0).abs() < 1e-10);  // ACF(0) = 1
/// assert!((theoretical_acf_ar1(phi1, 1) - 0.7).abs() < 1e-10);  // ACF(1) = φ₁
/// assert!((theoretical_acf_ar1(phi1, 2) - 0.49).abs() < 1e-10); // ACF(2) = φ₁²
/// ```
pub fn theoretical_acf_ar1(phi1: f64, lag: usize) -> f64 {
    phi1.powi(lag as i32)
}

/// Compute sample autocorrelation function (ACF) from a time series
///
/// Uses unbiased covariance estimator (dividing by n-k instead of n).
///
/// # Arguments
///
/// * `series` - Time series values
/// * `lag` - Lag k for ACF(k)
///
/// # Returns
///
/// Sample ACF(k) estimate
///
/// # Panics
///
/// Panics if series length <= lag
///
/// # Examples
///
/// ```
/// use powers_rs::ar_dynamics::sample_acf;
///
/// // Perfect AR(1) series with φ₁=0.7
/// let series: Vec<f64> = vec![100.0, 70.0, 49.0, 34.3, 24.01];
/// let acf1 = sample_acf(&series, 1);
///
/// // Should be close to 0.7
/// assert!((acf1 - 0.7).abs() < 0.1);
/// ```
pub fn sample_acf(series: &[f64], lag: usize) -> f64 {
    assert!(
        series.len() > lag,
        "Series length ({}) must be > lag ({})",
        series.len(),
        lag
    );

    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;

    // Compute variance
    let var: f64 =
        series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    // Compute autocovariance at lag k
    let cov: f64 = series[lag..]
        .iter()
        .zip(series[..n - lag].iter())
        .map(|(x, x_lag)| (x - mean) * (x_lag - mean))
        .sum::<f64>()
        / (n - lag) as f64;

    cov / var
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_model_no_ar() {
        // Test: Independent model → Xₜ = εₜ (no AR dynamics)
        let temporal_models = vec![TemporalModel::Independent];
        let initial_lags = HashMap::new(); // No lags needed

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

        let innovations = vec![vec![10.0], vec![5.0], vec![-3.0]];

        let (realizations, updated_lags) =
            applicator.apply_ar_dynamics(&innovations);

        // Independent model: output = input (no AR)
        assert_eq!(realizations.len(), 3);
        assert_eq!(realizations[0][0], 10.0);
        assert_eq!(realizations[1][0], 5.0);
        assert_eq!(realizations[2][0], -3.0); // Negative allowed for independent

        // No lag buffers for independent model
        assert!(updated_lags.is_empty());
    }

    #[test]
    fn test_ar1_model_recurrence() {
        // Test: AR(1) with φ₁=0.7 → Xₜ = 0.7*Xₜ₋₁ + εₜ
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.7],
        }];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![100.0]); // X₀ = 100

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

        let innovations = vec![vec![10.0], vec![5.0]];

        let (realizations, updated_lags) =
            applicator.apply_ar_dynamics(&innovations);

        // Scenario 0: X₁ = 0.7*100 + 10 = 80.0
        assert!((realizations[0][0] - 80.0).abs() < 1e-10);

        // Scenario 1: X₂ = 0.7*80 + 5 = 61.0
        assert!((realizations[1][0] - 61.0).abs() < 1e-10);

        // Updated lags: [X₂] = [61.0]
        assert_eq!(updated_lags.get(&0).unwrap()[0], 61.0);
    }

    #[test]
    fn test_ar2_model_recurrence() {
        // Test: AR(2) with φ₁=0.6, φ₂=0.2 → Xₜ = 0.6*Xₜ₋₁ + 0.2*Xₜ₋₂ + εₜ
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 2,
            coefficients: vec![0.6, 0.2],
        }];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![100.0, 90.0]); // [Xₜ₋₁, Xₜ₋₂]

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

        let innovations = vec![vec![10.0]];

        let (realizations, updated_lags) =
            applicator.apply_ar_dynamics(&innovations);

        // X = 0.6*100 + 0.2*90 + 10 = 60 + 18 + 10 = 88.0
        assert!((realizations[0][0] - 88.0).abs() < 1e-10);

        // Updated lags: [X, Xₜ₋₁] = [88.0, 100.0]
        let lags = updated_lags.get(&0).unwrap();
        assert_eq!(lags.len(), 2);
        assert!((lags[0] - 88.0).abs() < 1e-10);
        assert!((lags[1] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_lag_buffer_update_mechanism() {
        // Test: Lag buffer shifts correctly after each scenario
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 3,
            coefficients: vec![0.5, 0.3, 0.1],
        }];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![100.0, 90.0, 80.0]); // [Xₜ₋₁, Xₜ₋₂, Xₜ₋₃]

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

        let innovations = vec![vec![10.0], vec![5.0]];

        let (_realizations, updated_lags) =
            applicator.apply_ar_dynamics(&innovations);

        // After 2 scenarios, lags should be [X₂, X₁, Xₜ₋₁]
        let lags = updated_lags.get(&0).unwrap();
        assert_eq!(lags.len(), 3);
        // X₁ = 0.5*100 + 0.3*90 + 0.1*80 + 10 = 50 + 27 + 8 + 10 = 95.0
        // X₂ = 0.5*95 + 0.3*100 + 0.1*90 + 5 = 47.5 + 30 + 9 + 5 = 91.5
        assert!((lags[0] - 91.5).abs() < 1e-10);
        assert!((lags[1] - 95.0).abs() < 1e-10);
        assert!((lags[2] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_entities() {
        // Test: Mix of AR(1), Independent, AR(2)
        let temporal_models = vec![
            TemporalModel::Autoregressive {
                lag_order: 1,
                coefficients: vec![0.7],
            },
            TemporalModel::Independent,
            TemporalModel::Autoregressive {
                lag_order: 2,
                coefficients: vec![0.6, 0.2],
            },
        ];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![100.0]);
        initial_lags.insert(2, vec![50.0, 45.0]);

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

        let innovations = vec![vec![10.0, 5.0, 3.0]];

        let (realizations, _) = applicator.apply_ar_dynamics(&innovations);

        assert_eq!(realizations.len(), 1);
        assert_eq!(realizations[0].len(), 3);

        // Entity 0: AR(1) → X = 0.7*100 + 10 = 80.0
        assert!((realizations[0][0] - 80.0).abs() < 1e-10);

        // Entity 1: Independent → X = 5.0
        assert_eq!(realizations[0][1], 5.0);

        // Entity 2: AR(2) → X = 0.6*50 + 0.2*45 + 3 = 30 + 9 + 3 = 42.0
        assert!((realizations[0][2] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_negativity_enforcement() {
        // Test: Negative AR result clamped to 0
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.5],
        }];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![10.0]);

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

        // Large negative innovation: 0.5*10 + (-20) = -15 → clamped to 0
        let innovations = vec![vec![-20.0]];

        let (realizations, updated_lags) =
            applicator.apply_ar_dynamics(&innovations);

        // Result should be clamped to 0
        assert_eq!(realizations[0][0], 0.0);

        // Lag buffer should contain clamped value (0.0, not -15.0)
        assert_eq!(updated_lags.get(&0).unwrap()[0], 0.0);
    }

    #[test]
    fn test_validation_rejects_missing_lags() {
        // Test: AR entity without initial lags rejected
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.7],
        }];

        let initial_lags = HashMap::new(); // Missing lag for entity 0

        let result = ARDynamicsApplicator::new(temporal_models, initial_lags);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing initial lags"));
    }

    #[test]
    fn test_validation_rejects_wrong_lag_count() {
        // Test: Lag count doesn't match lag_order
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 2,
            coefficients: vec![0.6, 0.2],
        }];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![100.0]); // Only 1 lag, need 2

        let result = ARDynamicsApplicator::new(temporal_models, initial_lags);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("lag count"));
    }

    #[test]
    fn test_validation_rejects_invalid_lag_values() {
        // Test: NaN/infinity lag values rejected
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.7],
        }];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![f64::NAN]);

        let result = ARDynamicsApplicator::new(temporal_models, initial_lags);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not finite"));
    }

    #[test]
    fn test_validation_rejects_negative_lags() {
        // Test: Negative lag values rejected (physical constraint)
        let temporal_models = vec![TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.7],
        }];

        let mut initial_lags = HashMap::new();
        initial_lags.insert(0, vec![-10.0]);

        let result = ARDynamicsApplicator::new(temporal_models, initial_lags);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("negative"));
    }

    #[test]
    fn test_empty_temporal_models_rejected() {
        // Test: Empty temporal models vector rejected
        let temporal_models = vec![];
        let initial_lags = HashMap::new();

        let result = ARDynamicsApplicator::new(temporal_models, initial_lags);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot be empty"));
    }

    #[test]
    #[should_panic(expected = "Innovations cannot be empty")]
    fn test_empty_innovations_panics() {
        // Test: Empty innovations panic
        let temporal_models = vec![TemporalModel::Independent];
        let initial_lags = HashMap::new();

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();
        let _ = applicator.apply_ar_dynamics(&[]);
    }

    #[test]
    #[should_panic(expected = "must match temporal models count")]
    fn test_dimension_mismatch_panics() {
        // Test: Dimension mismatch panics
        let temporal_models =
            vec![TemporalModel::Independent, TemporalModel::Independent];
        let initial_lags = HashMap::new();

        let applicator =
            ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

        // Only 1 entity in innovations, but 2 temporal models
        let innovations = vec![vec![10.0]];
        let _ = applicator.apply_ar_dynamics(&innovations);
    }

    #[test]
    fn test_theoretical_acf_ar1() {
        // Test: Theoretical ACF for AR(1)
        let phi1 = 0.7;

        assert!((theoretical_acf_ar1(phi1, 0) - 1.0).abs() < 1e-10);
        assert!((theoretical_acf_ar1(phi1, 1) - 0.7).abs() < 1e-10);
        assert!((theoretical_acf_ar1(phi1, 2) - 0.49).abs() < 1e-10);
        assert!((theoretical_acf_ar1(phi1, 3) - 0.343).abs() < 1e-10);
    }

    #[test]
    fn test_sample_acf_perfect_ar1() {
        // Test: Sample ACF on perfect AR(1) series
        // Generate series: X₀=100, Xₜ = 0.7*Xₜ₋₁ (no innovation for simplicity)
        let phi1 = 0.7;
        let mut series = vec![100.0];
        for _ in 0..100 {
            let x_t = phi1 * series.last().unwrap();
            series.push(x_t);
        }

        let acf1 = sample_acf(&series, 1);
        let acf2 = sample_acf(&series, 2);

        // Should match theoretical ACF closely
        assert!((acf1 - phi1).abs() < 0.01);
        assert!((acf2 - phi1 * phi1).abs() < 0.01);
    }
}
