//! Unified internal representation for noise specifications
//!
//! This module provides `UnifiedNoiseSpec`, an internal data structure that separates
//! entity-level uncertainty specifications from season-level parameters.
//!
//! # Architecture
//!
//! The `UnifiedNoiseSpec` structure enables:
//! - **O(1) lookups**: Season parameters stored in `HashMap<usize, SeasonalNoiseParams>`
//! - **Clear semantics**: Entity-level temporal models separated from seasonal parameters
//! - **Performance**: Pre-computed lookups replace O(n) linear searches
//! - **Flexibility**: Supports both independent and PAR temporal models
//!
//! # Usage
//!
//! This is an **internal representation** used by the scenario generation pipeline.
//! It is not directly exposed in JSON input files.

use crate::input::{
    GraphInput, MarginalDistribution, SystemInput, TemporalModelInput,
    UncertaintySpecification, UncertaintyType,
};
use std::collections::{HashMap, HashSet};

/// Unified internal representation for noise specifications
///
/// Separates entity-level concerns (which entity, what temporal model) from
/// season-level parameters (mean, std_dev per season). This design enables:
/// - O(1) lookup of seasonal parameters via HashMap
/// - Clear separation of temporal dynamics from seasonal statistics
/// - Support for both independent and PAR models
#[derive(Debug, Clone)]
pub struct UnifiedNoiseSpec {
    pub uncertainty_type: UncertaintyType,
    pub entity_id: usize,
    pub temporal_model: TemporalModelSpec,
    /// Key: season_id (0..num_seasons-1)
    pub seasonal_params: HashMap<usize, SeasonalNoiseParams>,
}

/// Temporal model specification for internal representation
///
/// Distinguishes between independent (no correlation) and PAR (seasonal AR) models.
/// This is separate from the public `TemporalModel` enum to allow internal
/// optimizations and clearer semantics.
///
/// # Variants
///
/// - `Independent`: No temporal correlation (each season independent)
/// - `PeriodicAutoregressive`: PAR(p) with seasonal parameters
///
/// # Design Note
///
/// We use a separate enum from `input::TemporalModel` to:
/// - Decouple internal representation from JSON schema
/// - Allow future extensions (e.g., caching, pre-computed structures)
/// - Provide a stable interface for the scenario generation pipeline
#[derive(Debug, Clone)]
pub enum TemporalModelSpec {
    /// Independent process (no temporal correlation)
    ///
    /// Each time step is independent. No AR dynamics.
    /// Seasonal parameters are still respected (different mean/std_dev per season).
    Independent,

    /// Periodic Autoregressive PAR(p) model
    ///
    /// AR parameters vary by season. Contains:
    /// - `num_seasons`: Seasonal cycle length (e.g., 12 for monthly)
    /// - `seasonal_ar_params`: HashMap of AR parameters per season
    PeriodicAutoregressive {
        num_seasons: usize,
        seasonal_ar_params: HashMap<usize, SeasonalPARParams>,
    },
}

/// Seasonal noise parameters (mean, std_dev, optional marginal override)
///
/// Contains the statistical parameters for one season. These are used for:
/// - Independent models: Direct sampling parameters
/// - PAR models: Seasonal mean μₘ and std_dev σₘ in PAR equation
#[derive(Debug, Clone)]
pub struct SeasonalNoiseParams {
    pub mean: f64,
    pub std_dev: f64,
    pub marginal_override: Option<MarginalDistribution>,
}

/// Seasonal PAR parameters (AR order and coefficients)
///
/// Contains the AR structure for one season in a PAR model. Each season can
/// have a different AR order (e.g., AR(1) in dry season, AR(2) in wet season).
///
/// # Fields
///
/// - `ar_order`: AR order pₘ for this season (must be > 0)
/// - `ar_coefficients`: AR coefficients [φ₁ₘ, φ₂ₘ, ..., φₚₘ] (length = ar_order)
///
/// # Performance
///
/// Vec<f64> for coefficients is small (typically 1-3 elements) and stored inline
/// in the HashMap. No additional heap allocations beyond the Vec itself.
///
/// # Validation
///
/// - `ar_order` must be > 0
/// - `ar_coefficients.len()` must equal `ar_order`
#[derive(Debug, Clone)]
pub struct SeasonalPARParams {
    /// AR order for this season (pₘ, must be > 0)
    ///
    /// Example: ar_order = 2 means AR(2) for this season
    pub ar_order: usize,

    /// AR coefficients [φ₁ₘ, φ₂ₘ, ..., φₚₘ]
    ///
    /// Length must equal `ar_order`.
    /// Example: For AR(2), ar_coefficients = [0.6, 0.2]
    ///
    /// # Validation
    ///
    /// Must satisfy: `ar_coefficients.len() == ar_order`
    pub ar_coefficients: Vec<f64>,
}

impl UnifiedNoiseSpec {
    /// Validate semantic constraints of the unified noise spec
    ///
    /// Comprehensive validation checking:
    /// 1. **Seasonal parameters**: positive std_dev, finite values
    /// 2. **PAR consistency**: all seasons defined, matching counts
    /// 3. **AR structure**: coefficients match order, reasonable bounds
    /// 4. **Numerical stability**: finite values, reasonable ranges
    ///
    /// # Validation Rules
    ///
    /// ## Statistical Parameters
    /// - `std_dev > 0` (strictly positive, no zero variance)
    /// - `std_dev < 1e6` (reasonable upper bound, likely input error if exceeded)
    /// - `mean` is finite (no NaN or Inf)
    /// - `std_dev` is finite
    ///
    /// ## PAR Model Completeness
    /// - `num_seasons > 0`
    /// - All seasons 0..num_seasons-1 have seasonal_params
    /// - All seasons 0..num_seasons-1 have seasonal_ar_params
    ///
    /// ## AR Structure
    /// - `ar_order > 0` (at least AR(1))
    /// - `ar_coefficients.len() == ar_order`
    /// - AR coefficients are finite
    /// - `|φ| < 10` (warning threshold for unusually large coefficients)
    ///
    /// ## Season ID Range
    /// - For PAR: all season_ids in [0, num_seasons)
    /// - For independent: season_ids are non-negative
    ///
    /// # Errors
    ///
    /// Returns descriptive error string with entity_id, season_id, parameter name, and constraint.
    ///
    /// # Performance
    ///
    /// O(num_seasons) - single pass through seasonal parameters and AR params.
    /// Typical: 12 seasons × 2 checks = 24 operations → <1μs
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let spec = UnifiedNoiseSpec { /* ... */ };
    /// spec.validate()?;
    /// // Returns Ok(()) if valid, or Err with detailed message
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        // PERFORMANCE: Use Vec to collect errors for batch reporting
        let mut errors = Vec::new();

        // Validate seasonal parameters (statistical validity)
        for (season_id, params) in &self.seasonal_params {
            // Check std_dev strictly positive
            if params.std_dev <= 0.0 {
                errors.push(format!(
                    "{:?} entity {} season {}: std_dev must be > 0 (got {}). \
                     Zero or negative standard deviation causes division by zero.",
                    self.uncertainty_type, self.entity_id, season_id, params.std_dev
                ));
            }

            // Check std_dev reasonable upper bound (likely input error if too large)
            if params.std_dev >= 1e6 {
                errors.push(format!(
                    "{:?} entity {} season {}: std_dev = {} is extremely large (> 1e6). \
                     This likely indicates an input error (wrong units or typo).",
                    self.uncertainty_type, self.entity_id, season_id, params.std_dev
                ));
            }

            // Check mean is finite
            if !params.mean.is_finite() {
                errors.push(format!(
                    "{:?} entity {} season {}: mean = {} is not finite (NaN or Inf). \
                     All statistical parameters must be finite numbers.",
                    self.uncertainty_type, self.entity_id, season_id, params.mean
                ));
            }

            // Check std_dev is finite
            if !params.std_dev.is_finite() {
                errors.push(format!(
                    "{:?} entity {} season {}: std_dev = {} is not finite (NaN or Inf). \
                     All statistical parameters must be finite numbers.",
                    self.uncertainty_type, self.entity_id, season_id, params.std_dev
                ));
            }

            // Validate marginal distribution if present
            if let Some(ref dist) = params.marginal_override {
                if let Err(e) = validate_marginal_distribution(
                    dist,
                    self.entity_id,
                    *season_id,
                ) {
                    errors.push(e);
                }
            }
        }

        // Validate PAR-specific constraints
        if let TemporalModelSpec::PeriodicAutoregressive {
            num_seasons,
            seasonal_ar_params,
        } = &self.temporal_model
        {
            // Check num_seasons > 0
            if *num_seasons == 0 {
                errors.push(format!(
                    "{:?} entity {}: PAR model has num_seasons = 0 (must be > 0). \
                     PAR models require at least one season.",
                    self.uncertainty_type, self.entity_id
                ));
            }

            // Check all seasons 0..num_seasons-1 are present in seasonal_params
            for season in 0..*num_seasons {
                if !self.seasonal_params.contains_key(&season) {
                    errors.push(format!(
                        "{:?} entity {}: PAR model missing seasonal_params for season {} \
                         (expected all seasons 0..{}). PAR models require complete seasonal coverage.",
                        self.uncertainty_type, self.entity_id, season, num_seasons - 1
                    ));
                }
            }

            // Check all seasons 0..num_seasons-1 are present in seasonal_ar_params
            for season in 0..*num_seasons {
                if !seasonal_ar_params.contains_key(&season) {
                    errors.push(format!(
                        "{:?} entity {}: PAR model missing seasonal_ar_params for season {} \
                         (expected all seasons 0..{}). Each season needs AR parameters.",
                        self.uncertainty_type, self.entity_id, season, num_seasons - 1
                    ));
                }
            }

            // Validate each season's AR structure
            for (season_id, ar_params) in seasonal_ar_params {
                if ar_params.ar_order == 0 {
                    errors.push(format!(
                        "{:?} entity {} season {}: ar_order = 0 (must be > 0). \
                         PAR models require at least AR(1) for temporal correlation.",
                        self.uncertainty_type, self.entity_id, season_id
                    ));
                }

                if ar_params.ar_coefficients.len() != ar_params.ar_order {
                    errors.push(format!(
                        "{:?} entity {} season {}: ar_order = {} but {} coefficient(s) provided. \
                         AR({}) model requires exactly {} coefficient(s) [φ₁, φ₂, ..., φ_p].",
                        self.uncertainty_type,
                        self.entity_id,
                        season_id,
                        ar_params.ar_order,
                        ar_params.ar_coefficients.len(),
                        ar_params.ar_order,
                        ar_params.ar_order
                    ));
                }

                // Validate AR coefficients are finite and reasonable
                for (coef_idx, &coef) in
                    ar_params.ar_coefficients.iter().enumerate()
                {
                    if !coef.is_finite() {
                        errors.push(format!(
                            "{:?} entity {} season {}: AR coefficient φ_{} = {} is not finite. \
                             All AR coefficients must be finite numbers.",
                            self.uncertainty_type,
                            self.entity_id,
                            season_id,
                            coef_idx + 1,
                            coef
                        ));
                    }

                    // Check for unusually large coefficients (warning threshold)
                    if coef.abs() >= 10.0 {
                        errors.push(format!(
                            "{:?} entity {} season {}: AR coefficient φ_{} = {} is unusually large (|φ| >= 10). \
                             Typical AR coefficients are in [-1, 1]. Check for input errors (wrong units or typo).",
                            self.uncertainty_type,
                            self.entity_id,
                            season_id,
                            coef_idx + 1,
                            coef
                        ));
                    }
                }

                // Check season_id is within valid range
                if *season_id >= *num_seasons {
                    errors.push(format!(
                        "{:?} entity {}: season_id {} in seasonal_ar_params exceeds num_seasons {} \
                         (expected 0..{}). Season IDs must be sequential starting from 0.",
                        self.uncertainty_type,
                        self.entity_id,
                        season_id,
                        num_seasons,
                        num_seasons - 1
                    ));
                }
            }
        }

        // Check for out-of-range season IDs in seasonal_params
        if let Some(max_season) = self.num_seasons() {
            for season_id in self.seasonal_params.keys() {
                if *season_id >= max_season {
                    errors.push(format!(
                        "{:?} entity {}: season_id {} in seasonal_params exceeds num_seasons {} \
                         (expected 0..{}). Season IDs must match PAR model definition.",
                        self.uncertainty_type,
                        self.entity_id,
                        season_id,
                        max_season,
                        max_season - 1
                    ));
                }
            }
        }

        // Return aggregated errors or Ok
        if errors.is_empty() {
            Ok(())
        } else {
            // Join all errors with newlines for batch reporting
            Err(errors.join("\n"))
        }
    }

    /// Validate against graph and system structure
    ///
    /// Cross-validates noise specifications with:
    /// - **System**: Entity IDs must exist (hydro_id for inflow, bus_id for load)
    /// - **Graph**: Season IDs must match graph nodes' season_ids
    /// - **Consistency**: PAR num_seasons should match graph structure
    ///
    /// # Arguments
    ///
    /// - `graph`: Graph structure with nodes containing season_ids
    /// - `system`: System configuration with entity counts
    ///
    /// # Validation Rules
    ///
    /// ## Entity ID Validity
    /// - Inflow specs: `entity_id < system.hydros.len()`
    /// - Load specs: `entity_id < system.buses.len()`
    ///
    /// ## Season ID Consistency
    /// - All season_ids in specs must appear in graph nodes
    /// - PAR num_seasons should match unique season count in graph
    ///
    /// # Errors
    ///
    /// Returns error with context if:
    /// - Entity ID doesn't exist in system
    /// - Season ID not found in graph
    /// - PAR num_seasons mismatch with graph structure
    ///
    /// # Performance
    ///
    /// O(n + m) where n = graph nodes, m = seasonal_params entries.
    /// Builds HashSet of graph season_ids for O(1) lookups.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let spec = UnifiedNoiseSpec { /* ... */ };
    /// spec.validate_against_graph(&graph, &system)?;
    /// ```
    pub fn validate_against_graph(
        &self,
        graph: &GraphInput,
        system: &SystemInput,
    ) -> Result<(), String> {
        let mut errors = Vec::new();

        // Validate entity ID exists in system
        match self.uncertainty_type {
            UncertaintyType::Inflow => {
                if self.entity_id >= system.hydros.len() {
                    errors.push(format!(
                        "Inflow entity {} does not exist in system (system has {} hydro(s), IDs 0..{}). \
                         Check that entity_id matches a valid hydro ID.",
                        self.entity_id,
                        system.hydros.len(),
                        system.hydros.len().saturating_sub(1)
                    ));
                }
            }
            UncertaintyType::Load => {
                if self.entity_id >= system.buses.len() {
                    errors.push(format!(
                        "Load entity {} does not exist in system (system has {} bus(es), IDs 0..{}). \
                         Check that entity_id matches a valid bus ID.",
                        self.entity_id,
                        system.buses.len(),
                        system.buses.len().saturating_sub(1)
                    ));
                }
            }
        }

        // PERFORMANCE: Build HashSet of graph season_ids for O(1) lookups
        use std::collections::HashSet;
        let graph_season_ids: HashSet<usize> =
            graph.nodes.iter().map(|n| n.season_id).collect();

        // Validate all season_ids in specs appear in graph
        for season_id in self.seasonal_params.keys() {
            if !graph_season_ids.contains(season_id) {
                errors.push(format!(
                    "{:?} entity {}: season_id {} not found in graph structure. \
                     Graph contains season_ids: [{}]. Check that seasonal_params match graph definition.",
                    self.uncertainty_type,
                    self.entity_id,
                    season_id,
                    {
                        let mut sorted: Vec<_> = graph_season_ids.iter().copied().collect();
                        sorted.sort_unstable();
                        sorted
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ));
            }
        }

        // For PAR models, check num_seasons matches graph structure
        if let Some(num_seasons) = self.num_seasons() {
            let graph_unique_seasons = graph_season_ids.len();
            if num_seasons != graph_unique_seasons {
                errors.push(format!(
                    "{:?} entity {}: PAR num_seasons = {} but graph has {} unique season_id(s). \
                     PAR models should span all seasons in the planning horizon. \
                     Graph season_ids: [{}]",
                    self.uncertainty_type,
                    self.entity_id,
                    num_seasons,
                    graph_unique_seasons,
                    {
                        let mut sorted: Vec<_> = graph_season_ids.iter().copied().collect();
                        sorted.sort_unstable();
                        sorted
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }

    /// Get seasonal parameters for a specific season (O(1) lookup)
    ///
    /// # Arguments
    ///
    /// - `season_id`: Season index (0..num_seasons-1)
    ///
    /// # Returns
    ///
    /// - `Some(&SeasonalNoiseParams)`: Parameters for this season
    /// - `None`: Season not defined (sparse independent model)
    ///
    /// # Performance
    ///
    /// O(1) average case via HashMap lookup
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(params) = spec.get_seasonal_params(5) {
    ///     println!("Season 5: mean={}, std_dev={}", params.mean, params.std_dev);
    /// }
    /// ```
    #[inline]
    pub fn get_seasonal_params(
        &self,
        season_id: usize,
    ) -> Option<&SeasonalNoiseParams> {
        self.seasonal_params.get(&season_id)
    }

    /// Get PAR parameters for a specific season (O(1) lookup)
    ///
    /// # Arguments
    ///
    /// - `season_id`: Season index (0..num_seasons-1)
    ///
    /// # Returns
    ///
    /// - `Some(&SeasonalPARParams)`: PAR parameters for this season
    /// - `None`: Not a PAR model or season not defined
    ///
    /// # Performance
    ///
    /// O(1) average case via HashMap lookup (after enum match)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(ar_params) = spec.get_par_params(5) {
    ///     println!("Season 5: AR({}), coeffs={:?}",
    ///         ar_params.ar_order, ar_params.ar_coefficients);
    /// }
    /// ```
    #[inline]
    pub fn get_par_params(
        &self,
        season_id: usize,
    ) -> Option<&SeasonalPARParams> {
        match &self.temporal_model {
            TemporalModelSpec::PeriodicAutoregressive {
                seasonal_ar_params,
                ..
            } => seasonal_ar_params.get(&season_id),
            TemporalModelSpec::Independent => None,
        }
    }

    /// Check if this is a PAR model
    ///
    /// # Returns
    ///
    /// - `true`: PAR model with temporal correlation
    /// - `false`: Independent model (no correlation)
    ///
    /// # Performance
    ///
    /// O(1) - simple enum discriminant check
    #[inline]
    pub fn is_par_model(&self) -> bool {
        matches!(
            self.temporal_model,
            TemporalModelSpec::PeriodicAutoregressive { .. }
        )
    }

    /// Get number of seasons for PAR models
    ///
    /// # Returns
    ///
    /// - `Some(num_seasons)`: Number of seasons for PAR model
    /// - `None`: Independent model (no fixed num_seasons)
    ///
    /// # Performance
    ///
    /// O(1) - enum match and field access
    #[inline]
    pub fn num_seasons(&self) -> Option<usize> {
        match &self.temporal_model {
            TemporalModelSpec::PeriodicAutoregressive {
                num_seasons, ..
            } => Some(*num_seasons),
            TemporalModelSpec::Independent => None,
        }
    }

    /// Convert new format (`UncertaintySpecification`) to internal representation
    ///
    /// This is the inverse of `to_uncertainty_specification()` and enables validation
    /// of format equivalence by converting both formats to `UnifiedNoiseSpec`.
    ///
    /// # Algorithm
    ///
    /// 1. For each `UncertaintySpecification`:
    ///    - Independent model: Build `seasonal_params` from `seasonal_distributions`
    ///    - PAR model: Build `seasonal_params` and `seasonal_ar_params` from arrays
    /// 2. Create `UnifiedNoiseSpec` with appropriate `TemporalModelSpec`
    /// 3. Store marginal distribution if present
    ///
    /// # Performance
    ///
    /// - Time: O(n·s) where n = specs, s = seasons per spec
    /// - Space: O(n·s) for storing result
    /// - Typical: <1ms for 10 entities × 12 seasons
    ///
    /// # Returns
    ///
    /// - `Ok(specs)`: Converted specifications
    /// - `Err(msg)`: Validation error (e.g., missing data, inconsistent arrays)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let new_specs = vec![/* uncertainty specifications */];
    /// let unified = UnifiedNoiseSpec::from_uncertainty_specifications(&new_specs)?;
    /// // Use unified for validation or processing
    /// ```
    pub fn from_uncertainty_specifications(
        specs: &[UncertaintySpecification],
    ) -> Result<Vec<UnifiedNoiseSpec>, String> {
        if specs.is_empty() {
            return Err("No uncertainty specifications defined (empty input)"
                .to_string());
        }

        let mut unified_specs = Vec::with_capacity(specs.len());

        for spec in specs {
            // Independent model: build from seasonal_distributions
            let seasonal_dists = spec.seasonal_distributions.as_ref()
                .ok_or_else(|| {
                    format!(
                        "{:?} entity {}: Independent model requires seasonal_distributions",
                        spec.uncertainty_type, spec.entity_id
                    )
                })?;

            let mut seasonal_params =
                HashMap::with_capacity(seasonal_dists.len());

            for dist in seasonal_dists {
                // Use helper method to convert SeasonalDistribution to SeasonalNoiseParams
                // This handles both Normal and LogNormal3 distributions correctly
                seasonal_params
                    .insert(dist.season_id, dist.to_seasonal_params());
            }
            match &spec.temporal_model {
                TemporalModelInput::Independent => {
                    unified_specs.push(UnifiedNoiseSpec {
                        uncertainty_type: spec.uncertainty_type.clone(),
                        entity_id: spec.entity_id,
                        temporal_model: TemporalModelSpec::Independent,
                        seasonal_params,
                    });
                }

                TemporalModelInput::PeriodicAr {
                    num_seasons,
                    ar_orders,
                    ar_coefficients,
                    seasonal_means,
                    seasonal_stds,
                } => {
                    // PAR model: build from arrays

                    // Validate array lengths
                    if ar_orders.len() != *num_seasons
                        || ar_coefficients.len() != *num_seasons
                        || seasonal_means.len() != *num_seasons
                        || seasonal_stds.len() != *num_seasons
                    {
                        return Err(format!(
                            "{:?} entity {}: PAR model arrays have inconsistent lengths \
                             (num_seasons={}, ar_orders={}, ar_coefficients={}, \
                             seasonal_means={}, seasonal_stds={})",
                            spec.uncertainty_type,
                            spec.entity_id,
                            num_seasons,
                            ar_orders.len(),
                            ar_coefficients.len(),
                            seasonal_means.len(),
                            seasonal_stds.len()
                        ));
                    }

                    // Build seasonal_params and seasonal_ar_params
                    let mut seasonal_params =
                        HashMap::with_capacity(*num_seasons);
                    let mut seasonal_ar_params =
                        HashMap::with_capacity(*num_seasons);

                    for season in 0..*num_seasons {
                        seasonal_params.insert(
                            season,
                            SeasonalNoiseParams {
                                mean: seasonal_means[season],
                                std_dev: seasonal_stds[season],
                                marginal_override: Some(
                                    seasonal_dists[season].distribution.clone(),
                                ),
                            },
                        );

                        // Validate AR coefficients array length matches order
                        if ar_coefficients[season].len() != ar_orders[season] {
                            return Err(format!(
                                "{:?} entity {}: PAR model season {} has ar_order={} \
                                 but ar_coefficients has length {}",
                                spec.uncertainty_type,
                                spec.entity_id,
                                season,
                                ar_orders[season],
                                ar_coefficients[season].len()
                            ));
                        }

                        seasonal_ar_params.insert(
                            season,
                            SeasonalPARParams {
                                ar_order: ar_orders[season],
                                ar_coefficients: ar_coefficients[season]
                                    .clone(),
                            },
                        );
                    }

                    unified_specs.push(UnifiedNoiseSpec {
                        uncertainty_type: spec.uncertainty_type.clone(),
                        entity_id: spec.entity_id,
                        temporal_model:
                            TemporalModelSpec::PeriodicAutoregressive {
                                num_seasons: *num_seasons,
                                seasonal_ar_params,
                            },
                        seasonal_params,
                    });
                }
            }
        }

        Ok(unified_specs)
    }
}

/// Helper function to validate marginal distribution parameters
///
/// Checks finite values and reasonable ranges for distribution parameters.
fn validate_marginal_distribution(
    dist: &MarginalDistribution,
    entity_id: usize,
    season_id: usize,
) -> Result<(), String> {
    match dist {
        MarginalDistribution::Normal { mean, std_dev } => {
            if !mean.is_finite() {
                return Err(format!(
                    "Entity {} season {}: Normal mean = {} is not finite",
                    entity_id, season_id, mean
                ));
            }
            if !std_dev.is_finite() {
                return Err(format!(
                    "Entity {} season {}: Normal std_dev = {} is not finite",
                    entity_id, season_id, std_dev
                ));
            }
            if *std_dev <= 0.0 {
                return Err(format!(
                    "Entity {} season {}: Normal std_dev = {} must be > 0",
                    entity_id, season_id, std_dev
                ));
            }
        }
        MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
            if !gamma.is_finite() {
                return Err(format!(
                    "Entity {} season {}: LogNormal3 gamma = {} is not finite",
                    entity_id, season_id, gamma
                ));
            }
            if !mu.is_finite() {
                return Err(format!(
                    "Entity {} season {}: LogNormal3 mu = {} is not finite",
                    entity_id, season_id, mu
                ));
            }
            if !sigma.is_finite() {
                return Err(format!(
                    "Entity {} season {}: LogNormal3 sigma = {} is not finite",
                    entity_id, season_id, sigma
                ));
            }
            if *sigma <= 0.0 {
                return Err(format!(
                    "Entity {} season {}: LogNormal3 sigma = {} must be > 0",
                    entity_id, season_id, sigma
                ));
            }
        }
    }
    Ok(())
}

/// Validate a collection of noise specs for duplicates and coverage
///
/// Performs collection-level validation across all noise specs:
/// 1. **Duplicate detection**: No two specs can have same (entity_id, uncertainty_type)
/// 2. **Entity coverage**: All entities should have noise specs
///    - All hydros (0..system.hydros.len()) should have inflow specs
///    - All buses (0..system.buses.len()) should have load specs
/// 3. **Cross-validation**: Each spec validates against graph and system
///
/// # Arguments
///
/// - `specs`: Collection of unified noise specs to validate
/// - `graph`: Graph structure with temporal nodes
/// - `system`: System configuration with entity counts
///
/// # Validation Rules
///
/// ## Duplicate Detection
/// - Each (entity_id, uncertainty_type) tuple must be unique
/// - Duplicates indicate conflicting uncertainty models for same entity
///
/// ## Entity Coverage
/// - **Inflows**: Every hydro ID in [0, system.hydros.len()) needs exactly one inflow spec
/// - **Loads**: Every bus ID in [0, system.buses.len()) needs exactly one load spec
/// - Missing entities get default (typically zero variance or deterministic)
///
/// ## Cross-Validation
/// - Entity IDs exist in system
/// - Season IDs match graph structure
/// - PAR num_seasons consistent with graph
///
/// # Errors
///
/// Returns aggregated errors from all validation checks. Collects all errors
/// rather than failing fast to provide complete validation feedback.
///
/// # Performance
///
/// O(n) where n = number of specs. Uses HashSet for O(1) duplicate detection
/// and entity coverage checking.
///
/// Typical: 20 entities × 2 uncertainty types = 40 specs → <1ms validation
///
pub fn validate_noise_specs(
    specs: &[UnifiedNoiseSpec],
    graph: &GraphInput,
    system: &SystemInput,
) -> Result<(), String> {
    let mut errors = Vec::new();

    // PERFORMANCE: Use HashSet for O(1) duplicate detection
    let mut seen_keys: HashSet<(usize, UncertaintyType)> = HashSet::new();
    let mut inflow_entities: HashSet<usize> = HashSet::new();
    let mut load_entities: HashSet<usize> = HashSet::new();

    // Validate each spec and check for duplicates
    for spec in specs {
        // Check for duplicates (clone since UncertaintyType doesn't implement Copy)
        let key = (spec.entity_id, spec.uncertainty_type.clone());
        if !seen_keys.insert(key) {
            errors.push(format!(
                "Duplicate noise specification for {:?} entity {}. \
                 Each entity can have only one uncertainty model.",
                spec.uncertainty_type, spec.entity_id
            ));
        }

        // Track entity coverage
        match spec.uncertainty_type {
            UncertaintyType::Inflow => {
                inflow_entities.insert(spec.entity_id);
            }
            UncertaintyType::Load => {
                load_entities.insert(spec.entity_id);
            }
        }

        // Run individual validation
        if let Err(e) = spec.validate() {
            errors.push(e);
        }

        // Run cross-validation with graph and system
        if let Err(e) = spec.validate_against_graph(graph, system) {
            errors.push(e);
        }
    }

    // Check entity coverage for inflows
    let num_hydros = system.hydros.len();
    for hydro_id in 0..num_hydros {
        if !inflow_entities.contains(&hydro_id) {
            errors.push(format!(
                "Missing inflow specification for hydro {}. \
                 All hydros (0..{}) require inflow uncertainty models.",
                hydro_id, num_hydros
            ));
        }
    }

    // Check entity coverage for loads
    let num_buses = system.buses.len();
    for bus_id in 0..num_buses {
        if !load_entities.contains(&bus_id) {
            errors.push(format!(
                "Missing load specification for bus {}. \
                 All buses (0..{}) require load uncertainty models.",
                bus_id, num_buses
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_noise_spec_creation() {
        let mut seasonal_params = HashMap::new();
        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 0,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
        };

        assert!(!spec.is_par_model());
        assert_eq!(spec.num_seasons(), None);
        assert!(spec.get_seasonal_params(0).is_some());
        assert_eq!(spec.get_seasonal_params(0).unwrap().mean, 100.0);
    }

    #[test]
    fn test_par_noise_spec_creation() {
        let mut seasonal_params = HashMap::new();
        let mut par_params = HashMap::new();

        for season in 0..12 {
            seasonal_params.insert(
                season,
                SeasonalNoiseParams {
                    mean: 100.0 + (season as f64) * 10.0,
                    std_dev: 20.0,
                    marginal_override: None,
                },
            );
            par_params.insert(
                season,
                SeasonalPARParams {
                    ar_order: 1,
                    ar_coefficients: vec![0.7],
                },
            );
        }

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 12,
                seasonal_ar_params: par_params,
            },
            seasonal_params,
        };

        assert!(spec.is_par_model());
        assert_eq!(spec.num_seasons(), Some(12));
        assert!(spec.get_seasonal_params(5).is_some());
        assert_eq!(spec.get_seasonal_params(5).unwrap().mean, 150.0);
        assert!(spec.get_par_params(5).is_some());
        assert_eq!(spec.get_par_params(5).unwrap().ar_order, 1);
    }

    #[test]
    fn test_validation_rejects_negative_std_dev() {
        let mut seasonal_params = HashMap::new();
        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: -1.0, // Invalid
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 0,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
        };

        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_par_missing_seasons() {
        let mut seasonal_params = HashMap::new();
        let mut par_params = HashMap::new();

        // Only define seasons 0-10, missing season 11
        for season in 0..11 {
            seasonal_params.insert(
                season,
                SeasonalNoiseParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    marginal_override: None,
                },
            );
            par_params.insert(
                season,
                SeasonalPARParams {
                    ar_order: 1,
                    ar_coefficients: vec![0.7],
                },
            );
        }

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 12,
                seasonal_ar_params: par_params,
            },
            seasonal_params,
        };

        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_par_mismatched_coefficients() {
        let mut seasonal_params = HashMap::new();
        let mut par_params = HashMap::new();

        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );
        par_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 2,
                ar_coefficients: vec![0.7], // Should have 2 coefficients
            },
        );

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 1,
                seasonal_ar_params: par_params,
            },
            seasonal_params,
        };

        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_validation_accepts_valid_independent() {
        let mut seasonal_params = HashMap::new();
        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 0,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
        };

        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_validation_accepts_valid_par() {
        let mut seasonal_params = HashMap::new();
        let mut par_params = HashMap::new();

        for season in 0..12 {
            seasonal_params.insert(
                season,
                SeasonalNoiseParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    marginal_override: None,
                },
            );
            par_params.insert(
                season,
                SeasonalPARParams {
                    ar_order: 1,
                    ar_coefficients: vec![0.7],
                },
            );
        }

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 12,
                seasonal_ar_params: par_params,
            },
            seasonal_params,
        };

        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_o1_lookup_performance() {
        // Create a PAR spec with 12 seasons
        let mut seasonal_params = HashMap::with_capacity(12);
        let mut par_params = HashMap::with_capacity(12);

        for season in 0..12 {
            seasonal_params.insert(
                season,
                SeasonalNoiseParams {
                    mean: 100.0 + (season as f64) * 10.0,
                    std_dev: 20.0,
                    marginal_override: None,
                },
            );
            par_params.insert(
                season,
                SeasonalPARParams {
                    ar_order: 1,
                    ar_coefficients: vec![0.7],
                },
            );
        }

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 12,
                seasonal_ar_params: par_params,
            },
            seasonal_params,
        };

        // Verify O(1) lookups work correctly
        for season in 0..12 {
            let params = spec.get_seasonal_params(season).unwrap();
            assert_eq!(params.mean, 100.0 + (season as f64) * 10.0);

            let ar_params = spec.get_par_params(season).unwrap();
            assert_eq!(ar_params.ar_order, 1);
        }
    }

    #[test]
    fn test_independent_model_sparse_seasons() {
        // Independent models may only define seasons where entity appears
        let mut seasonal_params = HashMap::new();
        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );
        seasonal_params.insert(
            5,
            SeasonalNoiseParams {
                mean: 150.0,
                std_dev: 30.0,
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 0,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
        };

        assert!(spec.validate().is_ok());
        assert!(spec.get_seasonal_params(0).is_some());
        assert!(spec.get_seasonal_params(1).is_none()); // Sparse - not defined
        assert!(spec.get_seasonal_params(5).is_some());
    }

    #[test]
    fn test_validation_rejects_non_finite_mean() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: f64::NAN,
                std_dev: 10.0,
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("not finite"));
        assert!(err_msg.contains("mean"));
    }

    #[test]
    fn test_validation_rejects_non_finite_std_dev() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: f64::INFINITY,
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("not finite"));
        assert!(err_msg.contains("std_dev"));
    }

    #[test]
    fn test_validation_rejects_very_large_std_dev() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 2e6,
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("extremely large"));
        assert!(err_msg.contains("1e6"));
    }

    #[test]
    fn test_validation_rejects_non_finite_ar_coefficient() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        let mut ar_params = HashMap::new();
        ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 2,
                ar_coefficients: vec![0.5, f64::NAN],
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 1,
                seasonal_ar_params: ar_params,
            },
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("AR coefficient"));
        assert!(err_msg.contains("not finite"));
    }

    #[test]
    fn test_validation_warns_large_ar_coefficient() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        let mut ar_params = HashMap::new();
        ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 1,
                ar_coefficients: vec![15.0], // Unusually large coefficient
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 1,
                seasonal_ar_params: ar_params,
            },
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("unusually large"));
        assert!(err_msg.contains("AR coefficient"));
        assert!(err_msg.contains(">= 10"));
    }

    #[test]
    fn test_validation_rejects_out_of_range_season_id() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );
        params.insert(
            5, // Out of range (num_seasons = 3)
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        let mut ar_params = HashMap::new();
        ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 1,
                ar_coefficients: vec![0.5],
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 3,
                seasonal_ar_params: ar_params,
            },
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("season_id"));
        assert!(err_msg.contains("exceeds num_seasons"));
    }

    #[test]
    fn test_validation_aggregates_multiple_errors() {
        // Create spec with multiple validation errors
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: f64::NAN, // Error 1: non-finite mean
                std_dev: -1.0,  // Error 2: negative std_dev
                marginal_override: None,
            },
        );

        let mut ar_params = HashMap::new();
        ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 0, // Error 3: zero ar_order
                ar_coefficients: vec![],
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 1,
                seasonal_ar_params: ar_params,
            },
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();

        // Check that error message contains multiple errors
        assert!(err_msg.contains("std_dev must be > 0"));
        assert!(err_msg.contains("not finite"));
        assert!(err_msg.contains("ar_order = 0"));

        // Count newlines to verify multiple errors reported
        let error_count = err_msg.matches('\n').count() + 1;
        assert!(
            error_count >= 3,
            "Expected at least 3 errors, got {}",
            error_count
        );
    }

    #[test]
    fn test_validation_accepts_reasonable_ar_coefficients() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        let mut ar_params = HashMap::new();
        ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 2,
                ar_coefficients: vec![0.7, -0.3], // Reasonable coefficients
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 1,
                seasonal_ar_params: ar_params,
            },
            seasonal_params: params,
        };

        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_validation_rejects_non_finite_lognormal_params() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: Some(MarginalDistribution::LogNormal3 {
                    gamma: f64::INFINITY,
                    mu: 0.0,
                    sigma: 0.5,
                }),
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("LogNormal3"));
        assert!(err_msg.contains("gamma"));
        assert!(err_msg.contains("not finite"));
    }

    #[test]
    fn test_validation_rejects_negative_lognormal_sigma() {
        let mut params = HashMap::new();
        params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: Some(MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 0.0,
                    sigma: -0.5, // Invalid: negative
                }),
            },
        );

        let spec = UnifiedNoiseSpec {
            entity_id: 0,
            uncertainty_type: UncertaintyType::Inflow,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params: params,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("LogNormal3"));
        assert!(err_msg.contains("sigma"));
        assert!(err_msg.contains("must be > 0"));
    }
}
