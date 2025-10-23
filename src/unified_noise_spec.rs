//! Unified internal representation for noise specifications
//!
//! This module provides `UnifiedNoiseSpec`, an internal data structure that separates
//! entity-level uncertainty specifications from season-level parameters. This design
//! addresses the architectural inconsistency where PAR models (spanning all seasons)
//! were incorrectly nested within single-season `NoiseModel` entries.
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
//! It is not directly exposed in JSON input files. Instead, it is constructed from:
//! - Current format: `NoiseModel` structs (via converter in TICKET-02)
//! - Future format: `UncertaintySpecification` structs (TICKET-09)
//!
//! # Example (Independent noise)
//!
//! ```rust
//! use powers_rs::unified_noise_spec::{
//!     UnifiedNoiseSpec, TemporalModelSpec, SeasonalNoiseParams
//! };
//! use powers_rs::input::{MarginalDistribution, UncertaintyType};
//! use std::collections::HashMap;
//!
//! // Load uncertainty with independent noise in 2 seasons
//! let mut seasonal_params = HashMap::new();
//! seasonal_params.insert(0, SeasonalNoiseParams {
//!     mean: 100.0,
//!     std_dev: 20.0,
//!     marginal_override: None,
//! });
//! seasonal_params.insert(1, SeasonalNoiseParams {
//!     mean: 120.0,
//!     std_dev: 25.0,
//!     marginal_override: None,
//! });
//!
//! let spec = UnifiedNoiseSpec {
//!     uncertainty_type: UncertaintyType::Load,
//!     entity_id: 0,
//!     temporal_model: TemporalModelSpec::Independent,
//!     seasonal_params,
//!     marginal_distribution: Some(MarginalDistribution::Normal {
//!         mean: 0.0,
//!         std_dev: 1.0,
//!     }),
//! };
//!
//! // O(1) lookup for season 0
//! let params = spec.seasonal_params.get(&0).unwrap();
//! assert_eq!(params.mean, 100.0);
//! ```
//!
//! # Example (PAR noise)
//!
//! ```rust
//! use powers_rs::unified_noise_spec::{
//!     UnifiedNoiseSpec, TemporalModelSpec, SeasonalNoiseParams, SeasonalPARParams
//! };
//! use powers_rs::input::{MarginalDistribution, UncertaintyType};
//! use std::collections::HashMap;
//!
//! // Inflow uncertainty with PAR(1) model over 12 seasons
//! let mut seasonal_params = HashMap::new();
//! for season in 0..12 {
//!     seasonal_params.insert(season, SeasonalNoiseParams {
//!         mean: 100.0 + (season as f64) * 10.0,  // Seasonal variation
//!         std_dev: 20.0,
//!         marginal_override: None,
//!     });
//! }
//!
//! let mut par_params = HashMap::new();
//! for season in 0..12 {
//!     par_params.insert(season, SeasonalPARParams {
//!         ar_order: 1,
//!         ar_coefficients: vec![0.7],
//!     });
//! }
//!
//! let spec = UnifiedNoiseSpec {
//!     uncertainty_type: UncertaintyType::Inflow,
//!     entity_id: 0,
//!     temporal_model: TemporalModelSpec::PeriodicAutoregressive {
//!         num_seasons: 12,
//!         seasonal_ar_params: par_params,
//!     },
//!     seasonal_params,
//!     marginal_distribution: Some(MarginalDistribution::LogNormal3 {
//!         gamma: 1.0,
//!         mu: 0.0,
//!         sigma: 0.6,
//!     }),
//! };
//!
//! // O(1) lookup for season 5
//! let params = spec.seasonal_params.get(&5).unwrap();
//! assert_eq!(params.mean, 150.0);
//! ```
//!
//! # Performance
//!
//! - **HashMap lookups**: O(1) average case vs O(n) linear search
//! - **Memory overhead**: ~1KB per entity (12 seasons × 80 bytes/entry)
//! - **Pre-allocation**: Use `HashMap::with_capacity(num_seasons)` to avoid rehashing
//!
//! # References
//!
//! - Ticket: TICKET-01 (Foundation - Internal Representation)
//! - Architecture: `docs/implementation-plans/PAR_INPUT_REFACTOR.md`
//! - Current format: `src/input.rs::NoiseModel`
//! - PAR implementation: `src/par_generator.rs::PeriodicARGenerator`

use crate::input::{MarginalDistribution, UncertaintyType};
use std::collections::{HashMap, HashSet};

/// Unified internal representation for noise specifications
///
/// Separates entity-level concerns (which entity, what temporal model) from
/// season-level parameters (mean, std_dev per season). This design enables:
/// - O(1) lookup of seasonal parameters via HashMap
/// - Clear separation of temporal dynamics from seasonal statistics
/// - Support for both independent and PAR models
///
/// # Fields
///
/// - `uncertainty_type`: Whether this is inflow or load uncertainty
/// - `entity_id`: Which entity (hydro_id or bus_id)
/// - `temporal_model`: How the entity behaves across time (independent vs PAR)
/// - `seasonal_params`: Per-season parameters (mean, std_dev) with O(1) access
/// - `marginal_distribution`: Optional entity-level distribution (for PAR residuals)
///
/// # Design Rationale
///
/// The previous `NoiseModel` structure embedded PAR models (12 seasons) within
/// entries marked with a single `season_id`, causing:
/// - Semantic confusion (PAR spans all seasons, not one)
/// - O(n) lookups (iterate through all noise models to find match)
/// - Data duplication risk (same PAR definition could be repeated)
///
/// `UnifiedNoiseSpec` resolves this by:
/// - PAR model defined once at entity level
/// - Seasonal parameters stored in HashMap for O(1) access
/// - Clear separation of cross-stage (PAR) and per-stage (seasonal) concerns
///
/// # Usage Pattern
///
/// ```rust,ignore
/// // Construction (from converter)
/// let spec = UnifiedNoiseSpec::from_noise_model(&noise_model)?;
///
/// // Lookup seasonal parameters (O(1))
/// if let Some(params) = spec.seasonal_params.get(&season_id) {
///     let mean = params.mean;
///     let std_dev = params.std_dev;
/// }
///
/// // Check temporal model
/// match &spec.temporal_model {
///     TemporalModelSpec::Independent => { /* no correlation */ }
///     TemporalModelSpec::PeriodicAutoregressive { num_seasons, .. } => {
///         /* PAR dynamics */
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct UnifiedNoiseSpec {
    /// Type of uncertainty (inflow or load)
    ///
    /// Determines which entities this spec applies to:
    /// - `Inflow`: Maps to hydro plant IDs
    /// - `Load`: Maps to bus IDs
    pub uncertainty_type: UncertaintyType,

    /// Entity ID (hydro_id for inflow, bus_id for load)
    ///
    /// Must match an entity in `system.json`. This is the entity to which
    /// the uncertainty applies. For example:
    /// - Inflow uncertainty: `entity_id = 0` means hydro plant 0
    /// - Load uncertainty: `entity_id = 3` means bus 3
    ///
    /// # Validation
    ///
    /// Entity IDs must be validated against the system configuration to ensure
    /// the referenced entities exist.
    pub entity_id: usize,

    /// Temporal model specification (independent or PAR)
    ///
    /// Defines how the entity's uncertainty evolves across time:
    /// - `Independent`: No temporal correlation (white noise)
    /// - `PeriodicAutoregressive`: PAR(p) with seasonal AR parameters
    ///
    /// For PAR models, this contains the AR structure (orders, coefficients)
    /// for all seasons. The seasonal parameters (mean, std_dev) are stored
    /// separately in `seasonal_params` for O(1) access.
    pub temporal_model: TemporalModelSpec,

    /// Per-season parameters (mean, std_dev) for O(1) lookup
    ///
    /// Key: season_id (0..num_seasons-1)
    /// Value: SeasonalNoiseParams with mean, std_dev, optional marginal override
    ///
    /// # Storage
    ///
    /// - **Independent models**: May contain sparse entries (only seasons where entity appears)
    /// - **PAR models**: Should contain all seasons (0..num_seasons-1) for consistency
    ///
    /// # Performance
    ///
    /// HashMap provides O(1) lookup vs O(n) iteration through Vec. Pre-allocate
    /// capacity with `HashMap::with_capacity(num_seasons)` to avoid rehashing.
    pub seasonal_params: HashMap<usize, SeasonalNoiseParams>,

    /// Optional entity-level marginal distribution
    ///
    /// Semantics depend on temporal model:
    /// - **Independent**: Marginal distribution of final series Xₜ (may be unused if per-season overrides exist)
    /// - **PAR**: Distribution of residuals aₜ after de-seasonalization
    ///
    /// Set to `None` when:
    /// - Independent model with per-season marginal overrides
    /// - No entity-level distribution is specified
    pub marginal_distribution: Option<MarginalDistribution>,
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
    ///
    /// # Performance
    ///
    /// HashMap provides O(1) lookup of AR parameters by season_id.
    /// Pre-allocate with `HashMap::with_capacity(num_seasons)`.
    PeriodicAutoregressive {
        /// Number of seasons in the cycle (e.g., 12 for monthly)
        ///
        /// Must be > 0. All seasons 0..num_seasons-1 should have entries
        /// in `seasonal_ar_params`.
        num_seasons: usize,

        /// AR parameters for each season (O(1) lookup)
        ///
        /// Key: season_id (0..num_seasons-1)
        /// Value: SeasonalPARParams with AR order and coefficients
        ///
        /// # Validation
        ///
        /// - All seasons 0..num_seasons-1 must have entries
        /// - Each season's AR coefficient count must match its AR order
        seasonal_ar_params: HashMap<usize, SeasonalPARParams>,
    },
}

/// Seasonal noise parameters (mean, std_dev, optional marginal override)
///
/// Contains the statistical parameters for one season. These are used for:
/// - Independent models: Direct sampling parameters
/// - PAR models: Seasonal mean μₘ and std_dev σₘ in PAR equation
///
/// # Fields
///
/// - `mean`: Seasonal mean (μₘ)
/// - `std_dev`: Seasonal standard deviation (σₘ, must be > 0)
/// - `marginal_override`: Optional season-specific distribution override
///
/// # Performance
///
/// This struct is small (24 bytes without marginal_override) and designed for
/// efficient storage in HashMap. No heap allocations beyond the HashMap itself.
#[derive(Debug, Clone)]
pub struct SeasonalNoiseParams {
    /// Seasonal mean μₘ
    ///
    /// For PAR models: Mean before AR dynamics
    /// For independent models: Mean of marginal distribution
    pub mean: f64,

    /// Seasonal standard deviation σₘ (must be > 0)
    ///
    /// For PAR models: Std dev before AR dynamics
    /// For independent models: Std dev of marginal distribution
    ///
    /// # Validation
    ///
    /// Must be > 0 (validated in `UnifiedNoiseSpec::validate`)
    pub std_dev: f64,

    /// Optional season-specific marginal distribution override
    ///
    /// When present, overrides the entity-level `marginal_distribution`.
    /// Used for independent models where different seasons may have different
    /// distribution types (e.g., Normal in summer, LogNormal3 in winter).
    ///
    /// Set to `None` when entity-level distribution applies.
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

        // Validate entity-level marginal distribution if present
        if let Some(ref dist) = self.marginal_distribution {
            if let Err(e) =
                validate_marginal_distribution(dist, self.entity_id, usize::MAX)
            {
                errors.push(
                    e.replace("season 18446744073709551615", "entity-level"),
                );
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
        graph: &crate::input::GraphInput,
        system: &crate::input::SystemInput,
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

    /// Convert from legacy `NoiseModel` format to `UnifiedNoiseSpec`
    ///
    /// This method transforms `Vec<NoiseModel>` entries into the new internal
    /// representation, handling both PAR and independent models. It performs
    /// grouping by (uncertainty_type, entity_id) and validates consistency.
    ///
    /// # Algorithm
    ///
    /// 1. **Group by entity**: Group all `NoiseModel` entries by (uncertainty_type, entity_id)
    /// 2. **Detect temporal model**: Check if group contains PAR or independent models
    /// 3. **Validate consistency**: Ensure no mixed temporal models for same entity
    /// 4. **Extract parameters**: Build `UnifiedNoiseSpec` from grouped data
    /// 5. **Validate result**: Ensure converted specs pass validation
    ///
    /// # PAR Models
    ///
    /// PAR models in the old format are embedded in a single `NoiseModel` entry
    /// (despite having `season_id` set). The converter extracts all seasonal data
    /// from the `TemporalModel::PeriodicAutoregressive` variant.
    ///
    /// # Independent Models
    ///
    /// Independent models may appear in multiple entries (one per season).
    /// The converter aggregates these into a single `UnifiedNoiseSpec` with
    /// seasonal parameters for each season where the entity appears.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Duplicate PAR definitions for same entity
    /// - Mixed PAR and independent models for same entity
    /// - Empty input (no uncertainties defined)
    /// - Validation of converted spec fails
    ///
    /// # Performance
    ///
    /// - O(n) where n = number of NoiseModel entries
    /// - Typical: 12 seasons × 2 types × 5 entities = 120 entries → <1ms
    /// - Not performance-critical (runs once at initialization)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use powers_rs::input::NoiseModel;
    /// use powers_rs::unified_noise_spec::UnifiedNoiseSpec;
    ///
    /// let noise_models: Vec<NoiseModel> = /* load from JSON */;
    /// let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)?;
    ///
    /// // Use unified_specs in scenario generation
    /// for spec in &unified_specs {
    ///     println!("Entity {}: {} seasons",
    ///         spec.entity_id,
    ///         spec.seasonal_params.len());
    /// }
    /// ```
    pub fn from_noise_models(
        models: &[crate::input::NoiseModel],
    ) -> Result<Vec<UnifiedNoiseSpec>, String> {
        use crate::input::TemporalModel;

        // PERFORMANCE: Pre-check for empty input (common error case)
        if models.is_empty() {
            return Err("No noise models defined (empty input)".to_string());
        }

        // Step 1: Group by (uncertainty_type, entity_id)
        // PERFORMANCE: Pre-allocate capacity based on typical entity count
        let mut groups: HashMap<
            (UncertaintyType, usize),
            Vec<&crate::input::NoiseModel>,
        > = HashMap::with_capacity(16);

        for model in models {
            groups
                .entry((model.uncertainty_type.clone(), model.entity_id))
                .or_default()
                .push(model);
        }

        // Step 2: Convert each group to UnifiedNoiseSpec
        // PERFORMANCE: Pre-allocate result vector
        let mut unified_specs = Vec::with_capacity(groups.len());

        for ((uncertainty_type, entity_id), group_models) in groups {
            // Clone uncertainty_type for error messages (cheap clone)
            let unc_type_copy = uncertainty_type.clone();

            // Step 3: Detect temporal model type
            let par_models: Vec<_> = group_models
                .iter()
                .filter(|m| {
                    matches!(
                        m.temporal_model,
                        TemporalModel::PeriodicAutoregressive { .. }
                    )
                })
                .collect();

            let independent_models: Vec<_> = group_models
                .iter()
                .filter(|m| {
                    matches!(m.temporal_model, TemporalModel::Independent)
                })
                .collect();

            // Step 4: Validate consistency
            if par_models.len() > 1 {
                // Duplicate PAR definition
                let season_ids: Vec<usize> =
                    par_models.iter().map(|m| m.season_id).collect();
                return Err(format!(
                    "Duplicate PAR definition for {:?} entity {} (found at season_ids: {:?}). \
                     PAR models span all seasons and should appear in only one entry.",
                    unc_type_copy, entity_id, season_ids
                ));
            }

            if !par_models.is_empty() && !independent_models.is_empty() {
                // Mixed temporal models
                return Err(format!(
                    "Mixed temporal models for {:?} entity {}. \
                     Found {} PAR entry(ies) and {} independent entry(ies). \
                     All entries for the same entity must use the same temporal model.",
                    unc_type_copy,
                    entity_id,
                    par_models.len(),
                    independent_models.len()
                ));
            }

            // Step 5: Convert based on temporal model type
            let unified_spec = if par_models.len() == 1 {
                // PAR model: Extract from single entry
                Self::convert_par_model(
                    uncertainty_type,
                    entity_id,
                    par_models[0],
                )?
            } else if !independent_models.is_empty() {
                // Independent models: Aggregate across seasons
                let models_slice: Vec<&crate::input::NoiseModel> =
                    independent_models.iter().map(|&&m| m).collect();
                Self::convert_independent_models(
                    uncertainty_type,
                    entity_id,
                    &models_slice,
                )?
            } else {
                // This should never happen (group must contain at least one model)
                return Err(format!(
                    "Internal error: Empty group for {:?} entity {}",
                    unc_type_copy, entity_id
                ));
            };

            // Step 6: Validate converted spec
            unified_spec.validate().map_err(|e| {
                format!(
                    "Validation failed for converted {:?} entity {}: {}",
                    unc_type_copy, entity_id, e
                )
            })?;

            unified_specs.push(unified_spec);
        }

        Ok(unified_specs)
    }

    /// Convert a PAR model from a single NoiseModel entry
    ///
    /// Extracts all seasonal parameters from the PeriodicAutoregressive variant.
    ///
    /// # Performance
    ///
    /// O(num_seasons) - iterates through seasonal arrays to build HashMaps
    fn convert_par_model(
        uncertainty_type: UncertaintyType,
        entity_id: usize,
        model: &crate::input::NoiseModel,
    ) -> Result<UnifiedNoiseSpec, String> {
        use crate::input::TemporalModel;

        match &model.temporal_model {
            TemporalModel::PeriodicAutoregressive {
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => {
                // PERFORMANCE: Pre-allocate HashMaps with known capacity
                let mut seasonal_params = HashMap::with_capacity(*num_seasons);
                let mut seasonal_ar_params = HashMap::with_capacity(*num_seasons);

                // Extract seasonal parameters
                for season in 0..*num_seasons {
                    seasonal_params.insert(
                        season,
                        SeasonalNoiseParams {
                            mean: seasonal_means[season],
                            std_dev: seasonal_stds[season],
                            marginal_override: None,
                        },
                    );

                    seasonal_ar_params.insert(
                        season,
                        SeasonalPARParams {
                            ar_order: ar_orders[season],
                            ar_coefficients: ar_coefficients[season].clone(),
                        },
                    );
                }

                Ok(UnifiedNoiseSpec {
                    uncertainty_type,
                    entity_id,
                    temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                        num_seasons: *num_seasons,
                        seasonal_ar_params,
                    },
                    seasonal_params,
                    marginal_distribution: Some(model.distribution.clone()),
                })
            }
            _ => Err(format!(
                "Internal error: convert_par_model called with non-PAR model for entity {}",
                entity_id
            )),
        }
    }

    /// Convert independent models from multiple NoiseModel entries
    ///
    /// Aggregates seasonal parameters across all entries for the entity.
    ///
    /// # Performance
    ///
    /// O(n) where n = number of entries for this entity (typically 12 seasons)
    fn convert_independent_models(
        uncertainty_type: UncertaintyType,
        entity_id: usize,
        models: &[&crate::input::NoiseModel],
    ) -> Result<UnifiedNoiseSpec, String> {
        // PERFORMANCE: Pre-allocate HashMap with typical season count
        let mut seasonal_params = HashMap::with_capacity(models.len());

        for model in models {
            let season_id = model.season_id;

            // Check for duplicate season definitions
            if seasonal_params.contains_key(&season_id) {
                return Err(format!(
                    "Duplicate independent model definition for {:?} entity {} season {}",
                    uncertainty_type, entity_id, season_id
                ));
            }

            // Extract mean and std_dev from distribution
            let (mean, std_dev) = match &model.distribution {
                MarginalDistribution::Normal { mean, std_dev } => {
                    (*mean, *std_dev)
                }
                MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                    // For LogNormal3, we approximate mean and std_dev
                    // Mean ≈ γ + exp(μ + σ²/2)
                    // This is approximate since LogNormal3 is used for residuals
                    let approx_mean = gamma + (mu + sigma * sigma / 2.0).exp();
                    let approx_std = (mu + sigma * sigma / 2.0).exp()
                        * (sigma * sigma).exp_m1().sqrt();
                    (approx_mean, approx_std)
                }
            };

            seasonal_params.insert(
                season_id,
                SeasonalNoiseParams {
                    mean,
                    std_dev,
                    marginal_override: Some(model.distribution.clone()),
                },
            );
        }

        Ok(UnifiedNoiseSpec {
            uncertainty_type,
            entity_id,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
            marginal_distribution: None, // Per-season distributions in seasonal_params
        })
    }

    /// Convert UnifiedNoiseSpec to new public UncertaintySpecification format
    ///
    /// This is the reverse conversion for migration purposes. Enables automatic
    /// conversion from old `noise_models` format to new `uncertainty_specifications` format.
    ///
    /// # Algorithm
    ///
    /// 1. **Independent models**: Extract seasonal_params to SeasonalDistribution vec
    /// 2. **PAR models**: Extract seasonal_params and ar_params to PeriodicAr arrays
    /// 3. **Validate**: Ensure result is well-formed (all seasons present for PAR)
    ///
    /// # Returns
    ///
    /// - `Ok(UncertaintySpecification)`: Successfully converted
    /// - `Err(String)`: Conversion failed (e.g., missing seasons for PAR)
    ///
    /// # Performance
    ///
    /// O(num_seasons) - iterates through seasonal_params and ar_params once
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let unified = UnifiedNoiseSpec::from_noise_models(&old_models)?;
    /// let new_spec = unified[0].to_uncertainty_specification()?;
    /// // Use new_spec in new format JSON
    /// ```
    pub fn to_uncertainty_specification(
        &self,
    ) -> Result<crate::input::UncertaintySpecification, String> {
        use crate::input::{
            SeasonalDistribution, TemporalModelInput, UncertaintySpecification,
        };

        match &self.temporal_model {
            TemporalModelSpec::Independent => {
                // Independent model: extract seasonal distributions
                let mut seasonal_distributions =
                    Vec::with_capacity(self.seasonal_params.len());

                // Sort by season_id for deterministic output
                let mut seasons: Vec<_> =
                    self.seasonal_params.keys().copied().collect();
                seasons.sort_unstable();

                for season_id in seasons {
                    let params = &self.seasonal_params[&season_id];
                    // Convert back to SeasonalDistribution
                    let distribution = if let Some(ref marginal_dist) =
                        params.marginal_override
                    {
                        marginal_dist.clone()
                    } else {
                        MarginalDistribution::Normal {
                            mean: params.mean,
                            std_dev: params.std_dev,
                        }
                    };
                    seasonal_distributions.push(SeasonalDistribution {
                        season_id,
                        distribution,
                    });
                }

                Ok(UncertaintySpecification {
                    uncertainty_type: self.uncertainty_type.clone(),
                    entity_id: self.entity_id,
                    temporal_model: TemporalModelInput::Independent,
                    marginal_distribution: None,
                    seasonal_distributions: Some(seasonal_distributions),
                })
            }

            TemporalModelSpec::PeriodicAutoregressive {
                num_seasons,
                seasonal_ar_params,
            } => {
                // PAR model: extract arrays in season order

                // Validate all seasons present
                for season in 0..*num_seasons {
                    if !self.seasonal_params.contains_key(&season) {
                        return Err(format!(
                            "{:?} entity {}: Missing seasonal_params for season {} (required for PAR conversion)",
                            self.uncertainty_type, self.entity_id, season
                        ));
                    }
                    if !seasonal_ar_params.contains_key(&season) {
                        return Err(format!(
                            "{:?} entity {}: Missing seasonal_ar_params for season {} (required for PAR conversion)",
                            self.uncertainty_type, self.entity_id, season
                        ));
                    }
                }

                // Build arrays in season order (0..num_seasons)
                let mut ar_orders = Vec::with_capacity(*num_seasons);
                let mut ar_coefficients = Vec::with_capacity(*num_seasons);
                let mut seasonal_means = Vec::with_capacity(*num_seasons);
                let mut seasonal_stds = Vec::with_capacity(*num_seasons);

                for season in 0..*num_seasons {
                    let params = &self.seasonal_params[&season];
                    let ar_params = &seasonal_ar_params[&season];

                    seasonal_means.push(params.mean);
                    seasonal_stds.push(params.std_dev);
                    ar_orders.push(ar_params.ar_order);
                    ar_coefficients.push(ar_params.ar_coefficients.clone());
                }

                Ok(UncertaintySpecification {
                    uncertainty_type: self.uncertainty_type.clone(),
                    entity_id: self.entity_id,
                    temporal_model: TemporalModelInput::PeriodicAr {
                        num_seasons: *num_seasons,
                        ar_orders,
                        ar_coefficients,
                        seasonal_means,
                        seasonal_stds,
                    },
                    marginal_distribution: self.marginal_distribution.clone(),
                    seasonal_distributions: None,
                })
            }
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
        specs: &[crate::input::UncertaintySpecification],
    ) -> Result<Vec<UnifiedNoiseSpec>, String> {
        use crate::input::TemporalModelInput;

        if specs.is_empty() {
            return Err("No uncertainty specifications defined (empty input)"
                .to_string());
        }

        let mut unified_specs = Vec::with_capacity(specs.len());

        for spec in specs {
            match &spec.temporal_model {
                TemporalModelInput::Independent => {
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

                    unified_specs.push(UnifiedNoiseSpec {
                        uncertainty_type: spec.uncertainty_type.clone(),
                        entity_id: spec.entity_id,
                        temporal_model: TemporalModelSpec::Independent,
                        seasonal_params,
                        marginal_distribution: spec
                            .marginal_distribution
                            .clone(),
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
                                marginal_override: None,
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
                        marginal_distribution: spec
                            .marginal_distribution
                            .clone(),
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
/// # Example
///
/// ```rust,ignore
/// let specs = UnifiedNoiseSpec::from_noise_models(&noise_models)?;
/// validate_noise_specs(&specs, &graph, &system)?;
/// ```
pub fn validate_noise_specs(
    specs: &[UnifiedNoiseSpec],
    graph: &crate::input::GraphInput,
    system: &crate::input::SystemInput,
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
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }),
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
            marginal_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 0.6,
            }),
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
        };

        assert!(spec.validate().is_ok());
        assert!(spec.get_seasonal_params(0).is_some());
        assert!(spec.get_seasonal_params(1).is_none()); // Sparse - not defined
        assert!(spec.get_seasonal_params(5).is_some());
    }

    // ========================================================================
    // Converter Tests (from_noise_models)
    // ========================================================================

    #[test]
    fn test_convert_empty_input() {
        use crate::input::NoiseModel;

        let models: Vec<NoiseModel> = vec![];
        let result = UnifiedNoiseSpec::from_noise_models(&models);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty input"));
    }

    #[test]
    fn test_convert_simple_independent_load() {
        use crate::input::{NoiseModel, TemporalModel};

        // Independent load model across 3 seasons
        let models = vec![
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 1,
                distribution: MarginalDistribution::Normal {
                    mean: 75.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 2,
                distribution: MarginalDistribution::Normal {
                    mean: 85.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
        ];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);

        let spec = &specs[0];
        assert_eq!(spec.entity_id, 0);
        assert!(matches!(spec.uncertainty_type, UncertaintyType::Load));
        assert!(!spec.is_par_model());
        assert_eq!(spec.seasonal_params.len(), 3);
        assert_eq!(spec.get_seasonal_params(0).unwrap().mean, 80.0);
        assert_eq!(spec.get_seasonal_params(1).unwrap().mean, 75.0);
        assert_eq!(spec.get_seasonal_params(2).unwrap().mean, 85.0);
    }

    #[test]
    fn test_convert_par_inflow_12_seasons() {
        use crate::input::{NoiseModel, TemporalModel};

        // PAR inflow model with 12 seasons
        let models = vec![NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 0, // Misleading - PAR spans all seasons
            distribution: MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 0.6,
            },
            temporal_model: TemporalModel::PeriodicAutoregressive {
                num_seasons: 12,
                ar_orders: vec![1; 12],
                ar_coefficients: vec![vec![0.7]; 12],
                seasonal_means: vec![
                    100.0, 120.0, 150.0, 180.0, 200.0, 180.0, 150.0, 120.0,
                    100.0, 90.0, 80.0, 90.0,
                ],
                seasonal_stds: vec![
                    20.0, 25.0, 30.0, 35.0, 40.0, 35.0, 30.0, 25.0, 20.0, 18.0,
                    15.0, 18.0,
                ],
            },
        }];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);

        let spec = &specs[0];
        assert_eq!(spec.entity_id, 0);
        assert!(matches!(spec.uncertainty_type, UncertaintyType::Inflow));
        assert!(spec.is_par_model());
        assert_eq!(spec.num_seasons(), Some(12));
        assert_eq!(spec.seasonal_params.len(), 12);

        // Check specific season values
        assert_eq!(spec.get_seasonal_params(0).unwrap().mean, 100.0);
        assert_eq!(spec.get_seasonal_params(0).unwrap().std_dev, 20.0);
        assert_eq!(spec.get_seasonal_params(4).unwrap().mean, 200.0);
        assert_eq!(spec.get_seasonal_params(11).unwrap().mean, 90.0);

        // Check PAR parameters
        assert_eq!(spec.get_par_params(0).unwrap().ar_order, 1);
        assert_eq!(spec.get_par_params(5).unwrap().ar_coefficients, vec![0.7]);
    }

    #[test]
    fn test_convert_mixed_par_and_independent() {
        use crate::input::{NoiseModel, TemporalModel};

        // PAR inflow + independent load
        let models = vec![
            NoiseModel {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 0.0,
                    sigma: 0.6,
                },
                temporal_model: TemporalModel::PeriodicAutoregressive {
                    num_seasons: 12,
                    ar_orders: vec![1; 12],
                    ar_coefficients: vec![vec![0.7]; 12],
                    seasonal_means: vec![100.0; 12],
                    seasonal_stds: vec![20.0; 12],
                },
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
        ];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 2); // One for inflow, one for load

        // Find each spec
        let inflow_spec = specs
            .iter()
            .find(|s| matches!(s.uncertainty_type, UncertaintyType::Inflow))
            .unwrap();
        let load_spec = specs
            .iter()
            .find(|s| matches!(s.uncertainty_type, UncertaintyType::Load))
            .unwrap();

        assert!(inflow_spec.is_par_model());
        assert!(!load_spec.is_par_model());
    }

    #[test]
    fn test_convert_duplicate_par_definition() {
        use crate::input::{NoiseModel, TemporalModel};

        // Duplicate PAR definition (appears in multiple entries)
        let models = vec![
            NoiseModel {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 0.0,
                    sigma: 0.6,
                },
                temporal_model: TemporalModel::PeriodicAutoregressive {
                    num_seasons: 12,
                    ar_orders: vec![1; 12],
                    ar_coefficients: vec![vec![0.7]; 12],
                    seasonal_means: vec![100.0; 12],
                    seasonal_stds: vec![20.0; 12],
                },
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0,
                season_id: 5,
                distribution: MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 0.0,
                    sigma: 0.6,
                },
                temporal_model: TemporalModel::PeriodicAutoregressive {
                    num_seasons: 12,
                    ar_orders: vec![1; 12],
                    ar_coefficients: vec![vec![0.7]; 12],
                    seasonal_means: vec![100.0; 12],
                    seasonal_stds: vec![20.0; 12],
                },
            },
        ];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Duplicate PAR definition"));
        assert!(err.contains("entity 0"));
        assert!(err.contains("season_ids"));
    }

    #[test]
    fn test_convert_mixed_temporal_models_same_entity() {
        use crate::input::{NoiseModel, TemporalModel};

        // PAR and independent for same (uncertainty_type, entity_id)
        let models = vec![
            NoiseModel {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 0.0,
                    sigma: 0.6,
                },
                temporal_model: TemporalModel::PeriodicAutoregressive {
                    num_seasons: 12,
                    ar_orders: vec![1; 12],
                    ar_coefficients: vec![vec![0.7]; 12],
                    seasonal_means: vec![100.0; 12],
                    seasonal_stds: vec![20.0; 12],
                },
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0,
                season_id: 3,
                distribution: MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
        ];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Mixed temporal models"));
        assert!(err.contains("entity 0"));
    }

    #[test]
    fn test_convert_sparse_independent_model() {
        use crate::input::{NoiseModel, TemporalModel};

        // Independent model with only 4 seasons specified
        let models = vec![
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 1,
                season_id: 0,
                distribution: MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 1,
                season_id: 3,
                distribution: MarginalDistribution::Normal {
                    mean: 85.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 1,
                season_id: 6,
                distribution: MarginalDistribution::Normal {
                    mean: 90.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 1,
                season_id: 9,
                distribution: MarginalDistribution::Normal {
                    mean: 75.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
        ];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);

        let spec = &specs[0];
        assert_eq!(spec.seasonal_params.len(), 4);
        assert!(spec.get_seasonal_params(0).is_some());
        assert!(spec.get_seasonal_params(1).is_none()); // Not defined
        assert!(spec.get_seasonal_params(3).is_some());
        assert!(spec.get_seasonal_params(6).is_some());
        assert!(spec.get_seasonal_params(9).is_some());
    }

    #[test]
    fn test_convert_duplicate_independent_season() {
        use crate::input::{NoiseModel, TemporalModel};

        // Duplicate season definition for independent model
        let models = vec![
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 0, // Duplicate!
                distribution: MarginalDistribution::Normal {
                    mean: 85.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
        ];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Duplicate independent model"));
        assert!(err.contains("season 0"));
    }

    #[test]
    fn test_converted_specs_pass_validation() {
        use crate::input::{NoiseModel, TemporalModel};

        // Multiple entities with various configurations
        let models = vec![
            // PAR inflow for entity 0
            NoiseModel {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 0.0,
                    sigma: 0.6,
                },
                temporal_model: TemporalModel::PeriodicAutoregressive {
                    num_seasons: 12,
                    ar_orders: vec![1; 12],
                    ar_coefficients: vec![vec![0.7]; 12],
                    seasonal_means: vec![100.0; 12],
                    seasonal_stds: vec![20.0; 12],
                },
            },
            // Independent load for entity 0 season 0
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
            // Independent load for entity 0 season 1
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 1,
                distribution: MarginalDistribution::Normal {
                    mean: 75.0,
                    std_dev: 0.001,
                },
                temporal_model: TemporalModel::Independent,
            },
        ];

        let result = UnifiedNoiseSpec::from_noise_models(&models);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 2);

        // All specs should pass validation
        for spec in &specs {
            assert!(spec.validate().is_ok());
        }
    }

    // ================================
    // TICKET-03: Enhanced validation tests
    // ================================

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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
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
            marginal_distribution: None,
        };

        let result = spec.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("LogNormal3"));
        assert!(err_msg.contains("sigma"));
        assert!(err_msg.contains("must be > 0"));
    }

    // ================================
    // Reverse conversion tests (UnifiedNoiseSpec -> UncertaintySpecification)
    // ================================

    #[test]
    fn test_to_uncertainty_specification_independent_model() {
        use crate::input::TemporalModelInput;

        // Create independent model with 3 seasons
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
            1,
            SeasonalNoiseParams {
                mean: 120.0,
                std_dev: 25.0,
                marginal_override: None,
            },
        );
        seasonal_params.insert(
            2,
            SeasonalNoiseParams {
                mean: 90.0,
                std_dev: 15.0,
                marginal_override: None,
            },
        );

        let spec = UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 5,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
            marginal_distribution: None,
        };

        let result = spec.to_uncertainty_specification();
        assert!(result.is_ok());

        let unc_spec = result.unwrap();
        assert_eq!(unc_spec.entity_id, 5);
        assert!(matches!(unc_spec.uncertainty_type, UncertaintyType::Load));
        assert!(matches!(
            unc_spec.temporal_model,
            TemporalModelInput::Independent
        ));
        assert!(unc_spec.marginal_distribution.is_none());
        assert!(unc_spec.seasonal_distributions.is_some());

        let seasonal_dists = unc_spec.seasonal_distributions.unwrap();
        assert_eq!(seasonal_dists.len(), 3);

        // Check season 0
        let s0 = seasonal_dists.iter().find(|s| s.season_id == 0).unwrap();
        match &s0.distribution {
            MarginalDistribution::Normal { mean, std_dev } => {
                assert_eq!(*mean, 100.0);
                assert_eq!(*std_dev, 20.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        // Check season 1
        let s1 = seasonal_dists.iter().find(|s| s.season_id == 1).unwrap();
        match &s1.distribution {
            MarginalDistribution::Normal { mean, std_dev } => {
                assert_eq!(*mean, 120.0);
                assert_eq!(*std_dev, 25.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        // Check season 2
        let s2 = seasonal_dists.iter().find(|s| s.season_id == 2).unwrap();
        match &s2.distribution {
            MarginalDistribution::Normal { mean, std_dev } => {
                assert_eq!(*mean, 90.0);
                assert_eq!(*std_dev, 15.0);
            }
            _ => panic!("Expected Normal distribution"),
        }
    }

    #[test]
    fn test_to_uncertainty_specification_par_model() {
        use crate::input::TemporalModelInput;

        // Create PAR model with 12 seasons
        let mut seasonal_params = HashMap::new();
        let mut par_params = HashMap::new();

        for season in 0..12 {
            seasonal_params.insert(
                season,
                SeasonalNoiseParams {
                    mean: 100.0 + (season as f64) * 10.0,
                    std_dev: 20.0 + (season as f64),
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
            entity_id: 3,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 12,
                seasonal_ar_params: par_params,
            },
            seasonal_params,
            marginal_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 0.6,
            }),
        };

        let result = spec.to_uncertainty_specification();
        assert!(result.is_ok());

        let unc_spec = result.unwrap();
        assert_eq!(unc_spec.entity_id, 3);
        assert!(matches!(unc_spec.uncertainty_type, UncertaintyType::Inflow));
        assert!(unc_spec.seasonal_distributions.is_none());
        assert!(unc_spec.marginal_distribution.is_some());

        match unc_spec.temporal_model {
            TemporalModelInput::PeriodicAr {
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => {
                assert_eq!(num_seasons, 12);
                assert_eq!(ar_orders.len(), 12);
                assert_eq!(ar_coefficients.len(), 12);
                assert_eq!(seasonal_means.len(), 12);
                assert_eq!(seasonal_stds.len(), 12);

                // Check first season
                assert_eq!(ar_orders[0], 1);
                assert_eq!(ar_coefficients[0], vec![0.7]);
                assert_eq!(seasonal_means[0], 100.0);
                assert_eq!(seasonal_stds[0], 20.0);

                // Check last season
                assert_eq!(seasonal_means[11], 210.0);
                assert_eq!(seasonal_stds[11], 31.0);
            }
            _ => panic!("Expected PeriodicAr temporal model"),
        }
    }

    #[test]
    fn test_round_trip_conversion_independent() {
        // Test: Independent model -> UnifiedNoiseSpec -> UncertaintySpecification
        use crate::input::{NoiseModel, TemporalModel};

        let models = vec![
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 0,
                distribution: MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 10.0,
                },
                temporal_model: TemporalModel::Independent,
            },
            NoiseModel {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0,
                season_id: 1,
                distribution: MarginalDistribution::Normal {
                    mean: 90.0,
                    std_dev: 12.0,
                },
                temporal_model: TemporalModel::Independent,
            },
        ];

        // Convert old -> unified
        let unified = UnifiedNoiseSpec::from_noise_models(&models).unwrap();
        assert_eq!(unified.len(), 1);

        // Convert unified -> new
        let new_spec = unified[0].to_uncertainty_specification().unwrap();

        // Verify structure
        assert_eq!(new_spec.entity_id, 0);
        assert!(new_spec.seasonal_distributions.is_some());
        let seasonal = new_spec.seasonal_distributions.unwrap();
        assert_eq!(seasonal.len(), 2);
    }

    #[test]
    fn test_round_trip_conversion_par() {
        // Test: PAR model -> UnifiedNoiseSpec -> UncertaintySpecification
        use crate::input::{NoiseModel, TemporalModel};

        let models = vec![NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 0,
            distribution: MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 0.6,
            },
            temporal_model: TemporalModel::PeriodicAutoregressive {
                num_seasons: 12,
                ar_orders: vec![1; 12],
                ar_coefficients: vec![vec![0.7]; 12],
                seasonal_means: vec![100.0; 12],
                seasonal_stds: vec![20.0; 12],
            },
        }];

        // Convert old -> unified
        let unified = UnifiedNoiseSpec::from_noise_models(&models).unwrap();
        assert_eq!(unified.len(), 1);

        // Convert unified -> new
        let new_spec = unified[0].to_uncertainty_specification().unwrap();

        // Verify structure
        assert_eq!(new_spec.entity_id, 0);
        assert!(new_spec.marginal_distribution.is_some());
        assert!(new_spec.seasonal_distributions.is_none());
    }
}
