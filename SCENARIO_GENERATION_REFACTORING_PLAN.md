# Scenario Generation Refactoring Plan - Detailed Implementation

**Date**: 2025-11-02  
**Status**: Architectural Redesign Proposal - Detailed Implementation Specification  
**Objective**: Unify uncertainty handling across all entity types and temporal models

---

## Executive Summary

This document provides a detailed implementation plan for refactoring the scenario generation pipeline with specific:
- Struct and module names (current → intermediate → final)
- Code locations and affected files
- LP model changes (Variables, Constraints)
- Migration path for each component
- Dependencies between changes

**Core Insight**: Independent models are just PAR(0). By unifying them and using proper inverse CDF transformations, we get a simpler, mathematically correct architecture that works identically for loads and inflows.

---

## Naming Strategy and Module Evolution

### Phase 1: Parallel Implementation (New alongside Old)

Create new implementations without breaking existing code:

**New Modules**:
- `src/temporal_model.rs` - New unified temporal model (will replace `uncertainty_model.rs` parts)
- `src/uncertainty_constraints.rs` - Unified constraint management (will replace `inflow_constraints.rs`)

**New Structs**:
- `TemporalModel` - Unified representation (independent = PAR(0))
- `UncertaintyConstraintManager` - Manages lag buffers for all entities
- `UncertaintyConstraintData` - Precomputed data for constraint updates

### Phase 2: Migration (Old marked deprecated)

**Deprecations**:
- `uncertainty_model::UncertaintyModel` enum → kept for compatibility, deprecated
- `inflow_constraints` module → kept for compatibility, deprecated

**New Primary Names**:
- `temporal_model::TemporalModel` - The unified model
- `uncertainty_constraints::UncertaintyConstraintManager` - Unified for all entities

### Phase 3: Cleanup (Remove old code)

After 1-2 releases with deprecation warnings:
- Remove `UncertaintyModel::Independent` and `UncertaintyModel::PeriodicAR` enum
- Remove `inflow_constraints` module entirely
- Rename `temporal_model` → `uncertainty_model` if desired (but `temporal_model` is actually clearer)

---

## Detailed Module and Struct Specifications

### Module: `src/temporal_model.rs` (NEW)

This new module will contain the unified temporal model representation.

```rust
//! Unified temporal model for uncertainty representation
//!
//! This module provides a single, unified representation for all temporal
//! uncertainty models, eliminating the artificial distinction between
//! "Independent" and "PAR" models. Independent models are simply PAR(0)
//! models with ar_orders = [0, 0, ...].

use crate::input::{MarginalDistribution, UncertaintyType};
use crate::error::PowersError;

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

impl TemporalModel {
    /// Create from old Independent model (backward compatibility)
    pub fn from_independent(/* ... */) -> Result<Self, PowersError> {
        // Convert Independent to PAR(0)
    }
    
    /// Create from old PeriodicAR model (backward compatibility)
    pub fn from_par(/* ... */) -> Result<Self, PowersError> {
        // Use existing PAR data
    }
    
    /// Create from JSON specification (new unified format)
    pub fn from_specification(/* ... */) -> Result<Self, PowersError> {
        // Parse unified JSON format
    }
    
    /// Get seasonal parameters for a given season
    pub fn seasonal_params(&self, season_id: usize) -> SeasonalParams {
        // Return params for this season
    }
    
    /// Check if this model has AR dynamics (max_ar_order > 0)
    pub fn is_autoregressive(&self) -> bool {
        self.max_ar_order > 0
    }
}

/// Lightweight seasonal parameters (copied per use)
#[derive(Debug, Clone, Copy)]
pub struct SeasonalParams {
    pub mean: f64,
    pub std_dev: f64,
    pub distribution: MarginalDistribution,
}
```

### Module: `src/uncertainty_constraints.rs` (NEW)

This new module replaces `inflow_constraints.rs` and works for ALL entities.

```rust
//! Unified uncertainty constraint management for LP subproblems
//!
//! This module handles observation-space constraints for all uncertain entities
//! (both loads and inflows), using a unified approach:
//!
//! Y_t[i] = deterministic_base[i] + σ[i]·η_t[i] + Σ_{k=1}^{p} ψ_k[i]·Y_{t-k}[i]
//!
//! Where:
//! - Y_t[i]: Observation variable (load or inflow)
//! - η_t[i]: Innovation variable (from SAA)
//! - deterministic_base[i]: Precomputed μ - Σ(φ_k·μ_{season-k})
//! - σ[i]: Seasonal standard deviation
//! - ψ_k[i]: Transformed AR coefficients
//! - p: AR order (can be 0 for independent models)

use crate::temporal_model::TemporalModel;
use crate::input::UncertaintyType;

/// Constraint indices for observation-space formulation
///
/// Stores LP constraint indices for uncertainty constraints.
/// One constraint per entity (load or inflow).
#[derive(Debug, Clone)]
pub struct UncertaintyConstraintIndices {
    /// Constraint indices for all entities (loads + inflows)
    ///
    /// Constraint: Y_t[i] = deterministic_base + σ·η_t + Σ(ψ_k·Y_{t-k})
    pub observation_constraints: Vec<usize>,
}

/// Unified lag buffer for all uncertain entities
///
/// Manages lag observations [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}] for both
/// loads and inflows in a single, cache-friendly data structure.
///
/// # Memory Layout
///
/// Entities (loads + inflows) with lag counts [2, 0, 3, 1]:
/// Offsets:  [0, 2, 2, 5, 6]
/// Data:     [e0_lag0, e0_lag1, e2_lag0, e2_lag1, e2_lag2, e3_lag0]
///
/// Entity 1 has ar_order=0, so no lags stored.
#[derive(Debug, Clone)]
pub struct UnifiedLagBuffer {
    data: Vec<f64>,
    offsets: Vec<usize>,
    n_entities: usize,
}

impl UnifiedLagBuffer {
    /// Create lag buffer from temporal models
    ///
    /// Automatically determines lag counts from max_ar_order of each model.
    pub fn from_temporal_models(models: &[TemporalModel]) -> Self {
        let lag_counts: Vec<usize> = models
            .iter()
            .map(|m| m.max_ar_order)
            .collect();
        Self::new(&lag_counts)
    }
    
    pub fn new(lag_counts: &[usize]) -> Self {
        // Same implementation as OptimizedLagBuffer
    }
    
    /// Get lag observations for entity i: [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    #[inline]
    pub fn get_lags(&self, entity: usize) -> &[f64] {
        // Return empty slice if ar_order == 0
    }
    
    /// Get mutable lag observations for entity i
    #[inline]
    pub fn get_lags_mut(&mut self, entity: usize) -> &mut [f64] {
        // Return empty slice if ar_order == 0
    }
    
    /// Update lag buffer with new observation
    ///
    /// Shifts lags: [Y_{t-1}, Y_{t-2}, ...] → [Y_t, Y_{t-1}, ...]
    pub fn update_lags(&mut self, entity: usize, new_observation: f64) {
        // Shift and insert, or no-op if ar_order == 0
    }
}

/// Manager for uncertainty constraints (loads and inflows)
///
/// Replaces `ObservationSpaceConstraintManager` with unified handling.
#[derive(Debug, Clone)]
pub struct UncertaintyConstraintManager {
    /// Total number of uncertain entities (loads + inflows)
    dimension: usize,
    
    /// Lag buffer for all entities
    lag_buffer: UnifiedLagBuffer,
    
    /// Maximum lag order across all entities
    max_lag: usize,
    
    /// LP constraint indices (set during subproblem construction)
    constraint_indices: Option<UncertaintyConstraintIndices>,
}

impl UncertaintyConstraintManager {
    /// Create from temporal models
    pub fn from_temporal_models(models: &[TemporalModel]) -> Self {
        let dimension = models.len();
        let max_lag = models.iter().map(|m| m.max_ar_order).max().unwrap_or(0);
        let lag_buffer = UnifiedLagBuffer::from_temporal_models(models);
        
        Self {
            dimension,
            lag_buffer,
            max_lag,
            constraint_indices: None,
        }
    }
    
    /// Get lag observations for entity i
    #[inline]
    pub fn get_lag_observations(&self, entity: usize) -> &[f64] {
        self.lag_buffer.get_lags(entity)
    }
    
    /// Update lag buffer after LP solve
    pub fn update_lag_buffer(&mut self, entity: usize, observation: f64) {
        self.lag_buffer.update_lags(entity, observation);
    }
    
    /// Set constraint indices (called during subproblem construction)
    pub fn set_constraint_indices(&mut self, indices: UncertaintyConstraintIndices) {
        self.constraint_indices = Some(indices);
    }
}
```

### Changes to `src/subproblem.rs`

This is where the major LP model changes happen.

#### New/Modified Structs

```rust
/// LP variable indices
///
/// CHANGES:
/// - Add: `load_observation: Vec<usize>` - Load observation variables Y_load[bus]
/// - Add: `innovation: Vec<usize>` - Innovation variables η[entity] for all entities
/// - Keep: `inflow: Vec<usize>` - Inflow observation variables Y_inflow[hydro]
/// - Remove: `lagged_inflow_state` - unified in `lagged_observation_state`
/// - Add: `lagged_observation_state: Option<Vec<Vec<usize>>>` - Lags for all entities
#[derive(Clone)]
pub struct Variables {
    // Existing physical variables (unchanged)
    pub deficit: Vec<usize>,
    pub direct_exchange: Vec<usize>,
    pub reverse_exchange: Vec<usize>,
    pub thermal_gen: Vec<usize>,
    pub turbined_flow: Vec<usize>,
    pub spillage: Vec<usize>,
    pub stored_volume: Vec<usize>,
    
    // NEW: Load observation variables (one per bus)
    /// Load observations Y_load[bus] in observation space
    ///
    /// These are the actual load values used in load balance constraints.
    /// Related to innovations via: Y_load[b] = μ + σ·η_load[b] + Σ(ψ_k·Y_{t-k})
    pub load_observation: Vec<usize>,
    
    // NEW: Innovation variables (one per entity: loads + inflows)
    /// Innovation variables η[entity] for all uncertain entities
    ///
    /// These receive values from SAA during realize_uncertainties.
    /// Ordering: [η_load[0], η_load[1], ..., η_inflow[0], η_inflow[1], ...]
    pub innovation: Vec<usize>,
    
    // Modified: Inflow observation variables (unchanged indexing)
    /// Inflow observations Y_inflow[hydro] in observation space
    ///
    /// Related to innovations via: Y_inflow[h] = μ + σ·η_inflow[h] + Σ(ψ_k·Y_{t-k})
    pub inflow: Vec<usize>,
    
    // REMOVED: lagged_inflow_state (replaced by lagged_observation_state)
    
    // NEW: Unified lagged observation state variables
    /// Lagged observation state variables for all entities with AR dynamics
    ///
    /// Only present if state includes lagged observations (StorageAndObservationState).
    /// Ordering: Same as `innovation` (loads first, then inflows)
    /// Structure: lagged_observation_state[entity][lag_index]
    pub lagged_observation_state: Option<Vec<Vec<usize>>>,
    
    // Existing (unchanged)
    pub alpha: usize,
}

/// Constraint indices
///
/// CHANGES:
/// - Keep: `load_balance: Vec<usize>` - Load balance constraints (modified to use Y_load)
/// - Keep: `hydro_balance: Vec<usize>` - Hydro balance constraints (unchanged)
/// - Rename: `ar_dynamics` → `uncertainty_observation` (more general name)
/// - Add: `uncertainty_observation: Vec<usize>` - Observation constraints for all entities
#[derive(Clone)]
pub struct Constraints {
    /// Load balance constraints at each bus
    ///
    /// MODIFIED: Now references load_observation variables instead of direct RHS
    /// 
    /// Old: Σ generation = load (RHS set directly)
    /// New: Σ generation = Y_load[bus]
    pub load_balance: Vec<usize>,
    
    /// Hydro balance constraints (UNCHANGED)
    pub hydro_balance: Vec<usize>,
    
    // REMOVED: ar_dynamics (replaced by uncertainty_observation)
    
    /// Observation-space constraints for all uncertain entities
    ///
    /// One constraint per entity (loads + inflows):
    /// Y[i] = deterministic_base[i] + σ[i]·η[i] + Σ_k ψ_k[i]·Y_{t-k}[i]
    ///
    /// Ordering: [loads..., inflows...]
    pub uncertainty_observation: Vec<usize>,
}

/// Preprocessed constraint data for fast constraint updates
///
/// REPLACES: HydroConstraintData
/// BECOMES: UncertaintyConstraintData (works for both loads and inflows)
#[derive(Debug, Clone)]
pub struct UncertaintyConstraintData {
    /// Entity type (Load or Inflow)
    pub entity_type: UncertaintyType,
    
    /// Entity ID within its type
    pub entity_id: usize,
    
    /// Global entity index (in innovations vector)
    pub global_entity_idx: usize,
    
    /// LP constraint index
    pub constraint_idx: usize,
    
    /// LP observation variable index (load_observation[bus] or inflow[hydro])
    pub observation_var_idx: usize,
    
    /// LP innovation variable index
    pub innovation_var_idx: usize,
    
    /// Season ID for this subproblem
    pub season_id: usize,
    
    /// Seasonal mean μ_s
    pub seasonal_mean: f64,
    
    /// Seasonal std dev σ_s
    pub seasonal_std: f64,
    
    /// AR order for this entity in this season
    pub ar_order: usize,
    
    /// Transformed AR coefficients [ψ_1, ψ_2, ..., ψ_p]
    pub psi_coefficients: Vec<f64>,
    
    /// Precomputed deterministic base: μ_s - Σ(φ_k·μ_{s-k})
    pub deterministic_base: f64,
}

/// Main subproblem struct
///
/// CHANGES:
/// - Replace: `inflow_manager` → `uncertainty_manager`
/// - Replace: `hydro_data: Vec<HydroConstraintData>` → `entity_data: Vec<UncertaintyConstraintData>`
#[derive(Clone)]
pub struct Subproblem {
    pub model: Option<solver::Model>,
    pub state: Box<dyn state::State>,
    pub variables: Variables,
    pub constraints: Constraints,
    pub season_id: usize,
    
    // REMOVED: inflow_manager
    
    // NEW: Unified uncertainty constraint manager
    /// Manages lag buffers and constraints for all uncertain entities
    pub uncertainty_manager: uncertainty_constraints::UncertaintyConstraintManager,
    
    // REMOVED: hydro_data
    
    // NEW: Unified entity constraint data
    /// Preprocessed constraint data for all uncertain entities
    ///
    /// Ordered: [loads..., inflows...]
    /// Enables fast constraint updates in realize_uncertainties
    pub entity_data: Vec<UncertaintyConstraintData>,
}
```

#### Modified Methods in Subproblem

```rust
impl Subproblem {
    /// Constructor (MODIFIED)
    pub fn new_from_temporal_models(
        system: &system::System,
        state_choice: &str,
        temporal_models: &[temporal_model::TemporalModel],  // NEW: use TemporalModel
        season_id: usize,
    ) -> Self {
        let state = state::factory(state_choice, system, temporal_models);
        
        // NEW: Create unified uncertainty manager
        let mut uncertainty_manager =
            uncertainty_constraints::UncertaintyConstraintManager::from_temporal_models(
                temporal_models,
            );
        
        let mut pb = solver::Problem::new();
        
        // MODIFIED: add_variables now handles loads + inflows
        let variables = Self::add_variables_to_subproblem(
            &mut pb,
            system,
            state.as_ref(),
            temporal_models,
        );
        
        // MODIFIED: add_constraints now handles loads + inflows
        let constraints = Self::add_constraints_to_subproblem(
            &mut pb,
            &variables,
            system,
            state.as_ref(),
            temporal_models,
            season_id,
            &mut uncertainty_manager,
        );
        
        // NEW: Build entity_data (precomputed constraint data)
        let entity_data = Self::build_entity_constraint_data(
            temporal_models,
            &variables,
            &constraints,
            season_id,
        );
        
        // ... rest of constructor
        
        Self {
            model: Some(model),
            state,
            variables,
            constraints,
            season_id,
            uncertainty_manager,
            entity_data,
        }
    }
    
    /// Add variables (HEAVILY MODIFIED)
    fn add_variables_to_subproblem(
        pb: &mut solver::Problem,
        system: &system::System,
        state: &dyn state::State,
        temporal_models: &[temporal_model::TemporalModel],
    ) -> Variables {
        // ... existing physical variables (unchanged)
        
        // NEW: Add load observation variables
        let load_observation: Vec<usize> = system
            .buses
            .iter()
            .map(|bus| pb.add_column(0.0, 0.0..))  // Cost=0, non-negative
            .collect();
        
        // NEW: Add innovation variables for all entities
        let n_entities = temporal_models.len();
        let innovation: Vec<usize> = (0..n_entities)
            .map(|_| pb.add_column(0.0, ..))  // Cost=0, unbounded (can be negative!)
            .collect();
        
        // MODIFIED: Add inflow observation variables (same as before)
        let inflow: Vec<usize> = temporal_models
            .iter()
            .filter(|m| m.entity_type == UncertaintyType::Inflow)
            .map(|_| pb.add_column(0.0, 0.0..))  // Cost=0, non-negative
            .collect();
        
        // NEW: Add lagged observation state variables (unified for loads + inflows)
        let lagged_observation_state = if state.has_lagged_observation_state() {
            let mut lags = Vec::new();
            for model in temporal_models {
                let mut entity_lags = Vec::new();
                for lag_idx in 0..model.max_ar_order {
                    let var = pb.add_column(0.0, ..);  // Cost=0, unbounded
                    entity_lags.push(var);
                }
                lags.push(entity_lags);
            }
            Some(lags)
        } else {
            None
        };
        
        Variables {
            deficit,
            direct_exchange,
            reverse_exchange,
            thermal_gen,
            turbined_flow,
            spillage,
            stored_volume,
            load_observation,
            innovation,
            inflow,
            lagged_observation_state,
            alpha,
        }
    }
    
    /// Add constraints (HEAVILY MODIFIED)
    fn add_constraints_to_subproblem(
        pb: &mut solver::Problem,
        variables: &Variables,
        system: &system::System,
        _state: &dyn state::State,
        temporal_models: &[temporal_model::TemporalModel],
        _season_id: usize,
        uncertainty_manager: &mut uncertainty_constraints::UncertaintyConstraintManager,
    ) -> Constraints {
        // MODIFIED: Load balance now uses Y_load[bus] instead of RHS
        let mut load_balance: Vec<usize> = vec![0; system.meta.buses_count];
        for bus in system.buses.iter() {
            let mut factors = vec![
                (variables.deficit[bus.id], 1.0),
                (variables.load_observation[bus.id], -1.0),  // NEW: use variable instead of RHS
            ];
            // ... add generators, lines (same as before)
            load_balance[bus.id] = pb.add_row(0.0..0.0, &factors);
        }
        
        // Hydro balance (UNCHANGED)
        let mut hydro_balance: Vec<usize> = vec![0; system.meta.hydros_count];
        for hydro in system.hydros.iter() {
            // ... same as before
            hydro_balance[hydro.id] = pb.add_row(0.0..0.0, &factors);
        }
        
        // NEW: Add observation-space constraints for all entities
        let uncertainty_observation = Self::add_uncertainty_observation_constraints(
            pb,
            variables,
            temporal_models,
            uncertainty_manager,
        );
        
        Constraints {
            load_balance,
            hydro_balance,
            uncertainty_observation,
        }
    }
    
    /// Add observation constraints (NEW METHOD)
    fn add_uncertainty_observation_constraints(
        pb: &mut solver::Problem,
        variables: &Variables,
        temporal_models: &[temporal_model::TemporalModel],
        uncertainty_manager: &mut uncertainty_constraints::UncertaintyConstraintManager,
    ) -> Vec<usize> {
        let mut constraint_indices = Vec::new();
        
        let mut load_idx = 0;
        let mut inflow_idx = 0;
        
        for (entity_idx, model) in temporal_models.iter().enumerate() {
            let observation_var = match model.entity_type {
                UncertaintyType::Load => variables.load_observation[load_idx++],
                UncertaintyType::Inflow => variables.inflow[inflow_idx++],
            };
            
            let innovation_var = variables.innovation[entity_idx];
            
            // Build constraint: Y[i] - η[i] = 0 (RHS updated during realize_uncertainties)
            // Full form: Y[i] = deterministic_base + σ·η[i] + Σ(ψ_k·Y_{t-k})
            let factors = [
                (observation_var, 1.0),
                (innovation_var, -1.0),  // Will be modified to -σ during updates
            ];
            let row = pb.add_row(0.0..=0.0, factors);
            constraint_indices.push(row);
        }
        
        // Store indices in manager
        let indices = uncertainty_constraints::UncertaintyConstraintIndices {
            observation_constraints: constraint_indices.clone(),
        };
        uncertainty_manager.set_constraint_indices(indices);
        
        constraint_indices
    }
    
    /// Build precomputed entity data (NEW METHOD)
    fn build_entity_constraint_data(
        temporal_models: &[temporal_model::TemporalModel],
        variables: &Variables,
        constraints: &Constraints,
        season_id: usize,
    ) -> Vec<UncertaintyConstraintData> {
        let mut entity_data = Vec::new();
        
        let mut load_idx = 0;
        let mut inflow_idx = 0;
        
        for (global_idx, model) in temporal_models.iter().enumerate() {
            let (observation_var, entity_id) = match model.entity_type {
                UncertaintyType::Load => {
                    let var = variables.load_observation[load_idx];
                    let id = load_idx;
                    load_idx += 1;
                    (var, id)
                }
                UncertaintyType::Inflow => {
                    let var = variables.inflow[inflow_idx];
                    let id = inflow_idx;
                    inflow_idx += 1;
                    (var, id)
                }
            };
            
            entity_data.push(UncertaintyConstraintData {
                entity_type: model.entity_type,
                entity_id,
                global_entity_idx: global_idx,
                constraint_idx: constraints.uncertainty_observation[global_idx],
                observation_var_idx: observation_var,
                innovation_var_idx: variables.innovation[global_idx],
                season_id,
                seasonal_mean: model.seasonal_means[season_id],
                seasonal_std: model.seasonal_stds[season_id],
                ar_order: model.ar_orders[season_id],
                psi_coefficients: model.psi_coefficients[season_id].clone(),
                deterministic_base: model.deterministic_bases[season_id],
            });
        }
        
        entity_data
    }
    
    /// REMOVED: set_load_balance_rhs() - no longer needed
    
    /// Update uncertainty constraints (REPLACES update_ar_constraints_optimized)
    fn update_uncertainty_constraints(&mut self, innovations: &[f64]) {
        if let Some(model) = self.model.as_mut() {
            for data in &self.entity_data {
                let innovation = innovations[data.global_entity_idx];
                let stochastic_term = data.seasonal_std * innovation;
                let mut rhs = data.deterministic_base + stochastic_term;
                
                // Add AR lag contribution (if ar_order > 0)
                if data.ar_order > 0 {
                    let lag_obs = self.uncertainty_manager.get_lag_observations(
                        data.global_entity_idx
                    );
                    let lag_contribution = crate::utils::dot_product(
                        &data.psi_coefficients,
                        lag_obs,
                    );
                    rhs += lag_contribution;
                }
                
                // Update constraint: Y[i] = rhs
                model.change_rows_bounds(data.constraint_idx, rhs, rhs);
            }
        }
    }
    
    /// Realize uncertainties (MODIFIED)
    pub fn realize_uncertainties(
        &mut self,
        noises: &scenario::NoiseRealization,
        realization_container: &mut Realization,
    ) -> Result<RealizeUncertaintiesTiming, String> {
        // ... timing setup
        
        // OLD: separate load and inflow handling
        // let load = noises.get_load_innovations();
        // self.set_load_balance_rhs(load);
        // let innovations = noises.get_inflow_innovations();
        // self.update_ar_constraints_optimized(innovations);
        
        // NEW: unified innovation handling
        let all_innovations = noises.get_all_innovations();  // NEW METHOD
        self.update_uncertainty_constraints(all_innovations);
        
        // ... solve LP, extract solution (mostly unchanged)
        
        // NEW: Update lag buffers for all entities (not just inflows)
        if let Some(sol) = &solution {
            for data in &self.entity_data {
                if data.ar_order > 0 {
                    let observation = sol.col_value[data.observation_var_idx];
                    self.uncertainty_manager.update_lag_buffer(
                        data.global_entity_idx,
                        observation
                    );
                }
            }
        }
        
        // ... rest of method
    }
}
```

### Changes to `src/scenario.rs`

Currently has separate `load_innovations` and `inflow_innovations`. Need to unify.

```rust
/// Noise realization for a single stage
///
/// CHANGES:
/// - REMOVE: `load_innovations: Vec<f64>`
/// - REMOVE: `inflow_innovations: Vec<f64>`
/// - ADD: `innovations: Vec<f64>` - All innovations (loads + inflows)
/// - Keep counts for indexing
#[derive(Clone, Debug)]
pub struct NoiseRealization {
    // REMOVED: load_innovations, inflow_innovations
    
    // NEW: Unified innovations vector
    /// All innovations in order: [loads..., inflows...]
    pub innovations: Vec<f64>,
    
    /// Number of load entities (for splitting innovations)
    pub num_load_entities: usize,
    
    /// Number of inflow entities (for splitting innovations)
    pub num_inflow_entities: usize,
}

impl NoiseRealization {
    /// Get all innovations (NEW METHOD)
    #[inline]
    pub fn get_all_innovations(&self) -> &[f64] {
        &self.innovations
    }
    
    /// Get load innovations only (BACKWARD COMPATIBILITY)
    #[deprecated(note = "Use get_all_innovations() instead")]
    #[inline]
    pub fn get_load_innovations(&self) -> &[f64] {
        &self.innovations[0..self.num_load_entities]
    }
    
    /// Get inflow innovations only (BACKWARD COMPATIBILITY)
    #[deprecated(note = "Use get_all_innovations() instead")]
    #[inline]
    pub fn get_inflow_innovations(&self) -> &[f64] {
        let start = self.num_load_entities;
        &self.innovations[start..]
    }
}
```

### Changes to `src/input.rs`

Need to support new unified JSON format while maintaining backward compatibility.

```rust
/// Temporal model specification (MODIFIED)
///
/// OLD: enum with Independent vs PeriodicAr variants
/// NEW: struct with unified fields (ar_orders can be all zeros)
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct TemporalModelInput {
    pub num_seasons: usize,
    pub seasonal_means: Vec<f64>,
    pub seasonal_stds: Vec<f64>,
    pub ar_orders: Vec<usize>,
    pub ar_coefficients: Vec<Vec<f64>>,
}

/// OLD temporal model enum (DEPRECATED, for backward compatibility)
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[deprecated(note = "Use TemporalModelInput struct instead")]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LegacyTemporalModelInput {
    Independent,
    #[serde(rename = "periodic_ar")]
    PeriodicAr {
        num_seasons: usize,
        ar_orders: Vec<usize>,
        ar_coefficients: Vec<Vec<f64>>,
        seasonal_means: Vec<f64>,
        seasonal_stds: Vec<f64>,
    },
}

impl LegacyTemporalModelInput {
    /// Convert to new unified format
    pub fn to_unified(
        &self,
        seasonal_distributions: &[SeasonalDistribution],
    ) -> Result<TemporalModelInput, PowersError> {
        match self {
            Self::Independent => {
                let num_seasons = seasonal_distributions.len();
                
                // Extract means/stds from seasonal distributions
                let seasonal_means = seasonal_distributions
                    .iter()
                    .map(|d| extract_mean(&d.distribution))
                    .collect::<Result<Vec<_>, _>>()?;
                let seasonal_stds = seasonal_distributions
                    .iter()
                    .map(|d| extract_std(&d.distribution))
                    .collect::<Result<Vec<_>, _>>()?;
                
                Ok(TemporalModelInput {
                    num_seasons,
                    seasonal_means,
                    seasonal_stds,
                    ar_orders: vec![0; num_seasons],
                    ar_coefficients: vec![vec![]; num_seasons],
                })
            }
            Self::PeriodicAr {
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => {
                Ok(TemporalModelInput {
                    num_seasons: *num_seasons,
                    seasonal_means: seasonal_means.clone(),
                    seasonal_stds: seasonal_stds.clone(),
                    ar_orders: ar_orders.clone(),
                    ar_coefficients: ar_coefficients.clone(),
                })
            }
        }
    }
}

/// Helper to extract mean from marginal distribution
fn extract_mean(dist: &MarginalDistribution) -> Result<f64, PowersError> {
    match dist {
        MarginalDistribution::Normal { mean, .. } => Ok(*mean),
        MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
            // True mean of LogNormal3: γ + exp(μ + σ²/2)
            Ok(gamma + (mu + sigma * sigma / 2.0).exp())
        }
    }
}

/// Helper to extract std from marginal distribution
fn extract_std(dist: &MarginalDistribution) -> Result<f64, PowersError> {
    match dist {
        MarginalDistribution::Normal { std_dev, .. } => Ok(*std_dev),
        MarginalDistribution::LogNormal3 { mu, sigma, .. } => {
            // Std dev of LogNormal: exp(μ + σ²/2) * sqrt(exp(σ²) - 1)
            let exp_mu_sigma2 = (mu + sigma * sigma / 2.0).exp();
            let var_factor = (sigma * sigma).exp() - 1.0;
            Ok(exp_mu_sigma2 * var_factor.sqrt())
        }
    }
}

/// Uncertainty specification (MODIFIED to support both formats)
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct UncertaintySpecification {
    pub uncertainty_type: UncertaintyType,
    pub entity_id: usize,
    
    // Support both old and new formats during transition
    #[serde(flatten)]
    pub temporal_model: TemporalModelInputWrapper,
    
    pub seasonal_distributions: Option<Vec<SeasonalDistribution>>,
}

/// Wrapper to support both legacy and new formats
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum TemporalModelInputWrapper {
    New(TemporalModelInput),
    Legacy(LegacyTemporalModelInput),
}

impl TemporalModelInputWrapper {
    pub fn to_unified(
        &self,
        seasonal_distributions: &[SeasonalDistribution],
    ) -> Result<TemporalModelInput, PowersError> {
        match self {
            Self::New(new) => Ok(new.clone()),
            Self::Legacy(legacy) => legacy.to_unified(seasonal_distributions),
        }
    }
}
```

### Changes to `src/uncertainty_model.rs` (Input module)

Add proper inverse CDF transformation for marginal distributions.

```rust
impl MarginalDistribution {
    /// Transform standard normal to target distribution via inverse CDF
    ///
    /// This implements the probability integral transform:
    /// 1. Z ~ N(0,1) → U ~ Uniform(0,1) via Φ(Z)
    /// 2. U → target distribution via F⁻¹(U)
    ///
    /// This is the mathematically correct way to transform distributions
    /// while preserving correlation structure (via Gaussian copula).
    ///
    /// # Arguments
    ///
    /// - `z`: Standard normal sample Z ~ N(0,1)
    ///
    /// # Returns
    ///
    /// Sample from target distribution in innovation space
    pub fn inverse_cdf(&self, z: f64) -> f64 {
        use statrs::distribution::{Normal, LogNormal, ContinuousCDF};
        
        // Step 1: Z ~ N(0,1) → U ~ Uniform(0,1)
        let standard_normal = Normal::standard();
        let u = standard_normal.cdf(z);
        
        // Step 2: U → target distribution via inverse CDF
        match self {
            Self::Normal { mean, std_dev } => {
                let target = Normal::new(*mean, *std_dev).unwrap();
                target.inverse_cdf(u)
            }
            Self::LogNormal3 { gamma, mu, sigma } => {
                // LogNormal3: X = γ + Y where Y ~ LogNormal(μ, σ)
                let log_normal = LogNormal::new(*mu, *sigma).unwrap();
                gamma + log_normal.inverse_cdf(u)
            }
        }
    }
}
```

---

## Implementation Phases with Detailed Steps

### Phase 1: Add Inverse CDF and Parallel Modules (Week 1-2)

**Goal**: Add new implementations without breaking existing code

**Steps**:

1. **Create `src/temporal_model.rs`** (new module)
   - Implement `TemporalModel` struct
   - Implement `from_independent()`, `from_par()` constructors
   - Add conversion methods from old `UncertaintyModel`
   - Add tests

2. **Create `src/uncertainty_constraints.rs`** (new module)
   - Implement `UnifiedLagBuffer`
   - Implement `UncertaintyConstraintManager`
   - Implement `UncertaintyConstraintData`
   - Add tests

3. **Add inverse CDF to `MarginalDistribution`** (in `input.rs`)
   - Implement `inverse_cdf()` method using statrs
   - Add unit tests comparing with old `transform()` for Normal
   - Verify LogNormal3 correctness

4. **Update `scenario_generator.rs`** to use inverse CDF
   - Keep old code, add feature flag `#[cfg(feature = "new-marginal")]`
   - Use `params.distribution.inverse_cdf(base_noise)`
   - Test with Normal distribution (should be identical)
   - Test with LogNormal3 (values will change - this is the fix!)

**Deliverables**:
- Two new modules alongside old ones
- Inverse CDF implementation
- Feature-flagged scenario generator
- Comprehensive tests

**Risk**: Low - all additive, no breaking changes

---

### Phase 2: Refactor Subproblem to Use New Modules (Week 3-4)

**Goal**: Update LP model to use unified approach

**Steps**:

1. **Add new structs to `subproblem.rs`**
   - Add `UncertaintyConstraintData` struct
   - Keep old `HydroConstraintData` for now

2. **Extend `Variables` struct**
   - Add `load_observation: Vec<usize>`
   - Add `innovation: Vec<usize>`
   - Add `lagged_observation_state: Option<Vec<Vec<usize>>>`
   - Keep old fields, mark some as `#[deprecated]`

3. **Extend `Constraints` struct**
   - Add `uncertainty_observation: Vec<usize>`
   - Keep `ar_dynamics` for now, mark as `#[deprecated]`

4. **Add parallel constructor** `new_from_temporal_models_v2()`
   - Takes `&[TemporalModel]` instead of `&[UncertaintyModel]`
   - Uses new variable/constraint layout
   - Uses `UncertaintyConstraintManager`
   - Generates `entity_data`

5. **Add new methods**
   - `add_variables_v2()` - with load observation variables
   - `add_constraints_v2()` - unified for loads + inflows
   - `update_uncertainty_constraints()` - replaces `update_ar_constraints_optimized()`
   - `realize_uncertainties_v2()` - uses unified innovations

6. **Testing strategy**
   - Convert one example to use new constructor
   - Run side-by-side with old implementation
   - Compare LP models (variable/constraint counts)
   - Compare solutions (should be similar for Normal, different for LogNormal3)

**Deliverables**:
- Parallel implementation in subproblem
- At least one example working with new approach
- Validation that LP model structure is correct

**Risk**: Medium - core LP changes, but parallel implementation reduces risk

---

### Phase 3: Update JSON Schema and Migrate Examples (Week 5)

**Goal**: Simplify JSON format, migrate examples

**Steps**:

1. **Design new unified JSON schema**
   - Document new `TemporalModelInput` struct format
   - Show example conversions

2. **Implement backward compatibility**
   - Add `LegacyTemporalModelInput` enum
   - Add `TemporalModelInputWrapper` with serde(untagged)
   - Add conversion logic `to_unified()`
   - Add deprecation warnings

3. **Create migration tool** (optional but helpful)
   - Script to convert old JSON to new format
   - Validates both formats
   - Can process entire example directories

4. **Migrate examples one by one**
   - Start with simplest: `03-multistage`
   - For each example:
     - Convert JSON to new format
     - Run with both old and new implementations
     - Compare outputs (document differences)
     - Update README

5. **Update documentation**
   - JSON schema documentation
   - Migration guide
   - Examples overview

**Deliverables**:
- All examples in new JSON format
- Backward compatibility maintained
- Migration guide
- Updated documentation

**Risk**: Low - backward compatibility ensures no breakage

---

### Phase 4: Remove Old Implementation (Week 6)

**Goal**: Clean up deprecated code

**Steps**:

1. **Remove old implementations**
   - Remove `UncertaintyModel::Independent` and `::PeriodicAR` enum variants
   - Remove `inflow_constraints` module (rename to `_deprecated_inflow_constraints`)
   - Remove old subproblem methods
   - Remove feature flag from scenario_generator

2. **Rename modules** (optional, can defer)
   - `temporal_model.rs` is good name, keep it
   - `uncertainty_constraints.rs` is good name, keep it
   - Could rename `uncertainty_model.rs` → `marginal_distributions.rs` to avoid confusion

3. **Update all call sites**
   - Update SDDP algorithm code
   - Update tests
   - Update benchmarks

4. **Final cleanup**
   - Remove `#[deprecated]` attributes
   - Remove unused imports
   - Run clippy
   - Format code

**Deliverables**:
- Clean codebase without old implementations
- All tests passing
- Code quality checks passing

**Risk**: Low - everything already working in Phase 2-3

---

### Phase 5: Unify Load/Inflow in State (Week 7, Optional)

**Goal**: Update state to track lagged observations for all entities

**Current**: `StorageAndInflowState` only tracks inflow lags

**Proposed**: `StorageAndObservationState` tracks all entity lags

**Steps**:

1. **Extend state interface** (`src/state.rs`)
   - Add `has_lagged_observation_state()` method
   - Add `get_lagged_observations()`, `set_lagged_observations()` methods

2. **Create new state variant**
   - `StorageAndObservationState` - tracks storage + all entity observations
   - Replaces `StorageAndInflowState`

3. **Update state factory**
   - Keep old state for backward compatibility
   - Add new state option

4. **Test with AR loads** (if desired)
   - Create example with load AR dynamics
   - Verify lag tracking works correctly

**Deliverables**:
- Extended state interface
- New state variant
- Example with AR loads (if desired)

**Risk**: Low - additive feature, old states still work

---

## Validation and Testing Strategy

### Unit Tests

For each new module:

1. **`temporal_model.rs`**
   - Test conversion from Independent → TemporalModel (ar_orders all zero)
   - Test conversion from PAR → TemporalModel
   - Test precomputation of psi coefficients
   - Test precomputation of deterministic bases

2. **`uncertainty_constraints.rs`**
   - Test `UnifiedLagBuffer` with various ar_orders including zeros
   - Test lag updates and retrieval
   - Test `UncertaintyConstraintManager` creation

3. **`MarginalDistribution::inverse_cdf`**
   - Test Normal distribution (compare with direct transform)
   - Test LogNormal3 distribution (verify non-negativity, moments)
   - Test correlation preservation (copula property)

4. **`subproblem.rs`**
   - Test LP model structure (variable/constraint counts)
   - Test constraint update logic
   - Test lag buffer updates

### Integration Tests

1. **Example comparison**
   - Run each example with old and new implementations
   - Compare objective values (should be close for Normal)
   - Document LogNormal3 differences (expected - fixing bug)

2. **SDDP algorithm**
   - Full SDDP runs with new implementation
   - Compare convergence behavior
   - Compare final policies

3. **Performance benchmarks**
   - Scenario generation time
   - LP solve time
   - Memory usage

### Regression Tests

1. **JSON parsing**
   - Test old format still works
   - Test new format works
   - Test mixed formats in same file

2. **State transitions**
   - Test lag buffer tracking across stages
   - Test state serialization/deserialization

---

## Migration Guide for Users

### JSON Format Changes

#### Independent Model

**Before**:
```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "type": "independent"
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 40.0, "std_dev": 10.0}
  ]
}
```

**After**:
```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [40.0, 40.0, ..., 40.0],
    "seasonal_stds": [10.0, 10.0, ..., 10.0],
    "ar_orders": [0, 0, ..., 0],
    "ar_coefficients": [[], [], ..., []]
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 40.0, "std_dev": 10.0}
  ]
}
```

#### PAR Model

**Before**:
```json
{
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 12,
    "ar_orders": [1, 1, ...],
    "ar_coefficients": [[0.7], [0.7], ...],
    "seasonal_means": [70.0, 65.0, ...],
    "seasonal_stds": [20.0, 20.0, ...]
  }
}
```

**After** (just remove `"type": "periodic_ar"`):
```json
{
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [70.0, 65.0, ...],
    "seasonal_stds": [20.0, 20.0, ...],
    "ar_orders": [1, 1, ...],
    "ar_coefficients": [[0.7], [0.7], ...]
  }
}
```

### Expected Result Changes

1. **Normal distribution**: Results should be identical

2. **LogNormal3 distribution**: Results will change because the old implementation had incorrect semantics. New results are mathematically correct.

3. **Performance**: Expect 5-10% improvement in scenario generation time due to unified code paths.

---

## Summary Checklist

### New Modules Created
- [ ] `src/temporal_model.rs` - Unified temporal model
- [ ] `src/uncertainty_constraints.rs` - Unified constraint management

### Modified Modules
- [ ] `src/scenario_generator.rs` - Use inverse CDF
- [ ] `src/subproblem.rs` - New Variables/Constraints, unified approach
- [ ] `src/scenario.rs` - Unified innovations
- [ ] `src/input.rs` - New JSON format, backward compatibility
- [ ] `src/state.rs` - Extended for lagged observations (optional)

### Deprecated/Removed
- [ ] `UncertaintyModel` enum (kept for compatibility, marked deprecated)
- [ ] `inflow_constraints` module (replaced by `uncertainty_constraints`)
- [ ] `set_load_balance_rhs()` method (load balance uses variables now)
- [ ] Separate `load_innovations`/`inflow_innovations` (unified to `innovations`)

### New Structs
- [ ] `TemporalModel` - Main temporal model representation
- [ ] `UnifiedLagBuffer` - Lag buffer for all entities
- [ ] `UncertaintyConstraintManager` - Constraint manager for all entities
- [ ] `UncertaintyConstraintData` - Precomputed constraint data

### Modified Structs
- [ ] `Variables` - Added `load_observation`, `innovation`, `lagged_observation_state`
- [ ] `Constraints` - Added `uncertainty_observation`, removed focus on just AR
- [ ] `Subproblem` - Uses `uncertainty_manager` and `entity_data`
- [ ] `NoiseRealization` - Unified `innovations` vector

### JSON Schema
- [ ] New `TemporalModelInput` struct format
- [ ] Backward compatibility with `LegacyTemporalModelInput`
- [ ] All examples migrated

### Documentation
- [ ] Architecture documentation updated
- [ ] JSON schema documentation
- [ ] Migration guide
- [ ] CHANGELOG entries

---

## Timeline Summary

| Week | Phase | Focus | Risk |
|------|-------|-------|------|
| 1-2 | Phase 1 | Parallel modules, inverse CDF | Low |
| 3-4 | Phase 2 | Subproblem refactoring | Medium |
| 5 | Phase 3 | JSON schema, examples | Low |
| 6 | Phase 4 | Cleanup, deprecation | Low |
| 7 | Phase 5 | Optional: AR loads | Low |
| 8 | Testing | Validation, benchmarks | - |

**Total**: 6-8 weeks for complete refactoring with thorough testing.

---

## Questions for Review

1. **Naming**: Do you prefer `TemporalModel` or something else? Should we eventually rename to just `Model`?

2. **Module names**: `temporal_model.rs` and `uncertainty_constraints.rs` vs other names?

3. **Backward compatibility duration**: How many releases should we keep deprecated code?

4. **Phase 5 priority**: Should AR load support be in Phase 2-3 or deferred to Phase 5?

5. **State refactoring**: Should we unify Storage and Observation states now or later?

---

This detailed plan provides a concrete roadmap with specific struct names, file locations, and implementation steps. Ready to begin Phase 1?
