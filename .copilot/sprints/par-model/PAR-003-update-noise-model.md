# PAR-003: Update NoiseModel to Support Periodic Residual Semantics

## Context

Currently, `MarginalDistribution` in `NoiseModel` is applied to the final series for Independent and to innovations for stationary AR. However, CEPEL's PAR(p) methodology requires a **semantic shift**: the marginal distribution (LogNormal3) is applied to **residuals** (aₜ), not the final series (Zₜ).

This ticket clarifies the semantics and prepares `NoiseModel` for PAR integration without breaking existing functionality.

## Acceptance Criteria

- [ ] `NoiseModel` documentation clearly states when marginal is applied to residuals vs final series
- [ ] New field or semantic marker distinguishes PAR residual application
- [ ] Backward compatible: existing Independent and AR configs work unchanged
- [ ] Schema supports explicit `residual_distribution` field for PAR models
- [ ] Clear migration path documented for users moving from stationary AR to PAR
- [ ] All existing tests pass without modification

## Tasks

### Implementation

- [ ] Add doc comments to `NoiseModel` clarifying distribution semantics:
  - Independent: marginal applied to final series
  - Stationary AR: marginal applied to innovations εₜ
  - Periodic AR: marginal applied to residuals aₜ (NEW!)
- [ ] Add optional `residual_distribution` field to `NoiseModel` (for PAR only)
  - Type: `Option<MarginalDistribution>`
  - Only populated when `temporal_model` is `PeriodicAutoregressive`
- [ ] Update `NoiseModel::validate()` to check consistency:
  - If PeriodicAR: `residual_distribution` should be `Some(...)`
  - If Independent/AR: `residual_distribution` should be `None`
- [ ] Add conversion logic in `NoiseModel` to route distribution correctly
- [ ] Ensure serde `#[serde(default)]` on `residual_distribution` for backward compat

### Testing

- [ ] Unit test: NoiseModel with Independent uses marginal_distribution for final series
- [ ] Unit test: NoiseModel with Autoregressive uses marginal_distribution for innovations
- [ ] Unit test: NoiseModel with PeriodicAutoregressive uses residual_distribution
- [ ] Unit test: Validation catches PeriodicAR without residual_distribution
- [ ] Unit test: Validation accepts Independent with no residual_distribution
- [ ] Regression test: Existing JSON configs deserialize correctly (backward compat)
- [ ] Integration test: Old AR configs continue to generate correct scenarios

### Documentation

- [ ] Update `NoiseModel` doc comment with semantic table
- [ ] Add migration guide section explaining residual vs final series distinction
- [ ] Document example JSON for each temporal model type
- [ ] Add inline comments explaining CEPEL pipeline stages
- [ ] Reference PAR_MODEL_SUPPORT.md architecture section

## Technical Notes

### Implementation Approach

```rust
/// Noise model for a single entity (hydro inflow or bus load)
///
/// # Distribution Semantics (CRITICAL!)
///
/// The meaning of `marginal_distribution` **changes** based on `temporal_model`:
///
/// | Temporal Model        | Distribution Applied To         | Pipeline Stage |
/// |-----------------------|---------------------------------|----------------|
/// | Independent           | Final series Xₜ                 | Direct         |
/// | Autoregressive        | Innovations εₜ                  | Before AR      |
/// | PeriodicAutoregressive| **IGNORED** (use residual_dist) | N/A            |
///
/// For PAR models, use `residual_distribution` instead, which is applied to
/// de-seasonalized residuals aₜ before re-seasonalization.
///
/// # CEPEL Pipeline for PAR
///
/// 1. Generate w ~ N(0,1) (base noise)
/// 2. Apply correlation: b = D·w
/// 3. Transform residuals: a = residual_distribution.transform(b)  ← Uses residual_dist!
/// 4. Re-seasonalize: Z = μₘ + σₘ·[AR_term + a]
///
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NoiseModel {
    pub uncertainty_type: UncertaintyType,
    pub entity_id: usize,
    pub season_id: usize,

    /// Marginal distribution for final series (Independent) or innovations (AR)
    ///
    /// **Ignored for PeriodicAutoregressive** - use `residual_distribution` instead.
    pub marginal_distribution: MarginalDistribution,

    /// Innovation distribution for stationary AR models
    ///
    /// Only used when temporal_model is Autoregressive.
    pub innovation_distribution: Option<InnovationDistribution>,

    /// Temporal correlation model
    pub temporal_model: TemporalModel,

    /// Residual distribution for PAR models (CEPEL methodology)
    ///
    /// Applied to de-seasonalized residuals aₜ, not final series Zₜ.
    /// **Required** when temporal_model is PeriodicAutoregressive.
    /// **Must be None** for Independent or Autoregressive models.
    ///
    /// # CEPEL Semantics
    ///
    /// In PAR models, the residual aₜ is the "surprise" after accounting for
    /// seasonal mean, seasonal variation, and AR correlation. CEPEL applies
    /// LogNormal3 to these residuals to guarantee non-negativity while
    /// preserving the AR correlation structure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_distribution: Option<MarginalDistribution>,
}

impl NoiseModel {
    /// Validate consistency between temporal model and distribution fields
    pub fn validate(&self) -> Result<(), String> {
        match &self.temporal_model {
            TemporalModel::Independent => {
                if self.residual_distribution.is_some() {
                    return Err(
                        "Independent models must not have residual_distribution".to_string()
                    );
                }
                // marginal_distribution is used directly
                Ok(())
            }
            TemporalModel::Autoregressive { .. } => {
                if self.residual_distribution.is_some() {
                    return Err(
                        "Stationary AR models must not have residual_distribution".to_string()
                    );
                }
                // marginal_distribution applied to innovations (if specified)
                Ok(())
            }
            TemporalModel::PeriodicAutoregressive { .. } => {
                if self.residual_distribution.is_none() {
                    return Err(
                        "Periodic AR models require residual_distribution (applied to aₜ)".to_string()
                    );
                }
                // residual_distribution is mandatory, marginal_distribution ignored
                Ok(())
            }
        }
    }

    /// Get the distribution to apply at the current pipeline stage
    ///
    /// Returns the correct distribution based on temporal model semantics.
    pub fn get_active_distribution(&self) -> DistributionTarget {
        match &self.temporal_model {
            TemporalModel::Independent => {
                DistributionTarget::FinalSeries(&self.marginal_distribution)
            }
            TemporalModel::Autoregressive { .. } => {
                DistributionTarget::Innovations(&self.marginal_distribution)
            }
            TemporalModel::PeriodicAutoregressive { .. } => {
                DistributionTarget::Residuals(
                    self.residual_distribution.as_ref()
                        .expect("PAR model requires residual_distribution")
                )
            }
        }
    }
}

/// Target for distribution application (internal helper)
#[derive(Debug)]
pub enum DistributionTarget<'a> {
    FinalSeries(&'a MarginalDistribution),
    Innovations(&'a MarginalDistribution),
    Residuals(&'a MarginalDistribution),
}
```

### Example JSON Configs

```json
// Independent (existing, unchanged)
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "season_id": 1,
  "marginal_distribution": {
    "type": "lognormal3",
    "gamma": 1.0,
    "mu": 4.5,
    "sigma": 0.3
  },
  "temporal_model": {
    "type": "independent"
  }
}

// Stationary AR (existing, unchanged)
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "season_id": 1,
  "marginal_distribution": {
    "type": "normal",
    "mean": 0.0,
    "std_dev": 15.0
  },
  "temporal_model": {
    "type": "autoregressive",
    "lag_order": 1,
    "coefficients": [0.7]
  }
}

// Periodic AR (NEW!)
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "season_id": 1,
  "marginal_distribution": {
    "type": "normal",
    "mean": 0.0,
    "std_dev": 1.0
  },
  "residual_distribution": {
    "type": "lognormal3",
    "gamma": 1.0,
    "mu": 4.5,
    "sigma": 0.3
  },
  "temporal_model": {
    "type": "periodic_ar",
    "period": 12,
    "ar_orders": [1, 1, 2, ...],
    "ar_coefficients": [[0.7], [0.75], [0.6, 0.2], ...],
    "seasonal_means": [100.0, 120.0, ...],
    "seasonal_stds": [20.0, 25.0, ...]
  }
}
```

### Migration Notes

Users upgrading from stationary AR to PAR:

1. Keep `marginal_distribution` as placeholder (will be ignored)
2. Add `residual_distribution` with LogNormal3 parameters
3. Change `temporal_model` type to `periodic_ar`
4. Add seasonal parameters (means, stds, ar_coeffs)

## Dependencies

- **Blocked by**: PAR-001 (needs PeriodicAutoregressive variant)
- **Blocks**: PAR-009 (scenario generator needs these semantics)
- **Related**: PAR-002 (uses SeasonalStats types)

## Estimated Effort

**2 story points** (confidence: high)

- 2 hours implementation (field addition, validation, routing)
- 1.5 hours testing (7 tests)
- 1.5 hours documentation (table, examples, migration guide)
- 1 hour review

### Breakdown

- Add residual_distribution field and serde: 0.5 hours
- Implement validation logic: 1 hour
- Add get_active_distribution() helper: 0.5 hours
- Unit tests: 1.5 hours
- Documentation with semantic table: 1.5 hours
- Code review and refinement: 1 hour
