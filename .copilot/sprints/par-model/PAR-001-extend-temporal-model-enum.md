# PAR-001: Extend TemporalModel Enum with PeriodicAutoregressive Variant

## Context

The current `TemporalModel` enum in `src/input.rs` supports only `Independent` and `Autoregressive` (stationary AR). To implement CEPEL's PAR(p) methodology, we need to add a `PeriodicAutoregressive` variant that supports season-varying parameters (monthly means, standard deviations, and AR coefficients).

This is the foundational ticket for PAR support. All subsequent tickets depend on having these types in place.

## Acceptance Criteria

- [ ] `TemporalModel` enum has new `PeriodicAutoregressive` variant
- [ ] Variant includes all required periodic parameters (period, ar_orders, ar_coefficients, seasonal_means, seasonal_stds)
- [ ] Serde serialization/deserialization works correctly
- [ ] Backward compatible: existing `Independent` and `Autoregressive` variants unchanged
- [ ] Compiles without breaking existing code
- [ ] All existing tests pass

## Tasks

### Implementation

- [ ] Add `PeriodicAutoregressive` variant to `TemporalModel` enum in `src/input.rs`
- [ ] Define fields:
  - `period: usize` (12 for monthly, 4 for quarterly)
  - `ar_orders: Vec<usize>` (length = period, each element is pₘ for month m)
  - `ar_coefficients: Vec<Vec<f64>>` (length = period, each Vec has length ar_orders[m])
  - `seasonal_means: Vec<f64>` (length = period, μₘ for each month)
  - `seasonal_stds: Vec<f64>` (length = period, σₘ for each month)
- [ ] Add `#[serde(rename = "periodic_ar")]` attribute
- [ ] Add doc comments explaining each field with CEPEL notation
- [ ] Ensure `PartialEq`, `Debug`, `Clone`, `Deserialize`, `Serialize` derive macros

### Testing

- [ ] Unit test: Deserialize valid periodic AR JSON config
- [ ] Unit test: Serialize PeriodicAutoregressive back to JSON
- [ ] Unit test: Verify all three variants (Independent, Autoregressive, PeriodicAutoregressive) coexist
- [ ] Regression test: Existing AR configs still deserialize correctly

### Documentation

- [ ] Add comprehensive doc comment to `PeriodicAutoregressive` variant
- [ ] Include example JSON in doc comment
- [ ] Document relationship to CEPEL equation (12)
- [ ] Add inline comment referencing PAR_MODEL_SUPPORT.md

## Technical Notes

### Implementation Approach

````rust
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TemporalModel {
    Independent,

    Autoregressive {
        lag_order: usize,
        coefficients: Vec<f64>,
    },

    /// Periodic Autoregressive PAR(p) model (CEPEL methodology)
    ///
    /// Implements the CEPEL PAR(p) model where parameters vary by season/month:
    /// Zₜ = μₘ + σₘ·[Σφᵢₘ·(Zₜ₋ᵢ - μₘ₋ᵢ)/σₘ₋ᵢ + aₜ]
    ///
    /// # Example (Monthly PAR with varying order)
    /// ```json
    /// {
    ///   "type": "periodic_ar",
    ///   "period": 12,
    ///   "ar_orders": [1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1],
    ///   "ar_coefficients": [
    ///     [0.7], [0.75], [0.6, 0.2], ...
    ///   ],
    ///   "seasonal_means": [100.0, 120.0, 150.0, ...],
    ///   "seasonal_stds": [20.0, 25.0, 30.0, ...]
    /// }
    /// ```
    #[serde(rename = "periodic_ar")]
    PeriodicAutoregressive {
        /// Seasonal cycle length (12 for monthly, 4 for quarterly, etc.)
        /// Maps to season_id values in graph nodes
        period: usize,

        /// AR order per period (can differ by season!)
        /// Length must equal `period`
        ar_orders: Vec<usize>,

        /// AR coefficients per period
        /// ar_coefficients[m] has length ar_orders[m]
        ar_coefficients: Vec<Vec<f64>>,

        /// Mean per period (μₘ in CEPEL notation)
        seasonal_means: Vec<f64>,

        /// Standard deviation per period (σₘ in CEPEL notation)
        seasonal_stds: Vec<f64>,
    },
}
````

### Validation Considerations (for later tickets)

These will be validated in PAR-005, but keep in mind:

- `period` must be > 0 (typically 12 or 4)
- `ar_orders.len() == period`
- `ar_coefficients.len() == period`
- `seasonal_means.len() == period`
- `seasonal_stds.len() == period`
- `ar_coefficients[m].len() == ar_orders[m]` for all m
- All `seasonal_stds[m] > 0`
- AR coefficients should satisfy stationarity (will check in PAR-006)

### Edge Cases

- Empty vectors: Will be caught by validation in PAR-005
- Mismatched lengths: Will be caught by validation in PAR-005
- Zero or negative std dev: Will be caught by validation in PAR-005

## Dependencies

- **Blocked by**: None (foundational ticket)
- **Blocks**: PAR-002, PAR-003, PAR-004 (all need this type)
- **Related**: None

## Estimated Effort

**2 story points** (confidence: high)

- 2 hours implementation
- 1 hour testing
- 1 hour documentation
- 1 hour review and polish

### Breakdown

- Type definition and serde setup: 1 hour
- Doc comments and examples: 1 hour
- Unit tests (deserialize, serialize, coexistence): 1 hour
- Integration with existing code verification: 1 hour
- Buffer for unexpected issues: 1 hour
