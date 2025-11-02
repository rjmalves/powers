# JSON Schema v2 - Unified Temporal Model Format

**Version**: 2.0  
**Date**: 2025-11-02  
**Status**: Active (v1 deprecated but supported)

---

## Overview

Version 2 of the JSON schema introduces a **unified temporal model format** that eliminates the artificial distinction between Independent and PAR models. This design recognizes that Independent models are simply PAR(0) models (all AR orders are zero).

### Key Changes

1. **Unified `temporal_model` structure**: Single format for all temporal models
2. **Explicit seasonal parameters**: All models specify `seasonal_means` and `seasonal_stds`
3. **Zero-based independence**: Independent models use `ar_orders: [0, 0, ...]`
4. **Backward compatibility**: Old format (`{"type": "independent"}`) still supported

### Benefits

- **Consistency**: Same structure for all uncertainty types
- **Clarity**: Explicit parameters eliminate ambiguity
- **Extensibility**: Easy to add features like seasonal AR orders
- **Performance**: Unified code paths improve efficiency

---

## Temporal Model Specification

### Structure

```json
{
  "temporal_model": {
    "num_seasons": <integer>,
    "seasonal_means": [<float>, ...],
    "seasonal_stds": [<float>, ...],
    "ar_orders": [<integer>, ...],
    "ar_coefficients": [[<float>, ...], ...]
  }
}
```

### Fields

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `num_seasons` | integer | Number of seasons in the cycle | ≥ 1 |
| `seasonal_means` | array[float] | Mean for each season: μₛ | length = num_seasons |
| `seasonal_stds` | array[float] | Standard deviation for each season: σₛ | length = num_seasons, all > 0 |
| `ar_orders` | array[integer] | AR order for each season | length = num_seasons, all ≥ 0 |
| `ar_coefficients` | array[array[float]] | AR coefficients per season | length = num_seasons |

### Constraints

- **Array lengths**: All arrays must have length = `num_seasons`
- **AR coefficients**: For season `s`, `ar_coefficients[s]` must have length = `ar_orders[s]`
- **Standard deviations**: All values in `seasonal_stds` must be > 0
- **AR orders**: Non-negative integers
- **Independence**: If all `ar_orders` are 0, then all `ar_coefficients` must be empty arrays `[]`

---

## Model Types

### Independent Model (PAR(0))

Independent model with no temporal correlation between seasons.

**Characteristics**:
- All AR orders are zero: `ar_orders: [0, 0, ..., 0]`
- No AR coefficients: `ar_coefficients: [[], [], ..., []]`
- Each observation is independent: `Y_t = μ_s + σ_s · η_t`

**Example**: Two-season independent model

```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 2,
    "seasonal_means": [100.0, 80.0],
    "seasonal_stds": [20.0, 15.0],
    "ar_orders": [0, 0],
    "ar_coefficients": [[], []]
  },
  "seasonal_distributions": [
    {
      "season_id": 0,
      "type": "normal",
      "mean": 100.0,
      "std_dev": 20.0
    },
    {
      "season_id": 1,
      "type": "normal",
      "mean": 80.0,
      "std_dev": 15.0
    }
  ]
}
```

### Periodic Autoregressive Model (PAR)

Model with seasonal autoregressive dynamics.

**Characteristics**:
- AR orders can vary by season: `ar_orders: [p₀, p₁, ..., pₙ₋₁]`
- Each season has its own AR coefficients
- Observations depend on past values: `Y_t = μ_s + σ_s · η_t + Σ(φᵢ · Y_{t-i})`

**Example**: PAR(1) model with uniform order

```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [
      70.0, 65.0, 60.0, 55.0, 50.0, 45.0,
      50.0, 55.0, 60.0, 65.0, 70.0, 75.0
    ],
    "seasonal_stds": [
      20.0, 18.0, 16.0, 14.0, 12.0, 10.0,
      12.0, 14.0, 16.0, 18.0, 20.0, 22.0
    ],
    "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "ar_coefficients": [
      [0.7], [0.7], [0.7], [0.7], [0.7], [0.7],
      [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]
    ]
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 70.0, "std_dev": 20.0},
    {"season_id": 1, "type": "normal", "mean": 65.0, "std_dev": 18.0},
    ...
  ]
}
```

**Example**: PAR(2) model with varying orders

```json
{
  "temporal_model": {
    "num_seasons": 4,
    "seasonal_means": [100.0, 90.0, 80.0, 95.0],
    "seasonal_stds": [20.0, 18.0, 15.0, 22.0],
    "ar_orders": [2, 2, 1, 1],
    "ar_coefficients": [
      [0.6, 0.2],  // Season 0: PAR(2)
      [0.5, 0.3],  // Season 1: PAR(2)
      [0.7],       // Season 2: PAR(1)
      [0.65]       // Season 3: PAR(1)
    ]
  }
}
```

---

## Marginal Distributions

Marginal distributions define the innovation distribution for each season.

### Normal Distribution

```json
{
  "season_id": <integer>,
  "type": "normal",
  "mean": <float>,
  "std_dev": <float>
}
```

**Parameters**:
- `mean`: μ - Location parameter
- `std_dev`: σ > 0 - Scale parameter

**Distribution**: `η ~ N(μ, σ²)`

### Three-Parameter Log-Normal Distribution

```json
{
  "season_id": <integer>,
  "type": "lognormal3",
  "gamma": <float>,
  "mu": <float>,
  "sigma": <float>
}
```

**Parameters**:
- `gamma`: γ - Location shift (lower bound)
- `mu`: μ - Log-scale location
- `sigma`: σ > 0 - Log-scale scale

**Distribution**: `X = γ + Y` where `Y ~ LogNormal(μ, σ)`

**Properties**:
- Support: `x ∈ [γ, ∞)`
- Mean: `γ + exp(μ + σ²/2)`
- Variance: `exp(2μ + σ²)(exp(σ²) - 1)`

---

## Complete UncertaintySpecification

### Structure

```json
{
  "uncertainty_type": "inflow" | "load",
  "entity_id": <integer>,
  "temporal_model": { ... },
  "seasonal_distributions": [ ... ]
}
```

### Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `uncertainty_type` | string | Type of uncertain entity: "inflow" or "load" | Yes |
| `entity_id` | integer | Entity identifier (hydro_id for inflows, bus_id for loads) | Yes |
| `temporal_model` | object | Temporal model specification | Yes |
| `seasonal_distributions` | array | Marginal distributions for each season | Yes |

---

## Migration Guide

### From v1 to v2

#### Independent Model Migration

**Old Format (v1)**:
```json
{
  "temporal_model": {
    "type": "independent"
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 40.0, "std_dev": 10.0},
    {"season_id": 1, "type": "normal", "mean": 45.0, "std_dev": 12.0}
  ]
}
```

**New Format (v2)**:
```json
{
  "temporal_model": {
    "num_seasons": 2,
    "seasonal_means": [40.0, 45.0],
    "seasonal_stds": [10.0, 12.0],
    "ar_orders": [0, 0],
    "ar_coefficients": [[], []]
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 40.0, "std_dev": 10.0},
    {"season_id": 1, "type": "normal", "mean": 45.0, "std_dev": 12.0}
  ]
}
```

**Migration Steps**:
1. Extract `num_seasons` from length of `seasonal_distributions`
2. Extract `seasonal_means` and `seasonal_stds` from distributions
3. Set `ar_orders` to all zeros: `[0, 0, ...]`
4. Set `ar_coefficients` to all empty arrays: `[[], [], ...]`

#### PAR Model Migration

**Old Format (v1)**:
```json
{
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 2,
    "ar_orders": [1, 1],
    "ar_coefficients": [[0.7], [0.6]],
    "seasonal_means": [100.0, 90.0],
    "seasonal_stds": [20.0, 18.0]
  }
}
```

**New Format (v2)**:
```json
{
  "temporal_model": {
    "num_seasons": 2,
    "seasonal_means": [100.0, 90.0],
    "seasonal_stds": [20.0, 18.0],
    "ar_orders": [1, 1],
    "ar_coefficients": [[0.7], [0.6]]
  }
}
```

**Migration Steps**:
1. Remove `"type": "periodic_ar"` field
2. Keep all other fields unchanged

### Backward Compatibility

**The v1 format is still supported** through a backward compatibility layer:
- Old JSON files will continue to work
- No immediate migration required
- Both formats can coexist in the same codebase
- Deprecation warnings will guide eventual migration

---

## Validation Rules

### Structural Validation

1. **Array lengths match**: All seasonal arrays must have `length = num_seasons`
2. **AR coefficient lengths**: For each season `s`, `ar_coefficients[s].length == ar_orders[s]`
3. **Positive std devs**: All values in `seasonal_stds` must be > 0
4. **Non-negative AR orders**: All values in `ar_orders` must be ≥ 0

### Semantic Validation

1. **Entity ID exists**: `entity_id` must correspond to existing hydro or bus
2. **Distribution consistency**: `seasonal_distributions` parameters should match `temporal_model` parameters
3. **AR stability**: For PAR models, coefficients should satisfy stationarity conditions
4. **Season alignment**: Number of seasons must be consistent across all entities

### Example Validation Errors

```json
// ❌ ERROR: Array length mismatch
{
  "num_seasons": 2,
  "seasonal_means": [100.0, 90.0],
  "seasonal_stds": [20.0],  // Wrong: length 1 ≠ num_seasons 2
  "ar_orders": [0, 0],
  "ar_coefficients": [[], []]
}

// ❌ ERROR: AR coefficient length mismatch
{
  "num_seasons": 2,
  "ar_orders": [1, 1],
  "ar_coefficients": [[0.7], [0.6, 0.2]]  // Wrong: second has 2 coeffs but order is 1
}

// ❌ ERROR: Negative std dev
{
  "seasonal_stds": [20.0, -5.0]  // Wrong: std dev must be positive
}
```

---

## Examples by Use Case

### Single-Season System

```json
{
  "uncertainty_type": "load",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 1,
    "seasonal_means": [50.0],
    "seasonal_stds": [10.0],
    "ar_orders": [0],
    "ar_coefficients": [[]]
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 50.0, "std_dev": 10.0}
  ]
}
```

### Multi-Season with Varying Patterns

Wet season (high flow, low variance) vs. dry season (low flow, high variance):

```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 2,
    "seasonal_means": [200.0, 50.0],
    "seasonal_stds": [30.0, 15.0],
    "ar_orders": [1, 1],
    "ar_coefficients": [[0.8], [0.5]]
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 200.0, "std_dev": 30.0},
    {"season_id": 1, "type": "normal", "mean": 50.0, "std_dev": 15.0}
  ]
}
```

### Log-Normal Inflows

For strictly positive quantities like reservoir inflows:

```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [150.0, 145.0, 140.0, ...],
    "seasonal_stds": [25.0, 23.0, 22.0, ...],
    "ar_orders": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ar_coefficients": [[], [], [], [], [], [], [], [], [], [], [], []]
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "lognormal3", "gamma": 0.0, "mu": 5.0, "sigma": 0.15},
    {"season_id": 1, "type": "lognormal3", "gamma": 0.0, "mu": 4.97, "sigma": 0.15},
    ...
  ]
}
```

---

## Best Practices

### 1. Consistency Between Parameters

Ensure `temporal_model` parameters align with `seasonal_distributions`:

```json
// ✅ GOOD: Means and std devs match between temporal_model and distributions
{
  "temporal_model": {
    "seasonal_means": [100.0],
    "seasonal_stds": [20.0]
  },
  "seasonal_distributions": [
    {"type": "normal", "mean": 100.0, "std_dev": 20.0}
  ]
}

// ⚠️  INCONSISTENT: Different values (will use temporal_model values)
{
  "temporal_model": {
    "seasonal_means": [100.0],
    "seasonal_stds": [20.0]
  },
  "seasonal_distributions": [
    {"type": "normal", "mean": 95.0, "std_dev": 18.0}  // Different!
  ]
}
```

### 2. AR Coefficient Selection

For stable PAR models:
- Sum of coefficients should generally be < 1.0
- For PAR(1): |φ₁| < 1
- For PAR(2): |φ₂| < 1 and φ₁ + φ₂ < 1

```json
// ✅ GOOD: Stable PAR(1) with φ = 0.7
{
  "ar_orders": [1],
  "ar_coefficients": [[0.7]]
}

// ⚠️  POTENTIALLY UNSTABLE: φ = 1.2 > 1
{
  "ar_orders": [1],
  "ar_coefficients": [[1.2]]
}
```

### 3. Season Count Selection

Choose `num_seasons` based on your planning horizon:
- **Monthly** planning: `num_seasons = 12`
- **Quarterly** planning: `num_seasons = 4`
- **Biannual** (wet/dry): `num_seasons = 2`
- **Annual** (single period): `num_seasons = 1`

### 4. Distribution Selection

- **Normal**: General purpose, symmetric
- **LogNormal3**: Strictly positive with right skew (good for inflows)

```json
// Inflows: Use LogNormal3 for physical realism
{
  "uncertainty_type": "inflow",
  "seasonal_distributions": [
    {"type": "lognormal3", "gamma": 0.0, "mu": 5.0, "sigma": 0.2}
  ]
}

// Loads: Normal often appropriate
{
  "uncertainty_type": "load",
  "seasonal_distributions": [
    {"type": "normal", "mean": 100.0, "std_dev": 15.0}
  ]
}
```

---

## Troubleshooting

### Common Issues

#### Issue: "Array length mismatch"

**Error**: All arrays in temporal_model must have same length

**Solution**: Check that `seasonal_means`, `seasonal_stds`, `ar_orders`, `ar_coefficients`, and `seasonal_distributions` all have length = `num_seasons`

#### Issue: "AR coefficient length incorrect"

**Error**: ar_coefficients[i] length doesn't match ar_orders[i]

**Solution**: For season `i` with `ar_orders[i] = p`, ensure `ar_coefficients[i]` has exactly `p` elements

#### Issue: "Negative standard deviation"

**Error**: Standard deviations must be positive

**Solution**: All values in `seasonal_stds` and all `std_dev` in distributions must be > 0

#### Issue: "Cannot find entity"

**Error**: entity_id doesn't correspond to existing hydro/bus

**Solution**: Verify that:
- For `"uncertainty_type": "inflow"`, `entity_id` matches a hydro ID
- For `"uncertainty_type": "load"`, `entity_id` matches a bus ID

---

## Technical Notes

### Implementation Details

1. **Inverse CDF Transform**: The system uses proper probability integral transform via inverse CDF for all distributions, ensuring correct copula properties

2. **AR Dynamics**: PAR models are implemented using observation-space formulation:
   ```
   Y_t = μ_s + σ_s · η_t + Σ φ_i · (Y_{t-i} - μ_{s-i})
   ```

3. **Independent Models**: Treated as special case of PAR with all orders = 0, enabling unified code paths

### Performance Considerations

- **Memory**: Unified format uses consistent memory layout regardless of model type
- **Parsing**: Backward compatibility adds minimal overhead (one enum variant check)
- **Generation**: Independent and PAR models use same scenario generation pipeline

---

## References

- **Refactoring Plan**: See `SCENARIO_GENERATION_REFACTORING_PLAN.md`
- **Implementation Tickets**: See `docs/refactoring-tickets.md`
- **Migration Guide**: See section above
- **API Documentation**: Run `cargo doc --open`

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-02  
**Authors**: Powers-RS Development Team
