# Migration Guide: From Stationary AR to Seasonic AR (PAR)

## Table of Contents

1. [Overview](#overview)
2. [Should You Migrate?](#should-you-migrate)
3. [Comparison: AR vs PAR](#comparison-ar-vs-par)
4. [Step-by-Step Migration](#step-by-step-migration)
5. [Backward Compatibility](#backward-compatibility)
6. [Common Migration Issues](#common-migration-issues)

---

## Overview

This guide helps you migrate existing stationary Autoregressive (AR) models to Seasonic Autoregressive (PAR) models in POWE.RS. PAR models provide more accurate uncertainty representation for systems with seasonal patterns.

### What Changes?

- **JSON Configuration**: Update `graph.json` and `recourse.json`
- **State Variables**: Change from `"storage"` to `"storage_and_inflow"`
- **Parameters**: Replace single μ, σ, φ with seasonal arrays μₘ, σₘ, φₖₘ
- **Residual Distribution**: New `residual_distribution` field (replaces `marginal_distribution` for PAR)

### What Stays the Same?

- **Algorithm**: SDDP logic unchanged
- **System**: `system.json` and `config.json` unchanged
- **Performance**: Similar runtime (<10% overhead for typical systems)
- **Results**: Should be similar if seasonal variation is small

---

## Should You Migrate?

### Migrate to PAR If:

✅ **Seasonal inflow variation >30%**: Example: Wet season 120 m³/s, Dry season 40 m³/s (3x difference)

✅ **Long planning horizon >12 months**: PAR captures annual cycles

✅ **Historical data shows seasonality**: Plot historical inflows - if there's a clear annual pattern, use PAR

✅ **Critical operational decisions**: Reservoir management where timing (wet vs dry) affects strategy

### Stay with Stationary AR If:

⚠️ **Seasonal variation <20%**: Minimal benefit from PAR complexity

⚠️ **Short planning horizon <6 months**: Seasonal effects don't manifest fully

⚠️ **Limited data <3 years**: Can't reliably estimate 12 seasonal parameters

⚠️ **Preliminary studies**: Stationary AR sufficient for what-if scenarios

### Example Decision

**System**: Brazilian hydro, 60-month planning horizon, historical data shows:
- January: Mean = 150 m³/s
- July: Mean = 45 m³/s
- Variation: 150/45 = 3.3x (233% difference)

**Decision**: ✅ **Migrate to PAR** - strong seasonality justifies the added complexity.

---

## Comparison: AR vs PAR

### Stationary AR

```
Xₜ = φ₁·Xₜ₋₁ + φ₂·Xₜ₋₂ + εₜ

where:
  εₜ ~ F (fixed distribution, e.g., Normal(μ, σ²))
  φ coefficients are constant
```

**Assumptions**:
- Mean μ is constant over time
- Standard deviation σ is constant
- AR coefficients φ are constant

**Suitable for**: Stationary processes, short horizons, low seasonal variation

### Seasonic AR (PAR)

```
Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + aₜ]

where:
  m = t mod num_seasons
  μₘ, σₘ, φₖₘ vary by season
  aₜ ~ residual_distribution
```

**Assumptions**:
- Mean μₘ varies by season m
- Standard deviation σₘ varies by season
- AR coefficients φₖₘ can vary by season

**Suitable for**: Seasonal processes, long horizons, high seasonal variation

### Side-by-Side Comparison

| Aspect | Stationary AR | PAR |
|--------|---------------|-----|
| **Parameters** | 3-4 global | 3T+1 (T = num_seasons) |
| **Seasonality** | ❌ None | ✅ Captured |
| **Persistence** | ✅ Captured | ✅ Captured |
| **JSON Complexity** | Low | Medium |
| **Estimation Complexity** | Low | Medium |
| **Data Requirements** | 3-5 years | 5-10 years |
| **State Variables** | +p per hydro | +p per hydro (same) |
| **Runtime Overhead** | Baseline | +5-10% |
| **Memory Overhead** | Baseline | +5-10% |
| **Realism (seasonal systems)** | Fair | Excellent |

---

## Step-by-Step Migration

### Step 1: Assess Current Configuration

Check your current `recourse.json`:

```json
// OLD: Stationary AR(1)
{
  "noise_models": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,
      "season_id": 0,
      "marginal_distribution": {
        "type": "normal",
        "mean": 80.0,
        "std_dev": 20.0
      },
      "temporal_model": {
        "type": "autoregressive",
        "lag_order": 1,
        "coefficients": [0.7]
      }
    }
  ]
}
```

### Step 2: Decide on Season

Choose num_seasons based on data granularity:

- **Monthly data** → `num_seasons = 12`
- **Quarterly data** → `num_seasons = 4`
- **Bi-annual data** → `num_seasons = 2`

**Recommendation**: Use `num_seasons = 12` for annual planning.

### Step 3: Estimate Seasonal Parameters

**Option A: Use Historical Data (Recommended)**

```bash
# Estimate from CSV
powers estimate-par historical_inflows.csv \
  --periods 12 \
  --order 1 \
  --output par_params.json

# This generates seasonal_means, seasonal_stds, ar_coefficients
```

**Option B: Manual Estimation**

If no data, estimate based on domain knowledge:

1. **Seasonal means**: Sketch out expected monthly averages
2. **Seasonal std devs**: 20-30% of mean for high variability, 10-15% for low
3. **AR coefficients**: Start with 0.7 for all periods, tune later

Example manual estimation:

```
Season: 0 (Jan) → μ=120, σ=30, φ=0.75
Season: 1 (Feb) → μ=110, σ=28, φ=0.72
...
Season: 6 (Jul) → μ=40, σ=12, φ=0.55
...
Season: 11 (Dec) → μ=130, σ=32, φ=0.78
```

### Step 4: Update `graph.json`

Add `season_id` to all nodes:

```json
// OLD: No season_id
{
  "nodes": [
    {
      "id": 0,
      "stage_id": 0,
      "inflow_stochastic_process": "naive",
      "state_variables": "storage",
      ...
    }
  ]
}

// NEW: With season_id and PAR
{
  "nodes": [
    {
      "id": 0,
      "stage_id": 0,
      "season_id": 0,  // ADD THIS
      "inflow_stochastic_process": "par",  // CHANGE THIS
      "state_variables": "storage_and_inflow",  // CHANGE THIS
      ...
    },
    {
      "id": 1,
      "stage_id": 1,
      "season_id": 1,  // Increments by 1
      "inflow_stochastic_process": "par",
      "state_variables": "storage_and_inflow",
      ...
    },
    // ... continue pattern ...
    {
      "id": 12,
      "stage_id": 12,
      "season_id": 0,  // Wraps back to 0 after num_seasons-1
      "inflow_stochastic_process": "par",
      "state_variables": "storage_and_inflow",
      ...
    }
  ]
}
```

**Critical**: `season_id` must cycle: 0, 1, 2, ..., num_seasons-1, 0, 1, 2, ...

### Step 5: Update `recourse.json`

Replace stationary AR with PAR:

```json
// OLD: Stationary AR
{
  "initial_condition": {
    "storage": [{"hydro_id": 0, "value": 150.0}],
    "inflow": [{"hydro_id": 0, "lag": 1, "value": 80.0}]  // Keep this
  },
  "noise_models": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,
      "season_id": 0,
      "marginal_distribution": {
        "type": "normal",
        "mean": 80.0,
        "std_dev": 20.0
      },
      "temporal_model": {
        "type": "autoregressive",
        "lag_order": 1,
        "coefficients": [0.7]
      }
    }
  ]
}

// NEW: PAR
{
  "initial_condition": {
    "storage": [{"hydro_id": 0, "value": 150.0}],
    "inflow": [{"hydro_id": 0, "lag": 1, "value": 85.0}]  // Same structure
  },
  "noise_models": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,
      "season_id": 0,  // Applied to all nodes with this season_id
      "marginal_distribution": {  // Keep but ignored for PAR
        "type": "normal",
        "mean": 0.0,
        "std_dev": 1.0
      },
      "temporal_model": {
        "type": "periodic_ar",  // CHANGED
        "num_seasons": 12,  // NEW
        "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // NEW
        "ar_coefficients": [  // EXPANDED from single [0.7]
          [0.75], [0.72], [0.68], [0.65],
          [0.60], [0.58], [0.55], [0.57],
          [0.60], [0.63], [0.68], [0.72]
        ],
        "seasonal_means": [  // NEW: replaces mean in marginal_distribution
          120.0, 110.0, 95.0, 80.0,
          60.0, 45.0, 35.0, 40.0,
          55.0, 75.0, 95.0, 115.0
        ],
        "seasonal_stds": [  // NEW: replaces std_dev in marginal_distribution
          30.0, 28.0, 25.0, 22.0,
          18.0, 15.0, 12.0, 15.0,
          18.0, 22.0, 26.0, 29.0
        ]
      },
      "residual_distribution": {  // NEW FIELD
        "type": "lognormal3",
        "gamma": 1.0,
        "mu": 0.0,
        "sigma": 0.6
      }
    }
  ]
}
```

### Step 6: Validate Configuration

Run validation checks:

```bash
# Check JSON syntax
jq . graph.json > /dev/null && echo "✓ graph.json valid"
jq . recourse.json > /dev/null && echo "✓ recourse.json valid"

# Verify season_id pattern
jq '.nodes[].season_id' graph.json  # Should print 0,1,2,...,11,0,1,2,...
```

### Step 7: Test Run

```bash
# Run with PAR configuration
cargo run --release -- examples/my-system

# Compare results with stationary AR baseline
# - Expected cost should be similar if seasonality is moderate
# - Policy should show seasonal patterns (store water in wet, use in dry)
```

### Step 8: Validate Results

1. **Check convergence**: Lower bound should stabilize
2. **Inspect policy**: Review decisions by season
3. **Plot scenarios**: Verify inflow distributions match expectations
4. **Compare costs**: PAR vs stationary AR (should be within 5-10% if both are valid)

---

## Backward Compatibility

### What's Preserved?

✅ **Old stationary AR configurations still work** - no breaking changes

✅ **Independent noise models unchanged**

✅ **Existing examples run as before**

### Mixing Model Types

You can mix PAR and stationary AR in the same system:

```json
{
  "noise_models": [
    {
      "entity_id": 0,  // Major hydro: Use PAR
      "temporal_model": {"type": "periodic_ar", ...}
    },
    {
      "entity_id": 1,  // Minor hydro: Keep stationary AR
      "temporal_model": {"type": "autoregressive", ...}
    },
    {
      "entity_id": 2,  // Load: Independent
      "temporal_model": {"type": "independent"}
    }
  ]
}
```

This is useful for large systems where full PAR for all entities would be overkill.

### Gradual Migration

**Phase 1**: Migrate one critical hydro to PAR

**Phase 2**: Validate results, compare with baseline

**Phase 3**: Migrate remaining hydros incrementally

**Phase 4**: Finalize all configurations

No need to migrate everything at once!

---

## Common Migration Issues

### Issue 1: Missing `season_id`

**Error**: `season_id field is required when inflow_stochastic_process="par"`

**Solution**: Add `season_id` to all graph nodes

```json
// Before
{"id": 0, "stage_id": 0, ...}

// After  
{"id": 0, "stage_id": 0, "season_id": 0, ...}
```

### Issue 2: Wrong State Variables

**Error**: `PAR requires state_variables="storage_and_inflow"`

**Solution**: Change state variables in graph nodes

```json
// Before
"state_variables": "storage"

// After
"state_variables": "storage_and_inflow"
```

### Issue 3: Array Length Mismatch

**Error**: `seasonal_means has length 10, but num_seasons=12`

**Solution**: Ensure all seasonal arrays have exactly `num_seasons` elements

```json
// Wrong
"num_seasons": 12,
"seasonal_means": [100, 90, 80, ...]  // Only 10 values!

// Correct
"num_seasons": 12,
"seasonal_means": [100, 95, 90, 85, 75, 65, 50, 55, 65, 75, 90, 98]  // 12 values
```

### Issue 4: Missing `residual_distribution`

**Error**: `PAR requires residual_distribution field`

**Solution**: Add `residual_distribution` to noise model

```json
{
  "temporal_model": {"type": "periodic_ar", ...},
  "residual_distribution": {  // ADD THIS
    "type": "normal",
    "mean": 0.0,
    "std_dev": 1.0
  }
}
```

### Issue 5: Non-Cycling `season_id`

**Error**: `season_id sequence invalid: expected 0 after 11, got 12`

**Solution**: Ensure `season_id` wraps around

```json
// Wrong
{"stage_id": 11, "season_id": 11, ...},
{"stage_id": 12, "season_id": 12, ...}  // Should be 0!

// Correct
{"stage_id": 11, "season_id": 11, ...},
{"stage_id": 12, "season_id": 0, ...}   // Wraps to 0
```

### Issue 6: AR(2) Without Lag-2 Initial Condition

**Error**: `AR(2) requires lag=1 and lag=2 initial values`

**Solution**: Provide both lags in initial condition

```json
// Wrong
"inflow": [
  {"hydro_id": 0, "lag": 1, "value": 85.0}
]

// Correct
"inflow": [
  {"hydro_id": 0, "lag": 1, "value": 85.0},
  {"hydro_id": 0, "lag": 2, "value": 80.0}  // ADD lag-2
]
```

---

## Migration Checklist

Before finalizing migration, verify:

- [ ] **Graph Nodes**:
  - [ ] All nodes have `season_id` field
  - [ ] `season_id` cycles correctly (0 to num_seasons-1)
  - [ ] `inflow_stochastic_process = "par"`
  - [ ] `state_variables = "storage_and_inflow"`

- [ ] **Recourse Configuration**:
  - [ ] `temporal_model.type = "periodic_ar"`
  - [ ] `num_seasons` is set correctly (12 for monthly)
  - [ ] All seasonal arrays have length = `num_seasons`
  - [ ] `ar_coefficients[m].len() == ar_orders[m]`
  - [ ] `residual_distribution` is defined
  - [ ] Initial condition includes required lags

- [ ] **Validation**:
  - [ ] JSON syntax is valid (use `jq` or online validator)
  - [ ] Test run completes without errors
  - [ ] Generated scenarios look reasonable (plot distribution)
  - [ ] Results are comparable to stationary AR baseline

- [ ] **Documentation**:
  - [ ] Configuration is documented for future reference
  - [ ] Parameter sources noted (historical data vs manual)
  - [ ] Migration date and author recorded

---

## Example Migration

### Before (Stationary AR)

**graph.json**:
```json
{
  "nodes": [
    {"id": 0, "stage_id": 0, "inflow_stochastic_process": "naive", "state_variables": "storage", ...}
  ]
}
```

**recourse.json**:
```json
{
  "noise_models": [{
    "entity_id": 0,
    "temporal_model": {
      "type": "autoregressive",
      "lag_order": 1,
      "coefficients": [0.7]
    },
    "marginal_distribution": {"type": "normal", "mean": 80.0, "std_dev": 20.0}
  }]
}
```

### After (PAR)

**graph.json**:
```json
{
  "nodes": [
    {"id": 0, "stage_id": 0, "season_id": 0, "inflow_stochastic_process": "par", "state_variables": "storage_and_inflow", ...},
    {"id": 1, "stage_id": 1, "season_id": 1, "inflow_stochastic_process": "par", "state_variables": "storage_and_inflow", ...},
    // ... (10 more nodes with season_id 2-11)
    {"id": 12, "stage_id": 12, "season_id": 0, "inflow_stochastic_process": "par", "state_variables": "storage_and_inflow", ...}
  ]
}
```

**recourse.json**:
```json
{
  "noise_models": [{
    "entity_id": 0,
    "season_id": 0,
    "temporal_model": {
      "type": "periodic_ar",
      "num_seasons": 12,
      "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      "ar_coefficients": [[0.75], [0.72], [0.68], [0.65], [0.60], [0.58], [0.55], [0.57], [0.60], [0.63], [0.68], [0.72]],
      "seasonal_means": [120, 110, 95, 80, 60, 45, 35, 40, 55, 75, 95, 115],
      "seasonal_stds": [30, 28, 25, 22, 18, 15, 12, 15, 18, 22, 26, 29]
    },
    "marginal_distribution": {"type": "normal", "mean": 0.0, "std_dev": 1.0},
    "residual_distribution": {"type": "lognormal3", "gamma": 1.0, "mu": 0.0, "sigma": 0.6}
  }]
}
```

---

## Getting Help

- **Guide**: See `docs/guides/PAR-MODEL-GUIDE.md` for detailed PAR documentation
- **Examples**: Check `examples/06-par-model/01-simple-par1/` for working configuration
- **Schema**: Review `schemas/recourse.schema.json` for field specifications
- **Troubleshooting**: See `docs/guides/TROUBLESHOOTING.md`

**Still stuck?** Open an issue on GitHub with your configuration files and error messages.

---

**Success!** You've migrated to PAR. Your model now captures seasonal patterns for more realistic uncertainty representation. 🎉
