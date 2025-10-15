# PAR Model Examples

This directory contains example configurations demonstrating Periodic Autoregressive (PAR) model usage for stochastic inflow modeling in hydrothermal dispatch optimization.

## Overview

PAR models capture both **seasonal patterns** (e.g., wet/dry seasons) and **temporal persistence** (e.g., high inflows tend to follow high inflows) in hydrological time series. They are essential for realistic long-term planning.

### What is PAR?

A PAR(p) model with period T is defined as:

```
Zₜ = μₘ + σₘ · [Σₖ φₖₘ · aₜ₋ₖ + aₜ]

where:
  t = time step (stage index)
  m = t mod T (seasonal period, mapped from season_id in graph nodes)
  aₜ = standardized residual (innovation)
  μₘ = seasonal mean for period m
  σₘ = seasonal standard deviation for period m
  φₖₘ = k-th AR coefficient for period m
  p = AR order (can vary by period!)
```

**Key Properties:**
- **Seasonality**: μₘ and σₘ change with seasons (e.g., wet/dry periods)
- **Persistence**: φ coefficients create temporal correlation
- **Flexibility**: AR order can vary by season (e.g., AR(2) in wet season, AR(1) in dry)
- **Stationarity**: Within each season, process is stationary

## Examples

### 01-simple-par1: Monthly PAR(1) for Single Reservoir

**Learning Focus**: Basic PAR configuration, period mapping, state variables

- **System**: 1 hydro + 1 thermal
- **Period**: 12 (monthly)
- **AR Order**: PAR(1) uniform across all months
- **Seasonality**: Wet season (Dec-Feb, 90 m³/s), Dry season (Jun-Aug, 50 m³/s)
- **AR Coefficient**: φ = 0.7 (moderate persistence)
- **Residuals**: Standard Normal N(0,1)

**Key Concepts:**
- `season_id` in graph nodes maps to PAR periods (0-11)
- `state_variables: "storage_and_inflow"` required for PAR
- `inflow_stochastic_process: "par"` activates PAR generation
- Initial condition includes past inflow lags

**Run**: 
```bash
cargo run --release -- examples/06-par-model/01-simple-par1
```

**Expected Runtime**: ~10 seconds

---

### 02-cascade-par: Multi-Reservoir with Correlation (TODO)

**Learning Focus**: Correlation blocks, spatial dependencies

- **System**: 3 hydros in cascade + 2 thermals
- **Correlation**: 0.8 between adjacent reservoirs, 0.5 for distant
- **Period**: 12 (monthly)
- **AR Order**: PAR(1)

Demonstrates:
- Correlation blocks in `recourse.json`
- Gaussian copula with Cholesky decomposition
- Realistic upstream/downstream correlation

---

### 03-lognormal-par: PAR with LogNormal3 Residuals (TODO)

**Learning Focus**: Non-negative guarantee, marginal transformations

- **System**: 1 hydro + 1 thermal
- **Period**: 12
- **Residuals**: LogNormal3(γ=1.0, μ=0.0, σ=0.6)
- **Guarantee**: All inflows ≥ γ = 1.0 m³/s

Demonstrates:
- `residual_distribution`: LogNormal3 vs Normal
- Non-negativity for physical realism
- Parameter fitting from historical data

---

### 04-mixed-models: Hybrid PAR + Independent (TODO)

**Learning Focus**: Multiple noise models, model selection

- **System**: 2 hydros + 1 thermal
- **Hydro 1**: PAR(1) - seasonal inflows
- **Hydro 2**: Independent Normal - constant mean

Demonstrates:
- Different models for different entities
- When to use PAR vs Independent
- Configuration flexibility

---

### 05-quarterly-par: 4-Period PAR (TODO)

**Learning Focus**: Period flexibility, non-monthly data

- **System**: 1 hydro + 1 thermal
- **Period**: 4 (quarterly)
- **AR Order**: Varying [2, 2, 1, 1] (higher order in wet quarters)

Demonstrates:
- Quarterly planning horizons
- Varying AR orders by season
- Period choice trade-offs

---

## Configuration Guide

### Graph Nodes (`graph.json`)

```json
{
  "id": 0,
  "stage_id": 0,
  "season_id": 0,  // Maps to PAR period index!
  "inflow_stochastic_process": "par",  // Activate PAR
  "state_variables": "storage_and_inflow",  // Required for PAR
  ...
}
```

**Critical**: `season_id` determines which PAR parameters are used:
- `season_id = 0` → uses `seasonal_means[0]`, `seasonal_stds[0]`, `ar_coefficients[0]`
- Must be contiguous 0..period-1 across all stages

### Recourse Specification (`recourse.json`)

```json
{
  "initial_condition": {
    "storage": [...],
    "inflow": [
      {"hydro_id": 0, "lag": 1, "value": 85.0}  // Past inflow for AR
    ]
  },
  "noise_models": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,  // Must match hydro_id
      "season_id": 0,  // Applies to all nodes with this season_id
      "temporal_model": {
        "type": "periodic_ar",
        "period": 12,  // Number of seasons
        "ar_orders": [1, 1, ..., 1],  // AR order per period
        "ar_coefficients": [[0.7], [0.7], ..., [0.7]],  // Coefficients per period
        "seasonal_means": [90.0, 85.0, ..., 90.0],  // Mean per period
        "seasonal_stds": [25.0, 23.0, ..., 25.0]   // Std dev per period
      },
      "residual_distribution": {
        "type": "normal",  // or "lognormal3"
        "mean": 0.0,
        "std_dev": 1.0
      }
    }
  ]
}
```

### Period Configuration

| Period | Use Case | Stages per Period | Notes |
|--------|----------|------------------|-------|
| 12 | Monthly | 30-365 days | Standard for annual planning |
| 4 | Quarterly | 90 days | Medium-term planning |
| 52 | Weekly | 7 days | Short-term or detailed studies |
| 2 | Bi-annual | 6 months | Simple wet/dry distinction |

**Rule**: `season_id` in graph must cycle through 0..period-1

### AR Coefficient Guidelines

| φ Value | Interpretation | Effect |
|---------|---------------|--------|
| 0.0 | No persistence | White noise (seasonality only) |
| 0.3-0.5 | Weak persistence | Short memory (1-2 periods) |
| 0.6-0.8 | Moderate persistence | Medium memory (3-5 periods) |
| 0.9+ | Strong persistence | Long memory (>10 periods) |

**Stationarity**: For AR(1), require |φ| < 1. For AR(p), spectral radius < 1.

### Residual Distributions

| Distribution | Use Case | Parameters |
|--------------|----------|------------|
| Normal | Balanced pos/neg, can go negative | μ=0, σ=1 (standard) |
| LogNormal3 | Non-negative guarantee (inflows) | γ≥0 (min), μ, σ |

**Recommendation**: Use LogNormal3 for inflows to guarantee physical realism (no negative flows).

## Common Pitfalls

1. **Mismatched season_id**: Ensure graph nodes cycle through 0..period-1
2. **Wrong state_variables**: Must use `"storage_and_inflow"` for PAR
3. **Missing initial lags**: Include past inflows in `initial_condition.inflow`
4. **Non-stationary coefficients**: Check AR stationarity conditions
5. **Period/array mismatch**: All seasonal arrays must have length = period

## Performance Considerations

**State Space Size**: PAR adds `max(ar_orders)` state variables per hydro:
- AR(1): +1 state per hydro
- AR(2): +2 states per hydro
- Impact: Polynomial increase in SDDP cut storage and solve time

**Recommendations**:
- Use lowest AR order that captures persistence (usually p=1 or p=2)
- For large systems (>10 hydros), consider mixed models (PAR for key hydros, Independent for minor ones)
- Profile memory usage for systems with >5 hydros and AR(2)+

## Parameter Estimation

Use the CLI tool to estimate PAR parameters from historical data:

```bash
# Estimate monthly PAR(1) from CSV
powers estimate-par historical_inflows.csv \
  --periods 12 \
  --order 1 \
  --output my_par_params.json

# Output is ready to paste into recourse.json
```

See `docs/guides/PARAMETER_ESTIMATION.md` for details.

## References

- **Algorithm**: See `docs/algorithm/PAR_MODEL_SUPPORT.md`
- **Schema**: `schemas/recourse.schema.json` (search for `periodic_ar`)
- **Implementation**: `src/par_generator.rs`, `src/seasonal_params.rs`
- **Tests**: `tests/test_par_*.rs`

## Questions?

- **What period should I use?** Match your planning horizon and data granularity. Monthly (12) is most common.
- **What AR order?** Start with AR(1). Use AR(2) only if data shows strong 2-step autocorrelation.
- **Normal vs LogNormal3?** LogNormal3 for inflows (non-negative), Normal for loads (can be negative in deficit scenarios).
- **How to handle correlation?** Use correlation blocks in `recourse.json` (see Example 02).

---

**Next Steps**: After mastering these examples, see `examples/05-large-scale-brazilian` for a production-scale PAR application with 150+ hydros.
