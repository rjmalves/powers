# PAR Model Guide: Seasonic Autoregressive Processes for Hydrothermal Optimization

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [When to Use PAR](#when-to-use-par)
4. [Configuration Guide](#configuration-guide)
5. [Parameter Estimation](#parameter-estimation)
6. [Examples and Best Practices](#examples-and-best-practices)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

Seasonic Autoregressive (PAR) models are essential for realistic long-term hydrothermal dispatch optimization. Unlike stationary AR models that assume constant statistical properties, PAR models capture the **seasonal variation** inherent in hydrological processes.

### Why PAR?

Real-world inflows exhibit two key characteristics:

1. **Seasonality**: Mean inflows vary significantly by season (e.g., wet vs dry periods)
2. **Temporal Persistence**: High (or low) inflows tend to persist across multiple periods

Stationary AR models can capture persistence but fail to represent seasonality. PAR models capture both, providing more accurate uncertainty representation for long-term planning.

### CEPEL Methodology

POWE.RS implements the PAR methodology developed by CEPEL (Electric Energy Research Center, Brazil), the gold standard for hydrothermal optimization in systems with strong seasonal patterns. The Brazilian electrical system, with 150+ hydro plants and pronounced wet/dry seasons, relies on PAR models for operational planning.

---

## Mathematical Background

### The PAR(p) Equation

A Seasonic Autoregressive model of order p with num_seasons T is defined as:

```
Zₜ = μₘ + σₘ · [∑ₖ₌₁ᵖᵐ φₖₘ · aₜ₋ₖ + aₜ]

where:
  t = time step (stage index in SDDP)
  m = t mod T (seasonal num_seasons index, 0 ≤ m < T)
  Zₜ = generated value at time t (e.g., inflow in m³/s)
  μₘ = seasonal mean for num_seasons m
  σₘ = seasonal standard deviation for num_seasons m
  φₖₘ = k-th AR coefficient for num_seasons m (can vary by season!)
  pₘ = AR order for num_seasons m (can vary by season!)
  aₜ = transformed residual (innovation) at time t
```

### Components Explained

#### 1. Seasonality (μₘ, σₘ)

The mean and standard deviation change with the seasonal num_seasons:

- **Wet season**: μₘ = 120 m³/s, σₘ = 30 m³/s (high mean, high variability)
- **Dry season**: μₘ = 40 m³/s, σₘ = 10 m³/s (low mean, low variability)

This captures the annual hydrological cycle.

#### 2. Temporal Persistence (φₖₘ)

AR coefficients create temporal correlation:

- **φₖₘ = 0**: No persistence (white noise + seasonality)
- **φₖₘ = 0.7**: Moderate persistence (memory of 3-5 periods)
- **φₖₘ = 0.9**: Strong persistence (long memory, slow transitions)

Example: If current inflow is high (aₜ > 0), and φ = 0.7, next num_seasons's inflow is likely also above average.

#### 3. Innovations (aₜ)

The innovations aₜ represent the unpredictable component after removing seasonality and autocorrelation. They follow a specified distribution:

- **Standard Normal**: N(0, 1) - simplest, allows negative values
- **LogNormal3**: γ + exp(μ + σZ) where Z ~ N(0,1) - guarantees Zₜ ≥ γ (non-negative)

### Multi-Season Dynamics

The PAR equation is applied sequentially:

```
Stage 0 (Jan, m=0): Z₀ = μ₀ + σ₀ · [φ₁₀·a₋₁ + a₀]
Stage 1 (Feb, m=1): Z₁ = μ₁ + σ₁ · [φ₁₁·a₀ + a₁]
Stage 2 (Mar, m=2): Z₂ = μ₂ + σ₂ · [φ₁₂·a₁ + a₂]
...
Stage 12 (Jan, m=0): Z₁₂ = μ₀ + σ₀ · [φ₁₀·a₁₁ + a₁₂]  # Season wraps around
```

Note: The num_seasons index m cycles: 0, 1, 2, ..., T-1, 0, 1, 2, ...

### Stationarity

While parameters change with seasons, within each season the process is stationary. For AR(1), stationarity requires:

```
|φₘ| < 1  for all m
```

For higher orders AR(p), the spectral radius of the companion matrix must be less than 1.

---

## When to Use PAR

### Use PAR When:

✅ **Strong seasonal patterns**: Inflows vary >30% between wet/dry seasons  
✅ **Long planning horizon**: >12 months (captures full annual cycle)  
✅ **Historical data available**: Need 5-10 years of monthly data for estimation  
✅ **Persistence observed**: Autocorrelation at lag-1 is >0.3  
✅ **Critical decisions**: Reservoir management where timing matters

### Use Stationary AR When:

⚠️ **Weak seasonality**: Inflows vary <20% across seasons  
⚠️ **Short planning horizon**: <6 months (seasonal effects minimal)  
⚠️ **Limited data**: <3 years of historical data  
⚠️ **Low persistence**: Autocorrelation <0.2 (nearly independent)

### Use Independent When:

⛔ **No persistence**: Inflows are uncorrelated over time  
⛔ **Very short horizon**: <3 months  
⛔ **Data-limited**: No historical data for fitting  
⛔ **Preliminary studies**: Quick what-if scenarios

### Comparison Table

| Feature | Independent | Stationary AR | PAR |
|---------|-------------|---------------|-----|
| Seasonality | ❌ No | ❌ No | ✅ Yes |
| Persistence | ❌ No | ✅ Yes | ✅ Yes |
| State variables | Storage only | Storage + lags | Storage + lags |
| Parameters/entity | 2 (μ, σ) | 3-4 (μ, σ, φ₁, φ₂) | 3T+1 (μₘ, σₘ, φₖₘ, T) |
| Estimation complexity | Low | Medium | High |
| Computational cost | Low | Medium | Medium |
| Realism (seasonal systems) | Poor | Fair | Excellent |

---

## Configuration Guide

### Overview

PAR configuration involves four JSON files:

1. `config.json` - SDDP algorithm parameters (unchanged)
2. `system.json` - Physical system (unchanged)
3. `graph.json` - **Scenario tree with season mapping**
4. `recourse.json` - **PAR parameters and initial conditions**

### Step 1: Graph Configuration (`graph.json`)

#### Set Season IDs

Each node must have a `season_id` that maps to PAR periods:

```json
{
  "nodes": [
    {
      "id": 0,
      "stage_id": 0,
      "season_id": 0,  // January → num_seasons 0
      "inflow_stochastic_process": "par",
      "state_variables": "storage_and_inflow",
      ...
    },
    {
      "id": 1,
      "stage_id": 1,
      "season_id": 1,  // February → num_seasons 1
      "inflow_stochastic_process": "par",
      "state_variables": "storage_and_inflow",
      ...
    },
    // ... more nodes ...
    {
      "id": 12,
      "stage_id": 12,
      "season_id": 0,  // January again → num_seasons 0 (wraps around)
      "inflow_stochastic_process": "par",
      "state_variables": "storage_and_inflow",
      ...
    }
  ]
}
```

**Critical Requirements**:
- `season_id` must cycle through 0, 1, 2, ..., num_seasons-1, 0, 1, ...
- `inflow_stochastic_process` must be `"par"`
- `state_variables` must be `"storage_and_inflow"` (not `"storage"`)

#### Season Examples

**Monthly (num_seasons = 12)**:
```
Stage:     0    1    2   ...  11   12   13  ...  23   24
Month:    Jan  Feb  Mar  ... Dec  Jan  Feb  ... Dec  Jan
season_id: 0    1    2   ...  11    0    1  ...  11    0
```

**Quarterly (num_seasons = 4)**:
```
Stage:     0    1    2    3    4    5    6    7
Quarter:  Q1   Q2   Q3   Q4   Q1   Q2   Q3   Q4
season_id: 0    1    2    3    0    1    2    3
```

**Weekly (num_seasons = 52)**:
```
Stage:     0    1   ...  51   52   53  ... 103  104
Week:     W1   W2  ... W52   W1   W2  ... W52   W1
season_id: 0    1   ...  51    0    1  ...  51    0
```

### Step 2: Recourse Configuration (`recourse.json`)

#### Initial Conditions

PAR requires past inflow values for AR initialization:

```json
{
  "initial_condition": {
    "storage": [
      {"hydro_id": 0, "value": 150.0},
      {"hydro_id": 1, "value": 100.0}
    ],
    "inflow": [
      {"hydro_id": 0, "lag": 1, "value": 85.0},  // Last month's inflow
      {"hydro_id": 1, "lag": 1, "value": 45.0}
    ]
  }
}
```

**Rules**:
- For AR(1): Provide lag=1 values
- For AR(2): Provide lag=1 and lag=2 values
- Values should be realistic (within seasonal range)
- Missing lags default to seasonal mean (suboptimal)

#### PAR Noise Model

```json
{
  "noise_models": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,  // Must match hydro_id
      "season_id": 0,  // Applied to all nodes with season_id=0
      "marginal_distribution": {
        "type": "normal",
        "mean": 0.0,
        "std_dev": 1.0
      },
      "temporal_model": {
        "type": "periodic_ar",
        "num_seasons": 12,
        "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "ar_coefficients": [
          [0.7], [0.75], [0.8], [0.75],  // Wet season: higher persistence
          [0.65], [0.6], [0.55], [0.6],  // Dry season: lower persistence
          [0.65], [0.7], [0.75], [0.7]   // Transition: medium persistence
        ],
        "seasonal_means": [
          120.0, 110.0, 95.0, 80.0,   // Wet → transition
          60.0, 45.0, 35.0, 40.0,     // Dry season (low)
          55.0, 75.0, 95.0, 115.0     // Transition → wet
        ],
        "seasonal_stds": [
          30.0, 28.0, 25.0, 22.0,     // Wet: high variability
          18.0, 15.0, 12.0, 15.0,     // Dry: low variability
          18.0, 22.0, 26.0, 29.0      // Transition
        ]
      },
      "residual_distribution": {
        "type": "lognormal3",  // Non-negative guarantee
        "gamma": 1.0,  // Minimum inflow = 1.0 m³/s
        "mu": 0.0,
        "sigma": 0.6
      }
    }
  ]
}
```

#### Parameter Arrays

All seasonal arrays must have length = `num_seasons`:

- `ar_orders`: AR order for each num_seasons [p₀, p₁, ..., p_{T-1}]
- `ar_coefficients`: AR coefficients for each num_seasons [[φ coeffs for num_seasons 0], [num_seasons 1], ...]
- `seasonal_means`: Mean for each num_seasons [μ₀, μ₁, ..., μ_{T-1}]
- `seasonal_stds`: Std dev for each num_seasons [σ₀, σ₁, ..., σ_{T-1}]

**Important**: `ar_coefficients[m]` must have length = `ar_orders[m]`

#### Residual Distribution

Two options:

**1. Standard Normal (simplest)**:
```json
"residual_distribution": {
  "type": "normal",
  "mean": 0.0,
  "std_dev": 1.0
}
```
- Pros: Simple, symmetric, standard
- Cons: Can generate negative inflows

**2. LogNormal3 (recommended for inflows)**:
```json
"residual_distribution": {
  "type": "lognormal3",
  "gamma": 1.0,   // Minimum value (safety buffer)
  "mu": 0.0,      // Mean of log(X - gamma)
  "sigma": 0.6    // Std dev of log(X - gamma)
}
```
- Pros: Guarantees Zₜ ≥ gamma (non-negative)
- Cons: Slightly more complex, right-skewed

**Recommendation**: Use LogNormal3 with gamma ≥ 1.0 for physical realism.

---

## Parameter Estimation

### Using the CLI Tool

POWE.RS provides a command-line tool to estimate PAR parameters from historical data:

```bash
powers estimate-par historical_inflows.csv \
  --periods 12 \
  --order 1 \
  --min-samples 20 \
  --output par_params.json
```

**Arguments**:
- `--periods` (`-p`): Number of seasonal periods (12 for monthly, 4 for quarterly)
- `--order` (`-o`): AR order (1 for AR(1), 2 for AR(2))
- `--min-samples`: Minimum samples per num_seasons (default: 10)
- `--output` (`-O`): Output file path (omit for stdout)
- `--has-header`: Set if CSV has header row

### CSV Input Format

```csv
hydro_1,hydro_2,hydro_3
120.5,45.2,78.3
115.3,42.8,75.1
...
```

- **Columns**: One per entity (hydro plant)
- **Rows**: Consecutive time steps (e.g., months)
- **Units**: Match your system configuration (m³/s, hm³/month, etc.)
- **Order**: Chronological (oldest first)

### Estimation Process

The tool performs:

1. **Seasonal decomposition**: Compute μₘ, σₘ for each num_seasons
2. **Deseasonalization**: Transform data to standardized residuals
3. **Yule-Walker estimation**: Fit AR coefficients via autocorrelation
4. **Validation**: Check stationarity, convergence, sample size

### Output Format

The tool outputs JSON ready to paste into `recourse.json`:

```json
{
  "num_seasons": 12,
  "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "ar_coefficients": [
    [0.72], [0.68], [0.74], ...
  ],
  "seasonal_means": [
    115.3, 108.7, 92.4, ...
  ],
  "seasonal_stds": [
    28.5, 26.3, 23.1, ...
  ]
}
```

Copy these values into the `temporal_model` section of your noise model.

### Data Requirements

| Season | Recommended Data | Minimum Data | Samples/Season |
|--------|------------------|--------------|----------------|
| 12 (monthly) | 10 years (120 months) | 5 years (60 months) | 5-10 |
| 4 (quarterly) | 7 years (28 quarters) | 4 years (16 quarters) | 4-7 |
| 52 (weekly) | 5 years (260 weeks) | 3 years (156 weeks) | 3-5 |

**Rule of thumb**: Need at least 5 samples per num_seasons for reliable estimation.

### Manual Parameter Tuning

If historical data is unavailable, set parameters based on domain knowledge:

1. **Seasonal means**: Expert estimates or literature values
2. **Seasonal std devs**: 20-30% of mean (high variability) or 10-15% (low variability)
3. **AR coefficients**: Start with φ = 0.6-0.7 (moderate persistence)
4. **Season**: Match planning granularity (12 for annual planning)

---

## Examples and Best Practices

### Example 1: Monthly PAR(1) for Brazilian System

**Scenario**: Single reservoir in Brazilian southeast region, annual planning.

```json
// graph.json excerpt
{
  "nodes": [
    {"id": 0, "season_id": 0, "inflow_stochastic_process": "par", ...},  // Jan
    {"id": 1, "season_id": 1, "inflow_stochastic_process": "par", ...},  // Feb
    // ... 
    {"id": 11, "season_id": 11, "inflow_stochastic_process": "par", ...} // Dec
  ]
}

// recourse.json excerpt
{
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 12,
    "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "ar_coefficients": [
      [0.75], [0.72], [0.68], [0.65],  // Summer (wet): Dec-Mar
      [0.60], [0.58], [0.55], [0.57],  // Autumn/Winter (dry): Apr-Jul
      [0.60], [0.63], [0.68], [0.72]   // Spring (transition): Aug-Nov
    ],
    "seasonal_means": [
      150.0, 145.0, 125.0, 100.0,  // High wet season inflows
      75.0, 55.0, 42.0, 45.0,      // Low dry season inflows
      58.0, 72.0, 95.0, 130.0      // Increasing transition
    ],
    "seasonal_stds": [
      40.0, 38.0, 32.0, 28.0,
      20.0, 15.0, 12.0, 13.0,
      18.0, 24.0, 30.0, 38.0
    ]
  },
  "residual_distribution": {
    "type": "lognormal3",
    "gamma": 5.0,  // Safety buffer: min 5 m³/s
    "mu": 0.0,
    "sigma": 0.5
  }
}
```

### Example 2: Quarterly PAR(2) for Long-Term Planning

**Scenario**: Simplified quarterly model with AR(2) for slower dynamics.

```json
{
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 4,
    "ar_orders": [2, 2, 2, 2],  // AR(2) for all quarters
    "ar_coefficients": [
      [0.6, 0.25],  // Q1: φ₁=0.6, φ₂=0.25
      [0.65, 0.2],  // Q2
      [0.55, 0.3],  // Q3
      [0.6, 0.25]   // Q4
    ],
    "seasonal_means": [110.0, 85.0, 50.0, 75.0],
    "seasonal_stds": [30.0, 25.0, 15.0, 22.0]
  }
}
```

**Note**: AR(2) requires 2 initial lags in `initial_condition.inflow`.

### Best Practices

#### 1. Season Selection

✅ **Do**: Match data granularity
- Monthly data → num_seasons = 12
- Quarterly data → num_seasons = 4

❌ **Don't**: Use num_seasons = 52 (weekly) with monthly data

#### 2. AR Order

✅ **Do**: Start with AR(1), increase only if needed
- AR(1) captures most persistence (>90%)
- Higher orders increase state space

❌ **Don't**: Use AR(3)+ without strong justification

#### 3. Coefficient Tuning

✅ **Do**: Vary coefficients by season if data supports it
- Higher φ in wet season (persistent rain)
- Lower φ in dry season (random dry spells)

❌ **Don't**: Set φ > 0.95 (numerical instability, non-stationary behavior)

#### 4. Validation

✅ **Do**: Validate generated scenarios
- Run simulation, plot inflow distribution
- Check: Mean ≈ μₘ, StdDev ≈ σₘ
- Verify autocorrelation matches φ

❌ **Don't**: Trust parameters without validation

#### 5. Initial Conditions

✅ **Do**: Use realistic historical values
- Take last observed inflows
- Or use seasonal mean if unknown

❌ **Don't**: Use zeros (creates initial bias)

---

## Performance Considerations

### State Space Impact

PAR adds state variables:

| AR Order | Added States per Hydro | Example: 10 Hydros |
|----------|------------------------|-------------------|
| AR(0) | 0 | 10 states (storage only) |
| AR(1) | 1 | 20 states (storage + 1 lag) |
| AR(2) | 2 | 30 states (storage + 2 lags) |
| AR(3) | 3 | 40 states (storage + 3 lags) |

**Impact**:
- More states → more cuts stored
- More cuts → more memory, slower subproblem solves
- Polynomial growth: O(states²) for cut storage

### Computational Cost

| Component | Cost | Notes |
|-----------|------|-------|
| PAR generation | O(p) per scenario | p = AR order, typically 1-2 |
| State updates | O(p) per stage | Shift residual buffer |
| Memory | O(stages × hydros × p) | Store residual buffers |

**Overhead**: ~5-10% vs stationary AR for typical systems (p=1, <20 hydros)

### Optimization Tips

1. **Use AR(1) when possible**: Minimal overhead vs AR(2)
2. **Mixed models**: PAR for major hydros, Independent for minor ones
3. **Reduce scenarios**: If state space grows large (>50 states), reduce branching
4. **Profile first**: Measure actual overhead before optimizing

### Memory Profiling

```bash
# Profile memory usage
cargo build --release
/usr/bin/time -v ./target/release/powers examples/06-par-model/01-simple-par1

# Look for "Maximum resident set size"
# PAR(1): Expect <100 MB for single hydro
# PAR(2): Expect <150 MB for single hydro
```

---

## Troubleshooting

### Common Errors

#### 1. "season_id out of range"

**Error**: `season_id=12 exceeds num_seasons=12`

**Cause**: season_id must be in [0, num_seasons-1]

**Fix**: Check graph.json - season_id should cycle 0, 1, ..., 11, 0, 1, ...

```json
// ❌ Wrong
{"stage_id": 11, "season_id": 12, ...}  // Should be 11!

// ✅ Correct
{"stage_id": 11, "season_id": 11, ...}
{"stage_id": 12, "season_id": 0, ...}   // Wraps to 0
```

#### 2. "ar_coefficients length mismatch"

**Error**: `ar_coefficients[3] has length 1, but ar_orders[3] = 2`

**Cause**: Coefficient array doesn't match declared order

**Fix**: Ensure `ar_coefficients[m].len() == ar_orders[m]`

```json
// ❌ Wrong
"ar_orders": [1, 1, 2, 1],
"ar_coefficients": [[0.7], [0.6], [0.65], [0.7]]  // Missing 2nd coeff for num_seasons 2!

// ✅ Correct
"ar_orders": [1, 1, 2, 1],
"ar_coefficients": [[0.7], [0.6], [0.65, 0.2], [0.7]]
```

#### 3. "Missing initial inflow lags"

**Error**: `Hydro 0 requires lag=1 for AR(1), but no initial value provided`

**Cause**: AR models need past values to start

**Fix**: Add to `initial_condition.inflow`:

```json
"inflow": [
  {"hydro_id": 0, "lag": 1, "value": 85.0}
]
```

#### 4. "Non-stationary AR coefficients"

**Warning**: `AR coefficients for num_seasons 3 may be non-stationary: φ=0.98`

**Cause**: |φ| too close to 1.0

**Fix**: Reduce coefficient:

```json
// ❌ Risk of non-stationarity
"ar_coefficients": [[0.98], ...]

// ✅ Safe stationary value
"ar_coefficients": [[0.85], ...]
```

#### 5. "state_variables must be storage_and_inflow"

**Error**: `PAR process requires state_variables="storage_and_inflow"`

**Cause**: Wrong state variable configuration

**Fix**: Update graph.json:

```json
// ❌ Wrong
"state_variables": "storage",

// ✅ Correct
"state_variables": "storage_and_inflow",
```

### Validation Checklist

Before running, verify:

- [ ] `season_id` cycles correctly (0 to num_seasons-1)
- [ ] All seasonal arrays have length = num_seasons
- [ ] `ar_coefficients[m].len() == ar_orders[m]` for all m
- [ ] AR coefficients satisfy stationarity (|φ| < 1 for AR(1))
- [ ] Initial inflow lags provided for all hydros
- [ ] `state_variables = "storage_and_inflow"` in graph nodes
- [ ] `inflow_stochastic_process = "par"` in graph nodes
- [ ] Residual distribution is appropriate (LogNormal3 for non-negative)

### Debugging Tips

1. **Start simple**: Begin with num_seasons=2, AR(1), single hydro
2. **Validate incrementally**: Add complexity step-by-step
3. **Check generated scenarios**: Plot inflows, verify distributions
4. **Compare with Independent**: Run same problem with Independent noise, compare costs
5. **Enable logging**: Check for warnings about parameter values

---

## References

- **Algorithm Documentation**: `docs/algorithm/PAR_MODEL_SUPPORT.md`
- **Schema Reference**: `schemas/recourse.schema.json`
- **Examples**: `examples/06-par-model/`
- **Migration Guide**: `docs/guides/MIGRATION-TO-PAR.md`
- **API Documentation**: `src/par_generator.rs`, `src/seasonal_params.rs`

## Further Reading

- **Box, G.E.P., Jenkins, G.M. (1976)**: Time Series Analysis: Forecasting and Control
- **CEPEL Technical Notes**: Seasonic Autoregressive Models for Hydrothermal Systems
- **Maceira, M.E.P., et al. (2008)**: "TEN YEARS OF APPLICATION OF STOCHASTIC DUAL DYNAMIC PROGRAMMING IN OFFICIAL AND AGENT STUDIES IN BRAZIL"

---

**Need Help?** See `docs/guides/TROUBLESHOOTING.md` or open an issue on GitHub.
