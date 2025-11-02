# POWE.RS Input Specification

**Version**: 0.2.0  
**Last Updated**: 2025-10-05

This document provides a comprehensive specification of all input JSON files required by POWE.RS for SDDP hydrothermal dispatch optimization.

## Table of Contents

1. [Overview](#overview)
2. [Configuration (`config.json`)](#configuration-configjson)
3. [Power System (`system.json`)](#power-system-systemjson)
4. [Scenario Tree Graph (`graph.json`)](#scenario-tree-graph-graphjson)
5. [Uncertainty and Recourse (`recourse.json`)](#uncertainty-and-recourse-recoursejson)
6. [Validation Rules](#validation-rules)
7. [Common Errors](#common-errors)
8. [IDE Integration](#ide-integration)

---

## Overview

POWE.RS requires four JSON input files to define an SDDP problem:

| File            | Purpose                                                  | Schema                                                  |
| --------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| `config.json`   | Algorithm configuration (iterations, seeds, output)      | [config.schema.json](../schemas/config.schema.json)     |
| `system.json`   | Power system topology (buses, lines, generators, hydros) | [system.schema.json](../schemas/system.schema.json)     |
| `graph.json`    | Multistage scenario tree (nodes, edges, stages)          | [graph.schema.json](../schemas/graph.schema.json)       |
| `recourse.json` | Initial conditions and uncertainty distributions         | [recourse.schema.json](../schemas/recourse.schema.json) |

**Validation**: All schemas follow JSON Schema Draft 7 standard. Use the provided schemas for:

- IDE auto-completion (VS Code, IntelliJ)
- Pre-validation before running POWE.RS
- Documentation generation

---

## Configuration (`config.json`)

**Schema**: [`schemas/config.schema.json`](../schemas/config.schema.json)

Defines SDDP algorithm parameters and execution settings.

### Fields

#### `num_iterations` (required)

- **Type**: `integer`
- **Constraint**: `>= 1`
- **Description**: Number of SDDP training iterations
- **Examples**: `10`, `32`, `100`
- **Guidance**: More iterations improve convergence but increase computation time
  - Quick tests: 10-20 iterations
  - Production: 50-100+ iterations

#### `num_forward_passes` (required)

- **Type**: `integer`
- **Constraint**: `>= 1`
- **Description**: Number of forward passes per iteration for Monte Carlo sampling
- **Examples**: `4`, `10`, `20`
- **Guidance**: More passes give better lower bound estimates
  - Fast convergence: 4-10 passes
  - Accurate bounds: 20-50 passes

#### `num_simulation_scenarios` (required)

- **Type**: `integer`
- **Constraint**: `>= 1`
- **Description**: Number of out-of-sample scenarios for policy evaluation
- **Examples**: `100`, `128`, `1000`
- **Guidance**: More scenarios give better policy quality estimates
  - Quick evaluation: 100-200 scenarios
  - Statistical significance: 1000+ scenarios

#### `seed` (required)

- **Type**: `integer` (unsigned 64-bit)
- **Constraint**: `>= 0`
- **Description**: Random seed for reproducibility
- **Examples**: `0`, `42`, `123456`
- **Guidance**: Same seed produces identical results (deterministic)

#### `output_path` (optional)

- **Type**: `string` or `null`
- **Default**: `null` (no output)
- **Description**: Directory path for CSV output files
- **Examples**: `"./output"`, `"./examples/01-deterministic"`, `null`
- **Output Files Generated**:
  - `cuts.csv` - Benders cuts (intercept, slopes)
  - `states.csv` - Visited states during training
  - `simulation_buses.csv` - Bus-level simulation results
  - `simulation_lines.csv` - Transmission line flows
  - `simulation_thermals.csv` - Thermal generation
  - `simulation_hydros.csv` - Hydro generation and storage
- **Performance Note**: Omitting `output_path` (or setting to `null`) disables CSV generation, resulting in 10-30% faster execution

### Example

```json
{
  "num_iterations": 10,
  "num_forward_passes": 1,
  "num_simulation_scenarios": 1,
  "seed": 0,
  "output_path": "./examples/01-deterministic"
}
```

### Validation Rules

1. All numeric fields must be positive integers
2. `output_path` is optional; if provided, directory will be created if it doesn't exist
3. No additional properties allowed

---

## Power System (`system.json`)

**Schema**: [`schemas/system.schema.json`](../schemas/system.schema.json)

Defines the physical power system: buses, transmission lines, thermal generators, and hydroelectric plants.

### Structure

```json
{
  "buses": [...],
  "lines": [...],
  "thermals": [...],
  "hydros": [...]
}
```

### Buses

Electrical buses (nodes) in the power system.

#### Fields

| Field          | Type      | Constraint                | Description                  |
| -------------- | --------- | ------------------------- | ---------------------------- |
| `id`           | `integer` | `>= 0`, contiguous 0..N-1 | Unique bus identifier        |
| `deficit_cost` | `number`  | `>= 0`                    | Cost of unmet demand ($/MWh) |

**Guidance**:

- `deficit_cost`: Penalty for load shedding; typically high (e.g., $1000-$5000/MWh)
- IDs must form contiguous range: [0, 1, 2, ...] with no gaps

#### Example

```json
"buses": [
  {
    "id": 0,
    "deficit_cost": 50.0
  },
  {
    "id": 1,
    "deficit_cost": 100.0
  }
]
```

### Lines

Transmission lines connecting buses.

#### Fields

| Field              | Type      | Constraint                | Description                        |
| ------------------ | --------- | ------------------------- | ---------------------------------- |
| `id`               | `integer` | `>= 0`, contiguous 0..N-1 | Unique line identifier             |
| `source_bus_id`    | `integer` | Must exist in `buses`     | Source bus ID                      |
| `target_bus_id`    | `integer` | Must exist in `buses`     | Target bus ID                      |
| `direct_capacity`  | `number`  | `>= 0`                    | Max power flow source→target (MW)  |
| `reverse_capacity` | `number`  | `>= 0`                    | Max power flow target→source (MW)  |
| `exchange_penalty` | `number`  | `>= 0`                    | Penalty for power exchange ($/MWh) |

**Guidance**:

- Empty array `[]` is valid (isolated buses)
- Asymmetric capacities model different flow limits
- `exchange_penalty`: Transmission loss or wheeling cost

#### Example

```json
"lines": [
  {
    "id": 0,
    "source_bus_id": 0,
    "target_bus_id": 1,
    "direct_capacity": 100.0,
    "reverse_capacity": 100.0,
    "exchange_penalty": 0.5
  }
]
```

### Thermals

Thermal generation units (coal, gas, etc.).

#### Fields

| Field            | Type      | Constraint                | Description             |
| ---------------- | --------- | ------------------------- | ----------------------- |
| `id`             | `integer` | `>= 0`, contiguous 0..N-1 | Unique thermal unit ID  |
| `bus_id`         | `integer` | Must exist in `buses`     | Connected bus ID        |
| `cost`           | `number`  | `>= 0`                    | Generation cost ($/MWh) |
| `min_generation` | `number`  | `>= 0`                    | Minimum output (MW)     |
| `max_generation` | `number`  | `>= min_generation`       | Maximum output (MW)     |

**Guidance**:

- `min_generation = 0`: Unit can be turned off
- `min_generation > 0`: Minimum stable generation (e.g., due to thermal constraints)
- Cost-ordering: Merit order dispatch prefers lower-cost units

#### Example

```json
"thermals": [
  {
    "id": 0,
    "bus_id": 0,
    "cost": 5.0,
    "min_generation": 0.0,
    "max_generation": 15.0
  },
  {
    "id": 1,
    "bus_id": 0,
    "cost": 10.0,
    "min_generation": 0.0,
    "max_generation": 15.0
  }
]
```

### Hydros

Hydroelectric plants with reservoirs.

#### Fields

| Field                 | Type                | Constraint                        | Description                                  |
| --------------------- | ------------------- | --------------------------------- | -------------------------------------------- |
| `id`                  | `integer`           | `>= 0`, contiguous 0..N-1         | Unique hydro plant ID                        |
| `downstream_hydro_id` | `integer` or `null` | Must exist in `hydros`, no cycles | Downstream hydro ID for cascade              |
| `bus_id`              | `integer`           | Must exist in `buses`             | Connected bus ID                             |
| `productivity`        | `number`            | `> 0`                             | Water-to-power conversion (MW per unit flow) |
| `min_storage`         | `number`            | `>= 0`                            | Minimum reservoir storage                    |
| `max_storage`         | `number`            | `>= min_storage`                  | Maximum reservoir storage                    |
| `min_turbined_flow`   | `number`            | `>= 0`                            | Minimum turbine flow                         |
| `max_turbined_flow`   | `number`            | `>= min_turbined_flow`            | Maximum turbine flow                         |
| `spillage_penalty`    | `number`            | `>= 0`                            | Cost penalty for spillage ($/unit)           |

**Guidance**:

- **Cascade Modeling**: Set `downstream_hydro_id` to model river cascades
  - `null`: No downstream (terminal reservoir or independent plant)
  - Valid ID: Water flows to downstream plant
  - **Must not create cycles**: Validate cascade topology
- **Storage Units**: Volume units must be consistent across all hydros
- **Flow Units**: Time units (e.g., hourly flow) must match stage duration
- **Productivity**: Conversion factor from flow to power (depends on head, efficiency)
- **Spillage Penalty**: Small cost to discourage wasteful spillage (e.g., $0.01/unit)

#### Example

```json
"hydros": [
  {
    "id": 0,
    "downstream_hydro_id": null,
    "bus_id": 0,
    "productivity": 1.0,
    "min_storage": 0.0,
    "max_storage": 100.0,
    "min_turbined_flow": 0.0,
    "max_turbined_flow": 60.0,
    "spillage_penalty": 0.01
  }
]
```

### Complete Example

See [`example/system.json`](../examples/01-deterministic/system.json) for a complete working example.

---

## Scenario Tree Graph (`graph.json`)

**Schema**: [`schemas/graph.schema.json`](../schemas/graph.schema.json)

Defines the multistage stochastic scenario tree structure.

### Structure

```json
{
  "nodes": [...],
  "edges": [...]
}
```

### Nodes

Decision stages in the scenario tree.

#### Fields

| Field                       | Type      | Constraint                               | Description                         |
| --------------------------- | --------- | ---------------------------------------- | ----------------------------------- |
| `id`                        | `integer` | `>= 0`, unique                           | Unique node identifier              |
| `stage_id`                  | `integer` | `>= 0`                                   | Stage number (typically sequential) |
| `season_id`                 | `integer` | `>= 0`                                   | Season ID (matches `recourse.json`) |
| `start_date`                | `string`  | ISO 8601 date-time                       | Stage start date                    |
| `end_date`                  | `string`  | ISO 8601, `> start_date`                 | Stage end date                      |
| `risk_measure`              | `string`  | `"expectation"`, `"cvar"`, `"worstcase"` | Risk measure                        |
| `load_stochastic_process`   | `string`  | `"naive"`                                | Load uncertainty model              |
| `inflow_stochastic_process` | `string`  | `"naive"`                                | Inflow uncertainty model            |
| `state_variables`           | `string`  | `"storage"`                              | State variables                     |

**Guidance**:

- **stage_id**: Typically sequential (0, 1, 2, ...) but not required
- **season_id**: Links to `uncertainties` in `recourse.json` (same season → same distributions)
- **Date Format**: Use ISO 8601 format: `"2024-01-01T00:00:00Z"`
- **Risk Measures**:
  - `"expectation"`: Expected value (standard SDDP)
  - `"cvar"`: Conditional Value at Risk (risk-averse)
  - `"worstcase"`: Worst-case scenario (maximum risk aversion)
- **Stochastic Processes**: Currently only `"naive"` (direct sampling from distributions) is supported
- **State Variables**: Currently only `"storage"` (reservoir levels) is tracked

#### Example

```json
"nodes": [
  {
    "id": 0,
    "stage_id": 0,
    "season_id": 0,
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-02-01T00:00:00Z",
    "risk_measure": "expectation",
    "load_stochastic_process": "naive",
    "inflow_stochastic_process": "naive",
    "state_variables": "storage"
  }
]
```

### Edges

Transitions between nodes in the scenario tree.

#### Fields

| Field           | Type      | Constraint            | Description                    |
| --------------- | --------- | --------------------- | ------------------------------ |
| `source_id`     | `integer` | Must exist in `nodes` | Source node ID                 |
| `target_id`     | `integer` | Must exist in `nodes` | Target node ID                 |
| `probability`   | `number`  | `> 0`, `<= 1.0`       | Transition probability         |
| `discount_rate` | `number`  | `>= 0`                | Discount rate for future costs |

**Guidance**:

- **Probability Constraint**: Sum of outgoing edge probabilities from each node must equal 1.0
- **Discount Rate**: Annual rate (e.g., 0.05 = 5% discount)
  - 0.0 = no discounting (common for operational planning)
- **Graph Connectivity**: All nodes must be reachable from root (node 0)
- **Scenario Tree Structure**:
  - Single root node (stage 0)
  - Branching represents uncertainty realization
  - Common structures: fan (independent stages), tree (correlated stages)

#### Example

```json
"edges": [
  {
    "source_id": 0,
    "target_id": 1,
    "probability": 0.5,
    "discount_rate": 0.0
  },
  {
    "source_id": 0,
    "target_id": 2,
    "probability": 0.5,
    "discount_rate": 0.0
  }
]
```

### Complete Example

See [`example/graph.json`](../examples/01-deterministic/graph.json) for a 12-stage sequential scenario tree.

---

## Uncertainty and Recourse (`recourse.json`)

**Schema**: [`schemas/recourse.schema.json`](../schemas/recourse.schema.json)

Defines initial system state and uncertainty distributions for stochastic scenarios.

### Structure

```json
{
  "initial_condition": {...},
  "uncertainties": [...]
}
```

### Initial Condition

System state at the beginning of the planning horizon.

#### Storage

| Field      | Type      | Constraint                                     | Description             |
| ---------- | --------- | ---------------------------------------------- | ----------------------- |
| `hydro_id` | `integer` | Must exist in `system.json`, contiguous 0..N-1 | Hydro plant ID          |
| `value`    | `number`  | Within `[min_storage, max_storage]`            | Initial reservoir level |

**Guidance**:

- Must specify storage for **all hydro plants** (IDs 0..N-1)
- Values must respect bounds from `system.json`

#### Inflow

| Field      | Type      | Constraint                  | Description                           |
| ---------- | --------- | --------------------------- | ------------------------------------- |
| `hydro_id` | `integer` | Must exist in `system.json` | Hydro plant ID                        |
| `lag`      | `integer` | `>= 1`                      | Lag num_seasons for historical inflow |
| `value`    | `number`  | Any                         | Historical inflow value               |

**Guidance**:

- Optional: Used for AR/ARMA inflow models (future feature)
- Currently unused by `"naive"` stochastic process

#### Example

```json
"initial_condition": {
  "storage": [
    {
      "hydro_id": 0,
      "value": 83.222
    }
  ],
  "inflow": [
    {
      "hydro_id": 0,
      "lag": 1,
      "value": 50.0
    }
  ]
}
```

### Uncertainties

Probability distributions for each season in the scenario tree.

#### Seasonal Uncertainty

| Field            | Type      | Constraint                          | Description                   |
| ---------------- | --------- | ----------------------------------- | ----------------------------- |
| `season_id`      | `integer` | Matches `season_id` in `graph.json` | Season identifier             |
| `num_branchings` | `integer` | `>= 1`                              | Number of scenarios to sample |
| `distributions`  | `object`  | Contains `load` and `inflow` arrays | Distribution specifications   |

**Guidance**:

- **season_id**: Must match seasons defined in `graph.json`
- **num_branchings**: Number of Monte Carlo samples per season
  - More branches → better uncertainty representation
  - Typical: 10-50 branches per season

#### Load Distributions

Uncertainty in electrical load demand.

| Field          | Type              | Distribution | Description                            |
| -------------- | ----------------- | ------------ | -------------------------------------- |
| `bus_id`       | `integer`         | N/A          | Bus ID (must exist, contiguous 0..N-1) |
| `normal`       | `object`          | N(μ, σ²)     | Normal distribution                    |
| `normal.mu`    | `number`          | N/A          | Mean load (MW)                         |
| `normal.sigma` | `number` (`>= 0`) | N/A          | Standard deviation (MW)                |

**Guidance**:

- Must specify distribution for **all buses** (IDs 0..N-1)
- **Normal Distribution**: Symmetric, can produce negative values
  - Use `sigma = 0` for deterministic load
  - Typical: `sigma = 5-10%` of `mu` for realistic uncertainty

#### Example

```json
"load": [
  {
    "bus_id": 0,
    "normal": {
      "mu": 75.0,
      "sigma": 7.5
    }
  }
]
```

#### Inflow Distributions

Uncertainty in hydro inflows.

| Field             | Type              | Distribution | Description                                    |
| ----------------- | ----------------- | ------------ | ---------------------------------------------- |
| `hydro_id`        | `integer`         | N/A          | Hydro plant ID (must exist, contiguous 0..N-1) |
| `normal`          | `object`          | N(μ, σ²)     | Normal distribution                            |
| `normal.mu`       | `number`          | N/A          | Mean inflow                                    |
| `normal.sigma`    | `number` (`>= 0`) | N/A          | Standard deviation                             |
| `lognormal`       | `object`          | LogN(μ, σ²)  | Log-normal distribution                        |
| `lognormal.mu`    | `number`          | N/A          | Mean of underlying normal                      |
| `lognormal.sigma` | `number` (`> 0`)  | N/A          | Std dev of underlying normal                   |

**Guidance**:

- Must specify distribution for **all hydro plants** (IDs 0..N-1)
- **Choose Distribution**:
  - **Normal**: Symmetric, can be negative (use for flows with storage buffers)
  - **Log-normal**: Always positive, right-skewed (realistic for natural inflows)
    - Parameters are for the **underlying normal distribution**
    - Actual mean: `exp(μ + σ²/2)`
    - Use for rivers, rainfall-based inflows
- **Deterministic**: Set `sigma = 0` (normal only)

#### Example

```json
"inflow": [
  {
    "hydro_id": 0,
    "lognormal": {
      "mu": 3.6,
      "sigma": 0.6928
    }
  }
]
```

### Noise Models (Advanced)

**Note**: `noise_models` replaces the simpler `uncertainties` structure for advanced temporal models (PAR, AR). You cannot use both `uncertainties` and `noise_models` in the same file.

For systems with seasonal patterns or temporal correlation in inflows, use `noise_models` with Seasonic Autoregressive (PAR) models.

#### Noise Model Structure

| Field                   | Type      | Constraint                          | Description                    |
| ----------------------- | --------- | ----------------------------------- | ------------------------------ |
| `uncertainty_type`      | `string`  | `"inflow"` or `"load"`              | Type of uncertainty            |
| `entity_id`             | `integer` | Must exist (hydro or bus)           | Hydro/Bus ID                   |
| `season_id`             | `integer` | Matches `season_id` in `graph.json` | Season identifier              |
| `marginal_distribution` | `object`  | Normal or LogNormal3                | Legacy field (ignored for PAR) |
| `temporal_model`        | `object`  | See below                           | Temporal correlation model     |
| `residual_distribution` | `object`  | Normal or LogNormal3                | Distribution of PAR residuals  |

#### Temporal Model: Seasonic Autoregressive (PAR)

PAR models capture both **seasonality** (mean/variance changes by season) and **persistence** (correlation with past values).

**When to use PAR**:

- Seasonal inflow variation >30% (e.g., wet season 120 m³/s, dry season 40 m³/s)
- Long planning horizons >12 months
- Historical data shows clear annual patterns

See [PAR Model Guide](../guides/PAR-MODEL-GUIDE.md) for detailed documentation and [Migration Guide](../guides/MIGRATION-TO-PAR.md) for converting from stationary AR.

##### PAR Temporal Model Fields

| Field             | Type         | Constraint                                         | Description                   |
| ----------------- | ------------ | -------------------------------------------------- | ----------------------------- |
| `type`            | `string`     | `"periodic_ar"`                                    | Identifies PAR model          |
| `num_seasons`     | `integer`    | `>= 1` (typically 12 for monthly)                  | Number of seasons in cycle    |
| `ar_orders`       | `integer[]`  | Length = `num_seasons`, each `>= 0`                | AR order for each season      |
| `ar_coefficients` | `number[][]` | Outer length = `num_seasons`, inner length = order | φ coefficients per season     |
| `seasonal_means`  | `number[]`   | Length = `num_seasons`                             | Mean inflow for each season   |
| `seasonal_stds`   | `number[]`   | Length = `num_seasons`, all `> 0`                  | Std deviation for each season |

**Mathematical Model** :

```
Zₜ = μₘ + σₘ · [∑ᵢ₌₁ᵖ φᵢₘ·aₜ₋ᵢ + aₜ]

where:
  m = t mod num_seasons             (current season)
  Zₜ = generated inflow value
  μₘ = seasonal_means[m]
  σₘ = seasonal_stds[m]
  φᵢₘ = ar_coefficients[m][i-1]
  aₜ ~ residual_distribution
```

**Validation Rules**:

1. **Array Lengths**: All seasonal arrays (`ar_orders`, `ar_coefficients`, `seasonal_means`, `seasonal_stds`) must have length = `num_seasons`
2. **AR Orders**: `ar_coefficients[m].length == ar_orders[m]` for each season m
3. **Positive Stds**: All `seasonal_stds` values must be `> 0`
4. **Initial Condition**: Must provide lagged inflows for all lags up to `max(ar_orders)`
5. **Season Cycling**: `season_id` in `graph.json` must cycle: 0, 1, ..., num_seasons-1, 0, 1, ...
6. **State Variables**: Nodes using PAR must set `state_variables = "storage_and_inflow"`
7. **Stochastic Process**: Nodes using PAR must set `inflow_stochastic_process = "par"`

##### PAR Example: Monthly PAR(1) Model

**Graph Configuration** (`graph.json`):

```json
{
  "nodes": [
    {
      "id": 0,
      "stage_id": 0,
      "season_id": 0,
      "inflow_stochastic_process": "par",
      "state_variables": "storage_and_inflow",
      ...
    },
    {
      "id": 1,
      "stage_id": 1,
      "season_id": 1,
      "inflow_stochastic_process": "par",
      "state_variables": "storage_and_inflow",
      ...
    }
    // ... (continue with season_id: 2, 3, ..., 11, then wrap to 0)
  ]
}
```

**Recourse Configuration** (`recourse.json`):

```json
{
  "initial_condition": {
    "storage": [{ "hydro_id": 0, "value": 150.0 }],
    "inflow": [{ "hydro_id": 0, "lag": 1, "value": 85.0 }]
  },
  "noise_models": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,
      "season_id": 0,
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
          [0.75],
          [0.72],
          [0.68],
          [0.65],
          [0.6],
          [0.58],
          [0.55],
          [0.57],
          [0.6],
          [0.63],
          [0.68],
          [0.72]
        ],
        "seasonal_means": [
          120.0, 110.0, 95.0, 80.0, 60.0, 45.0, 35.0, 40.0, 55.0, 75.0, 95.0,
          115.0
        ],
        "seasonal_stds": [
          30.0, 28.0, 25.0, 22.0, 18.0, 15.0, 12.0, 15.0, 18.0, 22.0, 26.0, 29.0
        ]
      },
      "residual_distribution": {
        "type": "lognormal3",
        "gamma": 1.0,
        "mu": 0.0,
        "sigma": 0.6
      }
    }
  ]
}
```

**Interpretation**:

- **Season 12**: Monthly cycle (Jan=0, Feb=1, ..., Dec=11)
- **PAR(1)**: Each month's inflow depends on previous month (`ar_orders=[1,1,...,1]`)
- **Seasonal Means**: Peak inflow ~120 m³/s (Jan), lowest ~35 m³/s (Jul)
- **Seasonal Stds**: Higher uncertainty in wet season (30 m³/s), lower in dry (12 m³/s)
- **AR Coefficients**: Persistence varies by season (0.55-0.75)
- **Residuals**: LogNormal3 ensures positive values

##### PAR Example: Quarterly PAR(2) Model

For quarterly data with 2-num_seasons memory:

```json
{
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 4,
    "ar_orders": [2, 2, 2, 2],
    "ar_coefficients": [
      [0.6, 0.3],
      [0.5, 0.25],
      [0.55, 0.28],
      [0.65, 0.32]
    ],
    "seasonal_means": [150.0, 80.0, 40.0, 120.0],
    "seasonal_stds": [35.0, 20.0, 12.0, 30.0]
  }
}
```

**Initial Condition** (must provide lag-1 and lag-2):

```json
{
  "inflow": [
    { "hydro_id": 0, "lag": 1, "value": 130.0 },
    { "hydro_id": 0, "lag": 2, "value": 125.0 }
  ]
}
```

#### Residual Distribution

The `residual_distribution` defines the distribution of the noise term (aₜ in PAR equation).

**Supported Distributions**:

1. **Normal Distribution**:

   ```json
   {
     "type": "normal",
     "mean": 0.0,
     "std_dev": 1.0
   }
   ```

   - Symmetric, can produce negative values
   - Use when seasonal_stds are large enough to keep Zₜ > 0

2. **LogNormal3 Distribution** (3-parameter log-normal):
   ```json
   {
     "type": "lognormal3",
     "gamma": 1.0,
     "mu": 0.0,
     "sigma": 0.6
   }
   ```
   - Always positive, right-skewed
   - `gamma`: Location parameter (shifts distribution)
   - `mu`, `sigma`: Parameters of underlying normal distribution
   - **Recommended for inflows** to ensure non-negative values

**Choosing Residual Distribution**:

- **Normal**: Simple, fast, but can produce negative residuals
- **LogNormal3**: Realistic for natural inflows (always positive, skewed)

### Complete Example

See:

- [`examples/01-deterministic/recourse.json`](../examples/01-deterministic/recourse.json) - Simple independent uncertainties
- [`examples/06-par-model/01-simple-par1/recourse.json`](../examples/06-par-model/01-simple-par1/recourse.json) - PAR(1) model

---

## Validation Rules

POWE.RS performs validation at multiple levels:

### 1. Schema Validation (IDE + Pre-Run)

JSON schemas enforce:

- Correct types (integer, number, string)
- Required fields present
- Numeric constraints (min, max, exclusiveMinimum)
- Enum values for categorical fields
- No additional properties

### 2. Runtime Validation (T3.7 InputValidator)

Additional runtime checks:

- **Config**: Positive iteration counts, forward passes, simulation scenarios
- **System**: (Future) ID contiguity, min <= max, no hydro cascade cycles
- **Graph**: (Future) Node ID uniqueness, probability sums = 1.0, connectivity
- **Recourse**: (Future) Storage bounds, distribution parameter validity

### 3. Cross-File Validation

Consistency checks across files:

- Bus IDs in `system.json` match references in `lines`, `thermals`, `hydros`
- Hydro IDs in `system.json` match references in `recourse.json`
- Season IDs in `graph.json` match references in `recourse.json` uncertainties
- Initial storage values within bounds from `system.json`

### Validation Workflow

```
1. IDE Validation (immediate)
   ↓
2. Input::from_paths() loads files
   ↓
3. InputValidator checks config (Phase 1)
   ↓
4. Serde deserialization validates types
   ↓
5. build_sddp_graph() validates references
   ↓
6. generate_sddp_noises() validates distributions
```

---

## Common Errors

### Configuration Errors

#### ❌ Zero iterations

```json
{"num_iterations": 0, ...}
```

**Error**: `config.json: num_iterations must be positive (got 0)`  
**Fix**: Set `num_iterations >= 1`

#### ❌ Missing required field

```json
{ "num_iterations": 10, "num_forward_passes": 4 }
```

**Error**: `missing field 'num_simulation_scenarios'`  
**Fix**: Add all required fields

### System Errors

#### ❌ Non-contiguous IDs

```json
"buses": [{"id": 0, ...}, {"id": 2, ...}]
```

**Error**: `buses: ID 1 missing (must be contiguous 0..N-1)`  
**Fix**: Use sequential IDs: [0, 1, 2, ...]

#### ❌ Invalid bus reference

```json
"thermals": [{"id": 0, "bus_id": 99, ...}]
```

**Error**: `thermal 0: bus_id 99 does not exist`  
**Fix**: Use valid bus_id from `buses` array

#### ❌ Min > Max

```json
{ "min_generation": 10.0, "max_generation": 5.0 }
```

**Error**: `thermal 0: min_generation (10.0) > max_generation (5.0)`  
**Fix**: Ensure `min <= max`

### Graph Errors

#### ❌ Probability sum != 1.0

```json
"edges": [
  {"source_id": 0, "target_id": 1, "probability": 0.3, ...},
  {"source_id": 0, "target_id": 2, "probability": 0.3, ...}
]
```

**Error**: `node 0: outgoing edge probabilities sum to 0.6 (must be 1.0)`  
**Fix**: Ensure probabilities from each source sum to 1.0

#### ❌ Date order

```json
{ "start_date": "2024-02-01T00:00:00Z", "end_date": "2024-01-01T00:00:00Z" }
```

**Error**: `node 0: end_date must be after start_date`  
**Fix**: Swap dates or correct typo

### Recourse Errors

#### ❌ Missing distribution

```json
"load": [{"bus_id": 0, "normal": {...}}]
// system.json has 2 buses, missing bus_id=1
```

**Error**: `season 0: missing load distribution for bus 1`  
**Fix**: Provide distributions for all buses/hydros

#### ❌ Storage out of bounds

```json
{ "hydro_id": 0, "value": 150.0 }
// system.json: hydro 0 has max_storage = 100.0
```

**Error**: `initial storage for hydro 0 (150.0) exceeds max_storage (100.0)`  
**Fix**: Set value within `[min_storage, max_storage]`

#### ❌ Invalid log-normal sigma

```json
{ "hydro_id": 0, "lognormal": { "mu": 3.6, "sigma": 0.0 } }
```

**Error**: `lognormal distribution: sigma must be > 0`  
**Fix**: Use `sigma > 0` (or switch to normal with `sigma = 0` for deterministic)

---

## IDE Integration

### VS Code

JSON schemas are automatically detected via `.vscode/settings.json`:

```json
{
  "json.schemas": [
    {
      "fileMatch": ["**/config.json"],
      "url": "./schemas/config.schema.json"
    }
    // ... (other schemas)
  ]
}
```

**Features**:

- ✅ Auto-completion (Ctrl+Space)
- ✅ Inline documentation on hover
- ✅ Real-time validation (red squiggles for errors)
- ✅ Schema-aware refactoring

**Setup**:

1. Install "JSON Language Features" extension (usually built-in)
2. Open any JSON file in `example/` directory
3. Start typing - auto-completion should work immediately

### IntelliJ IDEA / PyCharm

1. Open Settings → Languages & Frameworks → Schemas and DTDs → JSON Schema Mappings
2. Add mappings:
   - Schema: `schemas/config.schema.json` → File: `*config*.json`
   - Repeat for other schemas
3. IntelliJ will provide auto-completion and validation

### Pre-Validation (Optional)

Validate JSON files before running POWE.RS using `jsonschema` CLI:

```bash
# Install jsonschema (Python)
pip install jsonschema

# Validate files
jsonschema -i example/config.json schemas/config.schema.json
jsonschema -i example/system.json schemas/system.schema.json
jsonschema -i example/graph.json schemas/graph.schema.json
jsonschema -i example/recourse.json schemas/recourse.schema.json
```

---

## Examples

Complete working examples are provided in the [`examples/`](../examples/) directory:

- [`config.json`](../examples/01-deterministic/config.json) - 10 iterations, 1 simulation scenario
- [`system.json`](../examples/01-deterministic/system.json) - 1 bus, 1 thermal, 1 hydro
- [`graph.json`](../examples/01-deterministic/graph.json) - 2-stage sequential tree
- [`recourse.json`](../examples/01-deterministic/recourse.json) - Deterministic inflows, deterministic load

**Run Example**:

```bash
cargo run --release example
```

**Expected Output**:

- Training: 10 iterations in ~0.5s
- Simulation: 1 scenarios in ~0.1s
- Output: CSV files in `examples/01-deterministic/` directory

---

## References

- **JSON Schema**: [json-schema.org](https://json-schema.org/)
- **SDDP Algorithm**: Pereira & Pinto (1991), "Multi-stage stochastic optimization applied to energy planning"
- **POWE.RS Repository**: [github.com/rjmalves/powers](https://github.com/rjmalves/powers)

---

**Questions or Issues?** Open an issue on GitHub: https://github.com/rjmalves/powers/issues
