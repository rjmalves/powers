---
status: approved
review_priority: 1-critical
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.7 (Stage Definitions — stages.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.8 (Uncertainty Models — inflow_models.parquet)"
  - "DATA_MODEL_SPECIFICATION.md §3.10 (Load Factors — load_factors.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.12 (Correlation — correlation.json)"
last_reviewed: 2026-02-17
reviewed_by: rogerio
review_notes: "Major restructuring. Added season_definitions, policy_graph, scenario pipeline flexibility, inflow history, external scenarios. Redesigned state_variables, block_mode, AR storage, correlation naming. Discount rate as annual with auto-conversion."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.7-§3.12"
  - date: 2026-02-15
    description: "Major review: added season_definitions, policy_graph with annual discount rate, scenario_source and pipeline flexibility, inflow_history schema, external scenario schema. Redesigned state_variables as boolean-flag object, block_mode per stage, AR coefficient logical schema. Renamed correlation blocks to correlation_groups. Fixed format rationale labels. Added block-hour and season-duration validation rules."
  - date: 2026-02-17
    description: "Format propagation: split inflow_models into inflow_seasonal_stats.parquet (with ar_order column) + inflow_ar_coefficients.parquet (long-form). Renamed load_models to load_seasonal_stats.parquet. All tabular files now .parquet. Removed §5 Exchange Factors (moved to input-constraints.md). Renumbered §6→§5 Correlation, §7→§6 Seasonal Override. Embedded correlation schedule in correlation.json as schedule array. Closed all format TBDs."
---

# Input Scenarios and Time Series

## Purpose

This spec defines the temporal structure (stages, seasons, blocks), policy graph (transitions, discounting), stochastic scenario pipeline (inflow history, uncertainty models, scenario sources), block-level scaling factors, and spatial correlation inputs. These inputs control how POWE.RS decomposes time, generates or consumes scenarios, and correlates random variables across the system.

For initial conditions (which bootstrap the stochastic process), see [Input Constraints](input-constraints.md) §1. For the PAR inflow model mathematics, see [PAR Inflow Model](../01-math/par-inflow-model.md). For discount rate mathematics, see [Discount Rate Formulation](../01-math/discount-rate.md).

## 1. Stage Definitions (`stages.json`)

> **Format Rationale — stages.json**
>
> **Complex nested object** — Stage definitions with nested block structures, season definitions, policy graph, scenario configuration, and risk parameters. JSON handles hierarchical config naturally.

> **⚠️ Order Invariance**: The order of stages and blocks in their arrays does NOT affect results. After loading, stages are sorted by `id`, and blocks within each stage are sorted by `id`. See [Design Principles §3](../00-overview/design-principles.md).

### 1.1 Season Definitions

The `season_definitions` section formally maps each `season_id` to a calendar period. This mapping is required whenever the system needs to aggregate inflow history into season-level values (see §2).

```json
{
  "season_definitions": {
    "cycle_type": "monthly",
    "seasons": [
      { "id": 0, "month_start": 1, "label": "January" },
      { "id": 1, "month_start": 2, "label": "February" },
      { "id": 2, "month_start": 3, "label": "March" },
      { "id": 3, "month_start": 4, "label": "April" },
      { "id": 4, "month_start": 5, "label": "May" },
      { "id": 5, "month_start": 6, "label": "June" },
      { "id": 6, "month_start": 7, "label": "July" },
      { "id": 7, "month_start": 8, "label": "August" },
      { "id": 8, "month_start": 9, "label": "September" },
      { "id": 9, "month_start": 10, "label": "October" },
      { "id": 10, "month_start": 11, "label": "November" },
      { "id": 11, "month_start": 12, "label": "December" }
    ]
  }
}
```

| `cycle_type` | Meaning                             | Season count | Calendar rule                                                               |
| ------------ | ----------------------------------- | ------------ | --------------------------------------------------------------------------- |
| `monthly`    | Each season = one calendar month    | 12           | `month_start` maps to the calendar month                                    |
| `weekly`     | Each season = one ISO calendar week | 52           | Season `id` maps to ISO week number                                         |
| `custom`     | User-defined date ranges            | any          | Each season has explicit `month_start`, `day_start`, `month_end`, `day_end` |

For `custom` cycle type, each season requires explicit date boundaries:

```json
{
  "cycle_type": "custom",
  "seasons": [
    {
      "id": 0,
      "month_start": 1,
      "day_start": 1,
      "month_end": 4,
      "day_end": 1,
      "label": "Q1"
    },
    {
      "id": 1,
      "month_start": 4,
      "day_start": 1,
      "month_end": 7,
      "day_end": 1,
      "label": "Q2"
    },
    {
      "id": 2,
      "month_start": 7,
      "day_start": 1,
      "month_end": 10,
      "day_end": 1,
      "label": "Q3"
    },
    {
      "id": 3,
      "month_start": 10,
      "day_start": 1,
      "month_end": 1,
      "day_end": 1,
      "label": "Q4"
    }
  ]
}
```

**Validation rules:**

- Each stage's `[start_date, end_date)` interval must fall entirely within the calendar period defined by its `season_id`.
- All stages sharing the same `season_id` must have exactly the same duration. This ensures PAR parameters are truly periodic and history aggregation produces comparable values across years.

**When required:** `season_definitions` is required whenever `inflow_history` is provided (see §2.3). Otherwise it is optional but recommended for validation and reporting.

### 1.2 Policy Graph and Transitions

The `policy_graph` section defines the graph structure of stage transitions, the horizon type, and the global discount rate.

```json
{
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 1, "target_id": 2, "probability": 1.0 }
    ]
  }
}
```

#### Policy Graph Types

| Type             | Description                                                                                                                                                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `finite_horizon` | Linear chain of stages with a terminal condition. The simplest and most common structure.                                                                                                                                     |
| `cyclic`         | Stages form a cycle (e.g., stage 59 transitions back to stage 48). Used for infinite periodic horizon. Requires `annual_discount_rate > 0` for convergence. See [Discount Rate Formulation §15](../01-math/discount-rate.md). |

#### Discount Rate

The `annual_discount_rate` is specified as a yearly rate (e.g., `0.06` = 6% per year). The system automatically converts this to a per-transition discount factor based on each stage's duration:

- Stage duration `Δt` is derived from `end_date - start_date`, expressed in years
- Transition discount factor: `β = 1 / (1 + annual_discount_rate) ^ Δt`
- The duration used is that of the **source** stage (the stage whose future cost is being discounted)

A value of `0.0` means no discounting (β = 1.0 for all transitions).

Individual transitions may override the global rate:

```json
{
  "source_id": 59,
  "target_id": 48,
  "probability": 1.0,
  "annual_discount_rate": 0.1
}
```

Per-transition overrides follow the same annual-rate-to-factor conversion.

#### Transition Fields

| Field                  | Type | Required | Default      | Description                                         |
| ---------------------- | ---- | -------- | ------------ | --------------------------------------------------- |
| `source_id`            | i32  | Yes      | —            | Source stage ID                                     |
| `target_id`            | i32  | Yes      | —            | Target stage ID                                     |
| `probability`          | f64  | Yes      | —            | Transition probability (must sum to 1.0 per source) |
| `annual_discount_rate` | f64  | No       | Global value | Override annual discount rate for this transition   |

### 1.3 Pre-Study Stages

Stages with negative IDs represent historical periods before the study horizon. Used only for PAR model initialization (providing lag values). Pre-study stages only need `id`, `start_date`, and `end_date`.

```json
{
  "pre_study_stages": [
    { "id": -6, "start_date": "2023-07-01", "end_date": "2023-08-01" },
    { "id": -5, "start_date": "2023-08-01", "end_date": "2023-09-01" },
    { "id": -4, "start_date": "2023-09-01", "end_date": "2023-10-01" },
    { "id": -3, "start_date": "2023-10-01", "end_date": "2023-11-01" },
    { "id": -2, "start_date": "2023-11-01", "end_date": "2023-12-01" },
    { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" }
  ]
}
```

### 1.4 Stage Fields

Each stage defines its temporal extent, block structure, scenario configuration, and risk parameters.

| Field             | Type          | Required | Default             | Description                                                                               |
| ----------------- | ------------- | -------- | ------------------- | ----------------------------------------------------------------------------------------- |
| `id`              | i32           | Yes      | —                   | Unique stage identifier (non-negative)                                                    |
| `start_date`      | string        | Yes      | —                   | Stage start date (ISO 8601)                                                               |
| `end_date`        | string        | Yes      | —                   | Stage end date (ISO 8601)                                                                 |
| `season_id`       | i32 \| null   | No       | null                | Season index linking to `season_definitions`. Null for stages without seasonal structure. |
| `blocks`          | array         | Yes      | —                   | Load blocks within stage (see §1.5)                                                       |
| `block_mode`      | string        | No       | `"parallel"`        | Block formulation: `"parallel"` or `"chronological"` (see §1.5)                           |
| `state_variables` | object        | No       | `{"storage": true}` | State variable configuration (see §1.6)                                                   |
| `risk_measure`    | string/object | No       | `"expectation"`     | Risk measure: `"expectation"` or `{"cvar": {...}}` (see §1.7)                             |
| `num_scenarios`   | i32           | Yes      | —                   | Number of scenarios for this stage                                                        |
| `sampling_method` | string        | No       | `"saa"`             | Sampling method (see §1.8)                                                                |

### 1.5 Blocks and Block Mode

Each stage contains one or more load blocks. Block IDs within each stage must be contiguous starting at 0 (validated: 0, 1, 2, …, n-1).

```json
{
  "blocks": [
    { "id": 0, "name": "LEVE", "hours": 168 },
    { "id": 1, "name": "MEDIA", "hours": 336 },
    { "id": 2, "name": "PESADA", "hours": 168 }
  ]
}
```

Block weights are computed internally from block hours.

**Validation rule:** The sum of all block hours within a stage must equal the total stage duration (derived from `end_date - start_date` converted to hours).

The `block_mode` field controls the block formulation for each stage:

| Mode              | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| `"parallel"`      | Blocks are independent sub-periods solved simultaneously within the stage (default). |
| `"chronological"` | Blocks are sequential within the stage, with inter-block state transitions.          |

Block mode can vary by stage, allowing adaptive strategies (e.g., chronological for near-term stages, parallel for distant stages). For the mathematical formulation of each mode, see [Block Formulations](../01-math/block-formulations.md).

### 1.6 State Variables

The `state_variables` field is an object with boolean flags indicating which variables carry state between stages:

```json
{
  "state_variables": {
    "storage": true,
    "inflow_lags": true
  }
}
```

| Flag          | Description                                    | Default |
| ------------- | ---------------------------------------------- | ------- |
| `storage`     | Reservoir storage volumes                      | `true`  |
| `inflow_lags` | Past inflow realizations used as AR model lags | `false` |

Storage is mandatory in most applications but kept as an explicit flag for transparency. Future extensions may add additional flags (e.g., `gnl_pipeline` for gas network state).

### 1.7 Risk Measure (CVaR)

The `risk_measure` field can be:

| Option            | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `"expectation"`   | Risk-neutral expected value (default)                                 |
| `{"cvar": {...}}` | CVaR parameters with `alpha` (confidence level) and `lambda` (weight) |

CVaR details: `alpha` = confidence level (e.g., 0.95 means 5% worst scenarios); `lambda` = weight of CVaR vs expectation. Final risk measure: `(1 - lambda) × E[cost] + lambda × CVaR_alpha[cost]`. CVaR parameters can vary by stage. See [Risk Measures](../01-math/risk-measures.md) for mathematical formulation.

### 1.8 Scenario Sampling Methods

| Method       | Description                                                                   | Use Case                             |
| ------------ | ----------------------------------------------------------------------------- | ------------------------------------ |
| `saa`        | **Sample Average Approximation** (default). Pure Monte Carlo random sampling. | General purpose, baseline            |
| `lhs`        | **Latin Hypercube Sampling**. Stratified sampling ensuring uniform coverage.  | Medium sample sizes (20–100)         |
| `qmc_sobol`  | **Quasi-Monte Carlo (Sobol sequences)**. Low-discrepancy sequences.           | High-dimensional, deterministic-like |
| `qmc_halton` | **Quasi-Monte Carlo (Halton sequences)**. Alternative low-discrepancy.        | Similar to Sobol                     |
| `selective`  | **Selective/Representative Sampling**. Clustering on historical data.         | Historical pattern-guided            |

Sampling method can vary by stage, allowing adaptive strategies.

### 1.9 Example

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/stages.schema.json",
  "season_definitions": {
    "cycle_type": "monthly",
    "seasons": [
      { "id": 0, "month_start": 1, "label": "January" },
      { "id": 1, "month_start": 2, "label": "February" },
      { "id": 2, "month_start": 3, "label": "March" }
    ]
  },
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 1, "target_id": 2, "probability": 1.0 }
    ]
  },
  "scenario_source": {
    "type": "generated",
    "seed": 42
  },
  "pre_study_stages": [
    { "id": -6, "start_date": "2023-07-01", "end_date": "2023-08-01" },
    { "id": -5, "start_date": "2023-08-01", "end_date": "2023-09-01" },
    { "id": -4, "start_date": "2023-09-01", "end_date": "2023-10-01" },
    { "id": -3, "start_date": "2023-10-01", "end_date": "2023-11-01" },
    { "id": -2, "start_date": "2023-11-01", "end_date": "2023-12-01" },
    { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" }
  ],
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "season_id": 0,
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 248 },
        { "id": 1, "name": "MEDIA", "hours": 248 },
        { "id": 2, "name": "PESADA", "hours": 248 }
      ],
      "block_mode": "chronological",
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.5 } },
      "state_variables": { "storage": true, "inflow_lags": true },
      "num_scenarios": 20,
      "sampling_method": "lhs"
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "season_id": 1,
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 232 },
        { "id": 1, "name": "MEDIA", "hours": 232 },
        { "id": 2, "name": "PESADA", "hours": 232 }
      ],
      "block_mode": "parallel",
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.25 } },
      "state_variables": { "storage": true, "inflow_lags": true },
      "num_scenarios": 20,
      "sampling_method": "lhs"
    },
    {
      "id": 2,
      "start_date": "2024-03-01",
      "end_date": "2024-04-01",
      "season_id": 2,
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 248 },
        { "id": 1, "name": "MEDIA", "hours": 248 },
        { "id": 2, "name": "PESADA", "hours": 248 }
      ],
      "risk_measure": "expectation",
      "state_variables": { "storage": true },
      "num_scenarios": 20,
      "sampling_method": "saa"
    }
  ]
}
```

> **Note:** The `$schema` field is a placeholder. No live schema URL exists yet. All JSON examples in this spec and other approved specs use placeholder `$schema` values for future JSON Schema validation support.

### 1.10 Validation Rules

1. Stage IDs must be unique and non-negative.
2. Block IDs within each stage must be contiguous starting at 0.
3. Block hours within each stage must sum to the total stage duration.
4. All stages sharing the same `season_id` must have exactly the same duration.
5. Each stage's `[start_date, end_date)` must fall within the calendar period defined by its `season_id` in `season_definitions`.
6. Transition probabilities must sum to 1.0 per source stage.
7. For `cyclic` policy graphs, the cumulative discount factor around each cycle must be strictly less than 1.0.

## 2. Scenario Pipeline

### 2.1 Scenario Source

The top-level `scenario_source` field in `stages.json` controls how inflow scenarios are produced for the SDDP forward pass:

```json
{ "scenario_source": { "type": "generated", "seed": 42 } }
```

| Type         | Description                                               | Required inputs                                                 |
| ------------ | --------------------------------------------------------- | --------------------------------------------------------------- |
| `generated`  | AR model generates scenarios from sampled noise (default) | Uncertainty models (§3) — user-provided or derived from history |
| `historical` | Replay actual historical inflow sequences                 | `inflow_history` + `season_definitions` (always)                |
| `external`   | User provides pre-computed scenario values per stage      | External scenario file indexed by `stage_id` (§2.5)             |

For `historical` and `external` sources, the system performs **reverse noise calculation**: back-computing the noise vector ε that would produce the given inflow values through the AR model. This is necessary because SDDP cuts are constructed in terms of state variables and the AR noise structure. This is an internal solver computation, not a data input concern.

### 2.2 Pipeline Flexibility

The scenario pipeline is a cascade of components, each of which can independently be **user-provided** or **derived from inflow history**:

| Component                  | User-provided via                       | Derived from                             |
| -------------------------- | --------------------------------------- | ---------------------------------------- |
| Seasonal statistics (μ, σ) | `inflow_seasonal_stats.parquet` (§3.1)  | Inflow history aggregated by season      |
| AR coefficients (ψ)        | `inflow_ar_coefficients.parquet` (§3.2) | Fitted from inflow history (Yule-Walker) |
| Correlation matrices       | `correlation.json` (§5)                 | Estimated from AR residuals of history   |

**Presence or absence of input files controls the pipeline.** No explicit mode flags are needed:

| `inflow_seasonal_stats` | `inflow_ar_coefficients` | `correlation.json` | `inflow_history` | System behavior                                                       |
| ----------------------- | ------------------------ | ------------------ | ---------------- | --------------------------------------------------------------------- |
| present                 | present                  | present            | —                | Use all directly. No history needed.                                  |
| present                 | present                  | absent             | present          | Use AR models directly. Estimate correlations from history.           |
| present                 | absent                   | present            | present          | Use seasonal stats directly. Fit AR from history. Use correlations.   |
| present                 | absent                   | absent             | present          | Use seasonal stats. Fit AR and correlations from history.             |
| absent                  | absent                   | present            | present          | Fit seasonal stats + AR from history. Use provided correlations.      |
| absent                  | absent                   | absent             | present          | Derive everything from history.                                       |
| present                 | absent                   | —                  | absent           | No AR structure (AR order = 0 for all). Only seasonal stats apply.    |
| absent                  | present                  | —                  | —                | **Error** — AR coefficients require seasonal stats for normalization. |
| absent                  | absent                   | —                  | absent           | **Error** — no stochastic model possible.                             |

All combinations of user-provided and derived components are valid. For example, a user may provide AR coefficients but override seasonal means (μ) to force conditioned inflow regimes, while letting the system estimate correlations from history.

**When any component is derived from history**, the system requires `inflow_history` and `season_definitions` to aggregate raw observations into season-level values.

### 2.3 History Aggregation

The user can provide inflow history at any time resolution (daily, weekly, monthly, or other). The system aggregates observations to match the season resolution defined in `season_definitions`:

- **Finer resolution → season**: The system averages all observations within each season's calendar period. For example, daily observations are averaged across each calendar month for `monthly` seasons. Weekly observations spanning a season boundary are included via weighted average based on overlap days.
- **Same resolution**: Direct mapping, no aggregation needed.
- **Coarser resolution → season**: Error. The system cannot disaggregate observations into finer seasons without additional assumptions.

### 2.4 Inflow History (`scenarios/inflow_history.parquet`)

The `inflow_history` file contains raw historical inflow observations at the user's chosen time resolution.

> **Format Rationale — inflow_history.parquet**
>
> **Entity-level time series** — Historical observations per hydro indexed by date. Parquet for efficient columnar access across potentially thousands of rows (hydros x dates).

| Column      | Type | Description                                     |
| ----------- | ---- | ----------------------------------------------- |
| `hydro_id`  | i32  | Hydro plant ID (must exist in system entities)  |
| `date`      | date | Start date of the observation period (ISO 8601) |
| `value_m3s` | f64  | Mean inflow for the period (m³/s)               |

The resolution of the history data must be declared explicitly via the `inflow_history` configuration:

```json
{
  "inflow_history": {
    "resolution": "daily",
    "path": "inflow_history.parquet"
  }
}
```

| `resolution` | Meaning                                      |
| ------------ | -------------------------------------------- |
| `daily`      | One observation per day per hydro            |
| `weekly`     | One observation per ISO week per hydro       |
| `monthly`    | One observation per calendar month per hydro |

Declaring the resolution explicitly (rather than inferring it from date intervals) ensures deterministic validation and aggregation. The system knows exactly what intervals to expect and can flag missing or duplicate records without guessing.

**Usage contexts:**

1. **Deriving seasonal statistics** — Compute μ, σ per hydro per season.
2. **Fitting AR models** — Estimate ψ coefficients via Yule-Walker equations. See [PAR Inflow Model](../01-math/par-inflow-model.md).
3. **Estimating correlations** — Compute cross-correlation from AR model residuals.
4. **Historical scenario replay** — When `scenario_source.type = "historical"`, forward passes use actual historical sequences mapped to stages via `season_definitions`.

### 2.5 External Scenarios (`scenarios/external_scenarios.parquet`)

When `scenario_source.type = "external"`, the user provides pre-computed scenario values indexed directly by `stage_id`. This eliminates any need for season-calendar mapping — the user is responsible for ensuring the values match the stage structure.

> **Format Rationale — external_scenarios.parquet**
>
> **Stage-indexed scenario table** — Pre-computed values per stage, scenario, and entity. Parquet for large scenario trees with efficient columnar access.

| Column        | Type | Description                                |
| ------------- | ---- | ------------------------------------------ |
| `stage_id`    | i32  | Stage ID (must exist in `stages.json`)     |
| `scenario_id` | i32  | Scenario index (0-based)                   |
| `hydro_id`    | i32  | Hydro plant ID                             |
| `value_m3s`   | f64  | Inflow value for this stage/scenario/hydro |

**Validation:** The number of distinct `scenario_id` values per stage must equal the stage's `num_scenarios`.

## 3. Uncertainty Models

### 3.1 Inflow Seasonal Statistics (`scenarios/inflow_seasonal_stats.parquet`)

> **Format Rationale — inflow_seasonal_stats.parquet**
>
> **Entity-stage parameter table** — Per-entity-per-stage seasonal statistics (mean and standard deviation). Parquet for typed columnar access across hydros and stages.

When provided, this table supplies pre-computed seasonal mean and standard deviation directly. When absent, the system derives these from `inflow_history` via season aggregation (see §2.2).

| Column     | Type | Description                                                                                                                                                                                                                                                                                        |
| ---------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hydro_id` | i32  | Hydro plant ID (must exist in system entities)                                                                                                                                                                                                                                                     |
| `stage_id` | i32  | Stage ID (must exist in `stages.json`)                                                                                                                                                                                                                                                             |
| `mean_m3s` | f64  | Seasonal mean inflow (μ)                                                                                                                                                                                                                                                                           |
| `std_m3s`  | f64  | Seasonal sample standard deviation ($s_m$). 0 = deterministic. Note: this is the sample std of historical observations, NOT the residual std ($\sigma_m$) — the residual std is computed at runtime from $s_m$ and the AR coefficients. See [PAR Inflow Model §3](../01-math/par-inflow-model.md). |
| `ar_order` | i32  | AR order for this (hydro, stage). Used for cross-validation with `inflow_ar_coefficients.parquet` — must match the number of coefficient rows.                                                                                                                                                     |

The `ar_order` column serves two purposes:

1. **Cross-validation**: When `inflow_ar_coefficients.parquet` is also provided, the system validates that each (hydro_id, stage_id) has exactly `ar_order` coefficient rows.
2. **Self-documenting**: When AR coefficients are absent (AR order 0 for all), the column makes the intent explicit.

An `ar_order` of 0 means independent noise — no AR structure for that (hydro, stage). No corresponding rows should exist in `inflow_ar_coefficients.parquet`.

### 3.2 Inflow AR Coefficients (`scenarios/inflow_ar_coefficients.parquet`) — Optional

> **Format Rationale — inflow_ar_coefficients.parquet**
>
> **Entity-stage-lag parameter table** — Long-form table of AR coefficients with one row per (hydro, stage, lag). Parquet for typed columnar access; long-form avoids null columns and imposes no maximum AR order.

When provided, this table supplies pre-computed AR coefficients. When absent, the system either fits AR coefficients from `inflow_history` (if present) or uses AR order 0 (independent noise).

The AR model for a given stage uses lags from previous stages. Coefficients are in **original units** (not standardized) — the natural output of standard fitting tools such as Yule-Walker. POWE.RS handles any internal transformations needed for the SDDP algorithm (e.g., reverse-standardization for residual std computation). Innovation terms (ε) are standard normal, transformed into correlated samples via Cholesky decomposition of the correlation matrix (see §5).

| Column        | Type | Description                                                                    |
| ------------- | ---- | ------------------------------------------------------------------------------ |
| `hydro_id`    | i32  | Hydro plant ID (must exist in system entities)                                 |
| `stage_id`    | i32  | Stage ID (must exist in `stages.json`)                                         |
| `lag`         | i32  | Lag index (1-based: 1 = first lag, 2 = second, etc.)                           |
| `coefficient` | f64  | AR coefficient $\psi_{m,\ell}$ for this lag (original units, not standardized) |

**Example rows** (hydro 0, stage 5, AR order 3):

| hydro_id | stage_id | lag | coefficient |
| -------- | -------- | --- | ----------- |
| 0        | 5        | 1   | 0.45        |
| 0        | 5        | 2   | 0.22        |
| 0        | 5        | 3   | 0.08        |

**Validation rules:**

- Lags must be contiguous starting at 1: `[1, 2, ..., p]` for AR order `p`.
- The number of rows per (hydro_id, stage_id) must match the `ar_order` in `inflow_seasonal_stats.parquet`.
- AR coefficients present without seasonal stats = **error** (AR needs μ, σ for normalization).

See [PAR Inflow Model](../01-math/par-inflow-model.md) for the mathematical formulation.

### 3.3 Load Seasonal Statistics (`scenarios/load_seasonal_stats.parquet`)

> **Format Rationale — load_seasonal_stats.parquet**
>
> **Entity-stage parameter table** — Per-bus-per-stage load statistics. Parquet for typed columnar data consistent with other tabular inputs.

| Column     | Type | Description                            |
| ---------- | ---- | -------------------------------------- |
| `bus_id`   | i32  | Bus ID (must exist in system entities) |
| `stage_id` | i32  | Stage ID (must exist in `stages.json`) |
| `mean_mw`  | f64  | Mean load for this stage               |
| `std_mw`   | f64  | Standard deviation (0 = deterministic) |

Load models are typically independent (no AR structure), so no AR columns are included.

## 4. Load Factors by Block — Optional

> **Format Rationale — load_factors.json**
>
> **Default-with-overrides** — Small number of load factor definitions that rarely change. JSON for readability and simplicity.

If missing, all block factors default to 1.0.

Load uncertainty generates a base load realization in **MW** per stage. Block factors are **multipliers** applied to the stochastic load value. For example, if stage load is 1000 MW and block factors are [0.85, 1.00, 1.15], the blocks get [850, 1000, 1150] MW.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/load_factors.schema.json",
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.85 },
        { "block_id": 1, "factor": 1.0 },
        { "block_id": 2, "factor": 1.15 }
      ]
    }
  ]
}
```

## 5. Correlation (`scenarios/correlation.json`)

> **Format Rationale — correlation.json**
>
> **Correlation / matrix data** — Symmetric correlation matrices between entities. JSON because data is small and structure is not tabular. Profile-based design avoids element-wise storage.

Defines spatial correlation between stochastic processes (inflows, loads, non-controllable generation). The system uses Cholesky decomposition to transform independent standard normal samples into correlated samples.

When provided, correlation matrices are used directly. When absent and `inflow_history` is available, the system estimates correlations from AR model residuals (see §2.2).

### 5.1 Profile-Based Time-Varying Correlation

Instead of storing element-wise overrides (O(stages × entities²) rows), POWE.RS uses a **profile-based system**:

1. **Named profiles** — Define multiple correlation matrices (e.g., `"default"`, `"wet_season"`, `"dry_season"`)
2. **Schedule table** — A compact tabular file maps each stage to a profile name

This reduces storage from potentially millions of rows to ~T rows plus a few matrix definitions.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/correlation.schema.json",
  "method": "cholesky",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "southeast_cascade",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 },
            { "type": "inflow", "id": 2 }
          ],
          "matrix": [
            [1.0, 0.75, 0.6],
            [0.75, 1.0, 0.7],
            [0.6, 0.7, 1.0]
          ]
        }
      ]
    },
    "wet_season": {
      "correlation_groups": [
        {
          "name": "southeast_cascade",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 },
            { "type": "inflow", "id": 2 }
          ],
          "matrix": [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0]
          ]
        }
      ]
    }
  }
}
```

### 5.2 Correlation Profile Fields

| Field                                           | Type   | Required | Description                                              |
| ----------------------------------------------- | ------ | -------- | -------------------------------------------------------- |
| `method`                                        | string | Yes      | Correlation method: `"cholesky"` (only supported method) |
| `profiles`                                      | object | Yes      | Map of profile names to correlation group definitions    |
| `profiles.<name>.correlation_groups`            | array  | Yes      | Array of correlation groups for this profile             |
| `profiles.<name>.correlation_groups[].name`     | string | Yes      | Unique name for correlation group                        |
| `profiles.<name>.correlation_groups[].entities` | array  | Yes      | Entities in this correlation group                       |
| `profiles.<name>.correlation_groups[].matrix`   | array  | Yes      | Correlation matrix (must be positive semi-definite)      |

The profile named `"default"` is required and used for any stage not explicitly mapped in the schedule.

### 5.3 Time-Varying Correlation Schedule — Optional

The correlation schedule is **embedded in `correlation.json`** as a `"schedule"` array. Each entry maps a stage to a named profile. If the schedule is absent or a stage is not listed, the `"default"` profile is used.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/correlation.schema.json",
  "method": "cholesky",
  "profiles": {
    "default": { "...": "..." },
    "wet_season": { "...": "..." },
    "dry_season": { "...": "..." }
  },
  "schedule": [
    { "stage_id": 0, "profile_name": "wet_season" },
    { "stage_id": 1, "profile_name": "wet_season" },
    { "stage_id": 4, "profile_name": "default" },
    { "stage_id": 5, "profile_name": "dry_season" }
  ]
}
```

| Field          | Type   | Description                                                  |
| -------------- | ------ | ------------------------------------------------------------ |
| `stage_id`     | i32    | Stage ID                                                     |
| `profile_name` | string | Profile name (must exist in `profiles` within the same file) |

Stages not listed in the schedule use the `"default"` profile. Only stages that deviate from the default need to be listed.

### 5.4 Validation

1. All profile names in the schedule must exist in `correlation.json`.
2. All correlation matrices must be positive semi-definite.
3. Entity IDs in correlation groups must exist in the system.

### 5.5 Correlation Input Options Summary

| Approach             | Files Required                                                          | Use Case                           |
| -------------------- | ----------------------------------------------------------------------- | ---------------------------------- |
| Static correlation   | `correlation.json` with only `"default"` profile                        | Same correlation for all stages    |
| Seasonal correlation | `correlation.json` with multiple profiles + embedded `"schedule"` array | Different profiles by season/stage |
| Derived from history | `inflow_history` (no `correlation.json`)                                | System estimates from AR residuals |

## 6. Seasonal Override Pattern (Cross-Cutting)

Several data model elements exhibit the same pattern: a value or configuration that varies by season or stage. This appears in production model selection, load factors, exchange factors, and correlation profiles.

Two approaches have been identified for this pattern:

### 6.1 Profile + Schedule

Define named profiles (complete configurations) and a separate schedule table that maps stages to profile names. For correlation, the schedule is embedded in the same JSON file (see §5.3).

**Strengths:** Clean separation of definitions and temporal assignment. Profiles are reusable. Schedule table is tiny. Good for complex objects (correlation matrices, production models).

**Weaknesses:** Requires two files per concept. Indirection may be confusing for simple cases.

**Used in:** Correlation (§5), production model selection (see [Input Hydro Extensions](input-hydro-extensions.md)).

### 6.2 Stage/Season Tagged Union

Include the varying parameter directly in each stage definition or in a per-stage table. The value is a tagged union selecting between variants.

**Strengths:** Self-contained — no external schedule file. Good for simple variant selection (e.g., `block_mode`, `risk_measure`).

**Weaknesses:** Repetitive for large stage counts. Doesn't scale for complex objects.

**Used in:** `block_mode` (§1.5), `risk_measure` (§1.7), `state_variables` (§1.6).

The final decision on which approach to use for each element will be made during implementation. Both are valid and may coexist.

## Cross-References

- [Input Constraints](input-constraints.md) — Initial conditions (§1) that bootstrap the stochastic process; time-varying entity bounds (§2)
- [Input System Entities](input-system-entities.md) — Buses and hydros referenced by uncertainty models
- [Input Directory Structure](input-directory-structure.md) — Overall case directory layout
- [PAR Inflow Model](../01-math/par-inflow-model.md) — Mathematical formulation of the AR inflow model
- [Risk Measures](../01-math/risk-measures.md) — CVaR mathematical formulation
- [Block Formulations](../01-math/block-formulations.md) — How blocks partition each stage and parallel vs chronological modes
- [Discount Rate Formulation](../01-math/discount-rate.md) — Discount factor mathematics and infinite periodic horizon
- [Design Principles §3](../00-overview/design-principles.md) — Order invariance and canonical ordering
