---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.7 (Stage Definitions — stages.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.8 (Uncertainty Models — inflow_models.parquet)"
  - "DATA_MODEL_SPECIFICATION.md §3.10 (Load Factors — load_factors.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.11 (Exchange Factors — exchange_factors.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.12 (Correlation — correlation.json)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.7-§3.12"
---

# Input Scenarios and Time Series

## Purpose

This spec defines the temporal structure (stages and blocks), stochastic uncertainty models (inflow and load), block-level scaling factors, and spatial correlation inputs. These files control how POWE.RS decomposes time, generates scenarios, and correlates random variables across the system.

For initial conditions (which bootstrap the stochastic process), see [Input Constraints](input-constraints.md) §1. For the PAR inflow model mathematics, see [PAR Inflow Model](../01-math/par-inflow-model.md).

## 1. Stage Definitions (`stages.json`)

> **Format Rationale — stages.json**
>
> **Complex nested object** — Stage definitions with nested block structures, optional seasonal risk parameters, and sampling configuration. JSON handles hierarchical config naturally.

> **⚠️ Order Invariance**: The order of stages and blocks in their arrays does NOT affect results. After loading, stages are sorted by `id`, and blocks within each stage are sorted by `id`. See [Design Principles §3](../00-overview/design-principles.md).

Each stage defines its own blocks (count and hours). Block IDs within each stage must be contiguous starting at 0 (validated: 0, 1, 2, …, n-1). Block weights are computed internally from block hours.

**Pre-study stages**: Stages with negative IDs represent historical periods before the study horizon. Used only for PAR model initialization (providing lag values). Pre-study stages only need `id`, `start_date`, and `end_date`.

### Risk Measure (CVaR)

The `risk_measure` field can be:

| Option            | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `"expectation"`   | Risk-neutral expected value (default)                                 |
| `{"cvar": {...}}` | CVaR parameters with `alpha` (confidence level) and `lambda` (weight) |

CVaR details: `alpha` = confidence level (e.g., 0.95 means 5% worst scenarios); `lambda` = weight of CVaR vs expectation. Final risk measure: `(1 - lambda) × E[cost] + lambda × CVaR_alpha[cost]`. CVaR parameters can vary by stage. See [Risk Measures](../01-math/risk-measures.md) for mathematical formulation.

### Scenario Sampling Methods

| Method       | Description                                                                          | Use Case                             |
| ------------ | ------------------------------------------------------------------------------------ | ------------------------------------ |
| `saa`        | **Sample Average Approximation** (default). Pure Monte Carlo random sampling.        | General purpose, baseline            |
| `lhs`        | **Latin Hypercube Sampling**. Stratified sampling ensuring uniform coverage.         | Medium sample sizes (20–100)         |
| `qmc_sobol`  | **Quasi-Monte Carlo (Sobol sequences)**. Low-discrepancy sequences.                  | High-dimensional, deterministic-like |
| `qmc_halton` | **Quasi-Monte Carlo (Halton sequences)**. Alternative low-discrepancy.               | Similar to Sobol                     |
| `selective`  | **Selective/Representative Sampling**. Clustering on historical data.                | Historical pattern-guided            |
| `historical` | **Historical Scenarios**. Actual historical sequences from `inflow_history.parquet`. | Backtesting, deterministic studies   |

Sampling method can vary by stage, allowing adaptive strategies.

### Example

```json
{
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
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 168 },
        { "id": 1, "name": "MEDIA", "hours": 336 },
        { "id": 2, "name": "PESADA", "hours": 168 }
      ],
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.5 } },
      "state_variables": "storage_and_inflow",
      "num_scenarios": 20,
      "sampling_method": "lhs"
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 168 },
        { "id": 1, "name": "MEDIA", "hours": 336 },
        { "id": 2, "name": "PESADA", "hours": 168 }
      ],
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.25 } },
      "state_variables": "storage_and_inflow",
      "num_scenarios": 20,
      "sampling_method": "lhs"
    },
    {
      "id": 2,
      "start_date": "2024-03-01",
      "end_date": "2024-04-01",
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 168 },
        { "id": 1, "name": "MEDIA", "hours": 336 },
        { "id": 2, "name": "PESADA", "hours": 168 }
      ],
      "risk_measure": "expectation",
      "state_variables": "storage_and_inflow",
      "num_scenarios": 20,
      "sampling_method": "saa"
    }
  ],
  "transitions": [
    {
      "source_id": 0,
      "target_id": 1,
      "probability": 1.0,
      "discount_rate": 0.0
    },
    { "source_id": 1, "target_id": 2, "probability": 1.0, "discount_rate": 0.0 }
  ]
}
```

### Stage Field Reference

| Field             | Type          | Required | Default          | Description                                        |
| ----------------- | ------------- | -------- | ---------------- | -------------------------------------------------- |
| `id`              | i32           | Yes      | —                | Unique stage identifier                            |
| `start_date`      | string        | Yes      | —                | Stage start date (ISO 8601)                        |
| `end_date`        | string        | Yes      | —                | Stage end date (ISO 8601)                          |
| `blocks`          | array         | Yes      | —                | Load blocks within stage                           |
| `risk_measure`    | string/object | No       | `"expectation"`  | Risk measure: `"expectation"` or `{"cvar": {...}}` |
| `state_variables` | string        | No       | `"storage_only"` | State configuration                                |
| `num_scenarios`   | i32           | Yes      | —                | Number of scenarios for this stage                 |
| `sampling_method` | string        | No       | `"saa"`          | Sampling method (see table above)                  |

## 2. Uncertainty Models (`scenarios/inflow_models.parquet`)

> **Format Rationale — inflow_models.parquet**
>
> **Time series** — Large per-entity-per-stage tabular data (PAR coefficients, historical inflows). Parquet for columnar compression and typed columns across potentially thousands of rows (hydros × stages).

Uncertainty models are defined per entity per stage in tabular format. This enables variable time resolutions, explicit parameters per stage (no cycling complexity), easy bulk editing, and AR order 0 for independent noise.

The AR model for stage `t` uses lags from previous stages. AR coefficients reference normalized residuals from stages `t-1`, `t-2`, …, `t-order`.

### Inflow Models Schema

| Column       | Type | Description                                    |
| ------------ | ---- | ---------------------------------------------- |
| `hydro_id`   | i32  | Hydro plant ID (must exist in `hydros.json`)   |
| `stage_id`   | i32  | Stage ID (must exist in `stages.json`)         |
| `mean_m3s`   | f64  | Mean inflow for this stage                     |
| `std_m3s`    | f64  | Standard deviation (0 = deterministic)         |
| `ar_order`   | i32  | AR order (0 = independent, 1–12 typical)       |
| `ar_coef_01` | f64  | AR coefficient for lag 1 (null if order < 1)   |
| `ar_coef_02` | f64  | AR coefficient for lag 2 (null if order < 2)   |
| …            | …    | …                                              |
| `ar_coef_12` | f64  | AR coefficient for lag 12 (null if order < 12) |

### Load Models Schema (`scenarios/load_models.parquet`)

| Column     | Type | Description                            |
| ---------- | ---- | -------------------------------------- |
| `bus_id`   | i32  | Bus ID                                 |
| `stage_id` | i32  | Stage ID (must exist in `stages.json`) |
| `mean_mw`  | f64  | Mean load for this stage               |
| `std_mw`   | f64  | Standard deviation (0 = deterministic) |

Load models are typically independent (no AR), so no AR columns are included.

## 3. Load Factors by Block (`scenarios/load_factors.json`) — Optional

> **Format Rationale — load_factors.json**
>
> **Default-with-overrides** — Small number of load factor definitions that rarely change. JSON for readability and simplicity.

If missing, all block factors default to 1.0.

Load uncertainty generates a base load realization in **MW** per stage. Block factors are **multipliers** applied to the stochastic load value. For example, if stage load is 1000 MW and block factors are [0.85, 1.00, 1.15], the blocks get [850, 1000, 1150] MW.

```json
{
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

## 4. Exchange Factors by Block (`scenarios/exchange_factors.json`) — Optional

> **Format Rationale — exchange_factors.json**
>
> **Default-with-overrides** — Small number of exchange factor definitions. JSON for readability.

If missing, all block factors default to 1.0.

Exchange (transmission) limits may vary by block due to thermal limits, contractual constraints, or operational patterns. Block factors are **multipliers** applied to the stage-level line capacity. For example, if a line has 5000 MW direct capacity and block factors are [0.90, 1.00, 1.10], the blocks get [4500, 5000, 5500] MW.

```json
{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.9, "reverse_factor": 0.9 },
        { "block_id": 1, "direct_factor": 1.0, "reverse_factor": 1.0 },
        { "block_id": 2, "direct_factor": 1.1, "reverse_factor": 1.1 }
      ]
    }
  ]
}
```

## 5. Correlation (`scenarios/correlation.json`)

> **Format Rationale — correlation.json**
>
> **Correlation / matrix data** — Symmetric correlation matrices between entities. JSON because data is small and structure is not tabular. Profile-based design avoids element-wise Parquet storage.

Defines spatial correlation between stochastic processes (inflows, loads, non-controllable generation). Uses Cholesky decomposition to transform independent standard normal samples into correlated samples.

### Profile-Based Time-Varying Correlation

Instead of storing element-wise overrides (O(stages × entities²) rows), POWE.RS uses a **profile-based system**:

1. **Named profiles** — Define multiple correlation matrices (e.g., `"default"`, `"wet_season"`, `"dry_season"`)
2. **Schedule table** — A compact Parquet file maps each stage to a profile name

This reduces storage from potentially millions of rows to ~T rows plus a few matrix definitions.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/correlation.schema.json",
  "method": "cholesky",
  "profiles": {
    "default": {
      "blocks": [
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
      "blocks": [
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

### Correlation Profile Fields

| Field                               | Type   | Required | Description                                              |
| ----------------------------------- | ------ | -------- | -------------------------------------------------------- |
| `method`                            | string | Yes      | Correlation method: `"cholesky"` (only supported method) |
| `profiles`                          | object | Yes      | Map of profile names to correlation block definitions    |
| `profiles.<name>.blocks`            | array  | Yes      | Array of correlation blocks for this profile             |
| `profiles.<name>.blocks[].name`     | string | Yes      | Unique name for correlation block                        |
| `profiles.<name>.blocks[].entities` | array  | Yes      | Entities in this correlation group                       |
| `profiles.<name>.blocks[].matrix`   | array  | Yes      | Correlation matrix (must be positive semi-definite)      |

The profile named `"default"` is required and used for any stage not explicitly mapped in the schedule.

### Time-Varying Correlation Schedule (`scenarios/correlation_schedule.parquet`) — Optional

Maps stages to correlation profiles. If missing, all stages use the `"default"` profile.

| Column         | Type   | Description                                     |
| -------------- | ------ | ----------------------------------------------- |
| `stage_id`     | i32    | Stage ID                                        |
| `profile_name` | string | Profile name (must exist in `correlation.json`) |

**Example** (12-month seasonal pattern):

| stage_id | profile_name |
| -------- | ------------ |
| 0        | wet_season   |
| 1        | wet_season   |
| 4        | default      |
| 5        | dry_season   |
| …        | …            |

**Storage comparison** (160 hydros, 60 stages):

| Format           | Storage                                             |
| ---------------- | --------------------------------------------------- |
| Old element-wise | 60 × 160 × 160 / 2 ≈ 768,000 rows                   |
| Profile-based    | 60 rows + ~3 profiles × matrix entries ≈ negligible |

### Validation

1. All profile names in schedule must exist in `correlation.json`
2. All correlation matrices must be positive semi-definite
3. Entity IDs in correlation blocks must exist in the system

### Correlation Input Options Summary

| Approach              | Files Required                                      | Use Case                              |
| --------------------- | --------------------------------------------------- | ------------------------------------- |
| Static correlation    | `correlation.json` with only `"default"` profile    | Same correlation for all stages       |
| Seasonal correlation  | `correlation.json` + `correlation_schedule.parquet` | Different profiles by season/stage    |
| Computed from history | External preprocessing → `correlation.json`         | Derive from inflow history externally |

Computing correlation from historical inflows is outside POWE.RS scope. Users should use external tools (GEVAZP, Python/R statistical packages) to estimate correlation matrices, then provide results in the profile format.

## Cross-References

- [Input Constraints](input-constraints.md) — Initial conditions (§1) that bootstrap the stochastic process; time-varying entity bounds (§2)
- [Input System Entities](input-system-entities.md) — Buses and hydros referenced by uncertainty models
- [Input Directory Structure](input-directory-structure.md) — Overall case directory layout
- [PAR Inflow Model](../01-math/par-inflow-model.md) — Mathematical formulation of the AR inflow model
- [Risk Measures](../01-math/risk-measures.md) — CVaR mathematical formulation
- [Block Model](../01-math/block-formulations.md) — How blocks partition each stage
- [Design Principles §3](../00-overview/design-principles.md) — Order invariance and canonical ordering
