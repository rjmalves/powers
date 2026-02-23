---
status: draft
review_priority: 3-medium
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §18 (18.1-18.10) Configuration-Driven LP Variants"
  - "MATHEMATICAL_FORMULATIONS.md §19 (19.1-19.4) Cross-Reference to Data Model"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
  - date: 2026-02-22
    description: "SDDP.jl refactoring step 6: Added §18.11 Scenario Source and Sampling Scheme with three sampling scheme variants (in_sample, external, historical). Added §18.12 Opening Tree Configuration. Updated §18.10 complete example with scenario_source. Updated §19.1 section mapping with scenario source row. Fixed inflow_models.parquet references to split files (inflow_seasonal_stats.parquet + inflow_ar_coefficients.parquet). Fixed §19.2 variable correspondence. Removed statistical stopping rule (removed per design decision). Fixed discount factor symbol from β to d."
---

# Configuration Reference

## Purpose

This spec provides a comprehensive mapping between POWE.RS configuration options and their effects on the LP subproblem formulation, including variable correspondence tables, Rust struct mappings, and a complete example configuration. It serves as the central reference for understanding how `config.json` and `stages.json` settings alter solver behavior.

## 18.1 Block Mode Configuration

| Option              | Value             | LP Effect                                             | Reference                                              |
| ------------------- | ----------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| modeling.block_mode | `"parallel"`      | Single water balance per stage, averaged generation   | [Block Formulations](../01-math/block-formulations.md) |
| modeling.block_mode | `"chronological"` | Per-block storage variables, sequential water balance | [Block Formulations](../01-math/block-formulations.md) |

## 18.2 Hydro Production Function

| Option                       | Value        | LP Effect                           | Reference                                                        |
| ---------------------------- | ------------ | ----------------------------------- | ---------------------------------------------------------------- |
| modeling.production_function | `"constant"` | Fixed productivity $\rho_h$         | [Hydro Production Models](../01-math/hydro-production-models.md) |
| modeling.production_function | `"fpha"`     | Piecewise-linear head approximation | [Hydro Production Models](../01-math/hydro-production-models.md) |

## 18.3 Inflow Non-Negativity Treatment

| Option                                      | Value                       | LP Effect                                     | Reference                                                   |
| ------------------------------------------- | --------------------------- | --------------------------------------------- | ----------------------------------------------------------- |
| modeling.inflow_non_negativity.method       | `"none"`                    | No slack, may cause infeasibility             | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md) |
| modeling.inflow_non_negativity.method       | `"penalty"`                 | Add $\sigma^{inf}_h$ slack with penalty       | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md) |
| modeling.inflow_non_negativity.method       | `"truncation"`              | Pre-truncate in scenario generation           | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md) |
| modeling.inflow_non_negativity.method       | `"truncation_with_penalty"` | Noise adjustment slack $\xi_h$                | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md) |
| modeling.inflow_non_negativity.penalty_cost | float                       | Penalty coefficient $c^{inf}$ (default: 1000) | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md) |

## 18.4 Cut Management

| Option                         | Value          | LP Effect                  | Reference                                      |
| ------------------------------ | -------------- | -------------------------- | ---------------------------------------------- |
| training.cut_selection.enabled | bool           | Enable/disable cut pruning | [Cut Management](../01-math/cut-management.md) |
| training.cut_selection.method  | `"level1"`     | Keep ever-active cuts      | [Cut Management](../01-math/cut-management.md) |
| training.cut_selection.method  | `"lml1"`       | Limited memory level-1     | [Cut Management](../01-math/cut-management.md) |
| training.cut_selection.method  | `"domination"` | Remove dominated cuts      | [Cut Management](../01-math/cut-management.md) |

## 18.5 Discount Rate

| Option                      | Location    | LP Effect                     | Reference                                    |
| --------------------------- | ----------- | ----------------------------- | -------------------------------------------- |
| transitions[].discount_rate | stages.json | Scale cuts by $d_{t \to t+1}$ | [Discount Rate](../01-math/discount-rate.md) |

## 18.6 Horizon Mode

| Option                     | Value                 | LP Effect                    | Reference                                      |
| -------------------------- | --------------------- | ---------------------------- | ---------------------------------------------- |
| horizon.mode               | `"finite"`            | Terminal value $V_{T+1} = 0$ | [SDDP Algorithm](../01-math/sddp-algorithm.md) |
| horizon.mode               | `"infinite_periodic"` | Cycle detection, cut sharing | [SDDP Algorithm](../01-math/sddp-algorithm.md) |
| horizon.max_horizon_length | int                   | Maximum forward pass length  | [SDDP Algorithm](../01-math/sddp-algorithm.md) |

## 18.7 Upper Bound Evaluation

| Option                                | Value    | LP Effect                        | Reference                                                      |
| ------------------------------------- | -------- | -------------------------------- | -------------------------------------------------------------- |
| upper_bound_evaluation.enabled        | bool     | Enable vertex-based inner approx | [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md) |
| upper_bound_evaluation.lipschitz.mode | `"auto"` | Auto-compute Lipschitz constants | [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md) |

## 18.8 Risk Measures

| Option                       | Location    | LP Effect              | Reference                                    |
| ---------------------------- | ----------- | ---------------------- | -------------------------------------------- |
| stages[].risk_measure.type   | stages.json | Risk measure selection | [Risk Measures](../01-math/risk-measures.md) |
| stages[].risk_measure.lambda | stages.json | Risk aversion weight   | [Risk Measures](../01-math/risk-measures.md) |
| stages[].risk_measure.alpha  | stages.json | CVaR confidence level  | [Risk Measures](../01-math/risk-measures.md) |

## 18.9 Penalty Coefficients

| Option                                      | Default | Objective Term                 | Reference                                                   |
| ------------------------------------------- | ------- | ------------------------------ | ----------------------------------------------------------- |
| modeling.deficit_penalty                    | 10000.0 | $c^{def} \cdot \delta_b$       | [LP Formulation](../01-math/lp-formulation.md)              |
| modeling.spillage_penalty                   | 0.001   | $c^{spill} \cdot s_h$          | [LP Formulation](../01-math/lp-formulation.md)              |
| modeling.inflow_non_negativity.penalty_cost | 1000.0  | $c^{inf} \cdot \sigma^{inf}_h$ | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md) |

## 18.10 Complete Example Configuration

```json
{
  "modeling": {
    "block_mode": "chronological",
    "production_function": "fpha",
    "inflow_non_negativity": {
      "method": "penalty",
      "penalty_cost": 1000.0
    },
    "deficit_penalty": 10000.0,
    "spillage_penalty": 0.001
  },
  "training": {
    "forward_passes": 10,
    "cut_selection": {
      "enabled": true,
      "method": "domination",
      "threshold": 0,
      "check_frequency": 10
    },
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 100 },
      { "type": "bound_stalling", "iterations": 10, "tolerance": 0.0001 }
    ]
  },
  "horizon": {
    "mode": "finite"
  },
  "upper_bound_evaluation": {
    "enabled": true,
    "initial_iteration": 10,
    "interval_iterations": 5
  }
}
```

> **Note:** The `scenario_source` and `n_openings` parameters are defined in `stages.json`, not `config.json`. See §18.11 and §18.12 below for those settings. The `config.json` example above shows only solver-side configuration.

## 18.11 Scenario Source and Sampling Scheme

The `scenario_source` field in `stages.json` configures how inflow scenarios are selected during the SDDP forward pass. The primary key is `sampling_scheme`, which names the forward sampling abstraction — one of three orthogonal SDDP concerns. See [Scenario Generation §3](../03-architecture/scenario-generation.md) for the full abstraction design.

| Option                          | Value          | Effect                                                                   | Reference                                                           |
| ------------------------------- | -------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| scenario_source.sampling_scheme | `"in_sample"`  | Forward pass samples from the fixed opening tree (PAR-generated noise)   | [Scenario Generation §3](../03-architecture/scenario-generation.md) |
| scenario_source.sampling_scheme | `"external"`   | Forward pass draws from user-provided scenario data                      | [Scenario Generation §4](../03-architecture/scenario-generation.md) |
| scenario_source.sampling_scheme | `"historical"` | Forward pass replays historical inflow sequences                         | [Scenario Generation §3](../03-architecture/scenario-generation.md) |
| scenario_source.seed            | i64            | Base seed for reproducible noise generation (required for `in_sample`)   | [Input Scenarios §2.1](../02-data-model/input-scenarios.md)         |
| scenario_source.selection_mode  | `"random"`     | Sample from external scenarios with replacement (default for `external`) | [Input Scenarios §2.1](../02-data-model/input-scenarios.md)         |
| scenario_source.selection_mode  | `"sequential"` | Cycle through external scenarios in order                                | [Input Scenarios §2.1](../02-data-model/input-scenarios.md)         |

**Backward pass noise source**: Regardless of the forward sampling scheme, the backward pass always evaluates all openings from the fixed opening tree. The opening tree is generated from a PAR model — either the user-provided model (for `in_sample`) or a PAR model fitted to the external/historical data (for `external` and `historical`). See [Scenario Generation §3.1](../03-architecture/scenario-generation.md).

**Example configurations:**

```json
{ "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 } }
```

```json
{
  "scenario_source": {
    "sampling_scheme": "external",
    "selection_mode": "random"
  }
}
```

```json
{ "scenario_source": { "sampling_scheme": "historical" } }
```

## 18.12 Opening Tree Configuration

The `n_openings` parameter in `stages.json` controls the number of noise vectors in the fixed opening tree — the set of discrete outcomes evaluated in the backward pass at each stage.

| Option       | Location    | Default | Effect                                   | Reference                                                             |
| ------------ | ----------- | ------- | ---------------------------------------- | --------------------------------------------------------------------- |
| `n_openings` | stages.json | —       | Number of backward pass noise branchings | [Scenario Generation §2.3](../03-architecture/scenario-generation.md) |

The opening tree is generated once before training and remains fixed throughout. Larger values improve cut quality but increase backward pass cost linearly.

> **Deferred**: Monte Carlo backward sampling — sample n < n_openings noise terms per backward step instead of evaluating all. See [Deferred Features C.14](../06-deferred/deferred-features.md).

## 19.1 Section Mapping

This table maps each mathematical formulation section to corresponding configuration options and data structures.

| Formulation Topic        | Config Path                                                                 | Data Files                                                                 | Spec Reference                                                      |
| ------------------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Block Formulation**    | config.json → modeling.block_mode                                           | `Stage.blocks[]`, per-block water balance                                  | [Block Formulations](../01-math/block-formulations.md)              |
| **Production Functions** | hydro_production_models.json, config.json → modeling.production_function    | `Hydro.productivity`, `fpha_hyperplanes.parquet`, `hydro_geometry.parquet` | [Hydro Production Models](../01-math/hydro-production-models.md)    |
| **PAR(p) Model**         | scenarios/`inflow_seasonal_stats.parquet`, `inflow_ar_coefficients.parquet` | Seasonal means/std, AR coefficients per (hydro, stage, lag)                | [PAR Inflow Model](../01-math/par-inflow-model.md)                  |
| **Non-Negativity**       | config.json → modeling.inflow_non_negativity                                | LP slack variables, penalty coefficients                                   | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md)         |
| **Cut Generation**       | N/A (runtime)                                                               | Cut intercept/coefficients, dual extraction                                | [Cut Management](../01-math/cut-management.md)                      |
| **Cut Selection**        | config.json → training.cut_selection                                        | Cut activity tracking                                                      | [Cut Management](../01-math/cut-management.md)                      |
| **Stopping Rules**       | config.json → training.stopping_rules[]                                     | Convergence metrics                                                        | [Stopping Rules](../01-math/stopping-rules.md)                      |
| **Discount Rate**        | stages.json → policy_graph.annual_discount_rate                             | Per-transition discount factor, cut scaling                                | [Discount Rate](../01-math/discount-rate.md)                        |
| **Infinite Horizon**     | config.json → horizon.mode, stages.json cycle                               | Policy graph cycle detection, cut sharing                                  | [SDDP Algorithm](../01-math/sddp-algorithm.md)                      |
| **Inner Approximation**  | config.json → upper_bound_evaluation                                        | Vertex storage, Lipschitz constants                                        | [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md)      |
| **Risk-Averse CVaR**     | stages.json → risk_measure                                                  | Risk-adjusted probability computation                                      | [Risk Measures](../01-math/risk-measures.md)                        |
| **Scenario Source**      | stages.json → scenario_source.sampling_scheme                               | Opening tree, `external_scenarios.parquet`, `inflow_history.parquet`       | [Scenario Generation §3](../03-architecture/scenario-generation.md) |

## 19.2 Variable Correspondence

| Math Symbol     | Field Name               | JSON/File Path                                     | Type       |
| --------------- | ------------------------ | -------------------------------------------------- | ---------- |
| $v_h$           | Hydro storage            | hydros.json → storage                              | `f64`      |
| $\hat{v}_h$     | Incoming storage (state) | Internal state vector                              | `Vec<f64>` |
| $a_h$           | Incremental inflow       | `inflow_seasonal_stats.parquet` (mean_m3s)         | `f64`      |
| $\psi_{m,\ell}$ | AR coefficients          | `inflow_ar_coefficients.parquet` → coefficient     | `f64`      |
| $\sigma_m$      | Residual std dev         | Computed from $s_m$ and AR coefficients at runtime | `f64`      |
| $\theta$        | Future cost variable     | LP variable                                        | `f64`      |
| $\alpha_k$      | Cut intercept            | policy/cuts/stage_XXX.bin                          | `f64`      |
| $\beta_k$       | Cut coefficients         | policy/cuts/stage_XXX.bin                          | `Vec<f64>` |
| $d_{t \to t+1}$ | Discount factor          | stages.json → policy_graph.annual_discount_rate    | `f64`      |
| $L_t$           | Lipschitz constant       | Computed from penalties                            | `f64`      |

## 19.3 Configuration Quick Reference

### Chronological Blocks

```json
{
  "modeling": {
    "block_mode": "chronological"
  }
}
```

**Effect**: Enables per-block storage variables $v_{h,k}$ and sequential water balance constraints.

### Discount Rate

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

**Effect**: Per-transition discount factor $d_{t \to t+1} = 1/(1 + r)^{\Delta t}$ where $r = 0.06$ and $\Delta t$ is the source stage duration in years. Individual transitions may override the global rate.

### PAR(p) Model

Configured via two files:

**`scenarios/inflow_seasonal_stats.parquet`:**

| Column     | Math Symbol | Description                               |
| ---------- | ----------- | ----------------------------------------- |
| `hydro_id` | —           | Hydro plant ID                            |
| `stage_id` | —           | Stage ID                                  |
| `mean_m3s` | $\mu_m$     | Season mean                               |
| `std_m3s`  | $s_m$       | Seasonal sample standard deviation        |
| `ar_order` | $p$         | Number of AR lags for this (hydro, stage) |

**`scenarios/inflow_ar_coefficients.parquet`** (optional):

| Column        | Math Symbol     | Description                                       |
| ------------- | --------------- | ------------------------------------------------- |
| `hydro_id`    | —               | Hydro plant ID                                    |
| `stage_id`    | —               | Stage ID                                          |
| `lag`         | $\ell$          | Lag index (1-based)                               |
| `coefficient` | $\psi_{m,\ell}$ | AR coefficient (original units, not standardized) |

### Inflow Non-Negativity

```json
{
  "modeling": {
    "inflow_non_negativity": {
      "method": "penalty",
      "penalty_cost": 1000.0
    }
  }
}
```

| Method                    | Math Formulation                          | LP Variables   |
| ------------------------- | ----------------------------------------- | -------------- |
| `none`                    | Direct AR output                          | None added     |
| `penalty`                 | $a_h + \sigma^{inf}_h = \text{AR output}$ | `inflow_slack` |
| `truncation`              | $a_h = \max(0, \text{AR output})$         | None           |
| `truncation_with_penalty` | $\eta_h^{adj} = \eta_h + \xi_h$           | `noise_slack`  |

### Cut Selection

```json
{
  "training": {
    "cut_selection": {
      "enabled": true,
      "method": "domination",
      "threshold": 0,
      "check_frequency": 10
    }
  }
}
```

| Method       | Algorithm                           |
| ------------ | ----------------------------------- |
| `level1`     | Keep cuts active at least once      |
| `lml1`       | Keep most recently active per state |
| `domination` | Remove Pareto-dominated cuts        |

### Stopping Rules

```json
{
  "training": {
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 100 },
      { "type": "bound_stalling", "iterations": 10, "tolerance": 0.0001 }
    ],
    "stopping_mode": "any"
  }
}
```

### Infinite Horizon

```json
{
  "horizon": {
    "mode": "infinite_periodic",
    "max_horizon_length": 240,
    "cycle_discretization_delta": 0.1
  }
}
```

**Requirements**:

- At least one transition must create a cycle
- Cycle transitions must have `annual_discount_rate > 0`

### Scenario Source

```json
{ "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 } }
```

| Sampling Scheme | Forward Noise Source                | Backward Noise Source                         |
| --------------- | ----------------------------------- | --------------------------------------------- |
| `in_sample`     | Opening tree (PAR-generated)        | Same opening tree                             |
| `external`      | User-provided scenario values       | Opening tree from PAR fitted to external data |
| `historical`    | Historical inflows mapped to stages | Opening tree from PAR fitted to history       |

See §18.11 for full parameter details.

### Inner Approximation

```json
{
  "upper_bound_evaluation": {
    "enabled": true,
    "initial_iteration": 10,
    "interval_iterations": 5,
    "lipschitz": {
      "mode": "auto",
      "fallback_value": 10000.0,
      "scale_factor": 1.1
    }
  }
}
```

## 19.4 Rust Struct Correspondence

| Math Entity                          | Struct        | File Location                          |
| ------------------------------------ | ------------- | -------------------------------------- |
| Benders cut $(\alpha_k, \beta_k)$    | `Cut`         | `powers-core/src/policy/cut.rs`        |
| Cut pool $\mathcal{K}_t$             | `CutPool`     | `powers-core/src/policy/cut_pool.rs`   |
| Vertex $(x^{(i)}, \bar{v}^{(i)})$    | `Vertex`      | `powers-core/src/policy/vertex.rs`     |
| Stage subproblem                     | `Subproblem`  | `powers-core/src/lp/subproblem.rs`     |
| State vector $x_t$                   | `StateVector` | `powers-core/src/state.rs`             |
| PAR(p) model                         | `InflowModel` | `powers-core/src/stochastic/inflow.rs` |
| Risk measure $\rho^{\lambda,\alpha}$ | `RiskMeasure` | `powers-core/src/risk.rs`              |
| Transition graph                     | `PolicyGraph` | `powers-core/src/graph.rs`             |

## Cross-References

- [SDDP Algorithm](../01-math/sddp-algorithm.md) -- Algorithm overview and policy graph structure
- [LP Formulation](../01-math/lp-formulation.md) -- Objective function, constraints, dual variables
- [Block Formulations](../01-math/block-formulations.md) -- Parallel and chronological block modes
- [Hydro Production Models](../01-math/hydro-production-models.md) -- Constant productivity and FPHA
- [PAR Inflow Model](../01-math/par-inflow-model.md) -- PAR(p) definition and fitting
- [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md) -- Non-negativity treatment methods
- [Cut Management](../01-math/cut-management.md) -- Cut generation, aggregation, selection
- [Stopping Rules](../01-math/stopping-rules.md) -- Convergence and stopping criteria
- [Discount Rate](../01-math/discount-rate.md) -- Discounted Bellman equation
- [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md) -- Inner approximation and gap computation
- [Risk Measures](../01-math/risk-measures.md) -- CVaR and risk-averse cut generation
- [Penalty System](../02-data-model/penalty-system.md) -- Three-tier penalty cascade
- [Input Directory Structure](../02-data-model/input-directory-structure.md) -- File layout and config.json schema
- [Scenario Generation](../03-architecture/scenario-generation.md) -- Sampling scheme abstraction (§3), opening tree (§2.3), external scenario integration (§4), complete tree mode (§7)
- [Input Scenarios](../02-data-model/input-scenarios.md) -- scenario_source schema (§2.1), inflow model files (§3)
- [Deferred Features](../06-deferred/deferred-features.md) -- Planned but not yet implemented features (C.13-C.16 for sampling scheme extensions)
