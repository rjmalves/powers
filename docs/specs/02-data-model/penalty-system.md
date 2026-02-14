---
status: approved
review_priority: 1-critical
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.2.1 (Penalties and Costs)"
last_reviewed: 2026-02-14
reviewed_by: rogerio
review_notes: "Approved. Reclassified penalties into 3 categories (recourse/constraint-violation/regularization). Completed constraint violation inventory. Fixed generation bounds, exchange modeling, removed pumping_cost. Stage override format remains TBD. Diversion confirmed as regularization-only (no violation slacks needed). CEPEL observations on lateral flow, Q_jus formulation, and travel time propagation curves flagged for P2 math spec reviews."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.2.1"
  - date: 2026-02-14
    description: "First review: rewrote penalty categories (3 types), completed constraint violation table, fixed generation bounds, fixed exchange/pumping in objective, added line overrides, opened stage-override format discussion, reset version"
  - date: 2026-02-14
    description: "Approved after diversion analysis (regularization-only, no slacks). CEPEL hydro modeling observations (lateral flow, Q_jus, travel time) flagged in CHANGE_TRACKER for P2 reviews"
---

# Penalty System

## Purpose

This spec defines the unified penalty system that ensures LP feasibility across all scenarios while correctly pricing operational costs and constraint violations. The penalty system uses a three-tier cascade resolution: global defaults → entity overrides → stage overrides.

This is a key area expected to evolve. See [LP Formulation](../01-math/lp-formulation.md) for how penalties enter the objective function.

## 1. Design Rationale

The LP must always be feasible. Several physical and operational constraints may be impossible to satisfy in extreme scenarios (droughts, equipment failures, etc.). The penalty system provides slack variables with graduated costs to maintain feasibility while signaling the severity of violations.

### Override Resolution Behavior

The penalty system supports three levels of specificity. The effective penalty for a given entity, stage, and penalty type is determined by the most specific value available:

1. **Stage-level override** — If a value is defined for this specific (entity, stage, penalty_type) tuple, it takes precedence.
2. **Entity-level override** — If no stage override exists, a per-entity default (defined in the entity registry file) is used.
3. **Global default** — If neither stage nor entity overrides exist, the global default from `penalties.json` applies.

| Query                       | Stage Override? | Entity Override? | Result                |
| --------------------------- | --------------- | ---------------- | --------------------- |
| Hydro 0, Stage 30, spillage | No              | Yes (0.005)      | 0.005 (entity)        |
| Hydro 0, Stage 60, spillage | Yes (0.02)      | Yes (0.005)      | 0.02 (stage)          |
| Hydro 1, Stage 30, spillage | No              | No               | 0.01 (global default) |

> **Implementation note**: The resolution behavior above describes the _semantics_ of the cascade, not the algorithm. In practice, the implementation should pre-resolve penalties during input loading (e.g., start from global defaults, apply entity overrides while parsing registries, then batch-apply stage overrides) rather than querying files at runtime. The exact resolution strategy is an implementation concern.

### Format Rationale

| Tier             | File                              | Format | Rationale                                                             |
| ---------------- | --------------------------------- | ------ | --------------------------------------------------------------------- |
| Global defaults  | `penalties.json`                  | JSON   | Hierarchical config with nested cost categories; natural for defaults |
| Entity overrides | `hydros.json`, `buses.json`, etc. | JSON   | Per-entity overrides co-located with the entity definition            |
| Stage overrides  | TBD                               | TBD    | Sparse overrides that apply only when values change over time         |

> **Open question — stage override file format**: The stage-varying overrides are sparse (most entities keep their default penalty for most stages), but this is not time-series data in the traditional sense — it is a set of (entity, stage) point overrides. Possible formats include:
>
> - **Parquet**: Efficient columnar storage and filtering, but may be over-engineered for what is typically a small, sparse dataset.
> - **JSON array**: Simple and human-readable; could be a flat list of `{entity_id, stage_id, field, value}` records or grouped by entity.
> - **CSV**: Simplest option for tabular override data; easy to inspect and edit.
>
> The choice should consider: typical dataset size (usually small), ease of manual editing by users, and consistency with other input file formats in the system.

## 2. Penalty Categories

Penalties serve three distinct purposes in the LP formulation. Understanding these categories is important for setting appropriate cost magnitudes and interpreting results.

### Category 1: Recourse Slacks (LP Feasibility)

These penalties ensure that the SDDP algorithm has relatively complete recourse — every subproblem must be feasible regardless of the scenario realization. Without these slacks, the LP would be infeasible when generation cannot meet demand or when excess uncontrollable generation cannot be absorbed.

| Penalty            | Units | Applied To                | Purpose                         | Typical Range      |
| ------------------ | ----- | ------------------------- | ------------------------------- | ------------------ |
| `deficit_segments` | $/MWh | Unmet load per bus        | Piecewise cost of load shedding | 1,000–10,000 $/MWh |
| `excess_cost`      | $/MWh | Excess generation per bus | Absorb uncontrollable surplus   | 0.001–0.1 $/MWh    |

Deficit and excess are conceptually slack variables on the load balance constraint, but they have special names because of their importance in the hydrothermal dispatch application. Deficit represents the value of lost load; excess is a regularization-level cost to eliminate spurious slack generation.

### Category 2: Constraint Violation Penalties (Policy Shaping)

These penalties provide slack for physical or operational constraints that may be impossible to satisfy under extreme conditions (e.g., drought, environmental directives from the system operator). Their cost must be high enough to affect the value function in earlier stages, signaling that the system should avoid states that lead to these violations.

| Penalty                           | Units      | Applied To                    | Purpose                                            | Typical Range      |
| --------------------------------- | ---------- | ----------------------------- | -------------------------------------------------- | ------------------ |
| `turbined_violation_below_cost`   | $/(m³/s·h) | Turbined flow < min           | Equipment limits / ecological flow                 | 500–1,000 $/unit   |
| `outflow_violation_below_cost`    | $/(m³/s·h) | Outflow < min                 | Environmental minimum flow (operator/regulatory)   | 500–1,000 $/unit   |
| `outflow_violation_above_cost`    | $/(m³/s·h) | Outflow > max                 | Downstream flooding prevention                     | 500–1,000 $/unit   |
| `generation_violation_below_cost` | $/MWh      | Generation < min              | Contractual or environmental minimum generation    | 1,000–2,000 $/unit |
| `evaporation_violation_cost`      | $/(m³/s·h) | Evaporation constraint        | Physical constraint (bidirectional, see Section 6) | 5,000+ $/unit      |
| `water_withdrawal_violation_cost` | $/(m³/s·h) | Unmet water withdrawal        | Human consumption / irrigation commitments         | 1,000–5,000 $/unit |
| `generic_violation_cost`          | varies     | Generic constraint violations | User-defined physical or operational constraints   | User-defined       |

These penalties create an artificial cost in the objective function that propagates backward through the value function, telling earlier stages to store more water (or dispatch differently) to avoid reaching states where violations are necessary.

### Category 3: Regularization Costs (Solution Guidance)

These are small costs inserted into the objective function to guide the solver toward physically preferred solutions when the LP would otherwise be indifferent. They do not represent real costs and should be orders of magnitude smaller than any economic cost to avoid distorting the optimal policy.

| Penalty          | Units      | Applied To          | Purpose                                                               | Typical Range     |
| ---------------- | ---------- | ------------------- | --------------------------------------------------------------------- | ----------------- |
| `spillage_cost`  | $/(m³/s·h) | Water spilled       | Prefer turbining over spilling when solver is indifferent             | 0.001–0.01 $/unit |
| `diversion_cost` | $/(m³/s·h) | Water diverted      | Prefer main channel flow; higher than spillage (water leaves cascade) | 0.01–0.1 $/unit   |
| `exchange_cost`  | $/MWh      | Power flow on lines | Prefer local supply; avoid unnecessary inter-bus power flows          | 0.01–1.0 $/unit   |

### Penalty Priority Ordering

When setting penalty magnitudes, the following ordering must be maintained:

$$\text{Physical violations} > \text{Deficit} > \text{Constraint violations} > \text{Resource costs} > \text{Regularization}$$

## 3. Global Penalty Defaults (`penalties.json`)

This file defines default penalty values for all entities. It is **required** and must be present in the case directory root.

```json
{
  "version": "1.0",
  "bus": {
    "deficit_segments": [
      { "depth_mw": 500, "cost": 1000.0 },
      { "depth_mw": 1000, "cost": 3000.0 },
      { "depth_mw": null, "cost": 5000.0 }
    ],
    "excess_cost": 100.0
  },
  "line": {
    "exchange_cost": 2.0
  },
  "hydro": {
    "spillage_cost": 0.01,
    "diversion_cost": 0.1,
    "turbined_violation_below_cost": 500.0,
    "outflow_violation_below_cost": 500.0,
    "outflow_violation_above_cost": 500.0,
    "generation_violation_below_cost": 1000.0,
    "evaporation_violation_cost": 5000.0,
    "water_withdrawal_violation_cost": 1000.0
  }
}
```

### Piecewise Deficit

Deficit is modeled as piecewise linear segments. Each segment specifies a depth (MW of unmet demand) and cost. Segments are cumulative: first `depth_mw` MW at first cost, next at second cost, etc. The last segment **MUST** have `depth_mw: null` to ensure LP feasibility (unbounded extension).

Deficit segments can be overridden per bus in `buses.json` (see [Input System Entities §1](input-system-entities.md)), but **cannot be stage-varying** (piecewise structure is too complex for per-stage override).

## 4. Constraint Violation Categories

This section enumerates all constraints in the LP that use slack variables, organized by the system element they belong to. For the full LP constraint formulations, see [LP Formulation](../01-math/lp-formulation.md).

### System-Level (Bus)

| Constraint Type | Slack Variable | Direction                  | Penalty            | Category |
| --------------- | -------------- | -------------------------- | ------------------ | -------- |
| Load balance    | `deficit`      | Lower (unmet demand)       | `deficit_segments` | Recourse |
| Load balance    | `excess`       | Upper (surplus generation) | `excess_cost`      | Recourse |

### Hydro — Flow and Generation Constraints

| Constraint Type    | Slack Variable               | Direction   | Penalty                           | Category             |
| ------------------ | ---------------------------- | ----------- | --------------------------------- | -------------------- |
| Minimum turbined   | `turbined_violation_below`   | Lower bound | `turbined_violation_below_cost`   | Constraint violation |
| Minimum outflow    | `outflow_violation_below`    | Lower bound | `outflow_violation_below_cost`    | Constraint violation |
| Maximum outflow    | `outflow_violation_above`    | Upper bound | `outflow_violation_above_cost`    | Constraint violation |
| Minimum generation | `generation_violation_below` | Lower bound | `generation_violation_below_cost` | Constraint violation |
| Evaporation        | `evaporation_violation_pos`  | Upper slack | `evaporation_violation_cost`      | Constraint violation |
| Evaporation        | `evaporation_violation_neg`  | Lower slack | `evaporation_violation_cost`      | Constraint violation |
| Water withdrawal   | `water_withdrawal_violation` | Lower bound | `water_withdrawal_violation_cost` | Constraint violation |

### Hydro — Storage Bounds

**Storage bounds** (`min_storage_hm3`, `max_storage_hm3`) are hard physical limits (reservoir capacity). No slack variables are used. If storage would exceed the maximum, emergency spillage handles the excess. If storage falls below the minimum, this indicates a data error or an impossible physical scenario.

### Lines

**Exchange bounds** are hard variable bounds on the direct and reverse flow variables. No slack variables. The `exchange_cost` is a regularization term on the flow variables themselves, not a violation penalty.

### Thermals

**Thermal bounds** (`min_generation`, `max_generation`) are hard constraints. No slack variables. Thermal dispatch is directly controllable (unlike hydro, which depends on exogenous inflows), so if bounds cannot be met, this indicates a data error.

### Generic Constraints

User-defined generic constraints (see [Input Constraints](input-constraints.md)) can optionally have slack variables with user-specified penalty costs. These are typically used for physical or operational directives from the system operator, and fall into the **constraint violation** category.

## 5. Stage-Varying Penalty Overrides

Stage-varying overrides allow penalty values to change at specific stages for specific entities. Only entries that differ from the entity or global defaults need to be specified (sparse storage).

> **Open question — file format**: See the discussion in Section 1. The schemas below define the _logical structure_ of the override data regardless of physical file format.

### Bus Penalty Overrides — Optional

| Column        | Type | Nullable | Description                                      |
| ------------- | ---- | -------- | ------------------------------------------------ |
| `bus_id`      | u32  | No       | Bus identifier                                   |
| `stage_id`    | u32  | No       | Stage identifier                                 |
| `excess_cost` | f64  | Yes      | $/MWh for excess generation (null = use default) |

### Line Penalty Overrides — Optional

| Column          | Type | Nullable | Description                                  |
| --------------- | ---- | -------- | -------------------------------------------- |
| `line_id`       | u32  | No       | Line identifier                              |
| `stage_id`      | u32  | No       | Stage identifier                             |
| `exchange_cost` | f64  | Yes      | $/MWh for exchange cost (null = use default) |

### Hydro Penalty Overrides — Optional

| Column                            | Type | Nullable | Description                            |
| --------------------------------- | ---- | -------- | -------------------------------------- |
| `hydro_id`                        | u32  | No       | Hydro identifier                       |
| `stage_id`                        | u32  | No       | Stage identifier                       |
| `spillage_cost`                   | f64  | Yes      | $/(m³/s·h) for spilled water           |
| `diversion_cost`                  | f64  | Yes      | $/(m³/s·h) for diverted water          |
| `turbined_violation_below_cost`   | f64  | Yes      | $/(m³/s·h) for turbined flow below min |
| `outflow_violation_below_cost`    | f64  | Yes      | $/(m³/s·h) for outflow below min       |
| `outflow_violation_above_cost`    | f64  | Yes      | $/(m³/s·h) for outflow above max       |
| `generation_violation_below_cost` | f64  | Yes      | $/MWh for generation below min         |
| `evaporation_violation_cost`      | f64  | Yes      | $/(m³/s·h) for evaporation violation   |
| `water_withdrawal_violation_cost` | f64  | Yes      | $/(m³/s·h) for unmet water withdrawal  |

**Sparse storage**: Only include entries where values differ from defaults to minimize file size and I/O.

## 6. Negative Evaporation (Condensation) Handling

While evaporation is typically positive (water loss), the evaporation coefficient can be negative:

| Condition               | Description                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| Condensation            | Water condensing on reservoir surface in humid climates                                         |
| Rainfall contribution   | When models include net precipitation effects                                                   |
| Linearization artifacts | The linear approximation may produce negative values at certain volume/coefficient combinations |

The evaporation constraint uses **bidirectional slack variables**:

```
Q_evaporated - evap_slack_positive + evap_slack_negative = EvapCoef × Area(V_avg)

where:
  evap_slack_positive ≥ 0  (actual evap > computed evap)
  evap_slack_negative ≥ 0  (actual evap < computed evap, including negative target)
```

Both slack variables receive the same penalty: `evaporation_violation_cost`.

## 7. Hydro Variables and Bounds Summary

| Variable        | Lower Bound          | Upper Bound         | Lower Slack  | Upper Slack     |
| --------------- | -------------------- | ------------------- | ------------ | --------------- |
| `storage`       | `min_storage_hm3`    | `max_storage_hm3`   | Hard         | Emergency spill |
| `turbined_flow` | `min_turbined_m3s`   | `max_turbined_m3s`  | With penalty | Hard            |
| `spillage`      | 0                    | ∞                   | Hard         | —               |
| `outflow`       | `min_outflow_m3s`    | `max_outflow_m3s`   | With penalty | With penalty    |
| `generation`    | `min_generation_mw`  | `max_generation_mw` | With penalty | Hard            |
| `evaporation`   | −∞ (can be negative) | +∞                  | With penalty | With penalty    |
| `withdrawal`    | `water_withdrawal`   | `water_withdrawal`  | With penalty | —               |

> **Generation bounds**: The user explicitly sets `min_generation_mw` and `max_generation_mw`. These are not derived from turbined flow bounds, because the production function is not always constant productivity. When a complete hydro model is available, the installed capacity provides a natural hard upper bound. The lower bound always requires a slack to maintain feasibility.

Relationship: `outflow = turbined_flow + spillage`, `generation = f(turbined_flow, storage)` (depends on production model)

### Dead-Volume Filling Specifics

During the filling period:

- **No turbined flow**: `turbined_flow = 0` (hard constraint — turbines not installed/operational)
- **Outflow = spillage**: All released water goes through non-turbine outlets
- **Min outflow requirement**: Environmental flow must be met via spillage
- If `inflow - filling_retention < min_outflow`, the `outflow_violation_below` slack absorbs the shortfall

The same `outflow_violation_cost` from `hydro_penalties.parquet` applies during filling. Spillage during filling also incurs `spillage_cost`.

## 8. LP Objective Function Impact

For each stage `t`, block `b`, scenario `s`:

```
minimize:
  // Resource costs
  + Σ_thermal (generation × cost_per_mwh)
  + Σ_contract (import × import_price - export × export_price)

  // Recourse slacks (LP feasibility)
  + Σ_bus Σ_segment (deficit_segment × segment_cost)
  + Σ_bus (excess × excess_cost)

  // Constraint violation penalties (policy shaping)
  + Σ_hydro (turbined_violation_below × turbined_violation_below_cost)
  + Σ_hydro (outflow_violation_below × outflow_violation_below_cost)
  + Σ_hydro (outflow_violation_above × outflow_violation_above_cost)
  + Σ_hydro (generation_violation_below × generation_violation_below_cost)
  + Σ_hydro (evaporation_violation_pos × evaporation_violation_cost)
  + Σ_hydro (evaporation_violation_neg × evaporation_violation_cost)
  + Σ_hydro (water_withdrawal_violation × water_withdrawal_violation_cost)

  // Regularization costs (solution guidance)
  + Σ_hydro (spillage × spillage_cost)
  + Σ_hydro (diversion × diversion_cost)
  + Σ_line (direct_flow × exchange_cost + reverse_flow × exchange_cost)

  // Future cost function
  + θ  // Cut approximation (future cost)
```

> **Note on pumping**: Pumping stations do not have an explicit cost in the objective. The cost of pumping is implicitly captured through the energy consumed at the connected bus (appears as negative demand in the load balance). The marginal cost at that bus determines the effective pumping cost.

## Cross-References

- [Input System Entities](input-system-entities.md) — Entity registries with optional penalty overrides
- [Input Constraints](input-constraints.md) — Time-varying entity bounds; generic constraint slack penalties
- [Input Hydro Extensions](input-hydro-extensions.md) — Hydro geometry for evaporation calculation
- [LP Formulation](../01-math/lp-formulation.md) — Cost taxonomy (§5.0) and slack penalties (§5.8)
- [Configuration Reference](../05-config/configuration-reference.md) — Penalty-related config settings
- [Design Principles](../00-overview/design-principles.md) — General design approach
