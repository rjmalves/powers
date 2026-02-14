---
status: needs-review
review_priority: 1-critical
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.2.1 (Penalties and Costs)"
last_reviewed: null
reviewed_by: null
review_notes: "User identified penalty system as an area expecting changes. Needs targeted review."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.2.1"
---

# Penalty System

## Purpose

This spec defines the unified penalty system that ensures LP feasibility across all scenarios while correctly pricing operational costs and constraint violations. The penalty system uses a three-tier cascade resolution: global defaults → entity overrides → stage overrides.

This is a key area expected to evolve. See [LP Formulation](../01-math/lp-formulation.md) for how penalties enter the objective function.

## 1. Design Rationale

The LP must always be feasible. Several physical and operational constraints may be impossible to satisfy in extreme scenarios (droughts, equipment failures, etc.). The penalty system provides slack variables with graduated costs to maintain feasibility while signaling the severity of violations.

### Override Resolution Algorithm

```
resolve_penalty(entity_id, stage_id, penalty_type):

    1. Check stage override in parquet file
       → If found, return parquet value

    2. Check entity override in entity JSON (e.g., hydros.json)
       → If found, return entity value

    3. Return global default from penalties.json
```

| Query                       | Stage Override? | Entity Override? | Result                |
| --------------------------- | --------------- | ---------------- | --------------------- |
| Hydro 0, Stage 30, spillage | No              | Yes (0.005)      | 0.005 (entity)        |
| Hydro 0, Stage 60, spillage | Yes (0.02)      | Yes (0.005)      | 0.02 (stage)          |
| Hydro 1, Stage 30, spillage | No              | No               | 0.01 (global default) |

### Format Rationale

> **Default-with-overrides** — Three-tier pattern using JSON for global defaults and entity overrides, Parquet for sparse stage-varying overrides.

| Tier             | File                              | Format  | Rationale                                                             |
| ---------------- | --------------------------------- | ------- | --------------------------------------------------------------------- |
| Global defaults  | `penalties.json`                  | JSON    | Hierarchical config with nested cost categories; natural for defaults |
| Entity overrides | `hydros.json`, `buses.json`, etc. | JSON    | Per-entity overrides co-located with the entity definition            |
| Stage overrides  | `*_penalties.parquet`             | Parquet | Sparse rows that override only when values change over time           |

## 2. Penalty Categories

Penalties are divided into two categories:

| Category                | Penalties                                          | Purpose                                               | Typical Range     |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------- | ----------------- |
| **Operational Costs**   | `spillage_cost`, `diversion_cost`, `exchange_cost` | Discourage undesirable but feasible operations        | 0.001–10 $/unit   |
| **Violation Penalties** | `deficit_*`, `excess_cost`, `*_violation_*_cost`   | Ensure LP feasibility, penalize constraint violations | 100–10,000 $/unit |

### Complete Penalty Reference

| Penalty                           | Category    | Units      | Applied To                | Purpose                                  |
| --------------------------------- | ----------- | ---------- | ------------------------- | ---------------------------------------- |
| `deficit_segments`                | Violation   | $/MWh      | Unmet load per bus        | Piecewise cost of load shedding          |
| `excess_cost`                     | Violation   | $/MWh      | Excess generation per bus | Dumping excess power                     |
| `exchange_cost`                   | Operational | $/MWh      | Power flow on lines       | Discourage unnecessary exchange          |
| `spillage_cost`                   | Operational | $/(m³/s·h) | Water spilled             | Opportunity cost, incentivizes turbining |
| `diversion_cost`                  | Operational | $/(m³/s·h) | Water diverted            | Opportunity cost (higher than spillage)  |
| `turbined_violation_below_cost`   | Violation   | $/(m³/s·h) | Turbined < min            | Equipment/ecological flow                |
| `outflow_violation_below_cost`    | Violation   | $/(m³/s·h) | Outflow < min             | Environmental minimum flow               |
| `outflow_violation_above_cost`    | Violation   | $/(m³/s·h) | Outflow > max             | Downstream flooding prevention           |
| `generation_violation_below_cost` | Violation   | $/MWh      | Generation < min          | Contractual/environmental minimum        |
| `evaporation_violation_cost`      | Violation   | $/(m³/s·h) | Evaporation constraint    | Physical constraint (bidirectional)      |
| `water_withdrawal_violation_cost` | Violation   | $/(m³/s·h) | Unmet water withdrawal    | Human consumption/irrigation             |

**Operational cost ordering**: `diversion_cost` should be higher than `spillage_cost` because diverted water typically leaves the main cascade entirely. Typical values: `spillage_cost ≈ 0.001-0.01`, `diversion_cost ≈ 0.01-0.1`.

## 3. Global Penalty Defaults (`penalties.json`)

This file defines default penalty values for all entities. It is **required** and must be present in the case directory root.

```json
{
  "version": "1.1",
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

| Category             | Constraint Type   | Slack Variable                                       | Direction                       |
| -------------------- | ----------------- | ---------------------------------------------------- | ------------------------------- |
| **System**           | Load balance      | `deficit`, `excess`                                  | Lower (deficit), Upper (excess) |
| **Hydro Generation** | Min generation    | `generation_violation_below`                         | Lower bound                     |
| **Hydro Turbined**   | Min turbined flow | `turbined_violation_below`                           | Lower bound                     |
| **Hydro Outflow**    | Min/Max outflow   | `outflow_violation_below`, `outflow_violation_above` | Both bounds                     |

**Storage bounds**: Storage min/max are typically hard physical limits (reservoir capacity). Violations are handled by emergency spillage (if above max) or infeasibility (if below min due to bad data). No slack variables for storage bounds.

**Thermals**: Thermal plants are modeled as always available within their bounds. No slack variables needed — if a thermal cannot meet its minimum generation, it indicates a data error.

## 5. Stage-Varying Penalty Overrides

### Bus Penalties (`constraints/bus_penalties.parquet`) — Optional

| Column        | Type | Nullable | Description                                      |
| ------------- | ---- | -------- | ------------------------------------------------ |
| `bus_id`      | u32  | No       | Bus identifier                                   |
| `stage_id`    | u32  | No       | Stage identifier                                 |
| `excess_cost` | f64  | Yes      | $/MWh for excess generation (null = use default) |

### Hydro Penalties (`constraints/hydro_penalties.parquet`) — Optional

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

**Sparse storage**: Only include rows where values differ from defaults to minimize file size and I/O.

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

| Variable        | Lower Bound           | Upper Bound           | Lower Slack  | Upper Slack     |
| --------------- | --------------------- | --------------------- | ------------ | --------------- |
| `storage`       | `min_storage_hm3`     | `max_storage_hm3`     | Hard         | Emergency spill |
| `turbined_flow` | `min_turbined_m3s`    | `max_turbined_m3s`    | With penalty | Hard            |
| `spillage`      | 0                     | ∞                     | Hard         | —               |
| `outflow`       | `min_outflow_m3s`     | `max_outflow_m3s`     | With penalty | With penalty    |
| `generation`    | Derived from turbined | Derived from turbined | With penalty | Hard            |
| `evaporation`   | −∞ (can be negative)  | +∞                    | With penalty | With penalty    |

Relationship: `outflow = turbined_flow + spillage`, `generation = productivity × turbined_flow`

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
  // Operational costs
  + Σ_thermal (generation × cost_per_mwh)
  + Σ_hydro (spillage × spillage_cost)
  + Σ_hydro (diversion × diversion_cost)
  + Σ_line (|exchange| × exchange_cost)
  + Σ_contract (import × import_price - export × export_price)
  + Σ_pumping (pumped_flow × pumping_cost)

  // Violation penalties
  + Σ_bus Σ_segment (deficit_segment × segment_cost)
  + Σ_bus (excess × excess_cost)
  + Σ_hydro (turbined_violation_below × turbined_violation_below_cost)
  + Σ_hydro (outflow_violation_below × outflow_violation_below_cost)
  + Σ_hydro (outflow_violation_above × outflow_violation_above_cost)
  + Σ_hydro (generation_violation_below × generation_violation_below_cost)
  + Σ_hydro (evaporation_violation_positive × evaporation_violation_cost)
  + Σ_hydro (evaporation_violation_negative × evaporation_violation_cost)
  + Σ_hydro (water_withdrawal_violation × water_withdrawal_violation_cost)

  // Future cost function
  + θ  // Cut approximation (future cost)
```

## Cross-References

- [Input System Entities](input-system-entities.md) — Entity registries with optional penalty overrides
- [Input Constraints](input-constraints.md) — Time-varying entity bounds; generic constraint slack penalties
- [Input Hydro Extensions](input-hydro-extensions.md) — Hydro geometry for evaporation calculation
- [LP Formulation](../01-math/lp-formulation.md) — Cost taxonomy (§5.0) and slack penalties (§5.8)
- [Configuration Reference](../05-config/configuration-reference.md) — Penalty-related config settings
- [Design Principles](../00-overview/design-principles.md) — General design approach
