---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.3 (Buses — buses.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.4 (Lines — lines.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.5 (Hydro Registry — core schema)"
  - "DATA_MODEL_SPECIFICATION.md §3.6 (Thermal Registry — thermals.json)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.3-§3.6"
---

# Input System Entities

## Purpose

This spec defines the JSON schemas for the four core system entity registries: buses, lines, hydros, and thermals. These files live under `system/` in the input case directory and define the physical model that the optimizer operates on.

For the overall directory layout and configuration, see [Input Directory Structure](input-directory-structure.md).

## 1. Buses (`system/buses.json`)

> **⚠️ Order Invariance**: The order of buses in this array does NOT affect results. After loading, all buses are sorted by `id`. See [Design Principles §3](../00-overview/design-principles.md).

> **Format Rationale — buses.json**
>
> **Registry** — Buses are a small set of entities with cross-references (lines, hydros, and thermals all reference a `bus_id`). JSON is a natural fit for registry data: each entity is a structured object with a unique ID, and the total count is small (typically 2–10 for national systems).

> **Deficit Modeling**: Deficit is modeled as piecewise linear segments. Each segment specifies a depth (MW of unmet demand) and cost. Segments are cumulative: first `depth_mw` MW at first cost, next `depth_mw` MW at second cost, etc. The last segment with `depth_mw: null` extends to infinity (required for LP feasibility). Global defaults are defined in `penalties.json`; entity-level overrides are defined inline here.

```json
{
  "buses": [
    {
      "id": 0,
      "name": "SUDESTE",
      "deficit_segments": [
        { "depth_mw": 1000, "cost": 2000.0 },
        { "depth_mw": 2000, "cost": 5000.0 },
        { "depth_mw": null, "cost": 10000.0 }
      ]
    },
    {
      "id": 1,
      "name": "SUL"
    }
  ]
}
```

> **Note on Bus 1 (SUL)**: No `deficit_segments` defined — uses global default from `penalties.json`.

### Bus Fields

| Field              | Type   | Required | Description                                               |
| ------------------ | ------ | -------- | --------------------------------------------------------- |
| `id`               | i32    | Yes      | Unique bus identifier                                     |
| `name`             | string | Yes      | Human-readable bus name                                   |
| `deficit_segments` | array  | No       | Piecewise deficit cost segments (uses default if omitted) |

### Deficit Segment Fields

| Field      | Type        | Required | Description                                                       |
| ---------- | ----------- | -------- | ----------------------------------------------------------------- |
| `depth_mw` | f64 \| null | Yes      | MW of deficit at this segment (`null` for final infinite segment) |
| `cost`     | f64         | Yes      | Cost per MWh of deficit in this segment                           |

For the mathematical formulation of bus load balance constraints and deficit variables, see [System Element Modeling](../01-math/system-elements.md).

## 2. Lines (`system/lines.json`)

> **⚠️ Order Invariance**: The order of lines in this array does NOT affect results. After loading, all lines are sorted by `id`. See [Design Principles §3](../00-overview/design-principles.md).

> **Format Rationale — lines.json**
>
> **Registry** — Lines are entity definitions referencing buses. Each line connects exactly two buses and has capacity, cost, and lifecycle attributes. JSON is a natural fit because the line count is small (typically 5–20) and each line is a self-contained structured object with cross-references to bus IDs.

> **Exchange Cost**: The `exchange_cost` field is an operational cost, NOT a violation penalty. It discourages unnecessary power flow between buses. Default is defined in `penalties.json`.

### Line Operative States

| State            | Condition                                    | LP Variables              |
| ---------------- | -------------------------------------------- | ------------------------- |
| `non_existing`   | Before `entry_stage_id`                      | None (buses isolated)     |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | direct_flow, reverse_flow |
| `decommissioned` | After `exit_stage_id`                        | None (buses isolated)     |

```json
{
  "lines": [
    {
      "id": 0,
      "name": "SE-NE",
      "source_bus_id": 0,
      "target_bus_id": 1,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "capacity": {
        "direct_mw": 5000.0,
        "reverse_mw": 3000.0
      },
      "exchange_cost": 0.01,
      "losses_percent": 2.5
    }
  ]
}
```

### Line Fields

| Field                 | Type        | Required | Description                                             |
| --------------------- | ----------- | -------- | ------------------------------------------------------- |
| `id`                  | i32         | Yes      | Unique line identifier                                  |
| `name`                | string      | Yes      | Human-readable line name                                |
| `source_bus_id`       | i32         | Yes      | Source bus for direct flow                              |
| `target_bus_id`       | i32         | Yes      | Target bus for direct flow                              |
| `entry_stage_id`      | i32 \| null | No       | Stage when line enters service (`null` = always exists) |
| `exit_stage_id`       | i32 \| null | No       | Stage when line is decommissioned (`null` = never)      |
| `capacity.direct_mw`  | f64         | Yes      | Maximum flow from source to target (MW)                 |
| `capacity.reverse_mw` | f64         | Yes      | Maximum flow from target to source (MW)                 |
| `exchange_cost`       | f64         | No       | Cost per MWh exchanged (uses default if omitted)        |
| `losses_percent`      | f64         | No       | Transmission losses as percentage (default: 0)          |

For the mathematical formulation of exchange variables and capacity constraints, see [System Element Modeling](../01-math/system-elements.md).

## 3. Hydro Registry (`system/hydros.json`) — Core Schema

> **⚠️ Order Invariance**: The order of hydros in this array does NOT affect results. After loading, hydros are sorted by `id`. See [Design Principles §3](../00-overview/design-principles.md).

> **Format Rationale — hydros.json**
>
> **Registry** — Hydro plants are complex entities with many optional nested fields: reservoir bounds, outflow limits, generation model selection, diversion channels, filling configuration, and optional penalty overrides. JSON handles optional/nested structures well — absent fields use defaults, and the hierarchical layout mirrors the physical structure of the plant.

> **Note**: The `generation` field supports multiple modeling approaches for the hydro production function. Different models can be used for different stages via `hydro_production_models.json`. See [Input Hydro Extensions](input-hydro-extensions.md) for geometry, FPHA, production data, pumping stations, and energy contracts.
>
> Inflow models are defined per hydro × stage in `scenarios/inflow_models.parquet`, linked by `hydro_id`.

### Hydro Operative States

| State            | Condition                                                 | LP Variables                                                                               |
| ---------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `non_existing`   | Before filling or entry (no filling defined)              | None                                                                                       |
| `filling`        | Between `filling.start_stage_id` and `entry_stage_id - 1` | storage, outflow (=spillage), violation slacks, evaporation                                |
| `operating`      | Between `entry_stage_id` and `exit_stage_id`              | storage, turbined, spillage, diversion, outflow, generation, evaporation, violation slacks |
| `decommissioned` | After `exit_stage_id`                                     | None                                                                                       |

> **Dead-volume filling**: During filling stages, the reservoir accumulates water. `turbined_flow = 0` (hard constraint), and all released water goes through spillways/bottom gates. Environmental flow must be met via spillage.
>
> **Cascade redirection**: The `downstream_id` always refers to the physical downstream plant. During stages when the downstream plant doesn't exist, outflows are automatically redirected to the next operating downstream in the cascade.
>
> **Penalties**: Penalty defaults are defined in `penalties.json`. Entity-level overrides can be specified in an optional `penalties` block in the hydro definition. Stage-varying overrides are defined in `hydro_penalties.parquet`. See [Penalty System](penalty-system.md).

```json
{
  "hydros": [
    {
      "id": 0,
      "name": "FURNAS",
      "bus_id": 0,
      "downstream_id": 2,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "filling": null,
      "diversion": null,
      "reservoir": {
        "min_storage_hm3": 5733.0,
        "max_storage_hm3": 22950.0
      },
      "outflow": {
        "min_outflow_m3s": 0.0,
        "max_outflow_m3s": null
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.8765,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1692.0,
        "min_generation_mw": null,
        "max_generation_mw": null
      }
    },
    {
      "id": 10,
      "name": "NEW_HYDRO",
      "bus_id": 0,
      "downstream_id": 0,
      "entry_stage_id": 60,
      "exit_stage_id": null,
      "filling": {
        "start_stage_id": 48,
        "target_storage_hm3": 2500.0
      },
      "diversion": null,
      "reservoir": {
        "min_storage_hm3": 1000.0,
        "max_storage_hm3": 5000.0
      },
      "outflow": {
        "min_outflow_m3s": 50.0,
        "max_outflow_m3s": null
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.95,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 500.0,
        "min_generation_mw": null,
        "max_generation_mw": null
      }
    }
  ]
}
```

### Core Hydro Fields

| Field                                | Type           | Required | Description                                                              |
| ------------------------------------ | -------------- | -------- | ------------------------------------------------------------------------ |
| `id`                                 | i32            | Yes      | Unique hydro identifier                                                  |
| `name`                               | string         | Yes      | Human-readable hydro name                                                |
| `bus_id`                             | i32            | Yes      | Bus where generation is injected                                         |
| `downstream_id`                      | i32 \| null    | Yes      | Physical downstream hydro (`null` if none)                               |
| `entry_stage_id`                     | i32 \| null    | No       | Stage when hydro enters operation (`null` = always)                      |
| `exit_stage_id`                      | i32 \| null    | No       | Stage when hydro is decommissioned (`null` = never)                      |
| `filling`                            | object \| null | No       | Dead-volume filling configuration (`null` = no filling)                  |
| `filling.start_stage_id`             | i32            | Yes¹     | First filling stage                                                      |
| `filling.target_storage_hm3`         | f64            | Yes¹     | Target storage at end of filling                                         |
| `diversion`                          | object \| null | No       | Diversion channel configuration (`null` = no diversion)                  |
| `diversion.downstream_id`            | i32            | Yes²     | Hydro plant receiving diverted water                                     |
| `diversion.max_flow_m3s`             | f64            | Yes²     | Maximum diversion flow                                                   |
| `reservoir.min_storage_hm3`          | f64            | Yes      | Minimum storage (dead volume)                                            |
| `reservoir.max_storage_hm3`          | f64            | Yes      | Maximum storage (full reservoir)                                         |
| `outflow.min_outflow_m3s`            | f64            | Yes      | Minimum total outflow (environmental flow)                               |
| `outflow.max_outflow_m3s`            | f64 \| null    | Yes      | Maximum outflow (`null` = unbounded)                                     |
| `generation.model`                   | string         | Yes      | `"constant_productivity"`, `"linearized_head"`, or `"fpha"`              |
| `generation.productivity_mw_per_m3s` | f64            | Yes      | Constant productivity factor                                             |
| `generation.min_turbined_m3s`        | f64            | Yes      | Minimum turbined flow                                                    |
| `generation.max_turbined_m3s`        | f64            | Yes      | Maximum turbined flow (machine capacity)                                 |
| `generation.min_generation_mw`       | f64 \| null    | No       | Minimum generation bound (`null` = derived)                              |
| `generation.max_generation_mw`       | f64 \| null    | No       | Maximum generation bound (`null` = derived)                              |
| `penalties`                          | object         | No       | Entity-level penalty overrides (see [Penalty System](penalty-system.md)) |

> ¹ Required when `filling` is not null. ² Required when `diversion` is not null.

### Diversion Channel

> **LP Modeling**: Diversion creates an additional flow variable `diversion_flow` bounded by `[0, max_flow_m3s]`. Diverted water is subtracted from the plant's balance and added to `diversion.downstream_id`'s inflow. It does NOT generate power. The `diversion_cost` from [Penalty System](penalty-system.md) incentivizes avoiding diversion unless necessary.

### Hydro Extensions

The following hydro-related files are documented in [Input Hydro Extensions](input-hydro-extensions.md):

| File                                   | Purpose                                            |
| -------------------------------------- | -------------------------------------------------- |
| `system/hydro_geometry.parquet`        | Volume-Height-Area relationship for evaporation    |
| `system/hydro_production_models.json`  | Stage-dependent production model selection         |
| `system/hydro_production_data.parquet` | Tailrace polynomials, hydraulic losses, efficiency |
| `system/fpha_hyperplanes.parquet`      | Pre-computed FPHA hyperplane coefficients          |
| `system/pumping_stations.json`         | Pumped storage and water transfer stations         |
| `system/energy_contracts.json`         | Import/export energy contracts                     |

For the mathematical formulation of hydro water balance, production function, and constraint details, see [System Element Modeling](../01-math/system-elements.md) and [Equipment Formulations](../01-math/equipment-formulations.md).

## 4. Thermal Registry (`system/thermals.json`)

> **⚠️ Order Invariance**: The order of thermals in this array does NOT affect results. After loading, thermals are sorted by `id`. See [Design Principles §3](../00-overview/design-principles.md).

> **Format Rationale — thermals.json**
>
> **Registry** — Thermal plants are entity definitions with cost curve nesting. Each thermal has a piecewise-linear cost curve represented as an array of `cost_segments`, plus generation bounds and lifecycle attributes. JSON is a natural fit because cost curves are naturally nested arrays within each plant object, and the total count is moderate (typically 50–200).

> **Note**: Use `entry_stage_id` and `exit_stage_id` to model plants entering or exiting the system. Alternatively, use `thermal_bounds.parquet` to set generation to 0 for stages where the plant is offline.

### Thermal Operative States

| State            | Condition                                    | LP Variables           |
| ---------------- | -------------------------------------------- | ---------------------- |
| `non_existing`   | Before `entry_stage_id`                      | None                   |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | generation per segment |
| `decommissioned` | After `exit_stage_id`                        | None                   |

```json
{
  "thermals": [
    {
      "id": 0,
      "name": "ANGRA1",
      "bus_id": 0,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "cost_segments": [{ "capacity_mw": 640.0, "cost_per_mwh": 15.0 }],
      "generation": {
        "min_mw": 500.0,
        "max_mw": 640.0
      }
    }
  ]
}
```

### Thermal Fields

| Field                          | Type           | Required | Description                                             |
| ------------------------------ | -------------- | -------- | ------------------------------------------------------- |
| `id`                           | i32            | Yes      | Unique thermal identifier                               |
| `name`                         | string         | Yes      | Human-readable thermal name                             |
| `bus_id`                       | i32            | Yes      | Bus where generation is injected                        |
| `entry_stage_id`               | i32 \| null    | No       | Stage when thermal enters service (`null` = always)     |
| `exit_stage_id`                | i32 \| null    | No       | Stage when thermal is decommissioned (`null` = never)   |
| `cost_segments`                | array          | Yes      | Piecewise-linear cost curve (ordered by ascending cost) |
| `cost_segments[].capacity_mw`  | f64            | Yes      | Capacity of this cost segment (MW)                      |
| `cost_segments[].cost_per_mwh` | f64            | Yes      | Marginal cost in this segment ($/MWh)                   |
| `generation.min_mw`            | f64            | Yes      | Minimum stable generation (0 if no minimum)             |
| `generation.max_mw`            | f64            | Yes      | Maximum generation capacity                             |
| `gnl_config`                   | object \| null | No       | GNL dispatch anticipation (see below)                   |

### Thermal Bounds Override (`constraints/thermal_bounds.parquet`)

Stage-varying generation bounds can override the base values from `thermals.json`:

| Column              | Type | Description                          |
| ------------------- | ---- | ------------------------------------ |
| `thermal_id`        | i32  | Thermal unit identifier              |
| `stage_id`          | i32  | Stage index                          |
| `min_generation_mw` | f64  | Minimum generation (null = use base) |
| `max_generation_mw` | f64  | Maximum generation (null = use base) |

> **Note**: If a thermal is not present for a stage, base bounds from `thermals.json` are used. Partial overrides allowed (only specify stages that differ from base).

### GNL Thermal Plants (Deferred)

> **🚧 Implementation Status**: Data model is ready. Implementation is planned but not yet complete.

GNL (Gas Natural Liquefeito) plants require **dispatch anticipation**: the dispatch decision must be made N stages ahead due to fuel ordering lead times. This creates additional state variables.

```json
{
  "id": 10,
  "name": "GNL_PLANT",
  "bus_id": 2,
  "gnl_config": {
    "lag_stages": 2
  },
  "cost_segments": [{ "capacity_mw": 500.0, "cost_per_mwh": 200.0 }],
  "generation": {
    "min_mw": 0.0,
    "max_mw": 500.0
  }
}
```

| Field                   | Type           | Description                                                 |
| ----------------------- | -------------- | ----------------------------------------------------------- |
| `gnl_config`            | object \| null | GNL configuration. If null or omitted, thermal is standard. |
| `gnl_config.lag_stages` | i32            | Number of stages ahead for dispatch decision.               |

When configured, the algorithm adds state variables for the committed dispatch pipeline:

| State Variable                   | Description                      |
| -------------------------------- | -------------------------------- |
| `gnl_committed[thermal_id, t+1]` | Dispatch committed for stage t+1 |
| `gnl_committed[thermal_id, t+2]` | Dispatch committed for stage t+2 |
| ...                              | ... up to `lag_stages` ahead     |

Initial values for the GNL pipeline are specified in `initial_conditions.json`.

For the mathematical formulation of thermal cost curves and generation constraints, see [System Element Modeling](../01-math/system-elements.md).

## Cross-References

- [Input Directory Structure](input-directory-structure.md) — case directory layout and `config.json` schema
- [Input Hydro Extensions](input-hydro-extensions.md) — geometry, FPHA, production data, pumping, contracts
- [Penalty System](penalty-system.md) — three-tier penalty cascade and all penalty schemas
- [Design Principles](../00-overview/design-principles.md) — format selection criteria and order invariance
- [System Element Modeling](../01-math/system-elements.md) — mathematical formulation of all system elements
- [Equipment Formulations](../01-math/equipment-formulations.md) — detailed per-equipment LP constraints
- [Hydro Production Models](../01-math/hydro-production-models.md) — FPHA and linearized head formulations
