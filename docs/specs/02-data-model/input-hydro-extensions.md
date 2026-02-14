---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.5.1 (Hydro Geometry — hydro_geometry.parquet)"
  - "DATA_MODEL_SPECIFICATION.md §3.5.2 (Hydro Production Models — hydro_production_models.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.5.3 (Hydro Production Data — hydro_production_data.parquet)"
  - "DATA_MODEL_SPECIFICATION.md §3.5.4 (FPHA Hyperplanes — fpha_hyperplanes.parquet)"
  - "DATA_MODEL_SPECIFICATION.md §3.5.5 (Pumping Stations — pumping_stations.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.5.6 (Energy Contracts — energy_contracts.json)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.5.1-§3.5.6"
---

# Input Hydro Extensions

## Purpose

This spec defines the optional extension files for the hydro subsystem: reservoir geometry, production function models, pre-computed FPHA hyperplanes, pumping stations, and energy contracts. These files augment the core hydro registry defined in [Input System Entities](input-system-entities.md) and live under `system/` in the input case directory.

For deferred hydro-related features (non-controllable sources, battery storage), see [Deferred Features](../06-deferred/deferred-features.md).

## 1. Hydro Geometry (`system/hydro_geometry.parquet`) — Optional

> **Format Rationale — hydro_geometry.parquet**
>
> **Time series** — Per-entity tabular data (head-storage curves) with many rows per hydro. Parquet gives columnar compression and efficient partial reads for this Volume-Height-Area relationship table.

Defines the Volume-Height-Area relationship for reservoirs, enabling accurate evaporation calculation. Instead of complex polynomials, uses a tabular approach with linear interpolation.

**Evaporation Calculation**: The evaporated flow depends on reservoir surface area and the evaporation coefficient:

```
Q_evap(V) = evap_coef_mm × A(V) × conversion_factor
```

where `conversion_factor = 1e-3 / (86400 × days_in_stage)` converts mm to m³/s.

**Linear Approximation in LP**: Since `A(V)` is nonlinear, a first-order Taylor approximation around reference volume `V_ref` is used:

```
Q_evap ≈ k_evap_0 + k_evap_V × V_avg
```

| Term       | Definition                                                            |
| ---------- | --------------------------------------------------------------------- |
| `V_avg`    | `(V_start + V_end) / 2` — average storage over the stage              |
| `k_evap_V` | `evap_coef × dA/dV` — slope (computed from geometry table at `V_ref`) |
| `k_evap_0` | `evap_coef × (A(V_ref) - dA/dV × V_ref)` — intercept                  |

Coefficients are recomputed per stage as the reference volume changes based on the previous stage's solution.

**Filling State Evaporation**: During the filling state (before `entry_stage_id`), the reservoir may operate below `min_storage_hm3`. Evaporation during filling uses the geometry at `min_storage_hm3` as a conservative simplification (smaller volumes have smaller areas).

### Schema

| Column       | Type | Description                                   |
| ------------ | ---- | --------------------------------------------- |
| `hydro_id`   | i32  | Hydro plant identifier                        |
| `volume_hm3` | f64  | Total volume (hm³) — must include dead volume |
| `height_m`   | f64  | Reservoir surface elevation (m)               |
| `area_km2`   | f64  | Water surface area (km²)                      |

**Example rows (Sobradinho):**

| hydro_id | volume_hm3 | height_m | area_km2 |
| -------- | ---------- | -------- | -------- |
| 42       | 5447.0     | 380.0    | 800.0    |
| 42       | 8000.0     | 385.0    | 1200.0   |
| 42       | 12500.0    | 390.0    | 2000.0   |
| 42       | 18000.0    | 395.0    | 3000.0   |
| 42       | 28000.0    | 400.0    | 4200.0   |

### Validation

| Rule                 | Description                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Volumes              | Must be monotonically increasing per hydro                                                                            |
| Heights              | Must be monotonically increasing with volume                                                                          |
| Areas                | Must be monotonically increasing with height                                                                          |
| Minimum volume entry | Should be at or below `min_storage_hm3`                                                                               |
| Maximum volume entry | Should be at or above `max_storage_hm3`                                                                               |
| Dead volume geometry | Geometry below `min_storage_hm3` is optional; if not provided, filling evaporation uses geometry at `min_storage_hm3` |

## 2. Hydro Production Models (`system/hydro_production_models.json`) — Optional

> **Format Rationale — hydro_production_models.json**
>
> **Registry** — Small config selecting model type per hydro with nested optional params and stage-range configuration. JSON handles optional/nested structures well.

Configures the hydro production function (HPF) modeling approach per stage range. Different stages can use different accuracy levels. See [Hydro Production Functions](../01-math/hydro-production-models.md) for the mathematical formulation.

**Model Hierarchy** (increasing complexity and accuracy):

1. **`constant_productivity`**: $g^{hy} = \rho \times q$ — single multiplication, fastest
2. **`linearized_head`**: $g^{hy} = \rho \times q \times (k_0 + k_v \times v_{avg})$ — accounts for head variation with storage
3. **`fpha`**: $g^{hy} \leq \gamma_0^m + \gamma_v^m \times v_{avg} + \gamma_q^m \times q + \gamma_s^m \times s$ — full piecewise-linear approximation

**Default Behavior**: If this file is not provided or a hydro is not listed, the model uses the `generation.model` field from `hydros.json` for all stages.

```json
{
  "production_models": [
    {
      "hydro_id": 0,
      "stage_ranges": [
        {
          "start_stage_id": 0,
          "end_stage_id": 24,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 7,
            "turbine_discretization_points": 15,
            "fitting_window": {
              "volume_min_hm3": null,
              "volume_max_hm3": null
            }
          }
        },
        {
          "start_stage_id": 25,
          "end_stage_id": 60,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 5,
            "turbine_discretization_points": 10,
            "fitting_window": {
              "volume_min_percentile": 10,
              "volume_max_percentile": 90
            }
          }
        },
        {
          "start_stage_id": 61,
          "end_stage_id": null,
          "model": "constant_productivity"
        }
      ]
    },
    {
      "hydro_id": 5,
      "stage_ranges": [
        {
          "start_stage_id": 0,
          "end_stage_id": null,
          "model": "fpha",
          "fpha_config": {
            "source": "precomputed"
          }
        }
      ]
    }
  ]
}
```

### Production Model Comparison

| Model                   | LP Complexity                                                                        | Accuracy | Use Case                                      |
| ----------------------- | ------------------------------------------------------------------------------------ | -------- | --------------------------------------------- |
| `constant_productivity` | 1 constraint: $g^{hy} = \rho \times q$                                               | Low      | Long-term stages, run-of-river, quick studies |
| `linearized_head`       | 1 constraint: $g^{hy} = \rho \times q \times h_{linear}(v)$                          | Medium   | Medium-term stages, moderate head variation   |
| `fpha`                  | M constraints: $g^{hy} \leq \gamma_0^m + \gamma_v^m v + \gamma_q^m q + \gamma_s^m s$ | High     | Near-term stages, significant head variation  |

### FPHA Configuration Fields

| Field                                  | Type   | Description                                                                           |
| -------------------------------------- | ------ | ------------------------------------------------------------------------------------- |
| `source`                               | string | `"computed"` (fit from topology) or `"precomputed"` (from `fpha_hyperplanes.parquet`) |
| `volume_discretization_points`         | i32    | Number of volume points in grid (default: 5)                                          |
| `turbine_discretization_points`        | i32    | Number of turbine flow points (default: 10)                                           |
| `fitting_window.volume_min_hm3`        | f64?   | Explicit minimum volume for fitting (null = physical min)                             |
| `fitting_window.volume_max_hm3`        | f64?   | Explicit maximum volume for fitting (null = physical max)                             |
| `fitting_window.volume_min_percentile` | f64?   | Minimum as percentile of operating range                                              |
| `fitting_window.volume_max_percentile` | f64?   | Maximum as percentile of operating range                                              |

Use absolute bounds (`volume_min_hm3`, `volume_max_hm3`) OR percentiles, not both. Percentiles are relative to `[min_storage_hm3, max_storage_hm3]` from `hydros.json`.

### Required Data by Model

| Model                   | `hydros.json`              | `hydro_geometry.parquet` | `hydro_production_data.parquet` | `fpha_hyperplanes.parquet` |
| ----------------------- | -------------------------- | ------------------------ | ------------------------------- | -------------------------- |
| `constant_productivity` | `productivity_mw_per_m3s`  | ✗                        | ✗                               | ✗                          |
| `linearized_head`       | `productivity_mw_per_m3s`  | ✓                        | ✗                               | ✗                          |
| `fpha` (computed)       | `productivity_mw_per_m3s`¹ | ✓                        | ✓ (for tailrace/losses)         | ✗                          |
| `fpha` (precomputed)    | `productivity_mw_per_m3s`¹ | ✗                        | ✗                               | ✓                          |

¹ Used as fallback for stages without FPHA configuration.

## 3. Hydro Production Data (`system/hydro_production_data.parquet`) — Optional

> **Format Rationale — hydro_production_data.parquet**
>
> **Time series** — Per-entity tabular data (turbine/efficiency curves) with many rows. Parquet for typed columns and compression.

Provides additional data for detailed production function modeling: tailrace polynomials, hydraulic losses, and efficiency curves.

| Column                 | Type  | Description                                   |
| ---------------------- | ----- | --------------------------------------------- |
| `hydro_id`             | i32   | Hydro plant identifier                        |
| `tailrace_type`        | str   | `"polynomial"` or `"piecewise"`               |
| `tailrace_coeffs`      | [f64] | Polynomial coefficients for h_jus(Q_jus)      |
| `hydraulic_loss_type`  | str   | `"factor"` (p.u.) or `"constant"` (m)         |
| `hydraulic_loss_value` | f64   | Loss factor or constant head loss             |
| `efficiency_type`      | str   | `"constant"`, `"flow_dependent"`, or `"grid"` |
| `efficiency_value`     | f64   | Constant efficiency (if type = `"constant"`)  |

**Fallback values** (when `fpha` model is used without this data):

| Assumption       | Fallback Value                                                       |
| ---------------- | -------------------------------------------------------------------- |
| Tailrace         | Constant downstream level from `hydro_geometry.parquet` lowest point |
| Hydraulic losses | Zero losses                                                          |
| Efficiency       | Constant from `productivity_mw_per_m3s`                              |

## 4. FPHA Hyperplanes (`system/fpha_hyperplanes.parquet`) — Optional

> **Format Rationale — fpha_hyperplanes.parquet**
>
> **Time series** — Large per-entity tabular data (pre-computed hyperplane coefficients). Parquet for efficient batch reads of many rows per hydro.

Pre-computed FPHA hyperplane coefficients for hydro production function modeling. Allows using externally-fitted planes instead of computing them at runtime.

| Use Case                 | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| Legacy system migration  | Import FPHA coefficients from DECOMP/DESSEM input files |
| External calibration     | Use coefficients fitted by specialized tools            |
| Performance optimization | Skip runtime fitting for large systems                  |

If not provided, POWE.RS computes hyperplanes from `hydro_geometry.parquet` and `hydro_production_data.parquet` during preprocessing.

### Schema

| Column            | Type | Required | Description                                                  |
| ----------------- | ---- | -------- | ------------------------------------------------------------ |
| `hydro_id`        | i32  | Yes      | Hydro plant identifier                                       |
| `plane_id`        | i32  | Yes      | Plane index within hydro (0 to M-1)                          |
| `gamma_0`         | f64  | Yes      | Intercept coefficient (MW)                                   |
| `gamma_v`         | f64  | Yes      | Volume coefficient (MW/hm³)                                  |
| `gamma_q`         | f64  | Yes      | Turbined flow coefficient (MW per m³/s)                      |
| `gamma_s`         | f64  | Yes      | Spillage coefficient (MW per m³/s, typically ≤ 0)            |
| `alpha_fpha`      | f64  | No       | Correction factor (default: 1.0, already applied to gamma_0) |
| `valid_v_min_hm3` | f64  | No       | Volume range minimum where plane is valid                    |
| `valid_v_max_hm3` | f64  | No       | Volume range maximum where plane is valid                    |
| `valid_q_max_m3s` | f64  | No       | Maximum turbined flow where plane is valid                   |

**Example rows (Itaipu):**

| hydro_id | plane_id | gamma_0 | gamma_v | gamma_q | gamma_s | alpha_fpha |
| -------- | -------- | ------- | ------- | ------- | ------- | ---------- |
| 66       | 0        | 1250.5  | 0.0023  | 0.892   | -0.015  | 0.985      |
| 66       | 1        | 1180.2  | 0.0031  | 0.875   | -0.012  | 0.985      |
| 66       | 2        | 1320.8  | 0.0018  | 0.901   | -0.018  | 0.985      |
| 66       | 3        | 1095.4  | 0.0042  | 0.858   | -0.010  | 0.985      |
| 66       | 4        | 1410.1  | 0.0012  | 0.915   | -0.022  | 0.985      |

**Constraint Form**: Each row defines a constraint:

```
GH ≤ alpha_fpha × (gamma_0 + gamma_v × V_avg + gamma_q × Q + gamma_s × S)
```

If `alpha_fpha` is provided, it is assumed **not yet applied** to `gamma_0`. The solver multiplies `gamma_0 × alpha_fpha` when building constraints. If `alpha_fpha = 1.0` (or null), `gamma_0` already includes any correction factor.

### Validation

| Rule            | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| Minimum planes  | Each `hydro_id` should have at least 3 planes                     |
| Typical range   | 5–30 planes per hydro                                             |
| `gamma_v`       | Should be positive (higher storage → higher generation)           |
| `gamma_q`       | Should be positive (more flow → more generation)                  |
| `gamma_s`       | Should be negative or zero (spillage reduces effective head)      |
| Validity ranges | If provided, planes are only activated when hydro is within range |

## 5. Pumping Stations (`system/pumping_stations.json`) — Optional

> **Format Rationale — pumping_stations.json**
>
> **Registry** — Small set of pump-turbine coupling definitions with cross-references to hydros and buses. JSON is natural for structured entities with unique IDs.

Models pumped storage and water transfer stations (elevatórias) that pump water from a downstream reservoir to an upstream reservoir, consuming electric power.

| Application             | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| Pumped hydro storage    | Store energy by pumping water to upper reservoir during low-demand periods     |
| Inter-basin transfers   | Move water between river basins for irrigation or energy optimization          |
| Reversible hydro plants | Plants that can both generate and pump (model as hydro + pumping station pair) |

**LP Variables**: `pumped_flow` (m³/s), `pumping_power_consumption` (MW)

| Constraint        | Formula                                                                             |
| ----------------- | ----------------------------------------------------------------------------------- |
| Power consumption | `pumping_power_consumption = pumped_flow × consumption_rate`                        |
| Power effect      | Pumping power is added to bus load (demand side)                                    |
| Water balance     | Pumped flow added to destination hydro inflow, subtracted from source hydro balance |

```json
{
  "pumping_stations": [
    {
      "id": 0,
      "name": "SANTA_CECILIA",
      "bus_id": 0,
      "source_hydro_id": 5,
      "destination_hydro_id": 10,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "consumption_mw_per_m3s": 0.85,
      "flow": {
        "min_m3s": 0.0,
        "max_m3s": 150.0
      }
    }
  ]
}
```

| Field                    | Type   | Description                           |
| ------------------------ | ------ | ------------------------------------- |
| `id`                     | i32    | Unique station identifier             |
| `name`                   | string | Station name                          |
| `bus_id`                 | i32    | Bus where power is consumed           |
| `source_hydro_id`        | i32    | Downstream hydro (water origin)       |
| `destination_hydro_id`   | i32    | Upstream hydro (water destination)    |
| `entry_stage_id`         | i32?   | First operating stage (null = always) |
| `exit_stage_id`          | i32?   | Last operating stage (null = forever) |
| `consumption_mw_per_m3s` | f64    | Power consumption rate                |
| `flow.min_m3s`           | f64    | Minimum pumped flow                   |
| `flow.max_m3s`           | f64    | Maximum pumped flow                   |

## 6. Energy Contracts (`system/energy_contracts.json`) — Optional

> **Format Rationale — energy_contracts.json**
>
> **Registry** — Contract definitions with nested structure and cross-references to buses. JSON is natural for structured entities with unique IDs.

Models energy import/export contracts with external systems (neighboring countries, bilateral contracts). Import adds to supply; export adds to demand in the bus load balance.

```json
{
  "contracts": [
    {
      "id": 0,
      "name": "ITAIPU_BR",
      "bus_id": 0,
      "type": "import",
      "entry_stage_id": null,
      "exit_stage_id": null,
      "price_per_mwh": 50.0,
      "limits": {
        "min_mw": 0.0,
        "max_mw": 6000.0
      }
    },
    {
      "id": 1,
      "name": "ARGENTINA_EXPORT",
      "bus_id": 0,
      "type": "export",
      "entry_stage_id": null,
      "exit_stage_id": null,
      "price_per_mwh": -30.0,
      "limits": {
        "min_mw": 0.0,
        "max_mw": 2000.0
      }
    }
  ]
}
```

| Field            | Type   | Description                                                  |
| ---------------- | ------ | ------------------------------------------------------------ |
| `id`             | i32    | Unique contract identifier                                   |
| `name`           | string | Contract name                                                |
| `bus_id`         | i32    | Bus connected to contract                                    |
| `type`           | string | `"import"` (external→system) or `"export"` (system→external) |
| `entry_stage_id` | i32?   | First active stage (null = always)                           |
| `exit_stage_id`  | i32?   | Last active stage (null = forever)                           |
| `price_per_mwh`  | f64    | Cost (import) or revenue (export, typically negative)        |
| `limits.min_mw`  | f64    | Minimum contract usage                                       |
| `limits.max_mw`  | f64    | Maximum contract usage                                       |

### Contract Bounds (`constraints/contract_bounds.parquet`) — Optional

Time-varying overrides for contract limits and prices:

| Column          | Type | Description                      |
| --------------- | ---- | -------------------------------- |
| `contract_id`   | i32  | Contract identifier              |
| `stage_id`      | i32  | Stage index                      |
| `min_mw`        | f64  | Minimum usage (null = use base)  |
| `max_mw`        | f64  | Maximum usage (null = use base)  |
| `price_per_mwh` | f64  | Price override (null = use base) |

## 7. Deferred Features

The following hydro-related features are designed but deferred for future implementation:

- **Non-Controllable Generation Sources** (`system/non_controllable_sources.json`) — Wind farms, solar plants, and other intermittent sources. See [Deferred Features](../06-deferred/deferred-features.md).
- **Battery Storage** (`system/batteries.json`) — Battery energy storage systems (BESS). See [Deferred Features](../06-deferred/deferred-features.md).

## Cross-References

- [Input System Entities](input-system-entities.md) — Core hydro registry (`hydros.json`) that these extensions augment
- [Input Directory Structure](input-directory-structure.md) — Overall case directory layout
- [Input Constraints](input-constraints.md) — Time-varying bounds for hydros, thermals, lines, and contracts
- [Hydro Production Functions](../01-math/hydro-production-models.md) — Mathematical formulation of HPF models
- [LP Formulation](../01-math/lp-formulation.md) — How hydro variables enter the LP
- [Deferred Features](../06-deferred/deferred-features.md) — Non-controllable sources and battery storage
- [Design Principles §3](../00-overview/design-principles.md) — Order invariance and canonical ordering
