---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §4.1 (Directory Structure Overview)"
  - "DATA_MODEL_SPECIFICATION.md §4.2 (Design Principles)"
  - "DATA_MODEL_SPECIFICATION.md §4.3 (Categorical Code Definitions)"
  - "DATA_MODEL_SPECIFICATION.md §4.4 (Dictionary Files)"
  - "DATA_MODEL_SPECIFICATION.md §4.5 (Simulation Output Schemas)"
  - "DATA_MODEL_SPECIFICATION.md §4.6 (Training Output Schemas)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §4.1-4.6"
---

# Output Schemas

## Purpose

This spec defines the Parquet schemas for all output files produced by POWE.RS during simulation (policy evaluation) and training (policy construction). It covers the output directory layout, column-level schemas for every entity type, and the dictionary/metadata files that make outputs self-documenting.

For output infrastructure (manifests, MPI partitioning, configuration, scale reference, validation), see [Output Infrastructure](output-infrastructure.md).

## 1. Directory Structure

```
output/
├── simulation/                              # Hive-partitioned simulation results
│   ├── _manifest.json                       # Checksums, row counts, partitions
│   ├── _SUCCESS                             # Marker on successful completion
│   ├── costs/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── hydros/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── thermals/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── exchanges/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── buses/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── pumping_stations/                    # Optional: only if entity exists
│   │   └── scenario_id=XXXX/data.parquet
│   ├── contracts/                           # Optional: only if entity exists
│   │   └── scenario_id=XXXX/data.parquet
│   ├── batteries/                           # 🚧 DEFERRED
│   │   └── scenario_id=XXXX/data.parquet
│   ├── non_controllables/                   # 🚧 DEFERRED
│   │   └── scenario_id=XXXX/data.parquet
│   ├── inflow_lags/                         # Optional: only if AR order > 0
│   │   └── scenario_id=XXXX/data.parquet
│   └── violations/
│       └── generic/
│           └── scenario_id=XXXX/data.parquet
│
└── training/                                # Training phase outputs
    ├── _manifest.json
    ├── _SUCCESS
    ├── convergence.parquet                  # Iteration-level convergence
    ├── timing/
    │   ├── iterations.parquet               # Per-iteration timing breakdown
    │   └── mpi_ranks.parquet                # Per-rank timing statistics
    ├── dictionaries/
    │   ├── codes.json                       # Categorical code mappings
    │   ├── bounds.parquet                   # Entity bounds by stage/block
    │   ├── state_dictionary.json            # State space definition
    │   ├── variables.csv                    # Variable metadata
    │   └── entities.csv                     # Entity metadata
    └── metadata.json                        # Run configuration and system info
```

> **Optional files**: Simulation entity directories are only written if the corresponding entities exist in the input model.

## 2. Design Principles

### 2.1 Hive Partitioning by Scenario

Simulation outputs use Hive-style partitioning by `scenario_id`:

- **Parallel writes**: Each MPI rank writes exclusively to its assigned scenario partitions
- **Partition pruning**: Queries filtering by scenario read only relevant files
- **Incremental updates**: Individual scenarios can be recomputed without rewriting all data

**Partition naming**: `{entity_type}/scenario_id={scenario_id:04d}/data.parquet`

The `scenario_id` column is NOT stored in Parquet data — it is derived from the partition path.

### 2.2 Categorical Encoding

All categorical columns use integer codes with mappings in `dictionaries/codes.json`:

- Reduces storage (i8 vs variable-length strings)
- Enables efficient filtering and grouping
- **Convention**: categorical columns end with `_code` suffix (e.g., `operative_state_code`)

### 2.3 Constraint Violation Handling

Two mechanisms:

1. **Slack columns in entity files**: Physical bound violations (e.g., `turbined_slack_m3s`) appear as dedicated columns, value 0 when no violation
2. **Generic violations file**: User-defined generic constraint violations in `violations/generic/`

### 2.4 File Naming Conventions

| Convention                   | Example                         | Rationale                |
| ---------------------------- | ------------------------------- | ------------------------ |
| Plural entity names          | `hydros.parquet`                | Multiple records         |
| Lowercase with underscores   | `pumping_stations/`             | Filesystem-safe          |
| `data.parquet` in partitions | `scenario_id=0001/data.parquet` | Standard Hive convention |

## 3. Categorical Codes (`dictionaries/codes.json`)

```json
{
  "version": "2.0.0",
  "generated_at": "2026-01-18T12:00:00Z",
  "operative_state": {
    "0": "non_existing",
    "1": "filling_dead_volume",
    "2": "operating",
    "3": "decommissioned"
  },
  "storage_binding": {
    "0": "none",
    "1": "min",
    "2": "max",
    "3": "target"
  },
  "contract_type": {
    "0": "import",
    "1": "export"
  },
  "entity_type": {
    "0": "hydro",
    "1": "thermal",
    "2": "bus",
    "3": "line",
    "4": "pumping_station",
    "5": "contract",
    "6": "battery",
    "7": "non_controllable"
  },
  "bound_type": {
    "0": "storage_min",
    "1": "storage_max",
    "2": "turbined_min",
    "3": "turbined_max",
    "4": "outflow_min",
    "5": "outflow_max",
    "6": "generation_min",
    "7": "generation_max",
    "8": "flow_min",
    "9": "flow_max"
  }
}
```

**Usage in analysis:**

```python
import json, polars as pl

with open("training/dictionaries/codes.json") as f:
    codes = json.load(f)

df = pl.read_parquet("simulation/hydros/")
df = df.with_columns(
    pl.col("operative_state_code")
      .map_dict({int(k): v for k, v in codes["operative_state"].items()})
      .alias("operative_state")
)
```

## 4. Dictionary Files

### 4.1 Bounds Dictionary (`dictionaries/bounds.parquet`)

Centralizes all entity bounds by stage and block, eliminating redundant bound columns from entity output files.

| Column             | Type | Nullable | Description                                |
| ------------------ | ---- | -------- | ------------------------------------------ |
| `entity_type_code` | i8   | No       | Entity type code (see `codes.json`)        |
| `entity_id`        | i32  | No       | Entity identifier                          |
| `stage_id`         | i32  | No       | Stage index (0-based)                      |
| `block_id`         | i32  | Yes      | Block index (null = applies to all blocks) |
| `bound_type_code`  | i8   | No       | Bound type code (see `codes.json`)         |
| `bound_value`      | f64  | No       | Bound value in native units                |

Bounds are stored only when they differ from default/infinite values.

### 4.2 State Dictionary (`dictionaries/state_dictionary.json`)

Documents the state space structure for the SDDP policy. See [Input Constraints §4](input-constraints.md) for full state schema.

### 4.3 Variables Metadata (`dictionaries/variables.csv`)

| Column        | Type   | Description                                   |
| ------------- | ------ | --------------------------------------------- |
| `file`        | string | Source file (e.g., `hydros`, `thermals`)      |
| `column`      | string | Column name                                   |
| `type`        | string | Data type (`i8`, `i32`, `i64`, `f64`, `bool`) |
| `unit`        | string | Physical unit or null                         |
| `description` | string | Human-readable description                    |
| `nullable`    | bool   | Whether null values are allowed               |

### 4.4 Entities Metadata (`dictionaries/entities.csv`)

| Column             | Type   | Description                   |
| ------------------ | ------ | ----------------------------- |
| `entity_type_code` | i8     | Entity type code              |
| `entity_id`        | i32    | Entity identifier             |
| `name`             | string | Entity name from input        |
| `bus_id`           | i32    | Connected bus (if applicable) |
| `system_id`        | i32    | System/subsystem identifier   |

## 5. Simulation Output Schemas

### 5.1 Costs (`simulation/costs/`)

Stage and block-level cost breakdown for economic analysis.

| Column            | Type | Nullable | Description                                   |
| ----------------- | ---- | -------- | --------------------------------------------- |
| `stage_id`        | i32  | No       | Stage index (0-based)                         |
| `block_id`        | i32  | Yes      | Block index (null for stage-level aggregates) |
| `total_cost`      | f64  | No       | Total stage cost (all components)             |
| `immediate_cost`  | f64  | No       | Stage immediate cost (excluding future cost)  |
| `thermal_cost`    | f64  | No       | Thermal generation cost                       |
| `deficit_cost`    | f64  | No       | Deficit (unmet demand) penalty                |
| `excess_cost`     | f64  | No       | Excess generation penalty                     |
| `spillage_cost`   | f64  | No       | Spillage penalty (all hydros)                 |
| `exchange_cost`   | f64  | No       | Exchange losses and tariffs                   |
| `pumping_cost`    | f64  | No       | Pumping energy cost                           |
| `contract_cost`   | f64  | No       | Import/export contract cost                   |
| `violation_cost`  | f64  | No       | Generic constraint violation penalties        |
| `future_cost`     | f64  | No       | Future cost function value (α)                |
| `discount_factor` | f64  | No       | Cumulative discount factor                    |

**Rows per scenario**: `num_stages × (1 + num_blocks)` (stage-level + block-level rows)

**Cost relationships**:

```
total_cost = immediate_cost + future_cost
immediate_cost = thermal_cost + deficit_cost + excess_cost + spillage_cost
                 + exchange_cost + pumping_cost + contract_cost + violation_cost
```

### 5.2 Hydros (`simulation/hydros/`)

| Column                    | Type | Nullable | Description                                  |
| ------------------------- | ---- | -------- | -------------------------------------------- |
| `stage_id`                | i32  | No       | Stage index (0-based)                        |
| `block_id`                | i32  | Yes      | Block index (null for stage-level)           |
| `hydro_id`                | i32  | No       | Hydro plant identifier                       |
| `turbined_m3s`            | f64  | No       | Turbined outflow (m³/s)                      |
| `spillage_m3s`            | f64  | No       | Spillage (m³/s)                              |
| `outflow_m3s`             | f64  | No       | Total outflow: turbined + spillage (m³/s)    |
| `evaporation_m3s`         | f64  | Yes      | Evaporation loss (m³/s), null if not modeled |
| `diverted_inflow_m3s`     | f64  | Yes      | Inflow diverted from upstream                |
| `diverted_outflow_m3s`    | f64  | Yes      | Outflow diverted to downstream               |
| `incremental_inflow_m3s`  | f64  | No       | Realized incremental inflow (m³/s)           |
| `inflow_m3s`              | f64  | No       | Total inflow including upstream              |
| `storage_initial_hm3`     | f64  | No       | Storage at start (hm³)                       |
| `storage_final_hm3`       | f64  | No       | Storage at end (hm³)                         |
| `generation_mw`           | f64  | No       | Power generation (MW)                        |
| `generation_mwh`          | f64  | No       | Energy generation (MWh)                      |
| `productivity_mw_per_m3s` | f64  | Yes      | Effective productivity                       |
| `turbined_slack_m3s`      | f64  | No       | Min turbined violation (0 if none)           |
| `outflow_slack_m3s`       | f64  | No       | Min outflow violation (0 if none)            |
| `generation_slack_mw`     | f64  | No       | Min generation violation (0 if none)         |
| `spillage_cost`           | f64  | No       | Spillage penalty cost                        |
| `water_value_per_hm3`     | f64  | No       | Marginal value of stored water ($/hm³)       |
| `storage_binding_code`    | i8   | No       | Storage bound binding status                 |
| `operative_state_code`    | i8   | No       | Operative state                              |

**Rows per scenario**: `num_stages × num_blocks × num_hydros`

**Water balance**: `storage_final = storage_initial + (inflow - outflow - evaporation + diverted_inflow - diverted_outflow) × duration`

**Slack interpretation**: value > 0 means the corresponding minimum constraint was relaxed.

### 5.3 Thermals (`simulation/thermals/`)

| Column                 | Type | Nullable | Description                                      |
| ---------------------- | ---- | -------- | ------------------------------------------------ |
| `stage_id`             | i32  | No       | Stage index (0-based)                            |
| `block_id`             | i32  | Yes      | Block index                                      |
| `thermal_id`           | i32  | No       | Thermal unit identifier                          |
| `generation_mw`        | f64  | No       | Power generation (MW)                            |
| `generation_mwh`       | f64  | No       | Energy generation (MWh)                          |
| `generation_cost`      | f64  | No       | Generation cost                                  |
| `is_gnl`               | bool | No       | Whether unit has GNL configuration               |
| `gnl_committed_mw`     | f64  | Yes      | GNL committed capacity (null if not GNL)         |
| `gnl_decision_mw`      | f64  | Yes      | GNL decision for future stages (null if not GNL) |
| `operative_state_code` | i8   | No       | Operative state                                  |

**Rows per scenario**: `num_stages × num_blocks × num_thermals`

**GNL notes**: `gnl_committed_mw` is capacity committed previously and available now; `gnl_decision_mw` is the decision made now for future availability. GNL decisions are state variables coupling stages.

### 5.4 Exchanges (`simulation/exchanges/`)

| Column                 | Type | Nullable | Description                      |
| ---------------------- | ---- | -------- | -------------------------------- |
| `stage_id`             | i32  | No       | Stage index (0-based)            |
| `block_id`             | i32  | Yes      | Block index                      |
| `line_id`              | i32  | No       | Transmission line identifier     |
| `net_flow_mw`          | f64  | No       | Net flow: direct − reverse (MW)  |
| `net_flow_mwh`         | f64  | No       | Net energy flow (MWh)            |
| `exchange_cost`        | f64  | No       | Exchange cost (losses + tariffs) |
| `operative_state_code` | i8   | No       | Operative state                  |

**Rows per scenario**: `num_stages × num_blocks × num_lines`

**Sign convention**: positive = bus_from → bus_to; negative = bus_to → bus_from.

### 5.5 Buses (`simulation/buses/`)

| Column        | Type | Nullable | Description                          |
| ------------- | ---- | -------- | ------------------------------------ |
| `stage_id`    | i32  | No       | Stage index (0-based)                |
| `block_id`    | i32  | Yes      | Block index                          |
| `bus_id`      | i32  | No       | Bus identifier                       |
| `load_mw`     | f64  | No       | Realized load after curtailment (MW) |
| `load_mwh`    | f64  | No       | Realized load energy (MWh)           |
| `deficit_mw`  | f64  | No       | Unmet demand (MW)                    |
| `deficit_mwh` | f64  | No       | Unmet demand energy (MWh)            |
| `excess_mw`   | f64  | No       | Excess generation (MW)               |
| `excess_mwh`  | f64  | No       | Excess generation energy (MWh)       |
| `spot_price`  | f64  | No       | Marginal cost of energy ($/MWh)      |

**Rows per scenario**: `num_stages × num_blocks × num_buses`

**Load balance**: `generation_total + imports − exports + deficit − excess = load`

To compute generation by source, join with `hydros`, `thermals`, etc. using `bus_id` from `entities.csv`.

### 5.6 Pumping Stations (`simulation/pumping_stations/`) — Optional

| Column                   | Type | Nullable | Description                |
| ------------------------ | ---- | -------- | -------------------------- |
| `stage_id`               | i32  | No       | Stage index (0-based)      |
| `block_id`               | i32  | Yes      | Block index                |
| `pumping_station_id`     | i32  | No       | Pumping station identifier |
| `pumped_flow_m3s`        | f64  | No       | Pumped water flow (m³/s)   |
| `pumped_volume_hm3`      | f64  | No       | Pumped volume (hm³)        |
| `power_consumption_mw`   | f64  | No       | Power consumed (MW)        |
| `energy_consumption_mwh` | f64  | No       | Energy consumed (MWh)      |
| `pumping_cost`           | f64  | No       | Total pumping cost         |
| `operative_state_code`   | i8   | No       | Operative state            |

**Rows per scenario**: `num_stages × num_blocks × num_pumping_stations`

### 5.7 Contracts (`simulation/contracts/`) — Optional

| Column                 | Type | Nullable | Description             |
| ---------------------- | ---- | -------- | ----------------------- |
| `stage_id`             | i32  | No       | Stage index (0-based)   |
| `block_id`             | i32  | Yes      | Block index             |
| `contract_id`          | i32  | No       | Contract identifier     |
| `contract_type_code`   | i8   | No       | 0=import, 1=export      |
| `power_mw`             | f64  | No       | Contracted power (MW)   |
| `energy_mwh`           | f64  | No       | Contracted energy (MWh) |
| `price_per_mwh`        | f64  | No       | Contract price ($/MWh)  |
| `total_cost`           | f64  | No       | Total contract cost     |
| `operative_state_code` | i8   | No       | Operative state         |

**Rows per scenario**: `num_stages × num_blocks × num_contracts`

### 5.8 Batteries (`simulation/batteries/`) — 🚧 DEFERRED

| Column                 | Type | Nullable | Description                    |
| ---------------------- | ---- | -------- | ------------------------------ |
| `stage_id`             | i32  | No       | Stage index (0-based)          |
| `block_id`             | i32  | Yes      | Block index                    |
| `battery_id`           | i32  | No       | Battery identifier             |
| `charge_mw`            | f64  | No       | Charging power (MW)            |
| `discharge_mw`         | f64  | No       | Discharging power (MW)         |
| `soc_initial_mwh`      | f64  | No       | State of charge at start (MWh) |
| `soc_final_mwh`        | f64  | No       | State of charge at end (MWh)   |
| `cycle_cost`           | f64  | No       | Cycling degradation cost       |
| `operative_state_code` | i8   | No       | Operative state                |

See [Deferred Features](../06-deferred/deferred-features.md) for implementation timeline.

### 5.9 Non-Controllables (`simulation/non_controllables/`) — 🚧 DEFERRED

| Column                 | Type | Nullable | Description                        |
| ---------------------- | ---- | -------- | ---------------------------------- |
| `stage_id`             | i32  | No       | Stage index (0-based)              |
| `block_id`             | i32  | Yes      | Block index                        |
| `non_controllable_id`  | i32  | No       | Non-controllable source identifier |
| `generation_mw`        | f64  | No       | Realized generation (MW)           |
| `generation_mwh`       | f64  | No       | Realized generation (MWh)          |
| `curtailment_mw`       | f64  | No       | Curtailed generation (MW)          |
| `curtailment_mwh`      | f64  | No       | Curtailed generation (MWh)         |
| `operative_state_code` | i8   | No       | Operative state                    |

See [Deferred Features](../06-deferred/deferred-features.md) for implementation timeline.

### 5.10 Inflow Lags (`simulation/inflow_lags/`) — Optional

| Column       | Type | Nullable | Description                      |
| ------------ | ---- | -------- | -------------------------------- |
| `stage_id`   | i32  | No       | Stage index (0-based)            |
| `hydro_id`   | i32  | No       | Hydro plant identifier           |
| `lag_index`  | i32  | No       | Lag index (1 = t−1, 2 = t−2, …)  |
| `inflow_m3s` | f64  | No       | Inflow value for this lag (m³/s) |

**Rows per scenario**: `num_stages × num_hydros × max_ar_order`

`lag_index` uses 1-based indexing. Maximum lag index equals the AR model order. These values are state variables affecting inflow sampling.

### 5.11 Generic Violations (`simulation/violations/generic/`)

| Column          | Type | Nullable | Description                         |
| --------------- | ---- | -------- | ----------------------------------- |
| `stage_id`      | i32  | No       | Stage index (0-based)               |
| `block_id`      | i32  | Yes      | Block index                         |
| `constraint_id` | i32  | No       | Generic constraint identifier       |
| `slack_value`   | f64  | No       | Violation amount (constraint units) |
| `slack_cost`    | f64  | No       | Penalty cost incurred               |

**Rows per scenario**: `num_stages × num_blocks × num_generic_constraints` (only non-zero violations may be stored).

## 6. Training Output Schemas

### 6.1 Convergence Log (`training/convergence.parquet`)

| Column             | Type | Nullable | Description                                    |
| ------------------ | ---- | -------- | ---------------------------------------------- |
| `iteration`        | i32  | No       | Iteration number (1-based)                     |
| `lower_bound`      | f64  | No       | Lower bound (expected cost-to-go from stage 0) |
| `upper_bound_mean` | f64  | Yes      | Upper bound mean (null if UB disabled)         |
| `upper_bound_std`  | f64  | Yes      | Upper bound standard deviation                 |
| `gap_percent`      | f64  | No       | Optimality gap: `(UB − LB) / LB × 100`         |
| `cuts_added`       | i32  | No       | Cuts added this iteration                      |
| `cuts_removed`     | i32  | No       | Cuts removed by cut selection                  |
| `cuts_active`      | i64  | No       | Total active cuts across all stages            |
| `time_forward_ms`  | i64  | No       | Forward pass wall time (ms)                    |
| `time_backward_ms` | i64  | No       | Backward pass wall time (ms)                   |
| `time_total_ms`    | i64  | No       | Total iteration wall time (ms)                 |
| `memory_peak_mb`   | i64  | No       | Peak memory usage (MB)                         |
| `forward_passes`   | i32  | No       | Forward scenarios this iteration               |
| `lp_solves`        | i64  | No       | Total LP solves this iteration                 |

**Rows**: `num_iterations`

When upper bound is available: `gap_percent = (upper_bound_mean − lower_bound) / |lower_bound| × 100`. If UB evaluation is disabled, gap shows change from previous iteration's lower bound.

### 6.2 Iteration Timing (`training/timing/iterations.parquet`)

| Column              | Type | Nullable | Description                      |
| ------------------- | ---- | -------- | -------------------------------- |
| `iteration`         | i32  | No       | Iteration number (1-based)       |
| `forward_solve_ms`  | i64  | No       | LP solve time in forward pass    |
| `forward_sample_ms` | i64  | No       | Scenario sampling time           |
| `backward_solve_ms` | i64  | No       | LP solve time in backward pass   |
| `backward_cut_ms`   | i64  | No       | Cut computation and storage time |
| `cut_selection_ms`  | i64  | No       | Cut selection/pruning time       |
| `mpi_allreduce_ms`  | i64  | No       | MPI AllReduce communication time |
| `mpi_broadcast_ms`  | i64  | No       | MPI Broadcast communication time |
| `io_write_ms`       | i64  | No       | Output writing time              |
| `overhead_ms`       | i64  | No       | Unaccounted overhead             |

**Rows**: `num_iterations`

### 6.3 MPI Rank Timing (`training/timing/mpi_ranks.parquet`)

| Column                  | Type | Nullable | Description                      |
| ----------------------- | ---- | -------- | -------------------------------- |
| `iteration`             | i32  | No       | Iteration number (1-based)       |
| `rank`                  | i32  | No       | MPI rank (0-based)               |
| `forward_time_ms`       | i64  | No       | Forward pass time on this rank   |
| `backward_time_ms`      | i64  | No       | Backward pass time on this rank  |
| `communication_time_ms` | i64  | No       | MPI communication time           |
| `idle_time_ms`          | i64  | No       | Time waiting for other ranks     |
| `lp_solves`             | i64  | No       | LP solves on this rank           |
| `scenarios_processed`   | i32  | No       | Scenarios processed on this rank |

**Rows**: `num_iterations × num_mpi_ranks`

Use `idle_time_ms` to identify load imbalance. Sum of `scenarios_processed` per iteration equals `forward_passes`. Communication patterns reveal MPI bottlenecks.

## Cross-References

- [Output Infrastructure](output-infrastructure.md) — Manifests, MPI partitioning, config, validation
- [Input System Entities](input-system-entities.md) — Entity registries defining entity IDs
- [Penalty System](penalty-system.md) — Penalty costs that appear in output cost columns
- [Input Constraints](input-constraints.md) — Generic constraints whose violations appear in output
- [LP Formulation](../01-math/lp-formulation.md) — Mathematical definitions of output variables
- [Deferred Features](../06-deferred/deferred-features.md) — Batteries, non-controllable sources
