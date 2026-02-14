---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.9 (Initial Conditions — initial_conditions.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.13 (Constraints — constraints/)"
  - "DATA_MODEL_SPECIFICATION.md §3.14 (Policy Directory — policy/)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.9, §3.13, §3.14"
---

# Input Constraints, Initial Conditions, and Policy

## Purpose

This spec defines the initial system state (storage, GNL pipeline), time-varying operational bounds for all entities, the generic constraint system for custom linear constraints, and the policy directory used for warm-starting and resuming SDDP training.

For entity base schemas (bounds in `hydros.json`, `thermals.json`, `lines.json`), see [Input System Entities](input-system-entities.md). For contract bounds, see [Input Hydro Extensions §6](input-hydro-extensions.md).

## 1. Initial Conditions (`initial_conditions.json`)

> **Format Rationale — initial_conditions.json**
>
> **Registry** — One-time snapshot of system state with cross-references to entities. JSON is natural for config-like data with nested structures.

Initial storage is the reservoir level at the start of the study. For hydros with `entry_stage_id`, this is the storage when they enter the system (not at stage 0).

```json
{
  "storage": [
    { "hydro_id": 0, "value_hm3": 15000.0 },
    { "hydro_id": 1, "value_hm3": 8500.0 },
    { "hydro_id": 10, "value_hm3": 2500.0 }
  ],
  "gnl_pipeline": [
    { "thermal_id": 10, "stage_offset": 1, "committed_mw": 250.0 },
    { "thermal_id": 10, "stage_offset": 2, "committed_mw": 300.0 }
  ]
}
```

### Validation

| Rule              | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| Hydro coverage    | Every hydro in `hydros.json` must have an entry in `storage`      |
| Storage bounds    | Storage value must be within `[min_storage_hm3, max_storage_hm3]` |
| Late-entry hydros | For hydros entering later, this is their initial storage at entry |

### GNL Pipeline Initial Conditions

When GNL thermals are configured (see [Input System Entities §4](input-system-entities.md)), their initial committed dispatch pipeline is specified here:

| Field          | Type | Description                                            |
| -------------- | ---- | ------------------------------------------------------ |
| `thermal_id`   | i32  | ID of the GNL thermal (must have `gnl_config` defined) |
| `stage_offset` | i32  | Future stage offset (1 = stage 1, 2 = stage 2, etc.)   |
| `committed_mw` | f64  | Committed dispatch in MW for that future stage         |

| Validation Rule   | Description                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| `gnl_pipeline`    | Optional. If omitted, GNL thermals start with zero committed dispatch              |
| Stage coverage    | Each GNL thermal with `lag_stages = N` should have entries for offsets 1 through N |
| Thermal reference | `thermal_id` must reference a thermal with `gnl_config` defined                    |
| Bounds            | `committed_mw` must be within the thermal's generation bounds                      |

### Inflow History (`scenarios/inflow_history.parquet`)

Contains realized inflow values for pre-study stages (negative stage IDs). Used to initialize AR model lags. See [Input Scenarios](input-scenarios.md) for the stochastic model definition.

| Column       | Type | Description                                        |
| ------------ | ---- | -------------------------------------------------- |
| `hydro_id`   | i32  | Hydro plant ID                                     |
| `stage_id`   | i32  | Pre-study stage ID (negative, e.g., -6, -5, …, -1) |
| `inflow_m3s` | f64  | Realized inflow value                              |

| Validation Rule    | Description                                                                        |
| ------------------ | ---------------------------------------------------------------------------------- |
| Pre-study coverage | Must cover at least the maximum AR order used                                      |
| Hydro entries      | Each hydro active from stage 0 must have entries for all required pre-study stages |
| Late-entry hydros  | Hydros entering later (entry_stage_id > 0) do not need pre-study history           |

## 2. Time-Varying Entity Bounds (`constraints/`)

Time-varying bounds allow entities to have different operational limits per stage. If an entity is not present for a stage, base bounds from the entity registry file are used. Partial overrides are supported (only specify stages that differ from base).

### Thermal Bounds (`constraints/thermal_bounds.parquet`) — Optional

| Column              | Type | Description                          |
| ------------------- | ---- | ------------------------------------ |
| `thermal_id`        | i32  | Thermal unit identifier              |
| `stage_id`          | i32  | Stage index                          |
| `min_generation_mw` | f64  | Minimum generation (null = use base) |
| `max_generation_mw` | f64  | Maximum generation (null = use base) |

### Hydro Bounds (`constraints/hydro_bounds.parquet`) — Optional

Useful for maintenance outages, seasonal restrictions, environmental constraints, and dead-volume filling periods.

| Column                 | Type | Description                                          |
| ---------------------- | ---- | ---------------------------------------------------- |
| `hydro_id`             | i32  | Hydro plant identifier                               |
| `stage_id`             | i32  | Stage index                                          |
| `min_turbined_m3s`     | f64  | Minimum turbined flow (null = use base)              |
| `max_turbined_m3s`     | f64  | Maximum turbined flow (null = use base)              |
| `min_storage_hm3`      | f64  | Minimum storage (null = use base)                    |
| `max_storage_hm3`      | f64  | Maximum storage (null = use base)                    |
| `min_outflow_m3s`      | f64  | Minimum outflow (null = use base)                    |
| `max_outflow_m3s`      | f64  | Maximum outflow (null = use base)                    |
| `filling_inflow_m3s`   | f64  | Water retained for filling                           |
| `water_withdrawal_m3s` | f64  | Water withdrawal (positive = remove, negative = add) |
| `evaporation_coef_mm`  | f64  | Monthly evaporation coefficient (mm/month)           |

**Filling inflow**: Water retained for reservoir filling (removed from cascade). If `inflow - filling_inflow < min_outflow`, a slack variable with `outflow_violation_penalty` is used.

**Water withdrawal (retirada de água)**: Water removed for consumption, irrigation, or industrial use. Positive values represent water leaving the system; negative values represent external additions (transpositions). A slack variable with `water_withdrawal_violation_cost` is used when inflow cannot meet the withdrawal target.

**Evaporation coefficient**: Monthly rate (mm/month) applied to reservoir surface area. Actual evaporated flow computed from the volume-area relationship (see [Input Hydro Extensions §1](input-hydro-extensions.md)).

### Line Bounds (`constraints/line_bounds.parquet`) — Optional

| Column       | Type | Description                             |
| ------------ | ---- | --------------------------------------- |
| `line_id`    | i32  | Transmission line identifier            |
| `stage_id`   | i32  | Stage index                             |
| `direct_mw`  | f64  | Direct flow capacity (null = use base)  |
| `reverse_mw` | f64  | Reverse flow capacity (null = use base) |

## 3. Generic Constraints (`constraints/generic_constraints.json`)

> **Format Rationale — generic_constraints.json**
>
> **Complex nested object** — Constraint definitions with variable references, coefficients, and conditional logic. JSON handles polymorphic structures well.

> **⚠️ Order Invariance**: The order of constraints does NOT affect results. After loading, constraints are sorted by `id`. See [Design Principles §3](../00-overview/design-principles.md).

Users can express custom linear constraints combining multiple LP variables:

| Constraint Type            | Example                                           |
| -------------------------- | ------------------------------------------------- |
| Regional generation limits | Minimum/maximum total hydro generation per region |
| Energy contracts           | Sum of generation from specific plants            |
| Irrigation agreements      | Sum of outflows                                   |
| Environmental corridors    | Combined outflow requirements                     |
| Fuel availability          | Sum of thermal generation                         |

### CEPEL Constraint Types Mapping

CEPEL models (NEWAVE, DECOMP) define specialized constraint types. In POWE.RS, all are expressed as generic constraints:

| CEPEL Type | Name                               | POWE.RS Expression                                           |
| ---------- | ---------------------------------- | ------------------------------------------------------------ |
| **RHQ**    | Restrição Hidráulica de Quantidade | `hydro_outflow(id) <= bound`                                 |
| **RE**     | Restrição Elétrica                 | `Σ hydro_generation(id) + Σ thermal_generation(id) >= bound` |
| **RHE**    | Restrição de Energia Hidráulica    | `Σ hydro_generation(id) >= bound`                            |
| **RHV**    | Restrição de Volume Hidráulico     | `hydro_storage(id) <= bound`                                 |
| **GHMIN**  | Geração Hidráulica Mínima          | `Σ hydro_generation(ids) >= min_gh`                          |
| **GTMIN**  | Geração Térmica Mínima             | `Σ thermal_generation(ids) >= min_gt`                        |
| **DEFMAX** | Déficit Máximo                     | `bus_deficit(bus_id) <= max_deficit`                         |

### Example

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/generic_constraints.schema.json",
  "constraints": [
    {
      "id": 0,
      "name": "min_southeast_hydro",
      "description": "Minimum hydro generation in Southeast region",
      "expression": "hydro_generation(0) + hydro_generation(1) + hydro_generation(2)",
      "sense": ">=",
      "slack": { "enabled": true, "penalty": 5000.0 }
    },
    {
      "id": 1,
      "name": "itaipu_contract",
      "description": "Itaipu energy contract - must deliver at least base amount",
      "expression": "hydro_generation(50)",
      "sense": ">=",
      "slack": { "enabled": true, "penalty": 8000.0 }
    },
    {
      "id": 2,
      "name": "environmental_flow",
      "description": "Combined outflow requirement for river stretch",
      "expression": "hydro_outflow(10) + hydro_outflow(11)",
      "sense": ">=",
      "slack": { "enabled": true, "penalty": 3000.0 }
    },
    {
      "id": 3,
      "name": "gas_availability",
      "description": "Total gas thermal generation limited by pipeline",
      "expression": "thermal_generation(5) + thermal_generation(6) + thermal_generation(7)",
      "sense": "<=",
      "slack": { "enabled": false }
    }
  ]
}
```

### Constraint Fields

| Field           | Type   | Required   | Description                                    |
| --------------- | ------ | ---------- | ---------------------------------------------- |
| `id`            | i32    | Yes        | Unique constraint identifier                   |
| `name`          | string | Yes        | Short name for reports                         |
| `description`   | string | No         | Human-readable description                     |
| `expression`    | string | Yes        | Linear expression (parsed using grammar below) |
| `sense`         | string | Yes        | `">="`, `"<="`, or `"=="`                      |
| `slack.enabled` | bool   | Yes        | Whether to add slack variable for feasibility  |
| `slack.penalty` | f64    | If enabled | Penalty per unit of violation                  |

### Variable Reference Syntax

Variables use function-like syntax: `variable_type(entity_id)`. For block-specific variables, use `variable_type(entity_id, block_id)` or omit `block_id` to sum over all blocks.

| Variable Name        | Syntax                             | Units |
| -------------------- | ---------------------------------- | ----- |
| `hydro_storage`      | `hydro_storage(id)`                | hm³   |
| `hydro_turbined`     | `hydro_turbined(id [, block])`     | m³/s  |
| `hydro_spillage`     | `hydro_spillage(id [, block])`     | m³/s  |
| `hydro_outflow`      | `hydro_outflow(id [, block])`      | m³/s  |
| `hydro_generation`   | `hydro_generation(id [, block])`   | MW    |
| `hydro_evaporation`  | `hydro_evaporation(id)`            | m³/s  |
| `hydro_withdrawal`   | `hydro_withdrawal(id)`             | m³/s  |
| `thermal_generation` | `thermal_generation(id [, block])` | MW    |
| `line_direct`        | `line_direct(id [, block])`        | MW    |
| `line_reverse`       | `line_reverse(id [, block])`       | MW    |
| `bus_deficit`        | `bus_deficit(id [, block])`        | MW    |
| `bus_excess`         | `bus_excess(id [, block])`         | MW    |
| `pumping_flow`       | `pumping_flow(id [, block])`       | m³/s  |
| `pumping_power`      | `pumping_power(id [, block])`      | MW    |
| `contract_import`    | `contract_import(id [, block])`    | MW    |
| `contract_export`    | `contract_export(id [, block])`    | MW    |

### Expression Grammar

```ebnf
expression    ::= term (('+' | '-') term)*
term          ::= coefficient? variable | number
coefficient   ::= number '*'
variable      ::= var_name '(' entity_id (',' block_id)? ')'
entity_id     ::= integer
block_id      ::= integer
number        ::= float | integer
```

**Examples**: `hydro_generation(10) + hydro_generation(11)` · `2.5 * thermal_generation(5) - hydro_generation(3)` · `hydro_turbined(5, 0) + hydro_turbined(5, 1)`

### Constraint Bounds (`constraints/constraint_bounds.parquet`)

Bounds can vary by stage (and optionally by block):

| Column          | Type | Description                                |
| --------------- | ---- | ------------------------------------------ |
| `constraint_id` | i32  | References constraint definition           |
| `stage_id`      | i32  | Stage index                                |
| `block_id`      | i32  | Block index (null = applies to all blocks) |
| `bound`         | f64  | RHS value for the constraint               |

### LP Integration

For constraint `c` with sense `>=`: `Σ (coef_i × var_i) + slack_below[c] >= bound[c]`

For constraint `c` with sense `<=`: `Σ (coef_i × var_i) - slack_above[c] <= bound[c]`

For constraint `c` with sense `==`: `Σ (coef_i × var_i) + slack_below[c] - slack_above[c] == bound[c]`

Slack variables are only created if `slack.enabled = true`.

### Validation Rules

1. All entity IDs referenced in expressions must exist in the system
2. Block IDs (if specified) must be valid for the stage
3. Expressions must parse correctly according to the grammar
4. `constraint_bounds.parquet` must have entries for all study stages for each constraint
5. Constraint IDs must be unique and contiguous (0, 1, 2, …)
6. If `slack.enabled = true`, `slack.penalty` must be provided and positive

## 4. Policy Directory (`policy/`)

> **Format Rationale — policy/**
>
> **Policy / binary data** — Pre-computed cuts for warm-start. FlatBuffers for zero-copy deserialization during hot-path loading. See [Binary Formats](binary-formats.md) for detailed FlatBuffers schemas.

POWE.RS uses a single `policy/` directory that serves both as input (loading existing cuts/states) and output (writing updated policy data).

### Directory Structure

```
policy/
├── metadata.json               # Algorithm state, RNG, bounds
├── state_dictionary.json       # State variable mapping
├── cuts/                       # Outer approximation (standard SDDP cuts)
│   ├── stage_000.bin
│   └── ...
├── states/                     # Visited states for cut selection
│   └── ...
├── vertices/                   # Inner approximation (upper bounds, if enabled)
│   └── ...
└── basis/                      # Solver basis for exact reproducibility (optional)
    └── ...
```

### Policy Modes

| Mode         | Reads From                                  | Behavior                                                       |
| ------------ | ------------------------------------------- | -------------------------------------------------------------- |
| `fresh`      | Nothing                                     | Start from scratch. Existing files ignored (not deleted).      |
| `warm_start` | `cuts/`, `states/`, `state_dictionary.json` | Load existing cuts/states, reset iteration count and RNG seed. |
| `resume`     | Everything including `metadata.json`        | Load full algorithm state, continue exactly where interrupted. |

### State Dictionary (`state_dictionary.json`)

Defines the mapping between coefficient indices and actual state variables. Follows canonical ordering (see [Design Principles §3](../00-overview/design-principles.md)): state variables ordered by entity type, then by entity ID.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/state_dictionary.schema.json",
  "version": "2.0.0",
  "state_dimension": 320,
  "state_variables": [
    {
      "index": 0,
      "type": "storage",
      "entity_type": "hydro",
      "entity_id": 0,
      "name": "FURNAS"
    },
    {
      "index": 1,
      "type": "storage",
      "entity_type": "hydro",
      "entity_id": 1,
      "name": "MARIMBONDO"
    },
    {
      "index": 160,
      "type": "inflow_lag_1",
      "entity_type": "hydro",
      "entity_id": 0,
      "name": "FURNAS"
    }
  ],
  "checksum": "sha256:abc123..."
}
```

| Field         | Type   | Description                                                       |
| ------------- | ------ | ----------------------------------------------------------------- |
| `index`       | i32    | Column index in cuts/states (`coefficient_N`, `component_N`)      |
| `type`        | string | Variable type: `"storage"`, `"inflow_lag_1"`, `"inflow_lag_2"`, … |
| `entity_type` | string | `"hydro"`, `"thermal"`, `"bus"`, etc.                             |
| `entity_id`   | i32    | Entity ID (matches entity registries)                             |
| `name`        | string | Human-readable entity name (for debugging)                        |

### Validation

**Resume** (`policy.mode = "resume"`):

1. Version compatibility with current POWE.RS version
2. Config hash match (or explicit override flag)
3. System hash match (entities must be identical)
4. State dictionary checksum match
5. All partitioned files exist for all stages

**Warm-start** (`policy.mode = "warm_start"`):

1. `state_dictionary.json` exists
2. State dimension matches current system
3. All entity IDs in dictionary exist in current system
4. At least one stage has cuts

### Reproducibility Guarantees

| Scenario                           | Reproducibility          | Notes                                                   |
| ---------------------------------- | ------------------------ | ------------------------------------------------------- |
| Straight run (no prior policy)     | ✅ Bit-for-bit identical | Same seed → same results                                |
| Resume with basis                  | ✅ Bit-for-bit identical | Solver basis restored → same pivots                     |
| Resume without basis               | ⚠️ Equivalent optimum    | Same optimum, possibly different duals → different cuts |
| Warm-start                         | ⚠️ Different run         | Fresh RNG, may converge differently                     |
| Resume with modified config/system | ❌ Not supported         | Hash mismatch → error                                   |

For detailed binary schemas of cuts, states, vertices, and basis files, see [Binary Formats](binary-formats.md).

## Cross-References

- [Input System Entities](input-system-entities.md) — Base entity schemas referenced by bounds and constraints
- [Input Hydro Extensions](input-hydro-extensions.md) — Hydro extension files and contract bounds
- [Input Scenarios](input-scenarios.md) — Stage definitions and stochastic models
- [Input Directory Structure](input-directory-structure.md) — Overall case directory layout
- [Binary Formats](binary-formats.md) — Detailed FlatBuffers schemas for policy binary files
- [LP Formulation](../01-math/lp-formulation.md) — How constraints enter the LP
- [Cut Management](../01-math/cut-management.md) — Cut storage and selection algorithms
- [SDDP Algorithm](../01-math/sddp-algorithm.md) — Forward/backward pass and convergence
- [Penalty System](penalty-system.md) — Penalty hierarchy for slack variables
- [Design Principles §3](../00-overview/design-principles.md) — Order invariance and canonical ordering
