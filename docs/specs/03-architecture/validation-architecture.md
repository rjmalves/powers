---
status: approved
review_priority: 3-medium
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §6 (6.1-6.4)"
  - "DATA_MODEL_SPECIFICATION.md §8 (8.1-8.2)"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted and merged from ARCHITECTURE §6 and DATA_MODEL §8"
  - date: 2026-02-22
    description: "Review rewrite: fix priority to P3. Strip all Rust code (§4, §5.2). Merge layer stack and phase sequence into single 5-layer model. Fix stale terminology (block weights → block hours, inflow_models → inflow_seasonal_stats). Align business rules with approved specs (stopping-rules, input-scenarios). Move Markovian validation to deferred. Scope GNL validation as deferred-aware. Fix validation report example (stale filename). Add cross-references to input-loading-pipeline.md, penalty-system.md, input-system-entities.md."
  - date: 2026-02-22
    description: "Deep-dive: §2.5 expanded from 11 rules to ~50 rules organized by domain (hydro, FPHA, thermal, stages, penalties, PAR, external scenarios, generic constraints, risk/discounting, declaration order). §2.5b expanded with resume mode, linearized head, warm-start block compatibility. AR stationarity downgraded to warning (per-season stability check). Added ModelQuality and ResumeIncompatible error kinds. Cross-references expanded to 17 entries."
---

# Validation Architecture

## Purpose

This spec defines the POWE.RS multi-layer input validation pipeline: the five validation layers, the error collection strategy, the error type catalog, and the validation report format. Validation runs during the Validation phase of the execution lifecycle (see [CLI and Lifecycle](./cli-and-lifecycle.md) §5.2).

For the file loading sequence and per-file validation checks, see [Input Loading Pipeline](./input-loading-pipeline.md) §2. This spec defines the _architectural framework_ for validation; the loading pipeline spec defines _what is validated per file_.

## 1. Validation Pipeline Overview

Validation runs on **rank 0 only** during the Validation phase. It collects all errors before failing, so the user sees every problem in a single report rather than fixing issues one at a time.

The pipeline comprises five sequential layers. Each layer depends on the previous one — e.g., referential integrity checks require that schema validation has already confirmed field presence, and semantic checks require that all cross-references have been resolved.

```
Layer 1          Layer 2         Layer 3            Layer 4           Layer 5
Structural    →  Schema       →  Referential     →  Dimensional    →  Semantic
(files, format)  (fields, types)  (foreign keys)    (coverage)        (business rules)
```

**Canonicalization** (sorting all entity collections by ID) occurs during loading, before any validation layer executes. This ensures bit-for-bit reproducibility regardless of declaration order in input files (see [Design Principles](../00-overview/design-principles.md) §3).

## 2. Validation Layers

### 2.1 Layer 1 — Structural Validation

Verifies that required files exist and are parseable:

- Required files exist on disk and are readable
- JSON files parse as valid JSON (UTF-8 encoding)
- Parquet files have valid Parquet headers and are readable
- Optional files that are absent are recorded and their defaults noted

Missing required files produce immediate errors. Missing optional files are not errors — the loader uses defaults (see [Input Loading Pipeline](./input-loading-pipeline.md) §4).

### 2.2 Layer 2 — Schema Validation

Validates that every file conforms to its expected structure:

- **JSON files:** Required vs. optional fields present, data types correct (string, number, boolean, array, object), value ranges valid (e.g., all IDs non-negative, probabilities in [0,1]), enum values valid (e.g., `block_mode` is `"parallel"` or `"chronological"`)
- **Parquet files:** Expected columns present with correct Arrow types, per-column value constraints (e.g., non-negative costs, valid entity IDs)

Schema validation is exhaustive for each file — every field and column is checked. See [Input Loading Pipeline](./input-loading-pipeline.md) §2 for the per-file validation notes.

### 2.3 Layer 3 — Referential Integrity

Validates cross-entity references (foreign keys):

- Bus IDs referenced by lines, hydros, thermals, NCS, pumping stations, and contracts exist in `buses.json`
- Downstream hydro IDs in cascade topology exist in `hydros.json`
- Stage IDs in time-series Parquet files exist in `stages.json`
- Season IDs in stage definitions match `season_definitions`
- Entity IDs in constraint bounds and penalty override files exist in their respective registries
- Policy graph transition source/target stage IDs exist
- Generic constraint entity references resolve to loaded entities

### 2.4 Layer 4 — Dimensional Consistency

Cross-file dimensional checks ensuring data completeness:

- Inflow model parameters (seasonal stats + AR coefficients) cover all hydros with PAR-based scenario generation
- Inflow history covers all hydros when `inflow_history.parquet` is provided
- Load seasonal stats cover all buses with load when `load_seasonal_stats.parquet` is provided
- Correlation matrix dimensions match the number of hydros in each correlation group
- `initial_conditions.json` storage array length matches hydro registry count
- FPHA hyperplanes exist for all hydros with FPHA source `"precomputed"` (hydros with source `"computed"` do not require this file — planes are generated during Initialization)

### 2.5 Layer 5 — Semantic Validation (Business Rules)

Domain-specific rules that require understanding of the model semantics. Rules are organized by domain. Unless otherwise noted, all rules produce **errors** that prevent execution.

#### Hydro System

| Rule                               | Description                                                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Acyclic cascade                    | Hydro cascade graph is a directed acyclic graph (DAG) — no cycles in downstream references                                                           |
| Storage bounds                     | `storage_min ≤ initial_storage ≤ storage_max` for each operating hydro                                                                               |
| Filling bounds                     | Filling storage within `[0, storage_min]` for each filling hydro                                                                                     |
| Filling/operating mutual exclusion | Each hydro appears in either `initial_storage` or `filling_storage`, never both                                                                      |
| Filling stage ordering             | `start_stage_id < entry_stage_id` for filling hydros                                                                                                 |
| Generation bounds                  | `generation_min ≤ generation_max` for each hydro                                                                                                     |
| Geometry monotonicity              | Volume-area-level curves: volumes monotonically increasing; heights monotonically increasing with volume; areas monotonically increasing with height |
| Geometry coverage                  | Geometry min volume entry ≤ `min_storage_hm3`; max volume entry ≥ `max_storage_hm3`                                                                  |
| Tailrace monotonicity              | Tailrace piecewise points sorted monotonically increasing by outflow                                                                                 |
| **Filling inflow sufficiency**     | **Warning.** Estimated natural inflow may be insufficient for the filling hydro to reach `min_storage` by `entry_stage_id`                           |

#### FPHA Production Model

| Rule              | Description                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Minimum planes    | At least 3 hyperplanes per hydro per stage                                                                                                      |
| Coefficient signs | $\gamma_v > 0$ (storage), $\gamma_q > 0$ (turbined flow), $\gamma_s \leq 0$ (spillage)                                                          |
| Penalty ordering  | `fpha_turbined_cost > spillage_cost` for each hydro using FPHA — ensures spillage is penalized less than exceeding the FPHA turbined flow bound |

#### Thermal System

| Rule              | Description                                                                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Generation bounds | `generation_min ≤ generation_max` for each thermal                                                                                                              |
| GNL rejection     | Thermals with `gnl_config` are rejected with a diagnostic error — GNL is not yet implemented. See [Deferred Features](../06-deferred/deferred-features.md) §C.1 |

#### Stages, Blocks, and Policy Graph

| Rule                     | Description                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| Stage IDs                | Unique and non-negative                                                                                                                 |
| Block IDs                | Contiguous starting at 0 within each stage                                                                                              |
| Block hours sum          | Sum of block hours within each stage equals the total stage duration (derived from `end_date - start_date`)                             |
| Season alignment         | Stage `[start_date, end_date)` falls within the corresponding season calendar period                                                    |
| Same-season duration     | Stages assigned to the same season have identical total duration                                                                        |
| Transition probabilities | Outgoing transition probabilities sum to 1.0 per source stage in the policy graph                                                       |
| Iteration limit          | At least one `iteration_limit` stopping rule is present (mandatory safety bound). See [Stopping Rules](../01-math/stopping-rules.md) §2 |

#### Penalty System

| Rule                 | Description                                                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Penalty values       | All penalty values in `penalties.json` are strictly positive                                                                                                       |
| Priority ordering    | Penalty hierarchy maintained: recourse slacks > constraint violation penalties > regularization costs. See [Penalty System](../02-data-model/penalty-system.md) §3 |
| Deficit last segment | Last deficit cost segment must have `depth_mw: null` (uncapped final segment)                                                                                      |
| Deficit monotonicity | Piecewise-linear deficit cost segments have monotonically increasing cost per MW                                                                                   |

#### PAR Inflow Model

| Rule                          | Description                                                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Positive residual variance    | $\sigma_m^2 > 0$ for all seasons                                                                                                                                                                                                  |
| Correlation PSD               | Correlation matrices are positive semi-definite (invertible)                                                                                                                                                                      |
| AR lag contiguity             | AR lags contiguous starting at 1 for each (hydro, stage)                                                                                                                                                                          |
| AR order consistency          | Number of AR coefficient rows per (hydro, stage) matches `ar_order` in seasonal stats                                                                                                                                             |
| AR coefficients require stats | AR coefficients without corresponding seasonal stats entry is an error                                                                                                                                                            |
| Default correlation profile   | A `"default"` correlation profile must exist                                                                                                                                                                                      |
| Profile name resolution       | All profile names referenced in the correlation schedule must exist in the profile definitions                                                                                                                                    |
| Correlation group entities    | Entity IDs in correlation groups must exist in the hydro registry                                                                                                                                                                 |
| Season definitions required   | `season_definitions` required when `inflow_history.parquet` is provided                                                                                                                                                           |
| **No systematic bias**        | **Warning.** Residuals $\varepsilon_t$ should have mean near zero — large bias suggests mis-specified AR model                                                                                                                    |
| **PAR seasonal stability**    | **Warning.** For each season $m$, the characteristic polynomial $1 - \sum_\ell \psi_{m,\ell} z^\ell$ should have all roots outside the unit circle. This is a per-season stability check, not a global PAR stationarity guarantee |

#### External Scenarios

| Rule           | Description                                                       |
| -------------- | ----------------------------------------------------------------- |
| Scenario count | Distinct `scenario_id` count must equal `num_scenarios` in config |

#### Generic Constraints

| Rule               | Description                                                                        |
| ------------------ | ---------------------------------------------------------------------------------- |
| Constraint IDs     | Unique and contiguous                                                              |
| Entity references  | All entity IDs in constraint expressions must exist in their respective registries |
| Block references   | Block IDs in constraint definitions must be valid for the referenced stage         |
| Expression parsing | Constraint expressions must parse successfully                                     |
| Slack penalty sign | Slack penalty must be strictly positive if slack is enabled                        |
| Bound references   | Constraint bounds must reference existing constraint IDs                           |

#### Risk and Discounting

| Rule                    | Description                                                                                                                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Discount rate in cycles | Cyclic policy graphs require `annual_discount_rate > 0`; cumulative discount factor around any cycle must be < 1.0 for convergence. See [Discount Rate](../01-math/discount-rate.md) §4 |
| CVaR confidence level   | $\alpha \in (0, 1]$. See [Risk Measures](../01-math/risk-measures.md) §2                                                                                                                |
| Risk aversion weight    | $\lambda \in [0, 1]$. See [Risk Measures](../01-math/risk-measures.md) §3                                                                                                               |

#### Declaration Order Invariance

| Rule              | Description                                                                                                                                                                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ID uniqueness** | Entity IDs unique within each registry (hydros, thermals, buses, lines, etc.) — combined with canonicalization during loading, this guarantees bit-for-bit identical results regardless of declaration order. See [Design Principles](../00-overview/design-principles.md) §3 |

#### General Warnings

| Rule              | Description                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| **Unused entity** | **Warning.** Entity defined but appears inactive (e.g., thermal with `max_generation = 0` for all stages) |

#### 2.5b Conditional Validation (mode-dependent)

Certain rules apply only when specific configuration modes or optional files are present:

##### Cyclic Policy Graph (`type = "cyclic"`)

- At least one back-edge transition exists
- Cycle transitions have `annual_discount_rate > 0`
- Cumulative discount factor around any cycle < 1.0 for convergence

##### Warm-Start (`policy.mode = "warm_start"`)

- State dictionary exists and dimension matches current system (entity count, state variable mapping)
- Cut stage IDs exist in current policy graph
- Entity IDs in state dictionary exist in current registries
- At least one stage has stored cuts
- Block mode and block count/durations per stage match between warm-start policy and current configuration. See [Block Formulations](../01-math/block-formulations.md) §4

##### Resume (`policy.mode = "resume"`)

- Version compatibility check
- Configuration hash match
- System hash match
- State dictionary checksum match
- All partitioned output files from prior run exist and are readable

##### External Scenarios (`external_scenarios.parquet` present)

- Entity/stage/scenario coverage is complete for all required entities
- Distinct `scenario_id` count equals `num_scenarios`

> **Note**: These validation rules apply regardless of whether external scenarios are used for simulation or training. External scenarios are NOT restricted to simulation — see [Scenario Generation §3](./scenario-generation.md). When used in training, additional validation applies: a PAR model must be fittable from the external data for backward pass opening tree generation.

##### Linearized Head Model

- Hydros using `linearized_head` production model are accepted only for simulation runs — rejected during training. See [Hydro Production Models](../01-math/hydro-production-models.md) §5

> **Note:** Markovian horizon validation (Markov states, transition probabilities per Markov state) is deferred. See [Deferred Features](../06-deferred/deferred-features.md) §C.5.

## 3. Error Collection Strategy

Validation collects **all errors** before failing, rather than stopping on the first error. This allows users to fix every problem in a single iteration.

The validation context tracks:

- **Errors** — Validation failures that prevent execution. Each error records the source file, entity (if applicable), error kind, and a human-readable message.
- **Warnings** — Non-fatal observations (e.g., an entity with `max_generation = 0` for all stages, suggesting it may be unused).

Within each layer, all checks run to completion. If a layer produces errors, subsequent layers may still execute for independent checks (e.g., referential integrity errors in one entity registry do not block schema validation of unrelated files). However, certain dependencies are hard — if schema validation fails for a file, no referential integrity checks are attempted for that file's references.

After all layers complete, the validation context is evaluated:

- If **any errors** exist, the program emits a validation report and exits with code 3 (see [CLI and Lifecycle](./cli-and-lifecycle.md) §4).
- If **only warnings** exist, the program emits the report to the log and continues execution.

## 4. Error Type Catalog

| Error Kind              | Severity | Description                                | Example                                                                                                         |
| ----------------------- | -------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `FileNotFound`          | Error    | Required file missing                      | `hydros.json` not found                                                                                         |
| `ParseError`            | Error    | Invalid JSON or Parquet format             | Malformed JSON syntax                                                                                           |
| `SchemaViolation`       | Error    | Schema mismatch                            | Missing required field `bus_id`                                                                                 |
| `InvalidReference`      | Error    | Foreign key references non-existent entity | `bus_id: 999` not in `buses.json`                                                                               |
| `DuplicateId`           | Error    | ID uniqueness violation                    | Two hydros with `id: 42`                                                                                        |
| `InvalidValue`          | Error    | Value out of valid range                   | `storage_max < storage_min`                                                                                     |
| `CycleDetected`         | Error    | Invalid graph structure                    | Hydro cascade forms a cycle                                                                                     |
| `DimensionMismatch`     | Error    | Cross-file coverage gap                    | Missing inflow params for hydro                                                                                 |
| `BusinessRuleViolation` | Error    | Semantic rule violated                     | `fpha_turbined_cost ≤ spillage_cost` for a hydro                                                                |
| `WarmStartIncompatible` | Error    | Policy incompatible with current system    | State dimension mismatch                                                                                        |
| `ResumeIncompatible`    | Error    | Resume state incompatible with current run | Config hash mismatch or missing partitioned files                                                               |
| `NotImplemented`        | Error    | Feature used but not yet implemented       | Thermal with `gnl_config` present                                                                               |
| `UnusedEntity`          | Warning  | Entity defined but appears inactive        | Thermal with `max_generation = 0` everywhere                                                                    |
| `ModelQuality`          | Warning  | Statistical quality concern in input model | PAR seasonal polynomial has root inside unit circle; residual bias detected; filling inflow may be insufficient |

Each error carries: **file path**, **entity identifier** (if applicable), **error kind** (from table above), and **message** (human-readable description).

## 5. Validation Report Format

When validation completes (pass or fail), POWE.RS emits a structured JSON report:

```json
{
  "valid": false,
  "timestamp": "2026-01-31T10:30:00Z",
  "case_directory": "/path/to/case",
  "errors": [
    {
      "file": "system/hydros.json",
      "entity": "hydro_042",
      "kind": "InvalidReference",
      "message": "bus_id 'BUS_99' not found in buses.json"
    },
    {
      "file": "scenarios/inflow_seasonal_stats.parquet",
      "entity": null,
      "kind": "DimensionMismatch",
      "message": "No seasonal stats for hydro 'hydro_015' at stage 48"
    }
  ],
  "warnings": [
    {
      "file": "system/thermals.json",
      "entity": "thermal_old",
      "kind": "UnusedEntity",
      "message": "Thermal 'thermal_old' has max_generation=0 for all stages"
    }
  ],
  "summary": {
    "files_checked": 24,
    "entities_validated": 456,
    "error_count": 2,
    "warning_count": 1
  }
}
```

The report is written to `{case_directory}/validation_report.json` and also emitted to the program log. In `--validate-only` mode, the report is the primary output.

## Cross-References

- [CLI and Lifecycle](./cli-and-lifecycle.md) — Validation phase within the execution lifecycle; `--validate-only` mode; exit codes
- [Input Loading Pipeline](./input-loading-pipeline.md) — File loading sequence, per-file validation checks (§2), conditional loading rules (§4)
- [Design Principles](../00-overview/design-principles.md) — Declaration order invariance (§3) enforced by canonicalization before validation
- [Input Directory Structure](../02-data-model/input-directory-structure.md) — File inventory and `config.json` schema
- [Input System Entities](../02-data-model/input-system-entities.md) — Entity registries and default bounds
- [Input Scenarios](../02-data-model/input-scenarios.md) — Policy graph, stage definitions, block hours, season definitions
- [Input Constraints](../02-data-model/input-constraints.md) — Generic constraints, penalty overrides, exchange factors, initial conditions, warm-start/resume
- [Input Hydro Extensions](../02-data-model/input-hydro-extensions.md) — Geometry validation, FPHA source modes (precomputed vs computed)
- [Penalty System](../02-data-model/penalty-system.md) — Penalty values, priority ordering, and three-tier cascade
- [Stopping Rules](../01-math/stopping-rules.md) — Mandatory `iteration_limit` rule
- [Discount Rate](../01-math/discount-rate.md) — Cycle discount factor convergence requirement
- [Risk Measures](../01-math/risk-measures.md) — CVaR confidence level and risk aversion weight bounds
- [Block Formulations](../01-math/block-formulations.md) — Policy compatibility validation for warm-start/resume
- [Hydro Production Models](../01-math/hydro-production-models.md) — FPHA coefficient signs, linearized head simulation-only restriction
- [PAR Inflow Model](../01-math/par-inflow-model.md) — AR stationarity, residual variance, correlation matrix validation
- [Scenario Generation](./scenario-generation.md) — PAR model validation rules (AR stationarity, sufficient history)
- [Deferred Features](../06-deferred/deferred-features.md) — GNL (§C.1) and Markovian (§C.5) validation deferred
