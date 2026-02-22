---
status: approved
review_priority: 3-medium
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §4 (4.1-4.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §5 (5.1-5.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §7 (7.1-7.4)"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: "Approved after two review passes. Schema validation general rule added, non-exhaustive note added, FPHA lifecycle gap resolved (precomputed vs computed paths)."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §4-5, §7"
  - date: 2026-02-22
    description: "Review rewrite: fix priority to P3, strip all Rust code (§3, §6, §8, §10), replace with behavioral descriptions. Update §2 file loading table to match approved input-directory-structure.md (split inflow files, added NCS/pumping/contracts/penalty overrides/exchange factors). Fix §5 conditional loading to match approved config structure. Rewrite §7 broadcast strategy as behavioral categories (removed size estimates). Fix §9 parallel policy loading (removed timing guesses). Remove §10 memory layout (duplicated internal-structures.md). Fix diagram placeholders. Fix cross-references."
  - date: 2026-02-22
    description: "Second review pass: add schema validation general rule and non-exhaustive note to §2. Fix FPHA lifecycle gap — §2.3 now conditional on source precomputed/computed, §4 split into two FPHA rows, §2.6 cross-ref check scoped to precomputed only. Added FPHA preprocessing paragraph in §8."
---

# Input Loading Pipeline

## Purpose

This spec defines the POWE.RS input loading architecture: the rank-0 centric loading pattern, file loading sequence with dependency ordering, dependency resolution, conditional loading rules, sparse time-series expansion, data broadcasting strategy, parallel policy loading for warm-start, and the transition to the in-memory data model.

For the file inventory and directory layout, see [Input Directory Structure](../02-data-model/input-directory-structure.md).
For the in-memory data model after loading completes, see [Internal Structures](../02-data-model/internal-structures.md).

## 1. Loading Architecture

Input loading follows a **rank-0 centric** pattern: rank 0 loads and validates all input data, then broadcasts to worker ranks. This design:

- Minimizes filesystem contention on parallel filesystems
- Centralizes validation logic on a single rank
- Reduces complexity of error handling across ranks
- Ensures all ranks receive identical, validated data

The one exception is **policy loading for warm-start** (§7), where all ranks load in parallel to avoid bottlenecking on large policy files.

> **Placeholder** — The input loading pipeline diagram (`../../diagrams/exports/svg/data/input-loading-pipeline.svg`) will be revised after the text review is complete.

## 2. File Loading Sequence

Files are loaded in dependency order so that each file can be validated against previously loaded data. Loading fails fast on the first error.

**Schema validation first:** Every JSON file undergoes full schema validation before any other checks. This includes required vs. optional fields, data types, value ranges, and structural constraints (e.g., array lengths, enum values). Similarly, every Parquet file is validated for expected columns, column types, and per-column value constraints. The "Validation" column in the tables below lists only the **additional** cross-reference and semantic checks performed after schema validation passes.

> **Note:** The validations listed in this section are illustrative, not exhaustive. Additional validations may be introduced during implementation or as a result of reviewing other specs. See [Validation Architecture](./validation-architecture.md) for the complete multi-layer validation design.

### 2.1 Root-Level Files

| Order | File                      | Dependencies | Validation                                              |
| ----- | ------------------------- | ------------ | ------------------------------------------------------- |
| 1     | `config.json`             | None         | JSON schema, execution mode, section structure          |
| 2     | `stages.json`             | config       | Stage count, season mapping, policy graph, block counts |
| 3     | `penalties.json`          | None         | All penalty values > 0, required categories present     |
| 4     | `initial_conditions.json` | None         | Array lengths deferred until entity registries loaded   |

### 2.2 System Entity Registries

| Order | File                                   | Dependencies  | Validation                                       |
| ----- | -------------------------------------- | ------------- | ------------------------------------------------ |
| 5     | `system/buses.json`                    | None          | Bus IDs unique                                   |
| 6     | `system/lines.json`                    | buses         | Source/target bus references valid               |
| 7     | `system/hydros.json`                   | buses         | Bus references valid, cascade references acyclic |
| 8     | `system/thermals.json`                 | buses         | Bus references valid                             |
| 9     | `system/non_controllable_sources.json` | buses         | Bus references valid (optional file)             |
| 10    | `system/pumping_stations.json`         | hydros, buses | Hydro and bus references valid (optional file)   |
| 11    | `system/energy_contracts.json`         | buses         | Bus references valid (optional file)             |

### 2.3 System Extension Data

| Order | File                                  | Dependencies   | Validation                                                        |
| ----- | ------------------------------------- | -------------- | ----------------------------------------------------------------- |
| 12    | `system/hydro_geometry.parquet`       | hydros         | Hydro ID coverage, monotonic volume-area-level curves             |
| 13    | `system/hydro_production_models.json` | hydros, stages | Hydro/stage references valid (optional file)                      |
| 14    | `system/fpha_hyperplanes.parquet`     | hydros         | Required only when FPHA source is `"precomputed"` (optional file) |

### 2.4 Scenario Data

| Order | File                                       | Dependencies     | Validation                                                    |
| ----- | ------------------------------------------ | ---------------- | ------------------------------------------------------------- |
| 15    | `scenarios/inflow_seasonal_stats.parquet`  | hydros, stages   | Hydro/stage coverage (optional file)                          |
| 16    | `scenarios/inflow_ar_coefficients.parquet` | hydros, stages   | Hydro/stage/lag coverage, consistent AR order (optional file) |
| 17    | `scenarios/inflow_history.parquet`         | hydros           | Hydro ID coverage (optional file)                             |
| 18    | `scenarios/load_seasonal_stats.parquet`    | buses, stages    | Bus/stage coverage (optional file)                            |
| 19    | `scenarios/load_factors.json`              | stages           | Block count consistency (optional file)                       |
| 20    | `scenarios/correlation.json`               | hydros           | Group membership covers all hydros (optional file)            |
| 21    | `scenarios/external_scenarios.parquet`     | entities, stages | Entity/stage/scenario coverage (optional file)                |

### 2.5 Constraints and Overrides

| Order | File                                            | Dependencies                | Validation                                        |
| ----- | ----------------------------------------------- | --------------------------- | ------------------------------------------------- |
| 22    | `constraints/thermal_bounds.parquet`            | thermals, stages            | Entity/stage references valid (optional file)     |
| 23    | `constraints/hydro_bounds.parquet`              | hydros, stages              | Entity/stage references valid (optional file)     |
| 24    | `constraints/line_bounds.parquet`               | lines, stages               | Entity/stage references valid (optional file)     |
| 25    | `constraints/pumping_bounds.parquet`            | pumping, stages             | Entity/stage references valid (optional file)     |
| 26    | `constraints/contract_bounds.parquet`           | contracts, stages           | Entity/stage references valid (optional file)     |
| 27    | `constraints/exchange_factors.json`             | lines, stages               | Line/stage references valid (optional file)       |
| 28    | `constraints/generic_constraints.json`          | entities                    | Entity references valid (optional file)           |
| 29    | `constraints/generic_constraint_bounds.parquet` | generic constraints, stages | Constraint/stage references valid (optional file) |
| 30    | `constraints/penalty_overrides_bus.parquet`     | buses, stages               | Entity/stage references valid (optional file)     |
| 31    | `constraints/penalty_overrides_line.parquet`    | lines, stages               | Entity/stage references valid (optional file)     |
| 32    | `constraints/penalty_overrides_hydro.parquet`   | hydros, stages              | Entity/stage references valid (optional file)     |
| 33    | `constraints/penalty_overrides_ncs.parquet`     | NCS, stages                 | Entity/stage references valid (optional file)     |

### 2.6 Cross-Reference Validation

After all files are loaded, a final cross-reference validation pass checks:

- `initial_conditions.json` storage array lengths match hydro registry count
- Inflow model coverage is complete for all operating hydros
- FPHA hyperplanes exist for all hydros with FPHA source `"precomputed"` (hydros with source `"computed"` do not require this file — planes are generated during Initialization)
- Generic constraint entity references resolve to loaded entities
- Policy graph stage transitions are valid (no dangling references)

### 2.7 Policy Files (Warm-Start Only)

| Order | File       | Dependencies | Validation                                                |
| ----- | ---------- | ------------ | --------------------------------------------------------- |
| 34    | `policy/*` | All above    | State dictionary matches current system, cut format valid |

Policy loading uses a different pattern — see §7 Parallel Policy Loading.

## 3. Dependency Graph

Input files form a directed acyclic graph (DAG) of dependencies. The loading sequence in §2 is a valid topological ordering of this DAG. Files at the same dependency level may be loaded in any order relative to each other.

> **Placeholder** — The dependency graph diagram (`../../diagrams/exports/svg/data/json-schema-dependencies.svg`) will be revised after the text review is complete.

## 4. Conditional Loading

Some files are loaded only when certain conditions are met. Missing optional files are not errors — the loader uses defaults (typically empty collections or identity values).

| Condition                                                 | Effect                                                                                         |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `training.enabled = false` in `config.json`               | Skip scenario noise generation (but still load models if simulation needs them)                |
| `simulation.enabled = false` in `config.json`             | Skip simulation-specific scenario setup                                                        |
| `policy.mode = "warm_start"` in `config.json`             | Load `policy/*` files (§7)                                                                     |
| Policy graph has cycle (`stages.json`)                    | Validate cycle structure per [Infinite Horizon](../01-math/infinite-horizon.md)                |
| Hydros with FPHA production model, source `"precomputed"` | Require `fpha_hyperplanes.parquet`                                                             |
| Hydros with FPHA production model, source `"computed"`    | FPHA hyperplanes computed during Initialization phase from geometry and topology data (see §8) |
| `pumping_stations.json` present                           | Load pumping bounds, validate hydro/bus references                                             |
| `energy_contracts.json` present                           | Load contract bounds                                                                           |
| `non_controllable_sources.json` present                   | Load NCS penalty overrides                                                                     |
| `external_scenarios.parquet` present                      | Use external scenarios instead of PAR-generated ones                                           |

## 5. Sparse Time-Series Expansion

Time-series Parquet files (bounds, penalty overrides) use **sparse representation**: only rows with non-default values are stored. Stages and entities not present in the file receive default values.

**Expansion behavior:**

1. Load the sparse Parquet file (only rows with explicit values).
2. Build a dense (stage × entity) structure initialized with defaults.
3. Overlay the sparse values onto the dense structure.
4. Validate that all stage IDs and entity IDs in the sparse data are valid (reference loaded registries).

**Default values:** Each file type has its own default. For bounds files, the default is the static bound from the entity registry. For penalty overrides, the default is the global value from `penalties.json`. This is defined per file in the relevant data model spec — see [Input System Entities](../02-data-model/input-system-entities.md) and [Penalty System](../02-data-model/penalty-system.md).

## 6. Broadcast Strategy

After rank 0 loads and validates all data, it broadcasts to worker ranks. Data is serialized to contiguous byte buffers for MPI broadcast. The broadcast uses a two-step protocol: first the buffer size, then the buffer contents.

**Broadcast categories:**

| Category          | Data                                                       | Strategy                                         |
| ----------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| Small objects     | Config, stages, penalties, initial conditions              | Single `MPI_Bcast`                               |
| Entity registries | Buses, lines, hydros, thermals, NCS, pumping, contracts    | Single `MPI_Bcast`                               |
| Scenario models   | PAR parameters, correlation matrices, load models          | Single `MPI_Bcast`                               |
| Bounds/overrides  | All constraint bounds, penalty overrides, exchange factors | Single `MPI_Bcast` (sparse form, expand locally) |
| Policy cuts       | FCF cuts for warm-start                                    | Parallel load (§7)                               |

**Sparse broadcast optimization:** Bounds and penalty override files are broadcast in their sparse Parquet form. Each rank performs the sparse-to-dense expansion locally (§5). This reduces broadcast volume — only non-default values are transmitted.

## 7. Parallel Policy Loading (Warm-Start)

Policy files (cuts, states, vertices, basis) can be large. Loading them on rank 0 and broadcasting would create a bottleneck. Instead, all ranks load in parallel:

1. Rank 0 loads and broadcasts the `policy/metadata.json` and `policy/state_dictionary.json` (small files).
2. All ranks validate the state dictionary against the current system (entity count, state variable mapping).
3. Each rank loads a subset of policy stage files from `policy/cuts/` — stages are assigned round-robin by rank.
4. After local loading, ranks exchange cuts so that every rank has the complete policy. The exchange strategy (e.g., `MPI_Allgatherv` or shared memory window + inter-node broadcast) is an implementation choice.

For the policy file format (FlatBuffers `.bin` files), see [Binary Formats](../02-data-model/binary-formats.md) §3.2.

## 8. Transition to In-Memory Model

After loading and broadcasting, each rank constructs its in-memory data model from the loaded data. The in-memory structures are defined in [Internal Structures](../02-data-model/internal-structures.md) and are not specified here.

**FPHA preprocessing:** For hydros with FPHA production model and source `"computed"`, the FPHA hyperplanes are fitted during the Initialization phase (after validation, before training). Fitting uses the hydro geometry (volume-area-level curves from `hydro_geometry.parquet`), topology data (productivity, tailrace, hydraulic losses, turbine efficiency from `hydros.json`), and fitting configuration (discretization points from `hydro_production_models.json`). See [Hydro Production Models](../01-math/hydro-production-models.md) for the mathematical formulation and [Input Hydro Extensions](../02-data-model/input-hydro-extensions.md) for the required input data.

The key invariant is: **after the Initialization phase completes, all ranks hold identical copies of the validated case data** (except for per-rank scenario assignments, which are determined during Scenario Generation). See [CLI and Lifecycle](./cli-and-lifecycle.md) §5.2 for the phase sequence.

## Cross-References

- [Input Directory Structure](../02-data-model/input-directory-structure.md) — File inventory, directory layout, `config.json` schema
- [Internal Structures](../02-data-model/internal-structures.md) — In-memory data model constructed from loaded data
- [CLI and Lifecycle](./cli-and-lifecycle.md) — Execution phases (Validation, Initialization) that orchestrate loading
- [Validation Architecture](./validation-architecture.md) — Multi-layer validation applied during and after loading
- [Design Principles](../00-overview/design-principles.md) — Declaration order invariance requiring canonicalization after loading
- [Configuration Reference](../05-config/configuration-reference.md) — Complete `config.json` schema loaded in step 1
- [Input System Entities](../02-data-model/input-system-entities.md) — Entity registries and default bounds
- [Input Scenarios](../02-data-model/input-scenarios.md) — Scenario model files and policy graph
- [Input Constraints](../02-data-model/input-constraints.md) — Generic constraints, penalty overrides, exchange factors
- [PAR Inflow Model](../01-math/par-inflow-model.md) — PAR(p) model parameters loaded from `inflow_seasonal_stats.parquet` and `inflow_ar_coefficients.parquet`
- [Binary Formats](../02-data-model/binary-formats.md) — Policy file format (FlatBuffers) for warm-start loading
- [Penalty System](../02-data-model/penalty-system.md) — Default penalty values and override cascade
