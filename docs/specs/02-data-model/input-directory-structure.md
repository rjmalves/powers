---
status: approved
review_priority: 1-critical
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.1 (Directory Structure)"
  - "DATA_MODEL_SPECIFICATION.md §3.2 (Configuration — config.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.2.1 (Penalties and Costs — summary only)"
last_reviewed: 2026-02-17
reviewed_by: rogerio
review_notes: "Format decisions closed. All tabular data uses Parquet. Correlation schedule embedded in correlation.json. exchange_factors moved to constraints/. config.json restructured with minimal example."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.1-§3.2"
  - date: 2026-02-14
    description: "First review: added explicit directory tree with all files, added root-level files detail table, updated penalty summary from 2 to 3 categories for consistency with approved penalty-system.md"
  - date: 2026-02-14
    description: "Deferred: filesystem structure depends on unresolved decisions in other specs (penalty override format, hydro modeling scope). Will resume after closing open edges."
  - date: 2026-02-16
    description: "Re-reviewed after all P1 specs approved. Fixed 17 issues: removed deleted hydro_production_data file, added non_controllable_sources.json, added inflow_history, added pumping_bounds, renamed constraint_bounds to generic_constraint_bounds, dropped all hardcoded .parquet extensions with format-agnostic names, updated config.json (removed global block_mode, removed horizon section moved to stages.json), rewrote §2.2/§2.3 (merged Block Mode and Horizon Mode into single Modeling Configuration note), updated root-level files table descriptions, added fpha_turbined_cost and curtailment_cost to penalty summary, updated directory purpose table with format-agnostic descriptions."
  - date: 2026-02-17
    description: "Closed all file format decisions. All tabular data files use Parquet with .parquet extensions. Split inflow_models into inflow_seasonal_stats.parquet + inflow_ar_coefficients.parquet. Renamed load_models to load_seasonal_stats.parquet. Moved exchange_factors.json from scenarios/ to constraints/ (affects LP bounds, not scenario pipeline). Removed correlation_schedule (embedded in correlation.json). Split penalty_overrides into 4 entity-specific .parquet files. Added external_scenarios.parquet. Restructured config.json with minimal example first, full example second. Made all MPI sub-sections optional with code defaults."
---

# Input Directory Structure

## Purpose

This spec defines the layout of a POWE.RS input case directory and the schema of the central configuration file `config.json`. It serves as the entry point for understanding how input data is organized and what options control solver behavior.

## 1. Directory Tree

![Directory Structure](../../diagrams/exports/svg/data/directory-structure.svg)

```
case/
├── config.json                                # Execution configuration (§2)
├── initial_conditions.json                    # Initial storage (operating + filling hydros)
├── stages.json                                # Stage/season definitions, policy graph, blocks
├── penalties.json                             # Global penalty defaults
│
├── system/                                    # Entity registries and extensions
│   ├── buses.json                             # Bus registry with deficit segments
│   ├── lines.json                             # Transmission line registry
│   ├── hydros.json                            # Hydro plant registry (includes tailrace,
│   │                                          #   losses, efficiency, evaporation)
│   ├── thermals.json                          # Thermal plant registry
│   ├── non_controllable_sources.json          # Wind/solar sources (optional)
│   ├── pumping_stations.json                  # Pumping station registry (optional)
│   ├── energy_contracts.json                  # Energy contract definitions (optional)
│   ├── hydro_geometry.parquet                 # Volume-area-level curves (optional)
│   ├── hydro_production_models.json           # Stage-varying production model config (optional)
│   └── fpha_hyperplanes.parquet               # Precomputed FPHA planes (optional)
│
├── scenarios/                                 # Stochastic models and time series
│   ├── inflow_history.parquet                 # Historical inflow observations (optional)
│   ├── inflow_seasonal_stats.parquet          # Seasonal mean/std per hydro/stage (optional)
│   ├── inflow_ar_coefficients.parquet         # PAR(p) AR coefficients per hydro/stage/lag (optional)
│   ├── external_scenarios.parquet             # Pre-computed scenario values (optional)
│   ├── load_seasonal_stats.parquet            # Load mean/std per bus/stage (optional)
│   ├── load_factors.json                      # Block-level load scaling factors (optional)
│   └── correlation.json                       # Spatial correlation profiles + schedule (optional)
│
├── constraints/                               # Time-varying bounds and generic constraints
│   ├── thermal_bounds.parquet                 # Stage-varying thermal limits (optional)
│   ├── hydro_bounds.parquet                   # Stage-varying hydro limits (optional)
│   ├── line_bounds.parquet                    # Stage-varying line limits (optional)
│   ├── pumping_bounds.parquet                 # Stage-varying pumping limits (optional)
│   ├── contract_bounds.parquet                # Stage-varying contract limits (optional)
│   ├── exchange_factors.json                  # Block-level exchange capacity factors (optional)
│   ├── generic_constraints.json               # Custom linear constraints (optional)
│   ├── generic_constraint_bounds.parquet      # RHS bounds for generic constraints (optional)
│   ├── penalty_overrides_bus.parquet          # Stage-varying bus penalties (optional)
│   ├── penalty_overrides_line.parquet         # Stage-varying line penalties (optional)
│   ├── penalty_overrides_hydro.parquet        # Stage-varying hydro penalties (optional)
│   └── penalty_overrides_ncs.parquet          # Stage-varying NCS penalties (optional)
│
└── policy/                                    # Warm-start / resume data (optional)
    ├── metadata.json                          # Algorithm state, RNG, bounds
    ├── state_dictionary.json                  # State variable mapping
    ├── cuts/                                  # Outer approximation (SDDP cuts)
    ├── states/                                # Visited states for cut selection
    ├── vertices/                              # Inner approximation (if enabled)
    └── basis/                                 # Solver basis for warm-start (optional)
```

The input case directory is organized into four top-level groups plus root-level configuration files:

| Directory      | Purpose                                                      | Format                      |
| -------------- | ------------------------------------------------------------ | --------------------------- |
| Root           | Configuration, penalties, stages, initial conditions         | JSON                        |
| `system/`      | Entity registries and extension data (all 7 element types)   | JSON + Parquet              |
| `scenarios/`   | Stochastic models, history, block factors, correlation       | JSON + Parquet              |
| `constraints/` | Stage-varying bounds, penalty overrides, generic constraints | JSON + Parquet              |
| `policy/`      | Warm-start and resume data (cuts, states, basis)             | JSON + FlatBuffers (binary) |

> **Format Rationale — Directory Layout**
>
> The separation follows the [Design Principles](../00-overview/design-principles.md) format selection criteria: **JSON** for human-editable structured objects with nested/optional fields (registries, configuration, correlation profiles); **Parquet** for typed columnar tabular data (entity-level lookup tables, stage-varying overrides, time series, scenario parameters). Root-level files are read once at startup; `system/` files define the physical model; `scenarios/` files define stochastic processes; `constraints/` files provide stage-varying overrides and block-level capacity factors; `policy/` stores algorithm state for warm-starting or resuming. Binary policy files use FlatBuffers for zero-copy deserialization — see [Binary Formats](binary-formats.md).
>
> **Why Parquet for all tabular data**: Parquet provides self-describing schemas with typed columns, columnar compression, efficient filtering, and excellent tooling in Python/R/Arrow. Even for small files (~100s of rows), the consistency benefit outweighs the minor overhead — users need Parquet tooling for the larger files regardless, and a future frontend will handle visual editing. See [Binary Formats §1](binary-formats.md) for the complete format decision framework.

### Root-Level Files

| File                      | Required | Description                                                                                                                                                                                                                                                | Spec Reference                               |
| ------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `config.json`             | Yes      | Central execution configuration: MPI/HPC parameters, modeling options, training settings, simulation settings, export controls. Controls all solver behavior.                                                                                              | §2 below                                     |
| `penalties.json`          | Yes      | Global default penalty values for the three-tier cascade: deficit segment costs, regularization costs, constraint violation penalties. Entity and stage overrides layer on top.                                                                            | [Penalty System](penalty-system.md)          |
| `stages.json`             | Yes      | Season definitions with calendar mapping, policy graph (transitions, horizon type, annual discount rate), stage definitions with per-stage block structure, block mode, state variables, risk measure (CVaR), scenario sampling method, and num_scenarios. | [Input Scenarios §1](input-scenarios.md)     |
| `initial_conditions.json` | Yes      | Initial system state: operating hydro storage levels (`storage` array) and filling hydro storage levels (`filling_storage` array, can be below dead volume). GNL pipeline state deferred — see [Input Constraints §1](input-constraints.md).               | [Input Constraints §1](input-constraints.md) |

## 2. Configuration (`config.json`)

> **Note**: Solver selection (HiGHS, CPLEX, Gurobi) is determined at compile time via Cargo features due to licensing constraints. Solver parameters, retry strategies, warm-start, and basis reuse are hardcoded per solver implementation and not user-configurable.

> **Format Rationale — config.json**
>
> JSON was chosen for the central configuration because it is human-readable, easily editable, and small in size. Configuration is a **nested object** with logical groupings (MPI, training, simulation) that map naturally to JSON's hierarchical structure. All sections have solid code defaults — the minimal valid config is very small.

**Minimal example** — only fields with no reasonable default:

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/config.schema.json",
  "version": "2.0.0",

  "training": {
    "seed": 42,
    "num_forward_passes": 200,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 50 }]
  }
}
```

All omitted sections (`mpi`, `modeling`, `upper_bound_evaluation`, `policy`, `simulation`, `exports`) use code defaults. The `mpi` section auto-detects thread counts, scheduler integration, and memory layout. See [Configuration Reference](../05-config/configuration-reference.md) for all defaults.

**Full example** — all sections with explicit overrides:

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/config.schema.json",
  "version": "2.0.0",

  "mpi": {
    "threads_per_rank": "auto",
    "thread_binding": "auto",
    "places": "auto",

    "scheduler_integration": {
      "enabled": true,
      "priority": ["slurm", "pbs", "lsf", "config"],
      "fallback_threads": 4,
      "memory_safety_factor": 0.9,
      "warn_on_override": true
    },

    "communication": {
      "cut_aggregation": "hierarchical",
      "aggregation_tree_fanout": 8,
      "backward_pipeline": true,
      "use_persistent_collectives": true,
      "use_shared_memory_windows": true
    },

    "memory": {
      "fcf_sharing": "intra_node_shared",
      "numa_aware_allocation": true,
      "first_touch_init": true
    },

    "io": {
      "parallel_warm_start": true,
      "checkpoint_writers": 4,
      "checkpoint_compression": "zstd"
    },

    "solver": {
      "threads_per_solve": 1
    }
  },

  "modeling": {
    "inflow_non_negativity": "truncate_zero"
  },

  "training": {
    "seed": 42,
    "num_forward_passes": 200,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 50 },
      { "type": "bound_stalling", "iterations": 10, "tolerance": 0.0001 }
    ],
    "stopping_mode": "any",
    "cut_formulation": "single",
    "forward_pass": {
      "type": "default"
    },
    "cut_selection": {
      "enabled": true,
      "method": "domination",
      "threshold": 0
    }
  },

  "upper_bound_evaluation": {
    "enabled": true,
    "initial_iteration": 10,
    "interval_iterations": 5
  },

  "policy": {
    "path": "./policy",
    "mode": "fresh",
    "checkpointing": {
      "enabled": true,
      "initial_iteration": 10,
      "interval_iterations": 10,
      "store_basis": true,
      "compress": true
    },
    "validate_compatibility": true
  },

  "simulation": {
    "enabled": true,
    "num_scenarios": 2000,
    "policy_type": "outer",
    "output_path": "./simulation",
    "output_mode": "streaming",
    "sampling_scheme": {
      "type": "in_sample"
    }
  },

  "exports": {
    "training": true,
    "cuts": true,
    "states": true,
    "vertices": true,
    "simulation": true,
    "forward_detail": false,
    "backward_detail": false,
    "compression": "zstd"
  }
}
```

The subsections below describe each configuration group. For the complete field-by-field reference with defaults and validation rules, see [Configuration Reference](../05-config/configuration-reference.md).

### 2.1 MPI Configuration (HPC Parameters) — Optional

> **Background**: POWE.RS uses hybrid MPI+OpenMP parallelism for distributed computing. The `mpi` section configures communication patterns, memory management, and I/O strategies optimized for production-scale SDDP on HPC clusters.
>
> **All `mpi` fields are optional.** When omitted, the system auto-detects thread counts from the job scheduler (SLURM/PBS/LSF), uses sensible defaults for communication and memory, and configures I/O based on available resources. The entire `mpi` section can be omitted for default behavior.

For thread binding, communication, memory, and I/O field details, see [Configuration Reference](../05-config/configuration-reference.md).

### 2.2 Modeling Configuration

| Field                   | Type   | Default           | Description                                                                                                            |
| ----------------------- | ------ | ----------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `inflow_non_negativity` | string | `"truncate_zero"` | Strategy for ensuring non-negative generated inflows. See [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md). |

> **Note**: Block mode (`parallel` or `chronological`) is configured **per stage** in `stages.json`, not globally. See [Input Scenarios §1.5](input-scenarios.md). Horizon mode (`finite_horizon` or `cyclic`) is configured in the `policy_graph` section of `stages.json`. See [Input Scenarios §1.2](input-scenarios.md).

### 2.3 Training Configuration

Key training parameters include the random seed, number of forward passes, stopping rules, cut formulation, and forward/backward pass modes. The `stopping_mode` controls how multiple rules combine (`"any"` = OR, `"all"` = AND).

> **⚠️ Validation**: At least one `iteration_limit` rule must be present in the `stopping_rules` array.

For the complete stopping rule types and their parameters, see [Configuration Reference](../05-config/configuration-reference.md).

### 2.4 Policy Directory Configuration

| Field                    | Type   | Default      | Description                                                  |
| ------------------------ | ------ | ------------ | ------------------------------------------------------------ |
| `path`                   | string | `"./policy"` | Directory for policy data (cuts, states, vertices, basis)    |
| `mode`                   | string | `"fresh"`    | How to initialize: `"fresh"`, `"warm_start"`, or `"resume"`  |
| `validate_compatibility` | bool   | true         | Verify state dimension and entity compatibility when loading |

**Policy Modes:**

| Mode         | Behavior                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------- |
| `fresh`      | Start from scratch. Ignore any existing data in `policy/`.                                    |
| `warm_start` | Load existing cuts/states to initialize, but reset iteration count and use fresh RNG seed.    |
| `resume`     | Load full algorithm state including RNG, iteration count. Continue exactly where interrupted. |

### 2.5 Simulation Configuration

| Field                  | Type   | Default       | Description                                       |
| ---------------------- | ------ | ------------- | ------------------------------------------------- |
| `enabled`              | bool   | false         | Enable post-training simulation                   |
| `num_scenarios`        | i32    | 2000          | Number of simulation scenarios                    |
| `policy_type`          | string | `"outer"`     | `"outer"` (cuts) or `"inner"` (vertices)          |
| `sampling_scheme.type` | string | `"in_sample"` | `"in_sample"`, `"out_of_sample"`, or `"external"` |

## 3. Penalties and Costs (Summary)

The LP must always be feasible. Penalty costs on slack variables ensure this by allowing constraint violations at a high cost. POWE.RS uses a **three-tier cascade** for penalty resolution:

1. **Global defaults** in `penalties.json` (required)
2. **Entity overrides** inline in entity JSON files (optional)
3. **Stage overrides** in per-entity-type Parquet files (optional, sparse)

Penalties are divided into three categories:

| Category                           | Examples                                                                                     | Purpose                                                     | Typical Range     |
| ---------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------- |
| **Recourse slacks**                | `deficit_*`, `excess_cost`                                                                   | Ensure LP feasibility when demand cannot be met             | 100–10,000 $/unit |
| **Constraint violation penalties** | `*_violation_*_cost`, `generic_violation_cost`                                               | Allow soft constraint violations at a cost (policy shaping) | 50–5,000 $/unit   |
| **Regularization costs**           | `spillage_cost`, `diversion_cost`, `exchange_cost`, `fpha_turbined_cost`, `curtailment_cost` | Discourage undesirable but feasible operations              | 0.001–10 $/unit   |

For the complete penalty specification — including `penalties.json` schema, entity override format, stage-varying override schemas, resolution semantics, and the full penalty inventory — see [Penalty System](penalty-system.md).

## Cross-References

- [Design Principles](../00-overview/design-principles.md) — format selection criteria and declaration order invariance
- [Configuration Reference](../05-config/configuration-reference.md) — complete field-by-field config.json reference
- [Penalty System](penalty-system.md) — full penalty specification with cascade resolution
- [Input System Entities](input-system-entities.md) — all 7 element type registries (buses, lines, hydros, thermals, non-controllable sources, pumping stations, energy contracts)
- [Input Hydro Extensions](input-hydro-extensions.md) — hydro geometry, production models, FPHA hyperplanes
- [Input Scenarios](input-scenarios.md) — stages.json schema, inflow models, load factors, correlations
- [Input Constraints](input-constraints.md) — initial conditions, stage-varying bounds, exchange factors, generic constraints
- [Internal Structures](internal-structures.md) — in-memory data model built from these input files
- [Binary Formats](binary-formats.md) — policy directory FlatBuffers schemas (cuts, states, basis)
- [Production Scale Reference](../00-overview/production-scale-reference.md) — LP sizing and performance targets
