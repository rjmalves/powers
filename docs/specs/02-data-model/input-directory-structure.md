---
status: deferred
review_priority: 1-critical
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.1 (Directory Structure)"
  - "DATA_MODEL_SPECIFICATION.md §3.2 (Configuration — config.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.2.1 (Penalties and Costs — summary only)"
last_reviewed: 2026-02-14
reviewed_by: rogerio
review_notes: "Deferred: directory structure depends on open decisions (penalty override file format TBD, potential hydro modeling changes from CEPEL observations). Will resume after closing open edges across other specs."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.1-§3.2"
  - date: 2026-02-14
    description: "First review: added explicit directory tree with all files, added root-level files detail table, updated penalty summary from 2 to 3 categories for consistency with approved penalty-system.md"
  - date: 2026-02-14
    description: "Deferred: filesystem structure depends on unresolved decisions in other specs (penalty override format, hydro modeling scope). Will resume after closing open edges."
---

# Input Directory Structure

## Purpose

This spec defines the layout of a POWE.RS input case directory and the schema of the central configuration file `config.json`. It serves as the entry point for understanding how input data is organized and what options control solver behavior.

## 1. Directory Tree

![Directory Structure](../../diagrams/exports/svg/data/directory-structure.svg)

```
case/
├── config.json                                # Execution configuration (§2)
├── initial_conditions.json                    # Initial storage, GNL pipeline state
├── stages.json                                # Stage definitions, blocks, transitions
├── penalties.json                             # Global penalty defaults
│
├── system/                                    # Entity registries and extensions
│   ├── buses.json                             # Bus registry with deficit segments
│   ├── lines.json                             # Transmission line registry
│   ├── hydros.json                            # Hydro plant registry
│   ├── thermals.json                          # Thermal plant registry
│   ├── hydro_geometry.parquet                 # Volume-area-level curves (optional)
│   ├── hydro_production_models.json           # Stage-varying production model config (optional)
│   ├── hydro_production_data.parquet          # Turbine efficiency curves (optional)
│   ├── fpha_hyperplanes.parquet               # Precomputed FPHA planes (optional)
│   ├── pumping_stations.json                  # Pumping station registry (optional)
│   └── energy_contracts.json                  # Energy contract definitions (optional)
│
├── scenarios/                                 # Stochastic models and time series
│   ├── inflow_models.parquet                  # PAR(p) coefficients per hydro/stage
│   ├── inflow_history.parquet                 # Pre-study realized inflows (AR lags)
│   ├── load_models.parquet                    # Load uncertainty parameters (optional)
│   ├── load_factors.json                      # Block-level load scaling factors (optional)
│   ├── exchange_factors.json                  # Block-level exchange scaling factors (optional)
│   ├── correlation.json                       # Spatial correlation profiles
│   └── correlation_schedule.parquet           # Time-varying correlation schedule (optional)
│
├── constraints/                               # Time-varying bounds and generic constraints
│   ├── thermal_bounds.parquet                 # Stage-varying thermal limits (optional)
│   ├── hydro_bounds.parquet                   # Stage-varying hydro limits (optional)
│   ├── line_bounds.parquet                    # Stage-varying line limits (optional)
│   ├── contract_bounds.parquet                # Stage-varying contract limits (optional)
│   ├── generic_constraints.json               # Custom linear constraints (optional)
│   ├── constraint_bounds.parquet              # RHS bounds for generic constraints (optional)
│   └── penalty overrides (format TBD)         # Stage-varying penalty overrides (optional)
│
└── policy/                                    # Warm-start / resume data (optional)
    ├── metadata.json                          # Algorithm state, RNG, bounds
    ├── state_dictionary.json                  # State variable mapping
    ├── cuts/                                  # Outer approximation (SDDP cuts)
    ├── states/                                # Visited states for cut selection
    ├── vertices/                              # Inner approximation (if enabled)
    └── basis/                                 # Solver basis (optional)
```

The input case directory is organized into four top-level groups plus root-level configuration files:

| Directory      | Purpose                                                  | Format         |
| -------------- | -------------------------------------------------------- | -------------- |
| Root           | Configuration, penalties, stages, initial conditions     | JSON           |
| `system/`      | Entity registries (buses, lines, hydros, thermals, etc.) | JSON + Parquet |
| `scenarios/`   | Stochastic models and time series (inflow, load)         | Parquet + JSON |
| `constraints/` | Stage-varying bounds, penalties, generic constraints     | Parquet + JSON |
| `policy/`      | Warm-start and resume data (cuts, states, basis)         | JSON + binary  |

> **Format Rationale — Directory Layout**
>
> The separation follows the [Design Principles](../00-overview/design-principles.md) format selection criteria: JSON for human-editable structured objects, Parquet for large tabular time-series data. Root-level files are read once at startup; `system/` files define the physical model; `scenarios/` files define stochastic processes; `constraints/` files provide stage-varying overrides; `policy/` stores algorithm state for warm-starting or resuming.

### Root-Level Files

| File                      | Required | Description                                                                                                                                                                     | Spec Reference                               |
| ------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `config.json`             | Yes      | Central execution configuration: MPI/HPC parameters, modeling options, training settings, simulation settings, export controls. Controls all solver behavior.                   | §2 below                                     |
| `penalties.json`          | Yes      | Global default penalty values for the three-tier cascade: deficit segment costs, regularization costs, constraint violation penalties. Entity and stage overrides layer on top. | [Penalty System](penalty-system.md)          |
| `stages.json`             | Yes      | Stage definitions with block structure (count and hours), transitions between stages, discount rates, risk measure parameters (CVaR), and scenario sampling method per stage.   | [Input Scenarios §1](input-scenarios.md)     |
| `initial_conditions.json` | Yes      | Initial system state: reservoir storage levels at study start (or at entry for late-entry hydros), and GNL thermal committed dispatch pipeline.                                 | [Input Constraints §1](input-constraints.md) |

## 2. Configuration (`config.json`)

> **Note**: Solver selection (HiGHS, CPLEX, Gurobi) is determined at compile time via Cargo features due to licensing constraints. Solver parameters, retry strategies, warm-start, and basis reuse are hardcoded per solver implementation and not user-configurable.

> **Format Rationale — config.json**
>
> JSON was chosen for the central configuration because it is human-readable, easily editable, and small in size. Configuration is a **nested object** with logical groupings (MPI, training, simulation) that map naturally to JSON's hierarchical structure.

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
    "block_mode": "parallel",
    "inflow_non_negativity": "truncate_zero"
  },

  "horizon": {
    "mode": "finite",
    "max_horizon_length": 240,
    "cycle_discretization_delta": 0.1
  },

  "training": {
    "seed": 42,
    "num_forward_passes": 200,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 50 },
      { "type": "statistical", "confidence": 0.95, "tolerance": 0.01 }
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

### 2.1 MPI Configuration (HPC Parameters)

> **Background**: POWE.RS uses hybrid MPI+OpenMP parallelism for distributed computing. The `mpi` section configures communication patterns, memory management, and I/O strategies optimized for production-scale SDDP on HPC clusters.
>
> **⚠️ SLURM/PBS/LSF Integration**: Thread and memory configuration should come from the job scheduler, not hardcoded in config. Use `"auto"` mode (default) to respect scheduler allocations.

For thread binding, communication, memory, and I/O field details, see [Configuration Reference](../05-config/configuration-reference.md).

### 2.2 Block Mode Configuration

| Mode            | Description                                                                                                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parallel`      | Blocks are independent within a stage. Storage balance applies to the whole stage (sum of block durations). Simpler model, common for medium/long-term planning.              |
| `chronological` | Blocks are sequential within a stage. Storage can vary between blocks (inter-block storage variables). Enables daily/weekly cycling patterns. More variables and constraints. |

### 2.3 Horizon Mode Configuration

| Mode                | Description                                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `finite`            | Standard finite horizon. Stages proceed from first to last, then stop. Terminal cost is zero or user-defined.                                                                                    |
| `infinite_periodic` | Infinite horizon with periodic structure. The algorithm detects cycles in the transition graph and shares cuts between stages of the same "season". Requires `discount_rate > 0` in transitions. |

For detailed configuration fields and validation rules, see [Configuration Reference](../05-config/configuration-reference.md).

### 2.4 Training Configuration

Key training parameters include the random seed, number of forward passes, stopping rules, cut formulation, and forward/backward pass modes. The `stopping_mode` controls how multiple rules combine (`"any"` = OR, `"all"` = AND).

> **⚠️ Validation**: At least one `iteration_limit` rule must be present in the `stopping_rules` array.

For the complete stopping rule types and their parameters, see [Configuration Reference](../05-config/configuration-reference.md).

### 2.5 Policy Directory Configuration

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

### 2.6 Simulation Configuration

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
3. **Stage overrides** (optional, sparse — file format TBD)

Penalties are divided into three categories:

| Category                           | Examples                                           | Purpose                                                     | Typical Range     |
| ---------------------------------- | -------------------------------------------------- | ----------------------------------------------------------- | ----------------- |
| **Recourse slacks**                | `deficit_*`, `excess_cost`                         | Ensure LP feasibility when demand cannot be met             | 100–10,000 $/unit |
| **Constraint violation penalties** | `*_violation_*_cost`, `generic_violation_cost`     | Allow soft constraint violations at a cost (policy shaping) | 50–5,000 $/unit   |
| **Regularization costs**           | `spillage_cost`, `diversion_cost`, `exchange_cost` | Discourage undesirable but feasible operations              | 0.001–10 $/unit   |

For the complete penalty specification — including `penalties.json` schema, entity override format, stage-varying override schemas, resolution semantics, and the full penalty inventory — see [Penalty System](penalty-system.md).

## Cross-References

- [Design Principles](../00-overview/design-principles.md) — format selection criteria and declaration order invariance
- [Configuration Reference](../05-config/configuration-reference.md) — complete field-by-field config.json reference
- [Penalty System](penalty-system.md) — full penalty specification with cascade resolution
- [Input System Entities](input-system-entities.md) — buses, lines, hydros, and thermals registries
- [Production Scale Reference](../00-overview/production-scale-reference.md) — LP sizing and performance targets
