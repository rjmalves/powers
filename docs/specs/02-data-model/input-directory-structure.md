---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.1 (Directory Structure)"
  - "DATA_MODEL_SPECIFICATION.md §3.2 (Configuration — config.json)"
  - "DATA_MODEL_SPECIFICATION.md §3.2.1 (Penalties and Costs — summary only)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §3.1-§3.2"
---

# Input Directory Structure

## Purpose

This spec defines the layout of a POWE.RS input case directory and the schema of the central configuration file `config.json`. It serves as the entry point for understanding how input data is organized and what options control solver behavior.

## 1. Directory Tree

> **Note**: Scenario noise generation always uses standard normal distributions. Non-negative inflow values are enforced at runtime via the configured `inflow_non_negativity` method. Deterministic load can be achieved by setting variance to 0 in the uncertainty model.

![Directory Structure](../../diagrams/exports/svg/data/directory-structure.svg)

The input case directory is organized into four top-level groups:

| Directory      | Purpose                                                  | Format         |
| -------------- | -------------------------------------------------------- | -------------- |
| `system/`      | Entity registries (buses, lines, hydros, thermals, etc.) | JSON           |
| `scenarios/`   | Stochastic models and time series (inflow, load)         | Parquet        |
| `constraints/` | Stage-varying bounds, penalties, generic constraints     | Parquet + JSON |
| Root           | Configuration, penalties, stages, initial conditions     | JSON           |

> **Format Rationale — Directory Layout**
>
> The separation follows the [Design Principles](../00-overview/design-principles.md) format selection criteria: JSON for human-editable structured objects, Parquet for large tabular time-series data. Root-level files are read once at startup; `system/` files define the physical model; `scenarios/` files define stochastic processes; `constraints/` files provide stage-varying overrides.

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
3. **Stage overrides** in Parquet files (optional, sparse)

Penalties are divided into two categories:

| Category                | Examples                                           | Purpose                                               | Typical Range     |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------- | ----------------- |
| **Operational Costs**   | `spillage_cost`, `diversion_cost`, `exchange_cost` | Discourage undesirable but feasible operations        | 0.001–10 $/unit   |
| **Violation Penalties** | `deficit_*`, `excess_cost`, `*_violation_*_cost`   | Ensure LP feasibility, penalize constraint violations | 100–10,000 $/unit |

For the complete penalty specification — including `penalties.json` schema, entity override format, stage-varying Parquet schemas, resolution algorithm, and the full penalty semantics table — see [Penalty System](penalty-system.md).

## Cross-References

- [Design Principles](../00-overview/design-principles.md) — format selection criteria and declaration order invariance
- [Configuration Reference](../05-config/configuration-reference.md) — complete field-by-field config.json reference
- [Penalty System](penalty-system.md) — full penalty specification with cascade resolution
- [Input System Entities](input-system-entities.md) — buses, lines, hydros, and thermals registries
- [Production Scale Reference](../00-overview/production-scale-reference.md) — LP sizing and performance targets
