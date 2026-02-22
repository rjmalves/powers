---
status: approved
review_priority: 3-medium
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §4.7 (Manifest Files)"
  - "DATA_MODEL_SPECIFICATION.md §4.8 (Metadata File)"
  - "DATA_MODEL_SPECIFICATION.md §4.9 (MPI Direct Hive Partitioning)"
  - "DATA_MODEL_SPECIFICATION.md §4.10 (Output Configuration)"
  - "DATA_MODEL_SPECIFICATION.md §4.11 (Production Scale Reference)"
  - "DATA_MODEL_SPECIFICATION.md §4.12 (Validation and Integrity)"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: "Approved after two revision rounds (12 structural + 11 cross-reference fixes). HPC write granularity note added pending work-distribution spec."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §4.7-4.12"
  - date: 2026-02-22
    description: "Review rewrite: fix priority, remove pseudocode/code examples, fix termination reasons, fix policy hash path, deduplicate config/scale sections, add risk-averse gap caveat, add block_mode note"
  - date: 2026-02-22
    description: "Cross-reference audit: configuration_snapshot aligned with approved config.json (removed phantom convergence_tolerance, fixed cut_selection field names, replaced upper_bound with upper_bound_evaluation, embedded stopping_rules verbatim). Training manifest iterations.target renamed to max_iterations. convergence.achieved semantics clarified. final_gap_percent made nullable. num_stages/num_blocks_per_stage consistency fixed."
  - date: 2026-02-22
    description: "Added intra-rank thread parallelism note to §3.2 Write Semantics — write granularity (per-rank vs per-thread) deferred to HPC work distribution specs."
---

# Output Infrastructure

## Purpose

This spec defines the infrastructure layer for POWE.RS output: manifest files for crash recovery, metadata for reproducibility, MPI-native Hive partitioning for parallel writes, and validation/integrity checks.

For output Parquet schemas (simulation and training column definitions), see [Output Schemas](output-schemas.md).
For output configuration options within `config.json`, see [Configuration Reference](../05-config/configuration-reference.md).

## 1. Manifest Files

Manifest files enable crash recovery and incremental writes. They track completion status and are updated atomically.

### 1.1 Simulation Manifest (`simulation/_manifest.json`)

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/simulation_manifest.schema.json",
  "version": "2.0.0",
  "status": "complete",
  "started_at": "2026-01-17T10:00:00Z",
  "completed_at": "2026-01-17T10:15:00Z",
  "scenarios": {
    "total": 2000,
    "completed": 2000,
    "failed": 0
  },
  "partitions_written": ["scenario_id=0/", "scenario_id=1/", "..."],
  "checksum": {
    "algorithm": "xxhash64",
    "value": "a1b2c3d4e5f6"
  },
  "mpi_info": {
    "world_size": 128,
    "ranks_participated": 128
  }
}
```

| Field                         | Type   | Description                                        |
| ----------------------------- | ------ | -------------------------------------------------- |
| `status`                      | string | `"running"`, `"complete"`, `"failed"`, `"partial"` |
| `started_at`                  | string | ISO 8601 timestamp                                 |
| `completed_at`                | string | ISO 8601 timestamp (null if not complete)          |
| `scenarios.total`             | i32    | Total scenarios to simulate                        |
| `scenarios.completed`         | i32    | Successfully completed scenarios                   |
| `scenarios.failed`            | i32    | Failed scenarios                                   |
| `partitions_written`          | array  | List of Hive partition directories written         |
| `checksum`                    | object | Integrity checksum for validation                  |
| `mpi_info.world_size`         | i32    | Number of MPI ranks                                |
| `mpi_info.ranks_participated` | i32    | Ranks that wrote data                              |

**Crash Recovery Protocol:**

1. On startup, check if `_manifest.json` exists with `status: "running"`
2. If found, read `partitions_written` to identify completed work
3. Resume from incomplete scenarios
4. Update manifest atomically on completion

### 1.2 Training Manifest (`training/_manifest.json`)

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/training_manifest.schema.json",
  "version": "2.0.0",
  "status": "complete",
  "started_at": "2026-01-17T08:00:00Z",
  "completed_at": "2026-01-17T12:30:00Z",
  "iterations": {
    "max_iterations": 100,
    "completed": 100,
    "converged_at": 87
  },
  "convergence": {
    "achieved": true,
    "final_gap_percent": 0.45,
    "termination_reason": "simulation"
  },
  "cuts": {
    "total_generated": 1250000,
    "total_active": 980000,
    "peak_active": 1100000
  },
  "checksum": {
    "algorithm": "xxhash64",
    "policy_value": "f1e2d3c4b5a6",
    "convergence_value": "1a2b3c4d5e6f"
  },
  "mpi_info": {
    "world_size": 128,
    "forward_passes_per_iteration": 8
  }
}
```

| Field                            | Type   | Description                                                                                                                                                                                 |
| -------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `status`                         | string | `"running"`, `"complete"`, `"failed"`, `"converged"`                                                                                                                                        |
| `iterations.max_iterations`      | i32    | Maximum iterations from `iteration_limit` stopping rule                                                                                                                                     |
| `iterations.completed`           | i32    | Iterations actually run                                                                                                                                                                     |
| `iterations.converged_at`        | i32    | Iteration where convergence-oriented rule triggered (null if terminated by safety limit)                                                                                                    |
| `convergence.achieved`           | bool   | Whether a convergence-oriented rule (`bound_stalling` or `simulation`) triggered, as opposed to a safety limit (`iteration_limit`, `time_limit`)                                            |
| `convergence.final_gap_percent`  | f64    | Final optimality gap (null if upper bound evaluation is disabled). Under CVaR risk measures, this gap is not a valid optimality bound; see [Risk Measures](../01-math/risk-measures.md) §10 |
| `convergence.termination_reason` | string | One of: `"iteration_limit"`, `"time_limit"`, `"bound_stalling"`, `"simulation"`. See [Stopping Rules](../01-math/stopping-rules.md)                                                         |
| `cuts.total_generated`           | i64    | Total cuts generated during training                                                                                                                                                        |
| `cuts.total_active`              | i64    | Active cuts at termination                                                                                                                                                                  |
| `cuts.peak_active`               | i64    | Peak active cuts during training                                                                                                                                                            |

## 2. Metadata File (`training/metadata.json`)

Comprehensive metadata for reproducibility, audit trails, and debugging.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/training_metadata.schema.json",
  "version": "2.0.0",
  "run_info": {
    "run_id": "uuid-v4-here",
    "started_at": "2026-01-17T08:00:00Z",
    "completed_at": "2026-01-17T12:30:00Z",
    "duration_seconds": 16200,
    "powers_version": "2.0.0",
    "solver": "highs",
    "solver_version": "1.7.2",
    "hostname": "compute-node-001",
    "user": "scheduler"
  },
  "configuration_snapshot": {
    "seed": 42,
    "num_forward_passes": 200,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 100 },
      {
        "type": "simulation",
        "replications": 100,
        "period": 20,
        "bound_window": 5,
        "distance_tol": 0.01,
        "bound_tol": 0.0001
      }
    ],
    "stopping_mode": "any",
    "cut_selection": {
      "enabled": true,
      "method": "level1"
    },
    "upper_bound_evaluation": {
      "enabled": true,
      "initial_iteration": 10,
      "interval_iterations": 5
    },
    "policy_mode": "fresh"
  },
  "problem_dimensions": {
    "num_stages": 12,
    "num_blocks_per_stage": [
      730, 730, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720
    ],
    "num_hydros": 160,
    "num_thermals": 200,
    "num_buses": 5,
    "num_lines": 8,
    "num_pumping_stations": 3,
    "num_contracts": 2,
    "num_generic_constraints": 15,
    "state_dimension": 320,
    "lp_dimensions": {
      "variables_per_stage_avg": 1500,
      "constraints_per_stage_avg": 2000,
      "nonzeros_per_stage_avg": 8500
    }
  },
  "performance_summary": {
    "total_lp_solves": 125000000,
    "avg_lp_time_us": 145,
    "median_lp_time_us": 132,
    "p99_lp_time_us": 450,
    "peak_memory_mb": 16384,
    "total_communication_time_seconds": 850,
    "io_write_time_seconds": 45
  },
  "data_integrity": {
    "input_hash": "sha256:abc123...",
    "config_hash": "sha256:def456...",
    "policy_hash": "sha256:789xyz...",
    "convergence_hash": "sha256:uvw012..."
  },
  "environment": {
    "mpi_implementation": "OpenMPI",
    "mpi_version": "4.1.5",
    "num_ranks": 128,
    "cpus_per_rank": 4,
    "memory_per_rank_gb": 32,
    "numa_binding": true,
    "omp_num_threads": 1
  }
}
```

**Notes on `configuration_snapshot`:**

- This is an informational record of the training configuration, not a normative schema. The canonical config schema is defined in [Configuration Reference](../05-config/configuration-reference.md).
- `stopping_rules` is recorded verbatim from `config.json` so that the termination behavior can be reconstructed from the output alone.
- `cut_selection.method` uses the values from [Cut Management §9](../01-math/cut-management.md): `"level1"`, `"lml1"`, or `"domination"`.
- `upper_bound_evaluation` mirrors the config section from [Input Directory Structure §2](input-directory-structure.md). Vertex-based (SIDP) upper bounds are enabled when the `upper_bound_evaluation` section is present with `enabled: true`; see [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md).

**Notes on `problem_dimensions`:**

- `num_blocks_per_stage` is an array because block count depends on each stage's `block_mode` (block mode is configured per stage, not globally). See [Block Formulations](../01-math/block-formulations.md).
- `lp_dimensions` are averages across stages; actual dimensions vary with block count.

**Notes on `data_integrity`:**

- `input_hash`: SHA-256 of concatenated input file hashes.
- `config_hash`: SHA-256 of normalized `config.json`.
- `policy_hash`: SHA-256 computed over the policy FlatBuffers files in `policy/cuts/stage_*.bin`. See [Binary Formats](binary-formats.md) §3.2 for the policy directory structure.
- `convergence_hash`: SHA-256 of `training/convergence.parquet` content.

## 3. MPI Direct Hive Partitioning

Each MPI rank writes directly to Hive partition directories without coordination. For Hive partitioning design principles, see [Output Schemas](output-schemas.md) §2.1.

### 3.1 Directory Layout

```
simulation/
├── costs/
│   ├── scenario_id=0/data.parquet      # Written by rank 0
│   ├── scenario_id=1/data.parquet      # Written by rank 0
│   ├── scenario_id=2/data.parquet      # Written by rank 1
│   └── ...
├── hydros/
│   ├── scenario_id=0/data.parquet
│   └── ...
└── _manifest.json                       # Written by rank 0 only
```

### 3.2 Write Semantics

**Scenario assignment:** Round-robin — `rank = scenario_id % world_size`. Each rank writes only its assigned scenarios.

**Write protocol:**

1. Each rank writes its assigned partitions independently (embarrassingly parallel — no inter-rank coordination during writes).
2. All ranks synchronize at a barrier after writes complete.
3. Rank 0 writes the manifest file after the barrier.

**Atomic write pattern:** Each file is written to a temporary path (`data.parquet.tmp`), flushed to disk, then atomically renamed to `data.parquet`. This prevents partial files from appearing as valid output.

> **Note — Intra-rank thread parallelism (pending HPC specs):** The write protocol above describes rank-level granularity. However, POWE.RS uses hybrid MPI+OpenMP parallelism where multiple threads within each rank may independently process scenarios. If all thread-owned scenarios funnel through a single rank-level writer, this becomes a serialization bottleneck at scale. The actual write responsibility assignment (per-rank vs per-thread) and synchronization strategy will be defined in the HPC work distribution specs. The invariants that must be preserved regardless of the final design are: (1) each partition is written by exactly one writer, (2) writes use the atomic temp-file-then-rename pattern, and (3) the manifest is written only after all partitions are confirmed complete.

### 3.3 Failure Handling

| Failure Type         | Detection                      | Recovery                            |
| -------------------- | ------------------------------ | ----------------------------------- |
| Rank crash mid-write | Missing partitions in manifest | Re-run failed scenarios only        |
| Partial file write   | Parquet read failure           | Delete and re-write partition       |
| Manifest corruption  | JSON parse error               | Rebuild from partition listing      |
| Disk full            | Write error                    | Alert, do not corrupt existing data |

## 4. Output Size Estimates

Reference output sizes for production-scale SDDP runs. For problem dimension profiles (Small through Extra Large), see [Production Scale Reference](../00-overview/production-scale-reference.md).

| Output                         | Small  | Medium | Large   | Extra Large |
| ------------------------------ | ------ | ------ | ------- | ----------- |
| `simulation/costs/`            | 50 MB  | 800 MB | 4 GB    | 20 GB       |
| `simulation/hydros/`           | 200 MB | 5 GB   | 30 GB   | 150 GB      |
| `simulation/thermals/`         | 150 MB | 4 GB   | 25 GB   | 120 GB      |
| `training/convergence.parquet` | 10 KB  | 50 KB  | 100 KB  | 250 KB      |
| `training/timing/`             | 1 MB   | 15 MB  | 120 MB  | 1.2 GB      |
| `policy/` (cuts)               | 500 MB | 8 GB   | 40 GB   | 200 GB      |
| **Total**                      | ~1 GB  | ~20 GB | ~100 GB | ~500 GB     |

**Storage recommendations:**

- Use SSD/NVMe for training (frequent random writes)
- Network filesystem acceptable for simulation (sequential writes)
- Consider parallel filesystem (Lustre, GPFS) for >100 GB outputs
- Enable compression for network transfers

### 4.1 I/O Bandwidth Requirements

| Scale       | Write Throughput | Duration | Bottleneck  |
| ----------- | ---------------- | -------- | ----------- |
| Small       | 50 MB/s          | 20s      | None        |
| Medium      | 200 MB/s         | 100s     | Network     |
| Large       | 500 MB/s         | 200s     | Filesystem  |
| Extra Large | 1+ GB/s          | 500s     | Parallel FS |

## 5. Validation and Integrity

### 5.1 Schema Validation

Each output entity must conform to the Parquet schema defined in [Output Schemas](output-schemas.md). Validation verifies column names, types, and nullability against the schema definitions.

### 5.2 Data Integrity Checks

| Check                  | Method                      | Frequency |
| ---------------------- | --------------------------- | --------- |
| Parquet file integrity | Footer checksum             | On read   |
| Partition completeness | Manifest comparison         | Post-run  |
| Row count consistency  | Cross-entity validation     | Post-run  |
| Value range validation | Min/max from bounds.parquet | Optional  |

**Cross-entity validation:** For each scenario, the number of rows in every entity output must be consistent with the stage and block counts for that scenario. For example, `costs/` has one row per (stage, block), while `hydros/` has one row per (stage, block, hydro). Missing or extra rows indicate a write failure.

### 5.3 Reproducibility Verification

The `data_integrity` section in `metadata.json` (§2) enables reproducibility verification:

- Given the same inputs (`input_hash`), configuration (`config_hash`), and random seed, the system must produce identical policy and convergence outputs.
- Two runs can be compared by checking whether their `policy_hash` and `convergence_hash` match.
- If `input_hash` differs between runs, the policy outputs are not directly comparable.

## Cross-References

- [Output Schemas](output-schemas.md) — Parquet column definitions for all entity types and Hive partitioning design
- [Binary Formats](binary-formats.md) — Policy file format (FlatBuffers `.bin` files) and directory structure
- [Input System Entities](input-system-entities.md) — Entity registries (entity IDs, names)
- [Penalty System](penalty-system.md) — Penalty costs affecting output values
- [Configuration Reference](../05-config/configuration-reference.md) — Full `config.json` reference including output configuration
- [Input Directory Structure](input-directory-structure.md) — `config.json` structure and `upper_bound_evaluation` section
- [Production Scale Reference](../00-overview/production-scale-reference.md) — Problem dimension profiles
- [Stopping Rules](../01-math/stopping-rules.md) — Termination criteria definitions
- [Cut Management](../01-math/cut-management.md) — Cut selection strategy names and parameters
- [Risk Measures](../01-math/risk-measures.md) — CVaR and lower bound validity
- [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md) — Simulation-based and SIDP upper bound mechanisms
- [Block Formulations](../01-math/block-formulations.md) — Block mode definitions (per-stage)
- [Design Principles](../00-overview/design-principles.md) — Overall design philosophy
