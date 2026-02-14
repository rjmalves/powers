---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §4.7 (Manifest Files)"
  - "DATA_MODEL_SPECIFICATION.md §4.8 (Metadata File)"
  - "DATA_MODEL_SPECIFICATION.md §4.9 (MPI Direct Hive Partitioning)"
  - "DATA_MODEL_SPECIFICATION.md §4.10 (Output Configuration)"
  - "DATA_MODEL_SPECIFICATION.md §4.11 (Production Scale Reference)"
  - "DATA_MODEL_SPECIFICATION.md §4.12 (Validation and Integrity)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §4.7-4.12"
---

# Output Infrastructure

## Purpose

This spec defines the infrastructure layer for POWE.RS output: manifest files for crash recovery, metadata for reproducibility, MPI-native Hive partitioning for parallel writes, output configuration, production scale reference, and validation/integrity checks.

For output Parquet schemas (simulation and training column definitions), see [Output Schemas](output-schemas.md).

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
    "target": 100,
    "completed": 100,
    "converged_at": 87
  },
  "convergence": {
    "achieved": true,
    "final_gap_percent": 0.45,
    "termination_reason": "gap_tolerance"
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

| Field                            | Type   | Description                                                               |
| -------------------------------- | ------ | ------------------------------------------------------------------------- |
| `status`                         | string | `"running"`, `"complete"`, `"failed"`, `"converged"`                      |
| `iterations.target`              | i32    | Maximum iterations configured                                             |
| `iterations.completed`           | i32    | Iterations actually run                                                   |
| `iterations.converged_at`        | i32    | Iteration where convergence achieved (null if not)                        |
| `convergence.achieved`           | bool   | Whether gap tolerance was reached                                         |
| `convergence.final_gap_percent`  | f64    | Final optimality gap                                                      |
| `convergence.termination_reason` | string | `"gap_tolerance"`, `"max_iterations"`, `"time_limit"`, `"user_interrupt"` |
| `cuts.total_generated`           | i64    | Total cuts generated during training                                      |
| `cuts.total_active`              | i64    | Active cuts at termination                                                |
| `cuts.peak_active`               | i64    | Peak active cuts during training                                          |

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
    "num_iterations": 100,
    "num_forward_passes": 8,
    "convergence_tolerance": 0.5,
    "cut_selection": {
      "enabled": true,
      "strategy": "level_one",
      "max_cuts_per_stage": 10000
    },
    "upper_bound": {
      "enabled": true,
      "frequency": 10,
      "num_scenarios": 1000
    },
    "policy_mode": "fresh",
    "seed": 42
  },
  "problem_dimensions": {
    "num_stages": 120,
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

## 3. MPI Direct Hive Partitioning

Each MPI rank writes directly to Hive partition directories without coordination.

### 3.1 Writing Strategy

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

**Scenario assignment**: Round-robin — `rank = scenario_id % world_size`. Each rank writes only its assigned scenarios. No inter-rank coordination during writes (embarrassingly parallel).

### 3.2 Write Protocol

```rust
// Pseudo-code for MPI Hive-partitioned writes
fn write_simulation_results(results: &SimulationResults, config: &OutputConfig) {
    let rank = mpi::comm_world().rank();
    let world_size = mpi::comm_world().size();

    // Each rank writes its assigned scenarios
    for scenario_id in (rank..num_scenarios).step_by(world_size) {
        let partition_path = format!(
            "{}/scenario_id={}/data.parquet",
            config.simulation_path, scenario_id
        );
        write_parquet(&results[scenario_id], &partition_path)?;
    }

    // Barrier before manifest write
    mpi::comm_world().barrier();

    // Only rank 0 writes manifest
    if rank == 0 {
        write_manifest(&manifest)?;
    }
}
```

### 3.3 Failure Handling

| Failure Type         | Detection                      | Recovery                            |
| -------------------- | ------------------------------ | ----------------------------------- |
| Rank crash mid-write | Missing partitions in manifest | Re-run failed scenarios only        |
| Partial file write   | Parquet read failure           | Delete and re-write partition       |
| Manifest corruption  | JSON parse error               | Rebuild from partition listing      |
| Disk full            | Write error                    | Alert, do not corrupt existing data |

**Atomic write pattern**: write to `data.parquet.tmp` → `fsync()` → atomic `rename()` to `data.parquet`.

### 3.4 Reading Partitioned Data

```python
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# Read all scenarios (automatic partition discovery)
dataset = ds.dataset("simulation/hydros/", format="parquet", partitioning="hive")
table = dataset.to_table()

# Filter to specific scenarios
table = dataset.to_table(filter=ds.field("scenario_id") < 100)

# Read single scenario
single = pq.read_table("simulation/hydros/scenario_id=42/data.parquet")
```

```rust
use polars::prelude::*;

// Read all partitions with lazy evaluation
let df = LazyFrame::scan_parquet(
    "simulation/hydros/**/data.parquet",
    ScanArgsParquet::default()
)?.collect()?;

// Single scenario (partition pruning)
let df = LazyFrame::scan_parquet(
    "simulation/hydros/scenario_id=42/data.parquet",
    ScanArgsParquet::default()
)?.collect()?;
```

## 4. Output Configuration

Control output generation via `config.json`:

```json
{
  "output": {
    "simulation_path": "./simulation",
    "training_path": "./training",
    "simulation": {
      "enabled": true,
      "entities": {
        "costs": true,
        "hydros": true,
        "thermals": true,
        "exchanges": true,
        "buses": true,
        "pumping_stations": true,
        "contracts": true,
        "batteries": false,
        "non_controllables": false,
        "inflow_lags": true,
        "violations": true
      },
      "compression": "zstd",
      "compression_level": 3
    },
    "training": {
      "enabled": true,
      "convergence": true,
      "timing": {
        "iterations": true,
        "mpi_ranks": true
      },
      "compression": "snappy"
    },
    "dictionaries": {
      "enabled": true,
      "codes": true,
      "bounds": true,
      "state_dictionary": true,
      "variables": true,
      "entities": true
    }
  }
}
```

| Field                          | Type   | Default          | Description                        |
| ------------------------------ | ------ | ---------------- | ---------------------------------- |
| `simulation_path`              | string | `"./simulation"` | Simulation output directory        |
| `training_path`                | string | `"./training"`   | Training output directory          |
| `simulation.enabled`           | bool   | `true`           | Enable simulation outputs          |
| `simulation.entities.*`        | bool   | varies           | Per-entity output control          |
| `simulation.compression`       | string | `"zstd"`         | Parquet compression codec          |
| `simulation.compression_level` | i32    | `3`              | Compression level (codec-specific) |
| `training.enabled`             | bool   | `true`           | Enable training outputs            |
| `training.timing.iterations`   | bool   | `true`           | Write iteration timing             |
| `training.timing.mpi_ranks`    | bool   | `true`           | Write per-rank timing              |
| `dictionaries.enabled`         | bool   | `true`           | Write dictionary files             |

**Compression options:**

| Codec    | Speed   | Ratio | Use Case                         |
| -------- | ------- | ----- | -------------------------------- |
| `none`   | Fastest | 1.0×  | Temporary/debugging              |
| `snappy` | Fast    | ~2×   | Training logs (frequent writes)  |
| `zstd`   | Medium  | ~4×   | Simulation outputs (recommended) |
| `gzip`   | Slow    | ~3.5× | Archival/compatibility           |

## 5. Production Scale Reference

Reference sizes for production-scale SDDP runs (Brazilian interconnected system scale).

### 5.1 Typical Problem Dimensions

| Dimension       | Small | Medium | Large | Extra Large |
| --------------- | ----- | ------ | ----- | ----------- |
| Stages          | 60    | 120    | 360   | 600         |
| Hydros          | 50    | 160    | 200   | 250         |
| Thermals        | 100   | 200    | 300   | 400         |
| Buses           | 4     | 5      | 8     | 12          |
| Scenarios (sim) | 200   | 2,000  | 5,000 | 10,000      |
| Iterations      | 50    | 100    | 200   | 500         |
| Forward passes  | 4     | 8      | 16    | 32          |
| MPI ranks       | 16    | 128    | 512   | 2,048       |

### 5.2 Output Size Estimates

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

### 5.3 I/O Bandwidth Requirements

| Scale       | Write Throughput | Duration | Bottleneck  |
| ----------- | ---------------- | -------- | ----------- |
| Small       | 50 MB/s          | 20s      | None        |
| Medium      | 200 MB/s         | 100s     | Network     |
| Large       | 500 MB/s         | 200s     | Filesystem  |
| Extra Large | 1+ GB/s          | 500s     | Parallel FS |

## 6. Validation and Integrity

### 6.1 Schema Validation

```bash
# Validate simulation output schema
powers validate-output --type simulation --path ./simulation/

# Validate training output schema
powers validate-output --type training --path ./training/

# Validate specific entity
powers validate-output --type simulation --entity hydros --path ./simulation/hydros/
```

### 6.2 Data Integrity Checks

| Check                  | Method                      | Frequency |
| ---------------------- | --------------------------- | --------- |
| Parquet file integrity | Footer checksum             | On read   |
| Partition completeness | Manifest comparison         | Post-run  |
| Row count consistency  | Cross-entity validation     | Post-run  |
| Value range validation | Min/max from bounds.parquet | Optional  |

**Cross-entity validation:**

```python
def validate_scenario(scenario_id: int) -> bool:
    costs = pq.read_table(f"simulation/costs/scenario_id={scenario_id}/")
    hydros = pq.read_table(f"simulation/hydros/scenario_id={scenario_id}/")

    expected_stages = costs.num_rows
    return validate_row_counts(costs, hydros, expected_stages)
```

### 6.3 Reproducibility Verification

The `data_integrity` section in `metadata.json` enables reproducibility verification:

```bash
# Verify inputs haven't changed since training
powers verify-inputs --metadata training/metadata.json --input-dir ./input/

# Compare two training runs
powers diff-runs --run1 ./training_v1/ --run2 ./training_v2/
```

**Hash computation:**

- `input_hash`: SHA-256 of concatenated input file hashes
- `config_hash`: SHA-256 of normalized config.json
- `policy_hash`: SHA-256 of policy/cuts.parquet content
- `convergence_hash`: SHA-256 of training/convergence.parquet content

## Cross-References

- [Output Schemas](output-schemas.md) — Parquet column definitions for all entity types
- [Input System Entities](input-system-entities.md) — Entity registries (entity IDs, names)
- [Penalty System](penalty-system.md) — Penalty costs affecting output values
- [Configuration Reference](../05-config/configuration-reference.md) — Full config.json reference
- [Design Principles](../00-overview/design-principles.md) — Overall design philosophy
