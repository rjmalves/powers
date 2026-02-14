---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §4 (4.1-4.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §5 (5.1-5.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §7 (7.1-7.4)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Input Loading Pipeline

## Purpose

This spec defines the POWE.RS input loading architecture: the rank-0 centric loading pattern, file loading sequence with dependency ordering, the loader interface, dependency resolution, conditional loading rules, sparse time-series expansion, data broadcasting strategy, serialization for MPI broadcast, parallel policy loading for warm-start, and the memory layout after broadcast completes.

## 1. Loading Architecture

Input loading follows a **rank-0 centric** pattern: the master rank loads and validates all input data, then broadcasts to worker ranks. This design:

- Minimizes filesystem contention on parallel filesystems
- Centralizes validation logic
- Reduces complexity of error handling across ranks

![Input Loading Pipeline](../../diagrams/exports/svg/data/input-loading-pipeline.svg)

## 2. File Loading Sequence

Files are loaded in dependency order to enable early validation:

| Order | File(s)                           | Dependencies          | Validation                   |
| ----- | --------------------------------- | --------------------- | ---------------------------- |
| 1     | `config.json`                     | None                  | Schema, execution mode       |
| 2     | `stages.json`                     | config (horizon mode) | Stage count, transitions     |
| 3     | `penalties.json`                  | None                  | Penalty values > 0           |
| 4     | `initial_conditions.json`         | None                  | Entity references (deferred) |
| 5     | `system/buses.json`               | None                  | Bus IDs unique               |
| 6     | `system/lines.json`               | buses                 | Source/target bus refs       |
| 7     | `system/hydros.json`              | buses                 | Bus refs, cascade refs       |
| 8     | `system/thermals.json`            | buses                 | Bus refs                     |
| 9     | `scenarios/inflow_models.parquet` | hydros, stages        | Hydro/stage coverage         |
| 10    | `scenarios/correlation.json`      | hydros                | Block membership             |
| 11    | `constraints/*.parquet`           | entities, stages      | Entity/stage refs            |
| 12    | `policy/*` (if warm-start)        | All above             | State dictionary match       |

## 3. Loader Interface

```rust
/// Main input loader - orchestrates the loading pipeline
pub struct InputLoader {
    case_dir: PathBuf,
    config: Option<Config>,
    validation_results: ValidationResult,
}

impl InputLoader {
    /// Load all inputs with validation
    /// Returns fully validated and canonicalized CaseData
    pub fn load(&mut self) -> Result<CaseData, LoadError> {
        // Phase 1: Load configuration
        self.config = Some(self.load_config()?);

        // Phase 2: Load stages (depends on horizon mode)
        let stages = self.load_stages()?;

        // Phase 3: Load system entities
        let system = self.load_system()?;

        // Phase 4: Load scenario data
        let scenarios = self.load_scenarios(&system, &stages)?;

        // Phase 5: Load constraints
        let constraints = self.load_constraints(&system, &stages)?;

        // Phase 6: Load policy (if warm-start)
        let policy = self.load_policy_if_needed(&system)?;

        // Phase 7: Final validation and canonicalization
        let case_data = CaseData {
            config: self.config.take().unwrap(),
            stages,
            system,
            scenarios,
            constraints,
            policy,
        };

        case_data.canonicalize();
        self.validate_cross_references(&case_data)?;

        Ok(case_data)
    }
}
```

## 4. Dependency Graph

Input files form a directed acyclic graph (DAG) of dependencies:

![JSON Schema Dependencies](../../diagrams/exports/svg/data/json-schema-dependencies.svg)

## 5. Conditional Loading

Some files are loaded conditionally based on configuration:

| Condition                            | Files Affected                   |
| ------------------------------------ | -------------------------------- |
| `training.enabled = false`           | Skip scenario noise generation   |
| `simulation.enabled = false`         | Skip simulation scenario loading |
| `policy.mode = "warm_start"`         | Load `policy/*` files            |
| `horizon.mode = "infinite_periodic"` | Validate cycle in stages         |
| Hydros with `fpha_enabled = true`    | Load `fpha_hyperplanes.parquet`  |
| Hydros with pumping                  | Load `pumping_stations.json`     |

## 6. Sparse Time-Series Handling

Time-series Parquet files use **sparse representation**: only non-default values are stored. The loader must:

1. Load sparse data from Parquet
2. Expand to dense representation using defaults
3. Validate stage coverage

```rust
/// Sparse-to-dense expansion for time series
pub fn expand_bounds<T: Default + Clone>(
    sparse: &[(StageId, EntityId, T)],
    stages: &[Stage],
    entities: &[EntityId],
    default: T,
) -> Vec<Vec<T>> {
    // Initialize with defaults
    let mut dense = vec![vec![default.clone(); entities.len()]; stages.len()];

    // Overlay sparse values
    for (stage_id, entity_id, value) in sparse {
        let stage_idx = stages.iter().position(|s| s.id == *stage_id)
            .expect("Invalid stage_id");
        let entity_idx = entities.iter().position(|e| *e == *entity_id)
            .expect("Invalid entity_id");
        dense[stage_idx][entity_idx] = value.clone();
    }

    dense
}
```

## 7. Broadcast Strategy

After rank 0 loads and validates all data, it must be distributed to workers. The strategy depends on data size and structure:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Data Broadcasting Strategy                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Data Type              │ Size     │ Strategy                                   │
│  ───────────────────────┼──────────┼────────────────────────────────────────    │
│  Config                 │ <10 KB   │ MPI_Bcast (serialized JSON)               │
│  Stages                 │ <100 KB  │ MPI_Bcast (serialized)                    │
│  System (entities)      │ 1-10 MB  │ MPI_Bcast (binary serialized)             │
│  PAR models             │ 10-50 MB │ MPI_Bcast (packed arrays)                 │
│  Correlation matrices   │ 1-5 MB   │ MPI_Bcast (dense matrix)                  │
│  Time-series bounds     │ 10-50 MB │ MPI_Bcast (sparse then expand locally)    │
│  FCF cuts (warm-start)  │ 1-20 GB  │ Parallel load (each rank loads subset)    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 8. Serialization for Broadcast

Data is serialized to contiguous byte buffers for efficient MPI broadcast:

```rust
/// Trait for MPI-broadcastable data
pub trait MpiBroadcast: Sized {
    /// Serialize to bytes for broadcast
    fn to_broadcast_bytes(&self) -> Vec<u8>;

    /// Deserialize from broadcast bytes
    fn from_broadcast_bytes(bytes: &[u8]) -> Self;
}

/// Broadcast wrapper handling size negotiation
pub fn broadcast_data<T: MpiBroadcast>(
    comm: &impl Communicator,
    data: Option<T>,  // Some on rank 0, None on workers
    root: Rank,
) -> T {
    let rank = comm.rank();

    if rank == root {
        let data = data.expect("Root must provide data");
        let bytes = data.to_broadcast_bytes();

        // First broadcast: size
        let size = bytes.len() as u64;
        comm.broadcast(&size, root);

        // Second broadcast: data
        comm.broadcast(&bytes, root);

        data
    } else {
        // Receive size
        let mut size: u64 = 0;
        comm.broadcast(&mut size, root);

        // Receive data
        let mut bytes = vec![0u8; size as usize];
        comm.broadcast(&mut bytes, root);

        T::from_broadcast_bytes(&bytes)
    }
}
```

## 9. Parallel Policy Loading (Warm-Start)

For large policy files (cuts), parallel loading improves startup time:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Parallel Policy Loading Pattern                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  120 stages, 8 ranks:                                                           │
│                                                                                  │
│  Rank 0: Load stages [0, 8, 16, 24, ...]    (15 stages)                        │
│  Rank 1: Load stages [1, 9, 17, 25, ...]    (15 stages)                        │
│  Rank 2: Load stages [2, 10, 18, 26, ...]   (15 stages)                        │
│  ...                                                                             │
│  Rank 7: Load stages [7, 15, 23, 31, ...]   (15 stages)                        │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ After local loading:                                                    │    │
│  │                                                                         │    │
│  │   MPI_Allgatherv to collect all cuts on all ranks                      │    │
│  │   OR                                                                    │    │
│  │   Use shared memory window (intra-node) + MPI_Bcast (inter-node)       │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Time comparison (20 GB policy, 200 MB/s parallel FS):                         │
│  Sequential: 20 GB / 200 MB/s = 100s                                           │
│  Parallel (8 ranks): 2.5 GB / 200 MB/s + sync overhead ≈ 15s                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 10. Memory Layout After Broadcast

After broadcasting, each rank has identical copies of:

```rust
/// Complete case data available on all ranks after initialization
pub struct DistributedCaseData {
    // Replicated on all ranks (small, read-only)
    pub config: Arc<Config>,
    pub stages: Arc<Vec<Stage>>,
    pub system: Arc<System>,

    // Replicated on all ranks (medium, read-only)
    pub par_models: Arc<ParModels>,
    pub correlation: Arc<CorrelationData>,

    // Shared within node (large, uses MPI shared memory window)
    pub fcf: SharedFcf,  // Future Cost Function cuts

    // Per-rank (scenario-specific data)
    pub my_scenarios: Vec<ScenarioId>,
    pub noise_samples: Vec<NoiseSample>,
}
```

## Cross-References

- [CLI and Lifecycle](./cli-and-lifecycle.md) — Program entrypoint, execution phases, and configuration resolution that precede input loading
- [Validation Architecture](./validation-architecture.md) — Multi-layer validation applied during and after loading
- [Design Principles](../00-overview/design-principles.md) — Declaration order invariance requiring canonicalization after loading
- [Configuration Reference](../05-config/configuration-reference.md) — Complete `config.json` schema loaded in phase 1
- [PAR Inflow Model](../01-math/par-inflow-model.md) — PAR(p) model parameters loaded from `inflow_models.parquet`
- [Cut Management](../01-math/cut-management.md) — Policy file structure loaded during warm-start
