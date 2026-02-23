---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §25.1 (Checkpoint Strategy)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §25.2 (Checkpoint Implementation)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §25.3 (Warm-Start from Checkpoint)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §25.5 (Policy Persistence Architecture)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §26.1 (Output Directory Structure)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §26.2 (Policy Output)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §26.3 (Simulation Summary Output)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §26.4 (Performance Logging)"
last_reviewed: null
reviewed_by: null
review_notes: "REVIEW NOTE (from block-formulations.md approval): Checkpoint format must include policy metadata (block modes, system dimensions, AR orders, etc.) for compatibility validation on resume. See Deferred Features §C.9."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from architecture §25 (25.1-25.3, 25.5) and §26 (26.1-26.4)"
---

# Checkpointing and Output Generation

## Purpose

This spec defines how POWE.RS persists training state for fault tolerance (checkpointing), saves trained policies for warm-start and simulation, and generates structured output including simulation summaries and performance logs. It covers the complete lifecycle from mid-training checkpoint to final output.

## 1. Checkpoint Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Checkpointing Strategy                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Goals:                                                                          │
│  - Resume training after job preemption or failure                              │
│  - Support long-running jobs (hours to days)                                    │
│  - Minimize checkpoint overhead (< 5% of iteration time)                        │
│  - Enable warm-start from previous runs                                         │
│                                                                                  │
│  Checkpoint Contents:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  1. FCF cuts (primary data, largest component)                          │   │
│  │     - All cuts for all stages                                            │   │
│  │     - Activity counters for cut selection                                │   │
│  │                                                                           │   │
│  │  2. Training state                                                        │   │
│  │     - Current iteration number                                            │   │
│  │     - Convergence monitor history (bounds, gaps)                         │   │
│  │     - RNG state for reproducibility                                       │   │
│  │                                                                           │   │
│  │  3. Configuration snapshot                                                │   │
│  │     - Hash of config.json for compatibility check                        │   │
│  │     - Timestamp                                                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Checkpoint Schedule:                                                           │
│  - Every N iterations (configurable, default 10)                               │
│  - On SIGTERM (graceful shutdown from scheduler)                               │
│  - On convergence (final checkpoint)                                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Checkpoint Implementation

Checkpoints are written by rank 0 only. Workers synchronize via MPI barrier. The checkpoint file uses a binary format with a magic header for integrity detection.

```rust
/// Checkpoint manager for training state persistence
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    interval: usize,
    last_checkpoint: usize,

    /// Signal handler state
    shutdown_requested: Arc<AtomicBool>,
}

impl CheckpointManager {
    pub fn new(case_dir: &Path, interval: usize) -> Self {
        let checkpoint_dir = case_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir).ok();

        let shutdown_requested = Arc::new(AtomicBool::new(false));

        // Register signal handler for graceful shutdown
        let shutdown_flag = shutdown_requested.clone();
        ctrlc::set_handler(move || {
            shutdown_flag.store(true, Ordering::SeqCst);
        }).ok();

        Self {
            checkpoint_dir,
            interval,
            last_checkpoint: 0,
            shutdown_requested,
        }
    }

    /// Check if checkpoint is needed
    pub fn should_checkpoint(&self, iteration: usize) -> bool {
        if iteration - self.last_checkpoint >= self.interval {
            return true;
        }
        if self.shutdown_requested.load(Ordering::SeqCst) {
            return true;
        }
        false
    }

    /// Write checkpoint (rank 0 only)
    pub fn write_checkpoint(
        &mut self,
        iteration: usize,
        fcf: &FutureCostFunction,
        monitor: &ConvergenceMonitor,
        comm: &Communicator,
    ) -> io::Result<()> {
        if comm.rank() != 0 {
            comm.barrier();
            return Ok(());
        }

        let checkpoint_path = self.checkpoint_dir.join(format!(
            "checkpoint_{:06}.bin", iteration
        ));

        let mut file = BufWriter::new(File::create(&checkpoint_path)?);

        // Header
        file.write_all(b"PWRSCHK\0")?;                     // Magic
        file.write_all(&1u32.to_le_bytes())?;               // Version
        file.write_all(&(iteration as u64).to_le_bytes())?; // Iteration
        file.write_all(&SystemTime::now()
            .duration_since(UNIX_EPOCH).unwrap()
            .as_secs().to_le_bytes())?;                     // Timestamp

        // FCF cuts
        self.write_fcf(&mut file, fcf)?;

        // Convergence monitor
        self.write_monitor(&mut file, monitor)?;

        file.flush()?;

        // Update symlink to latest
        let latest_link = self.checkpoint_dir.join("latest");
        std::fs::remove_file(&latest_link).ok();
        std::os::unix::fs::symlink(&checkpoint_path, &latest_link)?;

        // Cleanup old checkpoints (keep last 3)
        self.cleanup_old_checkpoints(3)?;

        self.last_checkpoint = iteration;
        comm.barrier();
        Ok(())
    }

    /// Load latest checkpoint if available
    pub fn load_latest(&self) -> Option<Checkpoint> {
        let latest_link = self.checkpoint_dir.join("latest");
        if !latest_link.exists() {
            return None;
        }
        let checkpoint_path = std::fs::read_link(&latest_link).ok()?;
        self.load_checkpoint(&checkpoint_path).ok()
    }
}

/// Loaded checkpoint data
pub struct Checkpoint {
    pub iteration: usize,
    pub timestamp: SystemTime,
    pub fcf: FutureCostFunction,
    pub monitor_history: ConvergenceHistory,
}
```

**Checkpoint file format:**

| Field        | Size    | Description                 |
| ------------ | ------- | --------------------------- |
| Magic        | 8 bytes | `PWRSCHK\0`                 |
| Version      | 4 bytes | Format version (u32 LE)     |
| Iteration    | 8 bytes | Current iteration (u64 LE)  |
| Timestamp    | 8 bytes | Unix epoch seconds (u64 LE) |
| FCF data     | varies  | Serialized cuts by stage    |
| Monitor data | varies  | Convergence history         |

**Retention policy:** Only the 3 most recent checkpoints are kept. Older checkpoints are automatically deleted after a successful write.

## 3. Warm-Start from Checkpoint

```rust
impl<R: RiskMeasure, C: CutFormulation, H: HorizonMode> TrainingLoop<R, C, H> {
    /// Initialize training with optional warm-start
    pub fn initialize(&mut self, case_data: &CaseData) -> io::Result<()> {
        let checkpoint_manager = CheckpointManager::new(
            &case_data.case_dir,
            self.config.checkpoint_interval,
        );

        if let Some(checkpoint) = checkpoint_manager.load_latest() {
            // Validate compatibility
            if checkpoint.fcf.state_dimension() != case_data.state_dimension() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Checkpoint state dimension mismatch",
                ));
            }

            // Restore state
            self.fcf = checkpoint.fcf;
            self.iteration = checkpoint.iteration;
            self.convergence_monitor.restore_history(checkpoint.monitor_history);

            if self.comm.rank() == 0 {
                println!(
                    "Resumed from checkpoint at iteration {} ({:?})",
                    self.iteration, checkpoint.timestamp,
                );
            }
        } else {
            // Cold start: initialize empty FCF
            self.fcf = FutureCostFunction::new(
                case_data.num_stages(),
                case_data.num_hydros(),
                case_data.par_models.max_order(),
            );
            self.iteration = 0;

            if self.comm.rank() == 0 {
                println!("Starting fresh training (no checkpoint found)");
            }
        }

        Ok(())
    }
}
```

## 4. Policy Persistence Architecture

A POWE.RS policy must be **fully self-contained** for warm-start, checkpointing, and simulation-only runs. This means persisting not just the cuts, but all information needed to reconstruct the exact optimization problem.

### 4.1 Policy Components

| Component              | Contents                                                | Size (Reference Case)         |
| ---------------------- | ------------------------------------------------------- | ----------------------------- |
| **Cuts (FCF)**         | α intercept + β gradient per cut per stage, metadata    | 120 stages × 10K cuts ≈ 20 GB |
| **PAR Coefficients**   | μ, ψ, σ, order per hydro per season                     | ~270 KB                       |
| **State Variable Map** | Canonical ordering: storage volumes then AR lag inflows | ~50 KB                        |
| **Policy Metadata**    | Version, iterations, gap, dimensions, checksum          | ~2 KB                         |

> ⚠️ **AR ORDER MISMATCH IS A FATAL ERROR**: The state dimension includes AR lags. Different orders = different cut dimensions = incompatible policy.

### 4.2 File Format

```
policy/
├── metadata.json           # Policy metadata + state mapping
├── par_models.json         # Complete PAR model coefficients
└── cuts/
    ├── stage_000.bin       # Binary cut data for stage 0
    ├── stage_001.bin       # Binary cut data for stage 1
    └── ...
```

**metadata.json:**

```json
{
  "version": "1.0",
  "training_iterations": 150,
  "convergence_gap": 0.0023,
  "state_dimension": 2080,
  "n_stages": 120,
  "n_hydros": 160,
  "max_ar_order": 12,
  "timestamp": "2026-01-31T14:30:00Z",
  "checksum": "sha256:a1b2c3...",
  "state_mapping": {
    "storage_variables": ["ITAIPU", "TUCURUI", "XINGO"],
    "ar_orders": { "ITAIPU": 12, "TUCURUI": 12, "XINGO": 10 }
  }
}
```

### 4.3 Compatibility Validation

Loading a policy requires strict compatibility checking:

```rust
/// Policy compatibility validation
pub struct PolicyValidator {
    tolerance: f64,
}

impl PolicyValidator {
    /// Validate policy against current case data
    pub fn validate(
        &self,
        policy: &Policy,
        case: &CaseData,
    ) -> Result<(), PolicyError> {
        // 1. State dimension match
        let expected_dim = case.compute_state_dimension();
        if policy.state_dimension != expected_dim {
            return Err(PolicyError::DimensionMismatch {
                policy_dim: policy.state_dimension,
                case_dim: expected_dim,
                detail: "State dimension mismatch - likely AR order difference".into(),
            });
        }

        // 2. Hydro count and IDs
        if policy.n_hydros != case.hydros.len() {
            return Err(PolicyError::HydroCountMismatch {
                policy: policy.n_hydros,
                case: case.hydros.len(),
            });
        }

        // 3. AR order consistency per hydro
        for (hydro_id, policy_order) in &policy.ar_orders {
            let case_order = case.par_models.order(hydro_id)
                .ok_or_else(|| PolicyError::UnknownHydro(hydro_id.clone()))?;
            if policy_order != &case_order {
                return Err(PolicyError::ArOrderMismatch {
                    hydro: hydro_id.clone(),
                    policy_order: *policy_order,
                    case_order,
                });
            }
        }

        // 4. PAR coefficients consistency (with tolerance)
        for (hydro_id, policy_par) in &policy.par_models {
            if let Some(case_par) = case.par_models.get(hydro_id) {
                let max_diff = policy_par.max_coefficient_diff(case_par);
                if max_diff > self.tolerance {
                    return Err(PolicyError::ParCoefficientMismatch {
                        hydro: hydro_id.clone(),
                        max_diff,
                        tolerance: self.tolerance,
                    });
                }
            }
        }

        // 5. Stage count
        if policy.n_stages != case.stages.len() {
            return Err(PolicyError::StageCountMismatch {
                policy: policy.n_stages,
                case: case.stages.len(),
            });
        }

        Ok(())
    }
}
```

### 4.4 Use Cases

| Scenario                | What's Loaded     | Validation                          |
| ----------------------- | ----------------- | ----------------------------------- |
| **Warm-start training** | Cuts + PAR models | Full validation, continue training  |
| **Simulation-only**     | Cuts + PAR models | Full validation, skip training      |
| **Checkpoint recovery** | Full state        | Same case required                  |
| **Policy transfer**     | Cuts only         | Dimension check only (experimental) |

> **CRITICAL WARNING**: Using a policy with mismatched PAR coefficients will produce subtly incorrect results. The cuts were computed with specific AR dynamics embedded in the LP constraints. Different AR coefficients mean different RHS values for the same state, leading to incorrect cost-to-go estimates, suboptimal decisions, and potential infeasibility.

## 5. Output Directory Structure

![Output Streaming Pipeline](../../PROGRAM_ARCHITECTURE_EXECUTION_FLOW/diagrams/exports/svg/data/output-streaming-pipeline.svg)

The output directory follows a structured layout separating policy, training artifacts, and simulation results. See [Design Principles](../00-overview/design-principles.md) for the distributed I/O rationale.

## 6. Policy Output

```rust
/// Policy output writer
pub struct PolicyWriter {
    output_dir: PathBuf,
}

impl PolicyWriter {
    /// Write trained policy to disk
    pub fn write_policy(
        &self,
        fcf: &FutureCostFunction,
        case_data: &CaseData,
        training_result: &TrainingResult,
    ) -> io::Result<PathBuf> {
        let policy_dir = self.output_dir.join("policy");
        std::fs::create_dir_all(&policy_dir)?;

        // Write metadata
        let metadata = PolicyMetadata {
            version: "1.0".to_string(),
            created: Utc::now(),
            case_name: case_data.name.clone(),
            n_stages: case_data.num_stages(),
            n_hydros: case_data.num_hydros(),
            state_variables: fcf.state_dictionary(),
            training_iterations: training_result.iterations,
            final_gap: training_result.gap,
            lower_bound: training_result.lower_bound,
            upper_bound: training_result.upper_bound,
        };

        let metadata_path = policy_dir.join("metadata.json");
        std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

        // Write cuts by stage
        let cuts_dir = policy_dir.join("cuts");
        std::fs::create_dir_all(&cuts_dir)?;
        for (stage_idx, pool) in fcf.cuts_by_stage.iter().enumerate() {
            let stage_path = cuts_dir.join(format!("stage_{:03}.bin", stage_idx));
            save_stage_cuts(&stage_path, pool.active_cuts())?;
        }

        // Write convergence history
        let convergence_path = policy_dir.join("convergence.json");
        std::fs::write(&convergence_path,
            serde_json::to_string_pretty(&ConvergenceReport::from(training_result))?)?;

        Ok(policy_dir)
    }
}
```

## 7. Simulation Summary Output

```rust
/// Simulation summary output
#[derive(Serialize)]
pub struct SimulationSummary {
    pub timestamp: DateTime<Utc>,
    pub n_scenarios: usize,
    pub scenario_source: String,
    pub cost: CostStatistics,
    pub operations: OperationalStatistics,
    pub risk: RiskMetrics,
}

#[derive(Serialize)]
pub struct CostStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub percentiles: HashMap<String, f64>,  // p5, p25, p75, p95
}

#[derive(Serialize)]
pub struct OperationalStatistics {
    pub deficit_probability: f64,
    pub deficit_mean_mwh: f64,
    pub deficit_max_mwh: f64,
    pub spill_mean_mwh: f64,
    pub thermal_generation_mean_mwh: f64,
    pub hydro_generation_mean_mwh: f64,
}

#[derive(Serialize)]
pub struct RiskMetrics {
    pub var_95: f64,   // Value at Risk (95%)
    pub cvar_95: f64,  // Conditional VaR (95%)
    pub var_99: f64,
    pub cvar_99: f64,
}
```

The summary is written to `simulation/summary.json` by calling `summary.write(&output_dir)`. It captures cost distributions, operational statistics (deficit probability, generation mix), and risk metrics (VaR, CVaR at 95th and 99th percentiles).

## 8. Performance Logging

```rust
/// Performance metrics collector
pub struct PerformanceLogger {
    metrics: Mutex<PerformanceMetrics>,
    start_time: Instant,
}

#[derive(Serialize, Default)]
pub struct PerformanceMetrics {
    pub total_runtime_seconds: f64,

    // Phase timings
    pub initialization_seconds: f64,
    pub training_seconds: f64,
    pub simulation_seconds: f64,

    // Training metrics
    pub training_iterations: usize,
    pub forward_passes: usize,
    pub backward_passes: usize,
    pub lp_solves: usize,
    pub cuts_generated: usize,

    // Parallel efficiency
    pub mpi_ranks: usize,
    pub threads_per_rank: usize,
    pub total_cores: usize,
    pub communication_overhead_percent: f64,

    // Memory
    pub peak_memory_gb: f64,
    pub fcf_memory_mb: f64,

    // I/O
    pub checkpoint_writes: usize,
    pub checkpoint_bytes_written: u64,
    pub output_bytes_written: u64,
}
```

Performance metrics are written to `logs/performance.json` at the end of execution. The logger records iteration-level metrics via `record_iteration()` and computes totals including parallel efficiency and I/O overhead.

## Cross-References

- [Memory Architecture](./memory-architecture.md) — memory budget, NUMA allocation, and pool design that checkpoints must serialize
- [Hybrid Parallelism](./hybrid-parallelism.md) — MPI rank 0 responsibility for checkpoint writes and barrier synchronization
- [Work Distribution](./work-distribution.md) — forward/backward pass distribution that feeds convergence monitoring
- [Design Principles](../00-overview/design-principles.md) — distributed I/O and reproducibility goals
- [SDDP Algorithm](../01-math/sddp-algorithm.md) — convergence criteria that trigger final checkpoint
- [Scenario Generation §2.3](../03-architecture/scenario-generation.md) — The opening tree is deterministically regenerable from the random seed, so it does NOT need explicit persistence in checkpoints. On resume, the system regenerates the same opening tree from the persisted seed.
