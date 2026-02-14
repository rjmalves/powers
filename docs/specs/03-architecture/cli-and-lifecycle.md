---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §1 (1.1-1.4)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §2 (2.1-2.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §3 (3.1-3.2)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# CLI and Lifecycle

## Purpose

This spec defines the POWE.RS program entrypoint, command-line interface, exit codes, execution phase lifecycle, conditional execution modes, configuration resolution hierarchy, and job scheduler integration. It covers everything from process invocation through phase orchestration to shutdown.

## 1. Design Philosophy

POWE.RS adopts a **single-entrypoint design** optimized for HPC batch execution. The program is always invoked via MPI launchers (`mpiexec`, `mpirun`, or SLURM's `srun`) and all runtime behavior is controlled through configuration files rather than command-line arguments.

**Rationale**:

- HPC job scripts benefit from stable command-line interfaces
- Configuration files provide auditability and reproducibility
- Complex nested options are better expressed in JSON than CLI flags
- Reduces parsing complexity in the hot initialization path

## 2. Invocation Pattern

```bash
# Standard invocation
mpiexec -n 8 powers /path/to/case_directory

# SLURM batch execution
srun powers /path/to/case_directory

# Validation-only mode
mpiexec -n 1 powers /path/to/case_directory --validate-only
```

## 3. Command-Line Interface

| Argument          | Required | Description                                     |
| ----------------- | -------- | ----------------------------------------------- |
| `CASE_DIR`        | Yes      | Path to case directory containing `config.json` |
| `--validate-only` | No       | Validate inputs and exit without execution      |
| `--version`       | No       | Print version and exit                          |
| `--help`          | No       | Print usage and exit                            |

**Design Decision**: All execution options (skip training, skip simulation, warm-start mode, etc.) are specified in `config.json`, not via CLI flags. This ensures:

1. Job scripts remain stable across configuration changes
2. Configuration is self-documenting and version-controlled
3. No ambiguity between CLI and config file settings

## 4. Exit Codes

| Code | Meaning                                   |
| ---- | ----------------------------------------- |
| 0    | Success                                   |
| 1    | Invalid command-line arguments            |
| 2    | Configuration validation error            |
| 3    | Input data validation error               |
| 4    | Runtime error (solver failure, MPI error) |
| 5    | Checkpoint recovery failed                |
| 130  | Interrupted (SIGINT)                      |
| 137  | Killed (SIGKILL, typically OOM)           |

## 5. Execution Phases Overview

### 5.1 Phase Diagram

![Execution Phases](../../diagrams/exports/svg/hpc/execution-phases.svg)

### 5.2 Phase Responsibilities

| Phase          | MPI Ranks      | Duration | Key Operations                                  |
| -------------- | -------------- | -------- | ----------------------------------------------- |
| Startup        | All            | <100ms   | MPI init, scheduler detection, CLI parsing      |
| Validation     | Rank 0 only    | 1-10s    | Load files, schema validation, cross-references |
| Initialization | All            | 1-5s     | Broadcast, memory allocation, solver setup      |
| Scenario Gen   | All (parallel) | 1-30s    | PAR fitting, noise sampling, correlation        |
| Training       | All (parallel) | 10min-2h | SDDP iterations                                 |
| Simulation     | All (parallel) | 1-30min  | Policy evaluation                               |
| Finalize       | All            | 1-10s    | Output writing, cleanup                         |

### 5.3 Conditional Execution

The execution flow supports several modes controlled by `config.json`:

| Mode            | Training | Simulation | Use Case                                   |
| --------------- | -------- | ---------- | ------------------------------------------ |
| Full Run        | Yes      | Yes        | Standard production run                    |
| Training Only   | Yes      | No         | Policy development, convergence analysis   |
| Simulation Only | No       | Yes        | Policy evaluation with existing cuts       |
| Validation Only | No       | No         | Input verification before batch submission |

```json
{
  "training": { "enabled": true },
  "simulation": { "enabled": true }
}
```

## 6. Configuration Resolution and Validation

### 6.1 Configuration Hierarchy

Configuration values are resolved in priority order (highest to lowest):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Configuration Resolution Priority                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  1. Environment Variables (highest priority)                                    │
│     └── SLURM_CPUS_PER_TASK, OMP_NUM_THREADS, POWERS_* variables               │
│                                                                                  │
│  2. Job Scheduler Detection                                                      │
│     └── SLURM, PBS, LSF environment → thread counts, memory limits             │
│                                                                                  │
│  3. config.json (explicit user configuration)                                   │
│     └── All algorithm parameters, execution options                             │
│                                                                                  │
│  4. Compiled Defaults (lowest priority)                                         │
│     └── Tolerances, buffer sizes, internal constants                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Scheduler Integration

POWE.RS automatically detects the job scheduler environment and respects its resource allocations:

```rust
/// Scheduler detection and configuration extraction
pub struct SchedulerConfig {
    pub scheduler_type: SchedulerType,
    pub cpus_per_task: Option<u32>,
    pub memory_per_node_mb: Option<u64>,
    pub job_id: Option<String>,
}

pub enum SchedulerType {
    Slurm,
    Pbs,
    Lsf,
    Local,  // No scheduler detected
}

impl SchedulerConfig {
    pub fn detect() -> Self {
        if std::env::var("SLURM_JOB_ID").is_ok() {
            Self::from_slurm()
        } else if std::env::var("PBS_JOBID").is_ok() {
            Self::from_pbs()
        } else if std::env::var("LSB_JOBID").is_ok() {
            Self::from_lsf()
        } else {
            Self::local_defaults()
        }
    }
}
```

## Cross-References

- [Configuration Reference](../05-config/configuration-reference.md) — Complete `config.json` schema and parameter documentation
- [Input Loading Pipeline](./input-loading-pipeline.md) — How input files are loaded after CLI parsing and config resolution
- [Validation Architecture](./validation-architecture.md) — Multi-layer validation executed during the Validation phase
- [Design Principles](../00-overview/design-principles.md) — Format selection and declaration order invariance governing input processing
- [Production Scale Reference](../00-overview/production-scale-reference.md) — Typical phase durations and resource requirements at production scale
