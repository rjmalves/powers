---
status: approved
review_priority: 3-medium
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §1 (1.1-1.4)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §2 (2.1-2.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §3 (3.1-3.2)"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: "Approved. Resource allocations are read-only from environment (not config.json). Graceful shutdown checkpoints last completed iteration. SLURM-first scheduler scope."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §1-3"
  - date: 2026-02-22
    description: "Review rewrite: fix priority to P3, strip Rust code from scheduler integration, replace config JSON with cross-reference, remove speculative phase durations, add graceful shutdown behavior, clarify SLURM-first scheduler scope, add diagram placeholder note."
  - date: 2026-02-22
    description: "Restructured §6: separated resource allocations (read-only from environment) from algorithm parameters (config.json hierarchy). Removed POWERS_* env vars. Added phase×mode matrix to §5.3. Fixed §7 shutdown to checkpoint last completed iteration (not current)."
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

| Argument          | Required | Description                                                  |
| ----------------- | -------- | ------------------------------------------------------------ |
| `CASE_DIR`        | Yes      | Path to case directory containing `config.json`              |
| `--validate-only` | No       | Run Startup and Validation phases only, then exit (see §5.3) |
| `--version`       | No       | Print version and exit                                       |
| `--help`          | No       | Print usage and exit                                         |

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

> **Placeholder** — The execution phases diagram (`../../diagrams/exports/svg/hpc/execution-phases.svg`) will be revised after the text review is complete.

### 5.2 Phase Responsibilities

| Phase          | MPI Ranks      | Key Operations                                  |
| -------------- | -------------- | ----------------------------------------------- |
| Startup        | All            | MPI init, scheduler detection, CLI parsing      |
| Validation     | Rank 0 only    | Load files, schema validation, cross-references |
| Initialization | All            | Broadcast, memory allocation, solver setup      |
| Scenario Gen   | All (parallel) | PAR fitting, noise sampling, correlation        |
| Training       | All (parallel) | SDDP iterations                                 |
| Simulation     | All (parallel) | Policy evaluation                               |
| Finalize       | All            | Output writing, cleanup                         |

### 5.3 Conditional Execution

The execution flow supports several modes. Which phases execute depends on the mode:

| Phase          | Full Run | Training Only | Simulation Only | Validation Only |
| -------------- | :------: | :-----------: | :-------------: | :-------------: |
| Startup        |   Yes    |      Yes      |       Yes       |       Yes       |
| Validation     |   Yes    |      Yes      |       Yes       |       Yes       |
| Initialization |   Yes    |      Yes      |       Yes       |        —        |
| Scenario Gen   |   Yes    |      Yes      |       Yes       |        —        |
| Training       |   Yes    |      Yes      |        —        |        —        |
| Simulation     |   Yes    |       —       |       Yes       |        —        |
| Finalize       |   Yes    |      Yes      |       Yes       |        —        |

**Mode selection:**

- **Full Run** — Default. Both training and simulation execute sequentially.
- **Training Only** — Produces a policy (cuts) without evaluating it. Useful for convergence analysis or when simulation will be run separately.
- **Simulation Only** — Evaluates an existing policy. Requires a `policy/` directory from a prior training run. Scenario generation still executes because simulation forward passes need scenario realizations.
- **Validation Only** — Validates all input files and configuration, then exits immediately after the Validation phase. No memory allocation, no solver setup, no outputs. Triggered by `--validate-only` on the command line (overrides config settings) or by disabling both `training.enabled` and `simulation.enabled` in `config.json`.

Training Only and Simulation Only are controlled by the `training.enabled` and `simulation.enabled` fields in `config.json`. See [Configuration Reference](../05-config/configuration-reference.md).

## 6. Configuration Resolution

There are two distinct categories of runtime settings, and they follow different resolution rules:

### 6.1 Resource Allocations (read-only from environment)

Resource allocations are determined by the MPI launcher and job scheduler. The program reads them from the environment and must not override them. These values are never sourced from `config.json` or compiled defaults.

| Parameter       | Source                                                 | Description                   |
| --------------- | ------------------------------------------------------ | ----------------------------- |
| MPI rank count  | MPI launcher (`mpiexec -n`, `srun`)                    | Number of processes           |
| CPUs per task   | Scheduler (`SLURM_CPUS_PER_TASK`) or `OMP_NUM_THREADS` | Threads per rank              |
| Memory per node | Scheduler (`SLURM_MEM_PER_NODE`)                       | Memory budget for pool sizing |
| Job ID          | Scheduler (`SLURM_JOB_ID`)                             | Recorded in output metadata   |

**Rationale:** Allowing config.json to override resource allocations would create dangerous mismatches — e.g., the program spawning 8 threads on a node where SLURM allocated 2 CPUs, causing oversubscription. Resource allocations are a contract between the job scheduler and the process; the program observes them, it does not negotiate.

If `OMP_NUM_THREADS` is not set and no scheduler is detected, the program defaults to 1 thread per rank.

### 6.2 Algorithm Parameters (config hierarchy)

Algorithm parameters (tolerances, buffer sizes, stopping rules, etc.) are resolved in priority order:

1. **`config.json`** — Explicit user configuration. See [Configuration Reference](../05-config/configuration-reference.md)
2. **Compiled defaults** — Internal constants for any parameter not specified in `config.json`

The resolved configuration is recorded in the training metadata file for reproducibility (see [Output Infrastructure](../02-data-model/output-infrastructure.md) §2).

### 6.3 Scheduler Detection

POWE.RS detects the job scheduler environment at startup to read resource allocations.

**Supported schedulers:**

| Scheduler  | Detection                      | Initial Support |
| ---------- | ------------------------------ | --------------- |
| SLURM      | `SLURM_JOB_ID` environment var | Yes             |
| PBS/Torque | `PBS_JOBID` environment var    | Future          |
| LSF        | `LSB_JOBID` environment var    | Future          |
| Local      | No scheduler env vars detected | Yes (fallback)  |

If no scheduler is detected, the program falls back to local defaults: 1 thread per rank, no memory budget constraint.

> **Scope note:** SLURM is the primary target scheduler. PBS and LSF support is planned for future releases and listed here for completeness. The detection mechanism is the same (environment variable probing), so adding new schedulers is straightforward.

## 7. Signal Handling and Graceful Shutdown

POWE.RS installs signal handlers to support graceful shutdown during long-running training and simulation phases.

| Signal    | Behavior                                                                              |
| --------- | ------------------------------------------------------------------------------------- |
| `SIGTERM` | Graceful shutdown: set shutdown flag, checkpoint last completed iteration, exit       |
| `SIGINT`  | Same as `SIGTERM` — checkpoint last completed iteration, exit with code 130           |
| `SIGKILL` | Immediate termination (cannot be caught). Recovery via crash protocol on next startup |

**Graceful shutdown protocol:**

1. The signal handler sets a global shutdown flag.
2. The program does **not** wait for the current iteration to finish — iterations at production scale can take minutes, and SIGTERM is expected to result in a fast exit.
3. A checkpoint is written from the **last fully completed iteration's** policy state. This state is always consistent and ready to serialize.
4. All MPI ranks coordinate shutdown via a barrier before finalization.
5. The training manifest is updated with `status: "partial"` and the last completed iteration number.

This ensures that a `SIGTERM` from SLURM (e.g., approaching wall-time limit) results in a prompt shutdown without corrupting policy or output files. The next invocation can detect the partial state via the manifest and resume from the checkpoint. See [Output Infrastructure](../02-data-model/output-infrastructure.md) §1.2 for manifest status values.

## Cross-References

- [Configuration Reference](../05-config/configuration-reference.md) — Complete `config.json` schema and parameter documentation
- [Input Loading Pipeline](./input-loading-pipeline.md) — How input files are loaded after CLI parsing and config resolution
- [Validation Architecture](./validation-architecture.md) — Multi-layer validation executed during the Validation phase
- [Design Principles](../00-overview/design-principles.md) — Format selection and declaration order invariance governing input processing
- [Production Scale Reference](../00-overview/production-scale-reference.md) — Typical phase durations and resource requirements at production scale
- [Output Infrastructure](../02-data-model/output-infrastructure.md) — Manifest files, metadata, crash recovery protocol
- [SLURM Deployment](../04-hpc/slurm-deployment.md) — Job scripts and multi-node deployment patterns
