---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md Appendix A.1 (Single-Node Job Script)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md Appendix A.2 (Multi-Node Production Job)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md Appendix A.3 (Job Array for Parameter Studies)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md Appendix B.1 (Key Performance Counters)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md Appendix B.2 (Timing Breakdown)"
  - "DATA_MODEL_SPECIFICATION.md §6.7 (NUMA-Aware Memory Management — SLURM Template)"
  - "DATA_MODEL_SPECIFICATION.md §6.9 (HPC Implementation Requirements)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from ARCHITECTURE Appendix A/B and DATA_MODEL §6.7/§6.9"
---

# SLURM Deployment

## Purpose

This spec defines SLURM job scripts, deployment patterns, and performance monitoring for POWE.RS on HPC clusters. It covers single-node development jobs, multi-node production runs with NUMA settings, parameter study sweeps via job arrays, performance counters, and timing expectations. It also captures the critical HPC implementation requirements from the data model specification.

## Shell → Rust Boundary

> **Important**: SLURM scripts are shell scripts that configure the job environment — they are preserved as-is. The Rust binary launched by `srun` uses `ferrompi` to detect placement:
>
> - **SLURM scripts** → set `--ntasks`, `--cpus-per-task`, `--mem-bind`, bind policies, module loads
> - **Rust startup** → calls `ferrompi::init_with_threading(Multiple)` to initialize MPI with thread support, and `ferrompi::slurm::local_rank()` to read SLURM topology variables (`SLURM_LOCALID`, `SLURM_CPUS_PER_TASK`, etc.) without manual `std::env::var` parsing
>
> See [Hybrid Parallelism](./hybrid-parallelism.md) §5 for the full initialization sequence and [Hybrid Parallelism](./hybrid-parallelism.md) §3 for `ParallelConfig::from_environment()` which delegates to `ferrompi::slurm` helpers.

## 1. Single-Node Job (Development/Testing)

For development, debugging, and small-scale testing on a single node with multiple MPI ranks sharing memory.

```bash
#!/bin/bash
#SBATCH --job-name=powers-test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --output=powers_%j.log

# Load modules
module load openmpi/4.1.5
module load rust/1.75

# Set OpenMP threads from SLURM allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Run POWE.RS
srun powers /path/to/case_directory
```

**Notes:**

- 8 ranks × 24 threads = 192 cores (full dual-socket EPYC node)
- `OMP_NUM_THREADS` is derived from `SLURM_CPUS_PER_TASK` — never hardcoded
- `debug` partition typically has shorter queue wait and 1-hour wall time limits
- The Rust binary reads `SLURM_CPUS_PER_TASK` via `ferrompi::slurm::cpus_per_task()` during `ParallelConfig::from_environment()`

## 2. Multi-Node Production Job

Full production configuration with NUMA memory binding, MPI tuning, and checkpoint signal handling.

```bash
#!/bin/bash
#SBATCH --job-name=powers-production
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=powers_%j.log
#SBATCH --error=powers_%j.err

# Load modules
module load openmpi/4.1.5
module load rust/1.75

# NUMA and memory settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MALLOC_MMAP_THRESHOLD_=1000000

# MPI settings for large jobs
export OMPI_MCA_btl_tcp_endpoint_cache=0
export OMPI_MCA_mpi_yield_when_idle=1

# Run with checkpoint signal handling
srun --signal=TERM@60 powers /scratch/user/case_directory

# On preemption, SIGTERM is sent 60s before kill
```

**Notes:**

- 8 nodes × 8 ranks/node × 24 threads/rank = 1,536 cores total
- `--signal=TERM@60` sends `SIGTERM` 60 seconds before SLURM kills the job, enabling graceful checkpoint writes
- `MALLOC_MMAP_THRESHOLD_` forces large allocations through `mmap` for NUMA-friendly placement
- `OMPI_MCA_mpi_yield_when_idle=1` reduces CPU waste on idle MPI ranks

## 3. NUMA-Optimized Template (Production Best Practice)

This template from the data model specification is optimized for high-NUMA-count systems (e.g., AWS c7a.48xlarge with 8 NUMA nodes) and uses 1 MPI rank per node with all cores available to OpenMP.

```bash
#!/bin/bash
#===============================================================================
# POWE.RS SDDP Solver - SLURM Job Script Template
# Optimized for hybrid MPI+OpenMP on NUMA systems
#===============================================================================

#SBATCH --job-name=powers-sddp
#SBATCH --output=powers-%j.out
#SBATCH --error=powers-%j.err

#===============================================================================
# RESOURCE ALLOCATION
#===============================================================================
#SBATCH --nodes=4                      # Number of compute nodes
#SBATCH --ntasks-per-node=1            # One MPI rank per node (recommended)
#SBATCH --cpus-per-task=192            # All cores for OpenMP threads
#SBATCH --mem=0                        # All available memory per node
#SBATCH --exclusive                    # Exclusive node access
#SBATCH --time=24:00:00                # Maximum runtime

#===============================================================================
# PARTITION (site-specific)
#===============================================================================
#SBATCH --partition=compute
#SBATCH --account=my_project

#===============================================================================
# ENVIRONMENT SETUP
#===============================================================================

module purge
module load openmpi/4.1.5

#===============================================================================
# OPENMP CONFIGURATION
# CRITICAL: Use SLURM's computed value - DO NOT hardcode!
#===============================================================================

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close             # Keep threads close for NUMA locality
export OMP_PLACES=cores                # One thread per physical core
export OMP_STACKSIZE=64M               # Stack for deep recursion

#===============================================================================
# JOB INFORMATION
#===============================================================================

echo "==============================================="
echo "POWE.RS SDDP Job Information"
echo "==============================================="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Nodes:            ${SLURM_JOB_NUM_NODES}"
echo "Tasks/Node:       ${SLURM_NTASKS_PER_NODE}"
echo "CPUs/Task:        ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS:  ${OMP_NUM_THREADS}"
echo "Memory/Node:      ${SLURM_MEM_PER_NODE:-all} MB"
echo "Node List:        ${SLURM_JOB_NODELIST}"
echo "==============================================="

#===============================================================================
# RUN APPLICATION (config uses "auto" - will read SLURM vars)
#===============================================================================

CASE_DIR="${1:-./case}"

srun --cpu-bind=verbose \
    --distribution=block:block \
    ./powers train --config "${CASE_DIR}/config.json"

exit $?
```

**Design choices:**

| Choice                       | Rationale                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `--ntasks-per-node=1`        | Single rank per node; OpenMP handles intra-node parallelism via shared memory |
| `--cpus-per-task=192`        | All cores available to OpenMP threads                                         |
| `--mem=0` + `--exclusive`    | Full node memory for shared FCF allocation (up to 26+ GB)                     |
| `--cpu-bind=verbose`         | Logs binding decisions for debugging NUMA placement                           |
| `--distribution=block:block` | Keeps rank-to-node mapping predictable                                        |
| `OMP_STACKSIZE=64M`          | Required for deep LP solver recursion stacks                                  |

**PBS/Torque equivalent:**

```bash
#!/bin/bash
#PBS -N powers-sddp
#PBS -l nodes=4:ppn=48
#PBS -l mem=512gb
#PBS -l walltime=24:00:00

export OMP_NUM_THREADS=${PBS_NUM_PPN}
cd ${PBS_O_WORKDIR}
mpirun -np $(cat ${PBS_NODEFILE} | wc -l) \
    -hostfile ${PBS_NODEFILE} \
    ./powers train --config case/config.json
```

## 4. Job Array for Parameter Studies

Parameterized job submission for sweeping forward scenario counts or other configuration values.

```bash
#!/bin/bash
#SBATCH --job-name=powers-sweep
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=powers_%A_%a.log

# Parameter values indexed by array task
SCENARIOS=(100 200 500 1000 2000 5000 10000 20000 50000 100000)
N_SCENARIOS=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

# Create case directory with modified config
CASE_DIR=/scratch/user/sweep_${N_SCENARIOS}
cp -r /home/user/base_case $CASE_DIR

# Modify config.json
jq ".training.forward_scenarios = ${N_SCENARIOS}" \
    $CASE_DIR/config.json > $CASE_DIR/config_new.json
mv $CASE_DIR/config_new.json $CASE_DIR/config.json

# Run
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun powers $CASE_DIR
```

**Notes:**

- `%A` = parent array job ID, `%a` = array task index — produces unique log files per sweep point
- Each array element runs independently with its own SLURM allocation
- `jq` modifies `config.json` in-place for each sweep point
- Sweep covers 3 orders of magnitude (100 → 100,000 scenarios)

## 5. Performance Monitoring

### 5.1 Key Performance Counters

```rust
/// Performance counters collected during execution
pub struct PerformanceCounters {
    // LP solver metrics
    pub lp_solves: AtomicU64,
    pub lp_iterations_total: AtomicU64,
    pub lp_time_ns: AtomicU64,

    // Communication metrics
    pub mpi_sends: AtomicU64,
    pub mpi_bytes_sent: AtomicU64,
    pub mpi_time_ns: AtomicU64,

    // Memory metrics
    pub allocations: AtomicU64,
    pub bytes_allocated: AtomicU64,
    pub peak_memory_bytes: AtomicU64,

    // Cache metrics (requires perf counters)
    pub l1_cache_misses: AtomicU64,
    pub llc_cache_misses: AtomicU64,
}

impl PerformanceCounters {
    pub fn summary(&self) -> PerformanceSummary {
        let lp_solves = self.lp_solves.load(Ordering::Relaxed);
        let lp_time_s = self.lp_time_ns.load(Ordering::Relaxed) as f64 / 1e9;

        PerformanceSummary {
            total_lp_solves: lp_solves,
            avg_lp_time_ms: lp_time_s * 1000.0 / lp_solves as f64,
            avg_lp_iterations: self.lp_iterations_total.load(Ordering::Relaxed) as f64
                / lp_solves as f64,
            mpi_overhead_percent: self.mpi_time_ns.load(Ordering::Relaxed) as f64
                / (lp_time_s * 1e9) * 100.0,
            peak_memory_gb: self.peak_memory_bytes.load(Ordering::Relaxed) as f64 / 1e9,
        }
    }
}
```

**Counter categories:**

| Category      | Counters                                              | Diagnostic Use                                    |
| ------------- | ----------------------------------------------------- | ------------------------------------------------- |
| LP solver     | `lp_solves`, `lp_iterations_total`, `lp_time_ns`      | Compute efficiency, warm-start effectiveness      |
| Communication | `mpi_sends`, `mpi_bytes_sent`, `mpi_time_ns`          | Communication overhead (target: <5% of total)     |
| Memory        | `allocations`, `bytes_allocated`, `peak_memory_bytes` | Memory pressure, leak detection                   |
| Cache         | `l1_cache_misses`, `llc_cache_misses`                 | NUMA placement quality (requires `perf` counters) |

### 5.2 Timing Breakdown

Expected per-iteration timing targets for a production configuration (8 nodes, 64 ranks):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Iteration Timing Breakdown (Target)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  SDDP Iteration (~30 seconds total)                                             │
│  ═══════════════════════════════════                                            │
│                                                                                  │
│  Forward Pass:                          20.0s (66.7%)                           │
│  ├── LP solves:                         18.5s (92.5%)                           │
│  ├── State transitions:                  0.8s (4.0%)                            │
│  ├── Noise sampling:                     0.2s (1.0%)                            │
│  └── Result collection:                  0.5s (2.5%)                            │
│                                                                                  │
│  Forward Sync (Allreduce):               0.5s (1.7%)                            │
│                                                                                  │
│  Backward Pass:                          8.0s (26.7%)                           │
│  ├── LP solves:                          7.2s (90.0%)                           │
│  ├── Cut computation:                    0.5s (6.3%)                            │
│  └── Local aggregation:                  0.3s (3.7%)                            │
│                                                                                  │
│  Backward Sync (Allgatherv):             1.0s (3.3%)                            │
│  ├── Serialization:                      0.3s                                   │
│  ├── MPI communication:                  0.5s                                   │
│  └── Deserialization:                    0.2s                                   │
│                                                                                  │
│  Convergence check + logging:            0.5s (1.7%)                            │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│  Efficiency metrics:                                                            │
│  - Compute/Communication ratio: ~18:1 (excellent)                               │
│  - Parallel efficiency (8 ranks): ~92%                                          │
│  - Thread utilization: ~85%                                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key efficiency targets:**

| Metric                      | Target        | Action if Missed                                   |
| --------------------------- | ------------- | -------------------------------------------------- |
| Compute/Communication ratio | ≥ 10:1        | Increase batch size, reduce sync frequency         |
| Parallel efficiency         | ≥ 85%         | Profile load imbalance, check NUMA binding         |
| Thread utilization          | ≥ 80%         | Verify `schedule(dynamic,1)`, check for contention |
| MPI overhead                | < 5% of total | Use non-blocking collectives, persistent comms     |

Additional per-rank metrics are written to `training/timing/mpi_ranks.parquet`:

| Metric                  | Description                           | Diagnostic Use                          |
| ----------------------- | ------------------------------------- | --------------------------------------- |
| `computation_time_ms`   | Time in LP solves and cut computation | Baseline work                           |
| `communication_time_ms` | Time in MPI calls                     | Communication overhead                  |
| `idle_time_ms`          | Time waiting at barriers              | Load imbalance                          |
| `gather_time_ms`        | Time in cut gather phase              | Aggregation bottleneck                  |
| `bcast_time_ms`         | Time in FCF broadcast                 | Distribution overhead                   |
| `memory_high_water_mb`  | Peak RSS during iteration             | Memory pressure                         |
| `numa_local_ratio`      | Fraction of local NUMA accesses       | Memory placement quality (target: >90%) |

## 6. HPC Implementation Requirements

These requirements are critical for correct and performant operation on HPC systems.

### 6.1 Thread-Safe Cut Slot Management

With full preallocation and deterministic slot assignment, cut slots are **computed, not allocated** at runtime:

```rust
// Deterministic slot computation - pure function, no mutation
slot = warm_start_count + iteration * forward_passes + forward_pass_idx
```

| Operation             | When                      | Thread-Safety                        |
| --------------------- | ------------------------- | ------------------------------------ |
| LP construction       | Startup (single-threaded) | N/A                                  |
| Warm-start loading    | Startup (single-threaded) | N/A                                  |
| Slot computation      | Runtime (parallel)        | Safe: pure function, no state        |
| Cut coefficient write | Runtime (parallel)        | Safe: threads write different rows   |
| Bound toggle          | Runtime (parallel)        | Safe: threads toggle different rows  |
| Bitmap update         | After parallel section    | Safe: single-threaded or batch merge |

No atomic operations or locks are needed during parallel execution.

### 6.2 NUMA-Aware FCF Allocation

> **⚠️ CRITICAL**: Standard `MPI_Win_allocate_shared` allocates the entire 18.6 GB FCF on a **single NUMA node**, causing 3x latency penalty for threads on remote NUMA nodes.

The fix is NUMA-distributed allocation with round-robin stage assignment across NUMA nodes. Each partition is allocated with first-touch initialization on the target NUMA node.

**Deployment requirements:**

- Use `libnuma` bindings for NUMA node binding
- Configure SLURM with `--mem-bind=local` when available
- Monitor `numa_local_ratio` metric (target: >90%)

### 6.3 False Sharing Prevention

Adjacent cuts sharing cache lines cause false sharing when threads write `domination_count` concurrently. The fix is **thread-local accumulation** with a post-parallel merge:

```rust
/// Thread-local workspace for cut evaluation (cache-line padded)
pub struct CutEvaluationWorkspace {
    /// Indexed by: local_counts[thread_id][cut_index]
    local_counts: Vec<Vec<u32>>,
    cut_values: Vec<f64>,
    _padding: [u8; 64],
}
```

Each thread accumulates into its own buffer during the parallel region; a single thread merges after the barrier.

### 6.4 Load Balancing Strategy

Work-stealing operates at two levels:

| Level                   | Mechanism                    | Latency                 | Implementation              |
| ----------------------- | ---------------------------- | ----------------------- | --------------------------- |
| **Intra-rank** (OpenMP) | `schedule(dynamic,1)`        | ~100 ns (shared memory) | Built into OpenMP runtime   |
| **Inter-rank** (MPI)    | Dynamic dispatch from rank 0 | ~10–100 μs (network)    | Dedicated dispatcher thread |

Cross-rank work-stealing was rejected: the overhead of serializing LP state across MPI exceeds any benefit for 1–10 second SDDP iterations.

### 6.5 Asynchronous Checkpointing

Synchronous checkpoint I/O can block all ranks at large scale. The required approach is double-buffered async checkpointing: fill buffer N while writing buffer N−1, with ZSTD level 1–3 compression.

### 6.6 Issue Priority Summary

| Issue                           | Severity     | Impact                              |
| ------------------------------- | ------------ | ----------------------------------- |
| NUMA FCF allocation             | **CRITICAL** | 3x latency penalty on EPYC systems  |
| False sharing in cut evaluation | HIGH         | 10–20% perf degradation             |
| Work-stealing load balance      | HIGH         | 15–25% efficiency improvement       |
| Async checkpointing             | MEDIUM       | Required at scale (>100 iterations) |

## Cross-References

- [Hybrid Parallelism](./hybrid-parallelism.md) — MPI+OpenMP initialization, `ferrompi` usage, OpenMP FFI, build configuration
- [Work Distribution](./work-distribution.md) — forward/backward pass distribution, dynamic dispatch protocol, rank 0 bottleneck mitigation
