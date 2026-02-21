---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.1 (Hybrid Parallelism Overview)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.2 (Design Rationale)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.3 (Parallel Configuration)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.4 (OpenMP FFI Bindings)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.5 (Initialization Sequence)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.6 (Build Configuration)"
  - "DATA_MODEL_SPECIFICATION.md §6.1 (Hybrid MPI+OpenMP Architecture Overview)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): Thread-trajectory affinity is the primary parallelization pattern — each thread owns a complete forward trajectory and the corresponding backward pass. Forward pass state save/restore (solver basis, scenario state) needed when M forward passes > N threads, analogous to CPU context switching but only at stage boundaries. These observations should be validated and detailed during P4 review. See sddp-algorithm.md §3.4."
---

# Hybrid Parallelism

## Purpose

This spec defines the hybrid MPI+OpenMP parallelization strategy used by POWE.RS: why native OpenMP is chosen over Rayon, how MPI ranks and OpenMP threads are configured, the OpenMP C FFI bindings for Rust, the parallel initialization sequence using `ferrompi`, and the build configuration for compiling the OpenMP wrapper.

## 1. Hybrid Parallelism Overview

POWE.RS employs a hybrid MPI+OpenMP parallelization strategy optimized for modern HPC architectures with multi-socket, many-core nodes. **Native OpenMP is used via FFI** (not Rayon) to leverage vendor-optimized runtimes (Intel, AMD, GCC) and provide direct control over scheduling, affinity, and synchronization.

MPI communication uses the `ferrompi` crate, which provides safe, idiomatic Rust bindings. The `ferrompi::Communicator` type is `Send + Sync` by design, enabling hybrid MPI+threads without unsafe sharing of raw `MPI_Comm` handles.

| Component                       | Responsibility                                                        |
| ------------------------------- | --------------------------------------------------------------------- |
| **MPI (Inter-Node/Inter-NUMA)** | Distributes work across NUMA nodes, handles cut aggregation           |
| **OpenMP (Intra-NUMA)**         | Parallelizes forward passes within each rank, leverages shared memory |
| **Shared Memory (MPI Windows)** | Scenarios and cuts shared within node, not replicated                 |

> This hybrid approach provides the load balancing benefits of NEWAVE's dynamic dispatch while avoiding its memory replication bottleneck.

**Shared Memory Contents:**

| Region           | Size    | Access Pattern                                           |
| ---------------- | ------- | -------------------------------------------------------- |
| Scenario Storage | 7.68 GB | 1000 passes x 120 stages x 20 branches, read by any rank |
| Cut Storage      | 18.6 GB | Preallocated slots, written by rank 0, read by all       |

**Per-Rank Resources:**

- OpenMP threads: 16 per rank
- LP solver instances: 1 per thread
- Local memory: ~240 MB per rank

**Memory Comparison:**

| Approach                  | Memory Required                    |
| ------------------------- | ---------------------------------- |
| NEWAVE-style (replicated) | 16 ranks x 26.3 GB = **294 GB**    |
| POWE.RS hybrid (shared)   | 26.3 GB + 4 x 240 MB = **27.3 GB** |
| **Reduction**             | **~11x**                           |

## 2. Design Rationale

**Why Native OpenMP (not Rayon)?**

| Criterion                  | Native OpenMP via FFI                | Rayon                 |
| -------------------------- | ------------------------------------ | --------------------- |
| **Vendor optimization**    | Full (Intel, AMD, GCC runtimes)      | Limited (generic)     |
| **Scheduling control**     | `static`, `dynamic`, `guided`        | Work-stealing only    |
| **Affinity control**       | `OMP_PLACES`, `OMP_PROC_BIND`        | None                  |
| **NUMA awareness**         | First-touch, explicit placement      | Opaque                |
| **Reduction primitives**   | Hardware-optimized tree reduction    | Manual implementation |
| **HPC ecosystem**          | Standard (SLURM, modules, profilers) | Limited integration   |
| **LP solver coordination** | Explicit single-thread forcing       | Potential conflicts   |

For SDDP's compute pattern (many small LP solves with shared data), the ability to control scheduling and affinity directly translates to 15-25% better performance on NUMA systems.

| Aspect            | MPI Ranks                                     | OpenMP Threads                        |
| ----------------- | --------------------------------------------- | ------------------------------------- |
| **Purpose**       | Distributed memory, inter-node communication  | Shared memory, intra-node parallelism |
| **Granularity**   | Coarse: scenario batches                      | Fine: individual LP solves            |
| **Communication** | Explicit: cuts, bounds, statistics            | Implicit: shared FCF, case data       |
| **Memory**        | Replicated (cut data) or shared (MPI windows) | Shared (read-only case data)          |
| **Load Balance**  | Static distribution (scenarios)               | Dynamic scheduling within rank        |
| **Scheduling**    | N/A                                           | `schedule(dynamic,1)` for LP solves   |

## 3. Parallel Configuration

```rust
pub struct ParallelConfig {
    pub mpi: MpiConfig,
    pub openmp: OpenMpConfig,
    pub numa: NumaConfig,
}

pub struct MpiConfig {
    /// Expected number of ranks (validated at startup)
    pub expected_ranks: Option<usize>,

    /// Use shared memory windows for intra-node FCF
    pub use_shared_memory: bool,

    /// Communication backend hints
    pub eager_limit: Option<usize>,
}

pub struct OpenMpConfig {
    /// Threads per rank (auto-detected from SLURM/OMP_NUM_THREADS if not set)
    pub threads_per_rank: Option<usize>,

    /// Thread affinity strategy
    pub affinity: ThreadAffinity,

    /// Wait policy for idle threads
    pub wait_policy: WaitPolicy,

    /// Stack size per thread (needed for deep LP solver recursion)
    pub stack_size_mb: usize,
}

pub enum ThreadAffinity {
    /// Bind threads to cores within NUMA domain (best for compute-bound)
    Close,
    /// Spread threads across cores (better memory bandwidth)
    Spread,
    /// No explicit binding (scheduler decides)
    None,
}

pub enum WaitPolicy {
    /// Spin-wait (lowest latency, highest power)
    Active,
    /// Sleep (higher latency, lower power - recommended for I/O phases)
    Passive,
}

pub struct NumaConfig {
    /// Enable first-touch memory placement
    pub first_touch: bool,

    /// Prefer local memory allocation
    pub local_alloc: bool,

    /// Interleave large arrays across NUMA domains
    pub interleave_large_arrays: bool,
}

impl ParallelConfig {
    /// Detect configuration from environment (SLURM, PBS, or manual)
    pub fn from_environment() -> Self {
        let scheduler = SchedulerConfig::detect();
        let threads = scheduler.cpus_per_task
            .map(|c| c as usize)
            .unwrap_or_else(|| {
                std::env::var("OMP_NUM_THREADS").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(1)
            });
        Self {
            mpi: MpiConfig {
                expected_ranks: None, use_shared_memory: true,
                eager_limit: Some(256 * 1024),
            },
            openmp: OpenMpConfig {
                threads_per_rank: Some(threads), affinity: ThreadAffinity::Close,
                wait_policy: WaitPolicy::Passive, stack_size_mb: 64,
            },
            numa: NumaConfig {
                first_touch: true, local_alloc: true,
                interleave_large_arrays: false,
            },
        }
    }
}
```

## 4. OpenMP FFI Bindings

POWE.RS uses a C wrapper to access OpenMP parallel regions from Rust, since OpenMP pragmas require compiler support unavailable in rustc. Note that **OpenMP bindings remain as C FFI** — the `ferrompi` crate covers MPI only.

**Core FFI Bindings (openmp_ffi.rs):**

```rust
//! OpenMP FFI bindings for Rust
//!
//! Provides safe wrappers around OpenMP runtime functions via C FFI.
//! Uses native OpenMP for maximum HPC performance.

use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, Ordering};

static OPENMP_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[link(name = "omp")]
extern "C" {
    fn omp_get_num_threads() -> c_int;
    fn omp_get_max_threads() -> c_int;
    fn omp_get_thread_num() -> c_int;
    fn omp_set_num_threads(num_threads: c_int);
    fn omp_get_num_procs() -> c_int;
    fn omp_in_parallel() -> c_int;
    fn omp_set_dynamic(dynamic_threads: c_int);
    fn omp_get_wtime() -> f64;
}

#[link(name = "openmp_wrapper", kind = "static")]
extern "C" {
    fn omp_parallel_for_dynamic(
        start: c_int,
        end: c_int,
        chunk_size: c_int,
        callback: extern "C" fn(idx: c_int, thread_id: c_int, user_data: *mut std::ffi::c_void),
        user_data: *mut std::ffi::c_void,
    );

    fn omp_parallel_reduce_sum(
        start: c_int,
        end: c_int,
        callback: extern "C" fn(idx: c_int, thread_id: c_int, user_data: *mut std::ffi::c_void) -> f64,
        user_data: *mut std::ffi::c_void,
    ) -> f64;
}

/// Safe Rust wrappers for OpenMP functions
pub mod omp {
    use super::*;

    /// Initialize OpenMP environment
    pub fn init(num_threads: usize) {
        if OPENMP_INITIALIZED.swap(true, Ordering::SeqCst) {
            return;
        }
        unsafe {
            omp_set_dynamic(0);  // Disable dynamic adjustment
            omp_set_num_threads(num_threads as c_int);
        }
    }

    #[inline]
    pub fn get_num_threads() -> usize {
        unsafe { omp_get_num_threads() as usize }
    }

    #[inline]
    pub fn get_thread_num() -> usize {
        unsafe { omp_get_thread_num() as usize }
    }

    #[inline]
    pub fn get_wtime() -> f64 {
        unsafe { omp_get_wtime() }
    }
}

/// Thread-local storage with cache-line alignment (prevents false sharing)
#[repr(C, align(64))]
pub struct ThreadLocal<T> {
    data: T,
    _padding: [u8; 64 - std::mem::size_of::<T>() % 64],
}

/// Parallel for loop with dynamic scheduling (trampoline pattern for C FFI)
pub fn parallel_for_dynamic<F>(range: std::ops::Range<usize>, chunk_size: usize, f: F)
where
    F: Fn(usize, usize) + Sync,  // (index, thread_id)
{
    // Uses extern "C" trampoline function to bridge Rust closure to C callback.
    // The closure is passed as *mut c_void user_data, cast back in the trampoline.
    let data = CallbackData { callback: f };
    unsafe {
        omp_parallel_for_dynamic(
            range.start as c_int, range.end as c_int, chunk_size as c_int,
            trampoline::<F>, &data as *const _ as *mut std::ffi::c_void,
        );
    }
}
```

**C Wrapper (openmp_wrapper.c):**

```c
/* OpenMP C wrapper for Rust FFI — Compile: gcc -c -fopenmp -O3 -march=native openmp_wrapper.c */
#include <omp.h>
#include <stdint.h>

typedef void (*parallel_callback)(int32_t idx, int32_t thread_id, void* user_data);
typedef double (*reduce_callback)(int32_t idx, int32_t thread_id, void* user_data);

/* Parallel for with dynamic scheduling (primary pattern for LP solves) */
void omp_parallel_for_dynamic(
    int32_t start, int32_t end, int32_t chunk_size,
    parallel_callback callback, void* user_data
) {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, chunk_size) nowait
        for (int32_t i = start; i < end; i++) {
            callback(i, thread_id, user_data);
        }
    }
}

/* Parallel reduction with sum */
double omp_parallel_reduce_sum(
    int32_t start, int32_t end,
    reduce_callback callback, void* user_data
) {
    double total_sum = 0.0;
    #pragma omp parallel reduction(+:total_sum)
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (int32_t i = start; i < end; i++) {
            total_sum += callback(i, thread_id, user_data);
        }
    }
    return total_sum;
}

/* NUMA-aware first-touch initialization */
void omp_parallel_first_touch_f64(double* array, int64_t size, double init_value) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk = size / nthreads;
        int64_t start = tid * chunk;
        int64_t end = (tid == nthreads - 1) ? size : start + chunk;
        for (int64_t i = start; i < end; i++) {
            array[i] = init_value;
        }
    }
}
```

## 5. Initialization Sequence

The parallel environment is initialized by first calling `ferrompi::init_with_threading` for MPI, then configuring OpenMP via FFI. The `ferrompi::Communicator` is `Send + Sync` by design, which is exactly what enables hybrid MPI+threads without unsafe sharing of raw `MPI_Comm` handles.

```rust
/// Initialize parallel environment (ferrompi + OpenMP FFI)
pub fn init_parallel(config: &ParallelConfig) -> ParallelContext {
    // 1. Initialize MPI with thread support via ferrompi
    //    ferrompi::Communicator is Send + Sync — safe for hybrid MPI+threads
    let universe = ferrompi::init_with_threading(ferrompi::ThreadLevel::Multiple)
        .expect("MPI initialization failed");
    let world = universe.world();

    let rank = world.rank();
    let size = world.size();

    // 2. Validate rank count if specified
    if let Some(expected) = config.mpi.expected_ranks {
        if size != expected && rank == 0 {
            eprintln!("Warning: Expected {} ranks, got {}", expected, size);
        }
    }

    // 3. Initialize OpenMP via C FFI (not covered by ferrompi)
    let threads = config.openmp.threads_per_rank.unwrap_or(1);
    omp::init(threads);

    // 4. Set OpenMP environment variables for affinity
    match config.openmp.affinity {
        ThreadAffinity::Close => {
            std::env::set_var("OMP_PROC_BIND", "close");
            std::env::set_var("OMP_PLACES", "cores");
        }
        ThreadAffinity::Spread => {
            std::env::set_var("OMP_PROC_BIND", "spread");
            std::env::set_var("OMP_PLACES", "cores");
        }
        ThreadAffinity::None => {}
    }

    // 5. Set wait policy and stack size
    match config.openmp.wait_policy {
        WaitPolicy::Active => std::env::set_var("OMP_WAIT_POLICY", "active"),
        WaitPolicy::Passive => std::env::set_var("OMP_WAIT_POLICY", "passive"),
    }
    std::env::set_var("OMP_STACKSIZE", format!("{}M", config.openmp.stack_size_mb));

    // 6. Force single-threaded LP solver (outer parallelism handles it)
    std::env::set_var("HIGHS_PARALLEL", "false");
    std::env::set_var("MKL_NUM_THREADS", "1");

    // 7. Setup NUMA allocation policy
    #[cfg(target_os = "linux")]
    if config.numa.local_alloc {
        unsafe { libc::numa_set_localalloc(); }
    }

    // 8. Create shared memory communicator (ranks on same node)
    let shared_comm = world.split_shared_memory();

    if rank == 0 {
        println!("Parallel initialized: {} ranks x {} threads = {} cores",
            size, threads, size * threads);
    }

    ParallelContext { world, shared_comm, rank, size, threads }
}
```

## 6. Build Configuration

The build script compiles the OpenMP C wrapper and links it. MPI is handled by the `ferrompi` Cargo dependency — no manual MPI link flags are needed.

**Cargo.toml dependency:**

```toml
[dependencies]
ferrompi = { version = "0.2", features = ["rma", "numa"] }
```

**Build script (build.rs):**

```rust
fn main() {
    println!("cargo:rerun-if-changed=src/parallel/openmp_wrapper.c");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let (cc, omp_flag, omp_lib) = detect_openmp_config();

    // Compile C wrapper with OpenMP and archive into static library
    let status = std::process::Command::new(&cc)
        .args(&["-c", &omp_flag, "-O3", "-march=native", "-fPIC",
                "src/parallel/openmp_wrapper.c", "-o"])
        .arg(format!("{}/openmp_wrapper.o", out_dir))
        .status().expect("Failed to compile OpenMP wrapper");
    assert!(status.success(), "OpenMP wrapper compilation failed");

    std::process::Command::new("ar")
        .args(&["rcs", &format!("{}/libopenmp_wrapper.a", out_dir),
                &format!("{}/openmp_wrapper.o", out_dir)])
        .status().expect("Failed to create static library");

    // Link directives (OpenMP only — MPI handled by ferrompi crate)
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=openmp_wrapper");
    println!("cargo:rustc-link-lib={}", omp_lib);
}

fn detect_openmp_config() -> (String, String, String) {
    if check_compiler("icx", "-qopenmp") {       // Intel oneAPI (preferred for HPC)
        return ("icx".into(), "-qopenmp".into(), "iomp5".into());
    }
    if check_compiler("gcc", "-fopenmp") {        // GCC
        return ("gcc".into(), "-fopenmp".into(), "gomp".into());
    }
    if check_compiler("clang", "-fopenmp") {      // Clang/LLVM
        return ("clang".into(), "-fopenmp".into(), "omp".into());
    }
    panic!("No OpenMP-capable compiler found");
}
```

## 7. Communication and Design Summary

**Forward Pass** — Dynamic dispatch from rank 0 (no synchronization, optimal load balancing).

**Backward Pass** — Two-level reduction per stage (t = T-1 down to 1):

| Level      | Operation                            | Result                                             |
| ---------- | ------------------------------------ | -------------------------------------------------- |
| 1 - OpenMP | Thread-local reduction within rank   | local_alpha, local_beta per rank                   |
| 2 - MPI    | `comm.reduce()` + `comm.broadcast()` | All ranks have identical global_alpha, global_beta |

Synchronization: MPI reduce + broadcast per stage (implicit barrier). Deterministic order for reproducibility.

| Aspect                     | Decision                             | Rationale                                    |
| -------------------------- | ------------------------------------ | -------------------------------------------- |
| **Intra-rank parallelism** | OpenMP (not Rayon)                   | HPC cluster compatibility, HiGHS integration |
| **Work distribution**      | Dynamic dispatch from rank 0         | Optimal load balancing with shared memory    |
| **Batch size**             | = OpenMP thread count                | One forward pass per thread                  |
| **Scenario/Cut storage**   | MPI shared memory (MPI_Win)          | No replication within node                   |
| **Cut aggregation**        | `comm.reduce()` + `comm.broadcast()` | Deterministic for reproducibility            |

## 8. Deployment Configuration

```bash
# Example: 2-socket AMD EPYC, 64 cores/socket, 4 NUMA nodes/socket (128 cores, 8 NUMA)
export MPI_RANKS=8              # 1 rank per NUMA node
export OMP_NUM_THREADS=16       # All cores in NUMA node
export OMP_PROC_BIND=close      # Keep threads on same NUMA node
export OMP_PLACES=cores         # Bind to physical cores
export OMP_SCHEDULE=static      # Reproducible scheduling

mpirun -np 8 --map-by ppr:1:numa --bind-to numa \
    -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
    ./powers_sddp --config case/config.json
```

## Cross-References

- [Work Distribution](./work-distribution.md) — forward/backward pass scenario distribution and dynamic dispatch
- [Design Principles](../00-overview/design-principles.md) — foundational data model design goals including distributed I/O
- [SDDP Algorithm](../01-math/sddp-algorithm.md) — algorithmic structure of forward and backward passes
