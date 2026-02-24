---
status: approved
review_priority: 4-low
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.1 (Hybrid Parallelism Overview)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.2 (Design Rationale)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.3 (Parallel Configuration)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.4 (OpenMP FFI Bindings)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.5 (Initialization Sequence)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §20.6 (Build Configuration)"
  - "DATA_MODEL_SPECIFICATION.md §6.1 (Hybrid MPI+OpenMP Architecture Overview)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): Thread-trajectory affinity is the primary parallelization pattern — each thread owns a complete forward trajectory and the corresponding backward pass. Forward pass state save/restore (solver basis, scenario state) needed when M forward passes > N threads, analogous to CPU context switching but only at stage boundaries. These observations should be validated and detailed during P4 review. See sddp-algorithm.md §3.4."
  - date: 2026-02-23
    description: "P4 review rewrite. Fixed review_priority (2-high → 4-low). Stripped all Rust code (6 blocks: ParallelConfig structs, OpenMP FFI bindings, init_parallel, build.rs) and C code (1 block: openmp_wrapper.c), replaced with behavioral descriptions and tables. Fixed forward pass distribution from 'dynamic dispatch from rank 0' to static contiguous block distribution (per approved training-loop.md §4.3). Removed env var ordering issue (OpenMP ICVs set after runtime init). Split §4 into environment-derived (read-only) vs configurable parameters (per cli-and-lifecycle.md §6.1). Added §5 OpenMP C FFI strategy (behavioral). Added §6 initialization sequence (corrected ordering). §7 backward pass reduction corrected to MPI_Allgatherv (per training-loop.md §6.3). Cross-references expanded from 3 to 14."
  - date: 2026-02-23
    description: "Review feedback: Clarified ferrompi role — ferrompi is the backbone for all process-level parallelism (inter-node communication, intra-node shared memory windows, topology detection), not just inter-node. OpenMP C FFI fills only the threading gap that MPI cannot cover. Rewrote §1 overview and §1.1 to frame as 'ferrompi + OpenMP' rather than 'MPI + OpenMP'. Added §1.2 ferrompi Capabilities Used table (6 capabilities). Renumbered §1.2→§1.3 Shared Memory Layout."
---

# Hybrid Parallelism

## Purpose

This spec defines the hybrid parallelization strategy used by POWE.RS: ferrompi as the backbone for inter-node communication, intra-node shared memory, and topology detection; OpenMP via C FFI as the threading layer for intra-rank parallelism; the design rationale for this split; configuration; initialization; and build integration.

## 1. Hybrid Architecture Overview

POWE.RS employs a hybrid parallelization strategy optimized for modern HPC architectures with multi-socket, many-core nodes. **ferrompi** (safe Rust MPI bindings) is the backbone for all process-level parallelism and shared memory, while **OpenMP via C FFI** fills the one gap MPI cannot cover: thread-level parallelism within a single rank.

| Level                        | Technology                                            | Scope                            | Responsibility                                                                                 |
| ---------------------------- | ----------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Inter-node / Inter-NUMA**  | ferrompi (MPI point-to-point and collectives)         | Across NUMA domains and nodes    | Work distribution, cut synchronization (`MPI_Allgatherv`), bound aggregation (`MPI_Allreduce`) |
| **Intra-node shared memory** | ferrompi (`SharedWindow<T>`, MPI windows)             | Ranks on the same physical node  | Scenario storage and cut pool shared without replication                                       |
| **Topology detection**       | ferrompi (`split_shared_memory()`, SLURM integration) | Node and NUMA discovery          | Identify co-located ranks, NUMA domain mapping                                                 |
| **Intra-rank threading**     | OpenMP via C FFI                                      | Threads within a single MPI rank | Parallel LP solves across scenario trajectories                                                |

### 1.1 Why ferrompi + OpenMP

ferrompi covers inter-node communication, intra-node shared memory, and topology detection — everything at the process level. The one thing MPI does not provide is **thread-level parallelism within a single rank**. To parallelize LP solves across cores within a NUMA domain, a threading model is needed. OpenMP (via a thin C FFI wrapper) fills this gap, providing vendor-optimized scheduling, affinity control, and NUMA-aware thread placement that Rust's standard threading libraries (Rayon) cannot match on HPC hardware.

SDDP's compute pattern — many independent small LP solves sharing large read-only data (scenarios, cuts) — maps naturally to this split:

- **ferrompi ranks** provide memory isolation between NUMA domains and between nodes. Each rank has its own address space, avoiding false sharing across NUMA boundaries. Shared memory windows allow ranks on the same node to share large read-only data without replication, reducing memory footprint by an order of magnitude compared to per-rank replication.
- **OpenMP threads** within each rank share the rank's address space, enabling fine-grained parallelism over LP solves without data replication within a NUMA domain.

The typical deployment is one MPI rank per NUMA domain. See [SLURM Deployment](./slurm-deployment.md) for concrete job configurations.

### 1.2 ferrompi Capabilities Used

ferrompi provides safe, idiomatic Rust bindings for MPI. The capabilities used by POWE.RS:

| Capability                | ferrompi API                                 | SDDP Use Case                                                         |
| ------------------------- | -------------------------------------------- | --------------------------------------------------------------------- |
| Thread-safe communicators | `Communicator` is `Send + Sync`              | Hybrid MPI+threads without `unsafe` sharing of raw `MPI_Comm` handles |
| Shared memory windows     | `SharedWindow<T>`                            | Scenario storage and cut pool shared within a node                    |
| Intra-node communicator   | `split_shared_memory()`                      | Group co-located ranks for shared memory operations                   |
| Collectives               | `allreduce()`, `allgatherv()`, `broadcast()` | Bound aggregation, cut synchronization, statistics                    |
| SLURM/NUMA detection      | Topology query APIs                          | Read resource allocations from scheduler environment                  |
| Threading level           | `init_with_threading(ThreadLevel::Multiple)` | Full MPI thread support for hybrid parallelism                        |

### 1.3 Shared Memory Layout

Ranks on the same physical node share two large data regions via MPI windows, eliminating per-rank replication:

| Region           | Contents                                                | Access Pattern                                                                 |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Scenario storage | Pre-generated opening tree noise vectors for all stages | Read-only by all ranks on the node; written once during scenario generation    |
| Cut pool         | Accumulated Benders cuts for all stages                 | Written by the rank designated as the node-local aggregator; read by all ranks |

**Per-rank resources** (not shared):

- One LP solver instance per OpenMP thread (LP solvers are not thread-safe — see [Solver Workspaces §1.1](../03-architecture/solver-workspaces.md))
- Thread-local solution buffers, RHS patch buffers, basis cache
- Per-thread state for forward pass trajectory tracking

> The exact memory sizing for shared regions and per-rank resources depends on the problem scale. See [Memory Architecture](./memory-architecture.md) for budget computation and [Production Scale Reference](../00-overview/production-scale-reference.md) for representative dimensions.

## 2. Design Rationale: Why OpenMP (Not Rayon)

Native OpenMP is used via C FFI rather than Rayon (Rust's standard parallelism library). The decision is driven by SDDP's specific compute pattern: many small LP solves with shared data on NUMA-aware HPC hardware.

| Criterion                  | Native OpenMP via FFI                                | Rayon                                     |
| -------------------------- | ---------------------------------------------------- | ----------------------------------------- |
| **Vendor optimization**    | Full (Intel, AMD, GCC runtimes)                      | Limited (generic work-stealing)           |
| **Scheduling control**     | `static`, `dynamic`, `guided`                        | Work-stealing only                        |
| **Affinity control**       | `OMP_PLACES`, `OMP_PROC_BIND`                        | None                                      |
| **NUMA awareness**         | First-touch placement, explicit binding              | Opaque scheduler decisions                |
| **Reduction primitives**   | Hardware-optimized tree reduction                    | Manual implementation required            |
| **HPC ecosystem**          | Standard (SLURM, modules, profilers like VTune, MAP) | Limited HPC tool integration              |
| **LP solver coordination** | Explicit single-thread forcing via environment       | Potential conflicts with solver threading |

For SDDP's compute pattern, the ability to control scheduling and affinity translates to significantly better performance on NUMA systems — work-stealing can move tasks across NUMA boundaries, causing remote memory access penalties on every LP solve.

## 3. MPI vs OpenMP Responsibility Split

| Aspect            | MPI Ranks                                                                                         | OpenMP Threads                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Purpose**       | Distributed memory, inter-node communication                                                      | Shared memory, intra-NUMA parallelism                                         |
| **Granularity**   | Coarse: contiguous blocks of scenario trajectories                                                | Fine: individual stage LP solves within assigned trajectories                 |
| **Communication** | Explicit: cut synchronization (`MPI_Allgatherv`), bound aggregation (`MPI_Allreduce`), statistics | Implicit: shared read-only case data, scenario tree, cut pool                 |
| **Memory model**  | Replicated (per-rank solver instances) or shared (MPI windows for scenarios/cuts)                 | Shared within rank (read-only data), thread-local (solver instances, buffers) |
| **Load balance**  | Static distribution: scenarios assigned in contiguous blocks to ranks                             | Dynamic scheduling within rank: `schedule(dynamic,1)` for LP solves           |

**Thread-trajectory affinity**: Each OpenMP thread owns one or more complete forward trajectories and also executes the backward pass for those trajectories. This preserves cache locality — the solver basis, scenario data, and LP coefficients remain warm in the thread's cache lines across stages. When the number of forward passes $M$ exceeds the total thread count $N$, threads process multiple trajectories in batches with state save/restore at stage boundaries. See [Training Loop §4.3](../03-architecture/training-loop.md) and [SDDP Algorithm §3.4](../01-math/sddp-algorithm.md).

**Forward pass**: Scenarios are distributed across MPI ranks in contiguous blocks. Within each rank, scenarios are parallelized across OpenMP threads. No inter-rank synchronization during the forward pass — each rank processes its assigned trajectories independently. After all ranks complete, `MPI_Allreduce` aggregates lower bound and upper bound statistics.

**Backward pass**: Per-stage synchronization barrier — all threads across all ranks must complete cut generation at stage $t$ before proceeding to stage $t-1$. Within each stage:

| Level      | Operation                                                                     | Result                                                    |
| ---------- | ----------------------------------------------------------------------------- | --------------------------------------------------------- |
| 1 — OpenMP | Each thread evaluates its assigned trial points and all openings sequentially | Thread-local cut contributions                            |
| 2 — MPI    | `MPI_Allgatherv` collects new cuts from all ranks                             | All ranks have the complete set of new cuts for stage $t$ |

See [Training Loop §6.3](../03-architecture/training-loop.md) and [Synchronization](./synchronization.md).

## 4. Parallel Configuration

Parallel configuration follows the two-category separation established in [CLI and Lifecycle §6](../03-architecture/cli-and-lifecycle.md):

### 4.1 Resource Allocations (Read-Only from Environment)

These parameters are determined by the MPI launcher and job scheduler. The program reads them from the environment and **must not override them**.

| Parameter        | Source                                     | Description                   |
| ---------------- | ------------------------------------------ | ----------------------------- |
| MPI rank count   | MPI launcher (`mpiexec -n`, `srun`)        | Number of processes           |
| Threads per rank | `SLURM_CPUS_PER_TASK` or `OMP_NUM_THREADS` | OpenMP threads per rank       |
| Memory per node  | `SLURM_MEM_PER_NODE`                       | Memory budget for pool sizing |

If `OMP_NUM_THREADS` is not set and no scheduler is detected, the program defaults to 1 thread per rank.

### 4.2 OpenMP Runtime Parameters (Environment Variables)

These parameters are set by the user in the job script or shell environment **before** launching the program. POWE.RS reads but does not modify them — per the OpenMP specification, internal control variables (ICVs) derived from environment variables are read once at runtime initialization and cannot be reliably changed afterward.

| Variable          | Recommended Value | Description                                                                 |
| ----------------- | ----------------- | --------------------------------------------------------------------------- |
| `OMP_PROC_BIND`   | `close`           | Bind threads to cores within NUMA domain (best for compute-bound LP solves) |
| `OMP_PLACES`      | `cores`           | Bind to physical cores (avoid SMT contention)                               |
| `OMP_WAIT_POLICY` | `passive`         | Sleep idle threads (reduce power; `active` spin-wait for lowest latency)    |
| `OMP_STACKSIZE`   | `64M`             | Stack size per thread (needed for deep LP solver recursion)                 |
| `OMP_SCHEDULE`    | `dynamic,1`       | Default schedule for parallel regions                                       |

### 4.3 LP Solver Threading Suppression

LP solvers must be forced to single-threaded mode because POWE.RS handles parallelism at the outer level (one solver per thread). The following environment variables are validated at startup:

| Variable                            | Required Value | Affected Solver                                             |
| ----------------------------------- | -------------- | ----------------------------------------------------------- |
| `HIGHS_PARALLEL`                    | `false`        | HiGHS                                                       |
| `MKL_NUM_THREADS`                   | `1`            | Any solver using MKL (CPLEX, etc.)                          |
| `OMP_NUM_THREADS` (solver-internal) | N/A            | Controlled by outer OpenMP; solver uses calling thread only |

See [Solver Abstraction §4](../03-architecture/solver-abstraction.md) for the solver interface contract requiring non-thread-safe solvers.

### 4.4 NUMA Allocation Policy

NUMA memory allocation is configured at startup via the Linux `libnuma` API:

| Policy           | Behavior                                                           | When to Use                                                                      |
| ---------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Local allocation | Memory allocated on the NUMA node of the calling thread            | Default — ensures LP solver data is NUMA-local                                   |
| First-touch      | Pages allocated on the NUMA node of the first thread to write them | Used for shared arrays during initialization (each thread touches its partition) |

See [Memory Architecture](./memory-architecture.md) for detailed NUMA-aware allocation strategy and budget computation.

## 5. OpenMP C FFI Strategy

OpenMP parallel regions require compiler pragma support (`#pragma omp parallel`), which is unavailable in `rustc`. POWE.RS bridges this gap with a thin C wrapper compiled with an OpenMP-capable compiler, linked as a static library into the Rust binary.

### 5.1 Why a C Wrapper

The C wrapper exists solely to provide `#pragma omp` constructs. The wrapper functions are minimal — they set up the parallel region and call back into Rust via function pointers. All application logic remains in Rust.

### 5.2 Wrapper Primitives

The C wrapper exposes three primitives:

| Primitive                      | OpenMP Construct                                         | SDDP Use Case                                                                                         |
| ------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Parallel for (dynamic)**     | `#pragma omp parallel for schedule(dynamic, chunk_size)` | Forward pass: distribute scenario trajectories across threads. Backward pass: distribute trial points |
| **Parallel reduce (sum)**      | `#pragma omp parallel reduction(+:total)`                | Aggregate stage costs, compute per-rank bound contributions                                           |
| **First-touch initialization** | `#pragma omp parallel` with thread-partitioned writes    | NUMA-aware initialization of large arrays (solver buffers, scenario storage)                          |

### 5.3 Callback Trampoline Pattern

Each wrapper function accepts a C function pointer and an opaque `void*` user_data pointer. On the Rust side, a type-erased trampoline function (`extern "C"`) casts the user_data back to the original Rust closure, enabling ergonomic parallel iteration from Rust while crossing the C FFI boundary.

**Safety invariants**: The trampoline pattern requires that (1) the closure outlives the parallel region (guaranteed because the caller blocks until completion), and (2) the closure is `Sync` (the Rust type system enforces this via the bound `F: Fn + Sync`).

### 5.4 Direct OpenMP Runtime Functions

In addition to the wrapper primitives, the Rust FFI layer provides direct bindings to standard OpenMP runtime functions:

| Function                | Purpose                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `omp_get_thread_num()`  | Thread ID within the current parallel region                         |
| `omp_get_num_threads()` | Active thread count                                                  |
| `omp_get_wtime()`       | Wall-clock timer (for per-thread timing)                             |
| `omp_set_num_threads()` | Set thread count (called once at initialization)                     |
| `omp_set_dynamic(0)`    | Disable dynamic thread adjustment (thread count must be predictable) |

## 6. Initialization Sequence

The parallel environment is initialized during the Startup phase (see [CLI and Lifecycle §5](../03-architecture/cli-and-lifecycle.md)). The ordering is critical — MPI must be initialized before OpenMP because MPI_Init_thread with `MPI_THREAD_MULTIPLE` must be the first MPI call.

**Step 1 — MPI initialization**: Call `ferrompi::init_with_threading(ThreadLevel::Multiple)` to initialize MPI with full thread support. This must be called before any other MPI operations and before the OpenMP runtime is initialized (some MPI implementations interact with the threading layer during init).

**Step 2 — Topology detection**: Read the MPI communicator to determine rank count and rank ID. Detect the scheduler environment (SLURM, PBS, or local) to read resource allocations. Validate rank count if the job script specifies an expected value.

**Step 3 — Shared memory communicator**: Create an intra-node communicator via `ferrompi::Communicator::split_shared_memory()`. This groups ranks on the same physical node for shared memory window operations (scenario storage, cut pool).

**Step 4 — OpenMP configuration**: The OpenMP runtime initializes implicitly on the first parallel region entry (or explicitly via `omp_set_num_threads()`). The environment variables `OMP_PROC_BIND`, `OMP_PLACES`, `OMP_WAIT_POLICY`, `OMP_STACKSIZE` must already be set in the process environment (by the job script) before this point. The program calls `omp_set_dynamic(0)` to disable dynamic thread adjustment.

**Step 5 — LP solver suppression**: Validate that LP solver threading environment variables are set correctly (`HIGHS_PARALLEL=false`, `MKL_NUM_THREADS=1`). If not set, set them before any solver instance is created. These are process-level settings and do not conflict with OpenMP (they affect solver-internal threading, not the outer parallelism).

**Step 6 — NUMA allocation policy**: On Linux, set the NUMA local allocation policy via `libnuma`. This must happen before any large memory allocation so that subsequent allocations respect NUMA locality.

**Step 7 — Workspace allocation**: Allocate thread-local solver workspaces on each thread's NUMA domain using first-touch initialization. See [Solver Workspaces §1](../03-architecture/solver-workspaces.md).

**Step 8 — Startup logging**: Rank 0 logs the parallel configuration summary (rank count, threads per rank, total cores, detected scheduler, NUMA topology).

## 7. Build Integration

The OpenMP C wrapper must be compiled with an OpenMP-capable C compiler and linked as a static library into the Rust binary. The build script (`build.rs`) handles this:

### 7.1 Compiler Detection

The build script probes for OpenMP-capable compilers in priority order:

| Priority | Compiler             | OpenMP Flag | Runtime Library |
| -------- | -------------------- | ----------- | --------------- |
| 1        | Intel oneAPI (`icx`) | `-qopenmp`  | `iomp5`         |
| 2        | GCC (`gcc`)          | `-fopenmp`  | `gomp`          |
| 3        | Clang (`clang`)      | `-fopenmp`  | `omp`           |

The build fails if no OpenMP-capable compiler is found.

### 7.2 Compilation and Linking

The C wrapper file is compiled with the detected compiler using `-O3 -march=native -fPIC` and the appropriate OpenMP flag, then archived into a static library (`libopenmp_wrapper.a`). Cargo link directives add the static library and the OpenMP runtime library to the final binary.

MPI linking is handled entirely by the ferrompi crate dependency — no manual MPI link flags are needed in the build script.

### 7.3 Rebuild Triggers

The build script registers `cargo:rerun-if-changed` for the C wrapper source file, so the wrapper is recompiled only when modified.

## 8. Reference Deployment

Example deployment on a 2-socket AMD EPYC node (64 cores/socket, 4 NUMA domains/socket = 128 cores, 8 NUMA domains):

| Parameter        | Value   | Rationale                             |
| ---------------- | ------- | ------------------------------------- |
| MPI ranks        | 8       | 1 per NUMA domain                     |
| Threads per rank | 16      | All cores in the NUMA domain          |
| Total cores      | 128     | Full node utilization                 |
| `OMP_PROC_BIND`  | `close` | Keep threads within their NUMA domain |
| `OMP_PLACES`     | `cores` | Bind to physical cores (avoid SMT)    |

For complete SLURM job scripts and multi-node deployment patterns, see [SLURM Deployment](./slurm-deployment.md).

## Cross-References

- [SDDP Algorithm §3.4](../01-math/sddp-algorithm.md) — Thread-trajectory affinity, backward sync barriers, forward pass state saving
- [Training Loop §4.3](../03-architecture/training-loop.md) — Forward pass parallel distribution: contiguous blocks to ranks, thread-trajectory affinity within rank
- [Training Loop §6.3](../03-architecture/training-loop.md) — Backward pass parallel distribution: per-stage barrier, MPI_Allgatherv for cut synchronization
- [CLI and Lifecycle §6](../03-architecture/cli-and-lifecycle.md) — Resource allocation sourcing (read-only from environment) and algorithm parameter hierarchy
- [Solver Abstraction §4](../03-architecture/solver-abstraction.md) — Solver interface contract: solvers are not thread-safe, one instance per thread
- [Solver Workspaces §1](../03-architecture/solver-workspaces.md) — Thread-local solver infrastructure, NUMA-local allocation, per-stage basis cache
- [Work Distribution](./work-distribution.md) — Detailed forward/backward pass scenario distribution and load balancing
- [Synchronization](./synchronization.md) — Sync points, thread synchronization, lock-free cut aggregation
- [Communication Patterns](./communication-patterns.md) — ferrompi persistent collectives, SharedWindow\<T\>, async overlap
- [Memory Architecture](./memory-architecture.md) — Memory budget computation, NUMA-aware allocation, shared memory sizing
- [Shared Memory Aggregation](./shared-memory-aggregation.md) — Hierarchical cut aggregation, shared memory scenarios
- [Checkpointing](./checkpointing.md) — Checkpoint strategy and warm-start across MPI ranks
- [SLURM Deployment](./slurm-deployment.md) — Job scripts, multi-node configuration, performance monitoring
- [Design Principles](../00-overview/design-principles.md) — Foundational design goals including distributed I/O
