---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.1 (Memory Budget Overview)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.2 (Memory Layout Strategy)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.3 (NUMA-Aware Allocation)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §24.4 (Memory Pool for Temporary Allocations)"
  - "DATA_MODEL_SPECIFICATION.md §6.7 (NUMA-Aware Memory Management)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from architecture §24 and data model §6.7 with ferroMPI migration"
---

# Memory Architecture

## Purpose

This spec defines the memory budget, layout strategy, NUMA-aware allocation, and memory pooling used by POWE.RS to manage data efficiently across MPI ranks and OpenMP threads. Proper memory architecture is critical for performance on modern multi-socket, many-NUMA-node systems where access latency varies by ~3x between local and remote memory.

## 1. Memory Budget Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Memory Architecture Overview                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Target System: 768 GB RAM, 8 MPI ranks (96 GB per rank)                        │
│  Reference Case: Brazilian System (156 hydros, 120 stages, 200 scenarios)       │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Per-Rank Memory Budget                                                   │   │
│  │                                                                           │   │
│  │  Component                    │ Size      │ Type                         │   │
│  │  ────────────────────────────┼───────────┼──────────────────────────────│   │
│  │  Case Data (replicated)      │ 50 MB     │ Read-only, shared across thds │   │
│  │  PAR Models                   │ 100 MB    │ Read-only                    │   │
│  │  Correlation Matrices         │ 20 MB     │ Read-only                    │   │
│  │  FCF Cuts (growing)          │ 500 MB    │ Read-mostly, synced          │   │
│  │  Scenario Data               │ 2 GB      │ Per-rank subset              │   │
│  │  LP Solver Workspaces        │ 4 GB      │ Per-thread × 24 threads      │   │
│  │  Solution Buffers            │ 500 MB    │ Per-thread                   │   │
│  │  Output Buffers              │ 200 MB    │ Streaming queue              │   │
│  │  Overhead/Fragmentation      │ 1 GB      │ Safety margin                │   │
│  │  ────────────────────────────┼───────────┼──────────────────────────────│   │
│  │  TOTAL                       │ ~8.4 GB   │ (well within 96 GB)          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Scaling: 10× problem size (1560 hydros) → ~30 GB per rank (still fits)        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Shared Memory Contents (MPI Windows — one copy per node):**

| Region           | Size    | Access Pattern                                           |
| ---------------- | ------- | -------------------------------------------------------- |
| Scenario Storage | 7.68 GB | 1000 passes × 120 stages × 20 branches, read by any rank |
| Cut Storage      | 18.6 GB | Preallocated slots, written by rank 0, read by all       |

**Per-Rank Resources:**

- OpenMP threads: 16 per rank (configurable)
- LP solver instances: 1 per thread
- Local memory: ~240 MB per rank (excluding shared windows)

**Memory Comparison:**

| Approach                  | Memory Required                    |
| ------------------------- | ---------------------------------- |
| NEWAVE-style (replicated) | 16 ranks × 26.3 GB = **294 GB**    |
| POWE.RS hybrid (shared)   | 26.3 GB + 4 × 240 MB = **27.3 GB** |
| **Reduction**             | **~11×**                           |

## 2. Memory Layout Strategy

```rust
/// Memory management for SDDP execution
pub struct MemoryManager {
    /// Static read-only data (shared across threads)
    case_data: Arc<CaseData>,

    /// FCF cuts (read-mostly, periodic updates)
    fcf: RwLock<FutureCostFunction>,

    /// Per-thread LP solver workspaces
    solver_workspaces: Vec<ThreadLocal<SolverWorkspace>>,

    /// Scenario data (per-rank partition)
    scenario_buffer: ScenarioBuffer,

    /// Memory pool for temporary allocations
    temp_pool: MemoryPool,
}

/// Thread-local LP solver workspace (avoids allocation per solve)
pub struct SolverWorkspace {
    /// Reusable LP model
    model: LpModel,

    /// Solution vector (pre-allocated)
    solution: Vec<f64>,

    /// Dual vector (pre-allocated)
    duals: Vec<f64>,

    /// Basis information (for warm-starting)
    basis: LpBasis,

    /// Scratch buffers
    scratch: ScratchBuffers,
}

impl SolverWorkspace {
    /// Pre-allocate workspace for problem of given size
    pub fn new(n_vars: usize, n_constraints: usize) -> Self {
        Self {
            model: LpModel::with_capacity(n_vars, n_constraints),
            solution: vec![0.0; n_vars],
            duals: vec![0.0; n_constraints],
            basis: LpBasis::new(n_vars, n_constraints),
            scratch: ScratchBuffers::new(n_vars, n_constraints),
        }
    }

    /// Reset workspace for new problem (reuse allocations)
    pub fn reset(&mut self) {
        self.model.clear();
        self.solution.fill(0.0);
        self.duals.fill(0.0);
        self.basis.invalidate();
    }
}
```

**Design Rationale:**

- `Arc<CaseData>` — immutable case data is shared across threads without locking
- `RwLock<FutureCostFunction>` — many readers (forward pass) with infrequent writes (backward pass cut insertion)
- `ThreadLocal<SolverWorkspace>` — each OpenMP thread has its own solver workspace, avoiding allocation on the hot path
- `MemoryPool` — arena-style allocation for temporary LP vectors (see §4)

## 3. NUMA-Aware Allocation

### 3.1 Topology Detection via ferroMPI

> **ferroMPI migration**: Replace manual SLURM env var parsing (`SLURM_LOCALID`, `SLURM_NNODES`, etc.) with `ferrompi::slurm` helpers from the `numa` feature. ferroMPI provides safe, typed access to SLURM topology information.

| Legacy Pattern                           | ferroMPI Replacement                                                   |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| `std::env::var("SLURM_LOCALID").parse()` | `ferrompi::slurm::local_rank()`                                        |
| `std::env::var("SLURM_NNODES").parse()`  | `ferrompi::slurm::node_count()`                                        |
| Manual NUMA domain calculation           | `ferrompi::slurm` NUMA-aware helpers                                   |
| Raw `MPI_Win_allocate_shared` calls      | `ferrompi::SharedWindow<T>::new(comm, count)` (requires `rma` feature) |

### 3.2 NUMA Node Allocation

```rust
/// NUMA-aware memory allocation utilities
pub mod numa {
    use std::alloc::{alloc, dealloc, Layout};

    /// Allocate memory on specific NUMA node
    #[cfg(target_os = "linux")]
    pub fn alloc_on_node<T>(count: usize, node: i32) -> Box<[T]> {
        let layout = Layout::array::<T>(count).expect("Invalid layout");

        unsafe {
            // Set memory policy for this allocation
            libc::set_mempolicy(
                libc::MPOL_BIND,
                &(1u64 << node) as *const u64,
                64,
            );

            let ptr = alloc(layout) as *mut T;

            // Reset to default policy
            libc::set_mempolicy(libc::MPOL_DEFAULT, std::ptr::null(), 0);

            // First-touch initialization
            for i in 0..count {
                std::ptr::write(ptr.add(i), T::default());
            }

            Box::from_raw(std::slice::from_raw_parts_mut(ptr, count))
        }
    }

    /// Get NUMA node for current thread
    #[cfg(target_os = "linux")]
    pub fn current_node() -> i32 {
        unsafe {
            let cpu = libc::sched_getcpu();
            libc::numa_node_of_cpu(cpu)
        }
    }

    /// Allocate scenario buffer with NUMA-aware placement
    pub fn alloc_scenario_buffer(
        n_scenarios: usize,
        n_stages: usize,
        n_hydros: usize,
        threads_per_node: usize,
    ) -> Vec<f64> {
        let total_size = n_scenarios * n_stages * n_hydros;
        let mut buffer = vec![0.0; total_size];

        // Parallel first-touch to distribute across NUMA nodes
        buffer.par_chunks_mut(total_size / threads_per_node)
            .for_each(|chunk| {
                // Touch from thread on target NUMA node
                for x in chunk.iter_mut() {
                    *x = 0.0;
                }
            });

        buffer
    }
}
```

### 3.3 First-Touch Initialization Pattern

```rust
/// NUMA-aware array allocation with first-touch initialization
pub fn allocate_numa_aware<T: Default + Send>(size: usize) -> Vec<T> {
    // Allocate uninitialized
    let mut vec = Vec::with_capacity(size);
    unsafe { vec.set_len(size); }

    // Initialize in parallel — each thread touches its portion.
    // Memory pages are allocated on the NUMA node of the touching thread.
    let num_threads = omp::get_max_threads();
    let chunk_size = (size + num_threads - 1) / num_threads;

    vec.par_chunks_mut(chunk_size)
        .for_each(|chunk| {
            for elem in chunk.iter_mut() {
                *elem = T::default();  // First touch allocates on local NUMA
            }
        });

    vec
}
```

### 3.4 NUMA-Partitioned Scenario Data

```rust
/// Scenario data partitioned by NUMA node
pub struct NumaPartitionedScenarios {
    /// Scenario data, partitioned so each NUMA node's threads access local data
    partitions: Vec<Vec<ScenarioData>>,

    /// Mapping from scenario_id to (numa_node, local_index)
    scenario_map: Vec<(usize, usize)>,
}

impl NumaPartitionedScenarios {
    pub fn new(scenarios: Vec<ScenarioData>, num_numa_nodes: usize) -> Self {
        let scenarios_per_node = (scenarios.len() + num_numa_nodes - 1) / num_numa_nodes;

        // Partition scenarios across NUMA nodes
        let partitions: Vec<Vec<ScenarioData>> = (0..num_numa_nodes)
            .into_par_iter()
            .map(|numa_id| {
                let start = numa_id * scenarios_per_node;
                let end = std::cmp::min(start + scenarios_per_node, scenarios.len());
                // Clone data on each NUMA node (first-touch allocates locally)
                scenarios[start..end].to_vec()
            })
            .collect();

        // Build index map
        let scenario_map = (0..scenarios.len())
            .map(|s| {
                let numa = s / scenarios_per_node;
                let local = s % scenarios_per_node;
                (numa, local)
            })
            .collect();

        Self { partitions, scenario_map }
    }

    pub fn get(&self, scenario_id: usize) -> &ScenarioData {
        let (numa, local) = self.scenario_map[scenario_id];
        &self.partitions[numa][local]
    }
}
```

### 3.5 NUMA Topology Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EPYC 8-NUMA TOPOLOGY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NUMA 0          NUMA 1          NUMA 2          NUMA 3                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 24 cores │    │ 24 cores │    │ 24 cores │    │ 24 cores │              │
│  │ Local    │────│          │────│          │────│          │              │
│  │ Memory   │    │          │    │          │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │              │              │              │                        │
│       └──────────────┴──────────────┴──────────────┘                        │
│                           Interconnect                                      │
│       ┌──────────────┬──────────────┬──────────────┐                        │
│       │              │              │              │                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 24 cores │    │ 24 cores │    │ 24 cores │    │ 24 cores │              │
│  │          │────│          │────│          │────│ Local    │              │
│  │          │    │          │    │          │    │ Memory   │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│  NUMA 4          NUMA 5          NUMA 6          NUMA 7                     │
│                                                                             │
│  Memory Latency (ns):                                                       │
│  - Local NUMA: ~80ns                                                        │
│  - Adjacent NUMA: ~120ns                                                    │
│  - Remote NUMA: ~200ns                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 Shared Memory Windows for Intra-Node Data

For read-only data shared by all ranks on the same node (e.g., scenario storage, cut coefficients), use `ferrompi::SharedWindow<T>` from the `rma` feature to avoid per-rank duplication:

```rust
use ferrompi::SharedWindow;

// Create shared window — only one rank per node allocates physical memory
let shared_cuts: SharedWindow<f64> = SharedWindow::new(&shared_comm, cut_count);

// All ranks on the same node can read through the shared pointer
let cut_data: &[f64] = shared_cuts.as_slice();
```

This replaces the legacy pattern of `MPI_Win_allocate_shared` + `MPI_Win_shared_query` with a safe, typed API. See [Hybrid Parallelism](./hybrid-parallelism.md) §1 for shared memory region sizing.

## 4. Memory Pool for Temporary Allocations

```rust
/// Fixed-size memory pool to avoid allocation during hot paths
pub struct MemoryPool {
    /// Pool of pre-allocated buffers
    pools: Vec<Mutex<Vec<Vec<f64>>>>,

    /// Buffer sizes for each pool
    sizes: Vec<usize>,
}

impl MemoryPool {
    /// Create pool with common buffer sizes
    pub fn new(buffers_per_size: usize) -> Self {
        let sizes = vec![64, 256, 1024, 4096, 16384, 65536];

        let pools = sizes.iter()
            .map(|&size| {
                let buffers: Vec<Vec<f64>> = (0..buffers_per_size)
                    .map(|_| vec![0.0; size])
                    .collect();
                Mutex::new(buffers)
            })
            .collect();

        Self { pools, sizes }
    }

    /// Get buffer of at least requested size
    pub fn get(&self, min_size: usize) -> PooledBuffer {
        // Find smallest pool that fits
        let pool_idx = self.sizes.iter()
            .position(|&s| s >= min_size)
            .unwrap_or(self.sizes.len() - 1);

        let buffer = {
            let mut pool = self.pools[pool_idx].lock().unwrap();
            pool.pop()
        };

        match buffer {
            Some(buf) => PooledBuffer {
                buffer: buf,
                pool_idx,
                pool: self,
            },
            None => {
                // Pool exhausted, allocate new
                PooledBuffer {
                    buffer: vec![0.0; self.sizes[pool_idx]],
                    pool_idx,
                    pool: self,
                }
            }
        }
    }
}

/// RAII guard that returns buffer to pool on drop
pub struct PooledBuffer<'a> {
    buffer: Vec<f64>,
    pool_idx: usize,
    pool: &'a MemoryPool,
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) {
        let buffer = std::mem::take(&mut self.buffer);
        let mut pool = self.pool.pools[self.pool_idx].lock().unwrap();
        pool.push(buffer);
    }
}

impl std::ops::Deref for PooledBuffer<'_> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &self.buffer
    }
}

impl std::ops::DerefMut for PooledBuffer<'_> {
    fn deref_mut(&mut self) -> &mut [f64] {
        &mut self.buffer
    }
}
```

**Pool Sizing:**

| Buffer Size | Typical Use                    | Buffers Pre-allocated |
| ----------- | ------------------------------ | --------------------- |
| 64          | Small scratch vectors          | 32 per thread         |
| 256         | Single-hydro constraint rows   | 32 per thread         |
| 1,024       | LP variable/constraint vectors | 16 per thread         |
| 4,096       | Multi-hydro constraint blocks  | 8 per thread          |
| 16,384      | Full LP column sets            | 4 per thread          |
| 65,536      | Large temporary matrices       | 2 per thread          |

## Cross-References

- [Hybrid Parallelism](./hybrid-parallelism.md) — MPI+OpenMP architecture, shared memory regions, and deployment configuration
- [Work Distribution](./work-distribution.md) — how scenarios are distributed across ranks and threads
- [Checkpointing](./checkpointing.md) — checkpoint file format and memory footprint of persisted state
- [Design Principles](../00-overview/design-principles.md) — foundational design goals including memory efficiency
