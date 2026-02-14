---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §6.4 (Hierarchical Cut Aggregation)"
  - "DATA_MODEL_SPECIFICATION.md §6.6 (Intra-Node Shared Memory)"
  - "DATA_MODEL_SPECIFICATION.md §6.8 (Performance Monitoring)"
  - "DATA_MODEL_SPECIFICATION.md §6.11 (Shared Memory Scenario Storage)"
  - "DATA_MODEL_SPECIFICATION.md §6.12 (Two-Level Cut Aggregation)"
  - "DATA_MODEL_SPECIFICATION.md §6.13 (Reproducibility Guarantees)"
last_reviewed: null
reviewed_by: null
review_notes: "All MPI_Win references migrated to ferrompi::SharedWindow<T>; MPI_Reduce/MPI_Bcast references use ferrompi Communicator methods"
change_log:
  - date: 2026-02-14
    description: "Extracted from monolith docs (T-022); migrated all MPI references to ferrompi crate API"
---

# Shared Memory and Aggregation

## Purpose

This spec defines the intra-node shared memory architecture, hierarchical and two-level cut aggregation strategies, shared memory scenario storage, performance monitoring metrics, and reproducibility guarantees for POWE.RS. All MPI window operations use `ferrompi::SharedWindow<T>` with RAII semantics.

## 1. Hierarchical Cut Aggregation

> **Problem**: With flat gather-to-master pattern, rank 0 becomes a serialization bottleneck at scale. For 128 ranks sending 50 KB each, rank 0 must process 6.4 MB of receives sequentially, adding ~100 ms overhead per stage.

> **Solution**: Hierarchical tree-based aggregation distributes the aggregation work across intermediate ranks.

**Benefits:**

- Reduces master receive operations from N-1 to log_fanout(N)
- Distributes aggregation computation across tree
- Enables partial cut selection at intermediate levels (optional)

### 1.1 Aggregation Tree

```rust
/// Hierarchical aggregation tree node
pub struct AggregationNode {
    pub rank: i32,
    pub parent: Option<i32>,
    pub children: Vec<i32>,
    pub level: u32,
}

impl AggregationNode {
    /// Build aggregation tree for given world size and fanout
    pub fn build_tree(world_size: i32, fanout: i32) -> Vec<AggregationNode> {
        // Level 0: all ranks are leaves
        // Level 1+: ranks at positions 0, fanout, 2*fanout, ... are aggregators
        // Continue until single root (rank 0)
        todo!()
    }
}

/// Aggregation protocol per stage
pub fn hierarchical_aggregate(
    local_cuts: &[CutMessage],
    tree: &AggregationNode,
    comm: &ferrompi::Communicator,
) -> Option<Vec<CutMessage>> {
    // Step 1: Receive from children (if any)
    let mut all_cuts = local_cuts.to_vec();
    for child in &tree.children {
        let child_cuts: Vec<CutMessage> = comm.recv(*child);
        all_cuts.extend(child_cuts);
    }

    // Step 2: Optional local aggregation (cut selection at intermediate level)
    // This reduces data volume but may affect cut quality
    // let aggregated = local_cut_selection(&all_cuts);

    // Step 3: Send to parent (if not root)
    if let Some(parent) = tree.parent {
        comm.send(&all_cuts, parent);
        None // Non-root ranks don't return cuts
    } else {
        Some(all_cuts) // Root returns all aggregated cuts
    }
}
```

### 1.2 Configuration

| Ranks | Recommended Fanout | Tree Depth | Master Receives |
| ----- | ------------------ | ---------- | --------------- |
| 16    | 4                  | 2          | 4               |
| 64    | 8                  | 2          | 8               |
| 128   | 8                  | 3          | ~16             |
| 512   | 16                 | 2          | 32              |
| 2048  | 16                 | 3          | ~128            |

## 2. Intra-Node Shared Memory

> **Problem**: Each MPI rank maintains a full FCF replica (18.6 GB at production scale). With 4 ranks per node, this requires 42.8 GB just for cuts.

> **Solution**: Use `ferrompi::SharedWindow<T>` so ranks on the same node share a single FCF copy. The `SharedWindow<T>` type provides RAII memory management — `Drop` automatically calls the equivalent of `MPI_Win_free`.

### 2.1 Shared FCF Manager

```rust
use ferrompi::{Communicator, SharedWindow, SharedWindowLock};

/// Shared memory FCF manager using ferroMPI
pub struct SharedFcf {
    /// Shared memory window (RAII — Drop frees the window)
    window: SharedWindow<u8>,

    /// Total size in bytes
    total_size: usize,

    /// Shared memory communicator (ranks on same node)
    shm_comm: Communicator,

    /// Rank within shared memory communicator
    shm_rank: i32,

    /// Whether this rank is the node leader (allocates memory)
    is_leader: bool,
}

impl SharedFcf {
    pub fn new(world_comm: &Communicator, fcf_size: usize) -> Self {
        // Create shared memory communicator
        let shm_comm = world_comm.split_shared_memory();
        let shm_rank = shm_comm.rank();
        let is_leader = shm_rank == 0;

        // Only leader allocates; others get size 0
        let alloc_size = if is_leader { fcf_size } else { 0 };

        // Allocate shared memory window via ferroMPI
        let window = SharedWindow::new(&shm_comm, alloc_size);

        Self {
            window,
            total_size: fcf_size,
            shm_comm,
            shm_rank,
            is_leader,
        }
    }

    /// Read access (all ranks) — direct pointer, zero-copy
    pub fn read_cuts(&self, stage: u32) -> &[BendersCut] {
        let offset = self.stage_offset(stage);
        unsafe {
            let ptr = self.window.base_ptr().add(offset) as *const BendersCut;
            std::slice::from_raw_parts(ptr, self.cuts_per_stage(stage))
        }
    }

    /// Write access (leader only, with window lock)
    pub fn apply_update(&mut self, stage: u32, update: &FcfUpdateMessage) {
        assert!(self.is_leader, "Only leader can write to shared FCF");

        // Lock window for exclusive access (RAII guard)
        let _lock: SharedWindowLock = self.window.lock(0);

        // Apply updates directly to shared memory
        unsafe {
            let offset = self.stage_offset(stage);
            let ptr = self.window.base_ptr().add(offset) as *mut BendersCut;
            // ... apply update ...
        }

        // Lock released on drop of _lock

        // Memory barrier ensures visibility to other ranks
        self.window.fence();
    }
}

// SharedWindow<T> implements Drop — no manual MPI_Win_free needed
```

### 2.2 Memory Savings

| Configuration              | Without Sharing | With Sharing | Savings |
| -------------------------- | --------------- | ------------ | ------- |
| 4 ranks/node, 18.6 GB FCF  | 74.4 GB/node    | 18.6 GB/node | 75%     |
| 8 ranks/node, 18.6 GB FCF  | 148.8 GB/node   | 18.6 GB/node | 87.5%   |
| 16 ranks/node, 18.6 GB FCF | 297.6 GB/node   | 18.6 GB/node | 93.75%  |

## 3. Shared Memory Scenario Storage

> **Context**: With dynamic work distribution, any forward pass can be assigned to any rank. Scenarios must be accessible regardless of assignment.
>
> **Decision**: Shared memory within node using `SharedWindow<f64>` — 7.68 GB per node instead of 7.68 GB x ranks.

### 3.1 Architecture

**Array Layout:** `scenarios[pass_idx][stage][branch][variable]`

- 1000 passes x 120 stages x 20 branches x 320 vars x 8 bytes = 7.68 GB

| Operation | Who                               | When                                     |
| --------- | --------------------------------- | ---------------------------------------- |
| Write     | Each rank writes assigned portion | During scenario generation               |
| Read      | Any rank reads any pass           | During forward pass                      |
| Sync      | `window.fence()`                  | After generation, before iteration start |

### 3.2 Deterministic Scenario Seeding

> **CRITICAL for Reproducibility**: Scenarios must be identical regardless of number of MPI ranks, number of OpenMP threads, which rank generates which pass, or order of generation.

**Solution**: Each scenario is seeded by its **identity**, not its computational assignment.

```rust
/// Deterministic seed computation
/// INVARIANT: Same (master_seed, pass, stage, branch) -> same seed -> same scenario
#[inline]
pub fn scenario_seed(master_seed: u64, pass_idx: u32, stage: u32, branch: u32) -> u64 {
    // SplitMix64-style mixing for high-quality seed derivation
    let mut x = master_seed;

    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x ^= pass_idx as u64;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);

    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x ^= stage as u64;
    x = x.wrapping_mul(0x94d049bb133111eb);

    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x ^= branch as u64;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);

    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x
}
```

### 3.3 Distributed Generation Protocol

```rust
/// Distributed scenario generation using ferroMPI shared memory
pub fn generate_scenarios_distributed(
    config: &ScenarioConfig,
    correlation: &CorrelationFactors,
    node_comm: &ferrompi::Communicator,
) -> SharedScenarioStorage {
    let rank = node_comm.rank() as usize;
    let size = node_comm.size() as usize;
    let num_vars = config.num_hydros + config.num_loads;
    let num_passes = config.num_forward_passes as usize;

    // Calculate total storage
    let total_branches: usize = config.branches_per_stage.iter()
        .map(|&b| b as usize).sum();
    let total_floats = num_passes * total_branches * num_vars;

    // Allocate shared memory (only rank 0 provides size)
    let alloc_count = if rank == 0 { total_floats } else { 0 };
    let window = SharedWindow::<f64>::new(node_comm, alloc_count);

    // Divide generation work across ranks
    let passes_per_rank = num_passes / size;
    let remainder = num_passes % size;
    let my_start = rank * passes_per_rank + rank.min(remainder);
    let my_count = passes_per_rank + if rank < remainder { 1 } else { 0 };

    // Parallel generation within rank using OpenMP
    for local_idx in 0..my_count {
        let pass_idx = (my_start + local_idx) as u32;
        for (stage, &num_branches) in config.branches_per_stage.iter().enumerate() {
            let cholesky = &correlation.factors[stage];
            for branch in 0..num_branches {
                let seed = scenario_seed(config.master_seed, pass_idx, stage as u32, branch);
                let epsilon = generate_scenario(seed, cholesky, num_vars);
                let offset = compute_offset(
                    pass_idx as usize, stage, branch as usize,
                    &config.branches_per_stage, num_vars,
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        epsilon.as_ptr(),
                        window.base_ptr_mut().add(offset),
                        num_vars,
                    );
                }
            }
        }
    }

    // Synchronization fence — ensures all writes visible to all ranks
    window.fence();

    SharedScenarioStorage { window, num_passes, num_vars, /* ... */ }
}
```

### 3.4 Memory Layout and Access

```rust
/// Shared scenario storage with efficient indexing
pub struct SharedScenarioStorage {
    window: SharedWindow<f64>,
    num_passes: usize,
    num_stages: usize,
    branches_per_stage: Vec<u32>,
    num_vars: usize,
    /// Precomputed offsets for O(1) access
    stage_offsets: Vec<usize>,
}

impl SharedScenarioStorage {
    /// Get scenario noise for (pass, stage, branch)
    #[inline]
    pub fn get(&self, pass: usize, stage: usize, branch: usize) -> &[f64] {
        debug_assert!(pass < self.num_passes);
        debug_assert!(stage < self.num_stages);
        debug_assert!(branch < self.branches_per_stage[stage] as usize);

        let offset = self.compute_offset(pass, stage, branch);
        unsafe {
            std::slice::from_raw_parts(self.window.base_ptr().add(offset), self.num_vars)
        }
    }
}
```

## 4. Two-Level Cut Aggregation

The SDDP backward pass uses two levels of reduction for cut aggregation:

1. **Level 1 (Intra-Rank)**: OpenMP reduction across threads within each rank
2. **Level 2 (Inter-Rank)**: MPI reduction across ranks via `ferrompi`

For single-cut formulation, all contributions are summed into one cut per stage.

### 4.1 Two-Level Reduction Architecture

| Level      | Operation                            | Result                              |
| ---------- | ------------------------------------ | ----------------------------------- |
| 1 - OpenMP | Thread reduction within rank         | local_alpha, local_beta per rank    |
| 2 - MPI    | `comm.reduce(Op::Sum, root=0)`       | global_alpha, global_beta at rank 0 |
| Storage    | `cuts[slot] = {alpha, beta, active}` | Rank 0 writes to shared memory      |
| Broadcast  | `comm.broadcast(root=0)`             | All ranks have identical cut data   |

**Slot Calculation:** `slot = iteration * (num_stages - 1) + (stage - 1)`

### 4.2 Implementation

```rust
/// Two-level cut aggregation for single-cut formulation
pub fn aggregate_single_cut(
    stage: usize,
    iteration: u32,
    local_results: &[BackwardSubproblemResult],
    state_dim: usize,
    comm: &ferrompi::Communicator,
) -> SingleCut {
    // =====================================================================
    // LEVEL 1: OpenMP reduction within rank
    // =====================================================================
    let (local_alpha, local_beta) = {
        let mut alpha_sum = 0.0f64;
        let mut beta_sum = vec![0.0f64; state_dim];

        for result in local_results {
            let weight = result.probability;
            alpha_sum += result.alpha * weight;
            for (i, &coef) in result.beta.iter().enumerate() {
                beta_sum[i] += coef * weight;
            }
        }

        (alpha_sum, beta_sum)
    };

    // =====================================================================
    // LEVEL 2: MPI reduction across ranks
    // =====================================================================
    let mut local_buf = Vec::with_capacity(1 + state_dim);
    local_buf.push(local_alpha);
    local_buf.extend_from_slice(&local_beta);

    let mut global_buf = vec![0.0f64; 1 + state_dim];

    // Reduce to rank 0 (deterministic order for reproducibility)
    comm.reduce(&local_buf, &mut global_buf, Op::Sum, 0);

    // =====================================================================
    // BROADCAST: All ranks receive identical cut
    // =====================================================================
    comm.broadcast(&mut global_buf, 0);

    SingleCut {
        alpha: global_buf[0],
        beta: global_buf[1..].to_vec(),
        iteration,
        stage: stage as u32,
    }
}
```

### 4.3 Replicated Cut Selection

> **Key Insight**: After `comm.broadcast()`, all ranks have identical cut data. If all ranks run the **same deterministic algorithm** on this data, they will produce **identical selection decisions** without additional communication.

```rust
/// Replicated cut selection — runs identically on all ranks
/// CRITICAL: Must be deterministic (same inputs -> same outputs)
pub fn replicated_cut_selection(
    stage: usize,
    cuts: &mut SharedCutStorage,
    config: &CutSelectionConfig,
) {
    let stage_cuts = cuts.get_stage_cuts(stage);
    let num_cuts = stage_cuts.len();

    // Deterministic domination check (iterate in slot order)
    let mut active_mask = vec![true; num_cuts];

    for i in 0..num_cuts {
        if !stage_cuts[i].is_active || !active_mask[i] { continue; }
        for j in 0..num_cuts {
            if i == j || !stage_cuts[j].is_active || !active_mask[j] { continue; }
            if is_dominated(&stage_cuts[i], &stage_cuts[j], config) {
                active_mask[i] = false;
                break;
            }
        }
    }

    // Only rank 0 writes to shared storage (avoids races)
    // But all ranks compute the same active_mask!
    if cuts.comm_rank() == 0 {
        for (i, &active) in active_mask.iter().enumerate() {
            if !active && stage_cuts[i].is_active {
                cuts.deactivate_cut(stage, i);
            }
        }
    }

    // Fence ensures all ranks see the update
    cuts.fence(); // window.fence()
}
```

## 5. Performance Monitoring

The following metrics are collected and reported in `training/timing/mpi_ranks.parquet`:

| Metric                  | Description                           | Diagnostic Use           |
| ----------------------- | ------------------------------------- | ------------------------ |
| `computation_time_ms`   | Time in LP solves and cut computation | Baseline work            |
| `communication_time_ms` | Time in MPI calls                     | Communication overhead   |
| `idle_time_ms`          | Time waiting at barriers              | Load imbalance           |
| `gather_time_ms`        | Time in cut gather phase              | Aggregation bottleneck   |
| `bcast_time_ms`         | Time in FCF broadcast                 | Distribution overhead    |
| `memory_high_water_mb`  | Peak RSS during iteration             | Memory pressure          |
| `numa_local_ratio`      | Fraction of local NUMA accesses       | Memory placement quality |

### 5.1 Load Imbalance Detection

```rust
/// Analyze per-rank timing for load imbalance
pub fn analyze_load_balance(rank_timings: &[RankTiming]) -> LoadBalanceReport {
    let compute_times: Vec<f64> = rank_timings.iter()
        .map(|r| r.computation_time_ms as f64)
        .collect();

    let mean = compute_times.iter().sum::<f64>() / compute_times.len() as f64;
    let max = compute_times.iter().cloned().fold(0.0_f64, f64::max);
    let min = compute_times.iter().cloned().fold(f64::MAX, f64::min);

    let imbalance_ratio = (max - min) / mean;
    let slowest_rank = rank_timings.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.computation_time_ms.cmp(&b.computation_time_ms))
        .map(|(i, _)| i)
        .unwrap();

    LoadBalanceReport {
        mean_compute_ms: mean,
        max_compute_ms: max,
        min_compute_ms: min,
        imbalance_ratio,
        slowest_rank,
        recommendation: if imbalance_ratio > 0.2 {
            "Consider dynamic scenario distribution or adaptive load balancing"
        } else {
            "Load balance acceptable"
        },
    }
}
```

## 6. Reproducibility Guarantees

> **Requirement**: Given the same inputs and random seed, POWE.RS must produce **bit-for-bit identical** results regardless of:

| Must Be Independent Of                       |
| -------------------------------------------- |
| Number of MPI ranks                          |
| Number of OpenMP threads per rank            |
| Execution timing/ordering                    |
| Hardware platform (with IEEE 754 compliance) |

### 6.1 Reproducibility Mechanisms

| Component                | Mechanism                                       | Guarantee                                   |
| ------------------------ | ----------------------------------------------- | ------------------------------------------- |
| **Scenario Generation**  | Deterministic seeding by (pass, stage, branch)  | Same scenarios regardless of rank/thread    |
| **Forward Pass Results** | Results indexed by global pass ID               | Assignment to ranks doesn't affect indexing |
| **Cut Aggregation**      | `comm.reduce(Op::Sum, 0)` + `comm.broadcast(0)` | All ranks have identical aggregated cuts    |
| **Cut Selection**        | Replicated algorithm on identical data          | All ranks make identical decisions          |
| **LP Constraint Order**  | Cuts added by slot index (iteration order)      | Same LP structure across runs               |
| **RNG**                  | Xoshiro256++ with explicit seeding              | Portable, reproducible sequence             |

### 6.2 Potential Reproducibility Issues

**Issue 1: Floating-Point Associativity**

- **Problem**: `(a + b) + c != a + (b + c)` in floating-point. `comm.allreduce()` order may vary between runs.
- **Solution**: Use `comm.reduce(Op::Sum, 0)` (fixed tree order) then `comm.broadcast(0)`.

**Issue 2: OpenMP Reduction Order**

- **Problem**: `#pragma omp parallel for reduction(+:sum)` — summation order depends on thread scheduling.
- **Solution**: Use compensated summation (Kahan) for high precision, or accept small FP differences within documented tolerance.

**Issue 3: Dynamic Work Distribution Order**

- **Analysis**: NOT an issue — each pass's scenarios are determined by global pass ID (not completion order), cut aggregation is commutative, and results are indexed by pass ID.

### 6.3 Verification Protocol

```rust
/// Reproducibility verification (run in debug/test builds)
pub struct ReproducibilityChecker {
    scenario_hash: u64,
    cut_hashes: Vec<u64>,
    lower_bound: f64,
    upper_bound: f64,
}

impl ReproducibilityChecker {
    pub fn verify_scenarios(&mut self, scenarios: &SharedScenarioStorage) {
        let mut hasher = XxHash64::default();
        for pass in 0..scenarios.num_passes {
            for stage in 0..scenarios.num_stages {
                for branch in 0..scenarios.branches_per_stage[stage] as usize {
                    let noise = scenarios.get(pass, stage, branch);
                    for &value in noise {
                        hasher.write(&value.to_le_bytes());
                    }
                }
            }
        }
        self.scenario_hash = hasher.finish();
    }

    pub fn compare(&self, other: &ReproducibilityChecker) -> bool {
        self.scenario_hash == other.scenario_hash
            && self.cut_hashes == other.cut_hashes
            && (self.lower_bound - other.lower_bound).abs() < 1e-10
    }
}
```

### 6.4 Configuration

```json
{
  "reproducibility": {
    "enabled": true,
    "master_seed": 12345,
    "mpi_reduce_deterministic": true,
    "openmp_schedule": "static",
    "fp_tolerance": 1e-12
  }
}
```

**Environment variables for reproducibility:**

```bash
export OMP_SCHEDULE="static"
export OMP_PROC_BIND="close"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Cross-References

- [Communication Patterns](./communication-patterns.md) — persistent collectives, async overlap, performance targets
- [Synchronization](./synchronization.md) — sync points, thread barriers, lock-free cut accumulation
- [Hybrid Parallelism](./hybrid-parallelism.md) — MPI+OpenMP architecture and NUMA configuration
- [Work Distribution](./work-distribution.md) — scenario distribution strategy
- [Cut Management](../01-math/cut-management.md) — cut generation, selection, and storage algorithms
