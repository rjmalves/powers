---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §21.1 (Forward Pass Distribution)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §21.2 (Backward Pass Distribution)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §21.3 (Work Distribution Implementation)"
  - "DATA_MODEL_SPECIFICATION.md §6.5 (Backward Pass Computation Modes)"
  - "DATA_MODEL_SPECIFICATION.md §6.10 (Dynamic Work Distribution)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): The same thread that runs forward pass K must also run backward pass K (thread-trajectory affinity). This preserves cache locality for solver basis and scenario data, and eliminates cross-thread data handoff. Within the backward pass, each thread solves its branching scenarios sequentially to keep the solver state hot. Validate this pattern against the existing dynamic dispatch design during P4 review. See sddp-algorithm.md §3.4."
---

# Work Distribution

## Purpose

This spec defines how POWE.RS distributes computational work across MPI ranks and OpenMP threads: forward pass dynamic dispatch, backward pass scenario-based distribution with sequential and pipelined modes, and the rank 0 dispatcher/worker architecture. It merges the distribution patterns from the architecture specification with the computation modes and dynamic dispatch design from the data model specification.

## 1. Forward Pass Distribution

Forward passes are distributed dynamically from rank 0 to worker ranks. Each rank processes batches of forward passes in parallel using OpenMP threads (one complete forward pass per thread per batch).

**Architecture Overview:**

- **Rank 0 (Dispatcher + Worker):** Maintains a work queue of forward pass indices `[0, 1, 2, ..., N-1]`. A dedicated dispatcher thread handles MPI requests from workers while compute threads process local batches via OpenMP.
- **Workers (Ranks 1..N-1):** Send `READY` to rank 0, receive batch assignment, process batch with OpenMP parallel for.
- **Batch Size:** = `num_threads` (one forward pass per thread, full utilization)
- **Batch Processing:** `#pragma omp parallel for schedule(dynamic, 1)`

## 2. Backward Pass Distribution

### 2.1 Why Scenario-Based (Not State-Based)?

| Metric                          | Scenario-Based | State-Based | Difference              |
| ------------------------------- | -------------- | ----------- | ----------------------- |
| **Forward->Backward sync**      | 0 MB (zero)    | 800 MB      | Infinitely better       |
| **Warm-start applicability**    | 95%            | 5%          | 19x better              |
| **LP solve time per iteration** | 16 min         | 116 min     | 7x faster               |
| **Communication per iteration** | 388 MB         | 1,188 MB    | 3x less                 |
| **Implementation complexity**   | Simple         | Complex     | Much simpler            |
| **NUMA locality**               | Perfect        | Poor        | Perfect vs cache misses |

**Mathematical Validity**: The cuts produced are **mathematically identical** regardless of distribution strategy. A cut computed at state $x$ captures the marginal value of storage changes. Whether that state came from scenario 5 or scenario 50 doesn't affect the cut's validity for approximating the cost-to-go function.

The state-based approach's only theoretical advantage (deduplication of identical states) saves ~2% of LP solves — completely overwhelmed by the 7x penalty from losing warm-start.

### 2.2 Backward Pass Computation Modes

At iteration $k$, the backward pass computes cuts for stage $t$ by solving subproblems at stage $t+1$ and extracting dual information. The key design question is: **which approximation of the future cost function should stage $t+1$ use?**

**Option A — Sequential (uses $V_{t+1}^k$):**

$$Q_t^k(x_{t-1}, \omega_t) = \min_{x_t} \left\{ c_t^\top x_t + \mathcal{Q}_{t+1}^{k}(x_t) \right\}$$

The superscript $k$ means we use the freshly-computed approximation from the current iteration. This requires that stage $t+1$'s cuts have already been computed before we compute stage $t$.

**Option B — Pipelined (uses $V_{t+1}^{k-1}$):**

$$Q_t^k(x_{t-1}, \omega_t) = \min_{x_t} \left\{ c_t^\top x_t + \mathcal{Q}_{t+1}^{k-1}(x_t) \right\}$$

The superscript $k-1$ means we use the approximation from the previous iteration. This allows stages to be computed in parallel or with overlapped communication.

> **Mathematical Validity**: Both approaches produce valid cuts that are supporting hyperplanes of the true value function. The difference is only in tightness — sequential cuts are generally tighter because they incorporate more recent information.
>
> _"It doesn't matter what order we visit the nodes to generate these cuts for. For example, we could compute them all in parallel, using the current approximations of V^K_i."_ — SDDP.jl Documentation

#### Sequential Mode (Default)

The standard backward pass walks stages from $T$ to $1$. At each stage, it waits for the downstream stage's cuts to be computed and broadcast before proceeding.

**Sequential Backward Pass Timeline:**

| Time   | Stage T | Stage T-1 | Stage T-2 |
| ------ | ------- | --------- | --------- |
| 0-20   | Compute | Wait      | Wait      |
| 20-30  | Gather  | Wait      | Wait      |
| 30-40  | Bcast   | Wait      | Wait      |
| 40-60  | Done    | Compute   | Wait      |
| 60-70  | Done    | Gather    | Wait      |
| 70-80  | Done    | Bcast     | Wait      |
| 80-100 | Done    | Done      | Compute   |

> **Key**: Barrier between each stage ensures $V_{t+1}^k$ is available before computing $V_t^k$

```rust
/// Sequential backward pass (default mode)
///
/// Produces tighter cuts by using freshly-computed V_{t+1}^k at each stage.
/// This is the recommended mode and matches SDDP.jl's default behavior.
pub fn backward_pass_sequential(
    iteration: u32,
    forward_results: &ForwardResults,
    fcf: &mut FutureCostFunction,
    comm: &ferrompi::Communicator,
    num_stages: u32,
) {
    for stage in (1..num_stages).rev() {
        // Step 1: Compute cuts for current stage
        // Uses V_{stage+1}^k which was just updated in the previous loop iteration
        let local_cuts = compute_stage_cuts(stage, forward_results, fcf);

        // Step 2: Gather cuts from all ranks (hierarchical aggregation)
        let all_cuts = hierarchical_gather(&local_cuts, comm);

        // Step 3: Master selects/aggregates cuts
        let selected_cuts = if comm.rank() == MASTER_RANK {
            cut_selection(all_cuts.unwrap(), &fcf.selection_config)
        } else {
            Vec::new()
        };

        // Step 4: Broadcast selected cuts to all ranks (BLOCKING)
        let broadcast_cuts = comm.broadcast(&selected_cuts, MASTER_RANK);

        // Step 5: All ranks apply cuts to their local FCF copy
        // This updates V_stage^k, ready for stage-1 to use
        fcf.apply_cuts(stage, &broadcast_cuts);

        // Implicit barrier: all ranks synchronized before next stage
    }
}
```

**Advantages:**

- Tighter cuts (uses most recent $V_{t+1}^k$)
- Faster convergence (fewer iterations to optimality gap)
- Simpler implementation (no overlapping state to manage)
- Default in SDDP.jl and most production implementations

**Disadvantages:**

- Synchronization barriers at each stage
- Cannot overlap computation with communication

#### Pipelined Mode (Future Enhancement)

> **Status**: Documented for future implementation. Not available in v2.0.

The pipelined backward pass overlaps computation with communication by using $V_{t+1}^{k-1}$ (previous iteration's approximation) instead of $V_{t+1}^k$.

**Pipelined Backward Pass Timeline:**

| Time  | Stage T | Stage T-1 | Stage T-2 | Comm T | Comm T-1 |
| ----- | ------- | --------- | --------- | ------ | -------- |
| 0-10  | Compute | -         | -         | -      | -        |
| 10-20 | Compute | Compute   | -         | -      | -        |
| 20-25 | Done    | Compute   | Compute   | -      | -        |
| 25-30 | Done    | Compute   | Compute   | Ibcast | -        |
| 30-35 | Done    | Irecv     | Compute   | Ibcast | -        |
| 35-40 | Done    | Irecv     | Compute   | Ibcast | Ibcast   |
| 40-50 | Done    | Done      | Done      | Done   | Ibcast   |

> **Key Insight**: Stage T-1 uses $V_T^{k-1}$ (from previous iteration), **NOT** $V_T^k$. This allows overlapped execution but produces looser cuts.

The pipelined implementation uses `ferrompi::PersistentComm` with non-blocking broadcasts (`comm.ibroadcast()`) to overlap communication with computation. Each stage starts computing immediately using $V_{t+1}^{k-1}$ while the previous stage's cut broadcast completes in the background. A `pending_broadcast: Option<(u32, ferrompi::Request)>` tracks in-flight operations, calling `req.wait()` only when the result is needed.

#### Trade-off Analysis

| Aspect                 | Sequential (Default)            | Pipelined (Future)                   |
| ---------------------- | ------------------------------- | ------------------------------------ |
| Cut source             | $V_{t+1}^k$ (current iteration) | $V_{t+1}^{k-1}$ (previous iteration) |
| Cut quality            | Tighter                         | Looser                               |
| Iterations to converge | Fewer                           | More                                 |
| Time per iteration     | Longer (barriers)               | Shorter (overlapped)                 |
| Implementation         | Simple                          | Complex (async state)                |
| MPI primitives         | `comm.broadcast()`              | `comm.ibroadcast()`, `req.wait()`    |

**Latency Reduction Estimate (Pipelined):**

| Stages | Barrier Overhead (Sequential) | Pipelined Overhead | Reduction |
| ------ | ----------------------------- | ------------------ | --------- |
| 60     | ~60 x 5ms = 300ms             | ~60ms (overlapped) | 80%       |
| 120    | ~120 x 5ms = 600ms            | ~100ms             | 83%       |
| 240    | ~240 x 5ms = 1.2s             | ~180ms             | 85%       |

**When to consider pipelined mode (future):**

- Very large stage counts (>100 stages)
- High network latency between nodes
- When iteration count is less important than wall-clock time
- After profiling shows backward pass communication is the bottleneck

#### Configuration

```json
{
  "training": {
    "backward_pass": {
      "mode": "sequential"
    }
  }
}
```

| Mode         | Description                                            | Status       |
| ------------ | ------------------------------------------------------ | ------------ |
| `sequential` | Standard backward pass using $V_{t+1}^k$. **Default.** | Implemented  |
| `pipelined`  | Overlapped backward pass using $V_{t+1}^{k-1}$.        | **DEFERRED** |

## 3. Work Distribution Implementation

### 3.1 Static Distribution Utilities

```rust
/// Work distribution utilities
pub struct WorkDistributor {
    rank: usize,
    world_size: usize,
}

impl WorkDistributor {
    /// Distribute N scenarios across ranks (static, balanced)
    /// Returns the range of scenario indices for this rank
    pub fn distribute_scenarios(&self, total_scenarios: usize) -> Range<usize> {
        let base = total_scenarios / self.world_size;
        let remainder = total_scenarios % self.world_size;

        let start = self.rank * base + self.rank.min(remainder);
        let count = base + if self.rank < remainder { 1 } else { 0 };

        start..start + count
    }

    /// Get counts and displacements for MPI collective operations
    pub fn get_distribution_info(&self, total: usize) -> (Vec<i32>, Vec<i32>) {
        let base = total / self.world_size;
        let remainder = total % self.world_size;

        let counts: Vec<i32> = (0..self.world_size)
            .map(|r| (base + if r < remainder { 1 } else { 0 }) as i32)
            .collect();

        let displs: Vec<i32> = counts.iter()
            .scan(0, |acc, &c| {
                let d = *acc;
                *acc += c;
                Some(d)
            })
            .collect();

        (counts, displs)
    }
}
```

### 3.2 Backward Pass Execution with Scenario-Based Distribution

```rust
/// Backward pass execution with scenario-based distribution
pub fn execute_backward_pass(
    ctx: &ParallelContext,
    scenarios: &[Scenario],
    fcf: &mut Fcf,
    solver_pool: &mut SolverPool,
) -> Vec<Cut> {
    let my_scenarios = ctx.distribute_scenarios(scenarios.len());
    let mut all_cuts = Vec::new();

    // Process stages in reverse order (T-1, T-2, ..., 1)
    for stage in (1..ctx.n_stages).rev() {
        let mut stage_cuts = Vec::new();

        // Process each scenario assigned to this rank
        for scenario_idx in my_scenarios.clone() {
            let scenario = &scenarios[scenario_idx];
            let state = scenario.state_at_stage(stage);

            // Parallel over noise outcomes using OpenMP
            let outcome_cuts = parallel_reduce_cuts(
                0..ctx.n_outcomes,
                |outcome_idx, thread_id| {
                    // Get thread-local solver (warm-started from previous solve)
                    let solver = solver_pool.get_solver(thread_id);

                    // Build and solve outcome LP
                    let noise = ctx.get_noise(stage, outcome_idx);
                    let result = solve_outcome_lp(solver, state, stage, noise, fcf);

                    // Return partial cut contribution
                    CutContribution {
                        alpha: result.objective * ctx.outcome_probability(outcome_idx),
                        beta: result.dual_state.scale(ctx.outcome_probability(outcome_idx)),
                    }
                },
            );

            // Aggregate outcomes into single cut for this (scenario, stage)
            let cut = Cut::from_contributions(stage - 1, scenario_idx, &outcome_cuts);
            stage_cuts.push(cut);
        }

        // Synchronize cuts across all ranks for this stage
        let global_cuts = ctx.sync_cuts_allgatherv(&stage_cuts);

        // Add all cuts to FCF (for use in earlier stages)
        for cut in &global_cuts {
            fcf.add_cut(cut.target_stage, cut.clone());
        }

        all_cuts.extend(global_cuts);
    }

    all_cuts
}

/// Parallel reduction collecting cut contributions using OpenMP
fn parallel_reduce_cuts<F>(
    range: Range<usize>,
    f: F,
) -> Vec<CutContribution>
where
    F: Fn(usize, usize) -> CutContribution + Sync,
{
    // Use OpenMP parallel for with thread-local accumulation
    let n_threads = omp::get_max_threads();
    let mut thread_results: Vec<Vec<CutContribution>> =
        (0..n_threads).map(|_| Vec::new()).collect();

    parallel_for_dynamic(range, 1, |outcome_idx, thread_id| {
        let contribution = f(outcome_idx, thread_id);
        // Thread-local append (no contention)
        thread_results[thread_id].push(contribution);
    });

    // Flatten thread-local results
    thread_results.into_iter().flatten().collect()
}
```

## 4. Dynamic Work Distribution

### 4.1 Rank 0 Bottleneck Mitigation

> **Issue**: Rank 0 acts as both dispatcher and worker. If dispatching blocks computation or vice versa, efficiency suffers.

**Solution: Dedicated Dispatcher Thread with Non-Blocking MPI**

```rust
/// Rank 0 execution model with dedicated dispatcher
pub struct Rank0Executor {
    work_queue: Mutex<VecDeque<u32>>,
    local_batches: ArrayQueue<Vec<u32>>,
    batch_size: usize,
    /// ferrompi communicator (Send + Sync, safe across threads)
    comm: ferrompi::Communicator,
}

impl Rank0Executor {
    pub fn execute_forward_pass(&self, total_passes: u32) -> ForwardPassResults {
        // Fill work queue
        { self.work_queue.lock().extend(0..total_passes); }

        // Spawn dispatcher thread (uses 1 core)
        let dispatcher_handle = std::thread::spawn(|| self.dispatcher_loop());

        // Main thread coordinates OpenMP compute (uses remaining cores)
        let results = self.compute_loop();
        dispatcher_handle.join().unwrap();
        results
    }

    /// Dispatcher thread: handle MPI requests from workers
    fn dispatcher_loop(&self) {
        let mut active_workers = self.comm.size() - 1;

        while active_workers > 0 {
            if let Some(status) = self.comm.iprobe_any(TAG_READY) {
                let worker = status.source();
                self.comm.recv::<u32>(worker, TAG_READY);

                if let Some(batch) = self.try_pop_batch() {
                    self.comm.send(&(batch.len() as u32), worker, TAG_BATCH_SIZE);
                    self.comm.send_slice(&batch, worker, TAG_BATCH_DATA);
                } else {
                    self.comm.send(&0u32, worker, TAG_BATCH_SIZE);
                    active_workers -= 1;
                }
            } else {
                if let Some(batch) = self.try_pop_batch() {
                    let _ = self.local_batches.push(batch);
                }
                std::thread::yield_now();
            }
        }
    }
}
```

### 4.2 Worker Implementation

```rust
/// Worker rank execution model
pub struct WorkerExecutor {
    batch_size: usize,
    comm: ferrompi::Communicator,
    workspaces: WorkspaceManager,
}

impl WorkerExecutor {
    pub fn execute_forward_pass(&mut self) -> ForwardPassResults {
        let mut results = ForwardPassResults::new();
        loop {
            self.comm.send(&1u32, 0, TAG_READY);
            let batch_size: u32 = self.comm.recv(0, TAG_BATCH_SIZE);
            if batch_size == 0 { break; }

            let mut batch = vec![0u32; batch_size as usize];
            self.comm.recv_slice(&mut batch, 0, TAG_BATCH_DATA);
            results.merge(self.process_batch_openmp(&batch));
        }
        results
    }
}
```

### 4.3 Thread Allocation Strategy

| Node Cores | Rank 0 Threads             | Worker Rank Threads | Efficiency Loss |
| ---------- | -------------------------- | ------------------- | --------------- |
| 64         | 1 dispatcher + 63 compute  | 64 compute          | 1.5%            |
| 128        | 1 dispatcher + 127 compute | 128 compute         | 0.8%            |
| 16         | 1 dispatcher + 15 compute  | 16 compute          | 6%              |

**Recommendation**: Use dedicated dispatcher thread for simplicity. The efficiency loss is negligible at scale.

### 4.4 MPI Message Protocol

| Tag | Name             | Direction        | Data                                      |
| --- | ---------------- | ---------------- | ----------------------------------------- |
| 1   | `TAG_READY`      | Worker -> Rank 0 | uint32 (ignored, triggers receive)        |
| 2   | `TAG_BATCH_SIZE` | Rank 0 -> Worker | uint32 (batch_size, or 0 for DONE)        |
| 3   | `TAG_BATCH_DATA` | Rank 0 -> Worker | uint32[batch_size] (forward pass indices) |

## Cross-References

- [Hybrid Parallelism](./hybrid-parallelism.md) — MPI+OpenMP architecture, initialization, and build configuration
- [SDDP Algorithm](../01-math/sddp-algorithm.md) — forward/backward pass algorithmic structure
- [Cut Management](../01-math/cut-management.md) — cut generation, selection, and storage
- [Design Principles](../00-overview/design-principles.md) — distributed I/O and reproducibility goals
