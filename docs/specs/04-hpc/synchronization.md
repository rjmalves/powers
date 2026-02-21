---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.1 (Synchronization Points)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.2 (Synchronization Summary)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.3 (Thread Synchronization Within Rank)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §22.4 (Lock-Free Cut Aggregation)"
  - "DATA_MODEL_SPECIFICATION.md §6.3 (Synchronization Points)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from monolith docs (T-022)"
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): The backward pass has a hard synchronization barrier at each stage boundary — all threads must complete cut construction at stage t before proceeding to stage t-1. The forward pass has no such per-stage barrier (fully parallel trajectories). Validate that existing sync point design accounts for this asymmetry during P4 review. See sddp-algorithm.md §3.4."
---

# Synchronization

## Purpose

This spec defines the synchronization architecture for POWE.RS: where MPI ranks synchronize during SDDP iterations, how threads coordinate within a rank using spin barriers and cache-aligned buffers, and how cut aggregation is performed lock-free to maximize throughput.

## 1. Synchronization Points

With scenario-based distribution, synchronization is **minimal and well-defined**:

**Key Observation**: There is **NO** synchronization between forward and backward pass. Each rank seamlessly transitions from forward to backward using the states it already computed. This eliminates the 800 MB state-gathering step that would be required by state-based distribution.

### 1.1 Synchronization Summary

The following table summarizes all synchronization points in a single SDDP iteration. MPI operations reference the `ferrompi` crate API — see [Communication Patterns](./communication-patterns.md) for implementation details.

| Phase Transition        | Synchronization               | Data Volume | Latency  |
| ----------------------- | ----------------------------- | ----------- | -------- |
| Init -> Forward         | `comm.barrier()`              | 0           | ~1 ms    |
| Forward -> Backward     | **None**                      | 0           | 0        |
| Backward stage t -> t-1 | `comm.allgatherv()`           | 3.26 MB     | ~5-10 ms |
| Backward -> Convergence | `comm.allreduce(Op::Sum)`     | 64 bytes    | ~100 us  |
| Iteration k -> k+1      | None (or barrier for logging) | 0           | ~1 ms    |

### 1.2 Data Model Synchronization View

From the data model perspective, synchronization points map to specific data exchange operations:

| Phase              | Sync Type                 | Data Size      | Frequency          |
| ------------------ | ------------------------- | -------------- | ------------------ |
| Forward pass end   | None                      | -              | Per iteration      |
| Backward per-stage | `comm.gather()`           | ~50 KB x ranks | Per stage          |
| FCF update         | `comm.broadcast()`        | ~100 KB        | Per stage          |
| Lower bound        | `comm.allreduce(Op::Sum)` | 8 bytes        | Per iteration      |
| Checkpointing      | `comm.barrier()`          | -              | Every N iterations |

## 2. Thread Synchronization (Within Rank)

Thread synchronization within a rank uses lightweight primitives optimized for the SDDP access pattern: many small LP solves with thread-local accumulation followed by a brief merge phase.

### 2.1 Outcome Synchronization

```rust
/// Thread synchronization for outcome processing
pub struct OutcomeSync {
    /// Spin barrier (faster than OpenMP barrier for small teams)
    barrier: SpinBarrier,

    /// Per-thread cut contribution buffers (cache-line aligned)
    thread_contributions: Vec<ThreadLocal<Vec<CutContribution>>>,
}

impl OutcomeSync {
    pub fn new(num_threads: usize) -> Self {
        Self {
            barrier: SpinBarrier::new(num_threads),
            thread_contributions: (0..num_threads)
                .map(|_| ThreadLocal::new(Vec::with_capacity(32)))
                .collect(),
        }
    }

    /// Add contribution from current thread (lock-free)
    #[inline]
    pub fn add_contribution(&self, thread_id: usize, contrib: CutContribution) {
        self.thread_contributions[thread_id].get_mut().push(contrib);
    }

    /// Barrier and collect all contributions
    pub fn barrier_and_collect(&self) -> Vec<CutContribution> {
        self.barrier.wait();

        // Only thread 0 collects
        if omp::get_thread_num() == 0 {
            self.thread_contributions.iter()
                .flat_map(|tc| tc.get().drain(..))
                .collect()
        } else {
            self.barrier.wait(); // Wait for collection
            Vec::new()
        }
    }
}
```

### 2.2 Spin Barrier

A custom spin barrier is used instead of OpenMP barriers for small thread counts. The spin barrier avoids the overhead of the OpenMP runtime's general-purpose barrier implementation.

```rust
/// Spin barrier (faster than OpenMP barrier for small thread counts)
pub struct SpinBarrier {
    count: std::sync::atomic::AtomicUsize,
    generation: std::sync::atomic::AtomicUsize,
    num_threads: usize,
}

impl SpinBarrier {
    pub fn new(num_threads: usize) -> Self {
        Self {
            count: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicUsize::new(0),
            num_threads,
        }
    }

    pub fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        let arrived = self.count.fetch_add(1, Ordering::AcqRel) + 1;

        if arrived == self.num_threads {
            self.count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            while self.generation.load(Ordering::Acquire) == gen {
                std::hint::spin_loop();
            }
        }
    }
}
```

**Design Rationale**: The spin barrier uses `Acquire`/`Release` memory ordering rather than `SeqCst` for minimal overhead. The generation counter prevents ABA problems. For thread counts <= 16 (typical per-NUMA-node), spin-waiting outperforms OS-level barriers due to avoiding context switch overhead.

## 3. Lock-Free Cut Aggregation

Cut aggregation during the backward pass uses a lock-free architecture where each thread writes to its own cache-line-aligned buffer. There is **no contention** during the hot loop — merging happens only after the parallel region completes.

### 3.1 Cut Accumulator

```rust
/// Lock-free cut accumulation for backward pass
/// Each thread writes to its own buffer, then buffers are merged after barrier
pub struct CutAccumulator {
    /// Per-thread cut buffers (cache-line aligned, no contention)
    thread_cuts: Vec<ThreadLocal<Vec<Cut>>>,
    num_threads: usize,
}

impl CutAccumulator {
    pub fn new(num_threads: usize) -> Self {
        Self {
            thread_cuts: (0..num_threads)
                .map(|_| ThreadLocal::new(Vec::with_capacity(64)))
                .collect(),
            num_threads,
        }
    }

    /// Add cut from current thread (completely lock-free)
    #[inline]
    pub fn add_cut(&self, thread_id: usize, cut: Cut) {
        self.thread_cuts[thread_id].get_mut().push(cut);
    }

    /// Collect all cuts after OpenMP parallel region ends
    /// Must be called from single thread (outside parallel region)
    pub fn collect_and_clear(&mut self) -> Vec<Cut> {
        self.thread_cuts.iter_mut()
            .flat_map(|buf| std::mem::take(buf.get_mut()))
            .collect()
    }

    /// Get total cut count without clearing
    pub fn total_cuts(&self) -> usize {
        self.thread_cuts.iter()
            .map(|buf| buf.get().len())
            .sum()
    }
}
```

### 3.2 Cache-Line Alignment

The `ThreadLocal<T>` wrapper ensures each thread's buffer resides on a separate cache line (64 bytes), eliminating false sharing:

```rust
/// Thread-local storage with cache-line alignment (prevents false sharing)
#[repr(C, align(64))]
pub struct ThreadLocal<T> {
    data: T,
    _padding: [u8; 64 - std::mem::size_of::<T>() % 64],
}
```

**Why this matters**: Without cache-line alignment, adjacent thread buffers may share a cache line. When one thread writes (even to its own buffer), the shared cache line is invalidated for all other threads, causing expensive cache coherency traffic. With 16 threads, this can cause a 3-5x slowdown in the accumulation phase.

### 3.3 Aggregation Flow

The overall aggregation flow for a single backward pass stage:

1. **Parallel Phase** (OpenMP): Each thread solves outcome LPs and calls `add_cut()` — pure thread-local writes, zero contention
2. **Barrier**: Threads synchronize via `SpinBarrier`
3. **Collect Phase** (Single-threaded): Thread 0 calls `collect_and_clear()` to merge all buffers
4. **MPI Phase**: Merged cuts are sent to the [communication layer](./communication-patterns.md) for inter-rank aggregation

## Cross-References

- [Hybrid Parallelism](./hybrid-parallelism.md) — MPI+OpenMP architecture and initialization
- [Work Distribution](./work-distribution.md) — forward/backward pass scenario distribution
- [Communication Patterns](./communication-patterns.md) — MPI persistent collectives and async overlap
- [Shared Memory and Aggregation](./shared-memory-aggregation.md) — intra-node shared memory, two-level cut aggregation, reproducibility
