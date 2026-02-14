---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.1 (MPI Communication Summary)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.2 (MPI 4.0 Persistent Collectives)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.4 (Hybrid Shared Memory Architecture)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.5 (Communication Volume Analysis)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.6 (Asynchronous Communication Overlap)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §23.7 (Communication Performance Targets)"
  - "DATA_MODEL_SPECIFICATION.md §6.2 (Message Structures)"
last_reviewed: null
reviewed_by: null
review_notes: "§23.3 (Rust FFI bindings for persistent collectives) intentionally removed — ferroMPI provides safe generic bindings, no FFI wrapper needed"
change_log:
  - date: 2026-02-14
    description: "Extracted from monolith docs (T-022); migrated all MPI references to ferrompi crate API"
---

# Communication Patterns

## Purpose

This spec defines the MPI communication architecture for POWE.RS: the persistent collective operations used for iterative SDDP, the hybrid shared memory architecture for intra-node efficiency, asynchronous communication overlap strategies, message structures, and communication performance targets. All MPI operations use the `ferrompi` crate — no raw C FFI for MPI.

## 1. MPI Communication Summary

POWE.RS uses a **hierarchical communication architecture** that combines MPI 4.0 persistent collectives for inter-node communication with shared memory for intra-node data sharing. This hybrid approach minimizes latency for the iterative SDDP algorithm.

| Operation                 | When                | Data               | Pattern     | ferroMPI Feature  |
| ------------------------- | ------------------- | ------------------ | ----------- | ----------------- |
| `comm.broadcast()`        | Initialization      | Case data, config  | Root -> All | Standard          |
| `comm.allreduce(Op::Sum)` | Bound computation   | Lower/upper bounds | All -> All  | Persistent        |
| `comm.allgatherv()`       | Cut synchronization | New cuts           | All -> All  | Persistent        |
| `window.fence()`          | Intra-node sync     | FCF updates        | Node-local  | `SharedWindow<T>` |
| `comm.gatherv()`          | Output              | Results            | All -> Root | Standard          |

### 1.1 Why MPI 4.0 Persistent Collectives?

In SDDP, the same communication pattern repeats ~100+ times per training run (once per iteration). Persistent collectives amortize the setup cost:

| Aspect               | Standard Collective       | Persistent Collective       |
| -------------------- | ------------------------- | --------------------------- |
| Setup cost per call  | Full protocol negotiation | Zero (pre-negotiated)       |
| First call latency   | 100-500 us                | 500-1000 us (includes init) |
| Subsequent calls     | 100-500 us                | 10-50 us                    |
| 100 iterations total | 10-50 ms                  | 1.5-6 ms                    |
| **Speedup**          | Baseline                  | **5-10x**                   |

## 2. Persistent Collectives via ferroMPI

All persistent collective operations use the `ferrompi` crate's safe, generic API. The `ferrompi::PersistentRequest<T>` type provides RAII semantics — requests are automatically freed on drop, eliminating resource leaks.

> **Note**: The original architecture specified C FFI wrappers around `MPI_Allgatherv_init`, `MPI_Allreduce_init`, etc. (§23.2-23.3). These are **replaced entirely** by `ferrompi`'s safe Rust API. No C FFI code is needed for MPI persistent collectives.

### 2.1 ferroMPI API Mapping

| C MPI Function       | ferroMPI Equivalent                              |
| -------------------- | ------------------------------------------------ |
| `MPI_Bcast_init`     | `comm.bcast_init(&mut buf, root)`                |
| `MPI_Allreduce_init` | `comm.allreduce_init(&send, &mut recv, Op::Sum)` |
| `MPI_Allgather_init` | `comm.allgather_init(&send, &mut recv)`          |
| `MPI_Start`          | `request.start()`                                |
| `MPI_Wait`           | `request.wait()`                                 |
| `MPI_Test`           | `request.test()`                                 |
| `MPI_Request_free`   | Automatic via `Drop` on `PersistentRequest<T>`   |
| `MPI_Win_create`     | `SharedWindow::new(comm, count)`                 |
| `MPI_Win_lock`       | `window.lock(rank)`                              |
| `MPI_Win_free`       | Automatic via `Drop` on `SharedWindow<T>`        |

### 2.2 Cut Synchronization Manager

The `CutSyncManager` initializes persistent collectives once and reuses them across all SDDP iterations:

```rust
use ferrompi::{Communicator, Op, PersistentRequest};

/// Cut synchronization using ferroMPI persistent collectives
pub struct CutSyncManager {
    /// Persistent allgatherv for cut sharing across ranks
    cut_gather: PersistentRequest<u8>,

    /// Persistent allreduce for bound computation (in-place)
    bound_reduce: PersistentRequest<f64>,

    /// Staging buffers (64-byte aligned for cache efficiency)
    send_buffer: AlignedVec<u8>,
    recv_buffer: AlignedVec<u8>,

    /// Bound reduction buffer: [lower_bound, upper_bound, gap]
    bounds: AlignedVec<f64>,

    world_size: usize,
    world_rank: usize,
}

impl CutSyncManager {
    /// Initialize persistent collectives (called once at startup)
    pub fn new(
        comm: &Communicator,
        max_cuts_per_rank: usize,
        cut_size_bytes: usize,
    ) -> Self {
        let world_size = comm.size();
        let world_rank = comm.rank();

        let send_capacity = max_cuts_per_rank * cut_size_bytes;
        let recv_capacity = send_capacity * world_size;

        let mut send_buffer = AlignedVec::new(send_capacity, 64);
        let mut recv_buffer = AlignedVec::new(recv_capacity, 64);
        let mut bounds = AlignedVec::new(3, 64); // [lower, upper, gap]

        // Initialize persistent allgatherv for cuts
        let cut_gather = comm.allgather_init(
            &send_buffer.as_slice(),
            &mut recv_buffer.as_mut_slice(),
        );

        // Initialize persistent allreduce for bounds (in-place sum)
        let bound_reduce = comm.allreduce_init(
            &bounds.as_slice(),
            &mut bounds.as_mut_slice(),
            Op::Sum,
        );

        Self {
            cut_gather,
            bound_reduce,
            send_buffer,
            recv_buffer,
            bounds,
            world_size,
            world_rank,
        }
    }

    /// Start asynchronous cut gathering (non-blocking)
    pub fn start_gather(&mut self) {
        self.cut_gather.start();
    }

    /// Wait for cut gathering to complete (blocking)
    pub fn wait_gather(&mut self) {
        self.cut_gather.wait();
    }

    /// Start asynchronous bound reduction (non-blocking)
    pub fn start_bounds(&mut self) {
        self.bound_reduce.start();
    }

    /// Wait for bound reduction to complete (blocking)
    pub fn wait_bounds(&mut self) {
        self.bound_reduce.wait();
    }
}

// PersistentRequest<T> implements Drop — no manual cleanup needed
```

## 3. Message Structures

Cut and FCF update messages use `#[repr(C)]` layout for zero-copy MPI transmission:

```rust
/// Cut data for MPI transmission
#[repr(C)]
pub struct CutMessage {
    pub stage_id: u32,
    pub iteration: u32,
    pub forward_pass_idx: u32,
    pub rank_id: u32,
    pub rhs: f64,
    pub state_dimension: u32,
    _padding: u32,
    // Followed by: coefficients[state_dimension]
    // Followed by: state_coefficients[state_dimension]
}

impl CutMessage {
    /// Size in bytes for a given state dimension
    pub fn size_bytes(state_dim: usize) -> usize {
        std::mem::size_of::<Self>() + 2 * state_dim * std::mem::size_of::<f64>()
    }
}

/// FCF update message from master
#[repr(C)]
pub struct FcfUpdateMessage {
    pub stage_id: u32,
    pub iteration: u32,
    pub num_new_cuts: u32,
    pub num_removed_cuts: u32,
    pub num_returned_cuts: u32,
    _padding: u32,
    // Followed by: new_cut_data (variable size)
    // Followed by: removed_cut_ids[num_removed_cuts]
    // Followed by: returned_cut_ids[num_returned_cuts]
}

/// Persistent communication handles
pub struct PersistentComm {
    /// Gather: all ranks -> master (cut data)
    pub cut_gather: PersistentRequest<u8>,

    /// Broadcast: master -> all ranks (FCF updates)
    pub fcf_broadcast: PersistentRequest<u8>,

    /// Allreduce: lower bound computation
    pub bound_allreduce: PersistentRequest<f64>,

    /// Preallocated buffers
    pub cut_send_buffer: Vec<u8>,
    pub cut_recv_buffer: Vec<u8>,
    pub fcf_buffer: Vec<u8>,
}
```

## 4. Hybrid Shared Memory Architecture

POWE.RS uses a **hybrid architecture** that combines `ferrompi::SharedWindow<T>` for intra-node FCF storage with inter-node persistent collectives. This achieves 93% memory efficiency while maintaining good NUMA locality.

| Aspect                   | Fully Replicated | SharedWindow<T> | Hybrid (Selected) |
| ------------------------ | ---------------- | --------------- | ----------------- |
| **Memory per node**      | 168 GB           | 29 GB           | 22.5 GB           |
| **Memory efficiency**    | 12.5%            | 72%             | **93%**           |
| **Read latency (best)**  | 50 ns            | 50 ns           | 50 ns             |
| **Read latency (worst)** | 100 ns           | 200 ns          | 150 ns            |
| **Write throughput**     | 10M cuts/s       | 2M cuts/s       | 10M cuts/s        |
| **NUMA impact**          | None             | High (3-4x)     | Low (interleaved) |
| **Max problem size**     | 2x current       | 8x current      | **8x current**    |

See [Shared Memory and Aggregation](./shared-memory-aggregation.md) for the `SharedWindow<T>` implementation details.

## 5. Communication Volume Analysis

With scenario-based backward pass distribution (see [Work Distribution](./work-distribution.md) Section 2), communication is minimized. Each stage requires only a single `allgatherv` of ~3.26 MB for cut synchronization plus a 64-byte `allreduce` for bound computation.

**Per-iteration communication budget** (reference configuration: 4 ranks, 120 stages, 2000 state dimensions):

| Operation       | Per-stage | Per-iteration (120 stages) |
| --------------- | --------- | -------------------------- |
| Cut allgatherv  | 3.26 MB   | 391 MB                     |
| Bound allreduce | 64 bytes  | 64 bytes                   |
| FCF broadcast   | ~100 KB   | 12 MB                      |
| **Total**       | ~3.36 MB  | ~403 MB                    |

## 6. Asynchronous Communication Overlap

The persistent collective API enables overlapping communication with computation during the backward pass. Local cuts are applied to the FCF while remote cuts are still in transit.

```rust
/// Backward pass with compute/communication overlap
pub fn backward_pass_with_overlap(
    stages: &[Stage],
    scenarios: &[Scenario],
    fcf: &mut HybridFcfStorage,
    sync: &mut CutSyncManager,
) {
    for (stage_idx, stage) in stages.iter().rev().skip(1).enumerate() {
        // Phase 1: Generate cuts (OpenMP parallel over scenarios/outcomes)
        let local_cuts = generate_cuts_parallel(stage, scenarios, fcf);

        // Serialize cuts to persistent send buffer
        let bytes_written = serialize_cuts(&local_cuts, sync.send_buffer_mut());

        // Phase 2: Start async gather while processing local cuts
        sync.start_gather(); // request.start() — non-blocking

        // OVERLAP: Add local cuts to FCF (no communication wait needed)
        for cut in &local_cuts {
            fcf.buffer_cut(cut.to_cut_data());
        }
        fcf.commit_buffered_cuts(stage.id - 1);

        // Phase 3: Wait for remote cuts and integrate
        sync.wait_gather(); // request.wait() — blocks until complete

        // Deserialize and add remote cuts
        let recv_buffer = sync.recv_buffer();
        for rank in 0..sync.world_size() {
            if rank != sync.world_rank() {
                let offset = rank * sync.cut_size * sync.max_cuts;
                let cuts = deserialize_cuts(&recv_buffer[offset..]);
                for cut in cuts {
                    fcf.buffer_cut(cut);
                }
            }
        }
        fcf.commit_buffered_cuts(stage.id - 1);

        // Intra-node fence ensures all ranks see the updates
        fcf.sync_intra_node(); // window.fence()
    }
}
```

**Key pattern**: `request.start()` -> computation -> `request.wait()`. The `PersistentRequest<T>` type tracks in-flight status and prevents use-after-free via RAII.

## 7. Communication Performance Targets

| Metric                             | Target           | Measurement Method                  |
| ---------------------------------- | ---------------- | ----------------------------------- |
| Cut gather latency (4 ranks)       | < 10 ms          | `MPI_Wtime` around `wait_gather()`  |
| Bound reduce latency               | < 200 us         | `MPI_Wtime` around `wait_bounds()`  |
| Communication fraction             | < 15% of total   | `comm_time / (comm_time + compute)` |
| Overlap efficiency                 | > 60%            | Overlapped work / total comm time   |
| Persistent collective amortization | > 5x vs standard | Compare first vs subsequent calls   |

## Cross-References

- [Hybrid Parallelism](./hybrid-parallelism.md) — MPI+OpenMP initialization and `ferrompi` setup
- [Synchronization](./synchronization.md) — sync points, thread barriers, lock-free cut accumulation
- [Work Distribution](./work-distribution.md) — scenario distribution and dynamic dispatch
- [Shared Memory and Aggregation](./shared-memory-aggregation.md) — intra-node shared memory, hierarchical/two-level cut aggregation, reproducibility
