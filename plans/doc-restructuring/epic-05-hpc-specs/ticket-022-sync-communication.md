# T-022: Extract Synchronization and Communication Specs

## Epic

Epic 5: HPC Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract synchronization architecture and MPI communication patterns into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/04-hpc/synchronization.md` extracted from ARCHITECTURE §22 (22.1-22.4) and DATA_MODEL §6.3
- [ ] `docs/specs/04-hpc/communication-patterns.md` extracted from ARCHITECTURE §23 (23.1-23.7) and DATA_MODEL §6.2, §6.4, §6.6, §6.8, §6.11-6.13
- [ ] Synchronization covers: sync points, summary, thread sync, lock-free cut aggregation
- [ ] Communication covers: MPI summary, persistent collectives, shared memory, volume analysis, async overlap, performance targets, message structures, hierarchical cut aggregation, intra-node shared memory, performance monitoring, shared memory scenario storage, two-level aggregation, reproducibility guarantees
- [ ] **All MPI references use `ferrompi` crate API**: persistent collectives via `bcast_init`/`allreduce_init`/`PersistentRequest`, shared memory via `SharedWindow<T>`, communicator operations via `Communicator` methods
- [ ] **No raw C FFI for MPI** — all old references to `MPI_*` C functions replaced with ferroMPI equivalents
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/04-hpc/synchronization.md`
- `docs/specs/04-hpc/communication-patterns.md`

## Technical Details

### synchronization.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §22, `DATA_MODEL_SPECIFICATION.md` §6.3

- §22.1 Synchronization points (where ranks must sync)
- §22.2 Synchronization summary table
- §22.3 Thread synchronization within rank (OpenMP barriers, critical sections)
- §22.4 Lock-free cut aggregation (atomic operations, memory ordering)
- §6.3 Synchronization points from data model perspective

### communication-patterns.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §23, `DATA_MODEL_SPECIFICATION.md` §6.2, §6.4, §6.6, §6.8, §6.11-6.13

- §23.1 MPI communication summary
- §23.2 MPI 4.0 persistent collectives — **replace C implementation references with `ferrompi` API**: `comm.bcast_init()`, `comm.allreduce_init()`, `comm.allgather_init()` returning `PersistentRequest<T>`, started with `.start()` and completed with `.wait()`
- §23.3 ~~Rust FFI bindings for persistent collectives~~ — **remove entirely**: ferroMPI provides safe generic bindings, no FFI wrapper code needed. Instead, document that `ferrompi` exposes all persistent collective variants through its safe API
- §23.4 Hybrid shared memory architecture — **replace MPI Windows references with `ferrompi::SharedWindow<T>`** from the `rma` feature: RAII lock guards (`SharedWindowLock`), `get()`/`put()` operations with type safety
- §23.5 Communication volume analysis
- §23.6 Asynchronous communication overlap — reference `PersistentRequest::start()` + computation + `PersistentRequest::wait()` pattern
- §23.7 Communication performance targets
- §6.2 Message structures (FlatBuffers for MPI messages)
- §6.4 Hierarchical cut aggregation — use `comm.allreduce_init()` with custom reduction ops
- §6.6 Intra-node shared memory — **replace "MPI windows" with `ferrompi::SharedWindow<T>` from `rma` feature**
- §6.8 Performance monitoring
- §6.11 Shared memory scenario storage — **replace MPI_Win references with `SharedWindow<T>`**
- §6.12 Two-level cut aggregation
- §6.13 Reproducibility guarantees

**ferroMPI migration notes for extraction**:

- `MPI_Bcast_init` → `comm.bcast_init(&mut buf, root)`
- `MPI_Allreduce_init` → `comm.allreduce_init(&send, &mut recv, Op::Sum)`
- `MPI_Allgather_init` → `comm.allgather_init(&send, &mut recv)`
- `MPI_Start` / `MPI_Wait` → `request.start()` / `request.wait()`
- `MPI_Win_create` / `MPI_Win_lock` → `SharedWindow::new(comm, count)` / `window.lock(rank)`
- `MPI_Win_free` → automatic via `Drop` on `SharedWindow`
- All `MPI_Request` handling → `ferrompi::PersistentRequest<T>` with RAII semantics

Note: This may exceed 500 lines due to the volume of MPI content. If so, split into `communication-patterns.md` (§23, §6.2) and `shared-memory-aggregation.md` (§6.4, §6.6, §6.11-6.13).

## Definition of Done

File(s) created with complete content, valid cross-references, under 500 lines each.
