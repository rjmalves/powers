# T-021: Extract Hybrid Parallelism and Work Distribution Specs

## Epic

Epic 5: HPC Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the MPI+OpenMP hybrid parallelism strategy and work distribution patterns into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/04-hpc/hybrid-parallelism.md` extracted from ARCHITECTURE §20 (20.1-20.6) and DATA_MODEL §6.1
- [ ] `docs/specs/04-hpc/work-distribution.md` extracted from ARCHITECTURE §21 (21.1-21.3) and DATA_MODEL §6.5, §6.10
- [ ] Hybrid parallelism covers: overview, design rationale, parallel configuration, OpenMP FFI bindings, initialization sequence, build configuration
- [ ] Work distribution covers: forward pass distribution, backward pass distribution, implementation details, backward pass computation modes, dynamic work distribution
- [ ] **All MPI references use `ferrompi` crate API** (not raw C FFI): `Communicator`, `init_with_threading(Multiple)`, `Send + Sync` communicators for hybrid MPI+threads
- [ ] **OpenMP FFI bindings remain as C FFI** (ferroMPI does not cover OpenMP — that's a separate FFI layer)
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/04-hpc/hybrid-parallelism.md`
- `docs/specs/04-hpc/work-distribution.md`

## Technical Details

### hybrid-parallelism.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §20, `DATA_MODEL_SPECIFICATION.md` §6.1

- §20.1 Hybrid parallelism overview (MPI between nodes, OpenMP within)
- §20.2 Design rationale (why native OpenMP via FFI, not Rayon)
- §20.3 Parallel configuration (ranks, threads, NUMA)
- §20.4 OpenMP FFI bindings (C interface, Rust unsafe wrappers) — **note: OpenMP remains C FFI; only MPI uses ferroMPI**
- §20.5 Initialization sequence — **replace raw `MPI_Init_thread` C FFI with `ferrompi::init_with_threading(ThreadLevel::Multiple)`**; ferroMPI's `Communicator` is `Send + Sync`, which is exactly what enables hybrid MPI+threads without unsafe sharing
- §20.6 Build configuration — **replace raw MPI C FFI link flags with `ferrompi` Cargo dependency**: `ferrompi = { version = "0.2", features = ["rma", "numa"] }`; OpenMP still requires C link flags (`-fopenmp`)
- §6.1 Architecture overview from data model perspective

**ferroMPI migration notes for extraction**:

- The old docs describe `MPI_Init_thread(MPI_THREAD_MULTIPLE)` via C FFI → replace with `ferrompi::init_with_threading(ThreadLevel::Multiple)`
- The old docs describe manual `MPI_Comm_rank`/`MPI_Comm_size` → replace with `comm.rank()`, `comm.size()` on a `ferrompi::Communicator`
- The old docs describe unsafe `MPI_Comm` handles shared across threads → replace with noting that `ferrompi::Communicator` is `Send + Sync` by design, no unsafe sharing needed

### work-distribution.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §21, `DATA_MODEL_SPECIFICATION.md` §6.5, §6.10

- §21.1 Forward pass distribution (scenarios across ranks)
- §21.2 Backward pass distribution (scenario-based, not state-based)
- §21.3 Work distribution implementation
- §6.5 Backward pass computation modes
- §6.10 Dynamic work distribution

## Definition of Done

Both files created merging content from both source docs, valid cross-references, under 500 lines each.
