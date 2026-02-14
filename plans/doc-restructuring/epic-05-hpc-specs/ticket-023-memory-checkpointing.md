# T-023: Extract Memory Architecture and Checkpointing Specs

## Epic

Epic 5: HPC Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract memory architecture (budget, NUMA, pools) and checkpointing/fault tolerance into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/04-hpc/memory-architecture.md` extracted from ARCHITECTURE §24 (24.1-24.4) and DATA_MODEL §6.7
- [ ] `docs/specs/04-hpc/checkpointing.md` extracted from ARCHITECTURE §25 (25.1-25.3, 25.5) and §26 (26.1-26.4)
- [ ] Memory covers: budget overview, layout strategy, NUMA-aware allocation, memory pools, NUMA-aware memory management
- [ ] Checkpointing covers: strategy, implementation, warm-start, policy persistence (components, format, compatibility, use cases), output directory structure, policy output, simulation summary, performance logging
- [ ] **NUMA-aware allocation references use `ferrompi::slurm` helpers** from the `numa` feature for topology detection (node-local rank, NUMA domain mapping) instead of manual SLURM env var parsing
- [ ] **Shared memory windows in memory architecture reference `ferrompi::SharedWindow<T>`** from the `rma` feature where applicable (intra-node shared data)
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/04-hpc/memory-architecture.md`
- `docs/specs/04-hpc/checkpointing.md`

## Technical Details

### memory-architecture.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §24, `DATA_MODEL_SPECIFICATION.md` §6.7

- §24.1 Memory budget overview (how much memory per rank, breakdown by component)
- §24.2 Memory layout strategy (struct of arrays, cache-friendly access)
- §24.3 NUMA-aware allocation — **replace manual SLURM env var parsing (`SLURM_LOCALID`, `SLURM_NODELIST`, etc.) with `ferrompi::slurm` helpers** from the `numa` feature: `ferrompi::slurm::local_rank()`, `ferrompi::slurm::node_count()`, NUMA domain detection. ferroMPI provides safe, typed access to SLURM topology information
- §24.4 Memory pool for temporary allocations (arena-style allocation for LP vectors)
- §6.7 NUMA-aware memory management — **reference `ferrompi::slurm` for topology queries** and `SharedWindow<T>` from the `rma` feature for intra-node shared memory that avoids per-rank duplication of read-only data

**ferroMPI migration notes for extraction**:

- Manual `std::env::var("SLURM_LOCALID")` parsing → `ferrompi::slurm::local_rank()`
- Manual `std::env::var("SLURM_NNODES")` parsing → `ferrompi::slurm::node_count()`
- Manual NUMA domain calculation → `ferrompi::slurm` helpers for NUMA-aware rank placement
- MPI shared memory windows for read-only data → `SharedWindow<T>::new(comm, count)` with `rma` feature

### checkpointing.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §25-26

- §25.1 Checkpoint strategy (when and what to checkpoint)
- §25.2 Checkpoint implementation (file format, atomicity)
- §25.3 Warm-start from checkpoint (resume training)
- §25.5 Policy persistence architecture (components, file format, compatibility validation, use cases)
- §26.1 Output directory structure
- §26.2 Policy output (cuts, metadata)
- §26.3 Simulation summary output
- §26.4 Performance logging

## Definition of Done

Both files created with complete content, valid cross-references, under 500 lines each.
