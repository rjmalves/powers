# T-024: Extract SLURM Deployment Spec

## Epic

Epic 5: HPC Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract SLURM job scripts, deployment patterns, and performance monitoring into a focused spec.

## Acceptance Criteria

- [ ] `docs/specs/04-hpc/slurm-deployment.md` extracted from ARCHITECTURE Appendix A (A.1-A.3), Appendix B (B.1-B.2), and DATA_MODEL §6.9
- [ ] Covers: single-node job script, multi-node production job, job array for parameter studies, key performance counters, timing breakdown, HPC implementation requirements
- [ ] All SLURM script examples preserved verbatim
- [ ] **SLURM topology detection references use `ferrompi::slurm` helpers** from the `numa` feature instead of manual env var parsing in scripts/code
- [ ] **SLURM scripts themselves remain as-is** (they're shell scripts, not Rust) — the ferroMPI migration applies to how the Rust code _reads_ SLURM topology, not how SLURM _sets up_ the environment
- [ ] Under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/04-hpc/slurm-deployment.md`

## Technical Details

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` Appendix A, Appendix B, `DATA_MODEL_SPECIFICATION.md` §6.9

- Appendix A.1 Single-node job (development/testing) — complete SLURM script
- Appendix A.2 Multi-node production job — complete SLURM script with NUMA settings
- Appendix A.3 Job array for parameter studies — parameterized job submission
- Appendix B.1 Key performance counters — what to measure
- Appendix B.2 Timing breakdown — expected time distribution across phases
- §6.9 HPC implementation requirements (critical) — what must be true for correct HPC operation
- Also include the SLURM template from DATA_MODEL §6.7 if not already in memory-architecture.md

**ferroMPI integration note**: The SLURM scripts set up the environment (srun, ntasks, cpus-per-task, etc.), but the Rust binary launched by srun uses `ferrompi::slurm` helpers to _read_ that topology safely. Add a note in the spec explaining this boundary:

- SLURM scripts → shell-level config (preserved as-is)
- Rust startup code → uses `ferrompi::init_with_threading(Multiple)` and `ferrompi::slurm::local_rank()` etc. to detect placement
- Reference `hybrid-parallelism.md` and `memory-architecture.md` for how ferroMPI is used at initialization

## Definition of Done

File created with complete deployment content, valid cross-references, under 500 lines.
