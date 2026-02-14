# T-016: Extract CLI, Lifecycle, and Input Loading Specs

## Epic

Epic 4: Architecture Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the program entrypoint/CLI design and input loading pipeline into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/03-architecture/cli-and-lifecycle.md` extracted from ARCHITECTURE §1 (1.1-1.4), §2 (2.1-2.3), §3 (3.1-3.2)
- [ ] `docs/specs/03-architecture/input-loading-pipeline.md` extracted from ARCHITECTURE §4 (4.1-4.3), §5 (5.1-5.3), §7 (7.1-7.4)
- [ ] CLI spec covers: design philosophy, invocation pattern, CLI interface, exit codes, execution phases, conditional execution, config resolution, scheduler integration
- [ ] Loading spec covers: loading architecture, file sequence, loader interface, dependency graph, conditional loading, sparse time-series, broadcast strategy, serialization for broadcast, parallel policy loading, memory layout after broadcast
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/03-architecture/cli-and-lifecycle.md`
- `docs/specs/03-architecture/input-loading-pipeline.md`

## Technical Details

### cli-and-lifecycle.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §1-3

- §1.1 Design philosophy
- §1.2 Invocation pattern (standard, SLURM, validation-only)
- §1.3 CLI interface — all flags and subcommands
- §1.4 Exit codes
- §2.1 Phase diagram
- §2.2 Phase responsibilities
- §2.3 Conditional execution
- §3.1 Configuration hierarchy
- §3.2 Scheduler integration

### input-loading-pipeline.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §4-5, §7

- §4.1 Loading architecture overview
- §4.2 File loading sequence
- §4.3 Loader interface (trait design)
- §5.1 Dependency graph for file loading
- §5.2 Conditional loading (skip optional files)
- §5.3 Sparse time-series handling
- §7.1 Broadcast strategy (rank 0 loads, broadcasts to others)
- §7.2 Serialization for broadcast
- §7.3 Parallel policy loading (warm-start)
- §7.4 Memory layout after broadcast

## Definition of Done

Both files created with complete content, valid cross-references, under 500 lines each.
