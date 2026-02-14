# T-019: Extract Training Loop and Convergence Specs

## Epic

Epic 4: Architecture Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the SDDP training loop architecture and convergence monitoring into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/03-architecture/training-loop.md` extracted from ARCHITECTURE §12-15 (training structure, forward pass, backward pass, cut management/storage)
- [ ] `docs/specs/03-architecture/convergence-monitoring.md` extracted from ARCHITECTURE §16 (16.1-16.3)
- [ ] Training loop covers: core training structures, trait abstractions, forward pass overview/implementation/state management/parallel execution, backward pass overview/implementation/dual extraction/parallel execution, cut management structures/selection/serialization/synchronization
- [ ] Convergence covers: criteria, monitor implementation, bound computation details
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/03-architecture/training-loop.md`
- `docs/specs/03-architecture/convergence-monitoring.md`

## Technical Details

### training-loop.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §12-15

- §12.1 SDDP algorithm overview (architecture perspective)
- §12.2 Core training structures (Rust structs)
- §12.3 Trait abstractions for forward/backward passes
- §13.1-13.4 Forward pass: overview, implementation, state management, parallel execution
- §14.1-14.4 Backward pass: overview, implementation, dual extraction, parallel execution
- §15.1-15.4 Cut management: FCF structure, selection strategies, serialization, cross-rank synchronization

References:

- `01-math/sddp-algorithm.md` for algorithmic definition
- `01-math/cut-management.md` for cut math
- `04-hpc/work-distribution.md` for parallelism details

### convergence-monitoring.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §16

- §16.1 Convergence criteria (implementation of stopping rules)
- §16.2 Convergence monitor implementation (Rust struct/trait)
- §16.3 Bound computation details (lower/upper bound tracking)

References:

- `01-math/stopping-rules.md` for stopping rule definitions
- `01-math/upper-bound-evaluation.md` for upper bound math

## Definition of Done

Both files created with complete content, valid cross-references, under 500 lines each.
