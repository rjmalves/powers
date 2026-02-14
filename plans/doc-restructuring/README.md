# Documentation Restructuring Plan

> Break 3 monolithic docs (~22,400 lines) into ~37 atomic spec files for AI-assisted development and targeted review.

## Quick Links

- [Master Plan](./00-master-plan.md) — Full rationale, target structure, and process
- [Epic 1: Foundation](./epic-01-foundation/) — Directory structure, cross-cutting specs
- [Epic 2: Math Specs](./epic-02-math-specs/) — Mathematical formulations
- [Epic 3: Data Model Specs](./epic-03-data-model-specs/) — Input/output schemas
- [Epic 4: Architecture Specs](./epic-04-architecture-specs/) — Execution flow, pipelines
- [Epic 5: HPC Specs](./epic-05-hpc-specs/) — MPI, OpenMP, NUMA, SLURM
- [Epic 6: Review & Finalize](./epic-06-review-finalize/) — Validation, READMEs, review checklist

## Status

| Ticket | Epic         | Description                                            | Status      | Dependencies                 |
| ------ | ------------ | ------------------------------------------------------ | ----------- | ---------------------------- |
| T-001  | Foundation   | Create spec directory structure and template           | `completed` | —                            |
| T-002  | Foundation   | Extract design principles, notation, scale specs       | `completed` | T-001                        |
| T-003  | Foundation   | Extract config reference and deferred features specs   | `completed` | T-001                        |
| T-004  | Math         | Extract SDDP algorithm spec                            | `completed` | T-001, T-002                 |
| T-004b | Math         | Extract system element modeling overview spec          | `completed` | T-001, T-002                 |
| T-005  | Math         | Extract LP formulation spec                            | `completed` | T-001, T-002, T-004b         |
| T-006  | Math         | Extract block, hydro production, equipment specs       | `completed` | T-001, T-002, T-004b         |
| T-007  | Math         | Extract PAR inflow and non-negativity specs            | `completed` | T-001, T-002                 |
| T-008  | Math         | Extract cut management and stopping rules specs        | `completed` | T-001, T-002                 |
| T-009  | Math         | Extract discount rate and upper bound specs            | `completed` | T-001, T-002                 |
| T-010  | Math         | Extract risk measures spec                             | `completed` | T-001, T-002                 |
| T-011  | Data Model   | Extract input directory and system entity specs        | `completed` | T-001                        |
| T-012  | Data Model   | Extract hydro extensions, scenarios, constraints specs | `completed` | T-001, T-011                 |
| T-013  | Data Model   | Extract penalty system spec                            | `completed` | T-001                        |
| T-014  | Data Model   | Extract output schemas and infrastructure specs        | `completed` | T-001                        |
| T-015  | Data Model   | Extract binary formats and internal structures spec    | `completed` | T-001                        |
| T-016  | Architecture | Extract CLI/lifecycle and input loading specs          | `completed` | T-001                        |
| T-017  | Architecture | Extract validation architecture spec                   | `completed` | T-001                        |
| T-018  | Architecture | Extract scenario generation spec                       | `completed` | T-001                        |
| T-019  | Architecture | Extract training loop and convergence specs            | `completed` | T-001                        |
| T-020  | Architecture | Extract simulation and extension points specs          | `completed` | T-001                        |
| T-020b | Architecture | Extract solver abstraction layer spec                  | `completed` | T-001                        |
| T-021  | HPC          | Extract hybrid parallelism and work distribution specs | `completed` | T-001                        |
| T-022  | HPC          | Extract synchronization and communication specs        | `completed` | T-001                        |
| T-023  | HPC          | Extract memory architecture and checkpointing specs    | `completed` | T-001                        |
| T-024  | HPC          | Extract SLURM deployment spec                          | `completed` | T-001                        |
| T-025  | Finalize     | Cross-reference validation and traceability matrix     | `pending`   | T-001..T-024, T-004b, T-020b |
| T-026  | Finalize     | Update root README and specs README                    | `pending`   | T-025                        |
| T-027  | Finalize     | Create review checklist and change tracker             | `pending`   | T-025, T-026                 |

## Dependency Graph

```
T-001 (directory structure)
  │
  ├── T-002 (overview specs)
  │     │
  │     ├── T-004 (sddp-algorithm)          ┐
  │     ├── T-004b (system-elements)         │
  │     │     ├── T-005 (lp-formulation)     │ Epic 2: Math
  │     │     └── T-006 (block/hydro/equip)  │ (T-005, T-006 depend on T-004b)
  │     ├── T-007 (par-inflow)               │
  │     ├── T-008 (cuts/stopping)            │
  │     ├── T-009 (discount/upperbound)      │
  │     └── T-010 (risk-measures)            ┘
  │
  ├── T-003 (config + deferred)
  │
  ├── T-011 (input dir + entities)      ┐
  │     └── T-012 (hydro/scenarios)     │ Epic 3: Data Model
  ├── T-013 (penalty system)            │ (can run in parallel
  ├── T-014 (output schemas)            │  except T-012 after T-011)
  ├── T-015 (binary formats)            ┘
  │
  ├── T-016 (cli + loading)             ┐
  ├── T-017 (validation)                │ Epic 4: Architecture
  ├── T-018 (scenario generation)       │ (can run in parallel)
  ├── T-019 (training + convergence)    │
  ├── T-020 (simulation + extensions)   │
  └── T-020b (solver abstraction)       ┘
  │
  ├── T-021 (parallelism + distrib)     ┐
  ├── T-022 (sync + communication)      │ Epic 5: HPC
  ├── T-023 (memory + checkpointing)    │ (can run in parallel)
  └── T-024 (slurm deployment)          ┘
        │
        ▼
  T-025 (cross-reference validation)    ┐
    └── T-026 (update READMEs)          │ Epic 6: Finalize
          └── T-027 (review checklist)  ┘
```

## Execution Strategy

1. **T-001** first (creates directories)
2. **T-002, T-003** next (foundational specs)
3. **T-004..T-024** in parallel batches (Epics 2-5 are independent; within Epic 2, T-005 and T-006 wait for T-004b)
4. **T-025..T-027** sequentially after all above

Estimated effort: ~4-6 hours of agent time with human review checkpoints.

## Process Per Ticket

1. AI reads relevant sections from source doc(s)
2. AI extracts content into new spec file with frontmatter
3. AI validates: no content lost, formulas preserved, cross-references updated
4. Human reviews and updates frontmatter status
