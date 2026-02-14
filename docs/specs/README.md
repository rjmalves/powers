# Powers Specification Index

Technical specifications for the Powers SDDP optimization system, extracted from the original monolithic documentation into focused, reviewable spec files.

## Review Progress

**0 / 43 specs approved**

| Category        | Total | Draft | Under Review | Needs Changes | Approved |
| --------------- | ----: | ----: | -----------: | ------------: | -------: |
| 00-overview     |     3 |     3 |            0 |             0 |        0 |
| 01-math         |    13 |    13 |            0 |             0 |        0 |
| 02-data-model   |     9 |     9 |            0 |             0 |        0 |
| 03-architecture |     9 |     9 |            0 |             0 |        0 |
| 04-hpc          |     7 |     7 |            0 |             0 |        0 |
| 05-config       |     1 |     1 |            0 |             0 |        0 |
| 06-deferred     |     1 |     1 |            0 |             0 |        0 |

### Status Lifecycle

```
draft --> under-review --> approved
               |                ^
               +-> needs-changes -+
```

## Spec File Inventory

### 00-overview

| File                                                                       | Status | Review Priority | Source Sections              | Description                                             |
| -------------------------------------------------------------------------- | ------ | --------------- | ---------------------------- | ------------------------------------------------------- |
| [design-principles.md](00-overview/design-principles.md)                   | draft  | 2-high          | DATA_MODEL_SPECIFICATION.md  | Design philosophy, goals, invariants                    |
| [notation-conventions.md](00-overview/notation-conventions.md)             | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Mathematical notation, index sets, symbols              |
| [production-scale-reference.md](00-overview/production-scale-reference.md) | draft  | 3-medium        | DATA_MODEL_SPECIFICATION.md  | System dimensions, variable counts, performance targets |

### 01-math

| File                                                             | Status | Review Priority | Source Sections              | Description                                                       |
| ---------------------------------------------------------------- | ------ | --------------- | ---------------------------- | ----------------------------------------------------------------- |
| [sddp-algorithm.md](01-math/sddp-algorithm.md)                   | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | SDDP overview, policy graph, state variables, single vs multi-cut |
| [system-elements.md](01-math/system-elements.md)                 | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | What each physical element is, its variables, connections, role   |
| [lp-formulation.md](01-math/lp-formulation.md)                   | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Objective function, constraints, dual variables                   |
| [block-formulations.md](01-math/block-formulations.md)           | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Parallel blocks, chronological blocks                             |
| [hydro-production-models.md](01-math/hydro-production-models.md) | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Constant productivity, FPHA, linearized head                      |
| [equipment-formulations.md](01-math/equipment-formulations.md)   | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Thermal, transmission, contracts, pumping                         |
| [par-inflow-model.md](01-math/par-inflow-model.md)               | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | PAR(p) definition, fitting, validation                            |
| [inflow-nonnegativity.md](01-math/inflow-nonnegativity.md)       | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | None, penalty, truncation, truncation+penalty                     |
| [cut-management.md](01-math/cut-management.md)                   | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Cut generation, aggregation, selection, dominated detection       |
| [stopping-rules.md](01-math/stopping-rules.md)                   | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Iteration limit, time limit, statistical, simulation-based        |
| [discount-rate.md](01-math/discount-rate.md)                     | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Discounted Bellman equation, stage-dependent rates                |
| [upper-bound-evaluation.md](01-math/upper-bound-evaluation.md)   | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | Inner approximation, Lipschitz interpolation, gap computation     |
| [risk-measures.md](01-math/risk-measures.md)                     | draft  | 2-high          | MATHEMATICAL_FORMULATIONS.md | CVaR, convex combination, risk-averse cuts, per-stage profiles    |

### 02-data-model

| File                                                                       | Status | Review Priority | Source Sections             | Description                                                        |
| -------------------------------------------------------------------------- | ------ | --------------- | --------------------------- | ------------------------------------------------------------------ |
| [input-directory-structure.md](02-data-model/input-directory-structure.md) | draft  | 1-critical      | DATA_MODEL_SPECIFICATION.md | File layout, config.json schema                                    |
| [input-system-entities.md](02-data-model/input-system-entities.md)         | draft  | 1-critical      | DATA_MODEL_SPECIFICATION.md | Buses, lines, hydros, thermals (registry schemas)                  |
| [input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       | draft  | 1-critical      | DATA_MODEL_SPECIFICATION.md | Geometry, production models, FPHA hyperplanes, pumping, contracts  |
| [input-scenarios.md](02-data-model/input-scenarios.md)                     | draft  | 1-critical      | DATA_MODEL_SPECIFICATION.md | Stages, inflow models, load factors, exchange factors, correlation |
| [input-constraints.md](02-data-model/input-constraints.md)                 | draft  | 1-critical      | DATA_MODEL_SPECIFICATION.md | Generic constraints, initial conditions, policy directory          |
| [penalty-system.md](02-data-model/penalty-system.md)                       | draft  | 1-critical      | DATA_MODEL_SPECIFICATION.md | Three-tier cascade, piecewise deficit, schema, overrides           |
| [output-schemas.md](02-data-model/output-schemas.md)                       | draft  | 3-medium        | DATA_MODEL_SPECIFICATION.md | Simulation outputs, training outputs, categorical codes            |
| [output-infrastructure.md](02-data-model/output-infrastructure.md)         | draft  | 3-medium        | DATA_MODEL_SPECIFICATION.md | Manifest, metadata, hive partitioning, distributed writing         |
| [binary-formats.md](02-data-model/binary-formats.md)                       | draft  | 3-medium        | DATA_MODEL_SPECIFICATION.md | FlatBuffers schemas, Parquet configuration                         |

### 03-architecture

| File                                                                     | Status | Review Priority | Source Sections                        | Description                                                            |
| ------------------------------------------------------------------------ | ------ | --------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| [cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md)             | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | Entrypoint, CLI design, exit codes, execution phases                   |
| [input-loading-pipeline.md](03-architecture/input-loading-pipeline.md)   | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | Loading architecture, dependency resolution, sparse time-series        |
| [validation-architecture.md](03-architecture/validation-architecture.md) | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | 5-phase validation, error collection, error types                      |
| [scenario-generation.md](03-architecture/scenario-generation.md)         | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | PAR preprocessing, noise sampling, correlation, external scenarios     |
| [training-loop.md](03-architecture/training-loop.md)                     | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | SDDP training, forward/backward pass execution, state management       |
| [simulation-architecture.md](03-architecture/simulation-architecture.md) | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | Simulation execution, output writing                                   |
| [convergence-monitoring.md](03-architecture/convergence-monitoring.md)   | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | Convergence criteria, bound computation                                |
| [solver-abstraction.md](03-architecture/solver-abstraction.md)           | draft  | 3-medium        | DATA_MODEL_SPECIFICATION.md            | LpSolver trait, compile-time selection, pre-allocated cuts, LP scaling |
| [extension-points.md](03-architecture/extension-points.md)               | draft  | 3-medium        | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | Trait abstractions, factory pattern, horizon modes                     |

### 04-hpc

| File                                                          | Status | Review Priority | Source Sections                                                     | Description                                                        |
| ------------------------------------------------------------- | ------ | --------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)         | draft  | 4-low           | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md, DATA_MODEL_SPECIFICATION.md | MPI (ferroMPI) + OpenMP (C FFI) strategy, design rationale         |
| [work-distribution.md](04-hpc/work-distribution.md)           | draft  | 4-low           | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md                              | Forward/backward pass distribution, dynamic work distribution      |
| [synchronization.md](04-hpc/synchronization.md)               | draft  | 4-low           | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md, DATA_MODEL_SPECIFICATION.md | Sync points, thread sync, lock-free cut aggregation                |
| [communication-patterns.md](04-hpc/communication-patterns.md) | draft  | 4-low           | DATA_MODEL_SPECIFICATION.md                                         | ferroMPI persistent collectives, SharedWindow\<T\>, async overlap  |
| [memory-architecture.md](04-hpc/memory-architecture.md)       | draft  | 4-low           | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md, DATA_MODEL_SPECIFICATION.md | Memory budget, NUMA-aware allocation (ferrompi::slurm), pools      |
| [checkpointing.md](04-hpc/checkpointing.md)                   | draft  | 4-low           | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md                              | Checkpoint strategy, warm-start, policy persistence                |
| [slurm-deployment.md](04-hpc/slurm-deployment.md)             | draft  | 4-low           | PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md                              | Job scripts, multi-node, parameter studies, performance monitoring |

### 05-config

| File                                                               | Status | Review Priority | Source Sections              | Description                                     |
| ------------------------------------------------------------------ | ------ | --------------- | ---------------------------- | ----------------------------------------------- |
| [configuration-reference.md](05-config/configuration-reference.md) | draft  | 3-medium        | MATHEMATICAL_FORMULATIONS.md | All config-driven LP variants, complete example |

### 06-deferred

| File                                                     | Status | Review Priority | Source Sections              | Description                                               |
| -------------------------------------------------------- | ------ | --------------- | ---------------------------- | --------------------------------------------------------- |
| [deferred-features.md](06-deferred/deferred-features.md) | draft  | 4-low           | MATHEMATICAL_FORMULATIONS.md | GNL thermals, batteries, multi-cut, Markovian, wind/solar |

## Source Document Mapping

| Source Document                          |  Lines | Specs Derived From                                                  |
| ---------------------------------------- | -----: | ------------------------------------------------------------------- |
| `DATA_MODEL_SPECIFICATION.md`            | 10,637 | 00-overview (2), 02-data-model (9), 03-architecture (1), 04-hpc (3) |
| `MATHEMATICAL_FORMULATIONS.md`           |  4,812 | 00-overview (1), 01-math (13), 05-config (1), 06-deferred (1)       |
| `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` |  6,961 | 03-architecture (8), 04-hpc (6)                                     |

## How to Review a Spec

1. Tell the AI: "Let's review `docs/specs/02-data-model/penalty-system.md`"
2. The AI loads the spec and presents a summary
3. Read, ask questions, request changes
4. The AI updates the spec content and the `change_log` in frontmatter
5. When satisfied, say "approve" — the AI sets `status: approved`, `last_reviewed`, `reviewed_by`
6. The AI updates this dashboard
