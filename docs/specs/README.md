# Powers Specification Index

Technical specifications for the Powers SDDP optimization system, extracted from the original monolithic documentation into focused, reviewable spec files.

## Dashboard

| Metric           |     Value |
| ---------------- | --------: |
| Total spec files |        48 |
| Total lines      |    17,562 |
| Categories       |         7 |
| Status           | All draft |

| Category        | Specs | Lines |
| --------------- | ----: | ----: |
| 00-overview     |     3 |   811 |
| 01-math         |    13 | 3,857 |
| 02-data-model   |    10 | 3,839 |
| 03-architecture |    12 | 4,958 |
| 04-hpc          |     8 | 3,502 |
| 05-config       |     1 |   331 |
| 06-deferred     |     1 |   468 |

## Review Progress

**3 / 48 specs approved** | **1 deferred**

| Category        | Total | Draft | Deferred | Under Review | Needs Changes | Approved |
| --------------- | ----: | ----: | -------: | -----------: | ------------: | -------: |
| 00-overview     |     3 |     3 |        0 |            0 |             0 |        0 |
| 01-math         |    13 |    13 |        0 |            0 |             0 |        0 |
| 02-data-model   |    10 |     6 |        1 |            0 |             0 |        3 |
| 03-architecture |    12 |    12 |        0 |            0 |             0 |        0 |
| 04-hpc          |     8 |     8 |        0 |            0 |             0 |        0 |
| 05-config       |     1 |     1 |        0 |            0 |             0 |        0 |
| 06-deferred     |     1 |     1 |        0 |            0 |             0 |        0 |

### Status Lifecycle

```
draft --> under-review --> approved
               |                ^
               +-> needs-changes -+
               |
               +-> deferred (resume later)
```

## Spec File Inventory

### 00-overview

| File                                                                       | Status | Lines | Description                                             |
| -------------------------------------------------------------------------- | ------ | ----: | ------------------------------------------------------- |
| [design-principles.md](00-overview/design-principles.md)                   | draft  |   145 | Design philosophy, goals, invariants                    |
| [notation-conventions.md](00-overview/notation-conventions.md)             | draft  |   398 | Mathematical notation, index sets, symbols              |
| [production-scale-reference.md](00-overview/production-scale-reference.md) | draft  |   268 | System dimensions, variable counts, performance targets |

### 01-math

| File                                                             | Status | Lines | Description                                                       |
| ---------------------------------------------------------------- | ------ | ----: | ----------------------------------------------------------------- |
| [block-formulations.md](01-math/block-formulations.md)           | draft  |   127 | Parallel blocks, chronological blocks                             |
| [cut-management.md](01-math/cut-management.md)                   | draft  |   295 | Cut generation, aggregation, selection, dominated detection       |
| [discount-rate.md](01-math/discount-rate.md)                     | draft  |   298 | Discounted Bellman equation, stage-dependent rates                |
| [equipment-formulations.md](01-math/equipment-formulations.md)   | draft  |   180 | Thermal, transmission, contracts, pumping                         |
| [hydro-production-models.md](01-math/hydro-production-models.md) | draft  |   589 | Constant productivity, FPHA, linearized head                      |
| [inflow-nonnegativity.md](01-math/inflow-nonnegativity.md)       | draft  |   191 | None, penalty, truncation, truncation+penalty                     |
| [lp-formulation.md](01-math/lp-formulation.md)                   | draft  |   345 | Objective function, constraints, dual variables                   |
| [par-inflow-model.md](01-math/par-inflow-model.md)               | draft  |   199 | PAR(p) definition, fitting, validation                            |
| [risk-measures.md](01-math/risk-measures.md)                     | draft  |   274 | CVaR, convex combination, risk-averse cuts, per-stage profiles    |
| [sddp-algorithm.md](01-math/sddp-algorithm.md)                   | draft  |   221 | SDDP overview, policy graph, state variables, single vs multi-cut |
| [stopping-rules.md](01-math/stopping-rules.md)                   | draft  |   250 | Iteration limit, time limit, statistical, simulation-based        |
| [system-elements.md](01-math/system-elements.md)                 | draft  |   428 | What each physical element is, its variables, connections, role   |
| [upper-bound-evaluation.md](01-math/upper-bound-evaluation.md)   | draft  |   260 | Inner approximation, Lipschitz interpolation, gap computation     |

### 02-data-model

| File                                                                       | Status   | Lines | Description                                                        |
| -------------------------------------------------------------------------- | -------- | ----: | ------------------------------------------------------------------ |
| [binary-formats.md](02-data-model/binary-formats.md)                       | draft    |   379 | FlatBuffers schemas, Parquet configuration                         |
| [input-constraints.md](02-data-model/input-constraints.md)                 | draft    |   403 | Generic constraints, initial conditions, policy directory          |
| [input-directory-structure.md](02-data-model/input-directory-structure.md) | deferred |   307 | File layout, config.json schema                                    |
| [input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       | approved |   291 | Geometry, production models, FPHA hyperplanes                      |
| [input-scenarios.md](02-data-model/input-scenarios.md)                     | draft    |   353 | Stages, inflow models, load factors, exchange factors, correlation |
| [input-system-entities.md](02-data-model/input-system-entities.md)         | approved |   646 | Buses, lines, hydros, thermals, pumping stations, energy contracts |
| [internal-structures.md](02-data-model/internal-structures.md)             | draft    |   446 | Core runtime Rust structs for SDDP algorithm                       |
| [output-infrastructure.md](02-data-model/output-infrastructure.md)         | draft    |   475 | Manifest, metadata, hive partitioning, distributed writing         |
| [output-schemas.md](02-data-model/output-schemas.md)                       | draft    |   495 | Simulation outputs, training outputs, categorical codes            |
| [penalty-system.md](02-data-model/penalty-system.md)                       | approved |   313 | Three-tier cascade, piecewise deficit, schema, overrides           |

### 03-architecture

| File                                                                     | Status | Lines | Description                                                            |
| ------------------------------------------------------------------------ | ------ | ----: | ---------------------------------------------------------------------- |
| [cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md)             | draft  |   177 | Entrypoint, CLI design, exit codes, execution phases                   |
| [convergence-monitoring.md](03-architecture/convergence-monitoring.md)   | draft  |   288 | Convergence criteria, bound computation                                |
| [cut-management-impl.md](03-architecture/cut-management-impl.md)         | draft  |   315 | FCF data structure, cut selection, binary serialization, MPI sync      |
| [extension-points.md](03-architecture/extension-points.md)               | draft  |   476 | Trait abstractions, factory pattern, horizon modes                     |
| [input-loading-pipeline.md](03-architecture/input-loading-pipeline.md)   | draft  |   286 | Loading architecture, dependency resolution, sparse time-series        |
| [scenario-generation.md](03-architecture/scenario-generation.md)         | draft  |   490 | PAR preprocessing, noise sampling, correlation, external scenarios     |
| [simulation-architecture.md](03-architecture/simulation-architecture.md) | draft  |   495 | Simulation execution, output writing                                   |
| [solver-abstraction.md](03-architecture/solver-abstraction.md)           | draft  |   588 | LpSolver trait, compile-time selection, pre-allocated cuts, LP scaling |
| [solver-highs-impl.md](03-architecture/solver-highs-impl.md)             | draft  |   448 | HiGHS integration, warm-starting, retry strategy, memory footprint     |
| [solver-workspaces.md](03-architecture/solver-workspaces.md)             | draft  |   596 | Thread-local solver infrastructure, NUMA-aware allocation, LP scaling  |
| [training-loop.md](03-architecture/training-loop.md)                     | draft  |   473 | SDDP training, forward/backward pass execution, state management       |
| [validation-architecture.md](03-architecture/validation-architecture.md) | draft  |   322 | 5-phase validation, error collection, error types                      |

### 04-hpc

| File                                                                | Status | Lines | Description                                                            |
| ------------------------------------------------------------------- | ------ | ----: | ---------------------------------------------------------------------- |
| [checkpointing.md](04-hpc/checkpointing.md)                         | draft  |   530 | Checkpoint strategy, warm-start, policy persistence                    |
| [communication-patterns.md](04-hpc/communication-patterns.md)       | draft  |   316 | ferroMPI persistent collectives, SharedWindow\<T\>, async overlap      |
| [hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)               | draft  |   496 | MPI (ferroMPI) + OpenMP (C FFI) strategy, design rationale             |
| [memory-architecture.md](04-hpc/memory-architecture.md)             | draft  |   450 | Memory budget, NUMA-aware allocation (ferrompi::slurm), pools          |
| [shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) | draft  |   597 | Hierarchical cut aggregation, shared memory scenarios, reproducibility |
| [slurm-deployment.md](04-hpc/slurm-deployment.md)                   | draft  |   449 | Job scripts, multi-node, parameter studies, performance monitoring     |
| [synchronization.md](04-hpc/synchronization.md)                     | draft  |   219 | Sync points, thread sync, lock-free cut aggregation                    |
| [work-distribution.md](04-hpc/work-distribution.md)                 | draft  |   445 | Forward/backward pass distribution, dynamic work distribution          |

### 05-config

| File                                                               | Status | Lines | Description                                     |
| ------------------------------------------------------------------ | ------ | ----: | ----------------------------------------------- |
| [configuration-reference.md](05-config/configuration-reference.md) | draft  |   331 | All config-driven LP variants, complete example |

### 06-deferred

| File                                                     | Status | Lines | Description                                               |
| -------------------------------------------------------- | ------ | ----: | --------------------------------------------------------- |
| [deferred-features.md](06-deferred/deferred-features.md) | draft  |   468 | GNL thermals, batteries, multi-cut, Markovian, wind/solar |

## Source Document Mapping

| Source Document                          |  Lines | Specs Derived From                                                   |
| ---------------------------------------- | -----: | -------------------------------------------------------------------- |
| `DATA_MODEL_SPECIFICATION.md`            | 10,637 | 00-overview (2), 02-data-model (10), 03-architecture (4), 04-hpc (4) |
| `MATHEMATICAL_FORMULATIONS.md`           |  4,812 | 00-overview (1), 01-math (13), 05-config (1), 06-deferred (1)        |
| `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` |  6,961 | 03-architecture (8), 04-hpc (6)                                      |

## How to Review

Each spec file contains YAML frontmatter with review metadata. The review workflow uses this frontmatter to track progress.

### Frontmatter Fields

```yaml
---
status: draft # Current review status
review_priority: 2-high # 1-critical, 2-high, 3-medium, 4-low
source_sections: # Which monolithic doc sections this came from
  - "DATA_MODEL_SPECIFICATION.md §3.1"
last_reviewed: null # Date of last review (YYYY-MM-DD)
reviewed_by: null # Who reviewed it
review_notes: "" # Notes from reviewer if changes needed
change_log:
  - date: 2026-02-14
    description: "Initial extraction"
---
```

### Review Workflow

1. **Pick a spec** — Start with `1-critical` priority specs, then `2-high`, etc.
2. **Read the spec** — Review for technical accuracy, completeness, and clarity.
3. **Update `status`** in the frontmatter:
   - `draft` &rarr; `under-review` — You are actively reviewing it
   - `under-review` &rarr; `approved` — Content is correct and complete
   - `under-review` &rarr; `needs-changes` — Issues found, describe in `review_notes`
   - `needs-changes` &rarr; `under-review` — Changes applied, ready for re-review
4. **Set `last_reviewed`** to today's date (`YYYY-MM-DD`).
5. **Set `reviewed_by`** to your name or handle.
6. **Add `review_notes`** if status is `needs-changes` — describe what needs fixing.
7. **Append to `change_log`** with date and description of any changes made.

### Review with AI Assistant

For an AI-assisted review workflow:

1. Tell the AI: _"Let's review `docs/specs/02-data-model/penalty-system.md`"_
2. The AI loads the spec and presents a summary
3. Read, ask questions, request changes
4. The AI updates the spec content and the `change_log` in frontmatter
5. When satisfied, say _"approve"_ — the AI sets `status: approved`, `last_reviewed`, `reviewed_by`
6. The AI updates this dashboard
