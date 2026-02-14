# Master Plan: Documentation Restructuring

## Goal

Break the 3 monolithic documentation files (~22,400 lines total) into ~30 atomic spec files, each focused on a single concern and under ~500 lines. This enables:

1. **AI-assisted development** — Each spec file fits comfortably in an agent's context window, so implementation tickets can reference exactly the spec they need without noise.
2. **Targeted review** — The user can review and approve individual specs independently, marking each as "reviewed" or "needs changes".
3. **Incremental data model changes** — Since the user expects changes to input schemas, internal structures, output formats, and the penalty system, having isolated spec files makes it safe to modify one area without risking others.
4. **Implementation planning** — Once specs are finalized, each atomic spec maps directly to one or more implementation tickets.

## Non-Goals

- Rewriting or changing the technical content (that happens in the review phase)
- Creating implementation tickets for the Rust code (separate plan)
- Removing the original monolithic files (they remain as reference until specs are approved)

## Source Documents

| File                                     | Lines  | Sections                                                                                                                                                       |
| ---------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATA_MODEL_SPECIFICATION.md`            | 10,637 | 9 major sections: design principles, production scale, input model, output model, internal structures, MPI comm, file formats, validation, implementation plan |
| `MATHEMATICAL_FORMULATIONS.md`           | 4,812  | 6 parts: SDDP foundation, LP formulation, stochastic modeling, cut management, advanced formulations, config reference                                         |
| `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` | 6,961  | 8 parts: lifecycle, input processing, scenario generation, training, simulation, parallel execution, memory/IO, extensions                                     |

## Target Structure

```
docs/specs/
├── 00-overview/
│   ├── design-principles.md              # Design philosophy, goals, invariants
│   ├── notation-conventions.md           # Mathematical notation, index sets, symbols
│   └── production-scale-reference.md     # System dimensions, variable counts, performance targets
│
├── 01-math/
│   ├── sddp-algorithm.md                # SDDP overview, policy graph, state variables, single vs multi-cut
│   ├── system-elements.md               # What each physical element is, its variables, connections, role
│   ├── lp-formulation.md                # Objective function, constraints, dual variables
│   ├── block-formulations.md            # Parallel blocks, chronological blocks
│   ├── hydro-production-models.md       # Constant productivity, FPHA, linearized head
│   ├── equipment-formulations.md        # Thermal, transmission, contracts, pumping
│   ├── par-inflow-model.md              # PAR(p) definition, fitting, validation
│   ├── inflow-nonnegativity.md          # None, penalty, truncation, truncation+penalty
│   ├── cut-management.md               # Cut generation, aggregation, selection, dominated detection
│   ├── stopping-rules.md               # Iteration limit, time limit, statistical, simulation-based
│   ├── discount-rate.md                 # Discounted Bellman equation, stage-dependent rates
│   ├── upper-bound-evaluation.md        # Inner approximation, Lipschitz interpolation, gap computation
│   └── risk-measures.md                 # CVaR, convex combination, risk-averse cuts, per-stage profiles
│
├── 02-data-model/
│   ├── input-directory-structure.md     # File layout, config.json schema
│   ├── input-system-entities.md         # Buses, lines, hydros, thermals (registry schemas)
│   ├── input-hydro-extensions.md        # Geometry, production models, FPHA hyperplanes, pumping, contracts
│   ├── input-scenarios.md              # Stages, inflow models, load factors, exchange factors, correlation
│   ├── input-constraints.md            # Generic constraints, initial conditions, policy directory
│   ├── penalty-system.md               # Three-tier cascade, piecewise deficit, schema, overrides
│   ├── output-schemas.md               # Simulation outputs, training outputs, categorical codes
│   ├── output-infrastructure.md        # Manifest, metadata, hive partitioning, distributed writing
│   └── binary-formats.md              # FlatBuffers schemas, Parquet configuration
│
├── 03-architecture/
│   ├── cli-and-lifecycle.md            # Entrypoint, CLI design, exit codes, execution phases
│   ├── input-loading-pipeline.md       # Loading architecture, dependency resolution, sparse time-series
│   ├── validation-architecture.md      # 5-phase validation, error collection, error types
│   ├── scenario-generation.md          # PAR preprocessing, noise sampling, correlation, external scenarios
│   ├── training-loop.md               # SDDP training, forward/backward pass execution, state management
│   ├── simulation-architecture.md     # Simulation execution, output writing
│   ├── convergence-monitoring.md      # Convergence criteria, bound computation
│   ├── solver-abstraction.md          # LpSolver trait, compile-time selection, pre-allocated cuts, LP scaling
│   └── extension-points.md           # Trait abstractions, factory pattern, horizon modes
│
├── 04-hpc/
│   ├── hybrid-parallelism.md          # MPI (ferroMPI) + OpenMP (C FFI) strategy, design rationale
│   ├── work-distribution.md           # Forward/backward pass distribution, dynamic work distribution
│   ├── synchronization.md            # Sync points, thread sync, lock-free cut aggregation
│   ├── communication-patterns.md     # ferroMPI persistent collectives, SharedWindow<T>, async overlap
│   ├── memory-architecture.md        # Memory budget, NUMA-aware allocation (ferrompi::slurm), pools
│   ├── checkpointing.md             # Checkpoint strategy, warm-start, policy persistence
│   └── slurm-deployment.md          # Job scripts, multi-node, parameter studies, performance monitoring
│
├── 05-config/
│   └── configuration-reference.md    # All config-driven LP variants, complete example
│
└── 06-deferred/
    └── deferred-features.md          # GNL thermals, batteries, multi-cut, Markovian, wind/solar
```

**Total: 37 atomic spec files** across 7 categories.

## Review Status Tracking

### Frontmatter Schema

Each spec file will include a YAML frontmatter block with explicit review tracking:

```yaml
---
status: draft | under-review | approved | needs-changes
review_priority: 1-critical | 2-high | 3-medium | 4-low
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §3.2-3.5"
  - "MATHEMATICAL_FORMULATIONS.md §5.1-5.10"
last_reviewed: null # date of last human review (YYYY-MM-DD)
reviewed_by: null # who reviewed it
review_notes: "" # free-text notes from review
change_log: # track modifications after initial extraction
  - date: null
    description: ""
---
```

### Status Lifecycle

```
draft ──→ under-review ──→ approved
                │                ↑
                └──→ needs-changes ──→ under-review ──→ approved
```

- **draft**: AI has extracted the content, no human has reviewed it yet
- **under-review**: Human is actively reviewing this spec
- **needs-changes**: Human reviewed and identified issues (documented in `review_notes`)
- **approved**: Human has verified the content is correct and complete

### Review Priority Levels

| Priority   | Category                               | Rationale                                   |
| ---------- | -------------------------------------- | ------------------------------------------- |
| 1-critical | Data model input specs, penalty system | User expects changes; blocks implementation |
| 2-high     | Math formulations, system elements, LP | Must verify correctness before coding       |
| 3-medium   | Architecture, output specs, config     | Important but less likely to change         |
| 4-low      | HPC, SLURM, deferred features          | Stable or not needed for initial phases     |

### Review Dashboard

The `docs/specs/README.md` will contain a live dashboard showing:

```
Review Progress: 0/36 specs approved

 Category        | Total | Draft | Review | Needs Changes | Approved
 00-overview     |     3 |     3 |      0 |             0 |        0
 01-math         |    13 |    13 |      0 |             0 |        0
 02-data-model   |     9 |     9 |      0 |             0 |        0
 03-architecture |     8 |     8 |      0 |             0 |        0
 04-hpc          |     7 |     7 |      0 |             0 |        0
 05-config       |     1 |     1 |      0 |             0 |        0
 06-deferred     |     1 |     1 |      0 |             0 |        0
```

### Workflow for Reviewing a Spec

1. Tell the AI: "Let's review `docs/specs/02-data-model/penalty-system.md`"
2. AI loads the spec into context and presents a summary
3. Human reads, asks questions, requests changes
4. AI updates the spec content and the `change_log` in frontmatter
5. When satisfied, human says "approve" → AI sets `status: approved`, `last_reviewed`, `reviewed_by`
6. AI updates the dashboard in `docs/specs/README.md`

This way, at any point you can ask "what have I reviewed?" and the AI can scan the frontmatter across all specs and give you an accurate answer.

## Epic Breakdown

### Epic 1: Foundation (3 tickets)

Set up the directory structure and create cross-cutting reference specs.

### Epic 2: Mathematical Formulation Specs (7 tickets)

Extract math content from `MATHEMATICAL_FORMULATIONS.md` into focused specs.

### Epic 3: Data Model Specs (5 tickets)

Extract data model content from `DATA_MODEL_SPECIFICATION.md` into focused specs.

**Input Format Rationale Requirement**: Every data model input spec must include a "Format Rationale" section for each input file. This section explicitly documents _why_ a particular format (JSON, Parquet, or FlatBuffers) was chosen, based on the nature of the data. During the review phase, this rationale will be critically evaluated to ensure format choices are well-justified. The classification framework is:

| Data Nature                                                                                                     | Preferred Format                  | Rationale                                                                                 |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| **Registry / catalog** (entity definitions with nested structure, cross-references, optional fields)            | JSON                              | Human-readable, supports nesting, easy to version-control, natural for config-like data   |
| **Time series** (per-stage or per-entity-per-stage tabular data, potentially large)                             | Parquet                           | Columnar compression, efficient partial reads, typed columns, good for stage-indexed data |
| **Default-with-overrides** (base values that rarely change, with occasional per-entity or per-stage exceptions) | JSON (base) + Parquet (overrides) | Sparse override pattern: JSON for the base case, Parquet only for rows that differ        |
| **Complex nested object** (hierarchical config, deeply nested optional sections)                                | JSON                              | Natural for hierarchical data, supports optional fields and polymorphic structures        |
| **Correlation / matrix data** (symmetric matrices, coefficient tables)                                          | JSON                              | Small enough for JSON, structure is not tabular                                           |
| **Policy / binary data** (cuts, solutions, checkpoint data needing zero-copy access)                            | FlatBuffers                       | Zero-copy deserialization, efficient for repeated read during hot-path access             |
| **High-volume output** (per-scenario, per-stage results at production scale)                                    | Parquet                           | Columnar, partitionable, compressible, standard for analytics                             |

This rationale should be documented per-file (not just per-format) so that during review, every format decision can be independently questioned and justified.

### Epic 4: Architecture Specs (6 tickets)

Extract architecture content from `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` and the solver interface design from `DATA_MODEL_SPECIFICATION.md` §5.4-5.5 into focused specs.

**Solver Abstraction Requirement**: The solver interface specification (DATA_MODEL §5.4-5.5, ~1,250 lines) defines a complete multi-solver abstraction layer and MUST be extracted as its own dedicated spec (`solver-abstraction.md`), NOT buried inside `binary-formats.md`. This is a critical architectural component covering:

- **`LpSolver` trait hierarchy**: `LpProblem` (data) → `LpScaling` (preprocessing) → `LpSolver` (execution) → `LpSolution` (result)
- **Compile-time solver selection**: Cargo feature flags (`solver-highs`, `solver-cplex`, `solver-gurobi`) with `ActiveSolver` type alias — zero vtable overhead on hot path
- **Pre-allocated cut constraint design**: Static LP structure with bound-toggling instead of row insertion/deletion, deterministic slot assignment via `CutSlotManager`
- **Solver error handling**: `SolverError` enum with infeasibility rays, partial solutions, recovery suggestions
- **Encapsulated retry logic**: Per-solver retry strategies (clear basis → disable presolve → switch to IPM → relax tolerances)
- **Dual normalization**: Canonical sign convention across solvers with different conventions
- **Basis storage for warm-starting**: `Basis` struct in original problem space for portability
- **Thread-local solver workspaces**: `ThreadSolverWorkspace` with NUMA-aware allocation, one solver instance per OpenMP thread, basis caching, batch bound operations
- **LP scaling**: Row/column scaling for numerical stability, scaling impact on cut coefficients, scaling persistence via FlatBuffers
- **HiGHS reference implementation**: Complete `HighsSolver` implementation as the default open-source solver

### Epic 5: HPC Specs (4 tickets)

Extract HPC/parallelism content (spans both data model and architecture docs) into focused specs.

**ferroMPI Requirement**: All MPI references in extracted specs must use the `ferrompi` crate API (`v0.2`, https://github.com/rjmalves/ferrompi) instead of raw C FFI wrappers from the old docs. Key replacements:

| Old Doc Reference                                | ferroMPI Equivalent                                                    |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `MPI_Init_thread(MPI_THREAD_MULTIPLE)` via C FFI | `ferrompi::init_with_threading(ThreadLevel::Multiple)`                 |
| Raw `MPI_Comm` handles, `MPI_Comm_rank/size`     | `ferrompi::Communicator` (is `Send + Sync`), `.rank()/.size()`         |
| `MPI_Bcast_init` / `MPI_Allreduce_init` C FFI    | `comm.bcast_init()` / `comm.allreduce_init()` → `PersistentRequest<T>` |
| `MPI_Win_create` / `MPI_Win_lock`                | `ferrompi::SharedWindow<T>::new()` / `.lock()` (RAII, `rma` feature)   |
| Manual SLURM env var parsing                     | `ferrompi::slurm::local_rank()`, `::node_count()` (`numa` feature)     |
| Rust FFI wrapper code for MPI                    | Remove — ferroMPI provides safe generic bindings directly              |

OpenMP FFI bindings remain as C FFI (ferroMPI does not cover OpenMP).

### Epic 6: Review and Finalize (3 tickets)

Cross-reference validation, update the root README, and create the review checklist.

## Execution Order

```
Epic 1 (Foundation)
  ├── T-001 → T-002 → T-003
  │
Epic 2 (Math) ──────────────────── depends on T-001, T-002
  ├── T-004 (sddp-algorithm)
  ├── T-004b (system-elements) ← conceptual overview of all physical elements
  ├── T-005 (lp-formulation) ← depends on T-004b
  ├── T-006 (block-formulations, hydro-production, equipment) ← depends on T-004b
  ├── T-007 (par-inflow, inflow-nonnegativity)
  ├── T-008 (cut-management, stopping-rules)
  ├── T-009 (discount-rate, upper-bound)
  └── T-010 (risk-measures)
  │
Epic 3 (Data Model) ────────────── depends on T-001
  ├── T-011 (input-directory, input-system-entities)
  ├── T-012 (input-hydro-extensions, input-scenarios, input-constraints)
  ├── T-013 (penalty-system)
  ├── T-014 (output-schemas, output-infrastructure)
  └── T-015 (binary-formats)
  │
Epic 4 (Architecture) ──────────── depends on T-001
  ├── T-016 (cli-lifecycle, input-loading)
  ├── T-017 (validation-architecture)
  ├── T-018 (scenario-generation)
  ├── T-019 (training-loop, convergence)
  ├── T-020 (simulation, extension-points)
  └── T-020b (solver-abstraction) ← LpSolver trait, feature flags, pre-allocated cuts, LP scaling
  │
Epic 5 (HPC) ───────────────────── depends on T-001
  ├── T-021 (hybrid-parallelism, work-distribution)
  ├── T-022 (synchronization, communication)
  ├── T-023 (memory-architecture, checkpointing)
  └── T-024 (slurm-deployment)
  │
Epic 6 (Finalize) ──────────────── depends on ALL above
  ├── T-025 (cross-reference validation)
  ├── T-026 (update root README and docs README)
  └── T-027 (create review checklist)
```

Epics 2, 3, 4, and 5 can run **in parallel** once Epic 1 is complete.
Epic 6 runs after all others are done.

## Process

For each spec file:

1. **Extract** — AI reads relevant sections from the source doc(s) and extracts content into the new spec file
2. **Restructure** — Clean up cross-references (update internal links to point to other spec files instead of section numbers), add frontmatter
3. **Validate** — Ensure no content is lost, all formulas/tables/code blocks are preserved
4. **User Review** — User reviews and marks status in frontmatter (approved / needs-changes)

## Success Criteria

- All content from the 3 source docs is covered by at least one spec file
- No spec file exceeds ~500 lines (hard limit: 600)
- Each spec file is self-contained: it defines its own terms or explicitly references another spec
- Cross-references between specs use relative links, not section numbers from the old docs
- Frontmatter tracks review status per file
- A mapping table shows old section → new spec file for traceability
