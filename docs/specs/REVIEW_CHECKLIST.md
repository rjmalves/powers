# Spec Review Checklist

## How to Use

This checklist guides you through reviewing every specification in priority order. The goal is to surface data model issues early — before implementation locks in schemas that are hard to change.

### AI-Assisted Review Workflow

1. **Work through specs in priority order** — Priority 1 (critical) first, then 2, 3, 4
2. **For each spec**, tell the AI: _"Let's review `<spec path>`"_
3. The AI loads the spec and presents a structured summary for discussion
4. You discuss changes, flag issues, or approve the spec as-is
5. After review, the AI updates the spec's frontmatter:
   - `status`: `draft` → `approved` (or `needs-changes`, `deferred`, `needs-re-review`)
   - `last_reviewed`: today's date
   - `reviewed_by`: your name
   - `review_notes`: summary of decisions
6. The AI updates the dashboard in [`docs/specs/README.md`](README.md)
7. Any data model changes discovered during review get recorded in [`CHANGE_TRACKER.md`](CHANGE_TRACKER.md)

### Priority Levels

| Priority | Label    | Scope                        | Rationale                                              |
| -------- | -------- | ---------------------------- | ------------------------------------------------------ |
| P1       | Critical | Data model, input/output     | Defines the contract — changes here cascade everywhere |
| P2       | High     | Mathematical formulations    | Must be correct — errors propagate to solver results   |
| P3       | Medium   | Architecture, outputs, infra | Review after data model is stable                      |
| P4       | Low      | HPC, deferred, overview      | Stable or not needed for initial implementation        |

---

## Priority 1: Critical (Data Model) — COMPLETE

All 9 P1 specs are approved. These define the input/output contract.

- [x] [`02-data-model/penalty-system.md`](02-data-model/penalty-system.md) **[P1]** Approved 2026-02-14, re-approved 2026-02-17
- [x] [`02-data-model/input-directory-structure.md`](02-data-model/input-directory-structure.md) **[P1]** Approved 2026-02-17
- [x] [`02-data-model/input-system-entities.md`](02-data-model/input-system-entities.md) **[P1]** Approved 2026-02-14, re-approved 2026-02-17
- [x] [`02-data-model/input-hydro-extensions.md`](02-data-model/input-hydro-extensions.md) **[P1]** Approved 2026-02-15, re-approved 2026-02-17
- [x] [`02-data-model/input-scenarios.md`](02-data-model/input-scenarios.md) **[P1]** Approved 2026-02-15, re-approved 2026-02-17
- [x] [`02-data-model/input-constraints.md`](02-data-model/input-constraints.md) **[P1]** Approved 2026-02-15
- [x] [`02-data-model/internal-structures.md`](02-data-model/internal-structures.md) **[P1]** Approved 2026-02-16, re-approved 2026-02-17
- [x] [`02-data-model/binary-formats.md`](02-data-model/binary-formats.md) **[P1]** Approved 2026-02-16, re-approved 2026-02-17
- [x] [`00-overview/design-principles.md`](00-overview/design-principles.md) **[P1]** §5 Implementation Language & FFI Strategy added 2026-02-17

### Key decisions from P1 review:

- All tabular input data uses **Parquet**; JSON for structured/nested objects; FlatBuffers for policy data
- Penalties organized as **3-category taxonomy** (recourse slacks, constraint violations, regularization)
- **Block mode is per-stage**, not global (`block_mode` in `stages.json`)
- Inflow models split into 2 files: `inflow_seasonal_stats.parquet` + `inflow_ar_coefficients.parquet`
- Exchange factors moved from `scenarios/` to `constraints/`
- Penalty overrides split into 4 entity-specific Parquet files
- Correlation schedule embedded in `correlation.json`
- **Rust** chosen as implementation language (documented in `design-principles.md` §5)

---

## Priority 2: High (Mathematical Formulations) — COMPLETE

All 14 P2 specs are approved. These define what the solver computes.

Review approach: Read the approved P1 specs first to ensure math specs are consistent with the finalized data model (penalty names, file references, variable conventions, etc.).

- [x] [`01-math/system-elements.md`](01-math/system-elements.md) **[P2]** Approved 2026-02-19
  - All system elements fully defined (buses, lines, thermals, hydros, NCS, contracts, pumping)
  - Variable Units Convention documented (rate units: MW, m³/s)
  - GNL thermal subsection added (deferred implementation)
  - Dependencies: none — foundational math spec

- [x] [`01-math/lp-formulation.md`](01-math/lp-formulation.md) **[P2]** Approved 2026-02-19
  - 3-category penalty taxonomy aligned with approved `penalty-system.md`
  - Unidirectional contracts, NCS generation, FPHA turbined cost added
  - Storage violations placed outside τ_k sum in objective
  - Dependencies: `system-elements.md`

- [x] [`01-math/hydro-production-models.md`](01-math/hydro-production-models.md) **[P2]** Approved 2026-02-20
  - Two training models (constant productivity, FPHA) + one simulation-only (linearized head)
  - Linearized head reclassified as simulation-only: bilinear term ($q \times v^{avg}$) changes LP between iterations, breaking SDDP convergence
  - FPHA hyperplane fitting, correction factor κ, LP integration
  - Cross-spec updates applied to 5 approved specs (system-elements, lp-formulation, input-system-entities, input-hydro-extensions, internal-structures)
  - Dependencies: `system-elements.md`

- [x] [`01-math/equipment-formulations.md`](01-math/equipment-formulations.md) **[P2]** Approved 2026-02-20
  - Thermal piecewise-linear convex cost curve clarified (segment filling order, not commitment approximation)
  - Contracts rewritten: bidirectional → typed unidirectional (single χ, single c^ctr)
  - NCS promoted from DEFERRED to full section; open question on block distribution factors
  - §8 Simulation-only constraint enhancements (future): stepped thermal, storage-dependent hydro bounds
  - Open design question: stepped constraints vs. generic constraints input format
  - Dependencies: `system-elements.md`, `lp-formulation.md`

- [x] [`01-math/par-inflow-model.md`](01-math/par-inflow-model.md) **[P2]** Approved 2026-02-20
  - Restructured: model definition → parameter semantics → stored vs. computed → fitting → validation
  - Clarified std_m3s = seasonal sample std (s_m), not residual std (σ_m); σ_m computed at runtime
  - AR coefficients stored in original units; POWE.RS reverse-standardizes for σ_m computation
  - CEPEL PAR(p)-A moved to deferred-features.md (C.8)
  - Dependencies: none — standalone statistical model

- [x] [`01-math/cut-management.md`](01-math/cut-management.md) **[P2]** Approved 2026-02-20
  - Full rewrite: definitions → aggregation → validity → selection → convergence
  - Pseudocode replaced with behavioral definitions and mathematical properties
  - Discount factor applied to θ in objective (not to cut coefficients directly)
  - FPHA contribution to storage cut coefficient documented (½ΣπγV term)
  - Multi-cut deferred to deferred-features.md
  - Dependencies: `lp-formulation.md`, `hydro-production-models.md`

- [x] [`01-math/sddp-algorithm.md`](01-math/sddp-algorithm.md) **[P2]** Approved 2026-02-20
  - §3.1-§3.2 rewritten from pseudocode to behavioral descriptions
  - §3.4 added Execution Model and Performance Considerations (thread-trajectory affinity, backward sync barriers, LP rebuild cost, state save/restore, generic constraint dual preprocessing)
  - Discount factor symbol changed from β to d (avoids collision with cut coefficients)
  - GNL validation-rejection note added to §5
  - Review notes propagated to 6 downstream specs (4 HPC, 2 architecture)
  - Dependencies: `lp-formulation.md`, `cut-management.md`

- [x] [`01-math/block-formulations.md`](01-math/block-formulations.md) **[P2]** ✅ approved 2026-02-20
  - Parallel blocks, chronological blocks
  - Must align with per-stage `block_mode` from `input-scenarios.md`
  - Dependencies: `lp-formulation.md`

- [x] [`01-math/stopping-rules.md`](01-math/stopping-rules.md) **[P2]** ✅ approved 2026-02-22
  - Iteration limit, time limit, statistical, simulation-based
  - Dependencies: `sddp-algorithm.md`

- [x] [`01-math/inflow-nonnegativity.md`](01-math/inflow-nonnegativity.md) **[P2]** ✅ approved 2026-02-22
  - None, penalty, truncation, truncation+penalty
  - Dependencies: `par-inflow-model.md`

- [x] [`01-math/discount-rate.md`](01-math/discount-rate.md) **[P2]** ✅ approved 2026-02-22
  - Discounted Bellman equation, stage-dependent rates, discount factor on θ
  - Rewritten: $d$ symbol, aligned config with annual_discount_rate, infinite horizon extracted
  - Dependencies: `lp-formulation.md`, `sddp-algorithm.md`

- [x] [`01-math/infinite-horizon.md`](01-math/infinite-horizon.md) **[P2]** ✅ approved 2026-02-22
  - Periodic structure, cycle detection, cut sharing, modified passes, convergence
  - Extracted from `discount-rate.md` §15
  - Dependencies: `discount-rate.md`, `sddp-algorithm.md`, `cut-management.md`

- [x] [`01-math/upper-bound-evaluation.md`](01-math/upper-bound-evaluation.md) **[P2]** ✅ approved 2026-02-22
  - Inner approximation (SIDP), Lipschitz interpolation, deterministic gap computation
  - Rewritten: renumbered §1-11, $d$ symbol, removed duplicated schema/config, added infinite horizon note
  - Dependencies: `sddp-algorithm.md`, `discount-rate.md`

- [x] [`01-math/risk-measures.md`](01-math/risk-measures.md) **[P2]** ✅ approved 2026-02-22
  - Full rewrite: renumbered §1-11, discount factor d, α symbol disambiguated
  - §7 sorting-based greedy weight computation (replaces LP formulation)
  - Config aligned with approved input-scenarios.md §1.7
  - Dependencies: `lp-formulation.md`, `cut-management.md`, `sddp-algorithm.md`

### Key decisions from P2 review so far:

- **Variable units**: Rate units (MW, m³/s) adopted for all LP variables; τ_k as external multiplier
- **Contracts**: Single unidirectional variable per contract (not two like transmission lines)
- **Storage violations**: Outside τ_k sum in objective (apply to end-of-stage storage, not per-block)
- **FPHA constraints**: Hard (no slacks) — regularization via `fpha_turbined_cost` on turbined flow
- **PAR inputs in original units**: `std_m3s` = seasonal sample std ($s_m$), AR coefficients in original units; residual std ($\sigma_m$) computed at runtime via reverse-standardization
- **Discount factor on θ**: Discounting applied to θ in objective ($\beta_{t-1→t} · θ$), not to cut coefficients — cuts remain unmodified
- **FPHA in cut coefficients**: Storage cut coefficient includes FPHA hyperplane dual contribution ($\pi^{wb} + ½Σπ^{fpha}·γ_v$)
- **Linearized head is simulation-only**: Bilinear term changes LP between iterations, breaking SDDP convergence
- **Discount factor symbol**: Use $d$ (not $\beta$) to avoid collision with cut coefficient symbol $\beta$
- **Thread-trajectory affinity**: Each thread owns a complete forward trajectory AND the corresponding backward pass (documented in `sddp-algorithm.md` §3.4)
- **Risk-averse cut weights**: Sorting-based greedy allocation replaces LP formulation — equivalent but simpler
- **Backward pass per-stage sync barriers**: Hard synchronization at each stage boundary; forward pass is fully parallel

---

## Priority 3: Medium (Architecture & Outputs)

These specs define how the solver is built and what it produces. **Review after P2 math specs are stable** — architecture depends on finalized formulations.

- [x] [`02-data-model/output-schemas.md`](02-data-model/output-schemas.md) **[P3]** ✅ approved 2026-02-22
  - Full rewrite: 12 issues fixed, all penalty categories in costs, exchange direct/reverse flow, hydro violation columns
  - NCS promoted from DEFERRED to active optional; batteries kept as forward-compatible placeholder
  - Convergence log annotated for risk-averse interpretation and UB mechanism distinction
  - contract_type_code removed from contracts output (redundant with input registry)
  - Dependencies: P1 input specs + P2 `system-elements.md`, `penalty-system.md`

- [ ] [`02-data-model/output-infrastructure.md`](02-data-model/output-infrastructure.md) **[P3]**
  - Manifest, metadata, hive partitioning, distributed writing
  - Dependencies: `output-schemas.md`

- [ ] [`03-architecture/cli-and-lifecycle.md`](03-architecture/cli-and-lifecycle.md) **[P3]**
  - Entrypoint, CLI design, exit codes, execution phases
  - Dependencies: `input-directory-structure.md`

- [ ] [`03-architecture/input-loading-pipeline.md`](03-architecture/input-loading-pipeline.md) **[P3]**
  - Loading architecture, dependency resolution, sparse time-series
  - Dependencies: all P1 input specs

- [ ] [`03-architecture/validation-architecture.md`](03-architecture/validation-architecture.md) **[P3]**
  - 5-phase validation, error collection, error types
  - Dependencies: `input-loading-pipeline.md`

- [ ] [`03-architecture/scenario-generation.md`](03-architecture/scenario-generation.md) **[P3]**
  - PAR preprocessing, noise sampling, correlation, external scenarios
  - Dependencies: P2 `par-inflow-model.md`, P1 `input-scenarios.md`

- [ ] [`03-architecture/training-loop.md`](03-architecture/training-loop.md) **[P3]**
  - SDDP training, forward/backward pass execution, state management
  - Dependencies: P2 `sddp-algorithm.md`

- [ ] [`03-architecture/convergence-monitoring.md`](03-architecture/convergence-monitoring.md) **[P3]**
  - Convergence criteria, bound computation
  - Dependencies: P2 `stopping-rules.md`, `training-loop.md`

- [ ] [`03-architecture/simulation-architecture.md`](03-architecture/simulation-architecture.md) **[P3]**
  - Simulation execution, output writing
  - Dependencies: `training-loop.md`

- [ ] [`03-architecture/solver-abstraction.md`](03-architecture/solver-abstraction.md) **[P3]**
  - LpSolver trait, compile-time selection, pre-allocated cuts, LP scaling
  - Dependencies: P2 `lp-formulation.md`

- [ ] [`03-architecture/solver-highs-impl.md`](03-architecture/solver-highs-impl.md) **[P3]**
  - HiGHS integration, warm-starting, retry strategy, memory footprint
  - Dependencies: `solver-abstraction.md`

- [ ] [`03-architecture/solver-workspaces.md`](03-architecture/solver-workspaces.md) **[P3]**
  - Thread-local solver infrastructure, NUMA-aware allocation, LP scaling
  - Dependencies: `solver-abstraction.md`

- [ ] [`03-architecture/cut-management-impl.md`](03-architecture/cut-management-impl.md) **[P3]**
  - FCF data structure, cut selection, binary serialization, MPI sync
  - Dependencies: P2 `cut-management.md`, `solver-abstraction.md`

- [ ] [`03-architecture/extension-points.md`](03-architecture/extension-points.md) **[P3]**
  - Trait abstractions, factory pattern, horizon modes
  - Dependencies: most P3 architecture specs

- [ ] [`05-config/configuration-reference.md`](05-config/configuration-reference.md) **[P3]**
  - All config-driven LP variants, complete example
  - Must align with per-stage block_mode and annual_discount_rate from P1
  - Dependencies: P1 and P2 reviews

---

## Priority 4: Low (HPC, Deferred & Overview)

These specs are either stable, deferred to later phases, or foundational references that rarely change. **Review last or skip for initial implementation.**

- [ ] [`04-hpc/hybrid-parallelism.md`](04-hpc/hybrid-parallelism.md) **[P4]**
  - MPI (ferroMPI) + OpenMP (C FFI) strategy, design rationale
  - Dependencies: none — standalone HPC spec

- [ ] [`04-hpc/work-distribution.md`](04-hpc/work-distribution.md) **[P4]**
  - Forward/backward pass distribution, dynamic work distribution
  - Dependencies: `hybrid-parallelism.md`

- [ ] [`04-hpc/synchronization.md`](04-hpc/synchronization.md) **[P4]**
  - Sync points, thread sync, lock-free cut aggregation
  - Dependencies: `work-distribution.md`

- [ ] [`04-hpc/communication-patterns.md`](04-hpc/communication-patterns.md) **[P4]**
  - ferroMPI persistent collectives, SharedWindow\<T\>, async overlap
  - Dependencies: `synchronization.md`

- [ ] [`04-hpc/shared-memory-aggregation.md`](04-hpc/shared-memory-aggregation.md) **[P4]**
  - Hierarchical cut aggregation, shared memory scenarios, reproducibility
  - Dependencies: `hybrid-parallelism.md`

- [ ] [`04-hpc/memory-architecture.md`](04-hpc/memory-architecture.md) **[P4]**
  - Memory budget, NUMA-aware allocation, pools
  - Dependencies: P1 `internal-structures.md`

- [ ] [`04-hpc/checkpointing.md`](04-hpc/checkpointing.md) **[P4]**
  - Checkpoint strategy, warm-start, policy persistence
  - Dependencies: `memory-architecture.md`

- [ ] [`04-hpc/slurm-deployment.md`](04-hpc/slurm-deployment.md) **[P4]**
  - Job scripts, multi-node, parameter studies, performance monitoring
  - Dependencies: `hybrid-parallelism.md`

- [ ] [`06-deferred/deferred-features.md`](06-deferred/deferred-features.md) **[P4]**
  - GNL thermals, batteries, multi-cut, Markovian, wind/solar
  - Dependencies: none — review to confirm scope boundaries

- [ ] [`00-overview/notation-conventions.md`](00-overview/notation-conventions.md) **[P4]**
  - Mathematical notation, index sets, symbols
  - Should be updated after all P2 math specs are approved to ensure symbol consistency
  - Dependencies: none — reference document

- [ ] [`00-overview/production-scale-reference.md`](00-overview/production-scale-reference.md) **[P4]**
  - System dimensions, variable counts, performance targets
  - Dependencies: none — reference document

---

## Summary

| Priority  |  Count | Approved | Status       |
| --------- | -----: | -------: | ------------ |
| P1        |      9 |        9 | **Complete** |
| P2        |     14 |       14 | **Complete** |
| P3        |     15 |        1 | **Next**     |
| P4        |     11 |        0 | Deferred     |
| **Total** | **49** |   **24** |              |

> **Note**: `README.md`, `TEMPLATE.md`, `TRACEABILITY.md`, `REVIEW_CHECKLIST.md`, and `CHANGE_TRACKER.md` are infrastructure files, not specs — they are not included in the review count.
