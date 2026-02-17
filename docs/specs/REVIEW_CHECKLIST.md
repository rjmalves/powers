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

## Priority 1: Critical (Data Model — Expect Changes)

These specs define the input/output contract. **Must be approved before implementation begins.** Changes here cascade to architecture, output schemas, and configuration.

- [x] [`02-data-model/penalty-system.md`](02-data-model/penalty-system.md) **[P1]** ✓ Approved 2026-02-14, re-approved 2026-02-16
  - Is the three-tier cascade (element → system → global) correct?
  - Any new penalty types needed beyond what's described?
  - Is the override resolution logic correct?
  - Dependencies: none — review first

- [ ] [`02-data-model/input-directory-structure.md`](02-data-model/input-directory-structure.md) **[P1]** ⏸ Deferred 2026-02-14 — depends on open decisions (penalty override format TBD, potential hydro modeling changes). Will resume after closing open edges.
  - Is the file layout final?
  - Is the `config.json` schema complete?
  - Any files missing from the directory tree?
  - Dependencies: none — review first

- [x] [`02-data-model/input-system-entities.md`](02-data-model/input-system-entities.md) **[P1]** ✓ Approved 2026-02-14, re-approved 2026-02-16
  - Are the hydro/thermal registry schemas complete?
  - Any missing fields in entity definitions?
  - Are all entity relationships documented?
  - Dependencies: `input-directory-structure.md` (defines where these files live)

- [x] [`02-data-model/input-hydro-extensions.md`](02-data-model/input-hydro-extensions.md) **[P1]** ✓ Approved 2026-02-15
  - Which optional files are actually needed for v1?
  - Any schema changes to hydro extension tables?
  - Is the production function data format final?
  - Dependencies: `input-system-entities.md` (defines base hydro schema)

- [x] [`02-data-model/input-scenarios.md`](02-data-model/input-scenarios.md) **[P1]** ✓ Approved 2026-02-15
  - Is the inflow model format final?
  - Is the correlation structure adequate?
  - Are scenario tree dimensions correct?
  - Dependencies: `input-system-entities.md` (defines which entities have scenarios)

- [x] [`02-data-model/input-constraints.md`](02-data-model/input-constraints.md) **[P1]** ✓ Approved 2026-02-15
  - Is the generic constraint format adequate for all use cases?
  - Can all real-world constraints be expressed in this format?
  - Are constraint identifiers and references consistent?
  - Dependencies: `input-system-entities.md` (constraints reference entities)

- [x] [`02-data-model/internal-structures.md`](02-data-model/internal-structures.md) **[P1]** ✓ Approved 2026-02-16
  - Are the internal memory representations correct?
  - Do internal structures align with input schemas?
  - Are serialization/deserialization boundaries clear?
  - Dependencies: all input specs (internal structures mirror inputs)

- [ ] [`02-data-model/binary-formats.md`](02-data-model/binary-formats.md) **[P1]**
  - Is FlatBuffers still the right choice vs alternatives?
  - Are internal structure schemas correct?
  - Is the versioning strategy adequate?
  - Dependencies: `internal-structures.md` (binary formats serialize these)

---

## Priority 2: High (Mathematical Formulations — Verify Correctness)

These specs define what the solver computes. **Verify mathematical correctness.** Errors here propagate directly to solver results.

- [ ] [`01-math/system-elements.md`](01-math/system-elements.md) **[P2]**
  - Are all system elements (hydro, thermal, lines, etc.) described?
  - Are variable tables complete for each element?
  - Are parameter ranges and units specified?
  - **⚠ CEPEL flag**: Review lateral flow (`Q_lat`), downstream flow formulation (`Q_jus` with participation factors), and water travel time propagation curves. These may require new flow variables beyond current `o = q + s`. See `CHANGE_TRACKER.md` "Future Modeling Observations".
  - Dependencies: none — foundational math spec

- [ ] [`01-math/lp-formulation.md`](01-math/lp-formulation.md) **[P2]**
  - Are all constraints present in the LP?
  - Is slack variable handling correct?
  - Is the objective function complete?
  - **⚠ CEPEL flag**: Current outflow `o = q + s` is the simplest case. Some plants require `Q_jus` with participation factors for turbined, spilled, lateral post inflows, and other plants' outflows. Water balance and travel time may also need propagation curves. See `CHANGE_TRACKER.md` "Future Modeling Observations".
  - Dependencies: `system-elements.md` (defines variables used in LP)

- [ ] [`01-math/hydro-production-models.md`](01-math/hydro-production-models.md) **[P2]**
  - Is the FPHA (Four-Point Hyperplane Approximation) formulation verified?
  - Are all production function variants described?
  - Do linearization approaches maintain accuracy?
  - **⚠ CEPEL flag**: Lateral flows affect tailwater level and thus the production function (e.g. Belo Monte, Itaipu). The FPHA may need to account for `Q_lat` in the tailwater polynomial. Also review backwater effects (remanso) from downstream reservoir levels. See `CHANGE_TRACKER.md` "Future Modeling Observations".
  - Dependencies: `system-elements.md` (defines hydro variables)

- [ ] [`01-math/equipment-formulations.md`](01-math/equipment-formulations.md) **[P2]**
  - Are thermal plant formulations verified?
  - Are transmission line formulations correct?
  - Are contract formulations complete?
  - Dependencies: `system-elements.md` (defines equipment variables)

- [ ] [`01-math/par-inflow-model.md`](01-math/par-inflow-model.md) **[P2]**
  - Are PAR(p) fitting steps correct?
  - Is the parameter estimation procedure accurate?
  - Is the seasonal decomposition described correctly?
  - Dependencies: none — standalone statistical model

- [ ] [`01-math/cut-management.md`](01-math/cut-management.md) **[P2]**
  - Are cut selection strategies correct?
  - Is the cut coefficient computation accurate?
  - Are dominance criteria well-defined?
  - Dependencies: `lp-formulation.md` (cuts modify the LP)

- [ ] [`01-math/sddp-algorithm.md`](01-math/sddp-algorithm.md) **[P2]**
  - Is the algorithm description accurate?
  - Are forward/backward pass procedures correct?
  - Is the convergence theory well-stated?
  - Dependencies: `lp-formulation.md`, `cut-management.md`

- [ ] [`01-math/block-formulations.md`](01-math/block-formulations.md) **[P2]**
  - Is block-to-block coupling correct?
  - Are intra-stage block transitions accurate?
  - Are chronological constraints properly formulated?
  - Dependencies: `lp-formulation.md` (blocks extend the base LP)

- [ ] [`01-math/stopping-rules.md`](01-math/stopping-rules.md) **[P2]**
  - Are all stopping rules described correctly?
  - Are statistical tests properly specified?
  - Are threshold parameters well-justified?
  - Dependencies: `sddp-algorithm.md` (stopping rules terminate the algorithm)

- [ ] [`01-math/inflow-nonnegativity.md`](01-math/inflow-nonnegativity.md) **[P2]**
  - Is the method comparison accurate (truncation, shifting, etc.)?
  - Are the trade-offs between methods well-described?
  - Is the recommended approach justified?
  - Dependencies: `par-inflow-model.md` (nonnegativity applies to PAR model)

- [ ] [`01-math/discount-rate.md`](01-math/discount-rate.md) **[P2]**
  - Is the discounting formulation correct?
  - Are multi-stage discount factors properly compounded?
  - Is the relationship to objective function clear?
  - Dependencies: `lp-formulation.md` (discount rate appears in objective)

- [ ] [`01-math/upper-bound-evaluation.md`](01-math/upper-bound-evaluation.md) **[P2]**
  - Is the inner approximation math verified?
  - Are bound estimation procedures correct?
  - Is the statistical confidence interval computation accurate?
  - Dependencies: `sddp-algorithm.md` (upper bound evaluates algorithm quality)

- [ ] [`01-math/risk-measures.md`](01-math/risk-measures.md) **[P2]**
  - Is the CVaR formulation correct?
  - Is the convex combination of expectation and CVaR properly specified?
  - Are risk parameter ranges well-defined?
  - Dependencies: `lp-formulation.md` (risk measures modify the objective)

---

## Priority 3: Medium (Architecture & Outputs)

These specs define how the solver is built and what it produces. **Review after data model (P1) is stable** — architecture depends on finalized schemas.

- [ ] [`02-data-model/output-schemas.md`](02-data-model/output-schemas.md) **[P3]**
  - Are Parquet output schemas correct and complete?
  - Do output fields align with LP variables from P2 specs?
  - Are all user-requested output dimensions covered?
  - Dependencies: blocked by P1 input specs + P2 `system-elements.md`

- [ ] [`02-data-model/output-infrastructure.md`](02-data-model/output-infrastructure.md) **[P3]**
  - Is the Hive partitioning scheme correct?
  - Is the distributed writing approach adequate?
  - Are output file size estimates realistic?
  - Dependencies: `output-schemas.md` (infra implements schema writing)

- [ ] [`03-architecture/cli-and-lifecycle.md`](03-architecture/cli-and-lifecycle.md) **[P3]**
  - Is the CLI interface final?
  - Are all lifecycle phases described?
  - Is the error reporting strategy adequate?
  - Dependencies: `input-directory-structure.md` (CLI points to input dir)

- [ ] [`03-architecture/input-loading-pipeline.md`](03-architecture/input-loading-pipeline.md) **[P3]**
  - Is the loading sequence correct?
  - Are validation steps in the right order?
  - Are error messages actionable?
  - Dependencies: blocked by all P1 input specs

- [ ] [`03-architecture/validation-architecture.md`](03-architecture/validation-architecture.md) **[P3]**
  - Is the 5-phase validation pipeline complete?
  - Are all validation rules enumerated?
  - Are error codes and messages well-defined?
  - Dependencies: `input-loading-pipeline.md` (validation is part of loading)

- [ ] [`03-architecture/scenario-generation.md`](03-architecture/scenario-generation.md) **[P3]**
  - Is the implementation architecture adequate?
  - Does it align with the PAR model from P2?
  - Is the random number generation strategy sound?
  - Dependencies: P2 `par-inflow-model.md`, P1 `input-scenarios.md`

- [ ] [`03-architecture/training-loop.md`](03-architecture/training-loop.md) **[P3]**
  - Is the training loop structure correct?
  - Are iteration callbacks properly placed?
  - Is the forward/backward pass orchestration accurate?
  - Dependencies: P2 `sddp-algorithm.md` (training implements the algorithm)

- [ ] [`03-architecture/convergence-monitoring.md`](03-architecture/convergence-monitoring.md) **[P3]**
  - Are convergence criteria complete?
  - Is the monitoring infrastructure adequate?
  - Are logging and reporting formats well-defined?
  - Dependencies: P2 `stopping-rules.md`, `training-loop.md`

- [ ] [`03-architecture/simulation-architecture.md`](03-architecture/simulation-architecture.md) **[P3]**
  - Is the simulation flow correct?
  - Does it properly use the trained policy?
  - Are output collection points identified?
  - Dependencies: `training-loop.md` (simulation uses trained cuts)

- [ ] [`03-architecture/solver-abstraction.md`](03-architecture/solver-abstraction.md) **[P3]**
  - Is the solver trait/interface design adequate?
  - Are all required LP operations covered?
  - Is the abstraction testable with mock solvers?
  - Dependencies: P2 `lp-formulation.md` (defines what solvers must handle)

- [ ] [`03-architecture/solver-highs-impl.md`](03-architecture/solver-highs-impl.md) **[P3]**
  - Is the HiGHS integration approach correct?
  - Are HiGHS-specific parameters documented?
  - Is error mapping from HiGHS to internal errors complete?
  - Dependencies: `solver-abstraction.md` (HiGHS implements the abstraction)

- [ ] [`03-architecture/solver-workspaces.md`](03-architecture/solver-workspaces.md) **[P3]**
  - Is the workspace pooling strategy correct?
  - Are thread-safety considerations addressed?
  - Is workspace reuse efficient?
  - Dependencies: `solver-abstraction.md` (workspaces manage solver instances)

- [ ] [`03-architecture/cut-management-impl.md`](03-architecture/cut-management-impl.md) **[P3]**
  - Does the implementation match the math in P2 `cut-management.md`?
  - Is the storage and retrieval strategy efficient?
  - Are cut sharing mechanisms for parallel scenarios correct?
  - Dependencies: P2 `cut-management.md`, `solver-abstraction.md`

- [ ] [`03-architecture/extension-points.md`](03-architecture/extension-points.md) **[P3]**
  - Is the trait design adequate for extensibility?
  - Are extension registration and discovery well-defined?
  - Are the planned extension points sufficient?
  - Dependencies: most P3 architecture specs (extension points span the system)

- [ ] [`05-config/configuration-reference.md`](05-config/configuration-reference.md) **[P3]**
  - Are all config fields documented with types and defaults?
  - Do config fields align with P1 and P2 spec parameters?
  - Is validation of config values described?
  - Dependencies: blocked by P1 and P2 reviews (config exposes their parameters)

---

## Priority 4: Low (HPC, Deferred & Overview)

These specs are either stable, deferred to later phases, or foundational references that rarely change. **Review last or skip for initial implementation.**

- [ ] [`04-hpc/hybrid-parallelism.md`](04-hpc/hybrid-parallelism.md) **[P4]**
  - Is the ferroMPI integration described correctly?
  - Is the hybrid MPI+threads approach well-justified?
  - Are scaling expectations realistic?
  - Dependencies: none — standalone HPC spec

- [ ] [`04-hpc/work-distribution.md`](04-hpc/work-distribution.md) **[P4]**
  - Are distribution patterns correct for scenario allocation?
  - Is load balancing described?
  - Are edge cases (uneven scenario counts) handled?
  - Dependencies: `hybrid-parallelism.md` (distribution runs on parallel infra)

- [ ] [`04-hpc/synchronization.md`](04-hpc/synchronization.md) **[P4]**
  - Are sync points accurate between forward/backward passes?
  - Is barrier placement optimal?
  - Are deadlock avoidance strategies described?
  - Dependencies: `work-distribution.md` (sync coordinates distributed work)

- [ ] [`04-hpc/communication-patterns.md`](04-hpc/communication-patterns.md) **[P4]**
  - Are MPI communication patterns correct?
  - Is ferroMPI API usage accurate?
  - Are message sizes and frequencies estimated?
  - Dependencies: `synchronization.md` (communication implements sync)

- [ ] [`04-hpc/shared-memory-aggregation.md`](04-hpc/shared-memory-aggregation.md) **[P4]**
  - Is the shared-memory aggregation strategy correct?
  - Are thread-safety guarantees well-defined?
  - Does it integrate with the hybrid parallelism model?
  - Dependencies: `hybrid-parallelism.md` (aggregation is part of parallel strategy)

- [ ] [`04-hpc/memory-architecture.md`](04-hpc/memory-architecture.md) **[P4]**
  - Is the memory budget realistic for production scale?
  - Are allocation strategies described per component?
  - Are cache-friendly data layouts considered?
  - Dependencies: P1 `internal-structures.md` (memory holds internal data)

- [ ] [`04-hpc/checkpointing.md`](04-hpc/checkpointing.md) **[P4]**
  - Is the checkpoint strategy adequate for long runs?
  - Is the checkpoint format versioned?
  - Is restart from checkpoint well-defined?
  - Dependencies: `memory-architecture.md` (checkpoints serialize memory state)

- [ ] [`04-hpc/slurm-deployment.md`](04-hpc/slurm-deployment.md) **[P4]**
  - Are SLURM job scripts correct for target clusters?
  - Are resource request parameters realistic?
  - Are environment setup steps documented?
  - Dependencies: `hybrid-parallelism.md` (SLURM launches the parallel app)

- [ ] [`06-deferred/deferred-features.md`](06-deferred/deferred-features.md) **[P4]**
  - Is the deferred feature list complete?
  - Are deferral justifications adequate?
  - Are any deferred items actually needed for v1?
  - Dependencies: none — review to confirm scope boundaries

- [ ] [`00-overview/design-principles.md`](00-overview/design-principles.md) **[P4]**
  - Do the stated principles still hold after spec reviews?
  - Are there any new principles discovered during review?
  - Dependencies: none — foundational document

- [ ] [`00-overview/notation-conventions.md`](00-overview/notation-conventions.md) **[P4]**
  - Are all mathematical symbols used in P2 specs defined here?
  - Are naming conventions consistent across all specs?
  - Dependencies: none — reference document

- [ ] [`00-overview/production-scale-reference.md`](00-overview/production-scale-reference.md) **[P4]**
  - Are scale estimates (system sizes, scenario counts) current?
  - Do memory/time estimates align with HPC specs?
  - Dependencies: none — reference document

---

## Summary

| Priority  | Count  | Scope                                 |
| --------- | ------ | ------------------------------------- |
| P1        | 8      | Data model & input/output contracts   |
| P2        | 13     | Mathematical formulations             |
| P3        | 15     | Architecture, outputs, solver, config |
| P4        | 12     | HPC, deferred, overview               |
| **Total** | **48** | **All specs in `docs/specs/`**        |

> **Note**: `README.md`, `TEMPLATE.md`, and `TRACEABILITY.md` are infrastructure files, not specs — they are not included in the review checklist.
