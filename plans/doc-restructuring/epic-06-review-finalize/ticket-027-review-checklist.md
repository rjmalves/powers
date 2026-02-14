# T-027: Create Review Checklist and Data Model Change Tracker

## Epic

Epic 6: Review and Finalize

## Dependencies

- T-025 (cross-reference validation complete)
- T-026 (READMEs updated)

## Description

Create a review checklist document that guides the user through reviewing each spec. Also create a data model change tracker to document expected changes.

## Acceptance Criteria

- [ ] `docs/specs/REVIEW_CHECKLIST.md` created with per-spec review guidance, organized by review priority
- [ ] `docs/specs/CHANGE_TRACKER.md` created to track expected data model changes
- [ ] Review checklist assigns `review_priority` to every spec: 1-critical (data model inputs, penalty), 2-high (math), 3-medium (architecture, outputs), 4-low (HPC, deferred)
- [ ] Each checklist item includes: spec file link, review priority, key questions to answer during review, blocking dependencies (which specs must be approved first)
- [ ] Change tracker has sections for: input schemas, internal structures, output formats, penalty system
- [ ] `docs/specs/README.md` dashboard is populated with initial counts per category

## Files to Create

- `docs/specs/REVIEW_CHECKLIST.md`
- `docs/specs/CHANGE_TRACKER.md`

## Technical Details

### REVIEW_CHECKLIST.md

Structure:

```markdown
# Spec Review Checklist

## How to Use

1. Work through specs in priority order (1-critical first)
2. For each spec, tell the AI: "Let's review [spec path]"
3. The AI loads the spec and presents a summary for discussion
4. After review, the AI updates the frontmatter: status, last_reviewed, reviewed_by, review_notes
5. The AI updates the dashboard in docs/specs/README.md

## Priority 1: Critical (Data Model — Expect Changes)

These specs define the input/output contract. Must be approved before implementation begins.

- [ ] `penalty-system.md` [P1] — Is the three-tier cascade correct? Any new penalty types? Override resolution logic ok?
- [ ] `input-directory-structure.md` [P1] — Is the file layout final? config.json schema complete?
- [ ] `input-system-entities.md` [P1] — Are the hydro/thermal registry schemas complete? Any missing fields?
- [ ] `input-hydro-extensions.md` [P1] — Which optional files are actually needed for v1? Schema changes?
- [ ] `input-scenarios.md` [P1] — Is the inflow model format final? Correlation structure ok?
- [ ] `input-constraints.md` [P1] — Generic constraint format adequate for all use cases?
- [ ] `binary-formats.md` [P1] — FlatBuffers vs alternatives still the right call? Internal structures correct?

## Priority 2: High (Mathematical Formulations — Mostly Correct)

These define what the solver computes. Verify mathematical correctness.

- [ ] `system-elements.md` [P2] — All elements described? Variable tables complete?
- [ ] `lp-formulation.md` [P2] — All constraints present? Slack handling correct?
- [ ] `hydro-production-models.md` [P2] — FPHA formulation verified?
- [ ] `equipment-formulations.md` [P2] — Thermal/line/contract formulations verified?
- [ ] `par-inflow-model.md` [P2] — PAR(p) fitting steps correct?
- [ ] `cut-management.md` [P2] — Cut selection strategies correct?
- [ ] `sddp-algorithm.md` [P2] — Algorithm description accurate?
- [ ] `block-formulations.md` [P2] — Block coupling correct?
- [ ] `stopping-rules.md` [P2] — All rules described correctly?
- [ ] `inflow-nonnegativity.md` [P2] — Method comparison accurate?
- [ ] `discount-rate.md` [P2] — Discounting formulation correct?
- [ ] `upper-bound-evaluation.md` [P2] — Inner approximation math verified?
- [ ] `risk-measures.md` [P2] — CVaR/convex combination correct?

## Priority 3: Medium (Architecture & Outputs)

These define how the solver is built. Review after data model is stable.

- [ ] `output-schemas.md` [P3] — Parquet output schemas correct? Blocked by P1 input review
- [ ] `output-infrastructure.md` [P3] — Hive partitioning, distributed writing approach ok?
- [ ] `cli-and-lifecycle.md` [P3] — CLI interface final?
- [ ] `input-loading-pipeline.md` [P3] — Loading sequence correct? Blocked by P1 input specs
- [ ] `validation-architecture.md` [P3] — 5-phase validation pipeline complete?
- [ ] `scenario-generation.md` [P3] — Implementation architecture ok?
- [ ] `training-loop.md` [P3] — Training structure correct?
- [ ] `convergence-monitoring.md` [P3] — Convergence criteria complete?
- [ ] `simulation-architecture.md` [P3] — Simulation flow ok?
- [ ] `extension-points.md` [P3] — Trait design adequate?
- [ ] `configuration-reference.md` [P3] — Config fields complete? Blocked by P1/P2 reviews

## Priority 4: Low (HPC & Deferred)

These are stable or not needed for initial implementation phases.

- [ ] `hybrid-parallelism.md` [P4] — ferroMPI integration described correctly?
- [ ] `work-distribution.md` [P4] — Distribution patterns correct?
- [ ] `synchronization.md` [P4] — Sync points accurate?
- [ ] `communication-patterns.md` [P4] — MPI patterns correct? ferroMPI API usage accurate?
- [ ] `memory-architecture.md` [P4] — Memory budget realistic?
- [ ] `checkpointing.md` [P4] — Checkpoint strategy adequate?
- [ ] `slurm-deployment.md` [P4] — Job scripts correct?
- [ ] `deferred-features.md` [P4] — Deferred list complete?
- [ ] `design-principles.md` [P4] — Principles still hold?
- [ ] `notation-conventions.md` [P4] — All symbols defined?
- [ ] `production-scale-reference.md` [P4] — Scale estimates current?
```

### CHANGE_TRACKER.md

Structure:

```markdown
# Data Model Change Tracker

## Input Schemas

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |

## Internal Structures

...

## Output Formats

...

## Penalty System

...
```

Each section has a table for tracking changes: what was proposed, what was decided, and whether the spec has been updated.

## Definition of Done

Both files created, review checklist is comprehensive, change tracker is ready for the user to fill in during reviews.
