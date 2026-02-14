# T-017: Extract Validation Architecture Spec

## Epic

Epic 4: Architecture Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the validation architecture into a focused spec combining content from both the data model and architecture docs.

## Acceptance Criteria

- [ ] `docs/specs/03-architecture/validation-architecture.md` extracted from ARCHITECTURE §6 (6.1-6.4) and DATA_MODEL §8 (8.1-8.2)
- [ ] Covers: validation layers (5 phases), error collection strategy, error types, validation report format, input validation phases, validation error type catalog
- [ ] No duplication between architecture doc content and data model content — merge coherently
- [ ] Under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/03-architecture/validation-architecture.md`

## Technical Details

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §6 and `DATA_MODEL_SPECIFICATION.md` §8

Merge strategy:

- §6.1 Validation layers (architecture perspective — how validation fits in execution flow)
- §6.2 Error collection strategy (how errors accumulate vs fail-fast)
- §6.3 Validation error types (Rust enum/type design)
- §6.4 Validation report format (output for user)
- §8.1 Input validation phases (data model perspective — what is validated when)
- §8.2 Validation error types (may overlap with §6.3 — deduplicate)

The merged spec should present validation as a unified pipeline:

1. Schema validation (file structure)
2. Type validation (field types and ranges)
3. Referential validation (cross-entity references)
4. Consistency validation (cross-file invariants)
5. Semantic validation (domain-specific rules)

## Definition of Done

File created merging validation content from both source docs, valid cross-references, under 500 lines.
