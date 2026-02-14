# T-002: Extract Design Principles and Notation Specs

## Epic

Epic 1: Foundation

## Dependencies

- T-001 (directory structure)

## Description

Create the three foundational spec files that other specs will reference: design principles, notation conventions, and production scale reference.

## Acceptance Criteria

- [ ] `docs/specs/00-overview/design-principles.md` extracted from DATA_MODEL §1 (design philosophy, format selection, key goals, declaration order invariance)
- [ ] `docs/specs/00-overview/notation-conventions.md` extracted from MATH_FORMULATIONS §1.2 (notation) and §4 (index sets, parameters, decision variables, dual variables)
- [ ] `docs/specs/00-overview/production-scale-reference.md` extracted from DATA_MODEL §2 (state dimensions, variable counts, performance expectations)
- [ ] Each file has correct frontmatter with source_sections filled in
- [ ] Each file is under 500 lines
- [ ] Cross-references use relative links to other specs

## Files to Create

- `docs/specs/00-overview/design-principles.md`
- `docs/specs/00-overview/notation-conventions.md`
- `docs/specs/00-overview/production-scale-reference.md`

## Technical Details

### design-principles.md

Source: `DATA_MODEL_SPECIFICATION.md` §1 (1.1 through 1.4)

- Format selection criteria
- Key design goals
- Declaration order invariance (critical requirement)
- LP subproblem formulation reference (brief — point to `01-math/lp-formulation.md`)

### notation-conventions.md

Source: `MATHEMATICAL_FORMULATIONS.md` §1.2, §4.1-4.4

- All index sets (stages, hydros, thermals, buses, blocks, scenarios)
- All parameters with descriptions
- All decision variables with descriptions
- All dual variables with descriptions
- Notation conventions (subscript/superscript, bar/hat/tilde)

### production-scale-reference.md

Source: `DATA_MODEL_SPECIFICATION.md` §2 (2.1-2.3)

- State dimension estimates table
- Variable and constraint count tables
- Performance expectations by scale
- Reference to `scripts/lp_sizing.py` for computing dimensions

## Definition of Done

All three files are created, contain all source content, have valid cross-references, and are under 500 lines each.
