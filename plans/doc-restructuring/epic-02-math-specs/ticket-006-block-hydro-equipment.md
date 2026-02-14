# T-006: Extract Block Formulations, Hydro Production, and Equipment Specs

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md)
- T-004b (system-elements.md — `equipment-formulations.md` builds on the element descriptions in §3)

## Description

Extract three related spec files: block formulation variants, hydro production function models, and equipment-specific detailed formulations.

Note: `system-elements.md` (T-004b) describes _what_ each element is and its decision variables. `equipment-formulations.md` (this ticket) contains the _detailed mathematical constraints_ for each equipment type from §8. The reading order is: system-elements → lp-formulation (assembled LP) → equipment-formulations (per-equipment deep dives).

## Acceptance Criteria

- [ ] `docs/specs/01-math/block-formulations.md` extracted from MATH_FORMULATIONS §6 (6.1-6.3)
- [ ] `docs/specs/01-math/hydro-production-models.md` extracted from MATH_FORMULATIONS §7 (7.1-7.5)
- [ ] `docs/specs/01-math/equipment-formulations.md` extracted from MATH_FORMULATIONS §8 (8.1-8.4, deferred items reference deferred-features.md)
- [ ] All mathematical formulas preserved exactly
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/block-formulations.md`
- `docs/specs/01-math/hydro-production-models.md`
- `docs/specs/01-math/equipment-formulations.md`

## Technical Details

### block-formulations.md

Source: `MATHEMATICAL_FORMULATIONS.md` §6 (6.1-6.3)

- Parallel blocks (default) — how blocks operate independently
- Chronological blocks — how blocks couple temporally
- Comparison summary table

### hydro-production-models.md

Source: `MATHEMATICAL_FORMULATIONS.md` §7 (7.1-7.5)

- Constant productivity model
- FPHA (Four-Point Head Approximation) — full formulation
- Linearized head model
- Model selection guidelines
- FPHA data requirements summary

### equipment-formulations.md

Source: `MATHEMATICAL_FORMULATIONS.md` §8 (8.1-8.4)

- Thermal plants — cost curves, commitment, ramping
- Transmission lines — flow limits, losses
- Import/export contracts — fixed/flexible contracts
- Pumping stations — pump-turbine coupling
- §8.5 Batteries, §8.6 Non-controllable → just a reference to deferred-features.md

## Definition of Done

All three files created with complete content, valid cross-references, each under 500 lines.
