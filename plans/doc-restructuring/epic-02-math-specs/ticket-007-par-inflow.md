# T-007: Extract PAR Inflow Model and Non-Negativity Specs

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md)

## Description

Extract the stochastic modeling specs: PAR(p) inflow model and inflow non-negativity solution methods.

## Acceptance Criteria

- [ ] `docs/specs/01-math/par-inflow-model.md` extracted from MATH_FORMULATIONS §9 (9.1-9.11)
- [ ] `docs/specs/01-math/inflow-nonnegativity.md` extracted from MATH_FORMULATIONS §10 (10.1-10.7)
- [ ] PAR spec includes: model definition, fitting steps (1-5), residual computation, parameter set, model order selection, CEPEL variant note, validation checks
- [ ] Non-negativity spec includes: problem statement, all 4 methods, comparison summary
- [ ] All mathematical formulas preserved exactly
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/par-inflow-model.md`
- `docs/specs/01-math/inflow-nonnegativity.md`

## Technical Details

### par-inflow-model.md

Source: `MATHEMATICAL_FORMULATIONS.md` §9 (9.1-9.11)

- §9.1 PAR(p) model definition
- §9.2 Notation for fitting
- §9.3-9.7 Steps 1-5 of model fitting (means/stdev, autocorrelations, Yule-Walker, original units, residual stdev)
- §9.8 Complete PAR(p) parameter set
- §9.9 Model order selection
- §9.10 CEPEL PAR(p)-A variant (future extension note)
- §9.11 Validation checks

### inflow-nonnegativity.md

Source: `MATHEMATICAL_FORMULATIONS.md` §10 (10.1-10.7)

- §10.1 Problem statement — why negative inflows are an issue
- §10.2-10.5 Four methods: none, penalty, truncation, truncation+penalty
- §10.6 Comparison summary table
- §10.7 Reference

## Definition of Done

Both files created with complete stochastic modeling content, valid cross-references, under 500 lines each.
