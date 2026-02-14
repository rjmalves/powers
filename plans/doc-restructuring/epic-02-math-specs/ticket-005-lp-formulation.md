# T-005: Extract LP Formulation Spec

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md)
- T-004b (system-elements.md — referenced for element context)

## Description

Extract the complete LP subproblem formulation into a focused spec covering the objective function, all constraints, and slack/penalty variables. This spec assumes the reader has already read `system-elements.md` and understands what physical elements exist in the system.

## Acceptance Criteria

- [ ] `docs/specs/01-math/lp-formulation.md` extracted from MATH_FORMULATIONS §5 (5.0-5.10)
- [ ] Covers: cost/penalty taxonomy, objective function, load balance, water balance, AR dynamics, hydro generation, outflow, minimum constraints, slack penalties, generic constraints, Benders cuts
- [ ] All mathematical formulas preserved exactly
- [ ] References `system-elements.md` for element descriptions (NOT duplicating §3 content)
- [ ] References notation spec for variable definitions
- [ ] Under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/lp-formulation.md`

## Technical Details

Source: `MATHEMATICAL_FORMULATIONS.md` §5 (5.0-5.10)

Note: §3 (System Element Modeling Overview) is handled by T-004b → `system-elements.md`. This spec focuses exclusively on the assembled LP formulation.

- §5.0 Cost and penalty taxonomy → operational costs vs violation penalties
- §5.1 Objective function → complete formulation with all terms
- §5.2 Load balance → per-bus, per-block balance equation
- §5.3 Hydro water balance → storage dynamics
- §5.4 AR inflow dynamics → state transition for AR model
- §5.5 Hydro generation constraints → bounds, generation limits
- §5.6 Outflow constraints → turbining + spillage
- §5.7 Minimum constraints → minimum generation, storage
- §5.8 Slack penalties and soft constraints → deficit, surplus, slack variables
- §5.9 Generic constraints → user-defined linear constraints
- §5.10 Benders cuts → future cost function approximation

## Definition of Done

File created with all LP formulation content preserved, valid cross-references, under 500 lines.
