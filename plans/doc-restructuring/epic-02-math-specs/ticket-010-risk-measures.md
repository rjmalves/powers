# T-010: Extract Risk Measures Spec

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md)

## Description

Extract the risk-averse SDDP formulation (CVaR) into a focused spec.

## Acceptance Criteria

- [ ] `docs/specs/01-math/risk-measures.md` extracted from MATH_FORMULATIONS §17 (17.1-17.12)
- [ ] Covers: motivation, CVaR definition, convex combination risk measure, dual representation, risk-averse subgradient theorem, risk-averse Bellman equation, cut generation with risk, per-stage profiles, implementation notes, upper bound with risk, lower bound validity
- [ ] All mathematical formulas preserved exactly
- [ ] Under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/risk-measures.md`

## Technical Details

Source: `MATHEMATICAL_FORMULATIONS.md` §17 (17.1-17.12)

- §17.1 Motivation for risk-averse optimization
- §17.2 CVaR definition and properties
- §17.3 Convex combination risk measure (SDDP.jl convention: λ·E + (1-λ)·CVaR_α)
- §17.4 Dual representation of convex risk measures
- §17.5 Risk-averse subgradient theorem
- §17.6 Risk-averse Bellman equation
- §17.7 Cut generation with risk measures — modified dual weights
- §17.8 Per-stage risk profiles — different risk parameters per stage
- §17.9 Implementation notes — practical considerations
- §17.10 Upper bound with risk measures
- §17.11 Reference
- §17.12 Lower bound validity with risk measures

## Definition of Done

File created with complete risk measure content, valid cross-references, under 500 lines.
