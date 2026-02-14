# T-004: Extract SDDP Algorithm Spec

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md — referenced for notation)

## Description

Extract the SDDP algorithm overview into a focused spec covering the algorithm structure, policy graph, state variables, and single vs multi-cut formulation.

## Acceptance Criteria

- [ ] `docs/specs/01-math/sddp-algorithm.md` extracted from MATH_FORMULATIONS §1, §2 (2.1-2.5)
- [ ] Covers: problem context, multistage stochastic formulation, SDDP algorithm description, policy graph structure, state variables, Markov property, single-cut vs multi-cut
- [ ] References `notation-conventions.md` for notation instead of duplicating it
- [ ] References `deferred-features.md` for multi-cut details
- [ ] Under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/sddp-algorithm.md`

## Technical Details

Source: `MATHEMATICAL_FORMULATIONS.md` §1 (1.1, 1.3) and §2 (2.1-2.5)

- §1.1 Document purpose → adapt as Purpose section
- §1.3 Problem context → system overview for the SDDP problem
- §2.1 Multistage stochastic formulation → Bellman equation, dynamic programming decomposition
- §2.2 The SDDP algorithm → forward pass, backward pass, convergence concept
- §2.3 Policy graph → finite horizon, infinite horizon structures
- §2.4 State variables and Markov property → what constitutes state, continuity
- §2.5 Single-cut vs multi-cut → comparison, when to use each

Note: §1.2 (notation) goes to `notation-conventions.md` instead.

## Definition of Done

File created with complete algorithmic content, valid cross-references, under 500 lines.
