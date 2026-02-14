# T-009: Extract Discount Rate and Upper Bound Evaluation Specs

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md)

## Description

Extract the discount rate formulation and upper bound evaluation specs (advanced formulations).

## Acceptance Criteria

- [ ] `docs/specs/01-math/discount-rate.md` extracted from MATH_FORMULATIONS §14 (14.1-14.5)
- [ ] `docs/specs/01-math/upper-bound-evaluation.md` extracted from MATH_FORMULATIONS §15-16 (all subsections)
- [ ] Discount rate covers: motivation, discounted Bellman, stage-dependent rates, modified subproblem, cumulative discounting
- [ ] Upper bound covers: inner approximation, Lipschitz interpolation, constant computation, vertex values, evaluation LP, linearized LP, gap computation, vertex storage, configuration, computational considerations
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/discount-rate.md`
- `docs/specs/01-math/upper-bound-evaluation.md`

## Technical Details

### discount-rate.md

Source: `MATHEMATICAL_FORMULATIONS.md` §14 (14.1-14.5)

- §14.1 Motivation — why discount rates in long-horizon planning
- §14.2 Discounted Bellman equation
- §14.3 Stage-dependent discount rates
- §14.4 Modified stage subproblem with discounting
- §14.5 Cumulative discounting

### upper-bound-evaluation.md

Source: `MATHEMATICAL_FORMULATIONS.md` §15-16 (15.1-15.5, 16.1-16.12)

- §15.1-15.5 Upper bound methods overview
- §16.1 Motivation and theory
- §16.2 Vertex-based inner approximation
- §16.3 Lipschitz interpolation
- §16.4 Lipschitz constant computation
- §16.5 Vertex value computation
- §16.6 Upper bound evaluation LP
- §16.7 Linearized upper bound LP
- §16.8 Gap computation
- §16.9 Vertex storage
- §16.10 Configuration
- §16.11 Computational considerations
- §16.12 References

## Definition of Done

Both files created with complete content, valid cross-references, under 500 lines each.
