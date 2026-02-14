# T-008: Extract Cut Management and Stopping Rules Specs

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md)

## Description

Extract cut management (generation, selection, aggregation) and stopping rules into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/01-math/cut-management.md` extracted from MATH_FORMULATIONS §11 (11.1-11.7) and §12 (12.1-12.9)
- [ ] `docs/specs/01-math/stopping-rules.md` extracted from MATH_FORMULATIONS §13 (13.1-13.8)
- [ ] Cut management covers: dual extraction, coefficient computation, single-cut aggregation, multi-cut reference, cut addition algorithm, cut validity, activity definition, Level-1, LML1, dominated detection, threshold parameter, configuration, convergence guarantee
- [ ] Stopping rules covers: all available rules, combining rules, output on termination
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/cut-management.md`
- `docs/specs/01-math/stopping-rules.md`

## Technical Details

### cut-management.md

Source: `MATHEMATICAL_FORMULATIONS.md` §11 (11.1-11.5, 11.7) and §12 (12.1-12.9)

- §11.1 Dual variable extraction
- §11.2 Cut coefficient computation
- §11.3 Single-cut aggregation
- §11.4 Multi-cut → reference to deferred-features.md
- §11.5 Cut addition algorithm
- §11.7 Cut validity
- §12.1 Motivation for cut selection
- §12.2 Cut activity definition
- §12.3 Level-1 cut selection
- §12.4 Limited Memory Level-1 (LML1)
- §12.5 Dominated cut detection
- §12.6-12.9 Configuration, convergence guarantee, reference

### stopping-rules.md

Source: `MATHEMATICAL_FORMULATIONS.md` §13 (13.1-13.8)

- §13.1 Available stopping rules overview
- §13.2 Iteration limit (mandatory)
- §13.3 Time limit
- §13.4 Statistical stopping
- §13.5 Bound stalling
- §13.6 Simulation-based stopping (recommended)
- §13.7 Combining rules
- §13.8 Output on termination

## Definition of Done

Both files created with complete content, valid cross-references, under 500 lines each.
