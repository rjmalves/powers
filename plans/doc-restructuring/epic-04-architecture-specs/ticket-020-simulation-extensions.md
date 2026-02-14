# T-020: Extract Simulation Architecture and Extension Points Specs

## Epic

Epic 4: Architecture Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract simulation execution and extension points (trait abstractions, risk measure implementations, horizon modes) into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/03-architecture/simulation-architecture.md` extracted from ARCHITECTURE §17-19 (simulation execution, output writing)
- [ ] `docs/specs/03-architecture/extension-points.md` extracted from ARCHITECTURE §27-29 (trait abstractions, risk implementations, horizon modes)
- [ ] Simulation covers: simulation execution flow, output writer implementation, Parquet output schema, distributed output coordination
- [ ] Extension points covers: extensibility architecture, core trait definitions, factory pattern, risk-neutral/CVaR/convex implementations, finite/infinite/periodic horizon modes
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/03-architecture/simulation-architecture.md`
- `docs/specs/03-architecture/extension-points.md`

## Technical Details

### simulation-architecture.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §17-19

- §17 Simulation execution flow (how simulation differs from training)
- §18 (if exists) Additional simulation details
- §19.1 Policy evaluation
- §19.2 Output writer implementation
- §19.3 Parquet output schema (runtime)
- §19.4 Distributed output coordination (how ranks coordinate writing)

### extension-points.md

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §27-29

- §27.1 Extensibility architecture overview
- §27.2 Core trait definitions (trait hierarchies for algorithm variants)
- §27.3 Factory pattern for configuration-driven selection
- §28.1 Expected value (risk-neutral) implementation
- §28.2 CVaR implementation
- §28.3 Convex combination risk implementation
- §29.1 Finite horizon
- §29.2 Infinite horizon with uniform discounting
- §29.3 Infinite horizon with periodic structure
- §29.4 Stage configuration for periodic horizon

References:

- `01-math/risk-measures.md` for risk measure math
- `01-math/discount-rate.md` for discounting math

## Definition of Done

Both files created with complete content, valid cross-references, under 500 lines each.
