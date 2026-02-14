# T-018: Extract Scenario Generation Spec

## Epic

Epic 4: Architecture Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the scenario generation pipeline (PAR preprocessing, noise sampling, external scenarios, memory layout) into a focused spec.

## Acceptance Criteria

- [ ] `docs/specs/03-architecture/scenario-generation.md` extracted from ARCHITECTURE §8-11 (PAR preprocessing, noise sampling, external scenarios, scenario memory layout)
- [ ] Covers: PAR preprocessing workflow, memory layout for hot-path, PAR fitting from historical data, Yule-Walker implementation, correlated noise generation, reproducible sampling, noise caching, external scenario sources, adapter interface, noise inversion, per-rank distribution, NUMA-aware allocation
- [ ] Under 500 lines, correct frontmatter
- [ ] References `01-math/par-inflow-model.md` for the mathematical definition

## Files to Create

- `docs/specs/03-architecture/scenario-generation.md`

## Technical Details

Source: `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` §8-11

- §8.1 PAR(p) model overview (brief — point to math spec)
- §8.2 Preprocessing workflow
- §8.3 Memory layout for hot-path access (struct of arrays for cache efficiency)
- §8.5 PAR model fitting from historical data (Yule-Walker implementation details)
- §9.1 Correlated noise generation (Cholesky decomposition)
- §9.2 Reproducible sampling (seeded RNG, deterministic across ranks)
- §9.3 Noise caching strategy
- §10.1-10.2 External scenario sources and adapter interface
- §10.5 Noise inversion for external scenarios
- §11.1 Memory organization
- §11.2 Per-rank scenario distribution
- §11.3 NUMA-aware allocation

Note: The mathematical content of PAR(p) is in `01-math/par-inflow-model.md`. This spec focuses on the implementation architecture.

## Definition of Done

File created with complete scenario generation content, valid cross-references, under 500 lines.
