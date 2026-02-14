# T-003: Extract Configuration Reference and Deferred Features Specs

## Epic

Epic 1: Foundation

## Dependencies

- T-001 (directory structure)

## Description

Create the configuration reference (all config-driven LP variants) and deferred features spec. These are cross-cutting and referenced by many other specs.

## Acceptance Criteria

- [ ] `docs/specs/05-config/configuration-reference.md` extracted from MATH_FORMULATIONS §18 (all config subsections) and §19 (cross-references)
- [ ] `docs/specs/06-deferred/deferred-features.md` extracted from MATH_FORMULATIONS Appendix C and relevant "DEFERRED" markers from DATA_MODEL §3.5.7, §3.5.8
- [ ] Each file has correct frontmatter
- [ ] Each file is under 500 lines
- [ ] Config reference includes the complete example configuration from §18.10

## Files to Create

- `docs/specs/05-config/configuration-reference.md`
- `docs/specs/06-deferred/deferred-features.md`

## Technical Details

### configuration-reference.md

Source: `MATHEMATICAL_FORMULATIONS.md` §18 (18.1-18.10) and §19 (19.1-19.4)

- Block mode configuration
- Hydro production function config
- Inflow non-negativity treatment config
- Cut management config
- Discount rate config
- Horizon mode config
- Upper bound evaluation config
- Risk measures config
- Penalty coefficients config
- Complete example `config.json`
- Cross-reference mapping table (old section → new spec)
- Variable correspondence table
- Rust struct correspondence table

### deferred-features.md

Source: `MATHEMATICAL_FORMULATIONS.md` Appendix C (C.1-C.7), `DATA_MODEL_SPECIFICATION.md` §3.5.7, §3.5.8, `MATHEMATICAL_FORMULATIONS.md` §8.5, §8.6

- GNL thermal plants
- Battery energy storage systems
- Multi-cut formulation
- Markovian policy graphs
- Non-controllable sources (wind/solar)
- FPHA enhancements
- Temporal scope decoupling
- For each: brief description, why deferred, prerequisites, estimated effort

## Definition of Done

Both files created with complete content, valid cross-references, and under 500 lines.
