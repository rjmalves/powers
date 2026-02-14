# T-013: Extract Penalty System Spec

## Epic

Epic 3: Data Model Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the penalty system into a dedicated spec. This is one of the areas the user expects to change, so it needs to be isolated for targeted review.

## Acceptance Criteria

- [ ] `docs/specs/02-data-model/penalty-system.md` extracted from DATA_MODEL §3.2.1 (penalties in config), and the existing `schemas/penalties.schema.json` and `examples/penalties.example.json`
- [ ] Covers: three-tier cascade (global → entity → stage), piecewise deficit, operational costs vs violation penalties, penalty JSON schema, override mechanisms
- [ ] References the existing schema file at `schemas/penalties.schema.json`
- [ ] Each input file has a "Format Rationale" callout explaining the format choice based on data nature
- [ ] Under 500 lines, correct frontmatter
- [ ] Frontmatter marks `status: needs-review` since user identified this as an area for changes

## Files to Create

- `docs/specs/02-data-model/penalty-system.md`

## Technical Details

Source: `DATA_MODEL_SPECIFICATION.md` §3.2.1, plus `schemas/penalties.schema.json`, `examples/penalties.example.json`

### Format Rationale

The penalty system uses a **default-with-overrides** pattern:

- `penalties.json` → **Default-with-overrides (base)** — global penalty defaults that apply system-wide, with nested structure for cost tiers and categories; JSON is natural for hierarchical config with optional sections
- Entity-level overrides (in `hydros.json`, `thermals.json`, etc.) → **Registry (embedded)** — per-entity penalty overrides are embedded as optional fields within entity JSON files; keeps overrides co-located with the entity they modify
- Stage-level overrides (in Parquet time-series files) → **Time series (overrides)** — per-stage penalty variations for time-varying costs; Parquet for sparse override rows that override defaults only when values change over time

This three-tier pattern (JSON global defaults → JSON entity overrides → Parquet stage overrides) is documented as the **override resolution algorithm**: stage parquet > entity JSON > penalties.json > built-in defaults.

### Content

- Three-tier cascade logic: `penalties.json` sets global defaults, entity JSONs can override per-entity, parquet stage overrides for time-varying values
- Piecewise deficit: multiple cost tiers with final infinite segment for LP feasibility
- Cost taxonomy: operational costs (exchange_cost, thermal costs) vs violation penalties (deficit, surplus, slack)
- Complete penalty JSON schema documentation
- Override resolution algorithm: stage parquet > entity JSON > penalties.json > built-in defaults
- Example penalty configuration walkthrough

Cross-references:

- `01-math/lp-formulation.md` §5.0 (cost taxonomy) and §5.8 (slack penalties)
- `05-config/configuration-reference.md` §18.9 (penalty coefficients config)

## Definition of Done

File created with complete penalty system content, valid cross-references, under 500 lines. Frontmatter marked as `needs-review`.
