# T-011: Extract Input Directory Structure and System Entity Specs

## Epic

Epic 3: Data Model Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the input data model: directory structure, config.json, and core system entities (buses, lines, hydros, thermals).

## Acceptance Criteria

- [ ] `docs/specs/02-data-model/input-directory-structure.md` extracted from DATA_MODEL §3.1 (directory), §3.2 (config.json including penalties subsection)
- [ ] `docs/specs/02-data-model/input-system-entities.md` extracted from DATA_MODEL §3.3 (buses), §3.4 (lines), §3.5 (hydro registry — core schema only), §3.6 (thermals)
- [ ] All JSON schema examples preserved
- [ ] Each input file has a "Format Rationale" callout explaining why JSON was chosen for that file based on the nature of the data (registry, nested object, default-with-overrides, etc.)
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/02-data-model/input-directory-structure.md`
- `docs/specs/02-data-model/input-system-entities.md`

## Technical Details

### input-directory-structure.md

Source: `DATA_MODEL_SPECIFICATION.md` §3.1, §3.2

- §3.1 Complete input directory tree with descriptions
- §3.2 `config.json` schema — all fields, types, defaults, validation
- §3.2.1 Penalties and costs section within config → brief reference to penalty-system.md for full details

### input-system-entities.md

Source: `DATA_MODEL_SPECIFICATION.md` §3.3-3.6

For each entity file, include a **Format Rationale** box:

- `buses.json` → **Registry** — small set of entities with cross-references; JSON is natural
- `lines.json` → **Registry** — entity definitions referencing buses; JSON is natural
- `hydros.json` → **Registry** — complex entity with many optional nested fields (production model, storage limits, cascading topology); JSON handles optional/nested structures well
- `thermals.json` → **Registry** — entity definitions with cost curve nesting; JSON is natural

Then the schema content:

- §3.3 Buses (buses.json) — schema, required fields, validation rules
- §3.4 Lines (lines.json) — schema, flow limits, bus references
- §3.5 Hydro registry (hydros.json) — core schema only (name, bus, storage limits, etc.)
  - Hydro extensions (geometry, production models, FPHA, pumping, contracts) → reference to input-hydro-extensions.md
- §3.6 Thermals (thermals.json) — schema, cost curves, operational limits

## Definition of Done

Both files created with complete input schema content, valid cross-references, under 500 lines each.
