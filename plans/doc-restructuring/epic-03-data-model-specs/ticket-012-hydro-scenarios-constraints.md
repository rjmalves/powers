# T-012: Extract Hydro Extensions, Scenarios, and Constraints Specs

## Epic

Epic 3: Data Model Specs

## Dependencies

- T-001 (directory structure)
- T-011 (input-system-entities.md — referenced for hydro core schema)

## Description

Extract the hydro extension files, scenario/time-series inputs, and constraints into focused specs.

## Acceptance Criteria

- [ ] `docs/specs/02-data-model/input-hydro-extensions.md` extracted from DATA_MODEL §3.5.1-3.5.6 (geometry, production models, production data, FPHA hyperplanes, pumping stations, energy contracts)
- [ ] `docs/specs/02-data-model/input-scenarios.md` extracted from DATA_MODEL §3.7-3.12 (stages, inflow models, load factors, exchange factors, correlation)
- [ ] `docs/specs/02-data-model/input-constraints.md` extracted from DATA_MODEL §3.9 (initial conditions), §3.13 (generic constraints), §3.14 (policy directory)
- [ ] All schema examples and validation rules preserved
- [ ] Each input file has a "Format Rationale" callout explaining why JSON or Parquet was chosen based on the nature of the data
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/02-data-model/input-hydro-extensions.md`
- `docs/specs/02-data-model/input-scenarios.md`
- `docs/specs/02-data-model/input-constraints.md`

## Technical Details

### input-hydro-extensions.md

Source: `DATA_MODEL_SPECIFICATION.md` §3.5.1-3.5.6

For each file, include a **Format Rationale** box:

- `hydro_geometry.parquet` → **Time series** — per-entity tabular data (head-storage curves) with many rows per hydro; Parquet gives columnar compression and efficient partial reads
- `hydro_production_models.json` → **Registry** — small config selecting model type per hydro with nested optional params; JSON handles optional/nested structures well
- `hydro_production_data.parquet` → **Time series** — per-entity tabular data (turbine/efficiency curves) with many rows; Parquet for typed columns and compression
- `fpha_hyperplanes.parquet` → **Time series** — large per-entity tabular data (pre-computed hyperplane coefficients); Parquet for efficient batch reads
- `pumping_stations.json` → **Registry** — small set of pump-turbine coupling definitions with cross-references; JSON is natural
- `energy_contracts.json` → **Registry** — contract definitions with nested structure and cross-references; JSON is natural

Then the schema content:

- §3.5.1 Hydro geometry (parquet) — optional, head-storage curves
- §3.5.2 Hydro production models (JSON) — optional, model type selection
- §3.5.3 Hydro production data (parquet) — optional, turbine/efficiency data
- §3.5.4 FPHA hyperplanes (parquet) — optional, pre-computed hyperplanes
- §3.5.5 Pumping stations (JSON) — optional, pump-turbine coupling
- §3.5.6 Energy contracts (JSON) — optional, import/export contracts
- §3.5.7, §3.5.8 (deferred) → reference to deferred-features.md

### input-scenarios.md

Source: `DATA_MODEL_SPECIFICATION.md` §3.7-3.12

For each file, include a **Format Rationale** box:

- `stages.json` → **Complex nested object** — stage definitions with nested block structures and optional seasonal parameters; JSON handles hierarchical config naturally
- `scenarios/inflow_models.parquet` → **Time series** — large per-entity-per-stage tabular data (PAR coefficients, historical inflows); Parquet for columnar compression and typed columns
- `scenarios/load_factors.json` → **Default-with-overrides** — small number of load factor definitions that rarely change; JSON for readability and simplicity
- `scenarios/exchange_factors.json` → **Default-with-overrides** — small number of exchange factor definitions; JSON for readability
- `scenarios/correlation.json` → **Correlation / matrix data** — symmetric correlation matrices between entities; JSON because data is small and structure is not tabular

Then the schema content:

- §3.7 Stage definitions (stages.json)
- §3.8 Inflow models (scenarios/inflow_models.parquet)
- §3.10 Load factors by block (scenarios/load_factors.json) — optional
- §3.11 Exchange factors by block (scenarios/exchange_factors.json) — optional
- §3.12 Correlation (scenarios/correlation.json)

### input-constraints.md

Source: `DATA_MODEL_SPECIFICATION.md` §3.9, §3.13, §3.14

For each file, include a **Format Rationale** box:

- `initial_conditions.json` → **Registry** — one-time snapshot of system state with cross-references to entities; JSON is natural for config-like data
- `constraints/*.json` → **Complex nested object** — constraint definitions with variable references, coefficients, and conditional logic; JSON handles polymorphic structures well
- `policy/` → **Policy / binary data** — pre-computed cuts for warm-start; FlatBuffers for zero-copy deserialization during hot-path loading (see binary-formats.md)

Then the schema content:

- §3.9 Initial conditions (initial_conditions.json)
- §3.13 Generic constraints (constraints/) — constraint definition format
- §3.14 Policy directory (policy/) — warm-start policy loading

## Definition of Done

All three files created with complete content, valid cross-references, under 500 lines each.
