# T-014: Extract Output Schemas and Infrastructure Specs

## Epic

Epic 3: Data Model Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract the output data model into two focused specs: output schemas (simulation/training data) and output infrastructure (manifest, metadata, partitioning, distributed writing).

## Acceptance Criteria

- [ ] `docs/specs/02-data-model/output-schemas.md` extracted from DATA_MODEL §4.1-4.6 (directory structure, design principles, categorical codes, dictionary files, simulation schemas, training schemas)
- [ ] `docs/specs/02-data-model/output-infrastructure.md` extracted from DATA_MODEL §4.7-4.12 (manifest, metadata, hive partitioning, output config, production scale, validation/integrity)
- [ ] All Parquet schema definitions and code examples preserved
- [ ] Each file under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/02-data-model/output-schemas.md`
- `docs/specs/02-data-model/output-infrastructure.md`

## Technical Details

### output-schemas.md

Source: `DATA_MODEL_SPECIFICATION.md` §4.1-4.6

- §4.1 Output directory structure overview
- §4.2 Design principles for output
- §4.3 Categorical code definitions (how encoded values map to meanings)
- §4.4 Dictionary files
- §4.5 Simulation output schemas — per-entity Parquet schemas for hydro, thermal, bus, line, etc.
- §4.6 Training output schemas — convergence data, cut evolution, iteration stats

### output-infrastructure.md

Source: `DATA_MODEL_SPECIFICATION.md` §4.7-4.12

- §4.7 Manifest files
- §4.8 Metadata file (training/metadata.json)
- §4.9 MPI direct hive partitioning — how ranks write to partitioned directories
- §4.10 Output configuration — what outputs to enable/disable
- §4.11 Production scale reference — output file sizes at scale
- §4.12 Validation and integrity — schema validation, row count verification, comparison tools

## Definition of Done

Both files created with complete output model content, valid cross-references, under 500 lines each.
