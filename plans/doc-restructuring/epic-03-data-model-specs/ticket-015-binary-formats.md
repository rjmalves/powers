# T-015: Extract Binary Formats and Internal Data Structures Specs

## Epic

Epic 3: Data Model Specs

## Dependencies

- T-001 (directory structure)

## Description

Extract binary format decisions (FlatBuffers, Parquet config) and core algorithm data structures into focused specs. **Note**: the solver interface (§5.4) and LP scaling (§5.5) have been moved to T-020b (solver-abstraction) — they are NOT part of this ticket.

## Acceptance Criteria

- [ ] `docs/specs/02-data-model/binary-formats.md` extracted from DATA_MODEL §7 (7.1-7.3: summary table, FlatBuffers for policy, Parquet configuration) and §5.1-5.3 (core algorithm structures, LP subproblem structure, FCF with replication)
- [ ] **Solver interface (§5.4) and LP scaling (§5.5) are NOT included** — those are handled by T-020b (solver-abstraction.md)
- [ ] All FlatBuffers schema examples and Parquet config preserved
- [ ] Includes a "Format Decision Framework" section that consolidates the rationale for when to use JSON vs Parquet vs FlatBuffers, referencing the classification table from the master plan
- [ ] Under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/02-data-model/binary-formats.md`

## Technical Details

### binary-formats.md

Source: `DATA_MODEL_SPECIFICATION.md` §5.1-5.3, §7

This spec consolidates:

- §7.1 Summary table — which format for which purpose (JSON for config, Parquet for time-series, FlatBuffers for policy)
- §7.2 FlatBuffers for policy data — decision rationale, schema definition, zero-copy advantages
- §7.3 Parquet configuration — compression, row groups, column encoding
- §5.1 Core algorithm structures — Rust structs for stage data, scenario data, iteration state
- §5.2 LP subproblem structure — variable/constraint indexing, column layout
- §5.3 FCF with replication — future cost function storage and replication for thread safety

**Moved to T-020b**:

- ~~§5.4 Solver interface specification~~ → `solver-abstraction.md`
- ~~§5.5 LP scaling specification~~ → `solver-abstraction.md` or `solver-workspaces.md`

### Format Decision Framework

This spec should open with a consolidated **Format Decision Framework** that serves as the authoritative reference for all format choices across the data model. The framework maps data nature to format:

| Data Nature            | Format         | Key Examples in POWE.RS                                         |
| ---------------------- | -------------- | --------------------------------------------------------------- |
| Registry / catalog     | JSON           | buses.json, hydros.json, thermals.json, lines.json              |
| Time series            | Parquet        | inflow_models.parquet, hydro_geometry.parquet, fpha_hyperplanes |
| Default-with-overrides | JSON + Parquet | penalties.json (base) + stage overrides (Parquet)               |
| Complex nested object  | JSON           | config.json, stages.json, constraints/\*.json                   |
| Correlation / matrix   | JSON           | correlation.json                                                |
| Policy / binary        | FlatBuffers    | policy cuts, checkpoint data                                    |
| High-volume output     | Parquet        | simulation results, training outputs                            |

Each individual data model spec (T-011, T-012, T-013, T-014) references this framework when justifying its per-file format choices.

Note: §5.1-5.3 content (core algorithm structures, LP subproblem, FCF) is closely tied to data formats and internal representation. If it makes the file too long (>500 lines), split into a separate `internal-structures.md`.

## Definition of Done

File(s) created with complete binary format and internal structure content (excluding solver interface), valid cross-references, under 500 lines each.
