# Data Model Change Tracker

Track all data model changes discovered during spec reviews. Each change should be recorded here when identified, then marked as resolved once the corresponding spec is updated.

## How to Use

1. During a spec review, when a data model change is identified, add a row to the relevant table below
2. Set **Change Type** to one of: `add field`, `remove field`, `rename`, `change type`, `restructure`, `new file`, `remove file`
3. Set **Status** to one of: `identified`, `approved`, `applied`, `rejected`
4. After updating the spec, change status to `applied` and note the commit or date

---

## Future Modeling Observations

Issues identified during review that may require new variables, constraints, or penalty types in future versions. These are NOT changes to current specs but flags for P2 math spec reviews.

| Observation                                                                                                                                                                                                                                                                                                                                                                               | Affected Specs                                                                                      | CEPEL Reference                                                                                                                                      | Status  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **Lateral flow (Vazão lateral)**: Some plants (e.g. Belo Monte/Pimental, Itaipu) have lateral flows from river posts or other plants' outflows that affect the tailwater level and production function. Our current model does not include `Q_lat` as a variable. This may require additional flow variables and potentially associated penalties.                                        | `system-elements.md`, `lp-formulation.md`, `hydro-production-models.md`, `penalty-system.md`        | [Canal de fuga — Vazão lateral](https://see.cepel.br/manual/libs/latest/usinas_hidreletricas/componentes_usinas/canal_fuga.html#vazao-lateral)       | flagged |
| **Downstream flow formulation (Vazão de jusante)**: The downstream flow `Q_jus` can include participation factors (`k_jus^Q`, `k_jus^S`, `k_jus^qa`, `k_jus^qd`) for turbined flow, spillage, lateral post inflows, and other plants' outflows. Our current `o = q + s` is the simplest case. Plants with non-trivial `Q_jus` formulations may need additional variables and constraints. | `system-elements.md`, `lp-formulation.md`, `hydro-production-models.md`, `input-system-entities.md` | [Canal de fuga — Vazão de jusante](https://see.cepel.br/manual/libs/latest/usinas_hidreletricas/componentes_usinas/canal_fuga.html#vazao-de-jusante) | flagged |
| **Water travel time propagation curves (Curva de Propagação da Água)**: Beyond simple translation (`τ_ij` delay), CEPEL models support propagation curves where outflow arrives fractionally over `[τ_min, τ_max]` with participation percentages. Also requires special handling at end-of-horizon (water in transit added to downstream storage for FCF coupling).                      | `system-elements.md`, `lp-formulation.md`, `input-system-entities.md`, `input-hydro-extensions.md`  | [Tempo de viagem da água](https://see.cepel.br/manual/libs/latest/usinas_hidreletricas/curso_rios/tempo_viagem_agua.html)                            | flagged |

---

## General Observations

Cross-cutting review observations that apply to multiple specs.

| Observation                                                                                                                                                                                                                                                                                                                                                                                                                          | Affected Specs                                                                                                                                             | Status   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| **Diagrams will be revised after text**: All diagrams (SVG, Mermaid, etc.) referenced across specs will be reviewed and updated only after the text part of the documentation review is complete. Diagram accuracy depends on finalized text content.                                                                                                                                                                                | All specs referencing diagrams                                                                                                                             | noted    |
| **`block_mode` moved from global config to per-stage**: `input-scenarios.md` now defines `block_mode` per stage. `block-formulations.md` and `configuration-reference.md` still reference `modeling.block_mode` as a global setting — will be updated when those specs are reviewed.                                                                                                                                                 | `block-formulations.md`, `configuration-reference.md`                                                                                                      | noted    |
| **`discount_rate` moved from per-transition to `policy_graph`**: `input-scenarios.md` now defines discount rate as `annual_discount_rate` in `policy_graph` with per-transition override. `discount-rate.md` §14.3 still shows per-transition `discount_rate` as a plain rate — will be updated during P2 review.                                                                                                                    | `discount-rate.md`                                                                                                                                         | noted    |
| **`$schema` placeholders**: All JSON examples in approved data model specs now include `$schema` placeholder fields for future JSON Schema validation. Apply to remaining specs as they are reviewed.                                                                                                                                                                                                                                | All data model specs with JSON examples                                                                                                                    | noted    |
| **Filling model impact on math specs**: The filling model redesign (storage slack, terminal constraint) requires updates to `lp-formulation.md` (line 110 references `target_storage_hm3` as constraint target) and `system-elements.md` (line 195-197 filling hydro subset). Bottom discharge was deferred to simulation-only and does not affect math specs. These are P2 math specs and will be reviewed at their scheduled time. | `lp-formulation.md`, `system-elements.md`                                                                                                                  | resolved |
| **Variable units convention: rate units adopted**: All LP decision variables use rate units (MW, m³/s). Block duration τ_k is an external multiplier in the objective and water balance conversion. Duals require ÷τ_k post-processing for $/MWh output. Decision documented in `system-elements.md` Variable Units Convention section. Affects dual interpretation in output specs and solver abstraction.                          | `system-elements.md`, `output-schemas.md`, `lp-formulation.md`, `solver-abstraction.md`                                                                    | noted    |
| **Linearized head is simulation-only**: The linearized head production model is excluded from training because the bilinear term ($q \times v^{avg}$) changes the LP between iterations, breaking SDDP convergence guarantees. Only `constant_productivity` and `fpha` are valid during training. Linearized head is available during simulation (single forward pass, no cuts) for higher-fidelity analytics.                       | `hydro-production-models.md`, `system-elements.md`, `lp-formulation.md`, `input-system-entities.md`, `input-hydro-extensions.md`, `internal-structures.md` | noted    |

---

## Specs Pending Re-Review

Previously approved specs that received changes during the review of other specs. These need re-review to confirm the cross-cutting changes are correct.

| Spec File                   | Original Approval | Changed During                  | Changes Applied                                                                                                                                                                                                                                                                                 | Status      |
| --------------------------- | ----------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `input-system-entities.md`  | 2026-02-15        | `internal-structures.md` review | §7 non-controllable sources: replaced deferred stub with full entity definition (bus, lifecycle, max_generation_mw, curtailment_cost, JSON example, field table, operative states). Added GNL exceptional validation rule to §4. Added internal-structures.md cross-reference.                  | re-approved |
| `penalty-system.md`         | 2026-02-14        | `internal-structures.md` review | Added `fpha_turbined_cost` and `curtailment_cost` to Category 3 regularization table. Added both to penalties.json example. Added `fpha_turbined_cost` to hydro overrides table. Added non-controllable source penalty overrides and constraint violation sections. Updated objective function. | re-approved |
| `binary-formats.md`         | 2026-02-16        | Format propagation 2026-02-17   | Closed TBD rows in §1 framework table and §2 summary table. All tabular data → Parquet.                                                                                                                                                                                                         | re-approved |
| `input-hydro-extensions.md` | 2026-02-15        | Format propagation 2026-02-17   | Added `.parquet` extensions to `hydro_geometry` and `fpha_hyperplanes` headers. Updated format rationale text.                                                                                                                                                                                  | re-approved |
| `penalty-system.md`         | 2026-02-16        | Format propagation 2026-02-17   | Closed format TBD → Parquet. Removed open question. Split penalty overrides into 4 entity-specific `.parquet` files.                                                                                                                                                                            | re-approved |
| `input-constraints.md`      | 2026-02-15        | Format propagation 2026-02-17   | Added `.parquet` extensions to all bounds files. Added `exchange_factors.json` section. Closed `generic_constraint_bounds` format.                                                                                                                                                              | re-approved |
| `input-scenarios.md`        | 2026-02-15        | Format propagation 2026-02-17   | Split inflow_models → 2 Parquet files. Renamed load_models. Removed §5 Exchange Factors. Embedded correlation schedule. Renumbered §6→§5, §7→§6.                                                                                                                                                | re-approved |
| `internal-structures.md`    | 2026-02-16        | Format propagation 2026-02-17   | Updated inflow model source refs to split files. Renamed Load Models → Load Seasonal Statistics. Fixed correlation §-reference.                                                                                                                                                                 | re-approved |
| `system-elements.md`        | 2026-02-19        | Linearized head 2026-02-20      | §5 Production Function: linearized_head annotated as simulation-only, added training-models-only blockquote.                                                                                                                                                                                    | re-approved |
| `lp-formulation.md`         | 2026-02-19        | Linearized head 2026-02-20      | §6: removed linearized_head from training LP, updated section intro.                                                                                                                                                                                                                            | re-approved |
| `input-system-entities.md`  | 2026-02-16        | Linearized head 2026-02-20      | §3: linearized_head variant annotated as simulation-only.                                                                                                                                                                                                                                       | re-approved |
| `input-hydro-extensions.md` | 2026-02-15        | Linearized head 2026-02-20      | §2: model hierarchy annotated, JSON example note added, field tables updated.                                                                                                                                                                                                                   | re-approved |
| `internal-structures.md`    | 2026-02-17        | Linearized head 2026-02-20      | §3: Generation Model table expanded with Phase column, added simulation-only explanatory note.                                                                                                                                                                                                  | re-approved |

---

## Input Schemas

Changes to input file formats, directory structure, and entity definitions.

| Spec File                   | Change Type | Description                                                                                                                  | Status  |
| --------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------- | ------- |
| `input-system-entities.md`  | restructure | Fixed priority from 2-high to 1-critical                                                                                     | applied |
| `input-system-entities.md`  | restructure | Buses: removed "2-10" count assumption, clarified as general network concept (hundreds/thousands)                            | applied |
| `input-system-entities.md`  | restructure | Buses: added explicit penalty system reference for deficit as recourse slack (Category 1)                                    | applied |
| `input-system-entities.md`  | restructure | Lines: removed "5-20" count assumption, scales with buses                                                                    | applied |
| `input-system-entities.md`  | restructure | Lines: added explicit penalty system reference for exchange_cost as regularization (Category 3), added Default column        | applied |
| `input-system-entities.md`  | restructure | Hydros: made `min_generation_mw` and `max_generation_mw` mandatory (removed null = derived)                                  | applied |
| `input-system-entities.md`  | restructure | Hydros: restructured generation model as tagged union with 3 variants (constant_productivity, linearized_head, fpha)         | applied |
| `input-system-entities.md`  | restructure | Hydros: removed specific file format references (e.g., `inflow_models.parquet`), made format-agnostic                        | applied |
| `input-system-entities.md`  | restructure | Hydros: changed `outflow.max_outflow_m3s` null description to "no flood control constraint"                                  | applied |
| `input-system-entities.md`  | restructure | Hydros: added explicit penalty system reference for diversion as regularization (Category 3)                                 | applied |
| `input-system-entities.md`  | restructure | Hydros: added CEPEL input schema impact analysis as future extension notes (lateral flow, Q_jus, travel time)                | applied |
| `input-system-entities.md`  | restructure | Hydros: flagged decommissioned/non-existing LP behavior as open question for P2 review                                       | applied |
| `input-system-entities.md`  | restructure | Hydros: added future extension note for generating units                                                                     | applied |
| `input-system-entities.md`  | restructure | Thermals: removed `thermal_bounds.parquet` schema section (format TBD)                                                       | applied |
| `input-system-entities.md`  | add field   | Added §5 Pumping Stations as independent system element (moved from input-hydro-extensions.md)                               | applied |
| `input-system-entities.md`  | add field   | Added §6 Energy Contracts as independent system element (moved from input-hydro-extensions.md)                               | applied |
| `input-system-entities.md`  | add field   | Added §7 Non-Controllable Sources as deferred stub section                                                                   | applied |
| `input-hydro-extensions.md` | restructure | Removed §5 (Pumping Stations), §6 (Energy Contracts), §7 (Deferred Features) — moved to input-system-entities.md             | applied |
| `input-hydro-extensions.md` | restructure | Fixed priority from 2-high to 1-critical, updated purpose and cross-references                                               | applied |
| `input-hydro-extensions.md` | restructure | §1 Hydro Geometry: fixed format rationale label to "Entity-level lookup table", stripped evaporation math formulas           | applied |
| `input-hydro-extensions.md` | restructure | §2 Production Models: added tagged union selection modes (stage_ranges, seasonal) with JSON examples and field tables        | applied |
| `input-hydro-extensions.md` | restructure | §2 Required Data table: removed `hydro_production_data.parquet` column, references hydro object fields instead               | applied |
| `input-hydro-extensions.md` | remove file | Deleted §3 (Hydro Production Data) — tailrace/losses/efficiency moved to hydro object in input-system-entities.md            | applied |
| `input-hydro-extensions.md` | restructure | §4→§3 FPHA Hyperplanes: fixed format rationale to "Pre-computed coefficient table", stripped constraint form formula         | applied |
| `input-hydro-extensions.md` | add field   | §3 FPHA Hyperplanes: added `stage_id` column (nullable — null = valid for all stages)                                        | applied |
| `input-hydro-extensions.md` | rename      | §3 FPHA Hyperplanes: renamed `alpha_fpha` → `kappa` to match math spec terminology                                           | applied |
| `input-hydro-extensions.md` | restructure | Updated cross-references to remove production data refs, add system-elements and kappa references                            | applied |
| `input-system-entities.md`  | add field   | Hydros: added optional `tailrace` field (tagged union: polynomial/piecewise) — moved from hydro_production_data              | applied |
| `input-system-entities.md`  | add field   | Hydros: added optional `hydraulic_losses` field (tagged union: factor/constant) — moved from hydro_production_data           | applied |
| `input-system-entities.md`  | add field   | Hydros: added optional `efficiency` field (tagged union: constant, future flow_dependent) — moved from hydro_production_data | applied |
| `input-system-entities.md`  | add field   | Hydros: added optional `evaporation` field with `coefficients_mm: [f64; 12]` — 12 monthly values, previously missing         | applied |
| `input-system-entities.md`  | restructure | Hydro Extensions table: removed "Production data" row (data moved to hydro object)                                           | applied |
| `input-scenarios.md`        | restructure | Fixed priority from 2-high to 1-critical                                                                                     | applied |
| `input-scenarios.md`        | add field   | Added `season_definitions` section with `cycle_type` (monthly/weekly/custom) and season-to-calendar mapping                  | applied |
| `input-scenarios.md`        | add field   | Added same-duration validation rule: all stages sharing a `season_id` must have identical duration                           | applied |
| `input-scenarios.md`        | restructure | Added `policy_graph` top-level section with `type` (finite_horizon/cyclic), `annual_discount_rate`, and `transitions`        | applied |
| `input-scenarios.md`        | restructure | Discount rate specified as annual rate, system auto-converts to per-transition factor based on stage duration                | applied |
| `input-scenarios.md`        | add field   | Added `scenario_source` top-level field (generated/historical/external) controlling forward pass behavior                    | applied |
| `input-scenarios.md`        | add field   | Added §2 Scenario Pipeline section documenting cascade flexibility (each component independently provided or derived)        | applied |
| `input-scenarios.md`        | add field   | Added §2.3 History Aggregation: system aggregates user-provided history at any resolution to match seasons                   | applied |
| `input-scenarios.md`        | add field   | Added §2.4 Inflow History schema: (hydro_id, date, value_m3s), format TBD                                                    | applied |
| `input-scenarios.md`        | add field   | Added §2.5 External Scenarios schema: indexed by stage_id (hydro_id, stage_id, scenario_id, value_m3s), format TBD           | applied |
| `input-scenarios.md`        | add field   | Documented reverse-noise calculation requirement for historical/external sources                                             | applied |
| `input-scenarios.md`        | restructure | Redesigned `state_variables` from string enum to object with boolean flags: `{"storage": true, "inflow_lags": true}`         | applied |
| `input-scenarios.md`        | add field   | Added `block_mode` per stage ("parallel"/"chronological") instead of global config                                           | applied |
| `input-scenarios.md`        | add field   | Added block-hour validation rule: sum of block hours must equal stage duration                                               | applied |
| `input-scenarios.md`        | add field   | Added `season_id` optional field per stage (i32 \| null)                                                                     | applied |
| `input-scenarios.md`        | restructure | Fixed format rationale labels: "Entity-stage parameter table" for inflow/load models                                         | applied |
| `input-scenarios.md`        | rename      | Renamed correlation `blocks` → `correlation_groups` in JSON structure and field reference table                              | applied |
| `input-scenarios.md`        | restructure | Documented exchange factors >1.0 as intentional (higher block-level capacity)                                                | applied |
| `input-scenarios.md`        | restructure | Documented noise distribution as standard normal, transformed via Cholesky                                                   | applied |
| `input-scenarios.md`        | restructure | AR coefficient storage: documented logical schema (ordered list per entity per stage), physical format TBD                   | applied |
| `input-scenarios.md`        | add field   | Added §7 Seasonal Override Pattern (cross-cutting): documented profile+schedule and stage/season tagged union approaches     | applied |
| `input-scenarios.md`        | restructure | Removed `historical` from sampling methods (now a `scenario_source` type, not a sampling method)                             | applied |
| `input-scenarios.md`        | restructure | Correlation schedule file format noted as TBD (part of broader format discussion)                                            | applied |

### Format Decision Propagation (2026-02-17)

Cross-cutting changes applied across 7 specs after closing all file format decisions. All tabular input data now uses Parquet. Correlation schedule embedded in `correlation.json`. Exchange factors moved from `scenarios/` to `constraints/`.

| Spec File                      | Change Type | Description                                                                                                                                                                                                                                                                                                                                                                 | Status  |
| ------------------------------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `input-directory-structure.md` | restructure | Closed all format TBDs. All tabular files use `.parquet`. Split `inflow_models` → 2 files. Renamed `load_models` → `load_seasonal_stats`. Moved `exchange_factors` to `constraints/`. Split penalty overrides into 4 files. Restructured `config.json` examples. Status: `deferred` → `approved`.                                                                           | applied |
| `binary-formats.md`            | restructure | §1 framework table: closed "Entity-level tabular data" → Parquet, "Default-with-overrides" → JSON + Parquet. §2 summary: closed "Scenario Pipeline" → JSON + Parquet, "Stage Overrides" → Parquet.                                                                                                                                                                          | applied |
| `input-hydro-extensions.md`    | restructure | Added `.parquet` to `hydro_geometry` (§1) and `fpha_hyperplanes` (§3) headers. Updated format rationale text.                                                                                                                                                                                                                                                               | applied |
| `penalty-system.md`            | restructure | Closed format TBD → Parquet in §1. Removed open question block. Split penalty overrides into 4 named files: `penalty_overrides_bus.parquet`, `penalty_overrides_line.parquet`, `penalty_overrides_hydro.parquet`, `penalty_overrides_ncs.parquet`.                                                                                                                          | applied |
| `input-constraints.md`         | restructure | Closed format question → Parquet. Added `.parquet` to all 6 bounds file headers. Closed `generic_constraint_bounds` format. Added `exchange_factors.json` section (moved from `input-scenarios.md`). Updated pre-study inflow history reference.                                                                                                                            | applied |
| `input-scenarios.md`           | restructure | Split `inflow_models` → `inflow_seasonal_stats.parquet` (with `ar_order` column) + `inflow_ar_coefficients.parquet` (long-form). Renamed `load_models` → `load_seasonal_stats.parquet`. Removed §5 Exchange Factors (moved to `input-constraints.md`). Renumbered §6→§5 Correlation with embedded schedule in `correlation.json`. Renumbered §7→§6. Closed all format TBDs. | applied |
| `internal-structures.md`       | restructure | §14 inflow model source references updated to split files. "Load Models" → "Load Seasonal Statistics". Correlation cross-reference updated §6→§5.                                                                                                                                                                                                                           | applied |

### Cross-Spec Changes

| Spec File                   | Change Type | Description                                                                                               | Status  |
| --------------------------- | ----------- | --------------------------------------------------------------------------------------------------------- | ------- |
| `penalty-system.md`         | add field   | Added `$schema` placeholder to penalties.json example for consistency                                     | applied |
| `input-system-entities.md`  | add field   | Added `$schema` placeholders to buses, lines, hydros, thermals, pumping_stations, contracts JSON examples | applied |
| `input-hydro-extensions.md` | add field   | Added `$schema` placeholders to both production_models JSON examples                                      | applied |

## Internal Structures

Changes to in-memory representations, binary formats, and serialization.

| Spec File                | Change Type | Description                                                                                                                                                           | Status  |
| ------------------------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `internal-structures.md` | restructure | Complete rewrite: removed all Rust `struct` code blocks, replaced with logical in-memory data model descriptions                                                      | applied |
| `internal-structures.md` | restructure | Fixed priority from 3-medium to 1-critical (this spec defines what the solver holds at runtime — foundational)                                                        | applied |
| `internal-structures.md` | restructure | §1 System Representation: replaced Rust structs with entity collection table and system metadata description                                                          | applied |
| `internal-structures.md` | restructure | §2 Operative State: rewritten from Rust enum to behavioral table (non-existing, filling, operating, decommissioned) with LP treatment per state                       | applied |
| `internal-structures.md` | restructure | §3 Hydro Plant: restructured into sub-sections (core identity, reservoir, outflow bounds, generation model, optional data, diversion, filling)                        | applied |
| `internal-structures.md` | restructure | §3 Hydro: generation model documented as tagged union with 3 variants (constant_productivity, linearized_head, fpha) — aligned with input-system-entities.md          | applied |
| `internal-structures.md` | restructure | §3 Hydro: added production model selection (stage_ranges, seasonal) referencing input-hydro-extensions.md                                                             | applied |
| `internal-structures.md` | restructure | §3 Hydro: added optional data table (tailrace, hydraulic losses, efficiency, evaporation, geometry, FPHA hyperplanes)                                                 | applied |
| `internal-structures.md` | restructure | §3 Hydro: added filling configuration (CEPEL-based) with terminal constraint and filling_inflow_m3s entity default                                                    | applied |
| `internal-structures.md` | restructure | §3 Hydro: soft lower bound on storage (storage_violation_below slack), hard upper bound (emergency spill)                                                             | applied |
| `internal-structures.md` | add field   | §7 Pumping Station: new section — independent system element with consumption rate, flow bounds, no explicit cost                                                     | applied |
| `internal-structures.md` | add field   | §8 Energy Contract: new section — import/export with price, limits, bus assignment                                                                                    | applied |
| `internal-structures.md` | restructure | §9 Penalty Tables: rewritten with three-category system (recourse, constraint violation, regularization), full penalty type tables per entity, priority ordering      | applied |
| `internal-structures.md` | restructure | §10 Pre-Resolved Bounds: rewritten with complete hydro bounds table including slack associations and filling_inflow_m3s                                               | applied |
| `internal-structures.md` | restructure | §11 Stage/Block: rewritten to include season_id, block_mode per stage, policy graph (finite_horizon/cyclic), annual_discount_rate, season definitions with cycle_type | applied |
| `internal-structures.md` | restructure | §12 Scenario Pipeline: new section documenting inflow models (mu, sigma, AR coefficients), load models, correlation model with Cholesky, block factors                | applied |
| `internal-structures.md` | restructure | §13 Generic Constraints: rewritten with full variable reference catalog (17 variables across 6 entity types), constraint bounds, validation rules                     | applied |
| `internal-structures.md` | restructure | §14 Initial Conditions: updated with split operating/filling storage, GNL deferred                                                                                    | applied |

### Internal Structures — Second Review Round (2026-02-16)

Changes applied based on user feedback during second review of `internal-structures.md`.

| Spec File                | Change Type | Description                                                                                                                                             | Status  |
| ------------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `internal-structures.md` | add field   | Added "Role in Program Lifecycle" section: LP definition construction (built once, clarity > performance) vs scenario pipeline (runtime, perf-critical) | applied |
| `internal-structures.md` | add field   | §9 Non-Controllable Source: new section — entity with bus, lifecycle, max_generation_mw, availability, generation variable, curtailment penalty         | applied |
| `internal-structures.md` | add field   | §3 Hydro: added FPHA turbined flow penalty description — prevents interior FPHA solutions, must be > spillage_cost                                      | applied |
| `internal-structures.md` | add field   | §10 Penalty Tables: added `fpha_turbined_cost` (regularization, FPHA-only) and `curtailment_cost` (regularization, non-controllable sources)            | applied |
| `internal-structures.md` | restructure | §4 Thermal: fleshed out GNL dispatch anticipation data model (lag_stages, state variables, LP coupling, state dimension impact)                         | applied |
| `internal-structures.md` | add field   | §4 Thermal: added exceptional validation rule — GNL thermals rejected until implementation                                                              | applied |
| `internal-structures.md` | restructure | §13 renamed to Block Factors (separated from Scenario Pipeline): load factors + exchange factors loaded during initialization, not runtime pipeline     | applied |
| `internal-structures.md` | restructure | §14 renamed to Scenario Pipeline: inflow models, load models, correlation model with Cholesky                                                           | applied |
| `internal-structures.md` | add field   | §15 Generic Constraints: added `non_controllable_generation` and `non_controllable_curtailment` to variable reference catalog                           | applied |
| `internal-structures.md` | add field   | §16 Initial Conditions: added GNL pipeline initial conditions (thermal_id, stage_offset, committed_mw tuples)                                           | applied |

### Cross-Spec Changes from Internal Structures Review (2026-02-16)

| Spec File                  | Change Type | Description                                                                                                                                                               | Status  |
| -------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `penalty-system.md`        | add field   | Added `fpha_turbined_cost` to Category 3 regularization table, penalties.json, hydro overrides table, and objective function                                              | applied |
| `penalty-system.md`        | add field   | Added `curtailment_cost` to Category 3 regularization table, penalties.json, and objective function                                                                       | applied |
| `penalty-system.md`        | add field   | Added non-controllable source penalty overrides section and non-controllable sources constraint violation section                                                         | applied |
| `penalty-system.md`        | add field   | Added `non_controllable_source` section to penalties.json example                                                                                                         | applied |
| `penalty-system.md`        | add field   | Added FPHA validation rule: `fpha_turbined_cost > spillage_cost` per plant                                                                                                | applied |
| `penalty-system.md`        | restructure | Marked as `needs-re-review`                                                                                                                                               | applied |
| `input-system-entities.md` | restructure | §7: replaced deferred stub with full non-controllable sources definition (concept, operative states, JSON example, field table, generation/curtailment, no stage-varying) | applied |
| `input-system-entities.md` | add field   | §4: added GNL exceptional validation rule (reject GNL thermals until implementation)                                                                                      | applied |
| `input-system-entities.md` | restructure | Marked as `needs-re-review`                                                                                                                                               | applied |

### Serialization and Persistence Formats Review (2026-02-16)

| Spec File           | Change Type  | Description                                                                                                                                                                    | Status  |
| ------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `binary-formats.md` | restructure  | Renamed from "Binary Formats and Internal Structures" to "Serialization and Persistence Formats" — title now reflects actual scope                                             | applied |
| `binary-formats.md` | restructure  | Changed priority from 2-high to 1-critical (defines policy data persistence, core to SDDP checkpoint/resume)                                                                   | applied |
| `binary-formats.md` | remove field | Removed §3.4 Rust memory layout code (`BendersCutData`, `CutCoefficients` structs) — replaced with prose description of memory layout considerations                           | applied |
| `binary-formats.md` | remove field | Removed §4 LP Subproblem Structure entirely (`BlockSubproblem`, `VariableLayout` Rust structs) — covered by `internal-structures.md`                                           | applied |
| `binary-formats.md` | remove field | Removed §5 FCF Rust structs (`FutureCostFunction`, `CutPool`, `BendersCut`) — replaced with logical "Cut Pool Persistence" section focusing on serialization concerns only     | applied |
| `binary-formats.md` | remove field | Removed §6 Parquet config Rust struct (`ParquetConfig`) — replaced with prose requirements for Parquet output configuration                                                    | applied |
| `binary-formats.md` | restructure  | §1 Format Framework: removed stale filenames (`inflow_models.parquet`, `hydro_geometry.parquet`), fixed "Time series" label to "Entity-level tabular data", format TBD         | applied |
| `binary-formats.md` | restructure  | §1 Format Framework: changed "JSON + Parquet" to "JSON + TBD" for default-with-overrides (stage override format TBD)                                                           | applied |
| `binary-formats.md` | restructure  | §2 Format Summary: removed vague "Uncertainty Models", "Distributions", "Load Profiles", "Inflow History" entries. Added "Scenario Pipeline", "Correlation", "Stage Overrides" | applied |
| `binary-formats.md` | restructure  | §3.1 FlatBuffers schema: replaced Unicode math symbols with ASCII equivalents in comments                                                                                      | applied |
| `binary-formats.md` | restructure  | §3 "maps directly to Rust structs" → "maps directly to in-memory representation"                                                                                               | applied |
| `binary-formats.md` | restructure  | Updated Purpose section and cross-references: removed "Epic 4" reference, fixed `internal-structures.md` description                                                           | applied |

## P2 Math Specs

Changes applied during P2 math spec reviews (2026-02-19 onwards).

### system-elements.md (approved 2026-02-19)

| Change Type | Description                                                                                                                                     | Status  |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| restructure | §1 Buses: generalized from regional subsystems to general network concept, deficit sum fixed to match penalty-system.md                         | applied |
| restructure | §3 Transmission Lines: added exchange factor note, clarified two-variable decomposition rationale                                               | applied |
| restructure | §4 Thermals: documented LP relaxation of commitment constraints, added GNL subsection with state variables                                      | applied |
| restructure | §5 Hydros: added linearized_head as third production model, filling model section, FPHA turbined cost, diversion max flow, AR lag clarification | applied |
| restructure | §5 Hydros: evaporation/withdrawal clarification, pumping min flow bound added                                                                   | applied |
| restructure | §6 Non-Controllable Sources: fully defined (was DEFERRED) — generation, curtailment, penalty category                                           | applied |
| restructure | §7 Contracts: rewritten as typed unidirectional with single variable per contract                                                               | applied |
| restructure | §8 Slack Variables: expanded table with all approved penalty names from penalty-system.md                                                       | applied |
| add field   | Added Variable Units Convention section documenting rate-units decision (MW, m³/s)                                                              | applied |
| restructure | Updated summary table with all element types and their LP roles                                                                                 | applied |

### lp-formulation.md (approved 2026-02-19)

| Change Type  | Description                                                                                                                                                                                        | Status  |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| restructure  | §1 Penalty System: rewrote from 5-category to 3-category taxonomy aligned with approved penalty-system.md                                                                                          | applied |
| restructure  | §1 Penalty System: updated all penalty symbols to approved names (c^{sv-}, c^{tv-}, c^{ov±}, c^{gv-}, c^{ev}, c^{wv}, c^{fill}, c^{fpha}, c^{curt})                                                | applied |
| add field    | §1 Penalty System: added filling_target and storage_violation to priority ordering above deficit                                                                                                   | applied |
| remove field | §1 Penalty System: removed stale thermal_bounds.parquet, DATA_MODEL_SPECIFICATION, target_storage_hm3 references                                                                                   | applied |
| restructure  | §2 Objective: rewrote with unidirectional contracts (Σ*c c^{ctr}\_c χ*{c,k}), added FPHA turbined cost, curtailment cost, storage violations outside τ_k sum                                       | applied |
| restructure  | §3 Constraints: added NCS generation term (g^{nc}\_{r,k}), fixed contracts to unidirectional, added dual units note                                                                                | applied |
| add field    | §4: added Variable Units Convention cross-reference                                                                                                                                                | applied |
| restructure  | §6 Hydro Production: added linearized_head as third production model, added generation upper bound (hard)                                                                                          | applied |
| restructure  | §8 Variable Bounds: renamed to "Variable Bounds and Minimum Constraints" — added storage bounds (soft lower/hard upper + filling terminal), turbined flow bounds, diversion bounds, pumping bounds | applied |
| restructure  | §9 Penalties: rewritten with approved penalty symbols and cascade resolution reference                                                                                                             | applied |
| add field    | Cross-references: added penalty-system.md, input-system-entities.md, hydro-production-models.md                                                                                                    | applied |

### Linearized Head Simulation-Only Reclassification (2026-02-20)

**Architectural Decision**: The linearized head production model is reclassified as **simulation-only** — available during policy evaluation but excluded from training (policy construction). Only `constant_productivity` and `fpha` are valid during training.

**Rationale**: The linearized head model uses a bilinear term ($q \times v^{avg}$). To maintain LP linearity, $v^{avg}$ must be fixed from the previous iteration, but this changes the LP constraint coefficients between iterations. SDDP requires a fixed LP structure per stage for convergence guarantees — Benders cuts generated under one linearization point are not guaranteed valid under a different one. During simulation (single forward pass, no cuts), this is safe.

| Spec File                    | Change Type | Description                                                                                                                                           | Status  |
| ---------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `hydro-production-models.md` | restructure | §3 rewritten as "Linearized Head Model (Simulation-Only Enhancement)" with §3.1 Why Simulation-Only, §3.2 Simulation Use Case, §3.3 Data Requirements | applied |
| `hydro-production-models.md` | restructure | §4 split into Training and Simulation subsections; linearized head removed from training recommendations                                              | applied |
| `hydro-production-models.md` | restructure | Purpose paragraph updated: "Two models during training, third during simulation only"                                                                 | applied |
| `hydro-production-models.md` | restructure | §2.9 FPHA turbined cost: clarified linearized_head exclusion with simulation-only reference                                                           | applied |
| `system-elements.md`         | restructure | §5 Production Function: linearized_head annotated as "(simulation-only)", added training-models-only blockquote                                       | applied |
| `lp-formulation.md`          | restructure | §6: removed linearized_head from training LP constraints, updated section intro to reference two training models                                      | applied |
| `input-system-entities.md`   | restructure | §3: linearized_head variant header annotated as "(simulation-only — excluded from training)"                                                          | applied |
| `input-hydro-extensions.md`  | restructure | §2: model hierarchy annotated, JSON seasonal example note added, stage range and season field tables updated with "(simulation-only)"                 | applied |
| `internal-structures.md`     | restructure | §3: Generation Model table expanded with Phase column (Training + Simulation / Simulation-only), added explanatory blockquote                         | applied |

## Output Formats

Changes to output schemas, Parquet layouts, and output infrastructure.

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |

## Penalty System

Changes to penalty types, cascade logic, and override resolution.

| Spec File           | Change Type  | Description                                                                                                          | Status  |
| ------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------- | ------- |
| `penalty-system.md` | restructure  | Reclassified penalties from 2 categories to 3: recourse slacks, constraint violation penalties, regularization costs | applied |
| `penalty-system.md` | add field    | Added `generic_violation_cost` to global defaults                                                                    | applied |
| `penalty-system.md` | add field    | Added `water_withdrawal_violation_cost`, `evaporation_violation_cost` to constraint violations                       | applied |
| `penalty-system.md` | restructure  | Added explicit sections for lines/thermals/storage (hard bounds, no slacks)                                          | applied |
| `penalty-system.md` | add field    | Added line penalty overrides to stage-varying overrides schema                                                       | applied |
| `penalty-system.md` | restructure  | Fixed generation bounds: user-defined, not derived from turbined flow                                                | applied |
| `penalty-system.md` | restructure  | Fixed exchange model: uses direct_flow + reverse_flow, not absolute value                                            | applied |
| `penalty-system.md` | remove field | Removed `pumping_cost` from objective (cost is implicit via bus energy consumption)                                  | applied |
| `penalty-system.md` | restructure  | Reorganized LP objective by 3 penalty categories                                                                     | applied |
| `penalty-system.md` | restructure  | Reset version from 1.1 to 1.0                                                                                        | applied |
| `penalty-system.md` | restructure  | Decoupled logical schema from physical file format for stage overrides (format TBD)                                  | applied |

### Filling Model Redesign (discovered during input-constraints.md review)

Based on CEPEL dead-volume filling documentation (`enchimento de volume morto`). These changes span multiple already-approved specs, which were marked as `needs-re-review`.

| Spec File                  | Change Type  | Description                                                                                                                                   | Status   |
| -------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `input-system-entities.md` | remove field | Removed `filling.target_storage_hm3` — filling always targets `min_storage_hm3`                                                               | applied  |
| `input-system-entities.md` | add field    | Added `filling.bottom_discharge_m3s` — maximum outflow through non-spillway outlets during filling (default 0)                                | reverted |
| `input-system-entities.md` | restructure  | Added Filling Model section with timeline, target description, bottom discharge semantics, initial conditions, and validation                 | applied  |
| `input-constraints.md`     | restructure  | Split initial conditions: `storage` array for operating hydros, separate `filling_storage` array for filling hydros (can be below V_min)      | applied  |
| `input-constraints.md`     | restructure  | Updated validation rules: mutual exclusion, filling storage bounds `[0, min_storage_hm3]`, operating hydro coverage                           | applied  |
| `input-constraints.md`     | restructure  | Improved `filling_inflow_m3s` description: minimum retention during `[start_stage_id, entry_stage_id)`, goes directly to storage              | applied  |
| `input-constraints.md`     | add field    | Added filling inflow sufficiency validation warning (deterministic lower-bound check, warning not error)                                      | applied  |
| `input-constraints.md`     | restructure  | Fixed GNL stale reference (line 59 referenced a `gnl_pipeline` field no longer in JSON example)                                               | applied  |
| `penalty-system.md`        | add field    | Added `storage_violation_below` slack with `storage_violation_below_cost` — storage lower bound changed from hard to soft                     | applied  |
| `penalty-system.md`        | add field    | Added `filling_target_violation` slack with `filling_target_violation_cost` — terminal filling constraint at `entry_stage_id - 1`             | applied  |
| `penalty-system.md`        | restructure  | Updated penalty priority ordering: filling_target > storage_violation > deficit > constraint violations > resource costs > regularization     | applied  |
| `penalty-system.md`        | restructure  | Rewritten Hydro Storage Bounds section: min storage now soft (slack), max still hard (emergency spill), terminal filling constraint described | applied  |
| `penalty-system.md`        | restructure  | Rewritten Dead-Volume Filling Specifics: bottom discharge, relaxed storage bounds, terminal constraint, filling-to-operating transition       | applied  |
| `penalty-system.md`        | restructure  | Updated penalties.json example, constraint violation table, variables summary, and objective function with new slack variables                | applied  |
| `penalty-system.md`        | restructure  | Added `storage_violation_below_cost` and `filling_target_violation_cost` to hydro penalty overrides table                                     | applied  |

### Bottom Discharge Deferral (2026-02-16)

Bottom discharge (`descargas de fundo`) creates a conditional constraint — outflow capacity depends on whether the reservoir level is above/below the spillway crest — which requires nonlinear or binary constraints incompatible with LP-based SDDP. Deferred to simulation step only.

| Spec File                  | Change Type  | Description                                                                                                                | Status  |
| -------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- | ------- |
| `input-system-entities.md` | remove field | Removed `filling.bottom_discharge_m3s` from JSON example, field table, and footnote. Added deferred note in Filling Model. | applied |
| `penalty-system.md`        | restructure  | Removed bottom discharge references from Dead-Volume Filling Specifics. Outflow during filling now via spillage only.      | applied |
| `input-constraints.md`     | restructure  | Removed `bottom_discharge_m3s` losses from filling inflow sufficiency warning computation.                                 | applied |

### Filling Inflow Entity Default (2026-02-16)

Added `filling_inflow_m3s` to the filling config in the hydro object as an entity-level default. Per-stage overrides in `hydro_bounds` replace it for specific stages. Follows the entity default → stage override cascade pattern.

| Spec File                  | Change Type | Description                                                                                                                                                  | Status  |
| -------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `input-system-entities.md` | add field   | Added `filling.filling_inflow_m3s` — entity-level default filling inflow (m³/s), optional, default 0.0. Updated JSON example, field table, filling behavior. | applied |
| `input-constraints.md`     | restructure | Updated `filling_inflow_m3s` column description and filling inflow paragraph to reference entity default with stage override cascade.                        | applied |
