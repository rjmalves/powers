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

| Observation                                                                                                                                                                                                                                                                                                                                                           | Affected Specs                                        | Status |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------ |
| **Diagrams will be revised after text**: All diagrams (SVG, Mermaid, etc.) referenced across specs will be reviewed and updated only after the text part of the documentation review is complete. Diagram accuracy depends on finalized text content.                                                                                                                 | All specs referencing diagrams                        | noted  |
| **`block_mode` moved from global config to per-stage**: `input-scenarios.md` now defines `block_mode` per stage. `block-formulations.md` and `configuration-reference.md` still reference `modeling.block_mode` as a global setting — will be updated when those specs are reviewed.                                                                                  | `block-formulations.md`, `configuration-reference.md` | noted  |
| **`discount_rate` moved from per-transition to `policy_graph`**: `input-scenarios.md` now defines discount rate as `annual_discount_rate` in `policy_graph` with per-transition override. `discount-rate.md` §14.3 still shows per-transition `discount_rate` as a plain rate — will be updated during P2 review.                                                     | `discount-rate.md`                                    | noted  |
| **`$schema` placeholders**: All JSON examples in approved data model specs now include `$schema` placeholder fields for future JSON Schema validation. Apply to remaining specs as they are reviewed.                                                                                                                                                                 | All data model specs with JSON examples               | noted  |
| **Filling model impact on math specs**: The filling model redesign (storage slack, terminal constraint, bottom discharge) requires updates to `lp-formulation.md` (line 110 references `target_storage_hm3` as constraint target) and `system-elements.md` (line 195-197 filling hydro subset). These are P2 math specs and will be reviewed at their scheduled time. | `lp-formulation.md`, `system-elements.md`             | noted  |

---

## Specs Pending Re-Review

Previously approved specs that received changes during the review of other specs. These need re-review to confirm the cross-cutting changes are correct.

| Spec File                  | Original Approval | Changed During                | Changes Applied                                                                                                                                                                                                                 | Status          |
| -------------------------- | ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `input-system-entities.md` | 2026-02-15        | `input-constraints.md` review | Filling config: removed `target_storage_hm3`, added `bottom_discharge_m3s`, added Filling Model section (timeline, target, bottom discharge, initial conditions, validation)                                                    | needs-re-review |
| `penalty-system.md`        | 2026-02-14        | `input-constraints.md` review | Storage lower bound: hard → soft (`storage_violation_below` slack). New `filling_target_violation` slack. Updated penalty ordering, penalties.json, constraint violation table, variables summary, objective, filling specifics | needs-re-review |

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

### Cross-Spec Changes

| Spec File                   | Change Type | Description                                                                                               | Status  |
| --------------------------- | ----------- | --------------------------------------------------------------------------------------------------------- | ------- |
| `penalty-system.md`         | add field   | Added `$schema` placeholder to penalties.json example for consistency                                     | applied |
| `input-system-entities.md`  | add field   | Added `$schema` placeholders to buses, lines, hydros, thermals, pumping_stations, contracts JSON examples | applied |
| `input-hydro-extensions.md` | add field   | Added `$schema` placeholders to both production_models JSON examples                                      | applied |

## Internal Structures

Changes to in-memory representations, binary formats, and serialization.

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |

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

| Spec File                  | Change Type  | Description                                                                                                                                   | Status  |
| -------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `input-system-entities.md` | remove field | Removed `filling.target_storage_hm3` — filling always targets `min_storage_hm3`                                                               | applied |
| `input-system-entities.md` | add field    | Added `filling.bottom_discharge_m3s` — maximum outflow through non-spillway outlets during filling (default 0)                                | applied |
| `input-system-entities.md` | restructure  | Added Filling Model section with timeline, target description, bottom discharge semantics, initial conditions, and validation                 | applied |
| `input-constraints.md`     | restructure  | Split initial conditions: `storage` array for operating hydros, separate `filling_storage` array for filling hydros (can be below V_min)      | applied |
| `input-constraints.md`     | restructure  | Updated validation rules: mutual exclusion, filling storage bounds `[0, min_storage_hm3]`, operating hydro coverage                           | applied |
| `input-constraints.md`     | restructure  | Improved `filling_inflow_m3s` description: minimum retention during `[start_stage_id, entry_stage_id)`, goes directly to storage              | applied |
| `input-constraints.md`     | add field    | Added filling inflow sufficiency validation warning (deterministic lower-bound check, warning not error)                                      | applied |
| `input-constraints.md`     | restructure  | Fixed GNL stale reference (line 59 referenced a `gnl_pipeline` field no longer in JSON example)                                               | applied |
| `penalty-system.md`        | add field    | Added `storage_violation_below` slack with `storage_violation_below_cost` — storage lower bound changed from hard to soft                     | applied |
| `penalty-system.md`        | add field    | Added `filling_target_violation` slack with `filling_target_violation_cost` — terminal filling constraint at `entry_stage_id - 1`             | applied |
| `penalty-system.md`        | restructure  | Updated penalty priority ordering: filling_target > storage_violation > deficit > constraint violations > resource costs > regularization     | applied |
| `penalty-system.md`        | restructure  | Rewritten Hydro Storage Bounds section: min storage now soft (slack), max still hard (emergency spill), terminal filling constraint described | applied |
| `penalty-system.md`        | restructure  | Rewritten Dead-Volume Filling Specifics: bottom discharge, relaxed storage bounds, terminal constraint, filling-to-operating transition       | applied |
| `penalty-system.md`        | restructure  | Updated penalties.json example, constraint violation table, variables summary, and objective function with new slack variables                | applied |
| `penalty-system.md`        | restructure  | Added `storage_violation_below_cost` and `filling_target_violation_cost` to hydro penalty overrides table                                     | applied |
