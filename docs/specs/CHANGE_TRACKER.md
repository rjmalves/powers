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

## Input Schemas

Changes to input file formats, directory structure, and entity definitions.

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |

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
