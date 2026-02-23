---
status: approved
review_priority: 1-critical
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.1 (Core Algorithm Structures)"
last_reviewed: 2026-02-17
reviewed_by: rogerio
review_notes: "Approved after complete rewrite and two review rounds."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §5.1"
  - date: 2026-02-16
    description: "Complete rewrite: removed all Rust struct code, replaced with logical in-memory data model descriptions. Aligned with approved specs (penalty-system, input-system-entities, input-hydro-extensions, input-scenarios, input-constraints). Fixed review_priority from 3-medium to 1-critical. Added missing entity types (pumping stations, energy contracts). Updated penalty model to match three-category system with all slacks. Updated hydro model with tagged union generation, filling model (CEPEL-based), tailrace/losses/efficiency/evaporation. Updated stage model with season_id, block_mode, policy graph, discount rate. Updated generic constraints with full variable reference catalog. Added scenario pipeline section."
  - date: 2026-02-16
    description: "Review feedback: added lifecycle role section (LP definition vs scenario pipeline performance). Defined non-controllable sources (§9) with curtailment penalty. Added FPHA turbined flow penalty (§3 + §10). Fleshed out GNL dispatch anticipation data model in §4 and initial conditions §16 (validation rejects GNL thermals for now). Separated block factors from scenario pipeline into §13. Added non-controllable source variables to generic constraints catalog §15."
  - date: 2026-02-17
    description: "Format propagation: updated §14 inflow model source references to split files (inflow_seasonal_stats.parquet + inflow_ar_coefficients.parquet). Renamed Load Models to Load Seasonal Statistics. Updated correlation cross-reference from §6 to §5 (renumbered in input-scenarios.md)."
  - date: 2026-02-20
    description: "Post-approval amendment: §3 Generation Model table — added Phase column, linearized_head marked simulation-only with explanatory note."
---

# Internal Structures

## Purpose

This spec defines the logical in-memory data model that the SDDP solver requires at runtime. It describes _what_ data the solver must hold in memory and _why_, without prescribing implementation types or data structures. The actual implementation (struct layout, indexing strategy, memory layout) will emerge during development.

These structures are the result of loading, validating, and resolving all input files into a unified, ready-to-use representation. They differ from input schemas in several ways:

| Concern    | Input Schemas                                | Internal Structures                      |
| ---------- | -------------------------------------------- | ---------------------------------------- |
| Scope      | One file at a time                           | Unified, cross-referenced                |
| Defaults   | May be absent (use global/entity default)    | All defaults resolved to concrete values |
| Penalties  | Three-tier cascade (global → entity → stage) | Pre-resolved per (entity, stage)         |
| Bounds     | Base values with sparse stage overrides      | Pre-resolved per (entity, stage)         |
| Validation | Deferred or partial                          | Complete — all cross-references verified |
| Ordering   | User-defined (arbitrary)                     | Canonical (sorted by ID)                 |

For input file schemas, see [Input System Entities](input-system-entities.md), [Input Hydro Extensions](input-hydro-extensions.md), [Input Scenarios](input-scenarios.md), [Input Constraints](input-constraints.md). For binary format choices (LP subproblems, cuts), see [Binary Formats](binary-formats.md).

### Role in Program Lifecycle

These internal structures serve two distinct purposes in the program, each with different performance characteristics:

**1. LP Definition Construction (initialization — built once)**

The system entities (§1–§9), pre-resolved penalties (§10), pre-resolved bounds (§11), generic constraints (§14), block factors (§12), and AR models (§13) are all loaded and resolved during program initialization — whether for training or simulation. From these resolved structures plus algorithm configuration (number of iterations, forward passes for cut preallocation, etc.), the program constructs a set of LP definitions, one per (stage, block) pair. These LP definitions are built once and reused throughout the algorithm. Since this happens once at startup, **data clarity and correctness are more important than performance** in this phase.

**2. Scenario Generation Pipeline (runtime — performance-critical)**

The scenario pipeline (§13) is exercised repeatedly during both training and simulation. Each forward pass requires sampling from standard normal distributions, applying Cholesky-decomposed correlation matrices, and transforming the results into correlated scenario realizations. This is parallelized across MPI ranks with seed preprocessing to ensure reproducibility. **Performance is critical** in this phase — the noise generation step (sampling, correlation transforms, and method-specific operations) must be efficient at scale.

> **Opening tree lifecycle**: The noise openings used in the backward pass are generated **once before training begins** (fixed opening tree), not per-iteration. The forward pass may use the same opening tree or an alternative noise source (e.g., external scenarios). See [Scenario Generation §2.3 and §3](../03-architecture/scenario-generation.md).

## 1. System Representation

The top-level system object holds all entity collections needed by the solver. After loading from input files, all collections are sorted by entity ID (canonical ordering) to ensure deterministic behavior regardless of declaration order in input files. See [Design Principles §3](../00-overview/design-principles.md).

### Entity Collections

| Collection               | Source                                 | Description                                                     |
| ------------------------ | -------------------------------------- | --------------------------------------------------------------- |
| Buses                    | `system/buses.json`                    | Electrical network nodes with pre-resolved deficit segments     |
| Lines                    | `system/lines.json`                    | Transmission interconnections between buses                     |
| Hydros                   | `system/hydros.json` + extensions      | Hydro plants with generation model, reservoir, cascade topology |
| Thermals                 | `system/thermals.json`                 | Thermal plants with piecewise cost curves                       |
| Pumping stations         | `system/pumping_stations.json`         | Water transfer stations consuming power                         |
| Energy contracts         | `system/energy_contracts.json`         | Import/export agreements with external systems                  |
| Non-controllable sources | `system/non_controllable_sources.json` | Intermittent generation (wind/solar) with curtailment option    |

### System Metadata

The system object should also carry metadata needed for cross-cutting concerns:

- Total entity counts per type (for LP dimensioning)
- Cascade topology (downstream/upstream adjacency, resolved to skip non-existing plants)
- Global configuration references

## 2. Operative State

Every entity with lifecycle attributes (`entry_stage_id`, `exit_stage_id`) has a stage-dependent operative state that determines which LP variables and constraints are active. The operative state is computed per entity per stage.

| State          | Applies To       | LP Treatment                                                                                                                |
| -------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Non-existing   | All entity types | No LP variables or constraints                                                                                              |
| Filling        | Hydros only      | Storage, spillage, evaporation, violation slacks. No generation (turbined = 0). See [Penalty System §7](penalty-system.md). |
| Operating      | All entity types | Full LP variables and constraints per entity type                                                                           |
| Decommissioned | All entity types | See open question in [Input System Entities §3](input-system-entities.md)                                                   |

**Hydro filling timeline**: `[start_stage_id, entry_stage_id)` = filling period. At `entry_stage_id`, transitions to operating. Based on CEPEL's dead-volume filling model (`enchimento de volume morto`).

**Cascade redirection**: When a downstream hydro is non-existing, outflows are automatically redirected to the next operating downstream in the cascade. The internal representation should resolve this redirection so the LP builder does not need to walk the cascade at build time.

## 3. Hydro Plant

Hydro plants are the most complex entity in the system. The in-memory hydro representation must include:

### Core Identity and Topology

- Entity ID, name, and bus assignment
- Downstream hydro ID (physical cascade), upstream hydro IDs (derived from downstream references)
- Lifecycle: `entry_stage_id`, `exit_stage_id`

### Reservoir

- `min_storage_hm3` (dead volume) — soft lower bound, violation uses `storage_violation_below` slack
- `max_storage_hm3` — hard upper bound (emergency spillage handles excess)

### Outflow Bounds

- `min_outflow_m3s` — soft lower bound (environmental flow), violation slack available
- `max_outflow_m3s` — soft upper bound (flood control), violation slack available. Null = no upper bound.

### Generation Model (Tagged Union)

The generation model is a tagged union selected by a `model` field. Only fields relevant to the selected variant are meaningful. See [Input System Entities §3](input-system-entities.md) for input schema and [Hydro Production Functions](../01-math/hydro-production-models.md) for math.

| Variant                 | Key Data                                                            | Requires Geometry              | Phase                 |
| ----------------------- | ------------------------------------------------------------------- | ------------------------------ | --------------------- |
| `constant_productivity` | Productivity factor [MW/(m3/s)]                                     | No                             | Training + Simulation |
| `linearized_head`       | Base productivity, adjusted by storage-dependent head               | Yes                            | **Simulation-only**   |
| `fpha`                  | Hyperplane coefficients (gamma_0, gamma_v, gamma_q, gamma_s, kappa) | Computed: yes; Precomputed: no | Training + Simulation |

> **Simulation-only restriction**: The `linearized_head` model is excluded from training because the bilinear term ($q \times v^{avg}$) requires re-fixing $v^{avg}$ between iterations, changing the LP structure and breaking SDDP convergence guarantees. See [Hydro Production Models §3](../01-math/hydro-production-models.md).

All variants carry:

- `min_turbined_m3s`, `max_turbined_m3s` — machine flow limits (lower bound has violation slack)
- `min_generation_mw`, `max_generation_mw` — user-defined generation bounds (lower bound has violation slack, upper is hard)

**FPHA turbined flow penalty**: When using the `fpha` production model, the concave geometry of the hyperplane approximation means that every point inside the region bounded by the FPHA hyperplanes is a feasible LP solution — including solutions where the generation is less than what the physical production function would yield for the given turbined flow. Without correction, the solver can effectively "spill through the turbines" by choosing these interior points. To prevent this, a `fpha_turbined_cost` regularization penalty is applied to the turbined flow variable, and it must be strictly greater than the `spillage_cost` for the same plant. This ensures the solver prefers to turbine (and generate) rather than spill, and when it does turbine, the FPHA constraints push generation toward the physical envelope. This penalty is only active for hydros using the `fpha` production model.

### Production Model Selection

The active generation model can vary by stage or season. See [Input Hydro Extensions §2](input-hydro-extensions.md) for the selection modes (`stage_ranges`, `seasonal` with `default_model` fallback).

### Optional Hydro Data

| Data             | Purpose                                                            | Source             |
| ---------------- | ------------------------------------------------------------------ | ------------------ |
| Tailrace model   | Downstream water level vs. outflow (polynomial or piecewise)       | `hydros.json`      |
| Hydraulic losses | Head losses (factor or constant)                                   | `hydros.json`      |
| Efficiency       | Turbine-generator efficiency (constant; future: flow-dependent)    | `hydros.json`      |
| Evaporation      | 12 monthly coefficients (mm)                                       | `hydros.json`      |
| Geometry         | Volume-Height-Area table for forebay/area computation              | `hydro_geometry`   |
| FPHA hyperplanes | Pre-computed production function planes (optionally stage-varying) | `fpha_hyperplanes` |

### Diversion Channel (Optional)

- `downstream_id` — hydro receiving diverted water
- `max_flow_m3s` — maximum diversion flow
- Diversion variable bounded by `[0, max_flow_m3s]`; only regularization cost (no violation slacks)

### Filling Configuration (Optional)

Based on CEPEL's dead-volume filling model. See [Input System Entities §3](input-system-entities.md) and [Penalty System §7](penalty-system.md).

- `start_stage_id` — first filling stage
- `filling_inflow_m3s` — entity-level default for filling inflow (overridable per stage in `hydro_bounds`)
- Filling target is always `min_storage_hm3`
- Terminal constraint at `entry_stage_id - 1` with `filling_target_violation` slack (highest penalty in system)

## 4. Thermal Plant

In-memory thermal representation:

- Entity ID, name, bus assignment
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- **Cost segments**: Piecewise-linear cost curve — ordered list of (capacity_mw, cost_per_mwh) segments
- **Generation bounds**: `min_mw`, `max_mw` — hard constraints (no slack variables; thermal dispatch is directly controllable)

### GNL Dispatch Anticipation

> **Implementation Status**: Data model defined. Validation currently rejects GNL thermal plants (thermals with `gnl_config` present). This restriction will be removed when the GNL algorithm is implemented.

GNL (Gas Natural Liquefeito) thermals require dispatch decisions N stages ahead due to fuel ordering lead times. This fundamentally changes the LP structure for these plants:

- **`lag_stages`** — number of stages of dispatch anticipation (e.g., 2 means the dispatch must be decided 2 stages before execution)
- **GNL state variables**: For each GNL thermal, the algorithm adds `lag_stages` state variables to the SDDP state vector: `gnl_committed[thermal_id, t+1]`, `gnl_committed[thermal_id, t+2]`, ..., up to `lag_stages` ahead. These represent the MW dispatch already committed for future stages.
- **LP coupling**: At stage `t`, the GNL thermal's generation is constrained by the commitment made at stage `t - lag_stages`. The LP at stage `t` also includes decision variables for committing dispatch at stage `t + lag_stages`.
- **State dimension impact**: Each GNL thermal with `lag_stages = L` adds `L` state variables to the system, increasing the cut coefficient count and the FCF dimensionality.

See [Input System Entities §4](input-system-entities.md) for the input schema.

## 5. Transmission Line

In-memory line representation:

- Entity ID, name, source/target bus IDs
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- **Capacity**: `direct_mw`, `reverse_mw` — hard bounds (no slack variables)
- **Exchange cost**: Regularization cost per MWh (not a violation penalty)
- **Losses**: Transmission losses as percentage

## 6. Bus

In-memory bus representation:

- Entity ID, name
- **Deficit segments**: Pre-resolved piecewise-linear deficit cost curve. Each segment has `depth_mw` and `cost`. The final segment is unbounded (ensures LP feasibility). Deficit is a recourse slack — see [Penalty System §2](penalty-system.md).
- **Excess cost**: Cost for surplus generation absorption (recourse slack)

## 7. Pumping Station

In-memory pumping station representation. Independent system element — not a property of any hydro. See [Input System Entities §5](input-system-entities.md).

- Entity ID, name, bus assignment
- `source_hydro_id`, `destination_hydro_id` — water transfer endpoints
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- `consumption_mw_per_m3s` — power consumption rate
- Flow bounds: `min_m3s`, `max_m3s`
- No explicit cost — cost is implicit via energy consumption at the connected bus

## 8. Energy Contract

In-memory contract representation. See [Input System Entities §6](input-system-entities.md).

- Entity ID, name, bus assignment
- Type: `import` or `export`
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- `price_per_mwh` — cost (import) or revenue (export)
- Limits: `min_mw`, `max_mw`

## 9. Non-Controllable Source

In-memory non-controllable source representation. See [Input System Entities §7](input-system-entities.md).

Non-controllable sources represent intermittent generation (wind farms, solar plants, small hydros run-of-river, etc.) whose available output depends on external conditions rather than dispatch decisions. The solver receives a stochastic availability value per scenario and can only curtail generation below that availability — it cannot dispatch upward.

- Entity ID, name, bus assignment
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- `max_generation_mw` — installed capacity (hard upper bound, physical limit)
- **Availability**: stochastic generation available per (stage, scenario), provided by the scenario pipeline. Bounded by `[0, max_generation_mw]`.
- **Generation variable**: bounded by `[0, available_generation]` — the solver dispatches between 0 and whatever is available
- **Curtailment**: the difference between available generation and dispatched generation. Penalized with `curtailment_cost` (regularization, Category 3) to prioritize using non-controllable generation over curtailing it. Analogous to spillage cost for hydros — curtailment discards available "free" energy.

## 10. Penalty Tables (Pre-Resolved)

At runtime, the solver needs a single resolved penalty value for each (entity, stage, penalty_type) tuple. The three-tier cascade (global defaults → entity overrides → stage overrides) is resolved during input loading. See [Penalty System](penalty-system.md) for the cascade semantics and penalty categories.

### Pre-Resolution

During input loading, the implementation should build a resolved penalty lookup that can answer: "what is the effective penalty for entity X, stage T, penalty type P?" in constant time. The resolution walks:

1. Start with global defaults from `penalties.json`
2. Apply entity-level overrides from entity registry files (`hydros.json`, `buses.json`, etc.)
3. Apply stage-level overrides from stage override files

### Penalty Types Per Entity

**Bus penalties:**

| Penalty            | Category | Description                                |
| ------------------ | -------- | ------------------------------------------ |
| `deficit_segments` | Recourse | Piecewise deficit cost (not stage-varying) |
| `excess_cost`      | Recourse | Surplus generation absorption              |

**Line penalties:**

| Penalty         | Category       | Description         |
| --------------- | -------------- | ------------------- |
| `exchange_cost` | Regularization | Flow discouragement |

**Hydro penalties:**

| Penalty                           | Category             | Description                                                                  |
| --------------------------------- | -------------------- | ---------------------------------------------------------------------------- |
| `spillage_cost`                   | Regularization       | Prefer turbining over spilling                                               |
| `diversion_cost`                  | Regularization       | Prefer main channel flow                                                     |
| `fpha_turbined_cost`              | Regularization       | Prevent interior FPHA solutions (FPHA model only, must be > `spillage_cost`) |
| `storage_violation_below_cost`    | Constraint violation | Storage below dead volume                                                    |
| `filling_target_violation_cost`   | Constraint violation | Filling target not reached                                                   |
| `turbined_violation_below_cost`   | Constraint violation | Turbined flow below minimum                                                  |
| `outflow_violation_below_cost`    | Constraint violation | Outflow below environmental minimum                                          |
| `outflow_violation_above_cost`    | Constraint violation | Outflow above flood control limit                                            |
| `generation_violation_below_cost` | Constraint violation | Generation below contractual minimum                                         |
| `evaporation_violation_cost`      | Constraint violation | Evaporation constraint (bidirectional)                                       |
| `water_withdrawal_violation_cost` | Constraint violation | Unmet water withdrawal                                                       |

**Non-controllable source penalties:**

| Penalty            | Category       | Description                                                                          |
| ------------------ | -------------- | ------------------------------------------------------------------------------------ |
| `curtailment_cost` | Regularization | Penalizes curtailing available generation; prioritizes using non-controllable energy |

### Penalty Priority Ordering

$$\text{Filling target} > \text{Storage violation} > \text{Deficit} > \text{Constraint violations} > \text{Resource costs} > \text{Regularization}$$

Regularization costs (spillage, diversion, FPHA turbined, curtailment, exchange) are the lowest priority — they guide solution quality when the LP has degrees of freedom, but never override economic or feasibility objectives. Within FPHA hydros, the constraint `fpha_turbined_cost > spillage_cost` must hold per plant.

## 11. Pre-Resolved Entity Bounds

Similar to penalties, entity bounds are resolved during loading from the base values in entity registries with sparse stage overrides applied on top. The internal representation should provide: "what are the effective bounds for entity X, stage T?" in constant time.

### Bound Resolution

For each entity type, the base bounds come from the entity registry file (e.g., `hydros.json`). Stage-specific overrides from the constraints directory (e.g., `hydro_bounds`) replace individual fields where provided (null = keep base). See [Input Constraints §2](input-constraints.md).

### Hydro Bounds Per Stage

The resolved set of hydro bounds for a given (hydro, stage):

| Bound                  | Source                          | Slack                        |
| ---------------------- | ------------------------------- | ---------------------------- |
| `min_storage_hm3`      | Reservoir config                | `storage_violation_below`    |
| `max_storage_hm3`      | Reservoir config                | Hard (emergency spill)       |
| `min_turbined_m3s`     | Generation model                | `turbined_violation_below`   |
| `max_turbined_m3s`     | Generation model                | Hard                         |
| `min_outflow_m3s`      | Outflow config                  | `outflow_violation_below`    |
| `max_outflow_m3s`      | Outflow config                  | `outflow_violation_above`    |
| `min_generation_mw`    | Generation model                | `generation_violation_below` |
| `max_generation_mw`    | Generation model                | Hard                         |
| `max_diversion_m3s`    | Diversion config                | Hard                         |
| `filling_inflow_m3s`   | Filling config → stage override | —                            |
| `water_withdrawal_m3s` | Stage override                  | `water_withdrawal_violation` |

## 12. Stage and Block Definitions

The temporal structure must be fully loaded in memory for the solver to dimension LPs and iterate:

### Stage

- `id` — unique identifier (non-negative for study, negative for pre-study)
- `start_date`, `end_date` — temporal extent
- `season_id` — links to season definitions for PAR model cycling
- `blocks` — ordered list of blocks (see below)
- `block_mode` — `parallel` or `chronological` (can vary per stage)
- `state_variables` — boolean flags: `storage` (default true), `inflow_lags` (default false)
- `risk_measure` — `expectation` or CVaR parameters (`alpha`, `lambda`)
- `num_scenarios` — branching factor for this stage
- `sampling_method` — SAA, LHS, QMC, etc.

### Block

- `id` — contiguous within stage (0, 1, 2, ...)
- `name` — human-readable label
- `hours` — duration in hours
- Derived: `weight = hours / sum(all block hours in stage)`

### Policy Graph

The policy graph defines stage transitions and discounting:

- `type` — `finite_horizon` or `cyclic`
- `annual_discount_rate` — global rate, auto-converted to per-transition factor: `beta = 1/(1+r)^dt`
- Transitions: list of `(source_id, target_id, probability)` with optional per-transition rate overrides

### Season Definitions

Maps `season_id` to calendar periods. Required when deriving AR parameters from inflow history.

- `cycle_type`: `monthly` (12 seasons), `weekly` (52), or `custom`
- Validation: all stages sharing a `season_id` must have equal duration

## 13. Block Factors

Block factors modify LP bounds and demand values per (entity, stage, block). They are loaded during initialization and used when constructing LP definitions.

- **Load factors**: Per (bus, stage, block) multipliers on stochastic load demand. Affect the actual demand value in the load balance constraint.
- **Exchange factors**: Per (line, stage, block) multipliers on line capacity (direct and reverse). Affect the line flow bounds in the LP — a different role from load factors. Values >1.0 represent higher block-level capacity.

Default: 1.0 when not specified.

## 14. Scenario Pipeline

The scenario pipeline state loaded in memory:

### Inflow Models (Per Hydro, Per Stage)

Pre-resolved seasonal statistics and AR coefficients:

- `mean_m3s` (mu) — seasonal mean inflow
- `std_m3s` (sigma) — seasonal standard deviation
- AR order `p` and coefficient list `[psi_1, ..., psi_p]` (variable length, can be 0)

Source: user-provided `inflow_seasonal_stats.parquet` (mu, sigma, ar_order) + optional `inflow_ar_coefficients.parquet` (lag coefficients), OR derived from `inflow_history` via season aggregation and Yule-Walker fitting. See [Input Scenarios §2-3](input-scenarios.md).

### Load Seasonal Statistics (Per Bus, Per Stage)

- `mean_mw`, `std_mw` — load statistics (typically no AR structure)

### Correlation Model

- Named profiles, each containing correlation groups with entity lists and correlation matrices
- Stage-to-profile mapping (schedule) — pre-resolved so the solver can look up the active profile per stage
- Cholesky-decomposed matrices ready for scenario generation

Source: user-provided `correlation.json` OR estimated from AR residuals of inflow history. See [Input Scenarios §5](input-scenarios.md).

## 15. Generic Constraints

User-defined linear constraints parsed and validated during loading. See [Input Constraints §3](input-constraints.md).

### Constraint Definition

Each generic constraint has:

- `id`, `name`, `description`
- **Parsed expression**: A list of linear terms, each being `coefficient x variable_reference`, plus a constant
- **Sense**: `>=`, `<=`, or `==`
- **Slack config**: whether a slack variable is created, and its penalty cost

### Variable References

The expression parser produces typed variable references. The full catalog:

| Variable                       | Entity                  | Block-specific                      |
| ------------------------------ | ----------------------- | ----------------------------------- |
| `hydro_storage`                | hydro                   | No (stage-level)                    |
| `hydro_turbined`               | hydro                   | Yes                                 |
| `hydro_spillage`               | hydro                   | Yes                                 |
| `hydro_diversion`              | hydro                   | Yes                                 |
| `hydro_outflow`                | hydro                   | Yes (alias for turbined + spillage) |
| `hydro_generation`             | hydro                   | Yes                                 |
| `hydro_evaporation`            | hydro                   | No                                  |
| `hydro_withdrawal`             | hydro                   | No                                  |
| `thermal_generation`           | thermal                 | Yes                                 |
| `line_direct`                  | line                    | Yes                                 |
| `line_reverse`                 | line                    | Yes                                 |
| `bus_deficit`                  | bus                     | Yes                                 |
| `bus_excess`                   | bus                     | Yes                                 |
| `pumping_flow`                 | pumping station         | Yes                                 |
| `pumping_power`                | pumping station         | Yes                                 |
| `contract_import`              | contract                | Yes                                 |
| `contract_export`              | contract                | Yes                                 |
| `non_controllable_generation`  | non-controllable source | Yes                                 |
| `non_controllable_curtailment` | non-controllable source | Yes                                 |

### Constraint Bounds

Pre-resolved per (constraint_id, stage_id, block_id). If no bound exists for a (constraint, stage), the constraint is not active for that stage.

### Validation (at load time)

All entity IDs referenced in constraint expressions must exist in the loaded system. Block IDs (if specified) must be valid for the stage. Expression parsing errors are reported at load time, not at LP build time.

## 16. Initial Conditions

Pre-resolved initial state:

- **Storage** (operating hydros): `value_hm3` per hydro, validated within `[min_storage_hm3, max_storage_hm3]`
- **Filling storage** (filling hydros): `value_hm3` per hydro, validated within `[0, min_storage_hm3]`
- **GNL pipeline**: For each GNL thermal, the committed dispatch (MW) for each future stage within the lag window. Represented as `(thermal_id, stage_offset, committed_mw)` tuples where `stage_offset` ranges from 1 to `lag_stages`. These values initialize the GNL state variables at the start of the study. See [Input Constraints §1](input-constraints.md).

> **Note**: GNL pipeline initial conditions are currently rejected by validation (GNL thermals are not accepted). When GNL support is implemented, the validation rule will be relaxed and these initial conditions will be required for any thermal with `gnl_config` defined.

See [Input Constraints §1](input-constraints.md).

## Cross-References

- [Input System Entities](input-system-entities.md) — JSON schemas that map to entity representations
- [Input Hydro Extensions](input-hydro-extensions.md) — Geometry, production models, FPHA hyperplanes
- [Input Constraints](input-constraints.md) — Initial conditions, time-varying bounds, generic constraints
- [Input Scenarios](input-scenarios.md) — Stage/block, inflow, and load data, correlation
- [Penalty System](penalty-system.md) — Penalty cascade resolution and categories
- [Binary Formats](binary-formats.md) — Serialization format decisions, FlatBuffers policy schema, cut pool persistence
- [Scenario Generation](../03-architecture/scenario-generation.md) — Opening tree lifecycle (§2.3), sampling scheme abstraction (§3)
- [Design Principles](../00-overview/design-principles.md) — Order invariance requirement (§3)
