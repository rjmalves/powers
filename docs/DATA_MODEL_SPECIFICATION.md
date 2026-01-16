# POWE.RS v2.0 Data Model Specification

> **Document Purpose**: Complete specification of input/output data models for the refactored POWE.RS SDDP solver with MPI-based distributed computing.
>
> **Status**: DRAFT - Awaiting Review
> **Last Updated**: 2026-01-15
> **Version**: 0.1.0

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Production Scale Reference](#2-production-scale-reference)
3. [Input Data Model](#3-input-data-model)
4. [Output Data Model](#4-output-data-model)
5. [Internal Data Structures](#5-internal-data-structures)
6. [MPI Communication Structures](#6-mpi-communication-structures)
7. [File Format Decisions](#7-file-format-decisions)
8. [Validation Requirements](#8-validation-requirements)
9. [Migration Path](#9-migration-path)

---

## 1. Design Principles

### 1.1 Format Selection Criteria

| Data Type | Recommended Format | Rationale |
|-----------|-------------------|-----------|
| Configuration & Parameters | JSON | Human-readable, easily editable, small size |
| Entity Registries | JSON | Structured objects with relationships |
| Time Series Data | Parquet | Columnar, compressed, efficient for large data |
| Warm-start Data (Cuts/States) | Parquet | Large volumes, needs efficient I/O |
| Simulation Results | Parquet | High volume, per-entity indexing |
| Dictionaries/Metadata | CSV | Human-readable, small, universal |

### 1.2 Key Design Goals

1. **Separation of Concerns**: Static system data vs. dynamic algorithm data vs. stochastic data
2. **Scalability**: File formats that scale to production sizes without memory explosion
3. **Reproducibility**: All inputs deterministically produce same outputs
4. **Warm-start Support**: Efficient serialization/deserialization of algorithm state
5. **Distributed I/O**: Rank 0 loads, broadcasts to workers (or parallel loading where beneficial)
6. **Declaration Order Invariance**: Results must not depend on the order entities are declared in input files

### 1.3 Declaration Order Invariance (Critical Requirement)

> **⚠️ CRITICAL**: The optimization results MUST be identical regardless of the order in which entities are declared in input files. This is a fundamental correctness requirement.

**Principle**: If a user declares hydros A and B, runs the program, then exchanges the declaration order (B before A), the numerical results must be **bit-for-bit identical** (given the same random seed and same IDs).

**What determines identity**: The **entity ID** is the sole identifier. Two runs are equivalent if:
- All entity IDs are the same
- All entity properties are the same
- All relationships (by ID) are the same
- The random seed is the same

**What must NOT affect results**:
- Order of entities in JSON arrays (`hydros`, `thermals`, `buses`, `lines`)
- Order of rows in Parquet tables
- Order of constraints in `generic_constraints.json`
- Order of stages in `stages.json` (sorted by ID internally)
- Order of blocks within a stage (sorted by ID internally)
- Order of correlation blocks or entities within correlation blocks

**Implementation Requirements**:

1. **Canonical Ordering**: After loading, all entity collections must be sorted by ID before any processing
2. **Deterministic Iteration**: All iterations over entities must use the canonical (sorted by ID) order
3. **LP Variable Ordering**: LP variables must be created in canonical order (by entity ID, then by block ID)
4. **LP Constraint Ordering**: LP constraints must be added in canonical order
5. **Random Number Generation**: Scenario generation must iterate entities in canonical order
6. **Cut Coefficients**: Cut coefficient ordering must follow the canonical state variable order

**Validation**: The test suite must include order-invariance tests that:
1. Run the same case with entities in different declaration orders
2. Verify bit-for-bit identical results (costs, decisions, cuts)

```rust
// Canonical ordering example
impl System {
    /// Sort all entity collections by ID for order-invariant processing
    pub fn canonicalize(&mut self) {
        self.buses.sort_by_key(|b| b.id);
        self.lines.sort_by_key(|l| l.id);
        self.hydros.sort_by_key(|h| h.id);
        self.thermals.sort_by_key(|t| t.id);
    }
}

impl GenericConstraints {
    /// Sort constraints by ID for order-invariant processing
    pub fn canonicalize(&mut self) {
        self.constraints.sort_by_key(|c| c.id);
    }
}
```

**Why This Matters**:
- **Debugging**: Users can reorganize input files without worrying about result changes
- **Version Control**: Reordering entities for readability doesn't create spurious diffs in results
- **Correctness**: Non-deterministic behavior from ordering is a bug, not a feature
- **Parallelism**: MPI ranks must agree on ordering without communication

### 1.4 New Concepts

- **Blocks**: Subdivisions of a stage affecting LP construction and some outputs
- **Stage × Block**: Many operations are indexed by `(stage_id, block_id)`

---

## 2. Production Scale Reference

Based on the target production scenario:

| Dimension | Value | Memory Impact |
|-----------|-------|---------------|
| Stages | 120 | Graph size |
| Blocks per Stage | 1-3 (varies) | LP structure, outputs |
| Hydros | 160 | State dimension |
| Max AR Order | 6 | State dimension, Variables/Constraints |
| Thermals | 130 | Variables |
| Buses | 6 | Variables/Constraints |
| Lines | 10 | Variables/Constraints |
| Forward Passes | 200 | Parallelism |
| Iterations | 50 | Cut pool size |
| Scenarios per Node | 20 | Branching |
| Simulation Scenarios | 2000 | Output size |

### Derived Sizes

| Entity | Calculation | Size |
|--------|-------------|------|
| Max Cuts per Stage | 200 × 50 = 10,000 | Per stage |
| Total Cuts | 10,000 × 120 = 1,200,000 | Across all stages |
| State Dimension | 160 (storage) + 160×6 (lags) = 1120 | Per cut |
| Cut Memory | 1.2M × 1120 × 8B ≈ 10.7 GB | Total cuts |
| Simulation Rows | 2000 × 120 × 3 × ~500 vars ≈ 360M | Per output file |

---

## 3. Input Data Model

### 3.1 Directory Structure

> **Note**: Scenario noise generation always uses standard normal distributions. Non-negative inflow values are enforced at runtime via the configured `inflow_non_negativity` method. Deterministic load can be achieved by setting variance to 0 in the uncertainty model.

```
case_directory/
├── config.json                    # Algorithm configuration
├── system/
│   ├── topology.json              # Buses, lines
│   ├── hydros.json                # Hydro plant registry
│   ├── thermals.json              # Thermal plant registry
│   └── hydro_cascade.json         # Cascade topology (optional)
├── temporal/
│   ├── stages.json                # Stage definitions with blocks (incl. pre-study)
│   └── initial_conditions.json    # Initial storage
├── scenarios/
│   ├── correlation.json           # Correlation specification
│   ├── load_factors.json          # Load distribution by block (optional)
│   └── exchange_factors.json      # Exchange limits by block (optional)
├── constraints/
│   ├── generic_constraints.json   # User-defined linear constraints
│   └── constraint_bounds.parquet  # Time-varying constraint bounds
├── timeseries/
│   ├── inflow_models.parquet      # PAR model parameters per hydro × stage
│   ├── load_models.parquet        # Load model parameters per bus × stage
│   ├── inflow_history.parquet     # Historical inflows for AR initialization
│   ├── bus_penalties.parquet      # Deficit/excess costs per bus × stage
│   ├── hydro_penalties.parquet    # Spillage/violation costs per hydro × stage
│   ├── thermal_bounds.parquet     # Time-varying thermal bounds (optional)
│   ├── hydro_bounds.parquet       # Time-varying hydro bounds (optional)
│   ├── outflow_bounds.parquet     # Time-varying outflow bounds (optional)
│   ├── line_bounds.parquet        # Time-varying line bounds (optional)
│   └── filling_constraints.parquet # Dead-volume filling constraints (if needed)
└── warmstart/                     # Optional: continue from previous run
    ├── cuts.parquet               # Benders cuts
    ├── states.parquet             # Visited states
    └── metadata.json              # Iteration count, bounds, etc.
```

### 3.2 Configuration (`config.json`)

> **Note**: Solver selection (HiGHS, CPLEX, Gurobi) is determined at compile time via Cargo features due to licensing constraints. Solver parameters, retry strategies, warm-start, and basis reuse are hardcoded per solver implementation and not user-configurable.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/config.schema.json",
  "version": "2.0.0",
  
  "mpi": {
    "threads_per_rank": 192,
    "thread_binding": "close"
  },
  
  "modeling": {
    "block_mode": "parallel",
    "inflow_non_negativity": "truncate_zero"
  },
  
  "training": {
    "seed": 42,
    "num_iterations": 50,
    "num_forward_passes": 200,
    "convergence": {
      "method": "statistical",
      "confidence": 0.95,
      "tolerance": 0.01
    },
    "cut_selection": {
      "enabled": true,
      "method": "domination",
      "threshold": 0
    }
  },
  
  "checkpointing": {
    "enabled": true,
    "interval_iterations": 10,
    "path": "./checkpoints",
    "store_basis": true,
    "compress": true
  },
  
  "simulation": {
    "enabled": true,
    "num_scenarios": 2000,
    "output_mode": "streaming"
  },
  
  "output": {
    "path": "./output",
    "format": "parquet",
    "exports": {
      "training": true,
      "cuts": true,
      "states": true,
      "simulation": true,
      "forward_detail": false,
      "backward_detail": false
    },
    "compression": "zstd"
  }
}
```

#### Block Mode Configuration

| Mode | Description |
|------|-------------|
| `parallel` | Blocks are independent within a stage. Storage balance applies to the whole stage (sum of block durations). Simpler model, common for medium/long-term planning. |
| `chronological` | Blocks are sequential within a stage. Storage can vary between blocks (inter-block storage variables). Enables daily/weekly cycling patterns. More variables and constraints. |

**Chronological Blocks Details:**
- Storage continuity: `storage_end_block[b] = storage_start_block[b+1]`
- Load balance per block: Each block has its own load balance constraint
- Hydro balance per block: Inflow distributed across blocks, turbined/spillage per block
- State: Only end-of-stage storage is part of the SDDP state (not inter-block)

#### Inflow Non-Negativity Methods

| Method | Description |
|--------|-------------|
| `truncate_zero` | Truncate negative inflows to 0.0 (simple, fast, may bias distribution) |
| `resample_entity` | Resample noise for that specific entity until non-negative |
| `resample_correlation_block` | Resample noise for all entities in the same correlation block |
| `reflect` | Use absolute value: `inflow = abs(inflow)` (preserves some variance) |

#### Checkpointing Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable periodic checkpointing |
| `interval_iterations` | i32 | 10 | Checkpoint every N iterations |
| `path` | string | "./checkpoints" | Directory for checkpoint files |
| `store_basis` | bool | false | Store solver basis for exact reproducibility |
| `compress` | bool | true | Compress parquet files (zstd) |

> **⚠️ Reproducibility Note**: If `store_basis = false`, resuming from checkpoint may produce numerically different (but algorithmically equivalent) results compared to a straight run. This is because the solver may choose different pivots when reconstructing the basis. Set `store_basis = true` for exact reproducibility at the cost of ~20-30% larger checkpoint files and additional I/O time.

### 3.2.1 Penalties and Costs

> **Design Rationale**: The LP must always be feasible. Several physical and operational constraints may be impossible to satisfy in extreme scenarios (droughts, equipment failures, etc.). We define a **unified penalty system** using tabular Parquet files, consistent with how bounds are specified. This allows:
> 1. All penalty values explicitly declared per entity × stage
> 2. Easy validation and inspection
> 3. Consistent format with bounds tables
> 4. Clear separation between operational costs (spillage) and violation penalties (deficit)

#### Constraint Violation Categories

| Category | Constraint Type | Slack Variable | Direction |
|----------|----------------|----------------|-----------|
| **System** | Load balance | `deficit`, `excess` | Lower (deficit), Upper (excess) |
| **Hydro Generation** | Min generation | `generation_violation_below` | Lower bound |
| **Hydro Turbined** | Min turbined flow | `turbined_violation_below` | Lower bound |
| **Hydro Outflow** | Min/Max outflow | `outflow_violation_below`, `outflow_violation_above` | Both bounds |
| **Hydro Storage** | Min/Max storage | Usually hard bounds, but see note | - |

> **Note on Storage Bounds**: Storage min/max are typically hard physical limits (reservoir capacity). Violations are handled by emergency spillage (if above max) or infeasibility (if below min due to bad data). We don't add slacks for storage bounds.
>
> **Note on Thermals**: Thermal plants are modeled as always available within their bounds. No slack variables are needed—if a thermal cannot meet its minimum generation, it indicates a data error (bounds should be adjusted via `thermal_bounds.parquet`).

#### Penalty Files

All penalties are defined in tabular Parquet files in the `timeseries/` directory:

| File | Entity | Columns |
|------|--------|---------|
| `bus_penalties.parquet` | Buses | deficit_cost, excess_cost |
| `hydro_penalties.parquet` | Hydros | spillage_cost, turbined_violation_cost, outflow_violation_cost, generation_violation_cost |

#### Bus Penalties Schema (`timeseries/bus_penalties.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `bus_id` | i32 | Bus identifier |
| `stage_id` | i32 | Stage identifier |
| `deficit_cost` | f64 | $/MWh for unmet load |
| `excess_cost` | f64 | $/MWh for excess generation |

#### Hydro Penalties Schema (`timeseries/hydro_penalties.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `hydro_id` | i32 | Hydro identifier |
| `stage_id` | i32 | Stage identifier |
| `spillage_cost` | f64 | $/(m³/s·h) for spilled water (opportunity cost, not violation) |
| `turbined_violation_cost` | f64 | $/(m³/s·h) for turbined flow below min |
| `outflow_violation_cost` | f64 | $/(m³/s·h) for outflow outside [min, max] |
| `generation_violation_cost` | f64 | $/MWh for generation below min |

#### Penalty Semantics

| Penalty | Units | Applied To | Purpose |
|---------|-------|------------|---------|
| `deficit_cost` | $/MWh | Unmet load per bus per block | Cost of load shedding |
| `excess_cost` | $/MWh | Excess generation per bus per block | Dumping excess power |
| `spillage_cost` | $/(m³/s·h) | Water spilled (not turbined) | Opportunity cost, incentivizes turbining |
| `turbined_violation_cost` | $/(m³/s·h) | Turbined flow below min_turbined | Equipment/ecological flow |
| `outflow_violation_cost` | $/(m³/s·h) | Outflow outside [min, max] | Environmental flow requirements |
| `generation_violation_cost` | $/MWh | Generation below min_generation | Environmental/contractual min |

> **Note**: `spillage_cost` is NOT a violation penalty—it's a small incentive to turbine water rather than spill it. It should be much smaller than generation value (typically 0.001-0.01).

#### Hydro Variables and Bounds Summary

| Variable | Lower Bound | Upper Bound | Lower Slack | Upper Slack |
|----------|-------------|-------------|-------------|-------------|
| `storage` | `min_storage_hm3` | `max_storage_hm3` | Hard | Emergency spill |
| `turbined_flow` | `min_turbined_m3s` | `max_turbined_m3s` | With penalty | Hard |
| `spillage` | 0 | ∞ | Hard | - |
| `outflow` | `min_outflow_m3s` | `max_outflow_m3s` | With penalty | With penalty |
| `generation` | Derived from turbined | Derived from turbined | With penalty | Hard |

> **Relationship**: `outflow = turbined_flow + spillage`, `generation = productivity × turbined_flow`

#### Dead-Volume Filling Specifics

During the filling period:
- **No turbined flow**: `turbined_flow = 0` (hard constraint—turbines not installed/operational)
- **Outflow = spillage**: All released water goes through non-turbine outlets (spillways, bottom gates, etc.)
- **Min outflow requirement**: Environmental flow must be met via spillage
- If `inflow - filling_retention < min_outflow`, the `outflow_violation_below` slack absorbs the shortfall

The same `outflow_violation_cost` from `hydro_penalties.parquet` applies during filling. Spillage during filling also incurs `spillage_cost`, representing the lost potential of water that could have been stored.

#### LP Objective Function Impact

For each stage `t`, block `b`, scenario `s`:

```
minimize:
  // Operational costs
  + Σ_thermal (generation × cost_per_mwh)
  + Σ_hydro (spillage × spillage_cost)
  
  // Violation penalties
  + Σ_bus (deficit × deficit_cost)
  + Σ_bus (excess × excess_cost)
  + Σ_hydro (generation_violation_below × generation_violation_cost)
  + Σ_hydro (turbined_violation_below × turbined_violation_cost)
  + Σ_hydro (outflow_violation_below × outflow_violation_cost)
  + Σ_hydro (outflow_violation_above × outflow_violation_cost)
  
  // Future cost function
  + α[t+1]  // Cut approximation
```

### 3.3 System Topology (`system/topology.json`)

> **⚠️ Order Invariance**: The order of buses and lines in their arrays does NOT affect results. After loading, all are sorted by `id`. See Section 1.3.
>
> **Note**: Deficit is modeled as piecewise linear segments. Each segment specifies a depth (MW of unmet demand) and cost. Segments are cumulative: first `depth_mw` MW at first cost, next `depth_mw` MW at second cost, etc. The last segment with `depth_mw: null` extends to infinity.
>
> **Line Operative State**: Each line has an operative state per stage:
> - `non_existing`: Before `entry_stage_id` - no exchange variables, buses isolated
> - `operating`: Between `entry_stage_id` and `exit_stage_id` - normal exchange
> - `decommissioned`: After `exit_stage_id` - no exchange variables

```json
{
  "buses": [
    {
      "id": 0,
      "name": "SUDESTE",
      "deficit_segments": [
        {"depth_mw": 1000, "cost": 2000.0},
        {"depth_mw": 2000, "cost": 5000.0},
        {"depth_mw": null, "cost": 10000.0}
      ]
    }
  ],
  "lines": [
    {
      "id": 0,
      "name": "SE-NE",
      "source_bus_id": 0,
      "target_bus_id": 1,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "capacity": {
        "direct_mw": 5000.0,
        "reverse_mw": 3000.0
      },
      "exchange_penalty": 0.01,
      "losses_percent": 2.5
    }
  ]
}
```

#### Line Operative States

| State | Condition | LP Variables |
|-------|-----------|--------------|
| `non_existing` | Before `entry_stage_id` | None (buses isolated) |
| `operating` | Between `entry_stage_id` and `exit_stage_id` | direct_flow, reverse_flow |
| `decommissioned` | After `exit_stage_id` | None (buses isolated) |


### 3.4 Hydro Registry (`system/hydros.json`)

> **⚠️ Order Invariance**: The order of hydros in this array does NOT affect results. After loading, hydros are sorted by `id`. See Section 1.3.
>
> **Note**: The `generation` field supports extensibility for future modeling approaches. Currently, only `constant_productivity` is implemented. Future versions may add `height_volume_curve` (HPF with reservoir height functions), `tailwater_curve` (downstream level effects), or `efficiency_curve` (generator efficiency by operating point).
>
> Inflow models are defined per hydro × stage in `timeseries/inflow_models.parquet`, linked by `hydro_id`.
>
> **Operative State**: Each hydro has an operative state per stage, determined by `entry_stage_id`, `exit_stage_id`, and `filling.start_stage_id`:
> - `non_existing`: Before `filling.start_stage_id` (or `entry_stage_id` if no filling) - no variables in LP
> - `filling`: Between `filling.start_stage_id` and `entry_stage_id - 1` - reservoir fills, no generation
> - `operating`: Between `entry_stage_id` and `exit_stage_id` - normal operation
> - `decommissioned`: After `exit_stage_id` - no variables in LP
>
> **Dead-volume filling**: During filling stages, the reservoir accumulates water according to constraints in `timeseries/filling_constraints.parquet`. The `filling_inflow_m3s` is water retained for filling; the remainder (`inflow - filling_inflow`) must be released as outflow. Outflow must meet `min_outflow_m3s`. The modeling during filling is:
> - **turbined_flow = 0** (hard constraint, turbines not installed/operational)
> - **outflow = spillage** (all released water goes through bottom outlets)
> - **hydro_balance**: `storage_end = storage_start + (inflow - filling_inflow - outflow) × time_factor`
> - The `filling_inflow_m3s` is the target filling rate, but if `inflow - min_outflow < filling_inflow`, less water is retained
> - Slack variables handle infeasible scenarios (see `penalties.json`)
>
> **Outflow**: Outflow = turbined_flow + spillage. Outflow has explicit bounds (`min_outflow_m3s`, `max_outflow_m3s`) that can vary per stage via `outflow_bounds.parquet`.
>
> **Generation**: Generation = productivity × turbined_flow. Generation can have bounds (`min_generation_mw`, `max_generation_mw`) for contractual or operational reasons. These are derived from turbined bounds by default but can be explicitly constrained.
>
> **Penalties**: All violation penalties are defined in `penalties.json`. The hydro config only defines physical bounds, not penalty values.
>
> **Cascade redirection**: The `downstream_id` always refers to the physical downstream plant. During stages when the downstream plant doesn't exist (non_existing or filling), outflows are automatically redirected to the next operating downstream in the cascade.

```json
{
  "hydros": [
    {
      "id": 0,
      "name": "FURNAS",
      "bus_id": 0,
      "downstream_id": 2,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "filling": null,
      "reservoir": {
        "min_storage_hm3": 5733.0,
        "max_storage_hm3": 22950.0
      },
      "outflow": {
        "min_outflow_m3s": 0.0,
        "max_outflow_m3s": null
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.8765,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1692.0,
        "min_generation_mw": null,
        "max_generation_mw": null
      }
    },
    {
      "id": 10,
      "name": "NEW_HYDRO",
      "bus_id": 0,
      "downstream_id": 0,
      "entry_stage_id": 60,
      "exit_stage_id": null,
      "filling": {
        "start_stage_id": 48,
        "target_storage_hm3": 2500.0
      },
      "reservoir": {
        "min_storage_hm3": 1000.0,
        "max_storage_hm3": 5000.0
      },
      "outflow": {
        "min_outflow_m3s": 50.0,
        "max_outflow_m3s": null
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.95,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 500.0,
        "min_generation_mw": null,
        "max_generation_mw": null
      }
    }
  ]
}
```

#### Hydro Operative States

| State | Condition | LP Variables |
|-------|-----------|--------------|
| `non_existing` | Before filling or entry (no filling defined) | None |
| `filling` | Between `filling.start_stage_id` and `entry_stage_id - 1` | storage, outflow (=spillage), outflow_violation_below |
| `operating` | Between `entry_stage_id` and `exit_stage_id` | storage, turbined, spillage, outflow, generation, violation slacks |
| `decommissioned` | After `exit_stage_id` | None |

#### Hydro LP Variables by State

| Variable | `non_existing` | `filling` | `operating` | `decommissioned` |
|----------|----------------|-----------|-------------|------------------|
| `storage` | ✗ | ✓ | ✓ | ✗ |
| `turbined_flow` | ✗ | ✗ (=0) | ✓ | ✗ |
| `spillage` | ✗ | ✓ | ✓ | ✗ |
| `outflow` | ✗ | ✓ (=spillage) | ✓ | ✗ |
| `generation` | ✗ | ✗ (=0) | ✓ | ✗ |
| `turbined_violation_below` | ✗ | ✗ | ✓ | ✗ |
| `outflow_violation_below` | ✗ | ✓ | ✓ | ✗ |
| `outflow_violation_above` | ✗ | ✓ | ✓ | ✗ |
| `generation_violation_below` | ✗ | ✗ | ✓ | ✗ |


### 3.5 Thermal Registry (`system/thermals.json`)

> **⚠️ Order Invariance**: The order of thermals in this array does NOT affect results. After loading, thermals are sorted by `id`. See Section 1.3.
>
> **Note**: Use `entry_stage_id` and `exit_stage_id` to model plants entering or exiting the system. Alternatively, use `thermal_bounds.parquet` to set generation to 0 for stages where the plant is offline.
>
> **Operative State**: Each thermal has an operative state per stage:
> - `non_existing`: Before `entry_stage_id` - no variables in LP
> - `operating`: Between `entry_stage_id` and `exit_stage_id` - normal operation
> - `decommissioned`: After `exit_stage_id` - no variables in LP

```json
{
  "thermals": [
    {
      "id": 0,
      "name": "ANGRA1",
      "bus_id": 0,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "cost_segments": [
        {"capacity_mw": 640.0, "cost_per_mwh": 15.0}
      ],
      "generation": {
        "min_mw": 500.0,
        "max_mw": 640.0
      }
    }
  ]
}
```

#### Thermal Operative States

| State | Condition | LP Variables |
|-------|-----------|--------------|
| `non_existing` | Before `entry_stage_id` | None |
| `operating` | Between `entry_stage_id` and `exit_stage_id` | generation per segment |
| `decommissioned` | After `exit_stage_id` | None |


### 3.6 Stage Definitions (`temporal/stages.json`)

> **⚠️ Order Invariance**: The order of stages and blocks in their arrays does NOT affect results. After loading, stages are sorted by `id`, and blocks within each stage are sorted by `id`. See Section 1.3.
>
> **Note**: Each stage defines its own blocks (count and hours). The weight is computed internally from block hours, not user-specified. Block IDs within each stage must be contiguous, starting at 0 (validated: 0, 1, 2, ..., n-1).
>
> **Pre-study stages**: Stages with negative IDs represent historical periods before the study horizon. These are used only for PAR model initialization (providing lag values). Pre-study stages only need `id`, `start_date`, and `end_date`.

```json
{
  "pre_study_stages": [
    {"id": -6, "start_date": "2023-07-01", "end_date": "2023-08-01"},
    {"id": -5, "start_date": "2023-08-01", "end_date": "2023-09-01"},
    {"id": -4, "start_date": "2023-09-01", "end_date": "2023-10-01"},
    {"id": -3, "start_date": "2023-10-01", "end_date": "2023-11-01"},
    {"id": -2, "start_date": "2023-11-01", "end_date": "2023-12-01"},
    {"id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01"}
  ],
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [
        {"id": 0, "name": "LEVE", "hours": 168},
        {"id": 1, "name": "MEDIA", "hours": 336},
        {"id": 2, "name": "PESADA", "hours": 168}
      ],
      "risk_measure": "expectation",
      "state_variables": "storage_and_inflow",
      "num_scenarios": 20
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "blocks": [
        {"id": 0, "name": "LEVE", "hours": 168},
        {"id": 1, "name": "MEDIA", "hours": 336},
        {"id": 2, "name": "PESADA", "hours": 168}
      ],
      "risk_measure": "expectation",
      "state_variables": "storage_and_inflow",
      "num_scenarios": 20
    }
  ],
  "transitions": [
    {"source_id": 0, "target_id": 1, "probability": 1.0, "discount_rate": 0.0}
  ]
}
```

### 3.7 Uncertainty Models (`timeseries/inflow_models.parquet`)

> **Note**: Uncertainty models are now defined per entity per stage in tabular format. This enables:
> - Variable time resolutions (daily, weekly, monthly, quarterly stages)
> - No cycling/season complexity - each stage has explicit parameters
> - Easy bulk editing and programmatic generation
> - AR order 0 = independent noise (no temporal correlation)
>
> The AR model for stage `t` uses lags from previous stages. AR coefficients reference normalized residuals from stages `t-1`, `t-2`, ..., `t-order`.

#### Inflow Models Schema (`timeseries/inflow_models.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `hydro_id` | i32 | Hydro plant ID (must exist in hydros.json) |
| `stage_id` | i32 | Stage ID (must exist in stages.json) |
| `mean_m3s` | f64 | Mean inflow for this stage |
| `std_m3s` | f64 | Standard deviation (0 = deterministic) |
| `ar_order` | i32 | AR order (0 = independent, 1-12 typical) |
| `ar_coef_01` | f64 | AR coefficient for lag  1 (null if order <  1) |
| `ar_coef_02` | f64 | AR coefficient for lag  2 (null if order <  2) |
| `ar_coef_03` | f64 | AR coefficient for lag  3 (null if order <  3) |
| `ar_coef_04` | f64 | AR coefficient for lag  4 (null if order <  4) |
| `ar_coef_05` | f64 | AR coefficient for lag  5 (null if order <  5) |
| `ar_coef_06` | f64 | AR coefficient for lag  6 (null if order <  6) |
| `ar_coef_07` | f64 | AR coefficient for lag  7 (null if order <  7) |
| `ar_coef_08` | f64 | AR coefficient for lag  8 (null if order <  8) |
| `ar_coef_09` | f64 | AR coefficient for lag  9 (null if order <  9) |
| `ar_coef_10` | f64 | AR coefficient for lag 10 (null if order < 10) |
| `ar_coef_11` | f64 | AR coefficient for lag 11 (null if order < 11) |
| `ar_coef_12` | f64 | AR coefficient for lag 12 (null if order < 12) |

#### Load Models Schema (`timeseries/load_models.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `bus_id` | i32 | Bus ID |
| `stage_id` | i32 | Stage ID (must exist in stages.json) |
| `mean_mw` | f64 | Mean load for this stage |
| `std_mw` | f64 | Standard deviation (0 = deterministic) |

> **Note**: Load models are typically independent (no AR), so no AR columns are included. If AR load models are needed in the future, a similar structure can be added.


### 3.8 Initial Conditions (`temporal/initial_conditions.json`)

> **Note**: Initial storage is the reservoir level at the start of the study. For hydros with `entry_stage_id`, this is the storage when they enter the system (not at stage 0).
>
> Inflow history is in a separate Parquet file indexed by `hydro_id` and `stage_id` (using pre-study stage IDs). Only hydros active from the start need pre-study inflow history.

```json
{
  "storage": [
    {"hydro_id": 0, "value_hm3": 15000.0},
    {"hydro_id": 1, "value_hm3": 8500.0},
    {"hydro_id": 10, "value_hm3": 2500.0}
  ]
}
```

> **Validation**: 
> - Every hydro in `hydros.json` must have an entry in `storage`
> - Storage value must be within `[min_storage_hm3, max_storage_hm3]`
> - For hydros entering later, this is their initial storage at entry

#### Inflow History Schema (`timeseries/inflow_history.parquet`)

> **Note**: Contains realized inflow values for pre-study stages (negative stage IDs). Used to initialize AR model lags. 
>
> **Validation**:
> - Pre-study stages must cover at least the maximum AR order used (if max order is 6, need stages -6 to -1)
> - Each hydro active from stage 0 must have entries for all required pre-study stages based on its AR order
> - Hydros entering later (entry_stage_id > 0) do not need pre-study history; their AR is initialized from study stages

| Column | Type | Description |
|--------|------|-------------|
| `hydro_id` | i32 | Hydro plant ID |
| `stage_id` | i32 | Pre-study stage ID (negative, e.g., -6, -5, ..., -1) |
| `inflow_m3s` | f64 | Realized inflow value |

Example (for AR order up to 6, with 2 hydros active from start):
```
hydro_id | stage_id | inflow_m3s
---------|----------|------------
0        | -6       | 45.2
0        | -5       | 52.1
0        | -4       | 58.3
0        | -3       | 61.0
0        | -2       | 55.8
0        | -1       | 48.5
1        | -6       | 120.5
1        | -5       | 135.2
...
```

### 3.9 Load Factors by Block (`scenarios/load_factors.json`) - Optional

> **Note**: This file is **optional**. If missing, all block factors default to 1.0.
>
> Load uncertainty generates a base load realization in **MW** per stage (similar to how inflows are generated in m³/s). The MW unit is power, independent of time duration. To get energy (MWh), multiply by block hours.
>
> Block factors are **multipliers** applied to the stochastic load value to get block-level loads. For example, if stage load is 1000 MW and block factors are [0.85, 1.00, 1.15], the blocks get [850, 1000, 1150] MW respectively.

```json
{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [
        {"block_id": 0, "factor": 0.85},
        {"block_id": 1, "factor": 1.00},
        {"block_id": 2, "factor": 1.15}
      ]
    }
  ]
}
```

### 3.10 Exchange Factors by Block (`scenarios/exchange_factors.json`) - Optional

> **Note**: This file is **optional**. If missing, all block factors default to 1.0.
>
> Exchange (transmission) limits may vary by block due to thermal limits, contractual constraints, or operational patterns. Block factors are **multipliers** applied to the stage-level line capacity to get block-level limits.
>
> For example, if a line has 5000 MW direct capacity and block factors are [0.90, 1.00, 1.10], the blocks get [4500, 5000, 5500] MW respectively.

```json
{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        {"block_id": 0, "direct_factor": 0.90, "reverse_factor": 0.90},
        {"block_id": 1, "direct_factor": 1.00, "reverse_factor": 1.00},
        {"block_id": 2, "direct_factor": 1.10, "reverse_factor": 1.10}
      ]
    }
  ]
}
```

### 3.11 Correlation (`scenarios/correlation.json`)

```json
{
  "method": "cholesky",
  "blocks": [
    {
      "name": "cascade_correlation",
      "entities": [
        {"type": "inflow", "id": 0},
        {"type": "inflow", "id": 1},
        {"type": "inflow", "id": 2}
      ],
      "matrix": [
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
      ]
    }
  ]
}
```

### 3.11 Time Series Data (`timeseries/*.parquet`)

> **Note**: Time-varying bounds allow entities to have different operational limits per stage. This is more direct than availability factors - the LP uses these bounds directly.

#### Thermal Bounds Schema (`timeseries/thermal_bounds.parquet`) - Optional

> **Note**: If a thermal is not present for a stage, uses bounds from `thermals.json`. Partial overrides allowed (only specify stages that differ from base).

| Column | Type | Description |
|--------|------|-------------|
| `thermal_id` | i32 | Thermal unit identifier |
| `stage_id` | i32 | Stage index |
| `min_generation_mw` | f64 | Minimum generation (null = use base) |
| `max_generation_mw` | f64 | Maximum generation (null = use base) |

#### Hydro Bounds Schema (`timeseries/hydro_bounds.parquet`) - Optional

> **Note**: If a hydro is not present for a stage, uses bounds from `hydros.json`. Useful for maintenance outages, seasonal restrictions, environmental constraints.

| Column | Type | Description |
|--------|------|-------------|
| `hydro_id` | i32 | Hydro plant identifier |
| `stage_id` | i32 | Stage index |
| `min_turbined_m3s` | f64 | Minimum turbined flow (null = use base) |
| `max_turbined_m3s` | f64 | Maximum turbined flow (null = use base) |
| `min_storage_hm3` | f64 | Minimum storage (null = use base) |
| `max_storage_hm3` | f64 | Maximum storage (null = use base) |

#### Line Bounds Schema (`timeseries/line_bounds.parquet`) - Optional

> **Note**: If a line is not present for a stage, uses bounds from `topology.json`. Useful for planned transmission upgrades or temporary capacity reductions.

| Column | Type | Description |
|--------|------|-------------|
| `line_id` | i32 | Transmission line identifier |
| `stage_id` | i32 | Stage index |
| `direct_mw` | f64 | Direct flow capacity (null = use base) |
| `reverse_mw` | f64 | Reverse flow capacity (null = use base) |

#### Filling Constraints Schema (`timeseries/filling_constraints.parquet`) - Required for hydros with filling

> **Note**: Specifies the filling inflow and minimum outflow constraints during dead-volume filling stages. Required for each hydro with `filling` configured, for each stage in the filling period.
>
> - `filling_inflow_m3s`: Water retained for reservoir filling (removed from cascade)
> - `min_outflow_m3s`: Minimum outflow required (environmental/downstream needs)
> - If `inflow - filling_inflow < min_outflow`, a slack variable with `outflow_violation_penalty` is used

| Column | Type | Description |
|--------|------|-------------|
| `hydro_id` | i32 | Hydro plant identifier |
| `stage_id` | i32 | Stage index (must be within filling period) |
| `filling_inflow_m3s` | f64 | Water retained for filling |
| `min_outflow_m3s` | f64 | Minimum required outflow |

#### Outflow Bounds Schema (`timeseries/outflow_bounds.parquet`) - Optional

> **Note**: Specifies time-varying outflow bounds. If not present for a stage, uses base bounds from `hydros.json`. Outflow = turbined_flow + spillage.

| Column | Type | Description |
|--------|------|-------------|
| `hydro_id` | i32 | Hydro plant identifier |
| `stage_id` | i32 | Stage index |
| `min_outflow_m3s` | f64 | Minimum outflow (null = use base) |
| `max_outflow_m3s` | f64 | Maximum outflow (null = use base) |

### 3.12 Generic Constraints (`constraints/`)

> **⚠️ Order Invariance**: The order of constraints in `generic_constraints.json` does NOT affect results. After loading, constraints are sorted by `id`. See Section 1.3.
>
> **Design Rationale**: Users may need to express custom linear constraints that combine multiple LP variables. These "generic" or "free" constraints allow modeling:
> - Minimum/maximum total hydro generation per region
> - Energy contracts (sum of generation from specific plants)
> - Irrigation agreements (sum of outflows)
> - Environmental corridors (combined outflow requirements)
> - Fuel availability (sum of thermal generation)
> - Any other linear combination of optimization variables

#### Variable Reference Syntax

Variables are referenced using a function-like syntax: `variable_type(entity_id)`. For block-specific variables (when using chronological blocks), use `variable_type(entity_id, block_id)` or omit block_id to sum over all blocks.

| Variable Name | Syntax | Units | Description |
|---------------|--------|-------|-------------|
| `hydro_storage` | `hydro_storage(id)` | hm³ | End-of-stage storage |
| `hydro_turbined` | `hydro_turbined(id)` or `hydro_turbined(id, block)` | m³/s | Turbined flow |
| `hydro_spillage` | `hydro_spillage(id)` or `hydro_spillage(id, block)` | m³/s | Spillage |
| `hydro_outflow` | `hydro_outflow(id)` or `hydro_outflow(id, block)` | m³/s | Total outflow |
| `hydro_generation` | `hydro_generation(id)` or `hydro_generation(id, block)` | MW | Power generation |
| `thermal_generation` | `thermal_generation(id)` or `thermal_generation(id, block)` | MW | Power generation |
| `line_direct` | `line_direct(id)` or `line_direct(id, block)` | MW | Direct flow |
| `line_reverse` | `line_reverse(id)` or `line_reverse(id, block)` | MW | Reverse flow |
| `bus_deficit` | `bus_deficit(id)` or `bus_deficit(id, block)` | MW | Deficit |
| `bus_excess` | `bus_excess(id)` or `bus_excess(id, block)` | MW | Excess |

#### Expression Grammar

Expressions are linear combinations of variables with numeric coefficients:

```ebnf
expression    ::= term (('+' | '-') term)*
term          ::= coefficient? variable | number
coefficient   ::= number '*'
variable      ::= var_name '(' entity_id (',' block_id)? ')'
var_name      ::= 'hydro_storage' | 'hydro_turbined' | 'hydro_spillage' | 'hydro_outflow' 
                | 'hydro_generation' | 'thermal_generation' | 'line_direct' | 'line_reverse'
                | 'bus_deficit' | 'bus_excess'
entity_id     ::= integer
block_id      ::= integer
number        ::= float | integer
```

**Examples:**
- `hydro_generation(10) + hydro_generation(11) + hydro_generation(12)` — sum of generation from 3 hydros
- `2.5 * thermal_generation(5) - hydro_generation(3)` — weighted combination
- `hydro_outflow(7) + hydro_outflow(8)` — combined outflow from two plants
- `thermal_generation(0) + thermal_generation(1) + 100.0` — sum with constant offset
- `hydro_turbined(5, 0) + hydro_turbined(5, 1)` — sum of turbined in blocks 0 and 1

#### Constraint Definition (`constraints/generic_constraints.json`)

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/generic_constraints.schema.json",
  "constraints": [
    {
      "id": 0,
      "name": "min_southeast_hydro",
      "description": "Minimum hydro generation in Southeast region",
      "expression": "hydro_generation(0) + hydro_generation(1) + hydro_generation(2)",
      "sense": ">=",
      "slack": {
        "enabled": true,
        "penalty": 5000.0
      }
    },
    {
      "id": 1,
      "name": "itaipu_contract",
      "description": "Itaipu energy contract - must deliver at least base amount",
      "expression": "hydro_generation(50)",
      "sense": ">=",
      "slack": {
        "enabled": true,
        "penalty": 8000.0
      }
    },
    {
      "id": 2,
      "name": "environmental_flow",
      "description": "Combined outflow requirement for river stretch",
      "expression": "hydro_outflow(10) + hydro_outflow(11)",
      "sense": ">=",
      "slack": {
        "enabled": true,
        "penalty": 3000.0
      }
    },
    {
      "id": 3,
      "name": "gas_availability",
      "description": "Total gas thermal generation limited by pipeline",
      "expression": "thermal_generation(5) + thermal_generation(6) + thermal_generation(7)",
      "sense": "<=",
      "slack": {
        "enabled": false
      }
    }
  ]
}
```

#### Constraint Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | i32 | Yes | Unique constraint identifier |
| `name` | string | Yes | Short name for reports |
| `description` | string | No | Human-readable description |
| `expression` | string | Yes | Linear expression (parsed using grammar above) |
| `sense` | string | Yes | `">="` (lower bound), `"<="` (upper bound), or `"=="` (equality) |
| `slack.enabled` | bool | Yes | Whether to add slack variable for feasibility |
| `slack.penalty` | f64 | If enabled | Penalty per unit of violation |

#### Constraint Bounds (`constraints/constraint_bounds.parquet`)

Bounds can vary by stage (and optionally by block for block-specific constraints). This follows the same pattern as entity bounds.

| Column | Type | Description |
|--------|------|-------------|
| `constraint_id` | i32 | References constraint definition |
| `stage_id` | i32 | Stage index |
| `block_id` | i32 | Block index (null = applies to all blocks) |
| `bound` | f64 | RHS value for the constraint |

**Example rows:**
| constraint_id | stage_id | block_id | bound |
|---------------|----------|----------|-------|
| 0 | 0 | null | 5000.0 |
| 0 | 1 | null | 5200.0 |
| 1 | 0 | null | 6000.0 |
| 2 | 0 | null | 150.0 |
| 3 | 0 | 0 | 2000.0 |
| 3 | 0 | 1 | 2500.0 |

#### LP Integration

For constraint `c` with sense `>=`:
```
Σ (coef_i × var_i) + slack_below[c] >= bound[c]
```

For constraint `c` with sense `<=`:
```
Σ (coef_i × var_i) - slack_above[c] <= bound[c]
```

For constraint `c` with sense `==`:
```
Σ (coef_i × var_i) + slack_below[c] - slack_above[c] == bound[c]
```

Slack variables are only created if `slack.enabled = true`.

#### Validation Rules

1. All entity IDs referenced in expressions must exist in the system
2. Block IDs (if specified) must be valid for the stage
3. Expressions must parse correctly according to the grammar
4. `constraint_bounds.parquet` must have entries for all study stages (0 to num_stages-1) for each constraint
5. Constraint IDs must be unique and contiguous (0, 1, 2, ...)
6. If `slack.enabled = true`, `slack.penalty` must be provided and positive

### 3.13 Warm-start Data (`warmstart/`)

> **Purpose**: Enable resumption of training after checkpointing. The warm-start directory contains the complete algorithm state needed to continue from a previous run.
>
> **⚠️ Reproducibility Warning**: Resuming from a checkpoint may produce slightly different results than a straight run due to solver basis state. See "Solver Basis Persistence" below for mitigation strategies.
>
> **Partitioning**: For production-scale cases, cuts and states files can become very large (>10GB). Files are partitioned by stage for practical handling:
> ```
> warmstart/
> ├── metadata.json
> ├── state_dictionary.json       # Maps coefficient/component indices to entity IDs
> ├── cuts/
> │   ├── stage_000.parquet
> │   ├── stage_001.parquet
> │   └── ...
> ├── states/
> │   ├── stage_000.parquet
> │   └── ...
> └── basis/                       # Optional: solver basis for exact reproducibility
>     ├── stage_000.parquet
>     └── ...
> ```

#### State Dictionary (`state_dictionary.json`)

> **Critical for Compatibility**: This file defines the mapping between coefficient/component indices and the actual state variables. Without this, it's impossible to correctly interpret `coefficient_0`, `coefficient_1`, etc.
>
> The dictionary follows the **canonical ordering** (see Section 1.3): state variables are ordered by entity type, then by entity ID.

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/state_dictionary.schema.json",
  "version": "2.0.0",
  "state_dimension": 320,
  "state_variables": [
    {"index": 0, "type": "storage", "entity_type": "hydro", "entity_id": 0, "name": "FURNAS"},
    {"index": 1, "type": "storage", "entity_type": "hydro", "entity_id": 1, "name": "MARIMBONDO"},
    {"index": 2, "type": "storage", "entity_type": "hydro", "entity_id": 2, "name": "ITUMBIARA"},
    // ... storage for all hydros (canonical order by ID)
    {"index": 160, "type": "inflow_lag_1", "entity_type": "hydro", "entity_id": 0, "name": "FURNAS"},
    {"index": 161, "type": "inflow_lag_1", "entity_type": "hydro", "entity_id": 1, "name": "MARIMBONDO"},
    // ... inflow lags for all hydros (if using inflow as state)
  ],
  "checksum": "sha256:abc123..."
}
```

#### State Dictionary Fields

| Field | Type | Description |
|-------|------|-------------|
| `index` | i32 | Column index in cuts/states (coefficient_N, component_N) |
| `type` | string | Variable type: `"storage"`, `"inflow_lag_1"`, `"inflow_lag_2"`, ... |
| `entity_type` | string | `"hydro"`, `"thermal"`, `"bus"`, etc. |
| `entity_id` | i32 | Entity ID (matches entity registries) |
| `name` | string | Human-readable entity name (for debugging) |

#### Compatibility Validation

When loading warm-start data, the system MUST verify:
1. `state_dictionary.json` exists and matches current system
2. Entity IDs in dictionary exist in current system
3. State dimension matches current configuration
4. Checksum matches to detect file corruption

If validation fails, warm-start is rejected with a clear error message.

#### Cuts Schema (`warmstart/cuts/stage_XXX.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `cut_id` | i64 | Unique cut identifier (unique within stage) |
| `iteration` | i32 | Iteration when generated |
| `forward_pass_idx` | i32 | Forward pass index within iteration |
| `scenario_idx` | i32 | Scenario that generated this cut |
| `rhs` | f64 | Cut RHS value (α₀) |
| `is_active` | bool | Whether cut is active in final model |
| `coefficient_0` | f64 | Coefficient for state variable 0 (see dictionary) |
| `coefficient_1` | f64 | Coefficient for state variable 1 |
| ... | ... | (up to state_dimension - 1) |

> **Interpretation**: A cut for stage t is: `α[t+1] ≥ rhs + Σᵢ coefficient_i × (state_i - state_i_at_generation)`
> The coefficient indices map to state variables via `state_dictionary.json`.

#### States Schema (`warmstart/states/stage_XXX.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `state_id` | i64 | Unique state identifier (unique within stage) |
| `iteration` | i32 | Iteration when visited |
| `forward_pass_idx` | i32 | Forward pass index |
| `scenario_idx` | i32 | Scenario that visited this state |
| `dominating_cut_id` | i64 | Cut ID currently dominating this state |
| `dominating_objective` | f64 | Objective value at dominating cut |
| `component_0` | f64 | State variable 0 value (see dictionary) |
| `component_1` | f64 | State variable 1 value |
| ... | ... | (up to state_dimension - 1) |

#### Solver Basis Schema (`warmstart/basis/stage_XXX.parquet`) - Optional

> **Purpose**: Store solver basis information to achieve exact reproducibility when resuming. Without this, the solver may choose different pivots, leading to different (but equivalent) optimal solutions and thus different cuts.
>
> **Trade-off**: Storing basis significantly increases checkpoint size and I/O time. For most use cases, slight numerical differences are acceptable.

| Column | Type | Description |
|--------|------|-------------|
| `variable_idx` | i32 | LP variable index |
| `basis_status` | i8 | Basis status: 0=lower, 1=basic, 2=upper, 3=free, 4=fixed |
| `row_idx` | i32 | Constraint row index (for row status) |
| `row_status` | i8 | Row basis status |

> **Note**: Basis format is solver-dependent. The implementation should serialize in a generic format and translate to solver-specific format on load.

#### Metadata (`warmstart/metadata.json`)

```json
{
  "$schema": "https://powers-rs.io/schemas/v2/warmstart_metadata.schema.json",
  "version": "2.0.0",
  "created_at": "2026-01-15T12:00:00Z",
  "powers_version": "2.0.0",
  "solver": "highs",
  "solver_version": "1.7.0",
  
  "algorithm_state": {
    "completed_iterations": 25,
    "last_forward_pass": 199,
    "rng_state": "base64:...",
    "final_lower_bound": 1234567.89,
    "best_upper_bound": 1245678.90,
    "gap_percent": 0.89,
    "best_iteration": 23
  },
  
  "data_integrity": {
    "state_dimension": 320,
    "total_cuts": 500000,
    "active_cuts": 450000,
    "total_states": 500000,
    "config_hash": "sha256:abc123...",
    "system_hash": "sha256:def456...",
    "state_dictionary_checksum": "sha256:789xyz..."
  },
  
  "partitioning": {
    "num_stages": 120,
    "cuts_by_stage": [4000, 4200, ...],
    "states_by_stage": [4000, 4200, ...],
    "has_basis": true
  },
  
  "reproducibility": {
    "basis_stored": true,
    "exact_resume_supported": true,
    "notes": "Basis stored for all stages. Exact reproducibility expected."
  }
}
```

#### Metadata Fields

| Section | Field | Description |
|---------|-------|-------------|
| `algorithm_state` | `completed_iterations` | Number of iterations completed |
| | `last_forward_pass` | Last forward pass index (for scenario indexing) |
| | `rng_state` | Serialized RNG state for exact scenario reproducibility |
| | `final_lower_bound` | Best lower bound at checkpoint |
| | `best_upper_bound` | Best upper bound (simulation) at checkpoint |
| `data_integrity` | `config_hash` | SHA-256 hash of config.json |
| | `system_hash` | SHA-256 hash of system topology + entities |
| | `state_dictionary_checksum` | Checksum of state_dictionary.json |
| `partitioning` | `cuts_by_stage` | Number of cuts per stage file |
| | `has_basis` | Whether solver basis is stored |
| `reproducibility` | `basis_stored` | Whether basis files exist |
| | `exact_resume_supported` | Whether exact reproducibility is possible |

#### Resume Validation

On resume, the loader MUST verify:

1. **Version compatibility**: `powers_version` is compatible with current version
2. **Config hash match**: Current config matches `config_hash` (or explicit override flag)
3. **System hash match**: Current system matches `system_hash` (entities must be identical)
4. **State dictionary match**: Dictionary checksum matches and all entities exist
5. **File completeness**: All partitioned files exist for all stages
6. **Optional basis check**: If `exact_resume_supported = true` and user requests exact resume, verify basis files exist

#### Reproducibility Guarantees

| Scenario | Reproducibility | Notes |
|----------|-----------------|-------|
| Straight run (no checkpoint) | ✅ Bit-for-bit identical | Same seed → same results |
| Resume with basis | ✅ Bit-for-bit identical | Solver basis restored → same pivots |
| Resume without basis | ⚠️ Equivalent optimum | Same optimum, but possibly different dual values → different cuts |
| Resume with modified config | ❌ Not supported | Hash mismatch → error |
| Resume with modified system | ❌ Not supported | Hash mismatch → error |

> **User Guidance**: For critical studies requiring exact reproducibility:
> 1. Enable `checkpoint.store_basis = true` in config
> 2. Accept the additional I/O overhead (~20-30% larger checkpoints)
> 3. For less critical studies, disable basis storage and accept minor numerical differences

---

## 4. Output Data Model

### 4.1 Directory Structure

```
output_directory/
├── training.parquet               # Iteration-level convergence
├── cuts/                          # Cuts per stage (optional split)
│   ├── stage_000.parquet
│   ├── stage_001.parquet
│   └── ...
├── states/                        # States per stage (optional split)
│   ├── stage_000.parquet
│   └── ...
├── simulation/                    # Simulation results
│   ├── summary.parquet            # Per-scenario totals
│   ├── operational/               # Per-stage × block detail
│   │   ├── hydro.parquet          # Hydro operations
│   │   ├── thermal.parquet        # Thermal operations
│   │   ├── exchange.parquet       # Line flows
│   │   └── deficit.parquet        # Unmet demand
│   └── state/                     # State trajectories
│       ├── storage.parquet
│       └── inflow.parquet
├── dictionaries/
│   ├── variable_dictionary.csv
│   ├── coefficient_dictionary.csv
│   └── state_component_dictionary.csv
└── metadata.json                  # Run metadata
```

### 4.2 Training Output (`training.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `iteration` | i32 | Iteration number (1-based) |
| `lower_bound` | f64 | Lower bound (first-stage cost) |
| `upper_bound_mean` | f64 | Statistical upper bound mean |
| `upper_bound_std` | f64 | Upper bound standard deviation |
| `gap` | f64 | Absolute gap |
| `relative_gap` | f64 | Relative gap (%) |
| `cuts_added` | i32 | Cuts added this iteration |
| `cuts_removed` | i32 | Cuts removed (cut selection) |
| `cuts_returned` | i32 | Cuts returned to model |
| `active_cuts` | i64 | Total active cuts |
| `time_forward_ms` | i64 | Forward pass time (ms) |
| `time_backward_ms` | i64 | Backward pass time (ms) |
| `time_communication_ms` | i64 | MPI communication time |
| `time_total_ms` | i64 | Total iteration time |
| `memory_peak_mb` | i64 | Peak memory usage |

### 4.3 Simulation Summary (`simulation/summary.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `scenario_id` | i32 | Scenario index |
| `total_cost` | f64 | Total scenario cost |
| `total_deficit_mwh` | f64 | Total unmet demand |
| `total_spillage_hm3` | f64 | Total spillage |
| `total_thermal_mwh` | f64 | Total thermal generation |
| `final_storage_total_hm3` | f64 | Final total storage |

### 4.4 Simulation Hydro Detail (`simulation/operational/hydro.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `scenario_id` | i32 | Scenario index |
| `stage_id` | i32 | Stage index |
| `block_id` | i32 | Block index |
| `hydro_id` | i32 | Hydro plant ID |
| `turbined_m3s` | f64 | Turbined flow |
| `spillage_m3s` | f64 | Spillage |
| `inflow_m3s` | f64 | Realized inflow |
| `storage_initial_hm3` | f64 | Storage at block start |
| `storage_final_hm3` | f64 | Storage at block end |
| `generation_mw` | f64 | Power generation |
| `water_value` | f64 | Marginal water value |

### 4.5 Simulation Thermal Detail (`simulation/operational/thermal.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `scenario_id` | i32 | Scenario index |
| `stage_id` | i32 | Stage index |
| `block_id` | i32 | Block index |
| `thermal_id` | i32 | Thermal unit ID |
| `generation_mw` | f64 | Power generation |
| `generation_cost` | f64 | Generation cost |
| `segment_id` | i32 | Cost segment used |

### 4.6 Simulation Exchange Detail (`simulation/operational/exchange.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `scenario_id` | i32 | Scenario index |
| `stage_id` | i32 | Stage index |
| `block_id` | i32 | Block index |
| `line_id` | i32 | Transmission line ID |
| `flow_direct_mw` | f64 | Flow in direct direction |
| `flow_reverse_mw` | f64 | Flow in reverse direction |
| `net_flow_mw` | f64 | Net flow (direct - reverse) |

### 4.7 Cuts Output (`cuts/stage_XXX.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `cut_id` | i64 | Unique cut identifier |
| `iteration` | i32 | Generation iteration |
| `forward_pass_idx` | i32 | Forward pass index |
| `rhs` | f64 | Cut RHS |
| `is_active` | bool | Active in final model |
| `domination_count` | i32 | Number of dominated states |
| `coefficients` | binary | State coefficients (packed f64[]) |

*Note: Coefficients as binary blob to handle variable state dimensions. Alternative: explode into columns if dimension is fixed.*

---

## 5. Internal Data Structures

### 5.1 Core Algorithm Structures

> **Order Invariance**: All entity vectors must be sorted by ID after loading (see Section 1.3). The `canonicalize()` method must be called before any processing.

```rust
/// Production-scale system representation
/// 
/// # Order Invariance
/// After loading from input files, `canonicalize()` MUST be called to sort
/// all entity collections by ID. This ensures deterministic behavior regardless
/// of declaration order in input files.
pub struct System {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub thermals: Vec<Thermal>,
    pub hydros: Vec<Hydro>,
    pub meta: SystemMeta,
}

impl System {
    /// Sort all entity collections by ID for order-invariant processing.
    /// MUST be called after loading and before any algorithm execution.
    pub fn canonicalize(&mut self) {
        self.buses.sort_by_key(|b| b.id);
        self.lines.sort_by_key(|l| l.id);
        self.hydros.sort_by_key(|h| h.id);
        self.thermals.sort_by_key(|t| t.id);
    }
}

/// Operative state for entities (computed per stage)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OperativeState {
    NonExisting,    // Before entry or after exit - no LP variables
    Filling,        // Hydro only: reservoir filling, no generation
    Operating,      // Normal operation
    Decommissioned, // After exit - no LP variables
}

/// Hydro with full production features
pub struct Hydro {
    pub id: u32,
    pub name: String,
    pub bus_id: u32,
    pub downstream_id: Option<u32>,
    pub upstream_ids: Vec<u32>,
    
    // Entry/exit for system changes modeling
    pub entry_stage_id: Option<i32>,  // None = exists from start
    pub exit_stage_id: Option<i32>,   // None = exists until end
    
    // Dead-volume filling (for plants entering later)
    pub filling: Option<FillingConfig>,
    
    // Base reservoir bounds (can be overridden per stage)
    pub base_min_storage: f64,
    pub base_max_storage: f64,
    
    // Base outflow bounds (can be overridden per stage)
    pub base_min_outflow: f64,
    pub base_max_outflow: Option<f64>,  // None = unlimited
    
    // Generation (currently constant_productivity, extensible)
    pub generation: HydroGeneration,
}

impl Hydro {
    /// Compute operative state for a given stage
    pub fn operative_state(&self, stage_id: i32) -> OperativeState {
        let entry = self.entry_stage_id.unwrap_or(i32::MIN);
        let exit = self.exit_stage_id.unwrap_or(i32::MAX);
        let filling_start = self.filling.as_ref().map(|f| f.start_stage_id);
        
        if stage_id > exit {
            OperativeState::Decommissioned
        } else if stage_id >= entry {
            OperativeState::Operating
        } else if let Some(start) = filling_start {
            if stage_id >= start {
                OperativeState::Filling
            } else {
                OperativeState::NonExisting
            }
        } else {
            OperativeState::NonExisting
        }
    }
}

/// Dead-volume filling configuration
pub struct FillingConfig {
    pub start_stage_id: i32,
    pub target_storage_hm3: f64,
}

/// Penalty tables loaded from Parquet files
/// All penalties are explicitly declared per entity × stage
pub struct PenaltyTables {
    /// Bus penalties indexed by (bus_id, stage_id)
    pub bus_penalties: HashMap<(u32, i32), BusPenalties>,
    /// Hydro penalties indexed by (hydro_id, stage_id)
    pub hydro_penalties: HashMap<(u32, i32), HydroPenalties>,
}

/// Bus penalties (from bus_penalties.parquet)
#[derive(Clone, Copy)]
pub struct BusPenalties {
    pub deficit_cost: f64,    // $/MWh
    pub excess_cost: f64,     // $/MWh
}

/// Hydro penalties (from hydro_penalties.parquet)
#[derive(Clone, Copy)]
pub struct HydroPenalties {
    pub spillage_cost: f64,              // $/(m³/s·h) - opportunity cost
    pub turbined_violation_cost: f64,    // $/(m³/s·h) - min turbined violation
    pub outflow_violation_cost: f64,     // $/(m³/s·h) - outflow bounds violation
    pub generation_violation_cost: f64,  // $/MWh - min generation violation
}

impl PenaltyTables {
    /// Get bus penalties for a specific stage
    pub fn bus(&self, bus_id: u32, stage_id: i32) -> Option<&BusPenalties> {
        self.bus_penalties.get(&(bus_id, stage_id))
    }
    
    /// Get hydro penalties for a specific stage
    pub fn hydro(&self, hydro_id: u32, stage_id: i32) -> Option<&HydroPenalties> {
        self.hydro_penalties.get(&(hydro_id, stage_id))
    }
}

// ============================================================================
// Generic Constraints
// ============================================================================

/// Variable reference in a generic constraint expression
#[derive(Clone, Debug, PartialEq)]
pub enum VariableRef {
    HydroStorage { hydro_id: u32 },
    HydroTurbined { hydro_id: u32, block_id: Option<u32> },
    HydroSpillage { hydro_id: u32, block_id: Option<u32> },
    HydroOutflow { hydro_id: u32, block_id: Option<u32> },
    HydroGeneration { hydro_id: u32, block_id: Option<u32> },
    ThermalGeneration { thermal_id: u32, block_id: Option<u32> },
    LineDirect { line_id: u32, block_id: Option<u32> },
    LineReverse { line_id: u32, block_id: Option<u32> },
    BusDeficit { bus_id: u32, block_id: Option<u32> },
    BusExcess { bus_id: u32, block_id: Option<u32> },
}

/// A term in a linear expression: coefficient × variable
#[derive(Clone, Debug)]
pub struct LinearTerm {
    pub coefficient: f64,
    pub variable: VariableRef,
}

/// Parsed linear expression (sum of terms plus constant)
#[derive(Clone, Debug)]
pub struct LinearExpression {
    pub terms: Vec<LinearTerm>,
    pub constant: f64,
}

/// Constraint sense
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintSense {
    GreaterEqual,  // >=
    LessEqual,     // <=
    Equal,         // ==
}

/// Slack configuration for a constraint
#[derive(Clone, Debug)]
pub struct SlackConfig {
    pub enabled: bool,
    pub penalty: f64,  // Only used if enabled
}

/// A generic constraint definition (parsed from JSON)
#[derive(Clone, Debug)]
pub struct GenericConstraint {
    pub id: u32,
    pub name: String,
    pub description: Option<String>,
    pub expression: LinearExpression,
    pub sense: ConstraintSense,
    pub slack: SlackConfig,
}

/// Bound for a generic constraint at a specific stage/block
#[derive(Clone, Copy, Debug)]
pub struct ConstraintBound {
    pub constraint_id: u32,
    pub stage_id: i32,
    pub block_id: Option<u32>,  // None = applies to all blocks
    pub bound: f64,
}

/// Collection of all generic constraints
pub struct GenericConstraints {
    pub constraints: Vec<GenericConstraint>,
    /// Bounds indexed by (constraint_id, stage_id, block_id)
    /// block_id = u32::MAX means "all blocks"
    pub bounds: HashMap<(u32, i32, u32), f64>,
}

impl GenericConstraints {
    /// Get bound for a constraint at a specific stage and block
    pub fn get_bound(&self, constraint_id: u32, stage_id: i32, block_id: u32) -> Option<f64> {
        // Try specific block first
        self.bounds.get(&(constraint_id, stage_id, block_id))
            .or_else(|| self.bounds.get(&(constraint_id, stage_id, u32::MAX)))
            .copied()
    }
    
    /// Validate all entity references exist in the system
    pub fn validate_references(&self, system: &System) -> Result<(), String> {
        for constraint in &self.constraints {
            for term in &constraint.expression.terms {
                match &term.variable {
                    VariableRef::HydroStorage { hydro_id } |
                    VariableRef::HydroTurbined { hydro_id, .. } |
                    VariableRef::HydroSpillage { hydro_id, .. } |
                    VariableRef::HydroOutflow { hydro_id, .. } |
                    VariableRef::HydroGeneration { hydro_id, .. } => {
                        if !system.hydros.iter().any(|h| h.id == *hydro_id) {
                            return Err(format!(
                                "Constraint '{}': hydro {} not found",
                                constraint.name, hydro_id
                            ));
                        }
                    }
                    VariableRef::ThermalGeneration { thermal_id, .. } => {
                        if !system.thermals.iter().any(|t| t.id == *thermal_id) {
                            return Err(format!(
                                "Constraint '{}': thermal {} not found",
                                constraint.name, thermal_id
                            ));
                        }
                    }
                    VariableRef::LineDirect { line_id, .. } |
                    VariableRef::LineReverse { line_id, .. } => {
                        if !system.lines.iter().any(|l| l.id == *line_id) {
                            return Err(format!(
                                "Constraint '{}': line {} not found",
                                constraint.name, line_id
                            ));
                        }
                    }
                    VariableRef::BusDeficit { bus_id, .. } |
                    VariableRef::BusExcess { bus_id, .. } => {
                        if !system.buses.iter().any(|b| b.id == *bus_id) {
                            return Err(format!(
                                "Constraint '{}': bus {} not found",
                                constraint.name, bus_id
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Thermal with entry/exit support
pub struct Thermal {
    pub id: u32,
    pub name: String,
    pub bus_id: u32,
    
    // Entry/exit for system changes modeling
    pub entry_stage_id: Option<i32>,
    pub exit_stage_id: Option<i32>,
    
    // Cost segments
    pub cost_segments: Vec<CostSegment>,
    
    // Base generation bounds (can be overridden per stage)
    pub base_min_generation: f64,
    pub base_max_generation: f64,
}

impl Thermal {
    /// Compute operative state for a given stage
    pub fn operative_state(&self, stage_id: i32) -> OperativeState {
        let entry = self.entry_stage_id.unwrap_or(i32::MIN);
        let exit = self.exit_stage_id.unwrap_or(i32::MAX);
        
        if stage_id > exit {
            OperativeState::Decommissioned
        } else if stage_id >= entry {
            OperativeState::Operating
        } else {
            OperativeState::NonExisting
        }
    }
}

/// Transmission line with entry/exit support
pub struct Line {
    pub id: u32,
    pub name: String,
    pub source_bus_id: u32,
    pub target_bus_id: u32,
    
    // Entry/exit for transmission expansion
    pub entry_stage_id: Option<i32>,
    pub exit_stage_id: Option<i32>,
    
    // Base capacity (can be overridden per stage and block via factors)
    pub base_direct_mw: f64,
    pub base_reverse_mw: f64,
    pub exchange_penalty: f64,
    pub losses_percent: f64,
}

impl Line {
    /// Compute operative state for a given stage
    pub fn operative_state(&self, stage_id: i32) -> OperativeState {
        let entry = self.entry_stage_id.unwrap_or(i32::MIN);
        let exit = self.exit_stage_id.unwrap_or(i32::MAX);
        
        if stage_id > exit {
            OperativeState::Decommissioned
        } else if stage_id >= entry {
            OperativeState::Operating
        } else {
            OperativeState::NonExisting
        }
    }
}

/// Generation modeling for hydro plants (extensible)
pub enum HydroGeneration {
    ConstantProductivity {
        productivity: f64,
        base_min_turbined: f64,  // Can be overridden per stage
        base_max_turbined: f64,  // Can be overridden per stage
        base_min_generation: Option<f64>,  // Derived from turbined if None
        base_max_generation: Option<f64>,  // Derived from turbined if None
    },
    // Future: HeightVolumeTable, TailwaterCurve, EfficiencyCurve, etc.
}

/// Stage with blocks (study stages have id >= 0, pre-study have id < 0)
pub struct Stage {
    pub id: i32,  // Negative for pre-study stages
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub blocks: Vec<Block>,  // Empty for pre-study stages
    pub risk_measure: RiskMeasure,
    pub state_config: StateConfig,
    pub num_scenarios: u32,
}

/// Block within a stage
pub struct Block {
    pub id: u32,
    pub name: String,
    pub hours: f64,
    // weight is computed as hours / sum(all block hours in stage)
}

/// Per-stage inflow model parameters
pub struct InflowModel {
    pub hydro_id: u32,
    pub stage_id: i32,
    pub mean: f64,
    pub std: f64,
    pub ar_order: u32,
    pub ar_coefficients: [f64; 6],  // Fixed size, unused slots = 0
}

/// Per-stage load model parameters  
pub struct LoadModel {
    pub bus_id: u32,
    pub stage_id: i32,
    pub mean: f64,
    pub std: f64,
}

/// Per-stage bounds override
pub struct StageBounds {
    pub entity_id: u32,
    pub stage_id: i32,
    pub min_bound: Option<f64>,
    pub max_bound: Option<f64>,
}
```

### 5.2 LP Subproblem Structure

```rust
/// Subproblem for a (stage, block) pair
pub struct BlockSubproblem {
    pub stage_id: u32,
    pub block_id: u32,
    
    // Solver interface
    pub solver: Box<dyn LpSolver>,
    
    // Variable indices (for solution extraction)
    pub variables: VariableLayout,
    
    // Constraint indices (for dual extraction)
    pub constraints: ConstraintLayout,
    
    // State management
    pub state: Box<dyn State>,
    
    // Precomputed data for hot path
    pub uncertainty_data: Vec<UncertaintyObservationData>,
    pub cut_constraint_slots: Vec<usize>,
}

/// Variable layout for O(1) extraction
pub struct VariableLayout {
    pub deficit: Range<usize>,
    pub direct_exchange: Option<Range<usize>>,
    pub reverse_exchange: Option<Range<usize>>,
    pub thermal_gen: Range<usize>,
    pub thermal_gen_segments: Vec<Range<usize>>,
    pub turbined_flow: Range<usize>,
    pub spillage: Range<usize>,
    pub storage_final: Range<usize>,
    pub storage_inter_block: Option<Range<usize>>,
    pub alpha: usize,  // Future cost variable
}
```

### 5.3 FCF with Replication

```rust
/// Future Cost Function (replicated per MPI rank)
pub struct FutureCostFunction {
    /// Cut pool for this stage
    pub cuts: CutPool,
    
    /// Visited states for this stage
    pub states: StatePool,
    
    /// Stage metadata
    pub stage_id: u32,
    
    /// Current iteration watermark (for incremental sync)
    pub sync_iteration: u32,
}

/// Cut pool with preallocation
pub struct CutPool {
    /// Preallocated cuts (capacity = iterations × forward_passes)
    pub pool: Vec<BendersCut>,
    
    /// Number of populated cuts
    pub populated_count: usize,
    
    /// Active cut bitmap for O(1) lookup
    pub active_bitmap: BitVec,
    
    /// State dimension (for validation)
    pub state_dimension: u32,
}

/// Single Benders cut
#[repr(C, align(64))]  // Cache-line aligned
pub struct BendersCut {
    pub id: u64,
    pub rhs: f64,
    pub iteration: u32,
    pub forward_pass_idx: u32,
    pub is_active: AtomicBool,
    pub domination_count: AtomicU32,
    pub slot_index: AtomicU32,
    _padding: [u8; 4],
    /// Coefficients stored separately for SIMD alignment
    pub coefficients_offset: usize,
}
```

---

## 6. MPI Communication Structures

### 6.1 Communication Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MPI DISTRIBUTION PATTERN                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FORWARD PASS (Parallel across ranks and threads)                   │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│  Rank 0 (Master)          Rank 1           ...      Rank N-1       │
│  ┌─────────────┐     ┌─────────────┐          ┌─────────────┐      │
│  │ FP 0..K-1   │     │ FP K..2K-1  │          │ FP ...      │      │
│  │ (threads)   │     │ (threads)   │          │ (threads)   │      │
│  └─────────────┘     └─────────────┘          └─────────────┘      │
│         │                   │                        │              │
│         └───────────────────┴────────────────────────┘              │
│                             │                                       │
│                    [No sync needed]                                 │
│                                                                     │
│  BACKWARD PASS (Per stage, synchronized)                            │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│  For each stage (reverse order):                                    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Phase 1: Parallel Cut Computation                            │   │
│  │                                                               │   │
│  │   Each rank computes cuts for its forward passes              │   │
│  │   Threads within rank work in parallel                        │   │
│  │                                                               │   │
│  │   ┌──────────┐    ┌──────────┐        ┌──────────┐           │   │
│  │   │ Thread 0 │    │ Thread 1 │  ...   │Thread T-1│           │   │
│  │   │ CutData  │    │ CutData  │        │ CutData  │           │   │
│  │   └────┬─────┘    └────┬─────┘        └────┬─────┘           │   │
│  │        └───────────────┼───────────────────┘                 │   │
│  │                        ▼                                     │   │
│  │               [Thread reduction to rank buffer]              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Phase 2: MPI Gather to Master                                │   │
│  │                                                               │   │
│  │   Rank 1..N-1 ───► MPI_Gather ───► Rank 0                    │   │
│  │   [CutData[]]                      [All CutData[]]           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Phase 3: Master Aggregation (Rank 0 only)                    │   │
│  │                                                               │   │
│  │   1. Sort cuts by (iteration, forward_pass_idx)              │   │
│  │   2. Evaluate cut domination                                 │   │
│  │   3. Identify cuts to add/remove/return                      │   │
│  │   4. Prepare broadcast message                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Phase 4: MPI Broadcast FCF Updates                           │   │
│  │                                                               │   │
│  │   Rank 0 ───► MPI_Bcast ───► Rank 1..N-1                     │   │
│  │   [FCFUpdateMsg]                                              │   │
│  │                                                               │   │
│  │   Each rank applies updates to local FCF replica             │   │
│  │   Each rank updates local solver models                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Phase 5: Barrier Synchronization                             │   │
│  │                                                               │   │
│  │   MPI_Barrier (ensures all ranks have updated FCF)           │   │
│  │   Proceed to next stage                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Message Structures

```rust
/// Cut data for MPI transmission
#[repr(C)]
pub struct CutMessage {
    pub stage_id: u32,
    pub iteration: u32,
    pub forward_pass_idx: u32,
    pub rank_id: u32,
    pub rhs: f64,
    pub state_dimension: u32,
    _padding: u32,
    // Followed by: coefficients[state_dimension]
    // Followed by: state_coefficients[state_dimension]
}

impl CutMessage {
    /// Size in bytes for a given state dimension
    pub fn size_bytes(state_dim: usize) -> usize {
        std::mem::size_of::<Self>() + 2 * state_dim * std::mem::size_of::<f64>()
    }
}

/// FCF update message from master
#[repr(C)]
pub struct FcfUpdateMessage {
    pub stage_id: u32,
    pub iteration: u32,
    pub num_new_cuts: u32,
    pub num_removed_cuts: u32,
    pub num_returned_cuts: u32,
    _padding: u32,
    // Followed by: new_cut_data (variable size)
    // Followed by: removed_cut_ids[num_removed_cuts]
    // Followed by: returned_cut_ids[num_returned_cuts]
}

/// Persistent communication handles (MPI 4.0+)
pub struct PersistentComm {
    /// Gather: all ranks → master (cut data)
    pub cut_gather: PersistentRequest,
    
    /// Broadcast: master → all ranks (FCF updates)
    pub fcf_broadcast: PersistentRequest,
    
    /// Allreduce: lower bound computation
    pub bound_allreduce: PersistentRequest,
    
    /// Preallocated buffers
    pub cut_send_buffer: Vec<u8>,
    pub cut_recv_buffer: Vec<u8>,
    pub fcf_buffer: Vec<u8>,
}
```

### 6.3 Synchronization Points

| Phase | Sync Type | Data Size | Frequency |
|-------|-----------|-----------|-----------|
| Forward pass end | None | - | Per iteration |
| Backward per-stage | MPI_Gather | ~50KB × ranks | Per stage |
| FCF update | MPI_Bcast | ~100KB | Per stage |
| Lower bound | MPI_Allreduce | 8 bytes | Per iteration |
| Checkpointing | MPI_Barrier | - | Every N iterations |

---

## 7. File Format Decisions

### 7.1 Summary Table

| Data Category | Read/Write | Format | Rationale |
|---------------|------------|--------|-----------|
| Algorithm Config | Read | JSON | Small, editable |
| System Registry | Read | JSON | Structured objects |
| Stage/Block Def | Read | JSON | Graph structure |
| Uncertainty Models | Read | JSON | Complex nested |
| Distributions | Read | JSON | Parameters |
| Load Profiles | Read | Parquet | Large, indexed |
| Inflow History | Read | Parquet | Time series |
| Warm-start Cuts | Read/Write | Parquet | Large, columnar |
| Warm-start States | Read/Write | Parquet | Large, columnar |
| Training Results | Write | Parquet | Analytics-ready |
| Simulation Detail | Write | Parquet | Large volume |
| Cuts Output | Write | Parquet | Large volume |
| Dictionaries | Write | CSV | Human-readable |

### 7.2 Parquet Configuration

```rust
/// Parquet writer settings for output
pub struct ParquetConfig {
    /// Compression algorithm
    pub compression: Compression::ZSTD(ZstdLevel::try_new(3).unwrap()),
    
    /// Row group size (for parallel reading)
    pub row_group_size: 100_000,
    
    /// Enable statistics
    pub enable_statistics: true,
    
    /// Dictionary encoding threshold
    pub dictionary_enabled: true,
    pub dictionary_page_size_limit: 1_048_576,
}
```

---

## 8. Validation Requirements

### 8.1 Input Validation Phases

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT VALIDATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 0: Canonicalization (Order Invariance)                       │
│  ─────────────────────────────────────────────────────────          │
│  • Sort all entity collections by ID                                │
│  • Sort stages by ID                                                │
│  • Sort blocks within stages by ID                                  │
│  • Sort generic constraints by ID                                   │
│  • Verify IDs are unique within each collection                     │
│  • (See Section 1.3 for full order-invariance requirements)         │
│                                                                     │
│  Phase 1: Schema Validation                                         │
│  ─────────────────────────────────────────────────────────          │
│  • JSON Schema validation (config, system, temporal)                │
│  • Parquet schema validation (timeseries, warmstart)                │
│  • Required field presence                                          │
│  • Type correctness                                                 │
│                                                                     │
│  Phase 2: Reference Integrity                                       │
│  ─────────────────────────────────────────────────────────          │
│  • Bus IDs exist for lines, hydros, thermals                        │
│  • Downstream hydro IDs exist (cascade)                             │
│  • Model IDs in uncertainty_models exist                            │
│  • Stage IDs in transitions exist                                   │
│  • Season IDs match distribution definitions                        │
│                                                                     │
│  Phase 3: Business Rules                                            │
│  ─────────────────────────────────────────────────────────          │
│  • Hydro cascade is acyclic                                         │
│  • min ≤ initial ≤ max (storage, generation)                        │
│  • Probabilities sum to 1.0                                         │
│  • Correlation matrices are positive semi-definite                  │
│  • AR coefficients ensure stationarity                              │
│  • Block weights sum to 1.0                                         │
│  • Deficit segments are monotonically increasing                    │
│                                                                     │
│  Phase 4: Dimension Consistency                                     │
│  ─────────────────────────────────────────────────────────          │
│  • Load profiles cover all (stage, block, bus) combinations         │
│  • Inflow history covers all hydros with PAR models                 │
│  • Seasonal parameters have correct length (num_seasons)            │
│  • Correlation matrix dimensions match entity count                 │
│                                                                     │
│  Phase 5: Warm-start Compatibility                                  │
│  ─────────────────────────────────────────────────────────          │
│  • State dimension matches current system                           │
│  • Cut stage IDs exist in current stage graph                       │
│  • Config hash matches (optional strict mode)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Validation Error Types

```rust
#[derive(Error, Debug)]
pub enum ValidationError {
    // Schema errors
    #[error("JSON schema validation failed for {file}: {details}")]
    JsonSchema { file: String, details: String },
    
    #[error("Parquet schema mismatch in {file}: expected {expected}, got {actual}")]
    ParquetSchema { file: String, expected: String, actual: String },
    
    // Reference errors
    #[error("{entity_type} {entity_id} references non-existent {ref_type} {ref_id}")]
    BrokenReference {
        entity_type: String,
        entity_id: u32,
        ref_type: String,
        ref_id: u32,
    },
    
    // Business rule errors
    #[error("Hydro cascade contains cycle: {cycle:?}")]
    CyclicCascade { cycle: Vec<u32> },
    
    #[error("Value out of range: {field} = {value}, expected [{min}, {max}]")]
    OutOfRange { field: String, value: f64, min: f64, max: f64 },
    
    #[error("Correlation matrix is not positive semi-definite for block {block}")]
    NotPositiveSemiDefinite { block: String },
    
    // Dimension errors
    #[error("Missing data for ({stage}, {block}, {entity}): expected {expected} rows")]
    MissingTimeSeries {
        stage: u32,
        block: u32,
        entity: String,
        expected: usize,
    },
    
    // Warm-start errors
    #[error("Warm-start state dimension mismatch: expected {expected}, got {actual}")]
    WarmstartDimensionMismatch { expected: u32, actual: u32 },
}
```

---

## 9. Migration Path

### 9.1 From v1.x to v2.0

```
v1.x Input Files              v2.0 Input Structure
─────────────────             ────────────────────

config.json          ──────►  config.json (restructured)

system.json          ──────►  system/
                              ├── topology.json
                              ├── hydros.json
                              └── thermals.json

graph.json           ──────►  temporal/stages.json

recourse.json        ──────►  temporal/
                              ├── initial_conditions.json
                              └── ...
                              timeseries/
                              ├── inflow_models.parquet
                              ├── load_models.parquet
                              └── inflow_history.parquet
                              scenarios/
                              ├── correlation.json
                              └── load_factors.json
```

### 9.2 Migration Tool

```bash
# Convert v1.x case to v2.0 structure
powers migrate --from v1 --to v2 ./old_case ./new_case

# Validate converted case
powers validate ./new_case

# Run with compatibility mode (reads both formats)
powers run --compat-v1 ./old_case
```

---

## Next Steps

1. **Review this specification** - Confirm structure, formats, field names
2. **Define block semantics** - How blocks affect LP construction
3. **Solver trait specification** - Define the solver abstraction interface
4. **MPI protocol detail** - Message formats, buffer sizing, error handling
5. **Implementation plan** - Phased approach to refactoring

---

*This specification is a living document. Update as decisions are finalized.*
