---
status: draft
review_priority: 3-medium
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.1 (Core Algorithm Structures)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §5.1"
---

# Internal Structures

## Purpose

This spec defines the core Rust structs used by the SDDP algorithm at runtime. These are the in-memory representations loaded from input files and used throughout training and simulation. They are distinct from binary format choices (see [Binary Formats](binary-formats.md)) and from input file schemas (see [Input System Entities](input-system-entities.md)).

## 1. System Representation

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
```

## 2. Operative State

```rust
/// Operative state for entities (computed per stage)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OperativeState {
    NonExisting,    // Before entry or after exit — no LP variables
    Filling,        // Hydro only: reservoir filling, no generation
    Operating,      // Normal operation
    Decommissioned, // After exit — no LP variables
}
```

## 3. Hydro Plant

```rust
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

    // Diversion channel (optional)
    pub diversion: Option<DiversionConfig>,

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

/// Diversion channel configuration
pub struct DiversionConfig {
    pub downstream_id: u32,      // Hydro receiving diverted water
    pub max_flow_m3s: f64,       // Maximum diversion flow
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
```

## 4. Thermal Plant

```rust
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
```

## 5. Transmission Line

```rust
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
```

## 6. Penalty Tables

```rust
/// Penalty tables loaded from JSON/Parquet files
pub struct PenaltyTables {
    /// Bus penalties indexed by (bus_id, stage_id)
    pub bus_penalties: HashMap<(u32, i32), BusPenalties>,
    /// Hydro penalties indexed by (hydro_id, stage_id)
    pub hydro_penalties: HashMap<(u32, i32), HydroPenalties>,
}

#[derive(Clone, Copy)]
pub struct BusPenalties {
    pub deficit_cost: f64,    // $/MWh
    pub excess_cost: f64,     // $/MWh
}

#[derive(Clone, Copy)]
pub struct HydroPenalties {
    pub spillage_cost: f64,              // $/(m³/s·h)
    pub diversion_cost: f64,             // $/(m³/s·h)
    pub turbined_violation_cost: f64,    // $/(m³/s·h)
    pub outflow_violation_cost: f64,     // $/(m³/s·h)
    pub generation_violation_cost: f64,  // $/MWh
}

impl PenaltyTables {
    pub fn bus(&self, bus_id: u32, stage_id: i32) -> Option<&BusPenalties> {
        self.bus_penalties.get(&(bus_id, stage_id))
    }

    pub fn hydro(&self, hydro_id: u32, stage_id: i32) -> Option<&HydroPenalties> {
        self.hydro_penalties.get(&(hydro_id, stage_id))
    }
}
```

## 7. Generic Constraints

```rust
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
pub struct LinearTerm {
    pub coefficient: f64,
    pub variable: VariableRef,
}

/// Parsed linear expression (sum of terms plus constant)
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
pub struct SlackConfig {
    pub enabled: bool,
    pub penalty: f64,
}

/// A generic constraint definition (parsed from JSON)
pub struct GenericConstraint {
    pub id: u32,
    pub name: String,
    pub description: Option<String>,
    pub expression: LinearExpression,
    pub sense: ConstraintSense,
    pub slack: SlackConfig,
}

/// Bound for a generic constraint at a specific stage/block
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
```

## 8. Stage and Block Definitions

```rust
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
```

## 9. Inflow and Load Models

```rust
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

## Cross-References

- [Binary Formats](binary-formats.md) — Format decisions, FlatBuffers schema, LP subproblem and FCF structs
- [Input System Entities](input-system-entities.md) — JSON schemas that map to these structs
- [Input Hydro Extensions](input-hydro-extensions.md) — Parquet data loaded into hydro structs
- [Input Constraints](input-constraints.md) — Generic constraint JSON parsed into `GenericConstraints`
- [Input Scenarios](input-scenarios.md) — Stage/block, inflow, and load data
- [Penalty System](penalty-system.md) — Penalty resolution loaded into `PenaltyTables`
- [Design Principles](../00-overview/design-principles.md) — Order invariance requirement (§1.3)
