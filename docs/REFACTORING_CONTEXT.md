# POWE.RS SDDP Solver: Complete Architectural Context for Refactoring

> **Document Purpose**: Comprehensive context document for the major refactoring of POWE.RS, transitioning from rayon-based parallelism to MPI-based distributed computing with solver abstraction.
>
> **Last Updated**: 2026-01-15
> **Version**: 1.0.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Overview](#2-current-architecture-overview)
3. [Input Data Model](#3-input-data-model)
4. [Output Data Model](#4-output-data-model)
5. [Core Algorithm Structure](#5-core-algorithm-structure)
6. [Solver Integration](#6-solver-integration)
7. [Parallelization Architecture](#7-parallelization-architecture)
8. [Data Structures](#8-data-structures)
9. [Performance Characteristics](#9-performance-characteristics)
10. [Known Limitations & Technical Debt](#10-known-limitations--technical-debt)
11. [External Dependencies](#11-external-dependencies)
12. [Refactoring Goals](#12-refactoring-goals)

---

## 1. Executive Summary

### What is POWE.RS?

POWE.RS is a high-performance implementation of the **Stochastic Dual Dynamic Programming (SDDP)** algorithm in Rust, specifically designed for the **hydrothermal dispatch problem**. It solves multistage stochastic optimization problems for power systems with:

- Hydroelectric plants with reservoirs (state variables)
- Thermal generators (dispatch decisions)
- Transmission networks (bus topology with exchanges)
- Stochastic inflows and loads (uncertainty modeled via PAR processes)

### Current Capabilities

| Capability | Status | Description |
|-----------|--------|-------------|
| **SDDP Training** | ✅ Complete | Forward/backward passes with Benders cuts |
| **Cut Selection** | ✅ Complete | Domination-based cut management |
| **PAR Models** | ✅ Complete | Periodic Autoregressive inflow/load models |
| **Risk Measures** | ✅ Partial | Expectation (CVaR, worst-case: placeholders) |
| **Correlation** | ✅ Complete | Gaussian copula with Cholesky decomposition |
| **Simulation** | ✅ Complete | Out-of-sample policy evaluation |
| **Output Formats** | ✅ Complete | CSV and Parquet |
| **Parallelism** | ⚠️ Rayon-only | Thread-level parallelism (not distributed) |

### Why Refactor?

1. **Scalability**: Current rayon-based parallelism cannot scale beyond a single machine
2. **Solver Lock-in**: Direct HiGHS bindings preclude using other solvers (CPLEX, Gurobi, Xpress)
3. **Input/Output Rigidity**: JSON schema designed for small problems, not production scale
4. **Memory Efficiency**: Current design has optimization opportunities for large-scale problems

---

## 2. Current Architecture Overview

### Module Structure

```
src/
├── main.rs                 # CLI entry point
├── lib.rs                  # Public API (run function)
├── cli.rs                  # Clap-based CLI definitions
│
├── input.rs                # JSON deserialization (~60KB)
├── input_validation.rs     # Input validation logic
├── output/                 # Output generation
│   ├── mod.rs              # Main output coordinator
│   ├── csv/                # CSV writers
│   ├── parquet/            # Parquet writers
│   ├── dictionary.rs       # Variable dictionaries
│   └── writer.rs           # Output trait
│
├── sddp/                   # SDDP algorithm core
│   ├── mod.rs              # SddpInstance, TrainingResult (~124KB)
│   ├── builder.rs          # SDDP configuration builder
│   └── instance.rs         # Algorithm instance
│
├── algorithm/              # Algorithm phases
│   ├── forward_pass.rs     # Forward simulation
│   ├── backward_pass.rs    # Cut generation
│   ├── context.rs          # Phase contexts
│   ├── coordinator.rs      # Parallel handler coordination
│   ├── processor.rs        # Backward stage processor trait
│   └── cut_computation.rs  # Benders cut calculation
│
├── subproblem.rs           # LP subproblem formulation (~236KB)
├── solver.rs               # HiGHS bindings (~59KB)
├── model/                  # LP model components
│   ├── variable_indices.rs # Variable index ranges
│   ├── constraint_indices.rs # Constraint index ranges
│   └── constraints/        # Constraint generators
│
├── graph.rs                # Directed graph for scenario tree
├── fcf.rs                  # Future Cost Function management
├── cut.rs                  # Benders cut structures
├── state.rs                # State representations (~104KB)
│
├── system.rs               # Power system topology
├── scenario.rs             # Scenario tree generation
├── scenario_generator.rs   # SAA generation
├── temporal_model.rs       # PAR model representation
├── correlation_applicator.rs # Gaussian copula
│
├── risk_measure.rs         # Risk measure interface
├── initial_condition.rs    # Initial state handling
├── error.rs                # Error types
├── display/                # Terminal output formatting
├── timing/                 # Performance timing infrastructure
├── memory/                 # Memory management utilities
└── utils/                  # Utilities (dot product, etc.)
```

### Dependency Graph (Simplified)

```
                    ┌──────────────┐
                    │   main.rs    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   lib.rs     │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
│   input.rs    │  │  sddp/mod.rs  │  │  output/mod.rs│
└───────────────┘  └───────┬───────┘  └───────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │ forward_pass│  │backward_pass│  │   fcf.rs    │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │subproblem.rs│
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │  solver.rs  │  │  state.rs   │  │  system.rs  │
   └─────────────┘  └─────────────┘  └─────────────┘
```

---

## 3. Input Data Model

### Current File Structure

```
input_directory/
├── config.json     # Algorithm configuration
├── system.json     # Power system topology
├── graph.json      # Scenario tree structure
└── recourse.json   # Uncertainty specification
```

### 3.1 Config Schema (`config.json`)

```json
{
  "general": {
    "seed": 42,                    // u64: Random seed
    "num_threads": null            // Optional<usize>: Thread count
  },
  "training": {
    "num_iterations": 32,          // usize: SDDP iterations
    "num_forward_passes": 4,       // usize: Forward passes per iteration
    "enable_cut_selection": true   // bool: Cut management
  },
  "simulation": {
    "num_scenarios": 128           // Optional<usize>: Out-of-sample scenarios
  },
  "output": {
    "path": "./output",
    "format": "CSV|PARQUET|Auto",
    "export_training": true,
    "export_cuts": true,
    "export_states": true,
    "export_simulation": true,
    "export_forward_detail": false,
    "export_backward_detail": false,
    "export_training_noises": false
  },
  "display": { /* terminal output config */ }
}
```

### 3.2 System Schema (`system.json`)

```json
{
  "buses": [
    { "id": 0, "deficit_cost": 5000.0 }
  ],
  "lines": [
    {
      "id": 0,
      "source_bus_id": 0,
      "target_bus_id": 1,
      "direct_capacity": 100.0,
      "reverse_capacity": 100.0,
      "exchange_penalty": 0.01
    }
  ],
  "thermals": [
    {
      "id": 0,
      "bus_id": 0,
      "cost": 50.0,
      "min_generation": 0.0,
      "max_generation": 100.0
    }
  ],
  "hydros": [
    {
      "id": 0,
      "downstream_hydro_id": null,  // Optional cascade
      "bus_id": 0,
      "productivity": 0.5,          // MW per unit flow
      "min_storage": 0.0,
      "max_storage": 100.0,
      "min_turbined_flow": 0.0,
      "max_turbined_flow": 60.0,
      "spillage_penalty": 0.01
    }
  ]
}
```

### 3.3 Graph Schema (`graph.json`)

```json
{
  "nodes": [
    {
      "id": 0,
      "stage_id": 1,
      "season_id": 0,                    // For PAR model lookup
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-02-01T00:00:00Z",
      "risk_measure": "expectation",     // "expectation"|"cvar"|"worstcase"
      "state_variables": "storage",      // "storage"|"storage_and_inflow"
      "num_scenarios": 10                // Branching count
    }
  ],
  "edges": [
    {
      "source_id": 0,
      "target_id": 1,
      "probability": 1.0,
      "discount_rate": 0.0
    }
  ]
}
```

### 3.4 Recourse Schema (`recourse.json`)

```json
{
  "initial_condition": {
    "storage": [
      { "hydro_id": 0, "value": 30.0 }
    ],
    "inflow": [
      { "hydro_id": 0, "lag": 1, "value": 60.0 }
    ]
  },
  "uncertainty_specifications": [
    {
      "uncertainty_type": "inflow",       // "inflow"|"load"
      "entity_id": 0,                     // hydro_id or bus_id
      "temporal_model": {
        "num_seasons": 12,
        "ar_orders": [1, 1, ...],         // PAR order per season
        "ar_coefficients": [[0.7], ...],  // φ coefficients
        "seasonal_means": [60.0, ...],    // μ_s
        "seasonal_stds": [5.0, ...]       // σ_s
      },
      "seasonal_distributions": [
        {
          "season_id": 0,
          "type": "lognormal3",           // Distribution type
          "gamma": 0.001,
          "mu": 0.0,
          "sigma": 0.5
        }
      ]
    }
  ],
  "correlation": {
    "method": "cholesky",
    "blocks": [
      {
        "name": "cascade_correlation",
        "entities": [
          { "uncertainty_type": "inflow", "entity_id": 0 },
          { "uncertainty_type": "inflow", "entity_id": 1 }
        ],
        "correlation_matrix": [[1.0, 0.8], [0.8, 1.0]]
      }
    ]
  }
}
```

### Key Data Structures from Input

```rust
// Power System (from system.json)
pub struct System {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub thermals: Vec<Thermal>,
    pub hydros: Vec<Hydro>,
    pub meta: SystemMetadata,
}

// Temporal Model (from recourse.json)
pub struct TemporalModel {
    pub entity_type: UncertaintyType,  // Load or Inflow
    pub entity_id: usize,
    pub num_seasons: usize,
    pub seasonal_means: Vec<f64>,
    pub seasonal_stds: Vec<f64>,
    pub ar_orders: Vec<usize>,
    pub ar_coefficients: Vec<Vec<f64>>,
    pub max_ar_order: usize,
    pub psi_coefficients: Vec<Vec<f64>>,      // Transformed AR
    pub deterministic_bases: Vec<f64>,        // Precomputed
}
```

---

## 4. Output Data Model

### Output Files Generated

| File | Contents | Format |
|------|----------|--------|
| `training.csv/parquet` | Iteration convergence metrics | Per-iteration |
| `cuts.csv/parquet` | Benders cuts with coefficients | Per-node, per-cut |
| `states.csv/parquet` | Visited states | Per-node, per-state |
| `simulation.csv/parquet` | Simulation trajectories | Per-scenario, per-stage |
| `variable_dictionary.csv` | Variable name mapping | Static |
| `coefficient_dictionary.csv` | Cut coefficient mapping | Static |
| `state_component_dictionary.csv` | State component mapping | Static |

### Training Output Schema

```
iteration,lower_bound,upper_bound_mean,upper_bound_std,gap,relative_gap,
num_cuts_added,num_cuts_removed,num_active_cuts,time_forward,time_backward,time_total
```

### Simulation Output Schema (Normalized)

```
stage,series,variable_index,entity_id,value
```

---

## 5. Core Algorithm Structure

### SDDP Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SDDP TRAINING LOOP                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  for iteration in 1..=num_iterations:                               │
│                                                                     │
│    ┌──────────────────────────────────────────────────────────┐    │
│    │              FORWARD PASS (Parallel)                      │    │
│    │                                                           │    │
│    │  for forward_pass in 0..num_forward_passes:               │    │
│    │    trajectory = []                                        │    │
│    │    for stage in stages:                                   │    │
│    │      1. Prepare subproblem from trajectory                │    │
│    │      2. Apply uncertainty realization (noises)            │    │
│    │      3. Solve LP subproblem                               │    │
│    │      4. Record realization to trajectory                  │    │
│    │    end                                                    │    │
│    │    record trajectory_cost                                 │    │
│    │  end                                                      │    │
│    └──────────────────────────────────────────────────────────┘    │
│                                                                     │
│    ┌──────────────────────────────────────────────────────────┐    │
│    │              BACKWARD PASS (3-Phase per Stage)            │    │
│    │                                                           │    │
│    │  for stage in reverse(stages):                            │    │
│    │    if first_stage:                                        │    │
│    │      compute lower_bound from branching                   │    │
│    │    else:                                                  │    │
│    │      ┌─────────────────────────────────────────────┐     │    │
│    │      │ Phase 1: Parallel cut computation           │     │    │
│    │      │   for each forward_pass:                    │     │    │
│    │      │     for each branching scenario:            │     │    │
│    │      │       solve LP with child FCF               │     │    │
│    │      │       extract dual values                   │     │    │
│    │      │     compute Benders cut                     │     │    │
│    │      │   end                                       │     │    │
│    │      └─────────────────────────────────────────────┘     │    │
│    │      ┌─────────────────────────────────────────────┐     │    │
│    │      │ Phase 2: Sequential cut selection           │     │    │
│    │      │   sort cuts by forward_pass_idx             │     │    │
│    │      │   evaluate domination                       │     │    │
│    │      │   identify cuts to add/remove/return        │     │    │
│    │      └─────────────────────────────────────────────┘     │    │
│    │      ┌─────────────────────────────────────────────┐     │    │
│    │      │ Phase 3: Parallel cut application           │     │    │
│    │      │   for each handler:                         │     │    │
│    │      │     add new cuts to LP model                │     │    │
│    │      │     remove dominated cuts                   │     │    │
│    │      │     return previously removed cuts          │     │    │
│    │      └─────────────────────────────────────────────┘     │    │
│    │  end                                                      │    │
│    └──────────────────────────────────────────────────────────┘    │
│                                                                     │
│    record iteration metrics                                         │
│    check convergence                                                │
│                                                                     │
│  end                                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Algorithm Components

#### Forward Pass Context

```rust
pub struct ForwardPassContext<'a> {
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,
    pub realization_graph: &'a mut DirectedGraph<Realization>,
    pub sampled_noises: &'a HashMap<usize, OptimizedSampledBranchingNoises>,
    pub study_period_ids: &'a [usize],
    pub graph_bfs_table: &'a Vec<Vec<usize>>,
}
```

#### Backward Pass Context

```rust
pub struct BackwardPassContext<'a> {
    pub subproblem_graph: &'a DirectedGraph<Subproblem>,
    pub realization_graph: &'a DirectedGraph<Realization>,
    pub training_noises: &'a HashMap<usize, OptimizedSampledBranchingNoises>,
    pub backward_bfs: &'a [usize],
    // ... more fields
}
```

#### Realization (LP Solution State)

```rust
pub struct Realization {
    pub node_id: usize,
    pub stage_id: usize,
    pub total_stage_objective: f64,
    pub current_stage_objective: f64,
    pub future_cost: f64,
    // Operational variables
    pub deficit: Vec<f64>,
    pub direct_exchange: Vec<f64>,
    pub reverse_exchange: Vec<f64>,
    pub thermal_gen: Vec<f64>,
    pub turbined_flow: Vec<f64>,
    pub spillage: Vec<f64>,
    pub final_storage: Vec<f64>,
    // Uncertainty observations
    pub load: Vec<f64>,
    pub inflow: Vec<f64>,
    // Dual variables
    pub marginal_cost: Vec<f64>,
    pub water_value: Vec<f64>,
    // AR state
    pub load_lag_duals: Option<Vec<Vec<f64>>>,
    pub inflow_lag_duals: Option<Vec<Vec<f64>>>,
}
```

---

## 6. Solver Integration

### Current Design: Direct HiGHS FFI

The current implementation uses `highs-sys` directly rather than the `highs` crate for:

1. **Zero-copy solution access**: Direct pointer access to solution vectors
2. **Basis warm-starting**: Full control over simplex basis management
3. **Incremental model updates**: Efficient row addition/deletion/bound changes
4. **Thread-local buffers**: Elimination of allocations in hot path

### Solver API (from `solver.rs`)

```rust
pub struct Model {
    highs: *mut c_void,  // HiGHS model pointer
}

impl Model {
    pub fn new() -> Self;
    pub fn from_problem(problem: &Problem) -> Self;
    
    // Solve operations
    pub fn solve(&mut self) -> HighsStatus;
    pub fn get_status(&self) -> HighsModelStatus;
    pub fn get_solution_into(&self, buffer: &mut Solution) -> HighsStatus;
    
    // Model modification
    pub fn add_row(&mut self, lower: f64, upper: f64, 
                   cols: &[HighsInt], vals: &[f64]) -> usize;
    pub fn delete_row(&mut self, row: usize) -> HighsStatus;
    pub fn change_rows_bounds(&mut self, row: usize, lower: f64, upper: f64);
    pub fn change_coefficients(&mut self, row: usize, col: usize, value: f64);
    
    // Basis management
    pub fn get_basis(&self) -> Basis;
    pub fn set_basis(&mut self, basis: &Basis) -> HighsStatus;
    
    // Options
    pub fn set_option<T: HighsOptionValue>(&mut self, name: &str, value: T);
}
```

### LP Problem Structure

```rust
pub struct Problem {
    pub num_col: usize,
    pub num_row: usize,
    pub num_nz: usize,
    pub col_cost: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    columns: Vec<(Vec<c_int>, Vec<f64>)>,  // CSC format
    pub offset: f64,
}
```

---

## 7. Parallelization Architecture

### Current: Rayon Thread Pool

```rust
// Forward passes in parallel
trajectories
    .par_iter_mut()
    .enumerate()
    .map(|(idx, traj)| {
        forward_pass::execute(ctx, timing)
    })
    .collect::<Result<Vec<_>, _>>()?;

// Backward pass Phase 1: Parallel cut computation
training_states
    .par_iter()
    .map(|state| {
        compute_cut_for_state(state, branching_scenarios)
    })
    .collect::<Vec<_>>();
```

### Handler Coordinator Pattern

```rust
pub struct ParallelHandlerCoordinator {
    handlers: Vec<StageHandler>,  // One per forward pass
    // Each handler owns its own copy of:
    // - Subproblem graphs
    // - LP models (HiGHS instances)
    // - Solution buffers
}
```

### Thread-Local Optimizations

```rust
thread_local! {
    // Solution extraction buffer (avoids allocation per solve)
    static SOLUTION_BUFFER: RefCell<Solution>;
    
    // Row operation buffers
    static ROW_COLS_BUFFER: RefCell<Vec<HighsInt>>;
    static ROW_VALS_BUFFER: RefCell<Vec<f64>>;
    
    // Batch constraint update buffers
    static BATCH_ROW_INDICES: RefCell<Vec<HighsInt>>;
    static BATCH_LOWER_BOUNDS: RefCell<Vec<f64>>;
    static BATCH_UPPER_BOUNDS: RefCell<Vec<f64>>;
}
```

---

## 8. Data Structures

### Future Cost Function

```rust
pub struct FutureCostFunction {
    pub cut_pool: BendersCutPool,
    pub state_pool: VisitedStatePool,
}

pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,  // Water values, lag duals
    pub rhs: f64,
    active: AtomicBool,          // For cut selection
    non_dominated_state_count: AtomicUsize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
    slot_index: AtomicUsize,     // Preallocated LP row
    populated: bool,
}

pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub total_cut_count: usize,
    num_forward_passes: usize,
    preallocated: bool,
}
```

### State Representations

```rust
pub trait State: Send + Sync {
    fn coefficients(&self) -> &[f64];
    fn update_coefficients(&mut self, coefficients: &[f64]);
    fn dimension(&self) -> usize;
    fn extract_storage_from_trajectory(&mut self, 
        trajectory: &[&Realization]) -> Cow<'_, [f64]>;
    fn evaluate_cut_ref<'a>(&mut self,
        risk_measure: &dyn RiskMeasure,
        branching_realizations: &[Realization],
        buffers: &'a mut CutComputationBuffers,
    ) -> CutEvalResult<'a>;
    // ... more methods
}

// Storage-only state
pub struct StorageState {
    state_coefficients: Vec<f64>,  // [V₀, V₁, ..., Vₙ]
    // metadata fields
}

// Storage + inflow lags
pub struct StorageAndInflowState {
    state_coefficients: Vec<f64>,  // [V₀, V₁, ..., Y₀⁽¹⁾, Y₀⁽²⁾, ...]
    storage_range: Range<usize>,
    // per-hydro lag metadata
}
```

### Scenario Tree

```rust
pub struct ScenarioTree {
    pub stage_noises: Vec<StageNoises>,
    pub metadata: ScenarioTreeMetadata,
}

pub struct StageNoises {
    pub stage_id: usize,
    pub branchings: Vec<BranchingNoises>,
}

pub struct BranchingNoises {
    pub load_innovations: Vec<f64>,
    pub inflow_innovations: Vec<f64>,
}
```

---

## 9. Performance Characteristics

### Profiled Bottlenecks (Typical Problem)

| Phase | Time % | Description |
|-------|--------|-------------|
| LP Solve | 40-60% | HiGHS simplex iterations |
| Model Preprocessing | 15-25% | Constraint bound updates |
| Model Postprocessing | 10-15% | Solution extraction |
| Cut Selection | 5-10% | Domination evaluation |
| Cut Application | 5-10% | LP row updates |
| I/O & Overhead | 5-10% | Parallel coordination |

### Memory Usage Patterns

- **Preallocated pools**: Cuts and states preallocated to `iterations × forward_passes`
- **Thread-local buffers**: Solution vectors, row operations
- **Per-handler LP models**: Each parallel handler owns separate HiGHS instance
- **Scenario tree**: Fully materialized in memory (SAA)

### Scalability Limits (Current)

| Dimension | Current Limit | Bottleneck |
|-----------|---------------|------------|
| Hydros | ~500 | State dimension → cut evaluation O(n²) |
| Stages | ~60 | Forward pass memory |
| Iterations | ~500 | Cut pool growth |
| Forward Passes | ~64 | Handler memory |
| Branchings | ~1000 | Scenario generation |

---

## 10. Known Limitations & Technical Debt

### Architectural Limitations

1. **Single-node parallelism**: Rayon cannot scale across machines
2. **HiGHS coupling**: No abstraction layer for alternative solvers
3. **Monolithic input**: Single JSON files don't scale to large systems
4. **Synchronous phases**: Forward/backward passes fully synchronized

### Code Quality Issues

1. **Large modules**: `subproblem.rs` (~236KB), `sddp/mod.rs` (~124KB)
2. **Feature flags**: `timing`, `timing-detailed`, `simd-optimizations` add complexity
3. **Error handling**: Mix of `Result`, `panic!`, and `unwrap()`
4. **Testing coverage**: Integration tests exist, but unit test coverage varies

### Performance Technical Debt

1. **Cut coefficient storage**: Full dense vectors even for sparse cuts
2. **State extraction**: Some redundant copies in `prepare_from_trajectory`
3. **Scenario generation**: Sequential Cholesky application
4. **Basis caching**: Per-handler, not shared

---

## 11. External Dependencies

### Core Dependencies

```toml
[dependencies]
# Solver
highs-sys = "1.6.4"

# Parallelism (current)
rayon = "1.10.0"

# Numerics
nalgebra = "0.33"
statrs = "0.17"
rand = "0.9.0"
rand_distr = "0.5.1"
rand_xoshiro = "0.7.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Output
arrow = "57"
parquet = "57"
csv = "1.3"

# CLI & Display
clap = { version = "4.5", features = ["derive"] }
chrono = "0.4"
crossterm = "0.27"
```

### New Dependency (FerroMPI)

Located at `~/git/ferrompi`:

```rust
// MPI 4.0+ bindings with persistent collectives
use ferrompi::{Mpi, Communicator, ReduceOp, PersistentRequest};

// Key features:
// - Blocking: broadcast_f64, allreduce_f64, gather_f64, scatter_f64
// - Nonblocking: ibroadcast_f64, iallreduce_f64
// - Persistent (MPI 4.0+): bcast_init_f64, allreduce_init_f64
// - Large count: Automatic for >2GB transfers
```

---

## 12. Refactoring Goals

### Primary Objectives

1. **MPI-based distributed computing**
   - Replace rayon with ferrompi for inter-node parallelism
   - Hybrid approach: MPI across nodes, optional rayon (or hybrid MPI calls) within nodes
   - Master-worker pattern for forward/backward passes

2. **Solver abstraction layer**
   - Trait-based solver interface
   - Initial implementations: HiGHS, CPLEX, Gurobi
   - Build time solver selection due to licensing constraints

3. **New input/output format**
   - Binary/columnar format for large-scale problems
   - Use the best file format for each data kind: JSON for objects, csv/parquet/other for timeseries, etc.
   - Distributed output (per-rank files) with aggregation as postprocessing (or keep separated by entity: hydro_id, thermal_id, etc.)

4. **Production readiness**
   - Comprehensive error handling
   - Checkpointing and recovery
   - Performance monitoring

### Questions for Requirements Specification

To proceed with the refactoring design, we need to specify:

1. **Input Format**
   - What new formats should be supported? (Parquet, HDF5, binary?)
   - Should the input be splittable for distributed loading?
   - What validation and preprocessing steps are needed?

We don't have specific formats to support. We have freedom to choose the formats that best suit the information we want to input to the program. Some inputs are more document-friendly like algorithm parameters, the registry of the existing hydro plants, etc. Some other can be represented in other ways, such in a table-like format in parquet or csv files.

Most time we don't input large amounts of data to the program, except when we are running a warm-start, on which we continue the execution of a training call from another run. The amount of cuts and visited stats will be large for the production executions. Also, the simulation outputs will be large for the production executions too.
Consider these numbers for the production cases: 120 stages, 160 hydros, 130 thermals, 6 buses, 10 lines, 200 forward passes, 50 iterations, 20 scenarios per node, 2000 simulation scenarios. We will also introduce a new concept which will be the "blocks", which are subdivisions of a stage that will influence how the linear problem is constructed, and some outputs will be also indexed by block (3 blocks in production).

There will be many validation steps, and preprocessing ones too. You can begin by giving me a design document on how I could specify the first version of the entire data model so we can construct the proper documentation for the production scenario, including the input and output formats.

2. **Output Format**
   - What aggregation is needed across MPI ranks?
   - Should cuts be distributed or replicated?
   - How to handle simulation outputs at scale?

For the SDDP algorithm to be correct, we need to synchronize at each stage in the backward pass, so we can have reproducibility in our results. We should keep a copy of the cuts and visited states on each MPI rank, and by using hybrid MPI, this copy of the cuts and states is shared by the threads associated with each rank. We can output the simulation results either all at the same time, if it fits the memory, or if the memory becomes a bottleneck, we can output each scenario result separately and aggregate after.

3. **MPI Distribution Strategy**
   - How to partition forward passes across ranks?
   - Where to store the FCF (replicated vs. distributed)?
   - What communication pattern for cut sharing?

We want to distribute forward passes across ranks in an asynchronous manner. Ideally, what we do with rayon threads should be mapped to a 2-stage forward pass distribution: first we distribute forward passes across ranks, then each rank will distribute them across its threads. If I think that the ranks should have independent memory spaces, since they can run on different nodes, we will put one replica of the FCF per rank. The cut sharing should be done in the following: the cut computed by each forward pass, in the backward step, should be communicated from each thread to the rank that controls it, then the rank should communicate to the rank 0 (master process). Then, the rank 0 will aggregate and sort all the new cuts from the current iteration, and then we proceed updating all the replicas of the future cost function and solver problem definitions with this new set of cut constraints.


4. **Solver Abstraction**
   - What solver features are essential vs. optional?
   - How to handle solver-specific optimizations (warm-start, basis)?
   - What performance characteristics to maintain?

We can consider the current set of features that we already use now as essential when designing our first version for the solver abstraction. We will most surely want to use solver optimizations, but warm-start and basis reuse are common for all solvers. What we can optimize is more in the line of specific solver parameters, such as tolerances, solution methods, etc. We will design specific implementations for each supported solver, something like a trait. We will want to maintain the greatest memory determinism as possible, avoiding unnecessary heap usage.

5. **Hybrid Parallelism**
   - MPI + rayon or MPI + OpenMP?
   - How many threads per MPI rank?
   - Memory model for hybrid execution?

We should stick with the one that gives us more performance, I think MPI + OpenMP will do it. We will run in high-end EPYC instances on AWS that range from 64 to 192 vCPUs per node. So we might use up to 192 threads per rank to save memory because of the cut replication per rank. Lets design our memory model together as we go through the specification more deeply.

---

## Appendix A: File Metrics

| File | Lines | Size |
|------|-------|------|
| `subproblem.rs` | ~5500 | 236KB |
| `sddp/mod.rs` | ~2900 | 124KB |
| `state.rs` | ~2400 | 104KB |
| `solver.rs` | ~1400 | 59KB |
| `input.rs` | ~1400 | 60KB |
| `fcf.rs` | ~750 | - |
| `graph.rs` | ~270 | - |

## Appendix B: Example Execution

```bash
# Training with output
powers run examples/07-par-model-with-inflow-state/

# Output files generated:
# - training.parquet
# - cuts.parquet
# - states.parquet
# - simulation.parquet
# - variable_dictionary.csv
# - coefficient_dictionary.csv
# - state_component_dictionary.csv
```

## Appendix C: Typical Problem Sizes

| Problem Type | Hydros | Stages | Forward Passes | Branchings |
|--------------|--------|--------|----------------|------------|
| Tutorial | 1-2 | 12 | 4-10 | 10-20 |
| Small | 5-20 | 24-60 | 10-20 | 50-100 |
| Medium | 50-100 | 60-120 | 20-50 | 100-200 |
| Large (Brazil) | 156+ | 120+ | 50-200 | 200-1000 |
| Target (Future) | 500+ | 240+ | 200-500 | 1000-5000 |

---

*This document serves as the authoritative reference for the POWE.RS refactoring project. It should be updated as requirements are specified and design decisions are made.*
