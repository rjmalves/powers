# Code Refactoring Plan for Collaborative Development

**Status**: 📋 Planning Phase  
**Goal**: Transform codebase into clean, intuitive, collaboration-ready architecture  
**Timeline**: 8 weeks (phased approach)  
**Last Updated**: 2025-11-09

---

## 📊 Current State Analysis

### Codebase Metrics

**Size & Distribution:**
- **Total Lines**: ~26,583 LOC
- **Functions**: 643 total (209 public)
- **Modules**: ~15 mixed-responsibility modules
- **Test Coverage**: ✅ 446 tests passing
- **Quality Gates**: ✅ Clippy clean, ✅ All tests pass

**Largest Files (Technical Debt Hotspots):**

| File | Lines | Status | Priority |
|------|-------|--------|----------|
| `src/subproblem.rs` | 6,213 | ⚠️ God Object | 🔴 HIGH |
| `src/sddp/mod.rs` | 3,967 | ⚠️ Mixed Concerns | 🔴 HIGH |
| `src/state.rs` | 2,128 | ⚠️ Large | 🟡 MEDIUM |
| `src/input.rs` | 1,763 | ⚠️ Large | 🟡 MEDIUM |
| `src/sddp/builder.rs` | 1,494 | ⚠️ Builder Heavy | 🟢 LOW |
| `src/system.rs` | 1,147 | ✅ Manageable | 🟢 LOW |

### Complexity Hot Spots

#### 1. **Parameter Overload** (Constructor Complexity)

Functions with excessive parameters indicate missing abstraction layers:

| Function | Parameters | File | Impact |
|----------|------------|------|--------|
| `Subproblem::new()` | 13 | subproblem.rs | ⚠️⚠️⚠️ |
| `SddpAlgorithm::new()` | 12 | sddp/mod.rs | ⚠️⚠️⚠️ |
| `backward_step_at_node()` | 9 | sddp/mod.rs | ⚠️⚠️ |
| `update_future_cost_function()` | 9 | sddp/mod.rs | ⚠️⚠️ |
| `compute_cut_for_backward_step()` | 7 | sddp/mod.rs | ⚠️ |
| `train()` | 7 | sddp/mod.rs | ⚠️ |

**Clean Code Principle Violated**: Functions should have ≤4 parameters (Uncle Bob's Clean Code)

#### 2. **Long Functions** (Cognitive Complexity)

Functions exceeding 80 lines (excluding tests):

| Function | Lines | File | Refactor Priority |
|----------|-------|------|-------------------|
| `retry_solve()` | 99 | subproblem.rs | 🔴 HIGH |
| `build_sddp_system()` | 65 | input.rs | 🟡 MEDIUM |

**Note**: Test functions (90-145 lines) are acceptable for integration tests.

#### 3. **Architectural Issues**

**God Objects** (Single Responsibility Principle Violations):
- `Subproblem` (6,213 lines): Combines LP formulation, uncertainty handling, solving, state tracking
- `SddpAlgorithm` (3,967 lines): Training, simulation, forward/backward passes, cut management

**Tight Coupling**:
- Subproblem directly manipulates solver internals
- SDDP algorithm tightly coupled to subproblem implementation
- State management scattered across multiple modules

**Missing Abstractions**:
- No clear separation between "what" (algorithm) and "how" (LP formulation)
- Uncertainty handling mixed with LP solving
- Cut management embedded in algorithm loop

---

## 🎯 Refactoring Strategy

### Design Principles

This refactoring follows **Clean Architecture** and **Clean Code** principles:

1. **Single Responsibility**: Each module/class does one thing well
2. **Open/Closed**: Open for extension, closed for modification
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **Interface Segregation**: Small, focused interfaces
5. **DRY**: Don't Repeat Yourself
6. **KISS**: Keep It Simple, Stupid

### Success Metrics

**Quantitative Goals:**

| Metric | Current | Target | Rationale |
|--------|---------|--------|-----------|
| Max file size | 6,213 LOC | ≤1,500 LOC | Maintainability threshold |
| Avg function length | Mixed | <50 lines | Cognitive load limit |
| Max parameters | 13 | ≤4 | Clean Code standard |
| Module count | ~15 | 25-30 | Better separation |
| Cyclomatic complexity | High | <10/function | Testability |

**Qualitative Goals:**

- ✅ **15-Minute Rule**: New contributor understands a module in <15 minutes
- ✅ **One-Sentence Rule**: Can explain what a file does in one sentence
- ✅ **Mock Test**: Can mock dependencies easily for unit testing
- ✅ **Clear Boundaries**: Responsibilities clearly separated
- ✅ **Self-Documenting**: Code explains itself, minimal comments needed

---

## 📅 Phased Implementation Plan

### Phase 1: Extract Configuration Objects (Weeks 1-2)

**Goal**: Replace long parameter lists with structured configuration objects

**Priority**: 🔴 HIGH | **Risk**: 🟢 LOW | **Impact**: 🔴 HIGH

#### Why Start Here?

1. **Immediate Value**: Readability improves dramatically
2. **Low Risk**: Pure extraction, no logic changes
3. **Enables Future Work**: Makes Phase 2 decomposition easier
4. **Quick Wins**: Can complete in 1-2 weeks
5. **Team Morale**: Early success builds momentum

#### Tasks

##### 1.1 Create `SubproblemConfig`

**Before** (13 parameters):
```rust
pub fn new(
    system: &system::System,
    stage: usize,
    node_id: usize,
    initial_storage: &[f64],
    cuts: Vec<cut::Cut>,
    stage_count: usize,
    state_space: state::StateSpace,
    risk_measure: risk_measure::RiskMeasure,
    state_factory: Arc<dyn state::StateFactory>,
    model_solver_override: Option<solver::Solver>,
    enable_timing: bool,
    enable_aggregation: bool,
    uncertainty_observation_data: Vec<UncertaintyObservationData>,
) -> Self
```

**After** (1 parameter):
```rust
pub struct SubproblemConfig {
    pub system: System,
    pub stage: usize,
    pub node_id: usize,
    pub initial_storage: Vec<f64>,
    pub stage_count: usize,
    pub state_space: StateSpace,
    pub risk_measure: RiskMeasure,
    pub solver: Option<Solver>,
    pub timing_enabled: bool,
    pub aggregation_enabled: bool,
}

impl SubproblemConfig {
    pub fn builder() -> SubproblemConfigBuilder { ... }
}

pub fn new(
    config: SubproblemConfig,
    cuts: Vec<Cut>,
    state_factory: Arc<dyn StateFactory>,
    uncertainty_data: Vec<UncertaintyObservationData>,
) -> Self
```

**Benefits**:
- 13 parameters → 4 parameters (69% reduction)
- Optional parameters with sensible defaults
- Builder pattern for fluent construction
- Easy to extend without breaking changes
- Better IDE autocomplete

##### 1.2 Create `SddpConfig` Hierarchy

```rust
pub struct SddpConfig {
    pub training: TrainingConfig,
    pub simulation: SimulationConfig,
    pub output: OutputConfig,
    pub logging: LoggingConfig,
}

pub struct TrainingConfig {
    pub max_iterations: usize,
    pub num_forward_passes: usize,
    pub convergence_tolerance: f64,
    pub time_limit: Option<Duration>,
}

pub struct BackwardPassConfig {
    pub cut_selection: CutSelectionStrategy,
    pub aggregation_enabled: bool,
    pub parallel_execution: bool,
}
```

##### 1.3 Extract Solver Configuration

```rust
pub struct SolverConfig {
    pub solver_type: Solver,
    pub time_limit: Option<Duration>,
    pub tolerance: SolverTolerances,
    pub retry_strategy: RetryStrategy,
}

pub struct RetryStrategy {
    pub max_attempts: usize,
    pub backoff: BackoffStrategy,
    pub fallback_solvers: Vec<Solver>,
}
```

**Files to Modify**:
- ✏️ `src/subproblem.rs` - Add `SubproblemConfig`
- ✏️ `src/sddp/mod.rs` - Add `SddpConfig`, `TrainingConfig`, `BackwardPassConfig`
- ✏️ `src/solver.rs` - Add `SolverConfig`, `RetryStrategy`
- ✏️ `src/sddp/builder.rs` - Update to use new configs
- ✏️ `src/input.rs` - Parse into config structures

**Testing Strategy**:
- ✅ Keep all existing tests (regression suite)
- ✅ Add builder tests for config objects
- ✅ Validate default values
- ✅ Test builder chaining

**Success Criteria**:
- [ ] All constructors have ≤4 parameters
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] Documentation updated
- [ ] Examples in doc comments

---

### Phase 2: Decompose God Objects (Weeks 3-4)

**Goal**: Split mega-modules into focused, cohesive modules

**Priority**: 🔴 HIGH | **Risk**: 🟡 MEDIUM | **Impact**: 🔴 HIGH

#### Phase 2.1: Split `subproblem.rs` (6,213 → ~1,500 lines each)

**New Module Structure**:

```
src/subproblem/
├── mod.rs                      (200 lines)
│   └── Public API, Subproblem orchestration
│
├── config.rs                   (150 lines)
│   └── SubproblemConfig, builders
│
├── builder.rs                  (300 lines)
│   └── Construction logic, LP formulation setup
│
├── constraints/
│   ├── mod.rs                  (100 lines)
│   ├── hydro.rs                (250 lines) - Hydro balance, cascade
│   ├── thermal.rs              (150 lines) - Thermal generation
│   ├── transmission.rs         (200 lines) - Line flow, bus balance
│   └── future_cost.rs          (150 lines) - Cost-to-go cuts
│
├── uncertainty/
│   ├── mod.rs                  (100 lines)
│   ├── realization.rs          (300 lines) - Apply scenario to LP
│   ├── observation_data.rs     (200 lines) - Precomputed constraint data
│   └── lag_tracking.rs         (400 lines) - AR lag buffer management
│
├── solver_wrapper.rs           (400 lines)
│   ├── SolverWrapper struct
│   ├── retry_solve() - Extract from 99-line function
│   └── Solver interaction, error handling
│
├── state_extraction.rs         (300 lines)
│   └── Extract state from LP solution
│
└── tests/
    ├── mod.rs
    ├── construction_tests.rs   (600 lines) - Move from main file
    ├── constraint_tests.rs     (800 lines)
    ├── uncertainty_tests.rs    (500 lines)
    └── integration_tests.rs    (1,000 lines)
```

**Key Extractions**:

1. **`SolverWrapper`** - Encapsulate solving complexity
   ```rust
   pub struct SolverWrapper {
       model: solver::Model,
       config: SolverConfig,
   }
   
   impl SolverWrapper {
       pub fn solve(&mut self) -> Result<SolveResult>;
       fn retry_solve(&mut self) -> Result<SolveResult>;
       fn apply_retry_options(&mut self, attempt: usize);
   }
   ```

2. **`ConstraintManager`** - Manage LP constraints
   ```rust
   pub struct ConstraintManager {
       hydro_constraints: HydroConstraints,
       thermal_constraints: ThermalConstraints,
       transmission_constraints: TransmissionConstraints,
       future_cost_constraints: FutureCostConstraints,
   }
   
   impl ConstraintManager {
       pub fn add_to_model(&self, model: &mut Model) -> Result<()>;
       pub fn update_rhs(&self, model: &mut Model, data: &UpdateData) -> Result<()>;
   }
   ```

3. **`UncertaintyHandler`** - Handle scenario realization
   ```rust
   pub struct UncertaintyHandler {
       observation_data: Vec<UncertaintyObservationData>,
       lag_tracker: LagTracker,
   }
   
   impl UncertaintyHandler {
       pub fn realize_scenario(&mut self, innovations: &[f64]) -> Result<RealizedScenario>;
       pub fn update_model(&self, model: &mut Model, scenario: &RealizedScenario) -> Result<()>;
   }
   ```

4. **`LagTracker`** - AR lag buffer management
   ```rust
   pub struct LagTracker {
       load_lags: LagBuffers,
       inflow_lags: LagBuffers,
   }
   
   impl LagTracker {
       pub fn update_from_trajectory(&mut self, trajectory: &Trajectory) -> Result<()>;
       pub fn get_lag_variables(&self) -> Vec<(usize, f64)>;
   }
   ```

**Refactoring Process**:

1. **Extract Pure Functions First** (lowest risk)
   - Helper functions with no state mutation
   - Mathematical utilities
   - Validation functions

2. **Extract Data Structures** (low risk)
   - Move related data into new modules
   - Keep original struct as facade temporarily
   - Delegate to new structures

3. **Extract Behaviors** (medium risk)
   - Move methods to appropriate modules
   - Update callers incrementally
   - Keep integration tests passing

4. **Remove Facade** (final step)
   - Once all callers updated
   - Remove delegation layer
   - Clean up imports

**Risk Mitigation**:
- ✅ Run tests after each extraction
- ✅ Use deprecation warnings for old APIs
- ✅ Maintain facade during transition
- ✅ Benchmark performance critical paths
- ✅ Keep git commits small and focused

#### Phase 2.2: Split `sddp/mod.rs` (3,967 → ~800 lines each)

**New Module Structure**:

```
src/sddp/
├── mod.rs                      (200 lines)
│   └── Public API, SddpAlgorithm orchestration
│
├── builder.rs                  (1,494 lines - existing, keep)
│   └── Already well-structured
│
├── instance.rs                 (135 lines - existing, keep)
│   └── Already well-structured
│
├── algorithm.rs                (600 lines)
│   ├── Main train() loop
│   ├── Convergence checking
│   └── High-level orchestration
│
├── forward_pass/
│   ├── mod.rs                  (150 lines)
│   ├── executor.rs             (400 lines) - Forward pass execution
│   ├── trajectory.rs           (200 lines) - Trajectory building
│   └── timing.rs               (150 lines) - Forward timing
│
├── backward_pass/
│   ├── mod.rs                  (150 lines)
│   ├── executor.rs             (500 lines) - Backward pass execution
│   ├── cut_computation.rs      (300 lines) - Cut generation
│   └── timing.rs               (150 lines) - Backward timing
│
├── simulation/
│   ├── mod.rs                  (100 lines)
│   ├── executor.rs             (300 lines) - Simulation execution
│   └── trajectory.rs           (200 lines) - Simulation trajectories
│
├── convergence.rs              (200 lines)
│   └── Convergence criteria, gap computation
│
└── tests/
    └── (Move integration tests here)
```

**Key Extractions**:

1. **`ForwardPass`** - Self-contained forward pass
   ```rust
   pub struct ForwardPassExecutor {
       config: ForwardPassConfig,
       subproblems: Vec<Arc<Mutex<Subproblem>>>,
   }
   
   impl ForwardPassExecutor {
       pub fn execute(&self, saa: &[Vec<f64>]) -> Result<Vec<Trajectory>>;
       fn solve_stage(&self, stage: usize, state: &State) -> Result<StageResult>;
   }
   ```

2. **`BackwardPass`** - Self-contained backward pass
   ```rust
   pub struct BackwardPassExecutor {
       config: BackwardPassConfig,
       subproblems: Vec<Arc<Mutex<Subproblem>>>,
   }
   
   impl BackwardPassExecutor {
       pub fn execute(&self, trajectories: &[Trajectory]) -> Result<Vec<Cut>>;
       fn compute_cut_at_node(&self, node: &GraphNode, trajectory: &Trajectory) -> Result<Cut>;
   }
   ```

3. **`SimulationEngine`** - Simulation execution
   ```rust
   pub struct SimulationEngine {
       config: SimulationConfig,
       subproblems: Vec<Arc<Mutex<Subproblem>>>,
   }
   
   impl SimulationEngine {
       pub fn simulate(&self, num_scenarios: usize) -> Result<Vec<Trajectory>>;
   }
   ```

**Testing Strategy**:
- ✅ Keep existing integration tests
- ✅ Add unit tests for new modules
- ✅ Test forward/backward passes independently
- ✅ Property-based tests for convergence

---

### Phase 3: Extract Domain Services (Weeks 5-6)

**Goal**: Create focused, single-responsibility services

**Priority**: 🟡 MEDIUM | **Risk**: 🟢 LOW | **Impact**: 🟡 MEDIUM

#### 3.1 Create Specialized Services

**Service Layer Pattern**: Separate business logic from algorithm orchestration

```rust
// src/services/mod.rs
pub mod cut_manager;
pub mod uncertainty_realizer;
pub mod state_extractor;
pub mod dual_processor;
pub mod trajectory_builder;
```

##### `CutManager` - Cut lifecycle management

```rust
// src/services/cut_manager.rs
pub struct CutManager {
    cuts: Vec<Cut>,
    selection_strategy: CutSelectionStrategy,
}

impl CutManager {
    /// Add a new cut to the collection
    pub fn add_cut(&mut self, cut: Cut) -> Result<CutId>;
    
    /// Select dominant cuts for a given state
    pub fn select_cuts(&self, state: &[f64]) -> Vec<CutId>;
    
    /// Apply selected cuts to LP model
    pub fn apply_cuts(&self, model: &mut Model, cut_ids: &[CutId]) -> Result<()>;
    
    /// Get cut statistics
    pub fn statistics(&self) -> CutStatistics;
}
```

**Benefits**:
- Single source of truth for cuts
- Testable in isolation
- Swappable selection strategies
- Performance monitoring

##### `UncertaintyRealizer` - Scenario realization

```rust
// src/services/uncertainty_realizer.rs
pub struct UncertaintyRealizer {
    observation_data: Vec<UncertaintyObservationData>,
    lag_tracker: LagTracker,
}

impl UncertaintyRealizer {
    /// Realize a scenario from SAA innovations
    pub fn realize(&mut self, innovations: &[f64]) -> Result<RealizedScenario>;
    
    /// Update lag buffers from trajectory
    pub fn update_lags(&mut self, trajectory: &Trajectory) -> Result<()>;
    
    /// Get current lag values for state
    pub fn current_lags(&self) -> &[f64];
}

pub struct RealizedScenario {
    pub loads: Vec<f64>,
    pub inflows: Vec<f64>,
    pub observations: Vec<f64>,
}
```

##### `StateExtractor` - Extract state from trajectory

```rust
// src/services/state_extractor.rs
pub struct StateExtractor {
    state_space: StateSpace,
    hydro_ar_orders: Vec<usize>,
}

impl StateExtractor {
    /// Extract state coefficients from trajectory
    pub fn extract(&self, trajectory: &Trajectory) -> Result<Vec<f64>>;
    
    /// Extract state from LP solution
    pub fn from_solution(&self, solution: &SolveResult) -> Result<Vec<f64>>;
}
```

##### `DualProcessor` - Process LP duals

```rust
// src/services/dual_processor.rs
pub struct DualProcessor {
    config: DualProcessingConfig,
}

impl DualProcessor {
    /// Extract relevant duals from LP solution
    pub fn extract_duals(&self, solution: &SolveResult) -> Result<Duals>;
    
    /// Compute cut coefficients from duals
    pub fn compute_cut_coefficients(&self, duals: &Duals) -> Result<Vec<f64>>;
}

pub struct Duals {
    pub storage_duals: Vec<f64>,
    pub inflow_duals: Vec<f64>,
    pub load_duals: Vec<f64>,
}
```

#### 3.2 Implement Service Registry Pattern

```rust
// src/services/registry.rs
pub struct ServiceRegistry {
    cut_manager: Arc<Mutex<CutManager>>,
    uncertainty_realizer: Arc<Mutex<UncertaintyRealizer>>,
    state_extractor: Arc<StateExtractor>,
    dual_processor: Arc<DualProcessor>,
}

impl ServiceRegistry {
    pub fn new(config: &SddpConfig) -> Self;
    
    pub fn cut_manager(&self) -> Arc<Mutex<CutManager>>;
    pub fn uncertainty_realizer(&self) -> Arc<Mutex<UncertaintyRealizer>>;
    pub fn state_extractor(&self) -> Arc<StateExtractor>;
    pub fn dual_processor(&self) -> Arc<DualProcessor>;
}
```

**Benefits of Service Registry**:
- Centralized dependency management
- Easy to mock for testing
- Clear service lifecycle
- Enables dependency injection

**Testing Strategy**:
- ✅ Unit test each service independently
- ✅ Mock dependencies with traits
- ✅ Integration tests with real services
- ✅ Property-based tests for mathematical correctness

---

### Phase 4: Improve Naming & Documentation (Week 7)

**Goal**: Make code self-documenting

**Priority**: 🟡 MEDIUM | **Risk**: 🟢 LOW | **Impact**: 🔴 HIGH

#### 4.1 Naming Conventions

**Current Issues**:

| Current | Issue | Improved |
|---------|-------|----------|
| `fcf` | Cryptic abbreviation | `future_cost_function` |
| `saa` | Not self-explanatory | `sample_average_approximation` |
| `par` | Ambiguous | `periodic_autoregressive` |
| `realize_uncertainties` | Vague action | `apply_scenario_to_model` |
| `get_current_stage_objective` | Wordy | `extract_objective_value` |
| `tmp`, `x`, `i` | Single letter | Descriptive names |

**Naming Standards**:

```rust
// ✅ Good: Verb-noun pattern for methods
pub fn extract_state_from_trajectory(...) -> Vec<f64>;
pub fn compute_cut_coefficients(...) -> Vec<f64>;
pub fn apply_cuts_to_model(...) -> Result<()>;
pub fn update_lag_buffers(...) -> Result<()>;

// ✅ Good: Boolean predicates
pub fn is_converged(&self) -> bool;
pub fn has_feasible_solution(&self) -> bool;
pub fn can_aggregate_cuts(&self) -> bool;
pub fn should_retry(&self) -> bool;

// ✅ Good: Builder pattern
pub fn with_tolerance(mut self, tolerance: f64) -> Self;
pub fn with_max_iterations(mut self, max_iter: usize) -> Self;

// ❌ Bad: Cryptic
pub fn proc(&self) -> R;  // Process what? Return what?
pub fn upd(&mut self);    // Update what?
pub fn get(&self) -> X;   // Get what?
```

**Type Naming**:

```rust
// ✅ Good: Clear purpose
pub struct SubproblemConfig { ... }
pub struct ForwardPassExecutor { ... }
pub struct CutSelectionStrategy { ... }
pub struct UncertaintyObservationData { ... }

// ❌ Bad: Vague
pub struct Data { ... }
pub struct Manager { ... }
pub struct Handler { ... }
```

#### 4.2 Module Documentation

**Template for Every Module**:

```rust
//! # Module Name
//!
//! One-sentence summary of module purpose.
//!
//! ## Purpose
//!
//! Detailed explanation of what this module does and why it exists.
//!
//! ## Responsibilities
//!
//! - Responsibility 1
//! - Responsibility 2
//! - Responsibility 3
//!
//! ## Key Types
//!
//! - [`TypeA`] - Description
//! - [`TypeB`] - Description
//!
//! ## Example
//!
//! ```rust
//! use powers::module::TypeA;
//!
//! let config = TypeA::builder()
//!     .with_option(value)
//!     .build();
//! ```
//!
//! ## See Also
//!
//! - Related module links
//! - Relevant documentation

use crate::...;
```

**Example - Good Module Documentation**:

```rust
//! # Cut Management
//!
//! Manages the lifecycle of Benders cuts in the SDDP algorithm.
//!
//! ## Purpose
//!
//! This module provides centralized management for cuts (future cost function
//! approximations) including storage, selection, and application to LP models.
//! It implements various cut selection strategies to balance solution quality
//! with model size.
//!
//! ## Responsibilities
//!
//! - Store cuts generated during backward passes
//! - Select dominant cuts for a given state (cut selection strategies)
//! - Apply selected cuts to LP subproblem models
//! - Track cut statistics (age, usage, dominance)
//! - Implement cut aggregation for memory efficiency
//!
//! ## Key Types
//!
//! - [`CutManager`] - Main service for cut lifecycle management
//! - [`CutSelectionStrategy`] - Strategy trait for cut selection algorithms
//! - [`Cut`] - Represents a single Benders cut (hyperplane)
//! - [`CutStatistics`] - Performance metrics for cut collection
//!
//! ## Example
//!
//! ```rust
//! use powers::services::cut_manager::{CutManager, CutSelectionStrategy};
//!
//! let mut manager = CutManager::new(CutSelectionStrategy::AllCuts);
//!
//! // Add cuts during backward pass
//! manager.add_cut(cut)?;
//!
//! // Select cuts for forward pass
//! let state = vec![100.0, 150.0, 200.0];
//! let cut_ids = manager.select_cuts(&state);
//!
//! // Apply to model
//! manager.apply_cuts(&mut model, &cut_ids)?;
//! ```
//!
//! ## See Also
//!
//! - [`crate::cut`] - Cut data structure
//! - [`crate::sddp::backward_pass`] - Cut generation
//! - [`crate::subproblem`] - Cut application to LP
```

#### 4.3 Function Documentation

**Template**:

```rust
/// Brief one-line summary.
///
/// Longer description explaining what the function does, when to use it,
/// and any important caveats or assumptions.
///
/// # Arguments
///
/// * `param1` - Description
/// * `param2` - Description
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// This function returns an error if:
/// - Condition 1
/// - Condition 2
///
/// # Panics
///
/// This function panics if:
/// - Condition 1
/// - Condition 2
///
/// # Examples
///
/// ```rust
/// let result = function(arg1, arg2)?;
/// assert_eq!(result, expected);
/// ```
///
/// # Performance
///
/// Time complexity: O(n)
/// Space complexity: O(1)
///
/// # See Also
///
/// - [`related_function`]
pub fn function(param1: Type1, param2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

**Example - Good Function Documentation**:

```rust
/// Computes Benders cut coefficients from LP dual values.
///
/// This function uses the chain rule to compute cut coefficients for state
/// variables, accounting for autoregressive dynamics in the state space.
/// For storage-only states, coefficients are simply the storage duals.
/// For storage+inflow states, AR coefficients are applied via chain rule.
///
/// # Arguments
///
/// * `duals` - Dual values extracted from LP solution
/// * `state_space` - State space type (storage-only vs storage+inflow)
/// * `ar_coefficients` - Transformed AR coefficients (ψ) for each hydro
///
/// # Returns
///
/// Vector of cut coefficients in state space order: [V₀, ..., Vₙ, Y₀⁽¹⁾, ...]
///
/// # Errors
///
/// Returns error if:
/// - Dual vector length doesn't match expected state dimension
/// - AR coefficient dimensions are inconsistent
///
/// # Examples
///
/// ```rust
/// use powers::services::dual_processor::compute_cut_coefficients;
///
/// let duals = Duals {
///     storage: vec![1.0, 2.0, 3.0],
///     inflow: vec![0.5, 0.6],
/// };
/// let coeffs = compute_cut_coefficients(&duals, state_space, &ar_coeffs)?;
/// ```
///
/// # Performance
///
/// Time: O(n + Σ(pᵢ)) where n = num_hydros, pᵢ = AR order for hydro i
/// Space: O(n + Σ(pᵢ)) for output vector
///
/// # Mathematical Details
///
/// The chain rule application follows:
/// ```text
/// ∂FO/∂Y_{t-j} = (λ^BH + λ^AR) · ψ_j
/// ```
/// where λ^BH is hydro balance dual, λ^AR is AR dynamics dual, ψ_j is
/// observation-space AR coefficient for lag j.
///
/// # See Also
///
/// - [`extract_duals`] - Extracts duals from LP solution
/// - [`crate::state::StorageAndInflowState`] - State representation
pub fn compute_cut_coefficients(
    duals: &Duals,
    state_space: StateSpace,
    ar_coefficients: &[Vec<f64>],
) -> Result<Vec<f64>> {
    // Implementation
}
```

#### 4.4 Code Comments

**When to Comment**:

```rust
// ✅ Good: Explain WHY, not WHAT
// Use Kahan summation to avoid catastrophic cancellation in dot product
let result = kahan_dot_product(&coefficients, &state);

// ✅ Good: Mathematical derivation
// Chain rule: ∂FO/∂Y_{t-j} = (λ^BH + λ^AR) · ψ_j
let cut_coeff = (hydro_dual + ar_dual) * psi[j];

// ✅ Good: Non-obvious business logic
// Retry with relaxed tolerance only for numerically challenging instances
if is_numerically_unstable(&system) && retry_count < 2 {
    relax_tolerance(&mut model);
}

// ❌ Bad: Obvious code
// Increment counter
counter += 1;

// ❌ Bad: Redundant
// Get the value
let value = get_value();
```

**Comment Style**:

```rust
// Single-line comments for brief explanations

/// Doc comments for public APIs
/// (using markdown formatting)

/* Block comments for longer explanations
   spanning multiple lines */
```

---

### Phase 5: Reduce Function Complexity (Week 8)

**Goal**: Functions <50 lines, <4 parameters

**Priority**: 🟢 LOW | **Risk**: 🟢 LOW | **Impact**: 🟡 MEDIUM

#### 5.1 Extract Helper Functions

**Pattern: Long Function → Multiple Focused Functions**

**Before (99 lines)**:
```rust
fn retry_solve(
    model: &mut solver::Model,
    stage: usize,
    node_id: usize,
    enable_timing: bool,
) -> Result<(solver::SolveResult, Option<SubproblemTiming>)> {
    // Setup (10 lines)
    let mut timing = if enable_timing {
        Some(SubproblemTiming::default())
    } else {
        None
    };
    
    // Attempt 1 with default options (20 lines)
    set_default_solver_options(model);
    let start_solve = Instant::now();
    let mut result = model.solve();
    let solve_duration = start_solve.elapsed();
    
    if let Some(ref mut t) = timing {
        t.solver_time = solve_duration;
    }
    
    // Attempt 2 with adjusted options (20 lines)
    if result.is_err() {
        set_first_retry_solver_options(model);
        let start_retry = Instant::now();
        result = model.solve();
        // ... error handling ...
    }
    
    // Attempt 3 (20 lines)
    // Attempt 4 (20 lines)
    // Final error handling (9 lines)
}
```

**After (~20 lines each)**:
```rust
fn retry_solve(
    model: &mut solver::Model,
    stage: usize,
    node_id: usize,
    enable_timing: bool,
) -> Result<(SolveResult, Option<SubproblemTiming>)> {
    let config = RetryConfig::new(stage, node_id, enable_timing);
    
    for attempt in 0..config.max_attempts {
        match try_solve_with_config(model, &config, attempt) {
            Ok(result) => return Ok(result),
            Err(e) if attempt < config.max_attempts - 1 => {
                log::warn!("Solve attempt {} failed: {}", attempt, e);
                continue;
            }
            Err(e) => return Err(e.into()),
        }
    }
    
    unreachable!("Loop should return in last iteration")
}

fn try_solve_with_config(
    model: &mut solver::Model,
    config: &RetryConfig,
    attempt: usize,
) -> Result<(SolveResult, Option<SubproblemTiming>)> {
    apply_solver_options(model, attempt);
    
    let start = Instant::now();
    let result = model.solve()?;
    let duration = start.elapsed();
    
    let timing = config.timing_enabled.then(|| SubproblemTiming {
        solver_time: duration,
        ..Default::default()
    });
    
    Ok((result, timing))
}

fn apply_solver_options(model: &mut solver::Model, attempt: usize) {
    let options = match attempt {
        0 => get_default_options(),
        1 => get_first_retry_options(),
        2 => get_second_retry_options(),
        3 => get_third_retry_options(),
        _ => get_final_retry_options(),
    };
    
    model.set_options(options);
}
```

**Benefits**:
- Each function has clear purpose
- Easier to test independently
- Less cognitive load
- Reusable components

#### 5.2 Use Table-Driven Logic

**Before (Repetitive)**:
```rust
fn set_retry_solver_options(model: &mut solver::Model, retry: usize) {
    match retry {
        0 => set_default_solver_options(model),
        1 => set_first_retry_solver_options(model),
        2 => set_second_retry_solver_options(model),
        3 => set_third_retry_solver_options(model),
        _ => set_final_retry_solver_options(model),
    }
}

fn set_default_solver_options(model: &mut solver::Model) {
    model.set_time_limit(60.0);
    model.set_gap_tolerance(1e-6);
    model.set_method(Method::Primal);
}

fn set_first_retry_solver_options(model: &mut solver::Model) {
    model.set_time_limit(120.0);
    model.set_gap_tolerance(1e-5);
    model.set_method(Method::Dual);
}

// ... more repetition ...
```

**After (Table-Driven)**:
```rust
#[derive(Clone)]
struct SolverOptions {
    time_limit: f64,
    gap_tolerance: f64,
    method: Method,
}

fn get_retry_options(attempt: usize) -> SolverOptions {
    const OPTIONS_TABLE: [SolverOptions; 5] = [
        SolverOptions { time_limit: 60.0,  gap_tolerance: 1e-6, method: Method::Primal },
        SolverOptions { time_limit: 120.0, gap_tolerance: 1e-5, method: Method::Dual },
        SolverOptions { time_limit: 180.0, gap_tolerance: 1e-4, method: Method::Barrier },
        SolverOptions { time_limit: 240.0, gap_tolerance: 1e-3, method: Method::Network },
        SolverOptions { time_limit: 300.0, gap_tolerance: 1e-2, method: Method::Concurrent },
    ];
    
    OPTIONS_TABLE[attempt.min(4)].clone()
}

fn apply_options(model: &mut solver::Model, options: &SolverOptions) {
    model.set_time_limit(options.time_limit);
    model.set_gap_tolerance(options.gap_tolerance);
    model.set_method(options.method);
}
```

**Benefits**:
- Data-driven, not code-driven
- Easy to adjust options
- Less code duplication
- Clearer intent

#### 5.3 Early Returns vs Deep Nesting

**Before (Deep Nesting)**:
```rust
fn process_result(result: Result<Value>) -> Result<Output> {
    if let Ok(value) = result {
        if value.is_valid() {
            if let Some(processed) = value.process() {
                if processed.meets_criteria() {
                    return Ok(processed.into());
                } else {
                    return Err("Criteria not met".into());
                }
            } else {
                return Err("Processing failed".into());
            }
        } else {
            return Err("Invalid value".into());
        }
    } else {
        return Err("Result error".into());
    }
}
```

**After (Early Returns)**:
```rust
fn process_result(result: Result<Value>) -> Result<Output> {
    let value = result?;
    
    if !value.is_valid() {
        return Err("Invalid value".into());
    }
    
    let processed = value.process()
        .ok_or("Processing failed")?;
    
    if !processed.meets_criteria() {
        return Err("Criteria not met".into());
    }
    
    Ok(processed.into())
}
```

**Benefits**:
- Flatter structure
- Error handling upfront
- Happy path is clear

---

## 🛡️ Risk Mitigation & Testing Strategy

### Continuous Validation

**After Every Change**:

```bash
# 1. Format check
cargo fmt --all --check

# 2. Lint check
cargo clippy --all-targets --all-features -- -D warnings

# 3. Test suite
cargo test --all-features

# 4. Integration tests
cargo test --test '*'

# 5. Benchmark (if performance-critical)
cargo bench --bench <benchmark_name>

# 6. Documentation build
cargo doc --no-deps --document-private-items
```

### Testing Pyramid

```
                    ▲
                   / \
                  /   \
                 /     \
                /  E2E  \         (5%) - Full algorithm runs
               /---------\
              /           \
             / Integration \      (15%) - Module interactions
            /---------------\
           /                 \
          /   Unit Tests      \   (80%) - Individual functions
         /---------------------\
```

**Testing Strategy by Phase**:

| Phase | Testing Approach |
|-------|-----------------|
| Phase 1 | Unit tests for configs, builder tests |
| Phase 2 | Integration tests for extracted modules, keep existing tests |
| Phase 3 | Unit tests for services, mock dependencies |
| Phase 4 | Doc tests in examples, README validation |
| Phase 5 | Refactor existing tests, add edge case coverage |

### Performance Validation

**Benchmark Critical Paths**:

```rust
// benches/subproblem_bench.rs
#[bench]
fn bench_subproblem_solve(b: &mut Bencher) {
    let subproblem = create_test_subproblem();
    b.iter(|| {
        subproblem.solve_forward_step(black_box(&innovations))
    });
}

#[bench]
fn bench_cut_evaluation(b: &mut Bencher) {
    let cuts = create_test_cuts();
    let state = create_test_state();
    b.iter(|| {
        evaluate_cuts(black_box(&cuts), black_box(&state))
    });
}
```

**Performance Regression Detection**:
- ✅ Run benchmarks before refactoring (baseline)
- ✅ Run benchmarks after refactoring (comparison)
- ✅ Flag regressions >5% for review
- ✅ Document intentional trade-offs

### Git Workflow

**Commit Strategy**:

```
feature/phase1-config-extraction
├── Extract SubproblemConfig struct
├── Add SubproblemConfig builder
├── Update Subproblem::new() signature
├── Update all construction sites
├── Add config tests
└── Update documentation

feature/phase2-split-subproblem
├── Create subproblem module structure
├── Extract SolverWrapper
├── Extract ConstraintManager
├── Extract UncertaintyHandler
├── Update tests
└── Remove old structure
```

**Branch Naming**:
- `feature/phaseN-<description>` - For phased work
- `refactor/<module>-<what>` - For refactoring work
- `docs/<what>` - For documentation updates

**Commit Messages**:
```
refactor(subproblem): extract SolverWrapper

- Move retry logic to dedicated struct
- Reduce Subproblem complexity by 150 lines
- All tests pass, no behavior change

Relates to Phase 2.1 of refactoring plan
```

---

## 📊 Progress Tracking

### Checklist Template

```markdown
## Phase 1: Configuration Extraction

- [ ] SubproblemConfig
  - [ ] Define struct
  - [ ] Implement builder
  - [ ] Update constructors
  - [ ] Add tests
  - [ ] Update docs
  - [ ] **Verify**: All tests pass ✅
  - [ ] **Verify**: No clippy warnings ✅
  - [ ] **Verify**: Benchmarks OK ✅

- [ ] SddpConfig
  - [ ] Define struct hierarchy
  - [ ] Implement builders
  - [ ] Update algorithm
  - [ ] Add tests
  - [ ] Update docs
  - [ ] **Verify**: All tests pass ✅
  - [ ] **Verify**: No clippy warnings ✅
  - [ ] **Verify**: Benchmarks OK ✅

**Phase 1 Complete**: ☐
```

### Metrics Dashboard

Track progress with metrics:

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Target |
|--------|----------|---------|---------|---------|---------|---------|--------|
| Max file LOC | 6,213 | - | - | - | - | - | ≤1,500 |
| Max params | 13 | 4 | - | - | - | - | ≤4 |
| Avg func LOC | Mixed | - | - | - | - | - | <50 |
| Module count | 15 | - | - | - | - | - | 25-30 |
| Test count | 446 | - | - | - | - | - | ≥500 |
| Doc coverage | ~60% | - | - | - | - | - | 100% |

---

## 🎯 Success Criteria

### Definition of Done (for entire refactoring)

- [ ] **Code Quality**
  - [ ] No file >1,500 lines
  - [ ] No function >50 lines (except tests)
  - [ ] No function >4 parameters
  - [ ] All public APIs documented
  - [ ] No clippy warnings
  - [ ] Formatted with `cargo fmt`

- [ ] **Testing**
  - [ ] All existing tests pass
  - [ ] New modules have >80% coverage
  - [ ] Integration tests updated
  - [ ] Benchmarks show no regressions

- [ ] **Documentation**
  - [ ] All modules have header docs
  - [ ] All public functions documented
  - [ ] Examples in doc comments
  - [ ] README updated
  - [ ] Architecture guide created

- [ ] **Architecture**
  - [ ] Clear separation of concerns
  - [ ] Single responsibility per module
  - [ ] Testable in isolation
  - [ ] Mockable dependencies
  - [ ] Clear interfaces

### Review Criteria

Use the Code Reviewer agent with these standards:

**Blocking Issues**:
- ❌ Code not formatted
- ❌ Clippy warnings
- ❌ Failing tests
- ❌ Undocumented public APIs
- ❌ Performance regressions >10%

**Request Changes**:
- 🔧 Missing unit tests
- 🔧 Unclear naming
- 🔧 Code duplication
- 🔧 Deep nesting (>3 levels)

**Suggestions**:
- 💡 Further decomposition opportunities
- 💡 Performance optimizations
- 💡 Documentation improvements

---

## 📚 Resources & References

### Clean Code Principles

- **Single Responsibility Principle**: A class/module should have one reason to change
- **Open/Closed Principle**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes must be substitutable for base types
- **Interface Segregation**: Many specific interfaces > one general interface
- **Dependency Inversion**: Depend on abstractions, not concretions

### Rust-Specific Best Practices

- **Ownership**: Use borrowing to avoid unnecessary clones
- **Error Handling**: Use `Result<T, E>` for recoverable errors
- **Traits**: Define behavior through traits for testability
- **Modules**: Keep modules focused and cohesive
- **Documentation**: Use doc comments (`///`) for public APIs

### Recommended Reading

- "Clean Code" by Robert C. Martin
- "Refactoring" by Martin Fowler
- "The Rust Programming Language" (official book)
- "Rust API Guidelines" (official)

---

## 🚀 Next Steps

### Immediate Actions

1. **Review this plan** with the team
2. **Get buy-in** on approach and timeline
3. **Set up tracking** (GitHub project board)
4. **Create branches** for Phase 1 work
5. **Start Phase 1** with SubproblemConfig extraction

### Communication Plan

**Weekly Status Updates**:
- What was completed
- What's in progress
- Blockers/challenges
- Metrics update

**Review Checkpoints**:
- End of Phase 1: Review config extraction
- End of Phase 2: Review module decomposition
- End of Phase 3: Review service layer
- End of Phase 4: Review documentation
- Final review: Complete refactoring assessment

---

## 📝 Notes & Decisions

### Design Decisions

**Decision Log** (to be filled during refactoring):

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2025-11-09 | Use builder pattern for configs | Reduces params, enables defaults | Phase 1 |
| TBD | ... | ... | ... |

### Lessons Learned

(To be updated during refactoring)

---

## ✅ Sign-Off

**Plan Created**: 2025-11-09  
**Plan Status**: 📋 Awaiting Approval  
**Target Start**: TBD  
**Target Completion**: TBD (8 weeks from start)

**Approval**:
- [ ] Technical Lead
- [ ] Product Owner
- [ ] Team Review

---

**Ready to start? Let's begin with Phase 1! 🚀**
