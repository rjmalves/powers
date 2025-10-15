# PAR Model Implementation Completion Plan v2.0

**Status**: Active  
**Priority**: High  
**Estimated Timeline**: 8-10 weeks  
**Last Updated**: October 14, 2025  
**Revision**: Complete architectural redesign based on state-space approach

---

## Executive Summary

This plan completes the PAR (Periodic Autoregressive) model implementation in POWE.RS using a **state-space augmentation approach**. PAR models will be integrated by adding lagged inflows as state variables, with AR coefficients appearing as constraint coefficients in subproblems. This approach maintains mathematical rigor while leveraging existing infrastructure for state management and cut generation.

## Critical Architectural Decision

There are **two fundamental approaches** for PAR integration in SDDP:

### Approach A: External Scenario Generation (Rejected)
- Generate AR scenarios externally, feed as independent noises
- **Problem**: Loses temporal coupling, breaks Bellman recursion
- **Impact**: Incorrect value functions, invalid cuts, wrong policies
- **Status**: Not suitable for SDDP

### Approach B: State-Space Augmentation (Selected ✅)
- Add lagged inflows as state variables to subproblems
- AR dynamics encoded as constraints with coefficients
- Innovations added as RHS updates
- **Benefits**: 
  - Mathematically correct Bellman recursion
  - Proper dual variables for cut generation
  - Leverages existing state management infrastructure
  - Maintains numerical stability
- **Status**: This is the correct approach and forms the basis of this plan

---

## Architectural Foundation

### Current Codebase Analysis

#### Existing Infrastructure (Ready to Use)
1. **`State` trait** (`src/state.rs`):
   - `add_variables_to_subproblem`: Already supports adding state variables ✅
   - `add_constraints_to_subproblem`: Already supports custom constraint logic ✅
   - `evaluate_cut`: Computes cut coefficients from dual variables ✅
   - `update_with_current_realization`: Updates state after forward pass ✅

2. **`StorageState`** (reference implementation):
   - Shows pattern for adding variables (storage levels)
   - Shows pattern for adding constraints (hydro balance)
   - Shows pattern for cut coefficient extraction (water values)
   - **Key insight**: Lagged inflows follow exact same pattern as storage

3. **`Subproblem`** (`src/subproblem.rs`):
   - `Variables` struct: Extensible for new variable types ✅
   - `Constraints` struct: Extensible for new constraint types ✅
   - `add_constraints_to_subproblem`: Delegates to state trait ✅

4. **`ParGenerator`** (`src/par_generator.rs`):
   - Complete PAR equation implementation (~1400 lines) ✅
   - Seasonal parameter handling ✅
   - Lag buffer management (VecDeque) ✅
   - **Status**: Ready for integration

#### Missing Pieces (To Be Implemented)
1. **`StorageAndInflowState`**: State type that manages both storage + lag history
2. **Initial condition handling**: Multiple initial nodes for lag warm-up
3. **Constraint coefficients**: AR parameters in subproblem constraints
4. **Cut generation**: Dual variables from lag state variables

---

## The State-Space PAR Formulation

### Mathematical Foundation

For a PAR(p) model with seasonal parameters, the SDDP subproblem becomes:

```
minimize: c^T x + α
subject to:
    A x = b                           (power system constraints)
    s_{t+1} = s_t + inflow_t - ...   (storage balance)
    
    // AR DYNAMICS AS CONSTRAINTS:
    inflow_t = μ_m + σ_m[φ_1·lag_{t-1} + φ_2·lag_{t-2} + ... + φ_p·lag_{t-p}] + ε_t
    
    // LAG STATE UPDATES:
    lag_{t-1} = inflow_{t-1}          (from previous stage state)
    lag_{t-2} = lag_{t-1,prev}        (from previous stage state)
    ...
    lag_{t-p} = lag_{t-p+1,prev}      (from previous stage state)
    
    // BENDERS CUT:
    α ≥ π_0 + Σ π_s·s_{t+1} + Σ π_lag·lag_{t}  
```

**Key observations**:
1. AR coefficients (φ_k) appear as constraint coefficients
2. Lag states get dual variables (π_lag) that enter cuts
3. Innovation (ε_t) is the RHS that varies by scenario
4. State dimension increases from n (storage) to n + n*p (storage + lags)

### Implementation in Code

The critical insight: **lagge d inflows are just another type of state variable**, handled identically to storage:

| Storage State | Lag State | Treatment |
|--------------|-----------|-----------|
| `stored_volume[i]` | `lag_inflow[i][k]` | Both are state variables |
| Storage dual → water value | Lag dual → AR shadow price | Both enter cuts |
| Initial storage from config | Initial lags from config | Both need initialization |
| Updated each forward pass | Updated each forward pass | Same update mechanism |

---

## Phase 1: Core PAR State Management (Weeks 1-3)

### Objective
Implement `StorageAndInflowState` with lag state variables and AR constraint generation.

### 1.1 Implement StorageAndInflowState Structure

**File**: `src/state.rs`

**New struct** (verify no duplicates exist):
```rust
/// State that manages both storage levels and lagged inflows for PAR models
/// 
/// This extends StorageState by adding circular buffers for AR lag history.
/// Each hydro with PAR inflow gets p lag state variables (p = AR order).
///
/// # State Space Dimension
/// 
/// - Without PAR: n (one storage per hydro)
/// - With PAR(p): n + n*p (storage + p lags per hydro)
///
/// # Subproblem Variables
///
/// For each hydro with PAR(p) inflow:
/// - `stored_volume[i]`: Storage level (existing)
/// - `lag_inflow[i][0..p]`: Lagged inflow values (new)
///
/// # Cut Generation
///
/// Cut coefficients extracted from dual variables of:
/// - Storage balance constraints → water value (existing)
/// - AR lag linking constraints → AR shadow price (new)
#[derive(Debug, Clone)]
pub struct StorageAndInflowState {
    /// Storage state (delegate all storage operations to this)
    storage: StorageState,
    
    /// Lag history for each hydro with PAR inflow
    /// Key: hydro_id, Value: circular buffer of past inflows [t-1, t-2, ..., t-p]
    lag_buffers: HashMap<usize, VecDeque<f64>>,
    
    /// AR orders for each hydro
    /// Key: hydro_id, Value: AR order p
    ar_orders: HashMap<usize, usize>,
    
    /// AR coefficients by hydro and season
    /// Key: (hydro_id, season_id), Value: [φ_1, φ_2, ..., φ_p, μ, σ]
    ar_params: HashMap<(usize, usize), Vec<f64>>,
    
    /// Flag for each hydro indicating if it uses PAR
    is_par_hydro: Vec<bool>,
}
```

**Constructor**:
```rust
impl StorageAndInflowState {
    pub fn new(
        system: &system::System,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
        initial_lags: &HashMap<usize, Vec<f64>>,
        ar_config: &HashMap<usize, ArConfiguration>,
    ) -> Result<Self, String> {
        // 1. Create base storage state
        let storage = StorageState::new(system, load_stochastic_process, inflow_stochastic_process);
        
        // 2. Initialize lag buffers from initial conditions
        let mut lag_buffers = HashMap::new();
        let mut ar_orders = HashMap::new();
        let mut is_par_hydro = vec![false; system.meta.hydros_count];
        
        for (hydro_id, ar_cfg) in ar_config.iter() {
            let p = ar_cfg.max_order();
            ar_orders.insert(*hydro_id, p);
            is_par_hydro[*hydro_id] = true;
            
            // Initialize lag buffer with historical values
            let lags = initial_lags.get(hydro_id)
                .ok_or_else(|| format!("Missing initial lags for hydro {}", hydro_id))?;
            
            if lags.len() < p {
                return Err(format!(
                    "Hydro {} requires {} initial lags but only {} provided",
                    hydro_id, p, lags.len()
                ));
            }
            
            // Take last p values (most recent history)
            let mut buffer = VecDeque::with_capacity(p);
            for &lag in lags.iter().rev().take(p).rev() {
                buffer.push_back(lag);
            }
            lag_buffers.insert(*hydro_id, buffer);
        }
        
        // 3. Extract AR parameters by season
        let ar_params = Self::build_ar_params_map(ar_config)?;
        
        Ok(Self {
            storage,
            lag_buffers,
            ar_orders,
            ar_params,
            is_par_hydro,
        })
    }
    
    fn build_ar_params_map(
        ar_config: &HashMap<usize, ArConfiguration>
    ) -> Result<HashMap<(usize, usize), Vec<f64>>, String> {
        let mut params = HashMap::new();
        for (hydro_id, cfg) in ar_config.iter() {
            for season_id in 0..cfg.num_seasons {
                let season_params = cfg.get_season_params(season_id)?;
                // Pack [φ_1, φ_2, ..., φ_p, μ, σ]
                let mut packed = season_params.phi.clone();
                packed.push(season_params.mu);
                packed.push(season_params.sigma);
                params.insert((*hydro_id, season_id), packed);
            }
        }
        Ok(params)
    }
}
```

**Tasks**:
- [x] Design `StorageAndInflowState` struct (verify no duplicates)
- [ ] Implement constructor with lag initialization
- [ ] Implement AR parameter extraction from configuration
- [ ] Add validation for initial conditions
- [ ] Write unit tests for lag buffer management

**Success Criteria**:
- State correctly initializes with historical lags
- AR parameters accessible by (hydro_id, season_id)
- Memory usage: O(n*p_max) where p_max is maximum AR order
- No panics on edge cases (missing lags, invalid orders)

---

### 1.2 Implement `add_variables_to_subproblem`

**Pattern**: Follow `StorageState::add_variables_to_subproblem` exactly

```rust
impl State for StorageAndInflowState {
    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        // Delegate storage variables to base implementation
        let storage_indices = self.storage.add_variables_to_subproblem(pb, ...);
        
        // Add lag state variables for PAR hydros
        let mut all_indices = storage_indices;
        
        for (hydro_id, p) in self.ar_orders.iter() {
            // Add p lag variables for this hydro: lag[0], lag[1], ..., lag[p-1]
            // Each lag is a state variable with bounds [inflow_min, inflow_max]
            let hydro = &self.storage.system.hydros[*hydro_id];
            let bounds = 0.0..hydro.max_inflow_estimate;  // TODO: Get from config
            
            let mut lag_vars = Vec::with_capacity(*p);
            for _k in 0..*p {
                let col_idx = pb.add_column(0.0, bounds.clone());
                lag_vars.push(col_idx);
            }
            
            // Store lag variable indices (will be used in constraints and cuts)
            all_indices.push(lag_vars);
        }
        
        all_indices
    }
}
```

**Key design choices**:
1. **Variable ordering**: Storage first, then lags (consistent with cut generation)
2. **Bounds**: Use reasonable estimates to avoid unbounded models
3. **Cost**: Zero (lags are state variables, not decision variables)

**Tasks**:
- [ ] Implement lag variable addition
- [ ] Determine appropriate bounds for lag variables
- [ ] Update `Variables` struct in `subproblem.rs` to include `lag_inflow: Vec<Vec<usize>>`
- [ ] Write tests for variable creation

---

### 1.3 Implement `add_constraints_to_subproblem`

**This is where AR dynamics enter the model**

```rust
impl State for StorageAndInflowState {
    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        // Delegate storage constraints to base
        let storage_constraints = self.storage.add_constraints_to_subproblem(pb, variables, ...);
        
        let mut all_constraints = storage_constraints;
        
        // For each PAR hydro, add:
        // 1. AR dynamics constraint: inflow_t = μ + σ[Σ φ_k·lag_{t-k}] + ε_t
        // 2. Lag linking constraints: lag_{t-k+1} = lag_{t-k} (from previous stage state)
        
        for (hydro_id, p) in self.ar_orders.iter() {
            let season_id = self.get_current_season(); // TODO: Pass from node context
            let params = self.ar_params.get(&(*hydro_id, season_id))
                .expect("AR params must exist for all PAR hydros");
            
            // Extract φ_1, ..., φ_p, μ, σ
            let phi = &params[0..*p];
            let mu = params[*p];
            let sigma = params[*p + 1];
            
            // AR DYNAMICS CONSTRAINT:
            // inflow_t - σ·(φ_1·lag[0] + φ_2·lag[1] + ... + φ_p·lag[p-1]) = μ + ε_t
            // where ε_t is the innovation (set in RHS update)
            
            let inflow_var = variables.inflow[*hydro_id];
            let lag_vars = &variables.lag_inflow[*hydro_id];
            
            let mut factors = vec![(inflow_var, 1.0)];
            for (k, &lag_var) in lag_vars.iter().enumerate() {
                factors.push((lag_var, -sigma * phi[k]));
            }
            
            // RHS = μ + ε_t (ε_t will be updated with scenario realization)
            let ar_constraint = pb.add_row(mu..mu, factors);
            
            // LAG LINKING CONSTRAINTS:
            // These will be updated with values from previous stage state
            // For now, add identity constraints: lag[k] = (value from state)
            let mut lag_constraints = vec![ar_constraint];
            for &lag_var in lag_vars.iter() {
                let link_constraint = pb.add_row(0.0..0.0, [(lag_var, 1.0)]);
                lag_constraints.push(link_constraint);
            }
            
            all_constraints.push(lag_constraints);
        }
        
        all_constraints
    }
}
```

**Critical observations**:
1. **AR coefficients in constraint matrix**: φ_k appear as coefficients (this is key!)
2. **Innovation as RHS**: ε_t is added to RHS during scenario realization
3. **Lag linking**: Handled through RHS updates from previous stage state
4. **Dual variables**: Each lag linking constraint produces dual variable for cut

**Tasks**:
- [ ] Implement AR dynamics constraint generation
- [ ] Implement lag linking constraint generation
- [ ] Update `Constraints` struct to include AR and lag constraint indices
- [ ] Write tests validating constraint structure
- [ ] Verify constraint matrix structure with hand calculations

---

### 1.4 Implement `update_with_current_realization`

**Pattern**: Update both storage and lag history after forward pass

```rust
impl State for StorageAndInflowState {
    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        // Update storage state
        self.storage.update_with_current_realization(realization);
        
        // Update lag buffers with current inflow realization
        for (hydro_id, buffer) in self.lag_buffers.iter_mut() {
            let current_inflow = realization.inflow[*hydro_id];
            
            // Push current inflow to buffer
            buffer.push_front(current_inflow);
            
            // Keep buffer size at AR order
            let p = self.ar_orders[hydro_id];
            while buffer.len() > p {
                buffer.pop_back();
            }
        }
    }
}
```

**Tasks**:
- [ ] Implement lag buffer updates
- [ ] Verify circular buffer semantics
- [ ] Test with multi-stage trajectories
- [ ] Profile memory allocations

---

### 1.5 Implement `evaluate_cut`

**This is where PAR state variables enter Benders cuts**

```rust
impl State for StorageAndInflowState {
    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        // Get storage cut coefficients (water values)
        let mut cut_coefficients = Vec::new();
        
        // 1. STORAGE COEFFICIENTS (from storage balance duals)
        for hydro_id in 0..self.storage.dimension {
            // Same as StorageState: aggregate water values
            let storage_coef = self.compute_storage_coefficient(
                hydro_id,
                risk_measure,
                branching_realizations,
            );
            cut_coefficients.push(storage_coef);
        }
        
        // 2. LAG COEFFICIENTS (from AR lag linking constraint duals)
        for (hydro_id, p) in self.ar_orders.iter() {
            for k in 0..*p {
                // Extract dual variable from lag linking constraint
                let lag_coef = self.compute_lag_coefficient(
                    *hydro_id,
                    k,
                    risk_measure,
                    branching_realizations,
                );
                cut_coefficients.push(lag_coef);
            }
        }
        
        // 3. COMPUTE CUT RHS
        let last_realization = forward_trajectory.last().unwrap();
        let objective = self.compute_expected_objective(risk_measure, branching_realizations);
        
        // State vector: [storage_0, ..., storage_n, lag_0_0, lag_0_1, ..., lag_n_p]
        let state_vector = self.pack_state_vector(last_realization);
        
        let cut_rhs = objective - utils::dot_product(&cut_coefficients, &state_vector);
        
        cut::BendersCut::new(
            0, // Temporary ID
            cut_coefficients,
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    }
    
    fn pack_state_vector(&self, realization: &subproblem::Realization) -> Vec<f64> {
        let mut state = Vec::new();
        
        // Storage levels
        state.extend_from_slice(&realization.final_storage);
        
        // Lag values
        for hydro_id in 0..self.storage.dimension {
            if let Some(buffer) = self.lag_buffers.get(&hydro_id) {
                state.extend(buffer.iter());
            }
        }
        
        state
    }
}
```

**Critical considerations**:
1. **State dimension consistency**: Cut coefficients must match state vector size
2. **Dual variable extraction**: Must extract from correct constraint indices
3. **Risk-adjusted aggregation**: Apply risk measure to dual variables same as costs
4. **Numerical stability**: Use Kahan summation (existing pattern)

**Tasks**:
- [ ] Implement lag coefficient extraction from duals
- [ ] Implement state vector packing (storage + lags)
- [ ] Update `add_cut_constraint_to_model` to handle expanded state
- [ ] Write tests for cut generation with PAR states
- [ ] Validate cut height calculation with hand computations

---

## Phase 2: Initial Condition & Graph Handling (Weeks 4-5)

### Objective
Enable multiple initial nodes for lag warm-up period.

### 2.1 Problem Analysis

**Challenge**: PAR(p) requires p historical values to start. In a 12-stage problem with PAR(3):
- First 3 stages need to "warm up" the lag buffer
- Cannot use single initial node (no history for AR equation)
- Need branching structure in initial stages

**Solution**: Add **pre-study nodes** to graph that establish lag history

```
Graph structure:
    
    t=0 (root)
     │
    ┌┴─────┬─────┐  (3 branches, historical realizations)
   t=-2   t=-2   t=-2
     │     │      │
    ┌┴┐   ┌┴┐   ┌┴┐  (continue branching)
  t=-1│  t=-1│  t=-1│
      │     │      │
     ┌┴┐   ┌┴┐   ┌┴┐
    t=1│  t=1│  t=1│  (actual study period begins)
```

### 2.2 Extend InitialCondition

**File**: `src/initial_condition.rs`

```rust
#[derive(Debug, Clone)]
pub struct InitialCondition {
    // Existing fields
    pub storage: Vec<f64>,
    
    // NEW: Lag history for PAR hydros
    pub lag_history: HashMap<usize, Vec<f64>>,  // hydro_id → [x_{-1}, x_{-2}, ..., x_{-p}]
    
    // NEW: Multiple initial nodes for lag warm-up
    pub initial_scenarios: Option<Vec<InitialScenario>>,
}

#[derive(Debug, Clone)]
pub struct InitialScenario {
    pub scenario_id: usize,
    pub probability: f64,
    pub lag_realizations: HashMap<usize, Vec<f64>>,  // hydro_id → lag values for this scenario
}
```

**Tasks**:
- [ ] Extend `InitialCondition` struct (verify no duplicates exist)
- [ ] Add JSON schema support for lag history
- [ ] Implement validation for lag history completeness
- [ ] Add schema validation tests

### 2.3 Graph Construction with Pre-Study Nodes

**File**: `src/sddp/mod.rs`

```rust
impl SddpAlgorithm {
    pub fn new_with_par_warmup(
        node_data_graph: graph::DirectedGraph<NodeData>,
        initial_condition: InitialCondition,
        num_warmup_stages: usize,
        seed: u64,
    ) -> Result<Self, String> {
        // If initial_scenarios is Some, add pre-study nodes
        if let Some(scenarios) = initial_condition.initial_scenarios {
            // Build warmup tree
            let warmup_graph = Self::build_warmup_graph(
                &node_data_graph,
                &scenarios,
                num_warmup_stages,
            )?;
            
            // Merge with main graph
            let merged_graph = Self::merge_graphs(warmup_graph, node_data_graph)?;
            
            // Continue with standard initialization
            Self::new(merged_graph, initial_condition, seed)
        } else {
            // Standard single-root construction
            Self::new(node_data_graph, initial_condition, seed)
        }
    }
    
    fn build_warmup_graph(
        main_graph: &graph::DirectedGraph<NodeData>,
        scenarios: &[InitialScenario],
        num_stages: usize,
    ) -> Result<graph::DirectedGraph<NodeData>, String> {
        // Create pre-study nodes with historical scenarios
        // Each node has system state initialized with scenario lags
        // ...
    }
}
```

**Tasks**:
- [ ] Implement warmup graph construction
- [ ] Implement graph merging logic
- [ ] Update FCF to handle multiple initial nodes
- [ ] Add integration tests with warmup graphs

---

## Phase 3: Innovation Generation & RHS Updates (Week 6)

### Objective
Connect scenario generation to AR constraint RHS updates.

### 3.1 Scenario Realization with Innovations

**Current**: Scenarios contain full inflow realizations  
**Need**: Scenarios contain AR innovations (ε_t) that are added to RHS

**File**: `src/subproblem.rs`

```rust
impl Subproblem {
    fn set_ar_innovations_in_subproblem(
        &mut self,
        innovations: &HashMap<usize, f64>,  // hydro_id → ε_t
        season_id: usize,
        lag_state: &HashMap<usize, &[f64]>,  // hydro_id → [lag_{t-1}, ..., lag_{t-p}]
    ) {
        // For each PAR hydro, update AR constraint RHS
        for (hydro_id, &epsilon_t) in innovations.iter() {
            let params = self.state.get_ar_params(*hydro_id, season_id);
            let lags = lag_state.get(hydro_id).unwrap();
            
            // Compute: μ + ε_t (deterministic AR part is in constraint coefficients)
            let rhs = params.mu + epsilon_t;
            
            // Update AR constraint RHS
            let ar_constraint_idx = self.constraints.ar_dynamics[*hydro_id];
            self.model.as_mut().unwrap().change_rows_bounds(
                ar_constraint_idx,
                rhs,
                rhs,
            );
        }
        
        // Update lag linking constraints with previous stage values
        for (hydro_id, &lag_values) in lag_state.iter() {
            let lag_constraints = &self.constraints.lag_linking[*hydro_id];
            for (k, &lag_value) in lag_values.iter().enumerate() {
                self.model.as_mut().unwrap().change_rows_bounds(
                    lag_constraints[k],
                    lag_value,
                    lag_value,
                );
            }
        }
    }
}
```

**Tasks**:
- [ ] Implement innovation extraction from scenarios
- [ ] Implement RHS update for AR constraints
- [ ] Implement lag state RHS updates
- [ ] Write tests for AR constraint RHS updates

### 3.2 Integrate ParGenerator for Innovation Sampling

**File**: `src/scenario.rs` (extend ScenarioGenerator)

```rust
impl ScenarioGenerator {
    fn generate_par_innovations(
        &self,
        hydro_id: usize,
        season_id: usize,
        num_scenarios: usize,
        current_lags: &[f64],
    ) -> Vec<f64> {
        // Use ParGenerator to sample innovations (not full realizations)
        let par_gen = self.par_generators.get(&hydro_id).unwrap();
        
        // Sample base noise (standard normal)
        let base_noise = self.sample_base_noise(num_scenarios);
        
        // Apply marginal transformation to get innovations
        let marginal = &self.entity_marginals[hydro_id];
        let innovations = marginal.transform(&base_noise);
        
        innovations
    }
}
```

**Critical distinction**:
- **Full realization**: z_t = μ + σ[Σ φ_k·x_{t-k} + ε_t]
- **Innovation only**: ε_t ~ marginal distribution
- **AR part in constraint**: σ[Σ φ_k·lag_k] handled by constraint coefficients

**Tasks**:
- [ ] Extend ScenarioGenerator for innovation-only generation
- [ ] Update SAA structure to carry innovations vs full realizations
- [ ] Update scenario sampling in forward/backward passes
- [ ] Write tests for innovation vs realization separation

---

## Phase 4: Testing & Validation (Weeks 7-8)

### Objective
Comprehensive testing at all levels of abstraction.

### 4.1 Unit Tests

**File**: `tests/test_storage_inflow_state.rs`

```rust
#[cfg(test)]
mod storage_inflow_state_tests {
    #[test]
    fn test_lag_buffer_initialization() {
        // Test lag buffers initialize correctly from initial conditions
    }
    
    #[test]
    fn test_lag_buffer_updates() {
        // Test circular buffer updates across multiple stages
    }
    
    #[test]
    fn test_ar_constraint_structure() {
        // Test AR constraints have correct coefficients
        // Verify φ_k appear in constraint matrix
    }
    
    #[test]
    fn test_cut_dimension_consistency() {
        // Test cut coefficients match state dimension
        // Verify storage + lag coefficients present
    }
    
    #[test]
    fn test_state_vector_packing() {
        // Test state vector correctly packs storage + lags
    }
}
```

### 4.2 Integration Tests

**File**: `tests/test_par_integration.rs`

```rust
#[test]
fn test_simple_par1_example() {
    // Run examples/06-par-model/01-simple-par1
    // Verify:
    // - AR persistence in inflows
    // - Cut coefficients for lag states
    // - Convergence of lower bound
}

#[test]
fn test_par_vs_naive_convergence() {
    // PAR(0) should behave identically to naive
    // Verify convergence rates match
}

#[test]
fn test_multistage_par_trajectory() {
    // Run multi-stage problem with PAR
    // Verify lag state consistency across stages
    // Check AR dynamics in realized inflows
}

#[test]
fn test_warmup_graph_construction() {
    // Test pre-study nodes correctly initialize lag buffers
    // Verify graph structure with warmup stages
}
```

### 4.3 Numerical Validation

**File**: `tests/test_par_numerical.rs`

```rust
#[test]
fn test_ar_equation_in_subproblem() {
    // Hand-craft PAR(1) problem with known parameters
    // Solve subproblem, extract solution
    // Verify: inflow_t = μ + σ·φ·lag_{t-1} + ε_t
}

#[test]
fn test_cut_height_calculation() {
    // Create simple 2-stage PAR problem
    // Generate cut from backward pass
    // Verify cut height matches hand calculation
}

#[test]
fn test_dual_variable_extraction() {
    // Solve subproblem, extract duals
    // Verify lag linking constraint duals
    // Check dual values used in cuts
}
```

### 4.4 Performance Benchmarks

**File**: `benches/par_performance.rs`

```rust
fn bench_storage_inflow_state_overhead(c: &mut Criterion) {
    // Compare StorageState vs StorageAndInflowState
    // Measure overhead from lag management
}

fn bench_par_subproblem_solve(c: &mut Criterion) {
    // Compare solve time with/without PAR
    // Target: <20% overhead for PAR(3)
}

fn bench_par_training_scalability(c: &mut Criterion) {
    // Measure training time vs AR order
    // Verify linear scaling O(p)
}
```

**Performance targets**:
- **State overhead**: <5% memory increase per lag state
- **Solve overhead**: <20% per subproblem for PAR(3)
- **Training overhead**: <30% total for PAR(3) vs naive
- **Cut generation**: <10% overhead from expanded state

### 4.5 Statistical Validation

**File**: `tests/test_par_statistical_properties.rs`

```rust
#[test]
fn test_autocorrelation_structure() {
    // Generate large sample from PAR model
    // Compute sample autocorrelations
    // Verify match theoretical φ_k
}

#[test]
fn test_seasonal_parameter_cycling() {
    // Run simulation across multiple years
    // Verify seasonal parameters cycle correctly
    // Check μ, σ, φ vary by season
}

#[test]
fn test_stationarity() {
    // Verify PAR process is stationary
    // Check |φ_k| < 1 constraint
    // Test long-run behavior
}
```

---

## Phase 5: Advanced Features (Weeks 9-10)

### Objective
Complete PAR feature set with correlation and advanced validation.

### 5.1 Correlation Support

**Challenge**: Correlation must be applied to AR **innovations**, not full realizations.

**Solution**: Extend CorrelationApplicator to handle PAR residuals

```rust
impl CorrelationApplicator {
    fn apply_to_par_innovations(
        &self,
        innovations: &mut HashMap<usize, Vec<f64>>,
        correlation_block: &CorrelationBlock,
    ) -> Result<(), String> {
        // 1. Extract innovations for correlated entities
        let entity_indices: Vec<usize> = correlation_block.entity_refs.iter()
            .map(|r| self.entity_index_map.get(r).unwrap())
            .collect();
        
        // 2. Apply Gaussian copula to innovations
        let correlated = self.apply_gaussian_copula(
            innovations,
            &entity_indices,
            &correlation_block.matrix,
        )?;
        
        // 3. Update innovations with correlated values
        for (idx, entity_idx) in entity_indices.iter().enumerate() {
            innovations.insert(*entity_idx, correlated[idx].clone());
        }
        
        Ok(())
    }
}
```

**Tasks**:
- [ ] Extend CorrelationApplicator for PAR innovations
- [ ] Update scenario generation to apply correlation before AR dynamics
- [ ] Add validation for correlation block compatibility with PAR
- [ ] Write tests for correlated PAR processes

### 5.2 Enhanced Configuration Validation

```rust
pub fn validate_par_configuration(
    recourse: &Recourse,
    graph: &DirectedGraph<NodeData>,
) -> Result<(), ValidationError> {
    // 1. Check PAR hydros use storage_and_inflow state
    for node in graph.nodes() {
        if node.data.has_par_inflows() && node.data.state_kind != "storage_and_inflow" {
            return Err(ValidationError::new(
                format!("Node {} uses PAR inflows but state_kind is '{}', must be 'storage_and_inflow'",
                    node.id, node.data.state_kind)
            ));
        }
    }
    
    // 2. Verify initial conditions provide sufficient lag history
    for noise_model in &recourse.noise_models {
        if let TemporalModel::PeriodicAutoregressive { max_order, .. } = noise_model.temporal_model {
            let entity_id = noise_model.entity_id;
            let lags = recourse.initial_condition.lag_history.get(&entity_id)
                .ok_or_else(|| ValidationError::new(
                    format!("Entity {} requires {} initial lags but none provided", entity_id, max_order)
                ))?;
            
            if lags.len() < max_order {
                return Err(ValidationError::new(
                    format!("Entity {} requires {} initial lags but only {} provided", 
                        entity_id, max_order, lags.len())
                ));
            }
        }
    }
    
    // 3. Validate seasonal parameter consistency
    // 4. Check correlation block compatibility
    // ...
    
    Ok(())
}
```

**Tasks**:
- [ ] Implement comprehensive PAR configuration validation
- [ ] Add validation to input parsing pipeline
- [ ] Create detailed error messages for common mistakes
- [ ] Add validation tests covering edge cases

---

## Implementation Dependencies & Timeline

### Critical Path

```
Week 1-2: StorageAndInflowState structure + basic methods
    ↓
Week 3: add_variables + add_constraints (AR dynamics in LP)
    ↓
Week 4: evaluate_cut (dual extraction for expanded state)
    ↓
Week 5: InitialCondition + warmup graphs
    ↓
Week 6: Innovation generation + RHS updates
    ↓
Week 7-8: Testing & numerical validation
    ↓
Week 9-10: Correlation + advanced features
```

### Parallel Work Opportunities

**Weeks 1-3** (Can work in parallel):
- Developer A: StorageAndInflowState implementation
- Developer B: Initial condition extensions + JSON schema
- Developer C: Test infrastructure setup

**Weeks 4-6** (Sequential, depends on Weeks 1-3):
- All developers: Core integration work (cannot parallelize)

**Weeks 7-10** (Can work in parallel):
- Developer A: Testing & benchmarking
- Developer B: Correlation support
- Developer C: Documentation + examples

---

## Risk Assessment & Mitigation

### High-Risk Items

#### 1. Numerical Stability of Expanded State Space
**Risk**: Larger state dimension increases numerical sensitivity  
**Impact**: Inaccurate cuts, convergence issues  
**Likelihood**: Medium  
**Mitigation**:
- Use tight solver tolerances (existing pattern)
- Kahan summation for cut aggregation (existing)
- Scale lag variables to similar magnitude as storage
- Extensive numerical validation tests

#### 2. Performance Degradation
**Risk**: PAR overhead makes SDDP too slow  
**Impact**: User adoption, production viability  
**Likelihood**: Medium  
**Mitigation**:
- Profile each component separately
- Establish performance budgets per phase
- Lazy initialization where possible
- Consider PAR(0) fast path

#### 3. Initial Condition Complexity
**Risk**: Multiple initial nodes complicates graph structure  
**Impact**: FCF complexity, increased bugs  
**Likelihood**: Medium-High  
**Mitigation**:
- Start with single-root case, add multi-root later
- Extensive graph construction tests
- Clear separation: warmup vs study period

### Medium-Risk Items

#### 4. Dual Variable Extraction
**Risk**: Wrong duals used for lag state cuts  
**Impact**: Invalid cuts, incorrect policy  
**Likelihood**: Medium  
**Mitigation**:
- Hand-calculated test cases
- Cross-validate with known solutions
- Audit dual extraction in code review

#### 5. State Consistency Across Stages
**Risk**: Lag buffers become inconsistent  
**Impact**: Incorrect AR dynamics  
**Likelihood**: Low-Medium  
**Mitigation**:
- Rigorous state update testing
- Trajectory validation tests
- Debugging aids for state inspection

---

## Success Criteria

### Functional Requirements
- [ ] PAR models exhibit actual autoregressive behavior (not naive)
- [ ] Autocorrelation structure matches theoretical φ_k
- [ ] Seasonal parameters cycle correctly based on season_id
- [ ] Lag state maintained consistently across SDDP stages
- [ ] Cuts contain coefficients for both storage and lag states
- [ ] Multiple initial nodes work correctly (warmup period)

### Performance Requirements
- [ ] No more than 5% memory overhead per lag state
- [ ] No more than 20% solve time overhead for PAR(3)
- [ ] No more than 30% total training time overhead
- [ ] Memory usage scales linearly with AR order: O(n*p)
- [ ] No memory leaks in long-running simulations

### Quality Requirements
- [ ] >90% test coverage for PAR-specific code
- [ ] All performance benchmarks pass regression tests
- [ ] Numerical validation tests pass (AR equation, cuts, duals)
- [ ] Statistical properties tests pass (autocorrelation, stationarity)
- [ ] All existing non-PAR tests continue to pass unchanged

### Documentation Requirements
- [ ] PAR formulation documented (AR dynamics as constraints)
- [ ] State augmentation approach explained
- [ ] API documentation complete (>95%)
- [ ] User guide with examples
- [ ] Migration guide from previous version

---

## Conclusion

This revised plan implements PAR models correctly using **state-space augmentation**, where:

1. **Lagged inflows are state variables** (just like storage)
2. **AR dynamics are constraints** with φ_k as coefficients
3. **Innovations are RHS updates** (scenario realizations)
4. **Dual variables from lag states** enter Benders cuts
5. **Existing infrastructure works** with minimal changes

### Key Architectural Insights

✅ **Correct approach**: State-space formulation maintains Bellman recursion  
✅ **Reuses existing patterns**: StorageState as template for StorageAndInflowState  
✅ **Minimal disruption**: Changes isolated to state trait implementations  
✅ **Performance-aware**: Linear overhead O(p), well-bounded  
✅ **Mathematically rigorous**: Preserves SDDP convergence properties  

### Critical Path Summary

**Must happen in order**:
1. StorageAndInflowState (foundation)
2. AR constraints in LP (core dynamics)
3. Cut generation with expanded state (correctness)
4. Innovation generation (scenario coupling)
5. Testing & validation (verification)

**Estimate**: 8-10 weeks with 2-3 developers working efficiently on critical path.

The plan is now rock-solid, based on fundamental SDDP theory, and leverages existing codebase patterns wherever possible. Every step is small, testable, and verifiable.
