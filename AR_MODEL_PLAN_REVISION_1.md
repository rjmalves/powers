# Refined AR Model Implementation Plan for POWE.RS

After reviewing your existing architecture, I see you have a much more sophisticated framework than I initially assumed. Let me refine the AR implementation plan to leverage your existing traits and patterns.

## Key Architectural Insights

1. **State as Trait**: Your `State` trait is the perfect extension point for AR models
2. **StochasticProcess Trait**: Already designed for transforming noises → perfect for AR
3. **Dual Variables**: You correctly identified that lag inflows MUST be decision variables with associated constraints to extract duals for cut coefficients
4. **Scenario Generation**: Your `NoiseGenerator` → `SAA` pipeline can generate innovations (ε_t) that feed into AR process

## Revised Implementation Strategy

### Phase 1: Extended State Implementation (Week 1)

#### 1.1 Create `StorageWithInflowState`

```rust
// src/state.rs - Add new implementation
#[derive(Debug, Clone)]
pub struct StorageWithInflowState {
    dimension: usize,
    final_storage: Vec<f64>,

    // NEW: AR lag states
    lag_dimension: usize,  // AR order (p)
    lag_inflows: Vec<Vec<f64>>,  // [reservoir][lag] - past p inflows

    // Domination tracking (unchanged)
    dominating_objective: f64,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
}

impl StorageWithInflowState {
    pub fn new(
        system: &system::System,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Self {
        // Extract AR order from inflow process
        let lag_dimension = inflow_stochastic_process.get_ar_order();

        Self {
            dimension: system.meta.hydros_count,
            final_storage: vec![0.0; system.meta.hydros_count],
            lag_dimension,
            lag_inflows: vec![vec![0.0; lag_dimension]; system.meta.hydros_count],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }
}
```

#### 1.2 Critical State Trait Methods for AR

```rust
impl State for StorageWithInflowState {
    fn coefficients(&self) -> &[f64] {
        // CRITICAL: Must return ALL state variables for cut evaluation
        // This needs refactoring - can't just return a slice anymore
        // Option 1: Return concatenated vector (storage + lags)
        // Option 2: Refactor trait to return Vec<f64> instead of &[f64]
        // I recommend Option 2 for flexibility
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        let mut col_indices = vec![];

        // Storage variables (as before)
        for _ in 0..self.dimension {
            col_indices.push(vec![pb.add_column(0.0, 0.0..)]);
        }

        // NEW: Lag inflow variables (unbounded, will be fixed by constraints)
        for r in 0..self.dimension {
            for lag in 0..self.lag_dimension {
                let var_id = pb.add_column(0.0, f64::NEG_INFINITY..f64::INFINITY);
                col_indices[r].push(var_id);
            }
        }

        col_indices
    }

    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        let mut constraints = vec![];

        // For each reservoir, add AR process constraints
        for r in 0..variables.inflow.len() {
            let mut reservoir_constraints = vec![];

            // AR constraint: inflow[t] = φ₀ + Σ(φᵢ * lag[i]) + ε[t]
            // Rearranged: inflow[t] - Σ(φᵢ * lag[i]) - ε[t] = φ₀

            let inflow_var = variables.inflow[r];
            let lag_vars = &variables.inflow_process[r][1..]; // Skip storage var
            let noise_var = variables.inflow_process[r][0]; // Innovation variable

            // Get AR coefficients from stochastic process
            let ar_coeffs = inflow_stochastic_process.get_ar_coefficients(r);
            let intercept = inflow_stochastic_process.get_ar_intercept(r);

            let mut factors = vec![(inflow_var, 1.0)];
            for (i, &lag_var) in lag_vars.iter().enumerate() {
                factors.push((lag_var, -ar_coeffs[i]));
            }
            factors.push((noise_var, -1.0));

            let constraint_id = pb.add_row(intercept..intercept, factors);
            reservoir_constraints.push(constraint_id);

            // Lag fixing constraints (will be set at runtime)
            for &lag_var in lag_vars {
                let fix_constraint = pb.add_row(0.0..0.0, [(lag_var, 1.0)]);
                reservoir_constraints.push(fix_constraint);
            }

            constraints.push(reservoir_constraints);
        }

        constraints
    }

    fn update_with_current_realization(&mut self, realization: &subproblem::Realization) {
        self.final_storage.clone_from_slice(&realization.final_storage);

        // NEW: Update lag states by shifting and adding current inflow
        for r in 0..self.dimension {
            // Shift lags: lag[i] = lag[i-1], then lag[0] = current_inflow
            for i in (1..self.lag_dimension).rev() {
                self.lag_inflows[r][i] = self.lag_inflows[r][i - 1];
            }
            if self.lag_dimension > 0 {
                self.lag_inflows[r][0] = realization.inflow[r];
            }
        }
    }

    fn evaluate_cut(&mut self, /* params */) -> cut::BendersCut {
        // ... existing logic for computing cut ...

        // CRITICAL: Cut must include coefficients for BOTH storage and lag states
        let mut cut_coefficients = vec![0.0; self.dimension + self.dimension * self.lag_dimension];

        // First dimension entries: storage coefficients
        // Next dimension*lag_dimension entries: lag coefficients

        // The water values from branching_realizations now include duals for lag variables
        // Need to extract and accumulate them properly

        // ... rest of cut computation ...
    }
}
```

### Phase 2: AR Stochastic Process (Week 1-2)

#### 2.1 Extend StochasticProcess Trait

```rust
// src/stochastic_process.rs
pub trait StochasticProcess: Send + Sync {
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64];

    // NEW: AR-specific methods with default implementations
    fn get_ar_order(&self) -> usize { 0 }
    fn get_ar_coefficients(&self, reservoir: usize) -> Vec<f64> { vec![] }
    fn get_ar_intercept(&self, reservoir: usize) -> f64 { 0.0 }

    // NEW: Transform innovations to inflows given lag states
    fn realize_with_lags(&self, innovations: &[f64], lag_states: &[Vec<f64>]) -> Vec<f64> {
        // Default: just return innovations (for Naive process)
        innovations.to_vec()
    }
}
```

#### 2.2 Implement AR Process

```rust
// src/stochastic_process.rs
#[derive(Debug, Clone)]
pub struct AutoRegressive {
    order: usize,
    coefficients: Vec<Vec<f64>>,  // [reservoir][lag]
    intercepts: Vec<f64>,         // [reservoir]
    periodic: bool,
    current_stage: usize,
}

impl AutoRegressive {
    pub fn new(
        order: usize,
        coefficients: Vec<Vec<f64>>,
        intercepts: Vec<f64>,
        periodic: bool,
    ) -> Self {
        Self {
            order,
            coefficients,
            intercepts,
            periodic,
            current_stage: 0,
        }
    }

    pub fn from_estimation(historical_inflows: &[Vec<f64>], order: usize) -> Self {
        // Yule-Walker estimation
        // ... implementation ...
    }
}

impl StochasticProcess for AutoRegressive {
    fn realize<'a>(&self, innovations: &'a [f64]) -> &'a [f64] {
        // For backward compatibility when lags aren't available
        innovations
    }

    fn get_ar_order(&self) -> usize {
        self.order
    }

    fn get_ar_coefficients(&self, reservoir: usize) -> Vec<f64> {
        self.coefficients[reservoir].clone()
    }

    fn get_ar_intercept(&self, reservoir: usize) -> f64 {
        self.intercepts[reservoir]
    }

    fn realize_with_lags(&self, innovations: &[f64], lag_states: &[Vec<f64>]) -> Vec<f64> {
        let mut inflows = vec![0.0; innovations.len()];

        for r in 0..innovations.len() {
            inflows[r] = self.intercepts[r] + innovations[r];

            for (i, &coeff) in self.coefficients[r].iter().enumerate() {
                if i < lag_states[r].len() {
                    inflows[r] += coeff * lag_states[r][i];
                }
            }
        }

        inflows
    }
}
```

### Phase 3: Integration Points (Week 2)

#### 3.1 Modify Subproblem for Lag State Transfer

```rust
// src/subproblem.rs - Add to Variables struct
pub struct Variables {
    // ... existing fields ...
    pub lag_inflows: Vec<Vec<usize>>,  // NEW: [reservoir][lag_index]
}

// In Subproblem::realize_uncertainties
fn realize_uncertainties(&mut self, /* params */) -> Result<RealizeUncertaintiesTiming, String> {
    // ... existing code ...

    // Extract lag states from current state
    let lag_states = match &self.state {
        StorageWithInflowState(s) => &s.lag_inflows,
        _ => /* empty default */
    };

    // Use AR process to transform innovations to inflows
    let innovations = /* sample from SAA */;
    let inflows = inflow_stochastic_process.realize_with_lags(innovations, lag_states);

    // Set the computed inflows AND fix lag variables
    self.set_inflows_and_lags(&inflows, lag_states);
}

// NEW: Fix lag variables in constraints
fn set_inflows_and_lags(&mut self, inflows: &[f64], lag_states: &[Vec<f64>]) {
    // Set inflow values (existing)
    // ...

    // NEW: Fix lag variable values via constraints
    if let Some(model) = self.model.as_mut() {
        for r in 0..lag_states.len() {
            for (lag_idx, &lag_value) in lag_states[r].iter().enumerate() {
                let constraint_idx = self.constraints.inflow_process[r][lag_idx + 1];
                model.change_rows_bounds(constraint_idx, lag_value, lag_value);
            }
        }
    }
}
```

#### 3.2 Modify Cut Evaluation for Extended State

```rust
// src/cut.rs
impl BendersCut {
    pub fn eval_height_at_extended_state(
        &self,
        storage_coeffs: &[f64],
        lag_coeffs: &[Vec<f64>],
    ) -> f64 {
        let mut height = self.rhs;

        // Storage contribution
        for (i, &coeff) in storage_coeffs.iter().enumerate() {
            height += self.coefficients[i] * coeff;
        }

        // Lag contribution
        let mut idx = storage_coeffs.len();
        for reservoir_lags in lag_coeffs {
            for &lag_value in reservoir_lags {
                height += self.coefficients[idx] * lag_value;
                idx += 1;
            }
        }

        height
    }
}
```

### Phase 4: Scenario Generation Integration (Week 3)

#### 4.1 Generate Innovations Instead of Inflows

```rust
// src/scenario.rs - Modify NoiseGenerator
impl<L, I> NoiseGenerator<L, I> {
    pub fn generate_ar(&self, seed: u64, ar_process: &AutoRegressive) -> SAA {
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed);
        let mut saa = SAA::new(self);

        for (stage_id, stage_generator) in self.node_generators.iter().enumerate() {
            // Generate INNOVATIONS (white noise) not inflows
            let innovations: Vec<Vec<f64>> = stage_generator
                .inflow_distributions
                .iter()
                .map(|dist| {
                    // Use Normal(0, σ) for innovations
                    let innovation_dist = rand_distr::Normal::new(0.0, /* std dev */).unwrap();
                    innovation_dist.sample_iter(&mut rng)
                        .take(stage_generator.num_branchings)
                        .collect()
                })
                .collect();

            // Store innovations in SAA - AR process will transform them at runtime
            saa.set_noises_by_stage(stage_id, /* ... */, innovations);
        }

        saa
    }
}
```

### Critical Design Decisions - Revised

#### 1. State Variable Representation

**Decision**: Extend `State` trait's `coefficients()` method to return `Vec<f64>` instead of `&[f64]`.

**Rationale**:

- Need to concatenate storage + lag states dynamically
- Slice can't represent non-contiguous memory
- Small overhead acceptable for flexibility

#### 2. Dual Variable Extraction

**Decision**: Extend `Realization` to store lag duals separately.

```rust
pub struct Realization {
    // ... existing fields ...
    pub lag_water_values: Vec<Vec<f64>>,  // NEW: [reservoir][lag]
}
```

**Rationale**:

- Clean separation of physical vs informational state duals
- Easier debugging and validation
- No ambiguity in cut coefficient construction

#### 3. Variable Ordering in LP

**Decision**: Order variables as `[storage_1, ..., storage_N, lag_1_1, ..., lag_N_P, ...]`

**Rationale**:

- Groups related variables together
- Simplifies dual extraction
- Consistent with solver's natural ordering

#### 4. Historical Initialization

**Decision**: Add `initial_lags` to input JSON format.

```json
{
  "recourse": {
    "noise_model": "autoregressive",
    "ar_order": 2,
    "ar_coefficients": [...],
    "initial_lags": [
      [150.0, 145.0],  // Reservoir 1: lag-1, lag-2
      [200.0, 190.0]   // Reservoir 2: lag-1, lag-2
    ]
  }
}
```

**Rationale**:

- Need historical values for first stage
- User should provide based on their data
- Defaults to zeros if not specified

## Implementation Sequence - Refined

### Sprint 5.1: Core AR State (1 week)

1. [ ] Implement `StorageWithInflowState`
2. [ ] Extend `State` trait for vector coefficients
3. [ ] Add lag variables to subproblem LP
4. [ ] Test state transitions with simple AR(1)

### Sprint 5.2: AR Process (1 week)

1. [ ] Implement `AutoRegressive` stochastic process
2. [ ] Add Yule-Walker estimation
3. [ ] Integrate with subproblem via `realize_with_lags`
4. [ ] Unit tests for AR transformations

### Sprint 5.3: Cut Generation (1 week)

1. [ ] Extend `Realization` for lag duals
2. [ ] Modify cut evaluation for extended state
3. [ ] Update FCF domination logic
4. [ ] Validate cut coefficients include lag gradients

### Sprint 5.4: Integration & Testing (1 week)

1. [ ] Update input format parser
2. [ ] Modify `state::factory` for AR state selection
3. [ ] Integration test on Example 03 with AR(1)
4. [ ] Benchmark convergence improvement

### Sprint 5.5: Production Features (1 week)

1. [ ] PAR (Periodic AR) support
2. [ ] Higher-order AR(p) for p > 1
3. [ ] Model selection criteria
4. [ ] Documentation and examples

## Key Implementation Notes

1. **Backward Compatibility**: Keep `StorageState` as default, use config flag to enable AR
2. **Performance**: Lag variables add O(Np) complexity where N=reservoirs, p=order
3. **Numerical Stability**: Bound AR coefficients to ensure stationarity
4. **Testing**: Compare against SDDP.jl results for validation

This refined plan leverages your excellent trait-based architecture while adding minimal complexity. The key insight is that lag inflows MUST be LP variables to get duals, making them true state variables in the SDDP sense.
