---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §12 (12.1-12.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §13 (13.1-13.4)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §14 (14.1-14.4)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Training Loop

## Purpose

This spec defines the POWE.RS SDDP training loop architecture: the core training structures and trait abstractions, forward pass execution with state management and parallel distribution, and backward pass execution with dual extraction and cut generation.

## 1. SDDP Algorithm Overview

The training phase implements the Stochastic Dual Dynamic Programming (SDDP) algorithm, iteratively constructing piecewise-linear approximations of the expected future cost function (FCF) through forward simulation and backward cut generation.

Each iteration consists of three phases:

1. **Forward pass** — Sample N scenarios, solve the LP at each stage with the current FCF, record visited states and stage-1 costs for the lower bound
2. **Backward pass** — For each stage T down to 2, evaluate the cost-to-go from each visited state under multiple noise realizations, extract LP duals, and compute new cuts via the risk measure
3. **Convergence check** — Update the upper bound estimate (mean forward cost), compute the gap `(UB - LB) / |UB|`, and test stopping rules (gap tolerance, stable LB, iteration/time limits)

The loop terminates when converged or a limit is reached, outputting the FCF cuts and bound history.

## 2. Core Training Structures

```rust
/// Main training orchestrator
pub struct TrainingLoop<R: RiskMeasure, C: CutFormulation, H: HorizonMode> {
    // Algorithm components
    risk_measure: R,
    cut_formulation: C,
    horizon_mode: H,

    // State
    fcf: FutureCostFunction,
    iteration: usize,
    convergence_monitor: ConvergenceMonitor,

    // Configuration
    config: TrainingConfig,

    // MPI context
    comm: WorldCommunicator,
}

/// Training configuration from config.json
pub struct TrainingConfig {
    // Iteration limits
    pub max_iterations: usize,           // e.g., 1000
    pub min_iterations: usize,           // e.g., 10
    pub time_limit_seconds: Option<f64>, // e.g., 3600.0

    // Convergence criteria
    pub gap_tolerance: f64,              // e.g., 0.01 (1%)
    pub stable_iterations: usize,        // e.g., 5

    // Scenario sampling
    pub forward_scenarios: usize,        // e.g., 100
    pub backward_samples: usize,         // e.g., 50 (noise outcomes per state)

    // Cut management
    pub cut_selection: CutSelectionStrategy,
    pub max_cuts_per_stage: Option<usize>,

    // Checkpointing
    pub checkpoint_interval: usize,      // e.g., 10 (iterations)
}

impl<R: RiskMeasure, C: CutFormulation, H: HorizonMode> TrainingLoop<R, C, H> {
    /// Execute the SDDP training loop
    pub fn run(&mut self, case_data: &CaseData) -> TrainingResult {
        let start_time = Instant::now();

        while !self.should_stop(start_time) {
            self.iteration += 1;

            // Forward pass: simulate scenarios, compute lower bound
            let forward_result = self.forward_pass(case_data);

            // Synchronize forward results across ranks
            let global_forward = self.sync_forward_results(&forward_result);

            // Backward pass: generate cuts from visited states
            self.backward_pass(case_data, &global_forward);

            // Synchronize new cuts across ranks
            self.sync_cuts();

            // Update convergence statistics
            self.convergence_monitor.update(&global_forward, &self.fcf);

            // Checkpoint if needed
            if self.iteration % self.config.checkpoint_interval == 0 {
                self.checkpoint(case_data);
            }

            // Log progress
            self.log_iteration();
        }

        self.build_result(start_time)
    }

    fn should_stop(&self, start_time: Instant) -> bool {
        // Check iteration limits
        if self.iteration >= self.config.max_iterations {
            return true;
        }

        // Check time limit
        if let Some(limit) = self.config.time_limit_seconds {
            if start_time.elapsed().as_secs_f64() >= limit {
                return true;
            }
        }

        // Check convergence (only after min_iterations)
        if self.iteration >= self.config.min_iterations {
            if self.convergence_monitor.is_converged(&self.config) {
                return true;
            }
        }

        false
    }
}
```

## 3. Trait Abstractions

SDDP variants are expressed through trait abstractions:

```rust
/// Risk measure determines how cuts are computed from noise outcomes
pub trait RiskMeasure: Send + Sync {
    /// Compute cut coefficients from backward pass duals
    /// Returns (intercept_rhs, gradient_coefficients)
    fn compute_cut(
        &self,
        stage: StageId,
        state: &StatePoint,
        outcomes: &[BackwardOutcome],
        probabilities: &[f64],
    ) -> CutCoefficients;

    /// Name for logging
    fn name(&self) -> &'static str;
}

/// Cut formulation determines the structure of cuts
pub trait CutFormulation: Send + Sync {
    /// Build the cut constraint to add to stage t LP
    fn build_cut_constraint(
        &self,
        cut: &Cut,
        stage_vars: &StageVariables,
    ) -> LinearConstraint;
}

/// Horizon mode determines stage transitions and terminal conditions
pub trait HorizonMode: Send + Sync {
    /// Get successor stage(s) with transition probabilities
    fn successors(&self, stage: StageId) -> Vec<(StageId, f64)>;

    /// Is this the final stage? (terminal value function applies)
    fn is_terminal(&self, stage: StageId) -> bool;

    /// Discount factor for infinite horizon
    fn discount_factor(&self) -> f64;
}
```

## 4. Forward Pass

The forward pass simulates multiple scenarios through the horizon, solving the LP at each stage with the current FCF approximation. Scenarios are distributed across MPI ranks in contiguous blocks; within each rank, scenarios are parallelized across OpenMP threads via `into_par_iter()`. After all ranks complete, `MPI_Allreduce` aggregates global statistics.

```rust
/// Result from a single forward scenario
pub struct ScenarioTrajectory {
    pub scenario_id: ScenarioId,
    pub total_cost: f64,
    pub stage_costs: Vec<f64>,
    pub visited_states: Vec<StatePoint>,  // State at end of each stage
}

/// State vector used for cut generation
pub struct StatePoint {
    pub storage: Vec<f64>,          // Storage level per hydro
    pub inflow_history: Vec<Vec<f64>>, // Lags for PAR model [hydro][lag]
}

impl<R: RiskMeasure, C: CutFormulation, H: HorizonMode> TrainingLoop<R, C, H> {
    /// Execute forward pass for this rank's scenarios
    pub fn forward_pass(&self, case_data: &CaseData) -> ForwardResult {
        let my_scenarios = self.distribute_scenarios();

        // Parallel forward simulation (OpenMP threads)
        let trajectories: Vec<ScenarioTrajectory> = my_scenarios
            .into_par_iter()
            .map(|scenario_id| self.simulate_scenario(case_data, scenario_id))
            .collect();

        // Compute local lower bound estimate (mean of stage-1 costs)
        let local_lb = trajectories.iter()
            .map(|t| t.stage_costs[0])
            .sum::<f64>() / trajectories.len() as f64;

        ForwardResult {
            trajectories,
            local_lower_bound: local_lb,
        }
    }

    fn simulate_scenario(
        &self,
        case_data: &CaseData,
        scenario_id: ScenarioId,
    ) -> ScenarioTrajectory {
        let mut state = case_data.initial_state();
        let mut stage_costs = Vec::with_capacity(case_data.num_stages());
        let mut visited_states = Vec::with_capacity(case_data.num_stages());
        let noise_path = self.sample_noise_path(scenario_id);

        for (stage_idx, stage) in case_data.stages.iter().enumerate() {
            // Update inflows from PAR model
            state.update_inflows(&case_data.par_models, &noise_path[stage_idx]);

            // Build and solve stage LP
            let lp = self.build_stage_lp(case_data, stage, &state);
            let solution = lp.solve().expect("LP should be feasible");

            // Record results
            stage_costs.push(solution.immediate_cost);
            visited_states.push(state.clone());

            // Transition to next state
            state = solution.extract_end_state();
        }

        ScenarioTrajectory {
            scenario_id,
            total_cost: stage_costs.iter().sum(),
            stage_costs,
            visited_states,
        }
    }
}
```

## 5. State Management

The state vector contains all information needed to determine the optimal policy from a given point:

```rust
impl StatePoint {
    /// Create state from initial conditions
    pub fn from_initial(case_data: &CaseData) -> Self {
        let storage: Vec<f64> = case_data.hydros.iter()
            .map(|h| h.initial_storage)
            .collect();

        // Initialize inflow history from historical data
        let max_lag = case_data.par_models.max_order();
        let inflow_history: Vec<Vec<f64>> = case_data.hydros.iter()
            .map(|h| {
                case_data.inflow_history
                    .get_lags(h.id, max_lag)
                    .to_vec()
            })
            .collect();

        Self { storage, inflow_history }
    }

    /// Update inflows using PAR model
    pub fn update_inflows(&mut self, par_models: &ParModels, noise: &NoiseVector) {
        for (h_idx, model) in par_models.models.iter().enumerate() {
            let new_inflow = model.sample(
                &self.inflow_history[h_idx],
                noise[h_idx],
            );

            // Shift history and prepend new value
            self.inflow_history[h_idx].pop();
            self.inflow_history[h_idx].insert(0, new_inflow);
        }
    }

    /// Extract end-of-stage state from LP solution
    pub fn from_solution(solution: &LpSolution, hydro_ids: &[HydroId]) -> Self {
        Self {
            storage: hydro_ids.iter()
                .map(|id| solution.get_storage(*id))
                .collect(),
            inflow_history: solution.inflow_history.clone(),
        }
    }
}
```

## 6. Backward Pass

The backward pass constructs cuts by computing subgradients of the expected future cost function. Starting from the final stage and working backwards, it evaluates the cost-to-go from each visited state under multiple noise realizations. States are distributed across MPI ranks; noise outcomes are parallelized across threads within each rank. After processing each stage, `MPI_Allgatherv` collects all new cuts.

```rust
/// Result from solving backward LP at one state-outcome pair
pub struct BackwardOutcome {
    pub noise_index: usize,
    pub probability: f64,
    pub objective: f64,          // Q_t^ω(x)
    pub dual_storage: Vec<f64>,  // π for storage constraints
    pub dual_inflow: Vec<f64>,   // π for inflow constraints
}

impl<R: RiskMeasure, C: CutFormulation, H: HorizonMode> TrainingLoop<R, C, H> {
    /// Execute backward pass to generate cuts
    pub fn backward_pass(&mut self, case_data: &CaseData, forward: &GlobalForwardResult) {
        // Process stages in reverse order (T down to 2)
        for stage_idx in (1..case_data.num_stages()).rev() {
            let stage = &case_data.stages[stage_idx];
            let prev_stage = &case_data.stages[stage_idx - 1];

            // Get unique states visited at stage-1 (deduplicated across scenarios)
            let visited_states = self.collect_visited_states(forward, stage_idx - 1);

            // Distribute states across ranks for parallel processing
            let my_states = self.distribute_states(&visited_states);

            // Generate cuts for each state (parallel across threads)
            let new_cuts: Vec<Cut> = my_states
                .into_par_iter()
                .map(|state| self.generate_cut(case_data, stage, &state))
                .collect();

            // Collect cuts from all ranks
            let all_cuts = self.allgather_cuts(&new_cuts);

            // Add cuts to FCF for previous stage
            for cut in all_cuts {
                self.fcf.add_cut(prev_stage.id, cut);
            }
        }
    }

    fn generate_cut(
        &self,
        case_data: &CaseData,
        stage: &Stage,
        state: &StatePoint,
    ) -> Cut {
        // Sample noise outcomes for backward evaluation
        let noise_outcomes = self.sample_backward_noise();

        // Evaluate LP for each noise outcome (parallel across outcomes)
        let outcomes: Vec<BackwardOutcome> = noise_outcomes
            .iter()
            .map(|(noise, prob)| {
                // Compute realized inflows
                let inflows = case_data.par_models.realize(
                    &state.inflow_history,
                    noise,
                );

                // Build backward LP with fixed initial state
                let lp = self.build_backward_lp(case_data, stage, state, &inflows);

                // Solve and extract duals
                let solution = lp.solve().expect("Backward LP should be feasible");

                BackwardOutcome {
                    noise_index: 0, // For tracking
                    probability: *prob,
                    objective: solution.objective,
                    dual_storage: solution.get_duals("storage"),
                    dual_inflow: solution.get_duals("inflow"),
                }
            })
            .collect();

        // Compute cut via risk measure
        let probabilities: Vec<f64> = outcomes.iter().map(|o| o.probability).collect();
        self.risk_measure.compute_cut(stage.id, state, &outcomes, &probabilities)
    }
}
```

## 7. Dual Extraction for Cut Coefficients

The cut coefficients are derived from LP duality. For a stage-$t$ LP:

$$
Q_t(x_{t-1}, \omega) = \min_{x_t} \{ c_t^\top x_t + \theta_{t+1} : Ax_t \geq b_t(x_{t-1}, \omega) \}
$$

The cut for stage $t-1$ is:

$$
\theta_t \geq \pi^\top b_t(x_{t-1}, \omega) - \text{const}
$$

where $\pi$ are the dual variables. Since $b_t$ depends linearly on the state:

- Storage: $v_{h,t} = v_{h,t-1} + \text{inflow} - \text{outflow}$
- Inflows: From PAR model with state-dependent history

```rust
/// Cut structure for FCF
pub struct Cut {
    pub stage: StageId,          // Stage this cut applies to
    pub intercept: f64,          // RHS constant
    pub storage_coef: Vec<f64>,  // Coefficient per hydro storage
    pub inflow_coef: Vec<f64>,   // Coefficient per hydro inflow state
    pub iteration: usize,        // Iteration when cut was created
    pub active_count: usize,     // Times cut was binding (for selection)
}

impl Cut {
    /// Evaluate cut at a given state
    pub fn evaluate(&self, state: &StatePoint) -> f64 {
        let storage_term: f64 = self.storage_coef.iter()
            .zip(state.storage.iter())
            .map(|(c, v)| c * v)
            .sum();

        let inflow_term: f64 = self.inflow_coef.iter()
            .zip(state.inflow_history.iter().map(|h| h[0]))
            .map(|(c, a)| c * a)
            .sum();

        self.intercept + storage_term + inflow_term
    }

    /// Build LP constraint: θ >= intercept + Σ c_v * v + Σ c_a * a
    pub fn to_constraint(&self, theta_var: VarId, state_vars: &StateVariables) -> Constraint {
        let mut coeffs = vec![(theta_var, 1.0)];  // θ

        for (i, &coef) in self.storage_coef.iter().enumerate() {
            coeffs.push((state_vars.storage[i], -coef));
        }

        for (i, &coef) in self.inflow_coef.iter().enumerate() {
            coeffs.push((state_vars.inflow[i], -coef));
        }

        Constraint {
            coeffs,
            sense: ConstraintSense::Ge,
            rhs: self.intercept,
        }
    }
}
```

> **Note:** See [Cut Management Implementation](cut-management-impl.md) for FCF structure, cut selection, serialization, and cross-rank synchronization.

## Cross-References

- [SDDP Algorithm](../01-math/sddp-algorithm.md) — Mathematical definition of the SDDP algorithm that this training loop implements
- [Cut Management (Math)](../01-math/cut-management.md) — Mathematical foundations for cut coefficients, selection theory, and dominance criteria
- [Cut Management Implementation](cut-management-impl.md) — FCF structure, cut selection strategies, serialization, and cross-rank cut synchronization
- [Work Distribution](../04-hpc/work-distribution.md) — Detailed MPI+OpenMP parallelism patterns for forward and backward pass distribution
- [Convergence Monitoring](./convergence-monitoring.md) — Convergence criteria, bound computation, and stopping rules applied within this loop
- [Input Loading Pipeline](./input-loading-pipeline.md) — How `CaseData` and warm-start policy cuts are loaded before training begins
