---
name: highs-integration
description: Optimize HiGHS linear programming solver integration in the POWE.RS SDDP solver, focusing on warm starting, dual variable extraction, and numerical stability for efficient subproblem solving.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - rust
    - optimization
    - linear-programming
    - highs
    - solver
    - ffi
---

# HiGHS LP Solver Integration

## Overview

This skill guides agents in optimizing the integration of the HiGHS linear programming solver in POWE.RS. HiGHS is a high-performance LP/MIP solver that forms the computational core of SDDP subproblem solving.

## HiGHS Dependency

### Current Version
```toml
# From Cargo.toml
highs-sys = "1.6.4"
```

This provides FFI bindings to the HiGHS solver library.

### Upstream Repository
**ERGO-Code/HiGHS**: https://github.com/ERGO-Code/HiGHS

Key features:
- **Performance**: State-of-the-art LP solver
- **Dual Simplex**: Efficient for re-solving similar LPs (SDDP use case)
- **Warm Starting**: Reuse basis from previous solves
- **Dual Variables**: Essential for Benders cut generation
- **Numerical Stability**: Robust handling of ill-conditioned problems

## Solver Module

### Primary File
**`src/solver.rs`** (45KB): Core HiGHS integration

This file contains:
- HiGHS FFI wrapper
- LP problem formulation
- Solver configuration
- Solution extraction
- Error handling for solver status

### Subproblem Integration
**`src/subproblem.rs`** (235KB): Uses solver for SDDP subproblems

Key responsibilities:
- Construct constraint matrices
- Set up objective function
- Call HiGHS solver
- Extract dual variables for cuts

## Optimization Strategies

### 1. Warm Starting for SDDP

**Context**: SDDP solves similar LPs repeatedly (same structure, different right-hand sides)

#### Basic Warm Start Pattern
```rust
pub struct SolverWorkspace {
    highs: *mut highs_sys::Highs,
    last_basis: Option<Basis>,
}

impl SolverWorkspace {
    pub fn solve(&mut self, problem: &LPProblem) -> Result<Solution, SolverError> {
        // Set up problem (constraint matrix, bounds, objective)
        self.setup_problem(problem)?;
        
        // Apply warm start if available
        if let Some(ref basis) = self.last_basis {
            unsafe {
                highs_sys::Highs_setBasis(self.highs, basis.as_ptr());
            }
        }
        
        // Solve
        let status = unsafe { highs_sys::Highs_run(self.highs) };
        
        // Save basis for next solve
        self.last_basis = Some(self.extract_basis()?);
        
        self.extract_solution(status)
    }
}
```

#### Warm Start Benefits
- **20-50% faster** for similar LPs
- **Critical for backward pass**: Solving 100s of similar subproblems per iteration
- **Basis reuse**: Start from near-optimal point

#### When to Reset Basis
```rust
// Reset basis when problem structure changes significantly
if problem.num_variables != last_problem.num_variables {
    self.last_basis = None;
}

// Reset periodically to avoid basis degradation
if iteration_count % 100 == 0 {
    self.last_basis = None;
}
```

### 2. Dual Variable Extraction for Cuts

**SDDP requires dual variables** to generate Benders cuts in the backward pass.

#### Extract Dual Variables
```rust
pub fn extract_duals(&self, num_constraints: usize) -> Result<Vec<f64>, SolverError> {
    let mut duals = vec![0.0; num_constraints];
    
    unsafe {
        let status = highs_sys::Highs_getDualValues(
            self.highs,
            duals.as_mut_ptr()
        );
        
        if status != highs_sys::HighsStatus_OK {
            return Err(SolverError::DualExtractionFailed);
        }
    }
    
    Ok(duals)
}
```

#### Compute Benders Cut from Duals
```rust
// Given dual variables π from stage t+1 subproblem:
// Cut: V_t(x_t) ≥ α + β^T x_t
// where:
//   β = -B^T π  (B is state transition matrix)
//   α = c^T x* - π^T b  (optimal cost minus dual contribution)

pub fn compute_cut(
    duals: &[f64],
    solution: &Solution,
    transition_matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> Cut {
    // Compute cut slope: β = -B^T π
    let slope = -transition_matrix.transpose() * DVector::from_vec(duals.to_vec());
    
    // Compute cut intercept: α = c^T x* - π^T b
    let dual_contribution: f64 = duals.iter()
        .zip(rhs.iter())
        .map(|(pi, b)| pi * b)
        .sum();
    let intercept = solution.objective_value - dual_contribution;
    
    Cut::new(slope.as_slice().to_vec(), intercept)
}
```

### 3. Numerical Stability

#### Problem Scaling
```rust
// Scale constraint matrix to improve numerical stability
pub fn scale_problem(problem: &mut LPProblem) {
    // Row scaling: Normalize each constraint
    for row in problem.constraints.iter_mut() {
        let max_coeff = row.coefficients.iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        
        if max_coeff > 1e-10 {
            let scale = 1.0 / max_coeff;
            row.coefficients.iter_mut().for_each(|x| *x *= scale);
            row.rhs *= scale;
        }
    }
    
    // Column scaling: Normalize each variable
    // Similar process for columns
}
```

#### Tolerances
```rust
// Configure HiGHS solver tolerances
pub fn configure_tolerances(highs: *mut highs_sys::Highs) {
    unsafe {
        // Primal feasibility tolerance
        highs_sys::Highs_setDoubleOptionValue(
            highs,
            c_str("primal_feasibility_tolerance"),
            1e-7
        );
        
        // Dual feasibility tolerance
        highs_sys::Highs_setDoubleOptionValue(
            highs,
            c_str("dual_feasibility_tolerance"),
            1e-7
        );
        
        // Optimal condition tolerance
        highs_sys::Highs_setDoubleOptionValue(
            highs,
            c_str("ipm_optimality_tolerance"),
            1e-8
        );
    }
}
```

#### Handle Ill-Conditioned Problems
```rust
// Detect and handle numerical issues
pub fn solve_with_fallback(&mut self, problem: &LPProblem) -> Result<Solution, SolverError> {
    // Try normal solve
    match self.solve(problem) {
        Ok(solution) if solution.is_numerically_stable() => Ok(solution),
        _ => {
            // Fallback: Tighten tolerances and resolve
            self.configure_tight_tolerances();
            
            match self.solve(problem) {
                Ok(solution) => {
                    log::warn!("Required tight tolerances for numerical stability");
                    Ok(solution)
                }
                Err(e) => Err(e),
            }
        }
    }
}
```

### 4. Efficient Matrix Construction

#### Sparse Matrix Representation
```rust
// Use sparse format for HiGHS
pub struct SparseMatrix {
    num_rows: usize,
    num_cols: usize,
    row_starts: Vec<i32>,    // CSR format: start index for each row
    col_indices: Vec<i32>,   // Column index for each non-zero
    values: Vec<f64>,        // Non-zero values
}

impl SparseMatrix {
    pub fn from_dense(dense: &DMatrix<f64>) -> Self {
        let mut row_starts = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        
        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let value = dense[(row, col)];
                if value.abs() > 1e-15 {  // Skip near-zero values
                    col_indices.push(col as i32);
                    values.push(value);
                }
            }
            row_starts.push(col_indices.len() as i32);
        }
        
        SparseMatrix {
            num_rows: dense.nrows(),
            num_cols: dense.ncols(),
            row_starts,
            col_indices,
            values,
        }
    }
}
```

#### Pass to HiGHS
```rust
pub fn set_constraint_matrix(&mut self, matrix: &SparseMatrix) -> Result<(), SolverError> {
    unsafe {
        let status = highs_sys::Highs_passLp(
            self.highs,
            matrix.num_cols as i32,
            matrix.num_rows as i32,
            matrix.values.len() as i32,
            // ... other parameters
            matrix.row_starts.as_ptr(),
            matrix.col_indices.as_ptr(),
            matrix.values.as_ptr(),
        );
        
        if status != highs_sys::HighsStatus_OK {
            return Err(SolverError::MatrixSetupFailed);
        }
    }
    
    Ok(())
}
```

### 5. Error Handling for Solver Status

#### Comprehensive Status Handling
```rust
pub fn interpret_status(status: i32) -> Result<SolverStatus, SolverError> {
    match status {
        highs_sys::HighsModelStatus_OPTIMAL => Ok(SolverStatus::Optimal),
        highs_sys::HighsModelStatus_INFEASIBLE => Err(SolverError::Infeasible {
            stage: self.stage,
            scenario: self.scenario,
        }),
        highs_sys::HighsModelStatus_UNBOUNDED => Err(SolverError::Unbounded {
            stage: self.stage,
        }),
        highs_sys::HighsModelStatus_TIME_LIMIT => Err(SolverError::TimeLimit),
        highs_sys::HighsModelStatus_ITERATION_LIMIT => Err(SolverError::IterationLimit),
        _ => Err(SolverError::UnknownStatus(status)),
    }
}
```

#### Contextual Error Messages
Follow patterns from `src/error.rs`:

```rust
#[derive(Error, Debug)]
pub enum SolverError {
    #[error("LP infeasible at stage {stage}, scenario {scenario}. Check constraints and bounds.")]
    Infeasible { stage: usize, scenario: usize },
    
    #[error("LP unbounded at stage {stage}. Add finite bounds or check objective direction.")]
    Unbounded { stage: usize },
    
    #[error("Dual variable extraction failed. Solution may be invalid.")]
    DualExtractionFailed,
    
    #[error("HiGHS FFI error: {message}")]
    HighsFFIError { message: String },
}
```

## Performance Optimization

### 1. Reuse Solver Instance
```rust
// ✅ CORRECT - Reuse solver across solves
pub struct SubproblemSolver {
    highs: *mut highs_sys::Highs,
}

impl SubproblemSolver {
    pub fn new() -> Self {
        let highs = unsafe { highs_sys::Highs_create() };
        SubproblemSolver { highs }
    }
    
    pub fn solve(&mut self, problem: &LPProblem) -> Result<Solution, SolverError> {
        // Clear previous problem
        unsafe { highs_sys::Highs_clear(self.highs) };
        
        // Set up new problem
        self.setup_problem(problem)?;
        
        // Solve
        self.run_solver()
    }
}

impl Drop for SubproblemSolver {
    fn drop(&mut self) {
        unsafe { highs_sys::Highs_destroy(self.highs) };
    }
}

// ❌ WRONG - Create new solver each time
pub fn solve(problem: &LPProblem) -> Result<Solution, SolverError> {
    let highs = unsafe { highs_sys::Highs_create() };  // Expensive!
    // ...
    unsafe { highs_sys::Highs_destroy(highs) };
    Ok(solution)
}
```

### 2. Parallel Solving (Forward Pass)
```rust
// Use rayon for parallel scenario solving
use rayon::prelude::*;

pub fn parallel_forward_pass(
    scenarios: &[Scenario],
    num_threads: usize,
) -> Vec<Result<Solution, SolverError>> {
    // Each thread needs its own HiGHS instance (not thread-safe)
    scenarios
        .par_iter()
        .map(|scenario| {
            // Thread-local solver
            let mut solver = SubproblemSolver::new();
            solver.solve(scenario)
        })
        .collect()
}
```

### 3. Benchmark Solver Performance
```rust
// From benches/README.md targets
// Subproblem Solve: Single stage < 10ms

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_subproblem_solve(c: &mut Criterion) {
    let problem = create_test_problem(100, 200);  // 100 vars, 200 constraints
    let mut solver = SubproblemSolver::new();
    
    c.bench_function("subproblem_solve", |b| {
        b.iter(|| {
            black_box(solver.solve(&problem))
        });
    });
}
```

## Integration with SDDP

### Backward Pass Cut Generation
```rust
pub fn backward_pass(&mut self) -> Result<Vec<Cut>, Error> {
    let mut cuts = Vec::new();
    
    // For each stage (backward from T-1 to 1)
    for stage in (1..self.num_stages).rev() {
        // For each scenario at this stage
        for scenario in &self.scenarios[stage] {
            // Solve subproblem
            let solution = self.solver.solve(&scenario.problem)?;
            
            // Extract dual variables
            let duals = self.solver.extract_duals(scenario.num_constraints)?;
            
            // Compute Benders cut
            let cut = self.compute_cut(&duals, &solution, stage);
            
            cuts.push(cut);
        }
    }
    
    Ok(cuts)
}
```

### Forward Pass Simulation
```rust
pub fn forward_pass(&mut self, num_scenarios: usize) -> Result<Vec<f64>, Error> {
    let mut costs = Vec::with_capacity(num_scenarios);
    
    // Parallel scenario simulation
    let scenario_costs: Vec<_> = (0..num_scenarios)
        .into_par_iter()
        .map(|scenario_idx| {
            let mut solver = SubproblemSolver::new();
            let mut total_cost = 0.0;
            
            // Simulate forward through stages
            for stage in 0..self.num_stages {
                let problem = self.formulate_subproblem(stage, scenario_idx);
                let solution = solver.solve(&problem)?;
                total_cost += solution.objective_value;
            }
            
            Ok(total_cost)
        })
        .collect();
    
    // Handle errors
    for result in scenario_costs {
        costs.push(result?);
    }
    
    Ok(costs)
}
```

## Numerical Stability Considerations

### 1. Check Solution Validity
```rust
impl Solution {
    pub fn is_numerically_stable(&self) -> bool {
        // Check for NaN or Inf
        if !self.objective_value.is_finite() {
            return false;
        }
        
        // Check variable values
        if self.variables.iter().any(|x| !x.is_finite()) {
            return false;
        }
        
        // Check dual variables
        if self.duals.iter().any(|pi| !pi.is_finite()) {
            return false;
        }
        
        true
    }
}
```

### 2. Monitor Condition Number
```rust
// Log warnings for ill-conditioned problems
pub fn solve_with_monitoring(&mut self, problem: &LPProblem) -> Result<Solution, SolverError> {
    let solution = self.solve(problem)?;
    
    // Get condition number from HiGHS (if available)
    let condition_number = self.get_condition_number();
    
    if condition_number > 1e12 {
        log::warn!(
            "Ill-conditioned problem detected (cond = {:.2e}). Solution may be inaccurate.",
            condition_number
        );
    }
    
    Ok(solution)
}
```

## Best Practices Checklist

- [ ] Reuse HiGHS instance across solves
- [ ] Implement warm starting for similar LPs
- [ ] Extract and validate dual variables
- [ ] Handle all solver status codes (optimal, infeasible, unbounded)
- [ ] Scale constraint matrix for numerical stability
- [ ] Use sparse matrix format for large problems
- [ ] Validate solutions are numerically stable (no NaN/Inf)
- [ ] Benchmark subproblem solve time (target < 10ms)
- [ ] Use thread-local solvers for parallel forward pass
- [ ] Log warnings for ill-conditioned problems

## File References

- **Solver module**: `src/solver.rs` (45KB)
- **Subproblem formulation**: `src/subproblem.rs` (235KB)
- **SDDP algorithm**: `src/sddp/mod.rs` (138KB)
- **Cut generation**: `src/cut.rs`
- **Error handling**: `src/error.rs`
- **Dependency**: `highs-sys = "1.6.4"` in `Cargo.toml`
- **Performance targets**: `benches/README.md`

## Related Skills

- **sddp-development**: For understanding cut generation and algorithm flow
- **rust-benchmarking**: For measuring solver performance
- **rust-profiling**: For identifying HiGHS-related bottlenecks
- **hpc-optimization**: For parallel forward pass optimization

## Resources

- **HiGHS Repository**: https://github.com/ERGO-Code/HiGHS
- **HiGHS Documentation**: https://ergo-code.github.io/HiGHS/
- **highs-sys Crate**: https://docs.rs/highs-sys/
- **Benders Decomposition**: Classical reference for dual-based cuts
