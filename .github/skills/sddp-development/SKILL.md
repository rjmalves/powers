---
name: sddp-development
description: Guide SDDP (Stochastic Dual Dynamic Programming) algorithm development, testing, and enhancement for the POWE.RS solver, covering mathematical foundations, implementation patterns, and modern improvements.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - sddp
    - stochastic-optimization
    - dynamic-programming
    - benders-decomposition
    - algorithm
---

# SDDP Algorithm Development

## Overview

This skill guides agents in developing, testing, and enhancing the Stochastic Dual Dynamic Programming (SDDP) algorithm implementation in POWE.RS. SDDP is a decomposition algorithm for solving large-scale multistage stochastic linear programs, particularly suited for hydrothermal dispatch optimization.

## Mathematical Foundations

### Documentation Reference
**`.copilot/context/01-sddp-mathematical-foundations.md`** provides comprehensive coverage of:

1. **Problem Formulation**: Multistage stochastic linear programs
2. **Dynamic Programming Decomposition**: Bellman's equation and cost-to-go functions
3. **Benders Decomposition**: Cutting plane approximation of value functions
4. **SDDP Algorithm**: Forward pass (simulation) and backward pass (cut generation)
5. **Convergence Theory**: Statistical bounds and stopping criteria

**Key Concepts**:
- **Stagewise independence**: Uncertainty at stage t depends only on state at t-1
- **Benders cuts**: `V(x) ≥ α + β^T x` where `α` is intercept, `β` is slope
- **Forward pass**: Monte Carlo simulation to estimate policy cost
- **Backward pass**: Generate cuts via dual variables from LP subproblems
- **Convergence**: Gap between upper bound (forward) and lower bound (backward)

## Current Implementation

### Documentation Reference
**`.copilot/context/02-current-implementation-analysis.md`** analyzes POWE.RS architecture:

**Strengths**:
- Well-designed module structure with clear separation of concerns
- Performance-focused HPC implementation
- Comprehensive test infrastructure (396+ tests, 72%+ coverage)
- Production-ready CI/CD pipeline

**Key Modules**:
- `src/sddp/mod.rs` (138KB): Main algorithm implementation
- `src/subproblem.rs` (235KB): LP formulation and solving
- `src/cut.rs`: Benders cut data structures
- `src/fcf.rs`: Future Cost Function with cut management
- `src/state.rs` (137KB): State variable management
- `src/scenario.rs`, `src/scenario_generator.rs`: Uncertainty modeling

## Core SDDP Modules

### 1. SDDP Main Algorithm (`src/sddp/`)

**Files**:
- `mod.rs` (138KB): Training loop, forward/backward passes
- `builder.rs`: SDDP configuration and initialization
- `instance.rs`: Problem instance representation

**Key Responsibilities**:
- Training iteration orchestration
- Forward pass: Parallel scenario simulation
- Backward pass: Sequential cut generation
- Convergence detection and stopping criteria

### 2. Subproblem Formulation (`src/subproblem.rs` - 235KB)

**Responsibilities**:
- Construct LP constraint matrix
- Set up objective function
- Integrate existing cuts into LP
- Call HiGHS solver
- Extract dual variables for new cuts

**Critical for**:
- Correctness: LP must represent problem accurately
- Performance: Hot path called thousands of times
- Numerical stability: Handle ill-conditioned matrices

### 3. Cut Management (`src/cut.rs`, `src/fcf.rs`)

**Cut Structure**:
```rust
pub struct Cut {
    intercept: f64,        // α in V(x) ≥ α + β^T x
    coefficients: Vec<f64>, // β (slope/gradient)
}
```

**Future Cost Function (FCF)**:
- Stores cuts for each stage and node
- Evaluates cuts to approximate value function
- Implements cut selection strategies

### 4. State Management (`src/state.rs` - 137KB)

**Responsibilities**:
- Represent state variables (reservoir levels, etc.)
- Track visited states for cut placement
- State transition logic
- Boundary handling

### 5. Scenario Generation (`src/scenario.rs`, `src/scenario_generator.rs`)

**Current Implementation**:
- Sample Average Approximation (SAA)
- Supports various stochastic processes
- Correlation handling via `src/correlation_applicator.rs`

**Key for**:
- Forward pass simulation
- Testing convergence with different scenario counts
- Risk measure evaluation

## SDDP Algorithm Workflow

### Training Iteration

```rust
pub fn train(&mut self, max_iterations: usize) -> Result<TrainingResult, Error> {
    for iteration in 0..max_iterations {
        // Forward pass: Simulate scenarios
        let forward_results = self.forward_pass(num_scenarios)?;
        let upper_bound = forward_results.mean_cost();
        
        // Backward pass: Generate cuts
        let cuts = self.backward_pass()?;
        let lower_bound = self.evaluate_lower_bound()?;
        
        // Check convergence
        let gap = (upper_bound - lower_bound) / upper_bound.abs();
        if gap < self.tolerance {
            break;
        }
        
        // Store iteration results
        self.results.push(IterationResult {
            iteration,
            upper_bound,
            lower_bound,
            gap,
        });
    }
    
    Ok(self.results)
}
```

### Forward Pass (Parallel)
```rust
pub fn forward_pass(&self, num_scenarios: usize) -> Result<Vec<f64>, Error> {
    use rayon::prelude::*;
    
    // Parallel scenario simulation
    (0..num_scenarios)
        .into_par_iter()
        .map(|scenario_idx| {
            self.simulate_scenario(scenario_idx)
        })
        .collect()
}

fn simulate_scenario(&self, scenario_idx: usize) -> Result<f64, Error> {
    let mut total_cost = 0.0;
    let mut state = self.initial_state.clone();
    
    // Sequential through stages
    for stage in 0..self.num_stages {
        let scenario = self.generate_scenario(scenario_idx, stage);
        let solution = self.solve_subproblem(&state, &scenario)?;
        total_cost += solution.cost;
        state = solution.next_state;
    }
    
    Ok(total_cost)
}
```

### Backward Pass (Sequential)
```rust
pub fn backward_pass(&mut self) -> Result<Vec<Cut>, Error> {
    let mut all_cuts = Vec::new();
    
    // Must be sequential: stage t needs cuts from t+1
    for stage in (1..self.num_stages).rev() {
        // But scenarios within stage can be parallel
        let stage_cuts: Vec<Cut> = self.scenarios[stage]
            .par_iter()
            .map(|scenario| {
                // Solve subproblem
                let solution = self.solve_with_cuts(stage, scenario)?;
                
                // Extract dual variables
                let duals = solution.dual_values;
                
                // Compute cut: β = -B^T π, α = c^T x - π^T b
                let cut = self.compute_cut(&duals, &solution, stage);
                
                Ok(cut)
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        // Add cuts to FCF for stage t
        self.fcf.add_cuts(stage, stage_cuts.clone());
        all_cuts.extend(stage_cuts);
    }
    
    Ok(all_cuts)
}
```

## Modern SDDP Improvements

### Documentation Reference
**`.copilot/context/03-modern-sddp-improvements.md`** describes state-of-the-art enhancements:

### 1. Cut Management Strategies

#### Multi-Cut vs Single-Cut
**Current**: Single-cut (average over scenarios)
**Alternative**: Multi-cut (separate cut per scenario)
- **Benefit**: 2-5x faster convergence for many-scenario problems
- **Trade-off**: More LP constraints

#### Cut Selection
Limit number of active cuts to prevent LP bloat:
- **Level-based selection**: Keep cuts near current level set
- **Dominated cut removal**: Remove cuts that never bind
- **Distance-based pruning**: Keep cuts closest to visited states

### 2. Convergence Acceleration

#### Regularization
Add quadratic penalty to encourage state revisitation:
```
min c^T x + θ + (ρ/2) ||x - x̄||²
```

#### Adaptive Sampling
- Start with few scenarios (fast iterations)
- Increase scenarios as lower bound improves
- Final iterations with many scenarios for accuracy

### 3. Numerical Improvements

#### State Space Discretization
For continuous state spaces, use interpolation:
- Store cuts at grid points
- Interpolate for intermediate states
- Reduces cut count while maintaining accuracy

#### Warm Starting
Reuse LP basis between similar subproblems:
- 20-50% faster subproblem solves
- Critical for problems with many stages/scenarios

See `highs-integration` skill for implementation details.

## Testing and Validation

### Test Infrastructure

**Test Directory**: `tests/`
**Test Scripts**: 
- `scripts/golden-tests.sh`: Regression tests with known solutions
- `scripts/run_examples.sh`: Example problem validation

### Testing Patterns

#### 1. Unit Tests for Core Components
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cut_evaluation() {
        let cut = Cut::new(vec![1.0, 2.0], 5.0);
        let state = State::new(vec![3.0, 4.0]);
        
        // V(x) = 5.0 + 1.0*3.0 + 2.0*4.0 = 16.0
        assert_eq!(cut.evaluate(&state), 16.0);
    }
    
    #[test]
    fn test_cut_dominance() {
        let cut1 = Cut::new(vec![1.0], 5.0);
        let cut2 = Cut::new(vec![1.0], 6.0);  // Dominates cut1
        
        assert!(cut2.dominates(&cut1));
    }
}
```

#### 2. Integration Tests for SDDP
```rust
#[test]
fn test_sddp_deterministic_problem() {
    // Deterministic problem should converge in 1 iteration
    let sddp = SDDPBuilder::new()
        .stages(3)
        .scenarios(1)  // Deterministic
        .tolerance(1e-6)
        .build();
    
    let result = sddp.train(10).unwrap();
    
    assert!(result.converged);
    assert!(result.iterations <= 5);
}

#[test]
fn test_sddp_stochastic_convergence() {
    let sddp = SDDPBuilder::new()
        .stages(5)
        .scenarios(10)
        .tolerance(0.01)  // 1% gap
        .build();
    
    let result = sddp.train(100).unwrap();
    
    assert!(result.converged);
    assert!(result.final_gap < 0.01);
}
```

#### 3. Property-Based Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_lower_bound_monotonic(
        num_iterations in 1..100_usize,
    ) {
        let mut sddp = create_test_sddp();
        let mut lower_bounds = Vec::new();
        
        for _ in 0..num_iterations {
            sddp.backward_pass()?;
            lower_bounds.push(sddp.evaluate_lower_bound()?);
        }
        
        // Property: Lower bound should be non-decreasing
        for i in 1..lower_bounds.len() {
            prop_assert!(lower_bounds[i] >= lower_bounds[i-1] - 1e-6);
        }
    }
}
```

#### 4. Golden Tests
Compare against known optimal solutions:

```bash
# scripts/golden-tests.sh
cargo test --release -- golden_
```

```rust
#[test]
fn golden_newsvendor_problem() {
    // Classic newsvendor problem with known solution
    let sddp = create_newsvendor_sddp();
    let result = sddp.train(100).unwrap();
    
    // Known optimal cost for this problem
    const EXPECTED_COST: f64 = 55.0;
    const TOLERANCE: f64 = 0.1;
    
    assert!((result.final_cost - EXPECTED_COST).abs() < TOLERANCE);
}
```

### Test Coverage Targets

From `rust-coverage` skill:
- **`src/sddp/mod.rs`**: 95%+ (core algorithm)
- **`src/subproblem.rs`**: 95%+ (correctness critical)
- **`src/cut.rs`**: 95%+ (fundamental structure)
- **Error paths**: 100% (all error handling tested)

```bash
# Check SDDP module coverage
cargo llvm-cov --html -- sddp
xdg-open target/llvm-cov/html/index.html
```

## Development Workflow

### 1. Understand the Algorithm
Read mathematical foundations:
```bash
cat .copilot/context/01-sddp-mathematical-foundations.md
```

### 2. Review Current Implementation
Understand existing code:
```bash
cat .copilot/context/02-current-implementation-analysis.md
```

### 3. Identify Improvement Opportunity
Check modern techniques:
```bash
cat .copilot/context/03-modern-sddp-improvements.md
```

### 4. Implement with Tests
```bash
# Write tests first (TDD)
# Edit: tests/test_sddp_*.rs

# Implement feature
# Edit: src/sddp/mod.rs or related files

# Run tests
cargo test sddp

# Check coverage
cargo llvm-cov --html -- sddp
```

### 5. Benchmark Performance
```bash
# Create benchmark in benches/
# Run benchmark
cargo bench --bench sddp_e2e

# Profile if needed
cargo flamegraph --bench sddp_e2e -- --bench
```

### 6. Validate with Examples
```bash
# Run example problems
./scripts/run_examples.sh

# Run golden tests
./scripts/golden-tests.sh
```

## Common SDDP Issues and Solutions

### Issue 1: Slow Convergence
**Symptoms**: Gap decreases very slowly, hundreds of iterations needed

**Potential Causes**:
- Too many scenarios (slow forward pass)
- Poor cut selection (LP bloat)
- No regularization (state space not explored)

**Solutions**:
- Implement adaptive sampling (few scenarios early, more later)
- Add cut selection strategies
- Consider regularization or sigma-point sampling

### Issue 2: Infeasible Subproblems
**Symptoms**: Solver returns infeasible status

**Potential Causes**:
- Incorrectly formulated constraints
- Infeasible cuts from numerical errors
- State bounds too tight

**Solutions**:
- Validate LP formulation with simple test cases
- Add feasibility check before adding cuts
- Use cut strengthening to avoid invalid cuts

### Issue 3: Numerical Instability
**Symptoms**: Dual variables are NaN/Inf, erratic convergence

**Potential Causes**:
- Ill-conditioned constraint matrices
- Poor scaling
- Tolerance too tight

**Solutions**:
- Scale constraints (see `highs-integration` skill)
- Use HiGHS numerical stability features
- Validate solution numerically before extracting duals

### Issue 4: Memory Explosion
**Symptoms**: Memory grows unbounded, OOM errors

**Potential Causes**:
- No cut selection (cuts accumulate)
- Large scenario trees
- Memory leaks in cut storage

**Solutions**:
- Implement cut selection (dominated cut removal)
- Use scenario reduction techniques
- Profile with heaptrack (see `rust-memory-analysis` skill)

## SDDP Development Checklist

Algorithm Implementation:
- [ ] Forward pass correctly simulates policy
- [ ] Backward pass generates valid cuts
- [ ] Cuts properly added to FCF
- [ ] Convergence detection works correctly
- [ ] Upper/lower bounds tracked accurately

Testing:
- [ ] Unit tests for all components
- [ ] Integration test for full algorithm
- [ ] Property tests for invariants
- [ ] Golden tests against known solutions
- [ ] Coverage > 95% for critical modules

Performance:
- [ ] Forward pass parallelized with rayon
- [ ] Subproblem solver reuses HiGHS instance
- [ ] Warm starting implemented
- [ ] Cut evaluation optimized
- [ ] Benchmark shows acceptable performance

Numerical Stability:
- [ ] LP matrices scaled appropriately
- [ ] Dual variables validated (no NaN/Inf)
- [ ] Cuts checked for validity
- [ ] Solver tolerances configured
- [ ] Solution checked for feasibility

## File References

- **Mathematical foundations**: `.copilot/context/01-sddp-mathematical-foundations.md`
- **Current implementation**: `.copilot/context/02-current-implementation-analysis.md`
- **Modern improvements**: `.copilot/context/03-modern-sddp-improvements.md`
- **SDDP modules**: `src/sddp/mod.rs` (138KB), `src/sddp/builder.rs`, `src/sddp/instance.rs`
- **Subproblem**: `src/subproblem.rs` (235KB)
- **State**: `src/state.rs` (137KB)
- **Cuts**: `src/cut.rs`, `src/fcf.rs`
- **Scenarios**: `src/scenario.rs`, `src/scenario_generator.rs`
- **Test scripts**: `scripts/golden-tests.sh`, `scripts/run_examples.sh`
- **Tests**: `tests/` directory

## Related Skills

- **highs-integration**: For LP solver optimization and warm starting
- **rust-benchmarking**: For measuring SDDP iteration performance
- **rust-profiling**: For identifying algorithm bottlenecks
- **hpc-optimization**: For parallel forward pass and SIMD optimization
- **rust-coverage**: For ensuring comprehensive testing
- **rust-clean-code**: For maintainable algorithm code

## Resources

- **SDDP.jl**: https://github.com/odow/SDDP.jl (Reference implementation)
- **Pereira & Pinto (1991)**: Original SDDP paper
- **Shapiro et al. (2014)**: "Analysis of Stochastic Dual Dynamic Programming"
- **Philpott & Guan (2008)**: "On the convergence of stochastic dual dynamic programming"
- **Project Documentation**: `.copilot/context/` directory
