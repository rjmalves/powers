---
name: rust-clean-code
description: Enforce clean code practices, idiomatic Rust patterns, and code quality standards for the POWE.RS SDDP solver using rustfmt, clippy, and established error handling patterns.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - rust
    - code-quality
    - rustfmt
    - clippy
    - style
    - idiomatic
---

# Rust Clean Code Practices

## Overview

This skill guides agents in writing clean, idiomatic, and maintainable Rust code for the POWE.RS SDDP solver. Clean code is essential for a numerical library where correctness, readability, and performance must coexist.

## Code Formatting with rustfmt

### Configuration
POWE.RS includes a `rustfmt.toml` configuration:

```toml
max_width = 80
```

This enforces an 80-character line width limit, promoting readability and conventional Rust style.

### Running rustfmt
```bash
# Format all code in the project
cargo fmt

# Check formatting without modifying files
cargo fmt -- --check

# Format specific file
rustfmt src/subproblem.rs
```

### Best Practices
- **Run before committing**: Always format code before creating commits
- **Respect line width**: Keep lines ≤ 80 characters
- **Don't fight rustfmt**: Accept its decisions for consistency
- **Chain formatting**: Use `cargo fmt && cargo test` in workflow

## Linting with Clippy

### Running Clippy
```bash
# Run all lints
cargo clippy

# Run with all features
cargo clippy --all-features

# Run on specific package
cargo clippy --package powers-rs

# Treat warnings as errors (CI mode)
cargo clippy -- -D warnings
```

### Recommended Clippy Configuration
Add to `Cargo.toml` or `.clippy.toml`:

```toml
# Recommended clippy lints for numerical code
[lints.clippy]
# Performance lints (critical for HPC)
needless_range_loop = "warn"
manual_memcpy = "warn"
inefficient_to_string = "warn"

# Correctness lints
float_cmp = "warn"           # Dangerous for numerical code
modulo_one = "deny"
suspicious_arithmetic_impl = "deny"

# Style lints
used_underscore_binding = "allow"  # Common in generated bindings
```

### Critical Lints for Numerical Code

#### Float Comparison (CRITICAL)
```rust
// ❌ WRONG - Direct float comparison
if value == 0.0 {
    // May fail due to floating-point error
}

// ✅ CORRECT - Use epsilon comparison
const EPSILON: f64 = 1e-10;
if value.abs() < EPSILON {
    // Robust to floating-point error
}

// ✅ CORRECT - Use approx crate (available in dev-dependencies)
use approx::assert_relative_eq;
assert_relative_eq!(value, expected, epsilon = 1e-6);
```

#### Integer Division in Numerical Code
```rust
// ⚠️ CAREFUL - Integer division
let average = sum / count;  // Truncates if integers

// ✅ CORRECT - Explicit cast for numerical operations
let average = sum as f64 / count as f64;
```

## Error Handling with thiserror

### Dependency
POWE.RS uses `thiserror = "2.0"` for error handling.

### Error Module Pattern
Refer to `src/error.rs` for the established pattern:

```rust
use thiserror::Error;

/// Top-level error type for POWE.RS operations.
#[derive(Error, Debug)]
pub enum PowersError {
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),
    
    #[error("Solver error: {0}")]
    Solver(#[from] SolverError),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Specific error category with context
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Invalid field value in {file}: {field} = {value} (constraint: {constraint}). {suggestion}")]
    InvalidFieldValue {
        file: String,
        field: String,
        value: String,
        constraint: String,
        suggestion: String,
    },
}
```

### Error Handling Principles (from src/error.rs)

1. **Context-Rich**: Every error includes relevant context (file, field, value)
2. **Actionable**: Errors suggest fixes, not just what went wrong
3. **User-Friendly**: Production errors are clear; stack traces only in debug
4. **Typed**: Use specific error types for proper error handling
5. **Zero-Cost**: Error types are thin wrappers with no runtime overhead

### Error Handling Examples

#### Good Error Handling
```rust
pub fn validate_positive(value: usize, field: &str) -> Result<(), PowersError> {
    if value == 0 {
        return Err(ValidationError::InvalidFieldValue {
            file: "config.json".to_string(),
            field: field.to_string(),
            value: value.to_string(),
            constraint: "must be positive (> 0)".to_string(),
            suggestion: format!("Set {} to at least 1", field),
        }.into());
    }
    Ok(())
}
```

#### Error Propagation
```rust
// ✅ CORRECT - Use ? operator
pub fn solve_subproblem(&mut self) -> Result<Solution, PowersError> {
    let matrix = self.build_constraint_matrix()?;
    let solution = self.call_solver(&matrix)?;
    Ok(solution)
}

// ❌ WRONG - Unwrapping in library code
pub fn solve_subproblem(&mut self) -> Solution {
    let matrix = self.build_constraint_matrix().unwrap();  // DON'T DO THIS
    let solution = self.call_solver(&matrix).unwrap();
    solution
}
```

#### Contextual Errors
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SolverError {
    #[error("LP solver returned infeasible status for stage {stage}, scenario {scenario}")]
    Infeasible { stage: usize, scenario: usize },
    
    #[error("LP solver returned unbounded status for stage {stage}. Check constraints.")]
    Unbounded { stage: usize },
    
    #[error("HiGHS FFI error: {message}")]
    HighsError { message: String },
}
```

## Documentation Standards

### Public API Documentation
All public items must have documentation:

```rust
/// Solves a single SDDP subproblem using the HiGHS LP solver.
///
/// # Arguments
///
/// * `state` - The state at the beginning of the stage
/// * `scenario` - The uncertainty realization for this scenario
///
/// # Returns
///
/// Returns `Ok(Solution)` containing optimal decision and cost, or
/// `Err(SolverError)` if the LP is infeasible or unbounded.
///
/// # Examples
///
/// ```
/// use powers_rs::{Subproblem, State, Scenario};
///
/// let subproblem = Subproblem::new(/* ... */);
/// let state = State::new(vec![100.0, 200.0]);
/// let scenario = Scenario::new(/* ... */);
///
/// match subproblem.solve(&state, &scenario) {
///     Ok(solution) => println!("Optimal cost: {}", solution.cost),
///     Err(e) => eprintln!("Solver error: {}", e),
/// }
/// ```
///
/// # Performance
///
/// Typical solve time: 5-10ms for problems with 100 variables and 200 constraints.
/// See `benches/README.md` for detailed performance characteristics.
pub fn solve(&mut self, state: &State, scenario: &Scenario) -> Result<Solution, SolverError> {
    // Implementation
}
```

### Internal Documentation
Use comments for complex algorithms:

```rust
// SDDP backward pass: Generate Benders cuts via dual variables
// 
// For each stage t and scenario ω:
// 1. Solve LP to get dual variables π(t,ω)
// 2. Compute cut coefficients: ∂V/∂x = -B^T π
// 3. Compute cut intercept: α = c^T x^* - π^T b
// 4. Add cut: V(x) ≥ α + (∂V/∂x)^T x
fn generate_cuts(&mut self) -> Result<Vec<Cut>, Error> {
    // Implementation
}
```

## Idiomatic Rust Patterns

### Ownership and Borrowing
```rust
// ✅ CORRECT - Borrow when read-only
fn evaluate_cuts(cuts: &[Cut], state: &State) -> f64 {
    cuts.iter().map(|cut| cut.evaluate(state)).sum()
}

// ✅ CORRECT - Mutable borrow when modifying
fn update_state(state: &mut State, decision: &Decision) {
    state.apply_decision(decision);
}

// ✅ CORRECT - Take ownership when consuming
fn consume_solution(solution: Solution) -> f64 {
    solution.cost  // solution is moved, no longer accessible
}
```

### Iterator Chains (Performance-Compatible)
```rust
// ✅ CORRECT - Functional style with zero overhead
let total_cost: f64 = scenarios
    .iter()
    .map(|s| s.solve())
    .filter(|r| r.is_ok())
    .map(|r| r.unwrap().cost)
    .sum();

// ✅ ALSO CORRECT - Imperative style for complex logic
let mut total_cost = 0.0;
for scenario in scenarios {
    match scenario.solve() {
        Ok(solution) => total_cost += solution.cost,
        Err(e) => log::warn!("Scenario failed: {}", e),
    }
}
```

### Avoid Unnecessary Allocations
```rust
// ❌ WRONG - Unnecessary allocation
fn compute_sum(values: &[f64]) -> f64 {
    let owned: Vec<f64> = values.to_vec();  // Unnecessary copy
    owned.iter().sum()
}

// ✅ CORRECT - Use slice directly
fn compute_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}
```

### Use Standard Traits
```rust
// ✅ CORRECT - Implement standard traits
#[derive(Debug, Clone, PartialEq)]
pub struct Cut {
    coefficients: Vec<f64>,
    intercept: f64,
}

// ✅ CORRECT - Implement Display for user-facing types
use std::fmt;

impl fmt::Display for Cut {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cut(intercept={:.2}, dims={})", self.intercept, self.coefficients.len())
    }
}
```

## Performance-Compatible Clean Code

### Avoid Premature Pessimization
```rust
// ✅ CORRECT - Use Vec::with_capacity for known sizes
let mut values = Vec::with_capacity(num_scenarios);
for scenario in scenarios {
    values.push(scenario.solve()?);
}

// ❌ WRONG - Repeated reallocations
let mut values = Vec::new();
for scenario in scenarios {
    values.push(scenario.solve()?);
}
```

### Prefer Zero-Cost Abstractions
```rust
// ✅ CORRECT - Iterator chains (zero overhead)
let sum: f64 = values.iter().sum();

// ✅ ALSO CORRECT - Manual loop (same performance)
let mut sum = 0.0;
for &value in values {
    sum += value;
}
```

### Use Inline for Small Hot Functions
```rust
// ✅ CORRECT - Inline small hot functions
#[inline]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
```

## Code Organization

### Module Structure
```rust
// ✅ CORRECT - Public API at top
pub mod sddp;
pub mod solver;
pub mod subproblem;

// ✅ CORRECT - Internal modules after
mod algorithm;
mod utils;

// ✅ CORRECT - Re-exports for convenience
pub use sddp::{SDDP, SDDPBuilder};
pub use solver::{Solver, SolverConfig};
```

### File Size Management
POWE.RS has large files that should be monitored:
- `src/subproblem.rs` (235KB)
- `src/state.rs` (137KB)
- `src/sddp/mod.rs` (138KB)

**When a file exceeds 200KB**:
1. Consider splitting into submodules
2. Extract distinct responsibilities
3. Maintain coherent public API

## Testing Code Quality

### Test Organization
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solve_feasible_problem() {
        // Arrange
        let subproblem = create_test_subproblem();
        let state = State::new(vec![100.0]);
        
        // Act
        let result = subproblem.solve(&state);
        
        // Assert
        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.cost > 0.0);
    }
}
```

### Property-Based Testing
Use `proptest = "1.4"` for property tests:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_cut_evaluation_scales_linearly(
        coeffs in prop::collection::vec(-100.0..100.0_f64, 1..10),
        state in prop::collection::vec(-100.0..100.0_f64, 1..10),
        scale in 0.1..10.0_f64
    ) {
        let cut = Cut::new(coeffs, 0.0);
        let value1 = cut.evaluate(&State::new(state.clone()));
        
        let scaled_state: Vec<f64> = state.iter().map(|x| x * scale).collect();
        let value2 = cut.evaluate(&State::new(scaled_state));
        
        // Property: evaluation scales linearly
        prop_assert!((value2 - value1 * scale).abs() < 1e-6);
    }
}
```

## Code Review Checklist

- [ ] Code formatted with `cargo fmt`
- [ ] No clippy warnings: `cargo clippy -- -D warnings`
- [ ] Public functions documented with doc comments
- [ ] Error handling uses `Result<T, Error>`, not `unwrap()`
- [ ] Float comparisons use epsilon, not `==`
- [ ] Tests added for new functionality
- [ ] Performance considerations documented
- [ ] Error messages are actionable and context-rich
- [ ] Code follows patterns from `src/error.rs`
- [ ] No unnecessary allocations in hot paths

## Common Anti-Patterns to Avoid

### 1. Unwrap in Library Code
```rust
// ❌ WRONG - Will panic in production
pub fn solve(&mut self) -> Solution {
    let result = self.call_solver().unwrap();
    result
}

// ✅ CORRECT - Propagate error
pub fn solve(&mut self) -> Result<Solution, Error> {
    let result = self.call_solver()?;
    Ok(result)
}
```

### 2. String Allocation in Hot Paths
```rust
// ❌ WRONG - Allocates in loop
for i in 0..1000000 {
    let msg = format!("Processing {}", i);
    log::trace!("{}", msg);  // Hot path!
}

// ✅ CORRECT - Use &str or skip formatting
for i in 0..1000000 {
    log::trace!("Processing {}", i);  // Format only if trace enabled
}
```

### 3. Ignoring Performance in "Clean" Code
```rust
// ❌ WRONG - Multiple passes over data
let sum: f64 = values.iter().sum();
let count = values.len();
let mean = sum / count as f64;
let variance: f64 = values.iter()
    .map(|x| (x - mean).powi(2))
    .sum::<f64>() / count as f64;

// ✅ CORRECT - Single pass
let (sum, sum_sq, count) = values.iter()
    .fold((0.0, 0.0, 0), |(s, sq, c), &x| (s + x, sq + x * x, c + 1));
let mean = sum / count as f64;
let variance = (sum_sq / count as f64) - mean * mean;
```

## Integration with Other Skills

- **rust-benchmarking**: Verify performance after refactoring
- **rust-coverage**: Ensure tests cover error paths
- **hpc-optimization**: Balance clean code with performance needs

## File References

- **Formatting config**: `rustfmt.toml`
- **Error patterns**: `src/error.rs`
- **Large files**: `src/subproblem.rs` (235KB), `src/state.rs` (137KB), `src/sddp/mod.rs` (138KB)
- **Dependencies**: `thiserror = "2.0"` in `Cargo.toml`
- **Test utilities**: `approx = "0.5"`, `proptest = "1.4"` in dev-dependencies

## Resources

- **Rust API Guidelines**: https://rust-lang.github.io/api-guidelines/
- **Clippy Lints**: https://rust-lang.github.io/rust-clippy/
- **rustfmt Configuration**: https://rust-lang.github.io/rustfmt/
- **thiserror Documentation**: https://docs.rs/thiserror/
