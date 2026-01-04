---
name: rust-coverage
description: Guide agents to measure and improve test coverage using cargo-llvm-cov for the POWE.RS SDDP solver, ensuring high-quality testing of critical optimization algorithms.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - rust
    - testing
    - coverage
    - llvm-cov
    - quality
---

# Rust Test Coverage with cargo-llvm-cov

## Overview

This skill guides agents in measuring and improving test coverage for the POWE.RS SDDP solver using `cargo-llvm-cov`. High test coverage is essential for a numerical optimization library where correctness is paramount and bugs can lead to incorrect solutions.

## Preferred Tool: cargo-llvm-cov

POWE.RS uses `cargo-llvm-cov` as documented in `.copilot/development/COVERAGE-TOOLING.md`.

### Why llvm-cov?
- **Accuracy**: Uses LLVM's native coverage instrumentation
- **Performance**: Faster than tarpaulin (no ptrace overhead)
- **Compatibility**: Better support for inline functions and generics
- **Integration**: Works seamlessly with Rust's LLVM-based compilation
- **Output Formats**: HTML, lcov, JSON, and text formats

## Installation

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Install required component
rustup component add llvm-tools-preview
```

## Basic Usage

### Generate HTML Coverage Report
```bash
# Generate HTML report in target/llvm-cov/html/
cargo llvm-cov --html

# Open in browser
xdg-open target/llvm-cov/html/index.html  # Linux
open target/llvm-cov/html/index.html      # macOS
```

### Generate Text Summary
```bash
# Quick summary to stdout
cargo llvm-cov --summary-only

# Example output:
# Filename                      Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       Lines      Missed Lines     Cover
# src/sddp/mod.rs                   245                15    93.88%          89                 3    96.63%         890                147    83.48%
```

### Generate lcov for CI/CD
```bash
# Generate lcov.info for codecov.io or similar services
cargo llvm-cov --lcov --output-path target/llvm-cov/lcov.info
```

## Coverage Workflow

### 1. Clean Previous Coverage Data
```bash
# Clean previous coverage data
cargo llvm-cov clean

# Clean and regenerate
cargo llvm-cov clean --workspace && cargo llvm-cov --html
```

### 2. Run Coverage with Specific Tests
```bash
# Run specific test file
cargo llvm-cov --html -- --test test_sddp_algorithm

# Run tests matching pattern
cargo llvm-cov --html -- forward_pass

# Include ignored tests (like expensive_tests)
cargo llvm-cov --html -- --include-ignored

# Run with features
cargo llvm-cov --html --features timing
```

### 3. Review HTML Report
The HTML report shows:
- **Green lines**: Covered by tests
- **Red lines**: Not covered (need tests)
- **Orange lines**: Partially covered (some branches not tested)
- **Gray lines**: Not executable (comments, declarations)

## Coverage Target Thresholds

As documented in `.copilot/development/COVERAGE-TOOLING.md`:

### Target Thresholds
- **Critical modules**: 95-98%+ line coverage
  - `src/sddp/mod.rs` (138KB): Core SDDP algorithm
  - `src/solver.rs` (45KB): HiGHS LP solver integration
  - `src/subproblem.rs` (235KB): LP formulation and solving
- **Error paths**: 100% - All error handling tested
- **Public API**: 100% - All public functions exercised
- **Overall project**: 90-95%+ line coverage

### Coverage Metrics
- **Line Coverage**: % of lines executed (primary metric)
- **Region Coverage**: % of code regions executed (LLVM concept)
- **Function Coverage**: % of functions called
- **Branch Coverage**: % of conditional branches taken

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
# .github/workflows/coverage.yml
name: Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: llvm-tools-preview
      
      - name: Install cargo-llvm-cov
        run: cargo install cargo-llvm-cov
      
      - name: Generate coverage
        run: cargo llvm-cov --lcov --output-path lcov.info
      
      - name: Upload to codecov.io
        uses: codecov/codecov-action@v3
        with:
          files: lcov.info
          fail_ci_if_error: true
```

## Best Practices

### 1. Run Coverage Regularly
```bash
# Add to development workflow
alias cov='cargo llvm-cov clean --workspace && cargo llvm-cov --html && xdg-open target/llvm-cov/html/index.html'
```

### 2. Focus on Meaningful Coverage
- **Prioritize**: Error paths, edge cases, public API
- **Don't chase 100%**: Some code (unreachable panics, defensive checks) may not need tests
- **Document exceptions**: Add comments explaining why certain lines aren't tested

### 3. Use Coverage to Guide Testing
```bash
# Generate report
cargo llvm-cov --html

# Review uncovered lines in browser
# Write tests for critical uncovered paths
# Re-run to verify improvement
```

## Common Uncovered Patterns

### Acceptable Uncovered Code
```rust
// Unreachable panics (acceptable to leave uncovered)
let value = map.get(key).expect("key must exist by construction");

// Defensive checks (may be hard to test)
if index >= self.len() {
    panic!("index out of bounds"); // Hard to trigger safely in tests
}

// Debug-only code
#[cfg(debug_assertions)]
eprintln!("Debug info: {}", value);
```

### Should Be Covered
```rust
// Error paths - MUST be tested
Err(e) => {
    log::error!("Solver failed: {}", e);
    return Err(Error::SolverFailed(e));
}

// Public API - MUST be tested
pub fn solve(&mut self) -> Result<Solution, Error> {
    // All paths should be exercised
}

// Edge cases - SHOULD be tested
if cuts.is_empty() {
    return Err(Error::NoCutsAvailable);
}
```

## Testing Critical Modules

### SDDP Algorithm (`src/sddp/mod.rs` - 138KB)
**Priority**: Highest - Core algorithm

```bash
# Test SDDP with coverage
cargo llvm-cov --html -- sddp

# Focus on:
# - Forward pass scenarios
# - Backward pass cut generation
# - Convergence detection
# - Cut management
```

**Key test areas**:
- Forward pass with different scenario counts
- Backward pass cut computation
- Convergence criteria (absolute, relative gaps)
- Edge cases: zero scenarios, single stage, empty cuts

### Solver Integration (`src/solver.rs` - 45KB)
**Priority**: Highest - Correctness critical

```bash
# Test solver with coverage
cargo llvm-cov --html -- solver

# Focus on:
# - HiGHS FFI correctness
# - Infeasible/unbounded detection
# - Dual variable extraction
# - Error handling
```

**Key test areas**:
- Feasible LP solving
- Infeasible problem detection
- Unbounded problem detection
- Warm starting
- Dual variable retrieval for cuts

### Subproblem Formulation (`src/subproblem.rs` - 235KB)
**Priority**: Highest - Largest file, complex logic

```bash
# Test subproblem with coverage
cargo llvm-cov --html -- subproblem

# Focus on:
# - Constraint matrix construction
# - Objective function setup
# - State variable handling
# - Cut integration
```

**Key test areas**:
- Matrix construction for various problem sizes
- Constraint right-hand side updates
- State transition handling
- Cut evaluation and integration

### State Management (`src/state.rs` - 137KB)
**Priority**: High - Critical for correctness

```bash
cargo llvm-cov --html -- state
```

**Key test areas**:
- State variable updates
- Boundary handling
- State transition validation

### Error Handling (`src/error.rs`)
**Priority**: High - Using `thiserror` patterns

As documented in the file, POWE.RS uses `thiserror = "2.0"` for error handling.

```bash
cargo llvm-cov --html -- error
```

**Ensure coverage of**:
- All error variants
- Error context (file, field, value)
- Error conversion (From implementations)
- Display formatting

## Writing Tests for Coverage

### Pattern 1: Test Public API
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass_single_scenario() {
        let sddp = SDDPBuilder::new()
            .stages(5)
            .scenarios(1)
            .build();
        
        let result = sddp.forward_pass();
        assert!(result.is_ok());
    }

    #[test]
    fn test_forward_pass_multiple_scenarios() {
        // Test with realistic scenario count
    }
}
```

### Pattern 2: Test Error Paths
```rust
#[test]
fn test_infeasible_subproblem() {
    let subproblem = create_infeasible_subproblem();
    
    let result = subproblem.solve();
    assert!(result.is_err());
    
    match result {
        Err(Error::SolverError(SolverError::Infeasible)) => {
            // Expected error
        }
        _ => panic!("Expected infeasible error"),
    }
}
```

### Pattern 3: Test Edge Cases
```rust
#[test]
fn test_empty_cuts() {
    let sddp = SDDPBuilder::new()
        .stages(3)
        .build();
    
    // First iteration should handle no cuts gracefully
    let result = sddp.backward_pass();
    assert!(result.is_ok());
}
```

### Pattern 4: Property-Based Testing
POWE.RS includes `proptest = "1.4"` in dev-dependencies:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_cut_evaluation_commutative(
        a in -100.0..100.0_f64,
        b in -100.0..100.0_f64
    ) {
        let cut1 = Cut::new(vec![a, b], 0.0);
        let cut2 = Cut::new(vec![b, a], 0.0);
        
        let state1 = State::new(vec![1.0, 2.0]);
        let state2 = State::new(vec![2.0, 1.0]);
        
        // Property: evaluation should be symmetric
        assert_approx_eq!(
            cut1.evaluate(&state1),
            cut2.evaluate(&state2)
        );
    }
}
```

## Integration with Testing Scripts

POWE.RS includes testing scripts:

### Golden Tests
```bash
# Run golden tests with coverage
cargo llvm-cov --html -- --test-threads=1
./scripts/golden-tests.sh
```

### Example Runs
```bash
# Test with examples
./scripts/run_examples.sh
```

## Coverage Analysis Workflow

1. **Generate baseline coverage**
   ```bash
   cargo llvm-cov --html
   ```

2. **Identify low-coverage modules**
   - Open `target/llvm-cov/html/index.html`
   - Sort by coverage percentage
   - Focus on files < 90%

3. **Write targeted tests**
   - Review uncovered lines in HTML report
   - Identify critical uncovered paths
   - Write tests for those specific paths

4. **Verify improvement**
   ```bash
   cargo llvm-cov --html
   # Check coverage increased for target module
   ```

5. **Ensure correctness**
   ```bash
   cargo test
   cargo bench  # Ensure no performance regression
   ```

## Coverage Checklist

- [ ] Install cargo-llvm-cov and llvm-tools-preview
- [ ] Generate HTML coverage report: `cargo llvm-cov --html`
- [ ] Identify modules below target threshold (95% for critical modules)
- [ ] Review uncovered lines in HTML report
- [ ] Write tests for critical uncovered paths
- [ ] Verify error paths are covered (100% target)
- [ ] Verify public API is covered (100% target)
- [ ] Re-run coverage to confirm improvement
- [ ] Run full test suite: `cargo test`
- [ ] Document any intentionally uncovered code with comments

## Integration with Other Skills

- **rust-benchmarking**: Ensure tests don't regress performance
- **sddp-development**: Use SDDP documentation for realistic test scenarios
- **rust-clean-code**: Follow code quality standards in tests

## Common Issues

### Coverage Lower Than Expected
- Check if tests are actually running: `cargo test -- --nocapture`
- Ensure feature flags are enabled if needed
- Verify tests aren't being skipped (e.g., `#[ignore]` attribute)

### False Coverage (Green but Not Actually Tested)
- Review test assertions - ensure they actually verify behavior
- Use property-based testing with proptest for thorough coverage
- Add explicit error case tests

### Coverage Report Not Generated
- Ensure llvm-tools-preview is installed: `rustup component add llvm-tools-preview`
- Clean coverage data: `cargo llvm-cov clean`
- Check for compilation errors: `cargo build`

## File References

- **Coverage documentation**: `.copilot/development/COVERAGE-TOOLING.md`
- **Critical modules**: `src/sddp/mod.rs` (138KB), `src/solver.rs` (45KB), `src/subproblem.rs` (235KB)
- **State management**: `src/state.rs` (137KB)
- **Error handling**: `src/error.rs` (uses `thiserror`)
- **Test infrastructure**: `tests/` directory
- **Testing scripts**: `scripts/golden-tests.sh`, `scripts/run_examples.sh`
- **Dev dependencies**: `proptest = "1.4"`, `approx = "0.5"` in `Cargo.toml`

## Resources

- **cargo-llvm-cov Documentation**: https://github.com/taiki-e/cargo-llvm-cov
- **LLVM Coverage Mapping**: https://llvm.org/docs/CoverageMappingFormat.html
- **Rust Coverage Book**: https://doc.rust-lang.org/rustc/instrument-coverage.html
- **Project Documentation**: `.copilot/development/COVERAGE-TOOLING.md`
