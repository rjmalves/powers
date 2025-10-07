# Coverage Tooling Preferences

## Preferred Tool: cargo-llvm-cov

**Why llvm-cov over tarpaulin**:

- **Accuracy**: Uses LLVM's native coverage instrumentation (same as Clang/LLVM C/C++)
- **Performance**: Faster execution than tarpaulin (no ptrace overhead)
- **Compatibility**: Better support for inline functions and generics
- **Integration**: Works seamlessly with Rust's LLVM-based compilation
- **Output Formats**: Supports HTML, lcov, JSON, and text formats
- **Line-level precision**: More accurate line-by-line coverage tracking

## Installation

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Install llvm-tools-preview component (required)
rustup component add llvm-tools-preview
```

## Usage

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
# Filename                      Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       Lines      Missed Lines     Cover    Branches   Missed Branches     Cover
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# src/lib.rs                        245                15    93.88%          89                 3    96.63%         890                147    83.48%           0                 0         -
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TOTAL                             245                15    93.88%          89                 3    96.63%         890                147    83.48%           0                 0         -
```

### Generate lcov for CI/CD

```bash
# Generate lcov.info for codecov.io or similar
cargo llvm-cov --lcov --output-path target/llvm-cov/lcov.info
```

### Run Coverage with Specific Tests

```bash
# Run specific test file
cargo llvm-cov --html -- --test test_sddp_algorithm

# Run tests matching pattern
cargo llvm-cov --html -- forward_pass

# Include ignored tests
cargo llvm-cov --html -- --include-ignored
```

### Clean Coverage Data

```bash
# Clean previous coverage data
cargo llvm-cov clean

# Clean and regenerate
cargo llvm-cov clean --workspace && cargo llvm-cov --html
```

## Best Practices

### 1. Run Coverage Regularly

```bash
# Add to your development workflow
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

### 4. CI/CD Integration

```yaml
# .github/workflows/coverage.yml example
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

## Interpreting Results

### Coverage Metrics

- **Line Coverage**: % of lines executed (primary metric)
- **Region Coverage**: % of code regions executed (LLVM concept)
- **Function Coverage**: % of functions called
- **Branch Coverage**: % of conditional branches taken

### Target Thresholds

- **Production Libraries**: 90-95%+ line coverage
- **Critical Modules**: 95-98%+ (sddp.rs, solver.rs, subproblem.rs)
- **Error Paths**: 100% (all error handling tested)
- **Public API**: 100% (all public functions exercised)

### Reading HTML Report

1. **Green lines**: Covered by tests
2. **Red lines**: Not covered (need tests)
3. **Orange lines**: Partially covered (some branches not tested)
4. **Gray lines**: Not executable (comments, declarations)

### Common Uncovered Patterns

```rust
// Unreachable panics (acceptable to leave uncovered)
let value = map.get(key).expect("key must exist by construction");

// Defensive checks (may be hard to test)
if index >= self.len() {
    panic!("index out of bounds"); // Hard to trigger safely
}

// Error paths that require complex setup
Err(e) => {
    log::error!("Solver failed: {}", e);
    return Err(Error::SolverFailed(e)); // Need integration test
}
```

## Comparison with Tarpaulin

| Feature                  | cargo-llvm-cov | cargo-tarpaulin |
|--------------------------|----------------|-----------------|
| **Speed**                | ⚡ Fast        | 🐢 Slower       |
| **Accuracy**             | ✅ High        | ⚠️ Medium       |
| **Inline functions**     | ✅ Accurate    | ⚠️ Issues       |
| **Generic functions**    | ✅ Accurate    | ⚠️ Issues       |
| **LLVM integration**     | ✅ Native      | ❌ No           |
| **Output formats**       | ✅ Many        | ✅ Many         |
| **Rustup component**     | ✅ Yes         | ❌ No           |
| **Community adoption**   | 📈 Growing     | 📊 Established  |

## Migration from Tarpaulin

If you have existing tarpaulin configuration:

```toml
# Old: tarpaulin.toml (remove this file)
[tarpaulin]
exclude-files = ["tests/*"]

# New: Add to Cargo.toml [workspace.metadata] if needed
[package.metadata.coverage]
exclude-files = ["tests/*"]
```

Most projects can simply switch to `cargo llvm-cov` without configuration changes.

## References

- [cargo-llvm-cov GitHub](https://github.com/taiki-e/cargo-llvm-cov)
- [LLVM Coverage Mapping](https://llvm.org/docs/CoverageMappingFormat.html)
- [Rust Coverage Book](https://doc.rust-lang.org/rustc/instrument-coverage.html)

---

**Last Updated**: October 7, 2025  
**Project**: POWE.RS (powers-rs)  
**Status**: Active - Preferred tool for coverage analysis
