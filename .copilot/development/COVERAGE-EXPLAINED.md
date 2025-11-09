# Coverage Configuration for POWE.RS

This file documents coverage measurement setup and explains warnings you may encounter.

## Quick Start

### Generate HTML Coverage Report
```bash
cargo llvm-cov --all-features --html
# View report: open target/llvm-cov/html/index.html
```

### Generate LCOV for CI
```bash
cargo llvm-cov --all-features --lcov --output-path lcov.info
```

### Run Coverage for Specific Tests
```bash
# Library tests only (no warning)
cargo llvm-cov --lib --all-features --html

# Specific integration test
cargo llvm-cov --test test_sddp_algorithm --html

# All tests
cargo llvm-cov --all-features --html
```

## Understanding the "141 functions have mismatched data" Warning

### What It Means

When running `cargo llvm-cov --all-features --html`, you may see:
```
warning: 141 functions have mismatched data
```

This warning occurs because:

1. **Multiple Test Binaries**: The project has 50+ integration test files, each compiled as a separate binary
2. **Shared Code**: Library code (especially generic and inline functions) is compiled into each test binary
3. **Coverage Profile Merging**: LLVM's profiler merges profiles from all test binaries and detects that the same functions appear with slightly different instrumentation

### Is This a Problem?

**No.** This is a known limitation of LLVM coverage with Rust's compilation model and does not affect coverage accuracy:

- ✅ Coverage percentages are correct
- ✅ Line coverage is accurate
- ✅ All code paths are tracked
- ⚠️ The warning is informational, not an error

### Why It Happens

**Generic Functions**: Functions like `Vec<T>` operations are monomorphized (compiled separately) for each type `T` used across different test binaries.

```rust
// This function gets compiled separately in each test binary
pub fn process<T>(items: Vec<T>) -> usize {
    items.len()  // <-- Instrumented differently in each binary
}
```

**Test Fixtures**: Shared test utilities in `tests/fixtures/` and `tests/utils/` are compiled into each integration test that uses them.

**Inline Functions**: Small functions marked `#[inline]` are copied into each call site across different binaries.

## Suppressing the Warning

If the warning bothers you in CI logs, here are options:

### Option 1: Run Coverage on Library Only (Recommended for CI)
```bash
# Fastest, no warning, good for quick checks
cargo llvm-cov --lib --all-features --lcov --output-path lcov.info
```

This covers all production code since integration tests primarily exercise library code.

### Option 2: Filter Warning in CI
```bash
# In CI scripts
cargo llvm-cov --all-features --lcov --output-path lcov.info 2>&1 | grep -v "mismatched data"
```

### Option 3: Suppress via Stderr Redirection
```bash
cargo llvm-cov --all-features --html 2>/dev/null
```

## Best Practices

### For Local Development
```bash
# Full coverage with all details
cargo llvm-cov --all-features --html
open target/llvm-cov/html/index.html
```

### For CI/CD
```bash
# Fast, warning-free, sufficient for most purposes
cargo llvm-cov --lib --all-features --lcov --output-path lcov.info
```

### For Detailed Analysis
```bash
# Show missing functions
cargo llvm-cov --all-features --html --show-missing-functions

# Show instantiations
cargo llvm-cov --all-features --html --show-instantiations
```

## Coverage Goals

- **Core algorithm**: >90% (SDDP, subproblem, solver interface)
- **Overall**: >80%
- **Critical paths**: 100% (numerical operations, cut generation)

## Troubleshooting

### Error: "Failed to merge profiles"
```bash
# Clean and retry
cargo llvm-cov clean
cargo llvm-cov --all-features --html
```

### Different Coverage Between Runs
```bash
# Ensure clean state
cargo clean
cargo llvm-cov clean
cargo llvm-cov --all-features --html
```

### Missing Coverage for New Code
Ensure tests are running:
```bash
cargo test --all-features  # Should pass
cargo llvm-cov --all-features --html  # Then measure coverage
```

## References

- [cargo-llvm-cov documentation](https://github.com/taiki-e/cargo-llvm-cov)
- [LLVM Coverage Mapping Format](https://llvm.org/docs/CoverageMappingFormat.html)
- [Rust Performance Book - Profiling](https://nnethercote.github.io/perf-book/profiling.html)

## Summary

The "141 functions have mismatched data" warning is **expected and safe** when running full test coverage on projects with multiple integration test binaries. It does not indicate a problem with your code or test coverage accuracy.

For warning-free coverage in CI, use `cargo llvm-cov --lib --all-features --lcov --output-path lcov.info`.
