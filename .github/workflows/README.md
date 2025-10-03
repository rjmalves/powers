# GitHub Actions Workflows

This directory contains CI/CD workflows for the POWE.RS project.

## Available Workflows

### `test.yml` - Main Test Suite

**Triggers:**

- Push to `main` or `master` branch
- Pull requests targeting `main` or `master`

**What it does:**

1. **Code Formatting Check** - Ensures code follows Rust formatting standards
2. **Linting (Clippy)** - Catches common mistakes and enforces best practices
3. **Build** - Compiles the project
4. **Test Suite** - Runs all 312 tests (unit, integration, and doc tests)

**Performance:**

- **Target:** < 10 minutes
- **Typical:** 3-5 minutes (with caching)
- **First run:** 8-10 minutes (cold cache)

**Caching Strategy:**

- Cargo registry and git index
- Build artifacts (target directory)
- Cache key based on `Cargo.lock` and source files
- Significantly speeds up subsequent runs

---

## Running Tests Locally

To run the same checks that CI runs:

### 1. Format Check

```bash
cargo fmt --all -- --check
```

### 2. Linting

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### 3. Build

```bash
cargo build --verbose
```

### 4. Run Tests

```bash
cargo test --verbose --all-features
```

### Run All Checks at Once

```bash
# Quick script to run all CI checks
cargo fmt --all -- --check && \
cargo clippy --all-targets --all-features -- -D warnings && \
cargo build --verbose && \
cargo test --verbose --all-features
```

---

## Modifying CI

### Adding a New Test Stage

Edit `.github/workflows/test.yml`:

```yaml
- name: Your New Check
  run: cargo your-command
```

Add it after the existing test stages but before the summary.

### Changing Rust Version

Currently using `stable`. To test with nightly:

```yaml
- name: Install Rust toolchain
  uses: dtolnay/rust-toolchain@nightly # Change to nightly
  with:
    components: rustfmt, clippy
```

### Adjusting Timeout

Default timeout is 15 minutes:

```yaml
jobs:
  test:
    timeout-minutes: 20 # Increase if needed
```

### Modifying Caching

Cache configuration in the workflow:

```yaml
- name: Cache cargo build
  uses: actions/cache@v3
  with:
    path: target
    key: ${{ runner.os }}-cargo-build-${{ hashFiles('**/Cargo.lock') }}-${{ hashFiles('**/*.rs') }}
```

**Cache Strategy:**

- Key includes `Cargo.lock` hash (dependencies)
- Key includes source file hashes (rebuilds when code changes)
- Restore keys provide fallback caches

---

## Troubleshooting

### CI Fails but Local Tests Pass

**Common Causes:**

1. **Formatting differences** - Run `cargo fmt --all` locally
2. **Clippy warnings** - Fix warnings shown in CI output
3. **Missing dependencies** - HiGHS requires cmake and build-essential
4. **Environment differences** - CI runs on Ubuntu; check OS-specific issues

**Debug Steps:**

```bash
# Run exact CI commands locally
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --verbose --all-features
```

### Tests are Flaky in CI

**Potential Issues:**

- Non-deterministic tests (should use fixed seeds)
- Timing-sensitive tests
- Resource constraints in CI

**Solution:**

- All tests use fixed seeds (verified in T1.1-T1.6)
- Tests are deterministic
- If issues persist, check GitHub Actions logs

### CI is Slow

**Performance Tips:**

1. **Check cache effectiveness:**

   - Look for "Cache hit" in workflow logs
   - Verify cache keys are correct

2. **Parallel test execution:**

   - Cargo runs tests in parallel by default
   - Use `RUST_TEST_THREADS` to control parallelism

3. **Incremental compilation:**
   - Caching should preserve incremental builds
   - Check that `target/` cache is working

**Current Performance:**

```
Typical CI run (with cache):
- Checkout: ~5s
- Setup Rust: ~10s
- Restore cache: ~20s
- Install deps: ~15s
- Format check: ~5s
- Clippy: ~30s
- Build: ~60s (cached incremental)
- Tests: ~60s
- Total: ~3-4 minutes
```

### HiGHS Dependencies Fail to Install

**Error:** `cmake: command not found` or build failures

**Solution:**
The workflow installs dependencies:

```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y cmake build-essential
```

If this fails, check GitHub Actions logs for apt-get errors.

### Cache Size Issues

**Symptom:** Slow cache restore or cache misses

**Solution:**

- GitHub Actions has 10GB cache limit per repo
- Caches are evicted after 7 days of no use
- Consider clearing old caches if issues persist

---

## CI Best Practices

### For Contributors

1. **Run checks locally before pushing:**

   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features
   cargo test --all-features
   ```

2. **Keep tests deterministic:**

   - Use fixed seeds for random number generation
   - Avoid timing-sensitive assertions
   - Don't depend on external services

3. **Watch CI status:**
   - Check CI results on your PR
   - Fix failures promptly
   - Don't merge until CI passes ✅

### For Maintainers

1. **Monitor CI performance:**

   - Check workflow duration trends
   - Investigate if runs exceed 10 minutes
   - Optimize caching if needed

2. **Keep dependencies updated:**

   - Rust toolchain (stable channel auto-updates)
   - GitHub Actions versions
   - System dependencies

3. **Review CI logs:**
   - Check for warnings even if tests pass
   - Look for performance degradation
   - Monitor cache effectiveness

---

## Performance Characteristics

### Test Suite Breakdown

```
Total: 312 tests
├─ Unit Tests: 286
│  ├─ lib.rs: 40 tests
│  ├─ test_cut.rs: 57 tests
│  ├─ test_cut_pool.rs: 46 tests
│  ├─ test_infrastructure.rs: 36 tests
│  ├─ test_scenario.rs: 53 tests
│  └─ test_state.rs: 54 tests
├─ Integration Tests: 21
│  └─ integration_simple_2stage.rs: 21 tests
└─ Doc Tests: 5
```

### Typical Timing (with cache)

```
Format check:    ~5s
Clippy:         ~30s
Build:          ~60s (incremental)
Tests:          ~60s
────────────────────
Total:         ~3-4 minutes
```

### Cold Cache Performance

```
Format check:    ~5s
Clippy:         ~90s (full analysis)
Build:         ~180s (full build)
Tests:          ~60s
────────────────────
Total:        ~8-10 minutes
```

---

## GitHub Actions Environment

**Runner:** `ubuntu-latest` (currently Ubuntu 22.04)

**Pre-installed:**

- Git
- Curl, wget
- Build essentials
- Python

**Installed by workflow:**

- Rust (stable)
- rustfmt, clippy
- cmake, build-essential (for HiGHS)

**Resources:**

- CPU: 2 cores
- RAM: 7 GB
- Disk: 14 GB (SSD)

---

## Future Enhancements

**Planned (not in this ticket):**

1. **Code Coverage** (T1.9)

   - Add tarpaulin or cargo-llvm-cov
   - Upload to codecov.io or similar
   - Track coverage trends

2. **Multiple Rust Versions**

   - Test on stable, beta, and nightly
   - Catch regressions early

3. **Multi-OS Testing**

   - Linux (current)
   - macOS
   - Windows

4. **Benchmark Tracking** (Sprint 3)

   - Run benchmarks on each commit
   - Detect performance regressions
   - Track optimization improvements

5. **Automated Releases**
   - Publish to crates.io
   - Create GitHub releases
   - Generate changelog

---

## Support

**Issues with CI?**

- Check this documentation first
- Review GitHub Actions logs
- Search existing issues
- Open a new issue with CI logs attached

**Questions?**

- Open a discussion in the repository
- Tag maintainers in your PR if CI fails unexpectedly

---

_Last updated: October 3, 2025_  
_Workflow version: 1.0_
