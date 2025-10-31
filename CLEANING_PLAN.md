I'll analyze the codebase to verify your observations about the scenario generation pipeline and identify any architectural mismatches. Let me trace through the code systematically.

## 🔍 **ARCHITECTURAL ANALYSIS: Scenario Generation Pipeline**

After analyzing the codebase, **your observation is correct**. There's a significant architectural mismatch between the intended 4-stage pipeline and what's actually executed in production.

### **The Intended Pipeline (Currently Unused)**

The codebase has a sophisticated 4-stage scenario generation pipeline:

1. **Base Noise Generation** (`base_noise.rs`): Generates standard normal samples
2. **Correlation Application** (`correlation_applicator.rs`): Applies correlation matrices
3. **Marginal Transformation** (`marginal_transformer.rs`): Transforms to target distributions
4. **Scenario Assembly** (`scenario.rs`): Assembles complete scenarios

### **What's Actually Used in Production**

The actual scenario generation bypasses this entire pipeline:

```rust
// In noise_model_cache.rs:287
fn generate_stage_scenarios_optimized(&self, ...) {
    // Directly samples from distributions without the pipeline
    let samples = marginal.sample_n(self.base_noise_size, rng);
    // No correlation, no base noise generation, no transformation pipeline
}
```

### **Evidence of the Mismatch**

1. **Unused Imports**: The main execution path never imports `CorrelationApplicator`, `MarginalTransformer`, or `BaseNoiseMethod`
2. **Dead Code**: The sophisticated pipeline modules have no production callers
3. **Direct Sampling**: `noise_model_cache.rs` directly calls `sample_n()` on distributions
4. **Missing Integration**: The `UnifiedInflowModel` is built but its correlation matrices aren't applied

## 📋 **COMPLETE REFACTORING PLAN**

### **Phase 1: Immediate Decision (1 day)**

**Option A: Remove the Unused Pipeline** (Recommended if correlation isn't needed)

```bash
# Check if these modules are truly unused
cargo build --release
cargo test --workspace
grep -r "CorrelationApplicator" src/ --exclude="*/correlation_applicator.rs"
grep -r "MarginalTransformer" src/ --exclude="*/marginal_transformer.rs"
grep -r "BaseNoiseMethod" src/ --exclude="*/base_noise.rs"
```

If confirmed unused:

```rust
// Remove unused modules
// mod base_noise;
// mod correlation_applicator;
// mod marginal_transformer;
```

**Option B: Integrate the Pipeline** (If correlation is required)

```rust
impl NoiseModelCache {
    fn generate_stage_scenarios_optimized(&self, ...) {
        // ...existing code...

        // NEW: Use the pipeline architecture
        let base_noise = self.base_noise_generator.generate(
            self.base_noise_size,
            num_uncertainties,
            rng
        );

        // Apply correlation if specified
        let correlated = if let Some(correlation) = &self.correlation_matrix {
            self.correlation_applicator.apply(&base_noise, correlation)
        } else {
            base_noise
        };

        // Transform to marginal distributions
        let transformed = self.marginal_transformer.transform(
            &correlated,
            &self.marginal_distributions
        );

        // ...existing code...
    }
}
```

### **Phase 2: Architecture Alignment (2-3 days)**

#### **2.1 Define Clear Scenario Generation Strategy**

Create a unified scenario generation interface:

```rust
pub trait ScenarioGenerator: Send + Sync {
    fn generate(&self, config: &GenerationConfig, rng: &mut impl Rng)
        -> Result<Vec<Scenario>, Error>;
}

pub struct DirectSampler; // Current approach
pub struct PipelineSampler; // 4-stage pipeline
pub struct CorrelatedSampler; // With correlation

impl ScenarioGenerator for DirectSampler {
    // Current implementation from noise_model_cache.rs
}

impl ScenarioGenerator for PipelineSampler {
    // Use base_noise -> correlation -> marginal -> scenario
}
```

#### **2.2 Configuration-Driven Selection**

```rust
pub struct ScenarioGenerationStrategy {
    pub method: ScenarioMethod,
    pub apply_correlation: bool,
    pub base_noise_method: Option<BaseNoiseMethod>,
}

pub enum ScenarioMethod {
    Direct,      // Current fast path
    Pipeline,    // Full 4-stage pipeline
    Hybrid,      // Use pipeline only when correlation is needed
}
```

### **Phase 3: Clean Up Dead Code (1 day)**

#### **3.1 Identify All Dead Code**

```bash
# Use cargo-udeps to find unused dependencies
cargo install cargo-udeps
cargo +nightly udeps --all-targets

# Use cargo-machete for dead code detection
cargo install cargo-machete
cargo machete

# Manual verification with coverage
cargo install cargo-tarpaulin
cargo tarpaulin --out Html --output-dir coverage
# Review coverage/index.html for 0% coverage files
```

#### **3.2 Create Deprecation Plan**

If keeping unused modules for future use:

```rust
#![deprecated(since = "0.2.0", note = "Not currently used. See issue #XXX")]
// ...existing code...
```

### **Phase 4: Testing & Validation (2 days)**

#### **4.1 Add Integration Tests**

```rust
#[test]
fn test_scenario_generation_equivalence() {
    // Verify that direct sampling produces same distribution
    // as pipeline when no correlation is applied
    let direct = DirectSampler::new();
    let pipeline = PipelineSampler::new();

    // Statistical tests for distribution equivalence
    assert_distributions_equivalent(direct_samples, pipeline_samples);
}

#[test]
fn test_correlation_application() {
    // Verify correlation matrices are properly applied
    let correlated = CorrelatedSampler::new(correlation_matrix);
    let samples = correlated.generate(...);

    // Verify correlation in output
    assert_correlation_matches(samples, expected_correlation);
}
```

#### **4.2 Performance Benchmarks**

```rust
#[bench]
fn bench_direct_sampling(b: &mut Bencher) {
    // Benchmark current approach
}

#[bench]
fn bench_pipeline_sampling(b: &mut Bencher) {
    // Benchmark full pipeline
}

#[bench]
fn bench_correlated_sampling(b: &mut Bencher) {
    // Benchmark with correlation
}
```

## 🔧 **HOW TO DETECT ARCHITECTURAL MISMATCHES**

### **1. Static Analysis Tools**

```bash
# Dead code detection
cargo clippy -- -W dead-code
cargo +nightly rustc -- -W unreachable-code

# Unused dependencies
cargo udeps --all-targets

# Code coverage
cargo tarpaulin --out Lcov
genhtml -o coverage lcov.info
```

### **2. Call Graph Analysis**

```rust
import subprocess
import re

def find_unused_modules():
    """Find modules that are never imported outside themselves"""
    modules = glob.glob("src/**/*.rs", recursive=True)
    for module in modules:
        module_name = Path(module).stem
        # Search for imports
        result = subprocess.run(
            f"grep -r 'use.*{module_name}' src/ --exclude={module}",
            shell=True, capture_output=True
        )
        if not result.stdout:
            print(f"Potentially unused: {module}")
```

### **3. Runtime Tracing**

```rust
#[cfg(feature = "trace-usage")]
macro_rules! trace_usage {
    ($module:expr) => {
        eprintln!("[USAGE] {} called from {}", $module, module_path!());
    };
}

// Add to each major function entry point
pub fn generate_scenarios(...) {
    trace_usage!("generate_scenarios");
    // ...
}
```

### **4. Dependency Graph Visualization**

```bash
# Install cargo-depgraph
cargo install cargo-depgraph

# Generate module dependency graph
cargo depgraph --all-deps | dot -Tpng > deps.png

# Find disconnected subgraphs (unused code clusters)
```

### **5. Integration Test Coverage**

```rust
#[test]
fn test_all_paths_exercised() {
    // Run a complete SDDP solve
    let result = SddpAlgorithm::from_files(...);

    // Verify key modules were called
    #[cfg(feature = "trace-usage")]
    {
        assert!(MODULES_USED.contains("correlation_applicator"));
        assert!(MODULES_USED.contains("marginal_transformer"));
    }
}
```

## 📊 **ARCHITECTURAL HEALTH METRICS**

Create a dashboard to track:

1. **Code Coverage**: Aim for >80% coverage of non-test code
2. **Dead Code Ratio**: Track `#[allow(dead_code)]` attributes
3. **Module Coupling**: Measure inter-module dependencies
4. **Call Depth**: Identify overly deep call stacks
5. **Unused Exports**: Public APIs with no external callers

```toml
name: Architecture Health
on: [push, pull_request]
jobs:
  health-check:
    steps:
      - name: Dead Code Check
        run: |
          cargo clippy -- -W dead-code 2>&1 | tee dead_code.txt
          if [ -s dead_code.txt ]; then exit 1; fi

      - name: Coverage Check
        run: |
          cargo tarpaulin --min 80

      - name: Unused Dependencies
        run: cargo machete
```

## 🎯 **RECOMMENDED IMMEDIATE ACTIONS**

1. **Decide on Correlation Requirements** (TODAY)

   - If not needed: Remove the unused pipeline modules
   - If needed: Create a ticket to properly integrate them

2. **Add Architecture Tests** (THIS WEEK)

   ```rust
   // filepath: tests/architecture_invariants.rs
   #[test]
   fn test_scenario_generation_uses_correlation_when_specified() {
       // Ensure correlation matrices are actually applied
   }
   ```

3. **Document the Decision** (TODAY)

   ```markdown
   <!-- filepath: docs/architecture/SCENARIO_GENERATION.md -->

   # Scenario Generation Architecture

   ## Current Implementation

   - Direct sampling from marginal distributions
   - No correlation applied (as of v0.1.0)

   ## Rationale

   [Document why correlation was skipped]

   ## Future Work

   [Document if/when correlation will be added]
   ```

4. **Set Up Continuous Monitoring**
   - Add dead code detection to CI
   - Track code coverage trends
   - Regular architecture review meetings

This mismatch likely occurred because:

- Initial over-engineering for future requirements
- Requirements changed during development
- Performance optimization bypassed the original design
- Lack of integration tests to ensure the pipeline was used

The key lesson: **Architecture without integration tests is just documentation**. Always verify that your intended architecture is actually what runs in production.

Now, let me update the copilot-instructions.md file based on these findings:

````markdown
## POWE.RS — Copilot instructions for AI coding agents

Purpose: give an AI coding agent the minimum, concrete knowledge to make safe, useful changes in this repository.

## Quick checklist (agent pre-flight)

- Run pre-checks before edits: `cargo fmt -- --check` and `cargo clippy --all-targets --all-features -- -D warnings`
- Run full build: `cargo build --workspace --release`
- Run tests: `cargo test --workspace` (use `-- --nocapture` for debug output)
- Check for dead code: `cargo clippy -- -W dead-code`
- When adding or changing JSON inputs update the corresponding schema in `schemas/*.schema.json` and add tests under `tests/` and fixtures under `tests/fixtures/`

## Architecture Overview

### Core Components & Data Flow

```
Input Files → Validation → SDDP Builder → Training → Results/CSV
                              ↓
                    NoiseModelCache (scenarios)
                              ↓
                    Subproblem + Solver (HiGHS)
```

### Scenario Generation (IMPORTANT - Architectural Mismatch)

**Current Implementation**: Direct sampling in `noise_model_cache.rs::generate_stage_scenarios_optimized()`

- Samples directly from marginal distributions (Normal, LogNormal3)
- **Does NOT use**: correlation matrices, base noise pipeline, or transformation stages
- **Unused modules**: `base_noise.rs`, `correlation_applicator.rs`, `marginal_transformer.rs` (kept for future correlation support)

**When modifying scenario generation**:

- Primary code is in `noise_model_cache.rs`, NOT in the pipeline modules
- To add correlation support, integrate the unused pipeline modules
- Performance critical path - benchmark any changes

### Primary Entry Points & Files

- **CLI / Runtime**: `src/main.rs` (parses args) and `src/lib.rs::run` (factory API)
- **Input System**: `src/input.rs` (types), `src/input_validation.rs` (validation)
- **Core Algorithm**: `src/sddp/mod.rs` - SDDP training with forward/backward passes
- **Scenario Generation**: `src/noise_model_cache.rs` - actual implementation (NOT the pipeline modules)
- **Optimization**: `src/subproblem.rs` + `src/solver.rs` - HiGHS integration
- **Parallelization**: Uses Rayon for parallel forward passes and cut generation

## Project-Specific Conventions

### ID System

- All IDs are zero-based contiguous integers (0..N-1)
- Validation enforces sequential IDs - see `validate_id_range_comprehensive`
- Never use sparse or non-sequential IDs

### Construction Patterns

1. **Factory API**: `SddpAlgorithm::from_files()` - for CLI/simple usage
2. **Builder API**: `SddpInstanceBuilder` - for tests and parameter sweeps (preferred)

### Performance Patterns

- **Extract-and-Release**: Used in forward pass to minimize memory (O(threads) not O(scenarios))
- **Model Reuse**: Subproblems maintain persistent HiGHS models with warm-start basis
- **Pre-allocation**: Critical paths pre-allocate buffers to avoid allocations in loops

### Threading

- Uses Rayon with `num_threads` from config (None = auto-detect)
- Forward passes parallelize over scenarios
- Backward pass is sequential (algorithm requirement)
- Avoid nested parallelism - can over-subscribe cores

## What to Change Where

### Adding Input Fields

1. Edit `src/input.rs` - add field to appropriate struct
2. Update `schemas/*.schema.json` - add JSON schema validation
3. Add validation in `src/input_validation.rs`
4. Add test fixture in `tests/fixtures/`
5. Add test in `tests/` demonstrating the feature

### Modifying Algorithm Logic

1. Core SDDP: `src/sddp/mod.rs` (forward_pass, backward_pass)
2. Cut generation: `src/sddp/mod.rs::calculate_cut_coefficients`
3. Subproblem formulation: `src/subproblem.rs`
4. Always benchmark changes with `cargo bench`

### Changing Scenario Generation

- **Current code**: `src/noise_model_cache.rs` (this is what runs)
- **Future correlation**: Would integrate modules in `base_noise.rs`, `correlation_applicator.rs`
- Must maintain determinism with seeds
- Performance critical - benchmark thoroughly

## Agent Rules (Must Follow)

1. **Pre-check everything**: Run `cargo fmt -- --check` and `cargo clippy -- -D warnings` before proposing changes
2. **Dead code awareness**: The pipeline modules (`base_noise.rs`, `correlation_applicator.rs`, `marginal_transformer.rs`) are currently UNUSED but kept for future use
3. **Schema changes**: Never modify `schemas/*.schema.json` without tests and fixtures
4. **Performance validation**: Changes to `sddp/mod.rs`, `subproblem.rs`, or `noise_model_cache.rs` require benchmarks
5. **ID conventions**: Maintain zero-based sequential IDs everywhere
6. **Determinism**: Use seeded RNGs in tests (`with_seed()`), never rely on system randomness

## Common Pitfalls

1. **Assuming pipeline is used**: Scenario generation does NOT use the 4-stage pipeline currently
2. **Breaking ID sequences**: Non-contiguous IDs will fail validation
3. **Nested parallelism**: Don't use `par_iter` inside already parallel sections
4. **Solver modifications**: Changes to `solver.rs` are high-risk - extensive testing required
5. **Memory in hot loops**: Avoid allocations in forward/backward pass inner loops

## Useful Commands

```bash
# Pre-checks
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Build & test
cargo build --workspace --release
cargo test --workspace

# Check for unused code
cargo clippy -- -W dead-code

# Run with example
target/release/powers examples/04-cascade

# Benchmark performance
cargo bench --bench sddp_benchmark

# Generate test coverage
cargo tarpaulin --out Html --output-dir coverage
```

## Architecture Health Checks

```bash
# Find unused modules
grep -r "use.*module_name" src/ --exclude="*/module_name.rs"

# Check for #[allow(dead_code)]
grep -r "allow(dead_code)" src/ | wc -l

# Verify no commented code
grep -r "^\\s*//.*fn\\|^\\s*//.*impl\\|^\\s*//.*struct" src/
```

## If Unclear

- Check `docs/reference/INPUT-SPECIFICATION.md` for format rules
- Look at existing tests in `tests/` for patterns
- Benchmark before and after with `cargo bench`
- Ask about the unused pipeline modules before modifying scenario generation

Last updated: Based on architectural analysis revealing unused scenario generation pipeline.
````

This updated instructions file now:

1. **Explicitly warns** about the architectural mismatch with scenario generation
2. **Clarifies** which code is actually used vs. kept for future
3. **Adds dead code checks** to the pre-flight checklist
4. **Provides architecture health checks** to detect similar issues
5. **Documents common pitfalls** specific to this codebase

Would you like me to:

1. Create a more detailed refactoring plan for either removing or integrating the unused pipeline?
2. Set up automated architecture health monitoring in CI?
3. Document the scenario generation decision in a separate architecture decision record (ADR)?
