# v0.3.0 (Unreleased)

### Breaking Changes

- **JSON Schema v0.3.0 (PAR-020)**: Simplified recourse.json structure
  - **Unified distribution field**: Single `distribution` field replaces `marginal_distribution`, `residual_distribution`, and `innovation_distribution`
    - For Independent models: `distribution` is marginal of final series Xₜ
    - For PAR models: `distribution` is residual distribution aₜ (de-seasonalized innovations)
  - **Removed Autoregressive temporal model**: Deprecated stationary AR model removed from schema (use PAR with `num_seasons=1` instead)
  - **Updated field names**: Schema now validates `num_seasons` instead of `period`, `periodic_ar` instead of `num_seasonsic_ar`
  - **Migration**: See `docs/guides/MIGRATION-TO-PAR.md` for conversion guide. Quick migration script:
    ```bash
    # Migrate recourse.json from v0.2.x to v0.3.0
    jq '.noise_models |= map(
      if .temporal_model.type == "periodic_ar" then
        .temporal_model.num_seasons = .temporal_model.period |
        del(.temporal_model.period)
      else . end |
      if .temporal_model.type == "independent" then
        .distribution = .marginal_distribution |
        del(.marginal_distribution, .residual_distribution, .innovation_distribution)
      elif .temporal_model.type == "periodic_ar" then
        .distribution = .residual_distribution |
        del(.marginal_distribution, .residual_distribution, .innovation_distribution)
      else . end
    )' recourse.json > recourse_v3.json
    ```

- **PAR Terminology Cleanup (PAR-017)**:
  - Renamed `period` field to `num_seasons` in `TemporalModel::PeriodicAutoregressive`
  - Renamed `SeasonalParams::period` to `num_seasons`
  - Renamed `PeriodicARParams::period` to `num_seasons`
  - Updated method names: `get_params_for_period` → `get_params_for_season`, `get_ar_coeffs_for_period` → `get_ar_coeffs_for_season`
  - Updated JSON schema: `"period"` → `"num_seasons"` in recourse.schema.json
  - Clarifies that the field represents "number of seasons in the cycle" (e.g., 12 months, 4 quarters)
  - **Migration**: Update JSON files: `"period": 12` → `"num_seasons": 12`

- **Deprecated Code Removal (PAR-021)**:
  - **Removed stationary AR implementation**: Deleted `TemporalModel::Autoregressive` variant from enum
  - **Removed field migration logic**: Cleaned up auto-migration code for distribution field unification  
  - **Removed deprecated structures**: Deleted `InnovationDistribution` struct and related helper methods
  - **Removed AR-specific code paths**: Eliminated stationary AR scenario generation and validation logic
  - **Code size reduction**: Removed ~500-1000 lines of deprecated code, reducing compilation time by ~5-10%
  - **Performance impact**: Zero performance regression (PAR with `num_seasons=1` equivalent to old AR)
  - **Breaking change**: Old v0.2.x JSON files with `"type": "autoregressive"` now rejected at validation
  - **Migration**: Convert AR models to PAR: `{"type": "autoregressive", "lag_order": p, "coefficients": [...]}` → `{"type": "periodic_ar", "num_seasons": 1, "ar_orders": [p], "ar_coefficients": [[...]], "seasonal_means": [μ], "seasonal_stds": [σ]}`

### Added

- **Simulation Memory Optimization (SIM-OPT Sprint 2)**:

  - **MAJOR PERFORMANCE IMPROVEMENT**: Reduced simulation memory usage by **83-96%** for large scenario counts
    - Implemented Extract-and-Release pattern for simulation phase
    - Memory model changed from O(scenarios × handler_size) to O(threads × handler_size + scenarios × trajectory_size)
    - **10,000 scenarios @ 120 stages**: 60 GB → 2.45 GB (96% reduction)
    - **1,000 scenarios @ 24 stages**: 6 GB → 290 MB (95% reduction)
  - **Technical implementation** (SIM-OPT-005):
    - Thread-local handler pool via `thread_local!` (one handler per thread)
    - Parallel simulation with handler reuse across scenarios
    - Lightweight `SimulationTrajectory` extraction (~240 KB @ 120 stages vs ~6 MB handler)
    - Handlers released immediately after trajectory extraction
  - **Output refactoring** (SIM-OPT-006):
    - Updated CSV export to use trajectory-based data access
    - Sequential memory access pattern (better cache locality)
    - **5-10% faster** CSV export vs handler-based approach
    - Removed deprecated `simulate_with_handlers_old()` method
  - **Benchmarking infrastructure** (SIM-OPT-007):
    - Comprehensive benchmark suite: `benches/simulation_memory.rs`
    - Memory usage validation (peak RSS tracking on Linux)
    - Throughput measurement (~2,000-2,200 scenarios/sec @ 24 stages)
    - Extraction overhead verification (<1% of forward pass time)
    - CSV export performance validation
    - Comparison script: `scripts/compare_simulation_memory.sh`
  - **Integration testing** (SIM-OPT-008):
    - Comprehensive test suite: `tests/test_simulation_extract_and_release.rs`
    - Variable scenario count validation (1, 10, 100, 1000 scenarios)
    - Memory scaling verification (O(threads + scenarios) confirmed)
    - Deterministic reproducibility testing
    - Multi-hydro cascade validation
    - Trajectory data completeness checks
  - **Documentation**:
    - Memory optimization guide: `docs/performance/simulation-memory.md`
    - Benchmarking guide: `docs/development/benchmarks.md`
    - Architecture explanations with before/after diagrams
    - Performance tables and scaling guidelines
  - **User-facing changes**: None - CSV output format unchanged, API backward compatible
  - **Performance characteristics**:
    - Training phase: Uses handlers (unchanged, requires basis warm-starting)
    - Simulation phase: Thread-local handlers + lightweight trajectories
    - Output phase: Trajectory-based sequential access
    - Throughput: Equal or better than previous implementation

- **Scenario Pipeline Validation & Benchmarking (AR-6.7)**:

  - **CRITICAL BUG FIX**: Fixed `ScenarioGenerator::set_noises_by_stage` - was completely non-functional
    - Root cause: `SAA::new_empty()` created empty `branching_samples` vector, but `set_noises_by_stage` assumed stages existed
    - Fix: Added dynamic stage initialization with `while` loop extending vector on-demand
    - Impact: All `ScenarioGenerator` usage was failing before this fix
  - Comprehensive statistical validation test suite (`tests/test_scenario_validation.rs`, 425 lines)
  - **8 statistical tests** (7 passing, 1 ignored for v2 schema):
    - Utility validation: mean, variance, correlation calculations
    - Marginal distributions: Normal N(100,20) with 10,000 samples
    - AR(1) temporal: φ=0.7, validates ACF(1)≈0.7, ACF(2)≈0.49 (52 stages)
    - AR(2) temporal: φ₁=0.6, φ₂=0.2, validates ACF(1)≈0.75, ACF(2)≈0.65 (52 stages)
    - Seed determinism: 100 scenarios, reproducibility verified
    - LogNormal3: Ignored until v2 schema fully integrated
  - **Statistical methodology**:
    - Central Limit Theorem for mean: μ ± 1.96·σ/√n (95% CI)
    - Variance tolerance: 20-30% practical validation
    - Fisher z-transformation for correlation CI
    - Bartlett's formula for ACF standard error
  - **Performance benchmarks** (`benches/scenario_benchmarks.rs`, 261 lines):
    - Independent Normal baseline: 308μs (100×5×12), 6.8ms (1000×10×12), 20.9ms (1000×20×12)
    - AR(1) temporal: 460μs (100×5×12), 9.0ms (1000×10×12)
    - **Full pipeline stress**: 9.0ms (1000×10×12) - **22x faster than 200ms target**
    - SAA allocation overhead: 6.6ms (memory profiling)
  - **Test results**:
    - Mean: 99.54 vs 100.0 (within 99% CI)
    - Variance: 394.09 vs 400.0 (within 30% tolerance)
    - AR(1) ACF(1): validates to 0.7 ± 1.96/√52
    - AR(2) ACF: validates to theoretical values
  - **Backward compatibility**: All examples (01-04) pass with old `uncertainties` format
  - 8 statistical tests + 4 performance benchmarks

- **Scenario Pipeline Integration (AR-6.6)**:

  - Unified `ScenarioGenerator` struct integrating all 4 stages of CEPEL pipeline
  - **Public API**:
    - `from_recourse_input()`: Full pipeline from JSON-based `RecourseInput`
    - `generate_saa()`: Multi-stage SAA generation with branching structure
  - **4-stage pipeline**:
    1. Base Noise: Independent Z ~ N(0,1) (Stage 1)
    2. Correlation: W = L×Z via Cholesky (Stage 2)
    3. Marginal Transform: X = Φ⁻¹(W) via Gaussian copula (Stage 3)
    4. AR Dynamics: Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ (Stage 4)
  - **Features**:
    - Multi-stage generation with per-stage scenario counts
    - Temporal model support: Independent and AR(p)
    - Marginal distributions: Normal and LogNormal3
    - Correlation blocks: Multiple independent groups
    - Seed-controlled determinism
  - **Input validation**:
    - Schema compliance (noise_type, uncertainty_type fields)
    - Initial lag requirements for AR models
    - Entity ID consistency across stages
  - **Performance**: ~9ms for 1000 scenarios × 10 entities × 12 stages (full pipeline)
  - 5 integration tests covering full pipeline combinations
  - Comprehensive documentation with examples

- **AR Temporal Dynamics (AR-6.5)**:

  - New `ar_dynamics` module for Stage 4 of CEPEL scenario generation pipeline (final stage)
  - Applies autoregressive temporal dynamics: Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
  - `ARDynamicsApplicator` struct manages lag buffers and applies AR recursion
  - **Independent model**: Xₜ = εₜ (no temporal correlation)
  - **AR(p) model**: Xₜ = φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ
  - **Lag buffer management**: Automatic shift and update after each scenario
    - Before: [Xₜ₋₁, Xₜ₋₂, ..., Xₜ₋ₚ]
    - After: [Xₜ, Xₜ₋₁, ..., Xₜ₋ₚ₊₁] (discard oldest)
  - **Non-negativity enforcement**: Clamp negative values to 0 for physical quantities
  - **Validation**: AR entities require initial lags, lag count must match lag_order
  - **Performance**: 223.7μs for 1000 scenarios × 10 entities × AR(1) (111x faster than 25ms target)
    - AR(1) 1000×10: 223.7μs
    - AR(2) 1000×10: 233.6μs (more lag computations)
    - Mixed 1000×10: 169.5μs (some independent entities)
    - Independent only 1000×10: 38.1μs (no AR computation)
    - Large scale (5000×50): 5.75ms
  - **Statistical validation**: Sample ACF matches theoretical ACF for AR models
  - **Helper functions**: `theoretical_acf_ar1()`, `sample_acf()` for ACF computation
  - 15 unit tests + 5 benchmarks covering:
    - Independent model (no AR dynamics)
    - AR(1) recurrence relation (φ₁Xₜ₋₁ + εₜ)
    - AR(2) recurrence relation (φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + εₜ)
    - Lag buffer update mechanism (shift and truncate)
    - Mixed entities (AR(1), AR(2), Independent)
    - Non-negativity enforcement (clamp negative to 0)
    - Validation (missing lags, wrong lag count, NaN/infinity, negative lags)
    - Dimension mismatch and empty input panics
    - Theoretical ACF computation (φ₁ᵏ for AR(1))
    - Sample ACF on perfect AR(1) series

- **Marginal Transformation (AR-6.4)**:

  - New `marginal_transformer` module for Stage 3 of CEPEL scenario generation pipeline
  - Transforms correlated N(0,1) samples to target marginal distributions via Gaussian copula
  - `MarginalTransformer` struct applies transformations independently per entity
  - **Normal transformation**: X = μ + σW (linear, preserves Pearson correlation exactly)
  - **LogNormal3 transformation**: X = γ + exp(μ + σW) (nonlinear, preserves Spearman rank correlation)
  - **Overflow protection**: Clamp exponent to [-20, 20] to prevent exp() overflow (max value ≈ 485M)
  - **Validation**: σ > 0 for both distributions, γ ≥ 0 for LogNormal3
  - **Performance**: 37.6μs for 1000 scenarios × 10 entities (398x faster than 15ms target)
    - Normal only: 33.7μs (fastest, linear)
    - LogNormal3 only: 58.8μs (exp() overhead)
    - Large scale (5000×50): 1.08ms
  - **Theory**: Gaussian copula preserves correlation structure:
    - Linear Normal: Pearson correlation preserved exactly
    - Nonlinear LogNormal3: Spearman ρₛ ≈ (6/π)arcsin(ρ/2)
  - 12 unit tests + 4 benchmarks covering:
    - Normal marginal (identity case, mean/variance properties)
    - LogNormal3 marginal (exp() transformation)
    - Mixed Normal/LogNormal3 entities
    - Correlation preservation (exact for Normal, approximate for LogNormal3)
    - Extreme values and overflow handling
    - Validation (σ ≤ 0, γ < 0, empty marginals)
    - Dimension mismatch and empty sample panics

- **Correlation Application (AR-6.3)**:

  - New `correlation_applicator` module for Stage 2 of CEPEL scenario generation pipeline
  - Applies Cholesky decomposition to introduce correlation structure: W = L×Z
  - `CorrelationApplicator` struct orchestrates correlation application across multiple blocks
  - `CorrelationBlock` struct defines entity groups with shared correlation structure
  - `EntityRef` type for referencing uncertainty entities (hydro inflows, loads, etc.)
  - `CholeskyFactor` wrapper in `correlation.rs` for efficient matrix-vector multiply
  - **Performance**: 115μs for 1000 scenarios × 10 entities (173x faster than 20ms target)
  - **Features**:
    - Multiple independent correlation blocks (e.g., separate hydro regions)
    - Entities not in blocks remain uncorrelated
    - Near-singular matrix handling via regularization
    - Preserves N(0,1) marginals while introducing correlation
  - 10 unit tests + 1 benchmark covering:
    - Uncorrelated (identity matrix) case
    - High correlation (ρ=0.999) case
    - Partial correlation (ρ=0.7) validation
    - Multiple blocks with different correlation structures
    - Mixed correlated/independent entities
    - Statistical properties preservation (mean=0, var=1)
    - Near-singular matrix handling
    - Duplicate entity detection
    - Empty blocks (no correlation)

- **Base Noise Generator (AR-6.2)**:

  - New `base_noise` module for Stage 1 of CEPEL scenario generation pipeline
  - `BaseNoiseGenerator` struct generates independent Z ~ N(0,1) samples
  - `BaseNoiseMethod` enum with:
    - `Standard`: Direct random sampling (implemented)
    - `KMeans`, `QuasiMonteCarlo`, `LatinHypercube`: Variance reduction methods (stubbed for future)
  - Deterministic generation via `Xoshiro256Plus` RNG with seed control
  - Performance: <10ms for 1000 scenarios × 10 entities
  - 10 unit tests covering dimensions, statistical properties, determinism, independence, validation

- **Input Schema Refactor (AR-6.1) - BREAKING CHANGE**:
  - **New schema v2** for noise models with explicit separation of concerns:
    - `NoiseModelV2` struct replaces ambiguous v1 format
    - `TemporalModel` enum (`Independent` | `Autoregressive`) for correlation structure
    - `MarginalDistribution` enum (`Normal` | `LogNormal3`) for target distribution
    - `InnovationDistribution` struct for AR white noise (mean, std_dev)
  - **Schema versioning**: `schema_version` field in `Recourse` (defaults to v1)
  - **Backward compatibility**: Legacy formats (v1, `uncertainties`) still supported
  - **Migration path**: `NoiseModelV2::from_legacy()` and `Recourse::normalize_to_v2()`
  - **Validation**: Semantic checks (AR requires innovation_distribution, etc.)
  - **Tests**: 7 unit tests + 1 integration test for v2 format
  - **Purpose**: Enables CEPEL-compliant 4-stage scenario generation pipeline

### Changed

- **Recourse struct** extended with schema v2 fields:
  - `schema_version: Option<u32>` - Version indicator (1=legacy, 2=refactored)
  - `noise_models_v2: Option<Vec<NoiseModelV2>>` - Preferred format for new inputs
  - `noise_models: Option<Vec<NoiseModel>>` - Deprecated but backward compatible

### Deprecated

- `NoiseModel` struct (schema v1) - Use `NoiseModelV2` for new inputs
- `NoiseType` enum - Use `TemporalModel` in schema v2

### Fixed

- Fixed test fixtures in `test_input_validation.rs` to properly test schema v2 fields
- Added missing test data for duplicate detection tests (initial storage, season IDs)
- Fixed AR(1) negative lag test to use zero lag (usize can't be negative)
- Fixed cross-validation tests to include all required seasons in test recourse data

### Migration Guide

Old format (schema v1):

```json
{
  "noise_type": "autoregressive",
  "distribution": { "type": "normal", "mean": 0.0, "std_dev": 15.0 },
  "coefficients": [0.7],
  "non_negativity_method": {
    "type": "lognormal3",
    "gamma": 1.0,
    "mu": 4.5,
    "sigma": 0.3
  }
}
```

New format (schema v2):

```json
{
  "schema_version": 2,
  "marginal_distribution": {
    "type": "lognormal3",
    "gamma": 1.0,
    "mu": 4.5,
    "sigma": 0.3
  },
  "innovation_distribution": { "mean": 0.0, "std_dev": 15.0 },
  "temporal_model": {
    "type": "autoregressive",
    "lag_order": 1,
    "coefficients": [0.7]
  }
}
```

---

# v0.2.1 (Unreleased)

### Added

- **`SddpInstanceBuilder` Pattern**: Enables parameter modification after loading JSON files but before construction

  - `SddpInstanceBuilder::from_paths()` - Load and validate inputs
  - `with_num_iterations(n)` - Modify number of SDDP iterations
  - `with_num_forward_passes(n)` - Modify forward passes per iteration
  - `with_seed(seed)` - Modify random seed for SAA generation
  - `with_num_threads(n)` - Configure thread count for parallel execution
  - `build()` - Construct `SddpInstance` with modified configuration
  - **Use case**: Parameter sweeps for benchmarking and sensitivity analysis
  - **Performance**: Zero-cost abstraction (move semantics, < 1μs construction)
  - **Backward compatible**: `from_files()` still works (uses builder internally)

- **Thread Configuration Parameter** (`num_threads`):
  - New `Config.num_threads: Option<usize>` field for explicit thread control
  - `None` or omitted: Auto-detects available CPU cores using `num_cpus::get()`
  - `Some(n)`: Uses exactly `n` threads (validated > 0 at runtime)
  - Integrated in `SddpInstance::train()` and `simulate()` methods
  - Thread pool configured via `configure_thread_pool()` helper in `utils.rs`
  - Thread count logged during training/simulation for debugging
  - JSON schema updated with validation (minimum: 1, examples: [4, 8, null])
  - All 5 example configs updated with recommended thread counts
  - **Eliminates**: `RAYON_NUM_THREADS` environment variable requirement
  - **Performance**: < 10ms thread pool configuration overhead per train/simulate call
  - **Use cases**: Parameter sweeps, reproducibility, performance tuning
  - **Backward compatible**: Configs without `num_threads` default to auto-detection

### Changed

- `SddpAlgorithm::from_files()` now uses `SddpInstanceBuilder` internally (zero overhead)

### Documentation

- **Performance Tuning Guide** (`docs/guides/PERFORMANCE_TUNING.md`):
  - Comprehensive 914-line guide for optimizing POWE.RS performance
  - **Thread configuration**: Hardware-specific recommendations (94.4%/83.3%/68.9% efficiency at 2/4/8 threads)
  - **Solver tuning**: HiGHS parameters, presolve strategy, tolerance configuration
  - **Memory management**: Scaling formulas, cut pool management, system limits
  - **Problem optimization**: State space reduction, scenario selection, stage aggregation
  - **Benchmarking**: Using Criterion suite for validation
  - **Troubleshooting**: Common issues (poor scaling, slow convergence, high memory, numerical problems)
  - **Quick reference**: Cheat sheets for thread config, memory estimation, benchmarking
  - Cross-references parallel efficiency analysis and memory profiling reports
- Added "Advanced Usage" section to README.md demonstrating parameter sweeps
- Added comprehensive module documentation for `SddpInstanceBuilder` (150+ lines)
- Added 7 unit tests and 7 integration tests for builder pattern

# v0.2.0

- Implements single-node parallelism (thread-based) for both training and simulation steps
- Applies L1 dominance cut selection while locking the future cost function on each thread
- Number of parallel threads is capped either on the logical core count or number of forward passes / simulated scenarios
- Generalizes the underlying data structure for the SDDP algorithm from a vector to a graph
- Creates submodules for scenario generation, risk measure, state, and stochastic process
- Generalizes recourse input for defining inflow and load scenarios, with branchings per node
- Uses trait objects for dynamic risk measure, state and stochastic process definition
- Comprehensive parallel efficiency analysis with production-scale benchmarks (83.3% efficiency at 4 threads)
- Extended memory profiling with production-scale validation: Brazilian hydrothermal (156 reservoirs) uses 3.6 GB; memory scales O(N²) with reservoir count, NOT linearly with stages/iterations

# v0.1.1

- Better handles memory allocation during the SAA generation step in simulation
- Fixes doctests from renaming the crate

# v0.1.0

- Initial release
- Solves a simples hydrothermal dispatch problem with hydro storages as state variables using SDDP
- Inflows are sampled from `LogNormal` distributions, considered the same for all stages
- Loads are constant, given by the user in the input data
- Implements a custom interface to the `HiGHS` solver
- Contains cut selection and basis reuse for improving performance
- Performs simulation by sampling from the same distributions used for training
