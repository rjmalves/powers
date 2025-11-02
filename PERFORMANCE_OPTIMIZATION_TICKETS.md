# Performance Optimization Implementation Tickets

**Generated from**: PERFORMANCE_OPTIMIZATION_REPORT.md  
**Date**: 2025-11-02  
**Total Estimated Duration**: 9 weeks (6 sprints)

---

## Sprint 1: Foundation & Baseline (Week 1-2)

### [PERF-001] Define HydroConstraintData structure

#### Context

Create the core data structure that will hold preprocessed, hydro-specific constraint data. This eliminates the need to iterate through all uncertainty models in the hot path and provides O(1) access to hydro-specific parameters.

This is foundational for all subsequent optimizations and represents a shift from storing generic `UncertaintyModel` objects to storing preprocessed, ready-to-use constraint data.

#### Acceptance Criteria

- [ ] Given a hydro ID and seasonal parameters, when constructing HydroConstraintData, then all seasonal parameters are cached correctly
- [ ] Given AR coefficients, when constructing HydroConstraintData, then transformed coefficients (ψ_i) are pre-computed
- [ ] Given seasonal mean and AR coefficients, when constructing HydroConstraintData, then deterministic_noise_base is pre-computed correctly
- [ ] Structure size is ≤ 200 bytes per hydro (verified with std::mem::size_of)

#### Tasks

##### Implementation

- [ ] Create `HydroConstraintData` struct in `subproblem.rs` with fields:
  - hydro_id: usize
  - ar_constraint_idx: usize
  - season_id: usize
  - seasonal_params: SeasonalParams (copy from UncertaintyModel)
  - ar_coefficients: Vec<f64>
  - transformed_coefficients: Vec<f64>
  - ar_order: usize
  - deterministic_noise_base: f64
- [ ] Implement constructor method that takes UncertaintyModel and season_id
- [ ] Implement transformation logic to compute ψ_i from AR coefficients
- [ ] Implement deterministic base computation: μ_t - Σ[ψ_i * μ_{t-i}]
- [ ] Add Debug and Clone implementations

##### Testing

- [ ] Unit test: Construct HydroConstraintData from Independent model (AR order 0)
- [ ] Unit test: Construct HydroConstraintData from AR(1) model
- [ ] Unit test: Construct HydroConstraintData from AR(3) model
- [ ] Unit test: Verify transformed_coefficients match manual PAR transformation
- [ ] Unit test: Verify deterministic_noise_base is correct for known parameters
- [ ] Unit test: Validate struct memory size is within target

##### Documentation

- [ ] Add comprehensive doc comments explaining each field's purpose
- [ ] Document the mathematical formulation for deterministic_noise_base
- [ ] Document the PAR to standard AR transformation for ψ_i
- [ ] Add code example showing construction from UncertaintyModel

#### Technical Notes

**Transformation formulas:**
- For PAR(p) model: y_t = μ_t + Σ[φ_i(y_{t-i} - μ_{t-i})] + σ_t·ε_t
- Transformed: y_t = [μ_t - Σ(φ_i·μ_{t-i})] + Σ[φ_i·y_{t-i}] + σ_t·ε_t
- Therefore: ψ_i = φ_i and deterministic_noise_base = μ_t - Σ[φ_i·μ_{t-i}]

**Edge cases:**
- Independent models: ar_order = 0, empty vectors for coefficients
- Seasonal mean may vary by season (already captured in seasonal_params)

#### Dependencies

- Blocked by: None (foundational work)
- Blocks: PERF-002, PERF-003
- Related: None

#### Estimated Effort

2 story points (confidence: high)  
~1-1.5 days

---

### [PERF-002] Refactor Subproblem to use HydroConstraintData

#### Context

Replace the redundant `uncertainty_models: Vec<UncertaintyModel>` field in Subproblem with a preprocessed `hydro_data: Vec<HydroConstraintData>` vector. This is the critical architectural change that enables all subsequent hot path optimizations.

This refactoring should maintain identical behavior while setting up for performance improvements.

#### Acceptance Criteria

- [ ] Given a Subproblem constructor, when initialized with uncertainty_models, then hydro_data is correctly populated
- [ ] Given hydro_data vector, when accessing by index, then it's sorted by hydro_id for cache-friendly access
- [ ] All existing tests pass without modification
- [ ] Memory usage per Subproblem is reduced by 20-30% (measured with profiler)

#### Tasks

##### Implementation

- [ ] Modify `Subproblem` struct: add `pub hydro_data: Vec<HydroConstraintData>` field
- [ ] Update `Subproblem::new_from_uncertainty_models` to build hydro_data vector
- [ ] Filter uncertainty_models to only inflow types during construction
- [ ] Sort hydro_data by hydro_id for sequential cache-friendly access
- [ ] Validate that ar_constraint_idx is correctly set for each hydro
- [ ] Keep uncertainty_models field temporarily (mark as deprecated) for backward compatibility
- [ ] Add assertion that hydro_data.len() matches expected hydro count

##### Testing

- [ ] Unit test: Build Subproblem with 10 hydro Independent models
- [ ] Unit test: Build Subproblem with 50 hydros mixed AR(1), AR(2), AR(3)
- [ ] Unit test: Verify hydro_data is sorted by hydro_id
- [ ] Unit test: Validate ar_constraint_idx points to correct constraint for each hydro
- [ ] Integration test: Full Subproblem construction matches old behavior
- [ ] Memory test: Profile memory usage before/after with 100-hydro system

##### Documentation

- [ ] Update Subproblem struct doc comments
- [ ] Add migration note about deprecated uncertainty_models field
- [ ] Document the preprocessing that happens during construction
- [ ] Update module-level docs in subproblem.rs

#### Technical Notes

**Construction logic:**
```rust
let mut hydro_data = Vec::new();
for (idx, model) in uncertainty_models.iter().enumerate() {
    if model.entity_type() == UncertaintyType::Inflow {
        let hydro_id = extract_hydro_id(model);
        let ar_constraint_idx = constraints.ar_dynamics[hydro_id];
        let data = HydroConstraintData::new(model, season_id, hydro_id, ar_constraint_idx)?;
        hydro_data.push(data);
    }
}
hydro_data.sort_by_key(|h| h.hydro_id);
```

**Memory calculation:**
- Old: Vec<UncertaintyModel> ≈ 500 bytes × n_models
- New: Vec<HydroConstraintData> ≈ 200 bytes × n_hydros
- Savings: ~40-50% for typical systems

#### Dependencies

- Blocked by: PERF-001
- Blocks: PERF-003, PERF-004
- Related: None

#### Estimated Effort

3 story points (confidence: high)  
~2-2.5 days

---

### [PERF-003] Add baseline performance benchmarks

#### Context

Establish performance baselines before implementing hot path optimizations. These benchmarks will be used to validate the 2-5x speedup targets and ensure no regressions occur.

Benchmarks should cover both micro-level operations (realize_uncertainties) and macro-level operations (full SDDP iteration).

#### Acceptance Criteria

- [ ] Benchmarks run successfully on systems with 10, 50, 100 hydros
- [ ] Benchmarks measure both time and memory allocations
- [ ] Results are reproducible within 5% variance
- [ ] Baseline results are documented for comparison

#### Tasks

##### Implementation

- [ ] Create `benches/realize_uncertainties.rs` benchmark file
- [ ] Implement benchmark for realize_uncertainties with 10 hydros
- [ ] Implement benchmark for realize_uncertainties with 50 hydros
- [ ] Implement benchmark for realize_uncertainties with 100 hydros
- [ ] Add benchmark for generate_precomputed_scenarios (to be eliminated)
- [ ] Add benchmark for update_observation_space_ar_constraints
- [ ] Create helper function to set up test subproblems with AR(2) models
- [ ] Add memory allocation tracking using criterion or custom profiler

##### Testing

- [ ] Verify benchmarks compile and run
- [ ] Run benchmarks 10 times and verify variance < 5%
- [ ] Test on development machine and document specs
- [ ] Validate results align with profiling from report (~60-80μs for 50 hydros)

##### Documentation

- [ ] Create BENCHMARK_RESULTS.md with baseline numbers
- [ ] Document test system specs (CPU, RAM, OS)
- [ ] Add instructions for running benchmarks to README
- [ ] Document expected performance ranges for different system sizes

#### Technical Notes

**Benchmark structure:**
```rust
fn bench_realize_uncertainties_50_hydros(c: &mut Criterion) {
    let mut subproblem = setup_50_hydro_ar2_system();
    let noises = create_test_noises(50);
    let mut realization = Realization::default();
    
    c.bench_function("realize_uncertainties_50_hydros", |b| {
        b.iter(|| {
            subproblem.realize_uncertainties(&noises, &mut realization)
        });
    });
}
```

**Key metrics to track:**
- Time per call (mean, median, p95, p99)
- Allocations per call
- Cache miss rate (if using perf)

#### Dependencies

- Blocked by: PERF-002 (needs new structure in place)
- Blocks: PERF-004 (validation baseline)
- Related: None

#### Estimated Effort

2 story points (confidence: high)  
~1-1.5 days

---

## Sprint 2: Hot Path Optimization (Week 3-4)

### [PERF-004] Optimize realize_uncertainties to use hydro_data directly

#### Context

This is the highest-impact optimization. Replace the current two-step process (generate_precomputed_scenarios + update_observation_space_ar_constraints) with a direct constraint update loop using preprocessed hydro_data.

Expected speedup: 2-3x for the realize_uncertainties function, translating to 40-60% faster SDDP forward passes.

#### Acceptance Criteria

- [ ] Given 50-hydro system, when calling realize_uncertainties, then execution time is ≤40μs (down from 120-150μs)
- [ ] Given any hydro system, when running optimized code, then numerical results match original implementation within 1e-10
- [ ] No Vec<PrecomputedInflowScenario> allocations occur in hot path
- [ ] Benchmarks show 2-3x speedup vs baseline

#### Tasks

##### Implementation

- [ ] Create new `realize_uncertainties_optimized` method in Subproblem
- [ ] Remove call to `generate_precomputed_scenarios`
- [ ] Implement direct loop over hydro_data:
  - Extract innovation from noises
  - Compute stochastic term: seasonal_params.std_dev × innovation
  - Compute RHS: deterministic_noise_base + stochastic_term
  - Add lag contribution using dot product if ar_order > 0
  - Call model.change_rows_bounds directly
- [ ] Inline lag contribution computation (avoid extra function call)
- [ ] Add `#[inline]` hints for hot functions
- [ ] Replace old realize_uncertainties with optimized version
- [ ] Remove unused generate_precomputed_scenarios and PrecomputedInflowScenario

##### Testing

- [ ] Unit test: 10-hydro Independent model system matches old behavior
- [ ] Unit test: 50-hydro AR(2) system matches old behavior (numerical tolerance 1e-10)
- [ ] Unit test: 100-hydro mixed AR orders matches old behavior
- [ ] Integration test: Full SDDP forward pass produces identical results
- [ ] Regression test: Compare 5 random seeds with old implementation
- [ ] Performance test: Verify 2-3x speedup in realize_uncertainties benchmark
- [ ] Memory test: Verify no Vec allocations during realize_uncertainties

##### Documentation

- [ ] Update realize_uncertainties doc comments with optimization notes
- [ ] Document the mathematical equivalence to old implementation
- [ ] Add performance notes to PERFORMANCE_OPTIMIZATION_REPORT.md
- [ ] Update CHANGELOG.md with performance improvement note

#### Technical Notes

**Optimized hot path pseudocode:**
```rust
for hydro_data in &self.hydro_data {
    let innovation = innovations[hydro_data.hydro_id];
    let stochastic = hydro_data.seasonal_params.std_dev * innovation;
    let mut rhs = hydro_data.deterministic_noise_base + stochastic;
    
    if hydro_data.ar_order > 0 {
        let lags = self.inflow_manager.get_lag_observations(
            hydro_data.hydro_id, 
            hydro_data.ar_order
        );
        rhs += dot_product(&hydro_data.transformed_coefficients, lags);
    }
    
    model.change_rows_bounds(hydro_data.ar_constraint_idx, rhs, rhs);
}
```

**Critical points:**
- No allocations in loop body
- Sequential access to hydro_data (cache-friendly)
- All parameters pre-computed
- Minimal branching

#### Dependencies

- Blocked by: PERF-002, PERF-003
- Blocks: PERF-007 (validation)
- Related: PERF-005 (additional speedup)

#### Estimated Effort

5 story points (confidence: medium)  
~3 days

---

### [PERF-005] Add SIMD-optimized dot product utilities

#### Context

The lag contribution computation (dot product of transformed_coefficients and lag observations) is called thousands of times per SDDP iteration. Optimizing this with SIMD can provide an additional 4-5x speedup for this specific operation.

This is a lower-risk optimization that complements PERF-004.

#### Acceptance Criteria

- [ ] Given two f64 slices, when calling dot_product_simd, then result matches standard implementation within 1e-12
- [ ] Benchmark shows 4-5x speedup for 3-10 element vectors
- [ ] Works correctly on both x86_64 and ARM64 architectures
- [ ] Gracefully falls back to scalar code if SIMD unavailable

#### Tasks

##### Implementation

- [ ] Create `src/utils/simd.rs` module
- [ ] Implement `dot_product_simd` using unsafe unchecked indexing for LLVM vectorization
- [ ] Implement `dot_product_kahan_simd` for numerically stable version
- [ ] Add feature flag `simd-optimizations` in Cargo.toml
- [ ] Use conditional compilation for SIMD vs scalar fallback
- [ ] Replace dot_product calls in realize_uncertainties with SIMD version
- [ ] Add architecture detection and appropriate SIMD intrinsics

##### Testing

- [ ] Unit test: Dot product with known vectors (e.g., [1,2,3] · [4,5,6] = 32)
- [ ] Unit test: Zero-length vectors (edge case)
- [ ] Unit test: Single-element vectors
- [ ] Unit test: Large vectors (100 elements) for numerical stability
- [ ] Property test: SIMD and scalar versions match for random inputs
- [ ] Benchmark: Compare SIMD vs scalar for 3, 5, 10, 100 element vectors
- [ ] Cross-platform test: Verify on x86_64 and ARM64 (if available)

##### Documentation

- [ ] Add module-level docs explaining SIMD optimization
- [ ] Document when to use dot_product_simd vs dot_product_kahan_simd
- [ ] Add note about numerical precision tradeoffs
- [ ] Update README with SIMD feature flag
- [ ] Document expected speedup ranges

#### Technical Notes

**SIMD implementation approach:**
```rust
#[inline]
pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    
    // LLVM will auto-vectorize this with target-cpu=native
    for i in 0..a.len() {
        sum += unsafe { 
            a.get_unchecked(i) * b.get_unchecked(i) 
        };
    }
    sum
}
```

**Alternative:** Use `std::simd` (nightly) or `packed_simd` crate for explicit SIMD control.

**Numerical considerations:**
- Standard summation may have floating-point drift
- Kahan summation preserves precision at ~10% performance cost
- For AR(1)-AR(3) models, drift is negligible

#### Dependencies

- Blocked by: None (can be done in parallel with PERF-004)
- Blocks: None
- Related: PERF-004 (used together for maximum speedup)

#### Estimated Effort

3 story points (confidence: medium)  
~2 days

---

### [PERF-006] Remove deprecated code and cleanup

#### Context

After proving the optimized implementation works correctly, remove the old generate_precomputed_scenarios, PrecomputedInflowScenario, and uncertainty_models field. This reduces code complexity and maintenance burden.

#### Acceptance Criteria

- [ ] No references to generate_precomputed_scenarios remain
- [ ] PrecomputedInflowScenario struct is removed
- [ ] uncertainty_models field is removed from Subproblem
- [ ] All tests still pass
- [ ] Code coverage remains ≥90%

#### Tasks

##### Implementation

- [ ] Remove generate_precomputed_scenarios function
- [ ] Remove PrecomputedInflowScenario struct and related code
- [ ] Remove uncertainty_models field from Subproblem struct
- [ ] Update all Subproblem constructor signatures
- [ ] Remove now-unused imports
- [ ] Run cargo clippy and fix any warnings
- [ ] Run cargo fmt

##### Testing

- [ ] Verify all unit tests pass
- [ ] Verify all integration tests pass
- [ ] Verify all benchmarks still compile and run
- [ ] Check test coverage hasn't decreased

##### Documentation

- [ ] Update CHANGELOG.md noting removed internal APIs
- [ ] Add migration note if any public APIs changed
- [ ] Update architecture documentation if relevant

#### Technical Notes

**Search for all references:**
```bash
git grep "generate_precomputed_scenarios"
git grep "PrecomputedInflowScenario"
git grep "uncertainty_models"
```

This is primarily a cleanup task with low risk.

#### Dependencies

- Blocked by: PERF-004, PERF-007
- Blocks: None
- Related: PERF-002

#### Estimated Effort

1 story point (confidence: high)  
~0.5-1 day

---

## Sprint 3: Lag Buffer Optimization (Week 5)

### [PERF-007] Implement OptimizedLagBuffer with flattened storage

#### Context

Replace the current Vec<Vec<f64>> lag buffer with a flattened Vec<f64> with offset-based indexing. This reduces allocation overhead, improves cache locality, and reduces memory footprint by ~40%.

This optimization targets the inflow_manager which is accessed on every realize_uncertainties call.

#### Acceptance Criteria

- [ ] Given 100 hydros with AR(2), when using OptimizedLagBuffer, then memory usage is ≤2,500 bytes (vs 4,000 current)
- [ ] Lag access time is 3-4x faster than current implementation
- [ ] All lag buffer operations maintain correctness
- [ ] Integration with Subproblem works seamlessly

#### Tasks

##### Implementation

- [ ] Create OptimizedLagBuffer struct in `src/inflow_constraints.rs`
- [ ] Implement constructor that takes Vec<usize> of per-hydro lag counts
- [ ] Implement `get_lags(&self, hydro_id) -> &[f64]` method
- [ ] Implement `get_lags_mut(&mut self, hydro_id) -> &mut [f64]` method
- [ ] Implement `update_from_observations(&mut self, observations: &[f64])` method
- [ ] Add validation in constructor to ensure offsets are correct
- [ ] Replace Vec<Vec<f64>> in ObservationSpaceConstraintManager
- [ ] Update all lag buffer access sites to use new API

##### Testing

- [ ] Unit test: Construct buffer for [2, 3, 1, 2] lag counts
- [ ] Unit test: Verify offsets calculation is correct
- [ ] Unit test: get_lags returns correct slice for each hydro
- [ ] Unit test: update_from_observations correctly shifts and updates lags
- [ ] Unit test: Edge case with zero lag count for some hydros
- [ ] Integration test: Full Subproblem with optimized lag buffer matches old behavior
- [ ] Memory test: Verify memory usage reduction (std::mem::size_of or profiler)
- [ ] Performance test: Benchmark lag access speed

##### Documentation

- [ ] Add comprehensive doc comments for OptimizedLagBuffer
- [ ] Document the offset-based indexing scheme
- [ ] Add diagram showing memory layout
- [ ] Update module-level docs in inflow_constraints.rs

#### Technical Notes

**Memory layout:**
```
Hydro lags: [2, 3, 1]
Offsets:    [0, 2, 5, 6]
Data:       [h0_lag0, h0_lag1, h1_lag0, h1_lag1, h1_lag2, h2_lag0]
```

**Offset construction:**
```rust
let mut offsets = vec![0];
let mut cumulative = 0;
for &count in lag_counts.iter() {
    cumulative += count;
    offsets.push(cumulative);
}
```

**Update logic for rotate_right:**
```rust
// Shift lags: [old0, old1, old2] -> [new, old0, old1]
lags.rotate_right(1);
lags[0] = new_observation;
```

#### Dependencies

- Blocked by: None
- Blocks: PERF-008
- Related: PERF-004 (benefits from faster lag access)

#### Estimated Effort

3 story points (confidence: high)  
~2 days

---

### [PERF-008] Integrate OptimizedLagBuffer into realize_uncertainties

#### Context

Update the optimized realize_uncertainties implementation to use the new OptimizedLagBuffer. This should provide an additional 10-15% speedup due to improved cache locality during lag access.

#### Acceptance Criteria

- [ ] realize_uncertainties uses OptimizedLagBuffer for lag observations
- [ ] Benchmarks show additional 10-15% speedup over PERF-004
- [ ] All numerical tests still pass
- [ ] No performance regression in any operation

#### Tasks

##### Implementation

- [ ] Update inflow_manager to use OptimizedLagBuffer
- [ ] Update realize_uncertainties to call new lag buffer API
- [ ] Verify get_lag_observations signature is compatible
- [ ] Profile to ensure cache locality is improved

##### Testing

- [ ] Integration test: Full SDDP pass with optimized buffer
- [ ] Numerical regression test: Compare 10 random seeds
- [ ] Performance test: Re-run realize_uncertainties benchmark
- [ ] Verify total speedup vs baseline is 2.5-3.5x

##### Documentation

- [ ] Update performance notes in PERFORMANCE_OPTIMIZATION_REPORT.md
- [ ] Update CHANGELOG.md

#### Technical Notes

This should be straightforward integration since the API was designed to be drop-in compatible.

#### Dependencies

- Blocked by: PERF-007
- Blocks: None
- Related: PERF-004

#### Estimated Effort

1 story point (confidence: high)  
~0.5-1 day

---

### [PERF-009] Memory profiling and validation

#### Context

Validate that all memory optimization targets have been achieved. Use profiling tools to measure actual memory usage and identify any remaining allocation hotspots.

#### Acceptance Criteria

- [ ] Memory usage per Subproblem is reduced by 30-40% vs baseline
- [ ] No unexpected allocations in realize_uncertainties hot path
- [ ] Heap profiling shows expected memory layout
- [ ] Document actual memory savings

#### Tasks

##### Implementation

- [ ] Add memory profiling benchmark using criterion with allocation tracking
- [ ] Use valgrind/massif or heaptrack for detailed profiling
- [ ] Compare memory usage: baseline vs all optimizations
- [ ] Identify any remaining allocation hotspots

##### Testing

- [ ] Profile 10-hydro, 50-hydro, 100-hydro systems
- [ ] Measure peak memory and steady-state memory
- [ ] Verify no memory leaks during long SDDP runs
- [ ] Document memory layout with diagrams

##### Documentation

- [ ] Create MEMORY_PROFILE.md with before/after comparison
- [ ] Add graphs showing memory over time
- [ ] Document profiling methodology
- [ ] Update PERFORMANCE_OPTIMIZATION_REPORT.md with actual results

#### Technical Notes

**Tools to use:**
- Valgrind with massif for heap profiling
- heaptrack for allocation tracking
- criterion with custom allocator for microbenchmarks
- /proc/self/statm for Linux peak memory

#### Dependencies

- Blocked by: PERF-007, PERF-008
- Blocks: None
- Related: PERF-003 (validation against baseline)

#### Estimated Effort

2 story points (confidence: medium)  
~1-1.5 days

---

## Sprint 4: State Optimization (Week 6)

### [PERF-010] Refactor trajectory filtering in SDDP forward pass

#### Context

Currently, `update_from_trajectory` filters the trajectory on every call. Move this filtering to the SDDP forward pass level so it's done once per stage instead of once per subproblem update.

Expected speedup: 10-15% in state update operations.

#### Acceptance Criteria

- [ ] Trajectory is filtered once per forward pass stage
- [ ] update_from_trajectory receives pre-filtered trajectory
- [ ] All SDDP tests pass with identical results
- [ ] State update is 10-15% faster

#### Tasks

##### Implementation

- [ ] Locate trajectory filtering in state.rs update_from_trajectory
- [ ] Move filtering logic to SDDP forward pass (sddp/mod.rs or similar)
- [ ] Update update_from_trajectory signature to accept filtered trajectory
- [ ] Pass filtered trajectory through the forward pass call chain
- [ ] Remove redundant filtering from update_from_trajectory

##### Testing

- [ ] Unit test: Manual trajectory filtering produces expected results
- [ ] Integration test: Full forward pass with pre-filtering matches old behavior
- [ ] Regression test: 10 random SDDP runs produce identical policies
- [ ] Performance test: Benchmark state update time before/after

##### Documentation

- [ ] Update update_from_trajectory doc comments
- [ ] Document filtering semantics (PreStudy with non-zero inflows)
- [ ] Update CHANGELOG.md

#### Technical Notes

**Filtering logic:**
```rust
let filtered_trajectory: Vec<&Realization> = trajectory
    .iter()
    .filter(|r| {
        r.kind != StudyPeriodKind::PreStudy || 
        r.inflow_residual.iter().any(|&v| v.abs() > 1e-10)
    })
    .copied()
    .collect();
```

This filtering preserves PreStudy realizations that have non-zero inflows (which contribute to AR lags).

#### Dependencies

- Blocked by: None
- Blocks: None
- Related: PERF-011

#### Estimated Effort

2 story points (confidence: medium)  
~1-1.5 days

---

### [PERF-011] Vectorize lag buffer update in state.rs

#### Context

The nested loop in `update_from_trajectory` that updates lagged_inflows can be vectorized for better performance. This is a micro-optimization but contributes to overall state update speed.

#### Acceptance Criteria

- [ ] Lag buffer update uses vectorized operations where possible
- [ ] Cache-unfriendly access pattern is eliminated
- [ ] State update is an additional 5-10% faster
- [ ] Numerical results unchanged

#### Tasks

##### Implementation

- [ ] Analyze current nested loop access pattern
- [ ] Restructure to iterate trajectory once and update all hydros
- [ ] Consider using iterator adapters for better optimization
- [ ] Ensure memory access is sequential where possible

##### Testing

- [ ] Unit test: Lag buffer update with complex trajectory
- [ ] Performance test: Benchmark state update
- [ ] Verify vectorization with cargo asm or llvm-mca

##### Documentation

- [ ] Add comments explaining vectorization strategy
- [ ] Update performance notes

#### Technical Notes

Current pattern is cache-unfriendly:
```rust
for hydro in 0..n_hydros {
    for lag_idx in 0..lags {
        // Accesses filtered_trajectory[hist_idx].inflow_residual[hydro]
        // Jumps between trajectory entries
    }
}
```

Better pattern would process trajectory linearly.

#### Dependencies

- Blocked by: PERF-010
- Blocks: None
- Related: None

#### Estimated Effort

2 story points (confidence: low)  
~1.5-2 days

---

### [PERF-012] End-to-end SDDP performance validation

#### Context

Validate that all optimizations combine to achieve the target 40-50% speedup in forward pass and backward pass. Run comprehensive benchmarks on realistic problem sizes.

#### Acceptance Criteria

- [ ] Forward pass is 50-60% faster than baseline
- [ ] Backward pass is 40-50% faster than baseline
- [ ] Full SDDP convergence time is reduced by 45-55%
- [ ] All numerical results match baseline within 1e-8 tolerance

#### Tasks

##### Implementation

- [ ] Create end-to-end SDDP benchmarks for 10, 50, 100, 200 hydro systems
- [ ] Run benchmarks on baseline and optimized versions
- [ ] Profile critical path and verify no unexpected bottlenecks remain
- [ ] Measure convergence properties (iterations to convergence)

##### Testing

- [ ] Benchmark: 10 hydros, 12 stages, 100 iterations
- [ ] Benchmark: 50 hydros, 24 stages, 100 iterations
- [ ] Benchmark: 100 hydros, 24 stages, 50 iterations
- [ ] Benchmark: 200 hydros, 12 stages, 20 iterations (scalability test)
- [ ] Numerical validation: Compare objective values, cut coefficients, policies

##### Documentation

- [ ] Update PERFORMANCE_OPTIMIZATION_REPORT.md with actual results
- [ ] Create comparison tables and graphs
- [ ] Document any deviations from projected speedups
- [ ] Add case studies with real-world problem instances

#### Technical Notes

**Validation metrics:**
- Wall-clock time per iteration
- Time per forward pass
- Time per backward pass
- Memory usage (peak and steady-state)
- Convergence rate (iterations to target gap)
- Numerical differences (objective, cuts, policies)

#### Dependencies

- Blocked by: PERF-004, PERF-008, PERF-010, PERF-011
- Blocks: None
- Related: PERF-003 (baseline), PERF-009 (memory)

#### Estimated Effort

3 story points (confidence: high)  
~2 days

---

## Sprint 5: SIMD & Advanced Optimizations (Week 7-8)

### [PERF-013] Add compile-time SIMD feature flags

#### Context

Make SIMD optimizations optional through feature flags. This allows users to opt-in for maximum performance while maintaining portable fallback implementations.

#### Acceptance Criteria

- [ ] Feature flag `simd-optimizations` controls SIMD usage
- [ ] Default build uses portable scalar code
- [ ] Feature-enabled build uses SIMD where available
- [ ] Both configurations pass all tests

#### Tasks

##### Implementation

- [ ] Add `simd-optimizations` feature to Cargo.toml
- [ ] Use `#[cfg(feature = "simd-optimizations")]` for SIMD code paths
- [ ] Implement scalar fallbacks for all SIMD functions
- [ ] Document feature flag in Cargo.toml

##### Testing

- [ ] Test with feature disabled (default)
- [ ] Test with feature enabled
- [ ] Verify performance difference is as expected
- [ ] Cross-platform testing (x86_64, ARM64)

##### Documentation

- [ ] Add feature flag documentation to README
- [ ] Document performance implications
- [ ] Add build instructions for optimized builds

#### Technical Notes

```toml
[features]
default = []
simd-optimizations = []
```

```rust
#[cfg(feature = "simd-optimizations")]
use simd_impl;

#[cfg(not(feature = "simd-optimizations"))]
use scalar_impl;
```

#### Dependencies

- Blocked by: PERF-005
- Blocks: PERF-014
- Related: None

#### Estimated Effort

1 story point (confidence: high)  
~0.5-1 day

---

### [PERF-014] Multi-architecture SIMD optimization

#### Context

Optimize SIMD code for both x86_64 (AVX2) and ARM64 (NEON). Use conditional compilation and runtime feature detection where appropriate.

#### Acceptance Criteria

- [ ] SIMD code works on x86_64 with AVX2
- [ ] SIMD code works on ARM64 with NEON
- [ ] Runtime falls back gracefully if CPU features unavailable
- [ ] Benchmarks show speedup on both architectures

#### Tasks

##### Implementation

- [ ] Add x86_64 AVX2 intrinsics for dot product
- [ ] Add ARM64 NEON intrinsics for dot product
- [ ] Use std::is_x86_feature_detected! for runtime detection
- [ ] Implement portable fallback
- [ ] Test on both architectures

##### Testing

- [ ] Test on x86_64 with AVX2
- [ ] Test on x86_64 without AVX2 (fallback)
- [ ] Test on ARM64 with NEON (if available)
- [ ] Cross-compile tests for ARM64
- [ ] Benchmark on both architectures

##### Documentation

- [ ] Document supported SIMD instruction sets
- [ ] Add architecture-specific performance notes
- [ ] Document runtime detection mechanism

#### Technical Notes

This may require access to ARM64 hardware or CI runners for complete validation.

#### Dependencies

- Blocked by: PERF-013
- Blocks: None
- Related: PERF-005

#### Estimated Effort

4 story points (confidence: low)  
~2.5-3 days

---

### [PERF-015] Investigate parallel constraint updates

#### Context

Explore parallelizing constraint updates in realize_uncertainties using Rayon. This is exploratory work (spike) to determine if thread-safe model updates can be achieved and whether the overhead is worth it.

This is optional/stretch work depending on prior results.

#### Acceptance Criteria

- [ ] Spike determines feasibility of parallel constraint updates
- [ ] If feasible, prototype shows speedup potential
- [ ] If not feasible, document blockers and alternative approaches

#### Tasks

##### Implementation (Spike - time-boxed to 2 days)

- [ ] Research HiGHS solver thread safety for change_rows_bounds
- [ ] Prototype parallel updates using Rayon
- [ ] Benchmark parallel vs sequential for 100+ hydro systems
- [ ] Identify synchronization overhead
- [ ] Determine if batch updates + single-threaded apply is better

##### Testing

- [ ] Benchmark on large systems (100, 200, 500 hydros)
- [ ] Measure synchronization overhead
- [ ] Compare to single-threaded optimized version

##### Documentation

- [ ] Document spike findings
- [ ] If pursuing: Create follow-up ticket for full implementation
- [ ] If not: Document why and what alternatives exist

#### Technical Notes

**Challenge:** Most LP solver libraries are not thread-safe for model mutation.

**Alternatives:**
- Batch constraint updates and apply atomically
- Use lock-free queue for constraint updates
- Parallelize at SDDP iteration level instead

#### Dependencies

- Blocked by: PERF-012 (understand current bottlenecks first)
- Blocks: None
- Related: None

#### Estimated Effort

3 story points (confidence: low)  
~2 days (time-boxed spike)

---

## Sprint 6: Documentation & Validation (Week 9)

### [PERF-016] Comprehensive regression testing

#### Context

Run extensive regression tests comparing optimized implementation against baseline (or preserved reference implementation). Ensure no numerical drift or convergence issues introduced.

#### Acceptance Criteria

- [ ] All existing test suite passes
- [ ] 20+ real-world case studies produce identical results (within tolerance)
- [ ] Convergence properties are preserved (same iteration count to converge)
- [ ] No numerical stability issues detected

#### Tasks

##### Implementation

- [ ] Create comprehensive regression test suite
- [ ] Run tests on diverse problem types:
  - Small systems (10 hydros)
  - Medium systems (50-100 hydros)
  - Large systems (200+ hydros)
  - Mixed AR orders
  - Independent vs PAR models
  - Different seasonality patterns
- [ ] Compare objective values, cut coefficients, policies
- [ ] Test with different random seeds

##### Testing

- [ ] Run 20 case studies with baseline and optimized versions
- [ ] Compare results with tolerance: objective ε=1e-8, cuts ε=1e-10
- [ ] Test long SDDP runs (500+ iterations) for numerical drift
- [ ] Validate convergence rate is unchanged

##### Documentation

- [ ] Create REGRESSION_TEST_RESULTS.md
- [ ] Document test methodology
- [ ] Include statistical analysis of differences
- [ ] Add confidence statement for production use

#### Technical Notes

**Comparison methodology:**
- Objective value: absolute and relative difference
- Cut coefficients: Euclidean distance in cut space
- Policies: Statistical distance metrics
- Convergence: number of iterations to 1% gap

#### Dependencies

- Blocked by: All PERF-* implementation tickets
- Blocks: PERF-018 (production readiness)
- Related: PERF-003 (baseline), PERF-012 (validation)

#### Estimated Effort

3 story points (confidence: high)  
~2 days

---

### [PERF-017] Update documentation and examples

#### Context

Update all documentation to reflect performance improvements, new internal architecture, and any API changes. Provide guidance for users upgrading from previous versions.

#### Acceptance Criteria

- [ ] README.md includes performance improvements section
- [ ] PERFORMANCE_OPTIMIZATION_REPORT.md updated with actual results
- [ ] Architecture documentation reflects new HydroConstraintData design
- [ ] Migration guide created for any breaking changes
- [ ] Examples run successfully with optimized code

#### Tasks

##### Implementation

- [ ] Update README.md:
  - Add "Performance" section
  - Document SIMD feature flag
  - Add benchmark results
- [ ] Update PERFORMANCE_OPTIMIZATION_REPORT.md:
  - Replace projections with actual results
  - Add graphs and tables
  - Document lessons learned
- [ ] Create MIGRATION_GUIDE.md if needed:
  - Document any API changes
  - Provide upgrade instructions
  - Note any behavior changes
- [ ] Update doc comments in:
  - subproblem.rs
  - inflow_constraints.rs
  - state.rs
- [ ] Update CHANGELOG.md with comprehensive notes
- [ ] Verify all examples compile and run
- [ ] Add performance comparison example

##### Testing

- [ ] Verify all examples compile
- [ ] Run examples and verify output
- [ ] Check doc links are valid (cargo doc)
- [ ] Spell check documentation

##### Documentation

- [ ] Create performance comparison graphs
- [ ] Add before/after memory diagrams
- [ ] Document optimization techniques used
- [ ] Add troubleshooting section for performance issues

#### Technical Notes

**Key sections to update:**
- README: Performance section with concrete numbers
- Inline docs: Implementation notes for future maintainers
- Examples: Consider adding a performance-tuning example
- CHANGELOG: Clear communication of improvements and changes

#### Dependencies

- Blocked by: PERF-016, PERF-012 (need final results)
- Blocks: None
- Related: All tickets (comprehensive documentation)

#### Estimated Effort

3 story points (confidence: high)  
~2 days

---

### [PERF-018] Production readiness checklist and release

#### Context

Final validation that the optimized code is production-ready. Create a comprehensive checklist and prepare for release.

#### Acceptance Criteria

- [ ] All tests pass (unit, integration, regression)
- [ ] All benchmarks show expected performance improvements
- [ ] Documentation is complete and accurate
- [ ] No outstanding bugs or issues
- [ ] Release notes are prepared

#### Tasks

##### Implementation

- [ ] Run full test suite on multiple platforms
- [ ] Run all benchmarks and verify targets met
- [ ] Review all documentation
- [ ] Run cargo clippy --all-targets
- [ ] Run cargo audit for security issues
- [ ] Verify examples work
- [ ] Check code coverage (aim for ≥90%)

##### Testing

- [ ] Full CI pipeline passes
- [ ] Manual testing on representative systems
- [ ] Load testing on large systems (500+ hydros)
- [ ] Memory leak testing with long runs

##### Documentation

- [ ] Prepare release notes
- [ ] Update version numbers
- [ ] Tag release in git
- [ ] Prepare announcement for users

#### Technical Notes

**Production readiness criteria:**
- ✅ All tests pass
- ✅ Performance targets met
- ✅ No memory leaks
- ✅ Documentation complete
- ✅ API stable
- ✅ Backward compatible (or migration guide provided)

**Release checklist:**
- [ ] Version bump (semantic versioning)
- [ ] CHANGELOG.md updated
- [ ] Git tag created
- [ ] Release notes published
- [ ] Documentation deployed

#### Dependencies

- Blocked by: PERF-016, PERF-017
- Blocks: None
- Related: All tickets (final validation)

#### Estimated Effort

2 story points (confidence: high)  
~1 day

---

## Summary Statistics

### Total Effort Estimation

- **Sprint 1 (Foundation)**: 7 story points (~4-5 days)
- **Sprint 2 (Hot Path)**: 9 story points (~5-6 days)
- **Sprint 3 (Lag Buffer)**: 6 story points (~3.5-4.5 days)
- **Sprint 4 (State Optimization)**: 7 story points (~4.5-5.5 days)
- **Sprint 5 (SIMD & Advanced)**: 8 story points (~5-6 days)
- **Sprint 6 (Documentation & Validation)**: 8 story points (~5 days)

**Total: 45 story points (~27-32 days of focused development)**

### Performance Targets (from Report)

| Metric | Baseline | Target | Expected |
|--------|----------|--------|----------|
| realize_uncertainties (50 hydros) | 120-150μs | 40-60μs | 2-3x speedup |
| Forward pass (100 hydros) | ~5s | ~2-2.5s | 50-60% faster |
| Backward pass (100 hydros) | ~8s | ~4-5s | 40-50% faster |
| Memory per subproblem | ~15 KB | ~8-10 KB | 30-40% reduction |
| Total memory (100 hydros) | ~75 MB | ~40-50 MB | 35-45% reduction |

### Risk Mitigation

**High-risk areas:**
- PERF-014: Multi-architecture SIMD (requires hardware access, fallback needed)
- PERF-015: Parallel constraint updates (may not be feasible)
- PERF-011: Vectorization complexity (may be difficult to optimize)

**Mitigation strategies:**
- Time-box exploratory work (spikes)
- Maintain scalar fallbacks for SIMD
- Extensive regression testing
- Incremental rollout with monitoring

### Success Criteria

The performance optimization effort is successful if:

1. ✅ **Hot path is 2-3x faster** (realize_uncertainties)
2. ✅ **Memory usage reduced by 30-40%**
3. ✅ **Numerical correctness maintained** (within 1e-8 tolerance)
4. ✅ **All tests pass**
5. ✅ **Documentation is comprehensive**
6. ✅ **No performance regressions in non-optimized paths**

---

## Notes for Implementation

### Parallel Work Opportunities

These tickets can be worked on in parallel:

- **Sprint 1**: PERF-001 → PERF-002 (sequential), but PERF-003 can start after PERF-002
- **Sprint 2**: PERF-005 (SIMD) can be done in parallel with PERF-004
- **Sprint 3**: All sequential
- **Sprint 4**: PERF-010 and PERF-011 are sequential, but independent from prior sprints
- **Sprint 5**: PERF-013 → PERF-014 sequential, PERF-015 independent (spike)
- **Sprint 6**: PERF-016 and PERF-017 can overlap, both block PERF-018

### Critical Path

**Must-have tickets (critical path to 2-3x speedup):**
1. PERF-001 → PERF-002 → PERF-003 → PERF-004 → PERF-006 → PERF-016 → PERF-018

**High-value enhancements:**
- PERF-007 → PERF-008 (memory optimization)
- PERF-005 (SIMD for additional speedup)

**Nice-to-have:**
- PERF-010 → PERF-011 (state optimization)
- PERF-013 → PERF-014 (multi-arch SIMD)
- PERF-015 (parallel constraints - experimental)

### Testing Strategy

**Continuous testing throughout:**
- Run regression tests after each ticket
- Maintain benchmark dashboard
- Monitor memory usage
- Track code coverage

**Final validation:**
- Comprehensive regression suite (PERF-016)
- Real-world case studies
- Long-running stability tests
- Cross-platform validation

---

**Document Version**: 1.0  
**Generated**: 2025-11-02  
**Total Tickets**: 18  
**Estimated Timeline**: 9 weeks (6 sprints)
