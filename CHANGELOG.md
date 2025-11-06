# v1.0.0 (2025-XX-XX) - Breaking Changes: Deprecated API Removal

## Breaking Changes

### [STATE-REFACTOR-005] Removed update_from_trajectory() from State trait 🔄❗

- **Breaking Change**: Removed `update_from_trajectory()` method from State trait
- **Rationale**: Method violated separation of concerns - State should extract, not update model
- **Architecture**: State trait now has NO model dependencies for extraction operations
- **Migration**: Use `extract_storage_from_trajectory()` and let Subproblem handle model updates
- **Before**: `state.update_from_trajectory(trajectory, model, constraints, variables)`
- **After**: `let storage = state.extract_storage_from_trajectory(trajectory); subproblem.update_storage_constraints(&storage)`
- **Impact**: Custom State implementations must remove `update_from_trajectory()` and rely on extraction pattern
- **Testing**: Updated 7 baseline tests to use extraction pattern without Model dependencies
- **Benefit**: True independence - State can be tested without solver, easier to implement custom states
- **Result**: All 354 tests passing, cleaner trait signature, complete separation achieved

## Internal Refactoring

### [STATE-REFACTOR-004] Consolidated storage constraint updates in Subproblem ✅

- **Feature**: Moved model updates from State implementations to `Subproblem::update_storage_constraints()`
- **Architecture**: ALL model updates now in Subproblem scope - matches pattern from REFACTOR-003
- **Implementation**: `prepare_from_trajectory()` now uses three-phase model: buffers → extraction → updates
- **Phase 1**: Update lag buffers (internal data structures)
- **Phase 2**: Extract state-dependent values (via `State::extract_storage_from_trajectory()`)
- **Phase 3**: Update solver model constraints (lag constraints + storage constraints in Subproblem)
- **Testing**: Added 5 comprehensive integration tests verifying extraction pattern works correctly
- **Test coverage**: Extraction pattern usage, storage constraint updates, StorageAndInflowState, idempotency, architectural consistency
- **Benefit**: Complete separation of concerns - State extracts values, Subproblem updates model
- **Result**: All 354 tests passing, no performance regression, cleaner architecture

### [STATE-REFACTOR-003] Added extraction methods to State trait 🔄

- **Feature**: Introduced `extract_storage_from_trajectory()` method to State trait establishing extraction pattern
- **Architecture**: State now provides values without updating model directly - matches UncertaintyManager pattern from REFACTOR-003
- **Implementation**: Both `StorageState` and `StorageAndInflowState` now extract storage and return values for Subproblem
- **Refactoring**: Updated `update_from_trajectory()` to use new extraction method internally (maintains backward compatibility)
- **Testing**: Added 8 comprehensive tests verifying extraction logic works independently of solver Model
- **Test coverage**: Extraction without Model, coefficient updates, heterogeneous AR orders, behavioral equivalence
- **Performance**: <1µs extraction overhead, small allocation acceptable (typically <100 elements)
- **Benefit**: Loose coupling between State and Subproblem, easier testing without solver setup, clear separation of concerns
- **Next steps**: STATE-REFACTOR-004 will move model updates from State to Subproblem::update_storage_constraints()

### [STATE-REFACTOR-002] Added baseline test suite for State trait model updates 🧪

- **Feature**: Created comprehensive baseline tests documenting current State trait behavior
- **Coverage**: 6 new tests covering StorageState and StorageAndInflowState implementations
- **Tests**: Verification of state coefficient extraction, model updates, edge cases, and idempotency
- **Test cases**: Zero storage values, heterogeneous AR orders (AR0/AR1/AR2), idempotency validation
- **Purpose**: Establish baseline behavior before refactoring and serve as regression tests
- **Benefit**: Documents current behavior and will catch any changes during STATE-REFACTOR-003

### [STATE-REFACTOR-001] Documented current State trait model update pattern 📝

- **Feature**: Added comprehensive documentation of current State trait responsibilities
- **Details**: Documented architectural inconsistency where State implementations directly update solver models
- **Architectural note**: Explained coupling issue and comparison with UncertaintyManager pattern
- **Forward-looking**: Added TODO comments referencing STATE-REFACTOR-003 for planned refactoring
- **Module docs**: Updated `src/state.rs` module documentation with "Current Architecture" and "Target Architecture" sections
- **Inline comments**: Added detailed phase-by-phase comments in `StorageState` and `StorageAndInflowState` implementations
- **Benefit**: Makes architectural decisions explicit and helps future refactoring efforts

### [REFACTOR-005] Refactored preprocessing API for clarity 📐

- **Feature**: Introduced cleaner `realize_and_solve()` API taking innovations directly
- **Clarity**: Clear two-phase model - `prepare_from_trajectory()` + `realize_and_solve()`
- **Backward compatibility**: `realize_uncertainties_new()` remains as thin wrapper over new API
- **Documentation**: Comprehensive doc comments explaining when to use each method
- **Testing**: Added 3 tests validating new API produces identical results to legacy API

### [REFACTOR-003] Hoisted lag constraint updates outside branching loop ⚡

- **Feature**: Major performance optimization - moved lag constraint updates outside the branching loop
- **Performance**: **10-15% speedup in backward pass** for large problems (eliminates ~98% redundant work)
- **Implementation**: Created `prepare_from_trajectory()` method combining all trajectory-based preprocessing
- **Architecture**: Established clean two-phase preprocessing model (trajectory → innovations)
- **Testing**: Added 3 comprehensive tests verifying optimization correctness and efficiency tracking

### [REFACTOR-002] Added efficiency tracking for lag constraint updates

- **Feature**: Added test instrumentation to track `update_lag_fixing_constraints()` call frequency
- **Benefit**: Enables verification that optimization (REFACTOR-003) works correctly and prevents regression
- **Details**: Thread-local counter with test-only compilation, zero overhead in production builds
- **Testing**: Added 4 comprehensive tests: counter correctness, single update per node, identical lag values across branchings, and counter reset

### [REFACTOR-001] Extracted lag buffer update helper function

- **Feature**: Added `Subproblem::update_lag_buffers_from_trajectory()` helper method to eliminate code duplication
- **Benefit**: Cleaner code, easier maintenance, consistent lag buffer handling across forward and backward passes
- **Details**: Extracted 70+ lines of duplicated lag buffer update logic from `solve_all_branchings()` into a reusable, well-tested helper function
- **Testing**: Added 5 comprehensive unit tests covering AR(1), AR(2), mixed AR orders, empty trajectory, and insufficient trajectory cases

## Fixed

- **Critical**: Fixed AR model indexing bug that caused incorrect results when system has mixed entity types (loads + inflows) with AR dynamics. Previously, lag duals were extracted into a compressed vector, causing index mismatches when loads had AR dynamics.
- Fixed `first_cut_row_index()` to correctly account for lag-fixing constraints, preventing cut placement errors with AR models
- Fixed `evaluate_cut()` to use direct indexing by entity ID, eliminating potential index mismatches

## Internal

- Refactored `Realization` to use separate `load_lag_duals` and `inflow_lag_duals` vectors indexed by entity ID (bus_id and hydro_id respectively)
- Improved AR model data structure design for correctness by construction through direct entity ID indexing
- Enhanced type safety and performance for AR model lag dual access (O(1) direct indexing vs O(n) lookup)

## Breaking Changes

This release completes the unified uncertainty handling refactoring by removing all deprecated APIs introduced in v0.4.0. If you are upgrading from v0.3.x or earlier, please first upgrade to v0.4.x and migrate your code before upgrading to v1.0.0.

### Removed Deprecated API

**Removed methods**:
- `Subproblem::new_from_uncertainty_models()` → Use `new_from_temporal_models()`
- `Subproblem::add_variables_to_subproblem()` → Use `add_variables()`
- `Subproblem::add_constraints_to_subproblem()` → Use `add_constraints()`
- `Subproblem::add_observation_space_inflow_variables()` → Internal method removed
- `Subproblem::add_observation_space_ar_constraints()` → Internal method removed
- `Subproblem::build_hydro_data()` → Use `build_entity_constraint_data()`
- `Subproblem::set_load_balance_rhs()` → Use load observation variables in constraints

**Removed fields**:
- `Variables::lagged_inflow_state` → Use `lagged_state` for unified lag tracking
- `Constraints::ar_dynamics` → Use `uncertainty_observation` for unified constraints

**Removed types**:
- `UncertaintyModel` enum is now fully deprecated (use `TemporalModel` struct)

### API Cleanup

**Method renames** (removed `_v2` suffixes):
- Methods no longer have `_v2` suffix since old versions were removed
- `add_variables()`, `add_constraints()`, `realize_uncertainties()` are now the primary methods

### Migration Guide

If you are still using the old API from v0.3.x:

1. **Replace `UncertaintyModel` with `TemporalModel`**:
   ```rust
   // Old (v0.3.x - v0.4.x):
   let model = UncertaintyModel::Independent { 
       entity_type: UncertaintyType::Inflow,
       entity_id: 0,
       seasonal_mean: vec![100.0],
       seasonal_std: vec![10.0],
       marginal_distribution: vec![MarginalDistribution::Normal { mean: 0.0, std: 1.0 }],
   };
   
   // New (v1.0.0+):
   let model = TemporalModel::from_par(
       UncertaintyType::Inflow,
       0,
       1,  // num_seasons
       vec![100.0],  // seasonal_mean
       vec![10.0],   // seasonal_std
       vec![MarginalDistribution::Normal { mean: 0.0, std: 1.0 }],
       vec![0],      // ar_order per season (0 for independent)
       vec![vec![]],  // ar_coefficients (empty for independent)
   ).unwrap();
   ```

2. **Update constructor calls**:
   ```rust
   // Old:
   let subproblem = Subproblem::new_from_uncertainty_models(&system, "storage", &models, 0);
   
   // New:
   let subproblem = Subproblem::new_from_temporal_models(&system, "storage", &models, 0);
   ```

3. **Update field references**:
   ```rust
   // Old:
   if let Some(lags) = &subproblem.variables.lagged_inflow_state { ... }
   let constraint_idx = subproblem.constraints.ar_dynamics[hydro_id];
   
   // New:
   if let Some(lags) = &subproblem.variables.lagged_state { ... }
   let constraint_idx = subproblem.constraints.uncertainty_observation[entity_id];
   ```

For detailed migration instructions, see the [v0.4.0 migration guide](docs/migration-guide.md).

### Internal Changes

- Removed ~400 lines of deprecated code
- Unified constraint handling for all uncertain entities
- Improved code maintainability and reduced complexity

### What's Next

The `inflow_constraints` module remains deprecated in this release and will be removed in v2.0.0.

---

# v0.4.0 (2025-11-02) - Unified Uncertainty Handling Refactoring

## Major Changes

### Unified Temporal Model Architecture

- **Unified TemporalModel representation**: Independent models are now correctly represented as PAR(0), eliminating artificial dichotomy between Independent and PAR models
- **New modules**:
  - `temporal_model`: Unified temporal model for all entities (loads and inflows)
  - `uncertainty_constraints`: Unified constraint management replacing `inflow_constraints`
- **Proper inverse CDF transformation**: Fixed LogNormal3 distribution handling using probability integral transform via Gaussian copula
- **Unified lag buffer management**: Single `UnifiedLagBuffer` for all entities replacing separate systems

### Bug Fixes

- **LogNormal3 distribution correctness**: Fixed incorrect transformation that wasn't preserving distribution properties
  - Old: Direct mean/std transformation (mathematically incorrect)
  - New: Inverse CDF via probability integral transform (mathematically correct)
  - Impact: Results with LogNormal3 distributions will differ (corrected values)

### Deprecations

The following items are deprecated and will be removed in v0.5.0:

**Subproblem methods**:
- `Subproblem::new_from_uncertainty_models()` → Use `new_from_temporal_models()`
- `Subproblem::add_variables_to_subproblem()` → Use `add_variables_v2()`
- `Subproblem::add_constraints_to_subproblem()` → Use `add_constraints_v2()`
- `Subproblem::build_hydro_data()` → Use `build_entity_constraint_data()`
- `Subproblem::update_ar_constraints_optimized()` → Use `update_uncertainty_constraints()`
- `Subproblem::set_load_balance_rhs()` → Use load observation variables in constraints
- `Subproblem::realize_uncertainties()` → Use `realize_uncertainties_new()`

**Types and modules**:
- `UncertaintyModel` enum → Use `TemporalModel` (Independent models are just PAR(0))
- `inflow_constraints` module → Use `uncertainty_constraints` module
- `LegacyTemporalModelInput` enum → Use `TemporalModelInput` struct

**State interface** (deprecated but functional):
- `State::has_lagged_inflow_state()` → Use `has_lagged_observation_state()` for unified approach

**Fields** (deferred to v0.6.0 removal, pending further state refactoring):
- `Variables::lagged_inflow_state` → Will be replaced by unified lag tracking
- `Constraints::ar_dynamics` → Replaced by `uncertainty_observation`

### Backward Compatibility

- All deprecated methods remain functional with deprecation warnings
- Old JSON format still supported via `LegacyTemporalModelInput`
- No breaking changes in this release
- Clear migration paths documented for all deprecations

### Documentation

- New JSON schema v2 documentation (`docs/json-schema-v2.md`)
- Comprehensive migration guide (`docs/migration-guide.md`)
- Updated refactoring tickets tracking document

### Testing

- All 309 unit tests passing
- Full backward compatibility maintained
- Clean clippy output for main library

### Performance

- Expected 5-10% improvement in scenario generation (unified code paths)
- No change in LP solve performance
- Slight memory reduction from unified structures

## Migration Guide

For users wanting to migrate to new APIs:

1. Update constructor calls to `new_from_temporal_models()`
2. Convert UncertaintyModel to TemporalModel using `to_temporal_model()`
3. Use new constraint update methods
4. Update JSON files to new format (optional, old format still works)

See `docs/migration-guide.md` for detailed instructions.

---

# v0.3.0 (Unreleased)

### Documentation Improvements

- **SG-001: Documented dual lag buffer systems** (Sprint 1 - Foundation)

  - Added comprehensive documentation explaining two lag buffer systems in the codebase
  - **ScenarioGenerator.par_states** (LEGACY): Used only during SAA generation, computes observations that are discarded for inflows
  - **Subproblem.inflow_manager** (ACTIVE): Used during SDDP execution, tracks observations in observation space
  - Added module-level documentation to `scenario_generator.rs` explaining what gets stored in SAA
  - Added detailed doc comments to `par_states` field explaining legacy nature
  - Added detailed doc comments to `inflow_manager` field explaining active usage
  - Added critical comment in `input.rs` explaining why only innovations (not observations) are stored for inflows
  - Reference: SCENARIO_GENERATION_CLEANUP_TICKETS.md (SG-001), SCENARIO_GENERATION_ANALYSIS.md

- **SG-002: Added tests verifying PAR states independence** (Sprint 1 - Validation)

  - Created `tests/test_scenario_generation.rs` with 3 tests validating scenario generation behavior
  - `test_par_states_independence`: Proves innovations are deterministic given same RNG seed (what goes to SAA)
  - `test_par_scenario_generation_sanity`: Validates PAR models produce reasonable statistical properties
  - `test_scenario_structure_populated`: Documents that all fields are populated but only innovations used for inflows
  - Tests provide confidence that par_states lag buffer can be safely removed without affecting SDDP execution
  - Reference: SCENARIO_GENERATION_CLEANUP_TICKETS.md (SG-002)

- **SG-003: Removed par_states from ScenarioGenerator** (Sprint 2 - Code Simplification) ⚡

  - **Impact: ~800 bytes memory saved, ~5-10% faster SAA generation, cleaner codebase**
  - Removed legacy `par_states` lag buffer system from ScenarioGenerator
  - PAR models now only sample innovations ε_t during SAA generation (not full observations)
  - Simplified `generate_stage_scenarios` to directly store innovations with placeholder values
  - Removed `reset_par_states` method (no longer needed)
  - Removed ~100 lines of legacy AR dynamics code that was computing discarded observations
  - Updated module documentation to explain simplified architecture
  - Memory: ScenarioGenerator reduced from ~7KB to ~6.3KB (11% reduction)
  - Performance: Eliminated unnecessary residual space computation during generation
  - All 307 library tests pass + 3 scenario generation tests pass
  - Reference: SCENARIO_GENERATION_CLEANUP_TICKETS.md (SG-003)

- **SG-004: Removed residuals field from Scenario struct** (Sprint 2 - Code Simplification) ⚡
  - **Impact: ~33% memory reduction for Scenario structs**
  - Removed unused `residuals` field from Scenario struct
  - Scenario memory reduced from ~480 bytes to ~336 bytes per scenario (20 entities)
  - Removed all `scenario.residuals.push()` calls from generation code
  - Updated Scenario documentation to clarify field usage by entity type
  - Updated tests to validate new structure (values + innovations only)
  - All 307 library tests pass + 3 scenario generation tests pass
  - Reference: SCENARIO_GENERATION_CLEANUP_TICKETS.md (SG-004)

### Performance Optimizations

- **PERF-001: HydroConstraintData structure** (Sprint 1 - Foundation)

  - Added `HydroConstraintData` struct to cache preprocessed hydro-specific constraint data
  - Eliminates need to iterate through generic `UncertaintyModel` objects in hot path
  - Pre-computes transformed AR coefficients (ψ*i) and deterministic noise base (μ_t - Σ[φ_i·μ*{t-i}])
  - Memory: ~136-200 bytes per hydro (vs ~500 bytes for full UncertaintyModel)
  - Access: O(1) direct field access with excellent cache locality
  - Foundational structure for subsequent hot path optimizations (PERF-002, PERF-004)
  - Comprehensive test coverage: 7 tests covering Independent, AR(1), AR(3), seasonal variation, memory size validation, coefficient transformation, and deterministic base correctness
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-001)

- **PERF-002: Refactor Subproblem to use HydroConstraintData** (Sprint 1 - Foundation)

  - Added `hydro_data: Vec<HydroConstraintData>` field to Subproblem struct
  - Implemented `build_hydro_data()` method to construct preprocessed constraint data during subproblem initialization
  - Filters inflow models, extracts constraint indices, and sorts by hydro_id for cache-friendly access
  - Deprecated `uncertainty_models` field (will be removed in PERF-006)
  - Memory: Expected 20-30% reduction per Subproblem (to be measured in PERF-003)
  - Performance: Sets foundation for 2-3x speedup in realize_uncertainties (PERF-004)
  - Test coverage: 6 new tests covering field population, sorting, constraint mapping, AR orders, filtering, and memory targets
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-002)

- **PERF-003: Baseline performance benchmarks** (Sprint 1 - Validation)

  - Created `benches/realize_uncertainties.rs` with Criterion framework benchmarks
  - Measures subproblem construction overhead (~86 µs for 50 hydros)
  - Measures hydro_data access pattern performance (~18 ns for 50 hydros)
  - Established baseline for validating PERF-004 speedup targets
  - Documentation: BENCHMARK_RESULTS.md with system specs and performance metrics
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-003)

- **PERF-004: Optimize realize_uncertainties hot path** ⚡ **(Sprint 2 - MAJOR PERFORMANCE WIN)**

  - **Impact: 2-3x speedup in realize_uncertainties, 40-50% faster SDDP forward passes**
  - Replaced two-step process (generate_precomputed_scenarios + update_observation_space_ar_constraints) with direct constraint update loop
  - New `update_ar_constraints_optimized()` method uses preprocessed hydro_data directly
  - **Zero heap allocations** in hot path loop (eliminates Vec<PrecomputedInflowScenario>)
  - Sequential iteration over hydro_data for excellent cache locality
  - Expected performance: ~40-60 µs per realize_uncertainties call (down from ~120-150 µs)
  - All 292 tests pass - mathematical correctness proven by algebraic equivalence
  - Old methods marked unused (will be removed in PERF-006 cleanup)
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-004), PERF-004-COMPLETION-SUMMARY.md

- **PERF-005: SIMD-optimized dot product utilities** (Sprint 2 - Hot Path Enhancement)

  - Added `src/utils/simd.rs` module with SIMD-optimized dot product implementations
  - Implemented `dot_product_simd()` using unsafe unchecked indexing for LLVM auto-vectorization
  - Implemented `dot_product_kahan_simd()` for numerically stable version (~10% slower, better precision)
  - Added `simd-optimizations` feature flag in Cargo.toml for opt-in SIMD
  - Performance: 1.2-1.4x speedup for AR(1)-AR(3) lag contributions (typical case: 3-10 elements)
  - Conditional compilation provides safe scalar fallback when feature disabled
  - Comprehensive test suite: 15 tests covering edge cases, typical AR scenarios, numerical stability
  - Benchmark suite: `benches/simd_dot_product.rs` for performance validation
  - Documentation: README updated with SIMD feature flag usage and expected speedups
  - Build: `cargo build --features simd-optimizations` for SIMD-enabled builds
  - Future: PERF-014 will add explicit AVX2/NEON intrinsics for 2-3x additional speedup
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-005), PERF-005-COMPLETION-SUMMARY.md

- **PERF-006: Remove deprecated code and cleanup** (Sprint 2 - Code Quality)

  - Removed `generate_precomputed_scenarios()` dead code function (replaced by PERF-004)
  - Removed `update_observation_space_ar_constraints()` dead code function (replaced by PERF-004)
  - Eliminated dead code compiler warnings
  - Code formatted with cargo fmt
  - All 307 tests pass with no regressions
  - **Deferred**: Full removal of `PrecomputedInflowScenario` struct and `uncertainty_models` field
    - Blocked by scenario generator dependencies and backward compatibility requirements
    - Will be completed in follow-up tickets PERF-006a, PERF-006b, PERF-006c
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-006), PERF-006-COMPLETION-SUMMARY.md

- **PERF-007: Implement OptimizedLagBuffer** ⚡ **(Sprint 3 - Memory & Cache Optimization)**

  - **Impact: 40% memory reduction, 3-4x faster lag access, better cache locality**
  - Created `OptimizedLagBuffer` struct with flattened Vec<f64> storage
  - Replaces Vec<Vec<f64>> (n allocations) with single contiguous allocation
  - Offset-based indexing enables O(1) access per hydro
  - Memory layout: [h0_lag0, h0_lag1, h1_lag0, h1_lag1, h1_lag2, ...]
  - Offset array: [0, 2, 5, 6, ...] for O(1) slice access
  - Methods: new(), get_lags(), get_lags_mut(), update_from_observations(), set_lags(), clear()
  - In-place updates using rotate_right() (LLVM optimized)
  - Memory: 100 hydros AR(2) reduced from ~4,000 bytes to ~2,500 bytes (40% savings)
  - Performance: Contiguous memory provides 3-4x faster access vs Vec<Vec<f64>>
  - Comprehensive test suite: 9 new tests covering construction, offsets, updates, edge cases
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-007), PERF-007-008-COMPLETION-SUMMARY.md

- **PERF-008: Integrate OptimizedLagBuffer** ⚡ **(Sprint 3 - Hot Path Integration)**

  - **Impact: Additional 10-15% speedup in per-stage time, zero API changes**
  - Updated `ObservationSpaceConstraintManager` to use OptimizedLagBuffer
  - Replaced `lag_buffer: Vec<Vec<f64>>` with `lag_buffer: OptimizedLagBuffer`
  - Modified all lag buffer methods: from_uncertainty_models(), get_lag_observations(),
    initialize_from_initial_condition(), update_lag_buffer(), update_lag_buffer_from_hydro_data(),
    clear_lag_buffer(), set_lag_buffer()
  - Updated 3 existing tests to use new API (all pass)
  - Zero breaking changes - internal optimization only
  - All 315 tests passing with no regressions
  - Expected cumulative speedup with PERF-004/005/007: 2.5-3.5x in SDDP iterations
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-008), PERF-007-008-COMPLETION-SUMMARY.md

- **PERF-009: Memory profiling and validation** ⚡ **(Sprint 3 - Validation)**
  - **Impact: Validated 67.6% memory reduction, exceeds all targets**
  - Created comprehensive memory profiling benchmarks (6 groups, 14 benchmarks)
  - Validated OptimizedLagBuffer: **39.8% memory reduction** (target: 40%) ✅
  - Measured full system: **67.6% memory reduction** (exceeds 30-40% target) ✅
  - Confirmed zero heap allocations in hot path (~87ns for 50 hydros) ✅
  - Benchmarked 10, 50, 100, 200 hydro systems (excellent scalability)
  - HydroConstraintData: ~150 bytes (70% smaller than UncertaintyModel)
  - OptimizedLagBuffer: 2,408 bytes vs 4,000 bytes (100 hydros AR(2))
  - Construction: ~120ns for 100 hydros (linear scaling)
  - All acceptance criteria met with statistical significance
  - Reference: PERFORMANCE_OPTIMIZATION_TICKETS.md (PERF-009), PERF-009-COMPLETION-SUMMARY.md

### Bug Fixes

- **Fixed lognormal standard deviation calculation in AR models**
  - **Issue**: Inflow values in simulation were extremely high (100-700+ instead of expected 20-40)
  - **Root Cause**: Using lognormal distribution's statistical std_dev instead of log-space parameter sigma
  - **Fix**: Changed lognormal parameter conversion in `SeasonalParams::from_distribution`
    - Now uses `mean = gamma + exp(mu)` and `std_dev = sigma` (direct parameter usage)
    - Previously computed lognormal distribution mean/variance (mean ≈ 27, std_dev ≈ 14.5 instead of exp(mu) ≈ 24, sigma = 0.5)
  - **Key Insight**: The log-space parameter `sigma` should be used to scale innovations in AR constraints, not the distribution's statistical standard deviation
  - **Scenario Generator**: No changes needed - the "breaking mathematical purity" approach of using lognormal innovations is correct and prevents negative inflows
  - **Impact**: All examples now produce correct inflow ranges; examples 06 and 07 (PAR models) fixed from infeasible to working
  - **Testing**: All 307 tests pass, all 7 examples verified correct
  - Reference: BUG-FIX-LOGNORMAL-INNOVATIONS.md

### Breaking Changes

- **Removed deprecated struct fields and methods**: Cleaned up deprecated API surface for cleaner v0.3.0 release:

  - **Removed `Variables::inflow_process` field** (deprecated since 0.3.0)
    - Use `lagged_inflow_state` field instead for lag state variables
    - Old field was multi-dimensional with unclear semantics
  - **Removed `Constraints::inflow_process` field** (deprecated since 0.2.0)
    - Use `ar_dynamics` (AR constraints) and `inflow_transform` (transformation constraints) instead
    - New fields have clear, documented purposes
  - **Removed `Subproblem::set_uncertainties()` method** (deprecated since 0.3.0)
    - Use `set_load_balance_rhs(bus_loads)` for load updates
    - Use `update_ar_constraint_rhs(innovations)` for inflow/AR dynamics updates
    - New methods are more explicit about what they update
  - **Architectural note**: With `UnifiedInflowModel`, lag variables are bounded (not constrained), so `lag_duals` are always zero. This is by design - bounded variables don't have duals in LP. Cut construction already handles this correctly by using 0.0 coefficients for lags when `lag_duals` is empty.
  - **Migration**: These fields/methods were deprecated for 1-2 minor versions. Update code to use new APIs before upgrading to v0.3.0.

- **BaseNoiseMethod enum simplified**: Removed unimplemented variants (`KMeans`, `QuasiMonteCarlo`, `LatinHypercube`) that were never functional. Only `Standard` variant remains. These variants added unused API surface and were marked with TODO comments since initial implementation. If variance reduction methods are needed in the future, they will be re-added with proper implementations.

- **Removed unreachable SDDP termination variants**: Removed the `Converged` and `TimeLimit` variants from `TerminationReason` in `src/sddp/mod.rs`.
- These variants were never constructed by the `train()` code paths; only `IterationLimit` is returned today.
- Keeping unreachable variants is confusing to users who expect these termination modes to be supported/configurable.
- If gap-based or time-limit termination is implemented in the future, the variants can be reintroduced along with tests and documentation.

### Improvements

- **Documentation Reorganization**: Improved documentation structure by extracting tutorial-level content from source code to dedicated documentation files:

  - **`sddp/mod.rs` module docs**: Condensed from 68 lines to 14 lines (79% reduction)
    - Removed: Entity listings (Buses, Lines, etc.), external dependencies list, detailed performance characteristics (memory usage patterns, threading model, complexity analysis, optimization decisions)
    - Kept inline: Brief description, key algorithmic features (parallel execution, cut management, memory efficiency, basis warm-starting, risk measures), link to algorithm docs
    - Rationale: Performance details are implementation concerns that evolve over time; module docs should focus on "what" and "why" (algorithm purpose, key capabilities) rather than "how" (implementation details)
    - Link: References `docs/algorithm/SDDP-OVERVIEW.md` for comprehensive algorithm explanation
    - Result: Scannable overview that guides users to detailed documentation
  - **`solver.rs` module docs**: Condensed from 24 lines to 5 lines (79% reduction)
    - Extracted: Detailed comparison with `highs` crate, rationale for using `highs-sys`, key modifications (3 changes), implementation details
    - Moved to: New `docs/architecture/SOLVER.md` with comprehensive architectural analysis
    - Kept inline: Brief description (direct HiGHS bindings), purpose (zero-cost LP/MIP solving), link to architecture docs
  - **`docs/architecture/SOLVER.md`**: New comprehensive architecture document
    - Context: SDDP requirements (50,000+ LP solves/run), design constraints
    - Comparison table: `highs-sys` vs `highs` crate across 12 dimensions
    - Key modifications: Single Problem type, no SolvedModel, additional APIs (basis management, incremental updates)
    - Implementation details: Memory management (Arc), error handling, type conversions, thread safety
    - Performance analysis: Overhead comparison (~23 ns/solve savings), actual bottlenecks
    - Trade-offs: Benefits (full API, zero overhead, memory control) vs Costs (unsafe code, maintenance)
    - Future considerations: When to reconsider, alternative solver support, refactoring options
    - 7 references: HiGHS docs, papers, related POWE.RS files
  - **`docs/architecture/README.md`**: New architecture documentation index with document template
  - **`docs/README.md`**: Updated to include SOLVER.md in architecture section and quick navigation
  - **Result**: Architectural decisions documented separately from API reference, easier to find rationale for design choices

  - **`lognormal3.rs` module docs**: Condensed from 70 lines to 22 lines (69% reduction)
    - Extracted: Mathematical definition, statistical properties, sampling algorithm details, correlation integration details, performance analysis
    - Moved to: New `docs/reference/distributions.md` with comprehensive treatment
    - Kept inline: Brief description, key features, minimal usage example, link to detailed docs
  - **`docs/reference/distributions.md`**: New comprehensive reference document
    - Mathematical definition with formal notation
    - Statistical properties (moments, shape characteristics)
    - Sampling algorithm with complexity analysis
    - Integration with correlation framework (Gaussian copula details)
    - Performance characteristics with benchmarks
    - 5 practical code examples (basic usage, parameter estimation, correlation integration)
    - 6 academic references (Aitchison & Brown 1957, Crow & Shimizu 1988, etc.)
  - **`docs/README.md`**: Updated to include distributions.md in reference section and quick navigation
  - **Result**: Module docs more scannable, detailed mathematical background easily accessible in rendered documentation

- **Test Documentation Consolidation**: Improved readability of test files by consolidating mathematical derivations into doc comments:

  - **`test_lognormal3_correct_moments`**: Moved 3-parameter log-normal moment formulas (E[X], Var[X]) from inline comments to structured doc comment with proper mathematical notation
  - **`expected_solution_bounds` fixture**: Reorganized 11-line resource balance calculation into formatted doc comment with clear sections (Cost Structure, Resource Balance, Expected Range)
  - **Preservation**: All mathematical information preserved - no calculations removed
  - **Result**: Test code more scannable, derivations easier to find in generated documentation
  - Related: Most test files already follow best practices with doc comments (e.g., `test_par_validation.rs` hand-calculation derivations)

- **Stochastic Process Documentation Clarity**: Removed stale TODO comments in `stochastic_process.rs` and improved inline documentation:

  - Clarified that `realize()` method limitations are by design (lifetime constraints)
  - Documented that `realize_owned()` is the proper method for PAR process realizations
  - Improved comments explaining PAR initialization requirements (residual conversion from observed inflows)
  - Clarified that simple `factory()` intentionally cannot create PAR processes (use `factory_with_config()` instead)
  - No functional changes - all existing tests pass

- **Input Validation Enhancements**: Added comprehensive cross-file consistency validation in `InputValidator`:

  - **Entity ID validation**: Verifies `entity_id` in `uncertainty_specifications` references existing entities in `system.json`:
    - Inflow uncertainties must reference valid `hydro_id` (existing hydro plant)
    - Load uncertainties must reference valid `bus_id` (existing bus)
    - Provides clear error messages with available ID ranges
  - **Initial condition validation**: Validates all `initial_condition` references:
    - Storage `hydro_id` values must match existing hydros
    - Inflow lag `hydro_id` values must match existing hydros
    - Prevents silent dimension mismatches
  - **Season ID consistency**: Validates `season_id` in `seasonal_distributions` match seasons used in `graph.json` nodes
  - **Fail-fast behavior**: Invalid configurations caught at input validation time (before SDDP training starts)
  - **Testing**: Added 8 comprehensive tests covering valid/invalid entity references and season IDs
  - **Performance**: Validation uses HashSet lookups (O(1) per check), negligible overhead (~1ms for typical problems)

- **Unused Enum Removal**: Removed `HighsBasisStatus` enum from `src/solver.rs`:

  - Enum was defined but never used in the codebase
  - Basis functionality uses `Basis` struct with raw `usize` values, not enum variants
  - Enum added no type safety and was marked with `#[allow(dead_code)]` since introduction
  - Actual basis reuse implementation in SDDP works correctly without it
  - Can be easily recreated if typed basis status is needed in the future

- **Dead Code Attribute Audit**: Cleaned up spurious `#[allow(dead_code)]` attributes across the codebase:

  - **Removed false positives**: Attributes on actually-used code (`Sense` enum, `UnifiedInflowModel` struct, `SystemMetadata` struct)
  - **Removed truly unused code**: Deleted `validate_entity_count` function (src/input.rs) and `extract_seasonal_params` function (src/state.rs) that had no references
  - **Documented legitimate uses**: Added comments for remaining attributes:
    - `HighsPtr::ptr()`: Used only in tests (test_highs_ptr_clone_creates_independent_instance)
    - `Subproblem::set_uncertainties()`: Deprecated method kept for backward compatibility
  - **Result**: Reduced from 7 attributes to 2 with clear justifications
  - All code compiles cleanly with `-D warnings`, zero test breakage

- **Redundant Comment Cleanup**: Removed obvious "what" comments that simply restated code:

  - Removed 7 redundant comments from src/sddp/mod.rs, src/base_noise.rs, and src/stochastic_process.rs
  - Examples removed:
    - "Extract costs into separate vector for sorting" (map/collect pattern is self-explanatory)
    - "Validate parameters" (immediately followed by obvious validation code)
    - "Count total solver calls" (sum operation is self-explanatory)
    - "Calculate sample mean/std dev" (formula is self-documenting)
  - Preserved valuable comments explaining "why" (algorithm rationale, performance notes, edge cases)
  - Improves code readability by reducing noise without sacrificing understanding

- **Performance Documentation Consolidation**: Moved detailed performance analysis from inline comments to module-level documentation:

  - Added comprehensive "Performance Characteristics" section to `src/sddp/mod.rs` module docs covering:
    - Memory usage patterns (training vs simulation phases)
    - Extract-and-Release pattern: O(threads) memory vs naive O(scenarios) approach (96% reduction)
    - Threading model with Rayon work-stealing scheduler
    - Computational complexity for training and simulation
    - Optimization decisions (pre-allocation, basis reuse, cut batching)
  - Condensed verbose inline comment (12 lines) to brief reference: "Extract-and-Release: O(threads) memory vs O(scenarios). See module docs."
  - **Rationale**: Detailed performance analysis belongs in module docs (read once for understanding) not inline (creates noise during code navigation)
  - **Impact**: Improved code readability while making performance characteristics more discoverable via `cargo doc`

- **TODO Comment Cleanup**: Removed all low-priority TODO comments from source code and consolidated them into comprehensive `FUTURE_WORK.md` document:

  - **7 future enhancements documented** with context, use cases, and effort estimates:
    - Algorithm: Markovian/cyclic graph support, unified load uncertainty model
    - Input/Output: CSV transformation for PAR models, skewness parameter tracking
    - Configuration: Automatic seasonal parameter extraction, multi-season initial conditions
    - Numerical Methods: Eigenvalue-based stationarity check (vs current heuristic)
  - **Replaced inline TODOs** with clear references to FUTURE_WORK.md sections
  - **Zero functional changes**: All code behavior identical, only documentation improved
  - **Priority guidance**: Enhancements ordered by value/effort ratio for future implementation
  - **Link added to README.md**: Contributors can easily find planned enhancements
  - **Maintenance plan**: Document updated as enhancements are identified or implemented

- **Sprint 1 Validation and Summary**: Completed comprehensive validation of all Sprint 1 cleanup work:
  - **Validation suite executed**: All pre-flight, code quality, functional, performance, and documentation checks passed
  - **Metrics documented**: Created `SPRINT_1_SUMMARY.md` with complete sprint statistics
  - **Code impact**: Net removal of ~5,200 lines (11,741 deletions, 6,493 insertions across 76 files)
  - **Quality maintained**: Zero build warnings, 494/494 tests passing, all 5 examples working
  - **No regressions**: Benchmarks show no performance impact from cleanup work
  - **Velocity established**: Sprint 1 completed in ~4.5 hours (1.56 story points/hour)
  - **Handoff ready**: Sprint 2 preparation complete with lessons learned and recommendations

### Added

- **Backward Compatibility Layer for New Format (TICKET-14)**: Graph building now supports both old and new uncertainty formats

  - **Automatic conversion**: Added `Recourse::get_or_create_noise_models()` method
    - Handles both `noise_models` (old) and `uncertainty_specifications` (new) formats
    - Converts new format to old format on-the-fly for backward compatibility
    - Independent models: one NoiseModel per season
    - PAR models: single NoiseModel at season_id=0 (convention)
  - **Graph building updated**: Both study and pre-study graph construction use conversion helper
    - `GraphInput::add_sddp_study_period_to_graph()`: Uses converted noise models
    - `GraphInput::add_sddp_pre_study_period_to_graph()`: Uses converted noise models
    - Eliminates hard dependency on deprecated `noise_models` field
  - **Zero-copy when possible**: If old format present, returns reference directly (no conversion)
  - **Error handling**: Clear error messages if neither format is present
  - **Testing**: All 424 tests pass, all 4 basic examples work with both formats
  - **Future path**: New format is source of truth; core API will eventually accept new format directly

- **NoiseModelCache for Pre-Initialized Generators (TICKET-13)**: Added caching layer for 5-10% performance improvement

  - **Cache structure**: Pre-initialized PAR generators and cached distributions
    - `NoiseModelCache`: HashMap-based O(1) lookups for generators and distributions (single-threaded, RefCell)
    - `NoiseModelCacheSync`: Thread-safe variant using `std::sync::Mutex` for parallel scenario generation
    - PAR generators pre-initialized with warm start from initial conditions
    - Distributions cached with pre-validated parameters
  - **Performance improvements**:
    - Cache construction: ~1-2ms for typical problems (10 hydros, 12 seasons)
    - Scenario generation: 5-10% faster than legacy path
    - Memory overhead: ~23KB for typical problem (negligible vs 480KB scenario storage)
    - O(1) parameter lookups via HashMap vs O(n) linear searches
  - **Thread-safe parallel generation** (NoiseModelCacheSync):
    - Lock overhead: ~30ns per entity per scenario (std::Mutex)
    - Near-linear speedup for independent entities (N cores → ~N× faster)
    - Entity-level parallelism: Each entity has own Mutex (minimal contention)
    - `par_generate_scenarios()`: Rayon-based parallel generation across stages
    - Thread-local RNGs for zero contention on random number generation
  - **PAR generator enhancement**: Added `generate_next_for_season()` for explicit season control
  - **Integration**: Automatic cache usage in `generate_sddp_noises()` when unified specs available
    - Falls back to legacy path if cache construction fails (safe degradation)
    - Maintains backward compatibility with existing tests
  - **Cache operations**:
    - `from_unified_specs()`: Build cache with warm-started PAR generators from initial conditions
    - `generate_stage_scenarios()`: O(1) scenario generation using pre-built cache
    - `reset_par_generators()`: State management for multi-run scenarios
    - `validate()`: Completeness checking for all entities
  - **Memory efficiency**: Cached data reused across all stages, eliminating repeated allocations
  - **Testing**: 11 comprehensive unit tests covering construction, validation, generation, and thread-safety
  - **New module**: `src/noise_model_cache.rs` (~1400 lines) with single-threaded and thread-safe implementations

- **Deprecation Warnings and Logging (TICKET-12)**: Added user-facing warnings to guide migration from old to new format

  - **Deprecation warning**: Comprehensive terminal message when old `noise_models` format is loaded
    - Displays boxed warning with migration guide, command examples, and timeline
    - Includes migration command: `powers migrate-format <path> --backup`
    - Timeline: v0.3.x-v0.5.x (warnings), v0.6.0 (removal, Q1 2026)
    - Warning shown once per file load (not spammy)
  - **Suppression mechanism**: Environment variable to disable warnings in CI/automation
    - Set `POWERS_SUPPRESS_DEPRECATION_WARNINGS=1` to suppress warnings
    - Documented in warning text and migration guide
  - **Version enforcement**: Safety check to ensure old format removal at v0.6.0
    - `check_deprecation_version()`: Panics if v0.6.0+ still supports old format
    - Warning escalation at v0.5.6+ (6 patch releases before removal)
    - Protects against accidental timeline extension
  - **Debug logging**: Format detection logging for monitoring migration progress
    - `log_format_info()`: Logs which format was used (old vs new) at debug level
    - Called automatically when loading recourse files
    - Controlled by `POWERS_DEBUG` environment variable
  - **Testing**: 7 comprehensive tests for warning behavior, suppression, validation
  - **Performance**: <1μs overhead for version check (cold path only)

- **Format Migration and Validation Tools (TICKET-11)**: Added CLI commands for safe migration from old to new format

  - **New commands**:
    - `powers migrate-format`: Migrate `recourse.json` files from `noise_models` to `uncertainty_specifications`
    - `powers rollback-migration`: Restore files from `.backup` versions with safety checks
  - **Migration features**:
    - Format equivalence validation: Ensures old and new formats produce identical `UnifiedNoiseSpec`
    - Dry-run mode: Preview changes without modifying files (`--dry-run`)
    - Automatic backups: Create `.backup` files before migration (`--backup`)
    - Recursive migration: Migrate entire directories (`--recursive`)
    - Force mode: Continue even with validation warnings (`--force`, not recommended)
  - **Rollback safety**:
    - Verifies backup exists before rollback
    - Validates backup is valid JSON
    - Creates safety backup during rollback
    - Restores from safety backup if rollback fails
  - **Validation**:
    - `validate_format_equivalence()`: Compares old and new formats with epsilon tolerance (1e-10)
    - Checks entity coverage, temporal model types, seasonal parameters, AR coefficients
    - Detailed error messages with entity ID, season ID, and parameter names
  - **Performance**: ~1-2ms per file (I/O bound), can process 100+ files/second
  - **Testing**: 2 comprehensive tests for equivalence validation and mismatch detection
  - **New module**: `src/migration.rs` (~770 lines) with migration logic and utilities
  - **Documentation**: Comprehensive CLI help text with examples and recommended workflows

- **JSON Schema for New Uncertainty Format (TICKET-10)**: Added schema definitions for `uncertainty_specifications` format
  - Updated `schemas/recourse.schema.json` with dual format support (old + new)
  - **Backward compatibility**: Schema validates both `noise_models` (old) and `uncertainty_specifications` (new) formats using `oneOf` constraint
  - **Deprecation notices**: Old format marked deprecated, removal planned for v0.6.0
  - **New definitions**:
    - `UncertaintySpecification`: Entity-level uncertainty with PAR or independent temporal models
    - `TemporalModelInput`: Discriminated union (type: "periodic_ar" | "independent") for IDE autocomplete
    - `SeasonalDistribution`: Per-season parameters for independent models
  - **Comprehensive examples**: 3 examples in schema (PAR model, independent model, mixed models)
  - **Validation rules**: Documented schema rules vs runtime rules with migration notes
  - **Testing**: 13 new tests for dual format validation, discriminated unions, required fields, constraints, examples, and deprecation
  - **IDE integration**: Schema enables autocomplete and validation in VS Code/IntelliJ
  - **Documentation**: Validation rules, migration notes, and examples embedded in schema

### Internal

- **Unified Noise Specification (PAR-INPUT-01)**: Added internal `UnifiedNoiseSpec` representation

  - New module: `src/unified_noise_spec.rs` with entity-level temporal models separated from seasonal parameters
  - O(1) HashMap-based lookups for seasonal parameters (replaces O(n) linear search)
  - Foundation for input format refactoring (no breaking changes to public API)
  - Performance: ~1KB memory per entity, 10-20% speedup expected in scenario generation
  - Documentation: `docs/architecture/unified-noise-spec.md`

- **NoiseModel to UnifiedNoiseSpec Converter (PAR-INPUT-02)**: Added backward-compatible converter

  - Method: `UnifiedNoiseSpec::from_noise_models()` transforms legacy format to new internal representation
  - Handles PAR models (extracts 12 seasons from single entry) and independent models (aggregates across seasons)
  - Validates consistency: detects duplicate PAR definitions, mixed temporal models for same entity
  - O(n) conversion time where n = number of NoiseModel entries (typically <1ms for 120 entries)
  - Comprehensive error messages with entity_id and season_id context
  - Zero breaking changes: all existing JSON files continue to work

- **Comprehensive UnifiedNoiseSpec Validation (PAR-INPUT-03)**: Enhanced validation framework

  - **Enhanced `validate()` method**:
    - Finite value checks (NaN, Inf detection for all statistical parameters)
    - std_dev bounds (>0, <1e6 with helpful error messages)
    - AR coefficient validation (finite values, |φ| < 10 warning threshold)
    - Season ID range validation (0..num_seasons-1)
    - Error aggregation (collects all errors, not fail-fast)
  - **New `validate_against_graph()` method**:
    - Entity ID existence checks (hydro_id < num_hydros, bus_id < num_buses)
    - Season ID consistency with graph structure (all seasons in specs exist in graph)
    - PAR num_seasons matches graph unique season count
  - **New `validate_noise_specs()` function**:
    - Collection-level duplicate detection (unique entity_id × uncertainty_type)
    - Entity coverage validation (all hydros have inflows, all buses have loads)
    - Cross-validation with graph and system structures
  - **Performance**: O(n) validation, <1ms for typical problem (10 entities × 12 seasons)
  - **Error reporting**: Detailed messages with entity_id, season_id, parameter names, constraints, and actionable suggestions
  - **Testing**: 10 new comprehensive tests (finite values, bounds, graph mismatches, coverage, error aggregation)

- **Test Infrastructure for Format Conversion (PAR-INPUT-04)**: Comprehensive test suite for conversion validation

  - **Test fixture loaders**: Load examples from `examples/` directory for integration testing
  - **Builder utilities**: Helper functions for creating test specs (PAR and independent models)
  - **Comparison utilities**: Tolerance-based comparison for UnifiedNoiseSpec equivalence
  - **Conversion tests**: 11 tests covering simple PAR, mixed models, edge cases
  - **Collection validation tests**: Duplicate detection, missing entity detection
  - **Performance baseline**: Conversion takes <100μs for typical cases (42μs for 13 noise models)
  - **Test coverage**: Tests validate conversion correctness, structural integrity, and cross-validation
  - **Documentation**: Test organization, fixture patterns, utility usage

- **Integration Tests for Scenario Generation (PAR-INPUT-08)**: Comprehensive integration testing

  - **New test file**: `tests/test_scenario_generation_integration.rs` (9 tests, 456 lines)
  - **Example-based tests**: All examples (01-06) work unchanged with refactored code

- **Dual Format Support for Recourse Struct (PAR-INPUT-09)**: Added new public API while maintaining backward compatibility

  - **New format**: `uncertainty_specifications` field with clearer entity-level structure
  - **Old format**: `noise_models` field marked deprecated (removal in v0.6.0)
  - **New public structs**:
    - `UncertaintySpecification`: One entry per entity (vs N×M entries in old format)
    - `SeasonalDistribution`: Explicit per-season parameters for independent models
    - `TemporalModelInput`: Public-facing temporal model enum
  - **Validation**: `validate_format()` ensures exactly one format specified
  - **Conversion**: `get_unified_specs()` converts either format to internal UnifiedNoiseSpec
  - **Backward compatibility**: All existing noise_models JSON files continue to work with deprecation warning
  - **Benefits**: Clearer structure, no misleading season_id for PAR models, explicit marginal/temporal separation
  - **Testing**: 11 new tests in `tests/test_dual_format_support.rs` (deserialization, validation, conversion, serialization)
  - **Documentation**: Complete migration guide at `docs/migration/PAR_INPUT_FORMAT.md`
  - **Serialization**: Added Serialize trait to all input structs for round-trip testing
  - **Timeline**: Deprecated in v0.5.0, removed in v0.6.0
    - Test examples: deterministic, stochastic, multistage (60 stages), cascade (multi-hydro)
    - Coverage: 4 active examples tested (01-04), 2 expensive tests marked `#[ignore]` (05-large-scale, 06-par-model)
  - **Determinism tests**: Same seed → identical results (bit-for-bit), different seeds → different results
  - **Numerical stability tests**: All bounds finite, monotonic improvement, reasonable ranges
  - **Performance regression tests**: Small (<500ms), medium (<3s), large (<10s) baselines
  - **Test execution**: 9 tests passing, 3 ignored (expensive), 0.53s total runtime
  - **Coverage**: Validates correctness of TICKET-05, TICKET-06, TICKET-07 optimizations in full SDDP context
  - **Purpose**: Final validation before exposing new API (TICKET-09)

- **Optimized Scenario Generation (PAR-INPUT-05)**: Refactored scenario generation with O(1) lookups

  - **New `NoiseLookupTable` structure**: Pre-indexed lookup table for O(1) parameter access
    - HashMap-based indexing: (uncertainty_type, entity_id, season_id) → (mean, std_dev, marginal)
    - Temporal model caching: O(1) check if entity uses PAR or independent model
    - Memory: ~80 bytes per (entity, season) entry with pre-allocation to avoid rehashing
    - Performance: O(1) average case lookups vs O(n) linear search through noise models
  - **Optimized `generate_sddp_noises()`**: Reduced per-stage overhead
    - Pre-build HashMap index: Convert O(n×s) repeated filtering to O(n) preprocessing + O(1) lookups
    - Eliminates redundant iterations through noise models for each stage
    - Maintains exact numerical behavior (same seed → same scenarios)
    - All 400 tests passing (393 original + 7 new NoiseLookupTable tests)
  - **Performance improvement**: 10-20% faster scenario generation for multi-entity problems
    - Before: O(n×s) where n = noise model entries, s = stages
    - After: O(n + s) with HashMap preprocessing
    - Typical example (10 entities, 12 stages): ~15% speedup measured
  - **API stability**: Zero breaking changes to public API
    - Internal optimization only - existing code continues to work unchanged
    - `generate_sddp_noises()` signature and behavior preserved
  - **Testing**: 7 new comprehensive tests for NoiseLookupTable
    - Construction from empty/single/multiple specs
    - O(1) parameter lookups (independent and PAR models)
    - Missing entity/season handling (returns None, not panic)
    - Marginal distribution retrieval
    - Performance test with 100 entities

- **Bulk Retrieval Optimizations (PAR-INPUT-06)**: Added cache-friendly bulk parameter access

  - **New `get_all_params_for_season()` method**: Retrieve all entity parameters for a season at once
    - Returns Vec of (entity_id, params) sorted by entity_id for predictable access
    - More efficient than repeated `get_params()` calls when processing many entities
    - Enables cache-friendly iteration patterns in hot loops
    - CPU prefetcher benefits from sequential entity_id access
  - **Performance analysis methods**: Added `param_count()` and `entity_count()` for profiling
    - `param_count()`: Total (entity, season) parameter entries
    - `entity_count(uncertainty_type)`: Number of unique entities per type
    - Useful for pre-allocation and memory profiling
  - **Usage pattern**:

    ```rust
    // BEFORE: Multiple scattered HashMap lookups
    for entity_id in 0..num_hydros {
        if let Some(params) = lookup.get_params(Inflow, entity_id, season) {
            process(params);
        }
    }

    // AFTER: Single bulk retrieval, cache-friendly iteration
    let all_params = lookup.get_all_params_for_season(Inflow, season);
    for (entity_id, params) in &all_params {
        process(params);
    }
    ```

  - **Testing**: 7 new tests for bulk retrieval (empty, single, multiple entities, mixed types, multiple seasons)
  - **Documentation**: Comprehensive doc comments with usage patterns and performance notes
  - **When to use**: Processing all entities in scenario generation loops, stage-by-stage building
  - **When NOT to use**: Single entity lookups, sparse access patterns
  - **Total tests**: 414 passing (407 from PAR-INPUT-05 + 7 new bulk retrieval tests)

- **Performance Benchmarks for Lookup Optimizations (PAR-INPUT-07)**: Added comprehensive benchmark suite
  - **New benchmark file**: `benches/lookup_structures.rs` with 7 benchmark groups
    - `lookup_table_construction`: Measures O(n×s) pre-indexing cost during table construction
    - `construction_independent_vs_par`: Compares PAR vs independent model construction overhead
    - `single_lookup`: Benchmarks O(1) HashMap lookups (worst/best/average cases)
    - `lookup_scaling`: Validates O(1) complexity (constant time regardless of table size)
    - `bulk_vs_single_lookups`: Compares bulk retrieval vs repeated single lookups
    - `bulk_retrieval_scaling`: Verifies O(n) bulk retrieval maintains linear scaling
    - `profiling_helpers`: Benchmarks `param_count()` and `entity_count()` methods
  - **Scaling tests**: Validates performance from 10 to 1000 entities
    - Construction: Linear O(n×s) scaling confirmed
    - Single lookup: Constant O(1) time (~2-5ns) regardless of table size
    - Bulk retrieval: Linear O(n) scaling with better cache locality
  - **Comparison benchmarks**: Bulk retrieval vs repeated lookups
    - 100 entities: ~30-40% faster with bulk retrieval due to reduced HashMap overhead
    - Cache-friendly sequential access improves CPU prefetching
  - **Throughput metrics**: Elements/second for construction and retrieval operations
    - Enables detection of performance regressions in CI
  - **Purpose**: Validate 10-20% speedup from TICKET-05 and guide future optimizations
  - **Usage**: `cargo bench --bench lookup_structures` to run benchmarks
  - **Integration**: Foundation for CI performance regression detection

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

  - Unified `ScenarioGenerator` struct integrating all 4 stages of pipeline
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

  - New `ar_dynamics` module for Stage 4 of scenario generation pipeline (final stage)
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

  - New `marginal_transformer` module for Stage 3 of scenario generation pipeline
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

  - New `correlation_applicator` module for Stage 2 of scenario generation pipeline
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

  - New `base_noise` module for Stage 1 of scenario generation pipeline
  - `BaseNoiseGenerator` struct generates independent Z ~ N(0,1) samples
  - `BaseNoiseMethod` enum with `Standard` variant for direct random sampling
  - Deterministic generation via `Xoshiro256Plus` RNG with seed control
  - Performance: <10ms for 1000 scenarios × 10 entities
  - 10 unit tests covering dimensions, statistical properties, determinism, independence, validation
  - Note: Variance reduction methods (KMeans, QuasiMonteCarlo, LatinHypercube) were initially stubbed but removed in v0.3.0 as they were never implemented

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
