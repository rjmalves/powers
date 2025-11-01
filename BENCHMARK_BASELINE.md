# Benchmark Baseline (Post-Refactor)

**Date**: 2025-11-01  
**Commit**: $(git rev-parse HEAD)  
**Branch**: fix/test-modernization  
**Status**: Phase 3 Partial Completion

## 📊 Benchmark Status

### Working Benchmarks (7/14 - 50%)

The following benchmarks compile and run successfully after the uncertainty_model refactor:

#### Core Performance Benchmarks
1. ✅ **sddp_benchmarks.rs** - SDDP algorithm performance
   - Measures: Algorithm iterations, convergence, training time
   - Status: Fully functional
   - Priority: **CRITICAL** (core algorithm)

2. ✅ **memory_profiling.rs** - Memory usage profiling
   - Measures: Memory allocation patterns, peak usage
   - Status: Fully functional
   - Priority: **HIGH** (performance critical)

3. ✅ **simulation_memory.rs** - Simulation memory patterns
   - Measures: Simulation phase memory optimization
   - Status: Fully functional
   - Priority: **HIGH** (validates SIM-OPT-005/006)

#### Overall & Feature Benchmarks
4. ✅ **comprehensive_benchmarks.rs** - Overall benchmark suite
   - Measures: End-to-end performance across examples
   - Status: Fully functional
   - Priority: **MEDIUM**

5. ✅ **cut_id_lookup.rs** - Cut lookup performance
   - Measures: Cut selection lookup efficiency
   - Status: Fully functional
   - Priority: **MEDIUM**

6. ✅ **marginal_transformation.rs** - Transformation benchmarks
   - Measures: Marginal value transformations
   - Status: Fully functional
   - Priority: **LOW**

7. ✅ **parallel_efficiency.rs** - Parallel performance
   - Measures: Thread scaling, parallel efficiency
   - Status: Fully functional
   - Priority: **MEDIUM**

### Benchmarks Needing Work (7/14 - 50%)

The following benchmarks need API migration to work with uncertainty_model:

#### Needs API Migration
1. ❌ **subproblem_solve.rs** - Solver performance
   - Issue: Uses old stochastic_process API
   - Work done: 3/8 sections fixed
   - Priority: **HIGH** (measures hot path)
   - Estimate: 1 hour to complete

2. ❌ **par_performance.rs** - PAR generation performance
   - Issue: Uses deleted par_generator module
   - Priority: **MEDIUM**
   - Estimate: 1 hour

3. ❌ **correlation_application.rs** - Correlation performance
   - Issue: Uses old API
   - Priority: **LOW**
   - Estimate: 30 minutes

4. ❌ **cut_selection.rs** - Cut selection performance
   - Issue: Uses old API
   - Priority: **MEDIUM**
   - Estimate: 1 hour

5. ❌ **lookup_structures.rs** - Lookup structures
   - Issue: Uses old API
   - Priority: **LOW**
   - Estimate: 30 minutes

6. ❌ **state_operations.rs** - State operations
   - Issue: Uses old API
   - Priority: **MEDIUM**
   - Estimate: 1 hour

#### May Be Obsolete
7. ❌ **par_generator.rs** - PAR generator
   - Issue: Tests deleted par_generator module
   - Priority: **LOW** (may be obsolete)
   - Action: Evaluate if still relevant or remove

## 🎯 Running Working Benchmarks

You can run the working benchmarks immediately:

```bash
# Core SDDP performance (most important)
cargo bench --bench sddp_benchmarks

# Memory profiling
cargo bench --bench memory_profiling
cargo bench --bench simulation_memory

# Comprehensive suite
cargo bench --bench comprehensive_benchmarks

# Specific features
cargo bench --bench cut_id_lookup
cargo bench --bench marginal_transformation
cargo bench --bench parallel_efficiency

# Run all working benchmarks
cargo bench --bench sddp_benchmarks \
            --bench memory_profiling \
            --bench simulation_memory \
            --bench comprehensive_benchmarks \
            --bench cut_id_lookup \
            --bench marginal_transformation \
            --bench parallel_efficiency
```

## 📝 Notes

### API Migration Pattern

For benchmarks that need fixing, the pattern is:

**Old API:**
```rust
let load_sp = powers_rs::stochastic_process::factory("naive");
let inflow_sp = powers_rs::stochastic_process::factory("naive");
let inflow_processes = vec![inflow_sp];
let subproblem = Subproblem::new(
    &system,
    "storage",
    load_sp.as_ref(),
    &inflow_processes,
    &[],
    0,
);
```

**New API:**
```rust
let uncertainty_models = vec![
    powers_rs::uncertainty_model::UncertaintyModel::Deterministic; 
    num_hydros
];
let subproblem = Subproblem::new_from_uncertainty_models(
    &system,
    "storage",
    &uncertainty_models,
    0,
);
```

### Performance Baseline

**Note**: Baseline performance numbers will be documented after running benchmarks.
The working benchmarks can be run to establish baseline at any time.

Key metrics to track:
- SDDP iteration time (forward + backward pass)
- Memory usage per scenario
- Simulation memory patterns
- Solver overhead

### Regression Detection

Use the working benchmarks to detect performance regressions:
1. Run benchmarks before changes: `cargo bench --bench sddp_benchmarks`
2. Make code changes
3. Run benchmarks after: `cargo bench --bench sddp_benchmarks`
4. Compare results using criterion's built-in comparison

## 🚀 Future Work

See tracking issue for completing the remaining 7 benchmarks.

**Total estimated time to complete**: 5-6 hours

**Priority order**:
1. subproblem_solve.rs (finish fixing - 1 hour)
2. par_performance.rs (1 hour)
3. cut_selection.rs (1 hour)
4. state_operations.rs (1 hour)
5. correlation_application.rs (30 min)
6. lookup_structures.rs (30 min)
7. par_generator.rs (evaluate/remove)

---

**Conclusion**: Core benchmarking capability is functional. The 7 working benchmarks 
cover the most critical performance paths (SDDP algorithm, memory usage, simulation).
Remaining benchmarks can be fixed incrementally as needed.
