# PERF-003 Baseline Benchmark Results

## System Specifications

**Date**: 2025-11-02  
**Hostname**: rogerio  
**CPU**: 12th Gen Intel(R) Core(TM) i7-12700KF  
**CPU Cores**: 20  
**Memory**: 31 GiB  
**OS**: Linux 6.6.87.2-microsoft-standard-WSL2  
**Rust Version**: rustc 1.90.0 (1159e78c4 2025-09-14)

## Benchmark Results

### Subproblem Construction with HydroConstraintData

This benchmark measures the time to construct a Subproblem with preprocessed hydro_data structures (PERF-001 and PERF-002 implementations).

| Hydros | Mean Time | Std Dev | Notes |
|--------|-----------|---------|-------|
| 10     | 45.7 µs   | ±0.4 µs | Small system |
| 50     | 86.0 µs   | ±0.9 µs | Medium system (target for optimization) |
| 100    | 125.9 µs  | ±1.0 µs | Large system |

**Analysis**:
- Construction time scales linearly with number of hydros (~0.85 µs per hydro)
- HydroConstraintData preprocessing happens during construction
- This is a one-time cost per stage, not part of the hot path

### HydroConstraintData Access Pattern

This benchmark simulates the hot path access pattern that will be used in PERF-004 optimize_realize_uncertainties.

| Hydros | Mean Time | Std Dev | Notes |
|--------|-----------|---------|-------|
| 10     | 2.5 ns    | ±0.03 ns | Sequential access, excellent cache locality |
| 50     | 18.2 ns   | ±0.07 ns | Target system size |
| 100    | 46.3 ns   | ±0.26 ns | Larger system |

**Analysis**:
- Access time: ~0.46 ns per hydro (extremely fast, cache-friendly)
- Sequential iteration over hydro_data is highly optimized
- This confirms the O(1) access benefit of preprocessed data structure

## Baseline for PERF-004 Optimization

The current (pre-optimization) realize_uncertainties implementation uses:
1. `generate_precomputed_scenarios`: Iterates through uncertainty_models, creates temporary Vec
2. `update_observation_space_ar_constraints`: Updates LP constraints from scenarios

Expected current performance (from profiling report):
- **50 hydros**: ~120-150 µs per realize_uncertainties call

**PERF-004 Target** (after hot path optimization):
- **50 hydros**: ~40-60 µs per realize_uncertainties call
- **Speedup**: 2-3x
- **Method**: Direct constraint update using hydro_data (eliminate intermediate allocations)

**PERF-004 Status**: ✅ **IMPLEMENTED** (2025-11-02)
- Replaced two-step process with direct `update_ar_constraints_optimized()` method
- Zero heap allocations in hot path loop
- Direct iteration over preprocessed hydro_data
- All 292 tests pass
- Expected speedup: 2-3x in forward pass (measured in full SDDP context)

## Key Insights

1. **HydroConstraintData Structure (PERF-001)**: Successfully implemented, ~200 bytes per hydro
2. **Subproblem Refactoring (PERF-002)**: hydro_data vector provides sequential, cache-friendly access
3. **Access Pattern**: Confirmed to be extremely fast (~0.46 ns/hydro)
4. **Ready for PERF-004**: Infrastructure in place for hot path optimization

## Next Steps

- **PERF-004**: Implement optimized realize_uncertainties using direct hydro_data access
- **Expected Impact**: 2-3x speedup in realize_uncertainties (hot path)
- **Validation**: Re-run these benchmarks after PERF-004 to confirm speedup

## Running the Benchmarks

```bash
# Run all realize_uncertainties benchmarks
cargo bench --bench realize_uncertainties

# Run specific benchmark group
cargo bench --bench realize_uncertainties -- subproblem_construction
cargo bench --bench realize_uncertainties -- hydro_data_access

# Generate flamegraph (requires cargo-flamegraph)
cargo flamegraph --bench realize_uncertainties
```

## Benchmark Details

The benchmarks are located in `benches/realize_uncertainties.rs` and measure:

1. **subproblem_construction**: Time to construct a Subproblem with HydroConstraintData preprocessing
2. **hydro_data_access**: Time to sequentially access hydro_data fields (simulates hot path)

These benchmarks use the Criterion framework with:
- 100 samples per benchmark
- 3-second warmup period
- Statistical analysis of timing distribution
- Outlier detection

Results are saved in `target/criterion/` with HTML reports.
