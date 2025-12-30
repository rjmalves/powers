# Performance Baseline

> **Captured**: 2025-12-29
> **Commit**: 5a823cd53c3ade20391076a46a721bf08f7038d6
> **Purpose**: Reference baseline for detecting performance regressions during refactoring

---

## System Information

| Property | Value |
|----------|-------|
| **CPU** | Intel Core Ultra 7 165U @ 2.69 GHz |
| **RAM** | 16 GB |
| **OS** | Linux (WSL2) 6.6.87.2-microsoft-standard-WSL2 |
| **Rust** | 1.89.0 (29483883e 2025-08-04) |
| **Profile** | release (optimized + debuginfo) |

---

## Criterion Benchmarks (sddp_e2e)

Baseline saved as: `before-refactoring`

### Single Iteration Benchmarks

| Benchmark | Mean | Notes |
|-----------|------|-------|
| `sddp_single_iteration/forward_passes/2` | 1.65 s | 2 forward passes, 1 iteration |
| `sddp_single_iteration/forward_passes/4` | 2.25 s | 4 forward passes, 1 iteration |
| `sddp_single_iteration/forward_passes/8` | 2.85 s | 8 forward passes, 1 iteration |

### Training Phases

| Benchmark | Mean | Notes |
|-----------|------|-------|
| `sddp_training_phases/3_iterations_10_forward` | 9.79 s | 3 iterations, 10 forward passes |

**Phase Breakdown** (156 hydros, 60 stages):
- Forward pass: ~350 ms per iteration
- Backward pass: ~7.5 s per iteration
- **Backward/Forward ratio**: ~21x

### Simulation

| Benchmark | Mean | Notes |
|-----------|------|-------|
| `sddp_simulation/32_scenarios` | 780 ms | 32 scenario simulation |

---

## Full Training Run (Example 05)

| Metric | Value |
|--------|-------|
| **Total wall time** | ~25 s |
| **Iterations** | 8 |
| **Forward passes** | 4 |
| **Stages** | 60 |
| **Hydro plants** | 156 |
| **Thermal plants** | 121 |

> **Note**: Example 05 was reduced from 16 forward passes to 4 forward passes (and 32 to 16 simulation scenarios) to enable faster iteration during development and CI testing.

---

## Memory Benchmarks

### Peak Memory Usage

| Example | Peak RSS | Notes |
|---------|----------|-------|
| `05-large-scale-brazilian` | **~1.5 GB** | Full 8-iteration training run (4 forward passes) |

### Memory Growth Pattern

See [MEMORY_GROWTH_ANALYSIS.md](../MEMORY_GROWTH_ANALYSIS.md) for detailed analysis.

> **Note**: The table below reflects the original 16 forward passes configuration. With 4 forward passes, memory growth is proportionally lower (~1/4 of the values shown).

| Iteration | Active Cuts | Memory (MB) | Delta (MB/iter) |
|-----------|-------------|-------------|-----------------|
| 1 | 236 | ~500 | baseline |
| 2 | 455 | ~570 | +70 |
| 3 | 639 | ~650 | +80 |
| 4 | 832 | ~760 | +110 |
| 5 | 949 | ~880 | +120 |
| 6 | 1067 | ~1010 | +130 |
| 7 | 1222 | ~1140 | +130 |
| 8 | 1411 | ~1260 | +120 |

**Average growth**: ~95 MB/iteration (with 4 forward passes)

### Key Allocation Hotspots

| Location | Issue | Impact |
|----------|-------|--------|
| `get_solution()` / `get_basis()` | New allocations per solve | ~4.9 GB churn |
| FCF cut pool | Unbounded growth | ~720 MB total |
| HiGHS internal | Solver allocations | ~1.6 GB |

---

## Reproducing Benchmarks

### Run Criterion Benchmarks

```bash
# Build release
cargo build --release

# Run benchmarks and save baseline
cargo bench --bench sddp_e2e -- --save-baseline before-refactoring

# Compare after changes
cargo bench --bench sddp_e2e -- --baseline before-refactoring
```

### Measure Memory Usage

```bash
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep "Maximum resident set size"
```

### Full Training Time

```bash
time ./target/release/powers run examples/05-large-scale-brazilian
```

---

## Regression Thresholds

A PR will be blocked if:

| Metric | Threshold |
|--------|-----------|
| Single iteration time | > 5% regression |
| Training phase time | > 5% regression |
| Simulation time | > 5% regression |
| Peak memory | > 10% increase |

---

## Notes

- Example 05 takes ~2 minutes for full training - plan CI timeouts accordingly
- Backward pass dominates (21x slower than forward) - optimization target
- Memory growth is primarily from allocation churn, not permanent leaks
- Criterion baseline data stored in `target/criterion/*/before-refactoring/`
