# Performance Profiling Baseline

**Date**: Mon Nov 10 09:18:13 -03 2025
**Example**: examples/05-large-scale-brazilian
**Rust Version**: rustc 1.90.0 (1159e78c4 2025-09-14)
**CPU**: 12th Gen Intel(R) Core(TM) i7-12700KF
**RAM**: 31Gi

---

## Performance Metrics

### Runtime
- **Average**: 34.02s (3 runs)
- **Individual runs**:
  - Run 1: 33.892153555s
  - Run 2: 34.369745831s
  - Run 3: 33.799237116s

### Memory
- **Peak Memory**: 

---

## Top CPU Hotspots

```
     1.78%     1.78%  powers   libc.so.6            [.] __memset_avx2_unaligned_erms
     1.76%     1.76%  powers   powers               [.] HighsSparseMatrix::priceByRowWithSwitch(bool, HVectorBase<double>&, HVectorBase<double> const&, double, int, double, int) const
     1.67%     1.67%  powers   powers               [.] HEkkDualRow::choosePossible()
     1.41%     0.00%  powers   [kernel.kallsyms]    [k] 0xffffffffab200134
            |
            ---0xffffffffab200134
               |          
                --1.36%--0xffffffffab0785aa

     1.36%     0.00%  powers   [kernel.kallsyms]    [k] 0xffffffffab0785aa
```

---

## Files Generated

1. `timing.txt` - Quick timing results
2. `benchmark_output.txt` - Criterion benchmark results
3. `perf_report.txt` - Detailed CPU profiling
4. `flamegraph.svg` - CPU profiling visualization
5. `massif_report.txt` - Memory allocation analysis
6. `perf_stat.txt` - Cache and cycle statistics
7. `run_*.log` - Full execution logs

---

## Next Steps

### 1. Review Results

```bash
# View flamegraph
firefox profiling_results/baseline_20251110_080611/flamegraph.svg

# Review perf report
less profiling_results/baseline_20251110_080611/perf_report.txt

# Check memory allocations
less profiling_results/baseline_20251110_080611/massif_report.txt | head -100
```

### 2. Identify Bottlenecks

Look for:
- Functions consuming >5% CPU time
- Allocations in hot paths
- Growing data structures

### 3. Document Findings

Update `PROFILING_ANALYSIS.md` with:
- Top 5 bottlenecks identified
- Optimization opportunities
- Expected impact of changes

### 4. Make Optimizations

After making changes, compare with:
```bash
./scripts/profile_compare.sh examples/05-large-scale-brazilian profiling_results/baseline_20251110_080611
```

---

## Benchmark Baseline

Criterion benchmarks saved to: `target/criterion/before_refactoring/`

To compare after changes:
```bash
cargo bench -- --baseline before_refactoring
```

---

## Reproduction

To establish a new baseline:
```bash
./scripts/profile_baseline.sh examples/05-large-scale-brazilian
```

