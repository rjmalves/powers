# Profiling Results - Baseline

**Date**: TBD  
**System**: [Document your CPU, RAM, OS]  
**Rust Version**: [Run `rustc --version`]  
**Example Used**: examples/fourbus/ (or specify)

---

## Executive Summary

**Top Bottleneck**: TBD (run profiling first)  
**Expected Improvement Potential**: TBD%  
**Primary Optimization Target**: TBD

---

## 1. CPU Profiling (Flamegraph)

### Top 5 CPU Consumers

| Rank | Function | % CPU Time | File | Category |
|------|----------|------------|------|----------|
| 1 | TBD | XX% | TBD | HOT PATH |
| 2 | TBD | XX% | TBD | HOT PATH |
| 3 | TBD | XX% | TBD | HOT PATH |
| 4 | TBD | XX% | TBD | WARM PATH |
| 5 | TBD | XX% | TBD | WARM PATH |

### Analysis

**What to look for in flamegraph**:
- Wide bars = time-consuming functions (optimize these)
- Tall stacks = deep call chains (consider inlining)
- Repeated patterns = allocations/cloning (use buffers)

**Findings**:
[After running flamegraph, document observations here]

Example:
- `solve_forward_step` takes 35% of CPU time
- Nested calls suggest function call overhead
- `realize_uncertainties` shows allocation pattern (saw Vec::push)

---

## 2. Memory Profiling (Massif)

### Memory Usage Statistics

- **Peak Memory**: TBD MB
- **Allocations per Iteration**: TBD
- **Memory Growth**: Constant / Growing / Spiky

### Top Allocation Sites

| Rank | Location | Size | Frequency | Fix Priority |
|------|----------|------|-----------|--------------|
| 1 | TBD | TBD MB | Per iteration | 🔴 HIGH |
| 2 | TBD | TBD MB | Per stage | 🟡 MEDIUM |
| 3 | TBD | TBD MB | Once | 🟢 LOW |

### Analysis

**What to look for**:
- Sawtooth pattern = allocate/free cycle (pre-allocate instead)
- Growing memory = leak or unbounded collection
- Large peak = big temporary allocations

**Findings**:
[After running massif, document observations]

---

## 3. Cache Performance (Perf)

### Cache Statistics

```
Performance counter stats:

  cache-references:          TBD
  cache-misses:              TBD (XX% of all cache refs)
  L1-dcache-load-misses:     TBD
  instructions:              TBD
  cycles:                    TBD
  
  IPC (instructions/cycle):  TBD
```

### Analysis

**Good Performance**:
- Cache miss rate: <5%
- IPC (instructions per cycle): >1.0

**Poor Performance**:
- Cache miss rate: >10% (investigate data layout)
- IPC: <0.5 (too many stalls)

**Findings**:
[Document cache analysis]

---

## 4. Benchmark Results

### Critical Path Benchmarks

| Benchmark | Mean Time | Std Dev | Throughput |
|-----------|-----------|---------|------------|
| `subproblem/solve_forward` | TBD µs | ±X% | TBD ops/sec |
| `cut_evaluation/1000_cuts` | TBD µs | ±X% | TBD ops/sec |
| `backward_pass/single_node` | TBD ms | ±X% | TBD ops/sec |
| `forward_pass/10_scenarios` | TBD ms | ±X% | TBD ops/sec |

### Baseline Saved

```bash
# Benchmarks saved to:
target/criterion/before_refactoring/

# To compare after optimization:
cargo bench --baseline before_refactoring
```

---

## 5. Bottleneck Prioritization

### 🔴 Critical (Phase 1 targets)

1. **[Function Name]** - XX% of time
   - Issue: [Allocations / Cache misses / Algorithm]
   - Expected gain: XX%
   - Effort: Low / Medium / High

2. **[Function Name]** - XX% of time
   - Issue: TBD
   - Expected gain: XX%
   - Effort: Low / Medium / High

### 🟡 Important (Phase 2 targets)

3. **[Function Name]** - XX% of time
   - Issue: TBD
   - Expected gain: XX%
   - Effort: Low / Medium / High

### 🟢 Nice-to-have (Later phases)

4. **[Function Name]** - X% of time
   - Issue: TBD
   - Expected gain: X%
   - Effort: Low / Medium / High

---

## 6. Optimization Opportunities

### Identified Patterns

**Excessive Allocations**:
- [ ] Location 1: [file:line] - Allocates in loop
- [ ] Location 2: TBD
- [ ] Location 3: TBD

**Cache Inefficiency**:
- [ ] Structure 1: Nested Vecs causing pointer chasing
- [ ] Structure 2: TBD

**Algorithm Issues**:
- [ ] Function 1: O(n²) could be O(n log n)
- [ ] Function 2: TBD

**Clone/Copy Overhead**:
- [ ] Hot path 1: Clones state every iteration
- [ ] Hot path 2: TBD

**Call Overhead**:
- [ ] Deep call chain: func1 → func2 → func3 → func4
- [ ] Small functions not inlined

---

## 7. Expected Improvements

Based on profiling data, estimated improvements from planned optimizations:

| Phase | Target | Expected Gain | Confidence |
|-------|--------|---------------|------------|
| Phase 1 | Pre-allocation | 20-30% | High |
| Phase 2 | Cache layout | 10-15% | Medium |
| Phase 3 | Remove clones | 5-10% | High |
| Phase 4 | Inlining | 3-8% | Medium |
| Phase 5 | Algorithms | 10-25% | Low-Medium |
| **Total** | **All phases** | **40-60%** | **Cumulative** |

**Note**: Actual results may vary. These are estimates based on typical optimization patterns.

---

## 8. Test System Specifications

Document your hardware for reproducibility:

- **CPU**: TBD (e.g., AMD Ryzen 9 5950X)
- **Cores**: TBD physical, TBD logical
- **RAM**: TBD GB DDR4-XXXX
- **OS**: TBD (e.g., Ubuntu 22.04 LTS)
- **Rust**: TBD (run `rustc --version`)
- **LLVM**: TBD
- **Compiler Flags**: `RUSTFLAGS="-C target-cpu=native"`

---

## 9. Profiling Commands Used

For reproducibility:

```bash
# CPU profiling
cargo flamegraph --bin powers -- examples/fourbus/

# Memory profiling
valgrind --tool=massif ./target/release/powers examples/fourbus/
ms_print massif.out

# Cache analysis
perf stat -e cache-misses,cache-references ./target/release/powers examples/fourbus/
perf record --call-graph dwarf ./target/release/powers examples/fourbus/
perf report

# Benchmarks
cargo bench --save-baseline before_refactoring
```

---

## 10. Next Actions

- [ ] Review flamegraph and identify top 5 CPU consumers
- [ ] Review massif report and identify allocation hotspots
- [ ] Review perf report for cache analysis
- [ ] Fill in TBD sections in this document
- [ ] Update PERFORMANCE_REFACTORING_PLAN.md with actual priorities
- [ ] Create Phase 1 optimization tasks
- [ ] Start with highest-impact, lowest-risk optimizations

---

## Appendix: Raw Data Files

All profiling data saved in: `profiling_results/baseline_YYYYMMDD_HHMMSS/`

- `flamegraph.svg` - CPU profiling visualization
- `massif_report.txt` - Memory usage report
- `perf_stat.txt` - Cache statistics
- `perf_report.txt` - Detailed perf analysis
- `benchmark_output.txt` - Criterion results
- `timing.md` - Quick timing comparison

---

**Status**: 📋 Template - Run `./scripts/profile_baseline.sh` to fill in  
**Next**: Profile → Document → Optimize → Measure → Repeat
