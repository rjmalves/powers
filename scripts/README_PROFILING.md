# Profiling Scripts for Performance Optimization

This directory contains profiling scripts to measure and validate performance optimizations, particularly for TICKET-006b (nested allocation elimination).

## Quick Start

### Simple Profiling (No External Tools)

**Best for**: Quick baseline, CI/CD, systems without profiling tools

```bash
./scripts/profile_allocations_simple.sh examples/03-multistage
```

**Measures**:
- Runtime (3 runs, average)
- Peak memory usage (RSS)
- Page faults

**Output**: `profiling_results/simple_YYYYMMDD_HHMMSS/`

### Detailed Profiling (Requires Tools)

**Best for**: Detailed analysis, identifying allocation hotspots

```bash
./scripts/profile_allocations.sh examples/03-multistage
```

**Measures**:
- Runtime baseline
- Heap profiling (Massif)
- CPU profiling (perf)  
- Malloc overhead percentage
- Allocation tracking (DHAT)

**Requirements**:
```bash
# Install profiling tools (Ubuntu/Debian)
sudo apt-get install valgrind linux-tools-generic

# Allow perf without sudo
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

**Output**: `profiling_results/allocations_YYYYMMDD_HHMMSS/`

## TICKET-006b Workflow

### Before Implementation

Establish baseline:

```bash
# Quick baseline (always works)
./scripts/profile_allocations_simple.sh examples/03-multistage

# Detailed baseline (if tools available)
./scripts/profile_allocations.sh examples/03-multistage
```

**Save the output directory** for comparison.

### After Implementation

Re-run profiling:

```bash
# Quick comparison
./scripts/profile_allocations_simple.sh examples/03-multistage

# Compare timing
# Before: ~0.48s average
# After:  ~0.41s average (expected 10-15% improvement)
```

### Expected Results

| Metric | Before | After (Target) | Improvement |
|--------|--------|---------------|-------------|
| Runtime | ~0.48s | ~0.41s | 15% faster |
| Allocations | ~184,000 | ~100 | 99.9% reduction |
| Malloc overhead | 8-10% | <2% | 75% reduction |
| Max RSS | ~24 MB | ~24 MB | Same |

## Other Profiling Scripts

### profile_baseline.sh

Comprehensive baseline profiling for general performance work.

```bash
./scripts/profile_baseline.sh examples/05-large-scale-brazilian
```

Includes:
- Benchmarks (Criterion)
- CPU profiling (perf)
- Flamegraphs
- Memory profiling

### profile_compare.sh

Compare two profiling runs:

```bash
./scripts/profile_compare.sh \
    profiling_results/baseline_before \
    profiling_results/baseline_after
```

### test_profiling.sh

Verify profiling infrastructure works:

```bash
./scripts/test_profiling.sh
```

## Interpreting Results

### Malloc Overhead (from perf)

```bash
# View malloc symbols in perf report
grep -E "(malloc|free|realloc)" profiling_results/.../perf_report.txt | head -20
```

**Interpretation**:
- **<2%**: Excellent (minimal allocation overhead)
- **2-5%**: Good (acceptable for dynamic applications)
- **5-10%**: Moderate (optimization recommended)
- **>10%**: High (significant optimization potential)

### Allocation Count (from DHAT)

Upload `dhat.out` to: https://nnethercote.github.io/dh_view/dh_view.html

**Look for**:
- Total blocks allocated
- Allocation hotspots (stack traces)
- Short-lived allocations (candidates for buffer reuse)

### Memory Usage (from Massif)

```bash
ms_print profiling_results/.../massif.out | less
```

**Look for**:
- Peak memory usage
- Memory growth over time
- Allocation sites (stack traces)

## Example Session

```bash
# 1. Establish baseline
$ ./scripts/profile_allocations_simple.sh examples/03-multistage
Average: 0.478s
Max RSS: 23680 KB

# 2. Implement TICKET-006b
# ... code changes ...

# 3. Verify correctness
$ cargo test --lib
495 passed ✓

# 4. Re-measure
$ ./scripts/profile_allocations_simple.sh examples/03-multistage
Average: 0.406s (15% improvement ✓)
Max RSS: 23680 KB (unchanged ✓)

# 5. Detailed validation
$ ./scripts/profile_allocations.sh examples/03-multistage
Malloc overhead: ~2% (was ~9%, improved ✓)
```

## Troubleshooting

### "perf_event_paranoid = 2"

Allow perf without sudo:

```bash
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
# Or for this session only:
sudo sysctl -w kernel.perf_event_paranoid=1
```

### "Valgrind not installed"

Install profiling tools:

```bash
# Ubuntu/Debian
sudo apt-get install valgrind linux-tools-generic

# Fedora/RHEL
sudo dnf install valgrind perf

# macOS (limited perf support)
brew install valgrind
```

### "Debug symbols not found"

Ensure `Cargo.toml` has:

```toml
[profile.release]
debug = true  # Enables debug symbols in release builds
```

## Performance Optimization Checklist

- [ ] Establish baseline with profiling
- [ ] Identify bottleneck from profile data
- [ ] Implement optimization
- [ ] Verify correctness (cargo test --lib)
- [ ] Re-profile to measure improvement
- [ ] Document results with actual measurements
- [ ] Compare before/after metrics

## References

- **TICKET-006b**: `tickets/performance-memory-preallocation/TICKET-006b-nested-preallocation.md`
- **Strategy**: `MEMORY_OPTIMIZATION_STRATEGY.md`
- **Baseline Results**: `profiling_results/*/summary.txt`

---

**Remember**: Always profile before and after optimization to validate improvements with real data! 📊
