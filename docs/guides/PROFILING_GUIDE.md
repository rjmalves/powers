# 🔍 POWE.RS Profiling Guide

Complete guide for profiling and performance optimization workflow.

---

## Quick Start

### Establish Baseline

```bash
# Run full baseline profiling (takes ~5 minutes)
./scripts/profile_baseline.sh examples/05-large-scale-brazilian

# Results saved to: profiling_results/baseline_YYYYMMDD_HHMMSS/
```

### Make Optimizations

```bash
# Make your code changes
vim src/...

# Rebuild
cargo build --release
```

### Compare Performance

```bash
# Compare against latest baseline
./scripts/profile_compare.sh examples/05-large-scale-brazilian

# Results saved to: profiling_results/comparison_YYYYMMDD_HHMMSS/
```

---

## Detailed Workflow

### Phase 0: Establish Baseline

This is your "before" snapshot:

```bash
./scripts/profile_baseline.sh examples/05-large-scale-brazilian
```

**What it does:**
1. ✅ Builds release binary with debug symbols
2. ✅ Runs Criterion benchmarks (saves to `before_refactoring`)
3. ✅ Times 3 executions and calculates average
4. ✅ Records CPU profile with `perf`
5. ✅ Generates flamegraph visualization
6. ✅ Profiles memory allocations with `massif`
7. ✅ Collects cache statistics

**Output files:**
- `SUMMARY.md` - Quick overview
- `timing.txt` - Average runtime
- `perf_report.txt` - CPU hotspots
- `flamegraph.svg` - Visual CPU profile
- `massif_report.txt` - Memory allocations
- `benchmark_output.txt` - Criterion results

**Time**: ~5-8 minutes for large example

---

### Phase 1: Analyze Bottlenecks

#### Review Flamegraph

```bash
firefox profiling_results/baseline_*/flamegraph.svg
```

**Look for:**
- Wide bars = CPU hotspots
- Deep stacks = call overhead
- Your code vs library code
- std::map/BTreeMap = data structure overhead
- malloc/free = allocation overhead

**Red flags:**
- ❌ `std::_Rb_tree_increment` - BTreeMap iteration
- ❌ `malloc`/`_int_malloc` >5% - excessive allocation
- ❌ `memset` >2% - zeroing buffers repeatedly

#### Review Perf Report

```bash
less profiling_results/baseline_*/perf_report.txt
```

**Top section shows:**
```
Overhead  Command  Shared Object       Symbol
   11.44%  powers   libhighs.so        [.] HFactor::ftranU
    5.31%  powers   libstdc++.so.6     [.] std::_Rb_tree_increment
    3.85%  powers   libc-2.31.so       [.] _int_malloc
```

**What to do:**
1. Ignore HiGHS functions (solver overhead is unavoidable)
2. Focus on YOUR code and std library overhead
3. Note any function >3% CPU time

#### Review Memory Profile

```bash
less profiling_results/baseline_*/massif_report.txt | head -100
```

**Look for:**
- Peak memory usage
- Growing allocations
- Allocation sources (stack traces)

**Red flags:**
- Allocations in hot loops
- Growing collections (`Vec::push` without `reserve`)
- Repeated allocations of same size

---

### Phase 2: Make Optimizations

Based on analysis, implement optimizations:

#### Example: Replace BTreeMap with HashMap

```rust
// Before (BTreeMap iteration shows up in perf)
use std::collections::BTreeMap;
let mut cuts: BTreeMap<NodeId, Vec<Cut>> = BTreeMap::new();

// After (HashMap is faster for most cases)
use std::collections::HashMap;
let mut cuts: HashMap<NodeId, Vec<Cut>> = HashMap::new();
```

#### Example: Pre-allocate Buffers

```rust
// Before (allocates every iteration)
for scenario in scenarios {
    let temp = vec![0.0; size];  // 🔴 Allocation hotspot!
    // use temp
}

// After (reuse buffer)
let mut temp = vec![0.0; size];
for scenario in scenarios {
    temp.fill(0.0);  // ✅ No allocation
    // use temp
}
```

#### Example: Use with_capacity

```rust
// Before (grows incrementally)
let mut results = Vec::new();
for item in items {
    results.push(process(item));  // May reallocate multiple times
}

// After (pre-allocate)
let mut results = Vec::with_capacity(items.len());
for item in items {
    results.push(process(item));  // Never reallocates
}
```

---

### Phase 3: Compare Performance

After making changes:

```bash
# Rebuild
cargo build --release

# Compare against baseline
./scripts/profile_compare.sh examples/05-large-scale-brazilian
```

**What it does:**
1. ✅ Builds and times 3 runs
2. ✅ Compares average time vs baseline
3. ✅ Profiles CPU and memory
4. ✅ Generates comparison report
5. ✅ Shows verdict (improved/regression/no change)

**Interpreting results:**

```
✅ SUCCESS: 15% improvement!
```
→ **Great!** Your optimization worked. Document and continue.

```
⚠️ REGRESSION: -8% slower
```
→ **Investigate:** Check flamegraph for new bottlenecks, consider reverting.

```
≈ NO CHANGE: 2%
```
→ **Acceptable** if code quality improved, otherwise try different approach.

---

## Profiling Best Practices

### DO ✅

1. **Profile with realistic workloads**
   - Use `examples/05-large-scale-brazilian` (156 hydros)
   - Represents real production usage
   - Runs long enough for accurate profiling

2. **Measure before and after**
   - Never optimize without data
   - Always compare against baseline
   - Document improvements with numbers

3. **Focus on hot paths**
   - Target functions >5% CPU time
   - Optimize backward pass (biggest impact)
   - Ignore cold paths

4. **Validate correctness**
   ```bash
   cargo test  # All tests must pass!
   ```

5. **Document optimizations**
   ```rust
   // PERFORMANCE: Replaced BTreeMap with HashMap
   // Profiling showed 5.3% time in tree iteration.
   // HashMap reduced this to <0.1%
   // Benchmark: backward_pass improved 12% (4.2s → 3.7s)
   ```

### DON'T ❌

1. **Don't optimize without profiling**
   - Guessing is wrong 90% of the time
   - Profile first!

2. **Don't micro-optimize cold paths**
   - Focus on functions >3% CPU time
   - Ignore setup/initialization code

3. **Don't sacrifice correctness**
   - All tests must pass
   - Verify numerical results match

4. **Don't ignore regressions**
   - >5% slower? Investigate immediately
   - May indicate new bottleneck

5. **Don't optimize HiGHS code**
   - Solver takes 60% of time (unavoidable)
   - Focus on OUR code (cut management, state handling)

---

## Tools Reference

### Installed Tools

You should have these installed:

```bash
# Check what's installed
perf --version           # CPU profiling
valgrind --version       # Memory profiling
cargo --version          # Rust toolchain

# Install missing tools
cargo install inferno    # Flamegraph generation
sudo apt install linux-tools-generic  # perf
sudo apt install valgrind            # massif
```

### Script Reference

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `profile_baseline.sh` | Establish performance baseline | ~5 min | `baseline_*/` |
| `profile_compare.sh` | Compare against baseline | ~5 min | `comparison_*/` |
| `generate_flamegraph.sh` | Quick flamegraph only | ~1 min | `flamegraph.svg` |

### Manual Profiling

If scripts don't work:

#### CPU Profiling
```bash
# Record
sudo perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/05-large-scale-brazilian

# Generate report
perf report --stdio > perf_report.txt

# Generate flamegraph
perf script | inferno-collapse-perf | inferno-flamegraph > flame.svg
```

#### Memory Profiling
```bash
# Record
valgrind --tool=massif --massif-out-file=massif.out \
  ./target/release/powers examples/05-large-scale-brazilian

# Generate report
ms_print massif.out > massif_report.txt
```

#### Quick Timing
```bash
# Single run
time ./target/release/powers examples/05-large-scale-brazilian

# Multiple runs
for i in {1..3}; do
  /usr/bin/time -f "Time: %e seconds" \
    ./target/release/powers examples/05-large-scale-brazilian
done
```

---

## Troubleshooting

### "Need sudo password"

**Issue:** `perf record` requires sudo due to `perf_event_paranoid=2`

**Solution 1:** Just use sudo (current approach)
```bash
sudo perf record ...
```

**Solution 2:** Lower paranoid level
```bash
# Temporary (until reboot)
sudo sysctl -w kernel.perf_event_paranoid=1

# Permanent
echo "kernel.perf_event_paranoid=1" | sudo tee -a /etc/sysctl.conf
```

### "Example runs too fast"

**Issue:** Small examples (<5s) don't capture enough samples

**Solution:** Use larger example
```bash
./scripts/profile_baseline.sh examples/05-large-scale-brazilian
# Not: examples/01-deterministic (too fast!)
```

### "No stack traces in perf.data"

**Issue:** Call graphs not recorded properly

**Solution:** Use DWARF explicitly
```bash
perf record --call-graph dwarf --freq 99 ...
# Not: perf record -g (may use frame pointers)
```

### "Lost too many samples"

**Issue:** I/O overload during recording

**Solution 1:** Lower frequency
```bash
perf record --freq 49 ...  # Instead of 99
```

**Solution 2:** Use ramdisk
```bash
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk
perf record --output /mnt/ramdisk/perf.data ...
cp /mnt/ramdisk/perf.data .
```

### "Flamegraph is empty"

**Issue:** No stack traces captured

**Diagnosis:**
```bash
# Check perf.data
perf script -i perf.data | head -20

# Should see function names and addresses
# If not, rebuild with debug symbols
```

**Solution:**
```toml
# Verify Cargo.toml has:
[profile.release]
debug = true
```

Then:
```bash
cargo clean
cargo build --release
```

---

## Expected Results

### Baseline (Before Optimization)

**Example 05 (156 hydros, 8 iterations):**
- Runtime: ~37s
- Peak memory: ~2GB
- Top CPU: HiGHS (60%), backward_pass (30%)
- Bottlenecks: BTreeMap iteration (5.3%), malloc (6%)

### After Phase 1 Optimizations

**Target improvements:**
- Runtime: 26-29s (25-30% faster)
- Peak memory: ~1.6GB (20% reduction)
- Top CPU: HiGHS (70%), backward_pass (20%)
- Bottlenecks: Solver overhead only

### Performance Goals

| Metric | Baseline | Phase 1 Goal | Phase 2 Goal |
|--------|----------|--------------|--------------|
| Total time | 37.3s | <29s (-25%) | <26s (-30%) |
| Backward/iter | 4.7s | <3.5s (-25%) | <3.0s (-36%) |
| Memory peak | 2.0GB | <1.6GB (-20%) | <1.4GB (-30%) |
| Allocations | High | Medium | Low |

---

## Integration with Development

### Before Committing

```bash
# 1. Run tests
cargo test

# 2. Check formatting
cargo fmt --check

# 3. Run clippy
cargo clippy

# 4. Quick performance check
./scripts/profile_compare.sh examples/05-large-scale-brazilian

# 5. If >5% regression, investigate before committing
```

### Pull Request Checklist

When submitting performance optimizations:

- [ ] Profiling data shows bottleneck
- [ ] Baseline established
- [ ] Comparison shows improvement (>3%)
- [ ] All tests pass
- [ ] Code is documented (explain trade-offs)
- [ ] No correctness regressions

---

## Common Optimization Patterns

### Pattern 1: Data Structure Choice

**Problem:** BTreeMap iteration in hot path

**Solution:** Use HashMap or Vec with binary search
```rust
// Before: BTreeMap (ordered, slower iteration)
let cuts: BTreeMap<NodeId, Vec<Cut>> = ...;

// After: HashMap (faster lookup/iteration)
let cuts: HashMap<NodeId, Vec<Cut>> = ...;

// Or: Flat Vec (best for sequential access)
let cuts: Vec<Cut> = ...;
let node_ranges: Vec<Range<usize>> = ...;
```

### Pattern 2: Pre-allocation

**Problem:** Growing vectors in loops

**Solution:** Pre-allocate with capacity
```rust
// Before
let mut results = Vec::new();  // Grows incrementally

// After
let mut results = Vec::with_capacity(n);  // Pre-allocated
```

### Pattern 3: Buffer Reuse

**Problem:** Allocating temporary buffers repeatedly

**Solution:** Reuse buffers across iterations
```rust
// Before
for _ in 0..1000 {
    let temp = vec![0.0; 1000];  // 1000 allocations!
}

// After (struct field)
struct Solver {
    temp_buffer: Vec<f64>,  // Allocated once
}

impl Solver {
    fn solve(&mut self) {
        self.temp_buffer.fill(0.0);  // Reuse
    }
}
```

### Pattern 4: Parallel Execution

**Problem:** Sequential processing of independent tasks

**Solution:** Use Rayon
```rust
// Before
let results: Vec<_> = scenarios.iter()
    .map(|s| expensive(s))
    .collect();

// After (if expensive() is independent)
use rayon::prelude::*;
let results: Vec<_> = scenarios.par_iter()
    .map(|s| expensive(s))
    .collect();
```

---

## Next Steps

1. **Establish baseline**
   ```bash
   ./scripts/profile_baseline.sh examples/05-large-scale-brazilian
   ```

2. **Review results**
   - Open flamegraph in browser
   - Identify top 5 bottlenecks
   - Document in PROFILING_ANALYSIS.md

3. **Plan optimizations**
   - Prioritize by impact (% CPU time)
   - Start with highest impact items
   - Follow PERFORMANCE_REFACTORING_PLAN.md

4. **Implement and validate**
   - Make focused changes
   - Run comparison after each change
   - Document results

5. **Repeat**
   - Each successful optimization becomes new baseline
   - Continue until performance goals met

---

## Success Criteria

You're done when:
- ✅ Runtime improved by >25% (37s → <29s)
- ✅ Memory usage reduced by >20% (2GB → <1.6GB)
- ✅ No data structure overhead in perf report
- ✅ Allocation overhead <2% CPU time
- ✅ All tests pass
- ✅ Code is documented and maintainable

---

**Remember:** Profile. Optimize. Validate. Repeat. 🚀
