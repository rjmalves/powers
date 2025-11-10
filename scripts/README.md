# Profiling Scripts

Automated performance profiling and comparison tools for POWE.RS optimization.

## Quick Start

```bash
# 1. Establish baseline
./scripts/profile_baseline.sh examples/05-large-scale-brazilian

# 2. Make optimizations
vim src/...

# 3. Compare performance
./scripts/profile_compare.sh examples/05-large-scale-brazilian
```

---

## Scripts

### `profile_baseline.sh`

Establishes a complete performance baseline before making changes.

**Usage:**
```bash
./scripts/profile_baseline.sh [example_dir]
```

**What it does:**
- ✅ Builds release binary with debug symbols
- ✅ Runs Criterion benchmarks
- ✅ Times 3 executions (calculates average)
- ✅ CPU profiling with perf
- ✅ Generates flamegraph
- ✅ Memory profiling with valgrind
- ✅ Collects cache statistics

**Output:** `profiling_results/baseline_YYYYMMDD_HHMMSS/`

**Time:** ~5 minutes

---

### `profile_compare.sh` ⭐

Compares current performance against baseline. **Use this after every optimization!**

**Usage:**
```bash
./scripts/profile_compare.sh [example_dir] [baseline_dir]

# If baseline_dir omitted, uses latest baseline
./scripts/profile_compare.sh examples/05-large-scale-brazilian
```

**What it does:**
- ✅ Times current implementation (3 runs)
- ✅ Compares against baseline timing
- ✅ Calculates improvement percentage
- ✅ Profiles CPU and memory
- ✅ Generates comparison report with verdict
- ✅ Identifies performance regressions

**Output:** `profiling_results/comparison_YYYYMMDD_HHMMSS/`

**Verdict examples:**
```
✅ SUCCESS: 15% improvement!
⚠️ REGRESSION: -5% slower  
≈ NO CHANGE: 2%
```

**Time:** ~5 minutes

---

### `generate_flamegraph.sh`

Quick flamegraph generation without full profiling suite.

**Usage:**
```bash
./scripts/generate_flamegraph.sh [example_dir]
```

**What it does:**
- ✅ Records with perf (--call-graph dwarf)
- ✅ Generates flamegraph SVG
- ✅ Shows top 10 functions

**Output:** `flamegraph.svg`

**Time:** ~1 minute

**Note:** Use larger examples (05-large-scale-brazilian) for meaningful results. Small examples run too fast.

---

### `test_profiling.sh`

Validates that all profiling tools and infrastructure work correctly.

**Usage:**
```bash
./scripts/test_profiling.sh
```

**What it tests:**
- ✅ Rust toolchain installed
- ✅ perf, valgrind, inferno available
- ✅ Scripts are executable
- ✅ Debug symbols enabled
- ✅ Examples exist
- ✅ Quick smoke test

**When to run:**
- First time setup
- After system updates
- When troubleshooting

---

## Typical Workflow

### Initial Baseline

```bash
# Establish baseline (once)
./scripts/profile_baseline.sh examples/05-large-scale-brazilian

# Results in: profiling_results/baseline_YYYYMMDD_HHMMSS/
# Review: cat profiling_results/baseline_*/SUMMARY.md
# View flamegraph: firefox profiling_results/baseline_*/flamegraph.svg
```

### Optimization Loop

```bash
# 1. Identify bottleneck from baseline
cat profiling_results/baseline_*/perf_report.txt

# 2. Make optimization
vim src/fcf.rs  # e.g., Replace BTreeMap with HashMap

# 3. Rebuild
cargo build --release

# 4. Compare
./scripts/profile_compare.sh examples/05-large-scale-brazilian

# 5. Review report
cat profiling_results/comparison_*/COMPARISON_REPORT.md

# 6. Decision:
#    ✅ Improved >5%? → Document and continue
#    ⚠️ Regression? → Investigate or revert
#    ≈ No change? → Try different approach
```

### Update Baseline

After successful optimization phase:

```bash
# Establish new baseline
./scripts/profile_baseline.sh examples/05-large-scale-brazilian

# Future comparisons will use this as reference
```

---

## Requirements

### Required Tools

- **Rust/Cargo** - Build system
- **perf** - CPU profiling
- **inferno** - Flamegraph generation
- **bc** - Calculations

### Optional Tools

- **valgrind** - Memory profiling (recommended)
- **ms_print** - Massif report formatting

### Install Missing Tools

```bash
# Debian/Ubuntu
sudo apt install linux-tools-generic valgrind bc

# Rust tools
cargo install inferno
```

### System Configuration

**perf_event_paranoid:**
```bash
# Check current value
cat /proc/sys/kernel/perf_event_paranoid

# If > 1, you'll need sudo for perf
# Optional: Lower it (permanent)
echo "kernel.perf_event_paranoid=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## Examples to Use

### ✅ Recommended: Large Example

```bash
# Best for profiling (runs ~37 seconds)
./scripts/profile_baseline.sh examples/05-large-scale-brazilian
```

**Why:** Long enough to capture meaningful profiling data

### ⚠️ Not Recommended: Small Examples

```bash
# Too fast for profiling (runs ~20ms)
./scripts/profile_baseline.sh examples/01-deterministic
```

**Issue:** Completes too quickly, perf can't capture enough samples

---

## Output Structure

```
profiling_results/
├── baseline_20251109_212950/
│   ├── SUMMARY.md              # Quick overview
│   ├── timing.txt              # Average runtime
│   ├── perf_report.txt         # CPU hotspots (125KB)
│   ├── flamegraph.svg          # Visual CPU profile
│   ├── massif_report.txt       # Memory allocations (7.6MB)
│   ├── benchmark_output.txt    # Criterion results
│   └── run_*.log              # Execution logs
│
└── comparison_20251110_143052/
    ├── COMPARISON_REPORT.md    # Verdict and analysis
    ├── timing_raw.txt          # Current timings
    ├── perf_report.txt         # Current CPU profile
    ├── flamegraph.svg          # Current flamegraph
    └── massif_report.txt       # Current memory profile
```

---

## Interpreting Results

### Flamegraph

**Good signs:**
- HiGHS functions dominate (60%) - unavoidable solver work
- Flat, wide bars - well-parallelized
- Deep stacks in solver - actual computation

**Red flags:**
- `std::_Rb_tree_increment` - BTreeMap iteration overhead
- `malloc`/`_int_malloc` >5% - excessive allocations
- Your function names >10% - potential bottleneck

### Perf Report

```
Overhead  Symbol
   11.44%  HFactor::ftranU       ← HiGHS solver (expected)
    5.31%  _Rb_tree_increment    ← 🔴 Data structure overhead!
    3.85%  _int_malloc           ← 🔴 Allocations!
```

**Focus on:**
- Your code >3% CPU time
- std::library overhead >2%
- Allocation functions

### Comparison Verdict

```
✅ SUCCESS: 15% improvement!
```
→ Great! Document and continue.

```
⚠️ REGRESSION: -8% slower
```
→ Investigate: New bottleneck? Revert if no other benefits.

```
≈ NO CHANGE: 2%
```
→ Within noise. OK if code quality improved.

---

## Troubleshooting

### "Need sudo password"

Scripts will prompt for sudo when needed (perf_event_paranoid=2).

**Solution:** Enter password, or lower paranoid level permanently.

### "Example runs too fast"

Small examples complete too quickly for profiling.

**Solution:** Use `examples/05-large-scale-brazilian`

### "No stack traces captured"

perf.data exists but has no call graphs.

**Check:**
```bash
# Verify debug symbols
file target/release/powers | grep debug_info

# If missing, rebuild
cargo clean && cargo build --release
```

### "Lost too many samples"

I/O overload during recording.

**Solution:** Already handled by using --freq 99 Hz. If still occurs, try --freq 49.

---

## Further Documentation

- **`../PROFILING_GUIDE.md`** - Complete profiling workflow
- **`../PROFILING_ANALYSIS.md`** - Initial baseline analysis
- **`../FLAMEGRAPH_QUICKSTART.md`** - Flamegraph usage tips
- **`../PROFILING_INFRASTRUCTURE_REPORT.md`** - Setup audit

---

## Support

For issues or questions:
1. Run `./scripts/test_profiling.sh` to diagnose
2. Check `PROFILING_GUIDE.md` for detailed help
3. Review existing baseline in `profiling_results/`

---

**Ready to optimize! 🚀**
