# 🔥 Quick Flamegraph Usage Guide

## TL;DR

```bash
# For meaningful flamegraphs, use examples that run >5 seconds
./scripts/generate_flamegraph.sh examples/05-large-scale-brazilian
```

## Common Issue: "Example runs too fast"

**Problem**: Small examples (like `01-deterministic`) complete in ~20ms, which is too fast for perf to capture meaningful samples.

**Solution**: Use longer-running examples:

### ✅ Good Examples for Profiling

```bash
# Large-scale example (~36 seconds)
./scripts/generate_flamegraph.sh examples/05-large-scale-brazilian

# Cascade example (~5-10 seconds)
./scripts/generate_flamegraph.sh examples/04-cascade

# Multi-stage example
./scripts/generate_flamegraph.sh examples/03-multistage
```

### ❌ Too Fast for Profiling

```bash
# These complete in milliseconds:
examples/01-deterministic  (~20ms)
examples/02-stochastic     (~50ms)
```

## Alternative: Increase Iterations

For smaller examples, increase iterations in `config.json`:

```bash
# Edit config
cd examples/02-stochastic
jq '.training.num_iterations = 100' config.json > config_temp.json
mv config_temp.json config.json

# Now profile
cd ../..
./scripts/generate_flamegraph.sh examples/02-stochastic
```

## Understanding the Output

### Successful Run

```
🔥 Generating Flamegraph
========================
Example: examples/05-large-scale-brazilian
Output: flamegraph.svg

⚠️  perf_event_paranoid = 2 (requires sudo for call graphs)
📦 Building release binary with debug symbols...
🎯 Recording with perf (--call-graph dwarf)...
   Using: sudo perf record

[ perf record: Captured and wrote 2.5 MB perf.data (45000 samples) ]
                                                      ^^^^^ Many samples!

🔍 Verifying stack traces were captured...
✅ Found 180000 stack trace lines  ← Good!

🔥 Generating flamegraph SVG...
✅ Flamegraph generated successfully!
   Size: 2.3M
```

### Failed Run (Too Fast)

```
[ perf record: Captured and wrote 0.020 MB perf.data (1 samples) ]
                                                      ^ Only 1 sample!

❌ ERROR: Insufficient profiling data captured!
   - perf.data size: 8192 bytes
   - Stack trace lines: 0

   ⚠️  LIKELY CAUSE: Example runs too fast for profiling!
```

## Troubleshooting

### Issue: "Need sudo password"

**Why**: `perf_event_paranoid=2` requires root for call graph recording.

**Solutions**:

1. **Use sudo** (current approach):
   ```bash
   ./scripts/generate_flamegraph.sh examples/05-large-scale-brazilian
   # Will prompt for password
   ```

2. **Lower paranoid level** (system-wide, requires root):
   ```bash
   # Temporary (until reboot)
   sudo sysctl -w kernel.perf_event_paranoid=1
   
   # Permanent (add to /etc/sysctl.conf)
   echo "kernel.perf_event_paranoid=1" | sudo tee -a /etc/sysctl.conf
   ```

3. **Allow perf for your user** (CAP_PERFMON):
   ```bash
   sudo setcap cap_perfmon,cap_sys_ptrace=ep /usr/bin/perf
   ```

### Issue: "No stack counts found" with large file

**Diagnosis**:
```bash
# Check if binary has debug symbols
file target/release/powers | grep debug_info

# Check perf data manually
sudo perf script -i perf.data | head -100
```

**Solution**: Rebuild with debug symbols:
```toml
# Cargo.toml should have:
[profile.release]
debug = true
```

Then:
```bash
cargo clean
cargo build --release
```

## Performance Validation Workflow

After making optimizations (like BTreeMap → HashMap):

```bash
# 1. Profile the OPTIMIZED code
./scripts/generate_flamegraph.sh examples/05-large-scale-brazilian

# 2. Open flamegraph
firefox flamegraph.svg

# 3. Look for changes:
#    - Is std::_Rb_tree_increment gone? ✅
#    - Is backward_pass faster? ✅
#    - What's the new bottleneck?

# 4. Compare with baseline
perf report -i perf.data --stdio > new_report.txt
diff profiling_results/baseline_*/perf_report.txt new_report.txt
```

## What to Look For in Flamegraph

### Expected for POWE.RS

- **HiGHS solver functions** (60% of time - unavoidable)
  - `HFactor::ftranU`, `HFactor::btranL`, etc.
  - This is normal - LP solving is the core work

- **Backward pass** (30-40% of time)
  - `backward_pass`, `solve_backward_step`
  - Cut management, state updates

- **Forward pass** (<5% of time)
  - Should be fast

### Red Flags (After HashMap optimization)

- ❌ `std::_Rb_tree_increment` - Should be GONE
- ❌ `malloc`/`free` in hot loops - Need pre-allocation
- ❌ `std::vector::_M_realloc_insert` - Growing vectors

### Good Signs

- ✅ Flat, wide bars at top (well-parallelized)
- ✅ Deep stacks in HiGHS solver (actual work)
- ✅ Minimal std library collection overhead

## Advanced: Differential Flamegraph

Compare before/after optimizations:

```bash
# Before optimization (use baseline)
cp profiling_results/baseline_*/perf.data perf_before.data

# After optimization
./scripts/generate_flamegraph.sh examples/05-large-scale-brazilian
cp perf.data perf_after.data

# Generate differential
perf script -i perf_before.data | inferno-collapse-perf > before.folded
perf script -i perf_after.data | inferno-collapse-perf > after.folded
inferno-diff-folded before.folded after.folded | inferno-flamegraph > diff.svg

# Red = slower, Blue = faster
firefox diff.svg
```

## Quick Commands Reference

```bash
# Basic usage
./scripts/generate_flamegraph.sh examples/05-large-scale-brazilian

# Alternative: Manual perf + flamegraph
sudo perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/05-large-scale-brazilian
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg

# Use perf report (interactive TUI)
sudo perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/05-large-scale-brazilian
sudo perf report

# Generate text report
sudo perf report --stdio > perf_report.txt
```

## Expected Results (After HashMap Optimization)

When profiling the optimized code with HashMap:

- **Total runtime**: 35s → 33-34s (5-7% faster)
- **Backward pass**: Noticeably faster in flamegraph
- **std::_Rb_tree_increment**: Should be ABSENT
- **Top function**: Should still be HiGHS solver (60%)

If you see different results, document them in `PERFORMANCE_OPTIMIZATION_ASSESSMENT.md`!
