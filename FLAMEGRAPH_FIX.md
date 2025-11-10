# 🔥 Flamegraph Issue: "No Stack Counts Found"

## Problem Diagnosis

Your perf.data captured **46,870 samples** but with **NO STACK TRACES**:
```
powers   80485 13492.593185:   10101010 cpu-clock:upppH: 
                                                        ↑ Missing stack!
```

This is why flamegraph generation failed: no call stacks = no flamegraph.

---

## Root Cause

The issue is **call graph recording was not enabled** properly. The `cargo-flamegraph` tool likely didn't pass the right flags to `perf record`.

---

## Solutions (Try in Order)

### ✅ Solution 1: Use Explicit perf Record + Flamegraph (RECOMMENDED)

This gives you full control over perf options:

```bash
# Step 1: Build with debug info (already done - you have debug = true)
cargo build --release --bin powers

# Step 2: Record with DWARF call graphs (works reliably)
perf record \
  --call-graph dwarf \
  --freq 99 \
  --output perf.data \
  ./target/release/powers examples/05-large-scale-brazilian

# Step 3: Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# Or use inferno (Rust implementation):
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

**Install tools if needed:**
```bash
# FlameGraph scripts (Perl)
git clone https://github.com/brendangregg/FlameGraph
export PATH="$PWD/FlameGraph:$PATH"

# OR use inferno (Rust, faster)
cargo install inferno
```

---

### ✅ Solution 2: Fix cargo-flamegraph Invocation

Try forcing DWARF call graphs:

```bash
# Option A: Set environment variable
CARGO_PROFILE_RELEASE_DEBUG=true \
  cargo flamegraph \
  --freq 99 \
  --perfdata-output flamegraph.data \
  -- examples/05-large-scale-brazilian

# Option B: Use custom perf args (if supported)
cargo flamegraph \
  --freq 99 \
  --perfdata-output flamegraph.data \
  --perf-args="--call-graph dwarf" \
  -- examples/05-large-scale-brazilian
```

---

### ✅ Solution 3: Use Frame Pointers (Alternative)

DWARF call graphs can be slow. Frame pointers are faster but require rebuild:

```toml
# Add to Cargo.toml [profile.release]
[profile.release]
debug = true
debug-assertions = false
overflow-checks = false

# Enable frame pointers for better perf compatibility
[profile.release.build-override]
opt-level = 0
```

Then rebuild and use frame pointers:
```bash
cargo clean
cargo build --release --bin powers

# Record with frame pointers (fp)
perf record \
  --call-graph fp \
  --freq 99 \
  ./target/release/powers examples/05-large-scale-brazilian

# Generate flamegraph
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

---

### ✅ Solution 4: Reduce I/O Pressure

Your current issues ("lost 610 chunks") suggest I/O overload. To reduce:

```bash
# Option A: Lower sampling frequency (already at 99 Hz - good)
perf record --call-graph dwarf --freq 49 ...

# Option B: Use smaller buffer size but write more frequently
perf record --call-graph dwarf --freq 99 -m 512 ...

# Option C: Write to tmpfs (RAM disk) - MUCH faster
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk

perf record \
  --call-graph dwarf \
  --freq 99 \
  --output /mnt/ramdisk/perf.data \
  ./target/release/powers examples/05-large-scale-brazilian

# Copy out after recording
cp /mnt/ramdisk/perf.data ./perf.data
```

---

### ✅ Solution 5: Use Shorter Run

Profile a smaller problem to get usable flamegraph faster:

```bash
# Use a smaller example
perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/02-stochastic

# Or reduce iterations in config
jq '.training.num_iterations = 2' \
  examples/05-large-scale-brazilian/config.json > /tmp/config.json
cp /tmp/config.json examples/05-large-scale-brazilian/config.json

perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/05-large-scale-brazilian
```

---

## Complete Workflow (RECOMMENDED)

Here's the full, working approach:

```bash
#!/bin/bash
set -e

echo "🔥 Generating Flamegraph - Complete Workflow"

# 1. Install inferno if needed
if ! command -v inferno-collapse-perf &> /dev/null; then
    echo "Installing inferno..."
    cargo install inferno
fi

# 2. Build with debug symbols (already done)
echo "Building release binary with debug symbols..."
cargo build --release --bin powers

# 3. Create ramdisk for perf data (optional but recommended)
echo "Setting up ramdisk..."
sudo mkdir -p /mnt/ramdisk 2>/dev/null || true
sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk 2>/dev/null || true

# 4. Record with DWARF call graphs
echo "Recording with perf (this will take ~40 seconds)..."
sudo perf record \
  --call-graph dwarf \
  --freq 99 \
  --output /mnt/ramdisk/perf.data \
  ./target/release/powers examples/05-large-scale-brazilian

# 5. Copy perf data out
sudo cp /mnt/ramdisk/perf.data ./perf.data
sudo chown $USER:$USER ./perf.data

# 6. Generate flamegraph
echo "Generating flamegraph..."
perf script -i perf.data | \
  inferno-collapse-perf | \
  inferno-flamegraph > flamegraph.svg

echo "✅ Done! Open flamegraph.svg in browser"
echo "   file://$(pwd)/flamegraph.svg"

# 7. Cleanup
sudo umount /mnt/ramdisk 2>/dev/null || true
```

---

## Verification

After recording, verify you have stack traces:

```bash
# Should show function names and stack traces
perf script -i perf.data | head -50
```

**Good output (with stacks):**
```
powers  12345 [001] 123.456789:  cpu-clock:
    7f123abc4567 [unknown] (/lib/x86_64-linux-gnu/libc.so.6)
    55a1234abcde HFactor::ftranU+0x23 (/path/to/powers)
    55a1234bcd12 backward_pass+0x456 (/path/to/powers)
    55a1234def34 main+0x789 (/path/to/powers)
```

**Bad output (no stacks):**
```
powers  12345 123.456789:  cpu-clock:
                          ↑ Missing everything!
```

---

## Why This Happens

`cargo-flamegraph` is a convenience wrapper but sometimes doesn't pass the right flags to `perf record`. The most reliable approach is:

1. **Use `perf record --call-graph dwarf` directly**
2. **Use Rust tools (`inferno`) for processing**
3. **Write to ramdisk to avoid I/O issues**

---

## Expected Results

With proper call graphs, you should get:
- **Flamegraph shows**: HiGHS solver, backward_pass, cut operations
- **Sample loss**: <10% (acceptable)
- **File size**: 2-3 GB (normal for 40s recording with DWARF)

---

## Quick Test

Try this minimal test first:

```bash
# Quick test with small example (5 seconds)
perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/01-deterministic

# Generate flamegraph
perf script | inferno-collapse-perf | inferno-flamegraph > test.svg

# Check if it worked
ls -lh test.svg
# Should be > 1KB and contain function names
```

If this works, scale up to the larger example.

---

## Alternative: Use perf report Instead

If flamegraph continues to fail, `perf report` provides similar insights:

```bash
perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/05-large-scale-brazilian

# Interactive TUI (very useful!)
perf report -i perf.data

# Text report
perf report -i perf.data --stdio > perf_analysis.txt
```

The interactive `perf report` lets you:
- Navigate call chains
- See percentages per function
- Annotate assembly
- Filter by thread/CPU

---

## Summary

**Immediate Action:**
```bash
# Install inferno
cargo install inferno

# Record properly
perf record --call-graph dwarf --freq 99 \
  ./target/release/powers examples/05-large-scale-brazilian

# Generate flamegraph
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

This should give you a working flamegraph! 🔥
