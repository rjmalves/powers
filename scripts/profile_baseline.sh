#!/bin/bash
#
# Baseline Performance Profiling Script
# Establishes performance baseline before refactoring
#
# Usage: ./scripts/profile_baseline.sh [example_dir]
# Example: ./scripts/profile_baseline.sh examples/fourbus

set -e

EXAMPLE_DIR=${1:-"examples/05-large-scale-brazilian"}
OUTPUT_DIR="profiling_results/baseline_$(date +%Y%m%d_%H%M%S)"

echo "=================================="
echo "POWE.RS Baseline Profiling"
echo "=================================="
echo "Example: $EXAMPLE_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. Build release binary
echo "[1/6] Building release binary..."
cargo build --release --bin powers
echo "✓ Build complete"
echo ""

# 2. Run benchmarks and save baseline
echo "[2/6] Running benchmarks (this may take 5-10 minutes)..."
cargo bench --bench sddp_e2e -- --save-baseline before_refactoring 2>&1 | tee "$OUTPUT_DIR/benchmark_output.txt"
echo "✓ Benchmarks complete"
echo ""

# 3. Generate flamegraph (CPU profiling)
echo "[3/6] Generating flamegraph (CPU profiling)..."
if command -v flamegraph &> /dev/null; then
    cargo flamegraph --bin powers -o "$OUTPUT_DIR/flamegraph.svg" -- "$EXAMPLE_DIR" 2>&1 | tee "$OUTPUT_DIR/flamegraph_output.txt"
    echo "✓ Flamegraph saved to $OUTPUT_DIR/flamegraph.svg"
else
    echo "⚠ flamegraph not installed. Install with: cargo install flamegraph"
    echo "  Skipping CPU profiling..."
fi
echo ""

# 4. Memory profiling with valgrind massif
echo "[4/6] Memory profiling (this may take 2-5 minutes)..."
if command -v valgrind &> /dev/null; then
    valgrind --tool=massif --massif-out-file="$OUTPUT_DIR/massif.out" \
        ./target/release/powers "$EXAMPLE_DIR" 2>&1 | tee "$OUTPUT_DIR/massif_output.txt"
    
    if command -v ms_print &> /dev/null; then
        ms_print "$OUTPUT_DIR/massif.out" > "$OUTPUT_DIR/massif_report.txt"
        echo "✓ Memory profile saved to $OUTPUT_DIR/massif_report.txt"
    else
        echo "✓ Memory profile saved to $OUTPUT_DIR/massif.out"
        echo "  (run 'ms_print massif.out' to view report)"
    fi
else
    echo "⚠ valgrind not installed. Install with: sudo apt install valgrind"
    echo "  Skipping memory profiling..."
fi
echo ""

# 5. Cache analysis with perf (Linux only)
echo "[5/6] Cache analysis (Linux only)..."
if command -v perf &> /dev/null; then
    echo "Running perf stat..."
    perf stat -e cache-misses,cache-references,L1-dcache-load-misses,instructions,cycles \
        ./target/release/powers "$EXAMPLE_DIR" 2>&1 | tee "$OUTPUT_DIR/perf_stat.txt"
    
    echo ""
    echo "Recording perf data..."
    perf record -F 999 --call-graph dwarf -o "$OUTPUT_DIR/perf.data" \
        ./target/release/powers "$EXAMPLE_DIR" 2>&1 | tee "$OUTPUT_DIR/perf_record.txt"
    
    echo "Generating perf report..."
    perf report -i "$OUTPUT_DIR/perf.data" --stdio > "$OUTPUT_DIR/perf_report.txt" 2>&1
    
    echo "✓ Perf analysis saved to $OUTPUT_DIR/"
else
    echo "⚠ perf not installed. Install with: sudo apt install linux-tools-generic"
    echo "  Skipping cache analysis..."
fi
echo ""

# 6. Quick timing test
echo "[6/6] Quick timing test (3 runs)..."
if command -v hyperfine &> /dev/null; then
    hyperfine --warmup 1 --runs 3 \
        "./target/release/powers $EXAMPLE_DIR" \
        --export-markdown "$OUTPUT_DIR/timing.md" \
        2>&1 | tee "$OUTPUT_DIR/timing.txt"
    echo "✓ Timing saved to $OUTPUT_DIR/timing.md"
else
    echo "Running manual timing..."
    for i in {1..3}; do
        echo "Run $i:"
        time ./target/release/powers "$EXAMPLE_DIR" 2>&1 | tail -5
        echo ""
    done > "$OUTPUT_DIR/timing.txt"
    echo "✓ Timing saved to $OUTPUT_DIR/timing.txt"
fi
echo ""

# Generate summary report
echo "=================================="
echo "Generating summary report..."
echo "=================================="

cat > "$OUTPUT_DIR/SUMMARY.md" << 'EOF'
# Performance Profiling Summary

**Date**: $(date)
**Example**: $EXAMPLE_DIR
**Rust Version**: $(rustc --version)
**CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
**RAM**: $(free -h | grep Mem | awk '{print $2}')

## Files Generated

1. `benchmark_output.txt` - Criterion benchmark results
2. `flamegraph.svg` - CPU profiling visualization
3. `massif_report.txt` - Memory usage analysis
4. `perf_stat.txt` - Cache performance statistics
5. `perf_report.txt` - Detailed perf analysis
6. `timing.md` - Quick timing comparison

## Quick Analysis

### Top CPU Consumers (from flamegraph)
TODO: Open flamegraph.svg and list top 5 functions

### Memory Usage (from massif)
EOF

if [ -f "$OUTPUT_DIR/massif_report.txt" ]; then
    echo "**Peak Memory**: $(grep -A1 "Peak" "$OUTPUT_DIR/massif_report.txt" | tail -1)" >> "$OUTPUT_DIR/SUMMARY.md"
fi

cat >> "$OUTPUT_DIR/SUMMARY.md" << 'EOF'

### Cache Performance (from perf)
EOF

if [ -f "$OUTPUT_DIR/perf_stat.txt" ]; then
    grep -E "cache-misses|cache-references|instructions|cycles" "$OUTPUT_DIR/perf_stat.txt" >> "$OUTPUT_DIR/SUMMARY.md" || true
fi

cat >> "$OUTPUT_DIR/SUMMARY.md" << 'EOF'

## Next Steps

1. Review flamegraph.svg to identify CPU hotspots
2. Review massif_report.txt to identify allocation hotspots
3. Review perf_report.txt for cache analysis
4. Document top 5 bottlenecks in PROFILING_RESULTS.md
5. Prioritize optimizations based on data

## Benchmark Baseline

Benchmarks saved to: `target/criterion/before_refactoring/`

To compare after changes:
```bash
cargo bench --baseline before_refactoring
```
EOF

echo "✓ Summary report saved to $OUTPUT_DIR/SUMMARY.md"
echo ""

# Final summary
echo "=================================="
echo "Profiling Complete!"
echo "=================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Quick review:"
echo "  1. View flamegraph: firefox $OUTPUT_DIR/flamegraph.svg"
echo "  2. View summary: cat $OUTPUT_DIR/SUMMARY.md"
echo "  3. View memory: cat $OUTPUT_DIR/massif_report.txt | head -100"
echo ""
echo "Next steps:"
echo "  1. Review the profiling data"
echo "  2. Document findings in PROFILING_RESULTS.md"
echo "  3. Update PERFORMANCE_REFACTORING_PLAN.md with actual bottlenecks"
echo "  4. Start Phase 1 optimizations"
echo ""
echo "Happy optimizing! 🚀🔥"
