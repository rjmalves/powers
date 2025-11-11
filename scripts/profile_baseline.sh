#!/bin/bash
#
# Baseline Performance Profiling Script
# Establishes performance baseline before refactoring
#
# Usage: ./scripts/profile_baseline.sh [example_dir]
# Example: ./scripts/profile_baseline.sh examples/05-large-scale-brazilian

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

EXAMPLE_DIR=${1:-"examples/05-large-scale-brazilian"}
OUTPUT_DIR="profiling_results/baseline_$(date +%Y%m%d_%H%M%S)"

echo "=================================="
echo "POWE.RS Baseline Profiling"
echo "=================================="
echo -e "${BLUE}Example:${NC} $EXAMPLE_DIR"
echo -e "${BLUE}Output:${NC} $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. Build release binary
echo "[1/7] Building release binary..."
cargo build --release --bin powers
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# # 2. Run benchmarks and save baseline
# echo "[2/7] Running benchmarks (this may take 5-10 minutes)..."
# cargo bench --bench sddp_e2e -- --save-baseline before_refactoring 2>&1 | tee "$OUTPUT_DIR/benchmark_output.txt"
# echo -e "${GREEN}✓ Benchmarks complete${NC}"
# echo ""

# 3. Quick timing test (for easy comparison)
echo "[3/7] Quick timing test (3 runs)..."
TIMES=()
for i in {1..3}; do
    echo "  Run $i/3..."
    START=$(date +%s.%N)
    ./target/release/powers "$EXAMPLE_DIR" > "$OUTPUT_DIR/run_${i}.log" 2>&1
    END=$(date +%s.%N)
    RUNTIME=$(echo "$END - $START" | bc)
    TIMES+=($RUNTIME)
    echo "    Time: ${RUNTIME}s"
done

# Calculate average
TOTAL=0
for t in "${TIMES[@]}"; do
    TOTAL=$(echo "$TOTAL + $t" | bc)
done
AVG_TIME=$(echo "scale=2; $TOTAL / ${#TIMES[@]}" | bc)
echo -e "${GREEN}✓ Average time: ${AVG_TIME}s${NC}"
echo "Average: ${AVG_TIME}s" > "$OUTPUT_DIR/timing.txt"
for i in {1..3}; do
    echo "Run $i: ${TIMES[$i-1]}s" >> "$OUTPUT_DIR/timing.txt"
done
echo ""

# # 4. CPU profiling with perf (direct, reliable)
# echo "[4/7] CPU profiling with perf..."

# # Check if we need sudo
# PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
# if [ "$PARANOID" -gt 1 ]; then
#     echo -e "${YELLOW}⚠️  perf_event_paranoid = $PARANOID (requires sudo)${NC}"
#     SUDO="sudo"
# else
#     SUDO=""
# fi

# if command -v perf &> /dev/null; then
#     echo "Recording with perf (--call-graph dwarf)..."
#     $SUDO perf record \
#         --call-graph dwarf \
#         --freq 99 \
#         --output "$OUTPUT_DIR/perf.data" \
#         ./target/release/powers "$EXAMPLE_DIR" \
#         > "$OUTPUT_DIR/perf_record.txt" 2>&1 || echo "Perf recording completed"
    
#     # Fix ownership if sudo was used
#     if [ -n "$SUDO" ]; then
#         $SUDO chown $USER:$USER "$OUTPUT_DIR/perf.data" 2>/dev/null || true
#     fi
    
#     echo "Generating perf report..."
#     perf report -i "$OUTPUT_DIR/perf.data" --stdio > "$OUTPUT_DIR/perf_report.txt" 2>&1
#     echo -e "${GREEN}✓ Perf profiling complete${NC}"
# else
#     echo -e "${YELLOW}⚠️  perf not installed${NC}"
#     echo "  Install with: sudo apt install linux-tools-generic"
# fi
# echo ""

# # 5. Generate flamegraph
# echo "[5/7] Generating flamegraph..."
# if [ -f "$OUTPUT_DIR/perf.data" ]; then
#     # Check for inferno
#     if ! command -v inferno-collapse-perf &> /dev/null; then
#         echo "Installing inferno..."
#         cargo install inferno
#     fi
    
#     if perf script -i "$OUTPUT_DIR/perf.data" 2>/dev/null | \
#        inferno-collapse-perf 2>/dev/null | \
#        inferno-flamegraph > "$OUTPUT_DIR/flamegraph.svg" 2>/dev/null; then
#         SIZE=$(du -h "$OUTPUT_DIR/flamegraph.svg" | cut -f1)
#         echo -e "${GREEN}✓ Flamegraph generated (${SIZE})${NC}"
#     else
#         echo -e "${YELLOW}⚠️  Flamegraph generation skipped (insufficient data)${NC}"
#         echo "  This can happen if the example runs too quickly."
#         echo "  Try a larger example or increase iterations in config.json"
#     fi
# else
#     echo -e "${YELLOW}⚠️  Skipping (no perf.data)${NC}"
# fi
# echo ""

# 6. Memory profiling with valgrind massif
echo "[6/7] Memory profiling (this may take 2-5 minutes)..."
if command -v valgrind &> /dev/null; then
    valgrind --tool=massif --massif-out-file="$OUTPUT_DIR/massif.out" \
        ./target/release/powers "$EXAMPLE_DIR" 2>&1 | tee "$OUTPUT_DIR/massif_output.txt"
    
    if command -v ms_print &> /dev/null; then
        ms_print "$OUTPUT_DIR/massif.out" > "$OUTPUT_DIR/massif_report.txt"
        PEAK_MEM=$(grep -A1 "peak" "$OUTPUT_DIR/massif_report.txt" | tail -1 | awk '{print $3}')
        echo -e "${GREEN}✓ Memory profiling complete${NC}"
        echo "  Peak memory: $PEAK_MEM"
    else
        echo -e "${GREEN}✓ Memory profile saved to $OUTPUT_DIR/massif.out${NC}"
        echo "  (run 'ms_print massif.out' to view report)"
    fi
else
    echo -e "${YELLOW}⚠️  valgrind not installed${NC}"
    echo "  Install with: sudo apt install valgrind"
fi
echo ""

# 7. Perf stat for cache statistics
echo "[7/7] Cache statistics..."
if command -v perf &> /dev/null; then
    echo "Running perf stat (cache analysis)..."
    $SUDO perf stat -e cpu-clock,task-clock,cycles,instructions \
        ./target/release/powers "$EXAMPLE_DIR" 2>&1 | tee "$OUTPUT_DIR/perf_stat.txt"
    echo -e "${GREEN}✓ Cache statistics collected${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping (perf not available)${NC}"
fi
echo ""

# Generate summary report
echo "=================================="
echo "Generating summary report..."
echo "=================================="

cat > "$OUTPUT_DIR/SUMMARY.md" << EOF
# Performance Profiling Baseline

**Date**: $(date)
**Example**: $EXAMPLE_DIR
**Rust Version**: $(rustc --version)
**CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs || echo "Unknown")
**RAM**: $(free -h | grep Mem | awk '{print $2}' || echo "Unknown")

---

## Performance Metrics

### Runtime
- **Average**: ${AVG_TIME}s (3 runs)
- **Individual runs**:
  - Run 1: ${TIMES[0]}s
  - Run 2: ${TIMES[1]}s
  - Run 3: ${TIMES[2]}s

### Memory
EOF

if [ -f "$OUTPUT_DIR/massif_report.txt" ]; then
    PEAK_MEM=$(grep -A1 "peak" "$OUTPUT_DIR/massif_report.txt" | tail -1 | awk '{print $3}' || echo "Unknown")
    echo "- **Peak Memory**: $PEAK_MEM" >> "$OUTPUT_DIR/SUMMARY.md"
fi

cat >> "$OUTPUT_DIR/SUMMARY.md" << EOF

---

## Top CPU Hotspots

EOF

if [ -f "$OUTPUT_DIR/perf_report.txt" ]; then
    echo "\`\`\`" >> "$OUTPUT_DIR/SUMMARY.md"
    head -40 "$OUTPUT_DIR/perf_report.txt" | tail -12 | head -10 >> "$OUTPUT_DIR/SUMMARY.md"
    echo "\`\`\`" >> "$OUTPUT_DIR/SUMMARY.md"
else
    echo "*Perf report not available*" >> "$OUTPUT_DIR/SUMMARY.md"
fi

cat >> "$OUTPUT_DIR/SUMMARY.md" << EOF

---

## Files Generated

1. \`timing.txt\` - Quick timing results
2. \`benchmark_output.txt\` - Criterion benchmark results
3. \`perf_report.txt\` - Detailed CPU profiling
4. \`flamegraph.svg\` - CPU profiling visualization
5. \`massif_report.txt\` - Memory allocation analysis
6. \`perf_stat.txt\` - Cache and cycle statistics
7. \`run_*.log\` - Full execution logs

---

## Next Steps

### 1. Review Results

\`\`\`bash
# View flamegraph
firefox $OUTPUT_DIR/flamegraph.svg

# Review perf report
less $OUTPUT_DIR/perf_report.txt

# Check memory allocations
less $OUTPUT_DIR/massif_report.txt | head -100
\`\`\`

### 2. Identify Bottlenecks

Look for:
- Functions consuming >5% CPU time
- Allocations in hot paths
- Growing data structures

### 3. Document Findings

Update \`PROFILING_ANALYSIS.md\` with:
- Top 5 bottlenecks identified
- Optimization opportunities
- Expected impact of changes

### 4. Make Optimizations

After making changes, compare with:
\`\`\`bash
./scripts/profile_compare.sh $EXAMPLE_DIR $OUTPUT_DIR
\`\`\`

---

## Benchmark Baseline

Criterion benchmarks saved to: \`target/criterion/before_refactoring/\`

To compare after changes:
\`\`\`bash
cargo bench -- --baseline before_refactoring
\`\`\`

---

## Reproduction

To establish a new baseline:
\`\`\`bash
./scripts/profile_baseline.sh $EXAMPLE_DIR
\`\`\`

EOF

echo -e "${GREEN}✓ Summary report saved to $OUTPUT_DIR/SUMMARY.md${NC}"
echo ""

# Final summary
echo "=================================="
echo "Baseline Profiling Complete!"
echo "=================================="
echo ""
echo -e "${BLUE}Results saved to:${NC} $OUTPUT_DIR/"
echo ""
echo -e "${GREEN}Performance Baseline:${NC}"
echo "  Runtime: ${AVG_TIME}s"
if [ -f "$OUTPUT_DIR/massif_report.txt" ]; then
    PEAK_MEM=$(grep -A1 "peak" "$OUTPUT_DIR/massif_report.txt" | tail -1 | awk '{print $3}' || echo "Unknown")
    echo "  Memory: $PEAK_MEM"
fi
echo ""
echo "Quick review:"
echo "  1. View summary: cat $OUTPUT_DIR/SUMMARY.md"
echo "  2. View flamegraph: firefox $OUTPUT_DIR/flamegraph.svg"
echo "  3. View perf report: less $OUTPUT_DIR/perf_report.txt"
echo "  4. View memory: less $OUTPUT_DIR/massif_report.txt | head -100"
echo ""
echo "Next steps:"
echo "  1. Review profiling data and identify bottlenecks"
echo "  2. Document findings in PROFILING_ANALYSIS.md"
echo "  3. Make optimizations"
echo "  4. Compare: ./scripts/profile_compare.sh $EXAMPLE_DIR $OUTPUT_DIR"
echo ""
echo "Happy optimizing! 🚀🔥"
