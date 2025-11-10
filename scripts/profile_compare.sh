#!/bin/bash
#
# Performance Profiling Comparison Script
# Compares current performance against baseline
#
# Usage: ./scripts/profile_compare.sh [example_dir] [baseline_dir]
# Example: ./scripts/profile_compare.sh examples/05-large-scale-brazilian profiling_results/baseline_20251109_212950

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

EXAMPLE_DIR=${1:-"examples/05-large-scale-brazilian"}
BASELINE_DIR=${2:-""}

# Find latest baseline if not specified
if [ -z "$BASELINE_DIR" ]; then
    BASELINE_DIR=$(ls -td profiling_results/baseline_* 2>/dev/null | head -1)
    if [ -z "$BASELINE_DIR" ]; then
        echo -e "${RED}❌ No baseline found!${NC}"
        echo "Run: ./scripts/profile_baseline.sh first"
        exit 1
    fi
    echo -e "${BLUE}Using latest baseline: $BASELINE_DIR${NC}"
fi

OUTPUT_DIR="profiling_results/comparison_$(date +%Y%m%d_%H%M%S)"

echo "=================================="
echo "POWE.RS Performance Comparison"
echo "=================================="
echo -e "${BLUE}Example:${NC} $EXAMPLE_DIR"
echo -e "${BLUE}Baseline:${NC} $BASELINE_DIR"
echo -e "${BLUE}Output:${NC} $OUTPUT_DIR"
echo ""

# Verify baseline exists
if [ ! -d "$BASELINE_DIR" ]; then
    echo -e "${RED}❌ Baseline directory not found: $BASELINE_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "Step 1: Building"
echo "=================================="
cargo build --release --bin powers
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

echo "=================================="
echo "Step 2: Quick Timing Test"
echo "=================================="
echo "Running 3 timed executions..."

# Create timing script
cat > "$OUTPUT_DIR/timing_raw.txt" << EOF
# Timing comparison
# Example: $EXAMPLE_DIR
# Date: $(date)
# 
EOF

# Run 3 times and collect timing
TIMES=()
for i in {1..3}; do
    echo "Run $i/3..."
    START=$(date +%s.%N)
    ./target/release/powers "$EXAMPLE_DIR" > "$OUTPUT_DIR/run_${i}.log" 2>&1
    END=$(date +%s.%N)
    RUNTIME=$(echo "$END - $START" | bc)
    TIMES+=($RUNTIME)
    echo "  Time: ${RUNTIME}s"
    echo "Run $i: ${RUNTIME}s" >> "$OUTPUT_DIR/timing_raw.txt"
done

# Calculate average
TOTAL=0
for t in "${TIMES[@]}"; do
    TOTAL=$(echo "$TOTAL + $t" | bc)
done
AVG_TIME=$(echo "scale=2; $TOTAL / ${#TIMES[@]}" | bc)
echo ""
echo -e "${GREEN}Average runtime: ${AVG_TIME}s${NC}"
echo "Average: ${AVG_TIME}s" >> "$OUTPUT_DIR/timing_raw.txt"

# Extract baseline timing
BASELINE_TIME=$(grep "Average:" "$BASELINE_DIR/timing.txt" 2>/dev/null | awk '{print $2}' | sed 's/s//' || echo "0")
if [ "$BASELINE_TIME" == "0" ]; then
    # Try alternative format
    BASELINE_TIME=$(tail -1 "$BASELINE_DIR/timing.txt" 2>/dev/null | awk '{print $2}' | sed 's/s//' || echo "37.3")
fi

# Calculate improvement
if [ "$BASELINE_TIME" != "0" ]; then
    IMPROVEMENT=$(echo "scale=2; (($BASELINE_TIME - $AVG_TIME) / $BASELINE_TIME) * 100" | bc)
    SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $AVG_TIME" | bc)
    
    echo ""
    echo "Baseline time: ${BASELINE_TIME}s"
    
    if (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
        echo -e "${GREEN}⚡ Improvement: ${IMPROVEMENT}% faster (${SPEEDUP}x speedup)${NC}"
    elif (( $(echo "$IMPROVEMENT < -5" | bc -l) )); then
        echo -e "${RED}⚠️  Regression: ${IMPROVEMENT}% slower${NC}"
    else
        echo -e "${YELLOW}≈ No significant change (${IMPROVEMENT}%)${NC}"
    fi
fi
echo ""

echo "=================================="
echo "Step 3: CPU Profiling"
echo "=================================="

# Check if we need sudo
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [ "$PARANOID" -gt 1 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

echo "Recording CPU profile..."
$SUDO perf record \
    --call-graph dwarf \
    --freq 99 \
    --output "$OUTPUT_DIR/perf.data" \
    ./target/release/powers "$EXAMPLE_DIR" \
    > "$OUTPUT_DIR/perf_record.log" 2>&1 || echo "Perf record completed"

# Fix ownership if sudo was used
if [ -n "$SUDO" ]; then
    $SUDO chown $USER:$USER "$OUTPUT_DIR/perf.data" 2>/dev/null || true
fi

# Generate perf report
echo "Generating perf report..."
perf report -i "$OUTPUT_DIR/perf.data" --stdio > "$OUTPUT_DIR/perf_report.txt" 2>&1
echo -e "${GREEN}✓ Perf report saved${NC}"

# Compare top functions
echo ""
echo "Top 10 functions (current):"
head -40 "$OUTPUT_DIR/perf_report.txt" | tail -12 | head -10

echo ""
echo "Top 10 functions (baseline):"
if [ -f "$BASELINE_DIR/perf_report.txt" ]; then
    head -40 "$BASELINE_DIR/perf_report.txt" | tail -12 | head -10
fi
echo ""

echo "=================================="
echo "Step 4: Generating Flamegraph"
echo "=================================="

# Check for inferno
if ! command -v inferno-collapse-perf &> /dev/null; then
    echo "Installing inferno..."
    cargo install inferno
fi

echo "Generating flamegraph..."
if perf script -i "$OUTPUT_DIR/perf.data" 2>/dev/null | \
   inferno-collapse-perf 2>/dev/null | \
   inferno-flamegraph > "$OUTPUT_DIR/flamegraph.svg" 2>/dev/null; then
    SIZE=$(du -h "$OUTPUT_DIR/flamegraph.svg" | cut -f1)
    echo -e "${GREEN}✓ Flamegraph generated (${SIZE})${NC}"
else
    echo -e "${YELLOW}⚠️  Flamegraph generation skipped (insufficient data)${NC}"
fi
echo ""

echo "=================================="
echo "Step 5: Memory Profiling"
echo "=================================="

if command -v valgrind &> /dev/null; then
    echo "Running memory profiling (this may take a few minutes)..."
    valgrind --tool=massif \
        --massif-out-file="$OUTPUT_DIR/massif.out" \
        ./target/release/powers "$EXAMPLE_DIR" \
        > "$OUTPUT_DIR/massif_run.log" 2>&1
    
    if command -v ms_print &> /dev/null; then
        ms_print "$OUTPUT_DIR/massif.out" > "$OUTPUT_DIR/massif_report.txt"
        
        # Extract peak memory
        PEAK_MEM=$(grep -A1 "peak" "$OUTPUT_DIR/massif_report.txt" | tail -1 | awk '{print $3}')
        echo -e "${GREEN}✓ Memory profiling complete${NC}"
        echo "Peak memory: $PEAK_MEM"
        
        # Compare with baseline
        if [ -f "$BASELINE_DIR/massif_report.txt" ]; then
            BASELINE_PEAK=$(grep -A1 "peak" "$BASELINE_DIR/massif_report.txt" | tail -1 | awk '{print $3}')
            echo "Baseline peak: $BASELINE_PEAK"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Valgrind not installed, skipping memory profiling${NC}"
fi
echo ""

echo "=================================="
echo "Step 6: Analysis Report"
echo "=================================="

# Generate comparison report
cat > "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
# Performance Comparison Report

**Generated**: $(date)
**Example**: $EXAMPLE_DIR
**Baseline**: $BASELINE_DIR

---

## Summary

### Runtime Performance

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Average Runtime | ${BASELINE_TIME}s | ${AVG_TIME}s | ${IMPROVEMENT}% |
| Speedup | 1.00x | ${SPEEDUP}x | - |

EOF

# Add verdict
if (( $(echo "$IMPROVEMENT > 10" | bc -l) )); then
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
**Verdict**: ✅ **SIGNIFICANT IMPROVEMENT**

The optimization successfully improved performance by ${IMPROVEMENT}%.
This is a meaningful speedup for production workloads.

EOF
elif (( $(echo "$IMPROVEMENT > 3" | bc -l) )); then
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
**Verdict**: ✅ **MINOR IMPROVEMENT**

The optimization improved performance by ${IMPROVEMENT}%.
This is a positive result, though not dramatic.

EOF
elif (( $(echo "$IMPROVEMENT > -3" | bc -l) )); then
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
**Verdict**: ≈ **NO SIGNIFICANT CHANGE**

Performance changed by ${IMPROVEMENT}%, which is within measurement noise.
This is acceptable if code quality improved.

EOF
else
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
**Verdict**: ⚠️ **PERFORMANCE REGRESSION**

Performance decreased by ${IMPROVEMENT}%.
**Action required**: Investigate the regression or revert changes.

EOF
fi

cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF

---

## CPU Profiling

### Top 10 Hotspots (Current)

\`\`\`
$(head -40 "$OUTPUT_DIR/perf_report.txt" | tail -12 | head -10)
\`\`\`

### Top 10 Hotspots (Baseline)

\`\`\`
$(head -40 "$BASELINE_DIR/perf_report.txt" 2>/dev/null | tail -12 | head -10 || echo "Baseline perf report not found")
\`\`\`

### Analysis

EOF

# Compare specific functions
echo "Comparing key functions..." >> "$OUTPUT_DIR/COMPARISON_REPORT.md"

# Extract function times from both reports
if [ -f "$BASELINE_DIR/perf_report.txt" ]; then
    echo "" >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
    echo "| Function | Baseline | Current | Change |" >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
    echo "|----------|----------|---------|--------|" >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
    
    # Check for std::_Rb_tree_increment (should disappear if we optimized HashMap)
    BASELINE_RBTREE=$(grep "_Rb_tree_increment" "$BASELINE_DIR/perf_report.txt" 2>/dev/null | head -1 | awk '{print $1}' | sed 's/%//' || echo "0")
    CURRENT_RBTREE=$(grep "_Rb_tree_increment" "$OUTPUT_DIR/perf_report.txt" 2>/dev/null | head -1 | awk '{print $1}' | sed 's/%//' || echo "0")
    
    if [ "$BASELINE_RBTREE" != "0" ]; then
        RBTREE_CHANGE=$(echo "scale=2; $CURRENT_RBTREE - $BASELINE_RBTREE" | bc)
        echo "| std::_Rb_tree_increment | ${BASELINE_RBTREE}% | ${CURRENT_RBTREE}% | ${RBTREE_CHANGE}% |" >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
    fi
    
    # Check malloc (should decrease if we pre-allocated)
    BASELINE_MALLOC=$(grep "malloc" "$BASELINE_DIR/perf_report.txt" 2>/dev/null | head -1 | awk '{print $1}' | sed 's/%//' || echo "0")
    CURRENT_MALLOC=$(grep "malloc" "$OUTPUT_DIR/perf_report.txt" 2>/dev/null | head -1 | awk '{print $1}' | sed 's/%//' || echo "0")
    
    if [ "$BASELINE_MALLOC" != "0" ]; then
        MALLOC_CHANGE=$(echo "scale=2; $CURRENT_MALLOC - $BASELINE_MALLOC" | bc)
        echo "| malloc | ${BASELINE_MALLOC}% | ${CURRENT_MALLOC}% | ${MALLOC_CHANGE}% |" >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
    fi
fi

cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF

---

## Memory Usage

EOF

if [ -f "$OUTPUT_DIR/massif_report.txt" ]; then
    PEAK_MEM=$(grep -A1 "peak" "$OUTPUT_DIR/massif_report.txt" | tail -1 | awk '{print $3}' || echo "Unknown")
    BASELINE_PEAK=$(grep -A1 "peak" "$BASELINE_DIR/massif_report.txt" 2>/dev/null | tail -1 | awk '{print $3}' || echo "Unknown")
    
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
| Metric | Baseline | Current |
|--------|----------|---------|
| Peak Memory | $BASELINE_PEAK | $PEAK_MEM |

EOF
else
    echo "*Memory profiling not available*" >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
    echo "" >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
fi

cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF

---

## Files Generated

- \`perf_report.txt\` - Detailed CPU profiling
- \`flamegraph.svg\` - Visual CPU profile
- \`massif_report.txt\` - Memory allocation details
- \`timing_raw.txt\` - Raw timing data
- \`run_*.log\` - Execution logs

---

## Next Steps

EOF

if (( $(echo "$IMPROVEMENT > 5" | bc -l) )); then
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
1. ✅ **Success!** Document the optimization approach
2. Run full benchmark suite: \`cargo bench\`
3. Update baseline: \`./scripts/profile_baseline.sh\`
4. Continue to next optimization phase
EOF
elif (( $(echo "$IMPROVEMENT < -3" | bc -l) )); then
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
1. ⚠️ **Investigate regression**
2. Review changes with: \`git diff\`
3. Check flamegraph for new bottlenecks
4. Consider reverting if no other benefits
EOF
else
    cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF
1. Review perf report for optimization opportunities
2. Check if code quality/maintainability improved
3. Consider if this change is still worthwhile
4. Continue with next optimization
EOF
fi

cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << EOF

---

## Reproduction

To reproduce this comparison:

\`\`\`bash
# Run the same comparison
./scripts/profile_compare.sh $EXAMPLE_DIR $BASELINE_DIR

# Or compare current code against this result
./scripts/profile_compare.sh $EXAMPLE_DIR $OUTPUT_DIR
\`\`\`

EOF

echo -e "${GREEN}✓ Comparison report generated${NC}"
echo ""

echo "=================================="
echo "Comparison Complete!"
echo "=================================="
echo ""
echo -e "${BLUE}Results saved to: $OUTPUT_DIR/${NC}"
echo ""

# Display summary
if (( $(echo "$IMPROVEMENT > 10" | bc -l) )); then
    echo -e "${GREEN}✅ SUCCESS: ${IMPROVEMENT}% improvement!${NC}"
    echo "Great work! The optimization significantly improved performance."
elif (( $(echo "$IMPROVEMENT > 3" | bc -l) )); then
    echo -e "${GREEN}✅ IMPROVED: ${IMPROVEMENT}% faster${NC}"
    echo "The optimization showed positive results."
elif (( $(echo "$IMPROVEMENT > -3" | bc -l) )); then
    echo -e "${YELLOW}≈ NO CHANGE: ${IMPROVEMENT}%${NC}"
    echo "Performance remained roughly the same."
else
    echo -e "${RED}⚠️  REGRESSION: ${IMPROVEMENT}% slower${NC}"
    echo "Performance decreased. Investigation recommended."
fi

echo ""
echo "Quick review:"
echo "  1. View report: cat $OUTPUT_DIR/COMPARISON_REPORT.md"
echo "  2. View flamegraph: firefox $OUTPUT_DIR/flamegraph.svg"
echo "  3. Compare perf: diff $BASELINE_DIR/perf_report.txt $OUTPUT_DIR/perf_report.txt"
echo ""
echo "Done! 🚀"
