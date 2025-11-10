#!/bin/bash
#
# Allocation Profiling Script for TICKET-006b
# Measures memory allocation behavior in backward pass
#
# This script profiles allocation overhead to establish baseline before
# TICKET-006b implementation (nested allocation elimination).
#
# Usage: ./scripts/profile_allocations.sh [example_dir]
# Example: ./scripts/profile_allocations.sh examples/03-multistage

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

EXAMPLE_DIR=${1:-"examples/03-multistage"}
OUTPUT_DIR="profiling_results/allocations_$(date +%Y%m%d_%H%M%S)"
BINARY="./target/release/powers"

echo "=========================================="
echo "POWE.RS Allocation Profiling (TICKET-006b)"
echo "=========================================="
echo -e "${BLUE}Target:${NC} Backward pass allocation baseline"
echo -e "${BLUE}Example:${NC} $EXAMPLE_DIR"
echo -e "${BLUE}Output:${NC} $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. Build release binary with debug symbols
echo -e "${CYAN}[1/6]${NC} Building release binary with debug symbols..."
cargo build --release --bin powers > "$OUTPUT_DIR/build.log" 2>&1
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}✗ Build failed${NC}"
    echo "Check $OUTPUT_DIR/build.log for details"
    exit 1
fi
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# 2. Quick baseline run
echo -e "${CYAN}[2/6]${NC} Baseline run (timing)..."
START=$(date +%s.%N)
$BINARY "$EXAMPLE_DIR" > "$OUTPUT_DIR/baseline_run.log" 2>&1
END=$(date +%s.%N)
BASELINE_TIME=$(echo "$END - $START" | bc)
echo -e "${GREEN}✓ Baseline time: ${BASELINE_TIME}s${NC}"
echo "Baseline runtime: ${BASELINE_TIME}s" > "$OUTPUT_DIR/summary.txt"
echo ""

# 3. Memory profiling with massif (allocation count)
echo -e "${CYAN}[3/6]${NC} Memory profiling with Valgrind Massif..."
echo "  (This may take 5-10x longer than baseline)"
if command -v valgrind > /dev/null 2>&1; then
    valgrind --tool=massif \
             --massif-out-file="$OUTPUT_DIR/massif.out" \
             --pages-as-heap=yes \
             --time-unit=i \
             $BINARY "$EXAMPLE_DIR" > "$OUTPUT_DIR/massif_run.log" 2>&1
    
    # Parse massif output
    if [ -f "$OUTPUT_DIR/massif.out" ]; then
        ms_print "$OUTPUT_DIR/massif.out" > "$OUTPUT_DIR/massif_report.txt"
        
        # Extract peak memory
        PEAK_MB=$(grep "peak" "$OUTPUT_DIR/massif_report.txt" | head -1 | awk '{print $6}' | tr -d ',' || echo "unknown")
        echo -e "${GREEN}✓ Peak memory: ${PEAK_MB} KB${NC}"
        echo "Peak memory: ${PEAK_MB} KB" >> "$OUTPUT_DIR/summary.txt"
    else
        echo -e "${YELLOW}⚠ Massif output not generated${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Valgrind not installed, skipping massif${NC}"
    echo "  Install: sudo apt-get install valgrind"
fi
echo ""

# 4. CPU profiling with perf (malloc overhead)
echo -e "${CYAN}[4/6]${NC} CPU profiling with perf (malloc overhead)..."

# Check perf availability
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [ "$PARANOID" -gt 1 ]; then
    echo -e "${YELLOW}⚠ perf_event_paranoid = $PARANOID${NC}"
    echo "  Run: echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
    echo "  Or use: sudo perf record ..."
    USE_SUDO="sudo"
else
    USE_SUDO=""
fi

if command -v perf > /dev/null 2>&1; then
    # Record with call graph
    $USE_SUDO perf record -F 99 -g --call-graph dwarf \
                          -o "$OUTPUT_DIR/perf.data" \
                          -- $BINARY "$EXAMPLE_DIR" > "$OUTPUT_DIR/perf_run.log" 2>&1
    
    # Generate report focusing on malloc/free
    $USE_SUDO perf report -i "$OUTPUT_DIR/perf.data" \
                          --stdio \
                          --no-children \
                          --sort symbol \
                          > "$OUTPUT_DIR/perf_report.txt" 2>&1
    
    # Extract malloc statistics
    echo "" >> "$OUTPUT_DIR/summary.txt"
    echo "=== Malloc/Free Overhead ===" >> "$OUTPUT_DIR/summary.txt"
    grep -E "(malloc|free|realloc|calloc)" "$OUTPUT_DIR/perf_report.txt" | head -20 >> "$OUTPUT_DIR/summary.txt" || echo "No malloc samples found" >> "$OUTPUT_DIR/summary.txt"
    
    # Calculate malloc overhead percentage
    MALLOC_PCT=$(grep -E "(malloc|free|realloc|calloc)" "$OUTPUT_DIR/perf_report.txt" | head -1 | awk '{print $1}' | tr -d '%' || echo "0")
    echo -e "${GREEN}✓ Malloc overhead: ~${MALLOC_PCT}%${NC}"
    echo "Malloc overhead: ~${MALLOC_PCT}%" >> "$OUTPUT_DIR/summary.txt"
else
    echo -e "${YELLOW}⚠ perf not installed, skipping CPU profiling${NC}"
    echo "  Install: sudo apt-get install linux-tools-generic"
fi
echo ""

# 5. DHAT allocation profiling (detailed allocation tracking)
echo -e "${CYAN}[5/6]${NC} Allocation tracking with DHAT..."
if command -v valgrind > /dev/null 2>&1; then
    valgrind --tool=dhat \
             --dhat-out-file="$OUTPUT_DIR/dhat.out" \
             $BINARY "$EXAMPLE_DIR" > "$OUTPUT_DIR/dhat_run.log" 2>&1
    
    if [ -f "$OUTPUT_DIR/dhat.out" ]; then
        # Parse DHAT output for allocation count
        TOTAL_BLOCKS=$(grep "total blocks" "$OUTPUT_DIR/dhat.out" | awk '{print $1}' || echo "unknown")
        TOTAL_BYTES=$(grep "total bytes" "$OUTPUT_DIR/dhat.out" | awk '{print $1}' || echo "unknown")
        
        echo -e "${GREEN}✓ Total allocations: ${TOTAL_BLOCKS}${NC}"
        echo -e "${GREEN}✓ Total bytes allocated: ${TOTAL_BYTES}${NC}"
        
        echo "" >> "$OUTPUT_DIR/summary.txt"
        echo "=== Allocation Statistics ===" >> "$OUTPUT_DIR/summary.txt"
        echo "Total allocations: ${TOTAL_BLOCKS}" >> "$OUTPUT_DIR/summary.txt"
        echo "Total bytes: ${TOTAL_BYTES}" >> "$OUTPUT_DIR/summary.txt"
    else
        echo -e "${YELLOW}⚠ DHAT output not generated${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Valgrind not installed, skipping DHAT${NC}"
fi
echo ""

# 6. Generate summary report
echo -e "${CYAN}[6/6]${NC} Generating summary report..."

cat > "$OUTPUT_DIR/README.md" << 'REPORT_EOF'
# Allocation Profiling Results

## Baseline Metrics (Pre-TICKET-006b)

This profiling establishes baseline allocation behavior before implementing
nested allocation elimination (TICKET-006b).

### Files Generated

- `summary.txt` - Quick metrics summary
- `baseline_run.log` - Normal execution output
- `massif.out` - Massif heap profiling data
- `massif_report.txt` - Human-readable massif analysis
- `perf.data` - CPU profiling data
- `perf_report.txt` - Malloc overhead analysis
- `dhat.out` - Detailed allocation tracking
- `dhat_run.log` - DHAT execution log

### Key Metrics to Track

1. **Baseline Runtime**: How long does execution take?
2. **Peak Memory**: Maximum heap usage during execution
3. **Malloc Overhead**: % of CPU time spent in malloc/free
4. **Allocation Count**: Total number of heap allocations
5. **Hot Allocation Sites**: Where are allocations happening?

### Expected TICKET-006b Impact

Based on code analysis (`src/state.rs:evaluate_cut()`):
- **Current**: ~184,000 allocations per training run
- **Target**: ~100 allocations per training run (99.9% reduction)
- **Expected malloc overhead**: 8-10% → <2%
- **Expected speedup**: 10-15% on backward pass

### Viewing Results

```bash
# Quick summary
cat summary.txt

# Detailed malloc overhead
less perf_report.txt
grep -E "(malloc|free)" perf_report.txt | head -20

# Memory usage over time
ms_print massif.out | less

# Allocation hotspots (requires DHAT viewer)
# Upload dhat.out to: https://nnethercote.github.io/dh_view/dh_view.html
```

### Comparing Before/After

After implementing TICKET-006b, run this script again and compare:

```bash
# Before
Malloc overhead: ~8-10%
Total allocations: ~184,000
Backward pass: X seconds

# After (expected)
Malloc overhead: ~1-2%
Total allocations: ~100
Backward pass: X * 0.85 seconds (15% faster)
```

REPORT_EOF

echo -e "${GREEN}✓ Summary generated${NC}"
echo ""

# Print final summary
echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo ""
cat "$OUTPUT_DIR/summary.txt"
echo ""
echo -e "${BLUE}Results saved to:${NC} $OUTPUT_DIR"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Review $OUTPUT_DIR/summary.txt"
echo "  2. Check malloc overhead in perf_report.txt"
echo "  3. Analyze allocation hotspots with DHAT viewer"
echo "  4. If malloc overhead is 8-10%, proceed with TICKET-006b"
echo "  5. After TICKET-006b, re-run this script to compare"
echo ""
echo -e "${GREEN}Baseline established! ✓${NC}"
