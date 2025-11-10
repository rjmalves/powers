#!/bin/bash
#
# Simple Allocation Profiling (no external tools required)
# Uses timing and Rust's built-in capabilities
#
# Usage: ./scripts/profile_allocations_simple.sh [example_dir]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

EXAMPLE_DIR=${1:-"examples/03-multistage"}
OUTPUT_DIR="profiling_results/simple_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Simple Allocation Profiling (TICKET-006b)"
echo "=========================================="
echo -e "${BLUE}Example:${NC} $EXAMPLE_DIR"
echo -e "${BLUE}Output:${NC} $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Build
echo -e "${CYAN}[1/4]${NC} Building..."
cargo build --release --bin powers > "$OUTPUT_DIR/build.log" 2>&1
echo -e "${GREEN}✓${NC}"
echo ""

# Baseline timing (3 runs)
echo -e "${CYAN}[2/4]${NC} Timing baseline (3 runs)..."
TIMES=()
for i in {1..3}; do
    START=$(date +%s.%N)
    ./target/release/powers "$EXAMPLE_DIR" > "$OUTPUT_DIR/run_${i}.log" 2>&1
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    TIMES+=($TIME)
    echo "  Run $i: ${TIME}s"
done

TOTAL=0
for t in "${TIMES[@]}"; do
    TOTAL=$(echo "$TOTAL + $t" | bc)
done
AVG=$(echo "scale=3; $TOTAL / 3" | bc)
echo -e "${GREEN}✓ Average: ${AVG}s${NC}"
echo ""

# Memory usage with /usr/bin/time
echo -e "${CYAN}[3/4]${NC} Memory usage..."
if command -v /usr/bin/time > /dev/null 2>&1; then
    /usr/bin/time -v ./target/release/powers "$EXAMPLE_DIR" > "$OUTPUT_DIR/memory_run.log" 2>&1 || true
    
    # Extract metrics
    MAX_RSS=$(grep "Maximum resident set" "$OUTPUT_DIR/memory_run.log" | awk '{print $6}' || echo "unknown")
    PAGE_FAULTS=$(grep "Minor.*page faults" "$OUTPUT_DIR/memory_run.log" | awk '{print $1}' || echo "unknown")
    
    echo "  Max RSS: ${MAX_RSS} KB"
    echo "  Page faults: ${PAGE_FAULTS}"
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠ /usr/bin/time not available${NC}"
fi
echo ""

# Generate summary
echo -e "${CYAN}[4/4]${NC} Summary..."

cat > "$OUTPUT_DIR/summary.txt" << SUMMARY
ALLOCATION PROFILING BASELINE
==============================

Date: $(date)
Example: $EXAMPLE_DIR

Timing (3 runs):
  Run 1: ${TIMES[0]}s
  Run 2: ${TIMES[1]}s
  Run 3: ${TIMES[2]}s
  Average: ${AVG}s

Memory:
  Max RSS: ${MAX_RSS} KB
  Page faults: ${PAGE_FAULTS}

TICKET-006b Target:
  Current estimate: ~184,000 allocations/run
  Target: <100 allocations/run (99.9% reduction)
  Expected improvement: 10-15% faster backward pass

Next Steps:
  1. Implement TICKET-006b (thread-local buffers)
  2. Re-run this script
  3. Compare timing: should see 10-15% improvement
  4. Verify with: cargo test --lib

For detailed profiling (requires tools):
  ./scripts/profile_allocations.sh $EXAMPLE_DIR

SUMMARY

cat "$OUTPUT_DIR/summary.txt"
echo ""
echo -e "${BLUE}Results:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Baseline established! ✓${NC}"
