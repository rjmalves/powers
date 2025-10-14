#!/usr/bin/env bash
#
# Memory comparison script for simulation benchmarks
# 
# This script runs simulation memory benchmarks and generates a comparison report
# validating the 96% memory reduction from SIM-OPT-005 and SIM-OPT-006.
#
# Usage:
#   ./scripts/compare_simulation_memory.sh
#
# Environment variables:
#   BASELINE_TAG - Git tag/branch to compare against (default: none, current only)
#   OUTPUT_DIR   - Directory for reports (default: target/memory-reports)
#

set -euo pipefail

# Configuration
BASELINE_TAG="${BASELINE_TAG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-target/memory-reports}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/memory-report-${TIMESTAMP}.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Simulation Memory Benchmark Comparison"
echo "  SIM-OPT-005 + SIM-OPT-006 Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to run benchmarks and extract memory stats
run_benchmark() {
    local tag="$1"
    local output_file="$2"
    
    echo -e "${BLUE}Running benchmarks${NC} (tag: ${tag:-current})"
    
    if [ -n "$tag" ]; then
        git checkout "$tag" 2>&1 | grep -v "Already on" || true
        cargo build --release --bench simulation_memory 2>&1 | tail -5
    fi
    
    # Run benchmark with reduced sample size for speed
    cargo bench --bench simulation_memory -- --sample-size 10 2>&1 | tee "$output_file"
    
    if [ -n "$tag" ]; then
        git checkout - 2>&1 | grep -v "Already on" || true
    fi
}

# Function to extract memory statistics from benchmark output
extract_memory_stats() {
    local file="$1"
    local scenarios="$2"
    
    # Extract peak RSS for the given scenario count
    grep -A 10 "memory_peak/${scenarios}" "$file" | \
        grep "Peak RSS:" | \
        awk '{print $3}' | \
        head -1
}

# Function to calculate percentage
calc_percentage() {
    local value="$1"
    local total="$2"
    echo "scale=1; ($value / $total) * 100" | bc -l
}

# Run current benchmarks
echo -e "${YELLOW}▶ Running current implementation...${NC}"
CURRENT_OUTPUT="${OUTPUT_DIR}/current-${TIMESTAMP}.log"
run_benchmark "" "$CURRENT_OUTPUT"

# Run baseline benchmarks if requested
if [ -n "$BASELINE_TAG" ]; then
    echo ""
    echo -e "${YELLOW}▶ Running baseline implementation (${BASELINE_TAG})...${NC}"
    BASELINE_OUTPUT="${OUTPUT_DIR}/baseline-${TIMESTAMP}.log"
    run_benchmark "$BASELINE_TAG" "$BASELINE_OUTPUT"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Memory Report"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Generate report
{
    echo "Simulation Memory Benchmark Report"
    echo "Generated: $(date)"
    echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Git commit: $(git rev-parse --short HEAD)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Parse memory statistics from current run
    echo "Current Implementation (SIM-OPT-005 + SIM-OPT-006)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Extract stats for different scenario counts
    for scenarios in 100 500 1000 2000; do
        grep -A 8 "📊 Memory Report:" "$CURRENT_OUTPUT" | \
            grep -A 8 "Scenarios: $scenarios" | head -9
        echo ""
    done
    
    if [ -n "$BASELINE_TAG" ]; then
        echo ""
        echo "Baseline Implementation ($BASELINE_TAG)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        for scenarios in 100 500 1000 2000; do
            grep -A 8 "📊 Memory Report:" "$BASELINE_OUTPUT" | \
                grep -A 8 "Scenarios: $scenarios" | head -9
            echo ""
        done
        
        echo ""
        echo "Memory Reduction Analysis"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Scenario Count | Baseline (MB) | Current (MB) | Reduction (%)"
        echo "---------------|---------------|--------------|---------------"
        
        for scenarios in 100 500 1000 2000; do
            baseline_mem=$(extract_memory_stats "$BASELINE_OUTPUT" "$scenarios" || echo "N/A")
            current_mem=$(extract_memory_stats "$CURRENT_OUTPUT" "$scenarios" || echo "N/A")
            
            if [ "$baseline_mem" != "N/A" ] && [ "$current_mem" != "N/A" ]; then
                reduction=$(echo "scale=1; (($baseline_mem - $current_mem) / $baseline_mem) * 100" | bc -l)
                printf "%14s | %13s | %12s | %13s\n" \
                    "$scenarios" "$baseline_mem" "$current_mem" "$reduction%"
            else
                printf "%14s | %13s | %12s | %13s\n" \
                    "$scenarios" "$baseline_mem" "$current_mem" "N/A"
            fi
        done
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Expected Memory Model (120 stages, 8 threads):"
    echo "  Old: scenarios × 6 MB"
    echo "  New: 48 MB (handlers) + scenarios × 240 KB"
    echo ""
    echo "For 10,000 scenarios:"
    echo "  Old: 60,000 MB (60 GB)"
    echo "  New: 2,448 MB (2.45 GB)"
    echo "  Reduction: 96%"
    echo ""
    
} | tee "$REPORT_FILE"

echo ""
echo -e "${GREEN}✓ Report saved to: ${REPORT_FILE}${NC}"
echo ""

# Check for regressions
if [ -n "$BASELINE_TAG" ]; then
    echo "Checking for memory regressions..."
    
    # Extract 1000-scenario memory usage
    current_1000=$(extract_memory_stats "$CURRENT_OUTPUT" "1000" | sed 's/[^0-9.]//g' || echo "0")
    baseline_1000=$(extract_memory_stats "$BASELINE_OUTPUT" "1000" | sed 's/[^0-9.]//g' || echo "0")
    
    if [ "$current_1000" != "0" ] && [ "$baseline_1000" != "0" ]; then
        # Allow 10% tolerance
        threshold=$(echo "$baseline_1000 * 1.10" | bc -l)
        
        if (( $(echo "$current_1000 > $threshold" | bc -l) )); then
            echo -e "${RED}✗ REGRESSION DETECTED${NC}"
            echo "  Memory usage increased by >10% (baseline: ${baseline_1000} MB, current: ${current_1000} MB)"
            exit 1
        else
            echo -e "${GREEN}✓ No regression detected${NC}"
            reduction=$(echo "scale=1; (($baseline_1000 - $current_1000) / $baseline_1000) * 100" | bc -l)
            echo "  Memory reduction: ${reduction}%"
        fi
    else
        echo -e "${YELLOW}⚠ Could not extract memory stats for regression check${NC}"
    fi
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Benchmark comparison complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
