#!/bin/bash
#
# Example: Compare allocator RSS behavior
#
# This script demonstrates how to use the RSS monitoring tools to compare
# different allocators (glibc, mimalloc, jemalloc).
#

set -e

EXAMPLE="examples/05-large-scale-brazilian"
OUTPUT_DIR="rss_comparison_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=== Allocator RSS Comparison ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Build all variants
echo "Building variants..."
cd ..
cargo build -j1 --release 2>&1 | tail -3
cargo build -j1 --release --features mimalloc 2>&1 | tail -3
cargo build -j1 --release --features jemalloc 2>&1 | tail -3
echo "Build complete"
echo ""

cd "$OUTPUT_DIR"

# Test glibc
echo "=== Testing glibc (system allocator) ==="
../scripts/monitor_rss.py -i 0.5 -o glibc_rss.csv -- \
    ../target/release/powers run "../$EXAMPLE" --log-level debug
echo ""

# Test mimalloc
echo "=== Testing mimalloc ==="
../scripts/monitor_rss.py -i 0.5 -o mimalloc_rss.csv -- \
    ../target/release/powers run "../$EXAMPLE" --log-level debug
echo ""

# Test jemalloc (with safe -j1 build)
echo "=== Testing jemalloc ==="
../scripts/monitor_rss.py -i 0.5 -o jemalloc_rss.csv -- \
    ../target/release/powers run "../$EXAMPLE" --log-level debug
echo ""

# Generate individual plots (if matplotlib available)
if python3 -c "import matplotlib" 2>/dev/null; then
    echo "=== Generating plots ==="
    
    ../scripts/plot_rss.py glibc_rss.csv -o glibc_plot.png -t "glibc RSS Timeline"
    ../scripts/plot_rss.py mimalloc_rss.csv -o mimalloc_plot.png -t "mimalloc RSS Timeline"
    ../scripts/plot_rss.py jemalloc_rss.csv -o jemalloc_plot.png -t "jemalloc RSS Timeline"
    
    # Comparison plot
    ../scripts/plot_rss.py glibc_rss.csv -o comparison.png \
        --compare "mimalloc:mimalloc_rss.csv" "jemalloc:jemalloc_rss.csv"
    
    echo "Plots generated:"
    ls -lh *.png
else
    echo "matplotlib not installed, skipping plots"
fi

# Generate summary report
echo ""
echo "=== Summary Report ==="

for allocator in glibc mimalloc jemalloc; do
    csv_file="${allocator}_rss.csv"
    
    if [ -f "$csv_file" ]; then
        # Get first and last RSS values
        first_rss=$(head -2 "$csv_file" | tail -1 | cut -d',' -f3)
        last_rss=$(tail -1 "$csv_file" | cut -d',' -f3)
        
        # Count iterations
        json_file="${allocator}_rss.iterations.json"
        if [ -f "$json_file" ]; then
            iter_count=$(python3 -c "import json; d=json.load(open('$json_file')); print(len([e for e in d if e['phase']=='end']))")
        else
            iter_count="unknown"
        fi
        
        echo "$allocator:"
        echo "  Initial RSS: ${first_rss} MB"
        echo "  Final RSS: ${last_rss} MB"
        echo "  Iterations: $iter_count"
        echo ""
    fi
done

echo "Data files:"
ls -lh *.csv *.json

echo ""
echo "All tests complete. Results in: $OUTPUT_DIR"
