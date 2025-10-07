#!/bin/bash
# Memory profiling script for POWE.RS
# Runs comprehensive memory analysis and generates report

set -e

echo "=========================================================================="
echo "POWE.RS Memory Profiling"
echo "=========================================================================="
echo ""

# Build release binary
echo "Building release binary..."
cargo build --release --quiet

echo "Running memory profiling benchmarks..."
echo ""

# Run memory profiling benchmark with full output
cargo bench --bench memory_profiling -- --sample-size 10 2>&1 | \
    grep -E "(Memory Profile:|Initial RSS:|Peak RSS:|Final RSS:|Delta RSS:|Benchmarking)" | \
    tee memory_profile_results.txt

echo ""
echo "=========================================================================="
echo "Memory profiling complete!"
echo "Results saved to: memory_profile_results.txt"
echo "=========================================================================="
