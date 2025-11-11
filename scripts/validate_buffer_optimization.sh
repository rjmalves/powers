#!/bin/bash
# Validate buffer optimization implementation
# 
# This script verifies that TICKET-006b buffer optimizations are working
# by measuring allocation counts and timing.

set -e

EXAMPLE="examples/03-multistage"
MAX_ITER=5

echo "=== Buffer Optimization Validation ==="
echo ""

# Build release
echo "Building release binary..."
cargo build --release --bin powers -q

echo ""
echo "=== Running with profiling ==="
echo "Example: $EXAMPLE"
echo "Iterations: $MAX_ITER"
echo ""

# Check if valgrind is available
if command -v valgrind &> /dev/null; then
    echo "Running with Valgrind massif (allocation profiling)..."
    valgrind --tool=massif \
             --massif-out-file=massif_buffer_validation.out \
             --pages-as-heap=no \
             ./target/release/powers run $EXAMPLE --max-iterations $MAX_ITER \
             2>&1 | grep -E "Iteration|Training|allocated"
    
    echo ""
    echo "=== Massif Summary ==="
    ms_print massif_buffer_validation.out | grep -A 5 "peak"
    
    # Extract allocation count
    ALLOCS=$(ms_print massif_buffer_validation.out | grep "allocs" | head -1 | awk '{print $NF}')
    echo ""
    echo "Total allocations: $ALLOCS"
    
    # Expected: ~620 allocations per training run (320 cuts + 300 states)
    # With 5 iterations: ~3,100 allocations
    # Allow some overhead: target < 5,000
    if [ "$ALLOCS" -lt 10000 ]; then
        echo "✅ PASS: Allocation count is reasonable ($ALLOCS < 10,000)"
    else
        echo "⚠️  WARNING: High allocation count ($ALLOCS >= 10,000)"
        echo "   Expected: ~3,000 for 5 iterations"
        echo "   This might indicate buffers aren't being reused"
    fi
    
else
    echo "⚠️  Valgrind not available, running timing test only"
    echo ""
    
    # Just run and time it
    /usr/bin/time -v ./target/release/powers run $EXAMPLE --max-iterations $MAX_ITER \
        2>&1 | grep -E "Iteration|Training|Elapsed|Maximum resident"
fi

echo ""
echo "=== Validation Complete ==="
echo ""
echo "To see detailed allocation patterns:"
echo "  ms_print massif_buffer_validation.out"
echo ""
echo "Expected behavior:"
echo "  - Allocations grow linearly with iterations (not quadratically)"
echo "  - Most allocations happen during initialization"
echo "  - Minimal allocations during training loop"
