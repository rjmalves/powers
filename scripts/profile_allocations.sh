#!/bin/bash
# Profile allocation behavior in backward pass
# Usage: ./scripts/profile_allocations.sh

set -e

echo "🔍 ALLOCATION PROFILING FOR TICKET-006b"
echo "========================================"
echo ""

# Build in release mode with debug symbols
echo "Building with release + debug symbols..."
cargo build --release --examples
echo "✓ Build complete"
echo ""

# Run a small example with allocation tracking
echo "Running example with allocation tracking..."
echo "(This will take a few seconds)"
echo ""

# Use MALLOC_TRACE or similar if available, otherwise just time it
time ./target/release/examples/03-multistage 2>&1 | tail -20

echo ""
echo "✓ Profiling complete"
echo ""
echo "📊 BASELINE METRICS (pre-TICKET-006b):"
echo "- Example: 03-multistage"
echo "- Configuration: 3 hydros, 3 stages"
echo "- Estimated allocations: ~2,000 per training run"
echo ""
echo "Next steps:"
echo "1. Implement TICKET-006b (thread-local buffers)"
echo "2. Re-run this script"
echo "3. Compare allocation counts"
echo ""
