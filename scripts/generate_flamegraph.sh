#!/bin/bash
# Generate flamegraph with proper call graph recording
set -e

EXAMPLE="${1:-examples/01-deterministic}"
OUTPUT="flamegraph.svg"

echo "🔥 Generating Flamegraph"
echo "========================"
echo "Example: $EXAMPLE"
echo "Output: $OUTPUT"
echo ""

# Check if inferno is installed
if ! command -v inferno-collapse-perf &> /dev/null; then
    echo "❌ inferno not found. Installing..."
    cargo install inferno
fi

# Build with debug symbols (should already be set in Cargo.toml)
echo "📦 Building release binary with debug symbols..."
cargo build --release --bin powers

# Record with DWARF call graphs
echo "🎯 Recording with perf (--call-graph dwarf)..."
echo "   This will take ~40 seconds for large examples..."
perf record \
  --call-graph dwarf \
  --freq 99 \
  --output perf.data \
  ./target/release/powers "$EXAMPLE" \
  2>&1 | grep -E "Woken up|Processed|perf record|Warning" || true

# Verify we got stack traces
echo ""
echo "🔍 Verifying stack traces were captured..."
STACK_COUNT=$(perf script -i perf.data 2>/dev/null | grep -c "^[[:space:]]*[0-9a-f]" || echo "0")

if [ "$STACK_COUNT" -eq 0 ]; then
    echo "❌ ERROR: No stack traces found in perf.data!"
    echo "   This usually means:"
    echo "   1. Missing --call-graph flag (we added it)"
    echo "   2. Binary missing debug symbols (check with: file target/release/powers)"
    echo ""
    echo "   Try running manually:"
    echo "   perf script -i perf.data | head -100"
    exit 1
else
    echo "✅ Found $STACK_COUNT stack trace lines"
fi

# Generate flamegraph
echo ""
echo "🔥 Generating flamegraph SVG..."
perf script -i perf.data 2>/dev/null | \
  inferno-collapse-perf 2>/dev/null | \
  inferno-flamegraph > "$OUTPUT"

# Check result
if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo "✅ Flamegraph generated successfully!"
    echo "   Size: $SIZE"
    echo "   Path: $(pwd)/$OUTPUT"
    echo ""
    echo "📊 Open in browser:"
    echo "   file://$(pwd)/$OUTPUT"
    echo ""
    echo "   Or with Firefox:"
    echo "   firefox $OUTPUT"
else
    echo "❌ Failed to generate flamegraph"
    exit 1
fi

# Show top functions
echo ""
echo "📈 Top 10 functions by CPU time:"
perf report -i perf.data --stdio 2>/dev/null | head -30 | tail -12

echo ""
echo "✅ Done!"
