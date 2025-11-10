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

# Check if we need sudo for perf
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [ "$PARANOID" -gt 1 ]; then
    echo "⚠️  perf_event_paranoid = $PARANOID (requires sudo for call graphs)"
    SUDO="sudo"
else
    SUDO=""
fi

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
echo "   This will take a few seconds for small examples..."
echo "   Using: $SUDO perf record"
echo ""

$SUDO perf record \
  --call-graph dwarf \
  --freq 99 \
  --output perf.data \
  ./target/release/powers "$EXAMPLE" \
  2>&1 | grep -E "Woken up|Processed|perf record|Warning" || true

# Change ownership if we used sudo
if [ -n "$SUDO" ]; then
    $SUDO chown $USER:$USER perf.data 2>/dev/null || true
fi

# Verify we got stack traces
echo ""
echo "🔍 Verifying stack traces were captured..."
if [ ! -f perf.data ]; then
    echo "❌ ERROR: perf.data not created!"
    echo "   Perf record may have failed. Check permissions."
    exit 1
fi

# Count stack trace lines (lines that start with whitespace followed by hex address)
STACK_COUNT=$(perf script -i perf.data 2>/dev/null | grep -c "^[[:space:]]\+[0-9a-f]" 2>/dev/null || true)
STACK_COUNT=${STACK_COUNT:-0}

# Get file size for diagnostics
FILE_SIZE=$(stat -c%s perf.data 2>/dev/null || echo "0")

if [ "$STACK_COUNT" -eq 0 ] || [ "$FILE_SIZE" -lt 10000 ]; then
    echo "❌ ERROR: Insufficient profiling data captured!"
    echo ""
    echo "   Debugging info:"
    echo "   - perf.data size: $FILE_SIZE bytes"
    echo "   - Stack trace lines: $STACK_COUNT"
    
    if [ "$FILE_SIZE" -lt 10000 ]; then
        echo ""
        echo "   ⚠️  LIKELY CAUSE: Example runs too fast for profiling!"
        echo ""
        echo "   The example '$EXAMPLE' likely completes in milliseconds."
        echo "   Perf needs at least a few seconds to capture meaningful data."
        echo ""
        echo "   Solutions:"
        echo "   1. Use a longer-running example:"
        echo "      ./scripts/generate_flamegraph.sh examples/04-cascade"
        echo "      ./scripts/generate_flamegraph.sh examples/05-large-scale-brazilian"
        echo ""
        echo "   2. Increase iterations in config.json:"
        echo "      Edit $EXAMPLE/config.json"
        echo "      Change 'num_iterations' to a larger value (e.g., 100)"
        echo ""
        echo "   3. Profile with perf report instead:"
        echo "      sudo perf record --call-graph dwarf --freq 99 \\"
        echo "        ./target/release/powers $EXAMPLE"
        echo "      sudo perf report"
    else
        echo ""
        echo "   Other possible causes:"
        echo "   1. Binary missing debug symbols"
        echo "      Check: file target/release/powers | grep 'with debug_info'"
        echo ""
        echo "   2. Try checking manually:"
        echo "      sudo perf script -i perf.data | head -100"
    fi
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
