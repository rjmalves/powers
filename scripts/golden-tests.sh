#!/bin/bash
set -e

# Golden Test Script for POWE.RS
# This script generates and verifies golden outputs for regression testing.
# Timing information is filtered out since it varies between runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GOLDEN_DIR="$REPO_ROOT/tests/golden"
BINARY="$REPO_ROOT/target/release/powers"

# Function to filter out timing information
# Removes: all timing columns (| 00:00:00.XXX patterns, including multiple per line)
# Removes: timing header columns (fwd, bwd, total)
# Removes: standalone timing lines (Training time:, etc.)
# Removes: trailing pipes and whitespace
filter_timing() {
  sed -E '
    # Remove all timing values in format | 00:00:00.XXX (handles multiple per line)
    s/\| [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} //g
    # Remove trailing timing at end of line (no trailing |)
    s/ [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}$//g
    # Remove timing column headers from table
    s/\|[[:space:]]*fwd[[:space:]]*\|[[:space:]]*bwd[[:space:]]*\|[[:space:]]*total//g
    s/\|[[:space:]]*fwd[[:space:]]*//g
    # Clean up trailing pipe and whitespace
    s/[[:space:]]*\|[[:space:]]*$//g
  ' | \
  grep -Ev "^(\[INFO\] )?(Training time:|Simulation time:|Total running time:)"
}

# Ensure binary exists
if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found at $BINARY"
  echo "Run 'cargo build --release' first."
  exit 1
fi

mkdir -p "$GOLDEN_DIR"

case "${1:-verify}" in
  generate)
    echo "Generating golden outputs..."
    echo ""
    for example in "$REPO_ROOT"/examples/0*; do
      [ -d "$example" ] || continue
      name=$(basename "$example")
      echo "Generating: $name (example 05 takes up to 2 minutes)"
      if "$BINARY" run "$example" 2>&1 | filter_timing > "$GOLDEN_DIR/$name.txt"; then
        echo "  ✓ Created $GOLDEN_DIR/$name.txt"
      else
        echo "  ⚠ WARNING: $name failed, error output captured"
      fi
    done
    echo ""
    echo "Golden files generated in $GOLDEN_DIR"
    ;;
  
  verify)
    echo "Verifying golden outputs..."
    echo ""
    FAILED=0
    for example in "$REPO_ROOT"/examples/0*; do
      [ -d "$example" ] || continue
      name=$(basename "$example")
      golden_file="$GOLDEN_DIR/$name.txt"
      
      if [ ! -f "$golden_file" ]; then
        echo "MISSING: $golden_file"
        FAILED=1
        continue
      fi
      
      echo -n "Verifying: $name ... "
      if "$BINARY" run "$example" 2>&1 | filter_timing | diff -u "$golden_file" - > /dev/null 2>&1; then
        echo "PASS"
      else
        echo "FAIL"
        echo "--- Diff output ---"
        "$BINARY" run "$example" 2>&1 | filter_timing | diff -u "$golden_file" - || true
        echo "-------------------"
        FAILED=1
      fi
    done
    
    echo ""
    if [ $FAILED -eq 1 ]; then
      echo "⛔ GOLDEN TEST FAILURE - Algorithm outputs have changed!"
      echo "   If this is unexpected, STOP and investigate."
      echo "   If intentional, run: $0 generate"
      exit 1
    fi
    echo "✅ All golden tests passed"
    ;;
  
  *)
    echo "Usage: $0 [generate|verify]"
    echo ""
    echo "Commands:"
    echo "  generate  - Generate golden output files from current binary"
    echo "  verify    - Verify current outputs match golden files (default)"
    exit 1
    ;;
esac
