#!/bin/bash
# Run all example problems for POWE.RS
# Usage: ./scripts/run_examples.sh

set -e

echo "========================================================================"
echo "POWE.RS Example Suite Runner"
echo "========================================================================"
echo ""

EXAMPLES_DIR="examples"
CARGO_CMD="cargo run --release"
FAILED=0
PASSED=0

# Function to run a single example
run_example() {
    local example_name=$1
    local example_path="${EXAMPLES_DIR}/${example_name}"
    
    echo "------------------------------------------------------------------------"
    echo "Running: ${example_name}"
    echo "------------------------------------------------------------------------"
    
    if [ ! -d "${example_path}" ]; then
        echo "❌ ERROR: Example directory not found: ${example_path}"
        ((FAILED++))
        return 1
    fi
    
    # Run the example
    if ${CARGO_CMD} "${example_path}" > /dev/null 2>&1; then
        echo "✅ PASSED: ${example_name}"
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAILED: ${example_name}"
        echo "   Run manually for details: cargo run --release ${example_path}"
        FAILED=$((FAILED + 1))
        return 1
    fi
    
    echo ""
}

# Run all examples
echo "Running all examples..."
echo ""

run_example "01-deterministic"
run_example "02-stochastic"
run_example "03-multistage"
run_example "04-cascade"
run_example "05-large-scale-brazilian"
run_example "06-par-model"
run_example "07-par-model-with-inflow-state"

# Summary
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo ""

if [ ${FAILED} -eq 0 ]; then
    echo "✅ All examples completed successfully!"
    exit 0
else
    echo "❌ Some examples failed. Check output above for details."
    exit 1
fi
