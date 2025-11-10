#!/bin/bash
#
# Profiling Infrastructure Test
# Validates that all profiling tools and scripts work correctly
#
# Usage: ./scripts/test_profiling.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=================================="
echo "Profiling Infrastructure Test"
echo "=================================="
echo ""

FAILED=0
WARNINGS=0

# Helper function for test results
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        FAILED=$((FAILED + 1))
    fi
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

echo "Testing system requirements..."
echo "=================================="

# Test Rust toolchain
if cargo --version > /dev/null 2>&1; then
    VERSION=$(cargo --version)
    test_result 0 "Rust/Cargo installed: $VERSION"
else
    test_result 1 "Rust/Cargo not found"
fi

# Test build capability
if cargo build --release --bin powers > /dev/null 2>&1; then
    test_result 0 "Release build works"
else
    test_result 1 "Cannot build release binary"
fi

# Test debug symbols
if file target/release/powers | grep -q "with debug_info"; then
    test_result 0 "Debug symbols enabled"
else
    warning "Debug symbols not found in release binary"
    info "  Add to Cargo.toml: [profile.release] debug = true"
fi

echo ""
echo "Testing profiling tools..."
echo "=================================="

# Test perf
if command -v perf > /dev/null 2>&1; then
    VERSION=$(perf --version 2>&1 | head -1)
    test_result 0 "perf installed: $VERSION"
    
    # Check perf_event_paranoid
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
    if [ "$PARANOID" -gt 1 ]; then
        warning "perf_event_paranoid = $PARANOID (will need sudo)"
        info "  Run: sudo sysctl -w kernel.perf_event_paranoid=1"
    else
        test_result 0 "perf_event_paranoid = $PARANOID (no sudo needed)"
    fi
else
    test_result 1 "perf not found"
    info "  Install: sudo apt install linux-tools-generic"
fi

# Test inferno
if command -v inferno-collapse-perf > /dev/null 2>&1; then
    test_result 0 "inferno installed (flamegraph generation)"
else
    test_result 1 "inferno not found"
    info "  Install: cargo install inferno"
fi

# Test valgrind
if command -v valgrind > /dev/null 2>&1; then
    VERSION=$(valgrind --version 2>&1 | head -1)
    test_result 0 "valgrind installed: $VERSION"
    
    # Test ms_print
    if command -v ms_print > /dev/null 2>&1; then
        test_result 0 "ms_print available"
    else
        warning "ms_print not found (valgrind installed but no ms_print)"
    fi
else
    warning "valgrind not found (memory profiling unavailable)"
    info "  Install: sudo apt install valgrind"
fi

# Test bc (for calculations)
if command -v bc > /dev/null 2>&1; then
    test_result 0 "bc installed (needed for calculations)"
else
    test_result 1 "bc not found"
    info "  Install: sudo apt install bc"
fi

echo ""
echo "Testing scripts..."
echo "=================================="

# Check scripts exist and are executable
for script in profile_baseline.sh profile_compare.sh generate_flamegraph.sh; do
    if [ -x "scripts/$script" ]; then
        test_result 0 "scripts/$script exists and is executable"
    else
        test_result 1 "scripts/$script not found or not executable"
    fi
done

echo ""
echo "Testing with small example..."
echo "=================================="

# Test with a fast example
EXAMPLE="examples/01-deterministic"

if [ ! -d "$EXAMPLE" ]; then
    warning "Example directory not found: $EXAMPLE"
    info "  Skipping quick test"
else
    echo "Running quick smoke test..."
    
    # Build
    if cargo build --release --bin powers > /dev/null 2>&1; then
        test_result 0 "Build succeeded"
    else
        test_result 1 "Build failed"
    fi
    
    # Quick run
    if timeout 10 ./target/release/powers "$EXAMPLE" > /dev/null 2>&1; then
        test_result 0 "Example runs successfully"
    else
        test_result 1 "Example failed to run"
    fi
    
    # Quick perf test (if available)
    if command -v perf > /dev/null 2>&1; then
        echo "Testing perf record (may need sudo)..."
        
        PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
        if [ "$PARANOID" -gt 1 ]; then
            SUDO="sudo"
        else
            SUDO=""
        fi
        
        if $SUDO perf record --call-graph dwarf --freq 99 \
           --output /tmp/test_perf.data \
           ./target/release/powers "$EXAMPLE" > /dev/null 2>&1; then
            
            # Fix ownership
            if [ -n "$SUDO" ]; then
                $SUDO chown $USER:$USER /tmp/test_perf.data 2>/dev/null || true
            fi
            
            test_result 0 "perf record works"
            
            # Check if we got stack traces
            if [ -f /tmp/test_perf.data ]; then
                STACK_COUNT=$(perf script -i /tmp/test_perf.data 2>/dev/null | grep -c "^[[:space:]]\+[0-9a-f]" || echo "0")
                
                if [ "$STACK_COUNT" -gt 0 ]; then
                    test_result 0 "Stack traces captured ($STACK_COUNT lines)"
                else
                    warning "No stack traces captured (example may be too fast)"
                    info "  This is normal for small examples"
                fi
                
                rm -f /tmp/test_perf.data
            fi
        else
            warning "perf record failed (check permissions)"
        fi
    fi
fi

echo ""
echo "Testing directory structure..."
echo "=================================="

# Check for example directories
if [ -d "examples" ]; then
    test_result 0 "examples/ directory exists"
    
    if [ -d "examples/05-large-scale-brazilian" ]; then
        test_result 0 "Large example exists (recommended for profiling)"
    else
        warning "Large example not found: examples/05-large-scale-brazilian"
    fi
else
    test_result 1 "examples/ directory not found"
fi

# Check for profiling results directory
if [ -d "profiling_results" ]; then
    test_result 0 "profiling_results/ directory exists"
    
    BASELINE_COUNT=$(find profiling_results -maxdepth 1 -name "baseline_*" -type d 2>/dev/null | wc -l)
    if [ "$BASELINE_COUNT" -gt 0 ]; then
        info "Found $BASELINE_COUNT baseline(s)"
    else
        warning "No baselines found (run: ./scripts/profile_baseline.sh)"
    fi
else
    info "profiling_results/ will be created on first run"
fi

echo ""
echo "=================================="
echo "Test Summary"
echo "=================================="
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo "Your profiling infrastructure is ready."
    echo ""
    echo "Next steps:"
    echo "  1. Establish baseline: ./scripts/profile_baseline.sh"
    echo "  2. Read guide: cat PROFILING_GUIDE.md"
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s), but no critical failures${NC}"
    echo "The infrastructure should work, but check warnings above."
    echo ""
    echo "You can still run:"
    echo "  ./scripts/profile_baseline.sh"
else
    echo -e "${RED}❌ $FAILED critical failure(s), $WARNINGS warning(s)${NC}"
    echo "Please fix the issues above before profiling."
    echo ""
    echo "Common fixes:"
    echo "  - Install perf: sudo apt install linux-tools-generic"
    echo "  - Install inferno: cargo install inferno"
    echo "  - Install bc: sudo apt install bc"
    exit 1
fi

echo ""
echo "=================================="
echo "Recommended: Run a Full Test"
echo "=================================="
echo ""
echo "To verify everything works end-to-end:"
echo ""
echo "  # Quick test (30 seconds)"
echo "  ./scripts/generate_flamegraph.sh examples/01-deterministic"
echo ""
echo "  # Full baseline (5 minutes)"
echo "  ./scripts/profile_baseline.sh examples/05-large-scale-brazilian"
echo ""

exit 0
