#!/bin/bash
# Test Modernization Quick Start Script
# Run this to begin the test modernization process

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║        POWE.RS Test Modernization Quick Start                   ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check current status
echo "📊 Step 1: Checking current status..."
echo ""

echo "Library compilation:"
if cargo build --lib --quiet 2>/dev/null; then
    echo "  ✅ Library compiles"
else
    echo "  ❌ Library does not compile"
    echo "  ⚠️  Fix library first before proceeding!"
    exit 1
fi

echo ""
echo "Test compilation:"
ERROR_COUNT=$(cargo test --no-run 2>&1 | grep -c "error\[E" || echo "0")
if [ "$ERROR_COUNT" -eq "0" ]; then
    echo "  ✅ Tests compile!"
    echo "  🎉 You may not need this migration!"
else
    echo "  ❌ Tests do not compile: $ERROR_COUNT errors"
fi

echo ""
echo "Benchmark compilation:"
BENCH_ERROR_COUNT=$(cargo bench --no-run 2>&1 | grep -c "error\[E" || echo "0")
if [ "$BENCH_ERROR_COUNT" -eq "0" ]; then
    echo "  ✅ Benchmarks compile"
else
    echo "  ❌ Benchmarks do not compile: $BENCH_ERROR_COUNT errors"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Step 2: Find occurrences of old API
echo "🔍 Step 2: Finding old API references..."
echo ""

OLD_API_COUNT=$(rg "unified_noise_spec|unified_inflow_model" src/ tests/ benches/ 2>/dev/null | wc -l || echo "0")
echo "  Found $OLD_API_COUNT occurrences of old API references"

if [ "$OLD_API_COUNT" -gt "0" ]; then
    echo ""
    echo "  Top affected files:"
    rg "unified_noise_spec|unified_inflow_model" src/ tests/ benches/ --count-matches 2>/dev/null | head -10 || echo "    (none)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Step 3: Suggest next steps
echo "🎯 Step 3: Recommended next steps"
echo ""

if [ "$ERROR_COUNT" -eq "0" ] && [ "$BENCH_ERROR_COUNT" -eq "0" ]; then
    echo "  ✅ Great news! Everything compiles."
    echo ""
    echo "  Next steps:"
    echo "  1. Run tests: cargo test"
    echo "  2. Run benchmarks: cargo bench"
    echo "  3. Check coverage: cargo tarpaulin"
    exit 0
fi

echo "  Phase 1: Fix Source File Test Modules (START HERE)"
echo ""
echo "  Files to fix (in order):"
echo "    1. src/state.rs"
echo "    2. src/subproblem.rs"
echo "    3. src/fcf.rs"
echo "    4. tests/fixtures/*.rs"
echo ""
echo "  Search and replace:"
echo "    - unified_noise_spec::UnifiedNoiseSpec → uncertainty_model::UncertaintyModel"
echo "    - unified_noise_spec::TemporalModelSpec → uncertainty_model::TemporalModelSpec"
echo "    - unified_noise_spec::SeasonalNoiseParams → uncertainty_model::SeasonalParams"
echo "    - unified_inflow_model::* → uncertainty_model::*"
echo ""

# Step 4: Create branch and tracking
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📝 Step 4: Setup"
echo ""

BRANCH_EXISTS=$(git branch --list fix/test-modernization | wc -l)
if [ "$BRANCH_EXISTS" -eq "0" ]; then
    echo "  Creating branch: fix/test-modernization"
    git checkout -b fix/test-modernization 2>/dev/null || echo "  (branch may already exist)"
else
    echo "  Branch already exists: fix/test-modernization"
fi

# Create progress tracking file if it doesn't exist
if [ ! -f "TEST_MODERNIZATION_PROGRESS.md" ]; then
    cat > TEST_MODERNIZATION_PROGRESS.md << 'EOF'
# Test Modernization Progress

**Started**: $(date)
**Status**: In Progress

## Phase 1: Restore Test Compilation

### Source Files
- [ ] src/state.rs
- [ ] src/subproblem.rs
- [ ] src/fcf.rs

### Test Fixtures
- [ ] tests/fixtures/systems.rs
- [ ] tests/fixtures/scenarios.rs
- [ ] tests/fixtures/subproblems.rs
- [ ] tests/fixtures/benchmarks.rs
- [ ] tests/fixtures/oos.rs
- [ ] tests/fixtures/mock_solver.rs
- [ ] tests/fixtures/validation.rs
- [ ] tests/fixtures/simple_2stage_reservoir.rs

### Validation
- [ ] cargo test --lib compiles
- [ ] cargo test --no-run completes

## Phase 2: High-Priority Tests

- [ ] test_input_validation.rs
- [ ] test_solver_interface.rs
- [ ] test_sddp_algorithm.rs
- [ ] test_scenario_generation_integration.rs

## Phase 3: Benchmarks

- [ ] All benchmarks compile
- [ ] Baseline documented

## Notes

### Day 1
- Started test modernization

EOF
    echo "  Created: TEST_MODERNIZATION_PROGRESS.md"
else
    echo "  Progress file exists: TEST_MODERNIZATION_PROGRESS.md"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Setup complete!"
echo ""
echo "📖 Next: Read CODE_REVIEW_REPORT.md for full analysis"
echo "📋 Next: Read TEST_MODERNIZATION_PLAN.md for detailed plan"
echo "🏃 Next: Start with src/state.rs"
echo ""
echo "Quick fix command:"
echo ""
echo "  # Fix src/state.rs"
echo "  sed -i 's/unified_noise_spec::/uncertainty_model::/g' src/state.rs"
echo "  sed -i 's/unified_inflow_model::/uncertainty_model::/g' src/state.rs"
echo ""
echo "  # Verify"
echo "  cargo test --lib 2>&1 | grep -c \"error\[E\""
echo ""
echo "Good luck! 🚀"
