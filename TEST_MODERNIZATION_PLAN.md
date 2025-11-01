# Test & Benchmark Modernization - Action Plan

**Status**: 🔴 CRITICAL - Tests do not compile  
**Priority**: HIGHEST  
**Blocking**: All development, CI/CD pipeline

---

## 🎯 Goal

Modernize test suite and benchmarks to work with current `uncertainty_model` API, replacing references to deleted modules (`unified_noise_spec`, `unified_inflow_model`, deprecated `seasonal_params`).

---

## 📊 Current Situation

```
✅ Library compiles:        cargo build --lib
❌ Tests don't compile:     cargo test --no-run (100+ errors)
❌ Benchmarks don't compile: cargo bench --no-run
```

**Root cause**: API refactor consolidated modules but didn't update tests.

---

## 🚀 Execution Plan

### PHASE 1: Restore Test Compilation (URGENT)

**Goal**: Make tests compile again  
**Time**: 4-6 hours  
**Priority**: DO THIS FIRST

#### Task 1.1: Fix Source File Test Modules ⏱️ 2h

**Files**:
- `src/state.rs` (~30 occurrences)
- `src/subproblem.rs` (~20 occurrences)  
- `src/fcf.rs` (several occurrences)

**Action**:
```bash
# Search and replace
unified_noise_spec::UnifiedNoiseSpec → uncertainty_model::UncertaintyModel
unified_noise_spec::TemporalModelSpec → uncertainty_model::TemporalModelSpec
unified_noise_spec::SeasonalNoiseParams → uncertainty_model::SeasonalParams
seasonal_params::SeasonalParams → uncertainty_model::SeasonalParams (contextual)
```

**Validation**:
```bash
cargo test --lib 2>&1 | grep "error\[E"  # Should show progress
```

#### Task 1.2: Fix Test Fixtures ⏱️ 1h

**Files**: `tests/fixtures/*.rs` (8 files)

These are dependencies for other tests - fix first!

**Action**:
1. Apply same search-replace as Task 1.1
2. Update constructors to new API
3. Check each fixture compiles individually

**Validation**:
```bash
cargo test --test "*" --no-run  # Check test compilation
```

#### Task 1.3: Document API Migration ⏱️ 30min

**Output**: `API_MIGRATION.md`

Create mapping table:
```markdown
| Old API (deleted) | New API (current) |
|-------------------|-------------------|
| unified_noise_spec::* | uncertainty_model::* |
| unified_inflow_model::* | uncertainty_model::* |
| seasonal_params::SeasonalParams | uncertainty_model::SeasonalParams |
```

Include examples of common patterns.

#### Task 1.4: Create Test Audit ⏱️ 1h

**Output**: `TEST_AUDIT.md`

Categorize all 44 test files:
- ✅ Ready to fix (simple API update)
- ⚠️ Needs investigation (may need rewrite)
- ❌ Broken beyond repair (delete or full rewrite)

**Phase 1 Success Criteria**:
```bash
cargo test --no-run  # Should complete without errors
```

---

### PHASE 2: High-Priority Test Modernization (SHORT-TERM)

**Goal**: Core tests passing  
**Time**: 2-3 days  
**When**: After Phase 1 complete

#### Task 2.1: Fix Critical Path Tests ⏱️ 1 day

**Priority Order**:

1. **Input Validation** (`test_input_validation.rs` - 46KB!)
   - Critical for correctness
   - Guards against bad user input
   - Extensive coverage

2. **Solver Interface** (`test_solver_interface.rs` - 37KB!)
   - Integration with HiGHS
   - Hot path testing
   - Numerical correctness

3. **SDDP Algorithm** (`test_sddp_algorithm.rs`)
   - Core functionality
   - Forward/backward passes
   - Convergence criteria

4. **Scenario Generation** (`test_scenario_generation_integration.rs`)
   - Data pipeline
   - PAR model correctness
   - Distribution testing

**Validation**: Each test file should pass:
```bash
cargo test --test test_input_validation
cargo test --test test_solver_interface
# etc.
```

#### Task 2.2: Fix Domain Logic Tests ⏱️ 1 day

**Files**:
- `test_state.rs` - State management
- `test_subproblem_construction.rs` - Subproblem building
- `test_cut.rs`, `test_cut_pool.rs` - Cut management
- `test_policy_validation.rs` - Policy checks

#### Task 2.3: Fix Integration Tests ⏱️ 4h

**Files**:
- `integration_simple_2stage.rs` - End-to-end
- `test_sddp_par_e2e.rs` - PAR integration

**Phase 2 Success Criteria**:
```bash
cargo test  # High-priority tests passing (>50% pass rate)
```

---

### PHASE 3: Benchmark Modernization (SHORT-TERM)

**Goal**: Benchmarks running, baseline documented  
**Time**: 1 day  
**When**: Parallel with Phase 2

#### Task 3.1: Fix Benchmark Compilation ⏱️ 4h

**Priority Benchmarks**:
1. `sddp_benchmarks.rs` - Core algorithm
2. `subproblem_solve.rs` - Solver performance
3. `par_performance.rs` - PAR generation
4. `memory_profiling.rs` - Memory usage

**Action**: Same search-replace as Phase 1

**Validation**:
```bash
cargo bench --no-run  # Should compile
```

#### Task 3.2: Run & Document Baseline ⏱️ 2h

```bash
# Run benchmarks
cargo bench > benchmark_baseline.txt 2>&1

# Document results
cat > BENCHMARK_BASELINE.md << EOF
# Benchmark Baseline (Post-Refactor)

Date: $(date)
Commit: $(git rev-parse HEAD)

## Results
$(cat benchmark_baseline.txt)

## Notes
- First benchmark run after uncertainty_model refactor
- Use as baseline for regression detection
EOF
```

#### Task 3.3: Validate Benchmark Relevance ⏱️ 2h

For each benchmark:
- ✅ Does it test current code?
- ✅ Are results meaningful?
- ❌ Is it outdated? → Remove or rewrite

**Phase 3 Success Criteria**:
```bash
cargo bench  # All benchmarks run successfully
```

---

### PHASE 4: Complete Test Coverage (MEDIUM-TERM)

**Goal**: All tests passing, comprehensive coverage  
**Time**: 1 week  
**When**: After Phases 1-3

#### Task 4.1: Fix Remaining Tests ⏱️ 2 days

**Files**: All remaining test files not covered in Phase 2

#### Task 4.2: Rewrite Broken Tests ⏱️ 2 days

**Candidates for rewrite**:
- `test_unified_noise_spec_conversion.rs` (module deleted)
- Any test that's broken beyond simple fix

**Approach**:
1. Understand original test intent
2. Write new test using current API
3. Ensure same coverage

#### Task 4.3: Add Missing Coverage ⏱️ 2 days

**Focus**:
- `uncertainty_model.rs` (new module needs tests!)
- Any refactored code without tests
- Edge cases discovered during migration

**Target**: >80% code coverage

#### Task 4.4: Documentation ⏱️ 4h

**Create**:
- `docs/TESTING.md` - How to run and write tests
- Update `CONTRIBUTING.md` with test requirements
- Add examples of test patterns

**Phase 4 Success Criteria**:
```bash
cargo test         # All tests pass
cargo tarpaulin    # >80% coverage
```

---

## 📋 Quick Command Reference

### Check Status
```bash
# Library compilation
cargo build --lib

# Test compilation
cargo test --no-run

# Benchmark compilation  
cargo bench --no-run

# Count errors
cargo test --no-run 2>&1 | grep "error\[E" | wc -l
```

### Find & Replace
```bash
# Find old API usage
rg "unified_noise_spec|unified_inflow_model" src/ tests/ benches/

# Interactive search-replace
sed -i 's/unified_noise_spec::/uncertainty_model::/g' src/state.rs
```

### Run Tests
```bash
# Specific test file
cargo test --test test_input_validation

# Tests matching pattern
cargo test scenario_generation

# With output
cargo test -- --nocapture --test-threads=1
```

### Run Benchmarks
```bash
# All benchmarks
cargo bench

# Specific benchmark
cargo bench --bench sddp_benchmarks

# Save results
cargo bench | tee benchmark_results.txt
```

---

## 🎯 Milestones & Deadlines

### Week 1
- ✅ Phase 1 complete (tests compile)
- ✅ Phase 2 started (50% tests passing)
- ✅ Phase 3 complete (benchmarks running)

### Week 2
- ✅ Phase 2 complete (80% tests passing)
- ✅ Phase 4 started (remaining tests)

### Week 3
- ✅ Phase 4 complete (all tests passing)
- ✅ Documentation complete
- ✅ CI/CD green

---

## 🚦 Risk Management

### High Risk
- ⚠️ **Unknown breaking changes** in new API
  - Mitigation: Review `uncertainty_model.rs` docs first
  - Mitigation: Test incrementally

- ⚠️ **Large test files** (46KB) hard to update
  - Mitigation: Break into smaller chunks if needed
  - Mitigation: Use code-reviewer agent for verification

### Medium Risk
- ⚠️ **Some tests may need complete rewrite**
  - Mitigation: Budget extra time for Phase 4
  - Mitigation: Use test-engineer agent for rewrites

### Low Risk
- ⚠️ **Benchmark results may differ** post-refactor
  - Mitigation: Document baseline before/after
  - Mitigation: Investigate significant regressions

---

## 🤝 Team Coordination

### If Multiple People Working

**Divide work by phase**:
- Person A: Phase 1 (blocking everyone)
- Person B: Phase 3 (parallel with Phase 2)
- Person A: Phase 2 (after Phase 1)
- Everyone: Phase 4 (divide test files)

**Avoid conflicts**:
- Assign specific test files to people
- Merge Phase 1 ASAP (blocks everything)
- Use feature branches: `fix/test-{module-name}`

---

## 📞 Getting Help

### Stuck on Phase 1?
```bash
# Use rust-implementer agent
gh copilot suggest "Fix src/state.rs test module to use uncertainty_model API"
```

### Stuck on Phase 2?
```bash
# Use test-engineer agent  
gh copilot suggest "Rewrite test_input_validation tests for new uncertainty_model API"
```

### Stuck on Phase 3?
```bash
# Use perf-optimizer agent
gh copilot suggest "Update sddp_benchmarks.rs to use uncertainty_model and verify no regressions"
```

### Need code review?
```bash
# Use code-reviewer agent
gh copilot suggest "Review my changes to test_scenario_generation for correctness and coverage"
```

---

## 📈 Progress Tracking

**Daily Update Template**:
```markdown
## Day N Progress

### Completed
- [x] Fixed src/state.rs test modules (30 occurrences)
- [x] Fixed src/subproblem.rs test modules (20 occurrences)

### In Progress
- [ ] Fixing test fixtures (3/8 done)

### Blocked
- None

### Tomorrow
- Complete test fixtures
- Start high-priority tests
```

**Track in**: `TEST_MODERNIZATION_PROGRESS.md`

---

## ✅ Final Checklist

Before considering this work complete:

**Phase 1**:
- [ ] All source file test modules fixed
- [ ] `cargo test --no-run` succeeds
- [ ] API migration guide created
- [ ] Test audit document created

**Phase 2**:
- [ ] Input validation tests passing
- [ ] Solver interface tests passing
- [ ] SDDP algorithm tests passing
- [ ] Scenario generation tests passing

**Phase 3**:
- [ ] All benchmarks compile
- [ ] Benchmarks run successfully
- [ ] Baseline performance documented
- [ ] No significant regressions

**Phase 4**:
- [ ] All tests passing
- [ ] Test coverage >80%
- [ ] Testing documentation complete
- [ ] CI/CD pipeline green

**Quality**:
- [ ] Code formatted (`cargo fmt --all`)
- [ ] No clippy warnings (`cargo clippy -- -D warnings`)
- [ ] CHANGELOG.md updated
- [ ] README.md updated if needed

---

## 🎓 Success Metrics

**Quantitative**:
- Tests compiling: 0% → 100%
- Tests passing: 0% → 100%
- Code coverage: Unknown → >80%
- Benchmarks running: 0 → 14

**Qualitative**:
- Test suite represents current API
- Benchmarks provide regression detection
- Team can develop with confidence
- CI/CD catches issues early

---

**Ready to start? Begin with Phase 1, Task 1.1!**

*See CODE_REVIEW_REPORT.md for detailed analysis*
