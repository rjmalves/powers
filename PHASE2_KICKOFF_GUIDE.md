# Phase 2: High-Priority Test Modernization - Kickoff Guide

**Prerequisites**: Phase 1 complete ✅  
**Current State**: 274/277 tests passing (99%)  
**Estimated Time**: 4-8 hours

---

## 🎯 Phase 2 Goals

1. Fix 3 failing builder validation tests
2. Re-enable and modernize 6 disabled test files
3. Achieve 100% test pass rate

---

## 📋 Task Breakdown

### Task 2.1: Fix Builder Validation Tests (30 minutes)

**Files**: `src/sddp/builder.rs` (test module)

**Failing Tests**:
1. `test_builder_validates_load_stage_count`
2. `test_builder_validates_stochastic_loads_scenario_count`
3. `test_builder_rejects_stochastic_loads_with_deterministic_inflows`

**Issue**: Tests expect validation errors that no longer occur

**Action**:
```bash
# Run tests to see failures
cargo test test_builder_validates

# Options:
# A) Update tests if validation rules changed legitimately
# B) Fix validation logic if it was accidentally removed
# C) Remove tests if they test obsolete constraints
```

**Files to check**:
- `src/sddp/builder.rs` - validation logic
- Look for load/inflow validation in builder methods

---

### Task 2.2: Re-enable integration_simple_2stage.rs (1 hour)

**File**: `tests/integration_simple_2stage.rs.disabled`  
**Errors**: 4 (lowest count, highest priority)

**Known Issues**:
- Uses old Realization::new signature
- May use converged() method

**Action**:
```bash
# Re-enable file
git mv tests/integration_simple_2stage.rs.disabled tests/integration_simple_2stage.rs

# Check errors
cargo test --test integration_simple_2stage --no-run 2>&1 | grep "error\[E"

# Fix based on errors found
```

**Likely Fixes**:
- Update Realization::new calls (remove one parameter)
- Already did similar fix in test_sddp_algorithm.rs - use as reference

---

### Task 2.3: Re-enable test_subproblem_construction.rs (2 hours)

**File**: `tests/test_subproblem_construction.rs.disabled`  
**Errors**: 16

**Known Issues**:
- Uses deleted `create_naive_stochastic_processes()`
- Uses old `realize_uncertainties()` API

**Action**:
1. Update all uses of `create_naive_stochastic_processes()` → remove/replace
2. Check `realize_uncertainties()` signature in `src/subproblem.rs`
3. Update calls to match new signature

**Reference**: `tests/fixtures/subproblems.rs` - already fixed similar issues

---

### Task 2.4: Re-enable test_par_validation.rs (2 hours)

**File**: `tests/test_par_validation.rs.disabled`  
**Errors**: 5

**Known Issues**:
- Uses `par_generator` (deleted) → use `scenario_generator`
- Uses `seasonal_params` (not exported) → use `uncertainty_model::SeasonalParams`
- Uses `base_noise` (deleted)
- Uses `correlation_applicator::{EntityRef, UncertaintyType}` → update imports

**Action**:
```bash
# Check what's actually needed
grep -n "par_generator\|seasonal_params\|base_noise" tests/test_par_validation.rs.disabled

# Update imports:
# - Remove par_generator, base_noise
# - Add scenario_generator if needed
# - Use uncertainty_model::SeasonalParams
# - Fix EntityRef → EntityReference
```

---

### Task 2.5: Re-enable test_scenario.rs (1 hour)

**File**: `tests/test_scenario.rs.disabled`  
**Errors**: 1 (import error only)

**Known Issue**:
- Imports deleted `stochastic_process::{self, Naive, StochasticProcess}`
- But tests NoiseGenerator which still exists

**Action**:
1. Check what parts of stochastic_process are actually used
2. If just used in a few places, comment out those tests
3. If pervasive, may need larger rewrite

---

### Task 2.6: Re-enable test_policy_validation.rs (2-3 hours)

**File**: `tests/test_policy_validation.rs.disabled`  
**Errors**: 24 (highest count)

**Known Issue**:
- Uses old Realization::new signature (12 args → 11 args)
- May have cascading issues

**Action**:
- Start after easier files are done
- May want to rewrite sections rather than fix all 24 errors
- Consider if some tests are obsolete

---

### Task 2.7: Re-enable test_sddp_error_paths.rs (2 hours)

**File**: `tests/test_sddp_error_paths.rs.disabled`  
**Errors**: 10

**Known Issues**:
- Uses deleted `TerminationReason`
- Uses old API calls

**Action**:
1. Check if TerminationReason was renamed or removed
2. Search for termination/convergence concepts in new API
3. Update error path tests to use current error handling

---

## 🎯 Recommended Order

1. **Builder validation tests** (30 min) - Quick win
2. **integration_simple_2stage** (1 hr) - High priority, low effort
3. **test_scenario** (1 hr) - Low error count
4. **test_par_validation** (2 hrs) - Important functionality
5. **test_subproblem_construction** (2 hrs) - Core functionality
6. **test_sddp_error_paths** (2 hrs) - Error handling
7. **test_policy_validation** (3 hrs) - Most complex, do last

**Total**: ~12 hours (with buffer)

---

## 🔧 Useful Commands

### Quick Error Check
```bash
# Check errors for a disabled file
cargo test --test integration_simple_2stage --no-run 2>&1 | grep "error\[E" | wc -l

# See detailed errors
cargo test --test integration_simple_2stage --no-run 2>&1 | grep "error\[E" -A 5
```

### Re-enable File
```bash
git mv tests/FILENAME.rs.disabled tests/FILENAME.rs
```

### Run Specific Test
```bash
cargo test --test FILENAME
```

### Check Overall Progress
```bash
cargo test 2>&1 | grep "test result"
```

---

## 📚 Resources

- **API_MIGRATION.md** - API change reference
- **PHASE1_COMPLETION_SUMMARY.md** - What was fixed in Phase 1
- **Phase 1 commits** - Examples of similar fixes

### Key API Changes to Remember

```rust
// Old → New

StorageState::new(&system, load_sp, inflow_sp)
→ StorageState::new(&system)

state::factory(..., load_sp, inflow_processes)
→ state::factory(..., uncertainty_models)

Subproblem::new(..., load_sp, inflow_processes, ...)
→ Subproblem::new_from_uncertainty_models(..., uncertainty_models, ...)

NodeData::new(..., "naive", &[], ...)
→ NodeData::new(..., Arc::new(vec![]), ...)

EntityRef → EntityReference
par_generator → scenario_generator
seasonal_params → uncertainty_model::SeasonalParams
```

---

## ✅ Success Criteria

- All 6 disabled test files re-enabled
- All builder validation tests passing or updated
- **100% test pass rate** (0 failures)
- Clean, well-documented commits

---

**Ready to start? Begin with Task 2.1 (Builder tests)!**
