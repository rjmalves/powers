# Scenario Generation: Next Steps

**Date**: 2025-11-02  
**Context**: Analysis of scenario_generator lag buffer confusion

---

## TL;DR

You're right! The SAA should only contain innovations, and AR dynamics should be in LP constraints. The code does this correctly during SDDP execution, but has **legacy machinery** in `scenario_generator.rs` that computes unnecessary observations.

### The Issue

**Two lag buffer systems exist**:

1. **ScenarioGenerator.par_states** (LEGACY):
   - Operates in residual space
   - Used during SAA generation to compute `scenario.values`
   - These values are **DISCARDED** for inflow entities!
   - Only `scenario.innovations` are stored in SAA

2. **Subproblem.inflow_manager** (ACTIVE):
   - Operates in observation space
   - Used during SDDP execution for AR constraint RHS
   - Updated after each LP solve with realized observations
   - This is the correct system that actually matters

---

## Recommended Next Steps

### Phase 1: Document & Validate (1 day)

**Goal**: Make the current behavior explicit and verify removal is safe

**Tasks**:

1. **Add documentation comments** in `scenario_generator.rs`:
   ```rust
   // LEGACY: This lag buffer is only used during SAA generation to compute
   // scenario.values for display/validation. For inflow entities, these values
   // are DISCARDED - only scenario.innovations are stored in the SAA.
   // The actual lag buffer used during SDDP execution is Subproblem.inflow_manager.
   pub par_states: HashMap<(UncertaintyType, usize), LagBuffer>,
   ```

2. **Add test** to verify independence:
   ```rust
   #[test]
   fn test_saa_uses_only_innovations() {
       // Generate SAA with PAR models
       // Verify that only scenario.innovations are stored
       // Verify that scenario.values for inflows are not used
   }
   ```

3. **Add assertion** in SAA generation loop (`input.rs:1232-1236`):
   ```rust
   UncertaintyType::Inflow => {
       // Document: For inflows, we only store innovations
       // The scenario.values computed during generation are unused
       inflow_innovations[inflow_idx].push(scenario.innovations[model_idx]);
       inflow_idx += 1;
   }
   ```

**Verification**:
- All tests pass (307/307)
- All examples run correctly
- Documentation is clear about what's legacy

---

### Phase 2: Simplify Generation (2-3 days)

**Goal**: Remove unnecessary computation during SAA generation

**Tasks**:

1. **Simplify PAR scenario generation** in `generate_stage_scenarios`:
   ```rust
   UncertaintyModel::PeriodicAR { entity_type, entity_id, par_params } => {
       // Sample innovation (what actually goes to SAA)
       let innovation = params.distribution.transform(base_noise, 0.0, 1.0);
       
       // For PAR models, we only need the innovation
       // The observation Y_t will be computed during LP solve
       scenario.values.push(0.0);  // Placeholder (unused for inflows)
       scenario.innovations.push(innovation);
       scenario.residuals.push(0.0);  // Placeholder (legacy field)
       
       // Note: No lag buffer update needed - observations computed at solve time
   }
   ```

2. **Remove `par_states` field** from ScenarioGenerator:
   ```rust
   pub struct ScenarioGenerator {
       models: Vec<UncertaintyModel>,
       correlation: Option<CorrelationApplicator>,
       // REMOVED: par_states: HashMap<(UncertaintyType, usize), LagBuffer>,
       base_noise_buffer: Vec<f64>,
       transformed_buffer: Vec<f64>,
   }
   ```

3. **Update constructor** to skip lag buffer initialization:
   ```rust
   pub fn new(...) -> Result<Self, PowersError> {
       // ... existing code ...
       
       // REMOVED: PAR state initialization loop
       
       Ok(Self {
           models,
           correlation,
           // REMOVED: par_states,
           base_noise_buffer: Vec::with_capacity(n_entities),
           transformed_buffer: Vec::with_capacity(n_entities),
       })
   }
   ```

4. **Remove reset_par_states method** (no longer needed)

**Verification**:
- ✅ All tests pass (307/307)
- ✅ All examples produce identical results (check with git diff on outputs)
- ✅ Example 06 and 07 (PAR models) work correctly
- ✅ Memory footprint reduced by ~800 bytes per instance
- ✅ SAA generation ~10% faster (no residual space computation)

**Expected Impact**:
- Code: ~100 lines removed
- Memory: ~800 bytes saved per ScenarioGenerator instance
- Performance: ~10% faster SAA generation
- Clarity: Removes confusion about two lag buffer systems

---

### Phase 3: Clean Up Data Structures (1 day)

**Goal**: Remove unused fields from Scenario struct

**Tasks**:

1. **Deprecate residuals field**:
   ```rust
   #[derive(Debug, Clone)]
   pub struct Scenario {
       pub values: Vec<f64>,      // For Load entities and validation
       pub innovations: Vec<f64>, // What actually goes to SAA
       // #[deprecated] pub residuals: Vec<f64>,  // UNUSED - can be removed
   }
   ```

2. **Update construction** to skip residuals:
   ```rust
   impl Scenario {
       fn with_capacity(n_entities: usize) -> Self {
           Self {
               values: Vec::with_capacity(n_entities),
               innovations: Vec::with_capacity(n_entities),
               // residuals no longer allocated
           }
       }
   }
   ```

3. **Update tests** to not check residuals field

**Verification**:
- ✅ All tests pass after removing residuals checks
- ✅ Memory usage reduced by ~8 bytes per entity per scenario
- ✅ No functionality lost (residuals were never used)

**Expected Impact**:
- Memory: ~30% reduction in Scenario struct size for typical problems
- Clarity: Clear separation between what's for display (values) vs what's used (innovations)

---

### Phase 4: Type-Safe Scenarios (Future - Optional)

**Goal**: Prevent accidental misuse of observations vs innovations

**Design**:
```rust
pub enum ScenarioValue {
    Load(f64),           // Observation (used directly)
    Inflow(f64),         // Innovation (used in constraint RHS)
}

pub struct TypedScenario {
    values: Vec<ScenarioValue>,
}
```

**Benefits**:
- Type system prevents using innovations as observations
- Compiler-enforced correctness
- Self-documenting code

**Effort**: 3-5 days (touches many files)

---

## Testing Strategy

### Unit Tests

1. **Test SAA independence**:
   - Generate SAA with identical seed but different par_states initialization
   - Verify identical innovations are stored

2. **Test SDDP execution**:
   - Run training with and without par_states
   - Verify identical lower bounds and policies

3. **Test observation computation**:
   - Manually verify Y_t computed in `realize_uncertainties` matches expected formula
   - Check lag buffer updates are correct

### Integration Tests

1. **Example 06** (PAR model):
   - Before/after: Identical policy values
   - Before/after: Identical simulation outputs
   - Before/after: Same number of iterations to convergence

2. **Example 07** (PAR with inflow state):
   - Before/after: Identical policy values
   - Before/after: Identical simulation outputs

### Regression Tests

1. **Run full test suite**: 307/307 passing
2. **Run all examples**: Identical outputs (use deterministic seeds)
3. **Benchmark**: SAA generation time (expect ~10% improvement)

---

## Risk Assessment

### Low Risk Changes (Phase 1)
- ✅ Adding documentation: Zero risk
- ✅ Adding tests: Zero risk
- ✅ Adding assertions: Zero risk (fail fast if assumptions violated)

### Medium Risk Changes (Phase 2)
- ⚠️ Removing par_states: Risk of unexpected dependencies
- **Mitigation**: Comprehensive tests before/after, check all examples
- **Rollback**: Simple git revert if issues found

### Low-Medium Risk (Phase 3)
- ⚠️ Removing residuals field: May break external code using this field
- **Mitigation**: Deprecate first, remove later
- **Note**: Since residuals are never used, unlikely to have external dependencies

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Documentation explains both lag buffer systems
- [ ] Comments mark legacy code explicitly
- [ ] Test verifies par_states is unused during execution
- [ ] All existing tests pass

### Phase 2 Complete When:
- [ ] par_states removed from ScenarioGenerator
- [ ] All tests pass (307/307)
- [ ] All examples produce identical outputs
- [ ] Examples 06 and 07 work correctly
- [ ] Code review confirms no unintended changes

### Phase 3 Complete When:
- [ ] residuals field removed or deprecated
- [ ] Memory usage reduced by ~30% for Scenario structs
- [ ] All tests updated and passing

---

## Timeline Estimate

- **Phase 1**: 1 day (documentation + validation)
- **Phase 2**: 2-3 days (implementation + testing)
- **Phase 3**: 1 day (cleanup + verification)

**Total**: 4-5 days for complete cleanup

---

## Alternative: Minimal Change

If you prefer minimal disruption, just do **Phase 1**:
- Add documentation explaining the dual system
- Mark par_states as legacy with clear comments
- Add test verifying it's unused during execution
- **No code changes**, zero risk

This resolves the confusion without touching working code.

---

## Recommendation

**Start with Phase 1**, then decide:

1. If confusion is resolved by documentation → Stop there
2. If code cleanup is desired → Continue to Phase 2
3. If full modernization is wanted → Complete Phase 3

**Suggested approach**: Phase 1 + Phase 2 (remove par_states) provides the best balance of cleanup vs risk.
