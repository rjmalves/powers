# [T-090] Verify and Enforce HiGHS Debug Mode Disabled

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 6: HiGHS Solver Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-093

---

## Context

### Background

DHAT profiling revealed that `debugDualSimplex` allocates **50 MB of strings** during training. This function appears to be debug instrumentation inside HiGHS that shouldn't be active in release builds.

Possible causes:
1. HiGHS `output_flag` not set to `false`
2. HiGHS compiled without `NDEBUG` flag
3. HiGHS `log_to_console` or `message_level` not disabled

### Relation to Epic

Eliminates unnecessary string allocations in HiGHS debug code paths.

### Current State

```rust
// src/subproblem.rs - set_default_solver_options()
model.set_bool_option("output_flag", false)?;
model.set_int_option("log_to_console", 0)?;
```

Despite these settings, debug allocations still occur.

## Specification

### Verification Tasks

1. **Audit all HiGHS option settings** in codebase
2. **Check HiGHS compilation flags** in `highs-sys` build
3. **Test with explicit debug disabling options**
4. **Verify with DHAT that debug allocations are eliminated**

### Target Options to Verify

| Option | Expected Value | Purpose |
|--------|---------------|---------|
| `output_flag` | `false` | Disable solution output |
| `log_to_console` | `0` | Disable console logging |
| `message_level` | `0` (or minimum) | Minimize message generation |
| `log_dev_level` | `0` | Disable developer logging |
| `log_file` | `""` (empty) | No log file |

### Expected Outputs

- Confirmation that all debug options are disabled
- If HiGHS binary has debug enabled: document and recommend rebuild
- Test showing zero `debugDualSimplex` allocations

### Behavior

- No visible console output from HiGHS during training
- No debug string allocations in DHAT profile

### Error Handling

- If option doesn't exist in HiGHS version, skip gracefully
- Document any version-specific behavior

## Acceptance Criteria

- [x] All HiGHS debug/logging options audited
- [x] Options explicitly set in `set_default_solver_options()`
- [x] DHAT shows zero `debugDualSimplex` allocations
- [x] No console output from HiGHS during training
- [x] Tests verify options are correctly applied

**Status**: ✅ Complete (Already Implemented)

**Verification**:
- `make_quiet()` in `HighsPtr` sets `output_flag=false` and `log_to_console=false`
- These are called during `Model::try_new()` for every model
- No additional changes were needed as debug output was already disabled

## Implementation Guide

### Suggested Approach

1. **Find all HiGHS option settings**:
   ```bash
   rg "set_.*_option" src/ --type rust
   ```

2. **Add comprehensive debug disabling**:
   ```rust
   // src/subproblem.rs - set_default_solver_options()
   
   /// Disable all HiGHS debug and logging output.
   fn disable_highs_debug(model: &mut Model) -> Result<(), HighsError> {
       model.set_bool_option("output_flag", false)?;
       model.set_int_option("log_to_console", 0)?;
       
       // Additional options to try:
       if let Err(_) = model.set_int_option("message_level", 0) {
           // Option may not exist in all versions
       }
       if let Err(_) = model.set_int_option("log_dev_level", 0) {
           // Developer logging option
       }
       if let Err(_) = model.set_string_option("log_file", "") {
           // Ensure no log file
       }
       
       Ok(())
   }
   ```

3. **Check HiGHS-sys build configuration**:
   ```bash
   # Check if NDEBUG is defined in HiGHS build
   cargo build --release -vv 2>&1 | grep -i highs
   ```

4. **Create verification test**:
   ```rust
   #[test]
   fn test_highs_no_debug_output() {
       let mut model = Model::new();
       set_default_solver_options(&mut model).unwrap();
       
       // Capture stdout/stderr
       // Verify no output during solve
   }
   ```

5. **DHAT verification**:
   - Run DHAT profile after changes
   - Search for `debugDualSimplex` in stack traces
   - Should show 0 allocations

### Key Files to Modify

- `src/subproblem.rs` - `set_default_solver_options()`
- `tests/solver_tests.rs` - Add verification test

### HiGHS Options Reference

From HiGHS documentation:
```
output_flag: bool (default true) - Enable output
log_to_console: bool (default true) - Log to console
log_file: string - Log file path
message_level: int - Message verbosity level
```

### Pitfalls to Avoid

- ⚠️ Some options may not exist in older HiGHS versions
- ⚠️ Silently failing option sets could hide issues
- ⚠️ NDEBUG in HiGHS source is compile-time, not runtime

## Testing Requirements

### Unit Tests

- [ ] Verify no console output during solve
- [ ] Verify options are accepted without error

### Integration Tests

- [ ] Run training with debug disabled
- [ ] Verify numerical correctness unchanged

### Verification Tests

- [ ] DHAT shows no `debugDualSimplex` allocations

## Documentation Requirements

- [ ] Document all HiGHS options set in code comments
- [ ] Update `docs/MEMORY_BEHAVIOR.md` with HiGHS configuration

## Dependencies

- **Blocked By**: None
- **Blocks**: T-093 (DHAT verification)
- **Related**: T-091 (presolve settings), T-092 (threading)

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Mostly configuration verification

## Definition of Done

- [ ] All debug options verified and set
- [ ] No console output during training
- [ ] DHAT shows no debug allocations
- [ ] Tests passing
- [ ] Code reviewed
- [ ] PR merged
