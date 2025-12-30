# [T-097] Investigate HSimplexNla Debug Allocations

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None
> **Priority**: 2 (HiGHS Investigation)

## Files to Read Before Starting

- `docs/DHAT_SPRINT6_ANALYSIS.md` - Current allocation breakdown
- `src/solver.rs` - `make_quiet()` function
- `src/subproblem.rs` - `set_default_solver_options()` function

---

## Context

### Background

DHAT analysis shows 4.66 GB from HSimplexNla, which appears to be debug-related:

| Category | Bytes | Blocks |
|----------|-------|--------|
| HSimplexNla_debug | 4.66 GB | 0.60M |

The allocation sites reference `HSimplexNla::debugCheckData`, suggesting debug code may still be executing despite output being disabled.

### Current Output Configuration

```rust
// In make_quiet()
model.set_option("output_flag", false);
model.set_option("log_to_console", 0);
```

### Hypothesis

1. HiGHS may have debug checks that run regardless of output settings
2. The HiGHS library may not have been built with `NDEBUG` flag
3. There may be a different option to disable debug computations

---

## Specification

### Investigation Tasks

1. **Verify current HiGHS build configuration**:
   - Check how HiGHS was built (CMake flags)
   - Determine if `NDEBUG` is defined
   - Check HiGHS version and build type

2. **Analyze HSimplexNla source**:
   - Find `debugCheckData` function
   - Identify what triggers these debug allocations
   - Check if there's a runtime flag to disable

3. **Test potential solutions**:
   | Approach | Action |
   |----------|--------|
   | Check build type | Verify release vs debug HiGHS |
   | Find debug option | Search for `debug` options in HiGHS |
   | Rebuild if needed | Document how to build HiGHS with debug disabled |

4. **Document findings** with actionable recommendations

### Expected Outcomes

| Outcome | Likelihood | Action |
|---------|------------|--------|
| Debug enabled in HiGHS build | Medium | Document rebuild instructions |
| No way to disable at runtime | Medium | Accept as limitation |
| Found debug disable option | Low | Update `make_quiet()` |

---

## Acceptance Criteria

- [ ] HiGHS build configuration verified
- [ ] HSimplexNla debug code analyzed
- [ ] At least 2 approaches tested
- [ ] Findings documented
- [ ] If fixable: code changes proposed
- [ ] If not fixable: documented with explanation

---

## Implementation Guide

### Suggested Approach

1. **Check HiGHS build** (0.5 day):
   ```bash
   # Check if HiGHS was built in release mode
   nm -C target/release/libhighs*.a | grep -i debug
   
   # Check CMake configuration if building from source
   cmake -L path/to/highs/build | grep -i debug
   ```

2. **Analyze HiGHS source** (0.5 day):
   - Find `HSimplexNla::debugCheckData` in HiGHS source
   - Check what guards the debug code (`#ifndef NDEBUG`, etc.)
   - Check for runtime debug options

3. **Test options** (0.5 day):
   - Search HiGHS options for debug-related settings:
     ```rust
     // Possible options to test
     model.set_option("run_crosscheck", false);
     model.set_option("allow_unbounded_or_infeasible", true);
     ```

4. **Document findings** (0.5 day):
   - Add section to `DHAT_SPRINT6_ANALYSIS.md` or create new doc
   - Include build instructions if rebuild is needed

### Key Files to Investigate

- HiGHS: `src/simplex/HSimplexNla.cpp`
- HiGHS: `CMakeLists.txt` (build configuration)
- Powers: `src/solver.rs` (`make_quiet()`)

### Pitfalls to Avoid

- ⚠️ Don't assume debug is the only cause - verify with DHAT
- ⚠️ Rebuilding HiGHS may be complex; document carefully
- ⚠️ Debug checks may serve important validation purposes

---

## Testing Requirements

### Investigation Tests

- [ ] Verify HiGHS build type
- [ ] Test any discovered options with DHAT

### Validation Tests

- [ ] Any changes must pass golden tests (numerical correctness)
- [ ] Debug removal must not hide legitimate errors

---

## Documentation Requirements

- [ ] Document HiGHS build configuration
- [ ] Document findings (positive or negative)
- [ ] If rebuild needed: add to project build documentation

---

## Dependencies

- **Blocked By**: None
- **Blocks**: None (investigation ticket)
- **Related**: T-096 (HEkkDual investigation)

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Investigation with moderate scope; depends on HiGHS internals

---

## Definition of Done

- [ ] Investigation complete
- [ ] Root cause identified (or documented as unknown)
- [ ] Recommendation made
- [ ] If fixable: PR with changes
- [ ] Documentation updated
- [ ] PR merged
