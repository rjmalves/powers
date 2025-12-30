# [T-096] Investigate HEkkDual Allocation Sources

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None
> **Priority**: 2 (HiGHS Investigation)

## Files to Read Before Starting

- `docs/DHAT_SPRINT6_ANALYSIS.md` - Current allocation breakdown
- `src/subproblem.rs` - `set_default_solver_options()` function
- HiGHS documentation: https://ergo-code.github.io/HiGHS/

---

## Context

### Background

After Sprint 6 optimizations, HEkkDual allocations remain the largest contributor:

| Category | Bytes | Blocks |
|----------|-------|--------|
| HEkkDual (fill_assign) | 22.33 GB | 3.71M |
| HEkkDual (default_append) | 12.57 GB | 4.12M |
| **Total HEkkDual** | **34.90 GB** | **7.83M** |

This is 77% of remaining allocations. Investigation is needed to determine if these can be reduced or if they are inherent to the dual simplex algorithm.

### Current HiGHS Configuration

```rust
// In set_default_solver_options()
model.set_option("presolve", "off");
model.set_option("solver", "simplex");
model.set_option("simplex_strategy", 1);        // Dual simplex
model.set_option("simplex_update_limit", 5000);
model.set_option("simplex_price_strategy", 1);
model.set_option("simplex_scale_strategy", 0);
model.set_option("parallel", "off");
model.set_option("threads", 1);
model.set_option("simplex_dual_edge_weight_strategy", -1);  // Auto
model.set_option("simplex_primal_edge_weight_strategy", -1); // Auto
```

---

## Specification

### Investigation Tasks

1. **Analyze HiGHS source** for HEkkDual allocation triggers:
   - What causes `_M_fill_assign` allocations?
   - What causes `_M_default_append` allocations?
   - Are these per-iteration or per-solve?

2. **Test alternative simplex options**:
   | Option | Current | Test Values |
   |--------|---------|-------------|
   | `simplex_strategy` | 1 (dual) | 4 (primal) |
   | `simplex_update_limit` | 5000 | 1000, 10000, 100000 |
   | `simplex_dual_edge_weight_strategy` | -1 (auto) | 0, 1, 2 |

3. **Measure impact** with DHAT for each configuration:
   ```bash
   valgrind --tool=dhat --dhat-out-file=dhat-test.out \
       ./target/release/powers run examples/05-large-scale-brazilian
   ```

4. **Document findings** with:
   - Which options affect HEkkDual allocations
   - Performance tradeoffs (solve time vs allocations)
   - Recommended configuration changes (if any)

### Expected Outcomes

| Outcome | Likelihood | Action |
|---------|------------|--------|
| HEkkDual allocations are inherent | High | Document as limitation |
| Options reduce allocations | Medium | Update `set_default_solver_options()` |
| Primal simplex uses less memory | Low | Evaluate performance tradeoff |

---

## Acceptance Criteria

- [ ] HiGHS source analyzed for HEkkDual allocation sources
- [ ] At least 3 option configurations tested with DHAT
- [ ] Results documented in investigation report
- [ ] If reductions found: code changes proposed
- [ ] If no reductions: documented as inherent limitation

---

## Implementation Guide

### Suggested Approach

1. **Phase 1: Source Analysis** (1 day)
   - Read HiGHS `HEkkDual.cpp` and `HEkkDualRow.cpp`
   - Identify allocation patterns in `_M_fill_assign` and `_M_default_append`
   - Document when these allocations occur

2. **Phase 2: Option Testing** (1 day)
   - Create test script to run DHAT with different options
   - Test at minimum:
     - Baseline (current config)
     - `simplex_update_limit = 1000`
     - `simplex_update_limit = 100000`
     - `simplex_dual_edge_weight_strategy = 0` (Dantzig)
   - Record allocation counts and solve times

3. **Phase 3: Documentation** (0.5 day)
   - Create `docs/HEKKDUAL_INVESTIGATION.md` with findings
   - Update `DHAT_SPRINT6_ANALYSIS.md` with conclusions

### Test Script Template

```bash
#!/bin/bash
# Test different HiGHS configurations

for UPDATE_LIMIT in 1000 5000 10000 100000; do
    echo "Testing simplex_update_limit = $UPDATE_LIMIT"
    # Modify config and run DHAT
    time valgrind --tool=dhat --dhat-out-file=dhat-update-$UPDATE_LIMIT.out \
        ./target/release/powers run examples/05-large-scale-brazilian
done
```

### Key Files to Modify

- `src/subproblem.rs`: If option changes are beneficial
- `docs/HEKKDUAL_INVESTIGATION.md`: New investigation report

### Pitfalls to Avoid

- ⚠️ Don't change default options without measuring solve time impact
- ⚠️ DHAT runs are slow (~20x); use smaller examples for iteration
- ⚠️ Document negative results (no improvement) as valid findings

---

## Testing Requirements

### Investigation Tests

- [ ] Run DHAT with at least 3 configurations
- [ ] Measure solve time for each configuration
- [ ] Verify numerical correctness with golden tests

### Validation Tests

- [ ] Any configuration changes must pass all tests
- [ ] Performance benchmarks if options change

---

## Documentation Requirements

- [ ] Create `docs/HEKKDUAL_INVESTIGATION.md` with findings
- [ ] Update epic overview with conclusions
- [ ] If no optimizations found, document as "investigated, inherent to HiGHS"

---

## Dependencies

- **Blocked By**: None
- **Blocks**: None (investigation ticket)
- **Related**: T-097 (HSimplexNla investigation)

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Investigation with uncertain outcomes; requires HiGHS source analysis

---

## Definition of Done

- [ ] Investigation complete
- [ ] Findings documented
- [ ] Recommendation made (change options or accept limitation)
- [ ] If changing options: PR with changes and benchmarks
- [ ] PR merged
