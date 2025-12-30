# [T-101] DHAT Verification of Rust Allocation Reduction

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: T-094, T-095, T-096, T-097, T-098, T-099, T-100
> **Blocks**: None

---

## Context

### Background

After implementing all Sprint 7 Rust allocation optimizations, verify the reduction with DHAT profiling.

### Baseline (from Appendix B)

- Rust Application allocations: ~1.8 GB (2.0% of total)
- After Sprint 6 HiGHS optimizations, Rust allocations should be more visible

### Target

Reduce Rust application allocations by **≥50%** (targeting ~0.9 GB or less).

## Specification

### Tasks

1. **Run DHAT on example 05** with all Sprint 7 changes
2. **Analyze Rust-specific allocation sites**
3. **Compare to Sprint 6 baseline**
4. **Document findings**

### Expected Outputs

- DHAT output file: `dhat-sprint7.out`
- Analysis comparing Sprint 6 → Sprint 7
- Updated documentation

## Acceptance Criteria

- [ ] DHAT profiling completed
- [ ] Rust allocations analyzed
- [ ] ≥50% reduction in Rust allocations verified
- [ ] Documentation updated

## Implementation Guide

### Suggested Approach

1. **Run DHAT**:
   ```bash
   cargo build --release
   valgrind --tool=dhat \
       --dhat-out-file=dhat-sprint7.out \
       ./target/release/powers run examples/05-large-scale-brazilian
   ```

2. **Analyze Rust allocations** (filter out HiGHS):
   ```python
   import json
   
   with open('dhat-sprint7.out') as f:
       data = json.load(f)
   
   # Filter for Rust allocations (not HiGHS)
   rust_allocations = [
       pp for pp in data['pps']
       if not any(highs_pattern in str(data['ftbl'][f]) 
                  for f in pp['fs']
                  for highs_pattern in ['HEkk', 'HFactor', 'Highs'])
   ]
   
   total_rust = sum(pp['tb'] for pp in rust_allocations)
   print(f"Total Rust allocations: {total_rust:,} bytes")
   ```

3. **Check specific optimizations**:
   - Search for `uniform_prob_by_count` - should be zero
   - Search for `sample_scenario` - should be minimal
   - Search for `state.clone` - should be zero

4. **Document findings**:
   ```markdown
   ## Sprint 7 Rust Allocation Analysis
   
   | Allocation Site | Sprint 6 | Sprint 7 | Reduction |
   |-----------------|----------|----------|-----------|
   | uniform_prob_by_count | X MB | 0 | 100% |
   | sample_scenario | X MB | Y MB | Z% |
   | state.clone | X MB | 0 | 100% |
   | HashSet | X MB | 0 | 100% |
   | Total Rust | 1.8 GB | X GB | Y% |
   ```

### Key Analysis Points

1. **uniform_prob_by_count** - Should be zero (T-094)
2. **sample_scenario** - Minimal (T-095)
3. **noises.to_vec** - Zero (T-096)
4. **state.clone** - Zero (T-097)
5. **forward_costs.clone** - Zero (T-098)
6. **HashSet** - Zero or minimal (T-099)
7. **past_realizations** - Zero (T-100)

## Testing Requirements

### Profiling

- [ ] DHAT completes without errors
- [ ] Output file is valid

### Validation

- [ ] Results reproducible
- [ ] Categories match methodology

## Documentation Requirements

- [ ] Update `docs/HOT_PATH_ALLOCATION_AUDIT.md` with Sprint 7 results
- [ ] Create summary in epic overview

## Dependencies

- **Blocked By**: All Sprint 7 tickets (T-094 to T-100)
- **Blocks**: None
- **Related**: T-093 (Sprint 6 verification)

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Standard analysis work

## Definition of Done

- [ ] DHAT profiling complete
- [ ] Results analyzed
- [ ] ≥50% reduction verified
- [ ] Documentation updated
- [ ] PR merged
