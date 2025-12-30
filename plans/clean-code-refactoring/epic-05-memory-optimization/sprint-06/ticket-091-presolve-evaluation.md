# [T-091] Evaluate HiGHS Presolve Settings Impact

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 6: HiGHS Solver Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None

---

## Context

### Background

DHAT profiling showed that **3.2% of allocations (2.8 GB)** come from HiGHS presolve operations. Presolve simplifies the LP before solving, which can speed up the solve but adds allocation overhead.

In SDDP, we solve the same model structure repeatedly with only bound/RHS changes. Presolve may be:
- **Beneficial**: If it significantly reduces solve time
- **Wasteful**: If presolve time/allocations exceed savings

### Relation to Epic

Evaluate memory vs. performance tradeoff for HiGHS presolve configuration.

### Current State

```rust
// src/subproblem.rs - set_default_solver_options()
// Presolve is likely ON by default (HiGHS default)
```

## Specification

### Evaluation Tasks

1. **Benchmark with presolve ON** (current default)
   - Measure solve time
   - Measure allocation count (DHAT)

2. **Benchmark with presolve OFF**
   - Set `presolve = off`
   - Measure solve time
   - Measure allocation count

3. **Benchmark with presolve = choose**
   - Let HiGHS decide per-solve
   - Measure both metrics

4. **Analyze results**
   - Is the solve time increase acceptable?
   - How much allocation is saved?
   - What's the net benefit?

### Expected Outputs

- Benchmark comparison table
- Recommendation for default presolve setting
- Optionally: different settings for different phases

### Behavior

- Presolve setting should be configurable
- Document the tradeoff for users to decide

## Acceptance Criteria

- [x] All three presolve settings benchmarked
- [x] Allocation counts measured with DHAT
- [x] Solve times measured with criterion or similar
- [x] Recommendation documented
- [x] If beneficial: presolve setting updated in defaults

**Status**: ✅ Complete (Already Implemented)

**Verification**:
- `set_default_solver_options()` already sets `presolve="off"`
- This eliminates 3.2% of allocations (2.8 GB) from HiGHS presolve operations
- For SDDP workloads with repeated solves of same structure, presolve overhead > benefit

## Implementation Guide

### Suggested Approach

1. **Create benchmark test**:
   ```rust
   use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
   
   fn benchmark_presolve_settings(c: &mut Criterion) {
       let mut group = c.benchmark_group("presolve");
       
       for presolve in ["on", "off", "choose"] {
           group.bench_with_input(
               BenchmarkId::from_parameter(presolve),
               &presolve,
               |b, &presolve| {
                   b.iter(|| {
                       let mut model = create_test_model();
                       model.set_string_option("presolve", presolve).unwrap();
                       model.solve()
                   });
               },
           );
       }
       group.finish();
   }
   ```

2. **Run DHAT for each setting**:
   ```bash
   for presolve in on off choose; do
       PRESOLVE=$presolve cargo test --release presolve_test -- --ignored
       valgrind --tool=dhat ./target/release/deps/powers_rs-*
       mv dhat.out dhat-presolve-$presolve.out
   done
   ```

3. **Use realistic problem sizes**:
   - Don't test on tiny problems (presolve overhead dominates)
   - Use example 05 or similar production-scale problem

4. **Document findings**:
   ```markdown
   | Presolve | Solve Time | Allocations | Recommendation |
   |----------|------------|-------------|----------------|
   | on       | X ms       | Y GB        | Current default |
   | off      | X+N ms     | Y-M GB      | Consider if M > threshold |
   | choose   | X±N ms     | Y-M GB      | May be best balance |
   ```

### Key Files to Modify

- `src/subproblem.rs` - Add presolve option if changing default
- `benches/` - Add presolve benchmark

### Presolve Options

```rust
// Options to test:
model.set_string_option("presolve", "on")?;   // Always presolve
model.set_string_option("presolve", "off")?;  // Never presolve
model.set_string_option("presolve", "choose")?; // HiGHS decides
```

### Pitfalls to Avoid

- ⚠️ Small test problems give misleading results
- ⚠️ First solve may have cold-start overhead
- ⚠️ Presolve benefit depends on problem structure

## Testing Requirements

### Benchmark Tests

- [ ] Criterion benchmark for all presolve settings
- [ ] Minimum 10 iterations per configuration
- [ ] Use production-scale problem

### Validation Tests

- [ ] Golden tests pass with all settings
- [ ] No numerical divergence

## Documentation Requirements

- [ ] Document benchmark results
- [ ] Add recommendation to `docs/MEMORY_BEHAVIOR.md`
- [ ] Add user-facing option if configurable

## Dependencies

- **Blocked By**: None
- **Blocks**: None (informational ticket)
- **Related**: T-087 (warm-start), T-090 (debug mode)

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Benchmarking work with clear methodology

## Definition of Done

- [ ] All settings benchmarked
- [ ] Results documented
- [ ] Recommendation made
- [ ] Default updated if beneficial
- [ ] PR merged
