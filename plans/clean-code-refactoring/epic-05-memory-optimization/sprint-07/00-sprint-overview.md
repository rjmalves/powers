# Sprint 7: Comprehensive Memory Optimization

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint 6 Results**: [DHAT_SPRINT6_ANALYSIS.md](../../../../docs/DHAT_SPRINT6_ANALYSIS.md)
> **Batch Cut Analysis**: [BATCH_CUT_BOUNDS_ANALYSIS.md](../../../../docs/BATCH_CUT_BOUNDS_ANALYSIS.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

Sprint 6 achieved exceptional results (**48.5% byte reduction, 72.7% block reduction**) by:
1. Disabling `reuse_forward_basis()` → 95% HFactor reduction
2. Implementing batch bounds API → 99.6% block reduction in changeRowBounds

This sprint has three focus areas:
1. **Sprint 6 follow-up**: Complete removal of counterproductive code
2. **Remaining HiGHS optimization**: Investigate HEkkDual allocations (still 22+ GB)
3. **Rust allocation optimization**: Target remaining 5.44 GB from Rust code

### Post-Sprint 6 Allocation Breakdown

| Category | Bytes | Blocks | Notes |
|----------|-------|--------|-------|
| HEkkDual (fill_assign) | 22.33 GB | 3.71M | HiGHS internal - investigate |
| HEkkDual (default_append) | 12.57 GB | 4.12M | HiGHS internal - investigate |
| HSimplexNla (debug) | 4.66 GB | 0.60M | Possible debug remnants |
| Other HiGHS | 4.18 GB | 28.33M | Various |
| Single changeRowBounds | 0.17 GB | 0.37M | Cut bound updates - batch! |
| Rust/Powers | 5.44 GB | 20.2M | Target for optimization |

---

## Sprint Goals

### Priority 1: Sprint 6 Follow-up (Critical)
1. **Remove `reuse_forward_basis()` code entirely** - Confirmed 95% HFactor reduction when disabled
2. **Document basis reuse guidelines** - When IS basis reuse appropriate?

### Priority 2: Remaining HiGHS Investigation (High)
3. **Investigate HEkkDual allocations** - 35 GB remaining, may have optimization potential
4. **Investigate HSimplexNla debug allocations** - 4.66 GB suggests debug code still running
5. **Batch cut constraint bounds** - Eliminate remaining 0.17 GB single bounds calls

### Priority 3: Rust Allocation Optimization (Medium)
6. **Preallocated probability buffers** - `uniform_prob_by_count()` allocations
7. **Thread-local scenario buffers** - Per-iteration allocations
8. **Eliminate unnecessary clones** - `noises.to_vec()`, `forward_costs.clone()`

---

## Technical Approach

### 1. Remove `reuse_forward_basis()` Code

The function is currently commented out in `src/sddp/mod.rs`. Sprint 6 DHAT confirms it was counterproductive:
- Triggered HiGHS "alien basis" handling
- Forced full factorization rebuilds
- **Action**: Delete the function and all related code

### 2. Investigate HEkkDual Allocations

HEkkDual accounts for 35 GB of remaining allocations. Investigation areas:
- Are we triggering unnecessary dual simplex iterations?
- Is `simplex_dual_edge_weight_strategy` optimal?
- Can we reduce basis refactorization frequency?

### 3. Investigate HSimplexNla Debug Allocations

4.66 GB from HSimplexNla suggests debug code may still be active:
- Verify `Highs_setOptionValue("output_flag", "false")` is working
- Check if HiGHS was built with `NDEBUG` flag
- Investigate `HSimplexNla::debugCheckData` calls

### 4. Batch Cut Constraint Bounds

See [BATCH_CUT_BOUNDS_ANALYSIS.md](../../../../docs/BATCH_CUT_BOUNDS_ANALYSIS.md) for full analysis.

Refactor `apply_aggregated_cut_selection_result()` to:
1. Collect all cut additions into batch buffers
2. Collect all cut removals into batch buffers
3. Apply single `change_rows_bounds_batch()` call per category

### 5. Replace `uniform_prob_by_count()` Allocations

**Current** (allocates every call):
```rust
pub fn uniform_prob_by_count(count: usize) -> Vec<f64> {
    let p = 1.0 / count as f64;
    vec![p; count]
}
```

**Target** (compute in-place):
```rust
pub fn fill_uniform_probabilities(buffer: &mut [f64]) {
    let p = 1.0 / buffer.len() as f64;
    buffer.fill(p);
}
```

### 6. Thread-Local Scenario Sampling Buffers

**Current**:
```rust
let branching_indices: Vec<usize> = 
    self.index_samplers.iter().map(|d| d.sample(rng)).collect();
```

**Target**:
```rust
thread_local! {
    static BRANCHING_INDICES: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
}
```

### 7. Eliminate Unnecessary Clones

| Clone Site | Current | Target |
|------------|---------|--------|
| `noises.to_vec()` | Clone Vec per forward pass | Pass slice reference |
| `forward_costs.clone()` | Clone for IterationResult | Move ownership |

---

## Sprint Tickets

### Priority 1: Sprint 6 Follow-up

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-094 | Remove `reuse_forward_basis()` code entirely | 2 | None |
| T-095 | Document HiGHS basis reuse guidelines | 2 | T-094 |

### Priority 2: HiGHS Investigation

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-096 | Investigate HEkkDual allocation sources | 5 | None |
| T-097 | Investigate HSimplexNla debug allocations | 3 | None |
| T-098 | Batch cut constraint bound updates | 3 | None |

### Priority 3: Rust Allocations

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-099 | Preallocated probability buffers | 3 | None |
| T-100 | Thread-local scenario sampling buffers | 3 | None |
| T-101 | Eliminate noises.to_vec() and forward_costs.clone() | 2 | None |
| T-102 | Replace HashSet with BitVec in cut selection | 3 | None |

### Verification

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-103 | DHAT verification of Sprint 7 optimizations | 3 | All above |

**Total**: 29 points

---

## Acceptance Criteria

### Sprint Completion

- [ ] `reuse_forward_basis()` code removed entirely
- [ ] HEkkDual investigation complete with findings documented
- [ ] HSimplexNla debug allocations investigated
- [ ] Cut constraint bounds use batch API
- [ ] `uniform_prob_by_count()` uses preallocated buffers
- [ ] Scenario sampling uses thread-local buffers
- [ ] Unnecessary clones eliminated
- [ ] DHAT shows meaningful reduction in remaining allocations
- [ ] All tests pass
- [ ] Golden tests pass

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| HEkkDual allocations inherent to HiGHS | High | Medium | Document findings, accept as limitation |
| HSimplexNla debug in HiGHS binary | Medium | Medium | Rebuild HiGHS with NDEBUG if possible |
| Batch cut bounds breaks determinism | Low | High | Comprehensive golden test validation |
| Buffer size estimation wrong | Low | Low | Conservative sizing + resize if needed |

---

## Key Files

| Component | Location |
|-----------|----------|
| `reuse_forward_basis()` | `src/sddp/mod.rs` (commented) |
| HiGHS options | `src/subproblem.rs:set_default_solver_options()` |
| Cut selection result | `src/subproblem.rs:apply_aggregated_cut_selection_result()` |
| `uniform_prob_by_count()` | `src/utils/mod.rs:269` |
| `sample_scenario()` | `src/scenario.rs:452` |
| Forward pass noises | `src/sddp/mod.rs:1952` |
| HashSet in FCF | `src/fcf.rs:321-322` |

---

## HiGHS Options to Investigate

```rust
// Current options in set_default_solver_options()
model.set_option("presolve", "off");
model.set_option("solver", "simplex");
model.set_option("simplex_strategy", 1);        // Dual simplex
model.set_option("simplex_update_limit", 5000); // Refactorization frequency
model.set_option("simplex_price_strategy", 1);
model.set_option("simplex_scale_strategy", 0);  // No scaling
model.set_option("parallel", "off");
model.set_option("threads", 1);
model.set_option("simplex_dual_edge_weight_strategy", -1);  // Auto
model.set_option("simplex_primal_edge_weight_strategy", -1); // Auto
```

### Options to Test (T-096)

| Option | Current | Test Values | Expected Impact |
|--------|---------|-------------|-----------------|
| `simplex_update_limit` | 5000 | 1000, 10000 | May affect HEkkDual allocs |
| `simplex_dual_edge_weight_strategy` | -1 (auto) | 0, 1, 2 | May reduce dual iterations |
| `rebuild_refactor_solution` | default | true/false | May affect refactorization |

---

## Verification Steps

### DHAT Profiling

```bash
cargo build --release
valgrind --tool=dhat --dhat-out-file=dhat-sprint7.out \
    ./target/release/powers run examples/05-large-scale-brazilian
```

### Golden Tests

```bash
cargo test --release -- golden
```

### Benchmark

```bash
cargo bench --bench sddp_training
```

---

## Definition of Done

- [ ] All tickets complete and merged
- [ ] HiGHS investigation findings documented
- [ ] DHAT shows reduction in remaining allocations
- [ ] No numerical divergence (golden tests pass)
- [ ] No performance regression
- [ ] Documentation updated
- [ ] All tests pass

---

## Expected Outcomes

### Optimistic (if HEkkDual can be reduced)
- Additional 10-20 GB reduction
- Total allocations under 30 GB

### Realistic (HEkkDual inherent, Rust optimizations successful)
- 2-3 GB reduction from Rust allocations
- 0.2 GB reduction from batch cut bounds
- Total allocations ~42-43 GB

### Minimum (investigations yield no new optimizations)
- Confirm HiGHS allocations are at theoretical minimum
- Document limitations for future reference
- Clean up Sprint 6 follow-up items
