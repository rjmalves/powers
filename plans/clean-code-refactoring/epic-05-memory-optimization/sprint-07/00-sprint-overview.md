# Sprint 7: Rust Application Allocation Optimization

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

After Sprint 6 optimizes HiGHS allocations (94.7%), this sprint targets the remaining **2% of allocations from Rust application code** (~1.8 GB). While small in comparison, these allocations can still be eliminated for deterministic memory behavior.

### Key Allocation Sites (from HOT_PATH_ALLOCATION_AUDIT.md)

| Allocation Site | Impact | Strategy |
|-----------------|--------|----------|
| `uniform_prob_by_count()` | HIGH | Compute into preallocated buffer |
| `sample_scenario()` Vec allocations | MEDIUM | Thread-local scenario buffers |
| `noises.to_vec()` cloning | MEDIUM | Pass slice reference |
| `state.clone()` in `compute_cut_data()` | MEDIUM | State staging buffer |
| `forward_costs.clone()` | LOW | Move instead of clone |
| `past_realizations` Vec | LOW | Preallocated trajectory buffer |
| HashSet allocations in FCF | LOW | BitVec or Vec-based set |

---

## Goals

1. **Eliminate `uniform_prob_by_count()` allocations** by computing into preallocated buffers
2. **Preallocate scenario sampling buffers** to avoid per-iteration allocations
3. **Remove `noises.to_vec()` clone** by passing slice references
4. **Reduce state cloning** in cut computation
5. **Optimize HashSet usage** in cut selection

## Non-Goals

- Further HiGHS optimization (covered in Sprint 6)
- Algorithm changes
- External API changes

---

## Technical Approach

### 1. Replace `uniform_prob_by_count()` Allocations

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

### 2. Thread-Local Scenario Sampling Buffers

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

### 3. Remove `noises.to_vec()`

**Current**:
```rust
.map(|(handler, noises)| self.forward(noises.to_vec(), handler))
```

**Target**:
```rust
.map(|(handler, noises)| self.forward(noises, handler))  // Pass slice
```

### 4. State Staging Buffer

**Current**:
```rust
let mut visited_state = self.state.clone();  // Full clone
```

**Target**:
```rust
// Reuse staging buffer
staging_buffer.copy_from(&self.state);
```

### 5. Replace HashSet with BitVec

**Current**:
```rust
let mut new_cut_ids = HashSet::new();
```

**Target**:
```rust
// Use fixed-size bitvec for known max cut count
let mut new_cut_flags = BitVec::with_capacity(max_cuts);
```

---

## Sprint Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-094 | Replace uniform_prob_by_count with preallocated buffers | 3 | None |
| T-095 | Add thread-local scenario sampling buffers | 3 | None |
| T-096 | Remove noises.to_vec() clone in forward pass | 2 | None |
| T-097 | Implement state staging buffer for cut computation | 5 | None |
| T-098 | Replace forward_costs.clone() with move | 1 | None |
| T-099 | Replace HashSet with BitVec in cut selection | 3 | None |
| T-100 | Preallocate past_realizations trajectory buffer | 3 | None |
| T-101 | DHAT verification of Rust allocation reduction | 3 | T-094 to T-100 |

**Total**: 23 points

---

## Acceptance Criteria

### Sprint Completion

- [ ] `uniform_prob_by_count()` uses preallocated buffers
- [ ] Scenario sampling uses thread-local buffers
- [ ] `noises.to_vec()` eliminated
- [ ] State cloning reduced or eliminated
- [ ] HashSet replaced with BitVec in cut selection
- [ ] DHAT shows ≥50% reduction in Rust allocations
- [ ] All tests pass
- [ ] Golden tests pass

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Buffer size estimation wrong | Medium | Low | Use conservative max + resize if needed |
| BitVec introduces complexity | Low | Medium | Keep HashSet fallback |
| State staging lifetime issues | Medium | Medium | Careful borrow design |
| Slice lifetime issues | Low | Low | Rust compiler will catch |

---

## Key Files

| Component | Location |
|-----------|----------|
| `uniform_prob_by_count()` | `src/utils/mod.rs:269` |
| `sample_scenario()` | `src/scenario.rs:452` |
| Forward pass noises | `src/sddp/mod.rs:1952` |
| State clone | `src/subproblem.rs:1739` |
| HashSet in FCF | `src/fcf.rs:321-322` |
| Past realizations | `src/algorithm/forward_pass.rs:123` |

---

## Verification Steps

### DHAT Profiling

```bash
cargo build --release
valgrind --tool=dhat ./target/release/powers run examples/05-large-scale-brazilian
```

Compare Rust allocation categories before/after Sprint 7.

### Golden Tests

```bash
cargo test --release -- golden
```

---

## Definition of Done

- [ ] All tickets complete and merged
- [ ] DHAT shows ≥50% reduction in Rust allocations
- [ ] No numerical divergence
- [ ] No performance regression
- [ ] Documentation updated
- [ ] All tests pass
