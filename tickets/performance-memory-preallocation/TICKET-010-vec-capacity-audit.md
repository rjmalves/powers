# TICKET-010: Audit and replace Vec::new() with Vec::with_capacity() in hot paths

## Context

This ticket systematically audits the codebase for `Vec::new()` calls in hot paths and replaces them with `Vec::with_capacity()` to eliminate incremental reallocations. While individual reallocations are fast, they accumulate over thousands of iterations. This "quick win" optimization complements the buffer pre-allocation work and catches allocation sites missed by previous tickets.

**Why this matters**: Growing vectors cause multiple reallocations as they exceed capacity. By pre-sizing vectors when we know (or can estimate) their final size, we eliminate reallocation overhead. This is low-risk, high-impact work that improves performance across the codebase.

**Part of**: Performance Implementation Plan - Phase 3: Forward Pass and Subproblem Optimization

**Depends on**: Phase 2 and 3 buffer work (to avoid conflicts)

## Acceptance Criteria

- [ ] Given all hot path files audited, when `Vec::new()` usage is reviewed, then all have justified capacity decisions
- [ ] Given replaced `Vec::with_capacity()` calls, when profiled, then reallocation overhead is reduced
- [ ] Given modified code, when tested, then all functionality remains correct
- [ ] Performance: Reduction in reallocation-related overhead measurable in profiling
- [ ] Code quality: All capacity choices documented with rationale

## Tasks

### Implementation

- [ ] Create audit tracking document:
  - [ ] List of files to audit (hot paths)
  - [ ] Template for documenting decisions
  - [ ] Track progress (files completed)
- [ ] Audit `src/sddp/mod.rs`:
  - [ ] Search for `Vec::new()` calls
  - [ ] For each, determine if size is knowable
  - [ ] Replace with `with_capacity()` or document why not
  - [ ] Add comment explaining capacity choice
- [ ] Audit `src/subproblem.rs`:
  - [ ] Focus on constraint/variable creation
  - [ ] Focus on result collection
  - [ ] Replace or document each `Vec::new()`
- [ ] Audit `src/fcf.rs` (Future Cost Function):
  - [ ] Cut storage and retrieval
  - [ ] Active cut selection
  - [ ] Replace or document
- [ ] Audit `src/scenario_generator.rs`:
  - [ ] Scenario generation loops
  - [ ] SAA sample collection
  - [ ] Replace or document
- [ ] Audit `src/state.rs`:
  - [ ] State construction
  - [ ] Coefficient extraction
  - [ ] Replace or document
- [ ] Audit `src/cut.rs`:
  - [ ] Cut coefficient storage
  - [ ] Replace or document
- [ ] Audit `src/output/` modules:
  - [ ] Result aggregation
  - [ ] Report generation
  - [ ] Replace if hot path, document otherwise
- [ ] Add PERFORMANCE comments for each replacement:
  - [ ] Explain why capacity is chosen
  - [ ] Note allocation elimination
  - [ ] Reference profiling if relevant

### Testing

- [ ] Sanity test: Verify code compiles after changes
- [ ] Correctness test: Run full test suite
  - [ ] All 446+ tests must pass
  - [ ] No behavior changes expected
- [ ] Correctness test: Run integration tests
  - [ ] 03-multistage example
  - [ ] 05-large-scale-brazilian example
  - [ ] Verify numerical results unchanged
- [ ] Performance test: Profile before/after
  - [ ] Measure reallocation overhead with massif
  - [ ] Compare allocation counts
  - [ ] Verify reduction in malloc calls
- [ ] Performance test: Micro-benchmark affected functions
  - [ ] Benchmark functions with significant changes
  - [ ] Verify no regressions
  - [ ] Document improvements
- [ ] Stress test: Large-scale example
  - [ ] Run 50+ iterations
  - [ ] Verify memory usage is stable
  - [ ] Verify no leaks

### Documentation

- [ ] Create `VECTOR_CAPACITY_AUDIT.md`:
  - [ ] List all files audited
  - [ ] Document decisions for each Vec::new()
  - [ ] Rationale for each capacity choice
  - [ ] Measurements of impact
- [ ] Add inline comments for each with_capacity():
  - [ ] Explain capacity calculation
  - [ ] Note performance benefit
  - [ ] Example: "// Pre-size for num_nodes cuts (avoids 5+ reallocations)"
- [ ] Update PERFORMANCE_REFACTORING_PLAN.md:
  - [ ] Note completion of Vec audit
  - [ ] Document measured improvements
  - [ ] Add to Phase 3 completion
- [ ] Add entry to CHANGELOG.md:
  - [ ] "Performance: Pre-size vectors in hot paths to eliminate reallocations"

## Technical Notes

### Audit Methodology

**Step 1: Find all Vec::new() in hot paths**
```bash
# Find all Vec::new() calls (excluding tests)
rg "Vec::new\(\)" src/ --type rust -n | grep -v "test" | grep -v "#\[cfg(test)\]"

# Focus on hot path files
rg "Vec::new\(\)" src/sddp/ src/subproblem.rs src/fcf.rs src/scenario_generator.rs --type rust -n
```

**Step 2: Categorize each Vec::new()**

For each occurrence, ask:
1. **Is this in a hot path?** (executed thousands of times)
2. **Is the final size knowable?** (iteration count, input size, etc.)
3. **Is the size bounded?** (maximum possible size)
4. **What's the cost of over-allocation?** (memory vs performance trade-off)

**Step 3: Make decision**

| Category | Action | Example |
|----------|--------|---------|
| Hot path + Known size | Replace with `with_capacity(exact_size)` | Loop over nodes: `Vec::with_capacity(nodes.len())` |
| Hot path + Bounded size | Replace with `with_capacity(max_size)` | Active cuts: `Vec::with_capacity(total_cuts)` |
| Hot path + Estimated size | Replace with `with_capacity(estimate)` | Results: `Vec::with_capacity(est_result_count)` |
| Cold path | Leave as `Vec::new()` | Startup code, one-time init |
| Unknown size + small | Leave as `Vec::new()` | Rare error collections |

### Replacement Patterns

**Pattern 1: Loop with known iteration count**
```rust
// ❌ Before: Reallocates multiple times
let mut results = Vec::new();
for node in &self.nodes {
    results.push(process_node(node));
}

// ✅ After: Single allocation
// PERFORMANCE: Pre-size for self.nodes.len() to avoid reallocations
let mut results = Vec::with_capacity(self.nodes.len());
for node in &self.nodes {
    results.push(process_node(node));
}
```

**Pattern 2: Collect from iterator**
```rust
// ❌ Before: collect() doesn't know size
let results: Vec<_> = items.iter().filter(...).map(...).collect();

// ✅ After: Pre-size based on input
// PERFORMANCE: Pre-size for worst case (all items pass filter)
let mut results = Vec::with_capacity(items.len());
results.extend(items.iter().filter(...).map(...));

// OR if expensive to compute size:
let results: Vec<_> = items.iter().filter(...).map(...).collect();
// Keep as-is if iteration cost > allocation cost
```

**Pattern 3: Accumulation with estimated size**
```rust
// ❌ Before: Grows incrementally
let mut cuts = Vec::new();
for node in &self.nodes {
    cuts.extend(get_active_cuts(node));  // Unknown size per node
}

// ✅ After: Estimate based on average
// PERFORMANCE: Pre-size for estimated average of 10 cuts per node
let estimated_cuts = self.nodes.len() * 10;
let mut cuts = Vec::with_capacity(estimated_cuts);
for node in &self.nodes {
    cuts.extend(get_active_cuts(node));
}
```

**Pattern 4: Parallel collect**
```rust
// ❌ Before: Rayon doesn't pre-size
let results: Vec<_> = items.par_iter().map(...).collect();

// ✅ After: Use with_capacity if sequential
// Note: Can't directly pre-size parallel collect, but can use fold
let results = items.par_iter()
    .fold(
        || Vec::with_capacity(items.len() / rayon::current_num_threads()),
        |mut acc, item| {
            acc.push(process(item));
            acc
        }
    )
    .reduce(Vec::new, |mut a, b| { a.extend(b); a });

// OR: Accept that parallel collect doesn't pre-size (usually OK)
```

### Files and Priority

**High Priority** (definitely audit):
- `src/sddp/mod.rs` - Core algorithm loop
- `src/subproblem.rs` - Subproblem construction and solving
- `src/fcf.rs` - Cut management
- `src/scenario_generator.rs` - Scenario generation

**Medium Priority** (audit if time permits):
- `src/state.rs` - State extraction
- `src/cut.rs` - Cut operations
- `src/graph.rs` - Graph traversal
- `src/output/training_log.rs` - Result collection

**Low Priority** (likely cold paths):
- `src/input.rs` - One-time parsing
- `src/system.rs` - One-time construction
- `src/initial_condition.rs` - One-time setup

### Performance Validation

**Measure reallocation overhead**:
```bash
# Run with allocation profiler
valgrind --tool=massif --detailed-freq=1 ./target/release/powers examples/03-multistage

# Look for reallocation events
ms_print massif.out.* | grep -A5 "realloc"

# Compare before/after
# Before: X reallocation events
# After: Y reallocation events (expect 50-80% reduction in hot paths)
```

**Micro-benchmark**:
```rust
#[bench]
fn bench_collect_with_capacity(b: &mut Bencher) {
    let items: Vec<_> = (0..1000).collect();
    b.iter(|| {
        let mut results = Vec::with_capacity(items.len());
        for &item in &items {
            results.push(item * 2);
        }
        black_box(results)
    });
}

#[bench]
fn bench_collect_without_capacity(b: &mut Bencher) {
    let items: Vec<_> = (0..1000).collect();
    b.iter(|| {
        let mut results = Vec::new();
        for &item in &items {
            results.push(item * 2);
        }
        black_box(results)
    });
}
// Expect: with_capacity is 5-15% faster
```

### Trade-offs to Consider

**Over-allocation**:
- Risk: Allocate more memory than needed
- Mitigation: Use tight bounds or exact sizes when possible
- Acceptable: Temporary over-allocation in hot paths (reclaimed quickly)

**Code complexity**:
- Risk: More complex capacity calculations
- Mitigation: Clear comments explaining calculation
- Acceptable: Slight complexity increase for performance gain

**Maintainability**:
- Risk: Capacity assumptions may become invalid with code changes
- Mitigation: Document assumptions clearly
- Acceptable: Benefits outweigh maintenance cost

### Edge Cases

- [ ] Empty collections (capacity 0 is fine)
- [ ] Very large collections (>10M elements, be careful with capacity estimates)
- [ ] Parallel collection (Rayon handles sizing internally)
- [ ] Error paths (cold, can use Vec::new())

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 3.3
- Rust Vec documentation: https://doc.rust-lang.org/std/vec/struct.Vec.html#capacity-and-reallocation

## Dependencies

- Blocked by: TICKET-008, 009 (to avoid merge conflicts)
- Blocks: TICKET-011 (integration testing uses all optimizations)
- Related: All Phase 2 and 3 tickets (complementary optimization)

## Estimated Effort

**2 story points** (1 day)

**Confidence**: High

**Breakdown**:
- Audit and implementation: 0.5 day (search, replace, comment)
- Testing: 0.25 day (verify correctness, no regressions)
- Documentation: 0.25 day (document decisions, update plans)

## Validation Checklist

Before marking this ticket complete:

- [ ] All target files audited (checklist complete)
- [ ] All hot path `Vec::new()` replaced or documented
- [ ] All capacity choices have explanatory comments
- [ ] `cargo test` passes all tests
- [ ] Integration tests pass (numerical results unchanged)
- [ ] Performance measurements collected (allocation count reduced)
- [ ] No clippy warnings about unused capacity
- [ ] `cargo fmt --check` passes
- [ ] Code reviewed by team member
- [ ] VECTOR_CAPACITY_AUDIT.md created and complete
- [ ] CHANGELOG.md updated

## Notes

**Low Risk, Measurable Impact**: This is one of the safest optimizations. If capacity is too large, we slightly over-allocate (temporary). If too small, vector grows naturally. But getting it right eliminates reallocations entirely.

**Documentation is Key**: The value of this ticket isn't just the code changes—it's the documented rationale. Future developers will understand why each capacity was chosen.

**Quick Wins**: Many replacements are obvious (loop over known collection size). Focus on these first for quick impact.

**Don't Over-Optimize**: Cold paths (startup, error handling) don't need with_capacity(). Keep code simple where it doesn't matter.

**Measure Impact**: Use profiling to validate that this work actually reduces allocations. If impact is minimal, document why and move on.
