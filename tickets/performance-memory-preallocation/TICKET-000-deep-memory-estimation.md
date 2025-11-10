# TICKET-000: Implement Deep Memory Estimation with DeepSizeEstimate Trait

## Status: 📋 READY TO START (Priority: CRITICAL)

**Created**: 2025-11-10  
**Priority**: P0 - CRITICAL (Blocks all nested allocation optimization work)  
**Estimated Effort**: 8 story points (1-2 days)  
**Confidence**: High

---

## Context

During TICKET-006 implementation, we discovered that the current `SizingInfo::estimate_memory_bytes()` dramatically underestimates actual memory usage because it only accounts for **stack sizes** using `std::mem::size_of`, completely missing **heap allocations** in nested structures.

### The Problem: Shallow vs Deep Estimation

**Current (Shallow)**:
```rust
std::mem::size_of::<BendersCut>() * num_cuts
// Returns: 56 bytes per cut (stack only)
```

**Reality (Deep)**:
```rust
BendersCut {
    coefficients: Vec<f64>,  // ← 156 × 8 = 1,248 bytes HEAP (NOT COUNTED!)
    // ... other fields
}
// Actual: 1,304 bytes per cut (23× underestimate!)
```

### Impact

| Metric | Current (Shallow) | Actual (Deep) | Error |
|--------|------------------|---------------|-------|
| Memory per cut | 56 bytes | 1,304 bytes | **23× underestimate** |
| Malloc overhead estimate | ~2% | ~8-10% | **4-5× underestimate** |
| Optimization target | Wrong (outer vecs) | Right (nested vecs) | Misguided strategy |

**Critical Issue**: We've been optimizing outer allocations (like unzip in TICKET-006) while the much larger nested allocations remain untouched. This ticket provides the foundation to identify and fix nested allocation bottlenecks.

### Why This Must Come First

1. **Accurate measurement**: Can't optimize what we can't measure
2. **Right priorities**: Reveals where the real allocation costs are
3. **Validates improvements**: Can measure actual vs estimated allocation reduction
4. **Blocks optimization work**: TICKET-006b and beyond need accurate sizing for pre-allocation

**See**: `MEMORY_OPTIMIZATION_STRATEGY.md` for complete technical analysis

---

## Acceptance Criteria

- [ ] Given any type implementing `DeepSizeEstimate`, when `estimate_heap_bytes()` is called, then it returns the total heap memory including all nested allocations
- [ ] Given `SizingInfo`, when `estimate_memory_bytes_deep()` is called, then the estimate is within 10% of actual measured memory usage
- [ ] Given `BendersCut`, when estimating size, then it accounts for `coefficients: Vec<f64>` heap allocation
- [ ] Given `CutStatePair`, when estimating size, then it recursively accounts for both cut and state nested allocations
- [ ] Given production workload (156 hydros, 192 forward passes), when comparing estimate to actual, then error is <10%
- [ ] Given the trait implementation, when adding new types, then pattern is clear and documented
- [ ] Performance: `estimate_heap_bytes_static()` completes in <1ms for planning purposes
- [ ] All existing tests pass (no behavior changes to existing code)

---

## Tasks

### Implementation - Phase 1: Trait Definition

- [ ] Create `src/memory/deep_sizing.rs` module
- [ ] Define `DeepSizeEstimate` trait with two methods:
  - [ ] `estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize` (dynamic)
  - [ ] `estimate_heap_bytes_static(sizing: &SizingInfo) -> usize` (static planning)
- [ ] Add comprehensive module-level documentation with:
  - [ ] Purpose and motivation
  - [ ] Usage examples
  - [ ] When to use dynamic vs static estimation
  - [ ] How to implement for new types
- [ ] Export trait from `src/memory/mod.rs`

### Implementation - Phase 2: Primitive Implementations

- [ ] Implement `DeepSizeEstimate` for `f64`:
  - [ ] Return `std::mem::size_of::<f64>()` (stack only)
  - [ ] Add doc comment explaining zero heap
- [ ] Implement `DeepSizeEstimate` for `usize`
- [ ] Implement `DeepSizeEstimate` for `bool`
- [ ] Implement `DeepSizeEstimate` for `i64`
- [ ] Add unit test: Verify primitives return correct stack size

### Implementation - Phase 3: Collection Implementations

- [ ] Implement `DeepSizeEstimate` for `Vec<f64>`:
  - [ ] Dynamic: Use actual `capacity()` × element size
  - [ ] Static: Use `sizing.max_state_dimension` × element size
  - [ ] Add doc comments explaining difference
- [ ] Implement `DeepSizeEstimate` for generic `Vec<T: DeepSizeEstimate>`:
  - [ ] Dynamic: Recursively sum element heap sizes
  - [ ] Static: Use estimated count × element static size
  - [ ] Handle empty vecs correctly
- [ ] Implement `DeepSizeEstimate` for `HashMap<K, V>`:
  - [ ] Account for capacity, load factor, and element sizes
  - [ ] Document HashMap overhead (buckets + entries)
- [ ] Add unit tests for collections:
  - [ ] Empty vec returns 0
  - [ ] Pre-allocated vec returns capacity × element_size
  - [ ] Nested vecs are recursively counted
  - [ ] HashMap overhead is included

### Implementation - Phase 4: Domain Type Implementations

- [ ] Implement `DeepSizeEstimate` for `BendersCut`:
  ```rust
  fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
      std::mem::size_of::<Self>() +
      sizing.max_state_dimension * std::mem::size_of::<f64>()
  }
  ```
  - [ ] Add detailed doc comment with calculation breakdown
  - [ ] Include example showing 156 hydros → 1,304 bytes
- [ ] Implement `DeepSizeEstimate` for `State` trait objects:
  - [ ] Account for StorageState vs StorageAndInflow variants
  - [ ] Use sizing to determine state dimension
- [ ] Implement `DeepSizeEstimate` for `CutStatePair`:
  - [ ] Recursively include cut and state
  - [ ] Add stack overhead for struct itself
- [ ] Implement `DeepSizeEstimate` for `BendersCutPool`:
  - [ ] Include all cuts in pool
  - [ ] Include HashMap overhead for active_cut_indices
  - [ ] Document calculation
- [ ] Implement `DeepSizeEstimate` for `Trajectory`:
  - [ ] Include all stages
  - [ ] Include action vectors per stage
  - [ ] Account for realization buffers
- [ ] Implement `DeepSizeEstimate` for `BackwardPassBuffers`:
  - [ ] Include all result buffers
  - [ ] Use static sizing for capacity planning
- [ ] Add unit tests for domain types:
  - [ ] BendersCut size matches hand calculation
  - [ ] CutStatePair includes both components
  - [ ] Trajectory accounts for all stages

### Implementation - Phase 5: SizingInfo Integration

- [ ] Add `estimate_memory_bytes_deep()` method to `SizingInfo`:
  ```rust
  pub fn estimate_memory_bytes_deep(&self) -> usize {
      let cuts = self.estimate_total_cuts() * 
                 BendersCut::estimate_heap_bytes_static(self);
      let trajectories = self.num_forward_passes * 
                        Trajectory::estimate_heap_bytes_static(self);
      let buffers = self.num_threads * 
                   ThreadLocalBuffers::estimate_heap_bytes_static(self);
      cuts + trajectories + buffers
  }
  ```
- [ ] Add `estimate_total_cuts()` helper method:
  - [ ] Conservative estimate based on convergence patterns
  - [ ] Document assumption (10 cuts/node × 30% survival rate)
- [ ] Update `log_summary()` to show both shallow and deep estimates:
  - [ ] Show difference to highlight nested allocation impact
  - [ ] Add warning if deep estimate is 3× shallow estimate
- [ ] Keep existing `estimate_memory_bytes()` for backwards compatibility
- [ ] Add deprecation notice on old method

### Testing - Unit Tests

- [ ] Test: Primitives return correct stack size
- [ ] Test: Empty Vec<f64> returns 0 heap bytes
- [ ] Test: Vec<f64> with capacity returns correct heap bytes
- [ ] Test: Nested Vec<Vec<f64>> recursively counts
- [ ] Test: BendersCut with 156 coefficients = 1,304 bytes
- [ ] Test: CutStatePair includes both cut and state sizes
- [ ] Test: HashMap includes overhead calculation
- [ ] Test: Static vs dynamic methods agree when capacity matches
- [ ] Test: SizingInfo deep estimate > shallow estimate (validation)

### Testing - Integration Tests

- [ ] Integration test: Load example/03-multistage and compute deep estimate:
  - [ ] Estimate within reasonable range (100-500 MB)
  - [ ] Deep estimate > shallow estimate by expected factor
- [ ] Integration test: Load example/05-large-scale-brazilian:
  - [ ] Estimate within 10-20% of actual (measure with profiler)
  - [ ] Log breakdown by component (cuts, trajectories, buffers)
- [ ] Integration test: Compare static vs dynamic estimation:
  - [ ] Create actual structures
  - [ ] Dynamic estimate matches measured size
  - [ ] Static estimate is conservative (≥ dynamic)

### Testing - Validation Against Actual Memory

- [ ] Create `src/bin/memory_validator.rs` binary:
  - [ ] Load a problem instance
  - [ ] Compute deep memory estimate
  - [ ] Run training to completion
  - [ ] Measure actual peak memory (using system APIs)
  - [ ] Report estimate vs actual with error percentage
  - [ ] Exit with error if estimate error > 15%
- [ ] Run validator on 03-multistage (small problem):
  - [ ] Document results in ticket
  - [ ] Verify estimate within 10%
- [ ] Run validator on 05-large-scale-brazilian (large problem):
  - [ ] Document results in ticket
  - [ ] Verify estimate within 10%
- [ ] Add validator to CI as smoke test (on small problem only)

### Documentation

- [ ] Add module-level docs to `src/memory/deep_sizing.rs`:
  - [ ] Explain shallow vs deep estimation problem
  - [ ] Show "allocation iceberg" analogy
  - [ ] Provide implementation guide for new types
  - [ ] Include complete usage example
- [ ] Add trait-level docs to `DeepSizeEstimate`:
  - [ ] When to use dynamic vs static
  - [ ] How to implement for new types
  - [ ] Performance considerations
- [ ] Update `src/memory/sizing.rs` docs:
  - [ ] Document both estimation methods
  - [ ] Explain when to use each
  - [ ] Add migration guide from shallow to deep
- [ ] Add section to `MEMORY_OPTIMIZATION_STRATEGY.md`:
  - [ ] Document actual measurements vs estimates
  - [ ] Include validation results
  - [ ] Mark TICKET-000 as complete with results
- [ ] Update `CHANGELOG.md`:
  - [ ] Add entry under "Performance" section
  - [ ] Note: Internal change, no user-visible API change

---

## Technical Notes

### Design Decisions

**Why Two Methods (Dynamic vs Static)?**

- **Dynamic** (`estimate_heap_bytes(&self)`): For actual instances, uses real capacity
- **Static** (`estimate_heap_bytes_static(sizing: &SizingInfo)`): For planning, uses max expected sizes

This enables both accurate measurement of existing structures and conservative pre-allocation planning.

**Why Pass `SizingInfo`?**

Nested structures need context to estimate sizes:
- `Vec<f64>` in `BendersCut` should be sized to `max_state_dimension`
- `Vec<CutStatePair>` in buffers should be sized to `num_forward_passes × (num_stages - 1)`

The `SizingInfo` provides this context without hardcoding magic numbers.

### Implementation Approaches

**Generic Vec Implementation**:
```rust
impl<T: DeepSizeEstimate> DeepSizeEstimate for Vec<T> {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        // Vec overhead: capacity × size_of<T>
        let vec_overhead = self.capacity() * std::mem::size_of::<T>();
        
        // Element heap allocations (recursive)
        let elements_heap: usize = self.iter()
            .map(|item| item.estimate_heap_bytes(sizing))
            .sum();
        
        vec_overhead + elements_heap
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        // Conservative: use max expected count
        let max_count = sizing.estimate_max_vec_size::<T>();
        let vec_overhead = max_count * std::mem::size_of::<T>();
        let elements_heap = max_count * T::estimate_heap_bytes_static(sizing);
        
        vec_overhead + elements_heap
    }
}
```

**Specialized Vec<f64> Implementation**:
```rust
impl DeepSizeEstimate for Vec<f64> {
    fn estimate_heap_bytes(&self, _sizing: &SizingInfo) -> usize {
        // Primitives have no nested heap allocations
        self.capacity() * std::mem::size_of::<f64>()
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        // For coefficient vectors, use max state dimension
        sizing.max_state_dimension * std::mem::size_of::<f64>()
    }
}
```

### Edge Cases to Handle

1. **Empty collections**: Should return 0, not panic
2. **Unallocated vecs**: `Vec::new()` has 0 capacity
3. **Over-allocated vecs**: Capacity may exceed length
4. **Trait objects**: `Box<dyn State>` needs special handling
5. **Sparse structures**: Not all nodes may allocate all buffers

### Performance Considerations

- **Static estimation**: Should be fast (<1ms) for planning
- **Dynamic estimation**: May traverse large structures, acceptable for profiling
- **No allocation**: Estimation itself must not allocate
- **Cache-friendly**: Linear traversal where possible

### Validation Strategy

**Three-Level Validation**:

1. **Unit tests**: Verify math for known structures
2. **Integration tests**: Verify estimates on real examples
3. **Runtime validation**: Compare estimate to actual memory usage

**Acceptance Threshold**: 10% error is acceptable because:
- Actual memory includes allocator overhead
- Rust collections may over-allocate for growth
- System allocator varies by platform

### References

- See `MEMORY_OPTIMIZATION_STRATEGY.md` Section 3 for complete trait design
- See TICKET-006 completion report for shallow estimation limitations
- See profiling results showing actual 8-10% malloc overhead (not 2%)

---

## Dependencies

### Blocked By

None - This is the foundation ticket

### Blocks

- **TICKET-006b**: Nested pre-allocation (needs accurate sizing)
- **TICKET-007**: Performance validation (needs deep estimation for metrics)
- **TICKET-008**: Forward pass optimization (needs deep sizing)
- **TICKET-009**: Simulation optimization (needs deep sizing)

### Related

- **TICKET-001**: Provides `SizingInfo` base that this extends
- **TICKET-002**: Buffer pools will use deep sizing for accurate capacity

---

## Estimated Effort

**8 story points (1-2 days)**

**Breakdown**:
- Trait definition and docs: 2 hours
- Primitive implementations: 1 hour
- Collection implementations: 3 hours
- Domain type implementations: 4 hours
- SizingInfo integration: 2 hours
- Unit tests: 3 hours
- Integration tests: 2 hours
- Validation binary: 2 hours
- Documentation: 2 hours
- **Total**: ~21 hours (2.5 days)

**Confidence**: High
- Pattern is well-defined in strategy document
- Implementation is straightforward (math + recursion)
- No algorithm complexity or solver interaction
- Risk: Low (purely additive, no existing behavior changes)

---

## Validation Checklist

Before marking this ticket complete:

- [ ] All unit tests pass (>20 tests for trait implementations)
- [ ] Integration tests pass on both small and large examples
- [ ] Memory validator runs and reports <10% error on 05-large-scale-brazilian
- [ ] `cargo test` passes all 486 existing tests (no regressions)
- [ ] `cargo clippy` produces no warnings
- [ ] `cargo fmt --check` passes
- [ ] Documentation is comprehensive with examples
- [ ] `MEMORY_OPTIMIZATION_STRATEGY.md` updated with validation results
- [ ] TICKET-006b is unblocked and ready to start

---

## Success Metrics

**Technical**:
- ✅ Deep estimation within 10% of actual memory
- ✅ All major types implement `DeepSizeEstimate`
- ✅ Validation binary confirms accuracy

**Impact**:
- ✅ Reveals true memory allocation patterns (8-10% malloc overhead, not 2%)
- ✅ Enables accurate pre-allocation sizing in TICKET-006b
- ✅ Provides foundation for all subsequent optimization work

**Quality**:
- ✅ Zero unsafe code
- ✅ Comprehensive documentation
- ✅ Pattern is clear and extensible
- ✅ No existing tests broken

---

## Notes

This ticket is **CRITICAL** because it changes our optimization strategy from guessing to measuring. The discovery during TICKET-006 that nested allocations dominate (23× more memory than estimated) means we need this foundation before continuing optimization work.

**Key Insight**: We can't optimize allocation patterns we can't measure. This ticket provides the measurement infrastructure that makes all subsequent work data-driven instead of assumption-driven.

**Post-Completion**: After this ticket, we'll have accurate data showing where the real allocation costs are, enabling focused optimization in TICKET-006b (cut coefficient pre-allocation) and beyond.
