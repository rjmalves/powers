# TICKET-004: Create comprehensive test infrastructure for memory module

## Context

This ticket establishes testing infrastructure for the memory module, including unit tests, integration tests, property-based tests, and performance validation. Comprehensive testing is critical because this module is foundational—bugs here will propagate to all algorithm code that uses pre-allocated buffers.

**Why this matters**: The memory module's correctness directly impacts algorithm correctness. Buffer reuse bugs could cause data leakage between iterations, leading to wrong optimization results. Performance tests verify we're actually achieving the zero-allocation goal.

**Part of**: Performance Implementation Plan - Phase 1: Core Buffer Management Infrastructure

**Depends on**: TICKET-001, TICKET-002, TICKET-003 (needs implementations to test)

## Acceptance Criteria

- [ ] Given the memory module, when all tests run, then coverage is >90% for core functionality
- [ ] Given parallel buffer access, when tested with 100 threads, then no data races or leaks occur
- [ ] Given buffer reuse over 1000 iterations, when profiled, then zero allocations are detected
- [ ] Given property tests, when run with 1000 iterations, then all invariants hold
- [ ] All test files follow consistent naming and organization
- [ ] Test execution time is <5 seconds for unit tests
- [ ] Integration tests cover realistic usage patterns

## Tasks

### Implementation

- [ ] Create `src/memory/tests.rs` for unit tests
- [ ] Organize tests into modules:
  - [ ] `mod sizing_tests` - SizingInfo tests
  - [ ] `mod buffer_tests` - Buffer<T> tests
  - [ ] `mod pool_tests` - BufferPool tests
  - [ ] `mod thread_local_tests` - ThreadLocalBuffers tests
- [ ] Create `tests/integration/memory_integration.rs` for integration tests
- [ ] Add test utilities module:
  - [ ] `create_test_sizing()` - realistic SizingInfo for tests
  - [ ] `create_minimal_sizing()` - minimal valid SizingInfo
  - [ ] `create_large_sizing()` - large-scale test case
- [ ] Add property test framework (proptest crate)
- [ ] Create performance test harness (using criterion or custom)

### Testing - SizingInfo (TICKET-001)

- [ ] Unit test: Compute state dimension for StorageOnly
  - [ ] Given 3 hydros, when state_space=StorageOnly, then dimension=3
- [ ] Unit test: Compute state dimension for StorageAndInflow
  - [ ] Given 3 hydros with AR orders [2,3,1], then dimension=3+6=9
- [ ] Unit test: Compute variable count
  - [ ] Given 3 hydros, 2 thermals, 4 buses, then vars = 3*3 + 2 + 4 + 1 = 16
- [ ] Unit test: Compute constraint count
  - [ ] Given 3 hydros, 4 buses, 5 lines, then constraints = 3 + 4 + 5*2 = 17
- [ ] Unit test: Memory estimation for small system
  - [ ] Verify calculation matches sum of individual components
- [ ] Unit test: Memory estimation for realistic system (156 hydros)
  - [ ] Verify result is in 100-2000 MB range
- [ ] Unit test: Thread count defaults to Rayon pool size
- [ ] Unit test: Thread count respects config value when provided
- [ ] Integration test: Load example/03-multistage, verify all dimensions correct
- [ ] Integration test: Load example/05-large-scale-brazilian, verify dimensions

### Testing - Buffer<T> (TICKET-002)

- [ ] Unit test: Create buffer with capacity 100, verify length and capacity
- [ ] Unit test: Buffer::reset() sets all elements to default (0.0 for f64)
- [ ] Unit test: Buffer::clear() preserves capacity but sets length to 0
- [ ] Unit test: Buffer::resize() increases size correctly
- [ ] Unit test: Buffer::as_mut_slice() returns mutable reference
- [ ] Unit test: Buffer::as_slice() returns immutable reference
- [ ] Unit test: Multiple resets don't cause reallocation
- [ ] Property test: Buffer capacity never decreases after operations
- [ ] Property test: Reset always produces default values

### Testing - BufferPool<T> (TICKET-002)

- [ ] Unit test: Create pool with 5 buffers, verify count
- [ ] Unit test: Acquire cycles through buffers (0, 1, 2, 3, 4, 0, ...)
- [ ] Unit test: Buffers are independent (modify one, others unchanged)
- [ ] Unit test: Acquire out of bounds indexes (idx % len)
- [ ] Integration test: Pool with 100 buffers, acquire 500 times
- [ ] Property test: Pool size invariant (len() never changes)
- [ ] Performance test: Buffer acquire has O(1) time complexity

### Testing - ThreadLocalBuffers (TICKET-002)

- [ ] Unit test: Create ThreadLocalBuffers with realistic SizingInfo
- [ ] Unit test: All buffers have correct capacity
- [ ] Unit test: reset_all() clears all buffers
- [ ] Unit test: Initialize thread-local storage in single thread
- [ ] Unit test: Access thread-local buffers after initialization
- [ ] Unit test: Panic when accessing uninitialized thread-local
  - [ ] Verify error message is clear
- [ ] Integration test: ThreadLocalBuffers in parallel (10 threads)
  - [ ] Each thread gets independent buffers
  - [ ] No data leakage between threads
  - [ ] Buffer values from one thread don't appear in another
- [ ] Integration test: ThreadLocalBuffers with Rayon parallel iterator
- [ ] Stress test: 100 threads, 1000 operations each, verify independence
- [ ] Performance test: Thread-local access overhead is <10ns

### Testing - Integration Scenarios

- [ ] Integration test: Complete workflow
  - [ ] Compute SizingInfo from realistic input
  - [ ] Create BufferPools based on sizing
  - [ ] Initialize ThreadLocalBuffers
  - [ ] Simulate 10 iterations with buffer reuse
  - [ ] Verify no allocations after initial setup
- [ ] Integration test: Backward pass simulation
  - [ ] Create buffers for num_forward_passes trajectories
  - [ ] Simulate backward pass buffer usage
  - [ ] Verify buffer reuse across iterations
- [ ] Integration test: Forward pass simulation
  - [ ] Create buffers for trajectory storage
  - [ ] Simulate forward pass with buffer reuse
- [ ] Integration test: Mixed sequential and parallel access
  - [ ] Sequential BufferPool access
  - [ ] Parallel ThreadLocalBuffers access
  - [ ] No conflicts or data races

### Testing - Property-Based Tests

Use `proptest` crate for property testing:

- [ ] Property: Buffer capacity is monotonically non-decreasing
  - [ ] Forall operations, capacity(after) >= capacity(before)
- [ ] Property: Buffer reset always produces default values
  - [ ] Forall buffer states, after reset all elements are default
- [ ] Property: BufferPool size is constant
  - [ ] Forall pool operations, len() is unchanged
- [ ] Property: ThreadLocal buffers are independent
  - [ ] Forall parallel executions, buffer contents don't leak between threads
- [ ] Property: Memory estimate increases with system size
  - [ ] Forall system sizes (s1 > s2), memory(s1) > memory(s2)

### Testing - Performance Validation

- [ ] Performance test: Verify zero allocations during buffer reuse
  - [ ] Use `massif` or custom allocator to track allocations
  - [ ] Create buffers once
  - [ ] Perform 1000 reset/reuse cycles
  - [ ] Assert allocation count = 0 after initial allocation
- [ ] Performance test: Buffer operations are fast
  - [ ] acquire(): <10ns
  - [ ] reset(): <1µs for 1000-element buffer
  - [ ] with_thread_buffers(): <10ns
- [ ] Performance test: Memory estimation is fast
  - [ ] SizingInfo::from_input(): <1ms for realistic system
  - [ ] estimate_memory_bytes(): <1µs
- [ ] Benchmark: Compare buffer reuse vs fresh allocation
  - [ ] Measure time for 1000 allocations
  - [ ] Measure time for 1000 buffer reuses
  - [ ] Verify reuse is >10x faster

### Documentation

- [ ] Add comprehensive doc comment for `tests.rs` module
- [ ] Document test organization and naming conventions
- [ ] Document how to run specific test suites
- [ ] Document how to run performance tests
- [ ] Add README in tests directory explaining test strategy
- [ ] Document property test invariants being tested
- [ ] Add examples of running tests with different configurations:
  - [ ] `cargo test memory` - run all memory tests
  - [ ] `cargo test --test memory_integration` - integration tests only
  - [ ] `cargo test -- --nocapture` - see test output

## Technical Notes

### Test Organization

```
src/memory/tests.rs              (unit tests, co-located with code)
tests/integration/memory_integration.rs  (integration tests)
benches/memory_bench.rs          (performance benchmarks)
```

### Property Testing Strategy

Use `proptest` for property-based testing:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn buffer_capacity_never_decreases(
        initial_capacity in 1usize..10000,
        operations in prop::collection::vec(0usize..5, 0..100),
    ) {
        let mut buffer = Buffer::<f64>::with_capacity(initial_capacity);
        let initial_cap = buffer.capacity();
        
        for op in operations {
            match op {
                0 => buffer.clear(),
                1 => buffer.reset(),
                _ => buffer.resize(op * 10),
            }
            assert!(buffer.capacity() >= initial_cap);
        }
    }
}
```

### Performance Testing

Use custom allocator or `massif` to track allocations:

```rust
#[test]
fn test_zero_allocations_during_reuse() {
    // Pre-allocate
    let mut buffer = Buffer::<f64>::with_capacity(1000);
    
    // Start tracking allocations
    let allocs_before = get_allocation_count();
    
    // Reuse 1000 times
    for _ in 0..1000 {
        buffer.reset();
        // ... use buffer ...
    }
    
    // Check allocations
    let allocs_after = get_allocation_count();
    assert_eq!(allocs_before, allocs_after, "Buffer reuse caused allocations!");
}
```

### Integration Test Patterns

Test realistic usage patterns:

```rust
#[test]
fn test_complete_workflow() {
    // 1. Setup
    let sizing = create_test_sizing();
    let mut pool = BufferPool::new(sizing.num_forward_passes, 100);
    initialize_thread_local_buffers(&sizing);
    
    // 2. Simulate algorithm execution
    for iteration in 0..10 {
        // Forward pass
        let buffer = pool.acquire(iteration);
        buffer.reset();
        simulate_forward_pass(buffer);
        
        // Backward pass (parallel)
        let cuts: Vec<_> = (0..sizing.num_forward_passes)
            .into_par_iter()
            .map(|_| {
                with_thread_buffers(|buffers| {
                    simulate_backward_step(buffers)
                })
            })
            .collect();
    }
    
    // 3. Verify no data corruption
}
```

### Test Data Utilities

Create realistic test data:

```rust
fn create_test_sizing() -> SizingInfo {
    SizingInfo {
        num_hydros: 3,
        num_thermals: 2,
        num_buses: 4,
        num_lines: 5,
        state_dimension: 9,
        max_ar_order: 3,
        num_stages: 5,
        num_nodes: 20,
        max_scenarios_per_node: 3,
        max_iterations: 100,
        num_forward_passes: 10,
        num_simulations: 1000,
        num_threads: 4,
        // ... derived fields ...
    }
}
```

### Edge Cases to Test

- [ ] Empty buffers (capacity 0)
- [ ] Single-element buffers
- [ ] Very large buffers (>1GB)
- [ ] Thread-local in single-threaded context
- [ ] Pool with 1 buffer
- [ ] Pool with 10000 buffers
- [ ] Uninitialized thread-local access (should panic)
- [ ] Re-initialization of thread-local (should be safe)

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 1.4
- Proptest documentation: https://docs.rs/proptest/
- Rust testing guide: https://doc.rust-lang.org/book/ch11-00-testing.html

## Dependencies

- Blocked by: TICKET-001 (needs SizingInfo implementation)
- Blocked by: TICKET-002 (needs Buffer implementations)
- Blocked by: TICKET-003 (needs module integration)
- Blocks: None (but provides confidence for future work)
- Related: All future tickets depend on these tests for regression prevention

## Estimated Effort

**5 story points** (3 days)

**Confidence**: Medium (comprehensive testing takes time)

**Breakdown**:
- Implementation: 1.5 days (many test cases, test utilities)
- Property tests: 0.5 day (setting up proptest framework)
- Performance tests: 0.5 day (allocation tracking, benchmarking)
- Documentation: 0.5 day (test documentation, README)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all tests (>90% pass rate target)
- [ ] `cargo test memory` runs all memory module tests
- [ ] Property tests run successfully with 1000 iterations
- [ ] Performance tests verify zero-allocation guarantee
- [ ] Integration tests cover realistic usage patterns
- [ ] Code coverage >90% for memory module (use `cargo-tarpaulin` or similar)
- [ ] All edge cases tested
- [ ] Tests are well-documented and maintainable
- [ ] Test execution time <10 seconds for full suite
- [ ] Code reviewed by team member

## Notes

**Test Quality Over Quantity**: Focus on meaningful tests that catch real bugs. Each test should verify a specific behavior or invariant.

**Performance Testing**: The zero-allocation guarantee is critical. Invest time in reliable allocation tracking.

**Property Testing**: Properties are more valuable than individual test cases because they test invariants over a wide range of inputs.

**Maintenance**: Write tests that are easy to understand and maintain. Future developers should be able to understand what's being tested by reading the test name and assertion.
