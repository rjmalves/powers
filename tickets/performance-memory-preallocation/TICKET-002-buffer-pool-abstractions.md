# TICKET-002: Implement Buffer Pool abstractions for memory reuse

## Context

This ticket implements the core buffer management abstractions that enable zero-allocation hot paths. The `Buffer<T>` and `BufferPool<T>` types provide type-safe, pre-allocated memory that can be borrowed and reused across iterations. The `ThreadLocalBuffers` struct provides thread-local storage for parallel execution with Rayon.

**Why this matters**: Each iteration currently allocates ~60 buffers (for 10 forward passes × 5 stages). By pre-allocating and reusing buffers, we eliminate all these allocations, targeting a 15-20% overall performance improvement.

**Part of**: Performance Implementation Plan - Phase 1: Core Buffer Management Infrastructure

**Depends on**: TICKET-001 (SizingInfo provides dimensions for buffer sizing)

## Acceptance Criteria

- [ ] Given a buffer capacity, when `Buffer::with_capacity(n)` is created, then buffer has space for n elements
- [ ] Given a buffer with data, when `reset()` is called, then all elements are set to default values
- [ ] Given a buffer with data, when `clear()` is called, then length is 0 but capacity is preserved
- [ ] Given a BufferPool with N buffers, when `acquire(i)` is called, then buffer at index i % N is returned
- [ ] Given ThreadLocalBuffers initialized for a thread, when `with_thread_buffers()` is called, then closure has access to thread-local buffers
- [ ] Given uninitialized ThreadLocalBuffers, when `with_thread_buffers()` is called, then it panics with clear error message
- [ ] Performance: Buffer reuse has zero allocation overhead (verified with profiler)
- [ ] Thread safety: ThreadLocalBuffers work correctly in parallel Rayon execution

## Tasks

### Implementation

- [ ] Create `src/memory/buffers.rs` file
- [ ] Implement `Buffer<T>` struct with generic type parameter
  - [ ] Add `data: Vec<T>` field
  - [ ] Implement `with_capacity(capacity: usize)` constructor
  - [ ] Implement `clear()` method (Vec::clear)
  - [ ] Implement `reset()` method (fill with T::default())
  - [ ] Implement `as_mut_slice()` accessor
  - [ ] Implement `as_slice()` accessor
  - [ ] Implement `resize(new_size: usize)` method
  - [ ] Add trait bounds: `T: Clone + Default`
- [ ] Implement `BufferPool<T>` struct
  - [ ] Add `buffers: Vec<Buffer<T>>` field
  - [ ] Add `next_available: AtomicUsize` for thread-safe cycling
  - [ ] Implement `new(count: usize, capacity: usize)` constructor
  - [ ] Implement `acquire(&mut self, idx: usize)` method
  - [ ] Implement `len()` method
  - [ ] Add trait bounds: `T: Clone + Default`
- [ ] Implement `ThreadLocalBuffers` struct
  - [ ] Add `realization_buffer: Buffer<f64>` field
  - [ ] Add `gradient_buffer: Buffer<f64>` field
  - [ ] Add `state_buffer: Buffer<f64>` field
  - [ ] Add `lag_buffer: Buffer<f64>` field
  - [ ] Add `cut_eval_buffer: Buffer<f64>` field
  - [ ] Implement `new(sizing: &SizingInfo)` constructor
  - [ ] Implement `reset_all()` method
- [ ] Implement thread-local storage
  - [ ] Add `thread_local!` static with `RefCell<Option<ThreadLocalBuffers>>`
  - [ ] Implement `initialize_thread_local_buffers(sizing: &SizingInfo)`
  - [ ] Implement `with_thread_buffers<F, R>(f: F) -> R`
  - [ ] Use `rayon::broadcast` for initialization
- [ ] Add exports to `src/memory/mod.rs`

### Testing

- [ ] Unit test: Buffer creation with capacity 100, verify length and capacity
- [ ] Unit test: Buffer reset sets all elements to default (0.0 for f64)
- [ ] Unit test: Buffer clear preserves capacity but sets length to 0
- [ ] Unit test: Buffer resize increases size correctly
- [ ] Unit test: BufferPool with 5 buffers, acquire cycling (0, 1, 2, 3, 4, 0, ...)
- [ ] Unit test: BufferPool buffers are independent (modification doesn't affect others)
- [ ] Unit test: ThreadLocalBuffers creation with realistic SizingInfo
- [ ] Unit test: ThreadLocalBuffers reset_all clears all buffers
- [ ] Unit test: ThreadLocalBuffers initialization in single thread
- [ ] Integration test: ThreadLocalBuffers in parallel Rayon execution (10 threads)
- [ ] Integration test: Each thread gets independent buffers (no data leakage)
- [ ] Property test: Buffer reuse maintains capacity after N operations
- [ ] Performance test: Verify zero allocations during buffer reuse (use profiler)
- [ ] Stress test: 1000 buffer acquire/reset cycles with no leaks

### Documentation

- [ ] Add module-level doc comment for `buffers.rs`
- [ ] Add doc comment for `Buffer<T>` with usage example
- [ ] Add doc comment for `BufferPool<T>` with usage example
- [ ] Add doc comment for `ThreadLocalBuffers` with parallel usage example
- [ ] Document `initialize_thread_local_buffers()` with initialization requirements
- [ ] Document `with_thread_buffers()` with panic conditions
- [ ] Add performance note about zero-cost abstractions and inlining
- [ ] Add thread-safety notes for BufferPool
- [ ] Add example showing typical workflow: create → acquire → use → reset
- [ ] Document why `AtomicUsize` is used in BufferPool (thread-safe cycling)

## Technical Notes

### Implementation Approach

1. **Generic Buffer**: Use `Vec<T>` internally for flexibility (f64, Cut, State, etc.)
2. **Type Constraints**: Require `Clone + Default` for reset functionality
3. **Zero-Cost Abstraction**: Methods should inline; verify with `#[inline]` annotations
4. **Thread-Local Pattern**: Use Rust's `thread_local!` macro with `RefCell` for interior mutability

### Buffer Lifecycle

```
Create (once) → Acquire → Use → Reset → Release → [Acquire again...]
```

**Key insight**: Reset is cheaper than allocation (just memset, keeps capacity)

### Thread-Local Design

**Why RefCell?**: Allows mutable access through immutable reference (required by thread_local!)

**Why Option?**: Explicit initialization step (not initialized until first use)

**Initialization**: Use `rayon::broadcast()` to initialize in all worker threads

### Edge Cases

- **Empty buffer**: `with_capacity(0)` is valid but not useful
- **Uninitialized thread-local**: Should panic with clear error message
- **Resize during use**: Avoid in hot paths; size buffers correctly initially
- **Thread safety**: `BufferPool::acquire` with mutable reference (single-threaded); ThreadLocalBuffers for parallel

### Performance Considerations

- **Inlining**: Mark hot methods with `#[inline]` (especially accessors)
- **Capacity preservation**: Never shrink capacity in reset operations
- **Thread-local overhead**: Small (thread ID lookup), amortized over many uses
- **Atomic counter**: In BufferPool for potential future thread-safe acquire (currently unused)

### Memory Layout

**Buffer<f64> with capacity 1000**:
- Size: 8 bytes/element × 1000 = 8KB
- Overhead: Vec metadata (~24 bytes)
- Total: ~8KB per buffer

**ThreadLocalBuffers** (typical sizing):
- realization_buffer: ~2KB (subproblem vars)
- gradient_buffer: ~1KB (cut coefficients)
- state_buffer: ~500 bytes (state dimension)
- lag_buffer: ~500 bytes (AR lags)
- cut_eval_buffer: ~400 bytes (scenarios)
- Total: ~5KB per thread

### Integration Points

- Uses: `SizingInfo` from TICKET-001
- Used by: `BackwardPassBuffers` (TICKET-005), `ForwardPassBuffers` (TICKET-008)
- Rayon integration: `rayon::broadcast` for thread initialization

### Testing Strategy

**Unit tests**: Verify individual buffer operations
**Integration tests**: Verify thread-local behavior in parallel execution
**Property tests**: Verify invariants (capacity preservation, independence)
**Performance tests**: Verify zero-allocation guarantee

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 1.2
- Rust thread-local documentation: https://doc.rust-lang.org/std/macro.thread_local.html
- Rayon broadcast: https://docs.rs/rayon/latest/rayon/fn.broadcast.html

## Dependencies

- Blocked by: TICKET-001 (needs SizingInfo type)
- Blocks: TICKET-003 (module integration needs these types)
- Blocks: TICKET-005 (backward pass buffers use BufferPool)
- Related: TICKET-004 (testing infrastructure)

## Estimated Effort

**5 story points** (3 days)

**Confidence**: High

**Breakdown**:
- Implementation: 1.5 days (generic types, thread-local setup is straightforward but needs care)
- Testing: 1 day (many test cases, parallel testing needs validation)
- Documentation: 0.5 day (critical to document thread-safety and usage patterns)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all tests including parallel tests
- [ ] `cargo clippy` produces no warnings
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc --no-deps` builds without warnings
- [ ] All acceptance criteria met
- [ ] Parallel test with 10 threads shows independent buffers
- [ ] Profile confirms zero allocations during buffer reuse
- [ ] Code reviewed by team member
- [ ] Thread-safety verified with ThreadSanitizer (if available)

## Notes

**IMPORTANT**: The thread-local buffer initialization must happen BEFORE any parallel execution. Add clear documentation and a panic message if used before initialization.

**Performance Validation**: Use `valgrind --tool=massif` or similar to confirm no allocations during buffer reuse phase.
