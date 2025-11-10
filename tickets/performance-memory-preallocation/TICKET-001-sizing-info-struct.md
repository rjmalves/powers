# TICKET-001: Implement SizingInfo struct for buffer dimension computation

## Context

This is the foundational ticket for the memory pre-allocation optimization (Phase 1, Week 1). The `SizingInfo` struct is a critical component that computes all buffer dimensions at application startup from input configuration files (system.json, graph.json, recourse.json, config.json). This centralized sizing computation enables pre-allocation of all buffers used in hot paths, eliminating runtime allocations.

**Why this matters**: Profiling shows 5.28% CPU time spent in malloc/memset. By knowing exact buffer sizes upfront, we can pre-allocate everything once and reuse buffers across iterations, targeting <2% allocation overhead.

**Part of**: Performance Implementation Plan - Phase 1: Core Buffer Management Infrastructure

## Acceptance Criteria

- [ ] Given input configuration files, when `SizingInfo::from_input()` is called, then all buffer dimensions are computed correctly
- [ ] Given a realistic system configuration (156 hydros, 8 stages), when computing sizing, then memory estimate is within 10% of actual usage
- [ ] Given `SizingInfo` instance, when `log_summary()` is called, then comprehensive sizing information is logged at INFO level
- [ ] Given `SizingInfo` instance, when `estimate_memory_bytes()` is called, then returned value matches sum of individual buffer sizes
- [ ] Performance: `from_input()` completes in <10ms for largest expected configuration

## Tasks

### Implementation

- [ ] Create `src/memory/` module directory
- [ ] Create `src/memory/mod.rs` with module documentation
- [ ] Create `src/memory/sizing.rs` file
- [ ] Implement `SizingInfo` struct with all required fields:
  - [ ] System dimensions (num_hydros, num_thermals, num_buses, num_lines)
  - [ ] State space dimensions (state_dimension, max_ar_order)
  - [ ] Graph dimensions (num_stages, num_nodes, max_scenarios_per_node)
  - [ ] Training dimensions (max_iterations, num_forward_passes)
  - [ ] Simulation dimensions (num_simulations)
  - [ ] Parallelism (num_threads)
  - [ ] Derived dimensions (subproblem_var_count, etc.)
- [ ] Implement `SizingInfo::from_input()` method
- [ ] Implement helper function `compute_state_dimension()`
- [ ] Implement helper function `compute_variable_count()`
- [ ] Implement helper function `compute_constraint_count()`
- [ ] Implement `estimate_memory_bytes()` method
- [ ] Implement `log_summary()` method with formatted output
- [ ] Add `src/memory` module to `src/lib.rs`

### Testing

- [ ] Unit test: Verify state dimension computation for StorageOnly state space
- [ ] Unit test: Verify state dimension computation for StorageAndInflow with various AR orders
- [ ] Unit test: Verify variable count for system with 3 hydros, 2 thermals, 4 buses
- [ ] Unit test: Verify constraint count computation
- [ ] Unit test: Verify memory estimation for small test case (known values)
- [ ] Unit test: Verify memory estimation for realistic case (100-2000 MB range)
- [ ] Integration test: Load example/03-multistage and verify all dimensions
- [ ] Integration test: Load example/05-large-scale-brazilian and verify dimensions
- [ ] Property test: Memory estimate increases monotonically with system size
- [ ] Test: Verify thread count defaults to Rayon pool size when not specified

### Documentation

- [ ] Add comprehensive module-level doc comment for `src/memory/mod.rs`:
  - [ ] Purpose and motivation
  - [ ] Key components overview
  - [ ] Usage pattern example
  - [ ] Link to Performance Implementation Plan
- [ ] Add struct-level doc comment for `SizingInfo`
- [ ] Add doc comments for all public methods with examples
- [ ] Add doc comments for all public fields explaining what they represent
- [ ] Add code example in doc comment showing typical usage
- [ ] Document computational complexity of `from_input()` method
- [ ] Add inline comments explaining non-obvious calculations (e.g., AR lag buffer sizing)

## Technical Notes

### Implementation Approach

1. **Field Organization**: Group related fields together (system, state, graph, training, derived)
2. **Derived Dimensions**: Compute in constructor, don't expose setters (immutable after construction)
3. **Thread Count**: Use `rayon::current_num_threads()` as default if not specified in config
4. **Memory Estimation**: Account for all major buffers (cuts, trajectories, thread-local)

### Key Calculations

**State Dimension**:
```rust
match state_space {
    StateSpace::StorageOnly => num_hydros,
    StateSpace::StorageAndInflow => num_hydros + sum(ar_orders),
}
```

**Subproblem Variables**:
- Hydro: 3 per plant (generation, spillage, end-storage)
- Thermal: 1 per plant (generation)
- Deficit: 1 per bus
- Future cost: 1 (alpha variable)

**Subproblem Constraints**:
- Hydro balance: 1 per plant
- Bus balance: 1 per bus
- Line limits: 2 per line (forward/reverse)
- Future cost cuts: dynamic (not counted here)

### Edge Cases

- **Empty AR orders**: Handle systems with no autoregressive models (max_order = 0)
- **Zero threads**: Should default to 1 (or Rayon default), never 0
- **Missing config fields**: Use sensible defaults for optional parameters

### Performance Considerations

- All computations are O(n) where n is system size
- Should complete in microseconds even for large systems
- No allocations during computation (except result struct)

### Integration Points

- Requires access to: `System`, `Graph`, `RecourseStructure`, `SddpConfig` types
- Will be used by: `BufferPool` creation, `BackwardPassBuffers`, `ForwardPassBuffers`
- Should be computed once at application startup

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 1.1
- Related to profiling results in `PROFILING_ANALYSIS.md`
- Memory allocation overhead data in `PERFORMANCE_REFACTORING_PLAN.md`

## Dependencies

- Blocked by: None (foundational ticket)
- Blocks: TICKET-002 (Buffer Pool Abstractions)
- Blocks: TICKET-003 (Module Integration)
- Related: TICKET-004 (Testing Infrastructure)

## Estimated Effort

**3 story points** (2 days)

**Confidence**: High

**Breakdown**:
- Implementation: 1 day (straightforward data structure and calculations)
- Testing: 0.5 day (multiple test cases, property tests)
- Documentation: 0.5 day (comprehensive docs for foundational component)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` produces no warnings
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc --no-deps` builds without warnings
- [ ] All acceptance criteria met
- [ ] Code reviewed by team member
- [ ] Memory estimates validated against actual measurements
