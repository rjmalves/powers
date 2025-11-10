# TICKET-003: Integrate memory module into main codebase

## Context

This ticket integrates the newly created `memory` module into the main POWE.RS codebase. It ensures the module is properly exposed, documented, and accessible to other parts of the system. This includes updating the library root, creating comprehensive module documentation, and setting up examples.

**Why this matters**: Without proper integration, the memory management infrastructure cannot be used by algorithm code. This ticket ensures clean API boundaries and excellent developer experience.

**Part of**: Performance Implementation Plan - Phase 1: Core Buffer Management Infrastructure

**Depends on**: TICKET-001 (SizingInfo) and TICKET-002 (Buffer abstractions)

## Acceptance Criteria

- [ ] Given the memory module, when imported with `use powers::memory::*`, then all public types are accessible
- [ ] Given `cargo doc`, when built, then memory module documentation appears with examples
- [ ] Given a developer looking at the codebase, when they see `src/memory/mod.rs`, then they understand module purpose in <5 minutes
- [ ] Given example code in docs, when copied and run, then it compiles and executes correctly
- [ ] All public types are re-exported from `memory` module root
- [ ] Module follows Rust API guidelines for organization

## Tasks

### Implementation

- [ ] Update `src/lib.rs` to include memory module:
  - [ ] Add `pub mod memory;` declaration
  - [ ] Verify module appears in generated documentation
- [ ] Complete `src/memory/mod.rs` with comprehensive documentation:
  - [ ] Module-level doc comment with overview
  - [ ] Purpose section explaining allocation problem
  - [ ] Architecture section showing relationships
  - [ ] Key components list with links
  - [ ] Usage pattern with complete example
  - [ ] Performance notes section
- [ ] Add public re-exports in `src/memory/mod.rs`:
  - [ ] `pub use sizing::SizingInfo;`
  - [ ] `pub use buffers::{Buffer, BufferPool, ThreadLocalBuffers};`
  - [ ] `pub use buffers::{initialize_thread_local_buffers, with_thread_buffers};`
- [ ] Create `src/memory/README.md` with:
  - [ ] High-level overview
  - [ ] When to use this module
  - [ ] Quick start example
  - [ ] Link to full documentation
- [ ] Verify module organization follows best practices:
  - [ ] Check module hierarchy is logical
  - [ ] Ensure no circular dependencies
  - [ ] Verify private vs public boundaries

### Testing

- [ ] Integration test: Import and use SizingInfo from external code
- [ ] Integration test: Import and use Buffer from external code
- [ ] Integration test: Import and use BufferPool from external code
- [ ] Integration test: Import thread-local functions and use in parallel
- [ ] Documentation test: Verify all doc examples compile and run
- [ ] Build test: `cargo doc --no-deps --document-private-items` succeeds
- [ ] Lint test: `cargo clippy --all-targets` passes
- [ ] API test: Verify public API surface is minimal and intentional

### Documentation

- [ ] Write comprehensive module-level documentation including:
  - [ ] Problem statement (allocation overhead)
  - [ ] Solution approach (pre-allocation)
  - [ ] Usage workflow (compute sizes → allocate → use → reuse)
  - [ ] Performance impact (target metrics)
  - [ ] Example showing complete workflow
- [ ] Add architecture diagram in doc comment (ASCII art):
  - [ ] Input files → SizingInfo → BufferPools → Algorithm
  - [ ] Show data flow and dependencies
- [ ] Document relationships between components:
  - [ ] How SizingInfo feeds into buffer creation
  - [ ] When to use Buffer vs BufferPool vs ThreadLocalBuffers
  - [ ] Thread-local vs shared buffer patterns
- [ ] Add inline examples for each public type
- [ ] Add "See Also" section with links to:
  - [ ] PERFORMANCE_IMPLEMENTATION_PLAN.md
  - [ ] Related algorithm modules (sddp, subproblem)
- [ ] Create examples/memory_usage_example.rs (if appropriate)

## Technical Notes

### Module Organization

```
src/memory/
├── mod.rs           (public API, re-exports, module docs)
├── sizing.rs        (SizingInfo implementation)
├── buffers.rs       (Buffer, BufferPool, ThreadLocalBuffers)
└── README.md        (optional, for GitHub display)
```

### Public API Design

**Exported types**:
- `SizingInfo` - compute buffer dimensions
- `Buffer<T>` - single pre-allocated buffer
- `BufferPool<T>` - pool of reusable buffers
- `ThreadLocalBuffers` - thread-local buffer storage

**Exported functions**:
- `initialize_thread_local_buffers(sizing)` - initialize thread locals
- `with_thread_buffers(closure)` - access thread-local buffers

**Not exported** (keep private):
- Internal helper functions (compute_state_dimension, etc.)
- Implementation details

### Documentation Standards

Follow Rust documentation guidelines:
1. **One-line summary**: What the module does
2. **Detailed explanation**: Why it exists, how it works
3. **Examples**: Concrete usage code
4. **See Also**: Links to related items

### Module-Level Example

Include a complete, runnable example in module docs:

```rust
//! # Example: Complete Memory Management Workflow
//!
//! ```rust
//! use powers::memory::*;
//!
//! // 1. Compute buffer dimensions from input
//! let sizing = SizingInfo::from_input(&system, &graph, &recourse, &config);
//! sizing.log_summary();
//!
//! // 2. Pre-allocate buffers
//! let mut trajectory_pool = BufferPool::new(
//!     sizing.num_forward_passes,
//!     sizing.trajectory_buffer_size,
//! );
//!
//! // 3. Initialize thread-local buffers for parallel execution
//! initialize_thread_local_buffers(&sizing);
//!
//! // 4. Use buffers in hot path (zero allocation!)
//! for iteration in 0..max_iterations {
//!     let buffer = trajectory_pool.acquire(iteration);
//!     buffer.reset();
//!     // ... use buffer ...
//! }
//! ```
```

### Integration Checklist

Verify these integration points:
- [ ] Module appears in `cargo doc` output
- [ ] Module can be imported by other internal modules
- [ ] Module can be used by external code (as a library)
- [ ] No circular dependencies introduced
- [ ] No warnings about unused code
- [ ] No clippy warnings about API design

### Rust API Guidelines

Ensure module follows Rust conventions:
- **C-REEXPORT**: Re-export important types at crate root (if appropriate)
- **C-COMMON-TRAITS**: Implement common traits (Debug, Clone where appropriate)
- **C-GOOD-ERR**: Error types have good error messages
- **C-EXAMPLE**: All public items have examples
- **C-SEALED**: Traits that shouldn't be implemented externally are sealed

Reference: https://rust-lang.github.io/api-guidelines/

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 1.3
- Rust API Guidelines: https://rust-lang.github.io/api-guidelines/
- Rust module system: https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html

## Dependencies

- Blocked by: TICKET-001 (SizingInfo must exist)
- Blocked by: TICKET-002 (Buffer types must exist)
- Blocks: TICKET-005 (backward pass needs memory module available)
- Blocks: TICKET-008 (forward pass needs memory module available)
- Related: TICKET-004 (testing infrastructure)

## Estimated Effort

**2 story points** (1 day)

**Confidence**: High

**Breakdown**:
- Implementation: 0.25 day (mostly module declarations and re-exports)
- Documentation: 0.5 day (comprehensive module docs with examples)
- Testing: 0.25 day (integration tests, verify documentation examples)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all tests
- [ ] `cargo clippy --all-targets` produces no warnings
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc --no-deps` builds without warnings
- [ ] Module documentation is clear and comprehensive
- [ ] All doc examples are tested (doc tests pass)
- [ ] Public API is minimal and well-designed
- [ ] No circular dependencies
- [ ] Code reviewed by team member
- [ ] Developer can use module by reading docs alone (no need to read source)

## Notes

**Documentation Quality**: This module is foundational for performance work. Invest extra time in clear, comprehensive documentation with examples. Future developers should understand the module's purpose and usage within 5 minutes of reading the docs.

**API Stability**: Once this module is integrated and used by backward/forward passes, changing its API will be expensive. Get the public interface right in this ticket.
