# CLEANUP-009: Move Performance Comments to Module Documentation

## Context

The `src/sddp/mod.rs` file contains excellent performance analysis comments inline with the code (e.g., lines 2883-2892 analyzing Extract-and-Release pattern memory usage). While these are valuable, they interrupt code flow and are better suited for module-level documentation where they can be read once for understanding rather than encountered repeatedly during code navigation.

**Example Current State** (lines 2883-2892):
```rust
// PERFORMANCE: Extract-and-Release pattern with map_init
// - Init closure: Creates ONE handler per thread (lazy allocation)
// - Map closure: Runs forward pass, extracts trajectory, returns lightweight data
// - Handler is reused across scenarios on same thread
// - Handler is automatically dropped when thread finishes
//
// Memory: O(threads) × 6MB + O(scenarios) × 96KB
//   vs old O(scenarios) × 6MB
//
// Result: 96% memory reduction for large simulations
```

**Risk Level**: VERY LOW (documentation reorganization only)

## Acceptance Criteria

- [ ] All detailed performance comments identified in src/sddp/mod.rs
- [ ] Performance analysis section added to module-level documentation
- [ ] Inline performance comments condensed to brief references
- [ ] Module documentation includes:
  - Memory usage patterns
  - Threading model
  - Performance characteristics
  - Optimization decisions and trade-offs
- [ ] Code remains readable with condensed comments
- [ ] No functional changes to code
- [ ] Documentation builds correctly (`cargo doc`)

## Tasks

### Discovery
- [ ] Search for "PERFORMANCE" comments in src/sddp/mod.rs:
  ```bash
  rg "PERFORMANCE|Memory:|O\(" src/sddp/mod.rs
  ```
- [ ] Identify other multi-line comments explaining performance decisions
- [ ] Identify comments explaining threading model or memory patterns
- [ ] Identify comments explaining algorithmic complexity
- [ ] Create inventory of performance-related comments to consolidate

### Module Documentation Structure
- [ ] Design structure for performance documentation section:
  ```rust
  //! ## Performance Characteristics
  //!
  //! ### Memory Usage
  //! Details about memory patterns, allocations, thread-local storage
  //!
  //! ### Threading Model
  //! Rayon-based parallelism, thread pool sizing, work distribution
  //!
  //! ### Algorithmic Complexity
  //! Time/space complexity for key operations
  //!
  //! ### Optimization Decisions
  //! Key design decisions and trade-offs
  ```

### Content Migration

#### 1. Extract-and-Release Pattern (lines ~2883-2892)
- [ ] Move detailed analysis to module doc section "Memory Usage"
- [ ] Replace inline comment with brief reference:
  ```rust
  // Extract-and-Release pattern: O(threads) memory instead of O(scenarios)
  // See module docs for detailed analysis
  ```

#### 2. Threading Model Comments
- [ ] Consolidate Rayon threading explanations
- [ ] Document num_threads configuration
- [ ] Explain thread pool vs work-stealing behavior
- [ ] Move to "Threading Model" section

#### 3. Kahan Summation and Numerical Stability
- [ ] Consolidate floating-point accuracy comments
- [ ] Explain why specific numerical methods are used
- [ ] Document deterministic operation guarantee
- [ ] Move to "Algorithmic Complexity" or new "Numerical Stability" section

#### 4. Cut Selection and Batching
- [ ] Consolidate comments about cut pool performance
- [ ] Document batch processing trade-offs
- [ ] Move to "Optimization Decisions" section

### Inline Comment Condensation
- [ ] Replace each verbose performance comment with 1-2 line summary
- [ ] Add reference to module docs: "See module docs for details"
- [ ] Ensure code structure remains understandable without verbose comments
- [ ] Keep critical "gotcha" comments inline (safety, edge cases)

### Example Transformations

#### Before:
```rust
// PERFORMANCE: Extract-and-Release pattern with map_init
// - Init closure: Creates ONE handler per thread (lazy allocation)
// - Map closure: Runs forward pass, extracts trajectory, returns lightweight data
// - Handler is reused across scenarios on same thread
// - Handler is automatically dropped when thread finishes
//
// Memory: O(threads) × 6MB + O(scenarios) × 96KB
//   vs old O(scenarios) × 6MB
//
// Result: 96% memory reduction for large simulations
let results: Vec<_> = scenarios
    .par_iter()
    .map_init(|| ForwardPassHandler::new(), |handler, scenario| {
        handler.run(scenario)
    })
    .collect();
```

#### After:
```rust
// Extract-and-Release: O(threads) memory vs O(scenarios). See module docs.
let results: Vec<_> = scenarios
    .par_iter()
    .map_init(|| ForwardPassHandler::new(), |handler, scenario| {
        handler.run(scenario)
    })
    .collect();
```

And in module documentation:
```rust
//! ### Memory Usage: Extract-and-Release Pattern
//!
//! The simulation phase uses an "Extract-and-Release" pattern with Rayon's
//! `map_init` to minimize memory usage:
//!
//! - **Init closure**: Creates ONE handler per thread (lazy allocation)
//! - **Map closure**: Runs forward pass, extracts trajectory, returns lightweight data
//! - **Handler lifetime**: Reused across scenarios on same thread, dropped at thread completion
//!
//! **Memory Characteristics**:
//! - `O(threads) × 6MB` for handlers (typically 32-64MB for 8 threads)
//! - `O(scenarios) × 96KB` for trajectory data (3.2MB for 1000 scenarios)
//! - **Total**: ~35MB vs ~6GB with naive approach (96% reduction)
//!
//! This pattern is critical for large-scale simulations (10,000+ scenarios).
```

### Testing
- [ ] Run `cargo doc --open` to verify documentation builds
- [ ] Review rendered documentation for readability
- [ ] Ensure all code references in docs use correct formatting (backticks, links)
- [ ] Run `cargo test --doc` to verify doc examples compile
- [ ] Run `cargo clippy --all-targets` to check for doc comment issues
- [ ] Visual code review: Verify inline comments are still helpful

### Documentation
- [ ] Add section on performance characteristics to module docs
- [ ] Include relevant benchmarking commands in docs
- [ ] Link to benchmark results if available
- [ ] Document how to profile memory usage (reference scripts/)
- [ ] Add CHANGELOG.md entry: "Enhanced SDDP module documentation with performance characteristics"

## Technical Notes

### Module Documentation Best Practices

**✅ Module-Level Docs Should Include**:
- High-level architecture and design decisions
- Performance characteristics and trade-offs
- Threading model and concurrency safety
- Memory usage patterns
- Optimization strategies
- References to papers/algorithms

**✅ Inline Comments Should Include**:
- Brief context for non-obvious code
- Safety requirements or invariants
- Edge case handling
- Workarounds for specific issues
- References to detailed module docs

### Performance Documentation Template

```rust
//! ## Performance Characteristics
//!
//! ### Computational Complexity
//! - Training: O(iterations × stages × scenarios × states × actions)
//! - Simulation: O(scenarios × stages × cuts)
//!
//! ### Memory Usage
//! - Cut pool: O(iterations × stages × states) [typically 10-100MB]
//! - Forward pass: O(threads) using Extract-and-Release pattern
//! - Simulation: O(scenarios × trajectory_size) [typically 100MB for 10K scenarios]
//!
//! ### Threading Model
//! - Forward/backward passes parallelized across scenarios using Rayon
//! - Thread count configurable via `config.num_threads` (defaults to CPU count)
//! - Thread-safe cut storage using Arc<Mutex<CutPool>>
//!
//! ### Optimization Decisions
//! - Kahan summation for numerical stability in objective calculations
//! - Batch cut selection to amortize lock acquisition costs
//! - Lazy allocation of handlers to minimize memory overhead
//! - Deterministic RNG seeding for reproducibility
```

### Files to Review for Similar Comments

- [ ] `src/subproblem.rs` - Solver performance comments
- [ ] `src/cut.rs` - Cut selection performance
- [ ] `src/scenario.rs` - Scenario generation performance

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-008 (general comment cleanup), CLEANUP-011 (test comment consolidation)

## Estimated Effort

**1 story point** (4-5 hours, confidence: high)

Time breakdown:
- Discovery and inventory: 1 hour
- Module doc structure design: 1 hour
- Content migration: 2 hours
- Testing and review: 1 hour

Clear scope, well-defined task. Can be done in parallel with comment cleanup tickets.
