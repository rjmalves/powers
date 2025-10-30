# CLEANUP-013: Extract solver.rs Comparison to Architecture Documentation

## Context

The `src/solver.rs` file contains 24 lines of module documentation explaining differences from the upstream `highs` crate. While this comparison is valuable for understanding design decisions, it's architectural context that belongs in `docs/architecture/SOLVER.md` rather than inline source documentation.

**Current State**: Module doc includes detailed comparison with upstream crate, explaining why POWE.RS uses direct `highs-sys` bindings instead of the higher-level `highs` crate.

**Goal**: Create proper architecture document, keep brief summary in source.

**Risk Level**: LOW (documentation reorganization)

## Acceptance Criteria

- [ ] Architecture document created at `docs/architecture/SOLVER.md`
- [ ] Detailed comparison and rationale documented
- [ ] Module doc comment condensed to 3-5 lines with reference
- [ ] Design decisions and trade-offs explained
- [ ] Performance implications documented
- [ ] CHANGELOG.md updated

## Tasks

### Analysis
- [ ] Read current `solver.rs` module documentation (lines 1-24)
- [ ] Identify key points in the comparison
- [ ] Determine what information is architectural vs API-level
- [ ] Check if `docs/architecture/` directory exists; create if needed
- [ ] Review other solver-related documentation for consistency

### Create docs/architecture/SOLVER.md

Structure:
```markdown
# Solver Integration Architecture

## Overview
High-level description of solver integration approach

## Design Decision: Direct highs-sys Bindings

### Rationale
Why we use highs-sys instead of the highs crate

### Trade-offs
Benefits and costs of this approach

### Comparison with highs Crate
Detailed comparison of the two approaches

## Implementation Details

### Memory Management
How we handle HiGHS objects and memory

### Error Handling
How we map HiGHS errors to Rust errors

### Thread Safety
Considerations for parallel solver calls

## Performance Implications

### Benchmarks
Performance comparison if available

### Optimization Opportunities
Future improvements

## Future Considerations

### Potential Migration
Conditions under which we might switch approaches

### Alternative Solvers
How to add support for other solvers

## References
HiGHS documentation, papers
```

- [ ] Extract comparison content from solver.rs module docs
- [ ] Expand on rationale for design decision
- [ ] Document trade-offs (performance, safety, maintainability)
- [ ] Add any additional architectural context not in module docs
- [ ] Document memory management approach
- [ ] Document error handling strategy
- [ ] Add performance characteristics
- [ ] Include references to HiGHS documentation

### Condense Module Documentation

**Target (3-5 lines)**:
```rust
//! Direct bindings to HiGHS solver via `highs-sys` for optimal performance.
//!
//! Uses C FFI for zero-cost LP/MIP solving in SDDP subproblems.
//! See [`docs/architecture/SOLVER.md`](../../docs/architecture/SOLVER.md)
//! for design rationale and comparison with `highs` crate.
```

- [ ] Write brief module summary
- [ ] Mention key characteristic (direct FFI bindings)
- [ ] Add reference to architecture documentation
- [ ] Remove detailed comparison from module docs

### Expand Architectural Content

Beyond what's in current module docs, add:

#### Design Context
- [ ] Document when the decision was made
- [ ] Explain requirements that drove the decision
- [ ] Mention alternatives considered

#### Technical Details
- [ ] Memory management strategy (Arc, lifetimes)
- [ ] Thread safety considerations
- [ ] Error handling approach
- [ ] Type conversions (Rust ↔ C)

#### Performance Analysis
- [ ] Overhead of different approaches
- [ ] Benchmark results if available
- [ ] Memory usage comparison

#### Future Considerations
- [ ] Conditions for reconsidering the decision
- [ ] Path to support alternative solvers
- [ ] Potential for abstraction layer

### Create docs/architecture/ Directory Structure

If directory doesn't exist:
```
docs/
  architecture/
    README.md        # Overview of architectural docs
    SOLVER.md        # This ticket
    THREADING.md     # Future: Rayon threading model
    MEMORY.md        # Future: Memory management patterns
```

- [ ] Create `docs/architecture/` directory if needed
- [ ] Create `docs/architecture/README.md` with index
- [ ] Add SOLVER.md to the index
- [ ] Update `docs/README.md` to reference architecture section

### Update Cross-References

- [ ] Link from solver.rs module docs to SOLVER.md
- [ ] Link from SOLVER.md back to solver.rs API docs
- [ ] Update docs/README.md table of contents
- [ ] Add reference in CONTRIBUTING.md about architectural documentation
- [ ] Check if README.md mentions solver integration; update if needed

### Testing & Validation
- [ ] Run `cargo doc --open` to verify module doc renders correctly
- [ ] Verify all links work (internal and external)
- [ ] Build full documentation
- [ ] Proofread SOLVER.md for clarity and accuracy
- [ ] Verify code examples in SOLVER.md compile (if any added)
- [ ] Check markdown formatting

### Documentation
- [ ] Add CHANGELOG.md entry: "Moved solver integration rationale to docs/architecture/SOLVER.md"
- [ ] Update CONTRIBUTING.md to mention architecture documentation:
  ```markdown
  ### Architecture Documentation (docs/architecture/)
  Document major design decisions, trade-offs, and architectural patterns.
  Include rationale, alternatives considered, and future considerations.
  ```

## Technical Notes

### Content to Extract vs Keep

**EXTRACT to docs/architecture/SOLVER.md**:
- Detailed comparison with `highs` crate
- Rationale for using `highs-sys`
- Trade-offs analysis
- Memory management details
- Thread safety considerations
- Performance benchmarks
- Future migration considerations
- Alternative solver support plans

**KEEP in solver.rs module docs**:
- Brief description: "Direct HiGHS bindings"
- Key characteristic: "Zero-cost FFI"
- Link to architecture docs

### Architecture Document Template

```markdown
# Title: Clear Description

## Context
What problem does this design solve? What are the requirements?

## Decision
What approach did we choose?

## Rationale
Why did we choose this approach?

## Alternatives Considered
What other options were evaluated? Why were they rejected?

## Trade-offs
What are the benefits and costs of this decision?

## Implementation Details
How is this implemented? Key technical details.

## Performance Implications
How does this affect performance? Benchmarks if available.

## Future Considerations
When might we revisit this decision? What would trigger a change?

## References
Links to relevant documentation, papers, discussions.
```

### Comparison Content Structure

For the highs vs highs-sys comparison:

```markdown
| Aspect | highs-sys (chosen) | highs crate |
|--------|-------------------|-------------|
| API Level | C FFI | Rust-idiomatic |
| Performance | Zero overhead | Small wrapper cost |
| Safety | Requires unsafe | Safe interface |
| Flexibility | Full HiGHS API | Subset of features |
| Maintenance | Track HiGHS C API | Track highs crate |
| Learning Curve | Requires C knowledge | Rust-native |

**Decision**: Use highs-sys for:
- Zero performance overhead in hot loop (subproblem solve)
- Access to full HiGHS C API
- Direct control over memory management
- Acceptable safety trade-off with careful encapsulation
```

### Related Documentation Files
- [ ] Check if `docs/guides/` has solver usage guide
- [ ] Check if examples demonstrate solver usage
- [ ] Verify INPUT-SPECIFICATION.md doesn't need solver details

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-012 (lognormal3 docs extraction)

## Estimated Effort

**0.5 story points** (2-3 hours, confidence: high)

Time breakdown:
- Create SOLVER.md structure: 0.5 hours
- Extract and expand content: 1 hour
- Condense module docs: 0.25 hours
- Cross-references and testing: 0.5 hours
- Review: 0.25 hours

Smaller scope than lognormal3 extraction (less content to move).
