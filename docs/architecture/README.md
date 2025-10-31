# Architecture Documentation

This directory contains documentation for major design decisions, architectural patterns, and trade-off analyses in POWE.RS.

---

## Purpose

Architecture documents explain the **"why"** behind technical decisions:
- Rationale for choosing specific approaches
- Alternatives considered and rejected
- Trade-offs and their implications
- Future considerations for evolution

These docs help:
- **New contributors** understand design context
- **Maintainers** make consistent decisions
- **Users** understand performance characteristics
- **Researchers** adapt the code for their needs

---

## Documents

### [SOLVER.md](SOLVER.md) - Solver Integration Architecture

**Status**: ✅ Complete  
**Decision**: Use direct `highs-sys` FFI bindings instead of `highs` crate  
**Topics**:
- Rationale for low-level FFI approach
- Comparison with safe `highs` crate
- Key modifications for SDDP needs
- Performance implications
- Memory management strategy
- Future migration considerations

---

## Document Template

When creating new architecture documents, follow this structure:

```markdown
# [Title]: Clear, Descriptive Name

## Context
- What problem does this solve?
- What requirements drove the decision?
- When was this decision made?

## Decision
- What approach did we choose?
- Brief summary of the solution

## Rationale
- Why this approach?
- What makes it better than alternatives?

## Alternatives Considered
- What other options were evaluated?
- Why were they rejected?
- Comparison table if applicable

## Implementation Details
- Key technical details
- Code organization
- Integration points

## Trade-offs
### Benefits ✅
- Advantages of this approach

### Costs ❌
- Disadvantages and risks

## Performance Implications
- How does this affect runtime/memory?
- Benchmarks if available
- Comparison with alternatives

## Future Considerations
- When might we reconsider?
- Evolution path
- Related future work

## References
- Papers, documentation, discussions
```

---

## Related Documentation

- **Algorithm Theory**: [`../algorithm/`](../algorithm/) - SDDP mathematical background
- **Performance Analysis**: [`../performance/`](../performance/) - Profiling and optimization
- **User Guides**: [`../guides/`](../guides/) - Practical usage information
- **API Reference**: [`../reference/`](../reference/) - Detailed specifications

---

## Contributing

When making architectural decisions:

1. **Document early**: Write ADR before implementation
2. **Include alternatives**: Show what you considered
3. **Quantify trade-offs**: Provide data when possible
4. **Consider future**: How might this evolve?
5. **Link to code**: Reference implementation files

---

**Navigation**: [↑ Back to Documentation](../README.md) | [Repository Root](../../)
