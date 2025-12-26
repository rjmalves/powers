# [TICKET-009] Ensure with_capacity usage everywhere

> **Epic**: [Epic 2: FCF Full Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-008](./ticket-008-audit-fcf-sites.md)  
> **Blocks**: [TICKET-010](./ticket-010-validate-memory.md)

## Context

### Background

Based on TICKET-008 audit, update all FCF instantiation sites to use `with_capacity()`.

### Relation to Epic

Core implementation ticket that enables FCF preallocation.

## Files to Read Before Starting

- TICKET-008 audit results
- `src/fcf.rs` - with_capacity() signature
- `src/memory/sizing.rs` - SizingInfo fields

## Specification

### Required Changes

For each site identified in TICKET-008:

1. Ensure SizingInfo is available (thread through if needed)
2. Replace `new()` / `default()` with `with_capacity()`
3. Use correct sizing parameters

### Pattern

```rust
// Find sizing info
let num_forward_passes = sizing.num_forward_passes;
let num_iterations = sizing.max_iterations;
let max_state_dim = sizing.max_state_dimension;

// Create with preallocation
let fcf = FutureCostFunction::with_capacity(
    num_forward_passes,
    num_iterations,
    max_state_dim,
);
```

### Threading SizingInfo

If SizingInfo not available at instantiation site:
1. Add `SizingInfo` parameter to constructor
2. Thread through from builder/initialization
3. Store in handler if needed for later use

## Acceptance Criteria

- [ ] All FCF instances use with_capacity()
- [ ] No new() or default() calls for FCF
- [ ] SizingInfo properly threaded where needed
- [ ] Examples still work correctly
- [ ] Code compiles without warnings

## Implementation Guide

### Suggested Approach

1. Start with highest-level instantiation (builder)
2. Thread SizingInfo down to FCF creation points
3. Replace instantiation calls
4. Verify compilation and examples

### Key Files to Modify

Based on likely locations:
- `src/sddp/builder.rs`
- `src/sddp/mod.rs`

### Code Changes

```rust
// Example change in builder.rs

// Before
pub fn build(self) -> Result<SddpInstance, String> {
    // ...
    let fcf = FutureCostFunction::new();
    // ...
}

// After
pub fn build(self) -> Result<SddpInstance, String> {
    // Compute sizing
    let sizing = SizingInfo::from_input(&system, &graph, &config);
    
    // ...
    let fcf = FutureCostFunction::with_capacity(
        sizing.num_forward_passes,
        sizing.max_iterations,
        sizing.max_state_dimension,
    );
    // ...
}
```

### Pitfalls to Avoid

- ⚠️ Ensure sizing is computed before FCF creation
- ⚠️ Use max_iterations, not current iteration
- ⚠️ Thread to all handler types (train, simulation)

## Testing Requirements

### Integration Tests

- [ ] Example 01: Produces identical results
- [ ] Example 07: Produces identical results
- [ ] No warnings about capacity

## Effort Estimate

**Points**: 2  
**Confidence**: Medium  
**Rationale**: May require SizingInfo threading

## Definition of Done

- [ ] All FCF instances use with_capacity()
- [ ] SizingInfo available at all creation sites
- [ ] Code compiles
- [ ] Examples work
