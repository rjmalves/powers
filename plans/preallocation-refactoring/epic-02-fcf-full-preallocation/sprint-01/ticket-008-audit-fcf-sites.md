# [TICKET-008] Audit FCF instantiation sites

> **Epic**: [Epic 2: FCF Full Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-009](./ticket-009-ensure-with-capacity.md)

## Context

### Background

`FutureCostFunction::with_capacity()` exists but may not be used everywhere. We need to find all instantiation sites to ensure complete preallocation.

### Relation to Epic

This ticket identifies all locations that need modification.

## Files to Read Before Starting

- `src/fcf.rs` - FutureCostFunction implementation (lines 50-120)
- `src/sddp/mod.rs` - Main SDDP algorithm
- `src/sddp/builder.rs` - SDDP construction

## Specification

### Task

Search codebase for all FCF instantiation patterns:
- `FutureCostFunction::new()`
- `FutureCostFunction::default()`
- `FutureCostFunction { ... }` (direct construction)
- `FutureCostFunction::with_capacity()` (already correct)

### Output

Document all sites with:
- File and line number
- Current instantiation method
- Whether SizingInfo is available at that point
- Recommended fix

## Acceptance Criteria

- [ ] All FCF instantiation sites identified
- [ ] Each site documented with location and current method
- [ ] SizingInfo availability assessed for each site
- [ ] Fix recommendations documented

## Implementation Guide

### Search Commands

```bash
# Find all FCF references
cd /home/rogerio/git/powers
grep -rn "FutureCostFunction" src/

# Find instantiations
grep -rn "FutureCostFunction::new" src/
grep -rn "FutureCostFunction::default" src/
grep -rn "FutureCostFunction::" src/

# Find struct construction
grep -rn "FutureCostFunction {" src/
```

### Expected Findings

Based on code review, likely locations:
1. `src/sddp/mod.rs` - Handler creation
2. `src/sddp/builder.rs` - Instance construction
3. Tests (if any)

### Documentation Template

For each site found:

```markdown
### Site N: [file:line]

**Current code**:
```rust
let fcf = FutureCostFunction::new();
```

**SizingInfo available**: Yes/No

**Recommended fix**:
```rust
let fcf = FutureCostFunction::with_capacity(
    num_forward_passes,
    num_iterations,
    max_state_dimension,
);
```

**Notes**: [Any complications]
```

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Simple grep-based audit

## Definition of Done

- [ ] Audit complete
- [ ] All sites documented
- [ ] Findings shared with TICKET-009
