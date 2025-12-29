# [T-016] Extract Bound Constraints (Optional)

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Constraint Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-012](./ticket-012-constraints-module-structure.md)
> **Blocks**: [T-017](./ticket-017-refactor-subproblem-facade.md)

---

## ⚠️ ANALYSIS FIRST

This ticket may be **optional** or **minimal** depending on how bounds are currently handled. Start with analysis to determine scope.

---

## Files to Read Before Starting

- `src/subproblem.rs` - Search for "bounds" or variable limits
- `src/solver.rs` - How bounds are set on variables
- `src/system.rs` - Capacity limits on generators, storage

---

## Context

### Background

Variable bounds (min/max limits) may be handled in different ways:

1. **At variable creation**: Bounds set when `add_col` is called
2. **As explicit constraints**: Separate row constraints for limits
3. **Mixed**: Some bounds on variables, some as constraints

This ticket analyzes the current approach and extracts bound-related logic if it exists as explicit constraints.

---

## Specification

### Analysis Phase

1. **Search for bound handling**:
   ```bash
   grep -n "bound\|limit\|capacity\|min_\|max_" src/subproblem.rs
   grep -n "add_col" src/subproblem.rs
   ```

2. **Identify patterns**:
   - Are bounds set at variable creation?
   - Are there explicit bound constraints?
   - Are bounds updated dynamically?

3. **Document findings** in implementation

### Possible Outcomes

#### Outcome A: Bounds at Variable Creation (No Extraction Needed)

If bounds are set via `add_col(lower..upper, ...)`:
- No constraint extraction needed
- Document finding
- Mark ticket as N/A

#### Outcome B: Explicit Bound Constraints

If there are explicit constraints like `x <= max_capacity`:
- Create `BoundsBuilder` in `src/model/constraints/bounds.rs`
- Extract constraint building logic

#### Outcome C: Dynamic Bound Updates

If bounds are updated during `prepare_from_trajectory`:
- Document the update pattern
- Consider if extraction makes sense

---

## Implementation (If Needed)

If explicit bound constraints exist, create:

```rust
// src/model/constraints/bounds.rs

use super::ConstraintContext;

/// Builder for variable bound constraints.
///
/// Creates explicit bound constraints for variables that cannot use
/// simple column bounds (e.g., time-varying limits).
pub struct BoundsBuilder;

impl BoundsBuilder {
    /// Build storage capacity constraints.
    pub fn build_storage_bounds(ctx: &mut ConstraintContext) -> Vec<usize> {
        // Implementation based on findings
        todo!()
    }
    
    /// Build generation capacity constraints.
    pub fn build_generation_bounds(ctx: &mut ConstraintContext) -> Vec<usize> {
        // Implementation based on findings
        todo!()
    }
}
```

---

## Acceptance Criteria

### If Bounds Are At Variable Creation (Outcome A):
- [ ] Analysis documented
- [ ] Ticket marked as N/A or completed with "no extraction needed"
- [ ] No code changes

### If Explicit Bound Constraints Exist (Outcome B):
- [ ] `BoundsBuilder` created
- [ ] Constraint logic extracted
- [ ] Golden tests pass

### If Dynamic Updates (Outcome C):
- [ ] Analysis documented
- [ ] Recommendation made for future work
- [ ] No extraction in this ticket

---

## Implementation Guide

### Suggested Approach

1. **Analyze current bound handling**:
   ```bash
   cd /home/rogerio/git/powers
   
   # Find all bound-related code
   grep -n "bound" src/subproblem.rs
   grep -n "capacity" src/subproblem.rs
   
   # Check how variables are added
   grep -n "add_col" src/subproblem.rs | head -20
   ```

2. **Check the add_variables function**:
   - Look at lines around `add_variables` (line ~2177)
   - See how bounds are specified

3. **Document findings**:
   - If no extraction needed, update ticket and close
   - If extraction needed, implement

4. **If implementing**, follow T-013/T-014 pattern

### Key Files to Analyze

| File | What to Look For |
|------|------------------|
| `src/subproblem.rs` | Bound constraints, capacity limits |
| `src/solver.rs` | `add_col` signature, bound specification |
| `src/system.rs` | Physical limits (storage, generation) |

### Expected Finding

Based on typical LP formulations, bounds are usually:
- **Storage**: `0 <= stored_volume <= max_capacity` (at variable creation)
- **Generation**: `0 <= thermal_gen <= max_generation` (at variable creation)
- **Flow**: `0 <= turbined_flow <= max_turbining` (at variable creation)

If this is the case, **no extraction is needed**.

---

## Testing Requirements

### If No Extraction:
- [ ] Document analysis results
- [ ] Verify golden tests still pass (no changes made)

### If Extraction:
- [ ] Unit tests for extracted constraints
- [ ] Golden tests pass

---

## Documentation Requirements

- [ ] Document analysis findings
- [ ] Explain why extraction was/wasn't needed
- [ ] If extracted, document constraint types

---

## Effort Estimate

**Points**: 2
**Confidence**: Medium (depends on analysis findings)
**Rationale**: Analysis first, implementation only if needed

---

## Definition of Done

- [ ] Analysis complete
- [ ] Decision made (extract or not)
- [ ] If extracting: implementation complete
- [ ] If not extracting: documented why
- [ ] Golden tests pass
