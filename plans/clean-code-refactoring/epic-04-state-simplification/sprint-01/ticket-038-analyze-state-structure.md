# [T-038] Analyze state.rs Structure and Document Allocation Points

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Sprint**: [Sprint 1: State Consolidation](./00-sprint-overview.md)
> **Dependencies**: Epic 3 Complete
> **Blocks**: [T-039](./ticket-039-document-state-cut-relationship.md), [T-040](./ticket-040-extract-state-utilities.md)

## Files to Read Before Starting

- `src/state.rs` - Full file (3,087 lines)
- `src/fcf.rs` - FutureCostFunction and state pool usage
- `src/subproblem.rs` - State usage in subproblem solving
- `docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md` - Related architecture context

---

## Context

### Background

Before refactoring `state.rs`, we need a comprehensive analysis of its structure, the duplication patterns between implementations, and all allocation points. This ticket produces the documentation that guides subsequent refactoring tickets.

### Current State (Post Epic 3)

`state.rs` contains 3,087 lines with this approximate breakdown:
- State trait definition: ~260 lines
- VisitedStatePool: ~118 lines  
- StateLayout: ~86 lines
- StorageState: ~352 lines
- StorageAndInflowState: ~586 lines
- Tests: ~1,506 lines

Two implementations exist: `StorageState` (storage-only) and `StorageAndInflowState` (storage + lagged inflows).

---

## Specification

### Outputs

Create a section in this ticket (or update the epic overview) documenting:

1. **Duplication Analysis**
   - List all methods that have identical implementations in both state types
   - Categorize: (a) trivial getters/setters, (b) shared logic, (c) type-specific logic
   - Estimate lines saved by extracting common code

2. **Allocation Point Inventory**
   | Location | Function | Allocation Type | Frequency | Hot Path? |
   |----------|----------|-----------------|-----------|-----------|
   | file:line | function_name | Vec/Box/String | per-X | Yes/No |

3. **State-Cut Relationship Summary**
   - How states are stored in `VisitedStatePool`
   - How cuts reference states
   - Current slot indexing mechanism

4. **Recommendations**
   - Which extractions provide most value
   - Risk areas to watch during refactoring

---

## Acceptance Criteria

- [ ] All methods in `State` trait catalogued by implementation type
- [ ] Duplication between `StorageState` and `StorageAndInflowState` quantified
- [ ] All `Box<dyn State>` creation points documented with file:line
- [ ] All `Vec::new()` / `vec![]` in State implementations documented
- [ ] Slot indexing mechanism documented
- [ ] Recommendations for refactoring prioritized

---

## Implementation Guide

### Suggested Approach

1. **Catalog trait methods**:
   ```bash
   grep -n "fn " src/state.rs | grep -v "^.*//\|test"
   ```

2. **Compare implementations side-by-side**:
   - StorageState impl: lines 669-993
   - StorageAndInflowState impl: lines 1123-1560

3. **Find allocation patterns**:
   ```bash
   grep -n "Vec::new\|vec!\|Box::new\|clone()" src/state.rs
   ```

4. **Document clone_dyn usage**:
   ```bash
   grep -rn "clone_dyn" src/
   ```

### Key Patterns to Identify

**Identical implementations** (extract to shared module):
- Domination tracking methods (get/set dominating_objective, dominating_cut_id)
- Iteration tracking (get/set iteration, forward_pass_idx)
- Basic coefficient access (coefficients, update_coefficients, reset_to_zero)

**Similar but different** (may need generic extraction):
- `evaluate_cut` - same algorithm, different coefficient structure
- `extract_storage_from_trajectory` - same pattern, different layout

**Type-specific** (cannot extract):
- `add_cut_constraint_to_model` - depends on variable structure
- `get_lagged_observations` - only in StorageAndInflowState

---

## Testing Requirements

- [ ] No code changes in this ticket (analysis only)
- [ ] Documentation added to this ticket or epic overview

---

## Documentation Requirements

- [ ] Complete the analysis tables in this ticket
- [ ] Update epic overview if findings change scope

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Reading and documenting existing code, no changes

---

## Definition of Done

- [ ] Duplication analysis complete with line counts
- [ ] All allocation points documented
- [ ] State-Cut relationship documented
- [ ] Prioritized recommendations ready
- [ ] Ready for T-039 and T-040

---

## Analysis Results (To Be Completed)

### Duplication Analysis

| Method | StorageState Lines | StorageAndInflowState Lines | Identical? |
|--------|--------------------|-----------------------------|------------|
| `set_dimension` | | | |
| `get_dominating_objective` | | | |
| `set_dominating_objective` | | | |
| `get_dominating_cut_id` | | | |
| `set_dominating_cut_id` | | | |
| `get_iteration` | | | |
| `set_iteration` | | | |
| `get_forward_pass_idx` | | | |
| `set_forward_pass_idx` | | | |
| `coefficients` | | | |
| `update_coefficients` | | | |
| `reset_to_zero` | | | |
| `dimension` | | | |
| `evaluate_cut` | | | |
| `evaluate_cut_ref` | | | |
| `compute_cut_data` | | | |
| `clone_dyn` | | | |

### Allocation Points

| Location | Function | Type | Frequency | Hot Path? |
|----------|----------|------|-----------|-----------|
| | | | | |

### Recommendations

1. **High Priority**: 
2. **Medium Priority**: 
3. **Low Priority**: 
