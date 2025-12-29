# [T-006] Analyze subproblem.rs Extraction Points

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Solution Extraction](./00-sprint-overview.md)
> **Dependencies**: Epic 1 complete
> **Blocks**: [T-007](./ticket-007-variable-indices-struct.md), [T-008](./ticket-008-solution-extractor-scaffold.md)

---

## ⚠️ CRITICAL: Analysis Only

This ticket is **analysis only**—no code changes. The goal is to understand the extraction points in `subproblem.rs` before making any modifications. Document your findings thoroughly as they will guide subsequent tickets.

---

## Files to Read Before Starting

- `src/subproblem.rs` - The main file to analyze (6,631 lines)
- `src/solver.rs` - Solution struct definition
- `src/state.rs` - State trait and implementations
- `src/sddp/mod.rs` - How subproblem is used in SDDP loop
- `plans/clean-code-refactoring/00-master-plan.md` - Overall refactoring goals

---

## Context

### Background

Before extracting code, we need a complete map of what to extract and how it interconnects. The `subproblem.rs` file has grown to 6,631 lines with multiple responsibilities mixed together. This analysis will identify:

1. All solution extraction functions and their dependencies
2. All constraint building functions and their dependencies
3. Data structures that need to be shared between modules
4. Hidden dependencies and potential pitfalls

### Current State

`subproblem.rs` contains:
- `Variables` struct (line ~700): LP variable indices
- `Constraints` struct (line ~740): LP constraint indices
- 11 `get_*_from_solution` functions (lines 1862-2046)
- Constraint building functions (lines 2355-2584)
- Much more interleaved logic

---

## Specification

### Deliverable

Create an analysis document at `docs/extraction-analysis.md` containing:

1. **Solution Extraction Functions Map**
2. **Constraint Building Functions Map**
3. **Data Dependencies Graph**
4. **Extraction Order Recommendation**
5. **Risk Assessment**

### Analysis Tasks

#### 1. Map All Solution Extraction Functions

For each `get_*_from_solution` function, document:

| Function | Lines | Input Types | Output Target | Depends On |
|----------|-------|-------------|---------------|------------|
| `get_deficit_from_solution` | 1862-1872 | `&Solution`, `&mut Realization` | `realization.deficit` | `variables.deficit` |
| ... | ... | ... | ... | ... |

**Expected functions** (verify against code):
- `get_deficit_from_solution`
- `get_net_exchange_from_solution`
- `get_thermal_gen_from_solution`
- `get_spillage_from_solution`
- `get_turbined_flow_from_solution`
- `get_final_storage_from_solution`
- `get_load_from_solution`
- `get_inflow_from_solution`
- `get_water_values_from_solution`
- `get_lag_duals_from_solution`
- `get_marginal_cost_from_solution`
- `populate_initial_state_fields`
- `get_current_stage_objective` (if exists)

#### 2. Map Variable/Constraint Dependencies

Document which fields of `Variables` and `Constraints` each function uses:

```
get_deficit_from_solution:
  - Reads: self.variables.deficit (Vec<usize>)
  - Writes: realization.deficit (via clone_from_slice)
  - Access pattern: contiguous range [first..last+1]
```

#### 3. Identify Extraction Boundaries

For each function, determine:
- Can it be extracted as-is? (pure function of inputs)
- Does it need `&self` access? (needs struct context)
- Does it mutate state beyond output? (side effects)

#### 4. Document the Realization Struct

Find and document the `Realization` struct:
- Location in codebase
- All fields that solution extraction populates
- Size/capacity patterns

#### 5. Constraint Building Analysis

For constraint functions, document:
- `add_constraints` (line 2355)
- `add_uncertainty_observation_constraints` (line 2495)
- What data they need from `system`, `variables`, etc.

#### 6. Risk Assessment

Identify potential issues:
- Functions with hidden side effects
- Shared mutable state
- Performance-sensitive hot paths
- Functions that seem too entangled to extract cleanly

---

## Acceptance Criteria

- [ ] Analysis document created at `docs/extraction-analysis.md`
- [ ] All 11+ solution extraction functions documented
- [ ] All constraint building functions documented
- [ ] Data dependency graph included
- [ ] Extraction order recommendation provided
- [ ] Risks and concerns documented
- [ ] No code changes made

### Correctness Verification

- [ ] `cargo build` still works (no changes made)
- [ ] `cargo test` still passes (no changes made)
- [ ] Golden tests still pass (no changes made)

---

## Implementation Guide

### Suggested Approach

1. **Open `subproblem.rs`** and create a section-by-section outline

2. **Find all `get_*_from_solution` functions**:
   ```bash
   grep -n "fn get_.*_from_solution" src/subproblem.rs
   ```

3. **Find constraint building functions**:
   ```bash
   grep -n "fn add_\|fn build_" src/subproblem.rs
   ```

4. **Find struct definitions**:
   ```bash
   grep -n "^pub struct\|^struct" src/subproblem.rs
   ```

5. **Find `Realization` definition**:
   ```bash
   grep -rn "struct Realization" src/
   ```

6. **Analyze each function**:
   - Read the function body
   - Note all `self.X` accesses
   - Note all parameter usages
   - Determine if it can be made a free function

7. **Create dependency diagram** (text-based is fine):
   ```
   Variables ──→ get_deficit_from_solution ──→ Realization.deficit
            └──→ get_exchange_from_solution ──→ Realization.exchange
   ```

8. **Write extraction order** based on dependencies:
   - Start with functions that have minimal dependencies
   - Group related functions together

### Key Files to Analyze

| File | Purpose |
|------|---------|
| `src/subproblem.rs:700-750` | Variables and Constraints structs |
| `src/subproblem.rs:1862-2100` | Solution extraction functions |
| `src/subproblem.rs:2355-2584` | Constraint building functions |
| `src/solver.rs` | Solution struct definition |
| `src/sddp/mod.rs` | Realization struct (likely here) |

### Analysis Document Template

```markdown
# Extraction Analysis: subproblem.rs

## Overview

Total lines: 6,631
Functions analyzed: X
Structs analyzed: X

## Solution Extraction Functions

### 1. get_deficit_from_solution (lines 1862-1872)

**Signature**: `fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization)`

**Purpose**: Extract deficit values from LP solution into realization

**Dependencies**:
- `self.variables.deficit: Vec<usize>` - variable indices
- `solution.colvalue: Vec<f64>` - LP column values
- `realization.deficit: Vec<f64>` - output target

**Extraction Notes**:
- Pure extraction, no side effects
- Uses contiguous slice access
- Can be extracted to free function with indices parameter

...

## Constraint Building Functions

...

## Data Dependency Graph

```text
[diagram here]
```

## Recommended Extraction Order

1. **Phase 1** (no dependencies):
   - get_deficit_from_solution
   - get_spillage_from_solution
   ...

2. **Phase 2** (depends on Phase 1):
   ...

## Risks and Concerns

1. **Risk**: ...
   **Mitigation**: ...
```

### Pitfalls to Avoid

- ⚠️ Don't make any code changes—this is analysis only
- ⚠️ Don't miss hidden dependencies (look for `self.` accesses)
- ⚠️ Don't assume function behavior—read the actual code
- ⚠️ Don't skip the constraint functions—they're needed for Sprint 2
- ⚠️ Document ALL functions, not just the obvious ones

---

## Testing Requirements

### Verification

- [ ] Document is complete and readable
- [ ] All functions mentioned in epic overview are covered
- [ ] Dependency graph is accurate (spot-check 3 functions)
- [ ] Extraction order makes logical sense

### No Code Tests

This ticket produces documentation only. No code tests needed.

---

## Documentation Requirements

- [ ] Create `docs/extraction-analysis.md`
- [ ] Include table of contents
- [ ] Include line number references for easy navigation
- [ ] Include dependency diagram

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Analysis work with clear scope; main time is careful reading and documentation

---

## Definition of Done

- [ ] Analysis document complete
- [ ] All solution extraction functions mapped
- [ ] All constraint functions mapped
- [ ] Dependencies documented
- [ ] Extraction order recommended
- [ ] Risks identified
- [ ] No code changes
- [ ] Reviewed by team
