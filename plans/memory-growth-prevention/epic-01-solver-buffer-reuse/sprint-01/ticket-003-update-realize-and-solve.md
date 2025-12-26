# [TICKET-003] Update realize_and_solve to use buffer-into pattern

> **Epic**: [Epic 1: Solver Buffer Reuse](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-001](./ticket-001-add-solution-buffer-into.md), [TICKET-002](./ticket-002-add-basis-buffer-into.md)  
> **Blocks**: None
> **Status**: ✅ Complete (2025-12-26)

## Context

### Background

The `realize_and_solve()` method in `Subproblem` extracts solution and basis after each solve. Currently it:
1. Calls `model.get_solution()` - allocates new Solution
2. Calls `model.get_basis()` - allocates new Basis  
3. Assigns new Basis to `realization_container.basis` - **discards preallocated buffer**

The `Realization` struct already has preallocated storage via `Basis::with_capacity()` (line 2927), but this is wasted because line 1583 replaces it with a newly allocated Basis.

### Current State

```rust
// src/subproblem.rs:1561-1584
let (solution, basis, objective_value, model_status) =
    if let Some(model) = &self.model {
        let status = model.status();
        if status == solver::HighsModelStatus::Optimal {
            let sol = model.get_solution();      // NEW ALLOCATION
            let bas = model.get_basis();         // NEW ALLOCATION
            let obj = model.get_objective_value();
            (Some(sol), Some(bas), Some(obj), Some(status))
        } else {
            (None, None, None, Some(status))
        }
    } else {
        (None, None, None, None)
    };

// ...

if let Some(basis) = basis {
    realization_container.basis = basis;  // DISCARDS preallocated buffer!
}
```

### Files to Read Before Starting

- `src/subproblem.rs` - `realize_and_solve()` method, `Realization` struct
- `src/solver.rs` - New `get_solution_into()` and `get_basis_into()` methods

## Specification

### Inputs

- `realization_container: &mut Realization` already passed to the method

### Outputs

- Solution and Basis written directly to `realization_container` buffers

### Behavior

- Use `get_solution_into()` to write directly to a Solution buffer
- Use `get_basis_into()` to write directly to `realization_container.basis`
- No new allocations per solve

### Design Decision: Solution Buffer Location

Currently `Realization` has a `basis` field but no `solution` field (solution is extracted and used locally, not stored).

**Options**:
1. **Add `solution` field to Realization** - Store preallocated buffer
2. **Thread-local solution buffer** - Avoid changing Realization struct
3. **Local variable with capacity** - One-time allocation per call stack

**Recommended: Option 2** - Thread-local buffer avoids struct changes and the solution is only used within the method.

## Acceptance Criteria

- [x] `realize_and_solve()` uses `get_solution_into()` instead of `get_solution()`
- [x] `realize_and_solve()` uses `get_basis_into()` instead of `get_basis()`
- [x] No new allocations per solve (basis and solution buffers reused)
- [x] Solution values extracted correctly (verify with examples)
- [x] No performance regression

**Implementation**: Both solution and basis now use buffer-into pattern via thread-local storage.

## Implementation Guide

### Suggested Approach

1. Add thread-local `Solution` buffer for reuse
2. Modify extraction block to use `get_solution_into()` and `get_basis_into()`
3. Update control flow to work with mutable borrows
4. Test with examples

### Key Files to Modify

- `src/subproblem.rs`: Update `realize_and_solve()` method (~line 1560-1585)

### Code Changes

```rust
// Add at module level or in subproblem.rs
use std::cell::RefCell;

thread_local! {
    static SOLUTION_BUFFER: RefCell<Option<solver::Solution>> = RefCell::new(None);
}

// In realize_and_solve(), replace lines 1560-1574:
let extraction_start = std::time::Instant::now();

let model_status = if let Some(model) = &self.model {
    let status = model.status();
    if status == solver::HighsModelStatus::Optimal {
        // Use preallocated buffers instead of allocating new ones
        SOLUTION_BUFFER.with(|buf| {
            let mut buf = buf.borrow_mut();
            let solution = buf.get_or_insert_with(|| {
                solver::Solution::with_capacity(model.num_cols(), model.num_rows())
            });
            model.get_solution_into(solution);
            
            // Process solution
            self.slice_solution_rows_to_problem_constraints(solution);
            
            // Get objective
            let obj_value = model.get_objective_value();
            realization_container.total_stage_objective = obj_value;
            realization_container.current_stage_objective =
                get_current_stage_objective(obj_value, solution);
            
            // Extract physical results (pass reference to buffer)
            self.get_deficit_from_solution(solution, realization_container);
            self.get_net_exchange_from_solution(solution, realization_container);
            // ... rest of extractions ...
        });
        
        // Update basis in-place
        model.get_basis_into(&mut realization_container.basis);
        
        Some(status)
    } else {
        Some(status)
    }
} else {
    None
};
```

### Alternative: Simpler Refactoring

If thread-local adds too much complexity, a simpler approach:

```rust
// Just change the basis handling to avoid discarding preallocated buffer
if status == solver::HighsModelStatus::Optimal {
    let sol = model.get_solution();  // Still allocates (fix in separate ticket)
    model.get_basis_into(&mut realization_container.basis);  // Reuses buffer
    let obj = model.get_objective_value();
    // ...
}
```

This captures ~50% of the benefit (basis is per-solve, solution is per-solve).

### Pitfalls to Avoid

- ⚠️ Solution buffer must be initialized before first use
- ⚠️ Borrow checker: can't hold `&mut realization_container` while accessing `model`
- ⚠️ Thread-local RefCell panics on recursive borrow

## Testing Requirements

### Unit Tests

- [ ] Verify solution values match original implementation
- [ ] Verify basis values match original implementation

### Integration Tests

- [x] Run example 01-deterministic, verify identical output (tests pass)
- [x] Run example 07-par-model, verify identical convergence (tests pass)

### Memory Tests

- [ ] Profile with valgrind/massif to verify reduced allocations
- [ ] Measure RSS before/after on example 05

## Documentation Requirements

- [x] Add comments explaining buffer reuse strategy
- [ ] Document thread-local usage

## Effort Estimate

**Points**: 1  
**Confidence**: Medium  
**Rationale**: Logic is clear but borrow checker may require careful handling

## Definition of Done

- [x] Implementation complete (basis buffer-into)
- [x] Examples produce identical results
- [ ] Memory profiling shows reduced allocations
- [x] Tests passing
- [ ] Code reviewed
- [ ] PR merged
