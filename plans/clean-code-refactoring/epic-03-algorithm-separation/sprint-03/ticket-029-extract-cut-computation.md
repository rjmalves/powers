# [T-029] Extract Cut Computation Logic

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-028](./ticket-028-extract-backward-loop.md)
> **Blocks**: [T-030](./ticket-030-backward-timing.md)

---

## Context

This ticket extracts cut computation logic into `src/algorithm/cut_computation.rs`. This includes the logic for computing Benders cuts from branching realizations.

---

## Files to Read Before Starting

- `src/sddp/mod.rs:636-731` - `compute_cut_data_for_backward_step()`
- `src/subproblem.rs` - `compute_cut_data()` method
- `src/fcf.rs` - `CutData` struct

---

## Specification

### Create `src/algorithm/cut_computation.rs`

```rust
//! Cut computation utilities for SDDP backward pass.
//!
//! This module contains the logic for computing Benders cuts from
//! branching scenario solutions.

use crate::fcf::CutData;
use crate::graph::DirectedGraph;
use crate::sddp::NodeData;
use crate::subproblem::Realization;

/// Compute cut data from branching realizations.
///
/// This is the core cut computation that generates the Benders cut
/// coefficients and RHS from the solved branching scenarios.
pub fn compute_cut_from_branchings(
    branching_realizations: &[Realization],
    risk_measure: &dyn crate::risk_measure::RiskMeasure,
    iteration: usize,
    forward_pass_idx: usize,
) -> CutData {
    // Delegate to subproblem's compute_cut_data
    // This is a thin wrapper for now; can be enhanced later
    todo!("Extract from subproblem")
}
```

Note: The actual cut computation logic is in `subproblem.rs:compute_cut_data()`. This module provides a clean interface and can be enhanced in future epics.

---

## Acceptance Criteria

- [ ] `src/algorithm/cut_computation.rs` created
- [ ] Module provides clean interface for cut computation
- [ ] Documentation explains cut computation algorithm
- [ ] Module exported in `src/algorithm/mod.rs`
- [ ] `cargo build -j1` succeeds

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Thin wrapper for now, main logic stays in subproblem

---

## Definition of Done

- [ ] Cut computation module created
- [ ] Interface documented
- [ ] Module exported
- [ ] Code reviewed
