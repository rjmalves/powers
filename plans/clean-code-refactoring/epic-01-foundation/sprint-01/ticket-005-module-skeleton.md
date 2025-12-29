# [T-005] Create Module Directory Skeleton

> **Epic**: [Epic 1: Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Infrastructure Setup](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: Epic 2 (uses new module structure)

---

## ⚠️ CRITICAL: No Algorithm Changes

This ticket creates empty module directories and placeholder files. **No logic is moved or modified.** The existing code remains exactly as-is.

---

## Files to Read Before Starting

- [Master Plan: Target State](../../../00-master-plan.md#target-state) - Module structure diagram
- `src/lib.rs` - Current module organization

---

## Context

### Background

The master plan defines a target module structure that separates concerns. This ticket creates the directory skeleton with placeholder modules. We're not moving any code yet—just creating the structure so future tickets can populate it.

### Current State

```
src/
├── sddp/           # Monolithic SDDP implementation
├── memory/         # Already exists (some memory utilities)
├── logging/        # Already exists
├── output/         # Already exists
└── utils/          # Already exists
```

### Target State (from Master Plan)

```
src/
├── algorithm/      # SDDP algorithm phases (Epic 3)
├── model/          # LP model operations (Epic 2)
├── state/          # State management (Epic 4)
├── memory/         # Already exists, will be extended (Epic 5)
├── timing/         # Created in T-003/T-004
└── ...             # Existing modules unchanged
```

---

## Specification

### Directories to Create

```
src/algorithm/
src/model/
```

Note: `src/state/` is NOT created because `src/state.rs` already exists and will be refactored in Epic 4.

### Files to Create

Each new module gets a `mod.rs` with:
1. Module documentation explaining future purpose
2. Commented-out submodule declarations (roadmap)
3. Explicit note that this is a placeholder

### Algorithm Module

```rust
// src/algorithm/mod.rs

//! SDDP Algorithm Phases
//!
//! This module will contain the core SDDP algorithm logic, separated by phase:
//!
//! - `forward_pass`: Forward simulation through the scenario tree
//! - `backward_pass`: Backward cut generation and FCF updates
//! - `cut_computation`: Benders cut calculation
//! - `convergence`: Convergence checking logic
//!
//! # Status
//!
//! 🚧 **Placeholder**: This module is currently empty. Logic will be migrated
//! from `src/sddp/mod.rs` in Epic 3: Algorithm Separation.
//!
//! # Future Structure
//!
//! ```text
//! algorithm/
//! ├── mod.rs
//! ├── forward_pass.rs
//! ├── backward_pass.rs
//! ├── cut_computation.rs
//! └── convergence.rs
//! ```

// Future submodules (uncomment as implemented):
// pub mod forward_pass;
// pub mod backward_pass;
// pub mod cut_computation;
// pub mod convergence;
```

### Model Module

```rust
// src/model/mod.rs

//! LP Model Operations
//!
//! This module will contain LP model building and solver interaction:
//!
//! - `builder`: Model construction utilities
//! - `constraints/`: Constraint generation by type
//! - `solver_interface`: HiGHS solver interaction
//! - `solution_extract`: Solution extraction into domain types
//!
//! # Status
//!
//! 🚧 **Placeholder**: This module is currently empty. Logic will be migrated
//! from `src/subproblem.rs` in Epic 2: Core Extraction.
//!
//! # Future Structure
//!
//! ```text
//! model/
//! ├── mod.rs
//! ├── builder.rs
//! ├── constraints/
//! │   ├── mod.rs
//! │   ├── hydro_balance.rs
//! │   ├── bus_balance.rs
//! │   └── ar_dynamics.rs
//! ├── solver_interface.rs
//! └── solution_extract.rs
//! ```

// Future submodules (uncomment as implemented):
// pub mod builder;
// pub mod constraints;
// pub mod solver_interface;
// pub mod solution_extract;
```

### Updates to lib.rs

Add the new modules to `src/lib.rs`:

```rust
// Add these lines (do not remove any existing modules):
pub mod algorithm;
pub mod model;
```

---

## Acceptance Criteria

- [ ] `src/algorithm/mod.rs` exists with documentation
- [ ] `src/model/mod.rs` exists with documentation  
- [ ] Both modules are exported from `src/lib.rs`
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] `cargo doc` generates documentation for new modules
- [ ] Golden tests pass

### Correctness Verification

- [ ] No existing code modified (except `src/lib.rs` module declarations)
- [ ] All existing tests pass
- [ ] Golden tests pass

---

## Implementation Guide

### Suggested Approach

1. **Create directories**:
   ```bash
   mkdir -p src/algorithm src/model
   ```

2. **Create placeholder modules** with documentation (see Specification)

3. **Update `src/lib.rs`**:
   ```rust
   pub mod algorithm;
   pub mod model;
   // ... existing modules unchanged
   ```

4. **Verify build**:
   ```bash
   cargo build
   cargo test
   cargo doc --open  # Verify docs render correctly
   ```

5. **Run golden tests**:
   ```bash
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Create

- `src/algorithm/mod.rs`
- `src/model/mod.rs`

### Key Files to Modify

- `src/lib.rs` (add module declarations only)

### Pitfalls to Avoid

- ⚠️ Don't create `src/state/` - `src/state.rs` already exists
- ⚠️ Don't move any existing code
- ⚠️ Don't uncomment the submodule declarations yet
- ⚠️ Don't forget to run golden tests

---

## Testing Requirements

### Compilation Tests

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] `cargo doc` succeeds

### Verification

- [ ] New modules appear in generated documentation
- [ ] Golden tests pass

---

## Documentation Requirements

- [ ] Module-level documentation in each new `mod.rs`
- [ ] Clear "placeholder" status indicator
- [ ] Future structure roadmap in comments

---

## Effort Estimate

**Points**: 1
**Confidence**: High
**Rationale**: Simple directory and file creation, no logic involved

---

## Definition of Done

- [ ] Directories created
- [ ] Placeholder modules with documentation
- [ ] Modules exported from lib.rs
- [ ] All builds and tests pass
- [ ] Golden tests pass
- [ ] Documentation renders correctly
