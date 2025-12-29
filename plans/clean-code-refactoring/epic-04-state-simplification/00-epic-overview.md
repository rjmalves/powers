# Epic 4: State Simplification

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

This epic refactors `state.rs` (3,087 lines) which contains core state representations used throughout the algorithm.

**Algorithm correctness is non-negotiable.** State values must be preserved exactly. Golden tests must pass after every change. If ANY unexpected behavior occurs, **STOP and ask for clarification**.

---

## Summary

This epic simplifies the state management in `src/state.rs`. Currently there are multiple state implementations with duplicated logic. We consolidate into cleaner abstractions while preserving exact behavior, and prepare the foundation for pool-based allocation in Epic 5.

**Key principle**: Simplify structure, not behavior. All numerical values must remain identical.

---

## Scope

### Included

1. **State Trait Analysis**
   - Document current `State` trait usage
   - Identify duplication between `StorageState` and `StorageAndInflowState`
   - **Document trait object allocation patterns** (where `Box<dyn State>` is created/cloned)

2. **State Consolidation**
   - Reduce code duplication
   - Create shared extraction utilities
   - **Simplify state cloning** (prepare for pool-based allocation)

3. **State-Cut Relationship Documentation**
   - Document the 1:1 relationship between states and cuts
   - Prepare for `CutStatePair` pool-based storage
   - Document slot-based indexing using `(iteration, forward_pass_idx)`

4. **Trajectory Storage Analysis**
   - Analyze trajectory storage patterns
   - **Prepare interface for pool-based allocation** (Epic 5)

### Excluded

- Pool-based memory allocation implementation (Epic 5)
- Full SoA conversion (analyzed in Epic 5)
- Algorithm changes

---

## Dependencies

- **Requires**:
  - Epic 3 complete (clean state interface from algorithm separation)
- **Enables**:
  - Epic 5: Memory Optimization (clear state allocation points, pool-ready interfaces)

---

## Acceptance Criteria

- [ ] `state.rs` reduced by ≥30% lines through deduplication
- [ ] All state functions ≤50 lines
- [ ] State trait implementations consolidated where possible
- [ ] **Trait object allocation points documented** with specific locations
- [ ] **State-Cut 1:1 relationship documented** with slot indexing strategy
- [ ] **Interfaces prepared for pool-based allocation**
- [ ] Golden tests pass
- [ ] Benchmarks within 5%

---

## Technical Approach

### Understanding the Current State Architecture

#### State Trait and Implementations

The `State` trait exists because different state definitions require:
1. **Different state coefficients**: `StorageState` stores only storage volumes; `StorageAndInflowState` stores storage + past inflow observations
2. **Different cut evaluation logic**: The cut evaluation process differs based on which variables are in the state

```rust
pub trait State: Send + Sync {
    fn coefficients(&self) -> &[f64];        // State variable values
    fn evaluate_cut(&self, ...) -> CutEvalResult;  // Different logic per impl
    fn clone_box(&self) -> Box<dyn State>;   // ⚠️ ALLOCATION: Creates Box<dyn State>
    // ...
}
```

#### State is a Subset of Realization

**Key insight**: A state is merely a subset of entries from a subproblem solution, contained in `Realization` objects:

```
Realization (full solution)
├── deficit, exchange, thermal_generation, ...  # NOT part of state
├── final_storage                               # → StorageState coefficients
├── inflow                                      # → StorageAndInflowState (+ storage)
├── water_value, marginal_cost                  # Duals for cut computation
└── load_lag_duals, inflow_lag_duals           # Lag duals for AR models
```

The state extracts the relevant subset and stores it for:
1. **Cut construction**: State coefficients define the cut's hyperplane
2. **Cut selection**: States must be tracked to identify dominated cuts

#### Realization Lifecycle

**Current pattern** (allocation-heavy):
```
Training iteration:
  ├── Forward pass:
  │   └── Realizations allocated for current iteration only
  │       (new Vec<Realization> per forward pass)
  │
  └── Backward pass:
      └── For each stage:
          ├── Compute cut from branching realizations
          ├── Extract state from subproblem: state.clone() → Box<dyn State> ⚠️ ALLOC
          └── Create CutStatePair(cut, visited_state, forward_pass_idx)
```

**Problem**: We need state history for cut selection, but realizations only exist for current iteration.

### Trait Object Allocation Points

The following locations create `Box<dyn State>`:

| Location | Function | Frequency | Notes |
|----------|----------|-----------|-------|
| `subproblem.rs:1659` | `compute_new_cut` | Per cut creation | `self.state.clone()` |
| `state.rs` | `clone_box()` impl | Called by above | Actual allocation |
| `fcf.rs` | `CutStatePair::new` | Per cut | Stores the cloned state |

### State-Cut 1:1 Relationship

Each cut has exactly one originating state—the state that was visited when the cut was constructed:

```rust
// Current structure (from fcf.rs)
pub struct CutStatePair {
    pub cut: BendersCut,
    pub visited_state: Box<dyn State>,  // ⚠️ The state when cut was created
    pub forward_pass_idx: usize,
}
```

**Slot-based indexing strategy** (for Epic 5):
- Index: `(iteration, forward_pass_idx)` uniquely identifies a state/cut pair
- This indexing is already used in the current code
- Pool slots can use this composite key

### Preparing for Pool-Based Allocation

#### Option 1: Enum-Based State (Simpler)

Replace trait object with enum to eliminate dynamic dispatch and `Box`:

```rust
pub enum StateKind {
    Storage(StorageState),
    StorageAndInflow(StorageAndInflowState),
}

impl StateKind {
    pub fn coefficients(&self) -> &[f64] {
        match self {
            Self::Storage(s) => &s.storage,
            Self::StorageAndInflow(s) => s.coefficients(),
        }
    }
    
    pub fn evaluate_cut(&self, ...) -> CutEvalResult {
        match self {
            Self::Storage(s) => s.evaluate_cut(...),
            Self::StorageAndInflow(s) => s.evaluate_cut(...),
        }
    }
}
```

**Benefits**:
- No `Box<dyn State>` allocation
- State can be stored inline in pools
- Slightly better cache locality

**Tradeoffs**:
- Less extensible (adding new state types requires enum change)
- Larger size (size of largest variant)

#### Option 2: Pool-Friendly Trait (More Flexible)

Keep trait but add pool-compatible methods:

```rust
pub trait State: Send + Sync {
    // Existing methods...
    
    /// Copy state coefficients into preallocated buffer.
    /// This avoids allocation when storing in pools.
    fn copy_coefficients_into(&self, target: &mut [f64]);
    
    /// Size of coefficient vector (for preallocation).
    fn coefficient_count(&self) -> usize;
    
    /// State type identifier (for pool slot selection).
    fn state_type_id(&self) -> StateTypeId;
}

pub enum StateTypeId {
    Storage,
    StorageAndInflow,
}
```

#### Recommendation

**Start with Option 2** (pool-friendly trait) in this epic, as it:
1. Preserves backward compatibility
2. Enables incremental migration
3. Can be converted to Option 1 in Epic 5 if beneficial

### Simplifying State Cloning

Current cloning creates a new `Box<dyn State>`. Prepare for pool-based alternative:

```rust
impl State for StorageState {
    // Current: allocates
    fn clone_box(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }
    
    // NEW: copy into preallocated slot (for Epic 5)
    fn copy_into(&self, slot: &mut StateSlot) {
        slot.set_storage(self.storage.as_slice());
    }
}
```

---

## Module Structure After Epic

```
src/state/
├── mod.rs              # Public exports, State trait
├── storage.rs          # StorageState implementation
├── storage_inflow.rs   # StorageAndInflowState implementation
├── shared.rs           # Shared utilities (extracted from duplication)
├── extraction.rs       # State extraction from Realization
└── pool_interface.rs   # Pool-compatible trait extensions (NEW)
```

---

## Sprints

### [Sprint 1: State Consolidation](./sprint-01/00-sprint-overview.md)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-030 | Analyze state.rs structure and document allocation points | 3 | ⬜ |
| T-031 | Document State-Cut 1:1 relationship and slot indexing | 2 | ⬜ |
| T-032 | Extract common state utilities into shared.rs | 3 | ⬜ |
| T-033 | Consolidate StorageState methods | 3 | ⬜ |
| T-034 | Consolidate StorageAndInflowState methods | 3 | ⬜ |
| T-035 | Add pool-compatible trait extensions | 3 | ⬜ |
| T-036 | Create state extraction module | 2 | ⬜ |

**Sprint Points**: 19

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 19 (increased from 13 due to pool preparation)
- **Risk Level**: Medium

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| State behavior changes | Medium | **CRITICAL** | Golden tests after every change |
| Pool interface inadequate | Low | Medium | Review with Epic 5 requirements |
| Enum vs trait decision wrong | Low | Low | Can refactor in Epic 5 |
| Hidden state dependencies | Medium | Medium | Careful analysis in T-030 |

---

## Definition of Done

- [ ] All tickets complete
- [ ] `state.rs` significantly simplified (≥30% reduction)
- [ ] **Trait object allocation points documented** with file:line references
- [ ] **State-Cut 1:1 relationship documented**
- [ ] **Slot indexing strategy (iteration, forward_pass_idx) documented**
- [ ] **Pool-compatible trait extensions added**
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Benchmark within 5%
- [ ] Ready for Epic 5 pool implementation
