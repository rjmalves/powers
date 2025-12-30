# [T-072] Migrate VisitedStatePool to Enum Dispatch

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Pool Memory Model Optimization](./00-sprint-overview.md)
> **Dependencies**: [T-071](./ticket-071-concrete-state-enum.md)
> **Blocks**: [T-073](./ticket-073-cleanup-deprecated.md)

## Files to Read Before Starting

- `src/state.rs:480-592` - `VisitedStatePool` implementation
- `src/fcf.rs:180-220` - FCF state pool usage
- `src/state.rs` - `ConcreteState` enum (from T-071)

---

## Context

### Background

With `ConcreteState` enum from T-071, we can replace the dynamic dispatch pool with static dispatch.

### Current State

```rust
pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,
}

impl VisitedStatePool {
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        template_state: &dyn State,
    ) -> Self { ... }
    
    pub fn update_state(
        &mut self,
        slot: usize,
        coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
    ) -> &mut Box<dyn State> { ... }
}
```

### Target State

```rust
pub struct VisitedStatePool {
    pub pool: Vec<ConcreteState>,
}

impl VisitedStatePool {
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        state_config: StateConfig,  // New: describes state type
    ) -> Self { ... }
    
    pub fn update_state(
        &mut self,
        slot: usize,
        coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
    ) -> &mut ConcreteState { ... }
}
```

---

## Specification

### StateConfig for Pool Initialization

Instead of a template `&dyn State`, use a configuration:

```rust
/// Configuration for creating state pool.
#[derive(Debug, Clone)]
pub enum StateConfig {
    /// Storage-only states
    Storage { num_hydros: usize },
    /// Storage + inflow lag states
    StorageAndInflow {
        num_hydros: usize,
        per_hydro_state_dims: Vec<usize>,
    },
}

impl StateConfig {
    /// Create a ConcreteState from this config.
    pub fn create_state(&self) -> ConcreteState {
        match self {
            Self::Storage { num_hydros } => ConcreteState::storage(*num_hydros),
            Self::StorageAndInflow { num_hydros, per_hydro_state_dims } => {
                ConcreteState::storage_and_inflow(*num_hydros, per_hydro_state_dims.clone())
            }
        }
    }
}
```

### Updated VisitedStatePool

```rust
pub struct VisitedStatePool {
    pub pool: Vec<ConcreteState>,
}

impl VisitedStatePool {
    pub fn with_capacity(num_states: usize) -> Self {
        Self {
            pool: Vec::with_capacity(num_states),
        }
    }

    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        config: &StateConfig,
    ) -> Self {
        let total_states = num_iterations * num_forward_passes;
        let pool = (0..total_states)
            .map(|_| config.create_state())
            .collect();
        Self { pool }
    }

    pub fn update_state(
        &mut self,
        slot: usize,
        coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
    ) -> &mut ConcreteState {
        let state = &mut self.pool[slot];
        state.update_coefficients(coefficients);
        state.set_iteration(iteration);
        state.set_forward_pass_idx(forward_pass_idx);
        state
    }
}
```

---

## Acceptance Criteria

- [ ] `VisitedStatePool` uses `Vec<ConcreteState>`
- [ ] `preallocate()` takes `StateConfig` instead of `&dyn State`
- [ ] All callsites updated
- [ ] FCF pool integration works
- [ ] All tests pass
- [ ] Golden tests pass

---

## Implementation Guide

### Step 1: Add StateConfig

In `src/state.rs`:

```rust
/// Configuration for creating state pool.
#[derive(Debug, Clone)]
pub enum StateConfig {
    Storage { num_hydros: usize },
    StorageAndInflow {
        num_hydros: usize,
        per_hydro_state_dims: Vec<usize>,
    },
}
```

### Step 2: Update VisitedStatePool

Replace `Vec<Box<dyn State>>` with `Vec<ConcreteState>`.

### Step 3: Update FutureCostFunction::preallocate_pools

In `src/fcf.rs`:

```rust
pub fn preallocate_pools(
    num_iterations: usize,
    num_forward_passes: usize,
    state_dimension: usize,
    state_config: &StateConfig,  // Changed from &dyn State
) -> Self {
    Self {
        cut_pool: BendersCutPool::preallocate(...),
        state_pool: VisitedStatePool::preallocate(
            num_iterations,
            num_forward_passes,
            state_config,
        ),
    }
}
```

### Step 4: Update callsites in sddp/mod.rs

Find where FCF pools are created and update to use StateConfig:

```rust
// Before
let template: Box<dyn State> = ...;
let fcf = FutureCostFunction::preallocate_pools(..., &*template);

// After
let state_config = StateConfig::Storage { num_hydros: 10 };
// or
let state_config = StateConfig::StorageAndInflow {
    num_hydros: 10,
    per_hydro_state_dims: vec![2, 3, 1, ...],
};
let fcf = FutureCostFunction::preallocate_pools(..., &state_config);
```

### Step 5: Update domination evaluation

In `src/fcf.rs`, `eval_new_cut_domination`:

```rust
pub fn eval_new_cut_domination(&mut self, new_cut: &mut BendersCut) {
    for state in self.state_pool.pool.iter_mut() {
        let state_coefs = state.coefficients();  // Now ConcreteState
        let height = new_cut.eval_height_at_state(state_coefs);
        // ... rest of logic
    }
}
```

### Step 6: Create StateConfig from NodeData

Add helper to create config from problem data:

```rust
fn state_config_from_node_data(node_data: &NodeData) -> StateConfig {
    let num_hydros = node_data.system.hydros.len();
    let per_hydro_dims = per_hydro_state_dims(
        &node_data.uncertainty_models,
        num_hydros,
    );
    
    if per_hydro_dims.iter().all(|&d| d == 1) {
        StateConfig::Storage { num_hydros }
    } else {
        StateConfig::StorageAndInflow {
            num_hydros,
            per_hydro_state_dims: per_hydro_dims,
        }
    }
}
```

---

## Testing Requirements

### Compile

```bash
cargo build -j1
```

### Tests

```bash
RUST_TEST_THREADS=1 cargo test -j1
```

### Golden Tests

```bash
./scripts/golden-tests.sh verify
```

---

## Pitfalls to Avoid

- ⚠️ **Mixed state types**: All states in a pool must be the same variant. Verify problem setup enforces this.
- ⚠️ **Trait usage**: Some code may use `&dyn State`. Update to use `&ConcreteState`.
- ⚠️ **Clone behavior**: `ConcreteState::clone()` is direct, not via trait machinery.

---

## Documentation Requirements

- [ ] Update VisitedStatePool docs
- [ ] Document StateConfig usage
- [ ] Update FCF docs

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Many callsites to update. StateConfig pattern is new.

---

## Definition of Done

- [ ] Pool uses Vec<ConcreteState>
- [ ] StateConfig used for initialization
- [ ] All callsites updated
- [ ] Tests pass
- [ ] Golden tests pass
