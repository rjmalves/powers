# TICKET-004c: Fix Cut Buffer Initialization with Correct Dimensions

## Status: ✅ COMPLETE (2025-12-26)

## Context

### Background

The current cut buffer initialization in `sddp/mod.rs` extracts `max_state_dim` from node data, but the logic is duplicated and potentially fragile. We need to ensure:

1. **Correct state dimension**: `num_hydros + inflow_lags` for `StorageAndInflowState`
2. **All Rayon worker threads initialized**: Use `rayon::broadcast()`
3. **Single source of truth**: Extract dimensions once, use everywhere

### Relation to Epic

Part of Epic: [Epic 01c: Memory Module Cleanup](../00-epic-overview.md)  
Sprint: [Sprint 1](./00-sprint-overview.md)

### Current State

```rust
// src/sddp/mod.rs:1733-1761
let max_state_dim = self.node_data_graph
    .iter_nodes()
    .map(|node| {
        match node.data.state_choice.as_str() {
            "storage" => node.data.system.meta.hydros_count,
            "storage_and_inflow" => {
                let base = node.data.system.meta.hydros_count;
                let lags: usize = node.data.uncertainty_models
                    .iter()
                    .filter(|tm| matches!(tm.entity_type, crate::input::UncertaintyType::Inflow))
                    .map(|tm| tm.max_ar_order)
                    .sum();
                base + lags
            }
            _ => 0,
        }
    })
    .max()
    .unwrap();

crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);
```

**Problem**: This only initializes the main thread. Rayon worker threads are auto-initialized with arbitrary defaults (50, 20) which may cause panics after TICKET-003c.

## Specification

### Changes to Cut Buffer Initialization

1. **Use `rayon::broadcast()` to initialize all worker threads**:
```rust
// Initialize in main thread
crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);

// Initialize in all Rayon worker threads
rayon::broadcast(|_| {
    crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);
});
```

2. **Add helper function for state dimension calculation**:
```rust
// src/state.rs or src/sddp/mod.rs
pub fn compute_state_dimension(
    state_choice: &str,
    num_hydros: usize,
    uncertainty_models: &[TemporalModel],
) -> usize {
    match state_choice {
        "storage" => num_hydros,
        "storage_and_inflow" => {
            let inflow_lags: usize = uncertainty_models
                .iter()
                .filter(|tm| matches!(tm.entity_type, UncertaintyType::Inflow))
                .map(|tm| tm.max_ar_order)
                .sum();
            num_hydros + inflow_lags
        }
        _ => num_hydros,
    }
}
```

3. **Update `initialize_cut_buffers` to broadcast**:

Option A: Keep broadcast in caller (SDDP)
Option B: Move broadcast into `initialize_cut_buffers` function

We choose **Option A** for clarity - the caller explicitly broadcasts.

### Behavior

- All Rayon worker threads have identically-sized buffers
- Main thread and worker threads use same dimensions
- No lazy initialization fallback (panic if uninitialized)

## Acceptance Criteria

- [ ] Cut buffers initialized in main thread AND all Rayon worker threads
- [ ] State dimension correctly accounts for inflow lags
- [ ] `rayon::broadcast()` called after main thread initialization
- [ ] Examples run without panics
- [ ] All tests pass

## Implementation Guide

### Step 1: Add Broadcast After Initialization

In `src/sddp/mod.rs`, after the existing `initialize_cut_buffers` call:

```rust
crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);

// Initialize in all Rayon worker threads
rayon::broadcast(|_| {
    crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);
});
```

### Step 2: Verify State Dimension Calculation

Ensure the existing calculation handles all cases:
- `"storage"`: `num_hydros`
- `"storage_and_inflow"`: `num_hydros + sum(inflow AR orders)`

### Step 3: Log Dimensions for Debugging

Add logging to verify correct dimensions:

```rust
log::debug!(
    "Cut buffer dimensions: max_state_dim={}, max_scenarios={}",
    max_state_dim,
    max_scenarios
);
```

### Step 4: Test with Example 07

Example 07 uses `storage_and_inflow` state and should exercise the full state dimension:

```bash
RUST_LOG=debug cargo run --release -- run examples/07-par-model-with-inflow-state 2>&1 | grep "Cut buffer"
```

### Key Files to Modify

- `src/sddp/mod.rs` - Add broadcast call

### Pitfalls to Avoid

- ⚠️ Don't call `initialize_cut_buffers` inside `rayon::broadcast` closure without also calling it in main thread
- ⚠️ Ensure the values captured by closure are correct (max_state_dim, max_scenarios)
- ⚠️ The broadcast must happen BEFORE any parallel cut computation

## Testing Requirements

### Unit Tests

Update tests in `src/memory/buffers.rs`:

```rust
#[test]
fn test_parallel_cut_buffer_initialization() {
    use rayon::prelude::*;
    
    // Initialize main + workers
    initialize_cut_buffers(100, 10);
    rayon::broadcast(|_| {
        initialize_cut_buffers(100, 10);
    });
    
    // Parallel usage should work
    let results: Vec<_> = (0..8)
        .into_par_iter()
        .map(|i| {
            with_cut_buffers(|buffers| {
                buffers.reset_for_cut(50, 5);  // Within capacity
                buffers.coefficients[0] = i as f64;
                buffers.coefficients[0]
            })
        })
        .collect();
    
    for (i, &result) in results.iter().enumerate() {
        assert_eq!(result, i as f64);
    }
}
```

### Integration Test

```bash
# Should complete without panics
cargo run --release -- run examples/07-par-model-with-inflow-state
```

### Capacity Verification

Add a temporary log to verify dimensions:

```rust
// In state.rs compute_new_cut()
log::trace!(
    "Cut computation: state_dim={}, num_scenarios={}",
    self.dimension,
    branching_realizations.len()
);
```

## Documentation Requirements

- [ ] Document the initialization pattern in `src/memory/mod.rs`
- [ ] Add comment explaining why broadcast is needed

## Dependencies

- **Blocked By**: TICKET-003c (hardened buffers)
- **Blocks**: TICKET-005c (validation)

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Small change but critical for correctness

## Definition of Done

- [ ] Broadcast call added
- [ ] All worker threads initialized
- [ ] Examples run without panics
- [ ] Tests pass
