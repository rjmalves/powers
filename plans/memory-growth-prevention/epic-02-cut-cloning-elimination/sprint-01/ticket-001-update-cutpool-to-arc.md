# [TICKET-001] Update BendersCutPool to use Arc<BendersCut>

> **Epic**: [Epic 2: Cut Cloning Elimination](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-002](./ticket-002-update-cut-consumers.md)

## Context

### Background

Cuts are cloned during batch processing to allow lock-free handler application. Each clone copies ~1.3 KB of data. Using `Arc<BendersCut>` allows cheap reference counting (2 atomic ops) instead of full data cloning.

### Current State

```rust
// src/cut.rs:56-61
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub active_cut_indices: HashMap<usize, usize>,
    pub total_cut_count: usize,
}
```

### Files to Read Before Starting

- `src/cut.rs` - BendersCut and BendersCutPool definitions
- `src/fcf.rs` - FCF struct that contains cut pool

## Specification

### Changes

1. Change `BendersCutPool.pool` from `Vec<BendersCut>` to `Vec<Arc<BendersCut>>`
2. Update `with_capacity` constructor
3. Update pool access patterns (`.get()` returns `&Arc<BendersCut>`)

### Behavior

- Pool stores Arc-wrapped cuts
- Adding a cut wraps it in Arc
- Getting a cut returns Arc reference
- Existing cut data unchanged

## Acceptance Criteria

- [ ] `BendersCutPool.pool` is `Vec<Arc<BendersCut>>`
- [ ] `with_capacity` works with Arc storage
- [ ] Compiles without errors (consumers updated in TICKET-002)

## Implementation Guide

### Key Files to Modify

- `src/cut.rs`: Update BendersCutPool struct and methods

### Code Changes

```rust
use std::sync::Arc;

pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
    pub active_cut_indices: HashMap<usize, usize>,
    pub total_cut_count: usize,
}

impl BendersCutPool {
    pub fn new() -> Self {
        Self {
            pool: vec![],
            active_cut_indices: HashMap::new(),
            total_cut_count: 0,
        }
    }
    
    pub fn with_capacity(max_cuts: usize, max_state_dim: usize) -> Self {
        Self {
            pool: Vec::with_capacity(max_cuts),
            active_cut_indices: HashMap::with_capacity(max_cuts),
            total_cut_count: 0,
        }
    }
    
    // Pool access now returns &Arc<BendersCut>
}
```

### Pitfalls to Avoid

- ⚠️ This will cause compile errors until consumers are updated
- ⚠️ Don't try to mutate cuts through Arc (use interior mutability if needed)

## Testing Requirements

### Unit Tests

- [ ] Compile-time: struct compiles with Arc
- [ ] Pool creation works

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple struct change, complexity is in consumer updates

## Definition of Done

- [ ] Implementation complete
- [ ] Compiles (with consumer updates)
- [ ] Tests passing
