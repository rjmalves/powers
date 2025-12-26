# TICKET-002c: Remove DeepSizeEstimate Trait and Implementations

## Status: ✅ COMPLETE (2025-12-26)

## Context

### Background

The `DeepSizeEstimate` trait in `src/memory/deep_sizing.rs` (474 lines) provides heap memory estimation for types with nested allocations. However:

1. It's **never called** in production code
2. Its estimates are **wrong** for `StorageAndInflowState` (assumes `num_hydros` instead of actual state dimension)
3. It requires `SizingInfo` which is also being removed

### Relation to Epic

Part of Epic: [Epic 01c: Memory Module Cleanup](../00-epic-overview.md)  
Sprint: [Sprint 1](./00-sprint-overview.md)

### Current State

```bash
$ grep -rn "estimate_heap_bytes" src/ --include="*.rs" | grep -v "fn estimate" | grep -v test
# Only trait implementations, no actual production calls
```

## Specification

### Files to Remove

1. `src/memory/deep_sizing.rs` - Entire file (474 lines)

### Files to Modify

1. `src/memory/mod.rs` - Remove `deep_sizing` module and re-export
2. `src/cut.rs` - Remove `DeepSizeEstimate` impl for `BendersCut` and `BendersCutPool`
3. `src/fcf.rs` - Remove `DeepSizeEstimate` impl for `CutStatePair`
4. `src/state.rs` - Remove `DeepSizeEstimate` impls for `StorageState` and `StorageAndInflowState`

### Behavior

- All `impl DeepSizeEstimate for ...` blocks must be removed
- All `use crate::memory::DeepSizeEstimate` imports must be removed

## Acceptance Criteria

- [ ] `src/memory/deep_sizing.rs` deleted
- [ ] No `DeepSizeEstimate` trait or implementations in codebase
- [ ] `pub use deep_sizing::DeepSizeEstimate` removed from `mod.rs`
- [ ] Removed impls from: `cut.rs`, `fcf.rs`, `state.rs`
- [ ] Code compiles without errors
- [ ] All tests pass

## Implementation Guide

### Step 1: Remove Implementations from Domain Files

**src/cut.rs** - Remove lines ~117-195:
```rust
// DELETE:
use crate::memory::DeepSizeEstimate;

impl DeepSizeEstimate for BendersCut { ... }
impl DeepSizeEstimate for BendersCutPool { ... }
```

**src/fcf.rs** - Remove lines ~361-395:
```rust
// DELETE:
use crate::memory::DeepSizeEstimate;

impl DeepSizeEstimate for CutStatePair { ... }
```

**src/state.rs** - Remove lines ~47, ~726-747, ~1199-1240:
```rust
// DELETE:
use crate::memory::{DeepSizeEstimate, SizingInfo};

impl DeepSizeEstimate for StorageState { ... }
impl DeepSizeEstimate for StorageAndInflowState { ... }
```

### Step 2: Remove the Trait File

```bash
rm src/memory/deep_sizing.rs
```

### Step 3: Update mod.rs

Remove:
```rust
pub mod deep_sizing;
pub use deep_sizing::DeepSizeEstimate;
```

### Step 4: Remove Related Tests

In `src/state.rs`, remove the `DeepSizeEstimate Tests` section and helper functions like `create_test_sizing()`.

### Key Files to Modify

- `src/memory/deep_sizing.rs` - DELETE
- `src/memory/mod.rs` - Remove module and export
- `src/cut.rs` - Remove ~75 lines
- `src/fcf.rs` - Remove ~35 lines  
- `src/state.rs` - Remove ~100 lines + tests

### Pitfalls to Avoid

- ⚠️ Make sure to remove ALL `impl DeepSizeEstimate` blocks
- ⚠️ The `SizingInfo` type used in these impls is already removed (TICKET-001c)
- ⚠️ Some tests create `SizingInfo` manually - remove those too

## Testing Requirements

### Compilation Test

```bash
cargo build --release
```

### Unit Tests

```bash
cargo test --lib
```

### Verification

```bash
# Confirm no DeepSizeEstimate references remain
grep -rn "DeepSizeEstimate" src/ --include="*.rs"
# Should return nothing
```

## Documentation Requirements

- [ ] Update `src/memory/mod.rs` module documentation

## Dependencies

- **Blocked By**: TICKET-001c (SizingInfo removal)
- **Blocks**: TICKET-003c (CutComputationBuffers hardening)

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Mechanical removal of trait and implementations

## Definition of Done

- [ ] `deep_sizing.rs` deleted
- [ ] All `DeepSizeEstimate` impls removed
- [ ] No compilation errors
- [ ] All tests pass
- [ ] Examples run successfully
