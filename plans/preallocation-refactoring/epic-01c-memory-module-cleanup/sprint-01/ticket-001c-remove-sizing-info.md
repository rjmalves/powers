# TICKET-001c: Remove SizingInfo and Related Dead Code

## Status: ✅ COMPLETE (2025-12-26)

## Context

### Background

The `src/memory/sizing.rs` file (1,574 lines) contains `SizingInfo`, `NodeSizing`, and `MemoryBreakdown` types that are **never used in production code**. Analysis shows `SizingInfo::from_input()` is only called in test code within the memory module itself.

### Relation to Epic

Part of Epic: [Epic 01c: Memory Module Cleanup](../00-epic-overview.md)  
Sprint: [Sprint 1](./00-sprint-overview.md)

### Current State

```bash
$ grep -rn "SizingInfo::from_input" src/ --include="*.rs" | grep -v "^src/memory" | grep -v test
# (no results - never used in production)
```

The training code in `sddp/mod.rs` manually computes dimensions instead of using `SizingInfo`.

## Specification

### Files to Remove

1. `src/memory/sizing.rs` - Entire file (1,574 lines)

### Files to Modify

1. `src/memory/mod.rs` - Remove `sizing` module and re-exports
2. `src/memory/buffers.rs` - Remove `SizingInfo` dependency from `ThreadLocalBuffers`

### Behavior

- All code that imports `SizingInfo`, `NodeSizing`, or `MemoryBreakdown` must be removed or updated
- Tests within the memory module that depend on `SizingInfo` should be removed

## Acceptance Criteria

- [ ] `src/memory/sizing.rs` deleted
- [ ] No references to `SizingInfo`, `NodeSizing`, `MemoryBreakdown` in codebase
- [ ] `pub use sizing::*` removed from `mod.rs`
- [ ] Code compiles without errors
- [ ] All existing tests pass (except removed sizing tests)

## Implementation Guide

### Step 1: Verify No Production Usage

```bash
# Confirm no production usage
grep -rn "SizingInfo" src/ --include="*.rs" | grep -v "^src/memory" | grep -v test | grep -v "fn estimate"
```

### Step 2: Remove File

```bash
rm src/memory/sizing.rs
```

### Step 3: Update mod.rs

Remove:
```rust
pub mod sizing;
pub use sizing::{MemoryBreakdown, NodeSizing, SizingInfo};
```

### Step 4: Update buffers.rs

The `ThreadLocalBuffers::new()` takes `&SizingInfo`. Since `ThreadLocalBuffers` is also being removed (TICKET-002c), we can either:
- Remove `ThreadLocalBuffers` now (preferred)
- Or temporarily comment out the dependency

### Step 5: Fix Compilation Errors

Any remaining references will cause compilation errors. Fix by removing the referencing code.

### Key Files to Modify

- `src/memory/sizing.rs` - DELETE
- `src/memory/mod.rs` - Remove module and exports
- `src/memory/buffers.rs` - Remove `SizingInfo` usage

### Pitfalls to Avoid

- ⚠️ Don't remove `initialize_cut_buffers` - it's used in production
- ⚠️ The `mod.rs` integration tests use `SizingInfo` - they should be removed

## Testing Requirements

### Compilation Test

```bash
cargo build --release
```

### Unit Tests

```bash
cargo test --lib
```

### Smoke Test

```bash
cargo run --release -- run examples/01-deterministic
cargo run --release -- run examples/07-par-model-with-inflow-state
```

## Documentation Requirements

- [ ] Update `src/memory/mod.rs` module documentation to reflect simplified scope

## Dependencies

- **Blocked By**: None
- **Blocks**: TICKET-002c (DeepSizeEstimate removal)

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward file deletion and import cleanup

## Definition of Done

- [ ] `sizing.rs` deleted
- [ ] No compilation errors
- [ ] All tests pass
- [ ] Examples run successfully
