# [T-025] Remove old timing/metrics.rs types

> **Epic**: [Epic 5: Output & Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-023](./ticket-023-update-csv-output.md), [T-024](./ticket-024-update-parquet-output.md)  
> **Blocks**: [T-026](./ticket-026-dead-code-cleanup.md)

## Files to Read Before Starting

- `src/timing/metrics.rs` - Old timing types to remove
- `src/timing/mod.rs` - Public exports
- `src/timing/backward.rs` - New `NewBackwardTiming`
- `src/timing/forward.rs` - New `NewForwardTiming`
- `src/timing/iteration.rs` - New `NewIterationTiming`

## Context

### Background

The `timing/metrics.rs` file contains the original timing types that were created before the refactor. With the new types in place (`NewForwardTiming`, `NewBackwardTiming`, `NewIterationTiming`), the old types in `metrics.rs` are now dead code and should be removed.

### Current State

`timing/metrics.rs` contains:
- `TimingMetric` enum (may still be useful)
- `ForwardTiming` (replaced by `NewForwardTiming`)
- `BackwardTiming` (replaced by `NewBackwardTiming`)
- `IterationTiming` (replaced by `NewIterationTiming`)

`timing/mod.rs` exports both old and new types:
```rust
pub use metrics::{BackwardTiming, ForwardTiming, IterationTiming, TimingMetric};
```

### Target State

- Remove `ForwardTiming`, `BackwardTiming`, `IterationTiming` from `metrics.rs`
- Keep `TimingMetric` if still used, or remove if not
- Rename new types by removing `New` prefix (or keep as-is)
- Clean exports in `mod.rs`

## Specification

### Types to Remove

| Type | Location | Replacement |
|------|----------|-------------|
| `ForwardTiming` | `metrics.rs:52` | `NewForwardTiming` in `forward.rs` |
| `BackwardTiming` | `metrics.rs:93` | `NewBackwardTiming` in `backward.rs` |
| `IterationTiming` | `metrics.rs:135` | `NewIterationTiming` in `iteration.rs` |

### Types to Keep or Review

| Type | Decision | Reason |
|------|----------|--------|
| `TimingMetric` enum | Review | Check if still used; may be useful for metrics indexing |

### Export Updates

In `timing/mod.rs`:
```rust
// OLD
pub use metrics::{BackwardTiming, ForwardTiming, IterationTiming, TimingMetric};

// NEW
// Remove ForwardTiming, BackwardTiming, IterationTiming exports
// Keep TimingMetric if used
```

### Optional: Rename New Types

Consider removing `New` prefix after old types are gone:
- `NewForwardTiming` → `ForwardTiming`
- `NewBackwardTiming` → `BackwardTiming`
- `NewIterationTiming` → `IterationTiming`

This is optional and can be deferred to avoid churn.

## Acceptance Criteria

- [ ] Old `ForwardTiming` removed from `metrics.rs`
- [ ] Old `BackwardTiming` removed from `metrics.rs`
- [ ] Old `IterationTiming` removed from `metrics.rs`
- [ ] `timing/mod.rs` exports cleaned up
- [ ] No compilation errors
- [ ] `cargo test -p powers-rs timing` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Check for usages

```bash
# Check if old types are still used
grep -rn "use.*metrics::ForwardTiming" src/
grep -rn "use.*metrics::BackwardTiming" src/
grep -rn "use.*metrics::IterationTiming" src/
grep -rn "timing::ForwardTiming" src/
grep -rn "timing::BackwardTiming" src/
grep -rn "timing::IterationTiming" src/
```

### Step 2: Remove type definitions

Delete from `metrics.rs`:
- `ForwardTiming` struct and impl (lines ~52-89)
- `BackwardTiming` struct and impl (lines ~93-131)
- `IterationTiming` struct and impl (lines ~135-179)
- Associated tests

### Step 3: Clean up mod.rs exports

```rust
// Remove these from metrics re-export
// pub use metrics::{BackwardTiming, ForwardTiming, IterationTiming, TimingMetric};

// Keep only if TimingMetric is used
pub use metrics::TimingMetric;
```

### Step 4: Check TimingMetric usage

```bash
grep -rn "TimingMetric" src/
```

If not used, remove the entire `TimingMetric` enum and potentially the `metrics.rs` file.

### Pitfalls to Avoid

- ⚠️ Don't remove `TimingMetric` if it's used for metrics indexing
- ⚠️ Tests in `metrics.rs` reference old types - remove or update them
- ⚠️ Some code may import old types indirectly

## Testing Requirements

### Unit Tests

- Remove or update tests that reference old types
- Ensure new type tests still pass

### Integration Tests

```bash
cargo test -p powers-rs
```

### Compilation Tests

```bash
cargo build -p powers-rs --all-targets
```

## Documentation Requirements

- [ ] Update module docstring in `metrics.rs` if file is kept
- [ ] Remove if `metrics.rs` becomes empty

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward deletion once usages are verified gone
