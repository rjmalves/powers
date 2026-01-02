# [T-132] Add jemalloc as Optional Dependency

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-133

---

## Context

### Background

jemalloc is a widely-used allocator in production systems (used by Firefox, Redis, etc.) known for better memory management than glibc. We need to add it as an optional dependency to compare its RSS behavior against mimalloc.

### Current State

Cargo.toml has mimalloc but not jemalloc:
```toml
mimalloc = { version = "0.1", optional = true }
```

## Specification

### Changes Required

1. Add `tikv-jemallocator` to Cargo.toml as optional dependency
2. Add `jemalloc` feature flag
3. Add conditional global allocator in main.rs
4. Ensure mutual exclusivity with mimalloc (or document behavior)

### Target Configuration

```toml
# Cargo.toml
[dependencies]
tikv-jemallocator = { version = "0.6", optional = true }

[features]
jemalloc = ["dep:tikv-jemallocator"]
```

```rust
// src/main.rs
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

### Behavior

- `--features jemalloc`: Use jemalloc
- `--features mimalloc`: Use mimalloc  
- Both features: jemalloc wins (document this)
- Neither: System allocator (glibc on Linux)

## Acceptance Criteria

- [ ] `tikv-jemallocator` added to Cargo.toml
- [ ] `jemalloc` feature flag defined
- [ ] Global allocator conditional compilation works
- [ ] `cargo build --release --features jemalloc` succeeds
- [ ] `cargo build --release --features mimalloc` still works
- [ ] `cargo test --features jemalloc` passes

## Implementation Guide

### Suggested Approach

1. Add dependency to Cargo.toml
2. Update main.rs with conditional allocator
3. Build and verify both features work independently
4. Run basic tests to ensure no issues

### Key Files to Modify

- `Cargo.toml`: Add dependency and feature
- `src/main.rs`: Add conditional global allocator

### Cargo.toml Changes

```toml
[dependencies]
# ... existing deps ...

# Optional: jemalloc allocator for better memory management
tikv-jemallocator = { version = "0.6", optional = true }

[features]
# ... existing features ...
jemalloc = ["dep:tikv-jemallocator"]
```

### main.rs Changes

```rust
// When jemalloc feature is enabled, use jemalloc as the global allocator.
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// When mimalloc feature is enabled (and jemalloc is not), use mimalloc.
#[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

### Pitfalls to Avoid

- ⚠️ jemalloc may require system dependencies on some platforms
- ⚠️ Ensure cfg conditions are mutually exclusive
- ⚠️ Test on Linux (primary target platform)

## Testing Requirements

### Build Tests

- [ ] `cargo build --release --features jemalloc` succeeds
- [ ] `cargo build --release --features mimalloc` succeeds
- [ ] `cargo build --release` (no features) succeeds
- [ ] `cargo build --release --features "jemalloc mimalloc"` succeeds (jemalloc wins)

### Unit Tests

- [ ] `cargo test --features jemalloc` passes
- [ ] `cargo test --features mimalloc` passes

## Documentation Requirements

- [ ] Update Cargo.toml comments for jemalloc
- [ ] Document feature flag behavior in README or docs

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple dependency addition, well-documented crate

## Definition of Done

- [ ] jemalloc dependency added
- [ ] Feature flag works correctly
- [ ] Both allocators can be built independently
- [ ] Basic tests pass with jemalloc
