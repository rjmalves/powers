# [TICKET-001] Add mimalloc feature flag

> **Epic**: [Epic 4: Custom Allocator](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: None

## Context

### Background

The default system allocator may not return freed memory to the OS, causing high RSS even when actual memory usage is lower. mimalloc is a fast, general-purpose allocator that can improve this behavior.

### Files to Read Before Starting

- `Cargo.toml` - Feature flags
- `src/main.rs` - Entry point

## Specification

### Changes

1. Add `mimalloc` optional dependency
2. Add `mimalloc` feature flag
3. Set global allocator when feature enabled
4. Document in README

### Behavior

- Default: No change (system allocator)
- With `--features mimalloc`: Use mimalloc as global allocator

## Acceptance Criteria

- [ ] `cargo build --release` works (default)
- [ ] `cargo build --release --features mimalloc` works
- [ ] Runtime behavior unchanged
- [ ] README documents the feature

## Implementation Guide

### Step 1: Update Cargo.toml

```toml
[features]
default = []
mimalloc = ["dep:mimalloc"]

[dependencies]
mimalloc = { version = "0.1", optional = true, features = ["local_dynamic_tls"] }
```

### Step 2: Update src/main.rs

```rust
// At the top of main.rs, before other code

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

### Step 3: Update README.md

Add section:

```markdown
## Memory Optimization

For HPC deployments with strict memory requirements, enable the `mimalloc` allocator:

```bash
cargo build --release --features mimalloc
```

This can reduce memory fragmentation and improve RSS return to OS.
```

### Pitfalls to Avoid

- ⚠️ Don't make mimalloc the default (may have compatibility issues)
- ⚠️ `local_dynamic_tls` feature needed for thread-local performance

## Testing Requirements

### Build Tests

- [ ] `cargo build --release` succeeds
- [ ] `cargo build --release --features mimalloc` succeeds

### Integration Tests

- [ ] Example 01 runs correctly with mimalloc

## Documentation Requirements

- [ ] README section on memory optimization
- [ ] Feature flag documented in Cargo.toml

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Simple feature flag addition

## Definition of Done

- [ ] Feature flag added
- [ ] Compiles both ways
- [ ] Documentation added
- [ ] PR merged
