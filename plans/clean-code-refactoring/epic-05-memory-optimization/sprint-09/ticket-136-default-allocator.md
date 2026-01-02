# [T-136] Make Winning Allocator the Default

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: T-135
> **Blocks**: T-137, T-138, T-139, T-140

---

## Context

### Background

Based on the allocator comparison (T-135), we need to make the winning allocator the default, with an opt-out mechanism for users who need the system allocator.

### Current State

```toml
# Cargo.toml - current (allocators are opt-in)
[features]
mimalloc = ["dep:mimalloc"]
jemalloc = ["dep:tikv-jemallocator"]
```

## Specification

### Target Configuration

The winning allocator becomes the default. Users can opt-out via feature flag.

**If mimalloc wins:**

```toml
# Cargo.toml
[dependencies]
mimalloc = { version = "0.1", optional = true }

[features]
default = ["mimalloc"]
mimalloc = ["dep:mimalloc"]
system-allocator = []  # Opt-out flag
```

```rust
// src/main.rs
#[cfg(all(feature = "mimalloc", not(feature = "system-allocator")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**If jemalloc wins:**

```toml
# Cargo.toml
[dependencies]
tikv-jemallocator = { version = "0.6", optional = true }

[features]
default = ["jemalloc"]
jemalloc = ["dep:tikv-jemallocator"]
system-allocator = []  # Opt-out flag
```

```rust
// src/main.rs
#[cfg(all(feature = "jemalloc", not(feature = "system-allocator")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

### Behavior Matrix

| Build Command | Allocator Used |
|---------------|----------------|
| `cargo build` | Winner (default) |
| `cargo build --no-default-features` | System (glibc) |
| `cargo build --features system-allocator` | System (glibc) |
| `cargo build --no-default-features --features mimalloc` | mimalloc |
| `cargo build --no-default-features --features jemalloc` | jemalloc |

## Acceptance Criteria

- [ ] Winning allocator is in `default` features
- [ ] `system-allocator` feature flag allows opt-out
- [ ] `cargo build` uses winning allocator
- [ ] `cargo build --no-default-features` uses system allocator
- [ ] All feature combinations build successfully
- [ ] Conditional compilation is correct

## Implementation Guide

### Suggested Approach

1. Update Cargo.toml with new default feature
2. Update main.rs with proper cfg conditions
3. Test all feature combinations
4. Verify correct allocator is used in each case

### Key Files to Modify

- `Cargo.toml`: Update features section
- `src/main.rs`: Update allocator cfg conditions

### Verification Commands

```bash
# Default (should use winner)
cargo build --release
./target/release/powers run examples/01-single-stage --help

# System allocator
cargo build --release --no-default-features

# Explicit winner
cargo build --release --no-default-features --features mimalloc

# Explicit alternative
cargo build --release --no-default-features --features jemalloc
```

### Verifying Allocator in Use

Add temporary debug output or use:

```rust
// Temporary verification (remove after testing)
#[cfg(feature = "mimalloc")]
eprintln!("Using mimalloc allocator");

#[cfg(feature = "jemalloc")]
eprintln!("Using jemalloc allocator");

#[cfg(all(not(feature = "mimalloc"), not(feature = "jemalloc")))]
eprintln!("Using system allocator");
```

### Pitfalls to Avoid

- ⚠️ Ensure `system-allocator` properly disables custom allocators
- ⚠️ Test feature flag precedence carefully
- ⚠️ Don't break existing `--features mimalloc` users

## Testing Requirements

### Build Tests

- [ ] `cargo build` uses default allocator
- [ ] `cargo build --no-default-features` uses system allocator
- [ ] All feature combinations build successfully

### Runtime Tests

- [ ] Verify allocator choice at runtime (temporary debug)
- [ ] RSS test confirms expected behavior

## Documentation Requirements

- [ ] Update Cargo.toml comments
- [ ] Document allocator features in README.md
- [ ] Add section on memory management configuration

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Feature flag configuration requires careful testing

## Definition of Done

- [ ] Default allocator configured
- [ ] Opt-out mechanism works
- [ ] All feature combinations tested
- [ ] Documentation updated
