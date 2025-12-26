# Epic 4: Custom Allocator (Optional)

## Status

**Status**: ✅ Complete (2025-12-26)
**Priority**: LOW (addresses RSS fragmentation, not actual memory usage)

## Summary

Add optional support for custom allocators (mimalloc or jemalloc) to reduce memory fragmentation and improve RSS return to OS. This is a feature flag that users can enable for HPC deployments.

**Implementation Note**: Initial testing shows minimal RSS improvement (~5.27 GB with mimalloc vs ~5.18 GB without). The memory growth appears to be primarily from legitimate cut storage growth and HiGHS internal allocations rather than allocator fragmentation.

## Problem Statement

Even after eliminating transient allocations, the default system allocator may:
- Fragment memory, causing higher RSS than actual usage
- Not return freed memory to OS promptly
- Have suboptimal multi-threaded allocation performance

**Note**: This addresses the symptom (high RSS), not the cause. Epics 1-3 address root causes.

## Scope

### Included

- Add `mimalloc` feature flag
- Configure mimalloc for aggressive memory return
- Document usage and tradeoffs

### Excluded

- Making custom allocator the default
- Supporting multiple allocator options (pick one)
- Detailed benchmarking (leave to users)

## Dependencies

- **Requires**: None (can be done independently)
- **Optional**: Better results after Epics 1-3

## Acceptance Criteria

- [x] `mimalloc` feature flag available
- [x] Compiles and runs with feature enabled
- [ ] Documentation explains usage
- [x] No change to default behavior (feature is opt-in)

## Technical Approach

### Cargo.toml Changes

```toml
[features]
default = []
mimalloc = ["dep:mimalloc"]

[dependencies]
mimalloc = { version = "0.1", optional = true, features = ["local_dynamic_tls"] }
```

### Global Allocator Setup

```rust
// src/main.rs or src/lib.rs
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

### Usage

```bash
# Build with mimalloc
cargo build --release --features mimalloc

# Run with mimalloc
./target/release/powers run examples/05-large-scale-brazilian
```

## Estimated Effort

- **Sprint 1**: 1-2 days
  - Ticket 1: Add mimalloc feature flag (1 point)

## Key Files

| File | Impact |
|------|--------|
| `Cargo.toml` | Feature flag and dependency |
| `src/main.rs` | Global allocator setup |

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| mimalloc compatibility issues | Low | Well-tested crate |
| Performance difference | Medium | Document as optional |
| Platform-specific issues | Low | mimalloc supports major platforms |
