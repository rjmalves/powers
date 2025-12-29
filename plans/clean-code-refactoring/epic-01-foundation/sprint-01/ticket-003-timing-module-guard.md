# [T-003] Implement TimingGuard

> **Epic**: [Epic 1: Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Infrastructure Setup](./00-sprint-overview.md)
> **Dependencies**: [T-001](./ticket-001-golden-test-infrastructure.md) (for validation)
> **Blocks**: [T-004](./ticket-004-timing-module-collector.md)

---

## ⚠️ CRITICAL: No Algorithm Changes

This ticket creates new infrastructure code. **No existing algorithm code should be modified.** The timing module is additive—we are not yet integrating it with the SDDP implementation.

After completing this ticket, run golden tests to verify nothing was accidentally changed.

---

## Files to Read Before Starting

- [Master Plan: Timing Architecture](../../../00-master-plan.md#timing-architecture-zero-pollution-instrumentation) - Design specification
- `src/sddp/mod.rs` lines 32-130 - Current timing structs (for reference, not modification)
- `Cargo.toml` - Feature flag patterns

---

## Context

### Background

The current timing infrastructure is scattered and pollutes business logic with `Instant::now()` and `.elapsed()` calls. We're creating a new timing module with RAII guards that will eventually replace the existing approach.

**Key requirement**: The timing infrastructure must **never overwrite measured values**. It must preserve precise timing and track parallel overhead explicitly.

### Current State

- 48+ `Instant::now()` / `.elapsed()` calls scattered in code
- 4+ timing structs with overlapping fields
- No feature-gated timing elimination

---

## Specification

### Module Structure

Create `src/timing/` with:
```
src/timing/
├── mod.rs       # Public exports, feature gates
└── guard.rs     # TimingGuard implementation
```

### TimingGuard Implementation

```rust
// src/timing/guard.rs

use std::cell::Cell;
use std::time::{Duration, Instant};

/// Zero-cost timing guard that records duration on drop.
/// When `timing` feature is disabled, this compiles to nothing.
pub struct TimingGuard<'a> {
    #[cfg(feature = "timing")]
    start: Instant,
    #[cfg(feature = "timing")]
    target: &'a Cell<Duration>,
    #[cfg(not(feature = "timing"))]
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> TimingGuard<'a> {
    /// Create a new timing guard that will add elapsed time to `target` on drop.
    #[inline(always)]
    #[cfg(feature = "timing")]
    pub fn new(target: &'a Cell<Duration>) -> Self {
        Self {
            start: Instant::now(),
            target,
        }
    }
    
    #[inline(always)]
    #[cfg(not(feature = "timing"))]
    pub fn new(_target: &'a Cell<Duration>) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "timing")]
impl Drop for TimingGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.target.set(self.target.get() + elapsed);
    }
}

/// Macro for clean timing scope creation.
/// Usage: time_scope!(timing_struct, field_name);
#[macro_export]
macro_rules! time_scope {
    ($timing:expr, $field:ident) => {
        let _guard = $crate::timing::TimingGuard::new(&$timing.$field);
    };
}
```

### Feature Flag Configuration

Add to `Cargo.toml`:
```toml
[features]
timing = []
timing-detailed = ["timing"]
```

### Module Exports

```rust
// src/timing/mod.rs

mod guard;

pub use guard::TimingGuard;

// Re-export macro at crate root level
// (macro is already exported via #[macro_export])
```

---

## Acceptance Criteria

- [ ] `src/timing/mod.rs` exists with proper exports
- [ ] `src/timing/guard.rs` implements `TimingGuard`
- [ ] `time_scope!` macro is available at crate root
- [ ] Code compiles with `--features timing`
- [ ] Code compiles without `timing` feature (zero-cost)
- [ ] Unit tests verify timing accumulation
- [ ] Unit tests verify zero-cost when feature disabled
- [ ] Golden tests still pass (no algorithm changes)

### Correctness Verification

- [ ] `cargo test` passes
- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] No changes to existing source files (only additions)

---

## Implementation Guide

### Suggested Approach

1. **Create module directory**:
   ```bash
   mkdir -p src/timing
   ```

2. **Add feature flags** to `Cargo.toml`:
   ```toml
   [features]
   default = []
   timing = []
   timing-detailed = ["timing"]
   ```

3. **Create `src/timing/mod.rs`**:
   ```rust
   //! Zero-pollution timing infrastructure for performance measurement.
   //!
   //! This module provides RAII-based timing that can be completely eliminated
   //! at compile time when the `timing` feature is disabled.
   //!
   //! # Features
   //!
   //! - `timing`: Enable basic timing collection
   //! - `timing-detailed`: Enable per-stage timing breakdown (implies `timing`)
   //!
   //! # Example
   //!
   //! ```ignore
   //! use std::cell::Cell;
   //! use std::time::Duration;
   //! use powers_rs::timing::TimingGuard;
   //!
   //! struct MyTiming {
   //!     operation: Cell<Duration>,
   //! }
   //!
   //! let timing = MyTiming { operation: Cell::new(Duration::ZERO) };
   //! {
   //!     let _guard = TimingGuard::new(&timing.operation);
   //!     // ... do work ...
   //! } // timing.operation now contains elapsed time
   //! ```
   
   mod guard;
   
   pub use guard::TimingGuard;
   ```

4. **Create `src/timing/guard.rs`** (see Specification above)

5. **Add to `src/lib.rs`**:
   ```rust
   pub mod timing;
   ```

6. **Write tests** in `src/timing/guard.rs`:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use std::thread;
       
       #[test]
       #[cfg(feature = "timing")]
       fn test_timing_guard_accumulates() {
           let target = Cell::new(Duration::ZERO);
           
           {
               let _guard = TimingGuard::new(&target);
               thread::sleep(Duration::from_millis(10));
           }
           
           let elapsed = target.get();
           assert!(elapsed >= Duration::from_millis(10));
           assert!(elapsed < Duration::from_millis(50)); // Reasonable upper bound
       }
       
       #[test]
       #[cfg(feature = "timing")]
       fn test_timing_guard_adds_to_existing() {
           let target = Cell::new(Duration::from_millis(100));
           
           {
               let _guard = TimingGuard::new(&target);
               thread::sleep(Duration::from_millis(10));
           }
           
           let elapsed = target.get();
           assert!(elapsed >= Duration::from_millis(110));
       }
       
       #[test]
       fn test_timing_guard_compiles_without_feature() {
           // This test verifies the code compiles in both modes
           let target = Cell::new(Duration::ZERO);
           let _guard = TimingGuard::new(&target);
           // In non-timing mode, this should be a no-op
       }
   }
   ```

7. **Verify compilation in both modes**:
   ```bash
   cargo build
   cargo build --features timing
   cargo test --features timing
   ```

8. **Run golden tests**:
   ```bash
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Create

- `src/timing/mod.rs`
- `src/timing/guard.rs`

### Key Files to Modify

- `Cargo.toml` (add features)
- `src/lib.rs` (add `pub mod timing;`)

### Pitfalls to Avoid

- ⚠️ Don't modify any existing timing code in `src/sddp/mod.rs`
- ⚠️ Don't forget the `#[inline(always)]` attributes for zero-cost
- ⚠️ Don't use `&mut` for the target—use `Cell<Duration>` for interior mutability
- ⚠️ Don't forget to handle both feature-on and feature-off cases

---

## Testing Requirements

### Unit Tests

- [ ] Test that guard accumulates time correctly
- [ ] Test that guard adds to existing duration (not replaces)
- [ ] Test that multiple guards can accumulate to same target
- [ ] Test that code compiles without `timing` feature

### Compilation Tests

- [ ] `cargo build` succeeds (no features)
- [ ] `cargo build --features timing` succeeds
- [ ] `cargo test --features timing` passes

### Integration Tests

- [ ] Golden tests pass after changes

---

## Documentation Requirements

- [ ] Doc comments on `TimingGuard`
- [ ] Doc comments on `time_scope!` macro
- [ ] Module-level documentation in `mod.rs`
- [ ] Example in documentation

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear specification, straightforward implementation, but needs careful handling of feature flags

---

## Definition of Done

- [ ] TimingGuard implemented with feature flags
- [ ] time_scope! macro available
- [ ] Compiles with and without timing feature
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Documentation complete
- [ ] Code reviewed
