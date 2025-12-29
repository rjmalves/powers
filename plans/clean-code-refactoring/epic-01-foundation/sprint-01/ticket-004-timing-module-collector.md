# [T-004] Implement TimingCollector Trait

> **Epic**: [Epic 1: Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Infrastructure Setup](./00-sprint-overview.md)
> **Dependencies**: [T-003](./ticket-003-timing-module-guard.md)
> **Blocks**: None (enables Epic 3 timing integration)

---

## ⚠️ CRITICAL: No Algorithm Changes

This ticket extends the timing infrastructure. **No existing algorithm code should be modified.** Run golden tests after completion.

---

## Files to Read Before Starting

- [Master Plan: Timing Architecture](../../../00-master-plan.md#timing-architecture-zero-pollution-instrumentation)
- [Master Plan: Parallel Overhead Tracking](../../../00-master-plan.md#3-parallel-overhead-tracking-strategy)
- `src/timing/guard.rs` - TimingGuard from T-003
- `src/sddp/mod.rs` lines 32-130 - Current timing structs (reference only)

---

## Context

### Background

Building on the TimingGuard from T-003, we need a trait-based abstraction for timing collection. This enables:
1. Testing without actual timing (mock implementations)
2. Different collection strategies (atomic for parallel, cell for single-threaded)
3. Feature-gated elimination of timing entirely

**Key requirement from Master Plan**: The timing infrastructure must preserve precise values and track parallel overhead explicitly—never overwrite measured values.

### Current State

- TimingGuard exists from T-003
- No abstraction for timing collection strategy
- Current code uses ad-hoc structs with overlapping fields

---

## Specification

### Files to Create

```
src/timing/
├── mod.rs           # Updated exports
├── guard.rs         # From T-003
├── collector.rs     # TimingCollector trait
├── metrics.rs       # TimingMetric enum, timing structs
└── atomic.rs        # AtomicTimingCollector for parallel use
```

### TimingMetric Enum

```rust
// src/timing/metrics.rs

use std::time::Duration;
use std::cell::Cell;

/// All timing metrics tracked in SDDP execution.
/// Organized hierarchically by algorithm phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimingMetric {
    // Forward pass - sequential components
    SaaSampling,
    ForwardPostprocessing,
    
    // Forward pass - per-trajectory (parallel)
    ForwardModelPrep,
    ForwardSolver,
    ForwardExtraction,
    
    // Forward pass - parallel overhead (computed, not measured)
    ForwardParallelWallTime,
    ForwardParallelOverhead,
    
    // Backward pass - sequential components
    BackwardPreprocessing,
    CutSelection,
    FcfUpdate,
    HandlerApplication,
    
    // Backward pass - per-stage (sequential within stage, parallel across branchings)
    BackwardModelPrep,
    BackwardSolver,
    BackwardExtraction,
    CutComputation,
}

impl TimingMetric {
    pub const COUNT: usize = 16;
    
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Forward pass timing with explicit parallel overhead tracking.
/// 
/// ⚠️ CRITICAL: `model_preprocessing`, `solver`, and `model_postprocessing` contain
/// the PRECISE measured values. They are NEVER overwritten or redistributed.
/// `parallel_overhead` is COMPUTED as the difference between wall time and CPU time.
#[derive(Debug, Clone, Default)]
pub struct ForwardTiming {
    pub saa_sampling: Cell<Duration>,
    
    // Wall-clock time for parallel section
    pub parallel_wall_time: Cell<Duration>,
    
    // Precise CPU times (NEVER overwritten)
    pub model_preprocessing: Cell<Duration>,
    pub solver: Cell<Duration>,
    pub model_postprocessing: Cell<Duration>,
    
    // Computed: parallel_wall_time - avg(per_trajectory_cpu_time)
    pub parallel_overhead: Cell<Duration>,
    
    pub postprocessing: Cell<Duration>,
}

/// Backward pass timing.
#[derive(Debug, Clone, Default)]
pub struct BackwardTiming {
    pub preprocessing: Cell<Duration>,
    pub model_preprocessing: Cell<Duration>,
    pub solver: Cell<Duration>,
    pub model_postprocessing: Cell<Duration>,
    pub cut_computation: Cell<Duration>,
    pub cut_selection: Cell<Duration>,
    pub fcf_update: Cell<Duration>,
    pub handler_application: Cell<Duration>,
}

/// Complete timing for one SDDP iteration.
#[derive(Debug, Clone, Default)]
pub struct IterationTiming {
    pub forward: ForwardTiming,
    pub backward: BackwardTiming,
    pub total: Cell<Duration>,
}

impl IterationTiming {
    /// Compute parallel overhead for forward pass.
    /// Call this AFTER all trajectory timings have been collected.
    /// 
    /// parallel_overhead = parallel_wall_time - (model_prep + solver + post) 
    pub fn compute_forward_parallel_overhead(&self) {
        let cpu_time = self.forward.model_preprocessing.get()
            + self.forward.solver.get()
            + self.forward.model_postprocessing.get();
        
        let wall_time = self.forward.parallel_wall_time.get();
        let overhead = wall_time.saturating_sub(cpu_time);
        
        self.forward.parallel_overhead.set(overhead);
    }
}
```

### TimingCollector Trait

```rust
// src/timing/collector.rs

use std::time::Duration;
use super::metrics::TimingMetric;

/// Trait for timing collection strategies.
/// 
/// This abstraction enables:
/// - Testing without actual timing (NullCollector)
/// - Thread-safe collection (AtomicCollector) 
/// - Single-threaded collection (CellCollector)
/// - Feature-gated elimination
pub trait TimingCollector: Send + Sync {
    /// Record a duration for the given metric.
    fn record(&self, metric: TimingMetric, duration: Duration);
    
    /// Get the current value for a metric.
    fn get(&self, metric: TimingMetric) -> Duration;
    
    /// Reset all metrics to zero.
    fn reset(&self);
}

/// No-op implementation for when timing is disabled.
/// All methods compile to nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullTimingCollector;

impl TimingCollector for NullTimingCollector {
    #[inline(always)]
    fn record(&self, _metric: TimingMetric, _duration: Duration) {}
    
    #[inline(always)]
    fn get(&self, _metric: TimingMetric) -> Duration {
        Duration::ZERO
    }
    
    #[inline(always)]
    fn reset(&self) {}
}
```

### AtomicTimingCollector

```rust
// src/timing/atomic.rs

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use super::collector::TimingCollector;
use super::metrics::TimingMetric;

/// Thread-safe timing collector using atomic operations.
/// Suitable for parallel sections where multiple threads record timing.
pub struct AtomicTimingCollector {
    // Store nanoseconds as u64 for atomic operations
    metrics: [AtomicU64; TimingMetric::COUNT],
}

impl AtomicTimingCollector {
    pub fn new() -> Self {
        // Initialize all metrics to zero
        Self {
            metrics: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl Default for AtomicTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingCollector for AtomicTimingCollector {
    fn record(&self, metric: TimingMetric, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        self.metrics[metric.index()].fetch_add(nanos, Ordering::Relaxed);
    }
    
    fn get(&self, metric: TimingMetric) -> Duration {
        let nanos = self.metrics[metric.index()].load(Ordering::Relaxed);
        Duration::from_nanos(nanos)
    }
    
    fn reset(&self) {
        for metric in &self.metrics {
            metric.store(0, Ordering::Relaxed);
        }
    }
}

// Safety: AtomicU64 is Send + Sync
unsafe impl Send for AtomicTimingCollector {}
unsafe impl Sync for AtomicTimingCollector {}
```

### Updated mod.rs

```rust
// src/timing/mod.rs

//! Zero-pollution timing infrastructure for performance measurement.
//!
//! # Design Principles
//!
//! 1. **Preserve Precise Values**: Timing values are NEVER overwritten or redistributed
//! 2. **Explicit Parallel Overhead**: Track scheduling overhead as a separate metric
//! 3. **Zero-Cost When Disabled**: Compile-time elimination via feature flags
//! 4. **RAII-Based**: TimingGuard automatically records on drop
//!
//! # Usage
//!
//! ```ignore
//! use powers_rs::timing::{TimingGuard, IterationTiming};
//!
//! let timing = IterationTiming::default();
//! {
//!     let _guard = TimingGuard::new(&timing.forward.solver);
//!     // ... solver work ...
//! }
//! // timing.forward.solver now contains elapsed time
//! ```

mod guard;
mod collector;
mod metrics;

#[cfg(feature = "timing")]
mod atomic;

pub use guard::TimingGuard;
pub use collector::{TimingCollector, NullTimingCollector};
pub use metrics::{TimingMetric, ForwardTiming, BackwardTiming, IterationTiming};

#[cfg(feature = "timing")]
pub use atomic::AtomicTimingCollector;
```

---

## Acceptance Criteria

- [ ] `TimingMetric` enum covers all SDDP timing points
- [ ] `TimingCollector` trait defined with record/get/reset
- [ ] `NullTimingCollector` provides zero-cost no-op
- [ ] `AtomicTimingCollector` provides thread-safe collection
- [ ] `IterationTiming` struct matches master plan design
- [ ] `ForwardTiming` has explicit `parallel_overhead` field
- [ ] `compute_forward_parallel_overhead()` computes overhead correctly
- [ ] All code compiles with and without `timing` feature
- [ ] Unit tests for all implementations
- [ ] Golden tests pass

### Correctness Verification

- [ ] `cargo test` passes
- [ ] `cargo test --features timing` passes
- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] No changes to algorithm code

---

## Implementation Guide

### Suggested Approach

1. Create `src/timing/metrics.rs` with enums and structs
2. Create `src/timing/collector.rs` with trait and NullCollector
3. Create `src/timing/atomic.rs` with AtomicTimingCollector
4. Update `src/timing/mod.rs` with exports
5. Write comprehensive tests
6. Verify feature flag behavior

### Pitfalls to Avoid

- ⚠️ Don't use `Mutex` in AtomicTimingCollector—use atomic operations
- ⚠️ Don't forget `#[cfg(feature = "timing")]` for atomic module
- ⚠️ Don't modify any existing SDDP code
- ⚠️ Don't implement aggregation that overwrites values (see master plan)

---

## Testing Requirements

### Unit Tests

- [ ] AtomicTimingCollector records correctly from single thread
- [ ] AtomicTimingCollector records correctly from multiple threads
- [ ] NullTimingCollector methods are truly no-ops
- [ ] TimingMetric::index() returns unique values
- [ ] IterationTiming::compute_forward_parallel_overhead() calculates correctly
- [ ] ForwardTiming fields are independent (not overwritten)

### Compilation Tests

- [ ] Compiles without `timing` feature
- [ ] Compiles with `timing` feature
- [ ] AtomicTimingCollector only available with `timing` feature

---

## Documentation Requirements

- [ ] Doc comments on all public types
- [ ] Module documentation explaining design principles
- [ ] Examples in documentation
- [ ] Note about parallel overhead calculation

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear specification from master plan, straightforward trait implementation

---

## Definition of Done

- [ ] All types implemented per specification
- [ ] Feature flags working correctly
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Documentation complete
- [ ] Code reviewed
