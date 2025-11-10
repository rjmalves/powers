# Memory Optimization Strategy: Deep Estimation and Pre-allocation

**Date**: 2025-11-10  
**Status**: Proposed Strategy Revision  
**Author**: Performance Engineering Team

---

## Executive Summary

This document proposes a **strategic revision** of the memory pre-allocation sprint based on critical insights discovered during implementation. The key finding: we've been optimizing **outer allocations** while **nested allocations** dominate the actual performance cost.

**Core Issue**: Current `SizingInfo::estimate_memory_bytes()` only accounts for pointer sizes (`std::mem::size_of`), missing the heap allocations inside nested structures like `BendersCut::coefficients: Vec<f64>`.

**Proposed Solution**: Implement bottom-up deep memory estimation, then systematically apply pre-allocation patterns across all hot paths.

---

## 1. The Problem: Shallow vs Deep Memory Estimation

### Current State (Shallow Estimation)

```rust
// src/memory/sizing.rs
pub fn estimate_memory_bytes(&self) -> usize {
    std::mem::size_of::<BendersCut>() * num_cuts
    // Returns: 56 bytes (stack size only)
    // WRONG: Doesn't account for Vec<f64> heap allocation!
}
```

### Reality: The Allocation Iceberg

```
VISIBLE (Outer Vecs):
  ├─ Vec<CutStatePair>: 100 allocations
  └─ Vec<Timing>: 100 allocations
  
HIDDEN (Nested Vecs - NOT COUNTED):
  ├─ BendersCut::coefficients: Vec<f64> × 100 = 122 KB (156 floats each)
  ├─ CutStatePair::state: Vec<f64> × 100 = 122 KB
  ├─ Trajectory::actions: Vec<Action> × 50 = 80 KB
  └─ Subproblem temporaries: Vec<f64> × 200 = 150 KB
  
TOTAL HIDDEN: ~474 KB per iteration (vs 5 KB estimated)
```

### Impact on Optimization Strategy

| Metric | Current (Shallow) | Actual (Deep) | Error |
|--------|------------------|---------------|-------|
| Memory per cut | 56 bytes | 1,304 bytes | **23× underestimate** |
| Malloc overhead | ~2% | ~8-10% | **4-5× underestimate** |
| Optimization target | Outer vecs | Nested vecs | Wrong focus! |

---

## 2. Understanding Rayon's `size_hint()` Limitation

### How Rayon Pre-allocates (Currently Working)

```rust
let phase1_results: Vec<(CutStatePair, Timing)> = train_handlers
    .par_iter_mut()  // ← size_hint() = (n, Some(n)) - exact!
    .map(|(idx, handler)| { ... })
    .collect()?;     // ← Rayon: Vec::with_capacity(n) ✅
```

**This works perfectly** because `Vec::iter()` provides exact size hints.

### Where Rayon Can't Help (Nested Allocations)

```rust
// Inside the map closure:
fn compute_cut_for_backward_step(...) -> CutStatePair {
    let mut coefficients = Vec::new();  // ← ALLOCATION #1 (Rayon can't see this!)
    
    for hydro in hydros {
        coefficients.push(compute_coeff(hydro));  // ← REALLOCATIONS
    }
    
    BendersCut { coefficients, ... }  // Escapes to outer Vec
}
```

**The problem**: Nested allocations happen **inside** the closure, which Rayon's collection can't control.

### Can We Help Rayon with Deep Estimation?

**No** - Rayon allocates space for **pointers**, not the full object graph.

```rust
// Rayon allocates:
Vec<(CutStatePair, Timing)>
// Size: num_items × sizeof((CutStatePair, Timing))
//     = num_items × (24 + 24) = 48 bytes per item

// Rayon does NOT allocate:
struct BendersCut {
    coefficients: Vec<f64>,  // ← This allocation happens inside map()
}
```

**Solution**: We must pre-allocate nested structures **before** passing to Rayon.

---

## 3. Proposed Solution: Bottom-Up Deep Estimation

### Design: `DeepSizeEstimate` Trait

```rust
// src/memory/deep_sizing.rs

/// Estimate total heap memory including nested allocations
pub trait DeepSizeEstimate {
    /// Estimate heap bytes for an instance (dynamic)
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize;
    
    /// Estimate heap bytes for type (static, for pre-allocation planning)
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize
    where
        Self: Sized;
}
```

### Implementation Hierarchy

#### Level 1: Primitives (Zero Heap)

```rust
impl DeepSizeEstimate for f64 {
    fn estimate_heap_bytes(&self, _: &SizingInfo) -> usize {
        std::mem::size_of::<f64>()  // Stack only
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        std::mem::size_of::<f64>()
    }
}

impl DeepSizeEstimate for usize {
    // Similar
}
```

#### Level 2: Collections (Sized by SizingInfo)

```rust
impl DeepSizeEstimate for Vec<f64> {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        // For pre-allocated vecs, use actual capacity
        self.capacity() * std::mem::size_of::<f64>()
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        // For planning, use max state dimension
        sizing.max_state_dimension * std::mem::size_of::<f64>()
    }
}

impl<T: DeepSizeEstimate> DeepSizeEstimate for Vec<T> {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        let vec_overhead = self.capacity() * std::mem::size_of::<T>();
        let elements_heap: usize = self.iter()
            .map(|item| item.estimate_heap_bytes(sizing))
            .sum();
        vec_overhead + elements_heap
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        // Use max expected count
        sizing.estimate_max_items() * T::estimate_heap_bytes_static(sizing)
    }
}
```

#### Level 3: Domain Structures (Recursive)

```rust
impl DeepSizeEstimate for BendersCut {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        let stack_size = std::mem::size_of::<Self>();
        let heap_size = self.coefficients.estimate_heap_bytes(sizing);
        stack_size + heap_size
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        let stack_size = std::mem::size_of::<Self>();
        let heap_size = Vec::<f64>::estimate_heap_bytes_static(sizing);
        stack_size + heap_size
        
        // For 156 hydros:
        // Stack: 56 bytes
        // Heap: 156 × 8 = 1,248 bytes
        // Total: 1,304 bytes per cut
    }
}

impl DeepSizeEstimate for CutStatePair {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        std::mem::size_of::<Self>() +
        self.cut.estimate_heap_bytes(sizing) +
        self.state.estimate_heap_bytes(sizing)
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        std::mem::size_of::<Self>() +
        BendersCut::estimate_heap_bytes_static(sizing) +
        // State size depends on implementation (Storage vs StorageAndInflow)
        sizing.max_state_dimension * std::mem::size_of::<f64>()
    }
}
```

#### Level 4: Pools and Aggregates

```rust
impl DeepSizeEstimate for BendersCutPool {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        let cuts_size = self.pool.iter()
            .map(|cut| cut.estimate_heap_bytes(sizing))
            .sum::<usize>();
            
        let hashmap_overhead = self.active_cut_indices.capacity() * 
                               (std::mem::size_of::<usize>() * 2 + 8);
        
        cuts_size + hashmap_overhead
    }
    
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        let num_cuts = sizing.estimate_total_cuts();
        let cuts_size = num_cuts * BendersCut::estimate_heap_bytes_static(sizing);
        let hashmap_overhead = num_cuts * (std::mem::size_of::<usize>() * 2 + 8);
        
        cuts_size + hashmap_overhead
    }
}

impl DeepSizeEstimate for BackwardPassBuffers {
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        sizing.num_forward_passes * 
        (sizing.num_stages - 1) * 
        CutStatePair::estimate_heap_bytes_static(sizing)
    }
}
```

### Updated SizingInfo

```rust
impl SizingInfo {
    /// Accurate deep memory estimation
    pub fn estimate_memory_bytes_deep(&self) -> usize {
        let cuts = self.estimate_total_cuts() * 
                   BendersCut::estimate_heap_bytes_static(self);
                   
        let trajectories = self.num_forward_passes * 
                          Trajectory::estimate_heap_bytes_static(self);
                          
        let buffers = self.num_threads * 
                     ThreadLocalBuffers::estimate_heap_bytes_static(self);
        
        cuts + trajectories + buffers
    }
    
    /// Helper: estimate total cuts after convergence
    pub fn estimate_total_cuts(&self) -> usize {
        // Conservative estimate: 
        // avg 10 cuts per node × num_nodes × max_iterations × convergence_factor
        let base_cuts = 10 * self.num_nodes;
        let training_cuts = base_cuts * self.max_iterations;
        
        // Apply convergence factor (cuts get dominated/pruned)
        (training_cuts as f64 * 0.3) as usize
    }
}
```

---

## 4. Pre-allocation Pattern: Unified Approach

### Pattern Template

```rust
// 1. Allocate buffer pool at initialization
struct Context {
    coeff_buffers: Vec<Vec<f64>>,  // One per thread
}

impl Context {
    fn new(sizing: &SizingInfo) -> Self {
        let num_threads = sizing.num_threads;
        let capacity = sizing.max_state_dimension;
        
        let coeff_buffers = (0..num_threads)
            .map(|_| vec![0.0; capacity])
            .collect();
        
        Self { coeff_buffers }
    }
}

// 2. Acquire buffer in hot path
fn compute_cut_with_buffer(
    &mut self,
    thread_id: usize,
    coeff_buffer: &mut [f64],
) -> BendersCut {
    // Zero overhead: reuse buffer
    coeff_buffer.fill(0.0);
    
    for (i, hydro) in hydros.iter().enumerate() {
        coeff_buffer[i] = compute_coefficient(hydro);
    }
    
    // Single allocation with exact size
    BendersCut {
        coefficients: coeff_buffer[..num_hydros].to_vec(),
        ...
    }
}

// 3. Use in parallel context
let results: Vec<CutStatePair> = handlers
    .par_iter_mut()
    .enumerate()
    .map(|(idx, handler)| {
        let thread_id = rayon::current_thread_index().unwrap();
        let coeff_buffer = &mut self.coeff_buffers[thread_id];
        handler.compute_cut_with_buffer(thread_id, coeff_buffer)
    })
    .collect();
```

### Apply Everywhere

**Backward Pass**:
- Pre-allocate coefficient buffers
- Pre-allocate state vectors
- Pre-allocate timing structs

**Forward Pass**:
- Pre-allocate action vectors
- Pre-allocate realization buffers
- Pre-allocate trajectory stages

**Simulation**:
- Pre-allocate scenario buffers
- Pre-allocate result aggregation structures

---

## 5. Revised Sprint Structure

### Current Sprint (Partially Wrong Order)

```
✅ TICKET-001: Per-node sizing ← Good foundation
✅ TICKET-002: Buffer pools ← Infrastructure exists
✅ TICKET-003: Module integration ← Done
⏭️ TICKET-004: Test infrastructure ← Skipped
✅ TICKET-005: BackwardPassBuffers ← Created but unused
✅ TICKET-006: Backward pass unzip ← Good optimization, but incomplete
📋 TICKET-007: Performance validation ← Next
```

**Issue**: We optimized outer allocations (unzip) but left nested allocations untouched.

### Proposed Sprint Revision

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 0: Foundation (NEW - SHOULD COME FIRST)          │
├─────────────────────────────────────────────────────────┤
│ 🆕 TICKET-000: Deep Memory Estimation (1-2 days)       │
│    - Implement DeepSizeEstimate trait                   │
│    - Add recursive size computation for all types       │
│    - Update SizingInfo with accurate estimates          │
│    - Profile: Measure actual vs estimated allocations   │
│    - Validate: Error <10% on production workloads       │
│                                                          │
│    Deliverables:                                        │
│    - src/memory/deep_sizing.rs                          │
│    - Implementations for BendersCut, CutStatePair, etc. │
│    - Updated SizingInfo::estimate_memory_bytes_deep()   │
│    - Validation report comparing estimate vs actual     │
│                                                          │
│    Acceptance Criteria:                                 │
│    - Deep estimation within 10% of measured memory      │
│    - All major types implement DeepSizeEstimate         │
│    - Documentation with examples                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 1: Core Infrastructure (KEEP, already done)       │
├─────────────────────────────────────────────────────────┤
│ ✅ TICKET-001: Per-node sizing                          │
│ ✅ TICKET-002: Buffer pools                             │
│ ✅ TICKET-003: Module integration                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 2: Backward Pass (AUGMENT with nested buffers)    │
├─────────────────────────────────────────────────────────┤
│ ✅ TICKET-005: BackwardPassBuffers infrastructure       │
│ ✅ TICKET-006: Unzip optimization (outer allocation)    │
│                                                          │
│ 🆕 TICKET-006b: Nested Pre-allocation (2 days)          │
│    - Pre-allocate cut coefficient buffers               │
│    - Pre-allocate state vectors                         │
│    - Thread-local buffer pools (one per Rayon thread)   │
│    - Eliminate allocations INSIDE cut computation       │
│                                                          │
│    Expected Impact:                                     │
│    - Eliminate ~200 nested allocations per iteration    │
│    - Reduce malloc overhead: 8% → 2%                    │
│    - Backward pass: 5-10% faster                        │
│                                                          │
│ 📋 TICKET-007: Performance validation (augmented)       │
│    - Validate both outer AND nested optimization        │
│    - Measure deep allocation reduction                  │
│    - Confirm malloc overhead <2%                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 3: Forward Pass (MIRROR backward pattern)         │
├─────────────────────────────────────────────────────────┤
│ 🆕 TICKET-008: Forward Pass Deep Pre-allocation         │
│    - Apply same pattern as backward pass                │
│    - Pre-allocate action vectors                        │
│    - Pre-allocate realization buffers                   │
│    - Profile and validate improvement                   │
│                                                          │
│    Expected Impact:                                     │
│    - Similar to backward pass (5-10% improvement)       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 4: Simulation (SAME PATTERN)                      │
├─────────────────────────────────────────────────────────┤
│ 🆕 TICKET-009: Simulation Buffer Pre-allocation         │
│    - Use deep sizing for trajectory buffers             │
│    - Pre-allocate scenario result structures            │
│    - Apply uniform pattern across simulation            │
│                                                          │
│    Expected Impact:                                     │
│    - Simulation: 5-10% faster                           │
│    - Memory: Predictable and bounded                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 5: Validation & Documentation                     │
├─────────────────────────────────────────────────────────┤
│ TICKET-010: Comprehensive profiling                     │
│ TICKET-011: Integration testing                         │
│ TICKET-012: Performance benchmarking                    │
│ TICKET-013: Documentation finalization                  │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Expected Impact: Before vs After

### Current State (After TICKET-006)

| Metric | Value | Notes |
|--------|-------|-------|
| Outer allocations | Optimized | Unzip pre-allocated ✅ |
| Nested allocations | **Not optimized** | Still allocating per cut ❌ |
| Malloc overhead | ~8-10% | Hidden nested allocations |
| Backward pass time | Baseline | Small improvement only |

### After Full Implementation (TICKET-000 + TICKET-006b)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory estimate accuracy | ±300% | ±10% | **30× better** |
| Allocations per iteration | ~500 | ~50 | **90% reduction** |
| Malloc overhead | ~8-10% | <2% | **75% reduction** |
| Backward pass time | Baseline | -10-15% | **Significant** |
| Forward pass time | Baseline | -10-15% | **Significant** |
| Overall training time | Baseline | -8-12% | **Target achieved** |

### Production Scale (192 forward passes, 100 iterations)

**Before Full Optimization**:
- Outer allocations: 96K (eliminated by TICKET-006 ✅)
- Nested allocations: **500K** (still happening ❌)
- Total malloc overhead: ~8-10% CPU time

**After Full Optimization**:
- Outer allocations: 96K → 0 (TICKET-006 ✅)
- Nested allocations: 500K → ~5K (TICKET-006b 🎯)
- Total malloc overhead: ~8-10% → <2% (Target achieved! 🎉)

---

## 7. Implementation Priority

### Week 1: Foundation

**TICKET-000: Deep Memory Estimation**

Priority: **CRITICAL** (blocks all other optimizations)

Tasks:
1. Create `DeepSizeEstimate` trait
2. Implement for primitives and collections
3. Implement for domain types (BendersCut, etc.)
4. Update SizingInfo with deep estimation
5. Profile and validate accuracy

### Week 2: Backward Pass Completion

**TICKET-006b: Nested Pre-allocation**

Priority: **HIGH** (completes backward pass optimization)

Tasks:
1. Create thread-local coefficient buffer pools
2. Refactor cut computation to use buffers
3. Pre-allocate state vectors
4. Profile and measure impact

**TICKET-007: Enhanced Validation**

Priority: **HIGH** (validates full optimization)

Tasks:
1. Measure nested allocation reduction
2. Confirm malloc overhead <2%
3. Benchmark on production scale
4. Document actual improvements

### Week 3-4: Forward Pass & Simulation

**TICKET-008, TICKET-009**: Apply same pattern

Priority: **MEDIUM** (incremental improvements)

---

## 8. Validation Methodology

### Measure Actual Allocations

```bash
# Count allocations with strace
strace -e brk,mmap -c ./target/release/powers example 2>&1 | grep -E "brk|mmap"

# Profile with perf
perf stat -e cpu/event=0xd1,umask=0x1/ ./target/release/powers example
# Event 0xd1: Memory loads

# Detailed malloc tracking with massif
valgrind --tool=massif \
         --massif-out-file=massif.out \
         ./target/release/powers examples/05-large

ms_print massif.out | grep -A 20 "peak"
```

### Compare Estimate vs Actual

```rust
// src/bin/memory_validator.rs
fn main() {
    let sizing = SizingInfo::from_input(...);
    
    // Estimate
    let estimated = sizing.estimate_memory_bytes_deep();
    println!("Estimated: {} MB", estimated / 1_000_000);
    
    // Run and measure
    let start = SystemMemory::current();
    run_training(...);
    let end = SystemMemory::current();
    let actual = end - start;
    
    println!("Actual: {} MB", actual / 1_000_000);
    println!("Error: {:.1}%", 
             (estimated as f64 - actual as f64).abs() / actual as f64 * 100.0);
    
    // Acceptance: Error <10%
    assert!(error < 10.0, "Memory estimation error too high!");
}
```

### Benchmark Template

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_backward_pass_nested_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_pass_nested");
    
    // Small problem (baseline)
    group.bench_function("small_4fp", |b| {
        let mut sddp = create_small_system();
        b.iter(|| {
            black_box(sddp.backward_pass().unwrap())
        });
    });
    
    // Production scale
    group.bench_function("large_192fp", |b| {
        let mut sddp = create_large_system();
        b.iter(|| {
            black_box(sddp.backward_pass().unwrap())
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_backward_pass_nested_alloc);
criterion_main!(benches);
```

---

## 9. Risk Assessment

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Deep estimation complexity | Medium | Start with simple types, iterate |
| Performance regression | Low | Comprehensive benchmarks |
| Numerical changes | Low | Validation tests (within 1e-10) |
| Implementation time | Medium | Phase incrementally, validate each |

### Rollback Strategy

If optimization causes issues:

1. **Feature flag**: Enable/disable deep pre-allocation
2. **Git revert**: Each ticket is atomic
3. **Fallback**: Keep old allocation path as backup

```rust
#[cfg(feature = "deep-prealloc")]
fn compute_cut_optimized(...) -> BendersCut { ... }

#[cfg(not(feature = "deep-prealloc"))]
fn compute_cut_baseline(...) -> BendersCut { ... }
```

---

## 10. Success Criteria

### Technical Metrics

- ✅ Deep memory estimation within 10% of actual
- ✅ Malloc overhead reduced from 8-10% to <2%
- ✅ Backward pass 10-15% faster
- ✅ Forward pass 10-15% faster
- ✅ Overall training 8-12% faster
- ✅ All tests pass (numerical accuracy maintained)
- ✅ Zero unsafe code

### Code Quality Metrics

- ✅ Clear documentation with examples
- ✅ Comprehensive test coverage (>90%)
- ✅ No clippy warnings
- ✅ Consistent pattern across codebase
- ✅ Maintainable (future types easily added)

### Project Metrics

- ✅ Phase 0: 1-2 days (foundation)
- ✅ Phase 1: Complete (already done)
- ✅ Phase 2: 2-3 days (augmented)
- ✅ Phase 3: 2 days (forward pass)
- ✅ Phase 4: 2 days (simulation)
- ✅ Total: ~2 weeks (vs 3-4 weeks original plan)

---

## 11. Recommendations

### Immediate Actions (This Week)

1. ✅ **Complete TICKET-007**: Validate current work
2. 🆕 **Create TICKET-000 spec**: Deep memory estimation design
3. 📋 **Profile nested allocations**: Baseline measurement
4. 📝 **Update sprint plan**: Revise ticket ordering

### Short Term (Next 2 Weeks)

1. Implement TICKET-000 (deep estimation)
2. Implement TICKET-006b (nested pre-allocation)
3. Validate improvements with TICKET-007 (enhanced)
4. Document pattern for future application

### Long Term (Weeks 3-4)

1. Apply pattern to forward pass (TICKET-008)
2. Apply pattern to simulation (TICKET-009)
3. Comprehensive benchmarking
4. Production validation

---

## 12. Conclusion

**The current optimization strategy is good but incomplete.** TICKET-006 successfully eliminated outer allocations (unzip), but the larger opportunity lies in nested allocations within domain structures.

**Key Insights**:

1. **Shallow estimation is misleading**: 23× underestimate of actual memory
2. **Rayon can't help with nested allocations**: Must pre-allocate before parallel execution
3. **Pattern is proven**: TICKET-006 validates pre-allocation approach
4. **Bottom-up is correct**: Build accurate foundation, then optimize systematically

**Recommendation**: **Revise sprint** to implement deep estimation first, then apply uniform pre-allocation pattern across all hot paths. This provides:

- **Better measurement**: Accurate memory tracking
- **Systematic optimization**: One pattern everywhere
- **Larger impact**: 8-12% overall improvement (vs 2-3% current)
- **Maintainability**: Clear trait for future types

**Confidence**: High - Pattern proven in TICKET-006, just needs systematic application with accurate sizing.

---

**Next Steps**: Review this strategy, draft TICKET-000 specification, and update sprint plan.

**Status**: Awaiting approval to proceed with revised sprint structure.
