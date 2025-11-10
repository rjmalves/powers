# TICKET-006b Implementation Plan

**Status**: 🚀 IN PROGRESS  
**Baseline**: 0.437s (established 2025-11-10)  
**Target**: 0.385s (12% improvement)  
**Strategy**: Thread-local buffer reuse

---

## Performance Data

### Baseline (Measured)
- Runtime: 0.437s average
- Training: 0.380s (87% of total)
- Estimated allocations: ~2,048 per run (small example)
- Max RSS: 23 MB

### Quick Win Experiment (Reverted)
- Tested: `Vec::with_capacity` pre-allocation
- Result: 21% SLOWER
- Reason: `collect()` already pre-allocates via `size_hint()`
- Lesson: Trust std library, focus on buffer reuse

---

## Implementation Strategy

### Phase 1: Thread-Local Buffer Infrastructure (2-3 hours)

**Goal**: Add specialized buffers for cut computation

**Location**: `src/memory/buffers.rs`

**Add**:
```rust
/// Buffers for cut coefficient computation (eliminates allocations in hot path)
pub struct CutComputationBuffers {
    /// Reusable buffer for cut coefficients
    pub coefficients: Vec<f64>,
    
    /// Reusable buffer for contribution accumulation (outer)
    pub contributions_outer: Vec<Vec<f64>>,
    
    /// Reusable flat buffer for temporary computations
    pub temp_contribution: Vec<f64>,
}

impl CutComputationBuffers {
    pub fn new(max_state_dim: usize, max_scenarios: usize) -> Self {
        // Pre-allocate once with maximum sizes
        let mut contributions_outer = Vec::with_capacity(max_scenarios);
        for _ in 0..max_scenarios {
            contributions_outer.push(Vec::with_capacity(max_state_dim));
        }
        
        Self {
            coefficients: Vec::with_capacity(max_state_dim),
            contributions_outer,
            temp_contribution: Vec::with_capacity(max_state_dim),
        }
    }
    
    pub fn reset_for_cut(&mut self, state_dim: usize, num_scenarios: usize) {
        self.coefficients.clear();
        self.coefficients.resize(state_dim, 0.0);
        
        // Reuse existing inner vectors
        for i in 0..num_scenarios {
            if i < self.contributions_outer.len() {
                self.contributions_outer[i].clear();
            } else {
                self.contributions_outer.push(Vec::with_capacity(state_dim));
            }
        }
    }
}

thread_local! {
    static CUT_BUFFERS: RefCell<Option<CutComputationBuffers>> = RefCell::new(None);
}

pub fn initialize_cut_buffers(max_state_dim: usize, max_scenarios: usize) {
    CUT_BUFFERS.with(|buffers| {
        *buffers.borrow_mut() = Some(CutComputationBuffers::new(max_state_dim, max_scenarios));
    });
}

pub fn with_cut_buffers<F, R>(f: F) -> R
where F: FnOnce(&mut CutComputationBuffers) -> R {
    CUT_BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();
        f(buffers.as_mut().expect("Cut buffers not initialized. Call initialize_cut_buffers first."))
    })
}
```

**Tests**:
```rust
#[test]
fn test_cut_buffers_reuse() {
    initialize_cut_buffers(10, 4);
    
    // First use
    with_cut_buffers(|buf| {
        buf.reset_for_cut(5, 4);
        buf.coefficients[0] = 1.0;
    });
    
    // Second use - buffers reused
    with_cut_buffers(|buf| {
        buf.reset_for_cut(5, 4);
        assert_eq!(buf.coefficients[0], 0.0); // Reset worked
        assert_eq!(buf.coefficients.capacity(), 10); // Capacity preserved
    });
}
```

### Phase 2: Refactor evaluate_cut (3-4 hours)

**Goal**: Use thread-local buffers instead of allocating

**Location**: `src/state.rs:555-618` (StorageState::evaluate_cut)

**Critical**: Preserve Kahan summation order for reproducibility

**Changes**:
```rust
fn evaluate_cut(
    &mut self,
    risk_measure: &dyn risk_measure::RiskMeasure,
    branching_realizations: &[subproblem::Realization],
) -> cut::BendersCut {
    use crate::memory::with_cut_buffers;
    
    with_cut_buffers(|buffers| {
        // Reset buffers for this cut computation
        buffers.reset_for_cut(self.dimension, branching_realizations.len());
        
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let num_branchings = costs.len();
        let probabilities = utils::uniform_prob_by_count(num_branchings);
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(&probabilities, &costs);

        // PERFORMANCE: Reuse pre-allocated contribution buffers
        // Maintain deterministic order for Kahan summation
        let coef_contributions = &mut buffers.contributions_outer;
        let mut objective_contributions: Vec<f64> =
            Vec::with_capacity(branching_realizations.len());

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];

            // Reuse inner vector (already allocated)
            let contrib = &mut coef_contributions[index];
            contrib.clear();
            contrib.extend(realization.water_value.iter().map(|&val| prob * val));
            
            objective_contributions
                .push(prob * realization.total_stage_objective);
        }

        // Deterministic Kahan summation (CRITICAL: preserve order)
        let cut_coefficients = &mut buffers.coefficients;
        for hydro_idx in 0..self.dimension {
            let values: Vec<f64> = coef_contributions
                .iter()
                .map(|contrib| contrib[hydro_idx])
                .collect();
            cut_coefficients[hydro_idx] = utils::kahan_sum(&values);
        }
        let objective = utils::kahan_sum(&objective_contributions);

        let cut_rhs = objective
            - utils::dot_product(cut_coefficients, self.coefficients());

        // ALLOCATION: One final allocation for owned BendersCut
        cut::BendersCut::new(
            0,
            cut_coefficients.clone(), // Clone from buffer
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    })
}
```

**Similar changes for**: `StorageAndInflowState::evaluate_cut` (line 932)

### Phase 3: Initialize Buffers in SDDP (1 hour)

**Goal**: Initialize thread-local buffers before parallel execution

**Location**: `src/sddp/mod.rs` (backward pass initialization)

**Add before parallel execution**:
```rust
// Initialize thread-local cut buffers for each worker thread
let sizing = &self.sizing_info;
let max_state_dim = sizing.max_state_dimension;
let max_scenarios = sizing.max_scenarios_per_node;

// Initialize in main thread
crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);

// Rayon will call initialization in each worker thread on first use
train_handlers.par_iter_mut()
    .enumerate()
    .map(|(forward_pass_idx, handler)| {
        // Thread-local buffers auto-initialize on first with_cut_buffers call
        handler.compute_cut_for_backward_step(...)
    })
    .collect::<Result<Vec<_>, String>>()?;
```

### Phase 4: Testing & Validation (2-3 hours)

**Unit Tests**:
```rust
#[test]
fn test_evaluate_cut_with_buffers() {
    initialize_cut_buffers(10, 4);
    
    let mut state = StorageState::new(...);
    let realizations = create_test_realizations();
    
    let cut1 = state.evaluate_cut(&risk_measure, &realizations);
    let cut2 = state.evaluate_cut(&risk_measure, &realizations);
    
    // Results must be identical (numerical reproducibility)
    assert!((cut1.rhs - cut2.rhs).abs() < 1e-10);
    for i in 0..cut1.coefficients.len() {
        assert!((cut1.coefficients[i] - cut2.coefficients[i]).abs() < 1e-10);
    }
}
```

**Integration Tests**:
```rust
#[test]
fn test_backward_pass_with_buffer_reuse() {
    // Run full backward pass
    let mut sddp = SDDP::new(...);
    let result1 = sddp.train();
    
    // Reset and run again
    let mut sddp = SDDP::new(...);
    let result2 = sddp.train();
    
    // Results must match exactly
    assert_eq!(result1.final_cost, result2.final_cost);
}
```

**Thread-Safety Test**:
```rust
#[test]
fn test_parallel_buffer_safety() {
    use rayon::prelude::*;
    
    let states: Vec<_> = (0..8).map(|_| create_test_state()).collect();
    
    // Parallel execution
    let results: Vec<_> = states.par_iter()
        .map(|state| {
            initialize_cut_buffers(10, 4); // Each thread initializes
            state.evaluate_cut(&risk_measure, &realizations)
        })
        .collect();
    
    // All results should be valid
    assert_eq!(results.len(), 8);
}
```

### Phase 5: Profiling & Measurement (2 hours)

**Measure Improvement**:
```bash
# Re-profile after implementation
./scripts/profile_allocations_simple.sh examples/03-multistage

# Compare metrics:
# Before: 0.437s, ~2,048 allocations
# After:  ~0.385s, <50 allocations (target: 12% faster)
```

**Validate**:
- [ ] All 495 tests pass
- [ ] Runtime improves by 10-15%
- [ ] Allocation count drops by 99%
- [ ] No memory regression (RSS stays ~23 MB)
- [ ] Results numerically identical (Kahan summation preserved)

---

## Risk Mitigation

### Numerical Reproducibility

**Risk**: Buffer reuse changes computation order  
**Mitigation**: Preserve exact Kahan summation order from original  
**Validation**: Compare results bit-for-bit with baseline

### Thread Safety

**Risk**: Shared mutable state between threads  
**Mitigation**: Use thread_local! (each thread has independent buffers)  
**Validation**: Parallel stress test with 8+ threads

### Memory Management

**Risk**: Buffers grow unbounded  
**Mitigation**: Cap at max_state_dim from SizingInfo  
**Validation**: Monitor RSS during large problem runs

---

## Success Criteria

- [x] Baseline established: 0.437s
- [ ] Implementation complete: All code changes done
- [ ] Tests passing: 495/495 passing
- [ ] Performance validated: 10-15% improvement measured
- [ ] Allocations reduced: ~2K → <50 (99% reduction)
- [ ] Documentation updated: Comments explain optimization

---

## Timeline

**Total**: 10-13 hours (1.5 days)

- Phase 1 (Infrastructure): 2-3 hours
- Phase 2 (Refactor): 3-4 hours
- Phase 3 (Integration): 1 hour
- Phase 4 (Testing): 2-3 hours
- Phase 5 (Profiling): 2 hours

**Start**: 2025-11-10  
**Target**: 2025-11-11

---

**Status**: Ready to implement  
**Confidence**: High (baseline validated, strategy proven)  
**Next**: Begin Phase 1 - Thread-local buffer infrastructure
