# Epic 3: Algorithm Separation

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 3 weeks (2 sprints)
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

This epic touches the **core SDDP algorithm**. Read the [Critical Principles](../00-master-plan.md#️-critical-principles-correctness-first-then-performance) section carefully.

**The algorithm logic must remain EXACTLY the same.** We are separating structure, not modifying behavior. Golden tests must pass after every change. If ANY unexpected behavior occurs, **STOP and ask for clarification**.

---

## Summary

This epic separates the forward and backward passes from the monolithic `src/sddp/mod.rs` (3,913 lines) into dedicated modules in `src/algorithm/`. We also introduce **Context structs** that bundle related parameters, reducing function parameter counts from 8+ to 1-2.

**Key principle**: The algorithm logic is frozen. We're only changing how the code is organized and how data flows through function signatures.

---

## Scope

### Included

1. **Forward Pass Extraction**
   - Extract `forward()` and related functions to `src/algorithm/forward_pass.rs`
   - Create `ForwardPassContext` struct
   - Integrate new timing infrastructure

2. **Backward Pass Extraction**
   - Extract backward pass logic to `src/algorithm/backward_pass.rs`
   - Create `BackwardPassContext` struct
   - Extract cut computation to `src/algorithm/cut_computation.rs`

3. **Context Struct Introduction**
   - Replace 6-10 parameter functions with context-based APIs
   - Bundle mutable state into coherent structs
   - **Design for future preallocation**: Context structs should be aware that data sizes are known upfront from input data

4. **Timing Integration**
   - Replace scattered `Instant::now()` with `TimingGuard` from Epic 1
   - Use new `IterationTiming` struct
   - **Preserve precise values, track parallel overhead explicitly**

### Excluded

- Algorithm modifications (convergence criteria, cut selection, etc.)
- State representation changes (Epic 4)
- Memory pool implementation (Epic 5)

---

## Dependencies

- **Requires**:
  - Epic 1 complete (timing infrastructure)
  - Epic 2 complete (clean subproblem interface)
- **Enables**:
  - Epic 4: State Simplification (cleaner state interface)
  - Epic 5: Memory Optimization (clear allocation points)

---

## Acceptance Criteria

- [ ] `src/algorithm/forward_pass.rs` contains forward pass logic
- [ ] `src/algorithm/backward_pass.rs` contains backward pass logic
- [ ] `src/algorithm/cut_computation.rs` contains cut generation logic
- [ ] `ForwardPassContext` and `BackwardPassContext` structs exist
- [ ] All public functions have ≤4 parameters
- [ ] New timing infrastructure integrated (no more raw `Instant::now()`)
- [ ] **Context structs document preallocation opportunities**
- [ ] Golden tests pass (bit-for-bit identical output)
- [ ] Benchmarks show no regression (within 5%)

### Correctness Verification

After EVERY change:
- [ ] `cargo test` passes
- [ ] `./scripts/golden-tests.sh verify` passes

⚠️ **If any golden test fails, STOP and investigate. Do not proceed until resolved.**

---

## Technical Approach

### Context Structs with Preallocation Awareness

The context structs must be designed with awareness that:
1. **All data sizes are known at initialization** from reading input data
2. **Future migration will copy into preallocated slices** instead of allocating new Vecs
3. **The training step will preallocate buffers** at the beginning based on known sizes

**Before** (in `sddp/mod.rs`):
```rust
fn forward(
    &mut self,
    subproblems: &mut [Subproblem],
    realizations: &mut [Realization],
    noises: &[Noise],
    stage_count: usize,
    fcf: &Fcf,
    timing: &mut ForwardPassTimingAccumulator,
    rng: &mut StdRng,
    config: &Config,
) -> Result<f64, String> { ... }
```

**After** (in `algorithm/forward_pass.rs`):
```rust
/// Context for forward pass execution.
/// 
/// # Preallocation Design
/// 
/// This context is designed to support future preallocation:
/// - `realizations` slice is preallocated for all stages
/// - Data sizes are known from `config` and system dimensions
/// - Future: `solution_buffer` and `basis_buffer` per stage
/// 
/// When migrating to preallocated buffers:
/// 1. Add `solution_buffers: &'a mut [SolutionBuffer]` field
/// 2. Add `basis_buffers: &'a mut [BasisBuffer]` field  
/// 3. Solver will call `get_solution_into(&mut buffer)` instead of allocating
pub struct ForwardPassContext<'a> {
    pub subproblems: &'a mut [Subproblem],
    pub realizations: &'a mut [Realization],
    pub noises: &'a [Noise],
    pub fcf: &'a Fcf,
    pub config: &'a Config,
    pub timing: &'a IterationTiming,
    
    // Document known dimensions for future preallocation
    // These are available from input data:
    // - stage_count: config.stages
    // - num_hydros: system.hydros.len()
    // - num_thermals: system.thermals.len()
    // - num_buses: system.buses.len()
    // - forward_passes: config.forward_passes
}

pub fn execute(ctx: &mut ForwardPassContext, rng: &mut StdRng) -> Result<f64, Error> {
    // Same logic, cleaner interface
}
```

### Backward Pass Context

```rust
/// Context for backward pass execution.
/// 
/// # Preallocation Design
/// 
/// Similar to ForwardPassContext, designed for future pool-based allocation:
/// - `branchings` are preallocated for max_scenarios per stage
/// - Cut computation buffers are reusable
/// - State coefficients can be copied into preallocated slots
pub struct BackwardPassContext<'a> {
    pub subproblems: &'a mut [Subproblem],
    pub branchings: &'a mut [Vec<Realization>],
    pub cut_pool: &'a mut CutPool,
    pub timing: &'a IterationTiming,
    pub config: &'a Config,
    
    // Future: Add preallocated buffers
    // pub cut_buffers: &'a mut CutBufferPool,
    // pub state_buffers: &'a mut StateBufferPool,
}
```

### Timing Integration

**Before**:
```rust
let prep_start = Instant::now();
// ... work ...
timing.model_preprocessing_time += prep_start.elapsed();
```

**After**:
```rust
{
    let _guard = TimingGuard::new(&ctx.timing.forward.model_preprocessing);
    // ... work ...
} // Timing recorded automatically on drop
```

### Parallel Overhead Tracking

Per master plan requirements, we **preserve precise timing values** and **compute parallel overhead explicitly**:

```rust
// After parallel forward execution
let parallel_wall_time = parallel_begin.elapsed();
ctx.timing.forward.parallel_wall_time.set(parallel_wall_time);

// Aggregate precise CPU times (NEVER overwritten)
// These are the actual measurements from each trajectory
ctx.timing.forward.model_preprocessing.set(aggregated_model_prep);
ctx.timing.forward.solver.set(aggregated_solver);
ctx.timing.forward.model_postprocessing.set(aggregated_post);

// Compute overhead as the difference
ctx.timing.compute_forward_parallel_overhead();
```

---

## Module Structure After Epic

```
src/algorithm/
├── mod.rs
├── forward_pass.rs      # ForwardPassContext, execute()
├── backward_pass.rs     # BackwardPassContext, execute()
├── cut_computation.rs   # Cut calculation logic
└── context.rs           # Shared context types, preallocation documentation
```

---

## Sprints

### [Sprint 1: Forward Pass Extraction](./sprint-01/00-sprint-overview.md) (Week 1-2)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-020 | Design context structs with preallocation awareness | 3 | ⬜ |
| T-021 | Create ForwardPassContext | 3 | ⬜ |
| T-022 | Extract forward pass step logic | 5 | ⬜ |
| T-023 | Integrate forward timing infrastructure | 3 | ⬜ |
| T-024 | Update sddp/mod.rs to use forward_pass module | 3 | ⬜ |

**Sprint 1 Points**: 17

### [Sprint 2: Backward Pass Extraction](./sprint-02/00-sprint-overview.md) (Week 2-3)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-025 | Create BackwardPassContext | 3 | ⬜ |
| T-026 | Extract backward pass logic | 5 | ⬜ |
| T-027 | Extract cut computation | 3 | ⬜ |
| T-028 | Integrate backward timing infrastructure | 3 | ⬜ |
| T-029 | Update sddp/mod.rs to use backward_pass module | 3 | ⬜ |

**Sprint 2 Points**: 17

---

## Estimated Effort

- **Duration**: 2 sprints (3 weeks)
- **Story Points**: 34
- **Risk Level**: High (core algorithm code, but logic unchanged)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Subtle behavior change | Medium | **CRITICAL** | Golden tests after every change, careful code review |
| Parallel execution changes | Medium | High | Verify thread behavior unchanged |
| Timing integration errors | Low | Medium | Compare timing outputs before/after |
| Context struct overhead | Low | Low | Benchmark, use references |
| Preallocation design inadequate | Low | Medium | Document assumptions, review with future Epic 5 needs |

---

## Definition of Done

- [ ] All Sprint 1 and Sprint 2 tickets complete
- [ ] Forward pass in `src/algorithm/forward_pass.rs`
- [ ] Backward pass in `src/algorithm/backward_pass.rs`
- [ ] Context structs reduce parameter counts to ≤4
- [ ] **Context structs document preallocation opportunities**
- [ ] New timing infrastructure fully integrated
- [ ] **No raw `Instant::now()` in algorithm code**
- [ ] Golden tests pass
- [ ] Benchmarks within 5% of baseline
- [ ] All tests pass
- [ ] Code reviewed and merged
