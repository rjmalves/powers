# Master Plan: Clean Code Refactoring for HPC Performance

## Progress Tracking

| Epic | Name | Status | Completed |
|------|------|--------|-----------|
| 1 | Foundation | ✅ Complete | 2025-12-29 |
| 2 | Core Extraction | ⬜ Not Started | - |
| 3 | Algorithm Separation | ⬜ Not Started | - |
| 4 | State Simplification | ⬜ Not Started | - |
| 5 | Memory Optimization | ⬜ Not Started | - |
| 6 | Test Modernization | ⬜ Not Started | - |
| 7 | Performance Validation | ⬜ Not Started | - |

---

## Executive Summary

This master plan addresses the structural debt in the POWE.RS codebase that has prevented successful memory optimization efforts. The application has grown organically with large monolithic functions, high argument counts, and tightly coupled modules—patterns that make it difficult to reason about memory flows, apply targeted optimizations, and maintain correctness. This refactoring will decompose the codebase into small, focused functions with clear responsibilities, enabling both better human understanding and compiler optimizations.

**Root Cause**: Previous optimization efforts (documented in `MEMORY_GROWTH_ANALYSIS.md`, `REMAINING_ALLOCATIONS_ANALYSIS.md`, `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md`) identified specific allocation hotspots but struggled to address them due to:
1. Functions too large to safely modify (~140+ line functions with 8+ parameters)
2. Implicit data flows buried in complex control structures
3. Lack of clear abstraction boundaries between computation phases
4. Test infrastructure that tests implementation details rather than behavior

**Goal**: Transform the codebase into clean, modular Rust code that enables:
- Successful elimination of dynamic allocations in hot paths
- Cache-friendly memory access patterns
- Compiler auto-vectorization opportunities
- Maintainable, testable code with clear contracts

---

## ⚠️ CRITICAL PRINCIPLES: Correctness First, Then Performance

> **THIS SECTION MUST BE READ AND FOLLOWED FOR EVERY TICKET IN THIS REFACTORING**

### Principle 1: Algorithm Correctness is Non-Negotiable

The SDDP algorithm is mathematically complex and its correctness is paramount. During this refactoring:

1. **The algorithm logic MUST remain unchanged** - We are refactoring structure, not behavior
2. **Numerical outputs MUST be bit-for-bit identical** (within floating-point determinism limits with same seed)
3. **Any deviation in results is a BUG** that must be investigated and fixed before proceeding
4. **Golden output tests are mandatory** - Run them after EVERY change, not just at PR time

### Principle 2: When In Doubt, ASK

If during implementation you encounter:
- **Unexpected results** (different outputs, test failures you don't understand)
- **Code that seems incorrect** in the original implementation
- **Ambiguity** about whether a change preserves behavior
- **Performance characteristics** that don't match expectations
- **Any situation** where you're unsure if proceeding is safe

**⛔ STOP and ask the user for clarification before proceeding.**

Do NOT:
- ❌ "Fix" what appears to be a bug in the original code without asking
- ❌ Assume a test failure is due to a "bad test" without verification
- ❌ Make algorithmic changes to "improve" the code
- ❌ Skip validation steps to save time
- ❌ Proceed with uncertainty - uncertainty compounds into larger problems

### Principle 3: Clean Code WITH High Performance

Our dual goals are **clean code** AND **high performance**. These are not in conflict:

1. **Clean code enables performance** - Small functions inline better, clear data flow enables cache optimization
2. **Performance validates correctness** - Regressions often indicate behavioral changes
3. **Never sacrifice correctness for either** - A fast wrong answer is worse than a slow correct one

### Verification Checklist (For Every Change)

Before considering any change complete:

- [ ] Golden output tests pass (bit-for-bit match with baseline)
- [ ] All existing tests pass (or explicitly documented why ignored)
- [ ] Benchmark shows no regression (>5% slowdown requires investigation)
- [ ] Code review confirms structural change only (no algorithm modifications)

**If any checkbox fails: STOP, investigate, and ask for clarification if needed.**

---

## Goals & Non-Goals

### Goals

1. **Function Decomposition**: All functions ≤50 lines, ≤4 parameters (following clean code principles)
2. **Module Cohesion**: Each module has a single, well-defined responsibility
3. **Explicit Data Flows**: All data dependencies visible at call sites
4. **Zero-Allocation Hot Paths**: Training/simulation loops perform no heap allocations
5. **Test Modernization**: Replace broken/brittle tests with behavior-focused test suites
6. **Performance Parity**: No performance regressions; target 10-15% improvement from better compiler optimization

### Non-Goals (Explicit Scope Exclusions)

1. **Algorithm Changes**: SDDP algorithm logic remains unchanged ⚠️ *See Critical Principles above*
2. **API Breaking Changes**: Public API remains backward compatible
3. **Dependency Updates**: No major dependency version changes
4. **Feature Additions**: No new features during refactoring
5. **Full SoA Conversion**: Hybrid approach only, per `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md`

---

## Architecture Overview

### Current State

The codebase consists of **32,743 lines** across 24 source files with these structural issues:

| File | Lines | Issues |
|------|-------|--------|
| `subproblem.rs` | 6,631 | Monolithic, 60+ functions, mixed responsibilities |
| `sddp/mod.rs` | 3,913 | God module with training, simulation, handlers |
| `state.rs` | 3,087 | Two implementations with duplicated logic |
| `input.rs` | 1,763 | Parsing mixed with validation |
| `sddp/builder.rs` | 1,494 | Complex construction logic |
| `solver.rs` | 1,235 | HiGHS interface with retry logic |
| `system.rs` | 1,147 | Data structures and operations mixed |

**Key Anti-Patterns Identified**:

1. **Large Functions**: `realize_and_solve()` (~140 lines), `forward()` (~80 lines)
2. **High Parameter Counts**: Functions with 6-10 parameters
3. **Trait Object Allocation**: `Box<dyn State>` cloned in hot paths
4. **Implicit State**: Thread-local buffers, mutable borrowing chains
5. **Test Coupling**: Tests rely on internal structure, not behavior

### Target State

```
src/
├── algorithm/              # SDDP algorithm phases
│   ├── forward_pass.rs     # Forward pass logic
│   ├── backward_pass.rs    # Backward pass logic
│   ├── cut_computation.rs  # Benders cut generation
│   └── convergence.rs      # Convergence checking
│
├── model/                  # LP model operations
│   ├── builder.rs          # Model construction
│   ├── constraints/        # Constraint types
│   │   ├── hydro_balance.rs
│   │   ├── bus_balance.rs
│   │   └── ar_dynamics.rs
│   ├── solver_interface.rs # HiGHS interaction
│   └── solution_extract.rs # Solution extraction
│
├── state/                  # State management
│   ├── storage.rs          # StorageState (simplified)
│   ├── storage_inflow.rs   # StorageAndInflowState (simplified)
│   ├── extraction.rs       # Trajectory → state extraction
│   └── cut_evaluation.rs   # State-based cut evaluation
│
├── memory/                 # Memory management
│   ├── pools/              # Object pools
│   │   ├── cut_pool.rs
│   │   ├── state_pool.rs
│   │   └── realization_pool.rs
│   ├── buffers.rs          # Thread-local buffers
│   └── preallocator.rs     # Upfront allocation
│
├── scenario/               # Scenario generation
│   ├── tree.rs             # Scenario tree structure
│   ├── sampling.rs         # SAA sampling
│   └── noise_models.rs     # Distribution models
│
├── io/                     # Input/Output
│   ├── input/              # Input parsing
│   ├── output/             # Output generation
│   └── validation/         # Schema validation
│
└── core/                   # Shared primitives
    ├── types.rs            # Common type aliases
    ├── error.rs            # Error types
    └── timing.rs           # Performance timing
```

### Key Design Decisions

1. **Extract Algorithm Phases**: Split `sddp/mod.rs` into separate forward/backward modules
   - *Rationale*: Each phase has distinct data access patterns; separation enables targeted optimization

2. **Separate Model from Solver**: Model construction vs solver interaction
   - *Rationale*: Clear boundary for buffer reuse and error handling

3. **Pool-Based Memory**: Preallocated pools with slot-based access
   - *Rationale*: Eliminates hot-path allocations identified in `REMAINING_ALLOCATIONS_ANALYSIS.md`

4. **Behavior-Focused Tests**: Test observable outcomes, not internal structure
   - *Rationale*: Enables refactoring without breaking tests

---

## Technical Approach

### Core Abstractions

#### 1. `ForwardPassContext` - Forward Pass Execution Context

```rust
/// Encapsulates all mutable state needed for a single forward pass.
/// Replaces 8+ parameters passed through the call chain.
pub struct ForwardPassContext<'a> {
    pub subproblems: &'a mut [Subproblem],
    pub realizations: &'a mut [Realization],
    pub trajectory: TrajectoryBuilder<'a>,
    pub timing: &'a mut ForwardPassTiming,
}
```

**Benefits**: Single parameter replaces multiple mutable borrows; clear ownership.

#### 2. `BackwardPassContext` - Backward Pass Execution Context

```rust
pub struct BackwardPassContext<'a> {
    pub subproblems: &'a mut [Subproblem],
    pub branchings: &'a mut [Vec<Realization>],
    pub cut_pool: &'a mut CutPool,
    pub timing: &'a mut BackwardPassTiming,
}
```

#### 3. `SolutionExtractor` - Allocation-Free Solution Extraction

```rust
/// Extracts LP solution into preallocated Realization buffers.
/// Zero heap allocations after initialization.
pub struct SolutionExtractor {
    variable_indices: VariableIndices,
    constraint_indices: ConstraintIndices,
}

impl SolutionExtractor {
    /// Extract solution into preallocated container.
    pub fn extract_into(
        &self,
        solution: &Solution,
        target: &mut Realization,
    ) { ... }
}
```

#### 4. `CutEvaluator` - Stateless Cut Computation

```rust
/// Computes Benders cuts without state allocation.
/// All buffers passed in; no internal allocation.
pub struct CutEvaluator;

impl CutEvaluator {
    pub fn evaluate(
        state_coefficients: &[f64],
        branching_results: &[BranchingResult],
        risk_measure: &dyn RiskMeasure,
        buffers: &mut CutBuffers,
    ) -> CutResult { ... }
}
```

### Data Flow

**Current** (implicit, scattered):
```
forward() → step() → realize_and_solve() → get_X_from_solution() × 10
                  ↓
            thread-local SOLUTION_BUFFER (implicit)
```

**Target** (explicit, traceable):
```
forward_pass::execute(context)
    ├── for each stage:
    │   ├── trajectory.prepare_stage(stage_id)
    │   ├── solver_interface::solve(&mut context.subproblems[stage_id])
    │   └── solution_extract::into_realization(solution, &mut context.realizations[stage_id])
    └── return ForwardPassResult
```

### Parallelism Strategy

The existing Rayon-based parallelism is preserved, but with clearer ownership:

```rust
// Current: Complex mutable sharing via Arc<Mutex<...>>
let fcf_node = Arc::new(Mutex::new(fcf_data));

// Target: Separate parallel and sequential phases
fn parallel_forward(contexts: &mut [ForwardPassContext]) {
    contexts.par_iter_mut().for_each(|ctx| {
        forward_pass::execute(ctx);  // No Arc/Mutex needed
    });
}

fn sequential_cut_update(pool: &mut CutPool, cuts: &[CutData]) {
    // Single-threaded cut pool update
}
```

### Performance Strategy

| Optimization | Mechanism | Expected Impact |
|--------------|-----------|-----------------|
| Function inlining | Small functions + `#[inline]` | +5-8% (LLVM optimization) |
| Buffer reuse | Context structs with preallocated buffers | -30MB allocation churn |
| Cache locality | Sequential stage iteration | +3-5% (L1/L2 hits) |
| Branch prediction | Simplified control flow | +1-2% |

---

## Phases & Milestones

| Phase | Name | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | **Foundation** | 2 weeks | Module structure, error types, test infrastructure |
| 2 | **Core Extraction** | 3 weeks | Extract `subproblem.rs` functions, solution extraction |
| 3 | **Algorithm Separation** | 3 weeks | Split forward/backward passes, context structs |
| 4 | **State Simplification** | 2 weeks | Refactor `state.rs`, eliminate trait object allocations |
| 5 | **Memory Optimization** | 2 weeks | Pool-based allocation, verify zero-allocation hot paths |
| 6 | **Test Modernization** | 2 weeks | Replace brittle tests, add property-based tests |
| 7 | **Performance Validation** | 1 week | Benchmark comparison, regression testing |

**Total Duration**: ~15 weeks

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence from refactoring | Medium | **CRITICAL** | Golden-output regression tests before any changes; **STOP and ask if any divergence detected** |
| Performance regression during transition | Medium | Medium | Continuous benchmarking at each phase |
| Breaking existing tests | High | Low | Tests are already brittle; modernize as we go |
| Scope creep into algorithm changes | Low | High | Strict code review; separate algorithm PRs |
| Parallelism bugs from ownership changes | Medium | High | Careful Arc/Mutex audit; Miri testing |

---

## Success Metrics

### Quantitative

- [ ] **Function Size**: 95% of functions ≤50 lines (currently ~60%)
- [ ] **Parameter Count**: 100% of public functions ≤4 parameters (currently ~70%)
- [ ] **Hot Path Allocations**: 0 allocations in forward/backward loops (currently ~30MB)
- [ ] **Test Coverage**: Maintain ≥85% line coverage
- [ ] **Performance**: ≥10% speedup on `05-large-scale-brazilian` benchmark

### Qualitative

- [ ] New contributors can understand a module in <30 minutes
- [ ] Adding a new constraint type requires changes to ≤3 files
- [ ] Memory profile is deterministic (flat RSS during training)

---

## Dependencies

### External Dependencies (No Changes)

- `highs-sys`: HiGHS solver bindings
- `rayon`: Parallel iteration
- `rand`, `rand_distr`: Random number generation
- `serde`, `serde_json`, `csv`: Serialization

### Internal Dependencies (Restructured)

```
algorithm ──────→ model ──────→ solver
    │                │
    ↓                ↓
  state ←──────── memory
    │
    ↓
scenario
```

---

## Validation Plan

### Phase Gates

Each phase must pass before proceeding:

1. **All existing tests pass** (may be temporarily `#[ignore]` during transition)
2. **No benchmark regressions** (>5% slowdown blocks merge)
3. **Code review approval** with focus on clean code principles
4. **Documentation updated** for changed modules
5. **Golden output verification** - Numerical results identical to baseline ⚠️ *If this fails, STOP and ask*

### Golden Output Tests

Before refactoring begins, capture deterministic outputs:

```bash
# Generate golden outputs
for example in examples/0*; do
  ./target/release/powers run $example --seed 42 > golden/${example##*/}.txt
done

# Verify after changes
for example in examples/0*; do
  diff -u golden/${example##*/}.txt <(./target/release/powers run $example --seed 42)
done
```

### Continuous Benchmarking

```bash
# Run before/after each PR
cargo bench --bench sddp_e2e -- --save-baseline before
# ... make changes ...
cargo bench --bench sddp_e2e -- --baseline before
```

---

## Open Questions for Clarification

Before breaking this into epics and sprints, please clarify:

### 1. Test Infrastructure Priority

The README claims 89.42% coverage with "189 library tests + 473+ total tests", but you mention many tests are "old and broken". 

**Questions**:
- Should we prioritize fixing broken tests before refactoring, or refactor first and update tests as we go?
- Are there specific test files that are known to be unreliable?
- Do you have a list of tests that are currently `#[ignore]`d or failing?

### 2. Module Boundary Decisions

The proposed structure separates concerns significantly. 

**Questions**:
- Is the proposed `algorithm/`, `model/`, `state/`, `memory/` structure acceptable?
- Should we maintain backward compatibility for library users importing `powers_rs::sddp::*`?
- Are there any modules you want to keep unchanged (e.g., `graph.rs` is relatively small)?

### 3. Parallelism Strategy

The current code uses `Arc<Mutex<...>>` extensively for parallel forward passes.

**Questions**:
- Should we maintain the current parallelism model or explore lock-free alternatives?
- What is the target hardware (number of cores, memory bandwidth)?
- Is there a maximum memory footprint we should target?

### 4. Timeline and Resources

**Questions**:
- Is the 15-week estimate acceptable, or do we need to compress the timeline?
- Will there be multiple developers working on this, or is it primarily single-developer work?
- Should we prioritize certain phases (e.g., memory optimization before test modernization)?

### 5. Performance Targets

**Questions**:
- What is the target speedup? (Currently estimating 10-15%)
- Is memory reduction more important than speed?
- Are there specific benchmarks or problem sizes that are priorities?

---

## Next Steps

Once the above questions are clarified, I will:

1. **Create Epic definitions** with clear scope and acceptance criteria
2. **Break epics into 2-week sprints** with parallel work opportunities
3. **Generate atomic tickets** (1-3 days each) with implementation guides

Please review this master plan and provide feedback on the clarification questions.

---

## Timing Architecture: Zero-Pollution Instrumentation

### Problem Statement

The current timing infrastructure has significant design issues that pollute the codebase:

**Current State**:
- **48 `Instant::now()` / `.elapsed()` calls** scattered across `sddp/mod.rs` and `subproblem.rs`
- **4+ timing structs** with overlapping, inconsistent fields:
  - `ForwardPassTiming` (6 fields)
  - `ForwardPassTimingAccumulator` (4 fields)
  - `BackwardPassTiming` (9 fields)
  - `BackwardPassTimingAccumulator` (10 fields)
  - `RealizeUncertaintiesTiming` (2 fields)
  - `BranchingsTiming` (2 fields)
- **Timing logic interleaved with business logic**: Every function manually manages `let start = Instant::now()` and `timing.X += start.elapsed()`
- **Aggregation complexity**: Multiple aggregation strategies (sum, average) implemented ad-hoc
- **Timing value overwriting**: The current code recalibrates (overwrites) internal forward timing estimates to "account for parallel overhead" (lines 1802-1823 in `sddp/mod.rs`), **losing the precise measured values**:
  ```rust
  // Current problematic pattern - OVERWRITES actual timing with proportional redistribution
  forward_timing.model_preprocessing_time = forward_parallel_time
      .mul_f64(forward_timing.model_preprocessing_time.as_secs_f64()
               / internal_forward_timings.as_secs_f64());
  ```
  This means we cannot distinguish between actual CPU time spent in each phase versus parallel scheduling overhead.

**Impact**:
- Business logic obscured by instrumentation boilerplate
- Adding a new timing metric requires changes to 10+ locations
- Timing overhead (~48 syscalls per iteration) affects performance measurements
- Inconsistent timing granularity across the codebase

### Design Goals

1. **Separation of Concerns**: Timing infrastructure completely separate from algorithm logic
2. **Zero-Cost When Disabled**: Compile-time elimination when not needed
3. **Hierarchical Structure**: Timing naturally follows the call tree
4. **Single Source of Truth**: One timing type, one aggregation strategy
5. **Extensibility**: Adding new metrics requires minimal code changes
6. **Preserve Precise Timing Values**: Never overwrite measured timing values; keep the actual CPU time spent in each phase
7. **Explicit Parallel Overhead Tracking**: Track parallel scheduling/synchronization overhead as a separate, dedicated metric

### Target Architecture

#### 1. Scoped Timing Guard Pattern

Replace scattered `Instant::now()` / `.elapsed()` with RAII guards:

```rust
// src/timing/mod.rs

/// Zero-cost timing guard that records duration on drop.
/// When `timing` feature is disabled, this compiles to nothing.
pub struct TimingGuard<'a> {
    #[cfg(feature = "timing")]
    start: Instant,
    #[cfg(feature = "timing")]
    target: &'a Cell<Duration>,
}

impl<'a> TimingGuard<'a> {
    #[inline(always)]
    pub fn new(target: &'a Cell<Duration>) -> Self {
        Self {
            #[cfg(feature = "timing")]
            start: Instant::now(),
            #[cfg(feature = "timing")]
            target,
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

/// Macro for clean timing scope creation
macro_rules! time_scope {
    ($timing:expr, $field:ident) => {
        let _guard = $crate::timing::TimingGuard::new(&$timing.$field);
    };
}
```

**Usage in business logic**:
```rust
// Before (polluted):
fn forward(&mut self, ...) -> Result<...> {
    let prep_start = std::time::Instant::now();
    // ... 20 lines of prep logic ...
    timing.model_preprocessing_time += prep_start.elapsed();
    
    let solve_start = std::time::Instant::now();
    // ... solve logic ...
    timing.solver_time += solve_start.elapsed();
}

// After (clean):
fn forward(&mut self, timing: &Timing) -> Result<...> {
    {
        time_scope!(timing, model_preprocessing);
        // ... 20 lines of prep logic ...
    }
    {
        time_scope!(timing, solver);
        // ... solve logic ...
    }
}
```

#### 2. Hierarchical Timing Tree

Replace flat structs with a tree that mirrors the call hierarchy:

```rust
/// Unified timing tree for SDDP iteration.
/// Mirrors the algorithm's hierarchical structure.
#[derive(Debug, Clone, Default)]
pub struct IterationTiming {
    pub forward: ForwardTiming,
    pub backward: BackwardTiming,
    pub total: Cell<Duration>,
}

#[derive(Debug, Clone, Default)]
pub struct ForwardTiming {
    pub saa_sampling: Cell<Duration>,
    
    // --- Wall-clock time for the parallel section ---
    /// Total wall-clock time for the parallel forward execution (measured externally)
    pub parallel_wall_time: Cell<Duration>,
    
    // --- Precise aggregated CPU times (NEVER overwritten) ---
    /// Sum/average of actual model_preprocessing time across all trajectories
    pub model_preprocessing: Cell<Duration>,
    /// Sum/average of actual solver time across all trajectories  
    pub solver: Cell<Duration>,
    /// Sum/average of actual model_postprocessing time across all trajectories
    pub model_postprocessing: Cell<Duration>,
    
    // --- Derived parallel overhead (computed, not measured) ---
    /// Parallel overhead = parallel_wall_time - (model_preprocessing + solver + model_postprocessing) / num_trajectories
    /// This captures thread scheduling, synchronization, and load imbalance costs
    pub parallel_overhead: Cell<Duration>,
    
    pub postprocessing: Cell<Duration>,
    /// Per-trajectory breakdown (optional, for detailed analysis)
    pub trajectories: Vec<TrajectoryTiming>,
}

#[derive(Debug, Clone, Default)]
pub struct TrajectoryTiming {
    pub stages: Vec<StageTiming>,
}

#[derive(Debug, Clone, Default)]
pub struct StageTiming {
    pub model_prep: Cell<Duration>,
    pub solver: Cell<Duration>,
    pub extraction: Cell<Duration>,
}

#[derive(Debug, Clone, Default)]
pub struct BackwardTiming {
    pub preprocessing: Cell<Duration>,
    pub branching_solves: Cell<Duration>,
    pub cut_computation: Cell<Duration>,
    pub cut_selection: Cell<Duration>,
    pub fcf_update: Cell<Duration>,
    pub handler_application: Cell<Duration>,
}
```

#### 3. Parallel Overhead Tracking Strategy

**Critical Requirement**: The new timing infrastructure **MUST NOT** overwrite measured timing values. Instead, parallel overhead is tracked explicitly as a derived metric.

**Current Problematic Behavior** (to be eliminated):
```rust
// ❌ BAD: Overwrites actual measurements with proportional redistribution
forward_timing.model_preprocessing_time = forward_parallel_time
    .mul_f64(forward_timing.model_preprocessing_time.as_secs_f64()
             / internal_forward_timings.as_secs_f64());
```

**New Correct Behavior**:
```rust
// ✅ GOOD: Preserve precise values, compute overhead separately

// 1. Aggregate precise timing from all trajectories (NEVER overwritten)
let aggregated = ForwardTimingAggregated {
    model_preprocessing: timings.iter().map(|t| t.model_preprocessing).sum::<Duration>() / n,
    solver: timings.iter().map(|t| t.solver).sum::<Duration>() / n,
    model_postprocessing: timings.iter().map(|t| t.model_postprocessing).sum::<Duration>() / n,
};

// 2. Compute parallel overhead as the difference
let total_cpu_time_per_trajectory = aggregated.model_preprocessing 
    + aggregated.solver 
    + aggregated.model_postprocessing;
let parallel_overhead = parallel_wall_time.saturating_sub(total_cpu_time_per_trajectory);

// 3. Store BOTH the precise values AND the computed overhead
forward_timing.model_preprocessing = aggregated.model_preprocessing;  // Precise value preserved
forward_timing.solver = aggregated.solver;                            // Precise value preserved
forward_timing.model_postprocessing = aggregated.model_postprocessing; // Precise value preserved
forward_timing.parallel_wall_time = parallel_wall_time;               // Measured wall time
forward_timing.parallel_overhead = parallel_overhead;                  // Computed overhead
```

**Benefits**:
1. **Accurate profiling**: Know exactly how much CPU time is spent in solver vs model prep
2. **Parallel efficiency analysis**: `parallel_overhead / parallel_wall_time` gives scheduling efficiency
3. **No information loss**: All original measurements preserved for analysis
4. **Load balancing insights**: Compare `max(trajectory_times)` vs `avg(trajectory_times)` to identify imbalance

**Backward Pass**: Similar approach for backward pass parallel sections (branching solves).

#### 4. Feature-Gated Compilation

Make timing completely eliminable at compile time:

```toml
# Cargo.toml
[features]
default = []
timing = []           # Basic timing (iteration-level)
timing-detailed = ["timing"]  # Per-stage breakdown
```

```rust
// Conditional timing collection
#[cfg(feature = "timing")]
pub fn create_timing() -> IterationTiming {
    IterationTiming::default()
}

#[cfg(not(feature = "timing"))]
pub fn create_timing() -> () {
    ()
}
```

#### 5. Timing Collector Trait

Abstract timing collection for testing and flexibility:

```rust
/// Trait for timing collection strategies.
/// Enables testing without actual timing, custom aggregation, etc.
pub trait TimingCollector: Send + Sync {
    fn record(&self, metric: TimingMetric, duration: Duration);
    fn snapshot(&self) -> TimingSnapshot;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimingMetric {
    // Forward pass
    SaaSampling,
    ForwardModelPrep,
    ForwardSolver,
    ForwardExtraction,
    ForwardPostprocessing,
    
    // Backward pass
    BackwardPreprocessing,
    BranchingSolver,
    CutComputation,
    CutSelection,
    FcfUpdate,
    HandlerApplication,
}

/// Default implementation using atomic counters.
pub struct AtomicTimingCollector {
    metrics: [AtomicU64; TimingMetric::COUNT],
}

/// No-op implementation for production when timing disabled.
pub struct NullTimingCollector;

impl TimingCollector for NullTimingCollector {
    #[inline(always)]
    fn record(&self, _: TimingMetric, _: Duration) {}
    fn snapshot(&self) -> TimingSnapshot { TimingSnapshot::default() }
}
```

### Integration with Context Structs

The timing collector is passed through context structs, not individual parameters:

```rust
pub struct ForwardPassContext<'a> {
    pub subproblems: &'a mut [Subproblem],
    pub realizations: &'a mut [Realization],
    pub trajectory: TrajectoryBuilder<'a>,
    pub timing: &'a dyn TimingCollector,  // Single timing interface
}

// In forward pass implementation:
pub fn execute_forward_pass(ctx: &mut ForwardPassContext) -> Result<f64, Error> {
    for stage_id in ctx.stage_ids() {
        let _guard = ctx.timing.scope(TimingMetric::ForwardModelPrep);
        ctx.prepare_stage(stage_id)?;
        drop(_guard);
        
        let _guard = ctx.timing.scope(TimingMetric::ForwardSolver);
        ctx.solve_stage(stage_id)?;
        // guard dropped automatically
    }
    Ok(ctx.compute_cost())
}
```

### Aggregation Strategy

Centralize aggregation logic in the timing module:

```rust
impl IterationTiming {
    /// Aggregate parallel trajectory timings using specified strategy.
    pub fn aggregate_forward(&mut self, strategy: AggregationStrategy) {
        match strategy {
            AggregationStrategy::Average => {
                let n = self.forward.trajectories.len();
                for field in [&self.forward.model_prep, ...] {
                    field.set(field.get() / n as u32);
                }
            }
            AggregationStrategy::Sum => { /* ... */ }
            AggregationStrategy::Max => { /* ... */ }
        }
    }
}

pub enum AggregationStrategy {
    Average,  // For representative per-trajectory metrics
    Sum,      // For total work done
    Max,      // For load balancing analysis
}
```

### Correctness Note for Timing Changes

⚠️ **Timing changes must NOT affect algorithm behavior.** Timing is observational only. If any timing change causes:
- Different numerical outputs
- Test failures unrelated to timing assertions
- Performance characteristics that suggest behavioral change

**STOP and ask for clarification.** The timing infrastructure is measurement-only and must have zero semantic impact.

### Migration Path

1. **Phase 1**: Create new timing module with `TimingGuard` and `TimingCollector`
2. **Phase 2**: Add timing to context structs
3. **Phase 3**: Replace `Instant::now()` calls one module at a time
4. **Phase 4**: Remove old timing structs
5. **Phase 5**: Add feature gates for compile-time elimination

### Metrics Preserved

All current timing metrics will be preserved with cleaner organization. **Critically, the new infrastructure adds `parallel_overhead` as an explicit metric instead of redistributing timing values**:

| Current Field | New Location | Collection Point |
|--------------|--------------|------------------|
| `saa_sampling_time` | `forward.saa_sampling` | Training loop |
| `model_preprocessing_time` | `stage.model_prep` | `prepare_stage()` |
| `solver_time` | `stage.solver` | `solve_stage()` |
| `model_postprocessing_time` | `stage.extraction` | `extract_solution()` |
| `forward_postprocessing_time` | `forward.postprocessing` | Training loop |
| `backward_preprocessing_time` | `backward.preprocessing` | `backward_pass()` |
| `cut_selection_time` | `backward.cut_selection` | `select_cuts()` |
| `fcf_state_update_time` | `backward.fcf_update` | `update_fcf()` |
| `cut_cloning_time` | *eliminated* | No longer needed |
| `handler_application_time` | `backward.handler_application` | `apply_cuts()` |
| *N/A (implicit)* | `forward.parallel_overhead` | Computed: `parallel_wall_time - avg(per_trajectory_cpu)` |
| *N/A (implicit)* | `forward.parallel_wall_time` | Training loop (around `par_iter`) |

### Performance Impact

| Metric | Current | Target |
|--------|---------|--------|
| Timing syscalls per iteration | ~48 | ~12 (via batching) |
| Lines of timing code in business logic | ~200 | ~20 |
| Timing struct definitions | 6 | 2 |
| Feature-gated elimination | No | Yes |

