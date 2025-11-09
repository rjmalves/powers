# Performance-Oriented Refactoring Plan

**Status**: 📋 Planning Phase  
**Goal**: Transform codebase for optimal performance while maintaining collaborative readability  
**Focus**: **Performance First, Clean Code Second**  
**Timeline**: 8 weeks (phased approach)  
**Last Updated**: 2025-11-09

---

## 🎯 Core Philosophy

> **"Premature optimization is the root of all evil, but late optimization is the root of poor performance."**  
> — Adapted from Donald Knuth

This plan prioritizes **performance in hot paths** while maintaining code quality in cold paths. We measure everything and optimize based on data.

### Guiding Principles

1. **Profile Before Refactoring** - Never guess, always measure
2. **Hot Path Optimization** - Focus 80% effort on 20% of critical code
3. **Zero-Cost Abstractions** - Use abstractions that compile away
4. **Memory Efficiency** - Pre-allocate, reuse buffers, minimize clones
5. **Cache Friendliness** - Contiguous data layouts, avoid pointer chasing
6. **Readable Hot Paths** - Even performance code must be maintainable

---

## 📊 Current State Analysis

### Performance Baseline

**Measured on**: Reference system (document your specs)  
**Method**: `cargo bench` + `flamegraph` + `perf`

#### Critical Path Performance (TODO: Measure First)

| Component | Operation | Current | Target | Priority |
|-----------|-----------|---------|--------|----------|
| Subproblem solve | Single forward step | TBD ms | <10ms | 🔴 CRITICAL |
| Cut evaluation | 1000 cuts | TBD µs | <100µs | 🔴 CRITICAL |
| Backward pass | Per node | TBD ms | <50ms | 🔴 CRITICAL |
| Forward pass | 10 scenarios | TBD ms | <100ms | 🟡 HIGH |
| State extraction | Per stage | TBD µs | <50µs | 🟢 MEDIUM |

**Action Item**: Run comprehensive profiling session to establish baseline.

```bash
# Establish baseline
cargo bench --save-baseline before_refactoring
cargo flamegraph --bin powers -- examples/fourbus/
perf record --call-graph dwarf ./target/release/powers examples/fourbus/
```

### Code Quality Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Max file LOC | 6,213 | ≤2,000 | Allow larger files if justified by performance |
| Max params | 13 | ≤6 | More flexible for performance configs |
| Allocations/iter | TBD | Minimize | Profile with massif |
| Clone frequency | 41 in sddp/mod.rs | Minimize in hot paths | OK in cold paths |

### Performance Hot Spots (From Code Analysis)

**Identified Issues** (to be confirmed by profiling):

1. **Excessive Allocations**
   - 41 clones in `sddp/mod.rs`
   - Potential allocations in iteration loops
   - Temporary vectors not reused

2. **Parameter Passing Overhead**
   - 13-parameter functions → stack pressure
   - Passing large structs by value

3. **Cache Inefficiency Risks**
   - Scattered data structures
   - Potential pointer chasing in nested Vecs
   - Arc/Mutex overhead in hot paths

4. **Function Call Overhead**
   - Deep call chains in performance-critical paths
   - Potential for inlining opportunities

---

## 🔥 Hot Path Identification

### Critical Performance Paths

**Forward Pass** (executed 1000s of times):
```
train() 
  → forward_pass()
    → solve_forward_step()  ← HOT
      → realize_uncertainties()  ← HOT
      → model.solve()  ← HOT (FFI to HiGHS)
      → extract_state()  ← HOT
      → evaluate_cuts()  ← HOT
```

**Backward Pass** (executed 1000s of times):
```
train()
  → backward_pass()
    → backward_step_at_node()  ← HOT
      → solve_backward_step()  ← HOT
      → model.solve()  ← HOT (FFI to HiGHS)
      → extract_duals()  ← HOT
      → compute_cut_coefficients()  ← HOT
      → add_cut()  ← HOT
```

**Priority for Optimization**:
1. 🔴 **Tier 1**: `solve_forward_step`, `solve_backward_step`, `evaluate_cuts`
2. 🟡 **Tier 2**: `realize_uncertainties`, `extract_state`, `compute_cut_coefficients`
3. 🟢 **Tier 3**: Algorithm orchestration, initialization, I/O

---

## 📅 Phased Implementation Plan

### Phase 0: Measure & Profile (Week 0)

**Goal**: Establish performance baseline and identify actual bottlenecks

**Priority**: 🔴 CRITICAL | **Risk**: 🟢 NONE | **Impact**: Foundation for all work

#### Tasks

##### 0.1 Comprehensive Profiling Session

```bash
# 1. CPU profiling with flamegraph
cargo install flamegraph
cargo flamegraph --bin powers -- examples/fourbus/
# → Identify which functions consume most CPU

# 2. Perf analysis
perf record --call-graph dwarf -F 999 ./target/release/powers examples/fourbus/
perf report
# → Identify cache misses, branch mispredictions

# 3. Memory profiling
valgrind --tool=massif --massif-out-file=massif.out ./target/release/powers examples/fourbus/
ms_print massif.out
# → Identify allocation hotspots

# 4. Allocation profiling
# Using DHAT (Rust's built-in allocation profiler)
# → Count allocations per iteration

# 5. Benchmark baseline
cargo bench --save-baseline before_refactoring
```

##### 0.2 Document Findings

Create `PROFILING_RESULTS.md`:
```markdown
# Profiling Results - Baseline

## Flamegraph Analysis
- Function X: 35% of CPU time
- Function Y: 18% of CPU time
- ...

## Memory Profile
- Peak memory: X MB
- Allocations per iteration: Y
- Hot allocation sites: ...

## Cache Performance
- L1 miss rate: X%
- L2 miss rate: Y%
- ...

## Bottleneck Ranking
1. [Function/Component]: % of time, issue
2. ...
```

**Success Criteria**:
- [ ] Flamegraph generated and analyzed
- [ ] Perf report reviewed
- [ ] Memory profile completed
- [ ] Top 5 bottlenecks identified
- [ ] Findings documented
- [ ] Team reviewed and prioritized targets

**Timeline**: 2-3 days

---

### Phase 1: Memory Optimization - Pre-allocation (Weeks 1-2)

**Goal**: Eliminate allocations in hot paths through buffer reuse

**Priority**: 🔴 CRITICAL | **Risk**: 🟢 LOW | **Impact**: 🔴 HIGH

**Why First**: Memory allocations are often the #1 performance killer in numerical code. This phase has high impact with low risk.

#### Optimization Strategy

**Pattern**: Move from allocate-on-demand to pre-allocate-and-reuse.

##### 1.1 Add Pre-allocated Buffers to Hot Structures

**Target**: `Subproblem`, `SddpAlgorithm`, forward/backward pass executors

**Before** (allocates every call):
```rust
fn realize_uncertainties(&mut self, innovations: &[f64]) -> Result<()> {
    let mut realized_values = Vec::new();  // ❌ Allocates every call!
    for data in &self.uncertainty_data {
        realized_values.push(data.deterministic_base + data.seasonal_std * innovations[data.innovation_idx]);
    }
    // ... use realized_values
}
```

**After** (reuses buffer):
```rust
pub struct Subproblem {
    // ... existing fields ...
    
    // PERFORMANCE: Pre-allocated buffers for hot path reuse
    // These buffers are sized once during construction and reused
    // across all forward/backward pass iterations.
    realization_buffer: Vec<f64>,      // Size: num_uncertainties
    cut_evaluation_buffer: Vec<f64>,   // Size: max_cuts
    state_extraction_buffer: Vec<f64>, // Size: state_dimension
}

fn realize_uncertainties(&mut self, innovations: &[f64]) -> Result<()> {
    // PERFORMANCE: Reuse pre-allocated buffer (zero allocations)
    self.realization_buffer.clear();
    for data in &self.uncertainty_data {
        self.realization_buffer.push(
            data.deterministic_base + data.seasonal_std * innovations[data.innovation_idx]
        );
    }
    // ... use self.realization_buffer
    Ok(())
}
```

**Impact**: Eliminates ~N allocations per iteration where N = iterations * stages.

##### 1.2 Buffer Pooling for Parallel Execution

For parallel forward passes, each thread needs its own buffers:

```rust
pub struct ForwardPassExecutor {
    subproblems: Vec<Arc<Mutex<Subproblem>>>,
    
    // PERFORMANCE: Thread-local buffer pool
    // Pre-allocate one buffer set per thread to avoid contention
    thread_local_buffers: ThreadLocal<RefCell<WorkBuffers>>,
}

struct WorkBuffers {
    innovations_buffer: Vec<f64>,
    trajectory_buffer: Vec<f64>,
    state_buffer: Vec<f64>,
}

impl ForwardPassExecutor {
    pub fn new(subproblems: Vec<Arc<Mutex<Subproblem>>>, max_state_dim: usize) -> Self {
        Self {
            subproblems,
            thread_local_buffers: ThreadLocal::new(),
        }
    }
    
    fn execute_scenario(&self, scenario: &[Vec<f64>]) -> Result<Trajectory> {
        // PERFORMANCE: Get or create thread-local buffers (allocation only on first use per thread)
        let buffers = self.thread_local_buffers.get_or(|| {
            RefCell::new(WorkBuffers {
                innovations_buffer: Vec::with_capacity(scenario[0].len()),
                trajectory_buffer: Vec::with_capacity(self.subproblems.len()),
                state_buffer: Vec::with_capacity(100), // max state dim
            })
        });
        
        // Use buffers.borrow_mut() for zero-allocation execution
        // ...
    }
}
```

##### 1.3 Pre-size Collections

**Pattern**: Use `with_capacity` everywhere we know the size upfront.

```rust
// ❌ Bad: Grows incrementally (multiple allocations)
let mut cuts = Vec::new();
for node in nodes {
    cuts.push(compute_cut(node));
}

// ✅ Good: Pre-allocated (single allocation)
let mut cuts = Vec::with_capacity(nodes.len());
for node in nodes {
    cuts.push(compute_cut(node));
}

// ✅ Best: Use iterators (often optimizes to single allocation)
let cuts: Vec<_> = nodes.iter()
    .map(|node| compute_cut(node))
    .collect();
```

##### 1.4 Replace Config Objects with Inline Fields

**Clean Code Approach** (causes indirection):
```rust
pub struct SubproblemConfig {
    pub stage: usize,
    pub node_id: usize,
    pub timing_enabled: bool,
    // ... 10 more fields
}

pub struct Subproblem {
    config: SubproblemConfig,  // ❌ Extra indirection, cache miss risk
}

impl Subproblem {
    fn process(&self) {
        if self.config.timing_enabled {  // ❌ Pointer chase
            // ...
        }
    }
}
```

**Performance-Oriented Approach** (flat structure):
```rust
pub struct Subproblem {
    // Flatten config fields directly (better cache locality)
    stage: usize,
    node_id: usize,
    timing_enabled: bool,
    // ... other fields ...
    
    // Still group logically in comments
    // === Solver Configuration ===
    solver_type: Solver,
    solver_timeout: Duration,
    
    // === State Configuration ===
    state_dimension: usize,
    max_ar_order: usize,
}
```

**Trade-off**: More parameters in constructor, but better runtime performance.

**Mitigation**: Use builder pattern, but inline during `build()`:
```rust
pub struct SubproblemBuilder {
    // Config during construction
    config: SubproblemConfig,
}

impl SubproblemBuilder {
    pub fn build(self) -> Subproblem {
        // PERFORMANCE: Flatten config into inline fields
        Subproblem {
            stage: self.config.stage,
            node_id: self.config.node_id,
            timing_enabled: self.config.timing_enabled,
            // ... inline all fields for cache efficiency
            
            // Pre-allocate buffers based on config
            realization_buffer: Vec::with_capacity(self.config.num_uncertainties),
            // ...
        }
    }
}
```

**Best of Both Worlds**: Clean construction API, flat runtime structure.

#### Files to Modify

- ✏️ `src/subproblem.rs`
  - Add buffer fields to `Subproblem`
  - Update methods to use buffers
  - Document with PERFORMANCE comments
  
- ✏️ `src/sddp/mod.rs`
  - Add buffer fields to executor structs
  - Implement thread-local buffer pools
  
- ✏️ `src/state.rs`
  - Pre-allocate state coefficient buffers

#### Benchmarking & Validation

```bash
# Before optimization
cargo bench --save-baseline pre_phase1

# After optimization
cargo bench --baseline pre_phase1

# Expected improvement: 15-30% reduction in forward/backward pass time
# Expected result: 90%+ reduction in allocations (measure with massif)
```

**Success Criteria**:
- [ ] Allocation count per iteration reduced by >80%
- [ ] Benchmark shows ≥10% performance improvement
- [ ] All tests pass
- [ ] No performance regression in other areas
- [ ] Memory usage stays constant across iterations

---

### Phase 2: Cache Optimization - Data Layout (Weeks 3-4)

**Goal**: Improve cache efficiency through contiguous data layouts

**Priority**: 🔴 HIGH | **Risk**: 🟡 MEDIUM | **Impact**: 🔴 HIGH

#### Background: Cache-Friendly Data Structures

**Problem**: Nested vectors and indirection hurt cache performance.

##### 2.1 Flatten Nested Structures

**Current** (potential cache inefficiency):
```rust
pub struct FutureCostFunction {
    // Each node has its own Vec of cuts → scattered in memory
    cuts_by_node: Vec<Vec<Cut>>,  // ❌ Poor cache locality
}

impl FutureCostFunction {
    pub fn evaluate(&self, node_id: usize, state: &[f64]) -> f64 {
        // Accessing cuts_by_node[node_id] may cause cache miss
        let cuts = &self.cuts_by_node[node_id];
        cuts.iter()
            .map(|cut| cut.evaluate(state))  // Each cut access may cache miss
            .max()
            .unwrap_or(0.0)
    }
}
```

**Optimized** (flat, cache-friendly):
```rust
pub struct FutureCostFunction {
    // PERFORMANCE: Flat storage for cache efficiency
    // All cuts stored contiguously, indexed by ranges
    cuts: Vec<Cut>,                    // Contiguous cut storage
    node_ranges: Vec<Range<usize>>,    // node_id → range in cuts vec
}

impl FutureCostFunction {
    pub fn evaluate(&self, node_id: usize, state: &[f64]) -> f64 {
        // PERFORMANCE: Sequential access through contiguous memory
        let range = &self.node_ranges[node_id];
        self.cuts[range.start..range.end]
            .iter()
            .map(|cut| cut.evaluate(state))  // Sequential cache access
            .max()
            .unwrap_or(0.0)
    }
    
    pub fn add_cut(&mut self, node_id: usize, cut: Cut) {
        // PERFORMANCE: Append to flat vec, update range
        let range = &mut self.node_ranges[node_id];
        self.cuts.insert(range.end, cut);
        range.end += 1;
        
        // Update subsequent ranges
        for r in &mut self.node_ranges[node_id + 1..] {
            r.start += 1;
            r.end += 1;
        }
    }
}
```

**Trade-off**: Slightly more complex insertion logic, but much faster evaluation (hot path).

##### 2.2 Struct-of-Arrays (SoA) vs Array-of-Structs (AoS)

For data processed in bulk, SoA can be faster:

**AoS** (traditional):
```rust
#[derive(Clone)]
pub struct Cut {
    pub objective: f64,
    pub coefficients: Vec<f64>,  // ❌ Each cut has its own Vec → fragmented
    pub iteration: usize,
    pub forward_pass_idx: usize,
}

// Evaluating many cuts touches scattered memory
for cut in cuts {
    let value = cut.objective - dot_product(&cut.coefficients, state);
    // ...
}
```

**SoA** (better for SIMD and cache):
```rust
pub struct CutCollection {
    // PERFORMANCE: Struct-of-Arrays for better cache usage
    objectives: Vec<f64>,           // All objectives contiguous
    coefficients: Vec<Vec<f64>>,    // Still need per-cut coeffs
    iterations: Vec<usize>,
    forward_pass_indices: Vec<usize>,
}

impl CutCollection {
    pub fn evaluate_batch(&self, state: &[f64]) -> Vec<f64> {
        // PERFORMANCE: Sequential memory access, better vectorization
        self.objectives.iter()
            .zip(self.coefficients.iter())
            .map(|(obj, coeffs)| obj - dot_product(coeffs, state))
            .collect()
    }
}
```

**When to Use**:
- ✅ Use SoA when processing entire collections (batch evaluation)
- ❌ Use AoS when accessing individual items with all fields

##### 2.3 Reduce Pointer Chasing

**Problem**: `Arc<Mutex<T>>` adds two levels of indirection.

```rust
// ❌ Expensive in hot path (lock + deref)
let subproblem = subproblems[stage].lock().unwrap();
subproblem.solve();
```

**Solutions**:

**Option A**: Use indexes instead of Arc (zero-cost):
```rust
pub struct SddpAlgorithm {
    // PERFORMANCE: Direct vec storage, no Arc/Mutex in single-threaded paths
    subproblems: Vec<Subproblem>,
}

impl SddpAlgorithm {
    fn solve_stage(&mut self, stage: usize) {
        // Direct access, zero overhead
        self.subproblems[stage].solve();
    }
}
```

**Option B**: Split mutable and immutable access patterns:
```rust
pub struct SddpAlgorithm {
    // Immutable data (shared via Arc, no Mutex needed)
    system: Arc<System>,
    graph: Arc<Graph>,
    
    // Mutable data (owned, no Arc/Mutex overhead)
    subproblems: Vec<Subproblem>,
}
```

**Option C**: For parallel access, use more granular locking:
```rust
// Instead of locking entire subproblem
pub struct Subproblem {
    // Immutable parts (no lock needed)
    system: Arc<System>,
    stage: usize,
    
    // Mutable parts (fine-grained locks)
    state: Mutex<State>,
    cuts: RwLock<Vec<Cut>>,  // RwLock: many readers, one writer
}
```

#### Files to Modify

- ✏️ `src/fcf.rs` - Flatten `FutureCostFunction`
- ✏️ `src/cut.rs` - Consider SoA for cut storage
- ✏️ `src/sddp/mod.rs` - Reduce Arc/Mutex overhead
- ✏️ `src/subproblem.rs` - Flatten nested structures

#### Benchmarking & Validation

```bash
# Before
cargo bench --save-baseline pre_phase2

# Use perf to measure cache misses
perf stat -e cache-misses,cache-references cargo bench

# After
cargo bench --baseline pre_phase2

# Expected: 10-20% reduction in cache misses
# Expected: 5-15% improvement in hot path benchmarks
```

**Success Criteria**:
- [ ] Cache miss rate reduced (measure with perf)
- [ ] Benchmark improvements in cut evaluation
- [ ] Memory layout more sequential
- [ ] All tests pass

---

### Phase 3: Eliminate Clones in Hot Paths (Week 5)

**Goal**: Remove unnecessary clones and copies in performance-critical code

**Priority**: 🔴 HIGH | **Risk**: 🟢 LOW | **Impact**: 🟡 MEDIUM

#### Audit Clone Usage

```bash
# Find all clones in hot path files
grep -n "\.clone()" src/sddp/mod.rs src/subproblem.rs src/state.rs

# Current: 41 clones in sddp/mod.rs
# Goal: Reduce to <10 in hot paths
```

##### 3.1 Replace Clone with Borrow

**Pattern**: Pass by reference instead of cloning.

```rust
// ❌ Clones every call
fn process_state(state: State) {
    // ...
}
process_state(state.clone());

// ✅ Borrows (zero-cost)
fn process_state(state: &State) {
    // ...
}
process_state(&state);
```

##### 3.2 Use `Cow<'a, T>` for Conditional Ownership

When sometimes owned, sometimes borrowed:

```rust
use std::borrow::Cow;

fn process_scenario(scenario: Cow<[f64]>) {
    // If we need to modify, it clones on write
    let mut scenario = scenario.into_owned();
    // ...
}

// Call with borrow (no clone)
process_scenario(Cow::Borrowed(&scenario));

// Call with owned (no clone)
process_scenario(Cow::Owned(scenario));
```

##### 3.3 Clone Only When Necessary

Some clones are unavoidable (parallel execution, immutable data structures). Document them:

```rust
// PERFORMANCE: Clone necessary here because:
// 1. Parallel threads need independent copies
// 2. Alternative would be Arc/Mutex (worse performance)
// Profiling shows this clone takes <0.1% of total time.
let thread_scenario = scenario.clone();
```

#### Categorize Clones

1. **Hot Path Clones** 🔴 - Must eliminate
2. **Initialization Clones** 🟢 - OK (cold path)
3. **Parallel Clones** 🟡 - Evaluate case-by-case
4. **API Boundary Clones** 🟢 - OK for safety

#### Files to Modify

- ✏️ `src/sddp/mod.rs` - Audit 41 clones
- ✏️ `src/subproblem.rs` - Check state cloning
- ✏️ `src/state.rs` - Reduce coefficient clones

**Success Criteria**:
- [ ] Clone count in hot paths reduced by >70%
- [ ] Benchmark improvement of 5-10%
- [ ] Documentation for remaining clones
- [ ] All tests pass

---

### Phase 4: Function Inlining & Call Overhead (Week 6)

**Goal**: Reduce function call overhead in hot paths

**Priority**: 🟡 MEDIUM | **Risk**: 🟢 LOW | **Impact**: 🟢 LOW-MEDIUM

#### Strategy

Rust's optimizer is good at inlining, but we can help:

##### 4.1 Strategic `#[inline]` Annotations

**Hot, small functions** - Always inline:
```rust
// PERFORMANCE: Inline to eliminate call overhead (called millions of times)
#[inline(always)]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

**Hot, medium functions** - Hint to inline:
```rust
// PERFORMANCE: Inline across crate boundaries
#[inline]
pub fn evaluate_cut(&self, state: &[f64]) -> f64 {
    self.objective - dot_product(&self.coefficients, state)
}
```

**Cold or large functions** - Don't inline:
```rust
// Cold path: initialization, I/O, error handling
// No inline annotation needed
pub fn parse_input_file(path: &Path) -> Result<System> {
    // Large function, inlining would hurt instruction cache
}
```

##### 4.2 Reduce Call Chain Depth

**Deep call chains** (more overhead):
```rust
fn train() {
    forward_pass()  // Call 1
}

fn forward_pass() {
    execute_scenario()  // Call 2
}

fn execute_scenario() {
    solve_stage()  // Call 3
}

fn solve_stage() {
    // Actual work (4 calls deep!)
}
```

**Flattened** (less overhead):
```rust
fn train() {
    // Inline the orchestration
    for scenario in scenarios {
        for stage in stages {
            // Direct call to worker function
            solve_stage(stage, scenario);
        }
    }
}
```

##### 4.3 Use Iterators (Better Optimization)

Iterators often optimize better than explicit loops:

```rust
// ❌ Manual loop (less optimization opportunity)
let mut sum = 0.0;
for i in 0..n {
    sum += values[i] * weights[i];
}

// ✅ Iterator (LLVM can vectorize better)
let sum: f64 = values.iter()
    .zip(weights.iter())
    .map(|(v, w)| v * w)
    .sum();
```

#### Files to Modify

- ✏️ `src/utils/mod.rs` - Inline mathematical utilities
- ✏️ `src/cut.rs` - Inline cut evaluation
- ✏️ `src/sddp/mod.rs` - Reduce call chain depth

**Success Criteria**:
- [ ] Hot functions annotated with `#[inline]`
- [ ] Call chain depth reduced where possible
- [ ] Benchmark improvement of 3-8%
- [ ] Code remains readable

---

### Phase 5: Algorithmic & Parallelism Improvements (Week 7)

**Goal**: Improve algorithmic efficiency and parallel scaling

**Priority**: 🟡 MEDIUM | **Risk**: 🟡 MEDIUM | **Impact**: 🔴 HIGH

#### 5.1 Algorithmic Optimizations

##### Cut Selection Algorithm

**Current**: Linear search through all cuts (O(n))
```rust
fn select_dominated_cuts(&self, state: &[f64]) -> Vec<usize> {
    let mut dominated = Vec::new();
    for (i, cut) in self.cuts.iter().enumerate() {
        if self.is_dominated(cut, state) {  // O(n) per cut
            dominated.push(i);
        }
    }
    dominated
}
```

**Optimized**: Use spatial data structure (O(log n))
```rust
pub struct CutManager {
    cuts: Vec<Cut>,
    // PERFORMANCE: R-tree for spatial indexing of cuts
    spatial_index: RTree<CutNode>,
}

fn select_dominated_cuts(&self, state: &[f64]) -> Vec<usize> {
    // PERFORMANCE: Spatial query is O(log n + k) where k = result size
    self.spatial_index
        .locate_in_envelope(&state_envelope(state))
        .filter(|node| self.is_dominated(node.cut_id, state))
        .map(|node| node.cut_id)
        .collect()
}
```

**Trade-off**: More complex code, but O(n) → O(log n) for large cut collections.

##### Cut Evaluation

**Current**: Evaluate all cuts, then find max
```rust
fn future_cost(&self, state: &[f64]) -> f64 {
    self.cuts.iter()
        .map(|cut| cut.evaluate(state))  // Evaluates ALL cuts
        .max()
        .unwrap_or(0.0)
}
```

**Optimized**: Early exit with dominated cut pruning
```rust
fn future_cost(&self, state: &[f64]) -> f64 {
    let mut max_value = f64::NEG_INFINITY;
    
    // PERFORMANCE: Sort cuts by objective (descending) for early exit potential
    for cut in self.cuts_sorted_by_objective.iter() {
        let value = cut.evaluate(state);
        if value > max_value {
            max_value = value;
        }
        
        // PERFORMANCE: Early exit if this cut can't possibly be better
        if cut.objective < max_value {
            break;  // Remaining cuts can't beat current max
        }
    }
    
    max_value
}
```

#### 5.2 Parallel Execution Optimization

##### Improve Load Balancing

**Current**: Static work distribution
```rust
// Rayon default: may lead to imbalance
scenarios.par_iter()
    .map(|s| solve_scenario(s))
    .collect()
```

**Optimized**: Dynamic work stealing with chunking
```rust
// PERFORMANCE: Chunk size tuned for load balancing vs overhead
// Small chunks = better balance, more overhead
// Large chunks = less overhead, potential imbalance
scenarios.par_chunks(4)  // Tune based on scenario variance
    .flat_map(|chunk| {
        chunk.iter().map(|s| solve_scenario(s))
    })
    .collect()
```

##### Reduce Synchronization Overhead

**Current**: Lock per operation
```rust
fn add_cut(&self, cut: Cut) {
    let mut cuts = self.cuts.lock().unwrap();  // Lock per cut
    cuts.push(cut);
}
```

**Optimized**: Batch operations
```rust
fn add_cuts_batch(&self, new_cuts: Vec<Cut>) {
    let mut cuts = self.cuts.lock().unwrap();  // Lock once
    cuts.extend(new_cuts);  // Add all at once
}
```

#### 5.3 SIMD Optimization

**Current**: Scalar dot product
```rust
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

**Check Auto-Vectorization**:
```bash
# Check if LLVM is already vectorizing
cargo rustc --release -- --emit asm
# Look for SIMD instructions (addpd, mulpd, etc.)
```

**Explicit SIMD** (if needed):
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// PERFORMANCE: Explicit SIMD for guaranteed vectorization
#[inline]
pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    
    unsafe {
        let mut sum = _mm256_setzero_pd();
        let chunks = a.len() / 4;
        
        for i in 0..chunks {
            let va = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            let prod = _mm256_mul_pd(va, vb);
            sum = _mm256_add_pd(sum, prod);
        }
        
        // Horizontal sum + remainder
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum();
        
        for i in (chunks * 4)..a.len() {
            total += a[i] * b[i];
        }
        
        total
    }
}
```

**Note**: Only if profiling shows this is a bottleneck AND auto-vectorization isn't working.

#### Files to Modify

- ✏️ `src/fcf.rs` - Optimize cut selection
- ✏️ `src/sddp/mod.rs` - Improve parallel load balancing
- ✏️ `src/utils/simd.rs` - Enhance SIMD implementations

**Success Criteria**:
- [ ] Cut selection faster for >1000 cuts
- [ ] Parallel efficiency >75% for 8 threads
- [ ] SIMD code verified with `cargo asm`
- [ ] Benchmark improvements: 10-25%

---

### Phase 6: Code Quality & Documentation (Week 8)

**Goal**: Clean up performance code and document optimizations

**Priority**: 🟡 MEDIUM | **Risk**: 🟢 LOW | **Impact**: 🟡 MEDIUM (for collaboration)

Even performance-optimized code must be readable. This phase adds comments and structure without sacrificing performance.

#### 6.1 Performance Documentation Standards

##### Hot Path Comments

Every optimization should be documented:

```rust
// PERFORMANCE: [What] - [Why] - [Impact]
//
// What: Pre-allocated buffer reused across iterations
// Why: Eliminates N allocations per iteration (N = 1000)
// Impact: 25% speedup in forward pass (measured)
// Profiling: Before: 150ms, After: 112ms
// Alternatives considered:
//   - SmallVec: 3% faster but less readable
//   - Arena allocator: No benefit (profiled)
let mut realization_buffer = vec![0.0; self.max_uncertainties];
```

##### Performance Trade-off Comments

Document when sacrificing clean code for performance:

```rust
// PERFORMANCE TRADE-OFF:
// Using flat fields instead of config struct for cache locality.
// Trade-off: More constructor parameters (8 vs 1)
// Benefit: Eliminates pointer indirection in hot path
// Measured impact: 8% faster in backward pass
pub struct Subproblem {
    stage: usize,
    node_id: usize,
    state_dimension: usize,
    // ... 5 more fields ...
}
```

##### Benchmark References

Link code to benchmarks:

```rust
/// Evaluates all cuts at the given state.
///
/// # Performance
///
/// This is a hot path called thousands of times per iteration.
/// See benchmark: `cargo bench --bench cut_evaluation`
///
/// ## Optimizations Applied
/// - Pre-sorted cuts by objective for early exit
/// - Flat memory layout for cache efficiency
/// - SIMD dot product (see `utils::simd`)
///
/// ## Profiling Data
/// - CPU time: ~5% of total runtime
/// - Cache miss rate: <2%
/// - Throughput: ~1M evaluations/sec
pub fn evaluate_cuts(&self, state: &[f64]) -> f64 {
    // ...
}
```

#### 6.2 Maintain Hot/Cold Path Separation

Keep performance-critical code separate from setup/config code:

```
src/subproblem/
├── mod.rs              (100 lines) - Public API
├── builder.rs          (300 lines) - Construction (COLD PATH)
├── core.rs             (600 lines) - Hot path logic (PERFORMANCE CRITICAL)
├── constraints.rs      (400 lines) - LP setup (COLD PATH)
└── tests/
```

**Core Module** (hot path):
```rust
// src/subproblem/core.rs
//
// PERFORMANCE CRITICAL MODULE
//
// This module contains the hot path logic called thousands of times
// per SDDP iteration. All code here is performance-sensitive.
//
// Guidelines:
// - Minimize allocations (use pre-allocated buffers)
// - Avoid clones (use borrows)
// - Consider inlining (annotate with #[inline])
// - Flat data structures (avoid pointer chasing)
// - Profile changes (run benchmarks)

impl Subproblem {
    /// Execute forward pass step (HOT PATH)
    #[inline]
    pub fn solve_forward_step(&mut self, innovations: &[f64]) -> Result<ForwardStepResult> {
        // PERFORMANCE: All operations here use pre-allocated buffers
        self.realize_uncertainties(innovations)?;
        self.solve_lp()?;
        self.extract_state()
    }
}
```

**Builder Module** (cold path):
```rust
// src/subproblem/builder.rs
//
// CONSTRUCTION MODULE (Cold Path)
//
// This module handles subproblem construction and configuration.
// Clarity and correctness are prioritized over performance.

impl SubproblemBuilder {
    // Complex construction logic - OK to be slower
    pub fn build(self) -> Result<Subproblem> {
        // Validate configuration
        self.validate()?;
        
        // Pre-allocate all buffers based on config
        let realization_buffer = vec![0.0; self.config.num_uncertainties];
        let state_buffer = vec![0.0; self.config.state_dimension];
        
        // Build subproblem with pre-allocated resources
        Ok(Subproblem {
            // ... inline config fields ...
            realization_buffer,
            state_buffer,
        })
    }
}
```

#### 6.3 Add Performance Testing

**Regression Tests**:
```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_no_allocations_in_hot_path() {
        let mut subproblem = create_test_subproblem();
        let innovations = vec![0.5; 10];
        
        // Use allocation tracker or custom allocator
        let alloc_count_before = get_allocation_count();
        
        for _ in 0..1000 {
            subproblem.solve_forward_step(&innovations).unwrap();
        }
        
        let alloc_count_after = get_allocation_count();
        
        // PERFORMANCE: Hot path should have zero allocations after warmup
        assert_eq!(alloc_count_before, alloc_count_after,
            "Hot path should not allocate after warmup");
    }
}
```

#### 6.4 Create Performance Guide

**`docs/PERFORMANCE.md`**:
```markdown
# Performance Guide

## Hot Path Catalog

Functions called >1000 times per execution:

| Function | Calls/Iter | % Time | Optimizations |
|----------|------------|--------|---------------|
| `solve_forward_step` | 1000 | 35% | Pre-allocated buffers, inlined |
| `evaluate_cuts` | 1000 | 12% | Early exit, flat layout |
| `realize_uncertainties` | 1000 | 8% | Buffer reuse, SIMD |

## Optimization Checklist

Before optimizing a function:
- [ ] Profile to confirm it's a bottleneck (>5% of time)
- [ ] Measure baseline with benchmarks
- [ ] Consider algorithmic improvements first
- [ ] Document the optimization with data
- [ ] Verify correctness with tests
- [ ] Measure improvement

## Performance Patterns

### ✅ Good Patterns
- Pre-allocate buffers, reuse in loops
- Pass large structs by reference
- Use iterators for better optimization
- Inline small hot functions
- Flat data structures in hot paths

### ❌ Anti-Patterns
- Allocating in loops
- Cloning in hot paths
- Deep call chains
- Arc/Mutex in single-threaded code
- Premature abstraction in hot paths
```

#### Files to Create/Modify

- ✏️ `docs/PERFORMANCE.md` - Performance guide
- ✏️ Add `// PERFORMANCE:` comments to optimized code
- ✏️ Document trade-offs in commit messages
- ✏️ Link benchmarks in function docs

**Success Criteria**:
- [ ] All optimizations documented
- [ ] Performance guide created
- [ ] Code remains readable
- [ ] New developers can understand optimizations
- [ ] Benchmarks referenced in docs

---

## 🛡️ Risk Mitigation & Validation

### Continuous Performance Monitoring

**After Every Phase**:

```bash
# 1. Run benchmarks
cargo bench --baseline previous_phase

# 2. Check for regressions
# Criterion will show performance changes

# 3. Profile if needed
cargo flamegraph --bin powers -- examples/fourbus/

# 4. Memory check
valgrind --tool=massif ./target/release/powers examples/fourbus/

# 5. All tests must pass
cargo test --release
```

### Performance Regression Detection

**Automated in CI**:
```yaml
# .github/workflows/performance.yml
- name: Run benchmarks
  run: cargo bench --no-fail-fast
  
- name: Compare to baseline
  run: |
    # Fail if >10% regression in critical benchmarks
    cargo bench --baseline main | grep -E "change:.*\+[1-9][0-9]\." && exit 1 || exit 0
```

### Correctness Validation

**Property-Based Tests**:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn optimized_dot_product_matches_naive(
        a in prop::collection::vec(any::<f64>(), 0..1000),
        b in prop::collection::vec(any::<f64>(), 0..1000)
    ) {
        prop_assume!(a.len() == b.len());
        
        let naive = naive_dot_product(&a, &b);
        let optimized = simd_dot_product(&a, &b);
        
        prop_assert!((naive - optimized).abs() < 1e-10,
            "Optimized version must match naive implementation");
    }
}
```

---

## 📊 Success Metrics

### Performance Goals

| Metric | Baseline (Phase 0) | Target | Measurement |
|--------|-------------------|--------|-------------|
| Forward pass | TBD ms | -25% | `cargo bench` |
| Backward pass | TBD ms | -30% | `cargo bench` |
| Allocations/iter | TBD | -80% | `massif` |
| Cache miss rate | TBD % | -20% | `perf stat` |
| Peak memory | TBD MB | Constant | `massif` |
| Parallel efficiency | TBD % | >75% | Custom bench |

### Code Quality Goals

Even with performance focus, maintain quality:

| Metric | Target | Why |
|--------|--------|-----|
| Test coverage | >80% | Ensure correctness |
| Doc comments on hot paths | 100% | Explain optimizations |
| Benchmark coverage | All critical functions | Detect regressions |
| Clippy warnings | 0 | Maintain best practices |
| `unsafe` code | Minimal, documented | Safety |

---

## 🎯 Optimization Priority Matrix

```
High Impact  │ 
             │ Phase 1: Pre-allocate │ Phase 2: Cache Layout
             │ (Weeks 1-2)          │ (Weeks 3-4)
             │                      │
             │ ─────────────────────┼────────────────────
             │                      │
             │ Phase 3: Remove      │ Phase 5: Algorithms
             │ Clones (Week 5)      │ (Week 7)
             │                      │
Low Impact   │ ─────────────────────┼────────────────────
             │                      │
             │ Phase 4: Inlining    │ Phase 6: Docs
             │ (Week 6)             │ (Week 8)
             │                      │
             └──────────────────────┴────────────────────
               Low Risk                High Risk
```

**Interpretation**:
- **Start top-left**: High impact, low risk
- **Be careful bottom-right**: Low impact, high risk (don't do)
- **Measure everything**: Let data drive priorities

---

## 🔬 Profiling & Measurement Tools

### Essential Tools

```bash
# Install performance tools
cargo install flamegraph
cargo install cargo-flamegraph
cargo install hyperfine  # Command-line benchmarking
sudo apt install valgrind linux-tools-generic  # perf, massif
```

### Profiling Cookbook

#### CPU Profiling (Flamegraph)
```bash
# Profile the application
cargo flamegraph --bin powers -- examples/fourbus/

# Open flamegraph.svg in browser
# Look for wide bars (time-consuming functions)
```

#### Cache Analysis (perf)
```bash
# Build release binary
cargo build --release

# Profile cache behavior
perf stat -e cache-misses,cache-references,L1-dcache-load-misses \
    ./target/release/powers examples/fourbus/

# Detailed per-function cache analysis
perf record -e cache-misses ./target/release/powers examples/fourbus/
perf report
```

#### Memory Profiling (massif)
```bash
# Profile memory usage
valgrind --tool=massif --massif-out-file=massif.out \
    ./target/release/powers examples/fourbus/

# Analyze
ms_print massif.out

# Look for:
# - Peak memory usage
# - Allocation hotspots
# - Memory growth over time
```

#### Allocation Counting (DHAT)
```bash
# Build with DHAT support
cargo build --release --features dhat-heap

# Run with allocation tracking
./target/release/powers examples/fourbus/

# DHAT will output allocation stats
```

### Benchmark-Driven Development

```bash
# 1. Write benchmark for target function
cat > benches/my_optimization.rs << 'EOF'
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_target_function(c: &mut Criterion) {
    let data = setup_test_data();
    
    c.bench_function("target_function", |b| {
        b.iter(|| target_function(black_box(&data)))
    });
}

criterion_group!(benches, bench_target_function);
criterion_main!(benches);
EOF

# 2. Establish baseline
cargo bench --bench my_optimization --save-baseline before

# 3. Optimize

# 4. Measure improvement
cargo bench --bench my_optimization --baseline before

# 5. If <5% improvement, try different approach
```

---

## 💡 Performance Patterns Catalog

### Pattern: Buffer Reuse
**When**: Function called repeatedly with temp allocations  
**How**: Add buffer field to struct, reuse across calls  
**Gain**: Eliminates allocations (often 20-40% speedup)

### Pattern: Flat Layout
**When**: Accessing nested data structures in loops  
**How**: Use `Vec<T>` + `Vec<Range<usize>>` instead of `Vec<Vec<T>>`  
**Gain**: Better cache locality (5-15% speedup)

### Pattern: Inline Small Functions
**When**: Small function called millions of times  
**How**: Add `#[inline]` or `#[inline(always)]`  
**Gain**: Eliminates call overhead (3-10% speedup)

### Pattern: Eliminate Clones
**When**: Cloning in loops or hot paths  
**How**: Pass by reference, use `Cow<'a, T>`, or pre-allocate  
**Gain**: Eliminates copy overhead (10-30% speedup)

### Pattern: Struct-of-Arrays
**When**: Processing collections in bulk  
**How**: Store fields separately: `{xs: Vec<X>, ys: Vec<Y>}`  
**Gain**: Better vectorization, cache usage (10-25% speedup)

### Pattern: Early Exit
**When**: Searching sorted data  
**How**: Break when condition can't be met  
**Gain**: Reduces average-case work (varies)

### Pattern: Batch Operations
**When**: Locking or synchronization overhead  
**How**: Collect operations, execute as batch  
**Gain**: Reduces lock contention (20-50% in parallel code)

---

## 📚 Recommended Reading

### Performance Books
- "Systems Performance" - Brendan Gregg (profiling, analysis)
- "The Rust Performance Book" - nnethercote (Rust-specific)
- "Computer Architecture: A Quantitative Approach" - Hennessy & Patterson

### Online Resources
- Rust Performance Book: https://nnethercote.github.io/perf-book/
- Criterion.rs User Guide: https://bheisler.github.io/criterion.rs/book/
- Rust SIMD Working Group: https://rust-lang.github.io/packed_simd/

---

## ✅ Pre-Refactoring Checklist

Before starting Phase 1, ensure:

- [ ] **Profiling tools installed** (flamegraph, perf, valgrind)
- [ ] **Baseline benchmarks run** (`cargo bench --save-baseline`)
- [ ] **Flamegraph generated** (identify hotspots)
- [ ] **Memory profile captured** (massif output)
- [ ] **Top 5 bottlenecks documented**
- [ ] **All tests passing** (`cargo test`)
- [ ] **Team reviewed plan** (agreement on priorities)
- [ ] **Git branch created** (`feature/performance-refactoring`)

---

## 🚦 When to Stop Optimizing

**Stop when**:
1. ✅ Performance goals met (see Success Metrics)
2. ✅ Profiling shows no clear bottlenecks (flat profile)
3. ❌ Further optimization requires major architectural changes
4. ❌ Optimization makes code significantly less maintainable
5. ❌ Cost/benefit ratio is poor (<5% gain for >1 week work)

**Remember**: 
- Perfect is the enemy of good
- 80/20 rule: 80% of impact from 20% of optimizations
- Measure, don't guess

---

## 📋 Phase 0 Action Items (Start Here!)

Before any refactoring:

```bash
# 1. Setup profiling environment
cargo install flamegraph criterion
sudo apt install linux-tools-generic valgrind

# 2. Run comprehensive profiling (save to PROFILING_RESULTS.md)
./scripts/profile_baseline.sh  # Create this script

# 3. Establish baseline
cargo bench --save-baseline before_refactoring
git add target/criterion/before_refactoring
git commit -m "chore: establish performance baseline"

# 4. Document findings
cp PROFILING_RESULTS_TEMPLATE.md PROFILING_RESULTS.md
# Fill in actual numbers

# 5. Review with team
# Prioritize based on actual bottlenecks, not assumptions

# 6. Update this plan
# Adjust phases based on profiling data
```

---

**Ready to optimize! Start with Phase 0 profiling. 🚀🔥**

---

## Appendix A: Performance vs Clean Code Trade-offs

This plan intentionally differs from pure "clean code" approaches:

| Clean Code Principle | Our Approach | Rationale |
|---------------------|--------------|-----------|
| ≤4 function parameters | Allow up to 6-8 for perf configs | Avoid indirection overhead |
| Small, single-purpose functions | Inline hot paths | Reduce call overhead |
| Prefer abstraction | Flatten hot structures | Cache efficiency |
| DRY (Don't Repeat Yourself) | Allow duplication in hot/cold paths | Different optimization needs |
| Avoid premature optimization | Profile-guided optimization | HPC requires performance |

**Key Insight**: Separate hot paths (performance-critical) from cold paths (clarity-critical).

---

## Appendix B: Benchmarking Cheat Sheet

```bash
# Quick benchmark of single function
cargo bench --bench <name> -- <function>

# Save baseline before changes
cargo bench --save-baseline <name>

# Compare to baseline
cargo bench --baseline <name>

# Benchmark with flamegraph
cargo flamegraph --bench <name>

# Statistical comparison
cargo bench -- --save-baseline before
# ... make changes ...
cargo bench -- --baseline before

# Run specific benchmark pattern
cargo bench -- dot_product

# Generate HTML reports
cargo bench
firefox target/criterion/report/index.html
```

---

**Version**: 1.0  
**Status**: Ready for Phase 0  
**Next Action**: Run profiling session
