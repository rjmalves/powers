# Refactoring Approaches Comparison

This document compares the two refactoring approaches for POWE.RS:
1. **Clean Code Approach** (`REFACTORING_PLAN.md`) - Prioritizes maintainability
2. **Performance Approach** (`PERFORMANCE_REFACTORING_PLAN.md`) - Prioritizes speed

---

## Philosophy Comparison

| Aspect | Clean Code | Performance-Oriented |
|--------|------------|---------------------|
| **Primary Goal** | Collaborative development readiness | Optimal execution speed |
| **Measurement** | Code metrics (LOC, params, etc.) | Profiling data (time, memory) |
| **Starting Point** | Architectural analysis | Performance profiling |
| **Trade-offs** | Clarity over minor performance hits | Performance over some abstractions |
| **Success Metric** | Code understandable in 15 min | X% faster execution |

---

## Key Differences

### 1. Configuration Objects

**Clean Code Approach**:
```rust
pub struct SubproblemConfig {
    pub stage: usize,
    pub node_id: usize,
    // ... 10 more fields
}

pub struct Subproblem {
    config: SubproblemConfig,  // Config object
    // ...
}
```
- ✅ Clean API: 13 params → 1 param
- ✅ Easy to extend
- ❌ Extra indirection (pointer chase)
- ❌ Potential cache miss

**Performance Approach**:
```rust
pub struct Subproblem {
    // Inline all fields (flat structure)
    stage: usize,
    node_id: usize,
    // ... 10 more fields inline
}
```
- ✅ No indirection (cache-friendly)
- ✅ Direct field access
- ❌ More fields in struct
- ⚖️ Use builder for construction

**Winner**: Depends on context
- Hot paths: Performance approach
- Cold paths: Clean code approach

---

### 2. Module Decomposition

**Clean Code Approach**:
- Split `subproblem.rs` (6,213 LOC) → 6+ focused modules (~1,000 LOC each)
- Maximize separation of concerns
- Service layer with interfaces
- Easy to test in isolation

**Performance Approach**:
- Split but allow larger files if justified (up to 2,000 LOC)
- Keep hot path code together (fewer module boundaries)
- Separate hot/cold paths explicitly
- Inline critical functions

**Winner**: Hybrid
- Use clean code structure
- Keep hot paths contiguous
- Document performance-critical regions

---

### 3. Function Parameters

| Approach | Max Parameters | Rationale |
|----------|----------------|-----------|
| Clean Code | ≤4 | Uncle Bob's guideline |
| Performance | ≤6-8 for configs | Avoid indirection |

**Example**:

```rust
// Clean Code: Use config object
fn solve(config: &SolverConfig) -> Result<()>

// Performance: Inline hot fields, group cold
fn solve(
    // Hot fields (accessed frequently)
    state: &[f64],
    cuts: &[Cut],
    // Cold config (can be struct)
    options: &SolverOptions,
) -> Result<()>
```

---

### 4. Allocations

**Both Approaches Agree**: Minimize allocations in hot paths

**Clean Code**: May introduce abstractions that inadvertently allocate

**Performance**: Profile-driven, explicit buffer reuse
```rust
pub struct Subproblem {
    // Pre-allocated buffers (documented)
    realization_buffer: Vec<f64>,
    cut_eval_buffer: Vec<f64>,
}
```

---

### 5. Data Structures

**Clean Code Approach**:
```rust
// Idiomatic nested structures
pub struct FutureCostFunction {
    cuts_by_node: Vec<Vec<Cut>>,  // Clear intent
}
```

**Performance Approach**:
```rust
// Flat layout for cache efficiency
pub struct FutureCostFunction {
    cuts: Vec<Cut>,               // Contiguous
    node_ranges: Vec<Range<usize>>, // Index
}
```

**Trade-off**:
- Clean: Easier to understand
- Performance: 10-20% faster access

---

## Recommended Hybrid Approach

### Guiding Principle

> **"Use clean code everywhere, optimize hot paths explicitly."**

### Strategy

1. **Start with Clean Code structure**
   - Module organization
   - API design
   - Testing strategy

2. **Profile to identify hot paths** (Phase 0)
   - Run flamegraph
   - Measure allocations
   - Identify bottlenecks

3. **Apply Performance optimizations selectively**
   - Hot paths: Flatten, pre-allocate, inline
   - Cold paths: Keep clean abstractions
   - Document trade-offs

4. **Maintain both qualities**
   - Performance code must be readable
   - Clean code shouldn't hurt performance

### Example: Hybrid Subproblem

```rust
// Clean API for construction
pub struct SubproblemBuilder {
    config: SubproblemConfig,  // Config object during build
}

impl SubproblemBuilder {
    pub fn build(self) -> Subproblem {
        // PERFORMANCE: Flatten during construction
        Subproblem {
            // Hot fields inlined
            stage: self.config.stage,
            node_id: self.config.node_id,
            // ...
            
            // Pre-allocated buffers
            realization_buffer: vec![0.0; self.config.buffer_size],
            
            // Cold config can stay as object
            solver_config: self.config.solver_config,
        }
    }
}

pub struct Subproblem {
    // === Hot Path Fields (inline for cache) ===
    stage: usize,
    node_id: usize,
    state_dimension: usize,
    
    // Pre-allocated buffers
    realization_buffer: Vec<f64>,
    
    // === Cold Path Fields (can be wrapped) ===
    solver_config: Arc<SolverConfig>,  // Shared, rarely accessed
}

impl Subproblem {
    // PERFORMANCE CRITICAL: Hot path
    #[inline]
    pub fn solve_forward_step(&mut self, innovations: &[f64]) -> Result<State> {
        // Direct field access, no indirection
        // Uses pre-allocated buffers
        self.realize_uncertainties(innovations)?;
        self.solve_lp()?;
        self.extract_state()
    }
    
    // Cold path: configuration
    pub fn update_solver_options(&mut self, options: SolverOptions) {
        // Can use abstractions here
        self.solver_config = Arc::new(options.into());
    }
}
```

**Benefits**:
- ✅ Clean construction API
- ✅ Fast execution
- ✅ Clear hot/cold separation
- ✅ Documented trade-offs

---

## Phase Priority Comparison

### Clean Code Plan
1. Extract configs (reduce params)
2. Decompose god objects (split files)
3. Extract services (separation)
4. Improve naming (clarity)
5. Reduce function complexity

### Performance Plan
0. **Profile first** (measure baseline)
1. Pre-allocate buffers (eliminate allocations)
2. Cache optimization (data layout)
3. Remove clones (reduce copies)
4. Function inlining (reduce overhead)
5. Algorithmic improvements

**Key Difference**: Performance plan starts with profiling!

---

## Recommendation

### For POWE.RS (HPC application)

Use **Performance-Oriented Plan** because:

1. ✅ **Performance is Critical**: This is an HPC solver
   - Runs for hours/days
   - 30% speedup = significant cost savings
   
2. ✅ **Profile-Driven**: Makes decisions based on data
   - No premature optimization
   - No wasted effort on cold paths
   
3. ✅ **Maintains Quality**: Still requires
   - Clear documentation
   - Comprehensive tests
   - Readable code (with perf comments)

4. ✅ **Pragmatic**: Allows trade-offs
   - Hot paths: Performance first
   - Cold paths: Clarity first
   - Document the choice

### Execution Strategy

```
Week 0: Profiling (Phase 0)
  ↓
Weeks 1-2: Memory optimization (Phase 1)
  ↓ [Measure improvement]
Weeks 3-4: Cache optimization (Phase 2)
  ↓ [Measure improvement]
Week 5: Remove clones (Phase 3)
  ↓ [Measure improvement]
Week 6: Inlining (Phase 4)
  ↓ [Measure improvement]
Week 7: Algorithms (Phase 5)
  ↓ [Measure improvement]
Week 8: Documentation (Phase 6)
  ↓
Review: Did we meet performance goals?
```

**Each phase validated with benchmarks!**

---

## When to Use Each Approach

### Use Clean Code Approach When:
- Building a library/API for others
- Performance is not critical (< 1 second runtime)
- Team is large and changing frequently
- Code is more read than executed

### Use Performance Approach When:
- HPC/scientific computing
- Long-running applications (hours/days)
- Performance = cost savings
- Code is executed millions of times

### Use Hybrid (Recommended) When:
- Both performance and maintainability matter
- Team is medium-sized and stable
- Application has clear hot/cold paths
- **This describes POWE.RS perfectly!**

---

## Validation

Both approaches must pass:
- ✅ All tests pass
- ✅ No clippy warnings
- ✅ Code is documented
- ✅ Benchmarks show no regression

Performance approach adds:
- ✅ Profiling data supports changes
- ✅ Benchmark shows improvement
- ✅ Performance goals met

---

## Conclusion

**For POWE.RS**: Use the **Performance-Oriented Refactoring Plan**

**Why**:
1. HPC application where performance matters
2. Profile-driven approach prevents wasted effort
3. Still maintains code quality (with perf comments)
4. Pragmatic: optimize hot paths, clean code elsewhere

**Start with**: `./scripts/profile_baseline.sh`

**Follow**: `PERFORMANCE_REFACTORING_PLAN.md`

**Remember**: 
> "Measure twice, optimize once."  
> "Document the trade-offs."  
> "Keep hot paths hot, cold paths clean."

---

**Both plans are good. Choose based on your priorities.**

For collaboration-first: Use `REFACTORING_PLAN.md`  
For performance-first: Use `PERFORMANCE_REFACTORING_PLAN.md`  
For best-of-both: Use performance plan with extra documentation

**Your call! 🚀**
