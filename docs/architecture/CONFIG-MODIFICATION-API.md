# Configuration Modification API Design

**Status**: Proposal  
**Date**: October 9, 2025  
**Context**: Benchmarking memory scaling with varying training parameters

## Problem Statement

The current `SddpAlgorithm::from_files()` API is monolithic:

```rust
let sddp_instance = SddpAlgorithm::from_files(config, system, graph, recourse)?;
sddp_instance.train()?;  // Uses config's num_iterations and num_forward_passes
```

**Limitations**:

1. **Can't modify config after loading**: No way to change `num_iterations`, `num_forward_passes`, or `seed` for parameter sweeps
2. **Borrow checker prevents workarounds**: Can't call `instance.algorithm_mut().train(n, m, instance.saa())` due to simultaneous mutable + immutable borrows
3. **Benchmarking is difficult**: Need to create multiple config JSON files to test different parameters

## Root Cause: Borrow Granularity

`SddpInstance` owns three components as a monolithic struct:

```rust
pub struct SddpInstance {
    algorithm: SddpAlgorithm,  // needs &mut for train()
    config: Config,            // immutable, but embedded
    saa: SAA,                  // needs & for train()
}
```

When you call:

- `instance.saa()` → borrows entire struct immutably
- `instance.algorithm_mut()` → borrows entire struct mutably
- **Cannot do both**: Rust borrow checker prevents simultaneous borrows

This is a **struct-level borrow granularity problem** - Rust doesn't know that `algorithm` and `saa` are independent fields.

## Proposed Solution 1: Builder Pattern (Recommended)

Add an intermediate builder for staged construction:

```rust
pub struct SddpInstanceBuilder {
    system: System,
    graph: GraphInput,
    recourse: Recourse,
    config: Config,
}

impl SddpInstanceBuilder {
    /// Load inputs from JSON files (parsable, validatable)
    pub fn from_paths(
        config_path: impl AsRef<Path>,
        system_path: impl AsRef<Path>,
        graph_path: impl AsRef<Path>,
        recourse_path: impl AsRef<Path>,
    ) -> Result<Self, PowersError> {
        let input = Input::from_paths(
            config_path.as_ref(),
            system_path.as_ref(),
            graph_path.as_ref(),
            recourse_path.as_ref(),
        )?;

        Ok(Self {
            system: input.system,
            graph: input.graph,
            recourse: input.recourse,
            config: input.config,
        })
    }

    /// Modify num_forward_passes before building
    pub fn with_num_forward_passes(mut self, n: usize) -> Self {
        self.config.num_forward_passes = n;
        self
    }

    /// Modify num_iterations before building
    pub fn with_num_iterations(mut self, n: usize) -> Self {
        self.config.num_iterations = n;
        self
    }

    /// Modify seed before building
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Build the SddpInstance with (potentially modified) config
    pub fn build(self) -> Result<SddpInstance, PowersError> {
        // Build graph, SAA, algorithm (same logic as from_files)
        let node_data_graph = self.graph
            .build_sddp_graph(&self.system)
            .map_err(|e| PowersError::Other(format!("Failed to build graph: {}", e)))?;

        let initial_condition = self.recourse.build_sddp_initial_condition();
        let saa = self.recourse.generate_sddp_noises(&node_data_graph, self.config.seed);
        let algorithm = SddpAlgorithm::new(node_data_graph, initial_condition, self.config.seed)
            .map_err(PowersError::Other)?;

        Ok(SddpInstance::new(algorithm, self.config, saa))
    }
}
```

### Usage Example: Parameter Sweep

```rust
// Benchmark: Memory scaling with forward passes
for num_fwd in [1, 4, 8, 16, 32] {
    let mut instance = SddpInstanceBuilder::from_paths(
        "examples/05-large-scale-brazilian/config.json",
        "examples/05-large-scale-brazilian/system.json",
        "examples/05-large-scale-brazilian/graph.json",
        "examples/05-large-scale-brazilian/recourse.json",
    )?
    .with_num_forward_passes(num_fwd)
    .build()?;

    let result = instance.train()?;  // Uses modified config
    println!("Memory: {} MB, Forward passes: {}", get_memory(), num_fwd);
}
```

### Benefits

1. ✅ **Clean API**: Fluent builder pattern is idiomatic Rust
2. ✅ **Flexible**: Modify any config parameter before building
3. ✅ **Type-safe**: All validation happens in `build()`
4. ✅ **Zero overhead**: Builder is consumed, no runtime cost
5. ✅ **Backward compatible**: Keep existing `from_files()` as convenience method

## Proposed Solution 2: Split Borrow Method (Quick Fix)

Add a method to `SddpInstance` that handles split borrows internally:

````rust
impl SddpInstance {
    /// Train with custom parameters, bypassing the embedded config.
    ///
    /// This allows parameter sweeps without borrow checker issues.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of SDDP iterations
    /// * `num_forward_passes` - Number of forward passes per iteration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut sddp = SddpAlgorithm::from_files(...)?;
    /// let result = sddp.train_custom(10, 32)?;  // Override config
    /// ```
    pub fn train_custom(
        &mut self,
        num_iterations: usize,
        num_forward_passes: usize,
    ) -> Result<TrainingResult, String> {
        // Split borrows internally - safe because we own self
        let saa = &self.saa;
        let algorithm = &mut self.algorithm;
        algorithm.train(num_iterations, num_forward_passes, saa)
    }
}
````

### Usage Example

```rust
let mut instance = SddpAlgorithm::from_files(...)?;

// Override config for benchmarking
let result = instance.train_custom(8, 32)?;  // 8 iters, 32 forward passes
```

### Benefits

1. ✅ **Simple**: One method, minimal code changes
2. ✅ **Immediate**: Solves borrow checker issue now
3. ✅ **Safe**: Split borrow is legal inside method
4. ⚠️ **Limited**: Only works for train parameters, not seed or other config

## Recommendation

**Do BOTH**:

1. **Short-term** (Sprint 4.5): Add `train_custom()` method to unblock memory benchmarking
2. **Long-term** (Phase 2): Implement `SddpInstanceBuilder` for full flexibility

### Rationale

- `train_custom()` is a **1-hour fix** that unblocks current work
- `SddpInstanceBuilder` is a **4-6 hour refactor** with broader benefits:
  - Enables comprehensive parameter sweeps (benchmarking, sensitivity analysis)
  - Standard HPC pattern (PETSc, Trilinos, deal.II use builders)
  - Enables future features (warm-starting, checkpointing, custom graph construction)

## Implementation Priority

### Phase 1: Immediate (Sprint 4.5)

```rust
// src/sddp/instance.rs
impl SddpInstance {
    pub fn train_custom(&mut self, num_iterations: usize, num_forward_passes: usize)
        -> Result<TrainingResult, String>
    {
        let saa = &self.saa;
        self.algorithm.train(num_iterations, num_forward_passes, saa)
    }
}
```

**Effort**: 1 hour (implementation + tests)  
**Value**: HIGH (unblocks memory profiling)

### Phase 2: Future Enhancement

```rust
// src/sddp/builder.rs
pub struct SddpInstanceBuilder { /* ... */ }
```

**Effort**: 4-6 hours (implementation + tests + examples + docs)  
**Value**: MEDIUM-HIGH (enables advanced use cases, improves API ergonomics)

## Testing Strategy

### Unit Tests (train_custom)

```rust
#[test]
fn test_train_custom_overrides_config() {
    let mut sddp = SddpAlgorithm::from_files(...)?;

    // Config has 8 iters, 16 fwd passes
    assert_eq!(sddp.config().num_iterations, 8);
    assert_eq!(sddp.config().num_forward_passes, 16);

    // Override with custom values
    let result = sddp.train_custom(10, 32)?;

    // Should use custom values, not config
    assert_eq!(result.num_iterations(), 10);
}
```

### Integration Tests (Builder)

```rust
#[test]
fn test_builder_modifies_config() {
    let sddp = SddpInstanceBuilder::from_paths(...)?
        .with_num_forward_passes(32)
        .with_num_iterations(10)
        .build()?;

    assert_eq!(sddp.config().num_forward_passes, 32);
    assert_eq!(sddp.config().num_iterations, 10);
}
```

## Performance Considerations

### Builder Pattern Performance

- **Construction**: No overhead - builder is consumed via move semantics
- **Runtime**: Zero overhead - same code path as `from_files()`
- **Memory**: No extra allocations (moves, not copies)

### train_custom() Performance

- **Call overhead**: ~5-10 ns (one method call)
- **Negligible**: Training takes seconds, overhead is < 0.0001%

## Alternatives Considered

### 1. Multiple Config Files ❌

**Pros**: No code changes  
**Cons**:

- Requires external file management
- Not programmatic (can't automate parameter sweeps)
- Clutters repository with config variants

### 2. Unsafe Split Borrows ❌

**Pros**: Works today  
**Cons**:

- Unsafe code in public API
- Violates API safety guarantees
- Hard to maintain

### 3. Clone SAA ❌

**Pros**: Avoids borrow issues  
**Cons**:

- Expensive (SAA is large)
- Defeats the purpose of the Instance wrapper

## Conclusion

The **builder pattern + train_custom()** approach provides:

1. **Immediate relief**: `train_custom()` unblocks benchmarking now
2. **Future flexibility**: Builder enables advanced use cases later
3. **Best practices**: Standard HPC API design pattern
4. **Performance**: Zero overhead (move semantics, no allocations)

**Recommendation**: Implement `train_custom()` in Sprint 4.5, defer Builder to Phase 2.
