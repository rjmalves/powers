# [TICKET-007] Integration and validation

> **Epic**: [Epic 1: HiGHS Constraint Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 2](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-006](./ticket-006-slot-reuse.md)  
> **Blocks**: Epic 2 (FCF preallocation)

## Context

### Background

All infrastructure is in place. This ticket integrates preallocation into the training pipeline and validates correctness and performance.

### Relation to Epic

Final integration ticket that enables preallocation and validates the complete implementation.

## Files to Read Before Starting

- `src/sddp/mod.rs` - Training loop, handler creation
- `src/sddp/builder.rs` - SDDP instance construction
- `src/memory/sizing.rs` - SizingInfo computation

## Specification

### Integration Points

1. **At Training Start**: Call `preallocate_cut_constraints()` for all subproblems
2. **Location**: In `SddpAlgorithm::train()` or `SddpBuilder`
3. **Timing**: After subproblems are created, before training loop

### Integration Code

Find where training starts and add:

```rust
// After subproblems are created, before training loop
let max_cuts = sizing.estimate_max_cuts_per_node();

for node_id in 0..num_nodes {
    let state_dim = sizing.state_dimension_for_node(node_id).unwrap_or(sizing.max_state_dimension);
    let subproblem = /* get mutable reference */;
    subproblem.preallocate_cut_constraints(max_cuts, state_dim)?;
}
```

### Validation Checklist

1. **Correctness**: Examples produce identical results
2. **Memory**: Flat profile during training
3. **Performance**: ≥3% improvement
4. **No Regressions**: All edge cases work

## Acceptance Criteria

- [ ] Preallocation called at training start
- [ ] Example 01: Identical lower bound trajectory
- [ ] Example 07: Identical convergence (within 0.001%)
- [ ] Memory profile flat (±1%) during training
- [ ] Performance improvement ≥3%
- [ ] No warnings or errors in logs

## Implementation Guide

### Suggested Approach

1. Locate training initialization in `src/sddp/mod.rs`
2. Add preallocation loop after subproblem creation
3. Run examples and compare results
4. Profile memory with valgrind/massif
5. Benchmark with hyperfine

### Key Files to Modify

- `src/sddp/mod.rs`: Add preallocation in training initialization
- `src/sddp/builder.rs`: Alternative location if subproblems created there

### Validation Commands

```bash
# Correctness: Compare lower bounds
cargo run --release -- run examples/01-deterministic 2>&1 | grep "lower"
# Expected: All lower bounds match baseline

cargo run --release -- run examples/07-par-model-with-inflow-state 2>&1 | grep "lower"
# Expected: Final lower bound matches baseline (±0.001%)

# Memory profile
valgrind --tool=massif --massif-out-file=massif.out \
  ./target/release/powers run examples/07-par-model-with-inflow-state
ms_print massif.out | head -100
# Expected: Flat profile after initialization

# Performance benchmark
hyperfine --warmup 2 --runs 5 \
  'cargo run --release -- run examples/07-par-model-with-inflow-state'
# Expected: ≥3% faster than baseline

# Baseline (before preallocation)
# Record baseline times for comparison
```

### Memory Profile Expectations

**Before (dynamic allocation)**:
```
  MB
82.0 ^                                      #
     |                                   ####
     |                                ####
     |                             ####
     |                          ####
73.0 +-----------------------------##-------
     |                       ####
68.0 +                    ####
     |                 ####
     +---+---+---+---+---+---+---+---+---+--->
                         iterations
```

**After (preallocation)**:
```
  MB
73.0 ^#######################################
     |#######################################
     |#######################################
     |#######################################
     |#######################################
     +---+---+---+---+---+---+---+---+---+--->
                         iterations
```

### Integration Code Location

Look for training loop in `src/sddp/mod.rs`:

```rust
impl SddpAlgorithm {
    pub fn train(...) -> Result<TrainingResult, String> {
        // ... initialization ...
        
        // ADD PREALLOCATION HERE
        self.preallocate_all_cut_constraints()?;
        
        // Training loop
        for iteration in 0..num_iterations {
            // Forward pass
            // Backward pass (uses preallocated slots)
        }
    }
    
    fn preallocate_all_cut_constraints(&mut self) -> Result<(), String> {
        let max_cuts = self.sizing.estimate_max_cuts_per_node();
        
        for handler in &mut self.train_handlers {
            for node in handler.subproblem_graph.iter_nodes_mut() {
                let state_dim = /* get from sizing or state */;
                node.data.preallocate_cut_constraints(max_cuts, state_dim)?;
            }
        }
        
        log::info!(
            "Preallocated {} cut slots per subproblem ({} total)",
            max_cuts,
            max_cuts * self.node_count()
        );
        
        Ok(())
    }
}
```

### Pitfalls to Avoid

- ⚠️ Ensure preallocation happens AFTER subproblems have models
- ⚠️ State dimension may vary per node (use correct one)
- ⚠️ Don't preallocate for simulation handlers (only training)
- ⚠️ Verify `SizingInfo` is available at preallocation point

## Testing Requirements

### Regression Tests

- [ ] Example 01: Lower bound trajectory matches baseline exactly
- [ ] Example 07: Final lower bound matches baseline (±0.001%)

### Memory Tests

- [ ] Massif shows flat profile during training
- [ ] No memory growth >1% after initialization

### Performance Tests

- [ ] Hyperfine shows ≥3% improvement on example 07

### Edge Cases

- [ ] Zero cuts (early termination): No crash
- [ ] Maximum cuts reached: Fallback works

## Documentation Requirements

- [ ] Log message shows preallocation count
- [ ] Update CHANGELOG.md with performance improvement

## Effort Estimate

**Points**: 2  
**Confidence**: Medium  
**Rationale**: Integration and validation, depends on finding right location

## Definition of Done

- [ ] Preallocation integrated into training
- [ ] All examples produce correct results
- [ ] Memory profile validated as flat
- [ ] Performance improvement measured
- [ ] CHANGELOG.md updated
- [ ] Epic 1 complete ✅
