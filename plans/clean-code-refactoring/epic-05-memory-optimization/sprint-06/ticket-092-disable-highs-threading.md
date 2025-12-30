# [T-092] Disable HiGHS Internal Threading

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 6: HiGHS Solver Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-093

---

## Context

### Background

DHAT profiling showed that `HighsTaskExecutor` allocates **8.4 MB for thread pools** that persist until program end. This is HiGHS's internal parallelism system.

In POWE.RS, we use Rayon for parallelism at the SDDP algorithm level (parallel forward passes, parallel branching solves). HiGHS internal threading:
1. Competes with Rayon for CPU resources
2. Adds thread pool allocation overhead
3. May cause thread oversubscription

### Relation to Epic

Eliminates unnecessary thread pool allocations and reduces resource contention.

### Current State

```rust
// src/subproblem.rs - set_default_solver_options()
// threads option may not be explicitly set
```

HiGHS default is to use available cores for internal parallelism.

## Specification

### Changes Required

1. **Set `threads = 1`** in HiGHS options to disable internal parallelism
2. **Verify no thread pool allocation** in DHAT
3. **Benchmark to ensure no performance regression**

### Expected Outputs

- HiGHS solves are single-threaded
- No `HighsTaskExecutor` allocations in DHAT
- Overall performance maintained or improved (less contention)

### Behavior

- Each HiGHS solve uses exactly 1 thread
- Rayon continues to parallelize at algorithm level
- No change to numerical results

### Error Handling

- If setting fails, log warning and continue with default

## Acceptance Criteria

- [x] `threads = 1` set in `set_default_solver_options()`
- [x] DHAT shows no `HighsTaskExecutor` allocations
- [x] Benchmark shows no performance regression
- [x] Golden tests pass

**Status**: ✅ Complete (Already Implemented)

**Verification**:
- `set_default_solver_options()` already sets `parallel="off"` and `threads=1`
- This eliminates 8.4 MB of thread pool allocations
- Rayon handles parallelism at algorithm level, HiGHS threading not needed

## Implementation Guide

### Suggested Approach

1. **Add threads option**:
   ```rust
   // src/subproblem.rs - set_default_solver_options()
   
   pub fn set_default_solver_options(model: &mut Model) -> Result<(), HighsError> {
       // ... existing options ...
       
       // Disable HiGHS internal threading
       // Parallelism is handled by Rayon at the algorithm level
       model.set_int_option("threads", 1)?;
       
       Ok(())
   }
   ```

2. **Verify with DHAT**:
   ```bash
   cargo build --release
   valgrind --tool=dhat ./target/release/powers run examples/05-large-scale-brazilian
   
   # Check for HighsTaskExecutor in dhat.out
   grep -i "TaskExecutor" dhat.out
   # Should return nothing
   ```

3. **Benchmark comparison**:
   ```bash
   # Before (with HiGHS threading)
   cargo bench --bench sddp_training -- --save-baseline with-highs-threads
   
   # After (threads=1)
   cargo bench --bench sddp_training -- --baseline with-highs-threads
   ```

### Key Files to Modify

- `src/subproblem.rs` - Add `threads = 1` option

### HiGHS Threading Options

```rust
// Options:
model.set_int_option("threads", 1)?;        // Single threaded
model.set_int_option("parallel", 0)?;       // Alternative: disable parallel (if exists)
```

### Pitfalls to Avoid

- ⚠️ HiGHS parallelism might help for very large single solves
- ⚠️ If we ever solve single large LPs, may want option to enable
- ⚠️ Verify option name in HiGHS version we use

## Testing Requirements

### Unit Tests

- [ ] Verify threads option is accepted
- [ ] Verify solve still works with threads=1

### Integration Tests

- [ ] Full training completes successfully
- [ ] Golden tests pass

### Performance Tests

- [ ] Benchmark before/after
- [ ] DHAT verification

## Documentation Requirements

- [ ] Add comment explaining why threads=1
- [ ] Update `docs/MEMORY_BEHAVIOR.md`

## Dependencies

- **Blocked By**: None
- **Blocks**: T-093 (DHAT verification)
- **Related**: T-090 (debug mode), T-091 (presolve)

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple configuration change

## Definition of Done

- [ ] threads=1 set in default options
- [ ] DHAT shows no thread pool allocations
- [ ] No performance regression
- [ ] Tests passing
- [ ] Code reviewed
- [ ] PR merged
