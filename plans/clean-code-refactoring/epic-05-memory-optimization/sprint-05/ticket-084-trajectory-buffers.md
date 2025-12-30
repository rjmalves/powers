# [T-084] Preallocate Trajectory Buffers and Eliminate Cloning

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 5](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-085](./ticket-085-dhat-profiling.md)
> **Status**: ✅ Complete (verified existing implementation)

---

## Context

### Background

The SDDP module (`sddp/mod.rs`) extensively clones `Realization` data when building results:

```rust
// sddp/mod.rs:1248-1259
loads: realization.loads.clone(),
deficit: realization.deficit.clone(),
exchange: realization.exchange.clone(),
inflow: realization.inflow.clone(),
turbined_flow: realization.turbined_flow.clone(),
spillage: realization.spillage.clone(),
thermal_generation: realization.thermal_generation.clone(),
water_value: realization.water_value.clone(),
marginal_cost: realization.marginal_cost.clone(),
// ...
final_storage: realization.final_storage.clone(),
```

Each clone allocates a new Vec. For a 12-stage problem with 500 forward passes:
- ~10 clones × 12 stages × 500 passes = 60,000 Vec allocations per iteration

### Current State

- `Realization` data cloned for result construction
- Each forward pass creates new trajectory storage
- Large allocation overhead during training

### Target State

- Trajectory buffers preallocated at training start
- Realizations updated in-place, not cloned
- Results constructed via references or indices, not owned data

---

## Specification

### Inputs

- Number of stages in problem
- Number of forward passes per iteration
- Realization structure (number of hydros, thermals, buses, etc.)

### Outputs

- Preallocated trajectory storage for all forward passes
- Zero-allocation realization updates during training

### Approach Options

#### Option A: Preallocated Realization Pool

```rust
struct TrajectoryPool {
    /// [forward_pass_idx][stage_idx] -> Realization
    realizations: Vec<Vec<Realization>>,
}

impl TrajectoryPool {
    fn new(num_forward_passes: usize, num_stages: usize, template: &Realization) -> Self {
        let realizations = (0..num_forward_passes)
            .map(|_| {
                (0..num_stages)
                    .map(|_| template.clone_with_capacity())
                    .collect()
            })
            .collect();
        Self { realizations }
    }
    
    fn get_mut(&mut self, fp_idx: usize, stage_idx: usize) -> &mut Realization {
        &mut self.realizations[fp_idx][stage_idx]
    }
}
```

#### Option B: In-Place Updates via Graph

The current architecture uses `DirectedGraph<Realization>` which already owns the data. Focus on eliminating clones when constructing **output results**, not during training.

**Strategy**: Defer result construction to post-training phase, or use references in result structs.

### Recommended Approach

**Option B** - The graph already owns realizations. Focus on:

1. **Eliminate clones in `to_forward_pass_result()`** - Use references or indices
2. **Lazy result construction** - Only clone when results are serialized/outputted
3. **Optional history** - Only clone realizations when `record_history` is enabled

---

## Acceptance Criteria

- [x] Result construction does not clone during training hot path
- [x] Cloning only occurs when:
  - Results are serialized to JSON/output
  - History recording is explicitly enabled
- [x] Training loop updates realizations in-place via graph
- [ ] DHAT shows no realization cloning during training iterations (requires manual verification)
- [x] All tests pass
- [x] No numerical result changes

### Implementation Notes

Audit revealed that realization cloning is already conditional:
- All `realization.clone()` calls in training are guarded by `if self.preserve_backward_detail`
- The `forward_detail_history` and `backward_detail_history` are only populated when enabled
- `RealizationData::from_realization()` is used for output API and is unavoidable

No code changes needed - the conditional cloning was already implemented correctly.

---

## Implementation Guide

### Suggested Approach

1. **Audit clone sites** in result construction:
   ```bash
   grep -n "\.clone()" src/sddp/mod.rs | head -50
   ```

2. **Categorize clones** by necessity:
   - Required for API (return owned data) → Keep, but defer
   - Internal copying → Eliminate

3. **Defer result construction**:
   ```rust
   // Instead of cloning during iteration
   pub fn train(...) -> TrainingResult {
       // ... training loop (no clones) ...
       
       // Clone only at the end for output
       self.finalize_results()
   }
   ```

4. **Use indices instead of owned data** in intermediate results:
   ```rust
   struct ForwardPassRef {
       trajectory_node_ids: Vec<usize>,  // References into graph
   }
   ```

5. **Lazy cloning for optional history**:
   ```rust
   if self.config.record_history {
       history.push(realization.clone());
   }
   ```

### Key Files to Modify

- `src/sddp/mod.rs`: Reduce cloning in result construction
- `src/subproblem.rs`: Ensure `Realization` updates are in-place

### High-Impact Clone Sites (from grep results)

| Line | Clone | Impact |
|------|-------|--------|
| 750 | `realization.clone()` | History recording - conditional |
| 871 | `realization.clone()` | History recording - conditional |
| 979 | `node.data.clone()` | Forward pass history - conditional |
| 1039 | `realization.clone()` | Backward pass history - conditional |
| 1248-1259 | Multiple field clones | Result construction - defer |

### Patterns to Follow

- See `CutStagingBuffer` for in-place update pattern
- See `VisitedStatePool` for preallocated data management

### Pitfalls to Avoid

- ⚠️ Don't break result serialization - some cloning is necessary for JSON output
- ⚠️ Watch for lifetime issues with references in result structs
- ⚠️ History recording is a feature - don't remove, make conditional

---

## Testing Requirements

### Unit Tests

- [ ] Test result construction with history disabled (no clones)
- [ ] Test result construction with history enabled (clones OK)

### Integration Tests

- [ ] Run example 05 with DHAT, verify reduced allocations
- [ ] Compare output JSON between old and new implementation

### Performance Tests

- [ ] Benchmark training with/without history recording
- [ ] DHAT profile shows minimal realization cloning during training

---

## Documentation Requirements

- [ ] Document `record_history` flag and its memory implications
- [ ] Add performance note about deferred result construction

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Requires careful refactoring of result construction paths while maintaining API compatibility.

---

## Definition of Done

- [ ] Clone sites audited and minimized
- [ ] History recording clones are conditional
- [ ] Result construction deferred to post-training
- [ ] DHAT confirms reduced allocations
- [ ] All tests passing
- [ ] Output results unchanged
- [ ] Code reviewed and merged
