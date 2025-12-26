# Sprint 01: Memory Module Cleanup and Cut Buffer Hardening

## Status: ✅ COMPLETE (2025-12-26)

## Goals

1. ✅ Remove all unused code from `src/memory` module
2. ✅ Harden `CutComputationBuffers` to enforce zero-allocation guarantees
3. ✅ Fix cut buffer initialization to use correct dimensions

## Tickets

| ID | Title | Points | Status |
|----|-------|--------|--------|
| TICKET-001c | Remove SizingInfo and related dead code | 2 | ✅ Complete |
| TICKET-002c | Remove DeepSizeEstimate trait and implementations | 2 | ✅ Complete |
| TICKET-003c | Harden CutComputationBuffers with capacity enforcement | 3 | ✅ Complete |
| TICKET-004c | Fix cut buffer initialization with correct dimensions | 2 | ✅ Complete |
| TICKET-005c | Validation and testing | 1 | ✅ Complete |

## Dependencies

- **From Previous Epic**: Epic 1b complete (deterministic slots) ✅
- **To Next Epic**: Epic 2 (cleaner memory module)

## Risks

| Risk | Mitigation | Outcome |
|------|------------|---------|
| Removed code might be used somewhere | Grep for all usages before removal | ✅ No issues |
| Panic in cut buffers too strict | Add clear error message with debugging info | ✅ Clear messages |
| Worker thread initialization timing | Use `rayon::broadcast()` before training | ✅ Working |

## Definition of Done

- [x] All tickets complete
- [x] `src/memory` contains only `CutComputationBuffers` code (~437 lines)
- [x] No `SizingInfo`, `DeepSizeEstimate`, `Buffer<T>`, `BufferPool<T>`, `ThreadLocalBuffers`
- [x] Cut buffers panic on capacity overflow
- [x] All examples pass with identical results
- [x] No clippy warnings
- [x] Code reviewed and merged

## Summary of Changes

### Files Removed
- `src/memory/sizing.rs` (1,574 lines)
- `src/memory/deep_sizing.rs` (474 lines)

### Files Modified
- `src/memory/mod.rs` - Reduced to minimal re-exports
- `src/memory/buffers.rs` - Reduced to CutComputationBuffers only
- `src/cut.rs` - Removed DeepSizeEstimate impl
- `src/fcf.rs` - Removed DeepSizeEstimate impl
- `src/state.rs` - Removed DeepSizeEstimate impl and tests
- `src/sddp/mod.rs` - Added rayon::broadcast() and strict consistency validation

### Key Improvements
1. **87% reduction in memory module size** (3,424 → 437 lines)
2. **Capacity enforcement** prevents silent reallocations in hot path
3. **rayon::broadcast()** ensures all worker threads initialized
4. **Strict validation** - `train()` asserts `NodeData.num_scenarios == SAA.num_branchings`

### Architecture: num_scenarios Data Flow

```
graph.json (input)
    └── GraphNodeInput.num_scenarios
        └── NodeData.num_scenarios (stored in node_data_graph)
            └── Recourse::generate_sddp_noises() uses node.data.num_scenarios
                └── ScenarioTree.stage_scenarios[].num_branchings (SAA)
```

**Invariant:** `NodeData.num_scenarios == ScenarioTree.stage_scenarios[stage_id].num_branchings`

**Source of Truth:** The SAA (`ScenarioTree`) is the runtime source of truth since it contains
the actual scenarios that will be solved. However, it MUST be consistent with `NodeData.num_scenarios`
since production code generates the SAA from node data.

**Validation:** `train()` validates this invariant and panics with a clear error message if violated.
This catches bugs in test setup or input file generation immediately, rather than masking them
with defensive programming.

### Test Fixes
- `test_train_with_default_system`: Fixed nodes to have `num_scenarios=3` matching SAA
- `test_simulate_with_default_system`: Fixed nodes to have `num_scenarios=3` matching SAA
- `test_backward_with_default_system`: Already consistent (`num_scenarios=1`, SAA has 1 branching)
