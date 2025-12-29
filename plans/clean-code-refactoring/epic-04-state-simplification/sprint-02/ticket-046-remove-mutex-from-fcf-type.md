# [T-046] Remove Mutex from FCF Graph Type

> **Epic**: [Epic 4: State Simplification](../../00-epic-overview.md)
> **Sprint**: [Sprint 2: FCF Graph Wrapper Removal](./00-sprint-overview.md)
> **Dependencies**: [T-045](./ticket-045-analyze-fcf-access-patterns.md)
> **Blocks**: [T-047](./ticket-047-update-coordinator-fcf-access.md), [T-048](./ticket-048-update-output-fcf-access.md)

## Files to Read Before Starting

- `docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md` - Architecture analysis
- `src/sddp/mod.rs` - FCF graph type definition
- `src/graph.rs` - DirectedGraph implementation
- `src/fcf.rs` - FutureCostFunction type

---

## Context

### Background

The FCF graph currently uses `Arc<Mutex<FutureCostFunction>>` as node data. This ticket changes the type to just `FutureCostFunction`, removing unnecessary synchronization primitives.

### Current State

```rust
// src/sddp/mod.rs
pub struct SddpAlgorithm {
    pub future_cost_function_graph: graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
}
```

### Target State

```rust
// src/sddp/mod.rs
pub struct SddpAlgorithm {
    pub future_cost_function_graph: graph::DirectedGraph<fcf::FutureCostFunction>,
}
```

---

## Specification

### Changes Required

1. **Update type definition** in `SddpAlgorithm` struct
2. **Update graph construction** to not wrap in `Arc<Mutex<>>`
3. **Update trait bounds** if any depend on `Send + Sync` from Mutex

### Error Handling

- Compilation errors will guide remaining changes
- Each error indicates an access point needing update (T-047, T-048)

---

## Acceptance Criteria

- [ ] `SddpAlgorithm::future_cost_function_graph` type changed to `DirectedGraph<FutureCostFunction>`
- [ ] Graph construction updated (no `Arc::new(Mutex::new(...))`)
- [ ] Code compiles (may have errors in other modules—expected, fixed in T-047/T-048)

---

## Implementation Guide

### Suggested Approach

1. **Locate type definition**:
   ```bash
   grep -n "future_cost_function_graph" src/sddp/mod.rs | head -5
   ```

2. **Change the type**:
   ```rust
   // Before
   pub future_cost_function_graph: graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
   
   // After
   pub future_cost_function_graph: graph::DirectedGraph<fcf::FutureCostFunction>,
   ```

3. **Update construction** (likely in `SddpBuilder` or initialization):
   ```rust
   // Before
   let fcf = Arc::new(Mutex::new(FutureCostFunction::new(...)));
   graph.add_node(node_id, fcf);
   
   // After
   let fcf = FutureCostFunction::new(...);
   graph.add_node(node_id, fcf);
   ```

4. **Compile and note errors** for T-047/T-048

### Key Files to Modify

- `src/sddp/mod.rs` - Type definition
- `src/sddp/builder.rs` - Graph construction (if applicable)

### Pitfalls to Avoid

- ⚠️ Don't try to fix ALL compilation errors in this ticket
- ⚠️ Focus only on type definition and construction
- ⚠️ Access site updates are in T-047 and T-048

---

## Testing Requirements

### Build Verification

- [ ] `cargo build -j1` shows errors only in expected modules (coordinator, output)
- [ ] No errors in sddp/mod.rs after this change

### Note

Full test suite will not pass until T-047 and T-048 complete. This is expected.

---

## Documentation Requirements

- [ ] Update any doc comments on `future_cost_function_graph` field
- [ ] Note removed synchronization in commit message

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Simple type change with known compilation error cascade

---

## Definition of Done

- [ ] Type definition changed
- [ ] Graph construction updated
- [ ] Compilation errors isolated to expected modules
- [ ] Ready for T-047 and T-048
