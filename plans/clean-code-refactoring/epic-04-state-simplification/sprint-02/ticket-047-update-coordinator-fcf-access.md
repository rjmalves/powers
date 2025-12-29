# [T-047] Update Coordinator FCF Access

> **Epic**: [Epic 4: State Simplification](../../00-epic-overview.md)
> **Sprint**: [Sprint 2: FCF Graph Wrapper Removal](./00-sprint-overview.md)
> **Dependencies**: [T-046](./ticket-046-remove-mutex-from-fcf-type.md)
> **Blocks**: [T-049](./ticket-049-verify-fcf-refactoring.md)

## Files to Read Before Starting

- `docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md` - Appendix with all access points
- `src/algorithm/coordinator.rs` - Main file to modify
- `src/algorithm/backward_pass.rs` - May need FCF parameter updates

---

## Context

### Background

The coordinator's `select_cuts_batch` method accesses FCF via `.lock().unwrap()`. With Mutex removed (T-046), we need to change to direct `&mut` access.

### Current State

```rust
// src/algorithm/coordinator.rs
fn select_cuts_batch(...) -> Result<Phase2Result, String> {
    let mut fcf_locked = parent_fcf_node.data.lock().unwrap();  // ← Lock
    fcf_locked.add_cuts_batch_from_data(cut_data, enable_cut_selection)
}
```

### Target State

```rust
fn select_cuts_batch(
    &mut self,
    fcf_node: &mut graph::Node<FutureCostFunction>,  // ← Direct mutable reference
    ...
) -> Result<Phase2Result, String> {
    fcf_node.data.add_cuts_batch_from_data(cut_data, enable_cut_selection)
}
```

---

## Specification

### Changes Required

1. **Update function signatures** to take `&mut FutureCostFunction` or `&mut Node<FutureCostFunction>`
2. **Remove `.lock().unwrap()` calls**
3. **Update callers** to pass mutable references
4. **Ensure borrow checker is satisfied** (may require restructuring)

### Behavior

- **Identical behavior**: Same functions called, same data modified
- **No algorithm changes**: Only access mechanism changes

---

## Acceptance Criteria

- [ ] All `.lock().unwrap()` calls removed from `coordinator.rs`
- [ ] FCF access uses direct `&mut` references
- [ ] Compilation succeeds in coordinator module
- [ ] Trait signatures updated if needed

---

## Implementation Guide

### Suggested Approach

1. **Identify all lock sites in coordinator**:
   ```bash
   grep -n "\.lock()" src/algorithm/coordinator.rs
   ```

2. **Update `select_cuts_batch` signature**:
   ```rust
   // Before
   fn select_cuts_batch(
       &mut self,
       cut_data: Vec<CutData>,
       stage_ctx: &BackwardStageContext,
       fcf_graph: &DirectedGraph<Arc<Mutex<FutureCostFunction>>>,
   ) -> Result<Phase2Result, String>
   
   // After
   fn select_cuts_batch(
       &mut self,
       cut_data: Vec<CutData>,
       stage_ctx: &BackwardStageContext,
       fcf_node: &mut FutureCostFunction,  // Direct reference
   ) -> Result<Phase2Result, String>
   ```

3. **Update method body**:
   ```rust
   // Before
   let mut fcf_locked = parent_fcf_node.data.lock().unwrap();
   fcf_locked.add_cuts_batch_from_data(...)
   
   // After
   fcf_node.add_cuts_batch_from_data(...)
   ```

4. **Update callers** (likely in `backward_pass.rs` or `sddp/mod.rs`)

### Key Files to Modify

- `src/algorithm/coordinator.rs` - Main changes
- `src/algorithm/backward_pass.rs` - Caller updates
- `src/sddp/mod.rs` - If backward pass calls originate here

### Patterns to Follow

```rust
// Pattern: Pass mutable reference through the call chain
fn backward_stage<P: BackwardStageProcessor>(
    processor: &mut P,
    fcf_node: &mut FutureCostFunction,  // Passed down
    ...
) {
    processor.select_cuts_batch(cut_data, stage_ctx, fcf_node)?;
}
```

### Pitfalls to Avoid

- ⚠️ **Borrow checker conflicts**: If you need both `&mut fcf_node` and `&mut something_else` from same struct, you may need to restructure
- ⚠️ **Don't introduce new Mutex/RefCell**: Use Rust's ownership system
- ⚠️ **Preserve determinism**: Order of operations must remain identical

---

## Testing Requirements

### Build Verification

- [ ] `cargo build -j1` succeeds for coordinator module

### Integration Verification

- [ ] Combined with T-048, full build should succeed
- [ ] Golden tests pass (run after T-049)

---

## Documentation Requirements

- [ ] Update doc comments on modified functions
- [ ] Remove references to locking in comments

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: May require borrow checker restructuring depending on current code shape

---

## Definition of Done

- [ ] All lock calls removed from coordinator
- [ ] Direct `&mut` references used
- [ ] Module compiles
- [ ] Ready for T-049 verification
