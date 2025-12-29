# [T-048] Update Output Modules FCF Access

> **Epic**: [Epic 4: State Simplification](../../00-epic-overview.md)
> **Sprint**: [Sprint 2: FCF Graph Wrapper Removal](./00-sprint-overview.md)
> **Dependencies**: [T-046](./ticket-046-remove-mutex-from-fcf-type.md)
> **Blocks**: [T-049](./ticket-049-verify-fcf-refactoring.md)

## Files to Read Before Starting

- `docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md` - Appendix with output access points
- `src/output/csv/fcf.rs` - CSV export
- `src/output/parquet/fcf.rs` - Parquet export

---

## Context

### Background

The output modules iterate over FCF graph nodes to export cut and state data. They currently use `.lock().unwrap()` to access node data.

### Current State

```rust
// src/output/csv/fcf.rs or similar
for node in fcf_graph.nodes() {
    let fcf = node.data.lock().unwrap();  // ← Lock
    for cut in fcf.cut_pool.iter() {
        // Export cut data
    }
}
```

### Target State

```rust
for node in fcf_graph.nodes() {
    let fcf = &node.data;  // ← Direct reference
    for cut in fcf.cut_pool.iter() {
        // Export cut data
    }
}
```

---

## Specification

### Changes Required

1. **Remove `.lock().unwrap()` calls** in output modules
2. **Use direct references** to `FutureCostFunction`
3. **Update function signatures** if they take the graph as parameter

### Behavior

- **Identical behavior**: Same data exported
- **Read-only access**: Output modules only read FCF data

---

## Acceptance Criteria

- [ ] All `.lock().unwrap()` calls removed from output modules
- [ ] Direct references used for FCF access
- [ ] Compilation succeeds in output modules
- [ ] Output format unchanged

---

## Implementation Guide

### Suggested Approach

1. **Identify all lock sites in output**:
   ```bash
   grep -rn "\.lock()" src/output/ | grep -i fcf
   ```

2. **Update iteration patterns**:
   ```rust
   // Before
   for node in fcf_graph.nodes() {
       let fcf = node.data.lock().unwrap();
       // use fcf
   }
   
   // After
   for node in fcf_graph.nodes() {
       let fcf = &node.data;
       // use fcf
   }
   ```

3. **Update function signatures** if graph type is in signature:
   ```rust
   // Before
   pub fn export_fcf_csv(
       fcf_graph: &DirectedGraph<Arc<Mutex<FutureCostFunction>>>,
       ...
   )
   
   // After
   pub fn export_fcf_csv(
       fcf_graph: &DirectedGraph<FutureCostFunction>,
       ...
   )
   ```

### Key Files to Modify

- `src/output/csv/fcf.rs`
- `src/output/parquet/fcf.rs`
- Any other output modules that access FCF

### Patterns to Follow

Output modules are read-only, so all access should be `&`:

```rust
// Read-only iteration over cuts
for (node_id, node) in fcf_graph.nodes().iter() {
    let fcf: &FutureCostFunction = &node.data;
    
    for cut in fcf.cut_pool.active_cuts() {
        writer.write_record(&[
            cut.rhs.to_string(),
            // ...
        ])?;
    }
}
```

### Pitfalls to Avoid

- ⚠️ **Don't change output format**: Data should be identical
- ⚠️ **Maintain iteration order**: Same order as before

---

## Testing Requirements

### Build Verification

- [ ] `cargo build -j1` succeeds for output modules

### Output Verification (after T-049)

- [ ] CSV output identical to before
- [ ] Parquet output identical to before
- [ ] Golden tests pass

---

## Documentation Requirements

- [ ] Update doc comments if any reference locking

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple mechanical replacement, read-only access is straightforward

---

## Definition of Done

- [ ] All lock calls removed from output modules
- [ ] Direct references used
- [ ] Module compiles
- [ ] Ready for T-049 verification
