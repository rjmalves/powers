# [T-045] Analyze and Document FCF Access Patterns

> **Epic**: [Epic 4: State Simplification](../../00-epic-overview.md)
> **Sprint**: [Sprint 2: FCF Graph Wrapper Removal](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-046](./ticket-046-remove-mutex-from-fcf-type.md)

## Files to Read Before Starting

- `docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md` - Analysis document with all findings
- `src/sddp/mod.rs` - Main FCF graph usage
- `src/algorithm/coordinator.rs` - Parallel handler coordination
- `src/output/csv/fcf.rs` - CSV export of FCF data
- `src/output/parquet/fcf.rs` - Parquet export of FCF data

---

## Context

### Background

The FCF (Future Cost Function) graph uses `Arc<Mutex<FutureCostFunction>>` as node data. Analysis has shown this Mutex is unnecessary because:
1. FCF is only modified in single-threaded Phase 2 (batch cut selection)
2. Parallel phases receive pre-cloned data, not live FCF references
3. Manual synchronization already ensures deterministic ordering

This ticket documents all FCF access points to enable safe refactoring.

### Current State

```rust
// src/sddp/mod.rs
pub struct SddpAlgorithm {
    pub future_cost_function_graph: graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    // ...
}
```

---

## Specification

### Outputs

A comprehensive document (can be added to this ticket or a separate markdown file) listing:

1. **All FCF write locations** with:
   - File and line number
   - Function name
   - Threading context (single-threaded/parallel)
   - Current lock pattern

2. **All FCF read locations** with:
   - File and line number
   - Function name
   - Threading context
   - Current lock pattern

3. **Confirmation** that the analysis in `FCF_GRAPH_ARCHITECTURE_ANALYSIS.md` is still accurate

---

## Acceptance Criteria

- [ ] All FCF `.lock()` call sites documented with file:line
- [ ] Each call site annotated with threading context
- [ ] Confirmed: no concurrent FCF access exists
- [ ] Confirmed: analysis document is up-to-date with current code
- [ ] Any discrepancies from analysis document noted

---

## Implementation Guide

### Suggested Approach

1. Search for all `.lock()` calls on FCF-related types:
   ```bash
   grep -rn "\.lock()" src/ | grep -i fcf
   grep -rn "future_cost_function" src/ | grep "\.lock()"
   ```

2. For each call site, document:
   - File and line
   - Enclosing function
   - Is the function called from a parallel context (`par_iter`, Rayon)?
   - Is the function single-threaded?

3. Cross-reference with analysis document appendix

4. Note any new call sites not in the analysis

### Key Patterns to Look For

```rust
// Pattern 1: Direct lock on node data
let mut fcf = node.data.lock().unwrap();

// Pattern 2: Lock in coordinator
let mut fcf_locked = parent_fcf_node.data.lock().unwrap();

// Pattern 3: Lock in output generation
for node in fcf_graph.nodes() {
    let fcf = node.data.lock().unwrap();
}
```

---

## Testing Requirements

### Verification

- [ ] No code changes in this ticket (documentation only)
- [ ] Build passes (no changes to verify)

---

## Documentation Requirements

- [ ] Update this ticket with findings OR create `docs/FCF_ACCESS_AUDIT.md`
- [ ] Confirm analysis document accuracy

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Grep-based analysis, cross-referencing existing document

---

## Definition of Done

- [ ] All FCF access points documented
- [ ] Threading context verified for each
- [ ] Analysis document confirmed accurate
- [ ] Ready for T-046 implementation
