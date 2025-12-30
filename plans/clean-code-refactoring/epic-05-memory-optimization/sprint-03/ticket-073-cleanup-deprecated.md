# [T-073] Cleanup Deprecated CutData Path

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Pool Memory Model Optimization](./00-sprint-overview.md)
> **Dependencies**: [T-070](./ticket-070-remove-hashmap.md), [T-072](./ticket-072-migrate-state-pool.md)
> **Blocks**: [T-074](./ticket-074-final-validation.md)

## Files to Read Before Starting

- `src/fcf.rs:820-840` - `CutData` struct (deprecated)
- `src/state.rs:310-330` - `compute_cut_data()` (deprecated)
- `src/sddp/mod.rs:700-780` - `compute_cut_data_for_backward_step()` (deprecated)

---

## Context

### Background

With the new staging buffer path in production, the old allocating path is no longer used:
- `CutData` struct
- `CutData::from_refs()`
- `compute_cut_data()` method
- `compute_cut_data_for_backward_step()` method
- `compute_cuts_parallel()` coordinator method
- `select_cuts_batch()` coordinator method

These should be removed to:
1. Reduce code maintenance burden
2. Prevent accidental use of allocating path
3. Clarify the codebase

---

## Specification

### Items to Remove

| Item | Location | Status |
|------|----------|--------|
| `CutData` struct | `src/fcf.rs` | Remove |
| `CutData::from_refs()` | `src/fcf.rs` | Remove |
| `CutData::new()` | `src/fcf.rs` | Remove |
| `compute_cut_data()` | `src/state.rs` (trait) | Remove |
| `compute_cut_data()` | `src/state.rs` (impls) | Remove |
| `compute_cut_data_for_backward_step()` | `src/sddp/mod.rs` | Remove |
| `compute_cuts_parallel()` | `src/algorithm/coordinator.rs` | Remove |
| `select_cuts_batch()` | `src/algorithm/coordinator.rs` | Remove |
| `add_cuts_batch_from_data()` | `src/fcf.rs` | Remove |
| Trait methods | `src/algorithm/processor.rs` | Remove |

### Items to Keep (for reference/testing)

None - full removal. Tests should use the new path.

---

## Acceptance Criteria

- [ ] All deprecated items removed
- [ ] No references to `CutData` in production code
- [ ] No `#[deprecated]` attributes remaining (for removed items)
- [ ] Compile succeeds
- [ ] All tests pass (after updating any tests using old path)
- [ ] Golden tests pass

---

## Implementation Guide

### Step 1: Find all usages

```bash
grep -rn "CutData\|compute_cut_data\|from_refs\|select_cuts_batch\|compute_cuts_parallel" src/
```

Document each usage and whether it's production or test code.

### Step 2: Update tests using old path

Before removing, find tests that use the old path:

```bash
grep -rn "compute_cut_data\|CutData" src/ tests/
```

Update tests to use new path:
- `compute_cut_into_staging()` + `update_from_staging()`
- Or `compute_cuts_parallel_into_slots()`

### Step 3: Remove from trait

In `src/algorithm/processor.rs`:

```rust
// REMOVE:
fn compute_cuts_parallel(...) -> Result<Phase1Result, String>;
fn select_cuts_batch(...) -> Result<Phase2Result, String>;
```

### Step 4: Remove from coordinator

In `src/algorithm/coordinator.rs`:
- Remove `compute_cuts_parallel()` implementation
- Remove `select_cuts_batch()` implementation

### Step 5: Remove from State trait

In `src/state.rs`:

```rust
// REMOVE from trait:
fn compute_cut_data(...) -> CutData;

// REMOVE from StorageState impl
// REMOVE from StorageAndInflowState impl
```

### Step 6: Remove from handler

In `src/sddp/mod.rs`:

```rust
// REMOVE:
pub fn compute_cut_data_for_backward_step(...) -> Result<(CutData, ...), String>
```

### Step 7: Remove CutData struct

In `src/fcf.rs`:

```rust
// REMOVE entire struct and impl blocks:
pub struct CutData { ... }
impl CutData { ... }
```

### Step 8: Remove add_cuts_batch_from_data

In `src/fcf.rs`:

```rust
// REMOVE:
pub fn add_cuts_batch_from_data(...) { ... }
```

### Step 9: Clean up imports

Remove unused imports throughout the affected files.

### Step 10: Run tests

```bash
cargo build -j1
RUST_TEST_THREADS=1 cargo test -j1
```

Fix any remaining references.

---

## Testing Requirements

### Compile

Must compile without errors after removal.

### Tests

All 549+ tests must pass. Some tests may need updating.

### Golden Tests

```bash
./scripts/golden-tests.sh verify
```

---

## Pitfalls to Avoid

- ⚠️ **Test dependencies**: Some tests may directly test old path. Update or remove these tests.
- ⚠️ **Benchmarks**: Check if benchmarks use old path.
- ⚠️ **Documentation**: Remove references to old path in docs.

---

## Documentation Requirements

- [ ] Remove doc references to CutData
- [ ] Update CHANGELOG with removal note
- [ ] Update architecture docs if needed

---

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward removal once dependencies updated.

---

## Definition of Done

- [ ] All deprecated items removed
- [ ] No compilation errors
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] CHANGELOG updated
