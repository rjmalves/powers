# HiGHS Warm-Start Investigation

> **Sprint**: Epic 5, Sprint 6
> **Ticket**: T-087
> **Date**: 2025-12-30

## Executive Summary

Investigation of HiGHS warm-start capabilities to reduce `HFactor::setupGeneral` allocations
which account for **44.9% of total heap allocations** (39.6 GB) during SDDP training.

### Key Findings

1. **Warm-starting is already implemented** via `Model::try_set_basis()`
2. **HFactor allocations are unavoidable** with current HiGHS architecture
3. **Batch bounds API provides significant FFI overhead reduction** (implemented in T-088)
4. **Current HiGHS options are already optimized** (presolve off, scaling off, fixed strategies)
5. **`reuse_forward_basis` may trigger unnecessary allocations** (disabled for testing)

## Investigation Details

### 1. HiGHS Warm-Start API

HiGHS supports warm-starting through:

1. **Basis setting**: `Highs_setBasis()` - Set column/row basis status before solve
2. **Solution setting**: Not typically useful for LP (basis is sufficient)

Our codebase already uses `Model::try_set_basis()` in:
- `src/solver.rs`: `set_basis()` and `try_set_basis()` methods
- SDDP forward passes for stage-to-stage warm-starting

### 2. Why HFactor Allocations Still Occur

From HiGHS source analysis (`HFactor.cpp`), `setupGeneral` allocations occur because:

1. **LU factorization rebuilding**: Even with warm-start, HiGHS may rebuild factorization
2. **Bound changes invalidate basis**: When row bounds change, basis may become invalid
3. **Internal data structures**: HiGHS allocates working vectors each solve

**Critical insight**: These are internal HiGHS allocations, not controllable via API options.

### 3. HiGHS Options Already Optimized

Current `set_default_solver_options()` in `src/subproblem.rs` already applies:

| Option | Value | Effect |
|--------|-------|--------|
| `presolve` | `"off"` | Eliminates 3.2% of allocations |
| `parallel` | `"off"` | No thread pool allocations |
| `threads` | `1` | Single-threaded (Rayon handles parallelism) |
| `simplex_scale_strategy` | `0` | No scaling vector allocation |
| `simplex_update_limit` | `5000` | Stable refactorization memory |
| `simplex_dual_edge_weight_strategy` | `-1` | Fixed edge weight strategy |

### 4. Batch Bounds Optimization (T-088)

While HFactor allocations are unavoidable, we can reduce FFI overhead:

**Before**: 3 million individual `Highs_changeRowBounds()` calls
**After**: ~60 batch `Highs_changeRowsBoundsBySet()` calls per stage

This doesn't eliminate HiGHS internal allocations but:
- Reduces Rust-to-C FFI call overhead
- Reduces HiGHS bounds-update processing overhead
- Improves cache locality

### 5. Forward Basis Reuse Analysis (⚠️ SUSPECTED ALLOCATION SOURCE)

The `reuse_forward_basis()` function in `src/sddp/mod.rs` was identified as a potential
source of unnecessary allocations and has been **temporarily disabled** for testing.

#### What `reuse_forward_basis` Does

```rust
fn reuse_forward_basis(
    subproblem: &mut Subproblem,
    node_forward_realization: &Realization,
) -> Result<(), String> {
    // 1. Extracts basis from forward pass realization
    // 2. Adjusts row status vector size if model has more/fewer rows (cuts added)
    // 3. Calls model.set_basis() to hot-start the backward branching solve
}
```

#### Why It May Cause Excessive Allocations

From HiGHS source analysis (`Highs.cpp`, `HighsSolution.cpp`):

1. **Alien Basis Handling**: When `setBasis()` receives a basis with mismatched dimensions
   (which happens when cuts are added), HiGHS marks it as "alien" and triggers:
   
   ```cpp
   // From Highs::setBasis() in Highs.cpp:2594
   HighsStatus return_status = formSimplexLpBasisAndFactor(solver_object);
   ```

2. **`formSimplexLpBasisAndFactor`** performs:
   - LP scaling consideration (allocation)
   - `accommodateAlienBasis()` for rank-deficient basis repair
   - `ekk_instance.moveLp()` - moves LP to EKK (potential allocation)
   - `ekk_instance.setBasis()` - sets basis in EKK
   - `initialiseSimplexLpBasisAndFactor()` - **creates new factorization** (main allocation source)

3. **`accommodateAlienBasis`** (called for non-matching bases):
   - Allocates `basic_index` vector
   - May need to complete basis with logical variables
   - Triggers full factorization rebuild

4. **Key Insight**: Every backward branching solve with `reuse_forward_basis()`:
   - Passes a basis from forward pass (which has fewer rows - no backward cuts yet)
   - Forces HiGHS to treat it as "alien" and rebuild factorization
   - This happens **for every branching at every stage**, potentially 60×20×8 = 9,600 times per iteration

#### Current Status

The function has been **commented out** in `solve_all_branchings()`:

```rust
for branching_id in 0..num_branchings {
    // Note: reuse_forward_basis temporarily disabled pending investigation
    // reuse_forward_basis(
    //     &mut subproblem_node.data,
    //     _node_forward_realization,
    // )?;
    
    let step_timing = step(...);
}
```

#### Testing Recommendation

Run DHAT profiling with and without `reuse_forward_basis`:

```bash
# Without (current state)
valgrind --tool=dhat ./target/release/powers run examples/05-large-scale-brazilian

# With (re-enable function)
# ... then run DHAT again
```

Compare `HFactor::setupGeneral` allocation counts. If significantly lower without
`reuse_forward_basis`, the hypothesis is confirmed.

#### Potential Solutions (If Confirmed)

1. **Option A**: Keep disabled - Let HiGHS use logical basis for backward passes
2. **Option B**: Only reuse basis when row counts match exactly
3. **Option C**: Use `Highs_setLogicalBasis()` instead for backward passes
4. **Option D**: Investigate HiGHS `putIterate()` API for more efficient warm-start

## Recommendations

### Already Implemented
1. ✅ Basis warm-starting (`try_set_basis`)
2. ✅ Presolve disabled
3. ✅ Threading disabled
4. ✅ Scaling disabled
5. ✅ Fixed memory strategies

### Implemented in Sprint 6
1. ✅ Batch bounds API (`change_rows_bounds_batch`)
2. ✅ Integrated batch bounds in constraint updates
3. ✅ Disabled `reuse_forward_basis` for testing

### Future Considerations
1. **LP model caching**: For identical problem structures, could cache factorization
2. **HiGHS modifications**: Would require changes to HiGHS source (out of scope)
3. **Alternative solvers**: GLPK, CLP may have different allocation patterns
4. **⚠️ DHAT verification**: Compare allocations with/without `reuse_forward_basis`

## Conclusion

The investigation confirms that:

1. **Warm-starting is already properly implemented** in our codebase
2. ~~**HFactor allocations are inherent to HiGHS** and cannot be eliminated via API~~
3. **Current HiGHS options are already optimized** for our use case
4. **Batch bounds API** (T-088) is the primary actionable optimization
5. **`reuse_forward_basis` may be counterproductive** - the "alien basis" handling in HiGHS
   may cause more allocations than it saves by forcing factorization rebuilds

The 44.9% allocation from `HFactor::setupGeneral` is a characteristic of HiGHS's
internal implementation. The `reuse_forward_basis` function may be exacerbating this
by triggering alien basis handling on every backward branching solve. Testing with
this function disabled should be performed to validate this hypothesis.

---

## 🎉 DHAT Verification Results (T-093)

**The hypothesis was validated!** Disabling `reuse_forward_basis()` achieved:

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| **HFactor::setupGeneral** | 39.58 GB | 2.00 GB | **95.0%** |
| Total bytes allocated | 88.19 GB | 45.43 GB | **48.5%** |
| Total allocation blocks | 159.0M | 43.4M | **72.7%** |

**Conclusion (Revised)**: The initial assessment that "HFactor allocations are inherent to HiGHS" was **incorrect**. The allocations were caused by our misuse of the basis API. When `setBasis()` receives a basis with mismatched row counts, HiGHS treats it as "alien" and triggers full factorization rebuilds.

**Recommendation**: Keep `reuse_forward_basis()` **permanently disabled** or remove the code entirely. Consider documenting when basis reuse IS appropriate (only when row counts match exactly).

**Status (Sprint 7)**: The `reuse_forward_basis()` function has been **permanently removed** from the codebase (T-094).

Full analysis: [DHAT_SPRINT6_ANALYSIS.md](./DHAT_SPRINT6_ANALYSIS.md)

---

## Basis Reuse Guidelines

This section documents when HiGHS basis reuse is appropriate and when it causes problems.
These guidelines were developed from Sprint 6 investigation and validated with DHAT profiling.

### When Basis Reuse IS Appropriate

Basis reuse via `Model::set_basis()` is beneficial when:

1. **Model dimensions are unchanged**: Same number of rows and columns
2. **Only RHS/bounds changed**: Objective coefficients, constraint bounds, variable bounds
3. **Same constraint structure**: No rows added, removed, or reordered

Example valid use case:
```rust
// Same model, different RHS values
model.change_rows_bounds(row, new_lb, new_ub);
// Basis from previous solve is still valid
model.solve();  // Will warm-start automatically
```

### When Basis Reuse Causes Problems

**DO NOT** use `set_basis()` when:

1. **Row count changed**: Cuts added/removed between solves
2. **Column count changed**: Variables added/removed
3. **Constraint structure changed**: Different sparsity pattern

What happens when you violate these rules:
- HiGHS detects dimension mismatch
- Basis marked as "alien"
- Triggers `formSimplexLpBasisAndFactor()`
- Full factorization rebuild (defeats warm-start purpose)
- Allocates 39+ GB instead of 2 GB for large SDDP runs

### SDDP-Specific Guidance

In SDDP training:

| Scenario | Basis Reuse Works? | Reason |
|----------|-------------------|--------|
| Between stages (same node) | **Maybe** | Only if no cuts added between stages |
| Between forward and backward passes | **NO** | Cuts added between passes |
| Between iterations | **NO** | Model structure evolves as cuts accumulate |
| Same stage, same cuts | **YES** | Model dimensions unchanged |

### Validation Evidence

Sprint 6 DHAT profiling confirmed:
- With `reuse_forward_basis()`: 39.58 GB HFactor allocations
- Without `reuse_forward_basis()`: 2.00 GB HFactor allocations
- **95% reduction** by NOT using mismatched basis

### Code Patterns

**Incorrect (causes alien basis handling):**
```rust
// DON'T DO THIS - basis has different row count than current model
let forward_basis = forward_realization.basis.rows().to_vec();
let mut adjusted_basis = forward_basis.clone();
adjusted_basis.resize(model.num_rows(), 0);  // Padding doesn't help!
model.set_basis(Some(&cols), Some(&adjusted_basis));  // Triggers alien basis
```

**Correct (let HiGHS use its own basis):**
```rust
// Let HiGHS start fresh when model structure has changed
// No set_basis() call needed - HiGHS will use logical basis
model.solve();  // HiGHS builds optimal basis internally
```

---

## Appendix A: HiGHS Options Reference

```rust
// Current optimized configuration in set_default_solver_options()
model.set_option("presolve", "off");
model.set_option("solver", "simplex");
model.set_option("simplex_strategy", 1);
model.set_option("simplex_update_limit", 5000);
model.set_option("simplex_price_strategy", 1);
model.set_option("simplex_scale_strategy", 0);
model.set_option("parallel", "off");
model.set_option("threads", 1);
model.set_option("simplex_dual_edge_weight_strategy", -1);
model.set_option("simplex_primal_edge_weight_strategy", -1);
```

## Appendix B: HiGHS Alien Basis Code Path

From `Highs.cpp:2563-2628`:

```cpp
HighsStatus Highs::setBasis(const HighsBasis& basis, const std::string& origin) {
  if (basis.alien) {
    // An alien basis needs to be checked properly, since it may be
    // singular, or even incomplete.
    if (model_.lp_.num_row_ == 0) {
      // Special case...
    } else {
      // Check whether a new basis can be defined
      if (!isBasisRightSize(model_.lp_, basis)) {
        // Error: size mismatch
        return HighsStatus::kError;
      }
      HighsBasis modifiable_basis = basis;
      modifiable_basis.was_alien = true;
      HighsLpSolverObject solver_object(...);
      // THIS IS THE ALLOCATION TRIGGER:
      HighsStatus return_status = formSimplexLpBasisAndFactor(solver_object);
      if (return_status != HighsStatus::kOk) return HighsStatus::kError;
      basis_ = std::move(modifiable_basis);
    }
  }
  // ...
}
```

The key insight is that `formSimplexLpBasisAndFactor()` is called for alien bases,
which triggers the full factorization machinery including `HFactor::setupGeneral`.

