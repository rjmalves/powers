# PAR-V2-001: Create StorageAndInflowState Struct

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 1 (Foundation)  
**Story Points**: 3  
**Priority**: 🔥 Critical Path - START HERE  
**Status**: 🔵 Not Started

---

## Context

This is the **first ticket** in the PAR completion epic and establishes the foundational data structure for managing both storage levels and lagged inflow history. 

The current `StorageState` only tracks storage levels. For PAR models, we need to track both:
1. Storage levels (existing functionality)
2. Lagged inflow values (new: for AR equation evaluation)

This struct will extend the existing `State` trait pattern and serve as the foundation for all subsequent PAR integration work.

**Why This Matters**: PAR models require maintaining lag history to evaluate AR equations. Without this state management, we cannot implement PAR dynamics correctly in the SDDP algorithm.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 1, Section 1.1
- Template: `src/state.rs` → `StorageState` implementation
- Theory: Plan → "The State-Space PAR Formulation" section

---

## Acceptance Criteria

- [ ] `StorageAndInflowState` struct defined in `src/state.rs`
- [ ] Struct contains fields for storage state and lag buffers
- [ ] Struct uses composition (wraps `StorageState`) for storage functionality
- [ ] Lag buffers use `VecDeque<f64>` for efficient circular buffer operations
- [ ] AR parameter storage organized by (hydro_id, season_id)
- [ ] Struct is `Clone` + `Debug` for state management
- [ ] No duplicate struct definitions (verified against codebase)
- [ ] Compiles without errors or warnings

---

## Tasks

### Implementation

- [ ] Define `StorageAndInflowState` struct with fields:
  - `storage: StorageState` (composition for storage functionality)
  - `lag_buffers: HashMap<usize, VecDeque<f64>>` (hydro_id → lag values)
  - `ar_orders: HashMap<usize, usize>` (hydro_id → AR order p)
  - `ar_params: HashMap<(usize, usize), Vec<f64>>` (hydro_id, season_id → [φ_1, ..., φ_p, μ, σ])
  - `is_par_hydro: Vec<bool>` (flags indicating PAR hydros)

- [ ] Add derive macros: `#[derive(Debug, Clone)]`

- [ ] Add comprehensive doc comments:
  - Module-level docs explaining state-space augmentation
  - Field-level docs explaining each field's purpose
  - Examples showing typical usage pattern

- [ ] Verify no struct name conflicts with existing code

### Testing

- [ ] Compile test: Verify struct compiles without errors
- [ ] Documentation test: Run `cargo doc` and verify docs render correctly
- [ ] Size test: Verify struct size is reasonable (use `std::mem::size_of`)

### Documentation

- [ ] Add doc comment with PAR state-space formulation explanation
- [ ] Document state dimension: n (storage) → n + n*p (storage + lags)
- [ ] Add example showing struct initialization skeleton
- [ ] Reference the State trait and its role

---

## Technical Notes

### Design Decisions

**Composition over Inheritance**:
```rust
pub struct StorageAndInflowState {
    storage: StorageState,  // Delegate storage operations
    // ... PAR-specific fields
}
```

**Why**: Reuses all existing `StorageState` functionality without code duplication. Methods can delegate to `self.storage` for storage operations.

**Lag Buffer Structure**:
- Use `VecDeque<f64>` for O(1) front/back operations
- Key by `hydro_id` - only PAR hydros need lag buffers
- Buffer size = AR order p (fixed capacity)

**AR Parameter Packing**:
```rust
// Packed as: [φ_1, φ_2, ..., φ_p, μ, σ]
ar_params: HashMap<(usize, usize), Vec<f64>>
```

**Why**: Single lookup gets all parameters for a (hydro, season) pair. Avoids multiple hash lookups in hot paths.

### Memory Layout Considerations

For n hydros with PAR orders [p_1, p_2, ..., p_n]:
- Storage state: `O(n)` (existing)
- Lag buffers: `O(Σ p_i)` (sum of AR orders)
- AR params: `O(n * num_seasons * max(p))` (typically small)

**Typical case**: 10 hydros, PAR(3), 12 seasons → ~360 f64 values → ~3KB

### Edge Cases to Consider

1. **Mixed PAR/non-PAR hydros**: Some hydros PAR, others naive
   - Solution: `is_par_hydro` flag vector + sparse lag buffer storage

2. **Variable AR orders**: Different hydros have different p values
   - Solution: `ar_orders` map + variable-sized lag buffers

3. **Seasonal parameter variation**: φ, μ, σ change by season
   - Solution: Two-level key `(hydro_id, season_id)` in params map

### Integration Points

**Delegates to `StorageState`**:
- Storage level management
- Initial storage handling
- Storage-related State trait methods

**New PAR-specific logic**:
- Lag buffer management
- AR parameter lookups
- Lag state variable tracking

---

## Dependencies

### Blocked By
- None - this is the starting point

### Blocks
- PAR-V2-002 (Lag buffer management)
- PAR-V2-003 (Constructor implementation)
- PAR-V2-004 (State trait methods)

### Related
- Existing: `StorageState` in `src/state.rs` (template)
- Existing: `State` trait in `src/state.rs` (interface to implement)

---

## Implementation Hints

### Start with the Struct Definition

```rust
/// State that manages both storage levels and lagged inflows for PAR models
/// 
/// This extends StorageState by adding circular buffers for AR lag history.
/// Each hydro with PAR inflow gets p lag state variables (p = AR order).
///
/// # State Space Dimension
/// 
/// - Without PAR: n (one storage per hydro)
/// - With PAR(p): n + n*p (storage + p lags per hydro)
///
/// # Subproblem Variables
///
/// For each hydro with PAR(p) inflow:
/// - `stored_volume[i]`: Storage level (existing)
/// - `lag_inflow[i][0..p]`: Lagged inflow values (new)
///
/// # Cut Generation
///
/// Cut coefficients extracted from dual variables of:
/// - Storage balance constraints → water value (existing)
/// - AR lag linking constraints → AR shadow price (new)
#[derive(Debug, Clone)]
pub struct StorageAndInflowState {
    /// Storage state (delegate all storage operations to this)
    storage: StorageState,
    
    /// Lag history for each hydro with PAR inflow
    /// Key: hydro_id, Value: circular buffer of past inflows [t-1, t-2, ..., t-p]
    lag_buffers: HashMap<usize, VecDeque<f64>>,
    
    /// AR orders for each hydro
    /// Key: hydro_id, Value: AR order p
    ar_orders: HashMap<usize, usize>,
    
    /// AR coefficients by hydro and season
    /// Key: (hydro_id, season_id), Value: [φ_1, φ_2, ..., φ_p, μ, σ]
    ar_params: HashMap<(usize, usize), Vec<f64>>,
    
    /// Flag for each hydro indicating if it uses PAR
    is_par_hydro: Vec<bool>,
}
```

### Verify No Name Conflicts

```bash
# Check for existing StorageAndInflowState
cd /home/rogerio/git/powers
grep -r "StorageAndInflowState" src/
# Should return nothing (besides this new definition)
```

### Check Struct Size

```rust
#[test]
fn test_storage_inflow_state_size() {
    use std::mem::size_of;
    
    // Verify struct size is reasonable
    let size = size_of::<StorageAndInflowState>();
    println!("StorageAndInflowState size: {} bytes", size);
    
    // Should be dominated by HashMap overhead, not huge
    assert!(size < 1024, "Struct size unexpectedly large: {}", size);
}
```

---

## Estimated Effort

**3 story points** (1 day)

**Confidence**: High

**Breakdown**:
- Struct definition: 2 hours
- Documentation: 2 hours
- Testing: 2 hours
- Review: 2 hours

---

## Definition of Done

- [x] Struct definition complete with all fields
- [x] Comprehensive doc comments added
- [x] Derives `Clone` and `Debug`
- [x] Compiles without errors or warnings
- [x] `cargo doc` generates correct documentation
- [x] No naming conflicts with existing code
- [x] Code reviewed and approved
- [x] Merged to branch

---

## Notes

- This ticket is intentionally small and focused - just the struct definition
- Constructor and methods come in subsequent tickets
- This establishes the data model for all subsequent PAR work
- Review the `StorageState` implementation as a template for patterns to follow

**Next Ticket**: PAR-V2-002 (Lag buffer management methods)
