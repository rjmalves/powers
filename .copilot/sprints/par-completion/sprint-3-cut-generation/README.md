# Sprint 3: Cut Generation (Week 4-5)

**Goal**: Extend Benders cut generation to include lag state dual variables and coefficients.

**Duration**: 2 weeks  
**Total Story Points**: 13  
**Risk Level**: 🟠 HIGH

---

## Overview

With AR dynamics in the LP (Sprint 2), we now extract dual variables from AR constraints and include them in Benders cuts. This completes the Bellman recursion for the extended state space.

**Key Concept**: Cuts must include coefficients for lag states to correctly approximate future costs as a function of storage AND lags.

---

## Sprint Tickets

| ID | Title | Points | Status | Priority |
|----|-------|--------|--------|----------|
| PAR-V2-010 | Extract dual variables from lag states | 3 | 🔵 Not Started | 🔥🔥 Critical |
| PAR-V2-011 | Include lag duals in cut generation | 5 | 🔵 Not Started | 🔥🔥 Critical |
| PAR-V2-012 | Integrate PAR into backward pass | 3 | 🔵 Not Started | 🔥 High |
| PAR-V2-013 | Integrate PAR into cut pool | 2 | 🔵 Not Started | Medium |

---

## Critical Path

```
PAR-V2-010 (Extract Duals) → PAR-V2-011 (Lag in Cuts) → PAR-V2-012 (Backward Pass)
                                                              ↓
                                                        PAR-V2-013 (Cut Pool)
```

---

## Mathematical Foundation

**Standard Cut**:
```
θ_{t+1} ≥ α + Σ β_i · storage_i
```

**Extended Cut (PAR)**:
```
θ_{t+1} ≥ α + Σ β_storage_i · storage_i + Σ β_lag_{j,k} · lag_{j,k}
```

Where:
- `β_lag_{j,k}` = dual variable from AR constraint for hydro j, lag k
- `α` = intercept adjusted for lag state contributions

---

## Key Challenges

### Challenge 1: Dual Extraction

AR constraint has form:
```
inflow - σ·(Σ φ_k·lag[k]) = μ + ε
```

**Question**: How do individual lag dual variables relate to the single AR constraint dual?

**Answer**: The dual π_AR of the AR constraint affects all lags. Individual lag sensitivities are:
```
∂f/∂lag[k] = -σ·φ_k · π_AR
```

### Challenge 2: Intercept Calculation

Must subtract lag contributions:
```
α = f(x*) - Σ π_storage_i · storage_i* - Σ π_lag_{j,k} · lag_{j,k}*
```

### Challenge 3: Cut Evaluation

When evaluating cut at new state, must access lag values:
```rust
if let Some(storage_inflow_state) = state.as_any().downcast_ref::<StorageAndInflowState>() {
    // Access lag values
}
```

---

## Success Criteria

- [ ] Lag duals extracted from AR constraints
- [ ] Cut struct extended with lag coefficients
- [ ] Cut generation includes lag terms
- [ ] Cut evaluation includes lag terms
- [ ] Backward pass integrates extended cuts
- [ ] Cut pool handles extended cuts
- [ ] Numerical validation: cuts are valid lower bounds

---

## Validation Tests

- [ ] **Dual extraction**: Verify duals are non-zero and meaningful
- [ ] **Cut structure**: Verify cut includes lag coefficients for all PAR hydros
- [ ] **Cut evaluation**: Hand-calculate cut value, verify matches code
- [ ] **Lower bound property**: Cut value ≤ true future cost
- [ ] **Tightness**: Cut is tight at state where it was generated

---

## Dependencies

**Blocked By**: Sprint 2 (LP Integration) - needs AR constraints and duals

**Blocks**: Sprint 4 (Forward Pass) - forward pass needs cuts with lags

---

## Files to Edit

- `src/cut.rs` - Extend Cut struct
- `src/sddp/backward_pass.rs` - Extract lag duals, generate extended cuts
- `src/sddp/cut_pool.rs` - Handle extended cuts (if needed)
- `tests/test_cut.rs` - Add PAR cut tests
- `tests/test_sddp_algorithm.rs` - Validate cut generation

---

## Notes

- This sprint is where the Bellman equation gets extended
- Cuts with lag coefficients are what propagate lag state values backward
- Getting this right is essential for convergence
- Test cut validity property rigorously
- Compare cut coefficients with dual variables to verify correctness

**Key Insight**: If Sprint 2 was correct (AR constraints), this sprint should be straightforward - we're just extracting duals and using them in cuts.
