# Sprint 2: Constraint Extraction

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Duration**: 1.5 weeks (overlaps with Sprint 1 end)
> **Status**: ✅ Complete

---

## ⚠️ CRITICAL REMINDER

This sprint extracts constraint building from `subproblem.rs`. **The algorithm logic must remain unchanged.**

Run golden tests after EVERY extraction. If any test fails, **STOP immediately** and investigate.

---

## Goals

1. **Primary**: Extract constraint generation into `src/model/constraints/`
2. **Primary**: Organize by constraint type
3. **Secondary**: Reduce complexity in `subproblem.rs`
4. **Validation**: Maintain bit-for-bit identical outputs

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-012](./ticket-012-constraints-module-structure.md) | Create constraints module structure | 2 | Yes | Sprint 1 | ✅ |
| [T-013](./ticket-013-hydro-balance-constraints.md) | Extract hydro balance constraints | 3 | Yes | T-012 | ✅ |
| [T-014](./ticket-014-bus-balance-constraints.md) | Extract bus balance constraints | 3 | Yes | T-012 | ✅ |
| [T-015](./ticket-015-ar-dynamics-constraints.md) | Extract AR dynamics constraints | 3 | Yes | T-012 | ✅ |
| [T-016](./ticket-016-bound-constraints.md) | Extract bound constraints (optional) | 2 | Yes | T-012 | ⏭️ Skipped |
| [T-017](./ticket-017-refactor-subproblem-facade.md) | Refactor subproblem.rs to use new modules | 3 | Yes | T-013-T-016 | ✅ |

**Total Points**: 16

---

## Parallelization

```
T-012 (Structure) ──→ T-013 (Hydro) ────────────────┐
                 └──→ T-014 (Bus) ─────────────────├──→ T-017 (Facade)
                 └──→ T-015 (AR) ──────────────────┤
                 └──→ T-016 (Bounds) ──────────────┘
```

- **T-013**, **T-014**, **T-015**, **T-016** can all run in parallel after T-012
- **T-017** consolidates everything and integrates with subproblem.rs

---

## Dependencies

- **From Sprint 1**:
  - VariableIndices struct (T-007)
  - ConstraintIndices struct (T-007)
  - SolutionExtractor pattern (T-008)
  - Established extraction methodology
- **To Epic 3**:
  - Clean constraint building interface
  - Reduced `subproblem.rs` complexity
  - SolutionExtractor integrated

---

## Key Files

| File | Lines | Purpose in This Sprint |
|------|-------|------------------------|
| `src/subproblem.rs` | 6,631 | Source of extraction, integration point |
| `src/model/constraints/mod.rs` | - | New: constraint module root |
| `src/model/constraints/hydro_balance.rs` | - | New: hydro constraints |
| `src/model/constraints/bus_balance.rs` | - | New: bus constraints |
| `src/model/constraints/ar_dynamics.rs` | - | New: AR constraints |
| `src/model/constraints/bounds.rs` | - | New: bound constraints (if needed) |

---

## Constraint Types to Extract

Based on `subproblem.rs` analysis:

1. **Hydro Balance**: Water balance constraints for hydro plants
   - One constraint per hydro
   - Includes upstream cascade contributions
   
2. **Bus Balance**: Power balance at network buses
   - One constraint per bus
   - Includes thermal, hydro, and exchange contributions
   
3. **AR Dynamics**: Auto-regressive state transition constraints
   - Uncertainty observation constraints (Y = base + σ·η + lags)
   - Lag fixing constraints (Y_{t-k} = value)
   
4. **Bounds**: Variable bounds and capacity limits (may be at variable creation)

---

## Verification Checklist

After EVERY ticket:

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] `./scripts/golden-tests.sh verify` passes

After sprint complete:

- [ ] `cargo bench` shows no regression (within 5%)
- [ ] All constraint building in new modules
- [ ] `subproblem.rs` uses facade pattern
- [ ] All extracted functions ≤50 lines

---

## Risks

| Risk | Mitigation |
|------|------------|
| Constraint building entangled with other logic | Careful analysis, incremental extraction |
| Entity routing (load vs inflow) errors | Match original logic exactly |
| Performance regression from indirection | Use `#[inline]`, benchmark after sprint |
| T-017 integration failures | Test after every method delegation |

---

## Definition of Done

- [ ] All 6 tickets complete
- [ ] All tests passing
- [ ] Golden tests passing
- [ ] Benchmarks within 5% of baseline
- [ ] Code reviewed and merged
- [ ] `subproblem.rs` uses new modules via facade pattern
- [ ] Ready for Epic 3: Algorithm Separation
