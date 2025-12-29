# Sprint 2: Backward Pass Extraction

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Duration**: 1.5 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

This sprint extracts the **backward pass**—the cut generation phase of SDDP.

**The algorithm logic must remain EXACTLY unchanged.** Cut computation, selection, and application must produce identical results. Run golden tests after EVERY change.

If ANY test fails or output differs, **STOP IMMEDIATELY** and investigate.

---

## Goals

1. **Primary**: Extract backward pass logic to `src/algorithm/backward_pass.rs`
2. **Primary**: Extract cut computation to `src/algorithm/cut_computation.rs`
3. **Primary**: Create `BackwardPassContext` struct
4. **Validation**: Bit-for-bit identical outputs

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-023](./ticket-023-backward-pass-context.md) | Create BackwardPassContext | 3 | Yes | Sprint 1 | ⬜ |
| [T-024](./ticket-024-extract-backward-pass.md) | Extract backward pass logic | 5 | Yes | T-023 | ⬜ |
| [T-025](./ticket-025-extract-cut-computation.md) | Extract cut computation | 3 | Yes | T-024 | ⬜ |
| [T-026](./ticket-026-backward-timing-integration.md) | Integrate backward timing infrastructure | 3 | Yes | T-025 | ⬜ |
| [T-027](./ticket-027-update-sddp-backward.md) | Update sddp/mod.rs to use backward_pass module | 3 | Yes | T-026 | ⬜ |

**Total Points**: 17

---

## Sequence

```
T-023 (Context) ──→ T-024 (Extract) ──→ T-025 (Cuts) ──→ T-026 (Timing) ──→ T-027 (Integrate)
```

Sequential sprint—each ticket builds on the previous.

---

## Dependencies

- **From Sprint 1**:
  - `ForwardPassContext` pattern established
  - Forward pass module complete
  - Timing integration pattern proven
- **To Epic 4**:
  - Clean state interface for backward pass
- **To Epic 5**:
  - Identified cut allocation points

---

## Key Files

| File | Lines | Role |
|------|-------|------|
| `src/sddp/mod.rs` | 3,913 | Source of extraction (backward methods) |
| `src/algorithm/backward_pass.rs` | - | New: backward pass module |
| `src/algorithm/cut_computation.rs` | - | New: cut calculation |
| `src/algorithm/context.rs` | - | BackwardPassContext |

---

## Backward Pass Components to Extract

From analysis of `sddp/mod.rs`:

1. **Backward iteration** over stages (reverse order)
2. **Branching** - generating scenarios for cut computation
3. **Subproblem solves** for each branching
4. **Cut computation** - Benders cuts from dual values
5. **Cut selection** - choosing which cuts to add
6. **FCF update** - adding cuts to future cost function
7. **Multi-threaded coordination** (Phase 1, Phase 2, Phase 3)

---

## Critical Parallel Sections

The backward pass has complex parallel coordination:

```
Phase 1: Parallel cut computation (no FCF lock)
Phase 2: Sequential FCF updates (critical section)
Phase 3: Handler application
```

**Do NOT change the synchronization pattern.** Extract the logic but preserve:
- Lock ordering
- Thread coordination
- Deterministic cut ordering

---

## Timing Integration Requirements

Per master plan, use new timing infrastructure:

1. `BackwardTiming` struct for all backward metrics
2. `TimingGuard` for each timed section
3. Preserve precise values
4. No overwriting or redistribution

---

## Verification Protocol

After EVERY ticket:

```bash
cargo build && cargo test && ./scripts/golden-tests.sh verify
```

After sprint complete:
```bash
cargo bench -- --baseline before-refactoring
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| Parallel synchronization change | Do not modify lock patterns, verify thread behavior |
| Cut ordering change | Ensure deterministic ordering preserved |
| FCF update race condition | Keep existing synchronization exactly |
| Performance regression | Benchmark, minimize indirection |

---

## Definition of Done

- [ ] All 5 tickets complete
- [ ] Backward pass fully extracted
- [ ] Cut computation in separate module
- [ ] `BackwardPassContext` reduces parameters
- [ ] New timing integrated
- [ ] **Parallel behavior unchanged**
- [ ] Golden tests pass
- [ ] All tests pass
- [ ] Benchmark within 5%
- [ ] Code reviewed
