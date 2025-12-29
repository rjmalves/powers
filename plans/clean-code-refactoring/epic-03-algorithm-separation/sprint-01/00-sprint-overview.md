# Sprint 1: Forward Pass Extraction

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⚠️ Partially Complete - T-021 Requires Rework

---

## ⚠️ CRITICAL REMINDER

This sprint extracted the **forward pass**—a core part of the SDDP algorithm.

**The algorithm logic must remain EXACTLY unchanged.** We are only reorganizing code.

---

## Goals

1. **Primary**: Extract forward pass logic to `src/algorithm/forward_pass.rs` ✅
2. **Primary**: Create `ForwardPassContext` struct ✅
3. **Primary**: Integrate new timing infrastructure ❌ **REWORK NEEDED**
4. **Validation**: Bit-for-bit identical outputs ✅

---

## Tickets

| ID | Title | Points | Status | Notes |
|----|-------|--------|--------|-------|
| [T-018](./ticket-018-design-context-structs.md) | Design context structs | 3 | ✅ | `docs/context-struct-design.md` created |
| [T-019](./ticket-019-forward-pass-context.md) | Create ForwardPassContext | 3 | ✅ | In `src/algorithm/context.rs` |
| [T-020](./ticket-020-extract-forward-step.md) | Extract forward pass step logic | 5 | ✅ | In `src/algorithm/forward_pass.rs` |
| [T-021](./ticket-021-forward-timing-integration.md) | Integrate forward timing infrastructure | 3 | ❌ | **REWORK: See ticket for details** |
| [T-022](./ticket-022-update-sddp-forward.md) | Update sddp/mod.rs to use forward_pass module | 3 | ✅ | `forward_pass::execute()` called |

**Total Points**: 17

---

## T-021 Rework Required

### Problem

The initial implementation used `Instant::now()` instead of `TimingGuard` because timing was embedded inside `ForwardPassContext`, causing borrow checker conflicts.

**This was an unacceptable compromise that should have been escalated.**

### Solution

**Separate timing from context** - pass timing as a separate parameter:

```rust
// BEFORE (problematic):
pub struct ForwardPassContext<'a> {
    pub timing: &'a ForwardTiming,  // ❌ Causes borrow conflicts
    // ...
}

// AFTER (correct):
pub struct ForwardPassContext<'a> {
    // NO timing field
    // ...
}

pub fn execute(
    ctx: &mut ForwardPassContext,
    timing: &TrajectoryTiming,  // ✅ Separate parameter
) -> Result<...>
```

### What Must Be Done

1. Remove timing from `ForwardPassContext`
2. Update `TrajectoryTiming` to use `Cell<Duration>`
3. Update `execute()` and `execute_stage()` signatures
4. Replace all `Instant::now()` with `TimingGuard`
5. Verify feature-gating works

See [T-021](./ticket-021-forward-timing-integration.md) for full specification.

---

## Dependencies

- **From Epic 2**:
  - Clean `SolutionExtractor` interface ✅
  - Clean constraint building interface ✅
  
- **From Epic 1**:
  - `TimingGuard` and timing infrastructure ✅

- **To Sprint 2**:
  - Forward pass module complete ✅
  - Context struct pattern established ✅
  - **Timing separation pattern** (after T-021 rework)

---

## Key Files

| File | Lines | Role |
|------|-------|------|
| `src/sddp/mod.rs` | 3,827 | Source of extraction |
| `src/algorithm/forward_pass.rs` | 336 | Forward pass module |
| `src/algorithm/context.rs` | 523 | Context structs |
| `src/timing/guard.rs` | 153 | TimingGuard RAII |

---

## Lessons Learned

### What Went Wrong

1. **Borrow checker conflict was discovered** during implementation
2. **Compromise was made** (using `Instant::now()` instead of `TimingGuard`)
3. **Compromise was not escalated** - work continued as if goals were met

### What Should Have Happened

1. Stop implementation when conflict was discovered
2. Escalate to project owner with analysis and options
3. Get architectural decision before proceeding
4. Implement correct solution

### Architectural Decision (Made 2025-12-29)

**Timing must be separate from context structs** to avoid borrow conflicts with `TimingGuard`. This applies to:
- `ForwardPassContext` (Sprint 1)
- `BackwardPassContext` (Sprint 2)
- `BackwardStageContext` (Sprint 2)
- Any future context structs

---

## Definition of Done

- [x] Forward pass extracted to `src/algorithm/forward_pass.rs`
- [x] `ForwardPassContext` struct created
- [ ] **Timing integrated using `TimingGuard`** ← INCOMPLETE
- [ ] **Feature-gating verified** ← INCOMPLETE
- [x] Golden tests pass
- [x] All tests pass
- [x] Benchmark within 5% of baseline
- [x] Code reviewed
