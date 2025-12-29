# Sprint 1: Forward Pass Extraction

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

This sprint extracts the **forward pass**—a core part of the SDDP algorithm.

**The algorithm logic must remain EXACTLY unchanged.** We are only reorganizing code. Run golden tests after EVERY change. If ANY test fails or output differs, **STOP IMMEDIATELY** and investigate.

Do NOT:
- ❌ "Improve" the algorithm
- ❌ "Fix" anything that looks like a bug
- ❌ Change the order of operations
- ❌ Modify numerical computations

---

## Goals

1. **Primary**: Extract forward pass logic to `src/algorithm/forward_pass.rs`
2. **Primary**: Create `ForwardPassContext` struct
3. **Primary**: Integrate new timing infrastructure
4. **Validation**: Bit-for-bit identical outputs

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-018](./ticket-018-design-context-structs.md) | Design context structs | 3 | Yes | Epic 2 | ⬜ |
| [T-019](./ticket-019-forward-pass-context.md) | Create ForwardPassContext | 3 | Yes | T-018 | ⬜ |
| [T-020](./ticket-020-extract-forward-step.md) | Extract forward pass step logic | 5 | Yes | T-019 | ⬜ |
| [T-021](./ticket-021-forward-timing-integration.md) | Integrate forward timing infrastructure | 3 | Yes | T-020 | ⬜ |
| [T-022](./ticket-022-update-sddp-forward.md) | Update sddp/mod.rs to use forward_pass module | 3 | Yes | T-021 | ⬜ |

**Total Points**: 17

---

## Sequence

```
T-018 (Design) ──→ T-019 (Context) ──→ T-020 (Extract) ──→ T-021 (Timing) ──→ T-022 (Integrate)
```

This sprint is **sequential**—each ticket builds on the previous. Take extra care with verification at each step.

---

## Dependencies

- **From Epic 2**:
  - Clean `SolutionExtractor` interface
  - Clean constraint building interface
  - Reduced `subproblem.rs` complexity
- **From Epic 1**:
  - `TimingGuard` and `IterationTiming` structs
  - Golden test infrastructure
- **To Sprint 2**:
  - Established context struct pattern
  - Forward pass module complete

---

## Key Files

| File | Lines | Role |
|------|-------|------|
| `src/sddp/mod.rs` | 3,913 | Source of extraction (forward methods) |
| `src/algorithm/forward_pass.rs` | - | New: forward pass module |
| `src/algorithm/context.rs` | - | New: context struct definitions |
| `src/timing/metrics.rs` | - | IterationTiming, ForwardTiming |

---

## Forward Pass Components to Extract

From analysis of `sddp/mod.rs`:

1. **forward()** - Main forward pass entry point
2. **step()** or stage iteration logic
3. **SAA sampling** coordination
4. **Parallel trajectory execution**
5. **Result aggregation**

---

## Timing Integration Requirements

Per master plan:

1. Replace `Instant::now()` / `.elapsed()` with `TimingGuard`
2. Use `IterationTiming.forward` struct
3. **PRESERVE precise values** - do not redistribute
4. **COMPUTE parallel overhead** as `wall_time - avg(cpu_time)`
5. **NEVER overwrite** measured timing values

Example transformation:
```rust
// Before:
let start = Instant::now();
// ... work ...
timing.model_preprocessing_time += start.elapsed();

// After:
{
    let _guard = TimingGuard::new(&ctx.timing.forward.model_preprocessing);
    // ... work ...
}
```

---

## Verification Protocol

After EVERY ticket:

```bash
# 1. Build
cargo build

# 2. Test
cargo test

# 3. Golden test (CRITICAL)
./scripts/golden-tests.sh verify

# 4. If ANY failure, STOP and investigate
```

After sprint complete:
```bash
# Performance check
cargo bench -- --baseline before-refactoring
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| Subtle ordering change | Extract exactly as-is, verify with golden tests |
| Parallel behavior change | Verify same number of threads, same scheduling |
| Timing changes affect output | Timing is observational only, verify independence |
| Context struct lifetime issues | Careful lifetime annotations, test thoroughly |

---

## Definition of Done

- [ ] All 5 tickets complete
- [ ] Forward pass fully extracted to `src/algorithm/forward_pass.rs`
- [ ] `ForwardPassContext` reduces parameters from 8+ to ≤4
- [ ] New timing integrated (no raw `Instant::now()`)
- [ ] Golden tests pass
- [ ] All tests pass
- [ ] Benchmark within 5% of baseline
- [ ] Code reviewed
