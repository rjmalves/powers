# Sprint 8 Revision Summary: Per-Iteration Model with Optional Basis

> **Date**: 2025-12-31
> **Revision**: 2 (Added optional basis for simulation reproducibility)

---

## Design Evolution

| Version | Approach | Issue |
|---------|----------|-------|
| Original Sprint 8 | Periodic Model rebuild | Complex cut extraction/restoration |
| Revision 1 | Per-solve Model | 6000+ Model creations/iter (too slow) |
| Revision 2 | Per-iteration Model | ✅ Balanced performance + memory |
| **Final** | + Optional basis | ✅ Enables simulation from loaded FCF |

---

## Key Insight: Simulation Reproducibility

When users load a persisted FCF and run simulation:
- They don't have the basis (too large to store)
- Results must match running simulation immediately after training

**Solution**: Make basis usage **optional**:
- Training: `use_basis=true` → warm-start for performance
- Simulation: `use_basis=false` → cold-start for reproducibility

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Subproblem                                   │
├─────────────────────────────────────────────────────────────────────┤
│  problem: Problem           ← Persistent LP definition              │
│  model: Option<Model>       ← Transient per-iteration               │
│  cached_basis: Option<StoredBasis> ← Optional, for warm-start       │
└─────────────────────────────────────────────────────────────────────┘

Training:
  create_iteration_model(use_basis: true)   ← Apply cached basis
  ... forward/backward passes ...
  finalize_iteration(cache_basis: true)     ← Save basis

Simulation:
  clear_cached_basis()
  create_iteration_model(use_basis: false)  ← Cold-start
  ... forward pass only ...
  finalize_iteration(cache_basis: false)    ← Don't save
```

---

## Sprint Tickets (13 total, 52 points)

### Solver Interface (11 points)
| ID | Title | Points |
|----|-------|--------|
| T-110 | `Problem::create_model()` | 5 |
| T-111 | Problem modification methods | 3 |
| T-112 | StoredBasis and basis transfer | 3 |

### Subproblem Architecture (16 points)
| ID | Title | Points |
|----|-------|--------|
| T-113 | Dual Problem+Model storage | 5 |
| T-114 | Lifecycle with optional basis | 5 |
| T-115 | Dual cut update | 3 |
| T-116 | Update realize_and_solve | 3 |

### Integration (8 points)
| ID | Title | Points |
|----|-------|--------|
| T-117 | Training loop integration | 5 |
| T-118 | Simulation mode config | 3 |

### Deferred (6 points, carried over)
| ID | Title | Points |
|----|-------|--------|
| T-100-r | Scenario indices buffer | 3 |
| T-102-r | CutIdSet type | 3 |

### Validation (11 points)
| ID | Title | Points |
|----|-------|--------|
| T-119 | Determinism test (±basis) | 3 |
| T-120 | RSS verification | 3 |
| T-121 | Performance benchmarks | 3 |
| T-122 | Golden test validation | 2 |

---

## Dual Update Pattern

During backward pass, cuts update **both** Problem and Model:

```
Cut at stage t → updates FCF of stage t-1

┌──────────────┐     ┌──────────────┐
│ Problem[t-1] │     │ Model[t-1]   │
│ (next iter)  │     │ (this iter)  │
└──────┬───────┘     └──────┬───────┘
       │ update_cut_dual()  │
       ▼                    ▼
   For iteration N+1    For backward pass
                        (stage t-2 needs it)
```

---

## When to Update What

| Update Type | Problem? | Model? |
|-------------|----------|--------|
| Uncertainty RHS | ❌ | ✅ |
| Lag constraints | ❌ | ✅ |
| Hydro balance | ❌ | ✅ |
| New cut (backward) | ✅ | ✅ |

---

## Files Created

```
sprint-08-revised/
├── 00-sprint-overview.md
├── ticket-110-create-model.md
├── ticket-111-problem-modifications.md
├── ticket-112-stored-basis.md
├── ticket-113-subproblem-dual-storage.md
├── ticket-114-iteration-lifecycle.md
├── ticket-115-dual-cut-update.md
├── ticket-116-realize-and-solve.md
├── ticket-117-training-loop-integration.md
├── ticket-118-simulation-mode.md
├── ticket-119-determinism-test.md
├── ticket-120-rss-verification.md
├── ticket-121-performance-benchmarks.md
├── ticket-122-golden-tests.md
└── SPRINT_8_REVISION_SUMMARY.md
```

---

## Future Enablement: FCF Persistence

This architecture enables (in a future sprint):

```rust
// Save FCF (cuts only, no basis)
fcf.save("model.fcf")?;

// Later: Load and simulate
let fcf = FutureCostFunction::load("model.fcf")?;
algorithm.load_fcf(fcf);
algorithm.simulate(&config);  // use_basis=false, reproducible
```

---

## Superseded Tickets

Original Sprint 8 tickets are replaced:

| Original | Status |
|----------|--------|
| T-104 (rebuild_model) | ❌ Superseded |
| T-105 (RSS monitoring) | ⏸️ Merged into T-120 |
| T-107 (rebuild integration) | ❌ Superseded |
| T-108 (rebuild config) | ❌ Superseded |
