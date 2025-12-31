# Sprint 8 (Revised): Per-Iteration Model Architecture with Optional Basis

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Previous Sprint 8**: [Superseded - Model Rebuild Strategy](../sprint-08/00-sprint-overview.md)
> **Duration**: 3 weeks
> **Status**: 📋 Planned

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Architecture changes must not alter numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

This sprint implements the **Per-Iteration Model Architecture** with **optional basis caching**, enabling:

1. **Training mode**: Per-iteration Model lifecycle with basis warm-starting
2. **Simulation mode**: Per-iteration Model lifecycle WITHOUT basis (for reproducibility from persisted FCF)

### Key Design Insight

When users load a persisted FCF and run simulation only, they cannot have the basis from training (storing basis for large problems is infeasible). The simulation must produce the same results whether:
- Run immediately after training (with basis available)
- Run from loaded FCF (without basis)

This means **basis must be optional**, and results must be deterministic regardless of basis availability.

---

## Architecture Overview

### Mode-Aware Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ITERATION LIFECYCLE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. CREATE MODELS                                                        │
│     ┌─────────────────────────────────────────────────────────┐         │
│     │ for each stage:                                          │         │
│     │   model = problem.create_model()                         │         │
│     │   if use_basis && cached_basis.is_some():               │         │
│     │     model.apply_stored_basis(cached_basis)  ← OPTIONAL  │         │
│     └─────────────────────────────────────────────────────────┘         │
│                                                                          │
│  2. FORWARD/BACKWARD PASS                                                │
│     [unchanged - uses Model]                                             │
│                                                                          │
│  3. FINALIZE                                                             │
│     ┌─────────────────────────────────────────────────────────┐         │
│     │ for each stage:                                          │         │
│     │   if use_basis:                                          │         │
│     │     cached_basis = model.get_stored_basis() ← OPTIONAL  │         │
│     │   drop(model)                                            │         │
│     └─────────────────────────────────────────────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

TRAINING:  use_basis = true   → Warm-start between iterations
SIMULATION: use_basis = false → Cold-start, reproducible from FCF
```

### Why This Matters for FCF Persistence

```
Future Feature: Load FCF and Simulate

User workflow:
1. Train SDDP → produces FCF with cuts
2. Persist FCF to disk (cuts, coefficients, RHS)
3. Later: Load FCF from disk
4. Run simulation with loaded FCF

Challenge: Basis cannot be persisted (too large)
Solution: Simulation doesn't use basis → always cold-start
Guarantee: Same results whether simulating after training or from loaded FCF
```

---

## Sprint Goals

### Priority 1: Solver Interface Extensions

| Goal | Description |
|------|-------------|
| Non-consuming model creation | `Problem::create_model()` |
| Problem modification methods | Row bound/coefficient changes |
| Basis persistence | `StoredBasis` for optional warm-start |

### Priority 2: Subproblem Architecture

| Goal | Description |
|------|-------------|
| Dual storage | `problem: Problem` + `model: Option<Model>` |
| Optional basis | `cached_basis: Option<StoredBasis>` with `use_basis` flag |
| Per-iteration lifecycle | `create_iteration_model(use_basis: bool)` |
| Dual cut updates | Update both Problem and Model during backward pass |

### Priority 3: Configuration

| Goal | Description |
|------|-------------|
| Basis configuration | Runtime flag for basis usage |
| Mode detection | Training vs Simulation mode |

### Priority 4: Validation

| Goal | Description |
|------|-------------|
| Determinism test | Same results with/without basis |
| RSS verification | Memory reclaimed between iterations |
| Golden tests | Numerical correctness |

---

## Technical Specification

### Subproblem Structure

```rust
pub struct Subproblem {
    /// Persistent LP problem definition (source of truth).
    pub problem: solver::Problem,
    
    /// Transient Model for current iteration.
    pub model: Option<solver::Model>,
    
    /// Cached basis from previous iteration.
    /// Only populated if basis caching is enabled.
    pub cached_basis: Option<solver::StoredBasis>,
    
    // ... other fields unchanged
}
```

### Lifecycle Methods with Optional Basis

```rust
impl Subproblem {
    /// Create Model for a new iteration.
    ///
    /// # Arguments
    ///
    /// * `use_basis` - If true, apply cached basis for warm-starting.
    ///                 If false, cold-start (for simulation reproducibility).
    ///
    /// # Training vs Simulation
    ///
    /// - Training: `use_basis = true` → Faster solves via warm-start
    /// - Simulation: `use_basis = false` → Reproducible without stored basis
    pub fn create_iteration_model(&mut self, use_basis: bool) -> Result<(), String> {
        if self.model.is_some() {
            return Err("Model already exists".into());
        }
        
        let mut model = self.problem
            .create_model(solver::Sense::Minimise)
            .map_err(|e| format!("Model creation failed: {:?}", e))?;
        
        set_default_solver_options(&mut model);
        
        // OPTIONAL: Apply cached basis only if requested
        if use_basis {
            if let Some(ref basis) = self.cached_basis {
                if basis.is_compatible(model.num_cols(), model.num_rows()) {
                    let _ = model.apply_stored_basis(basis);
                }
            }
        }
        
        self.model = Some(model);
        Ok(())
    }
    
    /// Finalize iteration with optional basis caching.
    ///
    /// # Arguments
    ///
    /// * `cache_basis` - If true, cache basis for next iteration.
    ///                   If false, don't cache (saves memory, simulation mode).
    pub fn finalize_iteration(&mut self, cache_basis: bool) {
        if cache_basis {
            if let Some(ref model) = self.model {
                self.cached_basis = Some(model.get_stored_basis());
            }
        }
        // Drop Model regardless
        self.model = None;
    }
    
    /// Clear cached basis (for transitioning to simulation or freeing memory).
    pub fn clear_cached_basis(&mut self) {
        self.cached_basis = None;
    }
}
```

### Training Loop Integration

```rust
// Configuration
struct IterationConfig {
    /// Use basis warm-starting (training: true, simulation: false)
    use_basis: bool,
}

impl SddpAlgorithm {
    pub fn train(&mut self, config: &TrainingConfig) -> TrainingResult {
        let iter_config = IterationConfig { use_basis: true };
        
        for iteration in 1..=config.max_iterations {
            self.run_iteration(&iter_config)?;
        }
        // ...
    }
    
    pub fn simulate(&mut self, config: &SimulationConfig) -> SimulationResult {
        // Simulation: don't use basis for reproducibility
        let iter_config = IterationConfig { use_basis: false };
        
        // Clear any cached basis from training
        for handler in &mut self.handlers {
            handler.subproblem.clear_cached_basis();
        }
        
        for scenario in 0..config.num_scenarios {
            self.run_iteration(&iter_config)?;
        }
        // ...
    }
    
    fn run_iteration(&mut self, config: &IterationConfig) -> Result<(), String> {
        // Create models (with or without basis)
        for handler in &mut self.handlers {
            handler.subproblem.create_iteration_model(config.use_basis)?;
        }
        
        // Forward/backward passes...
        
        // Finalize (with or without basis caching)
        for handler in &mut self.handlers {
            handler.subproblem.finalize_iteration(config.use_basis);
        }
        
        Ok(())
    }
}
```

### Future: FCF Persistence Integration

```rust
// FUTURE FEATURE (not in this sprint, but enabled by this architecture)

impl FutureCostFunction {
    /// Persist FCF to disk for later simulation.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        // Save cuts, coefficients, RHS
        // Note: Basis is NOT saved (too large, not needed for simulation)
    }
    
    /// Load FCF from disk.
    pub fn load(path: &Path) -> Result<Self, Error> {
        // Load cuts, coefficients, RHS
        // No basis available - simulation will cold-start
    }
}

// User workflow:
// 1. algorithm.train() → FCF populated
// 2. fcf.save("model.fcf")
// 3. Later: fcf = FutureCostFunction::load("model.fcf")
// 4. algorithm.simulate() with use_basis=false
// 5. Results match simulating immediately after training
```

---

## Sprint Tickets

### Priority 1: Solver Interface

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-110 | Implement `Problem::create_model()` | 5 | None |
| T-111 | Add `Problem` modification methods | 3 | T-110 |
| T-112 | Implement `StoredBasis` and basis transfer | 3 | T-110 |

### Priority 2: Subproblem Architecture

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-113 | Refactor Subproblem for dual Problem+Model storage | 5 | T-110, T-111, T-112 |
| T-114 | Implement per-iteration Model lifecycle with optional basis | 5 | T-113 |
| T-115 | Implement dual cut update (`update_cut_dual`) | 3 | T-113, T-114 |
| T-116 | Update `realize_and_solve()` to use iteration Model | 3 | T-114 |

### Priority 3: Training Loop Integration

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-117 | Integrate iteration lifecycle into training loop | 5 | T-114, T-115, T-116 |
| T-118 | Add basis configuration for simulation mode | 3 | T-117 |

### Priority 4: Deferred Optimizations (Carried Over)

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-100-r | Scenario sampling indices buffer | 3 | None |
| T-102-r | CutIdSet type implementation | 3 | None |

### Priority 5: Validation

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-119 | Determinism test: with vs without basis | 3 | T-118 |
| T-120 | RSS verification tests | 3 | T-117 |
| T-121 | Performance benchmarks | 3 | T-117 |
| T-122 | Golden test validation | 2 | All above |

**Total**: 52 points (3 week sprint)

---

## Key Design Decisions

### 1. Why Optional Basis?

| Mode | `use_basis` | Behavior | Rationale |
|------|-------------|----------|-----------|
| Training | `true` | Warm-start with cached basis | Performance |
| Simulation | `false` | Cold-start, no basis | Reproducibility from FCF |

### 2. Determinism Guarantee

The solver must produce **identical results** regardless of basis:
- Same cuts active
- Same constraint bounds
- Same coefficients
- ⇒ Same optimal solution

Basis only affects **solver performance** (fewer iterations), not **results**.

### 3. When to Update What

| Update Type | Problem? | Model? | Reason |
|-------------|----------|--------|--------|
| Uncertainty RHS | ❌ | ✅ | Per-solve, not persisted |
| Lag constraints | ❌ | ✅ | Per-solve, not persisted |
| Hydro balance | ❌ | ✅ | Per-solve, not persisted |
| New cut | ✅ | ✅ | Persist + current backward |

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Results differ with/without basis | Low | Critical | Determinism test (T-119) |
| Model creation overhead | Low | Medium | Benchmark (T-121) |
| Dual update sync bugs | Medium | High | Comprehensive tests |
| Memory not reclaimed | Low | High | RSS tests (T-120) |

---

## Acceptance Criteria

### Architecture

- [ ] Problem is persistent source of truth
- [ ] Model is transient per-iteration
- [ ] Basis caching is optional (controlled by flag)
- [ ] Dual cut update works correctly

### Training Mode

- [ ] `use_basis=true` applies cached basis
- [ ] Basis cached at end of iteration
- [ ] Warm-starting reduces solve time

### Simulation Mode

- [ ] `use_basis=false` cold-starts every iteration
- [ ] No basis cached
- [ ] Results identical to training-then-simulate

### Validation

- [ ] Determinism test passes (same results ±/- basis)
- [ ] RSS decreases between iterations
- [ ] Golden tests pass
- [ ] All 567+ tests pass

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/solver.rs` | `create_model()`, modification methods, `StoredBasis` |
| `src/subproblem.rs` | Dual storage, lifecycle methods, optional basis |
| `src/sddp/mod.rs` | Iteration lifecycle, basis configuration |
| `tests/subproblem_*.rs` | New tests |
| `tests/determinism_test.rs` | New: verify basis doesn't affect results |
| `benches/` | Iteration overhead benchmark |

---

## Definition of Done

- [ ] `Problem::create_model()` implemented
- [ ] Problem modification methods working
- [ ] Basis transfer working with optional flag
- [ ] Per-iteration lifecycle with `use_basis` parameter
- [ ] Training uses `use_basis=true`
- [ ] Simulation uses `use_basis=false`
- [ ] Results identical with/without basis
- [ ] RSS decreases between iterations
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Documentation updated
