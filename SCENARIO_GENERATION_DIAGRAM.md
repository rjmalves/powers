# Scenario Generation Flow Diagram

## Current System: Two Lag Buffer Systems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SAA GENERATION (Once)                              │
│                      scenario_generator.rs                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 1                    Stage 2                    Stage 3
   │                          │                          │
   │ Sample Z ~ N(0,1)        │                          │
   ├──► Transform             │                          │
   │    ε₁ = f(Z)             │                          │
   │                          │                          │
   │ [LEGACY] Apply AR:       │                          │
   │    Z'₁ = ε₁              │                          │
   │    Y₁ = μ + σ·Z'₁        │                          │
   │                          │                          │
   │ par_states.push(Z'₁) ───┼─► [LEGACY] Apply AR:     │
   │         ↓                │    Z'₂ = φ·Z'₁ + ε₂      │
   │    ⚠️  UNUSED!           │    Y₂ = μ + σ·Z'₂        │
   │                          │                          │
   │ Store in SAA:            │ par_states.push(Z'₂) ───┼─► [LEGACY] Apply AR
   │    innovations = [ε₁]    │         ↓                │    Z'₃ = φ·Z'₂ + ε₃
   │    ❌ values = [Y₁]      │    ⚠️  UNUSED!           │    Y₃ = μ + σ·Z'₃
   │       (DISCARDED!)       │                          │
   │                          │ Store in SAA:            │ Store in SAA:
   │                          │    innovations = [ε₂]    │    innovations = [ε₃]
   │                          │    ❌ values = [Y₂]      │    ❌ values = [Y₃]
   │                          │       (DISCARDED!)       │       (DISCARDED!)
   │                          │                          │
   └──────────────────────────┴──────────────────────────┴──────────────────────

                              ⬇️  SAA TREE  ⬇️

        ┌────────────────────────────────────────────────┐
        │   SAA: Only stores innovations [ε₁, ε₂, ε₃]   │
        │   ✅ Compact, efficient                        │
        │   ❌ Y₁, Y₂, Y₃ are NEVER stored or used      │
        └────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                       SDDP EXECUTION (Every Iteration)                      │
│                         subproblem.rs                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Forward Pass: Stage t

    1. Sample innovation from SAA: ε_t
       │
    2. Get lag observations from inflow_manager:
       │  lag_buffer = [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
       │
    3. Update AR constraint RHS (realize_uncertainties):
       │
       │  RHS = deterministic_base + stochastic_term + lag_contribution
       │        └─────────────────┘   └─────────────┘   └────────────────┘
       │        μ_t - Σ(φᵢ·μ_{t-i})   σ_t · ε_t         Σ[φᵢ · Y_{t-i}]
       │        (pre-computed)         (from SAA)        (from lag_buffer)
       │
       │  Set constraint: Y_t = RHS
       │
    4. Solve LP
       │  │
       │  └──► Get Y_t from solution
       │
    5. Update lag buffer for next stage:
       │  inflow_manager.update_lag_buffer(Y_t)
       │  lag_buffer = [Y_t, Y_{t-1}, ..., Y_{t-p+1}]
       │
    6. Move to next stage →
```

---

## Key Insights

### ✅ What Actually Happens

```
SAA Generation:
   Input: Historical data, distributions
   Output: Tree of innovations [ε₁, ε₂, ..., εₜ] for each scenario
   Status: ✅ CORRECT

SDDP Execution:
   Input: Innovations from SAA + Lag buffer (observations)
   Process: Y_t = f(ε_t, Y_{t-1}, ..., Y_{t-p})
   Output: Policy cuts
   Status: ✅ CORRECT
```

### ⚠️ What's Confusing (Legacy Code)

```
SAA Generation (scenario_generator.rs):
   par_states.apply_ar(innovation)  ⚠️ LEGACY
        ↓
   Computes: Y_t = μ + σ · Z'_t
        ↓
   Stored in: scenario.values[t]
        ↓
   Used for: ❌ NOTHING! (discarded for inflows)
   
Why it exists:
   - Historical artifact from when SAA stored observations
   - Currently only used for Load entities
   - For Inflow entities, only scenario.innovations is used
```

---

## Data Flow Comparison

### For Load Entities (Deterministic/Simple)

```
generate_stage_scenarios:
   Sample → Transform → Store observation
                         ↓
                      SAA.load_observations
                         ↓
              realize_uncertainties: Use directly
```

**Status**: ✅ Correct, observations are used

### For Inflow Entities (PAR Models)

```
generate_stage_scenarios:
   Sample → Transform → [LEGACY] Apply AR → Compute observation
                         ↓                    ↓
                      ⚠️ Update             ❌ DISCARDED!
                         par_states          (never used)
                         ↓
                      Store ONLY innovation
                         ↓
                      SAA.inflow_innovations
                         ↓
              realize_uncertainties:
                         ↓
                 Combine with lag_buffer
                    (from inflow_manager)
                         ↓
                   Compute RHS: Y_t = ...
                         ↓
                      Solve LP → Get Y_t
                         ↓
              Update inflow_manager.lag_buffer(Y_t)
```

**Problem**: The par_states updates and observation computation during SAA generation are **never used**!

---

## Memory Layout

### Current (With Legacy par_states)

```
ScenarioGenerator:
   models:              ~6 KB    (20 entities × 300 bytes)
   par_states:          ~800 B   (10 PAR × 80 bytes)  ⚠️ UNUSED during execution
   buffers:             ~320 B   (2 × 20 × 8 bytes)
   ──────────────────────────
   Total:               ~7.1 KB

Scenario (per scenario):
   values:              160 B    (20 × 8 bytes)      ⚠️ Inflow values unused
   innovations:         160 B    (20 × 8 bytes)      ✅ Used in SAA
   residuals:           160 B    (20 × 8 bytes)      ⚠️ UNUSED
   ──────────────────────────
   Total:               480 B
```

### After Cleanup (Removing par_states)

```
ScenarioGenerator:
   models:              ~6 KB
   par_states:          REMOVED (-800 B)
   buffers:             ~320 B
   ──────────────────────────
   Total:               ~6.3 KB  (-11%)

Scenario (per scenario):
   values:              160 B    (Load entities only)
   innovations:         160 B    ✅ Used in SAA
   residuals:           REMOVED (-160 B)
   ──────────────────────────
   Total:               320 B    (-33%)
```

**Savings**: ~11% on ScenarioGenerator, ~33% on Scenario structs

---

## Correctness Verification

### Current Behavior

```python
# SAA Generation
for stage in stages:
    innovation = sample()
    residual = apply_ar(innovation)  # Uses par_states
    observation = transform(residual)
    
    saa.store(innovation)  # ✅ This is used
    # observation is DISCARDED for inflows!

# SDDP Execution  
for stage in stages:
    innovation = saa.get(stage)  # ✅ From above
    lags = inflow_manager.get_lags()  # ✅ From previous solves
    
    rhs = compute_rhs(innovation, lags)  # ✅ Correct formula
    solve_lp()
    
    observation = get_solution()
    inflow_manager.update(observation)  # ✅ For next stage
```

### After Cleanup

```python
# SAA Generation
for stage in stages:
    innovation = sample()
    # REMOVED: residual = apply_ar(innovation)
    # REMOVED: observation = transform(residual)
    
    saa.store(innovation)  # ✅ Same as before

# SDDP Execution  
for stage in stages:
    innovation = saa.get(stage)  # ✅ Unchanged
    lags = inflow_manager.get_lags()  # ✅ Unchanged
    
    rhs = compute_rhs(innovation, lags)  # ✅ Unchanged
    solve_lp()
    
    observation = get_solution()
    inflow_manager.update(observation)  # ✅ Unchanged
```

**Result**: Identical behavior, less code, less memory!

---

## Why The Bug Manifested

The lognormal bug was in `seasonal_params.std_dev`:

```
Bug Location: uncertainty_model.rs (parameter conversion)
   ↓
Used in: scenario_generator.rs (scenario.values computation)
   ↓
Impact: ❌ None for inflows! (values discarded)
   ↓
BUT ALSO used in: subproblem.rs (stochastic_term computation)
   ↓
Impact: ✅ BIG! Wrong constraint RHS → infeasible/wrong results
```

The bug **would not have manifested** if:
1. We only used innovations in SAA (already true!)
2. We didn't compute observations during SAA generation (should be true!)

The fix corrected `std_dev` which fixed the **actual problem** (constraint RHS), not the **apparent problem** (scenario.values computation which is unused).

---

## Conclusion

You were right to be confused! The code has two lag buffer systems:

1. **ScenarioGenerator.par_states** (residual space, LEGACY):
   - Used to compute scenario.values during generation
   - These values are DISCARDED for inflows
   - Should be removed

2. **Subproblem.inflow_manager** (observation space, ACTIVE):
   - Used during SDDP execution for AR constraints
   - Updated after each solve
   - This is the correct system

**Action**: Remove par_states to eliminate confusion and reduce memory usage.
