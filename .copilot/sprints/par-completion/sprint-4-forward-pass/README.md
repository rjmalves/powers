# Sprint 4: Forward Pass & Scenario Handling (Week 6-7)

**Goal**: Implement innovation sampling, state transition, and scenario handling for forward pass simulation.

**Duration**: 2 weeks  
**Total Story Points**: 16  
**Risk Level**: 🟡 MEDIUM

---

## Overview

The forward pass simulates system operation forward in time. For PAR models, this involves:

1. **Innovation sampling**: Sample ε ~ N(0,1)
2. **Innovation transformation**: Convert ε to inflow using AR equation
3. **Correlation**: Apply correlation across hydros
4. **State transition**: Update lag buffers with realized inflows
5. **RHS updates**: Update AR constraint RHS with scenarios

**Key Concept**: Scenarios are innovations (ε), not absolute inflows. This maintains consistency with AR structure.

---

## Sprint Tickets

| ID | Title | Points | Status | Priority |
|----|-------|--------|--------|----------|
| PAR-V2-014 | Forward pass initialization | 2 | 🔵 Not Started | 🔥 High |
| PAR-V2-015 | Forward pass state transition | 3 | 🔵 Not Started | 🔥🔥 Critical |
| PAR-V2-016 | Innovation sampling capability | 3 | 🔵 Not Started | 🔥 High |
| PAR-V2-017 | Transform innovations to inflows | 3 | 🔵 Not Started | 🔥 High |
| PAR-V2-018 | Handle innovation correlation | 3 | 🔵 Not Started | Medium |
| PAR-V2-019 | Scenario tree construction for PAR | 2 | 🔵 Not Started | Medium |

---

## Critical Path

```
PAR-V2-014 (Init) → PAR-V2-015 (State Transition) ────┐
                                                       ↓
PAR-V2-016 (Sample) → PAR-V2-017 (Transform) → [Forward Pass Complete]
         ↓
    PAR-V2-018 (Correlation)
         ↓
    PAR-V2-019 (Scenario Tree)
```

---

## State Evolution

**Critical Concept**: Lag state evolves through time:

```
t=0: state = [storage_0, lag_{-1}, lag_{-2}, ...]  (initial condition)
     ↓ solve stage 1, get inflow_1
t=1: state = [storage_1, inflow_1, lag_{-1}, ...]  (lag[0] updated)
     ↓ solve stage 2, get inflow_2
t=2: state = [storage_2, inflow_2, inflow_1, ...]  (lag[0] updated, lag[1]=old lag[0])
```

**Implementation**: Circular buffer automatically handles rotation.

---

## Innovation vs Inflow

**Key Distinction**:

**Innovation** (ε_t):
- Sampled from N(0,1)
- Stored in scenarios
- Applied via correlation
- Mean-zero by construction

**Inflow** (inflow_t):
- Computed from innovation + AR dynamics
- State-dependent (depends on lags)
- Not stored directly in scenarios
- Computed on-the-fly: `inflow = μ + σ·(Σ φ·lag) + σ·ε`

---

## Success Criteria

- [ ] Forward pass initializes with lag initial conditions
- [ ] State transitions correctly update lag buffers
- [ ] Innovation sampler generates N(0,1) variates
- [ ] Innovation-to-inflow transformation correct
- [ ] Correlation applied to innovations
- [ ] Scenario tree handles innovation-based scenarios
- [ ] End-to-end forward pass simulation succeeds

---

## Validation Tests

- [ ] **State evolution**: Track state through multiple stages, verify lag rotation
- [ ] **Innovation distribution**: Sample 10k innovations, verify mean≈0, var≈1
- [ ] **Transformation**: Verify inflow = μ + σ·(AR component) + σ·ε
- [ ] **Reproducibility**: Same seed → same scenario sequence
- [ ] **Correlation**: Verify correlated innovations have correct correlation matrix

---

## Dependencies

**Blocked By**: Sprint 3 (Cut Generation) - forward pass uses cuts

**Blocks**: Sprint 5 (Runtime) - needs working forward pass for SDDP loop

---

## Files to Edit

- `src/sddp/forward_pass.rs` - State initialization, transition logic
- `src/scenario.rs` - Innovation sampling, scenario generation
- `src/state.rs` - State transition methods
- `src/correlation_applicator.rs` - Extend for innovations (possibly)
- `tests/test_forward_pass.rs` - Add PAR forward pass tests
- `tests/test_scenario.rs` - Add innovation sampling tests

---

## Notes

### Ticket PAR-V2-015 is Critical

State transition is where lag buffers get updated. Bugs here break entire simulation.

**Test thoroughly**:
- Single stage transition
- Multi-stage sequences
- PAR(1), PAR(2), PAR(3)
- Mixed PAR and naive hydros

### Innovation Sampling Design

Consider using existing `rand` crate patterns:

```rust
use rand_distr::{Normal, Distribution};

let normal = Normal::new(0.0, 1.0).unwrap();
let innovation = normal.sample(&mut rng);
```

### Correlation Application

Existing `CorrelationApplicator` works with absolute values. May need adapter for innovations:

```rust
// Innovations are already N(0,1), just need Cholesky transformation
let correlated_innovations = cholesky_matrix * independent_innovations;
```

---

## Sprint Goal

**By end of Sprint 4**: Forward pass can simulate system forward in time with PAR models, generating state trajectories and objective values.

**Deliverable**: Complete forward pass implementation that can be integrated into SDDP loop in Sprint 5.
