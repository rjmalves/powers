# PAR-V2-015: Integrate PAR into Forward Pass State Transition

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 4 (Forward Pass & Scenario Handling)  
**Story Points**: 3  
**Priority**: 🔥🔥 Critical  
**Status**: 🔵 Not Started

---

## Context

Forward pass transitions state from stage t to t+1:

**Storage transition** (existing):
```
storage_{t+1} = solution.storage_{t}
```

**Lag transition** (NEW):
```
lag_{t+1}[0] = solution.inflow_t  (new realization becomes most recent lag)
lag_{t+1}[k] = lag_t[k-1]  for k > 0  (shift history)
```

This is handled by `StorageAndInflowState::update_from_solution` (PAR-V2-002), but needs integration into forward pass flow.

**Critical**: This is where lag states actually evolve during simulation. Bugs here break the entire simulation.

**References**:
- Code: `src/sddp/forward_pass.rs`
- Buffer update: PAR-V2-002
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 3, Section 3.2

---

## Acceptance Criteria

- [ ] Lag states transition correctly between stages
- [ ] Buffer rotation works (newest → lag[0], oldest discarded)
- [ ] State evolution tested numerically
- [ ] Forward pass completes successfully with PAR
- [ ] State trajectory is feasible and correct

---

## Tasks

### Implementation

- [ ] Verify `update_from_solution` called correctly in forward pass
- [ ] Ensure lag buffers updated before next stage
- [ ] Test state transition logic

### Testing

- [ ] **Test lag evolution through forward pass**
  ```rust
  #[test]
  fn test_lag_evolution_in_forward_pass() {
      // Initial: lag[0]=100, lag[1]=90
      // Stage 1 solution: inflow=110
      // After transition: lag[0]=110, lag[1]=100
      // Verify this propagates correctly
  }
  ```

- [ ] **Test multi-stage forward pass**
  ```rust
  #[test]
  fn test_forward_pass_multiple_stages() {
      // 3-stage problem
      // Track lag evolution through all stages
      // Verify lag[0] at stage t equals inflow from stage t-1
  }
  ```

- [ ] **Test PAR(p) for various p**
  ```rust
  #[test]
  fn test_lag_evolution_various_orders() {
      // Test PAR(1), PAR(2), PAR(3)
      // Verify buffer management for each
  }
  ```

### Documentation

- [ ] Document state transition logic
- [ ] Add diagram showing lag evolution

---

## Dependencies

### Blocked By

- ✅ PAR-V2-002: Buffer management
- ✅ PAR-V2-004: State trait update method

---

## Estimated Effort

**3 story points** (1-2 days)
