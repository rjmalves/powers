# PAR-V2-017: Transform Innovations to Inflow Realizations

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 4 (Forward Pass & Scenario Handling)  
**Story Points**: 3  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

Innovations (ε_t) must be transformed to inflow realizations (inflow_t) using the AR equation:

```
inflow_t = μ_m + σ_m · (Σ_{k=1}^p φ_{km} · lag_{t-k}) + σ_m · ε_t
```

This transformation happens:
1. **After** innovation sampling (PAR-V2-016)
2. **Before** setting subproblem RHS (PAR-V2-020)

**Key Insight**: This is the inverse of the residual calculation. Given lags and innovation, we reconstruct the inflow.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 3, Section 3.3
- Existing: Similar pattern in `src/stochastic_process.rs`

---

## Acceptance Criteria

- [ ] Transformation function implemented
- [ ] Uses current lag values from state
- [ ] Uses seasonal parameters (φ, μ, σ)
- [ ] Tested numerically against AR equation
- [ ] Handles all AR orders (p=1 to p=max)

---

## Tasks

### Implementation

- [ ] **Implement transformation function**
  ```rust
  impl StorageAndInflowState {
      pub fn innovation_to_inflow(
          &self,
          hydro_id: usize,
          season_id: usize,
          innovation: f64,
      ) -> f64 {
          let p = self.ar_orders[&hydro_id];
          let params = &self.ar_params[&(hydro_id, season_id)];
          
          let phi = &params[0..p];
          let mu = params[p];
          let sigma = params[p + 1];
          
          // AR component: Σ φ_k · lag[k]
          let ar_component: f64 = (0..p)
              .map(|k| phi[k] * self.get_lag(hydro_id, k))
              .sum();
          
          // inflow = μ + σ·(AR component) + σ·ε
          mu + sigma * ar_component + sigma * innovation
      }
  }
  ```

- [ ] **Integrate with scenario application**
  ```rust
  // In forward pass or scenario handling
  for (hydro_id, innovation) in scenario.innovations.iter() {
      let inflow = state.innovation_to_inflow(*hydro_id, season_id, *innovation);
      // Use inflow for RHS update
  }
  ```

### Testing

- [ ] **Test transformation correctness**
  ```rust
  #[test]
  fn test_innovation_to_inflow() {
      // PAR(2): φ=[0.6, 0.3], μ=100, σ=10
      // State: lag[0]=120, lag[1]=110
      // Innovation: ε=0.5
      // Expected: inflow = 100 + 10·(0.6·120 + 0.3·110) + 10·0.5
      //                  = 100 + 10·(72 + 33) + 5
      //                  = 100 + 1050 + 5 = 1155
      let state = create_test_state();
      let inflow = state.innovation_to_inflow(hydro_id, season_id, 0.5);
      assert!((inflow - 1155.0).abs() < 1e-10);
  }
  ```

- [ ] **Test inverse relationship**
  ```rust
  #[test]
  fn test_roundtrip_transformation() {
      // Given lag state and inflow realization
      // Compute residual (inflow - AR prediction) / σ
      // Transform back using innovation
      // Should recover original inflow
  }
  ```

- [ ] **Test various AR orders**
  ```rust
  #[test]
  fn test_transformation_various_orders() {
      // Test PAR(1), PAR(2), PAR(3)
  }
  ```

### Documentation

- [ ] Document transformation formula
- [ ] Explain AR equation components
- [ ] Add numerical example

---

## Dependencies

### Blocked By

- ✅ PAR-V2-016: Innovation sampling

### Blocks

- PAR-V2-020: RHS updates with innovations

---

## Estimated Effort

**3 story points** (1-2 days)
