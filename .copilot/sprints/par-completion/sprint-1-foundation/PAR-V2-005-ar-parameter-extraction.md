# PAR-V2-005: Extract AR Parameters from Config

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 1 (Foundation)  
**Story Points**: 2  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

AR parameters (φ_k, μ, σ) are specified in `system.json` or `graph.json`. We need to extract them and organize for efficient lookup during subproblem construction.

**Storage Format**: `HashMap<(hydro_id, season_id), Vec<f64>>`
- Key: (hydro_id, season_id)
- Value: [φ_1, φ_2, ..., φ_p, μ, σ] (p+2 elements)

**References**:
- Input specification: `docs/reference/INPUT-SPECIFICATION.md`
- Existing pattern: `src/input.rs` → `SystemInput`, `GraphInput`
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 1, Section 1.1

---

## Acceptance Criteria

- [ ] AR parameters extracted from system/graph config
- [ ] Organized in HashMap with (hydro_id, season_id) keys
- [ ] Validation: all PAR hydros have parameters for all seasons
- [ ] Validation: parameter vector has correct length (p+2)
- [ ] Validation: σ > 0, -1 < φ_k < 1 (stationarity)

---

## Tasks

### Implementation

- [ ] **Implement extraction function**
  ```rust
  pub fn extract_ar_parameters(
      system: &SystemInput,
      graph: &GraphInput,
  ) -> Result<HashMap<(usize, usize), Vec<f64>>, Error> {
      let mut params = HashMap::new();
      
      for hydro in &system.hydros {
          if let Some(process) = &hydro.inflow_process {
              if let StochasticProcessType::PAR { order, coefficients } = &process.process_type {
                  for (season_id, coef_set) in coefficients.iter().enumerate() {
                      let key = (hydro.id, season_id);
                      validate_ar_parameters(coef_set, *order)?;
                      params.insert(key, coef_set.clone());
                  }
              }
          }
      }
      
      Ok(params)
  }
  ```

- [ ] **Implement validation function**
  ```rust
  fn validate_ar_parameters(params: &[f64], order: usize) -> Result<(), Error> {
      if params.len() != order + 2 {
          return Err(Error::InvalidParameterCount { ... });
      }
      
      let phi = &params[0..order];
      let sigma = params[order + 1];
      
      // Check stationarity
      if !is_stationary(phi) {
          return Err(Error::NonStationaryAR { ... });
      }
      
      // Check sigma > 0
      if sigma <= 0.0 {
          return Err(Error::InvalidSigma { ... });
      }
      
      Ok(())
  }
  ```

- [ ] **Implement stationarity check**
  ```rust
  fn is_stationary(phi: &[f64]) -> bool {
      // For PAR(1): -1 < φ < 1
      // For PAR(2): φ_2 + φ_1 < 1, φ_2 - φ_1 < 1, -1 < φ_2 < 1
      // For higher orders: check characteristic polynomial roots
      // Start with simple bounds check
      phi.iter().all(|&p| p.abs() < 1.0)
  }
  ```

### Testing

- [ ] **Test extraction success**
- [ ] **Test validation catches invalid parameters**
- [ ] **Test stationarity check**
- [ ] **Test missing parameter error**

### Documentation

- [ ] Document parameter format
- [ ] Document validation rules

---

## Dependencies

### Blocked By

- ✅ PAR-V2-001: Struct exists

### Blocks

- PAR-V2-003: Constructor (needs extraction function)
- PAR-V2-007: AR constraints (needs parameters)

---

## Estimated Effort

**2 story points** (1 day)
