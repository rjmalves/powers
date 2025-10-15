# PAR-V2-016: Add Innovation Sampling Capability

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 4 (Forward Pass & Scenario Handling)  
**Story Points**: 3  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

For PAR models, scenario realizations are innovations (ε_t) sampled from N(0,1), not absolute inflow values. We need:

1. **Sampling**: Generate ε_t ~ N(0,1)
2. **Transformation**: inflow_t = μ + σ·(Σ φ_k·lag[k]) + σ·ε_t
3. **Storage**: Scenario is (ε_t), not (inflow_t)

**Why Store Innovations**: Scenarios are mean-zero, allowing easier correlation across hydros and consistency with AR structure.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 3, Section 3.3
- Existing: `src/scenario.rs` → ScenarioGenerator

---

## Acceptance Criteria

- [ ] Innovation sampler implemented (samples ε ~ N(0,1))
- [ ] Sampler integrated with scenario generation
- [ ] Sampling respects random seed for reproducibility
- [ ] Correct distribution (mean=0, variance=1)
- [ ] Works for multiple hydros independently

---

## Tasks

### Implementation

- [ ] **Implement innovation sampler**
  ```rust
  pub struct InnovationSampler {
      rng: Rng,  // Random number generator
  }
  
  impl InnovationSampler {
      pub fn new(seed: Option<u64>) -> Self { ... }
      
      pub fn sample(&mut self) -> f64 {
          // Sample from N(0,1)
          self.rng.sample(StandardNormal)
      }
      
      pub fn sample_multiple(&mut self, n: usize) -> Vec<f64> {
          (0..n).map(|_| self.sample()).collect()
      }
  }
  ```

- [ ] **Integrate with ScenarioGenerator**
  ```rust
  // In scenario.rs
  pub enum ScenarioType {
      Absolute,  // Existing: absolute values
      Innovation,  // NEW: N(0,1) innovations
  }
  ```

- [ ] **Add seed management**
  - Ensure reproducibility
  - Allow seed specification in config

### Testing

- [ ] **Test N(0,1) distribution**
  ```rust
  #[test]
  fn test_innovation_distribution() {
      let mut sampler = InnovationSampler::new(Some(42));
      let samples: Vec<f64> = (0..10000).map(|_| sampler.sample()).collect();
      let mean = samples.iter().sum::<f64>() / samples.len() as f64;
      let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
      assert!((mean - 0.0).abs() < 0.05);  // Close to 0
      assert!((variance - 1.0).abs() < 0.05);  // Close to 1
  }
  ```

- [ ] **Test reproducibility**
  ```rust
  #[test]
  fn test_sampling_reproducibility() {
      let mut sampler1 = InnovationSampler::new(Some(123));
      let mut sampler2 = InnovationSampler::new(Some(123));
      for _ in 0..100 {
          assert_eq!(sampler1.sample(), sampler2.sample());
      }
  }
  ```

- [ ] **Test multiple hydro sampling**
  ```rust
  #[test]
  fn test_multi_hydro_sampling() {
      let mut sampler = InnovationSampler::new(Some(42));
      let innovations = sampler.sample_multiple(5);  // 5 hydros
      assert_eq!(innovations.len(), 5);
  }
  ```

### Documentation

- [ ] Document innovation vs absolute scenarios
- [ ] Explain N(0,1) sampling
- [ ] Document seed management

---

## Dependencies

### Blocked By

- None (can start immediately)

### Blocks

- PAR-V2-017: Transform innovations to inflows
- PAR-V2-018: Correlation application

---

## Estimated Effort

**3 story points** (1-2 days)
