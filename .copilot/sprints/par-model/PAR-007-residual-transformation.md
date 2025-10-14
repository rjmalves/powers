# PAR-007: Implement Residual-Based Marginal Transformation

## Context

CEPEL methodology requires marginal distribution (LogNormal3) applied to **residuals `aₜ`** before seasonal re-scaling, not to the final series. This ticket creates a residual transformer that applies LogNormal3 to correlated normal variates while preserving correlation structure.

Current: `Xₜ = LogNormal3(corrected_noise)` ❌  
CEPEL: `aₜ = LogNormal3(corrected_noise)`, then `Zₜ = μₘ + σₘ·(AR_term + aₜ)` ✅

## Acceptance Criteria

- [ ] `ResidualTransformer` applies marginal distribution to correlated noise
- [ ] Supports LogNormal3, Normal, and other distributions
- [ ] Preserves spatial correlation structure
- [ ] Integration with existing `MarginalDistribution` enum
- [ ] Tests verify correlation preservation
- [ ] Performance comparable to current marginal transformation

## Tasks

### Implementation

- [ ] Create transformation function in `src/marginal_transformer.rs`
- [ ] Add `transform_residuals(correlated_noise, distribution)` method
- [ ] Ensure vectorized operation for efficiency
- [ ] Update `src/scenario.rs` to use residual transformation stage

### Testing

- [ ] Unit test: LogNormal3 transformation produces positive residuals
- [ ] Unit test: Normal transformation preserves mean/std
- [ ] Unit test: Correlation matrix eigenvalues unchanged
- [ ] Integration test: Full pipeline (noise → correlation → residual transform → PAR)

### Documentation

- [ ] Clarify residual vs final series semantics
- [ ] Add CEPEL pipeline diagram
- [ ] Performance notes

## Technical Notes

```rust
/// Transform correlated normal variates to residuals via marginal distribution
///
/// # CEPEL Pipeline Stage 3
///
/// Input: b ~ MVN(0, Σ) [correlated Gaussian noise]
/// Output: a = F⁻¹(Φ(b)) [residuals with target marginal]
///
/// where:
///   Φ = standard normal CDF
///   F⁻¹ = inverse CDF of target distribution (e.g., LogNormal3)
pub fn transform_to_residuals(
    correlated_noise: &[f64],
    distribution: &MarginalDistribution,
) -> Vec<f64> {
    correlated_noise.iter().map(|&b| {
        let u = crate::utils::standard_normal_cdf(b); // Φ(b)
        distribution.inverse_cdf(u) // F⁻¹(u)
    }).collect()
}
```

## Dependencies

- **Blocked by**: PAR-003 (NoiseModel semantics)
- **Blocks**: PAR-009 (scenario integration)
- **Related**: PAR-006 (PAR generator)

## Estimated Effort

**2 story points** (1 day)
