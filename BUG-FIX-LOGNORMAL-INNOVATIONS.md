# Bug Fix: Incorrect Lognormal Standard Deviation Calculation

**Date**: 2025-11-02  
**Issue**: Inflow values in simulation outputs were extremely high (100-700+ instead of expected 20-40)  
**Root Cause**: Using lognormal distribution's standard deviation instead of the log-space parameter sigma

## Problem Description

The simulation outputs showed inflow values that were orders of magnitude too large:
- Expected: ~20-40 (based on lognormal parameters mu=3.178, sigma=0.5)
- Actual: 162, 289, 778, etc.

Investigation revealed that the seasonal standard deviation (std_dev) was being set to ~14.49 instead of the log-space parameter sigma=0.5.

## Root Cause

### Lognormal Parameter Conversion (src/uncertainty_model.rs:143-149)

**Problem**: Converting lognormal parameters to the distribution's statistical mean and standard deviation.

```rust
// BEFORE (incorrect):
let mean = gamma + (mu + sigma.powi(2) / 2.0).exp();
let variance = (2.0 * mu + sigma.powi(2)).exp() * (sigma.powi(2).exp() - 1.0);
let std_dev = variance.sqrt();
```

This computed the mean and standard deviation of the **lognormal distribution itself**, which for mu=3.178, sigma=0.5 gives:
- mean ≈ 27.19 (statistical mean of the distribution)
- std_dev ≈ 14.49 (statistical standard deviation of the distribution)

**Solution**: Use lognormal parameters directly for the AR model formulation:

```rust
// AFTER (correct):
let mean = gamma + mu.exp();  // exp(mu) is the median (mode in log-space)
let std_dev = *sigma;         // Use sigma parameter directly (log-space std dev)
```

For mu=3.178, sigma=0.5:
- mean = exp(3.178) ≈ 24.0
- std_dev = 0.5

## Why This Fix Is Correct

### The AR Formulation

For AR models, the seasonal parameters (mean, std_dev) are used in the constraint:
```
Y_t = mean + std_dev * innovation + AR_terms
```

Where `innovation` is the sampled value (lognormal for LogNormal3 distributions).

### LogNormal3 Innovation Handling

The scenario generator deliberately uses transformed lognormal values as "innovations" to ensure non-negative inflows. This is a **controlled deviation** from canonical mathematical formulation:

```rust
// In scenario_generator.rs:
let innovation = params.distribution.transform(base_noise, 0.0, 1.0);
// For LogNormal3: innovation = gamma + exp(mu + sigma * base_noise)
```

This gives innovations with typical values around 20-40 for mu=3.178, sigma=0.5.

### The Bug

With the incorrect std_dev ≈ 14.49:
```
Y_t = 27.19 + 14.49 * (lognormal_value ~24) + AR_terms
    = 27.19 + 347.76 + ...
    = 374+ (way too high!)
```

With the correct std_dev = 0.5:
```
Y_t = 24.0 + 0.5 * (lognormal_value ~24) + AR_terms  
    = 24.0 + 12.0 + ...
    = ~36 (correct range!)
```

## Mathematical Explanation

The key insight is that for AR models with lognormal marginals, we're NOT working with the statistical properties of the lognormal distribution. Instead:

1. **Innovation sampling**: Sample from LogNormal3 to get non-negative values (controlled deviation)
2. **AR formulation**: Use `sigma` (log-space parameter) to scale these innovations, not the distribution's std_dev
3. **Mean**: Use `exp(mu)` (the median) as the reference point, not the distribution mean

This approach:
- ✅ Ensures non-negative inflows (via lognormal sampling)
- ✅ Uses AR parameters consistently with the log-space parameterization  
- ✅ Produces realistic inflow ranges

## Files Modified

1. **src/uncertainty_model.rs** (lines 143-149)
   - Changed lognormal parameter conversion
   - Now uses: mean = gamma + exp(mu), std_dev = sigma (direct parameter usage)
   - No longer computes statistical mean/variance of the distribution

2. **src/scenario_generator.rs** (NO CHANGES)
   - Kept original transformation approach for LogNormal3
   - This "breaks mathematical purity" but is the intended design to prevent negative inflows

## Impact

### Before Fix
```
std_dev=14.492833 (computed from distribution variance)
innovation=19.880877 (lognormal value)
stoch_term=135.699894 (14.49 * 19.88)
RHS=162.893993 (way too high!)
```

### After Fix
```
std_dev=0.500000 (sigma parameter directly)
innovation=19.880877 (same lognormal value)
stoch_term=9.940438 (0.5 * 19.88)  
RHS=33.939146 (correct range!)
```

## Testing

- ✅ All 307 tests pass
- ✅ Example 01-deterministic: inflow ≈ 20 (correct for exp(2.996))
- ✅ Example 02-stochastic: inflow ≈ 25-36 (correct range for lognormal(3.178, 0.5))
- ✅ Example 06-par-model: now runs successfully (was infeasible before)
- ✅ Example 07-par-model-with-inflow-state: now runs successfully
- ✅ All examples produce realistic output ranges

## Lessons Learned

1. **Parameter interpretation matters**: For AR models with lognormal marginals, use the log-space parameters (mu, sigma) directly, not the distribution's statistical properties.

2. **Controlled deviations can be correct**: The "breaking mathematical purity" approach of using lognormal values as innovations is a deliberate design choice to ensure non-negative inflows. The bug was in how we scaled these innovations, not in the transformation approach itself.

3. **Test with multiple examples**: The bug manifested as infeasibility in PAR model examples but as unrealistic values in stochastic examples. Testing across different scenarios helped identify the root cause.

## References

- PAR model formulation: par_derivation.pdf
- PERF-004: Optimize realize_uncertainties (where this bug manifested)
- User feedback: Correctly identified that the std_dev calculation was the issue, not the transformation approach

---

**Status**: ✅ Fixed and tested  
**Tests**: 307/307 passing  
**Examples**: All 7 examples producing correct outputs
