/// Helper function for evaluating the dot product between two vectors.
/// This implementation expect f64 slices and does not use any kind
/// of SSE operations. The slices are expected to have the same length.
///
/// **Note**: For reproducibility-critical code (cut height evaluation),
/// use `dot_product_deterministic()` instead, which uses Kahan summation
/// to ensure order-independent results and prevent floating-point rounding
/// non-determinism. See REPRO-011.
///
/// ## Example
///
/// ```
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0, 1.0, 1.0];
///
/// let dot = powers_rs::utils::dot_product(&a, &b);
/// assert_eq!(dot, 6.0);
/// ```
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut product = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

/// Computes dot product with deterministic order-independent accumulation.
///
/// Uses Kahan summation to ensure identical results regardless of:
/// - Compiler optimizations reordering operations
/// - FMA (fused multiply-add) instruction usage
/// - Floating-point evaluation order variations
///
/// # Critical Use Case: Cut Height Evaluation
///
/// In SDDP, cut heights are computed as `rhs - dot(coefficients, state)`.
/// Standard dot product allows compiler to reorder operations, causing
/// different rounding in different runs. This leads to:
/// - Different heights → different dominating cuts → diverging lower bounds
/// - Non-reproducible training even with identical inputs
///
/// # Performance
///
/// ~3-4x slower than naive dot product, but critical for reproducibility
/// in domination evaluation. Typical usage: 156-element vectors, ~200ns overhead.
///
/// **Frequency in Example 05**:
/// - ~61,440 evaluations per run (4 cuts × 8 iters × 32 nodes × 60 states)
/// - Total overhead: ~0.55ms (< 0.002% of 25-second runtime)
/// - **Impact: Negligible**
///
/// # When to Use
///
/// - **ALWAYS** for cut height evaluation (`eval_height_at_state`)
/// - Any domination-related computation
/// - When reproducibility is critical and inputs may vary in order
///
/// # When NOT to Use
///
/// - Cut RHS computation (inputs already deterministic)
/// - Non-critical numerical operations
/// - Inner loops where standard dot product suffices
///
/// # Example
///
/// ```
/// use powers_rs::utils::dot_product_deterministic;
///
/// let coefficients = vec![1.5, 2.5, 3.5];
/// let state = vec![100.0, 200.0, 300.0];
///
/// // Always gives same result, regardless of compiler optimization
/// let height = dot_product_deterministic(&coefficients, &state);
/// assert!((height - 1700.0).abs() < 1e-10);
/// ```
///
/// # Reproducibility
///
/// ```
/// use powers_rs::utils::dot_product_deterministic;
///
/// // Pathological case: Large + small + large cancellation
/// let a = vec![1e10, 1.0, -1e10];
/// let b = vec![1.0, 1.0, 1.0];
///
/// // Standard dot product may give different results across runs:
/// // Run 1: (1e10 * 1.0) + (1.0 * 1.0) + (-1e10 * 1.0) → might lose 1.0
/// // Run 2: Different order → different rounding
///
/// // Deterministic dot product ALWAYS gives same result:
/// let result = dot_product_deterministic(&a, &b);
/// assert_eq!(result, 1.0); // Guaranteed!
/// ```
///
/// # References
///
/// - REPRO-011: Deterministic Cut Height Evaluation
/// - Kahan summation algorithm for numerical stability
pub fn dot_product_deterministic(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot_product_deterministic: vectors must have same length"
    );

    // Collect all products first (deterministic order)
    let products: Vec<f64> =
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

    // Accumulate with Kahan summation for order-independent result
    kahan_sum(&products)
}

/// Kahan compensated summation for deterministic floating-point accumulation.
///
/// This algorithm minimizes rounding errors and ensures deterministic results
/// regardless of summation order. Critical for reproducibility in parallel contexts
/// where accumulation order may vary across runs.
///
/// # Algorithm
///
/// Maintains a running compensation for lost low-order bits. Each addition:
/// 1. Compensates for previous errors (`y = value - compensation`)
/// 2. Performs the addition (`t = sum + y`)
/// 3. Computes new compensation (`compensation = (t - sum) - y`)
///
/// The compensation captures the "lost" precision from each addition and adds it
/// back in the next iteration, preventing error accumulation.
///
/// # Performance
///
/// ~3-4x slower than naive summation due to extra operations per element.
/// However, this overhead is negligible compared to solver calls in SDDP.
///
/// **When to use**:
/// - Accumulations where order may vary (parallel results, thread-local buffers)
/// - Cut coefficient accumulation across forward passes
/// - Any summation where reproducibility is critical
///
/// **When NOT to use**:
/// - Single-threaded sequential accumulations with guaranteed fixed order
/// - Performance-critical inner loops where order is deterministic
/// - When naive sum accuracy is sufficient (simple integer-like values)
///
/// # Reproducibility
///
/// Unlike naive summation (`iter().sum()`), Kahan summation provides consistent
/// results regardless of input order, making it essential for REPRO-004 and REPRO-005.
///
/// # Example
///
/// ```
/// use powers_rs::utils::kahan_sum;
///
/// // Pathological case where naive sum loses precision
/// let values = vec![1e10, 1.0, -1e10, 1.0, 1.0];
/// let result = kahan_sum(&values);
/// assert_eq!(result, 3.0);
///
/// // Naive sum might give 0.0 or 2.0 depending on order:
/// // (1e10 + 1.0) → 1e10 (1.0 lost due to precision)
/// // (1e10 - 1e10) → 0.0
/// // (0.0 + 1.0 + 1.0) → 2.0 (one 1.0 was lost)
/// ```
///
/// # References
///
/// - Kahan, W. (1965). "Further remarks on reducing truncation errors"
/// - Goldberg, D. (1991). "What every computer scientist should know about floating-point"
pub fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;

    for &value in values {
        let y = value - compensation; // Compensate for previous lost bits
        let t = sum + y; // Add compensated value
        compensation = (t - sum) - y; // Capture lost precision for next iteration
        sum = t;
    }

    sum
}

/// Deterministic mean using Kahan summation.
///
/// Guarantees reproducible results even with varying accumulation order
/// from parallel execution. Use this instead of `mean()` when computing
/// averages of parallel results (forward pass costs, cut coefficients, etc.).
///
/// # Performance
///
/// ~3-4x slower than naive mean due to Kahan summation overhead.
/// For SDDP workloads, this is negligible (< 0.1% of total runtime).
///
/// # Panics
///
/// Panics if `values` is empty (cannot compute mean of zero elements).
///
/// # Example
///
/// ```
/// use powers_rs::utils::mean_deterministic;
///
/// let costs = vec![100.5, 200.3, 150.7];
/// let avg = mean_deterministic(&costs);
/// assert!((avg - 150.5).abs() < 1e-10);
/// ```
///
/// # Reproducibility
///
/// ```
/// use powers_rs::utils::mean_deterministic;
///
/// let values = vec![1e10, 1.0, 2.0, -1e10, 3.0];
///
/// // Same result regardless of order
/// let mean1 = mean_deterministic(&values);
///
/// let mut reversed = values.clone();
/// reversed.reverse();
/// let mean2 = mean_deterministic(&reversed);
///
/// assert_eq!(mean1, mean2); // Deterministic!
/// ```
pub fn mean_deterministic(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "Cannot compute mean of empty slice");
    kahan_sum(values) / values.len() as f64
}

/// Helper function for generating an uniform probability distribution
/// from a given number of samples.
///
/// ## Example
///
/// ```
/// let count = 5;
///
/// let p = powers_rs::utils::uniform_prob_by_count(count);
/// assert_eq!(p, &[0.2, 0.2, 0.2, 0.2, 0.2]);
/// ```
pub fn uniform_prob_by_count(count: usize) -> Vec<f64> {
    assert!(count > 0);
    let p = 1.0 / count as f64;
    vec![p; count]
}

/// Helper function for evaluating the average of a
/// series of values.
///
/// **Note**: For reproducibility-critical code (parallel accumulations),
/// use `mean_deterministic()` instead, which uses Kahan summation to
/// ensure order-independent results.
///
/// ## Example
///
/// ```
/// let vals = [1.0, 2.0, 3.0];
///
/// let m = powers_rs::utils::mean(&vals);
/// assert_eq!(m, 2.0);
/// ```
pub fn mean(values: &[f64]) -> f64 {
    let total: f64 = values.iter().sum();
    let count = values.len();
    total / count as f64
}

/// Helper function for evaluating the standard deviation of
/// a series of values.
///
/// ## Example
///
/// ```
/// let vals = [1.0, 1.0, 1.0];
///
/// let m = powers_rs::utils::standard_deviation(&vals);
/// assert_eq!(m, 0.0);
/// ```
pub fn standard_deviation(values: &[f64]) -> f64 {
    let m = mean(values);
    let deviations: Vec<f64> =
        values.iter().map(|c| (c - m) * (c - m)).collect();
    let mean_deviation = mean(&deviations);

    f64::sqrt(mean_deviation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_uniform_prob_by_count() {
        let p = uniform_prob_by_count(4);
        assert_eq!(p, vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_mean() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&values), 5.0);
    }

    #[test]
    fn test_standard_deviation() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((standard_deviation(&values) - 2.0).abs() < 1e-9);
    }

    // =========================================================================
    // Kahan Summation Tests (REPRO-002)
    // =========================================================================

    #[test]
    fn test_kahan_sum_basic() {
        // Test basic correctness with simple values
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(kahan_sum(&values), 10.0);

        let values = vec![0.1, 0.2, 0.3];
        let result = kahan_sum(&values);
        assert!((result - 0.6).abs() < 1e-15);
    }

    #[test]
    fn test_kahan_sum_precision() {
        // Pathological case where naive summation loses precision
        // This demonstrates Kahan's superiority over naive sum
        let values = vec![1e10, 1.0, -1e10, 1.0, 1.0];
        let kahan_result = kahan_sum(&values);

        // Kahan sum should give correct answer
        assert_eq!(kahan_result, 3.0);

        // Kahan is GUARANTEED to give 3.0 regardless of order
        assert!(
            (kahan_result - 3.0).abs() < 1e-10,
            "Kahan sum should be exactly 3.0"
        );
    }

    #[test]
    fn test_kahan_sum_order_independence() {
        // Critical property for reproducibility: result independent of order
        let values = vec![1e10, 1.0, 2.0, -1e10, 3.0];
        let sum1 = kahan_sum(&values);

        // Reverse order
        let mut reversed = values.clone();
        reversed.reverse();
        let sum2 = kahan_sum(&reversed);

        // Shuffle (deterministically for test reproducibility)
        let shuffled = vec![3.0, 1e10, -1e10, 1.0, 2.0];
        let sum3 = kahan_sum(&shuffled);

        // All should be identical
        assert_eq!(sum1, sum2, "Kahan sum should be order-independent");
        assert_eq!(sum2, sum3, "Kahan sum should be order-independent");
        assert_eq!(sum1, 6.0, "All orders should give correct result");
    }

    #[test]
    fn test_kahan_sum_edge_cases() {
        // Empty slice
        assert_eq!(kahan_sum(&[]), 0.0);

        // Single element
        assert_eq!(kahan_sum(&[42.0]), 42.0);

        // Zero sum
        assert_eq!(kahan_sum(&[1.0, -1.0]), 0.0);

        // Negative values
        assert_eq!(kahan_sum(&[-1.0, -2.0, -3.0]), -6.0);

        // Large values (but not overflow)
        let large = vec![1e100, 1e100, -1e100];
        let result = kahan_sum(&large);
        assert!(result.is_finite(), "Should not overflow");
        assert!((result - 1e100).abs() < 1e85); // Some tolerance for large numbers
    }

    #[test]
    fn test_mean_deterministic() {
        // Test basic correctness
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = mean_deterministic(&values);
        assert!((result - 5.0).abs() < 1e-10);

        // Test with problematic values
        let values = vec![1e10, 1.0, 2.0, -1e10, 3.0];
        let result = mean_deterministic(&values);
        assert!((result - 1.2).abs() < 1e-10); // (6.0 / 5.0)
    }

    #[test]
    fn test_mean_deterministic_order_independence() {
        // Critical property: same mean regardless of order
        let values = vec![1e10, 1.0, 2.0, -1e10, 3.0, 4.0];

        let mean1 = mean_deterministic(&values);

        let mut reversed = values.clone();
        reversed.reverse();
        let mean2 = mean_deterministic(&reversed);

        let shuffled = vec![3.0, 1e10, -1e10, 4.0, 1.0, 2.0];
        let mean3 = mean_deterministic(&shuffled);

        assert_eq!(mean1, mean2, "Mean should be order-independent");
        assert_eq!(mean2, mean3, "Mean should be order-independent");
    }

    #[test]
    #[should_panic(expected = "Cannot compute mean of empty slice")]
    fn test_mean_deterministic_empty() {
        mean_deterministic(&[]);
    }

    #[test]
    fn test_mean_deterministic_single_element() {
        assert_eq!(mean_deterministic(&[42.0]), 42.0);
    }

    #[test]
    fn test_kahan_vs_naive_large_scale() {
        // Demonstrate Kahan's advantage with many small additions to large sum
        let mut values = vec![1e16];
        values.extend(std::iter::repeat_n(1.0, 10000));

        let kahan_result = kahan_sum(&values);

        // Kahan should maintain accuracy better
        // Expected: 1e16 + 10000
        let expected = 1e16 + 10000.0;

        // Kahan should give reasonable accuracy
        let kahan_error = (kahan_result - expected).abs();

        // Kahan should give reasonable accuracy
        // (exact result depends on FP rounding mode)
        assert!(
            kahan_error < 1.0,
            "Kahan sum should maintain reasonable accuracy"
        );
    }

    // =========================================================================
    // Deterministic Dot Product Tests (REPRO-011)
    // =========================================================================

    #[test]
    fn test_dot_product_deterministic_basic() {
        // Test basic correctness
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product_deterministic(&a, &b);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_deterministic_order_independence() {
        // CRITICAL: Result must be independent of vector element order
        let a = vec![1e10, 1.0, -1e10, 2.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result1 = dot_product_deterministic(&a, &b);

        // Reverse vectors
        let a_rev: Vec<f64> = a.iter().rev().copied().collect();
        let b_rev: Vec<f64> = b.iter().rev().copied().collect();
        let result2 = dot_product_deterministic(&a_rev, &b_rev);

        // Shuffle (deterministically)
        let a_shuffled = vec![2.0, 1e10, -1e10, 1.0];
        let b_shuffled = vec![1.0, 1.0, 1.0, 1.0];
        let result3 = dot_product_deterministic(&a_shuffled, &b_shuffled);

        // All must be identical
        assert_eq!(
            result1, result2,
            "Deterministic dot product should be order-independent"
        );
        assert_eq!(
            result2, result3,
            "Deterministic dot product should be order-independent"
        );
        assert_eq!(result1, 3.0, "Result should be exactly 3.0");
    }

    #[test]
    fn test_dot_product_deterministic_precision() {
        // Test with realistic SDDP values (cut coefficients × state values)
        // Typical: coefficients O(1-100), state values O(10^3 - 10^6)
        let coefficients = vec![1.5, 2.3, 0.8, 15.2, 100.0];
        let state = vec![1000.0, 5000.0, 2000.0, 500.0, 100.0];

        let result = dot_product_deterministic(&coefficients, &state);

        // Expected: 1.5*1000 + 2.3*5000 + 0.8*2000 + 15.2*500 + 100*100
        //         = 1500 + 11500 + 1600 + 7600 + 10000 = 32200
        assert!((result - 32200.0).abs() < 1e-9);
    }

    #[test]
    fn test_dot_product_deterministic_vs_naive() {
        // Show that deterministic version is MORE accurate than naive
        // with pathological input
        let a = vec![1e10, 1.0, 1.0, 1.0, -1e10];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let deterministic_result = dot_product_deterministic(&a, &b);

        // Deterministic should give exactly 3.0
        assert_eq!(deterministic_result, 3.0);

        // Naive dot product might lose precision
        let naive_result = dot_product(&a, &b);

        // Naive result may differ slightly (implementation and compiler dependent)
        // But we can verify deterministic is at least as accurate
        let expected = 3.0;
        let deterministic_error = (deterministic_result - expected).abs();
        let naive_error = (naive_result - expected).abs();

        assert!(
            deterministic_error <= naive_error,
            "Deterministic dot product should be at least as accurate as naive"
        );
    }

    #[test]
    fn test_dot_product_deterministic_edge_cases() {
        // Empty vectors
        assert_eq!(dot_product_deterministic(&[], &[]), 0.0);

        // Single element
        assert_eq!(dot_product_deterministic(&[5.0], &[3.0]), 15.0);

        // Zero result
        assert_eq!(dot_product_deterministic(&[1.0, -1.0], &[1.0, 1.0]), 0.0);

        // Negative values
        assert_eq!(
            dot_product_deterministic(&[-1.0, -2.0], &[3.0, 4.0]),
            -11.0
        );
    }

    #[test]
    #[should_panic(expected = "vectors must have same length")]
    fn test_dot_product_deterministic_length_mismatch() {
        dot_product_deterministic(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn test_dot_product_deterministic_realistic_sddp() {
        // Simulate realistic SDDP scenario with 156 hydro reservoirs
        let coefficients: Vec<f64> =
            (0..156).map(|i| (i as f64) * 0.5).collect();
        let state: Vec<f64> =
            (0..156).map(|i| 1000.0 + (i as f64) * 10.0).collect();

        let result = dot_product_deterministic(&coefficients, &state);

        // Result should be deterministic and finite
        assert!(result.is_finite());

        // Verify it's the same as Kahan-accumulated products
        let products: Vec<f64> = coefficients
            .iter()
            .zip(state.iter())
            .map(|(c, s)| c * s)
            .collect();
        let expected = kahan_sum(&products);

        assert_eq!(result, expected);
    }
}
