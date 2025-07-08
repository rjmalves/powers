/// Helper function for evaluating the dot product between two vectors.
/// This implementation expect f64 slices and does not use any kind
/// of SSE operations. The slices are expected to have the same length.
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
}