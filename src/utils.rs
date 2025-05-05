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
