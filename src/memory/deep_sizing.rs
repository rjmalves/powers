//! Deep memory estimation for accurate allocation tracking.
//!
//! This module provides the `DeepSizeEstimate` trait for computing the total heap
//! memory usage of types, including nested allocations. This addresses a critical
//! limitation discovered during TICKET-006: shallow estimation using `std::mem::size_of`
//! only measures stack size, missing heap allocations in nested structures like
//! `Vec<f64>` fields.
//!
//! # The Problem: Shallow vs Deep Estimation
//!
//! **Shallow estimation** (using `std::mem::size_of`):
//! ```rust,ignore
//! std::mem::size_of::<BendersCut>()  // Returns: 56 bytes (stack only)
//! ```
//!
//! **Reality** (with nested allocations):
//! ```rust,ignore
//! struct BendersCut {
//!     coefficients: Vec<f64>,  // ← 156 × 8 = 1,248 bytes HEAP (NOT COUNTED!)
//!     // ... other fields
//! }
//! // Actual: 1,304 bytes per cut (23× underestimate!)
//! ```
//!
//! # Solution: Deep Estimation
//!
//! The `DeepSizeEstimate` trait provides two methods for memory estimation:
//!
//! 1. **Dynamic** (`estimate_heap_bytes(&self)`): For actual instances, uses real capacities
//! 2. **Static** (`estimate_heap_bytes_static(sizing)`): For planning, uses max expected sizes
//!
//! # Implementation Hierarchy
//!
//! ## Level 1: Primitives (Zero Heap)
//! ```rust,ignore
//! impl DeepSizeEstimate for f64 {
//!     fn estimate_heap_bytes(&self, _: &SizingInfo) -> usize {
//!         std::mem::size_of::<f64>()  // Stack only, no heap
//!     }
//! }
//! ```
//!
//! ## Level 2: Collections (Sized by Context)
//! ```rust,ignore
//! impl DeepSizeEstimate for Vec<f64> {
//!     fn estimate_heap_bytes(&self, _: &SizingInfo) -> usize {
//!         self.capacity() * std::mem::size_of::<f64>()
//!     }
//!     
//!     fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
//!         sizing.max_state_dimension * std::mem::size_of::<f64>()
//!     }
//! }
//! ```
//!
//! ## Level 3: Domain Types (Recursive)
//! ```rust,ignore
//! impl DeepSizeEstimate for BendersCut {
//!     fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
//!         std::mem::size_of::<Self>() +  // Stack
//!         self.coefficients.estimate_heap_bytes(sizing)  // Heap (recursive)
//!     }
//! }
//! ```
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use powers_rs::memory::{DeepSizeEstimate, SizingInfo};
//!
//! // Get sizing context
//! let sizing = SizingInfo::from_input(&system, &graph, &config);
//!
//! // Estimate for existing instance
//! let cut = BendersCut { ... };
//! let actual_bytes = cut.estimate_heap_bytes(&sizing);
//!
//! // Estimate for planning (before allocation)
//! let expected_bytes = BendersCut::estimate_heap_bytes_static(&sizing);
//!
//! // Pre-allocate with accurate size
//! let mut cuts = Vec::with_capacity(num_cuts);
//! let total_memory = num_cuts * expected_bytes;
//! ```
//!
//! # Performance Considerations
//!
//! - **Static estimation**: Should be fast (<1ms) for planning purposes
//! - **Dynamic estimation**: May traverse structures, acceptable for profiling
//! - **No allocation**: Estimation itself must not allocate memory
//! - **Cache-friendly**: Linear traversal where possible
//!
//! # Validation
//!
//! Deep estimation accuracy should be within 10% of actual memory usage because:
//! - Actual memory includes allocator overhead
//! - Rust collections may over-allocate for growth
//! - System allocator varies by platform
//!
//! See `MEMORY_OPTIMIZATION_STRATEGY.md` for complete design and validation strategy.

use super::SizingInfo;

/// Estimate total heap memory including nested allocations.
///
/// This trait provides accurate memory estimation for types that contain
/// heap-allocated data structures. Unlike `std::mem::size_of`, which only
/// measures stack size, this trait recursively computes the total memory
/// footprint including all nested allocations.
///
/// # Why Two Methods?
///
/// - **Dynamic** (`estimate_heap_bytes`): For actual instances, uses real capacities.
///   Useful for profiling and validation.
///
/// - **Static** (`estimate_heap_bytes_static`): For planning, uses maximum expected sizes.
///   Useful for buffer pre-allocation and capacity planning.
///
/// # Implementation Guide
///
/// ## For Primitives
/// Return stack size only (no heap allocations):
/// ```rust,ignore
/// impl DeepSizeEstimate for f64 {
///     fn estimate_heap_bytes(&self, _: &SizingInfo) -> usize {
///         std::mem::size_of::<f64>()
///     }
///     
///     fn estimate_heap_bytes_static(_: &SizingInfo) -> usize {
///         std::mem::size_of::<f64>()
///     }
/// }
/// ```
///
/// ## For Collections
/// Account for capacity and element sizes:
/// ```rust,ignore
/// impl DeepSizeEstimate for Vec<f64> {
///     fn estimate_heap_bytes(&self, _: &SizingInfo) -> usize {
///         // Dynamic: use actual capacity
///         self.capacity() * std::mem::size_of::<f64>()
///     }
///     
///     fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
///         // Static: use maximum expected size from context
///         sizing.max_state_dimension * std::mem::size_of::<f64>()
///     }
/// }
/// ```
///
/// ## For Domain Types
/// Recursively sum stack and nested heap:
/// ```rust,ignore
/// impl DeepSizeEstimate for MyStruct {
///     fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
///         std::mem::size_of::<Self>() +  // Stack size
///         self.field1.estimate_heap_bytes(sizing) +  // Nested heap
///         self.field2.estimate_heap_bytes(sizing)    // Nested heap
///     }
///     
///     fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
///         std::mem::size_of::<Self>() +
///         Field1Type::estimate_heap_bytes_static(sizing) +
///         Field2Type::estimate_heap_bytes_static(sizing)
///     }
/// }
/// ```
///
/// # Context via SizingInfo
///
/// The `SizingInfo` parameter provides context for sizing decisions:
/// - `max_state_dimension`: For coefficient vectors
/// - `num_forward_passes`: For trajectory counts
/// - `num_stages`: For stage-dependent structures
/// - `num_threads`: For thread-local buffers
///
/// This avoids hardcoding magic numbers and enables accurate estimation
/// based on actual problem configuration.
pub trait DeepSizeEstimate {
    /// Estimate heap bytes for this instance (dynamic, uses actual capacities).
    ///
    /// This method computes the actual heap memory used by this instance,
    /// including all nested allocations. It uses real capacities from
    /// existing data structures.
    ///
    /// # Use Cases
    /// - Profiling: Measure actual memory usage
    /// - Validation: Compare estimate to measured usage
    /// - Debugging: Understand memory footprint
    ///
    /// # Parameters
    /// - `sizing`: Context for computing nested sizes
    ///
    /// # Returns
    /// Total bytes of heap memory (stack + heap)
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize;

    /// Estimate heap bytes for this type (static, uses maximum expected sizes).
    ///
    /// This method computes the expected heap memory for an instance of this
    /// type using maximum expected sizes from the problem configuration.
    /// Used for pre-allocation planning.
    ///
    /// # Use Cases
    /// - Buffer pre-allocation: Size buffers before allocation
    /// - Capacity planning: Estimate total memory needs
    /// - Performance planning: Understand memory requirements
    ///
    /// # Parameters
    /// - `sizing`: Context providing maximum expected sizes
    ///
    /// # Returns
    /// Expected bytes of heap memory (conservative estimate)
    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize
    where
        Self: Sized;
}

// ============================================================================
// Level 1: Primitive Type Implementations
// ============================================================================

/// Primitives have zero heap allocation (stack only).
macro_rules! impl_deep_size_primitive {
    ($($t:ty),*) => {
        $(
            impl DeepSizeEstimate for $t {
                fn estimate_heap_bytes(&self, _sizing: &SizingInfo) -> usize {
                    std::mem::size_of::<$t>()
                }

                fn estimate_heap_bytes_static(_sizing: &SizingInfo) -> usize {
                    std::mem::size_of::<$t>()
                }
            }
        )*
    };
}

impl_deep_size_primitive!(
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
    bool, char
);

// ============================================================================
// Level 2: Collection Implementations
// ============================================================================

/// Vec<T> where T has no nested heap allocations (e.g., Vec<f64>).
///
/// For vectors of primitives, we only need to account for the vector's
/// capacity, not recursive element sizes.
impl<T> DeepSizeEstimate for Vec<T>
where
    T: DeepSizeEstimate,
{
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        // Vec overhead: capacity × size_of<T>
        let vec_overhead = self.capacity() * std::mem::size_of::<T>();
        
        // For primitives, elements have no heap allocations
        // For complex types, recursively sum element heap sizes
        let elements_heap: usize = if std::mem::size_of::<T>() <= 16 {
            // Optimization: primitives and small types have no heap
            0
        } else {
            // Complex types: recursively compute
            self.iter()
                .map(|item| item.estimate_heap_bytes(sizing))
                .sum()
        };
        
        vec_overhead + elements_heap
    }

    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        // Conservative estimate: use max state dimension as default size
        // Specific types can provide more accurate estimates
        let estimated_capacity = sizing.max_state_dimension;
        let vec_overhead = estimated_capacity * std::mem::size_of::<T>();
        
        // For complex element types, estimate their heap usage
        let per_element_heap = if std::mem::size_of::<T>() <= 16 {
            0
        } else {
            T::estimate_heap_bytes_static(sizing)
        };
        
        vec_overhead + (estimated_capacity * per_element_heap)
    }
}

/// String implementation.
///
/// Strings are heap-allocated, so we need to account for their capacity.
impl DeepSizeEstimate for String {
    fn estimate_heap_bytes(&self, _sizing: &SizingInfo) -> usize {
        std::mem::size_of::<Self>() + self.capacity()
    }

    fn estimate_heap_bytes_static(_sizing: &SizingInfo) -> usize {
        // Conservative estimate: 64 bytes for typical strings
        std::mem::size_of::<Self>() + 64
    }
}

/// Box<T> implementation.
///
/// Boxed values are always heap-allocated.
impl<T: DeepSizeEstimate> DeepSizeEstimate for Box<T> {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        std::mem::size_of::<Self>() + 
        std::mem::size_of::<T>() +
        self.as_ref().estimate_heap_bytes(sizing)
    }

    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        std::mem::size_of::<Self>() +
        std::mem::size_of::<T>() +
        T::estimate_heap_bytes_static(sizing)
    }
}

/// Option<T> implementation.
impl<T: DeepSizeEstimate> DeepSizeEstimate for Option<T> {
    fn estimate_heap_bytes(&self, sizing: &SizingInfo) -> usize {
        std::mem::size_of::<Self>() + 
        match self {
            Some(val) => val.estimate_heap_bytes(sizing),
            None => 0,
        }
    }

    fn estimate_heap_bytes_static(sizing: &SizingInfo) -> usize {
        std::mem::size_of::<Self>() + T::estimate_heap_bytes_static(sizing)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_sizing() -> SizingInfo {
        // Create minimal sizing for tests
        let sizing = SizingInfo {
            node_sizing: vec![],
            max_state_dimension: 100,
            min_state_dimension: 10,
            avg_state_dimension: 50.0,
            max_scenarios_per_node: 10,
            max_subproblem_vars: 300,
            num_hydros: 10,
            num_thermals: 5,
            num_buses: 5,
            num_lines: 8,
            num_stages: 5,
            num_nodes: 10,
            max_iterations: 50,
            num_forward_passes: 10,
            num_simulations: 100,
            num_threads: 4,
        };
        sizing
    }

    #[test]
    fn test_primitive_estimation() {
        let sizing = make_test_sizing();
        
        // Primitives return their size
        let f = 42.0_f64;
        assert_eq!(f.estimate_heap_bytes(&sizing), std::mem::size_of::<f64>());
        assert_eq!(f64::estimate_heap_bytes_static(&sizing), std::mem::size_of::<f64>());
        
        let i = 42_usize;
        assert_eq!(i.estimate_heap_bytes(&sizing), std::mem::size_of::<usize>());
    }

    #[test]
    fn test_vec_f64_estimation() {
        let sizing = make_test_sizing();
        
        // Empty vec: zero capacity
        let empty: Vec<f64> = Vec::new();
        assert_eq!(empty.estimate_heap_bytes(&sizing), 0);
        
        // Vec with capacity
        let mut with_capacity: Vec<f64> = Vec::with_capacity(100);
        with_capacity.push(1.0);
        with_capacity.push(2.0);
        
        let expected = 100 * std::mem::size_of::<f64>();
        assert_eq!(with_capacity.estimate_heap_bytes(&sizing), expected);
        
        // Static estimation uses max_state_dimension
        let static_expected = sizing.max_state_dimension * std::mem::size_of::<f64>();
        assert_eq!(Vec::<f64>::estimate_heap_bytes_static(&sizing), static_expected);
    }

    #[test]
    fn test_string_estimation() {
        let sizing = make_test_sizing();
        
        let s = String::from("hello");
        let heap_bytes = s.estimate_heap_bytes(&sizing);
        
        // Should be stack size + capacity
        assert!(heap_bytes >= std::mem::size_of::<String>());
        assert!(heap_bytes >= std::mem::size_of::<String>() + s.capacity());
    }

    #[test]
    fn test_box_estimation() {
        let sizing = make_test_sizing();
        
        let boxed = Box::new(42.0_f64);
        let heap_bytes = boxed.estimate_heap_bytes(&sizing);
        
        // Should include pointer, boxed value, and any nested heap
        assert!(heap_bytes >= std::mem::size_of::<Box<f64>>() + std::mem::size_of::<f64>());
    }

    #[test]
    fn test_option_estimation() {
        let sizing = make_test_sizing();
        
        let some_val = Some(42.0_f64);
        let none_val: Option<f64> = None;
        
        assert!(some_val.estimate_heap_bytes(&sizing) >= std::mem::size_of::<Option<f64>>());
        assert_eq!(none_val.estimate_heap_bytes(&sizing), std::mem::size_of::<Option<f64>>());
    }

    #[test]
    fn test_nested_vec_estimation() {
        let sizing = make_test_sizing();
        
        // Vec of vecs (nested)
        let nested: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, 5.0],
        ];
        
        let heap_bytes = nested.estimate_heap_bytes(&sizing);
        
        // Should account for:
        // - Outer vec capacity
        // - Each inner vec's capacity
        let expected_minimum = 
            nested.capacity() * std::mem::size_of::<Vec<f64>>() +  // Outer
            nested.iter().map(|v| v.capacity() * std::mem::size_of::<f64>()).sum::<usize>();  // Inner
        
        assert!(heap_bytes >= expected_minimum);
    }
}
