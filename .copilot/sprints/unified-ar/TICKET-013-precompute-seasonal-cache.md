# TICKET-013: Precompute and Cache Seasonal Transformations

**Sprint:** 4 - Optimization  
**Phase:** 4 - Optimize Performance  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

The observation-residual transformation `Y_t = μ_s + σ_s * Z'_t` is used in every subproblem for every season. Currently, seasonal parameters (μ, σ) may be looked up repeatedly from HashMaps or computed on-the-fly. This ticket precomputes and caches all transformation parameters at construction time, eliminating repeated lookups in hot paths.

This optimization targets the 15-25% speedup potential identified in the roadmap.

## Acceptance Criteria

- [ ] Given seasonal parameters, when UnifiedInflowModel is constructed, then all μ, σ, 1/σ values are precomputed and cached
- [ ] Given a subproblem being solved, when transformation constraints are added, then no HashMap lookups occur
- [ ] Given constraint RHS updates, when setting seasonal parameters, then cached values are used
- [ ] Given cache structure, when accessed, then it provides O(1) lookups by (hydro, season)
- [ ] Performance: Subproblem construction should be 5-10% faster due to eliminated lookups
- [ ] Performance: realize_uncertainties should show measurable speedup from caching

## Acceptance Criteria

- [ ] Given seasonal parameters, when UnifiedInflowModel is constructed, then all μ, σ, 1/σ values are precomputed and cached
- [ ] Given a subproblem being solved, when transformation constraints are added, then no HashMap lookups occur
- [ ] Given constraint RHS updates, when setting seasonal parameters, then cached values are used
- [ ] Given cache structure, when accessed, then it provides O(1) lookups by (hydro, season)
- [ ] Performance: Subproblem construction should be 5-10% faster due to eliminated lookups
- [ ] Performance: realize_uncertainties should show measurable speedup from caching

## Tasks

### Implementation

- [ ] Design `SeasonalTransformCache` struct:
  - Stores μ, σ, 1/σ per (hydro, season) combination
  - Optimized for cache locality (e.g., Vec<Vec<(f64, f64, f64)>>)
  - Shared via Arc for efficient cloning across subproblems
- [ ] Implement cache construction:
  - Extract all seasonal parameters at UnifiedInflowModel creation
  - Precompute 1/σ for inverse transform (if needed)
  - Store in cache-friendly layout
- [ ] Update UnifiedInflowModel to use cache:
  - Replace seasonal_params HashMap with cache
  - Update add_constraints_to_lp() to use cached values
  - Update any transformation methods to use cache
- [ ] Add cache accessor methods:
  - `get_transform_params(hydro, season) -> (μ, σ)`
  - `get_inverse_params(hydro, season) -> (μ, 1/σ)`
  - Inline where possible for maximum performance
- [ ] Update Subproblem to share cache via Arc:
  - Pass Arc<SeasonalTransformCache> during construction
  - No cloning of data, just Arc reference counting

### Testing

- [ ] Unit test: Construct cache with known seasonal parameters
- [ ] Unit test: Verify get_transform_params returns correct values
- [ ] Unit test: Verify 1/σ is correctly precomputed
- [ ] Unit test: Verify Arc sharing works (multiple subproblems, one cache)
- [ ] Unit test: Verify cache access is O(1) (measure with large n_hydros \* n_seasons)
- [ ] Integration test: Full algorithm run with cache, verify correctness
- [ ] Performance test: Benchmark subproblem construction before/after caching
- [ ] Performance test: Benchmark realize_uncertainties before/after caching
- [ ] Memory test: Verify cache memory overhead is acceptable (< 1MB for typical systems)

### Documentation

- [ ] Add doc comment to SeasonalTransformCache explaining purpose and layout
- [ ] Document cache construction and precomputation strategy
- [ ] Add inline comments explaining Arc sharing pattern
- [ ] Add performance notes to module docs explaining cache benefits
- [ ] Update CHANGELOG.md with "Performance: Precomputed seasonal transformation cache"

## Technical Notes

### Cache Design

```rust
/// Precomputed seasonal transformation parameters for efficient access.
///
/// Stores (μ, σ, 1/σ) tuples for each (hydro, season) combination.
/// Layout optimized for cache locality: hydros in outer Vec, seasons in inner Vec.
#[derive(Clone, Debug)]
pub struct SeasonalTransformCache {
    // cache[hydro][season] = (μ, σ, 1/σ)
    params: Vec<Vec<(f64, f64, f64)>>,
    n_hydros: usize,
    n_seasons: usize,
}

impl SeasonalTransformCache {
    pub fn new(
        seasonal_params: &HashMap<(usize, usize), (f64, f64)>,
        n_hydros: usize,
        n_seasons: usize,
    ) -> Self {
        let mut params = vec![vec![(0.0, 1.0, 1.0); n_seasons]; n_hydros];

        for (&(hydro, season), &(mu, sigma)) in seasonal_params.iter() {
            let inv_sigma = if sigma > 1e-10 { 1.0 / sigma } else { 1.0 };
            params[hydro][season] = (mu, sigma, inv_sigma);
        }

        Self { params, n_hydros, n_seasons }
    }

    #[inline]
    pub fn get_transform(&self, hydro: usize, season: usize) -> (f64, f64) {
        let (mu, sigma, _) = self.params[hydro][season];
        (mu, sigma)
    }

    #[inline]
    pub fn get_inverse_transform(&self, hydro: usize, season: usize) -> (f64, f64) {
        let (mu, _, inv_sigma) = self.params[hydro][season];
        (mu, inv_sigma)
    }
}
```

### Integration with UnifiedInflowModel

```rust
pub struct UnifiedInflowModel {
    dimension: usize,
    ar_coefficients: Vec<Vec<f64>>,
    seasonal_cache: Arc<SeasonalTransformCache>,  // CHANGED: Arc for sharing
    lag_buffer: Vec<Vec<f64>>,
    max_lag: usize,
}

impl UnifiedInflowModel {
    pub fn add_constraints_to_lp(
        &self,
        pb: &mut solver::Problem,
        vars: &Variables,
        season_id: usize,
    ) -> ConstraintIndices {
        // ...
        for hydro in 0..self.dimension {
            // OPTIMIZED: Direct cache access, no HashMap lookup
            let (mu, sigma) = self.seasonal_cache.get_transform(hydro, season_id);

            // Add constraint: Y - σZ' = μ
            let obs_factors = vec![
                (vars.inflow[hydro], 1.0),
                (vars.inflow_residual[hydro], -sigma),
            ];
            let obs_row = pb.add_row(mu..=mu, &obs_factors);
            // ...
        }
    }
}
```

### Memory Layout Optimization

**Why Vec<Vec> instead of HashMap?**

```rust
// SLOW: HashMap lookup O(log n) or O(1) with hash
HashMap<(usize, usize), (f64, f64, f64)>

// FAST: Direct indexing O(1) with better cache locality
Vec<Vec<(f64, f64, f64)>>
```

**Cache Performance:**

- Vec access: ~1-2 CPU cycles (cached)
- HashMap access: ~20-50 CPU cycles (hash + lookup)
- Speedup: 10-50x on parameter access

**Memory Overhead:**

- 3 f64 values per (hydro, season): 24 bytes
- 30 hydros × 12 seasons: 8.6 KB
- Negligible compared to solver memory

### Arc Sharing Pattern

```rust
// Cache created once
let cache = Arc::new(SeasonalTransformCache::new(...));

// Shared across all subproblems (no cloning data)
for node in nodes {
    let model = UnifiedInflowModel {
        seasonal_cache: Arc::clone(&cache),  // Just bumps ref count
        // ...
    };
}
```

**Benefits:**

- No data duplication
- Cheap Arc::clone (just pointer + ref count)
- Thread-safe sharing (Arc is Send + Sync)

### Performance Impact Estimation

**Before (HashMap lookups):**

- Lookup per constraint: ~50ns
- Constraints per subproblem: ~2n (n hydros × 2 constraints)
- Lookups per solve: ~2n lookups
- Cost: ~100n ns per subproblem

**After (Vec indexing):**

- Lookup per constraint: ~2ns
- Cost: ~4n ns per subproblem
- **Speedup: 25x on parameter access**

**Overall impact:**

- Subproblem construction: 5-10% faster
- realize_uncertainties: 2-5% faster (parameter access is small part)
- Full algorithm: 3-7% faster (amortized)

Combined with eliminated conditionals (15-25% from TICKET-008), total speedup could reach 20-30%.

### Edge Cases

- **Zero variance**: σ = 0 case, set inv_sigma = 1.0 and log warning
- **Missing parameters**: Should never happen after validation, but handle defensively
- **Large season count**: Cache still O(1), just more memory
- **Cache invalidation**: Not needed (parameters are immutable)

## Dependencies

- **Blocked by**:
  - TICKET-001 (needs UnifiedInflowModel)
  - TICKET-002 (needs constraint generation)
- **Blocks**: None (independent optimization)
- **Related**: TICKET-014 (batch RHS updates)

## References

- `src/seasonal_params.rs` - Current seasonal parameter handling
- `src/unified_inflow_model.rs` - UnifiedInflowModel implementation
- UNIFIED_AR_ROADMAP.md - Section 3.1 (Precompute seasonal transformations)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Performance benchmark shows 5-10% subproblem construction speedup
- [ ] Performance benchmark shows measurable realize_uncertainties speedup
- [ ] Memory overhead is acceptable (< 10KB for typical systems)
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Arc sharing verified (test Arc::strong_count)
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
