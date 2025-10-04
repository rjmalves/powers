# Architecture Note: Production API vs Builder API

**Date**: October 4, 2025  
**Context**: Sprint 2 Review - Sprint 3 Preparation  
**Decision**: Keep production `run()` using low-level API

---

## Executive Summary

The production `run()` function in `src/lib.rs` **should continue using the low-level API** (manual graph construction from JSON) and **should NOT be migrated to the Builder API**.

**Reason**: Different use cases with incompatible requirements.

---

## Two APIs, Two Use Cases

### Low-Level API (Production)

**Used by**: `src/lib.rs::run()`, `src/main.rs`  
**Purpose**: Production workloads with complex configurations

**Features**:

- ✅ Complex seasonal structures (12+ seasons)
- ✅ Distribution-based uncertainty (Normal, LogNormal, etc.)
- ✅ Markovian graph structures (not just linear paths)
- ✅ Per-node stochastic process configuration
- ✅ JSON-driven configuration (graph.json, recourse.json)
- ✅ Full flexibility for production use

**Example Input** (recourse.json):

```json
{
  "uncertainties": [
    {
      "season_id": 0,
      "num_branchings": 10,
      "distributions": {
        "load": [
          {
            "bus_id": 0,
            "normal": { "mu": 75.0, "sigma": 0.0 }
          }
        ],
        "inflow": [
          {
            "hydro_id": 0,
            "lognormal": { "mu": 3.6, "sigma": 0.6928 }
          }
        ]
      }
    }
  ]
}
```

**Code**:

```rust
// Low-level API (production)
let node_data_graph = graph_input.build_sddp_graph(&input.system)?;
let initial_condition = recourse.build_sddp_initial_condition();
let saa = recourse.generate_sddp_noises(&node_data_graph, seed);
let mut sddp = SddpAlgorithm::new(node_data_graph, initial_condition, seed)?;
```

---

### Builder API (Tests/Benchmarks)

**Used by**: `tests/fixtures/benchmarks.rs`, `tests/test_*.rs`  
**Purpose**: Testing, benchmarks, simple examples

**Features**:

- ✅ Simple linear path graphs (stage 0→1→2→...)
- ✅ Explicit scenario values (deterministic or stochastic)
- ✅ 90% less boilerplate (150 lines → 8 lines)
- ✅ Type-safe, fluent API
- ✅ Zero-cost abstraction (compiles to same code)
- ⚠️ Limited to simpler use cases

**Example Usage**:

```rust
// Builder API (tests/benchmarks)
let sddp = SddpAlgorithm::builder()
    .system(system)
    .initial_storage(vec![50.0])
    .num_stages(12)
    .deterministic_loads(vec![75.0; 12])  // Explicit values, not distributions
    .deterministic_inflows(vec![40.0; 12])
    .seed(42)
    .build()?;
```

---

## Why NOT Migrate Production to Builder?

### 1. Loss of Functionality

**Problem**: Builder doesn't support distribution-based input.

Current production input:

```json
"load": [{"bus_id": 0, "normal": {"mu": 75.0, "sigma": 5.0}}]
```

Builder API requires:

```rust
.deterministic_loads(vec![75.0])  // No distribution support!
```

**Impact**: Would break all production use cases that rely on stochastic load/inflow.

---

### 2. API Mismatch

**Current JSON format** (production):

- Distributions: Normal, LogNormal
- Per-season configuration
- Branching factors per season
- Complex graph structures

**Builder format** (tests):

- Explicit scenario values
- Linear path graphs only
- Simple configuration

**Conclusion**: Fundamentally different data models.

---

### 3. No Performance Benefit

**Both APIs compile to the same code**:

```rust
// Low-level API
let sddp = SddpAlgorithm::new(graph, initial_condition, seed)?;

// Builder API
let sddp = builder.build()?;  // Internally calls SddpAlgorithm::new()
```

**Benchmark verification** (from T2.6a):

- Zero performance overhead
- Identical machine code after optimization
- No allocations difference

**Conclusion**: No runtime benefit to migration.

---

### 4. Increased Complexity

**Migration would require**:

1. Convert distribution-based JSON to explicit scenarios
2. Pre-compute scenario values from distributions
3. Flatten seasonal structure into linear path
4. Rewrite all input parsing logic
5. Update all documentation and examples
6. Migrate existing production JSON files

**Estimated effort**: 40-60 hours  
**Benefit**: None (same functionality, same performance)

---

## Architectural Decision

### Decision: Keep Both APIs

**Production** (`src/lib.rs::run()`):

- ✅ Continue using low-level API
- ✅ Support full JSON-based configuration
- ✅ Maintain distribution-based uncertainty
- ✅ No changes needed

**Tests/Benchmarks** (`tests/**/*.rs`):

- ✅ Use Builder API for simplicity
- ✅ 90% less boilerplate
- ✅ Clearer test intent
- ✅ Already migrated in Sprint 2

---

## Documentation Update

### src/lib.rs::run() Documentation

Added comprehensive doc comment explaining:

- Purpose: Production use with full JSON configuration
- Features: Complex seasonal structures, distributions, Markovian graphs
- Alternative: Builder API for simpler use cases
- Performance notes: Critical hot path, Rayon parallelism

### Example of Builder API Usage

For developers who need simpler cases, the builder is documented and available:

```rust
use powers_rs::sddp::SddpAlgorithm;

// Simple deterministic case (e.g., for testing)
let sddp = SddpAlgorithm::builder()
    .system(system)
    .initial_storage(vec![50.0])
    .num_stages(12)
    .deterministic_loads(vec![75.0; 12])
    .deterministic_inflows(vec![40.0; 12])
    .seed(42)
    .build()?;

let result = sddp.train(30, 10)?;
```

See `tests/fixtures/benchmarks.rs` for complete examples.

---

## Sprint 3 Implications

### No Migration Needed

**Verdict**: Production `run()` function is correct as-is.

**Actions Taken**:

1. ✅ Added documentation explaining low-level vs builder API
2. ✅ Clarified use cases for each API
3. ✅ No code changes to `run()` (intentional)

### Builder API Already in Use

**Sprint 2 Achievement**:

- ✅ Builder API implemented (T2.6a)
- ✅ All benchmarks migrated to builder (T2.6)
- ✅ All tests use builder where appropriate
- ✅ 296 new tests created with builder API

**Sprint 3**: Continue using builder for new tests.

---

## Future Considerations

### Potential Enhancement: Builder Support for Distributions

**If** there's demand for builder to support distributions:

```rust
// Hypothetical future API
let sddp = SddpAlgorithm::builder()
    .system(system)
    .initial_storage(vec![50.0])
    .num_stages(12)
    .stochastic_loads_normal(vec![75.0; 12], vec![5.0; 12])  // mu, sigma
    .stochastic_inflows_lognormal(vec![3.6; 12], vec![0.69; 12])
    .seed(42)
    .build()?;
```

**Estimated Effort**: 10-15 hours  
**Benefit**: Simplified API for stochastic problems  
**Priority**: Low (low-level API works fine)

---

## Conclusion

**Decision**: Keep production `run()` using low-level API. No migration to builder.

**Rationale**:

1. Different use cases (production vs tests)
2. No performance benefit
3. Loss of functionality (distributions)
4. No user demand for migration
5. Both APIs work well for their intended purposes

**Documentation**: Updated `src/lib.rs::run()` with comprehensive comments explaining the decision.

**Sprint 3**: No action required. Proceed with simulation testing (T3.1, T3.2, T3.3).

---

**Prepared By**: HPC Developer  
**Reviewed By**: Software Reviewer & Quality Guardian  
**Date**: October 4, 2025  
**Status**: ✅ Architectural Decision Documented
