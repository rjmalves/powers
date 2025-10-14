# PAR(p) Model Support - Sprint Plan

## Epic: Periodic Autoregressive (PAR) Model Support for CEPEL Compliance

### Executive Summary

Implement full support for CEPEL's Periodic Autoregressive (PAR) model methodology, which is the production-standard approach for hydrological uncertainty modeling in Brazilian hydrothermal dispatch systems (NEWAVE, GEVAZP).

### Business Value

- **Production Compliance**: Match industry-standard CEPEL/NEWAVE/PSR methodology
- **Seasonal Hydrology**: Correctly model monthly/seasonal variation in inflows
- **Scientific Accuracy**: Use peer-reviewed, production-proven approach
- **Brazilian SIN Support**: Enable realistic modeling of Brazilian power system

### Current State

The existing implementation has a 4-stage pipeline that is **80% correct** but **critically flawed** for seasonal hydrology:

1. ✅ Base noise generation (correct)
2. ✅ Spatial correlation via Cholesky (correct)
3. ❌ Marginal transformation applied at wrong stage (should be on residuals)
4. ❌ Missing periodic/seasonal AR model (PAR(p))

### Target Architecture

```
CEPEL Pipeline (Correct):
Step 0: Fit PAR(p) to historical data → seasonal params (μₘ, σₘ, φₖₘ)
Step 1: Generate w ~ N(0,1) (base noise)
Step 2: Apply correlation: b = D·w (Cholesky)
Step 3: Transform residuals: a = LogNormal3(b) [residuals, not final values!]
Step 4: Re-seasonalize: Z = μₘ + σₘ·[Σφᵢₘ·(Z_{t-i} - μ_{m-i})/σ_{m-i} + a]

where m = season_id (period index), allowing flexible periodicity:
  - period=12 for monthly stages
  - period=4 for quarterly stages
  - any custom cycle (not restricted to months!)
```

**Key Design Decision**: POWE.RS uses the existing `season_id` field (from `GraphNodeInput` and `NoiseModel` structs) instead of hardcoding monthly periods. This is more general than CEPEL's original implementation and supports various time granularities.

### Sprint Structure

**Total Duration**: 4 sprints (8 weeks)
**Story Points**: 34 points total

- **Sprint 1 (Foundation)**: 8 points - Extend type system, add periodic AR types
- **Sprint 2 (Core Implementation)**: 10 points - Implement PAR(p) generator
- **Sprint 3 (Integration)**: 8 points - Update scenario generator, validation
- **Sprint 4 (Tooling & Polish)**: 8 points - Parameter estimation, examples, docs

### Sprints Overview

| Sprint | Focus                    | Points | Tickets        | Duration |
| ------ | ------------------------ | ------ | -------------- | -------- |
| 1      | Foundation & Type System | 8      | PAR-001 to 004 | 2 weeks  |
| 2      | Core PAR Implementation  | 10     | PAR-005 to 008 | 2 weeks  |
| 3      | Integration & Validation | 8      | PAR-009 to 012 | 2 weeks  |
| 4      | Tooling & Documentation  | 8      | PAR-013 to 016 | 2 weeks  |

### Dependencies

- **External**: None (pure Rust implementation)
- **Internal**: Builds on existing AR infrastructure in `ar_dynamics.rs`, `scenario.rs`
- **Blocking**: Must complete before Brazilian SIN examples can be production-ready

### Success Criteria

- [ ] PAR(p) model generates scenarios matching CEPEL methodology
- [ ] Backward compatible: existing AR models continue to work
- [ ] Schema v3 with explicit `periodic_ar` temporal model
- [ ] Comprehensive test coverage (>90% for new code)
- [ ] Parameter estimation tool for fitting PAR(p) to historical data
- [ ] Full documentation with examples
- [ ] Performance: PAR(p) overhead <10% vs current AR

### Risk Management

| Risk                                  | Likelihood | Impact | Mitigation                                          |
| ------------------------------------- | ---------- | ------ | --------------------------------------------------- |
| Breaking changes to AR semantics      | High       | High   | Keep both models, explicit schema version           |
| Complex periodic parameter handling   | Medium     | Medium | Comprehensive validation, unit tests                |
| Performance degradation               | Low        | High   | Benchmarks in PAR-011, optimization in PAR-012      |
| Parameter estimation tool complexity  | Medium     | Medium | Start with simple Yule-Walker, document limitations |
| Integration conflicts with other work | Low        | Medium | Clear module boundaries, early integration testing  |

### Migration Strategy

1. **Add, don't replace**: Keep existing `Autoregressive` for non-seasonal data
2. **Schema versioning**: Introduce schema v3 with explicit model type
3. **Gradual adoption**: Existing configs work unchanged; new configs opt-in to PAR
4. **Documentation**: Clear guidance on when to use PAR vs stationary AR
5. **Validation**: Extensive testing to ensure no regression in existing functionality

### Ticket List

#### Sprint 1: Foundation (8 points)

- [PAR-001](PAR-001-extend-temporal-model-enum.md): Extend TemporalModel enum with PeriodicAutoregressive variant (2 pts)
- [PAR-002](PAR-002-add-seasonal-stats-types.md): Add SeasonalStats and periodic parameter types (2 pts)
- [PAR-003](PAR-003-update-noise-model.md): Update NoiseModel to support periodic residual semantics (2 pts)
- [PAR-004](PAR-004-update-json-schemas.md): Update JSON schemas for periodic AR support (2 pts)

#### Sprint 2: Core Implementation (10 points)

- [PAR-005](PAR-005-implement-seasonal-params.md): Implement SeasonalParams container and validation (2 pts)
- [PAR-006](PAR-006-implement-periodic-ar-generator.md): Implement PeriodicARGenerator with PAR(p) equation (4 pts)
- [PAR-007](PAR-007-residual-distribution-transform.md): Implement residual-based marginal transformation (2 pts)
- [PAR-008](PAR-008-unit-tests-par-generator.md): Comprehensive unit tests for PAR generator (2 pts)

#### Sprint 3: Integration & Validation (8 points)

- [PAR-009](PAR-009-integrate-scenario-generator.md): Integrate PAR generator into scenario generation pipeline (2 pts)
- [PAR-010](PAR-010-validation-tests.md): Validation tests against CEPEL equations (2 pts)
- [PAR-011](PAR-011-performance-benchmarks.md): Performance benchmarks: PAR vs stationary AR (2 pts)
- [PAR-012](PAR-012-integration-tests.md): End-to-end integration tests with full pipeline (2 pts)

#### Sprint 4: Tooling & Documentation (8 points)

- [PAR-013](PAR-013-parameter-estimation-tool.md): Parameter estimation tool for fitting PAR(p) to historical data (3 pts)
- [PAR-014](PAR-014-example-configs.md): Create example configs demonstrating PAR usage (1 pt)
- [PAR-015](PAR-015-comprehensive-documentation.md): Comprehensive documentation and migration guide (2 pts)
- [PAR-016](PAR-016-update-changelog.md): Update CHANGELOG and final polish (2 pts)

### Post-Sprint Activities

- [ ] Monitor production usage for 1 month
- [ ] Collect user feedback on parameter estimation tool
- [ ] Consider advanced features (PAR-A, MS-PAR for ENSO)
- [ ] Potential optimization: SIMD vectorization for seasonal calculations

### References

- CEPEL GEVAZP manual: https://see.cepel.br/manual/libs/latest/incerteza_hidrologica/
- Maceira & Bezerra (1997): "Stochastic streamflow model for hydroelectric systems"
- Box-Jenkins time series analysis methodology
- Yule-Walker equations for AR parameter estimation

---

**Created**: October 13, 2025  
**Epic Owner**: Architecture Team  
**Sprint Planning**: 4 sprints × 2 weeks = 8 weeks total
