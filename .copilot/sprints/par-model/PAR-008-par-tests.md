# PAR-008: Comprehensive Unit Tests for PAR Generator

## Context

Dedicated testing ticket for PAR generator covering edge cases, numerical stability, and CEPEL compliance. Ensures production-grade quality with >95% code coverage.

## Acceptance Criteria

- [ ] Edge case tests (zero coefficients, max AR order, etc.)
- [ ] Numerical stability tests (1M+ iterations)
- [ ] Stationarity verification (long-run statistics)
- [ ] Known CEPEL test cases (if available)
- [ ] Property-based tests using proptest
- [ ] Code coverage >95% for PAR modules
- [ ] Performance benchmarks included

## Tasks

### Testing

- [ ] Edge case: All AR coefficients zero
- [ ] Edge case: Max AR order (p=12)
- [ ] Edge case: Single-period PAR (no seasonality)
- [ ] Stability: 1M iteration run without overflow
- [ ] Stationarity: Verify long-run mean matches seasonal mean
- [ ] Stationarity: Verify long-run variance converges
- [ ] Property test: Generated values always finite
- [ ] Property test: Stationarity implies bounded output
- [ ] Benchmark: PAR vs stationary AR generation speed
- [ ] Regression: Known CEPEL output (if test data available)

### Documentation

- [ ] Test strategy documentation
- [ ] Coverage report interpretation guide

## Dependencies

- **Blocked by**: PAR-006 (generator implementation)
- **Blocks**: None (parallel with PAR-009)

## Estimated Effort

**2 story points** (1 day)
