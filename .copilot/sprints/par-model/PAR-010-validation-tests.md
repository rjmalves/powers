# PAR-010: Validation Tests Against CEPEL Equations

## Context

Verify implementation correctness against CEPEL's published equations and methodology. This ticket ensures mathematical accuracy and catches any subtle implementation bugs.

## Acceptance Criteria

- [ ] Hand-calculated test cases for PAR(1) and PAR(2)
- [ ] Comparison with CEPEL reference outputs (if available)
- [ ] Statistical tests for mean/variance convergence
- [ ] Correlation structure preservation tests
- [ ] Seasonality pattern verification
- [ ] Stationarity condition verification

## Tasks

### Testing

- [ ] Create hand-calculated reference cases for PAR(1), PAR(2)
- [ ] Implement statistical convergence tests (1000+ scenarios)
- [ ] Verify seasonal mean μₘ emerges in long simulations
- [ ] Verify seasonal std σₘ emerges in long simulations
- [ ] Test spatial correlation matrix preservation
- [ ] Compare with GEVAZP/NEWAVE outputs if accessible
- [ ] Document validation methodology

### Documentation

- [ ] Validation report with test results
- [ ] CEPEL equation compliance checklist
- [ ] Known limitations documentation

## Dependencies

- **Blocked by**: PAR-009 (scenario integration)
- **Blocks**: None (parallel with PAR-011)

## Estimated Effort

**2 story points** (1 day)
