# PAR-012: End-to-End Integration Tests

## Context

Comprehensive end-to-end tests running full SDDP algorithm with PAR models. Verify PAR works correctly in the full optimization context, not just scenario generation.

## Acceptance Criteria

- [ ] E2E test: Single hydro with PAR inflows (varying period)
- [ ] E2E test: Multi-hydro cascade with PAR
- [ ] E2E test: Mixed PAR + stationary AR entities
- [ ] E2E test: PAR with LogNormal3 residuals
- [ ] E2E test: Multiple period configurations (12-period, 4-period, custom)
- [ ] Verify policy convergence with PAR scenarios
- [ ] Verify CSV output correctness
- [ ] No crashes or numerical instabilities

## Tasks

### Implementation

- [ ] Create `tests/test_sddp_par_e2e.rs`
- [ ] Test: 12-period PAR model (e.g., monthly inflows)
- [ ] Test: 4-period PAR model (e.g., quarterly inflows)
- [ ] Test: Multi-reservoir with correlated PAR inflows
- [ ] Test: PAR + Independent mixed configuration
- [ ] Test: Long horizon (36+ stages)
- [ ] Compare policy quality vs stationary AR baseline

### Testing

- [ ] Run full SDDP training with PAR (100+ iterations)
- [ ] Verify convergence metrics
- [ ] Check simulation outputs for anomalies
- [ ] Verify cut generation stability

### Documentation

- [ ] E2E test documentation
- [ ] Known issues or limitations
- [ ] Performance notes for large-scale cases

## Dependencies

- **Blocked by**: PAR-009 (scenario integration), PAR-010 (validation)
- **Blocks**: PAR-013 (tooling - needs working PAR pipeline)

## Estimated Effort

**2 story points** (1 day)
