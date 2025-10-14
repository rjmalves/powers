# PAR-014: Example Configs Demonstrating PAR Usage

## Context

Create example configurations in `examples/` demonstrating PAR model usage. Help users understand how to configure PAR models for real-world cases.

## Acceptance Criteria

- [ ] Example 1: Simple single-hydro PAR(1) with 12-period (monthly) inflows
- [ ] Example 2: Multi-reservoir cascade with correlated PAR inflows
- [ ] Example 3: PAR with LogNormal3 residuals (non-negative inflows)
- [ ] Example 4: Mixed PAR + Independent configuration
- [ ] Example 5: Quarterly PAR (4-period) demonstrating flexibility
- [ ] All examples validated and runnable
- [ ] README explaining each example and period configuration
- [ ] Documentation references from main docs

## Tasks

### Implementation

- [ ] Create `examples/06-par-model/` directory
- [ ] Create `01-simple-par1/` with single hydro, 12-period PAR(1)
- [ ] Create `02-cascade-par/` with multi-hydro PAR
- [ ] Create `03-lognormal-par/` with LogNormal3 residuals
- [ ] Create `04-mixed-models/` with PAR + Independent
- [ ] Create `05-quarterly-par/` with 4-period PAR (demonstrates flexibility)
- [ ] Add README to each example explaining setup and period choice
- [ ] Update `examples/README.md` with PAR section

### Testing

- [ ] Run all examples with `cargo run --release`
- [ ] Verify outputs are reasonable
- [ ] Document expected run times

### Documentation

- [ ] Example descriptions in `examples/README.md`
- [ ] Link from main PAR documentation
- [ ] Usage notes and tips

## Dependencies

- **Blocked by**: PAR-012 (needs working PAR pipeline)
- **Blocks**: None (parallel with PAR-013)

## Estimated Effort

**1 story point** (0.5 day)
