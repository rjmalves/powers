# PAR-015: Comprehensive Documentation and Migration Guide

## Context

Complete user-facing documentation for PAR model support. Include mathematical background, configuration guide, migration from stationary AR, and troubleshooting.

## Acceptance Criteria

- [ ] Documentation in `docs/` covering all PAR aspects
- [ ] Mathematical background (CEPEL methodology)
- [ ] Configuration guide with JSON examples
- [ ] Migration guide from stationary AR to PAR
- [ ] Troubleshooting section (common errors)
- [ ] API documentation complete
- [ ] Updated main README with PAR mention

## Tasks

### Documentation

- [ ] Create `docs/guides/PAR-MODEL-GUIDE.md`:
  - Mathematical background (CEPEL equation)
  - When to use PAR vs stationary AR
  - Configuration walkthrough
  - Parameter estimation workflow
  - Examples and best practices
- [ ] Create `docs/guides/MIGRATION-TO-PAR.md`:
  - Step-by-step migration instructions
  - Comparison table (stationary AR vs PAR)
  - Backward compatibility notes
  - Troubleshooting common issues
- [ ] Update `docs/reference/INPUT-SPECIFICATION.md`:
  - PeriodicAutoregressive section
  - residual_distribution field
  - Validation rules
- [ ] Update main `README.md`:
  - Add PAR to features list
  - Link to PAR guide
- [ ] Update `PAR_MODEL_SUPPORT.md`:
  - Mark implementation complete
  - Document any deviations from CEPEL

### Testing

- [ ] Review all docs for accuracy
- [ ] Verify all links work
- [ ] Spell check and grammar review

## Dependencies

- **Blocked by**: PAR-012 (needs complete implementation)
- **Blocks**: PAR-016 (CHANGELOG references docs)

## Estimated Effort

**2 story points** (1 day)
