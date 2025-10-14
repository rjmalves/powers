# PAR-013: Parameter Estimation Tool (Yule-Walker Fitting)

## Context

Users need a tool to estimate PAR parameters (μₘ, σₘ, φₖₘ) from historical time series data. Implement Yule-Walker method for fitting periodic AR models to data.

## Acceptance Criteria

- [ ] CLI tool or library function for parameter estimation
- [ ] Input: CSV with historical time series
- [ ] Output: JSON config with PAR parameters
- [ ] Implements Yule-Walker equations for each period
- [ ] Estimates seasonal means/stds from data
- [ ] Validates stationarity of estimated parameters
- [ ] Documentation with usage examples

## Tasks

### Implementation

- [ ] Create `src/estimation/mod.rs` module
- [ ] Implement `estimate_par_params(data: &[f64], period: usize) -> SeasonalParams`
- [ ] De-seasonalize data (subtract means, divide by stds)
- [ ] Apply Yule-Walker per period to get φₖₘ
- [ ] Validate estimated parameters
- [ ] Optional: Add to CLI as subcommand `powers estimate-par`

### Testing

- [ ] Unit test: Known AR(1) series recovers φ
- [ ] Unit test: Seasonal pattern detected correctly
- [ ] Integration test: Full estimation pipeline
- [ ] Test with synthetic PAR-generated data (roundtrip test)

### Documentation

- [ ] Usage guide for parameter estimation
- [ ] Mathematical background (Yule-Walker)
- [ ] Example workflow with sample data
- [ ] Limitations and assumptions

## Technical Notes

```bash
# CLI usage example
powers estimate-par \
  --input historical_inflows.csv \
  --period 12 \
  --ar-order 1 \
  --output par_config.json
```

Output: JSON fragment ready to paste into system.json

## Dependencies

- **Blocked by**: PAR-012 (needs working PAR pipeline)
- **Blocks**: None (parallel with PAR-014)

## Estimated Effort

**3 story points** (1.5 days)
