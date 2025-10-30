# Future Enhancements for POWE.RS

This document tracks planned enhancements that are not currently prioritized but would add value to the project. These items represent nice-to-have improvements that don't affect current functionality but could improve usability, performance, or capabilities.

## Algorithm Enhancements

### Markovian and Cyclic Graph Support

**Current Limitation**: Simulation trajectory extraction assumes path graphs (linear progression through stages). The BFS-based approach works well for path graphs but may need revision for more complex graph topologies.

**Proposed Enhancement**: Extend simulation to support:
- **Markovian graphs**: State transitions depend on current state (not just time)
- **Cyclic graphs**: Infinite horizon problems where stages can loop

**Use Cases**:
- Multi-reservoir systems with complex operating rules
- Infinite horizon problems (steady-state policies)
- Stochastic dual dynamic programming with stage-dependent uncertainty

**Potential Value**: Enables modeling of more complex hydrothermal systems and broadens applicability

**Estimated Effort**: Large (3-4 weeks)
- Requires graph traversal redesign
- Changes to simulation trajectory extraction logic
- Extensive testing for correctness

**Code Location**: `src/sddp/mod.rs:2120` (graph_bfs_table construction)

---

### Unified Load Uncertainty Model

**Current Limitation**: Load balance constraints use older approach separate from unified inflow model. Code comment indicates migration path: "Load balance RHS (old approach, TODO: migrate to unified model)".

**Proposed Enhancement**: Migrate load uncertainty to use the same unified model as inflow uncertainties:
- Single code path for all uncertainty types
- Consistent handling of AR dynamics vs independent noise
- Simplified subproblem update logic

**Use Cases**:
- Code maintainability and consistency
- Easier to add new uncertainty types
- Reduced cognitive load for developers

**Potential Value**: Medium - improves code quality and maintainability, no user-facing changes

**Estimated Effort**: Medium (1-2 weeks)
- Refactor load balance constraint updates
- Ensure backward compatibility with existing examples
- Comprehensive testing

**Code Location**: `src/subproblem.rs:1327` (set_load_balance_rhs call)

---

## Input/Output Enhancements

### CSV Output Transformation for PAR Models

**Current Limitation**: For PAR models with state expansion, CSV output contains residuals (Z'_t) rather than observations (Y_t). This is mathematically correct for the LP but less intuitive for users.

**Proposed Enhancement**: Transform residuals back to observation space for CSV output:
- Apply inverse transformation: Y_t = μ_m + σ_m · Z'_t
- Requires seasonal parameters (μ, σ) in output generation
- Add option to output both residuals and observations

**Use Cases**:
- More intuitive CSV files for users unfamiliar with PAR internals
- Direct comparison with historical data
- Post-processing and visualization

**Potential Value**: High user experience improvement, low technical risk

**Estimated Effort**: Small (2-3 days)
- Access seasonal params in output generation
- Add transformation logic
- Update CSV column names/documentation

**Code Location**: `src/output.rs:377` (HydroSimulationOutput serialization)

---

### Skewness Parameter in Seasonal Statistics

**Current Limitation**: `SeasonalStats` struct in PAR model has `skewness: None` hardcoded. Skewness is not currently computed or used.

**Proposed Enhancement**: Compute and store skewness in seasonal statistics:
- Add skewness calculation during estimation
- Store in SeasonalStats for completeness
- Potential use in distribution selection guidance

**Use Cases**:
- Statistical diagnostics and model selection
- Help users choose between Normal vs LogNormal3 distributions
- Model documentation and reporting

**Potential Value**: Low - nice diagnostic information, not critical

**Estimated Effort**: Small (1-2 days)
- Add skewness calculation to estimation module
- Populate SeasonalStats field
- Optional: Add to output/reports

**Code Location**: `src/input.rs:770` (SeasonalStats construction)

---

## Configuration Enhancements

### Extract Seasonal Configuration from PAR Model

**Current Limitation**: Pre-study period construction hardcodes `num_seasons = 12` with a TODO comment "Extract from PAR config when available". This works for monthly models but is inflexible.

**Proposed Enhancement**: Extract `num_seasons` from PAR configuration automatically:
- Query uncertainty_specifications for PAR models
- Determine num_seasons from temporal_model
- Remove hardcoded assumption

**Use Cases**:
- Quarterly models (num_seasons = 4)
- Weekly models (num_seasons = 52)
- Single-season models (num_seasons = 1)
- Automatic adaptation to input data

**Potential Value**: Medium - improves flexibility and removes magic numbers

**Estimated Effort**: Small (1-2 days)
- Add helper to extract num_seasons from config
- Update pre-study construction
- Test with various seasonal configurations

**Code Location**: `src/sddp/builder.rs:682` (pre-study period setup)

---

### Multi-Season Initial Conditions

**Current Limitation**: Initial lag transformation uses first season's parameters (season_id=0) regardless of when the planning horizon starts. This is a simplification that may not be accurate for all cases.

**Proposed Enhancement**: Use season-appropriate parameters for initial condition transformation:
- Determine actual season of initial lags based on study start time
- Apply correct μ_m, σ_m for each lag's season
- More accurate residual initialization

**Use Cases**:
- Planning horizons starting mid-year
- Improved accuracy for short horizons
- Better warm-start for PAR generators

**Potential Value**: Medium accuracy improvement for specific use cases

**Estimated Effort**: Small (2-3 days)
- Add season detection logic
- Update residual transformation
- Test with various start dates

**Code Location**: `src/noise_model_cache.rs:287` (initial residual transformation)

---

## Numerical Methods

### Eigenvalue-Based Stationarity Check

**Current Limitation**: PAR stationarity validation uses heuristic sufficient condition for AR(p > 2): sum(|φₖ|) < 1. This is conservative and may reject some stationary models.

**Proposed Enhancement**: Use nalgebra to compute eigenvalues of companion matrix for exact stationarity check:
- All eigenvalues must have magnitude < 1 for stationarity
- More accurate than sum-of-coefficients heuristic
- Would require adding `nalgebra` dependency

**Mathematical Background**:
For AR(p): X_t = φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t

Companion matrix:
```
C = [φ₁  φ₂  φ₃  ... φₚ]
    [1   0   0   ... 0 ]
    [0   1   0   ... 0 ]
    [... ... ... ... ...]
    [0   0   0   1   0 ]
```

Stationarity ⟺ max(|eigenvalues(C)|) < 1

**Use Cases**:
- More accurate validation for complex AR models
- Accept valid models currently rejected by heuristic
- Professional-grade numerical validation

**Potential Value**: Medium - improved accuracy, but current heuristic is adequate

**Estimated Effort**: Medium (3-5 days)
- Add nalgebra dependency
- Implement companion matrix construction
- Compute eigenvalues and check magnitudes
- Benchmark performance impact
- Update error messages

**Code Location**: `src/seasonal_params.rs:491` (PAR stationarity validation)

**Trade-offs**:
- ✅ More accurate stationarity detection
- ✅ Professional implementation
- ❌ Adds nalgebra dependency (~50KB, but well-maintained)
- ❌ Slightly slower validation (negligible for typical models)
- ❌ More complex implementation

---

## Implementation Priority

If these enhancements were to be implemented, suggested order based on value/effort ratio:

1. **CSV Output Transformation** - High user value, low effort
2. **Extract PAR Seasonal Config** - Removes magic numbers, low effort
3. **Multi-Season Initial Conditions** - Accuracy improvement, low effort
4. **Unified Load Model** - Code quality, medium effort
5. **Skewness Parameter** - Nice-to-have diagnostic, low priority
6. **Eigenvalue Stationarity** - Accuracy vs dependency trade-off
7. **Markovian/Cyclic Graphs** - Large effort, enables new use cases

---

## Contributing

If you're interested in implementing any of these enhancements:

1. Open an issue referencing this document
2. Discuss approach and acceptance criteria with maintainers
3. Submit a PR with implementation and tests
4. Update this document to mark item as implemented or move to CHANGELOG

---

## Maintenance

This document should be updated when:
- New future enhancements are identified (add here instead of inline TODOs)
- Enhancements are implemented (move to CHANGELOG, remove from here)
- Priorities or estimates change based on new information
- Use cases become more compelling or less relevant

Last Updated: 2025-10-30
