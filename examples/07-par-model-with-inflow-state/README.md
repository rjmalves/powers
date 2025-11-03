# Example 07: PAR Model with Inflow State

This example demonstrates Periodic Autoregressive (PAR) modeling with inflow state variables tracked across stages.

## System Description

- 1 hydro plant  
- 12 seasons (monthly)
- PAR(1) temporal model (first-order autoregression)
- Normal distributed innovations

## Files

- `system.json` - System configuration
- `graph.json` - SDDP tree structure
- `recourse.json` - Uncertainty specifications with PAR model
- `config.json` - SDDP algorithm configuration

## Uncertainty Model

This example uses the **new unified temporal model format** (v0.4.0+):

```json
{
  "temporal_model": {
    "num_seasons": 12,
    "ar_orders": [1, 1, ..., 1],
    "ar_coefficients": [[0.7], [0.7], ..., [0.7]],
    "seasonal_means": [70.0, 65.0, 55.0, ..., 70.0],
    "seasonal_stds": [20.0, 20.0, 20.0, ..., 20.0]
  }
}
```

### PAR(1) Model

Periodic Autoregressive model of order 1:
- `ar_orders`: All ones `[1, 1, ..., 1]` (order 1 for each season)
- `ar_coefficients`: `[[0.7], [0.7], ...]` (correlation coefficient 0.7)
- Inflows correlated across time with previous period

### Mathematical Form

For each season s:
```
Y_t = μ_s + φ_s · (Y_{t-1} - μ_{s-1}) + σ_s · ε_t
```

Where:
- `Y_t`: Inflow at time t
- `μ_s`: Seasonal mean
- `φ_s`: AR coefficient (0.7 in this example)
- `σ_s`: Seasonal standard deviation
- `ε_t`: Standard normal innovation

### State Variables

This example uses `storage_and_inflow` state choice, which tracks:
- Storage levels at each hydro
- **Lagged inflow observations** (Y_{t-1}) for PAR dynamics

The lagged inflows are state variables that affect the distribution of future inflows.

### Initial Conditions

The `recourse.json` file specifies initial conditions that directly affect first-stage optimization:

```json
{
  "initial_condition": {
    "storage": [100.0],
    "inflow": [[70.0]]
  }
}
```

**Initial Storage**: Starting reservoir level (100.0 MWh)

**Initial Lagged Inflows**: Recent historical inflows that seed the AR dynamics
- For AR(1): one lag value [Y_{t-1}]
- For AR(2): two lag values [Y_{t-1}, Y_{t-2}]
- Order: newest to oldest

**Impact on First Stage**: The initial lag directly affects first-stage inflow realizations through the AR equation:

```
Y_t = μ_s + φ_s · (Y_{t-1} - μ_{s-1}) + σ_s · ε_t
```

**Experiment**: Try changing the initial lag to see the impact:

```bash
# Edit recourse.json: change "inflow": [[70.0]] to [[100.0]]
# Then run the example and compare first-stage inflow values
```

With φ=0.7, changing the lag from 70.0 to 100.0 will shift first-stage expected inflows by approximately 21 m³/s (0.7 × 30).

## Seasonal Patterns

The example includes realistic seasonal patterns:

| Season | Mean (m³/s) | Std Dev (m³/s) | Period |
|--------|-------------|----------------|--------|
| 0-1    | 70.0, 65.0  | 20.0           | Wet    |
| 2-6    | 55.0 → 35.0 | 20.0 → 10.0    | Dry    |
| 7-9    | 45.0 → 65.0 | 10.0 → 20.0    | Transition |
| 10-11  | 65.0, 70.0  | 20.0           | Wet    |

## Running the Example

From the repository root:

```bash
# Train the model with PAR dynamics
cargo run --release --example 07-par-model-with-inflow-state

# Or with custom config
cargo run --release --example 07-par-model-with-inflow-state -- --config examples/07-par-model-with-inflow-state/config.json
```

## Migration Notes

This example was migrated from the old format (`{"type": "periodic_ar", ...}`) to the new unified format:

```bash
python3 tools/migrate_json.py examples/07-par-model-with-inflow-state/recourse.json -i -v
```

The migration simply removed the `"type"` field - all other fields remain unchanged.

## Expected Results

The SDDP algorithm should:
- Converge to optimal policy considering AR dynamics
- Track lagged inflows in state variables
- Produce cuts that depend on both storage and lagged inflows
- Show temporal correlation in simulated trajectories

## Comparison with Independent Model

Compared to independent (IID) models:
- State space is larger (includes lagged inflows)
- Policies are more sophisticated (react to recent inflow patterns)
- Value functions depend on previous observations
- Typically better performance in out-of-sample simulation

## See Also

- `docs/json-schema-v2.md` - JSON format specification  
- `docs/migration-guide.md` - Migration instructions
- `par_derivation.pdf` - Mathematical derivation of PAR models
- Example 03 - Independent model for comparison
- Example 06 - PAR model without state tracking
