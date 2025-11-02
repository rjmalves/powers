# Example 03: Multi-Stage Stochastic Optimization

This example demonstrates multi-stage stochastic optimization with LogNormal3 distributed inflows.

## System Description

- 1 hydro plant
- 12 seasons (monthly)
- Independent temporal model (IID noise across time)
- LogNormal3 distributed inflows

## Files

- `system.json` - System configuration (hydro plants, buses, etc.)
- `graph.json` - SDDP tree structure
- `recourse.json` - Uncertainty specifications
- `config.json` - SDDP algorithm configuration

## Uncertainty Model

This example uses the **new unified temporal model format** (v0.4.0+):

```json
{
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [51.11, ...],
    "seasonal_stds": [40.64, ...],
    "ar_orders": [0, 0, ..., 0],
    "ar_coefficients": [[], [], ..., []]
  }
}
```

### Independent Model (IID)

An independent (IID) model is represented as PAR(0):
- `ar_orders`: All zeros `[0, 0, ..., 0]`
- `ar_coefficients`: All empty `[[], [], ..., []]`
- No temporal correlation between stages

### Distribution

LogNormal3 distribution with:
- `gamma`: 0.0 (location parameter)
- `mu`: 3.689 (log-space mean)
- `sigma`: 0.7 (log-space std)

This results in:
- Mean inflow: ~51.11 m³/s
- Std dev: ~40.64 m³/s

## Running the Example

From the repository root:

```bash
# Train the model
cargo run --release --example 03-multistage

# Or with custom config
cargo run --release --example 03-multistage -- --config examples/03-multistage/config.json
```

## Migration Notes

This example was migrated from the old format (`{"type": "independent"}`) to the new unified format using:

```bash
python3 tools/migrate_json.py examples/03-multistage/recourse.json -i -v
```

The old format is still supported for backward compatibility, but the new format is recommended.

## Expected Results

The SDDP algorithm should converge, providing:
- Optimal policy for each stage
- Expected cost estimate
- Water value functions

Results may differ slightly from previous versions due to the corrected LogNormal3 inverse CDF implementation in v0.4.0.

## See Also

- `docs/json-schema-v2.md` - JSON format specification
- `docs/migration-guide.md` - Migration instructions
- `tools/README.md` - Migration tool usage
