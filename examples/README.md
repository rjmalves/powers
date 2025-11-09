# Powers-rs Examples

This directory contains example systems demonstrating various features of the powers-rs SDDP solver.

## Available Examples

### Basic Examples

**01-deterministic** - Deterministic hydrothermal system (no uncertainty)
- Single stage
- Demonstrates basic system setup

**02-stochastic** - Single-stage stochastic optimization  
- IID load and inflow uncertainty
- Introduction to uncertainty modeling

### Multi-Stage Examples

**03-multistage** ⭐ - Multi-stage with LogNormal3 inflows
- 12 seasons (monthly)
- Independent (IID) temporal model
- LogNormal3 distributed inflows
- Good starting point for learning

**04-cascade** - Cascade hydro system
- Multiple hydros in series
- Demonstrates upstream/downstream relationships

**05-large-scale-brazilian** - Large-scale Brazilian system
- 161 uncertainty specifications
- Real-world system complexity
- Performance benchmarking

### Advanced Temporal Models

**06-par-model** - PAR model without state tracking
- Periodic Autoregressive dynamics
- Simpler state space (storage only)

**07-par-model-with-inflow-state** ⭐ - PAR model with state tracking
- AR(1) temporal correlation
- Lagged inflows as state variables
- Advanced uncertainty modeling

## JSON Format (v0.4.0+)

All examples use the **new unified temporal model format**:

### Independent Model (IID)
```json
{
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [50.0, 55.0, ...],
    "seasonal_stds": [20.0, 25.0, ...],
    "ar_orders": [0, 0, ..., 0],
    "ar_coefficients": [[], [], ..., []]
  }
}
```

### PAR Model
```json
{
  "temporal_model": {
    "num_seasons": 12,
    "ar_orders": [1, 1, ..., 1],
    "ar_coefficients": [[0.7], [0.7], ...],
    "seasonal_means": [70.0, 65.0, ...],
    "seasonal_stds": [20.0, 20.0, ...]
  }
}
```

## Migration from Old Format

All examples have been migrated from the old format (`{"type": "independent"}` or `{"type": "periodic_ar"}`) to the new unified format.

To migrate your own JSON files:

```bash
# Dry run (preview changes)
python3 tools/migrate_json.py your_recourse.json -d

# Migrate in place
python3 tools/migrate_json.py your_recourse.json -i -v

# Migrate all examples (already done)
for file in examples/*/recourse.json; do
    python3 tools/migrate_json.py "$file" -i -v
done
```

See `tools/README.md` for more details.

## Running Examples

From the repository root:

```bash
# Run an example
cargo run --release --example 03-multistage

# With custom config
cargo run --release --example 03-multistage -- --config examples/03-multistage/config.json

# Run tests
cargo test --lib
```

## File Structure

Each example directory contains:
- `system.json` - System configuration (buses, hydros, thermals, lines)
- `graph.json` - SDDP tree structure (stages, branching)
- `recourse.json` - Uncertainty specifications (temporal models)
- `config.json` - Algorithm parameters (iterations, tolerance, etc.)
- `README.md` - Example-specific documentation (where available)

## Choosing an Example

**New users**: Start with **03-multistage** (simple, well-documented)

**Learning PAR models**: Progress through:
1. 03-multistage (Independent/IID)
2. 06-par-model (PAR without state)
3. 07-par-model-with-inflow-state (PAR with state)

**Performance testing**: Use **05-large-scale-brazilian**

**System design**: Study **04-cascade** for multi-reservoir systems

## Documentation

- `docs/json-schema-v2.md` - Complete JSON format specification
- `docs/migration-guide.md` - Migration from old to new format
- `docs/guides/` - User guides and tutorials
- `CHANGELOG.md` - Release notes and version history

## Backward Compatibility

The old JSON format (`{"type": "independent"}`) still works but is deprecated. The new unified format is recommended for:
- Consistency (Independent = PAR(0))
- Clarity (explicit parameters)
- Future features (load AR models)

## Expected Results

After the v0.4.0 LogNormal3 inverse CDF fix:
- Normal distribution results: Unchanged
- LogNormal3 results: Different (corrected values)

See migration guide for details on expected changes.

## Output Configuration (v0.4.0+)

**All examples use the new indexed output format.** This format uses integer indices instead of variable names for consistency and efficiency.

### Output Files Generated

When you run an example, you'll get:

**Dictionary Files** (decode indices):
- `variable_dictionary.csv` - Maps variable indices to variable info
- `coefficient_dictionary.csv` - Maps cut coefficient indices  
- `state_component_dictionary.csv` - Maps state component indices

**Result Files** (indexed format):
- `training.csv` - Iteration convergence metrics
- `simulation.csv` - Simulation results (single normalized file)
- `cuts.csv` - Benders cuts with coefficient indices
- `states.csv` - Visited states with component indices
- `forward_detail.csv` - Forward pass details (if enabled)
- `backward_detail.csv` - Backward pass details (if enabled)
- `sampled_noises.csv` - Sampled noises (if enabled)

### Configuration Example

```json
{
  "num_iterations": 32,
  "num_forward_passes": 4,
  "num_simulation_scenarios": 128,
  "output_path": "./examples/03-multistage",
  "output": {
    "export_training": true,
    "export_cuts": true,
    "export_states": true,
    "export_simulation": true
  }
}
```

### Reading Indexed Outputs

**Step 1**: Load dictionary
```python
import pandas as pd

# Load variable dictionary
var_dict = pd.read_csv('variable_dictionary.csv', index_col='variable_index')

# Load simulation results  
sim = pd.read_csv('simulation.csv')

# Decode variable names
sim['variable_name'] = sim['variable_index'].map(var_dict['variable_name'])
```

**Step 2**: Filter and analyze
```python
# Get all water values
water_values = sim[sim['variable_name'] == 'water_value']

# Group by stage
by_stage = water_values.groupby('stage')['value'].mean()
```

### Output Schema Reference

**simulation.csv**:
```
stage,series,variable_index,entity_id,value
0,0,3,0,54.5  # sampled_inflow for hydro 0
0,0,4,0,84.6  # final_storage for hydro 0
```

**cuts.csv**:
```
stage_index,stage_cut_id,iteration,forward_pass_idx,active,coefficient_index,value
0,0,1,0,true,0,59094.8  # RHS
0,0,1,0,true,1,-100.0   # storage coefficient
```

**states.csv**:
```
stage_index,dominating_cut_id,iteration,forward_pass_idx,state_component_index,value
0,116,1,0,0,825.0  # objective
0,116,1,0,1,84.5   # storage_0
```

### Breaking Change from v0.3.x

**Old Format** (v0.3.x):
- Multiple files: `simulation_buses.csv`, `simulation_hydros.csv`, etc.
- Wide format with named columns
- Optional indexed mode

**New Format** (v0.4.0+):
- Single `simulation.csv` file
- Normalized format (one value per row)
- Always indexed (requires dictionaries)
- 20-30% smaller files

## Contributing

When adding new examples:
1. Use the new unified JSON format
2. Include a README.md with description and usage
3. Provide realistic parameters
4. Document expected behavior
5. Test before committing

## See Also

- Main README: `/README.md`
- Documentation: `/docs/`
- Migration tool: `/tools/`
- Tests: `/tests/`
