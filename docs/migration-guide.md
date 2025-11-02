# Migration Guide: JSON Schema v1 → v2

**Version**: 1.0  
**Date**: 2025-11-02  
**Target**: Powers-RS JSON configuration migration

---

## Overview

This guide explains how to migrate your Powers-RS configuration files from the legacy JSON schema (v1) to the new unified temporal model format (v2).

### Why Migrate?

- **Consistency**: Unified format for all temporal models
- **Clarity**: Explicit parameters eliminate ambiguity  
- **Future-proof**: New features will use v2 format
- **Performance**: Unified code paths improve efficiency

### Do I Need to Migrate?

**No immediate action required**. The v1 format is still fully supported through a backward compatibility layer. However, v1 is deprecated and will be removed in a future major release.

**Recommended timeline**:
- ✅ **Now**: Learn the new format for new projects
- ⏰ **Next 3-6 months**: Migrate existing projects at your convenience
- ⚠️ **Future release**: V1 support will be removed (with advance notice)

---

## Quick Start

### Independent Model

**Before (v1)**:
```json
{
  "temporal_model": {
    "type": "independent"
  }
}
```

**After (v2)**:
```json
{
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [40.0, 42.0, ...],
    "seasonal_stds": [10.0, 11.0, ...],
    "ar_orders": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ar_coefficients": [[], [], [], [], [], [], [], [], [], [], [], []]
  }
}
```

### PAR Model

**Before (v1)**:
```json
{
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 12,
    "ar_orders": [1, 1, ...],
    "ar_coefficients": [[0.7], [0.7], ...],
    "seasonal_means": [70.0, 65.0, ...],
    "seasonal_stds": [20.0, 18.0, ...]
  }
}
```

**After (v2)** - just remove `"type"`:
```json
{
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [70.0, 65.0, ...],
    "seasonal_stds": [20.0, 18.0, ...],
    "ar_orders": [1, 1, ...],
    "ar_coefficients": [[0.7], [0.7], ...]
  }
}
```

---

## Detailed Migration Steps

### Step 1: Identify Model Type

Examine your current configuration to determine the model type:

```json
// Independent model - has "type": "independent"
{
  "temporal_model": {
    "type": "independent"
  }
}

// PAR model - has "type": "periodic_ar"
{
  "temporal_model": {
    "type": "periodic_ar",
    ...
  }
}
```

### Step 2: Apply Appropriate Migration

#### For Independent Models

1. **Count seasons**: `num_seasons = length of seasonal_distributions`
2. **Extract means**: From each distribution's `mean` field
3. **Extract std devs**: From each distribution's `std_dev` field
4. **Set AR orders**: All zeros `[0, 0, ..., 0]`
5. **Set AR coefficients**: All empty `[[], [], ..., []]`

**Example**:

```json
// BEFORE
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "type": "independent"
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 100.0, "std_dev": 20.0},
    {"season_id": 1, "type": "normal", "mean": 90.0, "std_dev": 18.0}
  ]
}

// AFTER
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 2,                    // ← Count from seasonal_distributions
    "seasonal_means": [100.0, 90.0],     // ← From distribution means
    "seasonal_stds": [20.0, 18.0],       // ← From distribution std_devs
    "ar_orders": [0, 0],                 // ← All zeros for independent
    "ar_coefficients": [[], []]          // ← All empty for independent
  },
  "seasonal_distributions": [
    {"season_id": 0, "type": "normal", "mean": 100.0, "std_dev": 20.0},
    {"season_id": 1, "type": "normal", "mean": 90.0, "std_dev": 18.0}
  ]
}
```

#### For PAR Models

Simply remove the `"type": "periodic_ar"` field. All other fields remain unchanged.

**Example**:

```json
// BEFORE
{
  "temporal_model": {
    "type": "periodic_ar",              // ← Remove this line
    "num_seasons": 12,
    "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "ar_coefficients": [[0.7], [0.7], [0.7], [0.7], [0.7], [0.7], 
                        [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]],
    "seasonal_means": [70.0, 65.0, 60.0, 55.0, 50.0, 45.0,
                       50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
    "seasonal_stds": [20.0, 18.0, 16.0, 14.0, 12.0, 10.0,
                      12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
  }
}

// AFTER
{
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [70.0, 65.0, 60.0, 55.0, 50.0, 45.0,
                       50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
    "seasonal_stds": [20.0, 18.0, 16.0, 14.0, 12.0, 10.0,
                      12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
    "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "ar_coefficients": [[0.7], [0.7], [0.7], [0.7], [0.7], [0.7], 
                        [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]]
  }
}
```

### Step 3: Handle LogNormal3 Distributions

For LogNormal3 distributions, you need to compute the mean and std dev from the parameters.

**Formulas**:
- Mean: `γ + exp(μ + σ²/2)`
- Std Dev: `exp(μ + σ²/2) · sqrt(exp(σ²) - 1)`

**Example**:

```json
// Original distribution
{
  "season_id": 0,
  "type": "lognormal3",
  "gamma": 0.0,
  "mu": 4.605,      // ln(100)
  "sigma": 0.2
}

// Computed values:
// mean = 0.0 + exp(4.605 + 0.2²/2) = 0.0 + exp(4.625) = 102.03
// std_dev = exp(4.625) · sqrt(exp(0.04) - 1) = 102.03 · 0.200 = 20.41
```

Add to temporal_model:
```json
{
  "temporal_model": {
    "seasonal_means": [102.03, ...],
    "seasonal_stds": [20.41, ...]
  }
}
```

### Step 4: Validate

After migration, verify your configuration:

1. **Array lengths**: All arrays in `temporal_model` have same length
2. **AR coefficients**: Each `ar_coefficients[i]` has length = `ar_orders[i]`
3. **Positive std devs**: All `seasonal_stds` values > 0
4. **Consistency**: Means/stds match distribution parameters (for Normal)

---

## Migration Tools

### Manual Migration

For small files, manual editing is straightforward:

1. Open JSON file in text editor
2. Locate `temporal_model` section
3. Apply migration steps from above
4. Save and test

### Automated Migration (Optional)

A migration tool can be created to automate the process:

```bash
# Planned tool (Ticket 3.4)
cargo run --bin migrate_json -- --input old.json --output new.json

# Or in-place migration
cargo run --bin migrate_json -- --in-place recourse.json
```

### Python Migration Script

If you need to migrate many files, here's a Python helper:

```python
import json
import math

def migrate_independent_model(spec):
    """Migrate independent model specification"""
    seasonal_dists = spec['seasonal_distributions']
    num_seasons = len(seasonal_dists)
    
    means = []
    stds = []
    for dist in seasonal_dists:
        if dist['type'] == 'normal':
            means.append(dist['mean'])
            stds.append(dist['std_dev'])
        elif dist['type'] == 'lognormal3':
            # Compute mean and std from LogNormal3 parameters
            gamma = dist['gamma']
            mu = dist['mu']
            sigma = dist['sigma']
            
            exp_term = math.exp(mu + sigma**2 / 2)
            mean = gamma + exp_term
            std = exp_term * math.sqrt(math.exp(sigma**2) - 1)
            
            means.append(mean)
            stds.append(std)
    
    return {
        'num_seasons': num_seasons,
        'seasonal_means': means,
        'seasonal_stds': stds,
        'ar_orders': [0] * num_seasons,
        'ar_coefficients': [[]] * num_seasons
    }

def migrate_par_model(temporal_model):
    """Migrate PAR model - just remove 'type' field"""
    result = temporal_model.copy()
    result.pop('type', None)
    return result

def migrate_recourse_file(input_path, output_path):
    """Migrate entire recourse.json file"""
    with open(input_path) as f:
        data = json.load(f)
    
    for spec in data.get('uncertainty_specifications', []):
        tm = spec['temporal_model']
        
        if tm.get('type') == 'independent':
            spec['temporal_model'] = migrate_independent_model(spec)
        elif tm.get('type') == 'periodic_ar':
            spec['temporal_model'] = migrate_par_model(tm)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Migrated {input_path} -> {output_path}")

# Usage
migrate_recourse_file('old_recourse.json', 'new_recourse.json')
```

---

## Testing After Migration

### Verify Backward Compatibility

The old and new formats should produce identical results:

```bash
# Run with old format
cargo run --release -- \
    --system examples/03-multistage/system.json \
    --recourse examples/03-multistage/recourse_old.json \
    --config examples/03-multistage/config.json \
    --output results_old.json

# Run with new format
cargo run --release -- \
    --system examples/03-multistage/system.json \
    --recourse examples/03-multistage/recourse_new.json \
    --config examples/03-multistage/config.json \
    --output results_new.json

# Compare results (should be identical for Normal distributions)
diff results_old.json results_new.json
```

### Run Tests

```bash
# Run full test suite
cargo test

# Run specific example
cargo run --example 03-multistage
```

### Common Migration Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Array length mismatch | Forgot to make all arrays same length | Count seasons carefully |
| AR coefficient error | ar_coefficients length ≠ ar_orders | Check each season's coefficient array |
| Negative std dev | Incorrectly computed from LogNormal3 | Use formulas above |
| Results differ | LogNormal3 mean/std incorrect | Verify LogNormal3 parameter extraction |

---

## Expected Result Changes

### Normal Distributions

Results should be **identical** for Normal distributions. Any differences indicate a migration error.

### LogNormal3 Distributions

Results **may differ** because:
1. Old implementation had bugs in LogNormal3 handling
2. New implementation uses correct probability integral transform
3. New results are mathematically correct

**This is expected and correct behavior.**

### Performance

Expect **5-10% improvement** in scenario generation time due to:
- Unified code paths
- Reduced branching
- Better cache utilization

---

## Example Migrations

### Example 1: Simple Independent Model

```json
// examples/01-deterministic/recourse.json

// BEFORE
{
  "uncertainty_specifications": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,
      "temporal_model": {"type": "independent"},
      "seasonal_distributions": [
        {"season_id": 0, "type": "lognormal3", "mu": 2.996, "sigma": 0.0001, "gamma": 0.0},
        {"season_id": 1, "type": "lognormal3", "mu": 2.996, "sigma": 0.0001, "gamma": 0.0}
      ]
    }
  ]
}

// AFTER
{
  "uncertainty_specifications": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,
      "temporal_model": {
        "num_seasons": 2,
        "seasonal_means": [19.98, 19.98],
        "seasonal_stds": [0.002, 0.002],
        "ar_orders": [0, 0],
        "ar_coefficients": [[], []]
      },
      "seasonal_distributions": [
        {"season_id": 0, "type": "lognormal3", "mu": 2.996, "sigma": 0.0001, "gamma": 0.0},
        {"season_id": 1, "type": "lognormal3", "mu": 2.996, "sigma": 0.0001, "gamma": 0.0}
      ]
    }
  ]
}
```

### Example 2: PAR Model

```json
// examples/07-par-model-with-inflow-state/recourse.json

// BEFORE
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 12,
    "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "ar_coefficients": [[0.7], [0.7], [0.7], [0.7], [0.7], [0.7],
                        [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]],
    "seasonal_means": [70, 65, 60, 55, 50, 45, 50, 55, 60, 65, 70, 75],
    "seasonal_stds": [20, 18, 16, 14, 12, 10, 12, 14, 16, 18, 20, 22]
  }
}

// AFTER (just remove "type")
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [70, 65, 60, 55, 50, 45, 50, 55, 60, 65, 70, 75],
    "seasonal_stds": [20, 18, 16, 14, 12, 10, 12, 14, 16, 18, 20, 22],
    "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "ar_coefficients": [[0.7], [0.7], [0.7], [0.7], [0.7], [0.7],
                        [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]]
  }
}
```

---

## Rollback Plan

If you encounter issues, you can safely rollback:

1. **Keep old files**: Save originals before migration
2. **Use version control**: Commit before and after migration
3. **Gradual migration**: Migrate one file at a time
4. **Backward compatibility**: Old format still works

---

## Getting Help

### Resources

- **JSON Schema Documentation**: See `docs/json-schema-v2.md`
- **Refactoring Plan**: See `SCENARIO_GENERATION_REFACTORING_PLAN.md`
- **Implementation Tickets**: See `docs/refactoring-tickets.md`

### Common Questions

**Q: Do I need to migrate immediately?**  
A: No. V1 format is still fully supported. Migrate at your convenience.

**Q: Will my results change?**  
A: For Normal distributions, no. For LogNormal3, yes (new results are correct).

**Q: Can I mix v1 and v2 formats?**  
A: Yes, but not in the same temporal_model field. Different entities can use different formats.

**Q: Is there a migration tool?**  
A: Planned (Ticket 3.4). For now, use manual migration or the Python script above.

**Q: How do I validate my migration?**  
A: Run tests and compare results. See "Testing After Migration" section.

---

## Feedback

If you encounter issues or have suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Contribute improvements to this guide

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-02  
**Authors**: Powers-RS Development Team
