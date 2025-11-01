# API Migration Guide: Old → New Uncertainty Model

**Date**: November 1, 2025  
**Purpose**: Guide for migrating from fragmented API to unified `uncertainty_model`

---

## Quick Reference Table

| Old API (Deleted) | New API (Current) | Notes |
|-------------------|-------------------|-------|
| `unified_noise_spec::UnifiedNoiseSpec` | `uncertainty_model::UncertaintyModel` | Enum variant changed |
| `unified_noise_spec::TemporalModelSpec` | No longer exists | Use `UncertaintyModel` variants directly |
| `unified_noise_spec::SeasonalNoiseParams` | `uncertainty_model::SeasonalParams` | Simplified struct |
| `unified_inflow_model::UnifiedInflowModel` | `uncertainty_model::UncertaintyModel` | Consolidated |
| `seasonal_params::SeasonalParams` | `uncertainty_model::SeasonalParams` | Moved and simplified |

---

## Architecture Changes

### Old Architecture (Fragmented)

```rust
// OLD: Multiple modules, unclear ownership
use crate::unified_noise_spec::UnifiedNoiseSpec;
use crate::unified_inflow_model::UnifiedInflowModel;
use crate::seasonal_params::SeasonalParams;

// OLD: Complex structure with HashMap
let spec = UnifiedNoiseSpec {
    uncertainty_type: UncertaintyType::Inflow,
    entity_id: 0,
    temporal_model: TemporalModelSpec::Independent,
    seasonal_params: HashMap::new(),  // HashMap with season -> params mapping
};
```

### New Architecture (Unified)

```rust
// NEW: Single module, clear ownership
use crate::uncertainty_model::{UncertaintyModel, SeasonalParams, DistributionType};

// NEW: Simplified enum-based structure
let model = UncertaintyModel::Independent {
    seasonal_params: vec![
        SeasonalParams {
            mean: 100.0,
            std_dev: 20.0,
            distribution: DistributionType::Normal,
        }
    ],
};
```

---

## Migration Patterns

### Pattern 1: Simple Independent Model (Test Code)

**OLD**:
```rust
let unified_specs = vec![unified_noise_spec::UnifiedNoiseSpec {
    uncertainty_type: input::UncertaintyType::Inflow,
    entity_id: 0,
    temporal_model: unified_noise_spec::TemporalModelSpec::Independent,
    seasonal_params: HashMap::new(),
}];
```

**NEW**:
```rust
// For test fixtures with minimal setup, use uncertainty_model::UncertaintyModel
// The exact API depends on how UncertaintyModel is used in the actual code.
// Check src/uncertainty_model.rs for the enum definition.

// If the test just needs a placeholder, typically:
let uncertainty_models = vec![
    uncertainty_model::UncertaintyModel::Independent {
        seasonal_params: vec![uncertainty_model::SeasonalParams {
            mean: 100.0,
            std_dev: 20.0,
            distribution: uncertainty_model::DistributionType::Normal,
        }],
    }
];
```

### Pattern 2: PAR Model with Seasonal Parameters

**OLD**:
```rust
let mut seasonal_params = HashMap::new();
seasonal_params.insert(
    0,  // season_id
    unified_noise_spec::SeasonalNoiseParams {
        mean: 100.0,
        std_dev: 20.0,
        distribution: MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 },
    },
);

let spec = unified_noise_spec::UnifiedNoiseSpec {
    temporal_model: unified_noise_spec::TemporalModelSpec::PeriodicAR {
        ar_orders: vec![1],
        ar_coefficients: vec![vec![0.7]],
        period: 12,
    },
    seasonal_params,
    // ...
};
```

**NEW**:
```rust
let model = uncertainty_model::UncertaintyModel::PeriodicAR {
    par_params: uncertainty_model::PARParams {
        ar_orders: vec![1],
        ar_coefficients: vec![vec![0.7]],
        seasonal_means: vec![100.0; 12],
        seasonal_stds: vec![20.0; 12],
        seasonal_distributions: vec![uncertainty_model::DistributionType::Normal; 12],
        period: 12,
    },
};
```

---

## Step-by-Step Migration

### Step 1: Update Imports

**Find and replace in test modules**:

```rust
// OLD imports (remove these)
use crate::unified_noise_spec;
use crate::unified_inflow_model;
use crate::seasonal_params;

// NEW imports (add these)
use crate::uncertainty_model::{UncertaintyModel, SeasonalParams, DistributionType};
```

### Step 2: Update Type References

**Simple search-replace** (works for ~90% of cases):

```bash
# In most files
unified_noise_spec::UnifiedNoiseSpec → uncertainty_model::UncertaintyModel
unified_noise_spec::TemporalModelSpec → [NEEDS MANUAL REVIEW - NO DIRECT EQUIVALENT]
unified_noise_spec::SeasonalNoiseParams → uncertainty_model::SeasonalParams
unified_inflow_model → uncertainty_model
seasonal_params::SeasonalParams → uncertainty_model::SeasonalParams
```

### Step 3: Update Constructors (Requires Manual Review)

This is the part that needs attention:

1. **Identify the temporal model** (Independent or PeriodicAR)
2. **Match to correct enum variant**
3. **Convert HashMap to Vec** (seasonal_params)
4. **Update field names** if changed

---

## Common Issues & Solutions

### Issue 1: "TemporalModelSpec no longer exists"

**Problem**: Old code uses `TemporalModelSpec::Independent`

**Solution**: Use enum variants directly

```rust
// OLD
temporal_model: unified_noise_spec::TemporalModelSpec::Independent

// NEW - This field doesn't exist anymore!
// Instead, use the appropriate UncertaintyModel variant:
UncertaintyModel::Independent { ... }  // or
UncertaintyModel::PeriodicAR { ... }
```

### Issue 2: "HashMap → Vec conversion"

**Problem**: Old API used `HashMap<season_id, params>`, new uses `Vec<params>`

**Solution**: Convert to vector indexed by season

```rust
// OLD
let mut map = HashMap::new();
map.insert(0, params);
map.insert(1, params);

// NEW
let vec = vec![params; 12];  // For 12 seasons
```

### Issue 3: "Field names changed"

**Problem**: `seasonal_params` field structure changed

**Solution**: Check new struct definition in `uncertainty_model.rs`

```rust
// OLD
SeasonalNoiseParams {
    mean, std_dev, distribution
}

// NEW
SeasonalParams {
    mean, std_dev, distribution  // Similar but may have different types
}
```

---

## Testing Migration

### Verification Checklist

After migrating test code:

```bash
# 1. Check compilation
cargo test --lib 2>&1 | grep "error\[E" | wc -l  # Should decrease

# 2. Format code
cargo fmt --all

# 3. Check clippy warnings
cargo clippy --all-targets --all-features -- -D warnings

# 4. Try running tests
cargo test --lib  # Should compile, may have test failures (fix those next)
```

### Progressive Testing

Test files one at a time:

```bash
# After fixing src/state.rs
cargo test --lib state  # Run only state tests

# After fixing src/subproblem.rs
cargo test --lib subproblem  # Run only subproblem tests
```

---

## Real Example: Fixing src/state.rs Tests

### Before (Broken):
```rust
#[test]
fn test_new_storage_state() {
    let system = system::System::default();
    let unified_specs = vec![unified_noise_spec::UnifiedNoiseSpec {
        uncertainty_type: input::UncertaintyType::Inflow,
        entity_id: 0,
        temporal_model: unified_noise_spec::TemporalModelSpec::Independent,
        seasonal_params: HashMap::new(),
    }];
    let state = StorageState::new(&system, &unified_specs);
    // ...
}
```

### After (Fixed):
```rust
#[test]
fn test_new_storage_state() {
    let system = system::System::default();
    
    // Check what StorageState::new() actually expects
    // It may expect Vec<UncertaintyModel> or a different structure
    // This is a placeholder - needs verification against actual API
    
    let uncertainty_models = vec![
        uncertainty_model::UncertaintyModel::Independent {
            seasonal_params: vec![uncertainty_model::SeasonalParams {
                mean: 100.0,
                std_dev: 20.0,
                distribution: uncertainty_model::DistributionType::Normal,
            }],
        }
    ];
    
    let state = StorageState::new(&system, &uncertainty_models);
    // ...
}
```

**Note**: The exact API needs to be checked in the actual function signatures.

---

## Next Steps

1. **Understand target API**: Read `src/uncertainty_model.rs` docs
2. **Check function signatures**: See what `StorageState::new()`, `SubProblem::new()`, etc. actually expect
3. **Fix imports first**: Update all `use` statements
4. **Fix types next**: Update type references
5. **Fix constructors last**: Update how objects are created
6. **Test incrementally**: Fix one file, test, move to next

---

## Need Help?

### Check These Files First

- `src/uncertainty_model.rs` - New API definition
- `src/input.rs` - Input types that uncertainty_model uses
- `src/scenario_generator.rs` - Example usage of new API

### Use Copilot Agents

```bash
# For understanding new API
gh copilot suggest "Explain uncertainty_model::UncertaintyModel enum variants and when to use each"

# For fixing specific test
gh copilot suggest "Fix this test to use uncertainty_model instead of unified_noise_spec: [paste test code]"
```

---

**Key Takeaway**: Most changes are straightforward find-replace, but constructor patterns need manual review. Start with imports, then types, then constructors.
