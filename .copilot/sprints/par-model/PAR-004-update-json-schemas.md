# PAR-004: Update JSON Schemas for Periodic AR Support

## Context

All JSON input schemas in `schemas/` need updates to support the new `periodic_ar` temporal model and `residual_distribution` field in `NoiseModel`. This ticket ensures IDE validation (VSCode JSON schema checking) works correctly for PAR configs.

Schema updates must maintain backward compatibility and provide clear documentation.

## Acceptance Criteria

- [ ] `system.schema.json` updated with `periodic_ar` variant in TemporalModel
- [ ] `system.schema.json` includes `residual_distribution` field in NoiseModel
- [ ] Schema validation enforces required fields for PAR (period, ar_orders, etc.)
- [ ] Schema provides examples and descriptions for all PAR fields
- [ ] Existing example JSON files validate against updated schema
- [ ] New PAR example in `tests/fixtures/` validates correctly
- [ ] Schema changes documented in CHANGELOG

## Tasks

### Implementation

- [ ] Update `schemas/system.schema.json`:
  - Add `periodic_ar` to TemporalModel oneOf options
  - Define all required/optional fields with types
  - Add field descriptions with CEPEL references
  - Add residual_distribution to NoiseModel properties
  - Update examples section
- [ ] Add schema validation test in `tests/test_json_schemas.rs`
- [ ] Create fixture `tests/fixtures/par_model_basic.json`
- [ ] Validate all existing fixtures still pass schema checks
- [ ] Run `cargo test` to ensure schema parsing works

### Testing

- [ ] Schema test: periodic_ar with all required fields validates
- [ ] Schema test: periodic_ar missing required field fails validation
- [ ] Schema test: ar_orders length matches ar_coefficients length
- [ ] Schema test: period matches seasonal_means/stds length
- [ ] Schema test: residual_distribution present with periodic_ar validates
- [ ] Schema test: existing Independent/AR fixtures still validate
- [ ] Integration test: deserialize PAR config via serde_json

### Documentation

- [ ] Add doc comments to schema JSON explaining PAR semantics
- [ ] Reference CEPEL methodology in schema description
- [ ] Document all field constraints (e.g., period > 0, |φ| < 1)
- [ ] Add example JSON snippet in schema for quick copy-paste
- [ ] Update `docs/reference/INPUT-SPECIFICATION.md` with PAR section

## Technical Notes

### Schema Updates

```json
{
  "definitions": {
    "TemporalModel": {
      "oneOf": [
        {
          "type": "object",
          "properties": {
            "type": { "const": "independent" }
          },
          "required": ["type"],
          "description": "Independent noise (no temporal correlation)"
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "autoregressive" },
            "lag_order": {
              "type": "integer",
              "minimum": 1,
              "description": "AR order (1 for AR(1), 2 for AR(2), etc.)"
            },
            "coefficients": {
              "type": "array",
              "items": { "type": "number" },
              "description": "AR coefficients [φ₁, φ₂, ..., φₚ]"
            }
          },
          "required": ["type", "lag_order", "coefficients"]
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "periodic_ar" },
            "period": {
              "type": "integer",
              "minimum": 1,
              "description": "Seasonal period (e.g., 12 for monthly data, 4 for quarterly). Maps to season_id values in graph nodes. All seasonal arrays must have this length."
            },
            "ar_orders": {
              "type": "array",
              "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": 12
              },
              "description": "AR order for each period (e.g., [1, 1, 2, 1, ...] for PAR(p) with varying p). Length must equal 'period'."
            },
            "ar_coefficients": {
              "type": "array",
              "items": {
                "type": "array",
                "items": { "type": "number" }
              },
              "description": "AR coefficients for each period. Outer array length = period, inner array[i] length = ar_orders[i]. Example: [[0.7], [0.75], [0.6, 0.2]] for AR(1), AR(1), AR(2)."
            },
            "seasonal_means": {
              "type": "array",
              "items": { "type": "number" },
              "description": "Seasonal mean μₘ for each period (e.g., [100, 120, 150, ...] for inflows). Length = period."
            },
            "seasonal_stds": {
              "type": "array",
              "items": {
                "type": "number",
                "exclusiveMinimum": 0
              },
              "description": "Seasonal standard deviation σₘ for each period (e.g., [20, 25, 30, ...]). Must be positive. Length = period."
            }
          },
          "required": [
            "type",
            "period",
            "ar_orders",
            "ar_coefficients",
            "seasonal_means",
            "seasonal_stds"
          ],
          "description": "Periodic Autoregressive PAR(p) model (CEPEL methodology). AR parameters vary by season/month. See PAR_MODEL_SUPPORT.md for details."
        }
      ]
    },
    "NoiseModel": {
      "type": "object",
      "properties": {
        "uncertainty_type": {
          "type": "string",
          "enum": ["inflow", "load"]
        },
        "entity_id": {
          "type": "integer",
          "minimum": 0
        },
        "season_id": {
          "type": "integer",
          "minimum": 0
        },
        "marginal_distribution": {
          "$ref": "#/definitions/MarginalDistribution",
          "description": "Distribution for final series (Independent) or innovations (Autoregressive). **Ignored for periodic_ar** - use residual_distribution instead."
        },
        "innovation_distribution": {
          "$ref": "#/definitions/InnovationDistribution",
          "description": "Optional innovation distribution for stationary AR models."
        },
        "temporal_model": {
          "$ref": "#/definitions/TemporalModel"
        },
        "residual_distribution": {
          "$ref": "#/definitions/MarginalDistribution",
          "description": "Distribution applied to de-seasonalized residuals aₜ in PAR models (CEPEL). **Required** for periodic_ar, **must be null** for independent/autoregressive."
        }
      },
      "required": [
        "uncertainty_type",
        "entity_id",
        "season_id",
        "marginal_distribution",
        "temporal_model"
      ],
      "description": "Noise model for a single entity. Distribution semantics depend on temporal_model type."
    }
  }
}
```

### Example Fixture (tests/fixtures/par_model_basic.json)

Create minimal valid PAR config:

```json
{
  "system": {
    "initial_stage": 0,
    "num_stages": 12,
    "hydros": [
      {
        "id": 0,
        "name": "Reservoir A",
        "storage_min": 0.0,
        "storage_max": 1000.0,
        "storage_initial": 500.0,
        "inflow_initial": [100.0],
        "production_coefficient": 1.0
      }
    ],
    "buses": [],
    "generators": [],
    "demands": []
  },
  "noise_models": [
    {
      "uncertainty_type": "inflow",
      "entity_id": 0,
      "season_id": 0,
      "marginal_distribution": {
        "type": "normal",
        "mean": 0.0,
        "std_dev": 1.0
      },
      "residual_distribution": {
        "type": "lognormal3",
        "gamma": 1.0,
        "mu": 4.5,
        "sigma": 0.3
      },
      "temporal_model": {
        "type": "periodic_ar",
        "period": 12,
        "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "ar_coefficients": [
          [0.7],
          [0.75],
          [0.8],
          [0.7],
          [0.65],
          [0.6],
          [0.6],
          [0.65],
          [0.7],
          [0.75],
          [0.8],
          [0.75]
        ],
        "seasonal_means": [
          100.0, 120.0, 150.0, 180.0, 200.0, 180.0, 150.0, 120.0, 100.0, 90.0,
          80.0, 90.0
        ],
        "seasonal_stds": [
          20.0, 25.0, 30.0, 35.0, 40.0, 35.0, 30.0, 25.0, 20.0, 18.0, 15.0, 18.0
        ]
      }
    }
  ],
  "correlation": {
    "spatial_correlation_matrix": [[1.0]],
    "temporal_correlation_lags": []
  }
}
```

### Test Structure

Add to `tests/test_json_schemas.rs`:

```rust
#[test]
fn test_par_model_schema_validation() {
    // Test 1: Valid PAR config
    let json = include_str!("fixtures/par_model_basic.json");
    let config: SystemInput = serde_json::from_str(json).expect("PAR config should deserialize");
    assert_eq!(config.noise_models.len(), 1);

    // Test 2: Verify fields
    let noise = &config.noise_models[0];
    match &noise.temporal_model {
        TemporalModel::PeriodicAutoregressive { period, ar_orders, .. } => {
            assert_eq!(*period, 12);
            assert_eq!(ar_orders.len(), 12);
        }
        _ => panic!("Expected PeriodicAutoregressive variant"),
    }

    // Test 3: Residual distribution present
    assert!(noise.residual_distribution.is_some());
}

#[test]
fn test_par_model_missing_residual_distribution() {
    // Schema should allow deserialization but runtime validation should catch it
    let json = r#"{
        "uncertainty_type": "inflow",
        "entity_id": 0,
        "season_id": 0,
        "marginal_distribution": {"type": "normal", "mean": 0.0, "std_dev": 1.0},
        "temporal_model": {
            "type": "periodic_ar",
            "period": 12,
            "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "ar_coefficients": [[0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]],
            "seasonal_means": [100.0; 12],
            "seasonal_stds": [20.0; 12]
        }
    }"#;

    let noise: NoiseModel = serde_json::from_str(json).expect("Should deserialize");

    // Runtime validation should catch missing residual_distribution
    let result = noise.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("residual_distribution"));
}

#[test]
fn test_par_model_length_mismatch() {
    // ar_orders length != period should fail validation
    let json = r#"{
        "uncertainty_type": "inflow",
        "entity_id": 0,
        "season_id": 0,
        "marginal_distribution": {"type": "normal", "mean": 0.0, "std_dev": 1.0},
        "residual_distribution": {"type": "lognormal3", "gamma": 1.0, "mu": 4.5, "sigma": 0.3},
        "temporal_model": {
            "type": "periodic_ar",
            "period": 12,
            "ar_orders": [1, 1, 1],
            "ar_coefficients": [[0.7], [0.7], [0.7]],
            "seasonal_means": [100.0, 100.0, 100.0],
            "seasonal_stds": [20.0, 20.0, 20.0]
        }
    }"#;

    let noise: NoiseModel = serde_json::from_str(json).expect("Should deserialize");
    let result = noise.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("length"));
}
```

## Dependencies

- **Blocked by**: PAR-001 (TemporalModel enum), PAR-003 (NoiseModel changes)
- **Blocks**: PAR-009 (integration tests need valid fixtures)
- **Related**: All PAR tickets (schema is contract for JSON input)

## Estimated Effort

**2 story points** (confidence: high)

- 2 hours schema updates (JSON definitions, descriptions)
- 2 hours fixture creation and validation
- 1 hour test implementation
- 1 hour documentation updates

### Breakdown

- Update system.schema.json: 1.5 hours
- Create PAR fixture JSON: 1 hour
- Write schema validation tests: 1 hour
- Update INPUT-SPECIFICATION.md: 0.5 hours
- Validate all existing fixtures: 0.5 hours
- Code review: 0.5 hours
