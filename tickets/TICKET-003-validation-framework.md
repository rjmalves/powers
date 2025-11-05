# [TICKET-003] Implement Validation Framework for Migration

**Sprint:** 1  
**Estimated Effort:** 3 story points (2 days)  
**Confidence:** High  
**Priority:** P1 - High (enables safe migration)

## Context

During the migration from unified to explicit lag structures, we need confidence that both implementations produce identical results. This ticket creates a comprehensive validation framework that verifies the new explicit structures contain the same data as the old unified structures.

This validation will run during the transition period and can be disabled with a feature flag once migration is complete.

## Acceptance Criteria

- [ ] Given subproblem with both old and new lag variables populated, when validation runs, then it verifies complete consistency or panics with detailed error
- [ ] Given a mismatch between old and new structures, when validation runs, then error message identifies exact entity, lag index, and conflicting values
- [ ] Given validation enabled via feature flag, when running existing tests, then all pass with validation active
- [ ] Given validation disabled, when compiling for release, then no runtime overhead from validation code
- [ ] Performance: Validation overhead should be < 1% of total subproblem creation time

## Tasks

### Implementation

- [ ] Add `migration_validation` feature flag to `Cargo.toml`
  ```toml
  [features]
  migration_validation = []
  ```

- [ ] Create `validation` module in `src/subproblem.rs`
  - Function: `validate_lag_variables_consistency`
  - Function: `validate_lag_constraints_consistency`
  - Helper: `format_validation_error`

- [ ] Implement variable validation logic
  - Iterate through old `lagged_state` with entity index
  - Determine entity type and id from temporal_models
  - Lookup corresponding entry in new structures
  - Compare variable indices
  - Generate detailed error on mismatch

- [ ] Implement constraint validation logic
  - Similar approach for `lag_fixing_constraints`
  - Validate against `load_lag_constraints` and `inflow_lag_constraints`

- [ ] Add validation calls in `Subproblem::new` (behind feature flag)
  ```rust
  #[cfg(feature = "migration_validation")]
  {
      self.validate_lag_variables()?;
      self.validate_lag_constraints()?;
  }
  ```

- [ ] Create detailed error types
  - `ValidationError::VariableMismatch`
  - `ValidationError::ConstraintMismatch`
  - `ValidationError::EntityTypeMismatch`
  - `ValidationError::MissingData`

### Testing

- [ ] Unit test: Validation passes when structures are identical
- [ ] Unit test: Validation detects variable index mismatch in load lag
- [ ] Unit test: Validation detects variable index mismatch in inflow lag
- [ ] Unit test: Validation detects missing entity in new structure
- [ ] Unit test: Validation detects extra entity in new structure
- [ ] Unit test: Validation detects wrong number of lags for entity
- [ ] Integration test: Run all existing tests with validation enabled
- [ ] Performance test: Measure validation overhead on large system (100+ entities)
- [ ] Feature flag test: Verify validation code not compiled when feature disabled

### Documentation

- [ ] Add doc comments to validation module explaining purpose
- [ ] Document how to enable validation feature for testing
- [ ] Add README note about migration validation
- [ ] Document error message format and interpretation
- [ ] Add inline comments explaining validation algorithm

## Technical Notes

### Implementation Structure

```rust
#[cfg(feature = "migration_validation")]
mod validation {
    use super::*;
    use crate::temporal_model::UncertaintyType;
    
    /// Validate that new explicit structures match old unified structure
    pub fn validate_lag_variables_consistency(
        old_vars: &Option<Vec<Vec<usize>>>,
        load_lags: &Option<LoadLagVariables>,
        inflow_lags: &Option<InflowLagVariables>,
        temporal_models: &[TemporalModel],
    ) -> Result<(), ValidationError> {
        let Some(old) = old_vars else {
            // No old variables, new should also be empty
            if load_lags.is_some() || inflow_lags.is_some() {
                return Err(ValidationError::MissingData {
                    message: "New structures have data but old is None".into(),
                });
            }
            return Ok(());
        };
        
        // Verify entity count matches
        if old.len() != temporal_models.len() {
            return Err(ValidationError::EntityCountMismatch {
                old_count: old.len(),
                model_count: temporal_models.len(),
            });
        }
        
        // For each entity in old structure
        for (entity_idx, old_entity_vars) in old.iter().enumerate() {
            let model = &temporal_models[entity_idx];
            
            // Get corresponding vars from new structure
            let new_entity_vars = match model.entity_type {
                UncertaintyType::Load => {
                    load_lags.as_ref()
                        .ok_or_else(|| ValidationError::MissingData {
                            message: format!("Load lags missing for bus {}", model.entity_id),
                        })?
                        .lags_by_bus
                        .get(model.entity_id)
                        .ok_or_else(|| ValidationError::EntityOutOfBounds {
                            entity_type: "Load",
                            entity_id: model.entity_id,
                        })?
                }
                UncertaintyType::Inflow => {
                    inflow_lags.as_ref()
                        .ok_or_else(|| ValidationError::MissingData {
                            message: format!("Inflow lags missing for hydro {}", model.entity_id),
                        })?
                        .lags_by_hydro
                        .get(model.entity_id)
                        .ok_or_else(|| ValidationError::EntityOutOfBounds {
                            entity_type: "Inflow",
                            entity_id: model.entity_id,
                        })?
                }
            };
            
            // Compare variable indices
            if old_entity_vars != new_entity_vars {
                return Err(ValidationError::VariableMismatch {
                    entity_type: format!("{:?}", model.entity_type),
                    entity_id: model.entity_id,
                    entity_idx,
                    old_vars: old_entity_vars.clone(),
                    new_vars: new_entity_vars.clone(),
                });
            }
        }
        
        Ok(())
    }
}

#[derive(Debug)]
enum ValidationError {
    VariableMismatch {
        entity_type: String,
        entity_id: usize,
        entity_idx: usize,
        old_vars: Vec<usize>,
        new_vars: Vec<usize>,
    },
    ConstraintMismatch {
        entity_type: String,
        entity_id: usize,
        entity_idx: usize,
        old_constraints: Vec<usize>,
        new_constraints: Vec<usize>,
    },
    EntityOutOfBounds {
        entity_type: &'static str,
        entity_id: usize,
    },
    EntityCountMismatch {
        old_count: usize,
        model_count: usize,
    },
    MissingData {
        message: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::VariableMismatch {
                entity_type, entity_id, entity_idx, old_vars, new_vars
            } => {
                write!(
                    f,
                    "Variable mismatch for {} {} (entity_idx={})\n  Old: {:?}\n  New: {:?}",
                    entity_type, entity_id, entity_idx, old_vars, new_vars
                )
            }
            ValidationError::ConstraintMismatch {
                entity_type, entity_id, entity_idx, old_constraints, new_constraints
            } => {
                write!(
                    f,
                    "Constraint mismatch for {} {} (entity_idx={})\n  Old: {:?}\n  New: {:?}",
                    entity_type, entity_id, entity_idx, old_constraints, new_constraints
                )
            }
            ValidationError::EntityOutOfBounds { entity_type, entity_id } => {
                write!(f, "{} {} is out of bounds", entity_type, entity_id)
            }
            ValidationError::EntityCountMismatch { old_count, model_count } => {
                write!(
                    f,
                    "Entity count mismatch: old has {} entities, temporal_models has {}",
                    old_count, model_count
                )
            }
            ValidationError::MissingData { message } => {
                write!(f, "Missing data: {}", message)
            }
        }
    }
}

impl std::error::Error for ValidationError {}
```

### Integration Points

Add validation calls after variable/constraint creation:

```rust
impl Subproblem {
    pub fn new(system: &System, ...) -> Result<Self, Error> {
        // ... create variables and constraints ...
        
        #[cfg(feature = "migration_validation")]
        {
            use validation::*;
            validate_lag_variables_consistency(
                &subproblem.variables.lagged_state,
                &subproblem.variables.load_lags,
                &subproblem.variables.inflow_lags,
                &temporal_models,
            ).map_err(|e| Error::ValidationFailed(e.to_string()))?;
            
            validate_lag_constraints_consistency(
                &subproblem.constraints.lag_fixing_constraints,
                &subproblem.constraints.load_lag_constraints,
                &subproblem.constraints.inflow_lag_constraints,
                &temporal_models,
            ).map_err(|e| Error::ValidationFailed(e.to_string()))?;
        }
        
        Ok(subproblem)
    }
}
```

### Edge Cases

- Empty systems (no entities)
- All entities with AR(0) - both old and new should be None
- Entity ID gaps (e.g., buses [0, 2, 5] - IDs 1, 3, 4 unused)
- Mismatched entity counts between system and temporal_models

### Performance Considerations

- Validation only runs when feature is enabled
- O(n) complexity where n = total number of lag variables
- Mostly pointer comparisons and small vector comparisons
- Should be negligible compared to solver operations

### Testing Strategy

Run CI with validation enabled:
```yaml
# .github/workflows/validation.yml
- name: Test with migration validation
  run: cargo test --features migration_validation
```

## Dependencies

- Blocked by: TICKET-001, TICKET-002
- Blocks: None (enables confidence in other tickets)
- Related: All migration tickets benefit from this

## Definition of Done

- [ ] Validation framework implemented and tested
- [ ] Feature flag works correctly
- [ ] All existing tests pass with validation enabled
- [ ] Error messages are clear and actionable
- [ ] CI includes validation test job
- [ ] Documentation complete
- [ ] No performance regression when validation disabled
