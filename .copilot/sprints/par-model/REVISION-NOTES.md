# PAR Model Sprint Tickets - Revision Notes

## Summary of Changes

The PAR model implementation tickets have been revised to reflect a **key architectural decision**: using the existing `season_id` field for flexible periodicity instead of hardcoding monthly periods.

## Rationale

### Original CEPEL Methodology

- CEPEL documentation references "month" as the unit for PAR periodicity
- Standard Brazilian hydrothermal dispatch models use monthly stages (period=12)
- Parameters denoted as μₘ, σₘ, φₖₘ where m = month (0..11)

### POWE.RS Implementation Decision

- **Use `season_id` field** (already present in `GraphNodeInput` and `NoiseModel` structs)
- **Support flexible periodicity**: monthly, quarterly, weekly, or custom cycles
- **More general than CEPEL**: Not restricted to monthly-only models
- **Preserves mathematical notation**: μₘ, σₘ, φₖₘ where m = period_index (0..period-1)

### Benefits

1. **Flexibility**: Users can model different time granularities
   - Monthly: `period=12` (CEPEL standard)
   - Quarterly: `period=4` (common in financial planning)
   - Custom: Any cycle matching graph structure
2. **Simplicity**: Reuses existing `season_id` infrastructure
3. **Consistency**: Aligns with existing input schema design
4. **Extensibility**: Easy to add seasonal variations for any time scale

## Tickets Revised

### Major Terminology Changes

**Changed**: "month" → "period" or "season"  
**Changed**: "monthly" → "per period" or "seasonal"  
**Changed**: "month index" → "period_index" or "season_id"

### Files Updated

1. **PAR-001**: `extend-temporal-model-enum.md`

   - Comment: "12 for monthly, 4 for quarterly, etc."
   - Comment: "Maps to season_id values in graph nodes"
   - Comment: "can differ by season!" (was "by month!")

2. **PAR-002**: `add-seasonal-stats-types.md`

   - Struct field: `period_index` (was `month`)
   - Comment: "Period index in cycle (0..period-1)"
   - Comment: "Maps to season_id in graph nodes"

3. **PAR-004**: `update-json-schemas.md`

   - Schema description: "Maps to season_id values in graph nodes"
   - Removed "monthly" from array descriptions

4. **PAR-005**: `seasonal-params-container.md`

   - Comment: "maps to season_id in graph nodes"
   - Comment: "12 for monthly, 4 for quarterly"

5. **PAR-006**: `par-generator.md`

   - Comment: "maps to season_id in graph nodes"
   - Comment: "Mean and std dev change each period (map to season_id)"

6. **PAR-009**: `scenario-integration.md`

   - Added context: "seasonal index (period m) is determined by season_id field"
   - Explains flexible periodicity support

7. **PAR-012**: `e2e-integration-tests.md`

   - Added test: "Multiple period configurations (12-period, 4-period, custom)"
   - Test naming: "12-period PAR" instead of "monthly"

8. **PAR-014**: `example-configs.md`

   - Added example: "Quarterly PAR (4-period) demonstrating flexibility"
   - Example 5: Demonstrates non-monthly periodicity
   - Title clarifications: "12-period (monthly)" to show flexibility

9. **README.md**: Sprint plan overview

   - Added "Key Design Decision" section
   - Explains season_id usage and flexibility
   - Shows examples: monthly, quarterly, custom

10. **TICKET-SUMMARY.md**: Sprint tracking
    - Added prominent "Key Design Decision" callout box
    - Lists period examples (12, 4, 52, custom)
    - Explains why tickets use "period" not "month"

## Mathematical Notation Preserved

The mathematical notation from CEPEL remains **unchanged**:

- μₘ: seasonal mean for period m
- σₘ: seasonal standard deviation for period m
- φₖₘ: AR coefficient k for period m
- pₘ: AR order for period m

The subscript "m" now means "period index" (0..period-1) instead of strictly "month".

## Code Structure Implications

### Input Types (`src/input.rs`)

```rust
// GraphNodeInput already has season_id
pub struct GraphNodeInput {
    pub id: usize,
    pub stage_id: usize,
    pub season_id: usize,  // ← Used for PAR periodicity!
    // ...
}

// NoiseModel already has season_id
pub struct NoiseModel {
    pub uncertainty_type: UncertaintyType,
    pub entity_id: usize,
    pub season_id: usize,  // ← Links to graph node's season_id!
    // ...
}
```

### PAR Generator Mapping

```rust
// In PAR generator (PAR-006):
let period_index = current_stage % params.period;

// This period_index corresponds to:
//   - season_id from GraphNodeInput
//   - season_id from NoiseModel
//   - 0..11 for monthly (period=12)
//   - 0..3 for quarterly (period=4)
//   - 0..(period-1) for any custom cycle
```

## Validation Strategy

### Schema Validation (PAR-004)

- `period` must be positive integer
- All seasonal arrays must have length = `period`
- No hardcoded maximum (was 12 in some places)

### Runtime Validation (PAR-005)

- Check `period == ar_orders.len() == seasonal_means.len() == seasonal_stds.len()`
- Check `ar_coefficients[m].len() == ar_orders[m]` for all m
- Stationarity checks apply regardless of period length

## Documentation Impact

### User-Facing Docs (PAR-015)

- Explain `season_id` → period mapping
- Show monthly example (period=12)
- Show quarterly example (period=4)
- Explain how to choose period based on graph structure

### Examples (PAR-014)

- Example 1-4: Use period=12 (standard CEPEL)
- Example 5: Use period=4 (demonstrates flexibility)
- README explains period choice rationale

## Migration Notes

### From CEPEL Monthly Models

If converting existing CEPEL monthly PAR configs:

1. Set `period: 12` in `PeriodicAutoregressive` variant
2. Ensure graph nodes have `season_id` values 0..11
3. Map μₘ, σₘ, φₖₘ directly (no changes needed)

### For Non-Monthly Models

If using quarterly or custom periods:

1. Set `period` to match your cycle length
2. Ensure graph nodes have `season_id` values 0..(period-1)
3. Provide seasonal parameters for each period

## Conclusion

This revision makes the PAR implementation **more general and flexible** than CEPEL's original specification while maintaining full backward compatibility with monthly models. The use of `season_id` aligns with POWE.RS's existing architecture and enables users to model various time granularities beyond just monthly stages.

**No functional changes** to the implementation plan—only terminology clarifications to reflect the flexible design.
