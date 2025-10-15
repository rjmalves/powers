# PAR-V2-027: Update JSON Schemas

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 6 (Documentation & Polish)  
**Story Points**: 2  
**Priority**: High  
**Status**: 🔵 Not Started

---

## Context

Update JSON schemas to include PAR-specific fields:

- `schemas/system.schema.json`: Add PAR stochastic process type
- `schemas/config.schema.json`: Add PAR-specific config if needed

**Critical**: Schemas provide IDE validation - must be correct.

**References**:
- Schemas: `schemas/`
- Validation: `tests/test_json_schemas.rs`

---

## Acceptance Criteria

- [ ] system.schema.json includes PAR fields
- [ ] Schema validates correctly
- [ ] Examples pass schema validation
- [ ] Test added in test_json_schemas.rs

---

## Tasks

- [ ] Update system.schema.json
  ```json
  {
    "stochastic_process_type": {
      "enum": ["Naive", "PAR"],
      "PAR": {
        "type": "object",
        "properties": {
          "order": {"type": "integer", "minimum": 1},
          "coefficients": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {"type": "number"}
            }
          }
        }
      }
    }
  }
  ```

- [ ] Add schema validation test
- [ ] Validate example systems against schema

---

## Estimated Effort

**2 story points** (1 day)
