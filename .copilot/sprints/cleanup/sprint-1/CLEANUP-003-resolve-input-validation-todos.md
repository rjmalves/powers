# CLEANUP-003: Resolve Input Validation TODOs

## Context

The `input_validation.rs` file contains two TODOs related to missing validation logic. These are marked as **MEDIUM RISK** because they could allow invalid input configurations to pass validation, leading to runtime errors or incorrect results.

**Locations**:
- Line 541: `// TODO: Add detailed validation for uncertainty_specifications`
- Line 553: `// TODO: Implement cross-file consistency checks as needed.`

**Decision Required**: Either implement comprehensive validation or explicitly document known limitations in INPUT-SPECIFICATION.md.

**Risk Level**: MEDIUM (data validation)

## Acceptance Criteria

- [ ] Decision made: implement validation OR document limitations
- [ ] If implementing: Validation logic added with comprehensive test coverage
- [ ] If documenting: INPUT-SPECIFICATION.md updated with clear limitations section
- [ ] TODO comments removed or replaced with issue references
- [ ] All existing tests pass
- [ ] New tests added if validation is implemented
- [ ] CHANGELOG.md updated if user-facing behavior changes

## Tasks

### Decision Making
- [ ] Review current validation coverage in `input_validation.rs`
- [ ] Identify what "detailed validation for uncertainty_specifications" means:
  - Schema validation already covered?
  - Cross-field constraints?
  - Value range checks?
  - Statistical consistency checks?
- [ ] Identify what "cross-file consistency checks" means:
  - References between system.json, graph.json, recourse.json?
  - ID consistency across files?
  - Dimensional compatibility?
- [ ] Determine if missing validation has caused issues in practice
- [ ] Decide: implement now, create ticket for later, or document as known limitation

### Option A: Implement Validation
- [ ] **Uncertainty Specifications Validation**:
  - [ ] Add validation for uncertainty spec structure
  - [ ] Add validation for parameter ranges (e.g., probabilities sum to 1)
  - [ ] Add validation for compatibility with system definition
  - [ ] Add validation for noise model parameters
- [ ] **Cross-File Consistency Checks**:
  - [ ] Add validation that system IDs referenced in graph.json exist in system.json
  - [ ] Add validation that resource IDs referenced in recourse.json exist
  - [ ] Add validation for dimensional consistency across files
  - [ ] Add validation for initial condition compatibility

### Option B: Document Limitations
- [ ] Add "Known Limitations" section to `docs/reference/INPUT-SPECIFICATION.md`
- [ ] Document what is NOT validated automatically
- [ ] Provide examples of invalid configurations that won't be caught
- [ ] Document recommended manual checks
- [ ] Add references to validation functions that do exist

### Testing (if implementing validation)
- [ ] Unit test: Valid uncertainty specifications pass validation
- [ ] Unit test: Invalid uncertainty specifications are rejected with clear error messages
- [ ] Unit test: Cross-file inconsistencies are detected
- [ ] Integration test: Complete valid input set passes all validation
- [ ] Integration test: Subtly invalid input set is caught by new validation
- [ ] Test error messages are clear and actionable

### Documentation
- [ ] Update `input_validation.rs` doc comments to reflect current validation coverage
- [ ] Update INPUT-SPECIFICATION.md if validation behavior changes
- [ ] Add CHANGELOG.md entry if user-facing: "Added validation for uncertainty specifications and cross-file consistency"
- [ ] Remove or update TODO comments

## Technical Notes

**Location**: `src/input_validation.rs`

**Current Validation Coverage**:
- Check existing `validate_*` functions to understand current coverage
- Review `src/input.rs` for what structures are being validated
- Check `tests/test_input_validation.rs` for what's already tested

**Validation Strategy Considerations**:
1. **Fail Fast**: Validate at input parsing time before algorithm starts
2. **Clear Errors**: Provide actionable error messages with line numbers/paths
3. **Performance**: Validation should be fast (runs once per invocation)
4. **Maintainability**: Validation logic should be easy to extend

**Example Invalid Configurations to Consider**:
```json
// Uncertainty spec with probabilities that don't sum to 1
// Graph references node ID that doesn't exist in system
// Recourse references reservoir not defined in system
// Initial condition incompatible with system dimensions
```

**Related Files**:
- `src/input.rs` - Input structures
- `src/error.rs` - Error types for validation failures
- `tests/test_input_validation.rs` - Existing validation tests
- `tests/test_input_error_paths.rs` - Error path tests
- `docs/reference/INPUT-SPECIFICATION.md` - Input format documentation

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-002 (both involve validation and correctness)

## Estimated Effort

**1.5 story points** (6-8 hours, confidence: medium)

**If implementing**: 6-8 hours (validation logic + comprehensive tests)  
**If documenting**: 2-3 hours (thorough documentation of limitations)

Recommend starting with investigation (1-2 hours) before committing to full implementation.
