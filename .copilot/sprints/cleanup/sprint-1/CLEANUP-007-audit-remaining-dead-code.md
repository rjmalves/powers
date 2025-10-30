# CLEANUP-007: Audit Remaining #[allow(dead_code)] Attributes

## Context

After addressing the solver and SDDP module dead code, there are 10 remaining `#[allow(dead_code)]` attributes scattered across the codebase that need individual review:

**Locations**:
1. `src/input.rs:101` - `validate_entity_count` function
2. `src/state.rs:857` - Unknown field
3. `src/system.rs:143` - Unknown field
4. `src/unified_inflow_model.rs:127` - Documented as used in TICKET-002/003
5. `src/subproblem.rs:1017` - Unknown field
6-10. Additional instances found in CLEANUP_PLAN analysis

**Risk Level**: LOW (individual assessment needed per item)

## Acceptance Criteria

- [ ] All remaining `#[allow(dead_code)]` attributes identified and documented
- [ ] Each attribute reviewed individually with context
- [ ] Decision made for each: REMOVE code, KEEP with documentation, or MOVE to tests
- [ ] All removals verified safe with tests
- [ ] All kept attributes have explanatory comments
- [ ] No new compiler warnings introduced
- [ ] Documentation updated for any API changes

## Tasks

### Discovery
- [ ] Generate complete list of `#[allow(dead_code)]` attributes:
  ```bash
  rg "#\[allow\(dead_code\)\]" src/ --line-number
  ```
- [ ] Cross-reference with cleanup plan to identify all instances
- [ ] Create inventory spreadsheet or checklist with:
  - File path and line number
  - Item name (function, struct, field, enum variant)
  - Context (what is it, why was it added)
  - Current usage (used in tests, truly unused, future feature)

### Investigation (For Each Attribute)
- [ ] Read surrounding code to understand purpose
- [ ] Search for references in entire codebase:
  ```bash
  rg "<item_name>" src/ tests/ benches/
  ```
- [ ] Check git history to understand when and why added:
  ```bash
  git log -p --all -S "<item_name>" -- <file_path>
  ```
- [ ] Determine if used only in tests
- [ ] Check if part of public API (would be breaking change to remove)
- [ ] Verify if documented as reserved for future use

### Specific Item Reviews

#### 1. `validate_entity_count` (src/input.rs:101)
- [ ] Search for calls to this function
- [ ] Check if validation is performed elsewhere
- [ ] Decision: Remove OR document why unused but kept

#### 2. Unknown field (src/state.rs:857)
- [ ] Identify the field name at line 857
- [ ] Check if field is read anywhere in codebase
- [ ] Check if field is serialized/deserialized (JSON)
- [ ] Decision: Remove OR document as reserved for future feature

#### 3. Unknown field (src/system.rs:143)
- [ ] Identify the field name at line 143
- [ ] Check if part of public API for system definition
- [ ] Check JSON schema to see if field is documented
- [ ] Decision: Remove OR keep with documentation

#### 4. `unified_inflow_model.rs:127` (documented as TICKET-002/003)
- [ ] Verify TICKET-002 and TICKET-003 exist and are active
- [ ] Check if this is being actively worked on
- [ ] If active: Keep with clear reference to ticket
- [ ] If stale: Remove and recreate when actually needed

#### 5. `subproblem.rs:1017` field
- [ ] Identify field and its purpose
- [ ] Check if part of solver interface
- [ ] Check if used in benchmarks or profiling
- [ ] Decision: Remove, move to tests, or document

### Resolution Actions

#### For Items to REMOVE:
- [ ] Delete the code
- [ ] Remove `#[allow(dead_code)]` attribute
- [ ] Run tests to verify no breakage
- [ ] Update CHANGELOG.md if user-facing

#### For Items to KEEP:
- [ ] Replace `#[allow(dead_code)]` with documented version:
  ```rust
  #[allow(dead_code)] // Reserved for future feature (see issue #XXX)
  // OR
  #[allow(dead_code)] // Used only in test infrastructure (see tests/helper.rs)
  // OR
  /// Reserved for future implementation of warm-start feature.
  #[allow(dead_code)]
  pub field_name: FieldType,
  ```
- [ ] Add doc comment explaining purpose if public API
- [ ] Reference issue tracker or FUTURE_WORK.md if planned feature

#### For Items to MOVE to Tests:
- [ ] Move code to appropriate test module
- [ ] Remove from main source
- [ ] Update test imports
- [ ] Verify tests still pass

### Testing
- [ ] Run full test suite after each removal: `cargo test --workspace`
- [ ] Run clippy to verify no new warnings: `cargo clippy --all-targets -- -D warnings`
- [ ] Run examples to verify no runtime issues: `scripts/run_examples.sh`
- [ ] Run benchmarks if any changes affect performance-critical code

### Documentation
- [ ] Update CHANGELOG.md with summary of removed dead code
- [ ] Update doc comments for any kept but unused items
- [ ] Document pattern in CONTRIBUTING.md for handling future dead code

## Technical Notes

**Investigation Tools**:
```bash
# Find all #[allow(dead_code)]
rg "#\[allow\(dead_code\)\]" src/ -A 2 -B 1

# Find references to specific item
rg "validate_entity_count" src/ tests/ --type rust

# Git history for item
git log -p --all -S "validate_entity_count"

# Check if item is in public API
rg "pub.*validate_entity_count" src/
```

**Decision Framework**:

| Condition | Action |
|-----------|--------|
| Truly unused, not in public API | **REMOVE** |
| Used only in tests | **MOVE** to test module |
| Part of public API, unused internally | **KEEP** with doc comment explaining external use |
| Reserved for near-term feature | **KEEP** with issue reference |
| Reserved for far-future feature | **REMOVE**, recreate when needed |
| Unclear purpose | **INVESTIGATE** deeper, ask maintainer |

**Checklist Template for Each Item**:
```markdown
## Item: <name> (<file>:<line>)
- [ ] Purpose understood
- [ ] References found (count: X)
- [ ] Git history reviewed
- [ ] Decision: REMOVE / KEEP / MOVE
- [ ] Rationale: ...
- [ ] Action taken: ...
- [ ] Tests passed: ✓
```

## Dependencies

- Blocked by: CLEANUP-005, CLEANUP-006 (handle specific dead code cases first)
- Blocks: None
- Related: All Priority 2 tickets

## Estimated Effort

**1.5 story points** (6-8 hours, confidence: medium)

Time breakdown:
- Discovery and inventory: 1 hour
- Individual item investigation: 4-5 hours (30-45 min per item)
- Resolution and testing: 1-2 hours

Effort varies based on how many items are found and complexity of decisions.
