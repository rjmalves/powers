# CLEANUP-002: Verify and Resolve Stochastic Process TODOs

## Context

The `stochastic_process.rs` file contains three TODO comments that are marked as **HIGH RISK** because they may affect core algorithm correctness. These need investigation to determine if:
1. The work is already complete (TODOs are stale)
2. The work is in progress (reference to active tickets)
3. The work is genuinely missing (requires implementation)

**Locations**:
- Line 383: `// TODO: Convert inflow to residual using inverse PAR transform`
- Line 418: `// TODO: Store realization for return`
- Line 469: `// TODO (PAR-011): Update factory or create builder pattern for PAR`

**Risk Level**: HIGH (algorithm correctness)

## Acceptance Criteria

- [ ] All three TODOs investigated and resolution path determined
- [ ] Line 383 (inverse transform): Verified correct or implementation ticket created
- [ ] Line 418 (realization storage): Verified complete or implementation ticket created
- [ ] Line 469 (PAR-011): Linked to existing ticket or new ticket created
- [ ] All TODOs either removed or replaced with issue references
- [ ] Algorithm correctness validated with existing tests
- [ ] No performance regressions

## Tasks

### Investigation
- [ ] **Line 383**: Examine surrounding code to determine if inverse PAR transform is implemented elsewhere or if current approach is correct
- [ ] **Line 383**: Check git history to see if this TODO is from recent development or legacy code
- [ ] **Line 418**: Trace code flow to determine if realization is already being stored/returned correctly
- [ ] **Line 469**: Search for TICKET-002 or PAR-011 references in codebase and git history
- [ ] Run PAR-related tests to verify current behavior is correct
- [ ] Check if `tests/test_par_*.rs` cover the scenarios mentioned in TODOs

### Resolution Path A: TODOs are Stale (Already Complete)
- [ ] Verify through code review and tests that functionality is present
- [ ] Remove TODO comments
- [ ] Add clarifying comments if needed about the implementation approach

### Resolution Path B: TODOs Reference Active Work
- [ ] Verify referenced tickets (PAR-011, TICKET-002, TICKET-003) exist in issue tracker
- [ ] Replace inline TODOs with issue references: `// See issue #XX for planned enhancement`
- [ ] Update issue tracker with current status if needed

### Resolution Path C: Work is Missing (Requires Implementation)
- [ ] Create detailed implementation tickets for each missing piece
- [ ] Determine priority and risk for each implementation
- [ ] Add to appropriate sprint or backlog
- [ ] Replace TODOs with issue references

### Testing
- [ ] Run `cargo test test_par_*` to verify PAR functionality
- [ ] Run `cargo test test_scenario*` to verify scenario generation
- [ ] Run full test suite: `cargo test --workspace`
- [ ] If changes made: Run benchmarks to ensure no performance regression

### Documentation
- [ ] Document findings in this ticket or linked investigation document
- [ ] Update CHANGELOG.md if any code changes are made
- [ ] Update inline documentation to clarify any ambiguous areas

## Technical Notes

**Location**: `src/stochastic_process.rs`

**Investigation Approach**:
1. Read code around each TODO carefully
2. Check git blame to understand when TODO was added
3. Search for related test coverage
4. Run debugger or add temporary print statements if needed
5. Consult with domain expert if algorithm correctness is unclear

**Key Questions**:
- Does the PAR model already perform inverse transforms correctly?
- Is the realization being returned through another mechanism?
- Is PAR-011 a known ticket that's documented elsewhere?

**Related Files to Check**:
- `src/par_generator.rs`
- `tests/test_par_*.rs`
- `src/unified_inflow_model.rs`
- Git history: `git log --all --grep=PAR-011`

## Dependencies

- Blocked by: None
- Blocks: CLEANUP-003, CLEANUP-004 (overall TODO resolution)
- Related: Any active PAR-model tickets

## Estimated Effort

**1 story point** (4-6 hours, confidence: medium)

Investigation-heavy ticket. Time depends on whether TODOs are stale or require new implementation tickets. Could spawn additional work if genuine issues are found.
