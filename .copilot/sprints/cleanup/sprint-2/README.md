# Sprint 2: Comment Cleanup - README

## Overview

Sprint 2 focuses on improving code readability by cleaning up comment bloat while preserving valuable information. This sprint addresses Priority 3 items from the CLEANING_PLAN: reducing excessive inline commentary and improving documentation organization.

**Sprint Goal**: Improve code readability by consolidating, condensing, and reorganizing comments and documentation.

**Total Estimated Effort**: 5 story points (approximately 1 day of work)

## Tickets

| Ticket | Title | Story Points | Confidence |
|--------|-------|--------------|------------|
| CLEANUP-011 | Consolidate Mathematical Derivation Comments in Tests | 1.5 | Medium |
| CLEANUP-012 | Extract lognormal3.rs Tutorial to Documentation | 1 | High |
| CLEANUP-013 | Extract solver.rs Comparison to Architecture Docs | 0.5 | High |
| CLEANUP-014 | Condense SDDP Module-Level Documentation | 0.25 | High |

**Note**: CLEANUP-008 and CLEANUP-009 from Sprint 1 also relate to comment cleanup but were included in Sprint 1 due to their tight coupling with TODO/dead code resolution.

## Sprint Objectives

### Primary Goals
1. ✅ Consolidate mathematical derivations in test comments
2. ✅ Extract verbose module documentation to proper docs/
3. ✅ Improve documentation organization (module docs vs external docs)
4. ✅ Maintain zero information loss

### Success Criteria
- [ ] Test derivation comments organized and readable
- [ ] Tutorial-level docs moved to docs/reference/
- [ ] Architecture docs created in docs/architecture/
- [ ] Module docs concise (10-15 lines for complex modules)
- [ ] All documentation builds correctly
- [ ] No test failures
- [ ] CHANGELOG.md updated

## Pre-Sprint Setup

Before starting Sprint 2:
- [ ] Verify Sprint 1 is complete and merged
- [ ] Create Sprint 2 feature branch from main
- [ ] Review Sprint 1 lessons learned
- [ ] Ensure docs/ directory structure is ready

## Work Sequence

### Recommended Order

**Week 1**:
1. **CLEANUP-014** - Quick win (1-2 hours)
2. **CLEANUP-013** - Create architecture doc structure (2-3 hours)
3. **CLEANUP-012** - Extract lognormal3 docs (4-5 hours)

**Week 2**:
4. **CLEANUP-011** - Consolidate test comments (6-8 hours)
5. **Validation** - Full documentation build and review (2 hours)

### Parallelization Opportunities

These tickets can be worked on in parallel:
- CLEANUP-012 and CLEANUP-013 (different files, different doc sections)
- CLEANUP-011 and CLEANUP-014 (different files, no conflicts)

## Sprint Validation

### Pre-Commit Checks (Per Ticket)
```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps
```

### Documentation Validation
```bash
# Build and review documentation
cargo doc --open

# Build with private items (to see test docs)
cargo doc --document-private-items

# If using mdBook for docs/
cd docs && mdbook build && mdbook serve

# Check internal links
# (manual review or use tool like markdown-link-check)
```

### Sprint Completion Checklist
- [ ] All 4 tickets completed
- [ ] All tests passing
- [ ] Documentation builds without errors
- [ ] Cross-references verified
- [ ] CHANGELOG.md updated
- [ ] No information loss from original comments
- [ ] Code readability improved

## Documentation Structure

After Sprint 2, documentation should be organized as:

```
docs/
  README.md                    # Overview and navigation
  algorithm/                   # Algorithm theory and details
    SDDP.md                   # (exists or to be created)
  architecture/                # NEW: Design decisions
    README.md                 # Architecture docs index
    SOLVER.md                 # NEW: Solver integration rationale
  reference/                   # API reference and specifications
    INPUT-SPECIFICATION.md    # (existing)
    distributions.md          # NEW: LN3 and other distributions
  guides/                      # User guides
    (existing guides)

src/
  lognormal3.rs               # Concise module docs (10-15 lines)
  solver.rs                   # Concise module docs (3-5 lines)
  sddp/mod.rs                 # Concise module docs (~10 lines)
  
tests/
  *.rs                        # Clean test docs with derivations in doc comments
```

## Key Principles

### Comment Organization
1. **Module docs (//!)**: Brief overview, features, links to detailed docs
2. **docs/reference/**: Detailed tutorials, mathematical background
3. **docs/architecture/**: Design decisions, trade-offs, rationale
4. **Item docs (///)**: API documentation, parameters, examples
5. **Inline comments (//)**: Brief context, non-obvious logic, why not what

### Documentation Targets
- Module docs: 10-15 lines for complex modules, 3-5 for simple
- Mathematical derivations: In doc comments, not inline
- Architecture decisions: In docs/architecture/, not inline
- Tutorial content: In docs/reference/, not in source

## Risk Management

### Low-Risk Activities
- All Sprint 2 tickets are documentation-only changes
- No functional code changes
- Tests verify no behavioral changes
- Easy to roll back if needed

### Potential Issues
1. **Link rot**: Cross-references break if files are moved
   - Mitigation: Test all links after changes
2. **Information loss**: Important context removed
   - Mitigation: Review all deleted content before removing
3. **Documentation build failures**: Markdown syntax errors
   - Mitigation: Build docs after each change

## Success Metrics

### Quantitative
- Module doc line count reduced: Before ___ → After ___
- Documentation pages created: ___ (architecture, reference)
- Cross-references added: ___
- Test comment organization improved: ___ functions updated

### Qualitative
- [ ] Module docs easier to scan
- [ ] Test files more readable
- [ ] Architecture decisions documented
- [ ] Detailed tutorials easily accessible
- [ ] New developers can find information faster

## Handoff to Sprint 3

After Sprint 2:
- [ ] All documentation organizational structure complete
- [ ] Ready for Sprint 3 (Phase 4 deferred to future work)
- [ ] Consider if additional architecture docs would be valuable
- [ ] Review if any other modules need similar documentation extraction

## Notes

Sprint 2 is lightweight and low-risk. Focus on preserving information while improving organization. This sprint creates the documentation structure that will serve the project long-term.
