# Epic 6: Integration & Documentation

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 1 week (1 sprint)
> **Status**: ⬜ Not Started

---

## Summary

This epic finalizes the performance evaluation infrastructure with comprehensive documentation, establishes baseline profiles for the current POWE.RS version, cleans up and removes old scripts, and validates the entire toolchain end-to-end.

---

## Scope

### Included

1. **Documentation**
   - Quick Start Guide
   - Tools Reference (all collectors, analyzers, reporters)
   - Analysis Guide (how to interpret results)
   - Troubleshooting Guide
   - API Reference (for extending)

2. **Baseline Establishment**
   - Run full profiling suite on current version
   - Commit baselines to `profiling_results/baselines/`
   - Document baseline system and conditions

3. **Cleanup**
   - Remove old scripts from `scripts/` directory
   - Update references in existing documentation
   - Update CHANGELOG.md

4. **End-to-End Validation**
   - Test all commands work together
   - Test comparison between baseline and HEAD
   - Validate JSON schema compliance
   - Test on both WSL2 and bare-metal (if available)

5. **Integration with Clean Code Refactoring**
   - Update `plans/clean-code-refactoring/README.md`
   - Add Epic 6 reference to master plan

### Excluded

- CI/CD integration (future work)
- Additional collectors

---

## Dependencies

- **Requires**: Epics 1-5 complete
- **Enables**: Test Modernization (Epic 7), Final Validation (Epic 8)

---

## Acceptance Criteria

- [ ] Complete documentation in `docs/profiling/`
- [ ] Baseline v0.2.0 profiles committed
- [ ] Old `scripts/monitor_rss.py`, `plot_rss.py`, etc. removed
- [ ] All `powers-profile` commands documented with examples
- [ ] End-to-end test passes
- [ ] Clean Code Refactoring plan updated

---

## Documentation Structure

```
docs/profiling/
├── QUICK_START.md          # 5-minute getting started
├── INSTALLATION.md         # Prerequisites, setup
├── TOOLS_REFERENCE.md      # All collectors and options
├── ANALYSIS_GUIDE.md       # Interpreting results
├── COMPARISON_GUIDE.md     # Version comparison
├── TROUBLESHOOTING.md      # Common issues
└── EXTENDING.md            # Adding new collectors
```

### QUICK_START.md Example

```markdown
# Quick Start

## Prerequisites

- Python 3.10+
- valgrind
- perf (Linux tools)

## Installation

```bash
cd profiling
pip install -e .
```

## First Run

```bash
# Build POWE.RS
cargo build --release

# Run full profiling suite
powers-profile run --suite full

# View results
powers-profile summary

# Generate dashboard
powers-profile dashboard --output report.html
open report.html
```
```

---

## Sprints

### [Sprint 1: Integration](./sprint-01/00-sprint-overview.md)

| ID | Title | Points | Status |
|----|-------|--------|--------|
| T-041 | Write Quick Start guide | 2 | ⬜ |
| T-042 | Write Tools Reference | 3 | ⬜ |
| T-043 | Write Analysis Guide | 3 | ⬜ |
| T-044 | Establish v0.2.0 baseline | 3 | ⬜ |
| T-045 | Remove old scripts | 2 | ⬜ |
| T-046 | Update clean-code-refactoring plan | 2 | ⬜ |
| T-047 | End-to-end validation | 3 | ⬜ |

**Sprint Points**: 18

---

## Estimated Effort

- **Duration**: 1 sprint (1 week)
- **Story Points**: 18
- **Risk Level**: Low (documentation and cleanup)

---

## Definition of Done

- [ ] All documentation written
- [ ] Baseline profiles committed
- [ ] Old scripts removed
- [ ] E2E validation passes
- [ ] Clean Code plan updated
- [ ] Ready for Epic 7 (Test Modernization)
