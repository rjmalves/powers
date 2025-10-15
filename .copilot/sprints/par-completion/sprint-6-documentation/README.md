# Sprint 6: Documentation & Polish (Week 10)

**Goal**: Document the PAR implementation, create examples, and prepare for release.

**Duration**: 1 week  
**Total Story Points**: 11  
**Risk Level**: 🟢 LOW

---

## Overview

With the PAR implementation working (validated in Sprint 5), this sprint polishes everything for production use:

- Documentation for users and developers
- Example system demonstrating PAR functionality
- JSON schema updates
- Final cleanup and code review

**Key Deliverable**: PAR feature ready for production release.

---

## Sprint Tickets

| ID | Title | Points | Status | Priority |
|----|-------|--------|--------|----------|
| PAR-V2-024 | Create PAR example system | 2 | 🔵 Not Started | Medium |
| PAR-V2-025 | Update input specification documentation | 2 | 🔵 Not Started | High |
| PAR-V2-026 | Update algorithm documentation | 2 | 🔵 Not Started | High |
| PAR-V2-027 | Update JSON schemas | 2 | 🔵 Not Started | High |
| PAR-V2-028 | Final integration and cleanup | 3 | 🔵 Not Started | Medium |

---

## Execution Order

**Parallel Work**:
- PAR-V2-024 (Example)
- PAR-V2-025 (Input Docs)
- PAR-V2-026 (Algorithm Docs)
- PAR-V2-027 (Schemas)

**Final**:
- PAR-V2-028 (Cleanup) - after all others complete

---

## Documentation Requirements

### Input Specification (PAR-V2-025)

Update `docs/reference/INPUT-SPECIFICATION.md`:

**Add Section**: "PAR Stochastic Process"

**Document**:
- `stochastic_process_type: "PAR"`
- `ar_order`: Integer (1-5 typical)
- `ar_coefficients`: Array of [φ_1, ..., φ_p, μ, σ] per season
- Initial lag values in `initial_condition`

**Provide Examples**:
```json
{
  "inflow_process": {
    "type": "PAR",
    "order": 2,
    "coefficients": {
      "wet": [0.6, 0.3, 100.0, 10.0],
      "dry": [0.7, 0.2, 50.0, 5.0]
    }
  }
}
```

### Algorithm Documentation (PAR-V2-026)

Create `docs/algorithm/PAR-MODELS.md`:

**Sections**:
1. **Introduction**: What are PAR models?
2. **Mathematical Formulation**: State-space approach
3. **Implementation**: AR constraints in LP
4. **Cut Generation**: Extended Benders cuts
5. **Comparison**: PAR vs Naive inflow models

**Include**:
- Mathematical equations
- Diagrams (state-space illustration)
- References to theory

### JSON Schemas (PAR-V2-027)

Update `schemas/system.schema.json`:

**Add PAR Process Type**:
```json
{
  "stochastic_process": {
    "oneOf": [
      {"$ref": "#/definitions/NaiveProcess"},
      {"$ref": "#/definitions/PARProcess"}
    ]
  },
  "PARProcess": {
    "type": "object",
    "properties": {
      "type": {"const": "PAR"},
      "order": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10
      },
      "coefficients": {
        "type": "object",
        "patternProperties": {
          ".*": {
            "type": "array",
            "items": {"type": "number"}
          }
        }
      }
    },
    "required": ["type", "order", "coefficients"]
  }
}
```

**Validate**: Test schema against example systems

---

## Example System (PAR-V2-024)

Create `examples/06-par-model/`:

**Files**:
- `system.json` - 2-stage, 2 hydros (1 PAR, 1 naive)
- `graph.json` - Simple 2-stage graph
- `config.json` - Standard SDDP config
- `README.md` - Explanation of example

**System Design**:
- Keep it simple (easy to understand)
- Show PAR(2) with seasonal parameters
- Demonstrate mixed PAR/naive hydros
- Include initial lag values
- Should train to convergence in < 1 minute

**README Content**:
- What this example demonstrates
- PAR parameter interpretation
- Expected output
- How to run: `cargo run --release -- examples/06-par-model`

---

## Final Cleanup (PAR-V2-028)

### Code Cleanup

- [ ] Remove debug prints
- [ ] Remove commented-out code
- [ ] Remove unused imports
- [ ] Fix clippy warnings
- [ ] Run cargo fmt
- [ ] Check for TODO/FIXME comments

### Testing

- [ ] Run full test suite:
  ```bash
  cargo test --workspace
  ```
- [ ] Run examples:
  ```bash
  ./scripts/run_examples.sh
  ```
- [ ] Run pre-checks:
  ```bash
  cargo fmt -- --check
  cargo clippy --all-targets --all-features -- -D warnings
  ```
- [ ] Build release:
  ```bash
  cargo build --workspace --release
  ```

### Documentation

- [ ] Update `CHANGELOG.md`:
  ```markdown
  ## [Unreleased]
  ### Added
  - PAR (Periodic AutoRegressive) stochastic process support
  - State-space formulation for AR dynamics in SDDP
  - Extended Benders cuts with lag state coefficients
  - Example system demonstrating PAR models
  ```

- [ ] Update README.md (if needed)
- [ ] Generate docs:
  ```bash
  cargo doc --no-deps
  ```

### Final Review

- [ ] Self code review (read through all changes)
- [ ] Peer code review (architect sign-off)
- [ ] Performance validation (compare benchmarks)
- [ ] Documentation review (check for completeness)

---

## Success Criteria

- [ ] Example system runs successfully
- [ ] Input specification documented
- [ ] Algorithm documentation complete
- [ ] JSON schemas updated and validated
- [ ] All tests pass
- [ ] Clippy clean
- [ ] Code formatted
- [ ] CHANGELOG updated
- [ ] Documentation builds without errors
- [ ] Code reviewed and approved
- [ ] Ready to merge to main

---

## Dependencies

**Blocked By**: Sprint 5 (Runtime) - must have working implementation

**Blocks**: None (final sprint)

---

## Files to Edit

- `docs/reference/INPUT-SPECIFICATION.md`
- `docs/algorithm/PAR-MODELS.md` (new)
- `schemas/system.schema.json`
- `examples/06-par-model/*` (new)
- `CHANGELOG.md`
- Various code files (cleanup)

---

## Sprint Goal

**By end of Sprint 6**: PAR feature is production-ready and fully documented.

**Deliverable**: Complete, tested, documented PAR implementation ready for merge to main branch and inclusion in next release.

---

## Post-Sprint Activities

After Sprint 6 completes:

1. **Merge to Main**:
   - Final review
   - Merge PR
   - Tag release (if appropriate)

2. **Announcement**:
   - Update documentation site
   - Notify users of new feature
   - Provide migration guide if needed

3. **Monitor**:
   - Watch for bug reports
   - Collect user feedback
   - Plan any necessary follow-up improvements

---

## Notes

This sprint should be straightforward if Sprint 5 validated correctness. Focus on clarity and completeness in documentation.

**Remember**: Good documentation is as important as good code. Users need to understand how to use PAR models effectively.

**Congratulations!** Completing Sprint 6 marks the end of a major feature implementation. The state-space formulation for PAR models is now part of the powe.rs toolkit.
