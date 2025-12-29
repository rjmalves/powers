# Epic 1: Foundation

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

Before starting any work in this epic, read the [Critical Principles](../00-master-plan.md#️-critical-principles-correctness-first-then-performance) section. Algorithm correctness is non-negotiable. If you encounter unexpected results, **STOP and ask for clarification**.

---

## Summary

This epic establishes the foundational infrastructure for the refactoring effort. It creates the module structure, sets up golden output tests for correctness validation, establishes baseline benchmarks (both time and memory), and creates the new timing infrastructure. **No algorithm logic is modified in this epic**—we are only creating the scaffolding for future changes.

---

## Scope

### Included

1. **Golden Output Test Infrastructure**
   - Capture deterministic outputs from all examples with fixed seeds
   - **Filter out timing information** from comparisons (timing varies between runs)
   - Create automated comparison scripts using relative paths
   - Document the baseline for regression detection

2. **Baseline Benchmark Capture**
   - Run and record current performance benchmarks
   - **Document memory usage patterns** (Peak RSS, allocation counts)
   - **Create memory analysis benchmarks** for example 05
   - Create benchmark comparison tooling

3. **New Timing Module**
   - Create `src/timing/mod.rs` with `TimingGuard` and `TimingCollector`
   - Implement feature-gated timing (compile-time elimination)
   - Create timing aggregation utilities

4. **Module Structure Skeleton**
   - Create new module directories (empty or with re-exports)
   - Set up `mod.rs` files with proper visibility
   - No logic migration yet—just structure

5. **Error Type Foundation**
   - Review existing error types
   - Plan unified error hierarchy (no implementation yet)

### Excluded

- Any modification to algorithm logic
- Migration of existing code to new modules
- Performance optimization work
- Test modernization (beyond golden tests)

---

## Dependencies

- **Requires**: None (this is the first epic)
- **Enables**: 
  - Epic 2: Core Extraction (uses new module structure)
  - Epic 3: Algorithm Separation (uses timing infrastructure)
  - Epic 5: Memory Optimization (uses memory baseline)
  - Epic 6: Test Modernization (uses golden test baseline)

---

## Acceptance Criteria

- [ ] Golden output tests exist for all examples in `examples/0*`
- [ ] Golden output tests pass with current codebase (baseline established)
- [ ] **Timing information filtered** from golden test comparisons
- [ ] Benchmark baseline documented with specific numbers (time AND memory)
- [ ] **Memory analysis benchmarks created** for example 05
- [ ] New `src/timing/` module exists with `TimingGuard` implementation
- [ ] Timing module compiles with and without `timing` feature flag
- [ ] New module directories exist: `src/algorithm/`, `src/model/`, `src/memory/`
- [ ] All existing tests still pass
- [ ] No performance regression (verified by benchmark)

### Correctness Verification

- [ ] `cargo test` passes with no changes to test results
- [ ] Golden outputs match exactly (bit-for-bit with same seed, timing filtered)
- [ ] Benchmark shows no regression from baseline

---

## Technical Approach

### Golden Output Tests

```bash
# Script: scripts/golden-tests.sh
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GOLDEN_DIR="$REPO_ROOT/tests/golden"
BINARY="$REPO_ROOT/target/release/powers"

# Filter timing info from output
filter_timing() {
  sed -E 's/\| [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} //g' | \
  grep -Ev "(Training time:|Simulation time:|Total running time:)"
}

# Generate or verify golden outputs
for example in "$REPO_ROOT"/examples/0*; do
  name=$(basename "$example")
  golden_file="$GOLDEN_DIR/$name.txt"
  
  if [ "$1" = "generate" ]; then
    "$BINARY" run "$example" 2>&1 | filter_timing > "$golden_file"
    echo "Generated: $golden_file"
  else
    # Verify mode
    "$BINARY" run "$example" 2>&1 | filter_timing | diff -u "$golden_file" - || {
      echo "MISMATCH: $name"
      exit 1
    }
    echo "PASS: $name"
  fi
done
```

### Example Execution Times

⚠️ **Important**: Plan timeouts appropriately
- Examples 01-04, 06-07: < 30 seconds each
- **Example 05 (large-scale-brazilian): Up to 2 minutes**

### Timing Module Structure

```
src/timing/
├── mod.rs          # Public exports, feature gates
├── guard.rs        # TimingGuard RAII implementation
├── collector.rs    # TimingCollector trait and implementations
├── metrics.rs      # TimingMetric enum, aggregation
└── tests.rs        # Unit tests for timing infrastructure
```

### Module Skeleton

```
src/
├── algorithm/      # Empty, to be populated in Epic 3
│   └── mod.rs      # pub mod forward_pass; pub mod backward_pass; (commented)
├── model/          # Empty, to be populated in Epic 2
│   └── mod.rs      # pub mod builder; pub mod constraints; (commented)
└── timing/         # Implemented in this epic
    └── mod.rs
```

---

## Sprints

### [Sprint 1: Infrastructure Setup](./sprint-01/00-sprint-overview.md)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| [T-001](./sprint-01/ticket-001-golden-test-infrastructure.md) | Create golden test infrastructure | 3 | ⬜ |
| [T-002](./sprint-01/ticket-002-baseline-benchmarks.md) | Capture baseline benchmarks | 3 | ⬜ |
| [T-003](./sprint-01/ticket-003-timing-module-guard.md) | Implement TimingGuard | 3 | ⬜ |
| [T-004](./sprint-01/ticket-004-timing-module-collector.md) | Implement TimingCollector trait | 3 | ⬜ |
| [T-005](./sprint-01/ticket-005-module-skeleton.md) | Create module directory skeleton | 1 | ⬜ |

**Total Points**: 13

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 13
- **Parallelizable**: Tickets T-001, T-002 can run in parallel; T-003, T-004 can run after T-001/T-002

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Golden tests not deterministic | Medium | High | Filter timing info, verify 3 runs with same seed |
| Benchmark variance too high | Low | Medium | Multiple runs, statistical analysis |
| Timing feature flags complex | Low | Low | Follow existing feature flag patterns |
| Example 05 timeout in CI | Medium | Medium | Set 5+ minute timeout for safety margin |

---

## Definition of Done

- [ ] All tickets in Sprint 1 complete
- [ ] All acceptance criteria met
- [ ] Golden test infrastructure documented in README
- [ ] Benchmark baseline documented (time AND memory)
- [ ] Memory analysis benchmarks created
- [ ] Timing module has >90% test coverage
- [ ] Code reviewed and merged to main branch
- [ ] No algorithm logic has been modified (verified by golden tests)
