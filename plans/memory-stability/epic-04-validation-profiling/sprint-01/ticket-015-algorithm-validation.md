# [TICKET-015] Algorithm correctness validation

> **Epic**: [Epic 4: Validation & Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-014](./ticket-014-memory-profiling.md)
> **Blocks**: [TICKET-016](./ticket-016-document-remaining-sources.md)

## Context

### Background

Memory optimizations must not change algorithmic behavior. The lower bounds produced by SDDP training must be identical to the baseline implementation.

### Relation to Epic

Ensures the implementation is production-ready without algorithmic regressions.

## Specification

### Validation Approach

1. **Determinism within implementation**:
   - Run example 05 twice with same seed
   - Lower bounds must be bit-for-bit identical

2. **Comparison to baseline**:
   - Compare lower bound sequences to pre-implementation run
   - Must be identical (or document acceptable differences)

3. **Cross-example validation**:
   - Run examples 01, 05, 07
   - Compare final lower bounds and convergence behavior

### Examples to Validate

| Example | Description | Critical Aspects |
|---------|-------------|------------------|
| 01-deterministic | Simple deterministic | Basic correctness |
| 05-large-scale-brazilian | Full scale with cut selection | Memory + correctness |
| 07-multistage | Multi-stage with uncertainty | Complex dynamics |

### Validation Commands

```bash
# Run and capture lower bounds
./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep -E "Iteration|lower" > post_impl_05.txt

# Compare with baseline (assuming baseline captured)
diff baseline_05.txt post_impl_05.txt
```

### Expected Results

- **Example 01**: Identical final cost
- **Example 05**: Identical lower bound sequence (8 values)
- **Example 07**: Identical lower bound sequence

## Acceptance Criteria

- [ ] Example 01 produces identical results
- [ ] Example 05 lower bounds match baseline
- [ ] Example 07 lower bounds match baseline
- [ ] Determinism verified (multiple runs identical)
- [ ] No test failures

## Implementation Guide

### Suggested Approach

1. Ensure baseline results are captured (before changes)
2. Run each example with post-implementation code
3. Compare outputs systematically
4. Run `cargo test` for automated tests
5. Document any differences and root cause

### Comparison Script

```bash
#!/bin/bash
# validate_correctness.sh

for example in 01-deterministic 05-large-scale-brazilian 07-multistage; do
    echo "=== Validating $example ==="
    
    # Run and capture
    ./target/release/powers run examples/$example 2>&1 | \
        grep -E "lower|final" > results_$example.txt
    
    # Compare (assumes baseline exists)
    if diff -q baseline_$example.txt results_$example.txt > /dev/null; then
        echo "✓ $example: PASS"
    else
        echo "✗ $example: DIFFERENCES FOUND"
        diff baseline_$example.txt results_$example.txt
    fi
done
```

## Testing Requirements

### Validation Tests

- [ ] Run all existing unit tests
- [ ] Run all integration tests
- [ ] Compare example outputs to baseline

### Determinism Tests

- [ ] Run example 05 twice, compare outputs
- [ ] Run with different thread counts, compare outputs

## Documentation Requirements

- [ ] Document validation results
- [ ] Note any acceptable differences (if any)
- [ ] Update test documentation if needed

## Deliverables

1. Validation report with:
   - Test results (pass/fail)
   - Lower bound comparisons
   - Any differences and explanations

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Testing and comparison task
