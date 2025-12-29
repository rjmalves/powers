# Golden Tests

Golden tests capture known-good outputs from the SDDP algorithm to ensure refactoring doesn't introduce regressions.

## Purpose

These tests verify **bit-for-bit correctness** of algorithm outputs. Any change to numerical results will cause test failures, protecting against accidental algorithm modifications during refactoring.

## Running Tests

### Verify current outputs match golden files (default)

```bash
./scripts/golden-tests.sh verify
```

### Regenerate golden files (only when intentional changes are made)

```bash
./scripts/golden-tests.sh generate
```

⚠️ **Only regenerate if you intentionally changed algorithm behavior** and have verified the new outputs are correct.

## What's Filtered

Timing information is filtered out of comparisons since it varies between runs:

- Timing columns in iteration tables (`| 00:00:00.XXX` patterns)
- Standalone timing lines (`Training time:`, `Simulation time:`, `Total running time:`)

## Expected Execution Times

| Example | Approximate Time |
|---------|------------------|
| 01-deterministic | < 5 seconds |
| 02-stochastic | < 10 seconds |
| 03-multistage | < 10 seconds |
| 04-cascade | < 5 seconds |
| 05-large-scale-brazilian | **~2 minutes** |
| 06-par-model | < 10 seconds |
| 07-par-model-with-inflow-state | < 10 seconds |

Full verification takes approximately 3-4 minutes.

## If Tests Fail

1. **STOP** - Do not proceed with other changes
2. Review the diff output to understand what changed
3. If the change was **unintentional**: revert your changes and investigate
4. If the change was **intentional** and verified correct: regenerate golden files

## Determinism

All examples use a fixed seed (`42` in `config.json`) to ensure deterministic outputs. The golden tests have been verified to produce identical results across 3 consecutive runs.

## Files

- `tests/golden/*.txt` - Golden output files (one per example)
- `scripts/golden-tests.sh` - Verification script
