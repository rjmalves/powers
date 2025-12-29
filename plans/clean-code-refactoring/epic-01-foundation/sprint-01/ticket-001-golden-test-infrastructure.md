# [T-001] Create Golden Test Infrastructure

> **Epic**: [Epic 1: Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Infrastructure Setup](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-003](./ticket-003-timing-module-guard.md)

---

## ⚠️ CRITICAL: Correctness Validation

This ticket creates the **primary mechanism for detecting algorithmic regressions**. The golden outputs captured here will be used to verify that every subsequent change preserves bit-for-bit correctness.

**If outputs are non-deterministic with the same seed, STOP and investigate.** Do not proceed with the refactoring until determinism is verified.

---

## Files to Read Before Starting

- `examples/` - All example directories to understand test cases
- `examples/02-stochastic/config.json` - Example config showing seed placement
- `src/cli.rs` - Command-line interface for running examples
- `src/main.rs` - Entry point
- `Cargo.toml` - Current feature flags

---

## Context

### Background

The SDDP algorithm produces numerical outputs that must remain unchanged during refactoring. We need a mechanism to capture "known good" outputs and verify them after every change. This is our primary defense against introducing bugs during structural refactoring.

### Current State

- Examples exist in `examples/0*` directories
- Seed is configured via `config.json` (field: `"seed": 42`), not CLI
- No golden test infrastructure currently exists

---

## Specification

### Inputs

- Example directories: `examples/01-*`, `examples/02-*`, etc.
- Fixed seed: `42` (configured in each example's `config.json`)
- Release build of the application

### Outputs

1. **Golden output files**: `tests/golden/<example-name>.txt` for each example
2. **Verification script**: `scripts/golden-tests.sh`
3. **CI integration**: Instructions for running in CI

### Behavior

- When `scripts/golden-tests.sh generate` is run:
  - Execute each example
  - **Filter out timing information** from stdout (lines containing time durations like `00:00:00.XXX`, "Training time:", "Simulation time:", "Total running time:")
  - Capture filtered output to `tests/golden/<name>.txt`
  - Report which files were generated

- When `scripts/golden-tests.sh verify` is run:
  - Execute each example
  - **Filter out timing information** (same pattern)
  - Compare filtered output to golden file using `diff`
  - Exit with non-zero if ANY difference detected
  - Report PASS/FAIL for each example

- When outputs differ:
  - Show exact diff
  - Exit immediately with error code
  - Developer must investigate before proceeding

### Timing Filtering Pattern

The output contains timing information that varies between runs. Filter these patterns:
- Lines with duration columns like `| 00:00:00.XXX |` (iteration table timing columns)
- Lines starting with `Training time:`, `Simulation time:`, `Total running time:`

Example sed/grep filter:
```bash
# Remove timing columns from iteration table (columns 4-6: fwd, bwd, total)
# Remove standalone timing lines
sed -E 's/\| [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} //g' | \
grep -v "^.*time: [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}"
```

### Error Handling

- If example execution fails: Report error, continue to next example, exit non-zero
- If golden file missing in verify mode: Report error, fail
- If config.json doesn't have seed: Document as known limitation

### Example Execution Times

⚠️ **Important timing expectations**:
- Examples 01-04, 06-07: Complete in **< 30 seconds** each
- Example 05 (large-scale-brazilian): Takes **up to 2 minutes**

Plan timeouts accordingly in scripts and CI.

---

## Acceptance Criteria

- [ ] Golden output files exist for all examples in `examples/0*`
- [ ] `scripts/golden-tests.sh generate` creates all golden files
- [ ] `scripts/golden-tests.sh verify` passes with current codebase
- [ ] Running verify **3 times** produces identical results (determinism check)
- [ ] Script is documented in `tests/golden/README.md`
- [ ] Script handles errors gracefully (continues after single failure)
- [ ] Timing information is properly filtered out of comparisons

### Correctness Verification

- [ ] Run `verify` **3 times** consecutively—all must pass
- [ ] Golden files are committed to git
- [ ] No changes to any algorithm code

---

## Implementation Guide

### Suggested Approach

1. **Create directory structure**:
   ```bash
   mkdir -p tests/golden scripts
   ```

2. **Build release binary**:
   ```bash
   cargo build --release
   ```

3. **Create generation script** (`scripts/golden-tests.sh`):
   ```bash
   #!/bin/bash
   set -e
   
   # Use paths relative to repo root
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
   
   GOLDEN_DIR="$REPO_ROOT/tests/golden"
   BINARY="$REPO_ROOT/target/release/powers"
   
   # Function to filter out timing information
   filter_timing() {
     # Remove timing columns from iteration table (| 00:00:00.XXX patterns)
     # Remove standalone timing lines (Training time:, Simulation time:, Total running time:)
     sed -E 's/\| [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} //g' | \
     grep -Ev "(Training time:|Simulation time:|Total running time:) [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}"
   }
   
   # Ensure binary exists
   if [ ! -f "$BINARY" ]; then
     echo "ERROR: Binary not found. Run 'cargo build --release' first."
     exit 1
   fi
   
   mkdir -p "$GOLDEN_DIR"
   
   case "${1:-verify}" in
     generate)
       for example in "$REPO_ROOT"/examples/0*; do
         [ -d "$example" ] || continue
         name=$(basename "$example")
         echo "Generating: $name (this may take up to 2 minutes for example 05)"
         "$BINARY" run "$example" 2>&1 | filter_timing > "$GOLDEN_DIR/$name.txt" || {
           echo "WARNING: $name failed, capturing error output"
         }
       done
       echo "Golden files generated in $GOLDEN_DIR"
       ;;
     
     verify)
       FAILED=0
       for example in "$REPO_ROOT"/examples/0*; do
         [ -d "$example" ] || continue
         name=$(basename "$example")
         golden_file="$GOLDEN_DIR/$name.txt"
         
         if [ ! -f "$golden_file" ]; then
           echo "MISSING: $golden_file"
           FAILED=1
           continue
         fi
         
         echo -n "Verifying: $name ... "
         if "$BINARY" run "$example" 2>&1 | filter_timing | diff -u "$golden_file" - > /dev/null; then
           echo "PASS"
         else
           echo "FAIL"
           # Show the diff for debugging
           "$BINARY" run "$example" 2>&1 | filter_timing | diff -u "$golden_file" - || true
           FAILED=1
         fi
       done
       
       if [ $FAILED -eq 1 ]; then
         echo ""
         echo "⛔ GOLDEN TEST FAILURE - Algorithm outputs have changed!"
         echo "   If this is unexpected, STOP and investigate."
         echo "   If intentional, run: $0 generate"
         exit 1
       fi
       echo ""
       echo "✅ All golden tests passed"
       ;;
     
     *)
       echo "Usage: $0 [generate|verify]"
       exit 1
       ;;
   esac
   ```

4. **Make script executable**:
   ```bash
   chmod +x scripts/golden-tests.sh
   ```

5. **Verify all examples have seed configured**:
   ```bash
   for config in examples/0*/config.json; do
     echo "$config: $(grep -o '"seed":[^,}]*' "$config" || echo 'NO SEED')"
   done
   ```

6. **Generate golden files**:
   ```bash
   ./scripts/golden-tests.sh generate
   ```

7. **Verify determinism** (run 3 times):
   ```bash
   for i in {1..3}; do
     echo "=== Run $i ==="
     ./scripts/golden-tests.sh verify
   done
   ```

8. **Create documentation** (`tests/golden/README.md`)

9. **Commit golden files**:
   ```bash
   git add tests/golden/ scripts/golden-tests.sh
   ```

### Key Files to Create

- `scripts/golden-tests.sh` - Main verification script
- `tests/golden/README.md` - Documentation
- `tests/golden/*.txt` - Golden output files (one per example)

### Patterns to Follow

- Use paths relative to repository root (not absolute paths)
- Filter timing information before comparison
- Capture both stdout and stderr
- Use `diff -u` for readable output
- Exit with meaningful error codes

### Pitfalls to Avoid

- ⚠️ Don't use absolute paths—use `$REPO_ROOT` relative paths for portability
- ⚠️ Don't compare timing values—they vary between runs, filter them out
- ⚠️ Don't assume examples run quickly—example 05 takes up to 2 minutes
- ⚠️ Don't ignore stderr—it may contain important warnings
- ⚠️ Don't skip the determinism verification—run 3 times
- ⚠️ Don't modify any algorithm code in this ticket

---

## Testing Requirements

### Manual Tests

- [ ] Run `generate` and verify files are created
- [ ] Run `verify` and confirm all pass
- [ ] Modify a golden file slightly, verify `verify` fails
- [ ] Run `verify` 3 times consecutively, all pass
- [ ] Test on a fresh clone (files should be in git)
- [ ] Verify timing info is NOT in golden files

### Edge Cases

- [ ] Handle example that fails to run
- [ ] Handle missing golden file gracefully
- [ ] Handle example with no output
- [ ] Verify timing filtering works correctly

---

## Documentation Requirements

- [ ] Create `tests/golden/README.md` explaining:
  - Purpose of golden tests
  - How to run verification
  - How to regenerate (and when it's appropriate)
  - What to do if tests fail
  - Why timing is filtered out
  - Expected execution times per example
- [ ] Add note to main README about golden tests

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward scripting, main complexity is ensuring determinism and correct timing filtering

---

## Definition of Done

- [ ] Script created and executable
- [ ] Golden files generated for all examples
- [ ] Timing information properly filtered
- [ ] Verification passes 3 consecutive times
- [ ] Documentation complete
- [ ] Files committed to git
- [ ] No algorithm code modified
