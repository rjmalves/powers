# CLEANUP-011: Consolidate Mathematical Derivation Comments in Tests

## Context

Test files contain excellent mathematical derivation comments that verify test correctness (e.g., manual calculations for percentiles, standard deviations). While valuable for verification, very long derivations can make test code harder to scan. The cleanup plan recommends KEEPING these (they're valuable) but improving their organization.

**Examples from cleanup plan**:
```rust
// Lines 4131-4138 - Percentile calculation
// 20th percentile: between index 0 and 1
// index = 0.2 * 4 = 0.8
// result = 1.0 * 0.2 + 2.0 * 0.8 = 1.8

// Lines 4215-4217 - Standard deviation
// Manual calculation: std = sqrt(((100-200)^2 + (200-200)^2 + (300-200)^2) / 3)
// = sqrt((10000 + 0 + 10000) / 3) = sqrt(20000/3) ≈ 81.65
```

**Goal**: Improve organization without losing valuable verification information.

**Risk Level**: VERY LOW (test comments only)

## Acceptance Criteria

- [ ] All mathematical derivation comments in tests/ identified and catalogued
- [ ] Long derivations (>5 lines) moved to test function doc comments
- [ ] Inline derivations kept concise (1-2 lines with result)
- [ ] Test readability improved (easier to scan test logic)
- [ ] All derivation information preserved (no information loss)
- [ ] Tests still pass (comments only, no logic changes)
- [ ] Doc comments render correctly in `cargo doc --document-private-items`

## Tasks

### Discovery
- [ ] Search for mathematical derivation comments in tests:
  ```bash
  rg "// Manual calculation|// Expected:|// Calculation:" tests/ -A 5
  rg "sqrt|pow|^2" tests/ --type rust -B 2 -A 2
  ```
- [ ] Identify tests with long mathematical comments (>5 lines)
- [ ] Create inventory of derivation comments with:
  - Test file and function name
  - Lines containing derivation
  - Length (line count)
  - Assessment: MOVE to doc comment or KEEP inline with condensing

### Classification

#### Long Derivations (>5 lines) - MOVE to Doc Comments
Examples:
- Complex statistical calculations
- Multi-step numerical derivations
- Detailed algorithm explanations

#### Short Derivations (1-2 lines) - KEEP Inline
Examples:
- Simple arithmetic: `// 0.2 * 4 = 0.8`
- Expected results: `// Expected: 1.8`

#### Medium Derivations (3-4 lines) - IMPROVE
- Keep concise version inline
- Move details to doc comment if needed

### Improvement Strategy

#### Pattern 1: Move Detailed Derivation to Doc Comment

**Before**:
```rust
#[test]
fn test_percentile_calculation() {
    // 20th percentile calculation:
    // Given data: [1.0, 2.0, 3.0, 4.0, 5.0]
    // index = 0.2 * 4 = 0.8
    // Interpolate between index 0 (1.0) and index 1 (2.0)
    // result = 1.0 * (1 - 0.8) + 2.0 * 0.8 
    //        = 1.0 * 0.2 + 2.0 * 0.8
    //        = 0.2 + 1.6 = 1.8
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = percentile(&data, 0.2);
    assert_eq!(result, 1.8);
}
```

**After**:
```rust
/// Test 20th percentile calculation with interpolation.
///
/// # Expected Calculation
/// Given data: [1.0, 2.0, 3.0, 4.0, 5.0]
/// - Index position: 0.2 × 4 = 0.8
/// - Interpolate between index 0 (1.0) and index 1 (2.0)
/// - Result: 1.0 × (1 - 0.8) + 2.0 × 0.8 = 1.0 × 0.2 + 2.0 × 0.8 = 0.2 + 1.6 = 1.8
#[test]
fn test_percentile_calculation() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = percentile(&data, 0.2);
    assert_eq!(result, 1.8); // See doc comment for derivation
}
```

#### Pattern 2: Condense Medium-Length Derivations

**Before**:
```rust
#[test]
fn test_standard_deviation() {
    // Manual calculation:
    // std = sqrt(((100-200)^2 + (200-200)^2 + (300-200)^2) / 3)
    // = sqrt((10000 + 0 + 10000) / 3)
    // = sqrt(20000/3) ≈ 81.65
    let data = vec![100.0, 200.0, 300.0];
    let result = std_dev(&data);
    assert!((result - 81.65).abs() < 0.01);
}
```

**After**:
```rust
/// Test standard deviation calculation.
///
/// # Expected Calculation
/// σ = sqrt(Σ(x - μ)² / n)
/// For data [100, 200, 300]: σ = sqrt((100² + 0² + 100²) / 3) ≈ 81.65
#[test]
fn test_standard_deviation() {
    let data = vec![100.0, 200.0, 300.0];
    let result = std_dev(&data);
    assert!((result - 81.65).abs() < 0.01); // σ ≈ 81.65
}
```

#### Pattern 3: Keep Short Derivations Inline

**Before/After (No Change)**:
```rust
#[test]
fn test_simple_sum() {
    let total = vec![10, 20, 30].iter().sum::<i32>();
    assert_eq!(total, 60); // 10 + 20 + 30 = 60
}
```

### File-by-File Review

Priority files (likely to have mathematical comments):
- [ ] `tests/test_scenario.rs` - Scenario generation math
- [ ] `tests/test_lognormal_scenarios.rs` - Distribution calculations
- [ ] `tests/test_numerical_validation.rs` - Numerical accuracy tests
- [ ] `tests/test_sddp_algorithm.rs` - Algorithm verification
- [ ] `tests/test_par_estimation.rs` - PAR model statistics
- [ ] Other test files as discovered

### Doc Comment Best Practices

When moving derivations to doc comments:
- [ ] Use `///` for doc comments (not `//`)
- [ ] Use markdown formatting for readability:
  - Math expressions: Use unicode symbols (×, ², √) or ASCII approximations
  - Code blocks for complex formulas
  - Lists for step-by-step derivations
- [ ] Include section heading: `# Expected Calculation` or `# Mathematical Derivation`
- [ ] Keep formulas readable (proper spacing, alignment)
- [ ] Reference external sources if applicable (papers, textbooks)

### Validation
- [ ] Run `cargo test --workspace` to verify no logic changes
- [ ] Run `cargo doc --document-private-items` to verify doc comments render
- [ ] Visual review of rendered documentation
- [ ] Spot-check test files for readability improvement
- [ ] Verify mathematical accuracy preserved in doc comments

### Documentation
- [ ] Add CHANGELOG.md entry: "Improved test documentation by consolidating mathematical derivations"
- [ ] Update CONTRIBUTING.md with test documentation guidelines:
  ```markdown
  ### Test Documentation
  - Use doc comments (///) for test functions with complex derivations
  - Keep inline comments concise (1-2 lines for simple calculations)
  - Document expected results and how they were calculated
  - Use markdown formatting for formulas and step-by-step derivations
  ```

## Technical Notes

### When to Move vs Keep Inline

**MOVE to Doc Comment**:
- Derivation >5 lines
- Multi-step calculations
- Statistical formulas with notation
- Algorithm explanations
- References to papers/textbooks

**KEEP Inline (but condense)**:
- Simple arithmetic (1-2 operations)
- Expected results
- Array index calculations
- Direct formula applications

### Doc Comment Formatting Examples

**Statistical Formula**:
```rust
/// # Expected Calculation
/// Sample variance: s² = Σ(xᵢ - x̄)² / (n - 1)
/// For data [2, 4, 6]: s² = ((2-4)² + (4-4)² + (6-4)²) / 2 = 8 / 2 = 4
```

**Algorithm Steps**:
```rust
/// # Expected Calculation
/// 1. Sort data: [1, 3, 5, 7, 9]
/// 2. Calculate position: p = 0.25 × (n - 1) = 0.25 × 4 = 1.0
/// 3. Exact index: return data[1] = 3.0
```

**Multi-Step Derivation**:
```rust
/// # Mathematical Derivation
/// Given PAR(1) model: Xₜ = φXₜ₋₁ + εₜ
/// 
/// Expected steady-state variance:
/// ```text
/// Var(X) = Var(φX + ε)
///        = φ² Var(X) + Var(ε)
/// Var(X) = Var(ε) / (1 - φ²)
/// ```
/// 
/// For φ=0.8, σ²=1.0: Var(X) = 1.0 / (1 - 0.64) = 2.78
```

### Search Patterns for Mathematical Comments

```bash
# Find calculation comments
rg "// [Mm]anual calculation|// [Cc]alculation:|// [Ee]xpected:" tests/ -A 5

# Find comments with math symbols/operations  
rg "sqrt|pow|\^2|\^3|\+|\-|\*|/" tests/ -B 1 -A 1 --type rust

# Find multi-line comment blocks in tests
rg "^[\s]*//" tests/ -A 3 --type rust | grep -C 3 "^[0-9]"

# Find statistical terms
rg "variance|deviation|percentile|quantile|mean|median" tests/ -B 1 -A 3
```

## Dependencies

- Blocked by: None (independent task)
- Blocks: None
- Related: CLEANUP-008, CLEANUP-009 (general comment improvement work)

## Estimated Effort

**1.5 story points** (6-8 hours, confidence: medium)

Time breakdown:
- Discovery and cataloging: 2 hours
- Doc comment migration: 3-4 hours
- Formatting and review: 1-2 hours
- Validation and testing: 1 hour

Moderate effort due to need for careful preservation of mathematical accuracy and readability assessment.
