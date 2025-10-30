# CLEANUP-012: Extract lognormal3.rs Tutorial to Documentation

## Context

The `src/lognormal3.rs` file has 70 lines of comprehensive module-level documentation for a 420-line module (17% of the file is documentation). While this documentation is excellent, it's tutorial-level content that belongs in user-facing documentation (`docs/reference/`) rather than inline with the source code.

**Current State**: Module doc includes:
- Mathematical background and theory
- Statistical properties and derivations
- Algorithm explanation
- Performance characteristics
- Integration examples
- Usage examples

**Goal**: Move comprehensive tutorial to `docs/reference/distributions.md`, keep concise module summary in source.

**Risk Level**: LOW (documentation reorganization)

## Acceptance Criteria

- [ ] Tutorial-level documentation extracted to `docs/reference/distributions.md`
- [ ] Module doc comment condensed to 10-15 lines with reference to docs
- [ ] All mathematical content preserved in new documentation
- [ ] Documentation includes working code examples
- [ ] Cross-references added between source and docs
- [ ] Documentation builds correctly (`cargo doc`, mdBook if used)
- [ ] CHANGELOG.md updated

## Tasks

### Analysis
- [ ] Read current `lognormal3.rs` module documentation (lines 1-70)
- [ ] Identify sections to extract vs keep inline
- [ ] Determine target structure for `docs/reference/distributions.md`
- [ ] Check if other distribution implementations exist (for consistency)
- [ ] Review existing docs/ structure for appropriate location

### Create docs/reference/distributions.md

Structure:
```markdown
# Probability Distributions

## 3-Parameter Log-Normal Distribution (LN3)

### Overview
Brief introduction and use cases

### Mathematical Definition
Detailed mathematical background

### Statistical Properties
Properties, moments, parameter relationships

### Implementation Details
Algorithm, performance characteristics

### Usage Examples
Code examples with explanations

### References
Academic papers, textbooks
```

- [ ] Extract mathematical background from module docs
- [ ] Extract statistical properties section
- [ ] Extract algorithm explanation
- [ ] Extract performance characteristics
- [ ] Extract usage examples
- [ ] Add references section (if applicable)
- [ ] Format with proper markdown (headings, code blocks, math notation)
- [ ] Add code examples that compile and run

### Condense Module Documentation

**Target (10-15 lines)**:
```rust
//! 3-Parameter Log-Normal Distribution (LN3) for non-negative scenario generation.
//!
//! Implements `X = γ + exp(Y)` where `Y ~ N(μ, σ²)`.
//!
//! # Features
//! - O(1) sampling with zero allocations
//! - Numerically stable parameter estimation
//! - Exact moments and quantiles
//!
//! # Usage
//! ```rust
//! # use powers::lognormal3::LogNormal3;
//! let dist = LogNormal3::from_moments(100.0, 1000.0, 0.0)?;
//! let sample = dist.sample(&mut rng);
//! ```
//!
//! For detailed mathematical background and properties, see
//! [`docs/reference/distributions.md`](../../docs/reference/distributions.md).
```

- [ ] Write concise module summary
- [ ] Include minimal usage example
- [ ] Add reference to detailed documentation
- [ ] Keep API-critical information inline
- [ ] Remove verbose explanations

### Update Cross-References

- [ ] Add link from lognormal3.rs module doc to distributions.md
- [ ] Add link from distributions.md back to lognormal3.rs API docs
- [ ] Update README.md if it references lognormal3 documentation
- [ ] Update docs/README.md to include distributions.md in table of contents
- [ ] Add distributions.md to SUMMARY.md if using mdBook

### Documentation Quality Checks

- [ ] Verify all mathematical notation is clear
- [ ] Ensure code examples compile:
  ```bash
  cargo test --doc lognormal3
  ```
- [ ] Check markdown formatting:
  ```bash
  mdl docs/reference/distributions.md  # if markdown linter available
  ```
- [ ] Verify internal links work
- [ ] Ensure math formulas render correctly (if using KaTeX/MathJax)
- [ ] Proofread for typos and clarity

### Testing
- [ ] Run `cargo doc --open` to verify module doc renders correctly
- [ ] Verify doc tests in lognormal3.rs still pass: `cargo test --doc`
- [ ] Build full documentation to verify cross-references work
- [ ] Visual review of rendered documentation
- [ ] Spot-check that API docs are still helpful for developers

### Documentation
- [ ] Add CHANGELOG.md entry: "Moved LN3 tutorial documentation to docs/reference/distributions.md"
- [ ] Update CONTRIBUTING.md with documentation organization guidelines:
  ```markdown
  ### Documentation Organization
  - Module docs (//!): Concise API overview, usage examples, links to detailed docs
  - docs/reference/: Detailed tutorials, mathematical background, algorithms
  - docs/guides/: User guides and tutorials
  - Function docs (///): API documentation, parameters, examples
  ```

## Technical Notes

### Content to Extract vs Keep

**EXTRACT to docs/reference/distributions.md**:
- Mathematical derivations and proofs
- Statistical theory and properties
- Detailed algorithm explanations
- Performance analysis and benchmarks
- Integration with other components
- Academic references and citations
- Extended usage examples

**KEEP in lognormal3.rs module docs**:
- Brief description of what it does
- Key features/characteristics
- Minimal usage example
- Link to detailed documentation
- API-critical information

### Distribution Documentation Template

```markdown
# Distribution Name

## Overview
[1-2 paragraphs: What it is, when to use it]

## Mathematical Definition
[Formal mathematical definition]
- PDF: f(x) = ...
- Parameters: ...
- Support: ...

## Statistical Properties
[Moments, quantiles, special properties]
- Mean: E[X] = ...
- Variance: Var(X) = ...
- Skewness: ...

## Implementation
[Algorithm used, computational complexity, numerical stability]

### Performance Characteristics
- Time: O(1) per sample
- Space: O(1) storage
- Numerical: Stable for σ < 3.0

## Usage

### Basic Example
```rust
[Minimal example]
```

### Parameter Estimation
```rust
[Example of fitting to data]
```

### Integration with SDDP
```rust
[Example in context]
```

## References
[Academic papers, textbooks]
```

### Related Files to Check
- [ ] Other distribution implementations (for consistency)
- [ ] `docs/README.md` - Update table of contents
- [ ] `README.md` - Check if lognormal3 is mentioned
- [ ] `docs/reference/INPUT-SPECIFICATION.md` - Check if cross-references needed

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-013 (solver.rs documentation), CLEANUP-014 (mod.rs documentation)

## Estimated Effort

**1 story point** (4-5 hours, confidence: high)

Time breakdown:
- Create distributions.md structure: 1 hour
- Extract and organize content: 2 hours
- Condense module docs: 0.5 hours
- Cross-references and formatting: 1 hour
- Testing and review: 0.5 hours

Clear scope, well-defined transformation.
