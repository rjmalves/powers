# CLEANUP-008: Remove Redundant "What" Comments

## Context

The codebase contains approximately 50 "what" comments that simply restate what the code obviously does. These comments add noise without value for developers familiar with Rust. The cleanup plan identifies this as a code quality improvement that will enhance readability.

**Examples from cleanup plan**:
- Line 552: "Extract costs into separate vector for sorting" (code clearly shows `costs.iter().map(|t| t.cost).collect()`)
- Line 2229: "Validate parameters" (followed by obvious validation code)
- Line 2338: "Count total solver calls across all trajectories" (followed by `total_solver_calls += ...`)

**Guideline**: If someone familiar with Rust could understand the line without the comment, remove the comment.

**Risk Level**: VERY LOW (comments only, no code changes)

## Acceptance Criteria

- [ ] All redundant "what" comments identified across src/ directory
- [ ] Comments reviewed to ensure they are truly redundant (not "why" comments)
- [ ] Redundant comments removed
- [ ] Valuable "why" comments preserved
- [ ] Code remains readable after comment removal
- [ ] No functional changes to code
- [ ] Pre-checks pass (fmt, clippy, tests)

## Tasks

### Discovery & Classification
- [ ] Search for candidate comments (variable assignments with preceding comment):
  ```bash
  rg "^[\s]*//" src/ --context 1 > comments_audit.txt
  ```
- [ ] Review comments_audit.txt to identify "what" vs "why" comments
- [ ] Create list of redundant comments with file:line references
- [ ] Classify comments into categories:
  - **REMOVE**: Pure "what" comments (restates obvious code)
  - **KEEP**: "Why" comments (explains reasoning/context)
  - **IMPROVE**: Could be made more valuable with slight rewording
  - **UNCERTAIN**: Need second opinion

### Pattern Recognition
Identify common redundant comment patterns:
- [ ] "Initialize X" before `let x = ...`
- [ ] "Loop through Y" before `for y in ...`
- [ ] "Return Z" before `return z;`
- [ ] "Calculate total" before sum operations
- [ ] "Check if valid" before obvious validation
- [ ] "Create new X" before constructor calls
- [ ] "Extract field Y" before `.map(|x| x.y)`

### Example Reviews (From Cleanup Plan)

#### Example 1: Extract costs (Redundant)
**Before**:
```rust
// Extract costs into separate vector for sorting
let mut costs: Vec<f64> = trajectories.iter().map(|t| t.cost).collect();
```
**After**:
```rust
let mut costs: Vec<f64> = trajectories.iter().map(|t| t.cost).collect();
```
**Rationale**: Code is self-explanatory. Any Rust developer understands map-collect pattern.

#### Example 2: Validate parameters (Redundant)
**Before**:
```rust
// Validate parameters
if risk_param < 0.0 || risk_param > 1.0 {
    return Err(Error::InvalidParameter);
}
```
**After**:
```rust
if risk_param < 0.0 || risk_param > 1.0 {
    return Err(Error::InvalidParameter);
}
```
**Rationale**: The validation is obvious. The error type makes it clear.

#### Example 3: Why comment (KEEP)
**Before/After**:
```rust
// Use Kahan summation to minimize floating point error accumulation
let mut sum = 0.0;
let mut compensation = 0.0;
```
**Rationale**: This explains WHY Kahan summation is used, not WHAT the code does. KEEP.

### File-by-File Cleanup
- [ ] `src/sddp/mod.rs` - Main algorithm file (likely highest concentration)
- [ ] `src/subproblem.rs` - Solver interaction code
- [ ] `src/state.rs` - State management
- [ ] `src/scenario.rs` - Scenario generation
- [ ] `src/input.rs` - Input parsing
- [ ] `src/output.rs` - Output generation
- [ ] `src/graph.rs` - Graph structures
- [ ] Other src/ files as needed

### Validation Rules
Before removing a comment, verify:
- [ ] Is this explaining "what" (remove) or "why" (keep)?
- [ ] Would a new team member benefit from this comment? (If yes, consider keeping)
- [ ] Does this comment reference a ticket, paper, or external spec? (Keep)
- [ ] Does this comment explain a non-obvious algorithm or optimization? (Keep)
- [ ] Does this comment explain a workaround or edge case? (Keep)

### Testing
- [ ] Run `cargo fmt` to ensure formatting is clean
- [ ] Run `cargo build --workspace` to verify no doc comment removal broke doc tests
- [ ] Run `cargo test --workspace` (comments shouldn't affect tests, but verify)
- [ ] Visual review: Read through modified files to ensure readability is improved
- [ ] Spot check: Pick 5 random files and verify comment removal makes sense

### Documentation
- [ ] Update CONTRIBUTING.md with comment guidelines:
  ```markdown
  ### Comment Guidelines
  - Explain WHY, not WHAT
  - Remove obvious comments that restate code
  - Keep comments that explain non-obvious algorithms
  - Keep comments that reference external specs/papers/tickets
  - Use doc comments (///) for public API documentation
  ```
- [ ] Add to CHANGELOG.md: "Cleaned up redundant inline comments"

## Technical Notes

### Comment Categories

**✅ KEEP - These Add Value**:
```rust
// Performance: O(n log n) sorting is acceptable here since n < 1000
// Kahan summation to minimize floating point errors
// See Algorithm 3.2 in Pereira et al. (2019)
// SAFETY: This is safe because we verified bounds above
// TODO: Replace with more efficient algorithm (see issue #123)
// Edge case: Handle when reservoir is empty
```

**❌ REMOVE - These Are Redundant**:
```rust
// Create new vector
let vec = Vec::new();

// Loop through items
for item in items { ... }

// Return the result
return result;

// Initialize counter to zero
let count = 0;

// Sort the costs
costs.sort_by(|a, b| a.partial_cmp(b));
```

**🤔 IMPROVE - These Could Be Better**:
```rust
// Calculate cost
let cost = base_cost * multiplier + fixed_cost;
// Better: Include marginal cost for shadow price calculation
```

### Search Patterns for Redundant Comments

```bash
# Find comments before variable declarations
rg "//.*\n\s*let" src/

# Find comments before loops
rg "//.*\n\s*for" src/

# Find comments before return statements
rg "//.*\n\s*return" src/

# Find "Calculate" comments
rg "//.*[Cc]alculate" src/

# Find "Initialize" comments  
rg "//.*[Ii]nitialize" src/
```

### Estimated Distribution
Based on cleanup plan (~50 redundant comments):
- 20-25 in `src/sddp/mod.rs` (large file with lots of algorithm code)
- 10-15 in `src/subproblem.rs`
- 5-10 in `src/state.rs`, `src/scenario.rs`
- 10-15 scattered across other files

## Dependencies

- Blocked by: None (independent task)
- Blocks: CLEANUP-011 (related comment cleanup work)
- Related: CLEANUP-012 (consolidates mathematical derivations in tests)

## Estimated Effort

**1 story point** (4 hours, confidence: high)

Time breakdown:
- Discovery and classification: 1.5 hours
- Removal and verification: 2 hours
- Testing and documentation: 0.5 hours

Low risk, high value. Can be done in parallel with other tickets.
