# Copilot CLI Agent Optimization Guide

## Overview

This directory contains **optimized agents for GitHub Copilot CLI**. These agents are specifically designed for CLI workflows and differ from VSCode agents in important ways.

## Key Differences: CLI vs VSCode Agents

| Aspect | VSCode Agents | CLI Agents |
|--------|---------------|------------|
| **Purpose** | Consultation & guidance | Action & execution |
| **Length** | Long, comprehensive | Concise, focused |
| **Style** | Educational | Directive |
| **Scope** | Broad expertise | Specific tasks |
| **Context** | Persistent in chat | Fresh per invocation |

## Agent Directory Structure

### Optimized CLI Agents (⚡ Use These)

- **`rust-implementer.md`** - Implements Rust features following specs
- **`doc-writer.md`** - Creates and updates documentation
- **`code-reviewer.md`** - Reviews code changes for quality
- **`test-engineer.md`** - Writes comprehensive tests
- **`perf-optimizer.md`** - Identifies and fixes performance issues
- **`bug-investigator.md`** - Debugs and fixes issues systematically
- **`sprint-planner.md`** - Breaks down features into tasks *(kept from original)*

### Legacy VSCode Agents (📚 Reference Only)

- **`architect.md`** - Architecture consultation (verbose)
- **`documentation-specialist.md`** - Documentation philosophy (too long)
- **`hpc-developer.md`** - Developer profile (consultative)
- **`reviewer.md`** - Review guide (educational)

## When to Use Each Agent

### 🔨 Implementation Tasks

**Use: `rust-implementer`**

```bash
# Implement a new feature
gh copilot suggest "Implement PAR model validation in src/stochastic_process.rs"

# Add functionality
gh copilot suggest "Add method to calculate storage bounds in SubProblem"

# Refactor code
gh copilot suggest "Refactor cut selection to use strategy pattern"
```

**Agent will**:
- ✅ Implement working Rust code
- ✅ Follow performance best practices
- ✅ Format with `cargo fmt`
- ✅ Fix clippy warnings
- ✅ Add basic tests

### 📝 Documentation Tasks

**Use: `doc-writer`**

```bash
# Create documentation
gh copilot suggest "Write user guide section for PAR model configuration"

# Update docs for new feature
gh copilot suggest "Document the new cut selection strategies in README"

# Write examples
gh copilot suggest "Create example showing risk-averse policy configuration"
```

**Agent will**:
- ✅ Write clear, example-rich documentation
- ✅ Structure for multiple audiences
- ✅ Include runnable code examples
- ✅ Update relevant doc files

### 👀 Code Review

**Use: `code-reviewer`**

```bash
# Review recent changes
gh copilot suggest "Review the changes in src/cut.rs"

# Check specific PR
gh copilot suggest "Review pull request #123"

# Pre-merge check
gh copilot suggest "Review code before merging feature branch"
```

**Agent will**:
- ✅ Check formatting and linting
- ✅ Identify correctness issues
- ✅ Flag performance concerns
- ✅ Verify tests exist
- ✅ Provide actionable feedback

### 🧪 Testing

**Use: `test-engineer`**

```bash
# Write tests for new code
gh copilot suggest "Write comprehensive tests for PAR sampling"

# Add missing coverage
gh copilot suggest "Add edge case tests for cut selection"

# Create benchmarks
gh copilot suggest "Benchmark backward pass performance"
```

**Agent will**:
- ✅ Write unit tests with edge cases
- ✅ Create integration tests
- ✅ Add numerical validation
- ✅ Set up benchmarks
- ✅ Test parallel correctness

### ⚡ Performance Optimization

**Use: `perf-optimizer`**

```bash
# Find bottlenecks
gh copilot suggest "Profile and identify performance bottlenecks"

# Optimize specific code
gh copilot suggest "Optimize memory allocations in forward pass"

# Improve algorithm
gh copilot suggest "Make cut selection more efficient"
```

**Agent will**:
- ✅ Profile code first
- ✅ Identify actual bottlenecks
- ✅ Implement data-driven optimizations
- ✅ Measure improvements
- ✅ Add benchmarks

### 🐛 Debugging

**Use: `bug-investigator`**

```bash
# Investigate failures
gh copilot suggest "Debug why test_parallel_forward fails intermittently"

# Fix issues
gh copilot suggest "Fix numerical instability in water balance calculation"

# Resolve errors
gh copilot suggest "Fix the infeasibility error in backward pass"
```

**Agent will**:
- ✅ Reproduce the bug
- ✅ Investigate systematically
- ✅ Identify root cause
- ✅ Implement minimal fix
- ✅ Add regression test

### 📋 Planning

**Use: `sprint-planner`**

```bash
# Break down features
gh copilot suggest "Create sprint plan for implementing multi-cut SDDP"

# Plan epic
gh copilot suggest "Break down risk measure implementation into tasks"

# Estimate work
gh copilot suggest "Create detailed tickets for PAR model improvements"
```

**Agent will**:
- ✅ Analyze requirements
- ✅ Break into atomic tickets
- ✅ Include testing & docs
- ✅ Identify dependencies
- ✅ Estimate effort

## Best Practices for CLI Agents

### 1. Be Specific in Prompts

❌ **Vague**: "Improve the code"  
✅ **Specific**: "Optimize memory allocations in backward_pass() function"

❌ **Vague**: "Add tests"  
✅ **Specific**: "Write unit tests for cut_selection with edge cases: empty cuts, single cut, max_cuts exceeded"

### 2. Provide Context When Needed

```bash
# Good: Includes what, where, and why
gh copilot suggest "Implement PAR model validation in src/stochastic_process.rs 
to verify parameters before sampling. Should check: periods match stages, 
positive variance, correlation matrices are valid."
```

### 3. Use Agents for Their Specialization

Don't ask `doc-writer` to implement code or `rust-implementer` to write tests - use specialized agents:

```bash
# Step 1: Implement
gh copilot suggest --agent rust-implementer "Add cut selection limit parameter"

# Step 2: Test
gh copilot suggest --agent test-engineer "Write tests for cut selection limit"

# Step 3: Document
gh copilot suggest --agent doc-writer "Document cut selection limit in API docs"

# Step 4: Review
gh copilot suggest --agent code-reviewer "Review cut selection changes"
```

### 4. Iterate Based on Output

If the agent's output isn't quite right:

```bash
# First attempt
gh copilot suggest "Implement cut selection strategy"

# If needs refinement
gh copilot suggest "Refine cut selection to use trait-based strategy pattern 
for extensibility, following the same pattern as risk measures"
```

### 5. Verify Agent Work

Always verify agent outputs:

```bash
# After implementation
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test

# After documentation
# Check that examples compile and run

# After optimization
cargo bench  # Compare before/after
```

## Migration from VSCode Agents

If you have existing VSCode agents, here's how to migrate:

### 1. Keep for Reference

Your original agents contain valuable domain knowledge - keep them in a `reference/` subdirectory:

```bash
mkdir -p .copilot/agents/reference
mv .copilot/agents/{architect,documentation-specialist,hpc-developer,reviewer}.md .copilot/agents/reference/
```

### 2. Use CLI-Optimized Versions

The new CLI agents distill the essence of your originals into action-oriented formats.

### 3. Gradually Refine

Based on usage:
- Add project-specific patterns you discover
- Include common commands for your workflow
- Document tricky aspects of your codebase

## Measuring Agent Effectiveness

Track how well agents work:

### Quality Metrics
- ✅ Code compiles without formatting/linting issues
- ✅ Tests pass on first run
- ✅ Documentation is clear and accurate
- ✅ Reviews catch real issues

### Efficiency Metrics
- ✅ Less time spent on boilerplate
- ✅ Fewer review iterations
- ✅ Faster onboarding of new patterns
- ✅ More consistent code quality

### When to Update Agents

Update agents when:
- You notice repeated corrections needed
- New patterns emerge in the codebase
- Team develops new conventions
- Technology stack changes

## Advanced Usage

### Combining Agents

Use multiple agents in sequence for complex tasks:

```bash
# 1. Plan the work
gh copilot suggest --agent sprint-planner "Plan multi-cut SDDP implementation"

# 2. Implement based on plan
gh copilot suggest --agent rust-implementer "Implement ticket #1: Refactor cut storage"

# 3. Write tests
gh copilot suggest --agent test-engineer "Test new cut storage"

# 4. Optimize if needed
gh copilot suggest --agent perf-optimizer "Profile and optimize cut storage"

# 5. Document
gh copilot suggest --agent doc-writer "Document cut storage API"

# 6. Review
gh copilot suggest --agent code-reviewer "Final review of cut storage changes"
```

### Custom Workflows

Create shell functions for common workflows:

```bash
# In your .bashrc or .zshrc
powers-implement() {
    gh copilot suggest --agent rust-implementer "$1" && \
    cargo fmt --all && \
    cargo clippy --all-targets --all-features -- -D warnings && \
    cargo test
}

powers-optimize() {
    cargo flamegraph --bin powers -- example && \
    gh copilot suggest --agent perf-optimizer "Optimize based on flamegraph" && \
    cargo bench
}
```

## Troubleshooting

### Agent Not Found

Make sure agents are in `.copilot/agents/` directory and have `.md` extension.

### Agent Gives Generic Responses

Make your prompt more specific and include relevant context from the codebase.

### Agent Makes Mistakes

Remember: agents are assistants, not infallible. Always:
- Review the output
- Run tests
- Use your judgment

## Continuous Improvement

Your agents should evolve with your project:

1. **Weekly**: Note any repeated corrections
2. **Monthly**: Review agent effectiveness
3. **Quarterly**: Update agents with new patterns

Keep agents concise, actionable, and focused on making you more productive!

---

**Remember**: CLI agents are your specialized team members. Use them for their strengths, verify their work, and refine them over time.
