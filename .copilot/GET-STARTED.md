# Get Started with CLI Agents - 5 Minute Guide

## Step 1: Try Your First Agent (1 minute)

Open your terminal and try this:

```bash
# Simple implementation task
gh copilot suggest "Add a simple getter method to SubProblem struct"
```

When prompted, select `@rust-implementer` agent.

**Expected**: You'll get working Rust code with doc comments.

## Step 2: Chain Agents for a Complete Task (2 minutes)

Let's add a simple feature end-to-end:

```bash
# 1. Implement
gh copilot suggest "@rust-implementer Add max_iterations field to Config struct with getter"

# 2. Test
gh copilot suggest "@test-engineer Write tests for max_iterations getter"

# 3. Document
gh copilot suggest "@doc-writer Add doc comment explaining max_iterations parameter"
```

## Step 3: Understand Agent Specializations (1 minute)

Quick reference:

| Need | Use Agent |
|------|-----------|
| Write Rust code | `@rust-implementer` |
| Write tests | `@test-engineer` |
| Write docs | `@doc-writer` |
| Review code | `@code-reviewer` |
| Fix performance | `@perf-optimizer` |
| Debug issues | `@bug-investigator` |
| Plan features | `@sprint-planner` |

## Step 4: Verify Agent Output (1 minute)

Always run after using agents:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Real-World Example

Let's add cut limiting feature:

```bash
# 1. Plan the work
gh copilot suggest "@sprint-planner Break down cut limiting feature into tickets"

# 2. Implement core function
gh copilot suggest "@rust-implementer Implement limit_cuts function that keeps most recent N cuts"

# 3. Add tests
gh copilot suggest "@test-engineer Write tests for limit_cuts: empty, single, at limit, over limit"

# 4. Verify implementation
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test

# 5. Document it
gh copilot suggest "@doc-writer Document limit_cuts in cut.rs with example"

# 6. Review
gh copilot suggest "@code-reviewer Review cut limiting implementation"
```

## Tips for Success

### ✅ Do This

**Be specific**:
```bash
@rust-implementer Add validate_storage() method to Hydro that checks:
- Storage is non-negative
- Storage is within bounds
- Returns Result<(), ValidationError>
```

**Provide context**:
```bash
@bug-investigator Debug test_parallel_forward failure.
Fails 30% of time with 8 threads, never with 1 thread.
ThreadSanitizer reports race at cut_storage.rs:145
```

**Chain related tasks**:
```bash
# Implement → Test → Document → Review
```

### ❌ Avoid This

**Too vague**:
```bash
@rust-implementer Improve the code
```

**Wrong agent**:
```bash
@doc-writer Fix the performance issue  # Use @perf-optimizer
```

**No verification**:
```bash
# Agent gives code → commit immediately  # Always test first!
```

## Common Workflows

### Feature Development
```bash
@sprint-planner → @rust-implementer → @test-engineer → @doc-writer → @code-reviewer
```

### Bug Fix
```bash
@bug-investigator → cargo test → @test-engineer (add regression test) → @code-reviewer
```

### Performance Work
```bash
cargo flamegraph → @perf-optimizer → cargo bench → @test-engineer → @code-reviewer
```

### Documentation Sprint
```bash
@doc-writer (guide) → @doc-writer (examples) → @code-reviewer
```

## Next Steps

1. **Read the full guide**: `.copilot/agents/README.md`
2. **Keep quick reference handy**: `.copilot/AGENT-QUICK-REFERENCE.md`
3. **Compare with VSCode agents**: `.copilot/agents/COMPARISON.md`
4. **Understand what changed**: `.copilot/OPTIMIZATION-SUMMARY.md`

## Getting Help

- **Agent not working?** Check `.copilot/agents/README.md` troubleshooting section
- **Need design advice?** Use VSCode agents in `.copilot/agents/reference/`
- **Unclear prompt?** Look at examples in this guide and README

## Remember

🎯 **Agents are your team** - Use the right specialist for each job  
🔍 **Always verify** - Run format, lint, test after agent work  
📝 **Be specific** - Better prompts = better results  
🔄 **Iterate** - Refine prompts if output isn't quite right  
📚 **Learn** - Read reference agents for deeper understanding  

---

**You're ready!** Start with simple tasks, build confidence, then tackle bigger features. The agents will help you ship faster while maintaining quality. 🚀
