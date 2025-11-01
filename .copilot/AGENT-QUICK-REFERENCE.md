# Copilot CLI Agent Quick Reference

## 🚀 Quick Start

1. **Implementation**: `@rust-implementer` - Writes Rust code
2. **Testing**: `@test-engineer` - Creates comprehensive tests
3. **Documentation**: `@doc-writer` - Writes clear docs
4. **Review**: `@code-reviewer` - Reviews code quality
5. **Performance**: `@perf-optimizer` - Finds and fixes bottlenecks
6. **Debugging**: `@bug-investigator` - Fixes bugs systematically
7. **Planning**: `@sprint-planner` - Breaks down features

## 📝 Usage Examples

### Implement Feature
```bash
@rust-implementer Implement PAR model parameter validation in 
src/stochastic_process.rs. Check: periods > 0, variance > 0, 
valid correlation matrices.
```

### Write Tests
```bash
@test-engineer Write comprehensive tests for cut_selection function.
Test cases: empty input, single cut, limit exceeded, numerical edge cases.
```

### Create Documentation
```bash
@doc-writer Add user guide section explaining PAR model configuration
with realistic example for 12-month seasonal pattern.
```

### Review Code
```bash
@code-reviewer Review changes in src/cut.rs for correctness,
performance, and test coverage.
```

### Optimize Performance
```bash
@perf-optimizer Profile backward_pass and optimize the top bottleneck.
Include before/after benchmarks.
```

### Debug Issue
```bash
@bug-investigator Fix intermittent test failure in test_parallel_forward.
Fails ~30% of time with 8 threads, never with 1 thread.
```

### Plan Work
```bash
@sprint-planner Break down "Add multi-cut SDDP support" into
atomic tickets with tests, docs, and benchmarks.
```

## 🎯 Agent Selection Guide

**I need to...**

- ✏️ Write new Rust code → `@rust-implementer`
- 🧪 Add tests → `@test-engineer`
- 📖 Write/update docs → `@doc-writer`
- 👀 Review changes → `@code-reviewer`
- ⚡ Make it faster → `@perf-optimizer`
- 🐛 Fix a bug → `@bug-investigator`
- 📋 Plan features → `@sprint-planner`

## 💡 Tips

### Be Specific
❌ "Improve the code"  
✅ "Reduce allocations in backward_pass by reusing buffer"

### Provide Context
Include:
- What file/function
- What the goal is
- Any constraints or requirements

### Verify Output
Always run after agent work:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

### Chain Agents
For complex work, use agents in sequence:
1. Plan → `@sprint-planner`
2. Implement → `@rust-implementer`
3. Test → `@test-engineer`
4. Document → `@doc-writer`
5. Review → `@code-reviewer`

## 🔧 Common Workflows

### New Feature
```bash
# 1. Plan
@sprint-planner Create implementation plan for [feature]

# 2. Implement
@rust-implementer Implement [specific task from plan]

# 3. Test
@test-engineer Write tests for [feature]

# 4. Document
@doc-writer Document [feature] in README and add example

# 5. Review
@code-reviewer Review [feature] implementation
```

### Bug Fix
```bash
# 1. Investigate
@bug-investigator Debug [failing test/issue]

# 2. Verify
cargo test

# 3. Document
@doc-writer Update troubleshooting guide if needed
```

### Performance Work
```bash
# 1. Profile
cargo flamegraph --bin powers -- example

# 2. Optimize
@perf-optimizer Optimize [identified bottleneck]

# 3. Benchmark
cargo bench

# 4. Verify
cargo test  # Ensure correctness maintained
```

## 📚 More Details

See [agents/README.md](README.md) for comprehensive guide.

---

**Pro tip**: Agents are your specialized team. Use the right specialist for each job!
