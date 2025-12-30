# GitHub Copilot CLI Agent Skills

This directory contains [Agent Skills](https://agentskills.io/specification) for the POWE.RS project. Agent Skills are structured instructions that guide GitHub Copilot CLI agents in performing specialized tasks for HPC Rust development and SDDP algorithm optimization.

## Available Skills

### Rust Development & Testing

1. **[rust-benchmarking](./rust-benchmarking/SKILL.md)**
   - Create and run performance benchmarks using Criterion.rs
   - Measure critical code paths and detect performance regressions
   - Reference: `benches/README.md`, `benches/sddp_e2e.rs`, `benches/simd_dot_product.rs`

2. **[rust-profiling](./rust-profiling/SKILL.md)**
   - Profile code using flamegraph, perf, and samply
   - Identify performance bottlenecks and optimization opportunities
   - Reference: `src/timing/` module, `[profile.release]` configuration

3. **[rust-memory-analysis](./rust-memory-analysis/SKILL.md)**
   - Analyze memory usage with DHAT, Heaptrack, and Valgrind
   - Optimize allocations using buffer pooling patterns
   - Reference: `src/memory/buffers.rs`, `mimalloc` feature flag

4. **[rust-coverage](./rust-coverage/SKILL.md)**
   - Measure and improve test coverage using cargo-llvm-cov
   - Target: 95%+ for critical modules (sddp, solver, subproblem)
   - Reference: `.copilot/development/COVERAGE-TOOLING.md`

5. **[rust-clean-code](./rust-clean-code/SKILL.md)**
   - Enforce clean code practices and idiomatic Rust
   - Use rustfmt, clippy, and thiserror for error handling
   - Reference: `rustfmt.toml`, `src/error.rs`

### Optimization & HPC

6. **[highs-integration](./highs-integration/SKILL.md)**
   - Optimize HiGHS LP solver integration
   - Implement warm starting and dual variable extraction
   - Reference: `src/solver.rs` (45KB), `highs-sys = "1.6.4"`

7. **[hpc-optimization](./hpc-optimization/SKILL.md)**
   - Optimize for high-performance computing workloads
   - Use SIMD, rayon parallelization, and cache optimization
   - Reference: `rayon = "1.10.0"`, `simd-optimizations` feature flag

### Algorithm Development

8. **[sddp-development](./sddp-development/SKILL.md)**
   - Guide SDDP algorithm development and testing
   - Understand mathematical foundations and modern improvements
   - Reference: `.copilot/context/01-sddp-mathematical-foundations.md`, `src/sddp/mod.rs` (138KB)

## Skill Structure

Each skill follows the [Agent Skills specification](https://agentskills.io/specification):

```
.github/skills/<skill-name>/
└── SKILL.md          # Skill definition with YAML frontmatter
```

### YAML Frontmatter
Each `SKILL.md` includes:
- `name`: Lowercase skill identifier (e.g., `rust-benchmarking`)
- `description`: Detailed skill purpose
- `license`: MIT
- `metadata.author`: rjmalves
- `metadata.version`: "1.0"
- `metadata.tags`: Relevant tags

### Content
Skills provide:
- **Overview**: High-level purpose and context
- **Infrastructure**: Existing codebase features to leverage
- **Instructions**: Step-by-step guidance with code examples
- **Best Practices**: Proven patterns and anti-patterns
- **File References**: Actual paths to relevant code/documentation
- **Related Skills**: Cross-references to complementary skills

## Using Skills with GitHub Copilot CLI

Skills guide Copilot agents in specialized tasks:

```bash
# Example: Use benchmarking skill to create performance tests
gh copilot --skill rust-benchmarking "Create benchmark for cut evaluation"

# Example: Use profiling skill to find bottlenecks
gh copilot --skill rust-profiling "Profile SDDP backward pass"

# Example: Use SDDP skill to implement algorithm improvement
gh copilot --skill sddp-development "Implement multi-cut SDDP"
```

## Skill Interconnections

Skills are designed to work together:

- **rust-benchmarking** → **rust-profiling**: Measure, then analyze
- **rust-profiling** → **hpc-optimization**: Find hotspots, then optimize
- **rust-coverage** → **sddp-development**: Test coverage for algorithm correctness
- **highs-integration** → **sddp-development**: Solver optimization for SDDP
- **hpc-optimization** → **sddp-development**: Parallel SDDP implementation

## Repository Context

POWE.RS is a high-performance SDDP solver with:
- **Large files**: `src/subproblem.rs` (235KB), `src/state.rs` (137KB), `src/sddp/mod.rs` (138KB)
- **Performance infrastructure**: Criterion benchmarks, timing module, feature flags
- **Documentation**: `.copilot/` directory with context and development guides
- **Testing**: Golden tests, property-based tests, 72%+ coverage

## Contributing

When adding new skills:
1. Follow the [Agent Skills specification](https://agentskills.io/specification)
2. Reference actual files in the POWE.RS codebase
3. Provide concrete examples using real code patterns
4. Include file paths and size information for context
5. Cross-reference related skills

## References

- **Agent Skills Specification**: https://agentskills.io/specification
- **POWE.RS Repository**: https://github.com/rjmalves/powers
- **Project Documentation**: `.copilot/` directory
- **Benchmarking Guide**: `benches/README.md`
- **Coverage Guide**: `.copilot/development/COVERAGE-TOOLING.md`

---

**Last Updated**: 2025-12-30
**Author**: rjmalves
**License**: MIT
