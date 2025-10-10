# POWE.RS Example Suite

This directory contains a progressive series of example problems demonstrating POWE.RS capabilities, from simple deterministic problems to large-scale stochastic systems.

## Learning Path

The examples are designed to be completed in order, with each building on concepts from the previous:

### 📘 Example 1: Deterministic 2-Stage
**Difficulty**: Beginner  
**Time**: 5 minutes  
**Concepts**: Basic SDDP, trade-offs, JSON format

Simple deterministic hydrothermal problem with 1 hydro + 1 thermal serving constant load over 2 stages. Perfect for verifying installation and understanding basic mechanics.

➡️ [Example 1 README](01-deterministic/README.md)

**Run**: `cargo run --release examples/01-deterministic`

---

### 📗 Example 2: Basic Stochastic
**Difficulty**: Beginner  
**Time**: 10 minutes  
**Concepts**: Stochasticity, scenario trees, risk management

Introduces uncertainty with 2 hydros + 2 thermals and stochastic inflows. Demonstrates how SDDP learns robust policies across scenario branches.

➡️ [Example 2 README](02-stochastic/README.md)

**Run**: `cargo run --release examples/02-stochastic`

---

### 📕 Example 3: Multi-Stage Hydrothermal
**Difficulty**: Intermediate  
**Time**: 15 minutes  
**Concepts**: Long-term planning, seasonality, network constraints

Scales to 24 stages (monthly) with seasonal patterns, 5 hydros + 5 thermals on 2 buses with network constraints. Realistic problem size for learning long-term storage management.

➡️ [Example 3 README](03-multistage/README.md)

**Run**: `cargo run --release examples/03-multistage`

---

### 📙 Example 4: Hydrothermal Cascade
**Difficulty**: Intermediate  
**Time**: 15 minutes  
**Concepts**: Cascade coupling, spillage management, downstream dependencies

Introduces hydro cascade constraints where upstream decisions affect downstream reservoirs. Demonstrates coupled state variables and spillage penalties.

➡️ [Example 4 README](04-cascade/README.md)

**Run**: `cargo run --release examples/04-cascade`

---

### 📓 Example 5: Large-Scale Brazilian System
**Difficulty**: Advanced  
**Time**: 10 minutes  
**Concepts**: Production-scale system, performance, multi-region optimization

Brazilian-sized system with 156 hydros, 121 thermals, 5 buses, 60 stages (5 years). Demonstrates production-quality performance and scalability with realistic cascade structures and seasonal patterns.

➡️ [Example 5 README](05-large-scale-brazilian/README.md)

**Run**: `cargo run --release examples/05-large-scale-brazilian`

---

## Running All Examples

Use the provided script to run all examples and verify they complete successfully:

```bash
# From project root
./scripts/run_examples.sh
```

This will run each example and report PASSED/FAILED status.

## Running Individual Examples

```bash
# General syntax
cargo run --release examples/<example-name>

# Or using the shorthand (if PATH is set)
powers examples/<example-name>
```

## Understanding the JSON Files

Each example directory contains 4 JSON files:

- **`config.json`**: SDDP algorithm configuration
  - Number of training iterations
  - Number of forward passes per iteration
  - Number of simulation scenarios
  - Random seed for reproducibility
  - Output path for results

- **`system.json`**: Physical power system
  - Buses (nodes in the electrical network)
  - Transmission lines (connections between buses)
  - Thermal generators (cost, capacity, min/max generation)
  - Hydro generators (storage, turbining capacity, productivity, spillage)

- **`graph.json`**: Scenario tree structure
  - Nodes (stages in time)
  - Edges (transitions between stages with probabilities)
  - Temporal configuration (start/end dates, stage IDs, season IDs)
  - Risk measures and stochastic processes

- **`recourse.json`**: Uncertainty and initial conditions
  - Initial storage levels (reservoir water)
  - Initial inflow history (lagged inflows for auto-regressive processes)
  - Uncertainty distributions by season (inflow and demand distributions)
  - Number of scenario branchings per season

## Key Concepts

### Resource Balancing

All examples are carefully balanced to avoid **trivial solutions**:

❌ **Avoid**: Demand >> Capacity → All deficit (no learning)  
❌ **Avoid**: Capacity >> Demand → Zero cost (no learning)  
❌ **Avoid**: Inflow >> Demand → Always hydro (no thermal trade-off)

✅ **Goal**: Create meaningful optimization where algorithm must learn trade-offs:
- Use cheap hydro now vs save for later
- Risk of low inflow vs cost of thermal
- Storage management across time
- Network flow optimization

### Deterministic vs Stochastic

| Aspect | Deterministic | Stochastic |
|--------|---------------|------------|
| Inflow variance | Zero (σ = 0 or very small) | Non-zero (σ > 0) |
| Scenario branches | 1 per stage | Multiple per stage |
| Convergence | Instant (1 iteration) | Gradual (multiple iterations) |
| Policy | Single optimal decision | Scenario-dependent decisions |
| Simulation std dev | Zero | Non-zero (reflects uncertainty) |

### Convergence and Gap

SDDP provides two metrics of solution quality:

- **Lower bound**: Optimistic estimate of optimal cost (underestimates)
- **Simulated cost**: Realistic estimate from forward simulation

**Gap** = (simulated - lower) / lower × 100%

- **Gap → 0%**: Policy is close to optimal
- **Negative gap**: Early iterations (policy improving)
- **Positive gap**: Later iterations (policy good but still learning)

Typically run until gap < 5% or stabilizes.

### Iteration Process

Each SDDP iteration consists of:

1. **Forward pass**: Simulate scenarios using current policy → Get simulated cost
2. **Backward pass**: Solve subproblems and construct cuts → Improve lower bound
3. **Repeat**: Continue until convergence (gap small and stable)

More iterations = better policy, but diminishing returns after convergence.

## Performance Tips

- **Use `--release` flag**: Release builds are ~10-100× faster than debug
- **Adjust `num_iterations`**: Start with 3-10 for testing, increase to 50-200 for production
- **Adjust `num_branchings`**: More branchings = better uncertainty approximation but slower
- **Parallel execution**: POWE.RS automatically uses multiple threads via Rayon
- **Memory usage**: Minimal allocations through model reuse and basis warm-starting

## Troubleshooting

### Common Issues

**Problem**: `Application error: File not found`  
**Solution**: Check that you're running from project root: `cargo run --release examples/<name>`

**Problem**: `JSON parse error: missing field`  
**Solution**: Check JSON syntax - all required fields must be present. See [Input Specification](../docs/reference/INPUT-SPECIFICATION.md)

**Problem**: `Application error: HiGHS solver failed`  
**Solution**: Check that your problem is feasible. Ensure demand can be met by generators + deficit.

**Problem**: `Gap is not converging`  
**Solution**: Increase `num_iterations` in `config.json`. Some problems need 50-200 iterations.

### Getting Help

- See [Troubleshooting Guide](../docs/guides/TROUBLESHOOTING.md)
- See [Input Specification](../docs/reference/INPUT-SPECIFICATION.md)
- See [SDDP Algorithm Overview](../docs/algorithm/SDDP-OVERVIEW.md)
- Check example README files for specific guidance

## Customizing Examples

Feel free to modify the examples to experiment with different configurations:

- **System changes**: Modify generator capacities, costs, storage in `system.json`
- **Temporal changes**: Modify number of stages, branching structure in `graph.json`
- **Uncertainty changes**: Modify distributions, variance in `recourse.json`
- **Algorithm changes**: Modify iterations, forward passes in `config.json`

After modifying, re-run the example and observe how the optimal policy changes!

## Contributing Examples

Have a interesting problem you'd like to share? Consider contributing:

1. Create example directory: `examples/XX-descriptive-name/`
2. Add all 4 JSON files
3. Create README.md documenting the problem
4. Test with `cargo run --release examples/XX-descriptive-name`
5. Add to `scripts/run_examples.sh`
6. Submit pull request

---

**Next**: Start with [Example 1 - Deterministic 2-Stage](01-deterministic/README.md) →
