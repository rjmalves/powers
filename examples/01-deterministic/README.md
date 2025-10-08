# Example 1: Deterministic 2-Stage Hydrothermal

## Problem Description

This is the simplest possible hydrothermal dispatch problem: a single hydroelectric plant and a single thermal plant serving a constant load over 2 time stages (days). The inflow is **deterministic** (no uncertainty), making this problem ideal for verifying your installation and understanding the basic SDDP mechanics.

## What You'll Learn

- Basic SDDP problem structure (stages, nodes, edges)
- JSON file format for system, graph, and recourse
- Trade-off between using cheap hydro now vs saving water for the future
- Deterministic optimization (no uncertainty)
- How to run POWE.RS and interpret output

## Problem Details

### System Configuration

- **Generators**:
  - 1 hydro plant: 30 MW turbining capacity, 50 MWh storage, productivity = 1.0
  - 1 thermal plant: 40 MW capacity, $80/MWh cost
- **Buses**: 1 bus (both generators connected)
- **Lines**: None
- **Demand**: 50 MW constant per stage
- **Inflow**: 20 MWh per stage (deterministic, near-zero variance)

### Temporal Configuration

- **Stages**: 2 (day 1 and day 2)
- **Scenario Tree**: Deterministic (1 branch per stage, no uncertainty)

### Resource Balance

This problem is **carefully balanced** to create a meaningful optimization:

- **Inflow per stage**: 20 MWh
- **Total inflow over 2 stages**: 40 MWh
- **Total demand over 2 stages**: 100 MWh (50 MW × 2 days)
- **Storage capacity**: 50 MWh (can store 1 day's worth of inflow)

**Key insight**: The inflow (40 MWh) is insufficient to meet demand (100 MWh) using hydro alone. The algorithm must decide:
- Use hydro now (cheap) and thermal later?
- Or save hydro for later and use thermal now?
- Or mix both?

With productivity = 1.0, the hydro can generate 1 MWh per 1 MWh of water. The thermal costs $80/MWh, so there's a strong incentive to use hydro whenever possible.

### Expected Policy

**Optimal strategy**: Use all available hydro capacity in both stages (30 MW) and supplement with thermal (20 MW) in both stages. This minimizes thermal usage.

- Stage 1: 30 MW hydro + 20 MW thermal = 50 MW total
- Stage 2: 30 MW hydro + 20 MW thermal = 50 MW total
- **Total cost**: $3,200 (40 MWh thermal × $80/MWh)

## How to Run

```bash
# From the project root directory:
cargo run --release examples/01-deterministic

# Or using the shorthand (if PATH is set):
powers examples/01-deterministic
```

## Expected Output

```
POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'examples/01-deterministic'

# Training
- Iterations: 3
- Forward passes: 2

--------------------------------------------------------------------------------
 iter |   lower ($) |   simul ($) |  gap (%) |  fwd (s) |  bwd (s) | total (s)
--------------------------------------------------------------------------------
    1 |     3200.00 |     3200.00 |     0.00 |    0.003 |    0.001 |    0.004
    2 |     3200.00 |     3200.00 |     0.00 |    0.000 |    0.001 |    0.001
    3 |     3200.00 |     3200.00 |     0.00 |    0.001 |    0.001 |    0.001
--------------------------------------------------------------------------------

Training time: 0.01 s

# Simulating
- Scenarios: 10

Expected cost ($): 3200.00 +- 0.00

Simulation time: 0.01 s

Total running time: 0.01 s
```

## What's Happening

1. **Iteration 1**: SDDP constructs an initial policy. The lower bound (cost-to-go) and simulated cost both start at $3,200.
2. **Gap = 0.00%**: The problem converges immediately because it's deterministic and simple. The algorithm found the optimal policy on the first iteration!
3. **Lower bound = Simulated cost**: Both are $3,200, confirming optimality.
4. **Simulation**: With 10 scenarios, all give the same cost ($3,200 ± 0.00) because the problem is deterministic.

The algorithm learned that using 30 MW hydro + 20 MW thermal in both stages minimizes cost. Any other policy (like saving all water for stage 2) would be more expensive.

## Key Takeaways

✅ **Deterministic problems converge instantly** - no learning needed across scenarios  
✅ **Resource balance matters** - insufficient hydro forces thermal usage  
✅ **Productivity = 1.0** means 1 MWh water → 1 MWh electricity  
✅ **Storage acts as a temporal buffer** - allows shifting water between stages  

## Next Steps

- **Modify the problem**:
  - Try changing inflow to 25 MWh (more water available)
  - Try changing thermal cost to $50/MWh (cheaper thermal)
  - Try changing storage capacity to 100 MWh (larger reservoir)
  - Observe how the optimal policy changes!

- **See next example**: [Example 2 - Basic Stochastic](../02-stochastic/README.md)  
  Introduces uncertainty with stochastic inflows and multiple scenario branches.

- **Read more**:
  - [SDDP Algorithm Overview](../../docs/algorithm/SDDP-OVERVIEW.md)
  - [Input Specification](../../docs/reference/INPUT-SPECIFICATION.md)
  - [Quickstart Guide](../../docs/guides/QUICKSTART.md)

## Files in This Example

- `config.json`: SDDP algorithm configuration (iterations, forward passes, simulation scenarios)
- `system.json`: Physical system (generators, buses, lines)
- `graph.json`: Scenario tree structure (nodes, stages, edges)
- `recourse.json`: Initial conditions and uncertainty distributions
