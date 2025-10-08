# Example 2: Basic Stochastic Hydrothermal

## Problem Description

This example introduces **stochasticity** (uncertainty) to the hydrothermal dispatch problem. We now have 2 hydroelectric plants and 2 thermal plants, with **stochastic inflows** that vary across scenario branches. This demonstrates how SDDP learns policies that handle uncertainty by building a scenario tree.

## What You'll Learn

- Stochastic optimization with uncertain inflows
- Scenario tree branching (5 branches per stage)
- Risk management across multiple scenarios
- Trade-offs between using hydro now vs conserving for uncertain future
- How SDDP converges across multiple iterations
- Interpreting confidence intervals in simulation results

## Problem Details

### System Configuration

- **Generators**:
  - **Hydro 1**: 40 MW turbining, 80 MWh storage, productivity = 1.0
  - **Hydro 2**: 35 MW turbining, 60 MWh storage, productivity = 1.0
  - **Thermal 1**: 50 MW capacity, $70/MWh cost (cheaper)
  - **Thermal 2**: 45 MW capacity, $90/MWh cost (more expensive)
- **Buses**: 1 bus (all generators connected)
- **Lines**: None
- **Demand**: 80 MW per stage (mean), with small variance (σ = 5 MW)
- **Inflow**: Stochastic with mean ~30 MWh per hydro, σ ~10 MWh

### Temporal Configuration

- **Stages**: 2
- **Scenario Tree**: 
  - Stage 0: 1 node (root)
  - Stage 1: 5 branches (5 possible inflow/demand scenarios)
  - **Total scenarios**: 5 (from stage 0 → stage 1 branching)

### Resource Balance

This problem is balanced to create meaningful optimization under uncertainty:

- **Expected total inflow**: ~60 MWh per stage (30 MWh × 2 hydros)
- **Demand**: 80 MW per stage
- **Hydro capacity**: 75 MW total (40 + 35)
- **Thermal capacity**: 95 MW total (50 + 45)

**Key challenge**: Expected inflow (~60 MWh) is less than demand (80 MWh), but there's uncertainty:
- **High-inflow scenarios**: Abundant water → use mostly hydro
- **Low-inflow scenarios**: Scarce water → use more thermal
- **Storage coupling**: Water saved in stage 0 can be used in stage 1

The algorithm must learn a **robust policy** that:
- Uses hydro aggressively in high-inflow scenarios
- Conserves water in low-inflow scenarios (use thermal instead)
- Balances expected cost across all scenarios

### Expected Policy

**Optimal strategy** (learned by SDDP after a few iterations):

- **Stage 0** (initial decision):
  - Use hydro conservatively (~50-60 MW)
  - Use thermal as needed to meet remaining demand (~20-30 MW)
  - Save some water for stage 1 (hedge against low-inflow scenarios)

- **Stage 1** (scenario-dependent):
  - **High inflow**: Use all available hydro, minimal thermal
  - **Medium inflow**: Mix hydro and cheap thermal
  - **Low inflow**: Deplete storage, use thermal heavily

**Expected cost**: ~$450-550 with uncertainty (±$300-400 std dev due to inflow variability)

## How to Run

```bash
# From the project root directory:
cargo run --release examples/02-stochastic

# Or using the shorthand (if PATH is set):
powers examples/02-stochastic
```

## Expected Output

```
POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'examples/02-stochastic'

# Training
- Iterations: 3
- Forward passes: 2

--------------------------------------------------------------------------------
 iter |   lower ($) |   simul ($) |  gap (%) |  fwd (s) |  bwd (s) | total (s)
--------------------------------------------------------------------------------
    1 |      477.50 |      303.59 |   -36.42 |    0.005 |    0.001 |    0.006
    2 |      477.50 |      454.66 |    -4.78 |    0.000 |    0.001 |    0.001
    3 |      477.50 |      635.69 |    33.13 |    0.000 |    0.001 |    0.001
--------------------------------------------------------------------------------

Training time: 0.01 s

# Simulating
- Scenarios: 20

Expected cost ($): 531.06 +- 354.00

Simulation time: 0.02 s

Total running time: 0.04 s
```

## What's Happening

1. **Iteration 1**: SDDP starts learning. Lower bound = $477.50, but simulated cost = $303.59 (gap = -36.42%). The negative gap indicates the policy is still being refined.

2. **Iteration 2**: Policy improves. Simulated cost increases to $454.66, getting closer to lower bound (gap = -4.78%). The algorithm is learning to handle uncertainty better.

3. **Iteration 3**: Further refinement. Simulated cost = $635.69 (gap = 33.13%). With only 3 iterations, the policy is still converging. More iterations would reduce the gap.

4. **Simulation**: With 20 scenarios, expected cost = $531.06 ± $354.00. The large standard deviation reflects inflow uncertainty:
   - **Best case** (high inflow): ~$180 (mostly hydro)
   - **Worst case** (low inflow): ~$885 (heavy thermal usage)

## Key Differences from Example 1

| Aspect | Example 1 (Deterministic) | Example 2 (Stochastic) |
|--------|---------------------------|------------------------|
| Uncertainty | None (deterministic) | Stochastic inflows |
| Scenario branches | 1 per stage | 5 per stage |
| Convergence | Instant (gap = 0%) | Gradual (multiple iterations) |
| Simulation std dev | 0.00 (no variance) | 354.00 (high variance) |
| Policy | Single optimal decision | Scenario-dependent decisions |
| Learning | No learning needed | Learns across scenarios |

## Key Takeaways

✅ **Stochastic problems require multiple iterations** to learn robust policies  
✅ **Scenario branching** captures uncertainty in inflows and demand  
✅ **Storage acts as a risk buffer** - save water for low-inflow scenarios  
✅ **Policy is scenario-dependent** - different decisions for different inflows  
✅ **Simulation confidence intervals** quantify uncertainty in expected cost  

## Experiments to Try

- **Increase branchings**: Change `num_branchings` from 5 to 10 in `recourse.json` → more scenarios, better approximation of uncertainty
- **Increase iterations**: Change `num_iterations` from 3 to 10 in `config.json` → tighter convergence gap
- **Reduce inflow variance**: Change `sigma` from 0.33 to 0.1 in `recourse.json` → less uncertainty, simpler problem
- **Increase storage**: Change `max_storage` from 80/60 to 150/120 in `system.json` → larger buffers, more flexibility

## Next Steps

- **See next example**: Example 3 - Multi-Stage Hydrothermal (coming soon)  
  Scales to 24 stages (monthly), introduces seasonal patterns and network constraints.

- **Read more**:
  - [SDDP Algorithm Overview](../../docs/algorithm/SDDP-OVERVIEW.md)
  - [Stochastic Process](../../docs/algorithm/SDDP-OVERVIEW.md#stochastic-processes)
  - [Input Specification](../../docs/reference/INPUT-SPECIFICATION.md)

## Files in This Example

- `config.json`: SDDP configuration (3 iterations, 2 forward passes, 20 simulation scenarios)
- `system.json`: Physical system (2 hydros + 2 thermals, capacities and costs)
- `graph.json`: Scenario tree (2 stages, deterministic structure)
- `recourse.json`: Initial conditions and **stochastic distributions** (lognormal inflows, normal demand)
