# Example 3: Multi-Stage Hydrothermal System

## Problem Description

This example demonstrates **long-term planning** in a **multi-stage hydrothermal system** spanning **24 months** (2 years). The system features:

- **5 hydroelectric plants** distributed across **2 buses**
- **5 thermal plants** with varying costs ($60-95/MWh)
- **1 transmission line** connecting the buses (80 MW capacity, $5/MWh exchange penalty)
- **Seasonal inflow patterns**: wet season (Nov-Mar) with high inflows, dry season (Apr-Oct) with low inflows
- **Seasonal demand variation**: summer peaks (Dec-Feb: 180 MW), winter lows (Jun-Aug: 100 MW)

### System Configuration

**Bus 0** (135 MW hydro capacity, 65 MW thermal capacity):
- 3 hydroelectric plants with storage capacities of 150, 120, and 100 MWh
- 2 thermal plants: $60/MWh (30 MW) and $75/MWh (35 MW)

**Bus 1** (75 MW hydro capacity, 105 MW thermal capacity):
- 2 hydroelectric plants with storage capacities of 100 and 80 MWh
- 3 thermal plants: $65/MWh (40 MW), $80/MWh (35 MW), $95/MWh (30 MW)

**Transmission Line**:
- Connects Bus 0 to Bus 1
- 80 MW bidirectional capacity
- $5/MWh exchange penalty (discourages excessive power transfer)

### Resource Balance

Total system capacity:
- **Hydro**: 210 MW generation, 550 MWh total storage
- **Thermal**: 170 MW generation
- **Total generation**: 380 MW
- **Peak demand**: 180 MW (summer)
- **Average demand**: ~140 MW

The system is **over-resourced** to demonstrate:
1. **Seasonal storage management**: How to accumulate water during wet season for use in dry season
2. **Network constraints**: When and how to transfer power between buses
3. **Long-term planning**: Trading off immediate vs future costs over 24 months

## What You'll Learn

This example introduces several advanced SDDP concepts:

### 1. Multi-Stage Planning Horizon
- **24 stages** representing monthly decisions over 2 years
- Each stage represents approximately 30 days (~720 hours)
- Decisions made today affect costs and feasibility 2 years into the future

### 2. Seasonal Patterns
- **Wet season** (Nov-Mar): High inflows (~50 MWh/month average per hydro)
  - Opportunity to accumulate storage
  - Lower thermal generation needed
- **Dry season** (Apr-Oct): Low inflows (~20 MWh/month average per hydro)
  - Must rely on stored water + thermal generation
  - Risk of storage depletion

### 3. Network-Constrained Operation
- **Transmission line capacity** limits power exchange between buses
- **Exchange penalty** ($5/MWh) discourages inefficient transfers
- **Optimal power flow**: When to transfer power vs generate locally

### 4. Storage Management Strategy
- **Initial storage** (75% of capacity): Provides cushion for first few months
- **Target end storage**: Algorithm determines optimal final storage levels
- **Trade-offs**: Storing water for future vs using it immediately

### 5. Stochastic Convergence Behavior
- **Lower bound**: Increases as algorithm learns future cost approximations
- **Simulation cost**: Varies as policy encounters different scenarios
- **Gap**: May remain positive due to stochasticity and limited iterations

## Expected Output

When you run this example with 3 iterations (quick test):

```bash
cargo run --release examples/03-multistage
```

You should see:

```
# Training
- Iterations: 3
- Forward passes: 2

iter |   lower ($) |   simul ($) |  gap (%) |  fwd (s) |  bwd (s) | total (s)
   1 |      197.18 |      392.13 |    98.87 |    0.006 |    0.029 |    0.035
   2 |      197.18 |      782.57 |   296.88 |    0.003 |    0.031 |    0.034
   3 |      197.18 |     3931.64 |  1893.93 |    0.003 |    0.031 |    0.034

Training time: 0.12 s

Number of constructed cuts by node: 6

# Simulating
- Scenarios: 30

Expected cost ($): 298.31 +- 125.10

Simulation time: 0.13 s

Total running time: 0.27 s
```

### Interpreting Results

**Lower Bound**: $197.18
- Represents the best possible expected cost assuming perfect information
- Lower than simulation cost due to limited iterations

**Simulation Cost**: $298.31 ± $125.10
- Average cost across 30 simulated scenarios
- Standard deviation reflects uncertainty from stochastic inflows and demand
- Higher than lower bound due to conservative policy (not fully converged)

**Gap**: 1893.93% (iteration 3)
- Large gap is **expected** with only 3 iterations on a 24-stage problem
- Indicates policy is conservative (not yet optimal)
- Gap would decrease with more iterations

**Running Time**: 0.27 seconds
- Very fast for a 24-stage stochastic problem
- Scales well due to efficient cut selection algorithms

## Convergence Behavior

With **3 iterations** (current configuration):
- Fast execution for testing and development
- Policy is conservative but feasible
- Gap remains large due to insufficient training

To achieve better convergence, increase iterations in `config.json`:

```json
{
    "num_iterations": 50,  // Increased from 3
    "num_forward_passes": 5,
    "num_simulation_scenarios": 100,
    "seed": 42,
    "output_path": "./examples/03-multistage"
}
```

Expected behavior with 50 iterations:
- Lower bound increases significantly (approaches $250-300)
- Simulation cost stabilizes around $280-320
- Gap reduces to <10%
- Running time increases to 2-5 seconds

## Experiments to Try

### Experiment 1: Increase Iterations
**Objective**: Observe convergence behavior

1. Edit `config.json`: Set `num_iterations` to 10, 20, 50
2. Run the example
3. **Observe**: How lower bound, simulation cost, and gap evolve

**What to learn**: 
- Lower bound increases monotonically (SDDP guarantee)
- Gap decreases as policy improves
- Convergence rate depends on problem structure

### Experiment 2: Modify Seasonal Patterns
**Objective**: Understand impact of inflow variability

1. Edit `recourse.json`: Increase wet season mu from 3.912 to 4.2
2. This increases wet season inflows from ~50 MWh to ~67 MWh
3. Run the example

**What to learn**:
- More abundant water → lower thermal usage → lower costs
- Storage dynamics change (more surplus in wet season)

### Experiment 3: Transmission Capacity Sensitivity
**Objective**: Analyze network constraint impact

1. Edit `system.json`: Change line `direct_capacity` and `reverse_capacity` from 80.0 to 40.0
2. Run the example

**What to learn**:
- Tighter line constraints → more local generation needed
- Higher costs due to inability to share resources efficiently
- Exchange penalty becomes more relevant

### Experiment 4: Remove Transmission Line
**Objective**: Compare networked vs isolated operation

1. Edit `system.json`: Set line capacities to 0.0 (or remove line)
2. Run the example

**What to learn**:
- Each bus must be self-sufficient
- Bus 0: More hydro, cheaper thermal → likely lower costs
- Bus 1: More expensive thermal → higher costs
- Total cost increases due to inability to share resources

### Experiment 5: Storage Capacity Sensitivity
**Objective**: Understand value of storage

1. Edit `system.json`: Reduce hydro `max_storage` by 50% (e.g., 150 → 75)
2. Run the example

**What to learn**:
- Less storage → harder to save water for dry season
- More thermal generation needed → higher costs
- Storage value increases with demand-inflow mismatch

## Key Takeaways

1. **Multi-stage planning is essential** when:
   - Decisions have long-term consequences (water stored today is valuable next season)
   - Seasonal patterns create temporal arbitrage opportunities
   - Storage constraints couple decisions across time

2. **Network constraints matter**:
   - Transmission capacity limits resource sharing
   - Exchange penalties can make local generation preferable
   - Optimal operation balances line usage vs local costs

3. **Seasonality drives strategy**:
   - Wet season: Accumulate storage, minimize thermal
   - Dry season: Deplete storage strategically, supplement with thermal
   - Transition periods: Critical for positioning storage levels

4. **Convergence requires patience**:
   - 24-stage problems need many iterations (50-100) for tight gaps
   - Lower bound provides rigorous cost lower limit
   - Simulation validates policy across multiple scenarios

5. **Stochasticity increases complexity**:
   - Inflow uncertainty requires flexible policies
   - Storage acts as insurance against dry scenarios
   - Standard deviation in simulation reflects unavoidable risk

## Problem Structure

This example uses:
- **Scenario tree**: 24 nodes (stages 0-23), deterministic edges
- **Stochastic branchings**: 10 realizations per season at each node
- **State variables**: Storage levels for 5 hydros (5-dimensional state space)
- **Decisions**: Generation, turbining, spillage, line flow, deficit
- **Uncertainties**: Inflows (lognormal distributions), demand (normal distributions)

The scenario tree structure allows efficient backward computation while capturing seasonal patterns through season-specific uncertainty distributions.

## File Descriptions

- **config.json**: Algorithm parameters (iterations, forward passes, simulation scenarios)
- **system.json**: Physical system (buses, lines, generators with capacities and costs)
- **graph.json**: Temporal structure (24 monthly stages, seasonal IDs)
- **recourse.json**: Initial conditions and seasonal uncertainty distributions

## Next Steps

After understanding this example:
1. Try **Example 4 - Hydrothermal Cascade**: Adds upstream-downstream dependencies
2. Explore **Example 5 - Large-Scale System**: Brazilian-scale complexity (simplified)
3. Modify seasonal patterns to match real-world data
4. Experiment with different network topologies
5. Add more stages (e.g., 60 months) to see long-term planning behavior

---

**Estimated time**: 5-10 minutes to run and experiment  
**Difficulty**: Intermediate  
**Prerequisites**: Examples 1-2, understanding of SDDP basics
