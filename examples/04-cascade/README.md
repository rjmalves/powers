# Example 4: Hydrothermal Cascade System

## Problem Description

This example demonstrates **cascade operation** in a hydrothermal system with **5 hydroelectric plants linked upstream-to-downstream** over **24 months**. Key features:

- **Cascade topology**: 5 hydros in series (0 → 1 → 2 → 3 → 4)
- **Upstream receives external inflow**: Only hydro 0 gets stochastic natural inflows
- **Downstream receives from upstream**: Each downstream hydro receives turbined water + spillage from its upstream neighbor
- **5 thermal plants**: $60-90/MWh providing backup generation
- **2 buses with transmission line**: 60 MW line capacity, $5/MWh exchange penalty
- **24 stages (monthly)** with seasonal patterns
- **Higher stochasticity**: 20 branchings per season (vs 10 in Example 3)

### System Configuration

**Cascade Structure** (Hydro 0 is most upstream, Hydro 4 is most downstream):

```
Hydro 0 (Bus 0) → Hydro 1 (Bus 0) → Hydro 2 (Bus 0) → Hydro 3 (Bus 1) → Hydro 4 (Bus 1)
  200 MWh          150 MWh          120 MWh          100 MWh          80 MWh
  50 MW            40 MW            35 MW            30 MW            25 MW
```

- **Total storage**: 650 MWh (largest upstream, smallest downstream)
- **Total turbining capacity**: 180 MW
- **Spillage penalty**: $1/MWh (discourages waste)

**Thermal Backup**:
- **Bus 0**: 2 thermals (70 MW total, $60-70/MWh)
- **Bus 1**: 3 thermals (90 MW total, $65-90/MWh)
- **Total thermal**: 160 MW capacity

**Transmission**:
- Line connects Bus 0 and Bus 1
- 60 MW bidirectional capacity
- $5/MWh exchange penalty

**Demand**:
- **Summer** (Dec-Feb): 190 MW
- **Winter** (Jun-Aug): 110 MW
- **Average**: ~150 MW

### Resource Balance

- **Total capacity**: 180 MW (hydro) + 160 MW (thermal) = 340 MW
- **Peak demand**: 190 MW
- **Over-resourced** to demonstrate:
  1. Cascade coordination without resource scarcity
  2. Trade-offs between storage, spillage, and thermal usage
  3. Value of proper cascade management

## What You'll Learn

### 1. Cascade Dynamics

**Upstream-Downstream Dependencies**:
- **Upstream release = Downstream inflow**: Water turbined or spilled at hydro N becomes inflow to hydro N+1
- **Coordination challenge**: Releasing upstream water helps downstream but depletes upstream storage
- **Spillage propagation**: Spillage at any hydro flows downstream (wasted energy if not captured)

**Key Insight**: The cascade creates **spatial coupling** in addition to temporal coupling:
- Temporal: Today's storage affects tomorrow's costs
- Spatial: Upstream decisions affect downstream feasibility and costs

### 2. External vs Internal Inflows

**External inflows** (from watershed):
- Only **Hydro 0** receives stochastic external inflows (~25-60 MWh/month seasonally)
- Uncertainty enters the system only at the most upstream point

**Internal inflows** (from cascade):
- **Hydros 1-4** receive deterministic inflows equal to upstream turbining + spillage
- No external uncertainty for downstream hydros

**Implication**: 
- Upstream decisions under uncertainty propagate deterministically downstream
- Uncertainty is "absorbed" by upstream storage management

### 3. Spillage Management

**Spillage penalty** ($1/MWh):
- **Purpose**: Discourage wasting water when storage is full
- **Trade-off**: Sometimes spillage is unavoidable (storage full, low demand)
- **Cascade effect**: Spillage can benefit downstream if they have capacity to store/use it

**Optimal strategy**:
- Avoid spillage at upstream (wastes potential energy)
- If spillage necessary, ensure downstream can capture it
- Spillage most likely during wet season + low demand periods

### 4. Seasonal Strategy in Cascade

**Wet season** (Nov-Mar):
- **Upstream (Hydro 0)**: Receives high inflows (~60 MWh/month)
  - Accumulate storage if possible
  - Release strategically to avoid spillage
- **Downstream**: Benefit from upstream releases
  - Can also accumulate storage
  - Less thermal generation needed

**Dry season** (Apr-Oct):
- **Upstream**: Receives low inflows (~25 MWh/month)
  - Must deplete storage strategically
  - Balance own needs vs downstream support
- **Downstream**: Rely on upstream releases + own storage
  - Coordinate storage depletion
  - More thermal generation needed

### 5. Multi-Bus Operation with Cascade

**Spatial distribution**:
- **Bus 0**: Hydros 0-2 (upstream/midstream), 70 MW thermal
- **Bus 1**: Hydros 3-4 (downstream), 90 MW thermal

**Network considerations**:
- Cascade crosses buses (Hydro 2 on Bus 0 → Hydro 3 on Bus 1)
- Water flows through cascade regardless of transmission constraints
- Power generation location matters for network constraints

**Optimization challenge**: 
- Where to generate power (upstream vs downstream)?
- When to use transmission vs local generation?
- How to coordinate hydro releases with network limits?

## Expected Output

With 3 iterations (quick test):

```bash
cargo run --release examples/04-cascade
```

Expected results:

```
# Training
- Iterations: 3
- Forward passes: 2

iter |   lower ($) |   simul ($) |  gap (%) |  fwd (s) |  bwd (s) | total (s)
   1 |   108618.27 |   135627.17 |    24.87 |    0.007 |    0.044 |    0.051
   2 |   119965.09 |   150663.19 |    25.59 |    0.003 |    0.048 |    0.051
   3 |   128156.36 |   149296.24 |    16.50 |    0.002 |    0.046 |    0.048

Training time: 0.16 s

Number of constructed cuts by node: 6

# Simulating
- Scenarios: 30

Expected cost ($): 135967.24 +- 4126.38

Simulation time: 0.10 s

Total running time: 0.26 s
```

### Interpreting Results

**Costs are much higher than Example 3** ($135k vs $300):
- **Why?** The cascade example has **higher spillage penalties** ($1/MWh vs $0.01/MWh)
- Spillage is heavily penalized, forcing more conservative storage management
- This demonstrates the cost of spillage avoidance in cascade systems

**Lower Bound**: $128k (iteration 3)
- Increasing across iterations (good convergence sign)
- Represents minimum expected cost with perfect information

**Simulation Cost**: $135k ± $4k
- Lower standard deviation than Example 3 (due to cascade buffering uncertainty)
- Cascade "smooths" uncertainty as it propagates downstream

**Gap**: 16.5% (iteration 3)
- Decreasing from 24.9% (iteration 1) → good progress
- Would decrease further with more iterations

**Solver Warnings**: "HiGHS emitted a warning: Highs_addRow"
- **Normal behavior** when adding many cuts dynamically
- Does not affect correctness or optimality

## Convergence Behavior

With **3 iterations**:
- Rapid lower bound improvement (108k → 128k)
- Gap reduces from 25% to 16%
- Fast execution (0.26 seconds)

To achieve better convergence:

```json
{
    "num_iterations": 50,
    "num_forward_passes": 5,
    "num_simulation_scenarios": 100,
    "seed": 42,
    "output_path": "./examples/04-cascade"
}
```

Expected with 50 iterations:
- Lower bound → ~$133-135k
- Simulation cost → ~$135-137k
- Gap → <3%
- Running time → 3-6 seconds

## Experiments to Try

### Experiment 1: Remove Spillage Penalty
**Objective**: Understand spillage cost impact

1. Edit `system.json`: Change all `spillage_penalty` from 1.0 to 0.01
2. Run the example

**What to learn**:
- Costs drop dramatically (from $135k to ~$300-400)
- More spillage occurs (especially in wet season)
- Demonstrates that spillage penalty dominates costs in this example

### Experiment 2: Break the Cascade
**Objective**: Compare cascade vs independent operation

1. Edit `system.json`: Set all `downstream_hydro_id` to `null`
2. This makes all hydros independent (no cascade)
3. Add external inflow to all hydros in `recourse.json`
4. Run the example

**What to learn**:
- Each hydro operates independently
- No benefit from upstream releases for downstream
- Total system performance may degrade (less coordination)

### Experiment 3: Increase Upstream Storage
**Objective**: Analyze value of upstream storage capacity

1. Edit `system.json`: Increase Hydro 0 `max_storage` from 200 to 400
2. Run the example

**What to learn**:
- More upstream storage → better wet season accumulation
- Can save more water for dry season
- Costs may decrease (less thermal usage in dry season)
- Demonstrates value of upstream reservoir capacity

### Experiment 4: Add More Branchings
**Objective**: Test stochastic convergence with higher uncertainty resolution

1. Edit `recourse.json`: Change `num_branchings` from 20 to 50
2. Run the example (may take longer)

**What to learn**:
- More branchings → better approximation of uncertainty
- Lower bound may increase (tighter approximation)
- Convergence may be slower (more scenarios to explore)
- Trade-off between accuracy and computational time

### Experiment 5: Seasonal Storage Target
**Objective**: Understand optimal storage positioning

1. Run the example with 50 iterations for better convergence
2. Check output files: `states.csv` to see storage trajectories
3. **Observe**: How storage levels evolve across wet/dry seasons

**What to learn**:
- Upstream fills during wet season, depletes in dry season
- Downstream storage levels depend on upstream strategy
- End-of-horizon effects (may drain storage toward final stage)

## Key Takeaways

1. **Cascade coupling is complex**:
   - Upstream decisions affect all downstream hydros
   - Spillage at one hydro can benefit downstream
   - Coordination is essential for optimal operation

2. **Uncertainty propagation**:
   - External uncertainty enters at upstream
   - Propagates deterministically downstream through releases
   - Downstream hydros have less direct exposure to inflow uncertainty

3. **Spillage penalty matters greatly**:
   - High penalty → conservative storage management → higher thermal costs
   - Low penalty → more spillage → lower total costs (if thermal is expensive)
   - Reflects real-world value of water vs cost of generation alternatives

4. **Storage positioning is strategic**:
   - Larger storage upstream provides system-wide flexibility
   - Downstream storage can buffer upstream release variability
   - Optimal cascade uses all storage levels coordinately

5. **Network topology interacts with cascade**:
   - Cascade water flows independent of transmission constraints
   - Power generation location must respect network limits
   - Optimal strategy coordinates water flows and power flows

## Problem Structure

- **Scenario tree**: 24 nodes (monthly stages 0-23)
- **Stochastic branchings**: 20 realizations per season
- **State variables**: Storage levels for 5 hydros (5-dimensional state)
- **Decisions**: Turbining, spillage, thermal generation, line flow, deficit
- **Uncertainties**: External inflow for Hydro 0 (lognormal), demand (normal)
- **Constraints**: Storage limits, turbining limits, cascade mass balance, line capacity

## File Descriptions

- **config.json**: Algorithm configuration (3 iterations for quick testing)
- **system.json**: Cascade topology (5 hydros with downstream links), thermal plants, line
- **graph.json**: 24 monthly stages with repeating seasonal pattern
- **recourse.json**: Initial conditions + seasonal distributions (only Hydro 0 has external inflow)

## Next Steps

After understanding cascade dynamics:
1. Try **Example 5 - Large-Scale System**: Brazilian-scale complexity (simplified)
2. Experiment with different cascade topologies
3. Add more hydros to create larger cascades
4. Test different spillage penalties to understand trade-offs
5. Analyze storage trajectories to understand optimal cascade management

---

**Estimated time**: 10-15 minutes to run and experiment  
**Difficulty**: Advanced  
**Prerequisites**: Examples 1-3, understanding of multi-stage and stochastic SDDP
