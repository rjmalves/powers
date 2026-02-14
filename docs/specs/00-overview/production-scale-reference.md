---
status: draft
review_priority: 3-medium
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §2 (2.1-2.3)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Production Scale Reference

## Purpose

This spec defines the production-scale dimensions of the POWE.RS SDDP solver: system sizes, LP variable and constraint counts, state dimension formulas, performance expectations by scale, and the sizing calculator tool. It serves as the reference for capacity planning, memory budgeting, and performance regression detection.

## 1. Production Scale Dimensions

Based on the target production scenario:

| Dimension            | Value                      | Memory Impact                          |
| -------------------- | -------------------------- | -------------------------------------- |
| Stages               | 120                        | Graph size                             |
| Blocks per Stage     | 1-24 (varies), typically 3 | LP structure, outputs                  |
| Hydros               | 160                        | State dimension                        |
| Max AR Order         | 12                         | State dimension, Variables/Constraints |
| Thermals             | 130                        | Variables                              |
| Buses                | 6                          | Variables/Constraints                  |
| Lines                | 10                         | Variables/Constraints                  |
| Forward Passes       | 200                        | Parallelism                            |
| Iterations           | 50                         | Cut pool size                          |
| Scenarios per Node   | 20                         | Branching                              |
| Simulation Scenarios | 2000                       | Output size                            |

## 2. State Dimension Estimates

### 2.1 State Variables and Dimension

The **state dimension** determines the size of Benders cuts. State variables include:

| Component     | Count                | Description                                  |
| ------------- | -------------------- | -------------------------------------------- |
| Storage       | $\mathcal{H}$        | End-of-stage reservoir volume for each hydro |
| AR Lags       | $\sum_{h} P_h$       | Inflow lag values for AR(P) models           |
| Battery SOC   | $\mathcal{BAT}$      | Battery state of charge (if batteries exist) |
| GNL Committed | $\sum_{gnl} L_{gnl}$ | GNL dispatch pipeline (if GNL exists)        |

**State Dimension Formula**:

$$
N_{state} = N_{hydro} + \sum_{h=1}^{N_{hydro}} P_h + N_{battery} + \sum_{gnl} L_{gnl}
$$

For production scale (160 hydros, AR order up to 12):

- Storage: 160
- AR lags: $160 \times 12 = 1920$ (worst case, all hydros use max order)
- Total: up to 2080

> **Note**: The actual state dimension depends on the AR orders specified in `inflow_models.parquet`. If most hydros use AR(6), the dimension would be $160 + 160 \times 12 = 1120$.

![State Variable Composition](../../diagrams/exports/svg/data/state-variables.svg)

## 3. Variable and Constraint Counts

### 3.1 Variable Count per Subproblem

| Component                   | Formula                                             | Typical Count       |
| --------------------------- | --------------------------------------------------- | ------------------- |
| Future cost                 | $1$                                                 | 1                   |
| Deficit                     | $N_{bus} \times N_{block} \times N_{seg}$           | 6 × 3 × 3 = 54      |
| Excess                      | $N_{bus} \times N_{block}$                          | 6 × 3 = 18          |
| Exchange (direct + reverse) | $2 \times N_{line} \times N_{block}$                | 2 × 10 × 3 = 60     |
| Hydro storage               | $N_{hydro}$                                         | 160                 |
| Hydro incremental inflow AR | $N_{hydro} \times P$                                | 160 x 12 = 1920     |
| Hydro turbined flow         | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro spillage              | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro generation            | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro diversion             | $N_{div} \times N_{block}$                          | ~10 × 3 = 30        |
| Hydro evaporation           | $N_{evap} \times N_{block}$                         | ~50 × 3 = 150       |
| Hydro withdrawal            | $N_{withdrawal} \times N_{block}$                   | ~20 × 3 = 60        |
| Hydro slacks                | $N_{hydro} \times N_{block} \times 6$               | 160 × 3 × 6 = 2880  |
| Thermal generation          | $N_{thermal} \times N_{block} \times \bar{N}_{seg}$ | 130 × 3 × 1.5 = 585 |
| Contracts                   | $(N_{imp} + N_{exp}) \times N_{block}$              | 5 × 3 = 15          |
| Pumping                     | $N_{pump} \times N_{block}$                         | 5 × 3 = 15          |
| **Total Variables**         |                                                     | **~7,500**          |

### 3.2 Constraint Count per Subproblem

| Component                        | Formula                                             | Typical Count      |
| -------------------------------- | --------------------------------------------------- | ------------------ |
| Load balance                     | $N_{bus} \times N_{block}$                          | 6 × 3 = 18         |
| Hydro water balance              | $N_{hydro}$                                         | 160                |
| Incremental inflow AR dynamics   | $N_{hydro}$                                         | 160                |
| Lagged incremental inflow fixing | $N_{hydro} \times P$                                | 160 x 12 = 1920    |
| Hydro generation (constant)      | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480      |
| Hydro generation (FPHA)          | $N_{fpha} \times N_{block} \times \bar{M}_{planes}$ | 50 × 3 × 10 = 1500 |
| Outflow definition               | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480      |
| Outflow bounds (min/max)         | $2 \times N_{hydro} \times N_{block}$               | 2 × 160 × 3 = 960  |
| Turbined min                     | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480      |
| Generation min                   | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480      |
| Evaporation                      | $N_{evap} \times N_{block}$                         | 50 × 3 = 150       |
| Water withdrawal                 | $N_{withdrawal} \times N_{block}$                   | 20 × 3 = 60        |
| Generic constraints              | $N_{generic}$                                       | ~50                |
| **Benders cuts (pre-allocated)** | $N_{cuts}$                                          | 10,000–15,000      |
| **Total Constraints**            |                                                     | **~17,000–22,000** |

> **Note**: The constraint count is dominated by pre-allocated Benders cut slots. During early iterations, most cut constraints are inactive (bounds set to $[-\infty, +\infty]$).

### 3.3 Counting Formulas (Exact)

For precise sizing, use the following formulas where parameters come from the configuration:

**Variables**:

```
N_VAR = 1                                                      # theta
      + N_BUS × N_BLOCK × (AVG_DEF_SEGMENTS + 1)              # deficit + excess
      + 2 × N_LINE × N_BLOCK                                   # exchange
      + N_HYDRO                                                # storage
      + N_HYDRO x AR_ORDER                                     # incremental inflow model
      + N_HYDRO × N_BLOCK × 4                                  # q, s, g, inflow
      + N_HYDRO_DIV × N_BLOCK                                  # diversion
      + N_HYDRO_EVAP × N_BLOCK                                 # evaporation
      + N_HYDRO_WITHDRAWAL × N_BLOCK                           # withdrawal
      + N_HYDRO × N_BLOCK × N_SLACK_TYPES                      # slack vars
      + N_THERMAL × N_BLOCK × AVG_COST_SEGMENTS                # thermal
      + (N_CONTRACT_IMP + N_CONTRACT_EXP) × N_BLOCK            # contracts
      + N_PUMP × N_BLOCK × 2                                   # pump flow + power
```

**Constraints**:

```
N_CON = N_BUS × N_BLOCK                                        # load balance
      + N_HYDRO                                                # water balance
      + N_HYDRO                                                # inflow AR dynamics
      + N_HYDRO x AR_ORDER                                     # lagged inflow fixing
      + N_HYDRO × N_BLOCK                                      # generation (constant)
      + N_HYDRO_FPHA × N_BLOCK × AVG_FPHA_PLANES               # FPHA (additional)
      + N_HYDRO × N_BLOCK                                      # outflow definition
      + N_HYDRO × N_BLOCK × 2                                  # outflow bounds
      + N_HYDRO × N_BLOCK                                      # turbined min
      + N_HYDRO × N_BLOCK                                      # generation min
      + N_HYDRO_EVAP × N_BLOCK                                 # evaporation
      + N_HYDRO_WITHDRAWAL × N_BLOCK                           # withdrawal
      + N_GENERIC                                              # generic constraints
      + N_CUT_CAPACITY                                         # Benders cuts
```

![LP Variable and Constraint Sizing](../../diagrams/exports/svg/data/lp-sizing.svg)

**State Dimension**:

```
N_STATE = N_HYDRO                                              # storage
        + SUM(AR_ORDER[h] for h in HYDROS)                     # AR lags
        + N_BATTERY                                            # SOC
        + SUM(GNL_LAG[t] for t in GNL_THERMALS)               # GNL pipeline
```

### 3.4 Sizing Calculator Tool

A Python script is provided to calculate LP dimensions from a JSON configuration:

```bash
# Calculate sizes for production configuration
python scripts/lp_sizing.py scripts/lp_sizing_production.json

# Interactive mode with prompts
python scripts/lp_sizing.py --interactive

# Output as JSON for programmatic use
python scripts/lp_sizing.py scripts/lp_sizing_production.json --json
```

The script outputs:

- Variable counts by category
- Constraint counts by type
- State dimension breakdown
- Memory estimates (LP matrix, cut storage, solver workspace)

See `scripts/lp_sizing.py` for the implementation and `scripts/lp_sizing_production.json` for a production-scale configuration example.

## 4. Performance Expectations by Scale

> **Purpose**: This table provides expected timing targets for different problem scales, enabling performance validation and regression detection. Timings are per-iteration unless otherwise noted.

> **Note**: Memory requirements are old placeholder values and need revision with the `lp_sizing` tool.

### 4.1 Hardware Assumptions

| Component | Specification                                         |
| --------- | ----------------------------------------------------- |
| CPU       | AMD EPYC 9R14 or equivalent (192 cores, 3.7 GHz base) |
| Memory    | DDR5, 384 GB/node                                     |
| Network   | InfiniBand HDR (200 Gb/s) or equivalent               |
| Storage   | NVMe SSD for I/O operations                           |

### 4.2 Test Systems

| Scale          | Stages | Hydros | Thermals | Inflow AR Order | Scenarios | Ranks | Threads/Rank | Forward Time | Backward Time | Memory/Rank |
| -------------- | ------ | ------ | -------- | --------------- | --------- | ----- | ------------ | ------------ | ------------- | ----------- |
| **Unit Test**  | 3      | 1      | 2        | 0               | 2         | 1     | 1            | <0.1s        | <1s           | <20 MB      |
| **Small**      | 6      | 1      | 2        | 1               | 10        | 1     | 2            | <0.2s        | <2s           | <50 MB      |
| **Medium**     | 12     | 80     | 1        | 1               | 100       | 4     | 12           | <5s          | <15s          | <2 GB       |
| **Large**      | 24     | 160    | 1        | 1               | 200       | 16    | 16           | <15s         | <45s          | <6 GB       |
| **Production** | 120    | 1      | 1        | 160             | 2000      | 200   | 16           | <30s         | <90s          | <20 GB      |

### 4.3 Key Performance Indicators

| Metric                | Target  | Measurement Point               |
| --------------------- | ------- | ------------------------------- |
| LP solve (warm-start) | <2 ms   | Hot-path, 500-row problem       |
| LP solve (cold-start) | <20 ms  | First solve or basis invalid    |
| RHS batch update      | <100 μs | 500 constraint updates          |
| Solution extraction   | <50 μs  | Primal + basis to buffers       |
| Cut broadcast         | <5 ms   | 1000 cuts × 2080 coefficients   |
| Parallel efficiency   | >80%    | At 128 ranks vs 1 rank          |
| Warm-start hit rate   | >70%    | Forward pass consecutive stages |

### 4.4 Scaling Expectations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPECTED SCALING BEHAVIOR                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Forward Pass:                                                              │
│    Time ∝ (stages × LP_solve_time) / (ranks × threads)                     │
│    Near-linear speedup expected up to scenarios/2 threads                   │
│                                                                             │
│  Backward Pass:                                                             │
│    Time ∝ stages × (branch_solves / threads + sync_overhead)               │
│    Sequential stage dependency limits parallelism                           │
│    Communication overhead: ~5-10% at 64 ranks, ~15-20% at 256 ranks        │
│                                                                             │
│  Memory:                                                                    │
│    Per-rank ∝ cuts_per_stage × state_dim × 8 + solver_instances × 15MB     │
│    Shared memory (MPI windows) reduces replication within node             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Convergence Reference

| Problem Type                       | Typical Iterations | Optimality Gap  |
| ---------------------------------- | ------------------ | --------------- |
| Simple (few hydros, short horizon) | 10-20              | <0.1%           |
| Medium (regional system)           | 30-50              | <0.5%           |
| Complex (full national grid)       | 50-100             | <1.0%           |
| With CVaR risk measure             | +20-50% iterations | Same gap target |

> **Note**: Iteration counts assume reasonable initial policy (warm-start from previous study). Cold-start may require 2-3× more iterations.

## Cross-References

- [Design Principles](./design-principles.md) — Format selection criteria and design goals
- [Notation Conventions](./notation-conventions.md) — Mathematical symbols used in formulas above
- [LP Formulation](../01-math/lp-formulation.md) — Complete LP subproblem that these dimensions describe
- [SDDP Algorithm](../01-math/sddp-algorithm.md) — Forward/backward pass structure driving performance targets
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — MPI+OpenMP architecture for achieving these scaling targets
- [Memory Architecture](../04-hpc/memory-architecture.md) — Memory budget and NUMA-aware allocation
- [SLURM Deployment](../04-hpc/slurm-deployment.md) — Job scripts for the test system scales above
