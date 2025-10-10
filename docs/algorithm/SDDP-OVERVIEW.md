# SDDP Algorithm Overview

**Stochastic Dual Dynamic Programming for Hydrothermal Dispatch**

This document provides an overview of the SDDP algorithm as implemented in POWE.RS, covering the mathematical background, implementation approach, and practical considerations.

## Table of Contents

- [What is SDDP?](#what-is-sddp)
- [The Hydrothermal Dispatch Problem](#the-hydrothermal-dispatch-problem)
- [SDDP Algorithm](#sddp-algorithm)
- [Convergence Properties](#convergence-properties)
- [Implementation in POWE.RS](#implementation-in-powers)
- [References](#references)

---

## What is SDDP?

**Stochastic Dual Dynamic Programming (SDDP)** is an algorithm for solving multistage stochastic optimization problems. It was originally proposed by Pereira and Pinto (1991) for hydrothermal scheduling in power systems.

### Key Idea

SDDP approximates the value function (expected future cost) using **Benders cuts** (linear constraints). Through iterative **forward** and **backward** passes, it refines this approximation until convergence.

### Why SDDP for Hydrothermal Dispatch?

**Problem characteristics**:

- **Multistage**: Decisions span multiple time periods (months, years)
- **Stochastic**: Uncertainty in hydro inflows, demand, etc.
- **Large-scale**: Many reservoirs, buses, scenarios → millions of variables
- **Nonlinear coupling**: Water value depends on future decisions

**Traditional approaches fail**:

- **Deterministic**: Ignores uncertainty → poor operational decisions
- **Scenario tree**: Exponential growth → computationally intractable
- **Stochastic Programming**: Too large for direct solution methods

**SDDP advantages**:

- ✅ Handles uncertainty via sampling
- ✅ Decomposes problem by stages (tractable subproblems)
- ✅ Provable convergence guarantees
- ✅ Scalable to real-world systems (100+ reservoirs, 60+ stages)

---

## The Hydrothermal Dispatch Problem

### Problem Statement

**Objective**: Minimize expected cost of meeting electricity demand over a planning horizon.

**Resources**:

- **Hydro**: Cheap ($0) but limited by storage and stochastic inflows
- **Thermal**: Expensive ($/MWh) but always available
- **Deficit**: Very expensive penalty ($/MWh) for unmet demand

**Trade-off**:

- Use hydro now → risk running out later (expensive thermal or deficit)
- Save hydro → insurance against low inflows, but use thermal now

**Uncertainty**: Hydro inflows are random (unknown at decision time).

### Mathematical Formulation

**Notation**:

- $t$ = stage (time period: month, week, etc.)
- $x_t$ = state (reservoir storage at stage $t$)
- $u_t$ = decision (hydro generation, thermal generation)
- $\xi_t$ = uncertainty (inflows)
- $c_t(x_t, u_t)$ = immediate cost at stage $t$

**Objective**:

$$
\min \mathbb{E}\left[\sum_{t=0}^{T-1} c_t(x_t, u_t)\right]
$$

**Subject to**:

- Power balance: $\text{hydro} + \text{thermal} + \text{deficit} = \text{demand}$
- Water balance: $x_{t+1} = x_t + \xi_t - u_t^\text{hydro}$
- Capacity limits: $0 \le u_t^\text{hydro} \le \bar{u}$, $0 \le x_t \le \bar{x}$

**Bellman's Principle of Optimality**:

$$
V_t(x_t) = \min_{u_t} \left\{ c_t(x_t, u_t) + \mathbb{E}_{\xi_{t+1}}[V_{t+1}(x_{t+1})] \right\}
$$

where $V_t(x_t)$ is the **value function** (optimal expected cost-to-go from state $x_t$).

**Challenge**: $V_t(x_t)$ is unknown and nonlinear → SDDP approximates it piecewise-linearly.

---

## SDDP Algorithm

SDDP alternates between two phases:

1. **Forward Pass**: Simulate decisions under current policy, visiting states
2. **Backward Pass**: Refine policy by adding Benders cuts at visited states

### Forward Pass

**Goal**: Sample scenarios and visit states under current policy.

**Procedure** (for each scenario $\omega$):

1. Start at initial state $x_0$
2. For each stage $t = 0, \ldots, T-1$:
   - Sample uncertainty $\xi_t^\omega$ from distribution
   - Solve subproblem:
     $$
     \min_{u_t} \left\{ c_t(x_t, u_t) + \alpha_t \right\}
     $$
     subject to:
     - Power balance, water balance, limits
     - Benders cuts: $\alpha_t \ge \beta_k + \pi_k^T (x_t - \bar{x}_k)$ for all cuts $k$
   - Record visited state $\bar{x}_t$
   - Update state: $x_{t+1} = x_t + \xi_t^\omega - u_t^\text{hydro}$
3. Accumulate total cost: $\sum_{t=0}^{T-1} c_t(x_t, u_t)$

**Output**: Visited states $\{\bar{x}_t\}$ and forward pass cost.

**Parallelization**: Multiple forward passes run in parallel (different scenarios).

### Backward Pass

**Goal**: Refine value function approximation by adding Benders cuts.

**Procedure**:

1. Start at final stage $t = T-1$ (no future cost)
2. For each stage $t = T-1, \ldots, 0$:
   - For each visited state $\bar{x}_t$ from forward pass:
     - Solve subproblem at $\bar{x}_t$ and all child scenarios $\xi_{t+1}^\omega$:
       $$
       Q_t(\bar{x}_t, \xi_{t+1}^\omega) = \min_{u_t} \left\{ c_t(\bar{x}_t, u_t) + \alpha_t \right\}
       $$
     - Extract dual variable $\pi_t^\omega$ (shadow price of water balance constraint)
     - Average over scenarios:
       $$
       \bar{\pi}_t = \frac{1}{|\Omega|} \sum_{\omega \in \Omega} \pi_t^\omega
       $$
     - Create Benders cut:
       $$
       \alpha_{t-1} \ge Q_t(\bar{x}_t) + \bar{\pi}_t^T (x_{t-1} - \bar{x}_t)
       $$
   - Add cut to stage $t-1$ subproblem

**Output**: New Benders cuts approximating value function.

**Single-Cut Variant**: One cut per stage per iteration (POWE.RS implementation).

### Convergence

**Lower Bound**: Computed from Benders cuts at root node (optimistic).

- Non-decreasing across iterations (monotonic)
- Converges to true optimal value from below

**Upper Bound**: Average of forward pass costs (pessimistic).

- Statistical estimate of policy value
- Converges to true optimal value from above

**Stopping Criterion**: When gap $\text{UB} - \text{LB}$ is sufficiently small.

---

## Convergence Properties

### Theoretical Guarantees

**Theorem** (Pereira & Pinto 1991):

- Lower bounds are non-decreasing: $\text{LB}_k \le \text{LB}_{k+1}$
- With infinite iterations and samples: $\text{LB}_k \to V^*$ (optimal value)
- For finite iterations: $\text{LB}_k \le V^* \le \mathbb{E}[\text{policy cost}]$

### Practical Convergence

**Typical behavior**:

- Lower bound increases quickly initially, then plateaus
- Gap decreases over iterations (statistical noise remains)
- Convergence in 50-500 iterations for hydrothermal problems

**Factors affecting convergence**:

- **Problem size**: More stages/reservoirs → more iterations
- **Uncertainty level**: High variance inflows → more iterations
- **Forward passes**: More passes → better lower bound estimates
- **Cut selection**: Removing dominated cuts speeds up solver

### SDDP vs. Deterministic Solution

**Deterministic Equivalent Problem (DEP)**:

- Solve full scenario tree directly (no decomposition)
- Optimal but intractable for $>5$ stages

**SDDP**:

- Decomposes by stage (tractable)
- Converges to DEP solution with enough iterations
- Can solve 50+ stage problems

**Value of Stochastic Solution (VSS)**:

- SDDP policy value vs. deterministic policy value
- Measures benefit of considering uncertainty
- Typically 5-20% cost reduction in hydrothermal systems

---

## Implementation in POWE.RS

### Key Design Decisions

#### 1. Single-Cut Variant

**Choice**: One cut per stage per iteration (average over scenarios).

**Alternative**: Multi-cut (one cut per scenario per iteration).

**Rationale**:

- ✅ More stable convergence (fewer cuts)
- ✅ Faster solver (fewer constraints)
- ⚠️ Slower convergence (fewer cuts per iteration)

**Future**: Multi-cut variant planned for v1.0+ (2-5× convergence acceleration).

#### 2. Sample Average Approximation (SAA)

**Choice**: Pre-generate scenarios, sample uniformly during training.

**Alternative**: Online sampling (generate scenarios on-the-fly).

**Rationale**:

- ✅ Reproducible (fixed seed → identical results)
- ✅ Efficient (pre-computed, cached)
- ✅ Easier to analyze (fixed scenario tree)

#### 3. Basis Warm-Starting

**Optimization**: Reuse LP basis from forward pass in backward pass.

**Impact**: 30-50% solver speedup.

**Mechanism**:

- Forward pass solves LP at state $x_t$, saves basis
- Backward pass starts with saved basis (near-optimal)
- HiGHS solver converges faster

#### 4. Batch Cut Selection

**Optimization**: Process cuts in batches to reduce lock contention.

**Impact**: 154× speedup in cut selection (5-10% overall).

**Mechanism**:

- Collect cuts from all parallel forward passes
- Add to pool in single lock acquisition (deterministic order)
- Eliminates thread synchronization overhead

See [performance/CUT-SELECTION.md](../performance/CUT-SELECTION.md) for analysis.

#### 5. Parallel Execution (Rayon)

**Strategy**: Thread-based parallelism with work-stealing.

**Parallelization points**:

- Forward passes (embarrassingly parallel)
- Backward pass cut generation (stage-wise synchronization)
- Simulation (embarrassingly parallel)

**Efficiency**: 14% parallelizable fraction (Amdahl's law analysis).

See [performance/PARALLELISM.md](../performance/PARALLELISM.md) for details.

### Algorithm Pseudocode

```python
def sddp_train(iterations, forward_passes):
    # Initialization
    cuts = {}  # Benders cuts by stage
    lower_bounds = []

    for k in range(iterations):
        # FORWARD PASS (parallel)
        visited_states = []
        forward_costs = []

        for scenario in parallel(range(forward_passes)):
            state = initial_state
            cost = 0

            for stage in range(num_stages):
                # Sample uncertainty
                inflow = sample_inflow(scenario, stage)

                # Solve subproblem with current cuts
                solution = solve_subproblem(state, inflow, cuts[stage])

                # Record and update
                visited_states.append((stage, state))
                cost += solution.immediate_cost
                state = solution.next_state

            forward_costs.append(cost)

        # BACKWARD PASS (sequential by stage, parallel within stage)
        for stage in reversed(range(num_stages)):
            for state in visited_states[stage]:
                # Solve for all child scenarios
                duals = []
                for child_inflow in child_scenarios:
                    sol = solve_subproblem(state, child_inflow, cuts[stage+1])
                    duals.append(sol.dual_water_balance)

                # Create Benders cut
                avg_dual = mean(duals)
                cut = BendersCut(intercept=sol.future_cost, slope=avg_dual, state=state)
                cuts[stage].add(cut)

        # Update lower bound (solve root subproblem)
        root_solution = solve_subproblem(initial_state, cuts[0])
        lower_bounds.append(root_solution.objective)

        # Statistics
        upper_bound = mean(forward_costs)
        gap = upper_bound - lower_bounds[-1]

        print(f"Iteration {k}: LB = {lower_bounds[-1]:.2f}, UB = {upper_bound:.2f}, Gap = {gap:.2f}")

    return cuts, lower_bounds
```

### Numerical Stability

POWE.RS includes several numerical robustness features:

**Multi-Level Solver Retry**:

1. Default: Tight tolerances (1e-7), simplex method
2. Retry 1: Relaxed tolerances (1e-6)
3. Retry 2: Further relaxed (1e-5)
4. Retry 3: Alternative simplex strategy
5. Retry 4: Interior point method with presolve

**Input Validation**:

- 26 validation rules across 4 phases
- Checks for infeasibility before training (early failure)
- Prevents ill-conditioned problems

See [guides/TROUBLESHOOTING.md](../guides/TROUBLESHOOTING.md) for common numerical issues.

---

## References

### Foundational Papers

1. **Pereira, M. V. F., & Pinto, L. M. V. G. (1991)**. "Multi-stage stochastic optimization applied to energy planning." _Mathematical Programming_, 52(1-3), 359-375.

   - Original SDDP algorithm for hydrothermal scheduling

2. **Shapiro, A., Tekaya, W., da Costa, J. P., & Soares, M. P. (2013)**. "Risk neutral and risk averse Stochastic Dual Dynamic Programming method." _European Journal of Operational Research_, 224(2), 375-391.

   - Statistical upper bound theory and risk measures

3. **Philpott, A. B., & de Matos, V. L. (2012)**. "Dynamic sampling algorithms for multi-stage stochastic programs with risk aversion." _Annals of Operations Research_, 200(1), 211-236.
   - Advanced sampling strategies

### Cut Selection

4. **de Matos, V. L., Philpott, A. B., & Finardi, E. C. (2015)**. "Improving the performance of Stochastic Dual Dynamic Programming." _Journal of Computational and Applied Mathematics_, 290, 196-208.
   - Level-1 cut selection (dominance checking)

### Textbooks

5. **Birge, J. R., & Louveaux, F. (2011)**. _Introduction to Stochastic Programming_. Springer.

   - Chapter 7: Benders decomposition and L-shaped method

6. **Powell, W. B. (2011)**. _Approximate Dynamic Programming_. Wiley.
   - Chapter 11: Stochastic programming perspective

---

## Further Reading

### POWE.RS Documentation

- [Performance Analysis](../performance/PARALLELISM.md) - Parallel efficiency and optimizations
- [Cut Selection](../performance/CUT-SELECTION.md) - Batch cut selection design
- [Architecture](../architecture/BATCH-CUT-SELECTION.md) - Implementation details

### Related Software

- **[SDDP.jl](https://github.com/odow/SDDP.jl)** - Julia implementation with advanced features
  - Multi-cut, risk measures, flexible stopping
  - Excellent documentation and tutorials
- **[PSRClassesInterface.jl](https://github.com/psrenergy/PSRClassesInterface.jl)** - Brazilian commercial SDDP tool

---

**Navigation**: [↑ Documentation Index](../README.md) | [API Reference](../reference/API-REFERENCE.md) | [Performance](../performance/PARALLELISM.md)
