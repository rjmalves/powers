---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §1.1 (Document Purpose)"
  - "MATHEMATICAL_FORMULATIONS.md §1.3 (Problem Context)"
  - "MATHEMATICAL_FORMULATIONS.md §2.1 (Multistage Stochastic Programming Formulation)"
  - "MATHEMATICAL_FORMULATIONS.md §2.2 (The SDDP Algorithm)"
  - "MATHEMATICAL_FORMULATIONS.md §2.3 (Policy Graph Structure)"
  - "MATHEMATICAL_FORMULATIONS.md §2.4 (State Variables and the Markov Property)"
  - "MATHEMATICAL_FORMULATIONS.md §2.5 (Single-Cut vs Multi-Cut Formulation)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# SDDP Algorithm

## Purpose

This spec describes the Stochastic Dual Dynamic Programming (SDDP) algorithm as implemented in POWE.RS: the multistage stochastic formulation, the iterative forward/backward pass structure, convergence monitoring, policy graph topologies, state variable requirements, and the single-cut vs multi-cut trade-off. It serves as the algorithmic foundation referenced by all other mathematical specs.

For notation conventions (index sets, parameters, decision variables, dual variables), see [Notation Conventions](../00-overview/notation-conventions.md).

## 1. Problem Context

POWE.RS solves the **hydrothermal dispatch problem**: determining optimal generation schedules for hydro and thermal plants over a multi-year planning horizon under inflow uncertainty. Key characteristics:

- **Flexible for short or long horizons**: 1 month - 5 years (daily, weekly or monthly stages)
- **Large state space**: 160+ hydro reservoirs with AR inflow models $\approx$ 2000 state dimensions
- **Stochastic inflows**: PAR(p) autoregressive models with seasonal patterns
- **Customize modeling complexity stagewise**: Scenario generation, hydro production and others are configurable stagewise and elementwise

## 2. Multistage Stochastic Programming Formulation

The hydrothermal dispatch problem is formulated as a multistage stochastic program:

$$
\min_{x_1, \ldots, x_T} \mathbb{E}\left[ \sum_{t=1}^{T} c_t(\omega_t)^\top x_t(\omega_{t}) \right]
$$

subject to stage-linking constraints and uncertainty realization. The nested formulation uses **value functions**:

$$
V_t(x_{t-1}) = \mathbb{E}_{\omega_t}\left[ \min_{x_t} \left\{ c_t^\top x_t + V_{t+1}(x_t) : A_t x_t = b_t - E_t x_{t-1}, \; x_t \in \mathcal{X}_t \right\} \right]
$$

with terminal condition $V_{T+1}(x) = 0$.

**Key insight**: The value function $V_t(x)$ is convex and piecewise linear (for LP subproblems), enabling outer approximation via Benders cuts.

![Value Function Approximation via Benders Cuts](../../diagrams/exports/svg/sddp/value-function-approximation.svg)

## 3. The SDDP Algorithm

SDDP iteratively builds piecewise-linear approximations $\hat{V}_t^k$ at iteration $k$ of the true value functions through:

1. **Forward pass**: Sample scenarios, make decisions using current approximation
2. **Backward pass**: Compute cuts to improve the approximation
3. **Convergence check**: Evaluate stopping criteria

![SDDP Iteration: Forward Pass, Backward Pass, and Convergence](../../diagrams/exports/svg/sddp/sddp-iteration.svg)

### 3.1 Forward Pass

The forward pass simulates the system under the current policy to generate **trial points** (visited states):

**Algorithm: Forward Pass** (iteration $k$, pass $m$)

- **Input:** Initial state $x_0$, cut approximations $\{\hat{V}_t^k\}$
- **Output:** Visited states $\{\hat{x}_t^m\}_{t=1}^{T-1}$, scenario costs

1. Set $\hat{x}_0 = x_0$
2. For $t = 1$ to $T$:
   - Sample $\omega_t \sim P(\Omega_t)$
   - Solve stage LP with incoming state $\hat{x}_{t-1}$ and realization $\omega_t$:
     $$\hat{x}_t, \hat{\theta}_t = \arg\min \{ c_t^\top x_t + \theta_t : \text{constraints}(x_t, \hat{x}_{t-1}, \omega_t), \theta_t \geq \alpha_i + \beta_i^\top x_t \; \forall \text{ cut } i \}$$
   - Record visited state $\hat{x}_t$
3. Return $\{\hat{x}_t^m\}_{t=0}^{T-1}$

**Parallelization**: Forward passes are parallel—each scenario trajectory is independent. POWE.RS distributes $M$ forward passes across MPI ranks and OpenMP threads below them.

### 3.2 Backward Pass

The backward pass computes cuts by walking stages in reverse order:

**Algorithm: Backward Pass** (iteration $k$)

- **Input:** Visited states from forward passes
- **Output:** New cuts for each stage

1. For $t = T$ down to $1$:
   - For each visited state $\hat{x}_{t-1}$ from forward passes:
     - For each $\omega \in \Omega_t$ (branching scenarios):
       - Solve stage LP with $(\hat{x}_{t-1}, \omega)$
       - Extract: $Q_t(\hat{x}_{t-1}, \omega) = \text{optimal value}$  
         $\pi_t(\omega) = \text{dual of state constraints}$
       - Compute per-scenario cut coefficients (see [Notation Conventions — §5.4](../00-overview/notation-conventions.md#54-cut-coefficient-derivation-from-duals) for sign convention):  
         $\beta(\omega) = \pi_t(\omega)$ (for state constraints $x_t = \hat{x}_{t-1} + \ldots$)  
         $\alpha(\omega) = Q_t - \beta(\omega)^\top \hat{x}_{t-1}$
     - Aggregate cut (single-cut formulation):  
       $\bar{\beta} = \sum_\omega p(\omega) \cdot \beta(\omega)$  
       $\bar{\alpha} = \sum_\omega p(\omega) \cdot \alpha(\omega)$
     - Add cut to stage $t-1$:  
       $\theta_{t-1} \geq \bar{\alpha} + \bar{\beta}^\top x_{t-1}$

**Warm-starting**: The forward pass solution provides a near-optimal basis for backward branching scenarios, significantly reducing solve times.

![Scenario Tree Branching](../../diagrams/exports/svg/sddp/scenario-tree-branching.svg)

### 3.3 Convergence Monitoring

**Lower Bound**: The deterministic lower bound is the first-stage LP value:

$$
\underline{z}^k = V_1^k(x) = \min_{x_1} \left\{ c_1^\top x_1 + \theta_1 : \text{constraints}, \; \theta_1 \geq \alpha_i + \beta_i^\top x_1 \; \forall i \right\}
$$

This bound increases monotonically as cuts are added.

**Upper Bound**: Estimated via Monte Carlo simulation or inner approximation (see [Deferred Features — Upper Bound Evaluation](../06-deferred/deferred-features.md)):

$$
\bar{z}^k = \frac{1}{M} \sum_{m=1}^{M} \sum_{t=1}^{T} c_t^\top x_t^{(m)}
$$

**Optimality Gap**:

$$
\text{gap}^k = \frac{\bar{z}^k - \underline{z}^k}{\max(1, |\bar{z}^k|)}
$$

## 4. Policy Graph Structure

### 4.1 Finite Horizon (Acyclic Graph)

The standard SDDP formulation uses an acyclic directed graph:

![Finite Horizon Policy Graph](../../diagrams/exports/svg/sddp/policy-graph-finite.svg)

- **Nodes**: Stages $t \in \{1, \ldots, T\}$
- **Arcs**: Transitions with probabilities (typically deterministic: $p = 1$)
- **Terminal**: $V_{T}(x) = 0$ (no future cost)

### 4.2 Cyclic Graph (Infinite Horizon)

For long-term planning, POWE.RS supports **infinite periodic horizon** with cyclic graphs:

![Cyclic Policy Graph — Infinite Horizon](../../diagrams/exports/svg/sddp/policy-graph-cyclic.svg)

- **Cycle**: Stage $T$ transitions back to stage $1$ (or a cycle start)
- **Discount**: Cycle transitions require discount rate $\beta < 1$ for convergence
- **Cut sharing**: Cuts at equivalent cycle positions are shared

See [Discount Rate and Infinite Horizon specs](../01-math/) for the complete infinite horizon formulation.

## 5. State Variables and the Markov Property

For SDDP to generate valid cuts, the subproblem must satisfy the **Markov property**: future costs depend only on the current state, not on how we arrived at that state.

**State variables in POWE.RS**:

| Component      | Variable       | Count                | Description                        |
| -------------- | -------------- | -------------------- | ---------------------------------- |
| Hydro storage  | $v_h$          | $N_{hydro}$          | Reservoir volume at end of stage   |
| AR inflow lags | $a_{h,\ell}$   | $\sum_h P_h$         | Lagged inflows for AR(P) models    |
| Battery SOC    | $soc_{bat}$    | $N_{battery}$        | Battery state of charge (DEFERRED) |
| GNL pipeline   | $gnl_{t,\ell}$ | $\sum_{gnl} L_{gnl}$ | Committed GNL dispatch (DEFERRED)  |

### 5.1 AR Lag State Expansion Trick

The AR inflow model requires past inflows $a_{h,t-1}, a_{h,t-2}, \ldots$ to compute current inflow. To maintain the Markov property, these lags are included as state variables with trivial dynamics:

$$
a_{h,\ell}^{out} = a_{h,\ell-1}^{in} \quad \text{for } \ell = 2, \ldots, P_h
$$

This allows cut coefficients to capture the value of inflow history.

See [PAR Inflow Model](../01-math/par-inflow-model.md) for the complete autoregressive formulation.

## 6. Single-Cut vs Multi-Cut Formulation

### 6.1 Single-Cut (Default)

One aggregated cut per iteration:

$$
\theta_{t-1} \geq \bar{\alpha} + \bar{\beta}^\top x_{t-1}
$$

where $\bar{\alpha} = \mathbb{E}[\alpha(\omega)]$ and $\bar{\beta} = \mathbb{E}[\beta(\omega)]$.

- **Pros**: Fewer cuts, smaller LP, faster solves
- **Cons**: May require more iterations to converge

### 6.2 Multi-Cut (DEFERRED)

One cut per scenario per iteration:

$$
\theta_{t-1,\omega} \geq \alpha(\omega) + \beta(\omega)^\top x_{t-1} \quad \forall \omega \in \Omega_t
$$

- **Pros**: Tighter approximation, fewer iterations
- **Cons**: More cuts, larger LP, memory-intensive

POWE.RS implements single-cut by default. Multi-cut is planned for future implementation. See [Deferred Features — C.3 Multi-Cut Formulation](../06-deferred/deferred-features.md#c3-multi-cut-formulation) for the full trade-off analysis.

## Cross-References

- [Notation Conventions](../00-overview/notation-conventions.md) — All index sets, parameters, decision variables, and dual variable definitions
- [Cut Management](../01-math/cut-management.md) — Cut generation mechanics, aggregation, and selection strategies
- [LP Formulation](../01-math/lp-formulation.md) — Complete stage subproblem LP that the forward/backward passes solve
- [PAR Inflow Model](../01-math/par-inflow-model.md) — Stochastic inflow model driving uncertainty in the forward pass
- [Risk Measures](../01-math/risk-measures.md) — CVaR and risk-averse extensions to the Bellman recursion
- [Deferred Features](../06-deferred/deferred-features.md) — Multi-cut formulation, Markovian policy graphs, and other planned extensions
- [Production Scale Reference](../00-overview/production-scale-reference.md) — Typical problem sizes and state dimensions
