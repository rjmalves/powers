---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §1.1 (Document Purpose)"
  - "MATHEMATICAL_FORMULATIONS.md §1.3 (Problem Context)"
  - "MATHEMATICAL_FORMULATIONS.md §2.1 (Multistage Stochastic Programming Formulation)"
  - "MATHEMATICAL_FORMULATIONS.md §2.2 (The SDDP Algorithm)"
  - "MATHEMATICAL_FORMULATIONS.md §2.3 (Policy Graph Structure)"
  - "MATHEMATICAL_FORMULATIONS.md §2.4 (State Variables and the Markov Property)"
  - "MATHEMATICAL_FORMULATIONS.md §2.5 (Single-Cut vs Multi-Cut Formulation)"
last_reviewed: 2026-02-20
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from MATHEMATICAL_FORMULATIONS.md §1-§2"
  - date: 2026-02-20
    description: "Review: §3.1-§3.2 rewritten from pseudocode to behavioral descriptions. §3.2 added discount factor on θ note and relatively complete recourse via penalty system. §3.3 fixed upper bound reference to upper-bound-evaluation.md. §4.2 fixed vague link to discount-rate.md, used d for discount factor to avoid β collision with cut coefficients. §5 added GNL validation-rejection note. §5.1 AR lag notation corrected to fixing constraints (matching approved lp-formulation.md §5). §3.4 added Execution Model and Performance Considerations section. Cross-references updated. Approved 2026-02-20."
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

The forward pass simulates the system under the current policy to generate **trial points** — the visited states that will be used by the backward pass to construct cuts.

For each of $M$ independent scenario trajectories:

1. Start from the known initial state $x_0$ (initial storage volumes and inflow history)
2. At each stage $t = 1, \ldots, T$, sample a scenario realization $\omega_t$ from the stage's scenario set, then solve the stage LP using the incoming state $\hat{x}_{t-1}$ and the sampled scenario. The LP includes all current Benders cuts as constraints on the future cost variable $\theta$
3. Record the optimal state $\hat{x}_t$ (end-of-stage storage volumes and updated AR lags) as the trial point for stage $t$
4. Pass $\hat{x}_t$ as the incoming state to stage $t+1$

The forward pass produces: (a) trial points $\{\hat{x}_t\}$ at each stage for each trajectory, and (b) stage costs for upper bound estimation.

**Parallelization**: Forward trajectories are independent — POWE.RS distributes $M$ trajectories across MPI ranks, with OpenMP threads solving individual stage LPs within each rank.

**Warm-starting**: The forward pass LP solution at stage $t$ provides a near-optimal basis for the backward pass solves at the same stage, significantly reducing solve times.

### 3.2 Backward Pass

The backward pass improves the value function approximation by generating new Benders cuts, walking stages in reverse order from $T$ down to $1$.

At each stage $t$, for each trial point $\hat{x}_{t-1}$ collected during the forward pass:

1. Solve the stage $t$ LP for **every** scenario $\omega \in \Omega_t$ (branching), using the trial state $\hat{x}_{t-1}$ as incoming state
2. From each LP solution, extract the optimal objective value $Q_t(\hat{x}_{t-1}, \omega)$ and the dual variables of state-linking constraints (water balance, AR lag fixing). For hydros using FPHA, the FPHA hyperplane duals also contribute to storage cut coefficients — see [Cut Management §2](cut-management.md)
3. Compute per-scenario cut coefficients $(\alpha(\omega), \beta(\omega))$ from the duals and trial point
4. Aggregate into a single cut via probability-weighted expectation — see [Cut Management §3](cut-management.md)
5. Add the aggregated cut to stage $t-1$'s cut pool

The backward pass produces one new cut per stage per trial point per iteration.

> **Feasibility**: Every backward LP solve must be feasible. This is guaranteed by the recourse slack system (Category 1 penalties) — see [Penalty System](../02-data-model/penalty-system.md). The relatively complete recourse property ensures valid cuts; see [Cut Management §4](cut-management.md).

> **Discount factor**: When discount rates are active, the discount factor is applied to the $\theta$ variable in the stage $t-1$ objective (i.e., $d_{t-1 \to t} \cdot \theta$), not to the cut coefficients. The cuts themselves remain unmodified. See [Discount Rate](discount-rate.md).

![Scenario Tree Branching](../../diagrams/exports/svg/sddp/scenario-tree-branching.svg)

### 3.3 Convergence Monitoring

**Lower Bound**: The deterministic lower bound is the first-stage LP value:

$$
\underline{z}^k = V_1^k(x) = \min_{x_1} \left\{ c_1^\top x_1 + \theta_1 : \text{constraints}, \; \theta_1 \geq \alpha_i + \beta_i^\top x_1 \; \forall i \right\}
$$

This bound increases monotonically as cuts are added.

**Upper Bound**: Estimated via Monte Carlo simulation over the forward pass trajectories (see [Upper Bound Evaluation](upper-bound-evaluation.md)):

$$
\bar{z}^k = \frac{1}{M} \sum_{m=1}^{M} \sum_{t=1}^{T} c_t^\top x_t^{(m)}
$$

**Optimality Gap**:

$$
\text{gap}^k = \frac{\bar{z}^k - \underline{z}^k}{\max(1, |\bar{z}^k|)}
$$

### 3.4 Execution Model and Performance Considerations

The SDDP iteration structure has specific properties that guide the parallelization strategy and solver lifecycle design. These are summarized here as architectural constraints; detailed design is in the HPC and architecture specs.

**Thread-trajectory affinity**: The dominant parallelization strategy assigns each thread ownership of a complete forward trajectory. With $N$ threads (summed across all OpenMP threads of all MPI ranks), $N$ forward passes execute in parallel, each thread solving its trajectory's stage LPs sequentially from $t = 1$ to $T$. The same thread that executed forward pass $k$ also performs the backward pass for the scenarios sampled by forward pass $k$. This affinity pattern preserves cache locality (solver basis, scenario data, LP coefficients remain warm in the thread's cache lines) and simplifies implementation by eliminating cross-thread data handoff.

**Backward pass synchronization**: Unlike the forward pass (fully parallel), the backward pass has a **hard synchronization barrier at each stage boundary**: all threads must complete cut construction at stage $t$ before any thread proceeds to stage $t-1$. Within a stage, each thread solves its branching scenarios sequentially, reusing the warm solver basis saved from the forward pass at that stage. This sequential branching keeps the solver state hot and avoids redundant LP setup.

**Forward pass state saving**: When the number of forward passes $M$ exceeds the number of available threads $N$, threads must process multiple trajectories in batches. This requires efficiently saving and restoring forward pass state (solver basis, scenario realization, visited states) at stage boundaries — analogous to CPU context switching for threads, but simpler because suspension only occurs at well-defined stage boundaries, not at arbitrary points.

**LP rebuild cost**: Memory constraints prevent keeping all stage LPs with their full cut sets resident simultaneously. The solver must rebuild LPs and add cut constraints when transitioning between stages, which lies on the critical performance path. The design must minimize this rebuild cost through strategies such as cut preallocation, basis persistence, and incremental constraint updates. See [Solver Abstraction](../03-architecture/solver-abstraction.md) and [Solver Workspaces](../03-architecture/solver-workspaces.md).

**Generic constraint dual preprocessing**: When generic constraints involve state variables (storage volumes), their duals contribute to cut coefficients (see [Cut Management §2](cut-management.md)). Since the mapping from generic constraint duals to state variable coefficients is static (determined at input loading time), it should be precomputed once and reused across all iterations. See [Cut Management Implementation](../03-architecture/cut-management-impl.md).

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
- **Discount**: Cycle transitions require a discount factor $d < 1$ for convergence
- **Cut sharing**: Cuts at equivalent cycle positions are shared

> **Symbol note**: All specs use $d$ for the discount factor to avoid collision with the cut coefficient symbol $\beta$ (see [Cut Management §1](cut-management.md)). The deficit variable uses $\delta$ (lowercase delta), so there is no conflict.

See [Discount Rate](discount-rate.md) for the discounted Bellman equation and [Infinite Horizon](infinite-horizon.md) for the complete cyclic formulation.

## 5. State Variables and the Markov Property

For SDDP to generate valid cuts, the subproblem must satisfy the **Markov property**: future costs depend only on the current state, not on how we arrived at that state.

**State variables in POWE.RS**:

| Component      | Variable       | Count                | Description                        |
| -------------- | -------------- | -------------------- | ---------------------------------- |
| Hydro storage  | $v_h$          | $N_{hydro}$          | Reservoir volume at end of stage   |
| AR inflow lags | $a_{h,\ell}$   | $\sum_h P_h$         | Lagged inflows for AR(P) models    |
| Battery SOC    | $soc_{bat}$    | $N_{battery}$        | Battery state of charge (DEFERRED) |
| GNL pipeline   | $gnl_{t,\ell}$ | $\sum_{gnl} L_{gnl}$ | Committed GNL dispatch (DEFERRED)  |

> **Note on deferred state variables**: Battery SOC and GNL pipeline state are planned extensions — see [Deferred Features](../06-deferred/deferred-features.md). For GNL thermals specifically, the data model already accepts GNL configurations but validation rejects them until the solver implementation is ready — see [Equipment Formulations §3](equipment-formulations.md).

### 5.1 AR Lag State Expansion

The PAR(p) inflow model requires past inflows $a_{h,t-1}, a_{h,t-2}, \ldots$ to compute current inflow. To maintain the Markov property, these lags are included as state variables with **fixing constraints** that bind each lag variable to the corresponding incoming state value:

$$
a_{h,\ell} = \hat{a}_{h,\ell} \quad \forall h \in \mathcal{H}, \; \ell \in \{1, \ldots, P_h\}
$$

where $\hat{a}_{h,\ell}$ is the lag $\ell$ inflow value passed from the previous stage. The duals of these fixing constraints ($\pi^{lag}_{h,\ell}$) contribute to cut coefficients, capturing the marginal value of inflow history — see [Cut Management §2](cut-management.md).

See [PAR Inflow Model](par-inflow-model.md) for the complete autoregressive formulation and [LP Formulation §5](lp-formulation.md) for how these constraints appear in the stage LP.

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
- [LP Formulation](lp-formulation.md) — Complete stage subproblem LP that the forward/backward passes solve
- [Cut Management](cut-management.md) — Cut generation, aggregation, selection, and validity conditions
- [PAR Inflow Model](par-inflow-model.md) — Stochastic inflow model driving uncertainty in the forward pass
- [Discount Rate](discount-rate.md) — Discounted Bellman equation, stage-dependent rates, discount factor on θ
- [Infinite Horizon](infinite-horizon.md) — Periodic structure, cycle detection, cut sharing, convergence
- [Upper Bound Evaluation](upper-bound-evaluation.md) — Upper bound estimation methods
- [Stopping Rules](stopping-rules.md) — Convergence criteria that terminate the iterative process
- [Risk Measures](risk-measures.md) — CVaR and risk-averse extensions to the Bellman recursion
- [Penalty System](../02-data-model/penalty-system.md) — Recourse slacks guaranteeing feasibility (relatively complete recourse)
- [Equipment Formulations](equipment-formulations.md) — GNL thermal validation-rejection rule
- [Deferred Features](../06-deferred/deferred-features.md) — Multi-cut formulation, Markovian policy graphs, batteries
- [Production Scale Reference](../00-overview/production-scale-reference.md) — Typical problem sizes and state dimensions
