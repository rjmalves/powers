---
status: draft
review_priority: 3-medium
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §14 (14.1-14.8) Discount Rate Formulation"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Discount Rate Formulation

## Purpose

This spec defines how discount rates are incorporated into the SDDP algorithm in POWE.RS: the discounted Bellman equation, stage-dependent discount factors, modified Benders cuts, cumulative discounting, and the effect on lower/upper bound computation. It also covers the infinite periodic horizon formulation where discounting is required for convergence.

For notation conventions (index sets, parameters, decision variables, dual variables), see [Notation Conventions](../00-overview/notation-conventions.md).

## 14.1 Motivation

The discount rate $\beta \in (0, 1]$ captures the time value of money or risk preference, where future costs are valued less than present costs. This is essential for:

1. **Infinite horizon problems**: Ensuring convergence of the value function
2. **Economic consistency**: Reflecting opportunity cost of capital
3. **Risk adjustment**: Implicitly reducing weight of distant uncertain outcomes

## 14.2 Discounted Bellman Equation

The standard risk-neutral Bellman recursion with discount factor $\beta$ is:

$$
V_t(x_{t-1}) = \mathbb{E}_{\omega_t}\left[\min_{x_t \in \mathcal{X}_t(\omega_t)} \left\{ c_t(x_t, u_t) + \beta \cdot V_{t+1}(x_t) \right\}\right]
$$

where:

- $c_t(x_t, u_t)$ is the immediate cost at stage $t$
- $V_{t+1}(x_t)$ is the future cost function (cost-to-go)
- $\beta = \frac{1}{1 + r}$ with $r$ being the discount rate per stage

> **Formulation Note**: The discount factor $\beta$ multiplies only the **future cost** $V_{t+1}$, not the immediate cost $c_t$. This is the standard SDDP convention:
>
> - **Immediate cost** $c_t$: Not discounted (incurred "now" at stage $t$)
> - **Future cost** $\beta \cdot V_{t+1}$: Discounted to present value at stage $t$
>
> This is mathematically equivalent to computing all costs at "time 0" (stage 1) present value, where stage $t$ costs are multiplied by $\prod_{s=1}^{t-1} \beta_s$ in the objective. The Bellman formulation above is the recursive form that SDDP exploits.
>
> **Alternative formulations** (not used in POWE.RS):
>
> - Some formulations discount both immediate and future cost by $\beta_t$ within the expectation
> - The choice affects cut coefficient scaling but not the optimal policy

## 14.3 Stage-Dependent Discount Rates

In POWE.RS, discount rates are specified per **transition** in `stages.json`:

```json
{
  "transitions": [
    {
      "source_id": 0,
      "target_id": 1,
      "probability": 1.0,
      "discount_rate": 0.005
    },
    {
      "source_id": 1,
      "target_id": 2,
      "probability": 1.0,
      "discount_rate": 0.005
    }
  ]
}
```

The discount factor for transition from stage $t$ to stage $t+1$ is:

$$
\beta_{t \to t+1} = \frac{1}{1 + r_{t \to t+1}}
$$

For configuration details, see [Configuration Reference](../05-config/configuration-reference.md).

## 14.4 Modified Stage Subproblem

The stage $t$ subproblem with discounting becomes:

$$
Q_t(x_{t-1}, \omega_t) = \min_{x_t, u_t, \theta} \left\{ c_t(x_t, u_t) + \theta \right\}
$$

subject to:

- All standard constraints (load balance, hydro balance, etc.)
- **Discounted Benders cuts**:

$$
\theta \geq \beta_{t \to t+1} \cdot \left( \alpha_i + \sum_{h} \beta^v_{i,h} \cdot v_h + \sum_{h,\ell} \beta^{lag}_{i,h,\ell} \cdot a_{h,\ell} \right) \quad \forall i
$$

The cut coefficients $(\alpha_i, \beta^v_{i,h}, \beta^{lag}_{i,h,\ell})$ are the undiscounted values from the backward pass. The discount factor $\beta_{t \to t+1}$ scales the entire cut when it is added to the LP. See [Cut Management](cut-management.md) for full details on cut generation and aggregation.

## 14.5 Cumulative Discounting

For a path from stage 1 to stage $T$, the cumulative discount factor is:

$$
\beta_{1 \to T} = \prod_{t=1}^{T-1} \beta_{t \to t+1}
$$

The present value at stage 1 of costs incurred at stage $T$ is:

$$
\text{PV}_1[c_T] = \beta_{1 \to T} \cdot c_T
$$

## 14.6 Lower Bound Computation with Discounting

The deterministic lower bound at iteration $k$ is computed as:

$$
\underline{z}^k = c_1(\hat{x}_1^k) + \theta_1^k
$$

where $\theta_1^k$ is the optimal value of the future cost variable at stage 1, which already includes all discounting through the cut coefficients.

## 14.7 Upper Bound (Simulation) with Discounting

When simulating the policy to estimate the upper bound:

$$
\bar{z}^k = \frac{1}{M} \sum_{m=1}^{M} \sum_{t=1}^{T} \beta_{1 \to t} \cdot c_t(\hat{x}_t^{k,m})
$$

where $\beta_{1 \to 1} = 1$ and $\beta_{1 \to t} = \prod_{s=1}^{t-1} \beta_{s \to s+1}$.

For stopping rules that use these bounds, see [Stopping Rules](stopping-rules.md).

## 14.8 Implementation Notes

- **Cut storage**: Cuts are stored in **undiscounted** form. Discounting is applied when adding to LP.
- **Cut coefficients in LP**: The LP stores $\beta \cdot \alpha$ and $\beta \cdot \beta^v$, not the raw values.
- **Consistent units**: All bounds and gaps are reported in present value terms (stage 1 currency).

> **Bound Interpretation**: Both the lower bound $\underline{z}$ and upper bound $\bar{z}$ represent total expected cost expressed in **present value at stage 1**. This means:
>
> - A lower bound of \$100M means the optimal policy costs at least \$100M in stage-1 dollars
> - Future costs are already discounted: a \$1M cost at stage 12 with $\beta_{1 \to 12} = 0.95$ contributes \$0.95M to the bounds
> - Comparisons between bounds and between iterations are valid because they use consistent discounting
> - When reporting per-stage costs in simulation outputs, POWE.RS reports both **nominal** (undiscounted) and **present value** costs

## Infinite Periodic Horizon

### 15.1 Motivation

Standard finite-horizon SDDP has a terminal condition $V_{T+1}(x) = 0$, causing "end-of-world" effects where the algorithm empties reservoirs toward the horizon. For long-term planning, an **infinite periodic horizon** better represents the ongoing nature of hydrothermal operations.

### 15.2 Periodic Structure

Consider a system with 12 monthly stages that repeat annually. Let $\tau(t)$ denote the **season** (position within the cycle) for stage $t$:

$$
\tau(t) = (t - 1) \mod 12 + 1 \in \{1, 2, \ldots, 12\}
$$

Stages with the same season share structural properties (demand patterns, inflow statistics).

### 15.3 Cycle Detection

The algorithm detects cycles by analyzing the transition graph in `stages.json`. A **cycle** exists when a transition points to a stage with ID less than or equal to the source (backward edge in a DAG sense).

**Example**:

```json
{
  "transitions": [
    {"source_id": 0, "target_id": 1, "probability": 1.0},
    ...
    {"source_id": 59, "target_id": 48, "probability": 1.0, "discount_rate": 0.05}
  ]
}
```

Here, stage 59 transitions back to stage 48, creating a 12-stage cycle.

### 15.4 Discounting for Convergence

For convergence, the cycle must include a **discount factor** $\beta < 1$ on the return edge. This ensures:

$$
\lim_{n \to \infty} \beta^n V_t(x) = 0
$$

The cumulative discount around one full cycle must satisfy:

$$
\beta_{cycle} = \prod_{t \in cycle} \beta_{t \to t+1} < 1
$$

**Typical setup**: Monthly discount rate of 0.5% gives $\beta = 1/1.005 \approx 0.995$, and annual discount $\beta_{cycle} = 0.995^{12} \approx 0.94$.

### 15.5 Cut Sharing Within Cycles

Stages in the same position of the cycle share their value function approximation. Let $\mathcal{C}_\tau = \{t : \tau(t) = \tau\}$ be all stages with season $\tau$.

**Cut sharing rule**: A cut generated at stage $t \in \mathcal{C}_\tau$ is applicable to all stages in $\mathcal{C}_\tau$:

$$
\underline{V}_\tau(x) = \max_{k \in \mathcal{K}_\tau} \left\{ \alpha_k + \beta_k^\top x \right\}
$$

**Implementation**: The cut pool is indexed by season $\tau \in \{1, \ldots, 12\}$, not by absolute stage ID.

### 15.6 Fixed-Point Iteration

The infinite-horizon SDDP finds the fixed point of the Bellman operator:

$$
V_\tau = T_\tau V_{\tau+1}
$$

where $T_\tau$ is the one-stage Bellman operator for season $\tau$:

$$
(T_\tau V)(x) = \mathbb{E}_{\omega_\tau}\left[\min_{x'} \left\{ c_\tau(x', u) + \beta \cdot V(x') \right\}\right]
$$

**Convergence criterion**: Value functions have converged when:

$$
\left\| V_\tau^{k+1} - V_\tau^k \right\|_\infty < \delta_{cycle}
$$

for all seasons $\tau$, where $\delta_{cycle}$ is the `cycle_discretization_delta` tolerance.

### 15.7 Modified Forward Pass

In infinite horizon, the forward pass continues until the discounted contribution becomes negligible:

**Algorithm: Infinite Horizon Forward Pass**

1. For forward pass $m$:
   - Initialize: $t = 0$, `cumulative_discount = 1.0`, $x = x_0$, `total_cost = 0`
   - While `cumulative_discount` $>$ `tolerance` AND $t <$ `max_horizon_length`:
     - Sample $\omega_t$ from stage $\tau(t)$
     - Solve subproblem: $(x', \text{cost}) = \text{solve\_subproblem}(t, x, \omega_t)$
     - Update: `total_cost += cumulative_discount * cost`
     - Update: $x = x'$
     - Increment: $t = t + 1$
     - Update discount: `cumulative_discount *= `$\beta_{t-1 \to t}$

The `max_horizon_length` provides a safety bound (e.g., 240 stages = 20 years for monthly).

### 15.8 Backward Pass Modifications

**Stopping condition**: The backward pass stops when it completes a full cycle with no significant improvement:

$$
\max_{\tau} \left| \underline{z}^{k,\tau} - \underline{z}^{k-12,\tau} \right| < \delta_{cycle}
$$

**Cut generation**: Same as finite horizon, but cuts are added to the season's cut pool, not a specific stage.

### 15.9 Configuration

```json
{
  "horizon": {
    "mode": "infinite_periodic",
    "max_horizon_length": 240,
    "cycle_discretization_delta": 0.1
  }
}
```

| Parameter                    | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `mode`                       | `"infinite_periodic"` enables this formulation |
| `max_horizon_length`         | Maximum stages in forward pass                 |
| `cycle_discretization_delta` | Convergence tolerance for cycle                |

For full configuration schema, see [Configuration Reference](../05-config/configuration-reference.md).

### 15.10 Reference

> Costa, B.S., de Matos, V.L., Philpott, A.B. (2025). "SDDP.jl approaches for infinite horizon problems." _Trends in Computational and Applied Mathematics_, 11(1). https://doi.org/10.5540/03.2025.011.01.0355

## Cross-References

- [SDDP Algorithm](sddp-algorithm.md) — Core Bellman equation and forward/backward pass structure that discount rates modify
- [Notation Conventions](../00-overview/notation-conventions.md) — Standard symbols for state variables, dual variables, and cost-to-go functions
- [Cut Management](cut-management.md) — Cut generation and aggregation affected by discount factor scaling
- [Stopping Rules](stopping-rules.md) — Convergence criteria using discounted lower/upper bounds
- [Upper Bound Evaluation](upper-bound-evaluation.md) — Inner approximation uses discounted vertex values
- [Configuration Reference](../05-config/configuration-reference.md) — `stages.json` transition discount rates and horizon configuration
