---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §14 (14.1-14.8) Discount Rate Formulation"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: "Rewritten during review: symbol $d$ (not $β$), aligned with approved input-scenarios.md annual_discount_rate, discount on θ convention, infinite horizon content extracted to infinite-horizon.md."
change_log:
  - date: 2026-02-22
    description: "Full rewrite: $d$ symbol, aligned config with policy_graph annual_discount_rate, discount-on-θ convention, §15 extracted to infinite-horizon.md, renumbered §1-9."
---

# Discount Rate Formulation

## Purpose

This spec defines how discount rates are incorporated into the POWE.RS SDDP solver: the discounted Bellman equation, stage-dependent discount factors, effect on the future cost variable $\theta$, cumulative discounting, and the effect on lower/upper bound computation.

For the infinite periodic horizon formulation (where discounting is required for convergence), see [Infinite Horizon](infinite-horizon.md).

For notation conventions (index sets, parameters, decision variables, dual variables), see [Notation Conventions](../00-overview/notation-conventions.md).

> **Symbol convention**: This spec uses $d$ for the discount factor to avoid collision with the cut coefficient symbol $\beta$ used in [Cut Management](cut-management.md). The deficit variable $\delta_{b,k,s}$ uses a different symbol (lowercase delta), so there is no conflict.

## 1 Motivation

The discount factor $d \in (0, 1]$ captures the time value of money or risk preference, where future costs are valued less than present costs. This is essential for:

1. **Infinite horizon problems**: Ensuring convergence of the value function
2. **Economic consistency**: Reflecting opportunity cost of capital
3. **Risk adjustment**: Implicitly reducing weight of distant uncertain outcomes

## 2 Discounted Bellman Equation

The standard risk-neutral Bellman recursion with discount factor $d$ is:

$$
V_t(x_{t-1}) = \mathbb{E}_{\omega_t}\left[\min_{x_t \in \mathcal{X}_t(\omega_t)} \left\{ c_t(x_t, u_t) + d_{t \to t+1} \cdot V_{t+1}(x_t) \right\}\right]
$$

where:

- $c_t(x_t, u_t)$ is the immediate cost at stage $t$
- $V_{t+1}(x_t)$ is the future cost function (cost-to-go)
- $d_{t \to t+1}$ is the discount factor for the transition from stage $t$ to $t+1$

> **Formulation Note**: The discount factor $d$ multiplies only the **future cost** $V_{t+1}$, not the immediate cost $c_t$. This is the standard SDDP convention:
>
> - **Immediate cost** $c_t$: Not discounted (incurred "now" at stage $t$)
> - **Future cost** $d \cdot V_{t+1}$: Discounted to present value at stage $t$
>
> This is mathematically equivalent to computing all costs at "time 0" (stage 1) present value, where stage $t$ costs are multiplied by $\prod_{s=1}^{t-1} d_{s \to s+1}$ in the objective. The Bellman formulation above is the recursive form that SDDP exploits.

## 3 Input Specification

Discount rates are specified as an **annual rate** in `stages.json`, within the `policy_graph` section:

```json
{
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 1, "target_id": 2, "probability": 1.0 }
    ]
  }
}
```

The system converts the annual rate to a per-transition discount factor based on each stage's duration:

- Stage duration $\Delta t$ is derived from `end_date - start_date`, expressed in years
- Transition discount factor: $d_{t \to t+1} = \frac{1}{(1 + r_{annual})^{\Delta t}}$
- The duration used is that of the **source** stage (the stage whose future cost is being discounted)

A value of `0.0` means no discounting ($d = 1.0$ for all transitions).

Individual transitions may override the global rate:

```json
{
  "source_id": 59,
  "target_id": 48,
  "probability": 1.0,
  "annual_discount_rate": 0.1
}
```

For the complete `policy_graph` schema, see [Input Scenarios §1.2](../02-data-model/input-scenarios.md).

## 4 Discount Factor in the Stage Subproblem

The discount factor is applied to the **future cost variable** $\theta$ in the stage $t$ objective, not to the cut coefficients:

$$
Q_t(x_{t-1}, \omega_t) = \min_{x_t, u_t, \theta} \left\{ c_t(x_t, u_t) + d_{t \to t+1} \cdot \theta \right\}
$$

subject to all standard constraints (load balance, hydro balance, etc.) and Benders cuts:

$$
\theta \geq \alpha_i + \sum_{h} \beta^v_{i,h} \cdot v_h + \sum_{h,\ell} \beta^{lag}_{i,h,\ell} \cdot a_{h,\ell} \quad \forall i
$$

The cut coefficients $(\alpha_i, \beta^v_{i,h}, \beta^{lag}_{i,h,\ell})$ are the **undiscounted** values from the backward pass. Cuts are stored and managed in undiscounted form — the discount factor appears only in the objective coefficient of $\theta$. See [Cut Management](cut-management.md) for cut generation and aggregation details.

> **Why discount on $\theta$, not on cuts**: Applying the discount factor to the $\theta$ variable rather than scaling each cut individually is simpler and avoids modifying cut coefficients. The mathematical result is identical — if $\theta \geq \alpha + \beta^\top x$ and $d \cdot \theta$ appears in the objective, it is equivalent to having $\theta' \geq d \cdot (\alpha + \beta^\top x)$ with $\theta'$ in the objective.

## 5 Cumulative Discounting

For a path from stage 1 to stage $T$, the cumulative discount factor is:

$$
d_{1 \to T} = \prod_{t=1}^{T-1} d_{t \to t+1}
$$

The present value at stage 1 of costs incurred at stage $T$ is:

$$
\text{PV}_1[c_T] = d_{1 \to T} \cdot c_T
$$

where $d_{1 \to 1} = 1$.

## 6 Lower Bound with Discounting

The deterministic lower bound at iteration $k$ is:

$$
\underline{z}^k = c_1(\hat{x}_1^k) + d_{1 \to 2} \cdot \theta_1^k
$$

where $\theta_1^k$ is the optimal value of the future cost variable at stage 1. Because the discount factor cascades through $\theta$ at each stage, the lower bound already reflects cumulative discounting to stage 1 present value.

## 7 Upper Bound (Simulation) with Discounting

When simulating the policy to estimate the upper bound:

$$
\bar{z}^k = \frac{1}{M} \sum_{m=1}^{M} \sum_{t=1}^{T} d_{1 \to t} \cdot c_t(\hat{x}_t^{k,m})
$$

Each stage's immediate cost is explicitly discounted to stage 1 present value using the cumulative discount factor.

For stopping rules that use these bounds, see [Stopping Rules](stopping-rules.md).

## 8 Reporting

Both the lower bound $\underline{z}$ and upper bound $\bar{z}$ represent total expected cost expressed in **present value at stage 1**:

- A lower bound of \$100M means the optimal policy costs at least \$100M in stage-1 dollars
- Future costs are already discounted: a \$1M cost at stage 12 with $d_{1 \to 12} = 0.95$ contributes \$0.95M to the bounds
- Comparisons between bounds and between iterations are valid because they use consistent discounting
- When reporting per-stage costs in simulation outputs, POWE.RS reports both **nominal** (undiscounted) and **present value** costs

## 9 Infinite Horizon Considerations

For cyclic policy graphs (infinite periodic horizon), discounting is **required** for convergence. The cumulative discount around one full cycle must satisfy:

$$
d_{cycle} = \prod_{t \in \text{cycle}} d_{t \to t+1} < 1
$$

This ensures the value function remains finite: $\lim_{n \to \infty} d_{cycle}^n \cdot V_t(x) = 0$.

**Typical setup**: An annual discount rate of 6% gives $d_{cycle} \approx 0.94$ per 12-month cycle.

For the complete infinite horizon formulation — including cut sharing within cycles, modified forward/backward passes, and convergence criteria — see [Infinite Horizon](infinite-horizon.md).

## Cross-References

- [Input Scenarios §1.2](../02-data-model/input-scenarios.md) — `policy_graph` schema with `annual_discount_rate` and per-transition overrides
- [SDDP Algorithm](sddp-algorithm.md) — Core Bellman equation and forward/backward pass structure that discount rates modify
- [Cut Management](cut-management.md) — Cut coefficients remain undiscounted; discount applied to $\theta$ in objective
- [Stopping Rules](stopping-rules.md) — Convergence criteria using discounted lower/upper bounds
- [Upper Bound Evaluation](upper-bound-evaluation.md) — Inner approximation uses discounted vertex values
- [Infinite Horizon](infinite-horizon.md) — Cycle detection, cut sharing, modified passes, convergence for periodic problems
- [Configuration Reference](../05-config/configuration-reference.md) — `stages.json` transition discount rates
