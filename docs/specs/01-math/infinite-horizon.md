---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §15 (15.1-15.10) Infinite Periodic Horizon"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: "Extracted from discount-rate.md during review. Approved as standalone spec."
change_log:
  - date: 2026-02-22
    description: "Extracted from discount-rate.md. Rewritten with $d$ discount symbol, behavioral descriptions (removed pseudocode), aligned config with approved input-scenarios.md policy_graph schema."
---

# Infinite Periodic Horizon

## Purpose

This spec defines the infinite periodic horizon formulation for POWE.RS SDDP: the periodic structure, convergence requirements, cut sharing within cycles, modified forward and backward pass behavior, and convergence criteria. This formulation eliminates "end-of-world" effects where finite-horizon SDDP empties reservoirs near the terminal stage.

For the discount factor mechanics that make infinite horizon convergence possible, see [Discount Rate](discount-rate.md).

> **Symbol convention**: This spec uses $d$ for the discount factor and $\beta$ for cut coefficients.

## 1 Motivation

Standard finite-horizon SDDP has a terminal condition $V_{T+1}(x) = 0$, causing the algorithm to attribute zero value to stored water at the end of the horizon. For long-term planning, an **infinite periodic horizon** better represents the ongoing nature of hydrothermal operations by allowing the value function to reflect perpetual future use.

## 2 Periodic Structure

Consider a system with $P$ stages per cycle (e.g., 12 monthly stages). Let $\tau(t)$ denote the **season** (position within the cycle) for stage $t$:

$$
\tau(t) = (t - 1) \mod P + 1 \in \{1, 2, \ldots, P\}
$$

Stages with the same season share structural properties: demand patterns, inflow statistics, block definitions, and stochastic process parameters.

## 3 Cyclic Policy Graph

A cyclic policy graph is defined when a transition in `stages.json` points from a later stage back to an earlier one, forming a cycle. The `policy_graph` type must be `"cyclic"`:

```json
{
  "policy_graph": {
    "type": "cyclic",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 59, "target_id": 48, "probability": 1.0 }
    ]
  }
}
```

In this example, stage 59 transitions back to stage 48, creating a 12-stage cycle (stages 48-59).

For the complete `policy_graph` schema and per-transition discount rate overrides, see [Input Scenarios §1.2](../02-data-model/input-scenarios.md).

## 4 Discount Requirement for Convergence

For the value function to remain finite, the cumulative discount around one full cycle must be strictly less than 1:

$$
d_{cycle} = \prod_{t \in \text{cycle}} d_{t \to t+1} < 1
$$

This ensures:

$$
\lim_{n \to \infty} d_{cycle}^n \cdot V_t(x) = 0
$$

**Typical setup**: A 6% annual discount rate gives $d_{cycle} \approx 0.94$ per 12-month cycle.

> **Validation**: The system rejects cyclic policy graphs where the cumulative cycle discount factor is $\geq 1$. See [Input Scenarios §1.2](../02-data-model/input-scenarios.md).

## 5 Cut Sharing Within Cycles

Stages at the same position within the cycle share their value function approximation. Let $\mathcal{C}_\tau = \{t : \tau(t) = \tau\}$ be all stages with season $\tau$.

A cut generated at any stage $t \in \mathcal{C}_\tau$ is valid for all stages in $\mathcal{C}_\tau$:

$$
\underline{V}_\tau(x) = \max_{k \in \mathcal{K}_\tau} \left\{ \alpha_k + \beta_k^\top x \right\}
$$

The cut pool is organized by season $\tau \in \{1, \ldots, P\}$, not by absolute stage ID. This means a single cycle's worth of cut pools represents the entire infinite horizon.

## 6 Forward Pass Behavior

In infinite horizon, the forward pass simulates the policy over multiple cycles until the discounted contribution becomes negligible:

- The forward pass starts at the cycle entry point and proceeds through stages
- At each stage, the immediate cost is accumulated with cumulative discounting: $d_{1 \to t} \cdot c_t$
- When the pass reaches the end of the cycle, it wraps back to the cycle start
- The pass terminates when either:
  - The cumulative discount factor drops below a tolerance threshold (the remaining contribution is negligible)
  - A maximum number of stages is reached (safety bound, e.g., 240 stages = 20 years for monthly stages)

The `max_horizon_length` parameter provides the safety bound.

## 7 Backward Pass Behavior

The backward pass generates cuts for each season in the cycle:

- Cuts are generated using the same mechanics as finite horizon (see [Cut Management](cut-management.md))
- Each cut is added to the season's cut pool, applicable to all stages at that cycle position
- The backward pass completes a full cycle per iteration

**Convergence of the backward pass**: The outer approximation has converged when the value functions are stable across consecutive cycles:

$$
\max_{\tau \in \{1, \ldots, P\}} \left| \underline{z}^{k,\tau} - \underline{z}^{k-P,\tau} \right| < \delta_{cycle}
$$

where $\delta_{cycle}$ is the `cycle_convergence_tolerance` parameter.

## 8 Fixed-Point Interpretation

The infinite-horizon SDDP finds the fixed point of the Bellman operator:

$$
V_\tau = T_\tau V_{\tau+1}
$$

where $T_\tau$ is the one-stage Bellman operator for season $\tau$:

$$
(T_\tau V)(x) = \mathbb{E}_{\omega_\tau}\left[\min_{x'} \left\{ c_\tau(x', u) + d \cdot V(x') \right\}\right]
$$

Convergence is achieved when the value functions at all seasons stabilize — the outer approximation is a sufficiently tight lower bound on the true fixed point.

## 9 Reference

> Costa, B.S., de Matos, V.L., Philpott, A.B. (2025). "SDDP.jl approaches for infinite horizon problems." _Trends in Computational and Applied Mathematics_, 11(1). https://doi.org/10.5540/03.2025.011.01.0355

## Cross-References

- [Discount Rate](discount-rate.md) — Discount factor mechanics, Bellman equation, cumulative discounting
- [Input Scenarios §1.2](../02-data-model/input-scenarios.md) — `policy_graph` schema with `type: "cyclic"`, `annual_discount_rate`, and transition definitions
- [SDDP Algorithm §4.2](sddp-algorithm.md) — High-level overview of cyclic policy graphs
- [Cut Management](cut-management.md) — Cut generation and aggregation (undiscounted cuts, discount on $\theta$)
- [Stopping Rules](stopping-rules.md) — Convergence criteria using discounted bounds
- [Configuration Reference](../05-config/configuration-reference.md) — Horizon and cycle configuration parameters
