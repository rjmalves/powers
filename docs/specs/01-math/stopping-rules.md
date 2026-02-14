---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §13 (13.1-13.8) Stopping Rules Evaluation"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Stopping Rules

## Purpose

This spec defines all available stopping rules for the POWE.RS SDDP solver, their mathematical formulations, configuration, and how they combine. It covers iteration limits, time limits, statistical stopping, bound stalling, and the recommended simulation-based stopping criterion.

## 13.1 Available Stopping Rules

SDDP can terminate based on multiple criteria. Each rule is evaluated independently, and the `stopping_mode` determines how they combine:

- `"any"`: Stop when **any** rule triggers (OR logic)

- `"all"`: Stop when **all** rules trigger (AND logic)

## 13.2 Iteration Limit (Mandatory)

**Configuration**:

```json
{ "type": "iteration_limit", "limit": 50 }
```

**Evaluation**:

$$
\text{STOP} \iff k \geq k_{max}
$$

where $k$ is the current iteration and $k_{max}$ is the limit.

**Purpose**: Safety bound to prevent infinite loops. **Must always be included.**

## 13.3 Time Limit

**Configuration**:

```json
{ "type": "time_limit", "seconds": 3600 }
```

**Evaluation**:

$$
\text{STOP} \iff t_{elapsed} \geq t_{max}
$$

**Implementation**: Check wall-clock time at end of each iteration.

## 13.4 Statistical Stopping

**Configuration**:

```json
{
  "type": "statistical",
  "num_replications": 100,
  "iteration_period": 5,
  "z_score": 1.96
}
```

**Algorithm**:

1. Every `iteration_period` iterations, run `num_replications` Monte Carlo simulations using the current policy

2. Compute sample statistics:

   $$
   \bar{z} = \frac{1}{M} \sum_{m=1}^{M} z^{(m)}
   $$

   $$
   s_z = \sqrt{\frac{1}{M-1} \sum_{m=1}^{M} (z^{(m)} - \bar{z})^2}
   $$

   where $z^{(m)}$ is the total cost of simulation $m$.

3. Compute confidence interval half-width:

   $$
   w = z_{\alpha/2} \cdot \frac{s_z}{\sqrt{M}}
   $$

   where $z_{\alpha/2}$ is the z-score (1.96 for 95% confidence).

4. **Stopping condition** (for minimization):

   $$
   \text{STOP} \iff \underline{z}^k \geq \bar{z} - w
   $$

   i.e., the deterministic lower bound lies within the confidence interval of the simulated upper bound.

**Interpretation**: With confidence $1 - \alpha$, the optimal value lies in $[\underline{z}^k, \bar{z} + w]$. If $\underline{z}^k \geq \bar{z} - w$, the gap is statistically zero.

> **Warning: Not Recommended for Production Use**
>
> The statistical stopping rule has several serious limitations:
>
> 1. **Half-width depends on replications**: With more replications $M$, the half-width $w \propto 1/\sqrt{M}$ shrinks, making the stopping condition easier to satisfy. This creates a perverse incentive: fewer replications = faster "convergence."
> 2. **Non-normality**: Cost distributions in stochastic optimization are often heavy-tailed or multimodal, violating the normality assumption underlying the z-score confidence interval.
> 3. **Sequential testing inflation**: Testing the stopping condition repeatedly at each period inflates the Type I error rate. The nominal 95% confidence level does not hold under repeated testing.
> 4. **Risk-averse incompatibility**: For risk-averse problems, the lower bound is not valid (see [Risk Measures](risk-measures.md)), making this rule fundamentally flawed.
>
> **Recommended Alternatives**:
>
> - Use **bound_stalling** (§13.5) for deterministic convergence monitoring
> - Use **simulation** stopping (§13.6) with explicit gap tolerance
> - For risk-averse problems, rely on iteration limits combined with policy stability metrics

## 13.5 Bound Stalling

**Configuration**:

```json
{
  "type": "bound_stalling",
  "iterations": 10,
  "tolerance": 0.0001
}
```

**Evaluation**:

Track the deterministic lower bound $\underline{z}^k$ over iterations. Compute relative improvement:

$$
\Delta_k = \frac{\underline{z}^k - \underline{z}^{k-\tau}}{\max(1, |\underline{z}^k|)}
$$

where $\tau$ is the `iterations` parameter.

**Stopping condition**:

$$
\text{STOP} \iff |\Delta_k| < \text{tolerance} \text{ for consecutive } \tau \text{ iterations}
$$

**Interpretation**: The bound has plateaued—further iterations provide diminishing returns.

## 13.6 Simulation-Based Stopping (Recommended)

**Configuration**:

```json
{
  "type": "simulation",
  "replications": 100,
  "period": 20,
  "distance_tol": 0.01,
  "bound_tol": 0.0001
}
```

**Algorithm**:

1. **Check bound stability first**:

   $$
   \text{Bound stable} \iff \left| \underline{z}^k - \underline{z}^{k-5} \right| < \text{bound\_tol} \times \max(1, |\underline{z}^k|)
   $$

2. **If bound is stable**, run simulation and compare to previous simulation:

   Let $\mathbf{x}^{new}$ and $\mathbf{x}^{old}$ be vectors of simulated state trajectories (or costs per stage).

   Compute normalized distance:

   $$
   d = \sqrt{\sum_i \left( \frac{x_i^{new} - x_i^{old}}{\max(1, |x_i^{old}|)} \right)^2}
   $$

3. **Stopping condition**:
   $$
   \text{STOP} \iff \text{Bound stable} \land d < \text{distance\_tol}
   $$

**Interpretation**: Both the outer approximation (bound) and the policy (simulated decisions) have stabilized.

**Why recommended**: Combines theoretical convergence indicator (bound) with practical policy quality (simulation), avoiding premature termination from statistical noise.

## 13.7 Combining Rules

**Mode: `"any"` (default)**:

$$
\text{STOP} \iff \text{Rule}_1 \lor \text{Rule}_2 \lor \ldots
$$

First rule to trigger causes termination.

**Mode: `"all"`**:

$$
\text{STOP} \iff \text{Rule}_1 \land \text{Rule}_2 \land \ldots
$$

All rules must trigger simultaneously.

**Example** (conservative setup):

```json
{
  "stopping_rules": [
    { "type": "iteration_limit", "limit": 500 },
    {
      "type": "simulation",
      "replications": 100,
      "period": 20,
      "distance_tol": 0.01,
      "bound_tol": 0.0001
    }
  ],
  "stopping_mode": "any"
}
```

This runs until simulation-based convergence OR 500 iterations, whichever comes first.

## 13.8 Output on Termination

When any stopping rule triggers, the output includes:

| Field             | Description                                |
| ----------------- | ------------------------------------------ | ------- | ----------- |
| `stopping_rule`   | Which rule triggered                       |
| `final_iteration` | Iteration count at termination             |
| `lower_bound`     | Final deterministic lower bound            |
| `upper_bound`     | Final simulated upper bound (if available) |
| `gap_percent`     | $(\bar{z} - \underline{z}) /               | \bar{z} | \times 100$ |

## Cross-References

- [Notation Conventions](../00-overview/notation-conventions.md) — Symbol definitions for bounds and statistical quantities
- [SDDP Algorithm](sddp-algorithm.md) — Main iteration loop that evaluates stopping rules
- [Cut Management](cut-management.md) — Cut generation and selection that affect convergence speed
- [Configuration Reference](../05-config/configuration-reference.md) — JSON schema for `stopping_rules` and `stopping_mode`
