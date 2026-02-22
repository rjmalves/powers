---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §16 (16.1-16.12) Upper Bound Evaluation LP (Inner Approximation / SIDP)"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: "Rewritten during review: renumbered §1-11, fixed discount symbol d, removed duplicated vertex schema and config (→ binary-formats.md, input-directory-structure.md), clarified deterministic vs simulation-based UB, added expectation in vertex values, added infinite horizon note."
change_log:
  - date: 2026-02-22
    description: "Renumbered §16.x → §1-11. Fixed discount symbol β → d. Removed duplicated vertex schema (→ binary-formats.md) and duplicated config (→ input-directory-structure.md). Clarified deterministic vs simulation-based upper bounds. Added expectation clarification for vertex values. Added infinite horizon note. Fixed stale cross-references."
---

# Upper Bound Evaluation

## Purpose

This spec defines the upper bound evaluation mechanism in POWE.RS via inner approximation (SIDP): the vertex-based value function approximation, Lipschitz interpolation, the linearized upper bound LP, gap computation, and configuration. It complements the outer approximation (cuts) described in [SDDP Algorithm](sddp-algorithm.md) by providing a convergence certificate through deterministic upper bounds.

For notation conventions (index sets, parameters, decision variables, dual variables), see [Notation Conventions](../00-overview/notation-conventions.md).

> **Symbol convention**: This spec uses $d$ for the discount factor (not $\beta$, which denotes cut coefficients). See [Discount Rate](discount-rate.md).

## 1 Motivation

Standard SDDP provides only a **lower bound** (outer approximation) through cuts. For convergence verification, we need an **upper bound** (inner approximation). This is especially important for:

1. **Risk-averse problems**: CVaR objectives cannot be reliably estimated via Monte Carlo simulation of the policy
2. **Convergence certificates**: Gap $= \bar{z} - \underline{z}$ provides a true optimality measure
3. **Conservative policies**: Inner approximation gives "at most $Y$" guarantees

> **Deterministic vs. simulation-based upper bounds**: The simulation-based stopping rules (see [Stopping Rules](stopping-rules.md)) estimate an upper bound by running the SDDP policy on sampled scenarios and averaging costs. This is a **statistical** estimate — valid for risk-neutral problems but not for risk-averse (CVaR) objectives. The inner approximation described here provides a **deterministic** upper bound that is valid regardless of the risk measure.

## 2 Vertex-Based Inner Approximation

The inner approximation $\bar{V}_t(x)$ is constructed from **vertices** (visited state-value pairs):

$$
\mathcal{V}_t = \{(x^{(1)}, \bar{v}^{(1)}), (x^{(2)}, \bar{v}^{(2)}), \ldots, (x^{(n)}, \bar{v}^{(n)})\}
$$

where each vertex stores:

- $x^{(i)}$: State vector visited during forward passes
- $\bar{v}^{(i)}$: Upper bound on expected cost-to-go from that state (computed recursively)

## 3 Lipschitz Interpolation

For a new state $x$ not in $\mathcal{V}_t$, the upper bound is computed via Lipschitz interpolation:

$$
\bar{V}_t(x) = \min_{(x^{(i)}, \bar{v}^{(i)}) \in \mathcal{V}_t} \left\{ \bar{v}^{(i)} + L_t \cdot \|x - x^{(i)}\|_1 \right\}
$$

where $L_t$ is the Lipschitz constant for stage $t$.

**Interpretation**: The upper bound at $x$ is the minimum over all vertices of "vertex value plus distance penalty." This forms a concave piecewise-linear function — the inner (concave) counterpart to the outer (convex) cut approximation.

## 4 Lipschitz Constant Computation

The Lipschitz constant bounds the maximum rate of change of the value function with respect to the state. For SDDP with penalty-based feasibility (relatively complete recourse):

**Backward accumulation**:

$$
L_T = c_{max}^{penalty}
$$

$$
L_t = d_{t \to t+1} \cdot L_{t+1} + c_{max}^{penalty,t}
$$

where:

- $c_{max}^{penalty,t}$ is the maximum penalty coefficient at stage $t$ (e.g., deficit penalty in \$/MWh)
- $d_{t \to t+1}$ is the discount factor for transition $t \to t+1$ (see [Discount Rate](discount-rate.md))

> **Note**: The discount factor $d$ appears in the Lipschitz accumulation because the future cost is discounted. Without discounting ($d = 1$), $L_t$ grows linearly with the remaining horizon.

**Example**: With deficit penalty $1000$ \$/MWh over 5 stages, no discounting:

| Stage $t$ | Lipschitz $L_t$ |
| --------- | --------------- |
| 5         | 1,000           |
| 4         | 2,000           |
| 3         | 3,000           |
| 2         | 4,000           |
| 1         | 5,000           |

## 5 Vertex Value Computation

During the upper bound evaluation pass (a backward pass variant):

**At terminal stage $T$**:

$$
\bar{v}^{(i)} = \mathbb{E}_{\omega_T}\left[c_T(x^{(i)}, \omega_T)\right] \quad \text{(expected immediate cost only)}
$$

**At stage $t < T$**:

For each vertex $(x^{(i)}, \cdot) \in \mathcal{V}_t$:

1. For each scenario $\omega_t$, solve the stage subproblem with incoming state $x^{(i)}$ and realization $\omega_t$
2. Obtain the optimal next-stage state $x_{t+1}^*(\omega_t)$
3. Evaluate the inner approximation at the next stage: $\bar{\theta}(\omega_t) = \bar{V}_{t+1}(x_{t+1}^*(\omega_t))$
4. Set vertex value as the expected discounted cost-to-go:

$$
\bar{v}^{(i)} = \mathbb{E}_{\omega_t}\left[c_t(x^{(i)}, \omega_t) + d_{t \to t+1} \cdot \bar{\theta}(\omega_t)\right]
$$

> **Expectation**: The vertex value is an expectation over scenarios, not a single-scenario value. This parallels the backward pass for cuts, which also computes expected cost-to-go.

## 6 Upper Bound Evaluation LP

For policy evaluation with inner approximation, the stage LP replaces the outer approximation (cut constraints on $\theta$) with inner approximation (vertex constraints on $\bar{\theta}$).

**Standard LP (outer approximation — lower bound)**:

$$
\min \; c_t^\top x_t + d_{t \to t+1} \cdot \theta
$$

$$
\text{s.t. } \theta \geq \alpha_k + \beta_k^\top x_t \quad \forall k \text{ (cuts)}
$$

**Inner approximation LP (upper bound)**:

$$
\min \; c_t^\top x_t + d_{t \to t+1} \cdot \bar{\theta}
$$

$$
\text{s.t. } \bar{\theta} \leq \bar{v}^{(i)} + L_t \sum_j |x_{t,j} - x_j^{(i)}| \quad \forall i \in \mathcal{V}_t \text{ (vertices)}
$$

> **Direction**: Cut constraints are **lower bounds** on $\theta$ ($\theta \geq \ldots$). Vertex constraints are **upper bounds** on $\bar{\theta}$ ($\bar{\theta} \leq \ldots$). The cut approximation is convex (piecewise-linear from below); the vertex approximation is concave (piecewise-linear from above).

## 7 Linearized Upper Bound LP

The absolute value $|x_j - x_j^{(i)}|$ in the vertex constraints is linearized using standard splitting:

$$
|x_j - x_j^{(i)}| = u_j^{(i)+} + u_j^{(i)-}
$$

$$
x_j - x_j^{(i)} = u_j^{(i)+} - u_j^{(i)-}
$$

$$
u_j^{(i)+}, u_j^{(i)-} \geq 0
$$

**Additional variables** (per vertex $i$, per state component $j$):

| Variable       | Domain   | Description                                         |
| -------------- | -------- | --------------------------------------------------- |
| $u_j^{(i)+}$   | $\geq 0$ | Positive deviation from vertex $i$ in dimension $j$ |
| $u_j^{(i)-}$   | $\geq 0$ | Negative deviation from vertex $i$ in dimension $j$ |
| $\bar{\theta}$ | free     | Upper bound on future cost                          |

**Constraints** (for each vertex $i \in \mathcal{V}_t$):

$$
\bar{\theta} \leq \bar{v}^{(i)} + L_t \sum_j (u_j^{(i)+} + u_j^{(i)-})
$$

$$
x_j - x_j^{(i)} = u_j^{(i)+} - u_j^{(i)-} \quad \forall j
$$

## 8 Gap Computation

At each iteration $k$ where the upper bound is evaluated:

**Lower bound** (from cuts at stage 1):

$$
\underline{z}^k = c_1(\hat{x}_1) + d_{1 \to 2} \cdot \underline{V}_2(\hat{x}_1)
$$

**Upper bound** (from vertices at stage 1):

$$
\bar{z}^k = c_1(\hat{x}_1) + d_{1 \to 2} \cdot \bar{V}_2(\hat{x}_1)
$$

**Relative gap**:

$$
\text{gap}^k = \frac{\bar{z}^k - \underline{z}^k}{\max(1, |\bar{z}^k|)} \times 100\%
$$

**Convergence**: As $k \to \infty$, $\text{gap}^k \to 0$ for convex problems with finitely many scenarios.

For stopping rules that use the gap, see [Stopping Rules](stopping-rules.md).

## 9 Computational Considerations

| Aspect                   | Impact                                                                   |
| ------------------------ | ------------------------------------------------------------------------ |
| **Vertices per stage**   | Typically $\mathcal{O}(\text{iterations} \times \text{forward\_passes})$ |
| **LP size increase**     | $2 \times n_{state} \times n_{vertices}$ additional variables            |
| **Evaluation frequency** | Trade-off between gap accuracy and runtime                               |
| **Memory**               | Vertices stored separately from cuts                                     |

**Recommendation**: Enable upper bound evaluation every 5–10 iterations after an initial burn-in period (10+ iterations) for convergence monitoring without excessive overhead.

## 10 Infinite Horizon

For cyclic policy graphs (see [Infinite Horizon](infinite-horizon.md)), the inner approximation operates on the same seasonal cut-pool structure: vertices are organized by season $\tau$, not by absolute stage ID. The Lipschitz constant must account for the cumulative discount around the cycle, which bounds the geometric series of future contributions.

The convergence guarantee still holds: with $d_{cycle} < 1$, both the outer (cut) and inner (vertex) approximations converge to the true value function at the fixed point.

## 11 References

> Costa, B.S., & Leclère, V. (2023). "Lipschitz-based Inner Approximation of Risk Measures." _Optimization Online_. https://optimization-online.org/?p=23738

> Philpott, A.B., de Matos, V.L., & Finardi, E.C. (2013). "On solving multistage stochastic programs with coherent risk measures." _Operations Research_, 61(4), 957–970. https://doi.org/10.1287/opre.2013.1200

## Cross-References

- [SDDP Algorithm](sddp-algorithm.md) — Core algorithm providing the outer approximation (lower bound) that this spec complements
- [Notation Conventions](../00-overview/notation-conventions.md) — Standard symbols for state variables, value functions, and cost-to-go
- [Discount Rate](discount-rate.md) — Discount factor $d$ used in vertex value computation (§5) and Lipschitz accumulation (§4)
- [Infinite Horizon](infinite-horizon.md) — Seasonal vertex organization for cyclic policy graphs
- [Cut Management](cut-management.md) — Outer approximation cuts that provide the lower bound counterpart
- [Stopping Rules](stopping-rules.md) — Convergence criteria that use the gap between inner and outer approximations
- [Risk Measures](risk-measures.md) — CVaR objectives where deterministic upper bounds are essential
- [Binary Formats §3.3](../02-data-model/binary-formats.md) — FlatBuffers `Vertex` and `StageVertices` schemas for persistence
- [Input Directory Structure §2.2](../02-data-model/input-directory-structure.md) — `upper_bound_evaluation` configuration in `config.json`
- [Configuration Reference](../05-config/configuration-reference.md) — Full configuration schema with defaults
