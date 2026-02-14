---
status: draft
review_priority: 3-medium
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §16 (16.1-16.12) Upper Bound Evaluation LP (Inner Approximation / SIDP)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Upper Bound Evaluation

## Purpose

This spec defines the upper bound evaluation mechanism in POWE.RS via inner approximation (SIDP): the vertex-based value function approximation, Lipschitz interpolation, the linearized upper bound LP, gap computation, and configuration. It complements the outer approximation (cuts) described in [SDDP Algorithm](sddp-algorithm.md) by providing a convergence certificate through deterministic upper bounds.

For notation conventions (index sets, parameters, decision variables, dual variables), see [Notation Conventions](../00-overview/notation-conventions.md).

## 16.1 Motivation

Standard SDDP provides only a **lower bound** (outer approximation) through cuts. For convergence verification, we need an **upper bound** (inner approximation). This is especially important for:

1. **Risk-averse problems**: CVaR objectives cannot be estimated via Monte Carlo
2. **Convergence certificates**: Gap $= \bar{z} - \underline{z}$ provides true optimality measure
3. **Conservative policies**: Inner approximation gives "at most $Y$" guarantees

## 16.2 Vertex-Based Inner Approximation

The inner approximation $\bar{V}_t(x)$ is constructed from **vertices** (visited state-value pairs):

$$
\mathcal{V}_t = \{(x^{(1)}, \bar{v}^{(1)}), (x^{(2)}, \bar{v}^{(2)}), \ldots, (x^{(n)}, \bar{v}^{(n)})\}
$$

where each vertex stores:

- $x^{(i)}$: State vector visited during forward passes
- $\bar{v}^{(i)}$: Upper bound on cost-to-go from that state (computed recursively)

## 16.3 Lipschitz Interpolation

For a new state $x$ not in $\mathcal{V}_t$, the upper bound is computed via Lipschitz interpolation:

$$
\bar{V}_t(x) = \min_{(x^{(i)}, \bar{v}^{(i)}) \in \mathcal{V}_t} \left\{ \bar{v}^{(i)} + L_t \cdot \|x - x^{(i)}\|_1 \right\}
$$

where $L_t$ is the Lipschitz constant for stage $t$.

**Interpretation**: The upper bound at $x$ is the minimum over all vertices of "vertex value plus distance penalty."

## 16.4 Lipschitz Constant Computation

The Lipschitz constant bounds the maximum rate of change of the value function. For SDDP with penalty-based feasibility:

**Backward accumulation**:

$$
L_T = c_{max}^{penalty}
$$

$$
L_t = L_{t+1} + c_{max}^{penalty,t}
$$

where $c_{max}^{penalty,t}$ is the maximum penalty coefficient at stage $t$ (e.g., deficit penalty in \$/MWh).

**Example**: With deficit penalty $1000$ \$/MWh over 5 stages:

| Stage $t$ | Lipschitz $L_t$ |
| --------- | --------------- |
| 5         | 1000            |
| 4         | 2000            |
| 3         | 3000            |
| 2         | 4000            |
| 1         | 5000            |

## 16.5 Vertex Value Computation

During upper bound evaluation (backward pass variant):

**At terminal stage $T$**:

$$
\bar{v}^{(i)} = c_T(x^{(i)}) \quad \text{(immediate cost only)}
$$

**At stage $t < T$**:

1. For each vertex $(x^{(i)}, \cdot) \in \mathcal{V}_t$:
2. Solve the stage subproblem with $x = x^{(i)}$
3. Compute future cost using inner approximation at stage $t+1$:

$$
\bar{\theta} = \bar{V}_{t+1}(x_{t+1}^*)
$$

4. Set vertex value:

$$
\bar{v}^{(i)} = c_t(x^{(i)}) + \beta \cdot \bar{\theta}
$$

where $\beta$ is the discount factor (see [Discount Rate](discount-rate.md) §14.2).

## 16.6 Upper Bound Evaluation LP

For policy simulation with inner approximation, the stage LP is modified.

**Standard LP (outer approximation)**:

$$
\min c_t^\top x_t + \theta
$$

$$
\text{s.t. } \theta \geq \alpha_k + \beta_k^\top x_t \quad \forall k \text{ (cuts)}
$$

**Inner approximation LP**:

$$
\min c_t^\top x_t + \bar{\theta}
$$

$$
\text{s.t. } \bar{\theta} \leq \bar{v}^{(i)} + L_t \sum_j |x_{t,j} - x_j^{(i)}| \quad \forall i \text{ (vertices)}
$$

The absolute value can be linearized using standard techniques:

$$
|x_j - x_j^{(i)}| = u_j^{(i)+} + u_j^{(i)-}
$$

$$
x_j - x_j^{(i)} = u_j^{(i)+} - u_j^{(i)-}
$$

$$
u_j^{(i)+}, u_j^{(i)-} \geq 0
$$

## 16.7 Linearized Upper Bound LP

**Additional Variables** (per vertex $i$, per state component $j$):

| Variable       | Domain   | Description                                         |
| -------------- | -------- | --------------------------------------------------- |
| $u_j^{(i)+}$   | $\geq 0$ | Positive deviation from vertex $i$ in dimension $j$ |
| $u_j^{(i)-}$   | $\geq 0$ | Negative deviation from vertex $i$ in dimension $j$ |
| $\bar{\theta}$ | free     | Upper bound on future cost                          |

**Constraints**:

For each vertex $i \in \mathcal{V}_t$:

$$
\bar{\theta} \leq \bar{v}^{(i)} + L_t \sum_j (u_j^{(i)+} + u_j^{(i)-})
$$

$$
x_j - x_j^{(i)} = u_j^{(i)+} - u_j^{(i)-} \quad \forall j
$$

See [SDDP Algorithm](sddp-algorithm.md) for the standard LP formulation that provides the outer approximation counterpart.

## 16.8 Gap Computation

At each iteration $k$ (when upper bound is evaluated):

**Lower bound** (from cuts at stage 1):

$$
\underline{z}^k = c_1(\hat{x}_1) + \underline{V}_2(\hat{x}_1)
$$

**Upper bound** (from vertices):

$$
\bar{z}^k = c_1(\hat{x}_1) + \bar{V}_2(\hat{x}_1)
$$

**Gap**:

$$
\text{gap}^k = \frac{\bar{z}^k - \underline{z}^k}{\max(1, |\bar{z}^k|)} \times 100\%
$$

**Convergence**: As $k \to \infty$, $\text{gap}^k \to 0$ for convex problems with finitely many scenarios.

For stopping rules that use the gap, see [Stopping Rules](stopping-rules.md).

## 16.9 Vertex Storage

Vertices are stored in `policy/vertices/stage_XXX.bin` with the following schema:

| Field       | Type             | Description                                 |
| ----------- | ---------------- | ------------------------------------------- |
| `state`     | `[f64; n_state]` | State vector at vertex                      |
| `value`     | `f64`            | Upper bound value at vertex                 |
| `iteration` | `u32`            | Iteration when vertex was created           |
| `lipschitz` | `f64`            | Per-vertex Lipschitz constant (if variable) |

## 16.10 Configuration

```json
{
  "upper_bound_evaluation": {
    "enabled": true,
    "initial_iteration": 10,
    "interval_iterations": 5,
    "lipschitz": {
      "mode": "auto",
      "fallback_value": 10000.0,
      "scale_factor": 1.1
    }
  }
}
```

| Parameter                  | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `enabled`                  | Enables upper bound evaluation                    |
| `initial_iteration`        | First iteration to compute upper bound            |
| `interval_iterations`      | Evaluate every N iterations after initial         |
| `lipschitz.mode`           | `"auto"` computes from penalty coefficients       |
| `lipschitz.fallback_value` | Used when auto-computation is not possible        |
| `lipschitz.scale_factor`   | Safety multiplier for computed Lipschitz constant |

For full configuration schema, see [Configuration Reference](../05-config/configuration-reference.md).

## 16.11 Computational Considerations

| Aspect                   | Impact                                                                   |
| ------------------------ | ------------------------------------------------------------------------ |
| **Vertices per stage**   | Typically $\mathcal{O}(\text{iterations} \times \text{forward\_passes})$ |
| **LP size increase**     | $2 \times n_{state} \times n_{vertices}$ additional variables            |
| **Evaluation frequency** | Trade-off between gap accuracy and runtime                               |
| **Memory**               | Vertices stored separately from cuts                                     |

**Recommendation**: Enable upper bound evaluation every 5-10 iterations after initial burn-in period (10+ iterations) for convergence monitoring without excessive overhead.

## 16.12 References

> Costa, B.S., & Leclère, V. (2023). "Lipschitz-based Inner Approximation of Risk Measures." _Optimization Online_. https://optimization-online.org/?p=23738

> Philpott, A.B., de Matos, V.L., & Finardi, E.C. (2013). "On solving multistage stochastic programs with coherent risk measures." _Operations Research_, 61(4), 957-970. https://doi.org/10.1287/opre.2013.1200

## Cross-References

- [SDDP Algorithm](sddp-algorithm.md) — Core algorithm providing the outer approximation (lower bound) that this spec complements
- [Notation Conventions](../00-overview/notation-conventions.md) — Standard symbols for state variables, value functions, and cost-to-go
- [Discount Rate](discount-rate.md) — Discount factor $\beta$ used in vertex value computation (§16.5) and infinite periodic horizon
- [Cut Management](cut-management.md) — Outer approximation cuts that provide the lower bound counterpart
- [Stopping Rules](stopping-rules.md) — Convergence criteria that use the gap between inner and outer approximations
- [Configuration Reference](../05-config/configuration-reference.md) — Full `upper_bound_evaluation` configuration schema
