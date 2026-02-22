---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §17.1 (Motivation)"
  - "MATHEMATICAL_FORMULATIONS.md §17.2 (CVaR Definition)"
  - "MATHEMATICAL_FORMULATIONS.md §17.3 (Convex Combination Risk Measure)"
  - "MATHEMATICAL_FORMULATIONS.md §17.4 (Dual Representation)"
  - "MATHEMATICAL_FORMULATIONS.md §17.5 (Risk-Averse Subgradient Theorem)"
  - "MATHEMATICAL_FORMULATIONS.md §17.6 (Risk-Averse Bellman Equation)"
  - "MATHEMATICAL_FORMULATIONS.md §17.7 (Cut Generation with Risk Measures)"
  - "MATHEMATICAL_FORMULATIONS.md §17.8 (Per-Stage Risk Profiles)"
  - "MATHEMATICAL_FORMULATIONS.md §17.9 (Implementation Notes)"
  - "MATHEMATICAL_FORMULATIONS.md §17.10 (Upper Bound with Risk Measures)"
  - "MATHEMATICAL_FORMULATIONS.md §17.11 (Reference)"
  - "MATHEMATICAL_FORMULATIONS.md §17.12 (Lower Bound Validity)"
last_reviewed: 2026-02-22
reviewed_by: rogerio
review_notes: "Full rewrite. Renumbered standalone §1-11. Discount factor d added to Bellman and cut generation. Symbol α disambiguated (CVaR confidence vs cut intercept â). Sorting-based weight computation replaces LP formulation in §7. Config aligned with approved input-scenarios.md §1.7. 8 cross-references added."
change_log:
  - date: 2026-02-14
    description: "Initial extraction from MATHEMATICAL_FORMULATIONS.md §17"
  - date: 2026-02-22
    description: "Full rewrite: renumbered to standalone §1-11. Fixed review_priority to 2-high. Added discount factor d to Bellman equation and cut generation. Disambiguated α (CVaR confidence) from cut intercept (now â). Renamed dual penalty function to ψ. Fixed config format to match approved input-scenarios.md §1.7. Fixed stale internal cross-references. Added cross-references to upper-bound-evaluation.md, input-scenarios.md, infinite-horizon.md, discount-rate.md. Rewrote §7 Step 2: sorting-based greedy weight computation replaces LP formulation."
---

# Risk Measures

## Purpose

This spec defines the risk-averse SDDP formulation used in POWE.RS, based on Conditional Value-at-Risk (CVaR). It covers the CVaR definition, the convex combination risk measure, dual representations, the risk-averse subgradient theorem, modified Bellman equation with discount factor, risk-averse cut generation, per-stage risk profiles, and the critical implications for bound validity.

For notation conventions (index sets, parameters, decision variables, dual variables), see [Notation Conventions](../00-overview/notation-conventions.md).

> **Symbol conventions**:
>
> - $\alpha$ denotes the **CVaR confidence level** (matching the `alpha` field in `stages.json`). This is the standard convention in the risk measure literature.
> - $\hat{\alpha}_t(\omega)$ denotes **per-scenario cut intercepts** within this spec. This corresponds to $\alpha$ in [Cut Management §1](cut-management.md), renamed here to avoid collision with the CVaR parameter.
> - $d$ denotes the **discount factor** (not $\beta$, which denotes cut coefficients). See [Discount Rate](discount-rate.md).
> - $\mu$ denotes the **risk-adjusted probability measure** (not $q$, which denotes turbined flow).
> - $\psi(p, \mu)$ denotes the **dual penalty function** in the general dual representation.

## 1 Motivation

Risk-neutral SDDP minimizes expected cost, which can lead to policies that perform poorly in adverse scenarios. **Risk-averse SDDP** incorporates a coherent risk measure (typically CVaR) to protect against tail risks while maintaining the convexity properties required for valid cut generation.

## 2 Conditional Value-at-Risk (CVaR)

For a random variable $Z$ representing cost and confidence level $\alpha \in (0, 1]$:

$$
\text{CVaR}_\alpha(Z) = \min_{\eta \in \mathbb{R}} \left\{ \eta + \frac{1}{\alpha} \mathbb{E}\left[(Z - \eta)^+\right] \right\}
$$

where $(Z - \eta)^+ = \max(0, Z - \eta)$ captures the excess cost above threshold $\eta$.

**Interpretation**: CVaR$_\alpha$ is the expected cost in the worst $\alpha$-fraction of scenarios.

| $\alpha$ | Risk Posture           | Meaning                                     |
| -------- | ---------------------- | ------------------------------------------- |
| 1.0      | Risk-neutral           | CVaR$_1$ = $\mathbb{E}[Z]$ (expected value) |
| 0.5      | Moderately risk-averse | Average of worst 50% of outcomes            |
| 0.2      | Risk-averse            | Average of worst 20% of outcomes            |
| 0.05     | Highly risk-averse     | Average of worst 5% of outcomes             |

## 3 Convex Combination Risk Measure

POWE.RS uses a convex combination of expectation and CVaR (following the SDDP.jl convention):

$$
\rho^{\lambda, \alpha}[Z] = (1 - \lambda) \mathbb{E}[Z] + \lambda \cdot \text{CVaR}_\alpha[Z]
$$

where:

- $\lambda \in [0, 1]$: Risk aversion weight (0 = risk-neutral, 1 = pure CVaR)
- $\alpha \in (0, 1]$: CVaR confidence level

This is sometimes called the **EAVaR** (Expectation + Average Value-at-Risk) risk measure.

## 4 Dual Representation of Convex Risk Measures

Convex risk measures have a **dual representation** that is essential for computing risk-averse cuts:

$$
\mathbb{F}[Z] = \sup_{\mu \in \mathcal{M}(p)} \mathbb{E}_\mu[Z] - \psi(p, \mu)
$$

where:

- $\mathcal{M}(p) \subseteq \mathcal{P}$ is a convex subset of the probability simplex
- $\psi(p, \mu)$ is a concave penalty function
- $\mathcal{P} = \{p \geq 0 : \sum_{\omega} p_\omega = 1\}$

**Interpretation**: The dual computes the expectation with respect to the **worst** probability vector $\mu$ within the set $\mathcal{M}$, less a penalty term $\psi(p, \mu)$.

### 4.1 CVaR Dual Representation

For CVaR$_\alpha$, the dual representation is:

$$
\text{CVaR}_\alpha[Z] = \sup_{\mu \in \mathcal{M}_\alpha(p)} \mathbb{E}_\mu[Z]
$$

where the risk set $\mathcal{M}_\alpha(p)$ is:

$$
\mathcal{M}_\alpha(p) = \left\{\mu \geq 0 : \sum_\omega \mu_\omega = 1, \; \mu_\omega \leq \frac{p_\omega}{\alpha} \; \forall \omega \right\}
$$

The penalty $\psi(p, \mu) = 0$ for CVaR (no penalty term).

**Interpretation**: CVaR puts more probability weight on the worst outcomes, with each scenario receiving at most $p_\omega / \alpha$ probability mass. For small $\alpha$, only the worst scenarios receive significant weight.

### 4.2 EAVaR Dual Representation

For the convex combination $\rho^{\lambda, \alpha}[Z] = (1-\lambda)\mathbb{E}[Z] + \lambda \cdot \text{CVaR}_\alpha[Z]$:

$$
\mathcal{M}^{EAVaR}(p) = \left\{\mu \geq 0 : \sum_\omega \mu_\omega = 1, \; \mu_\omega \leq (1-\lambda) p_\omega + \frac{\lambda p_\omega}{\alpha} \; \forall \omega \right\}
$$

## 5 Risk-Averse Subgradient Theorem

The key theorem for computing risk-averse cuts:

> **Theorem (Risk-Averse Subgradient)**: Let $V(x, \omega)$ be convex with respect to $x$ for all fixed $\omega \in \Omega$, and let $\lambda(\tilde{x}, \omega)$ be a subgradient of $V(x, \omega)$ at $x = \tilde{x}$.
>
> If $\mu^* = \text{argmax}_{\mu \in \mathcal{M}(p)} \mathbb{E}_\mu[V(\tilde{x}, \omega)] - \psi(p, \mu)$, then:
>
> $$\sum_{\omega \in \Omega} \mu^*_\omega \cdot \lambda(\tilde{x}, \omega)$$
>
> is a subgradient of $\mathbb{F}[V(x, \omega)]$ at $\tilde{x}$.

**Application to Cut Generation**: In SDDP, the subgradients $\lambda(\tilde{x}, \omega)$ are the cut coefficients $\beta_t(\omega)$ obtained from LP duals (see [Cut Management §2](cut-management.md)). The risk-averse cut coefficients are computed by replacing the uniform scenario probabilities with risk-adjusted probabilities $\mu^*$:

$$
\bar{\beta}_{t-1,h} = \sum_{\omega \in \Omega_t} \mu^*_\omega \cdot \beta_{t,h}(\omega)
$$

where $\mu^*$ is the optimal dual probability vector computed from the scenario costs $\{Q_t(\hat{x}, \omega)\}_{\omega \in \Omega_t}$.

## 6 Risk-Averse Bellman Equation

The risk-averse value function with discount factor $d$ satisfies:

$$
V_t(x_{t-1}) = \rho^{\lambda_t, \alpha_t}\left[\min_{x_t} \left\{ c_t^\top x_t + d_{t \to t+1} \cdot V_{t+1}(x_t) : (x_t, x_{t-1}) \text{ feasible} \right\}\right]
$$

This modifies the standard [Bellman recursion](sddp-algorithm.md) in two ways:

1. **Risk measure replaces expectation**: $\rho^{\lambda_t, \alpha_t}[\cdot]$ replaces $\mathbb{E}[\cdot]$
2. **Discount factor on future cost**: $d_{t \to t+1} \cdot V_{t+1}(x_t)$ discounts the cost-to-go (see [Discount Rate §2](discount-rate.md))

> **Time consistency**: The risk measure is applied stage-wise (nested formulation), not to the total cost. This guarantees time consistency, which is essential for the dynamic programming decomposition. Formally: $\rho_1[\rho_2[\cdots \rho_{T-1}[\cdot]]]$, not $\rho[\text{total cost}]$.

In the LP subproblem at stage $t$, the future cost variable $\theta$ appears in the objective as $d_{t \to t+1} \cdot \theta$ (see [Discount Rate §5](discount-rate.md)). Cuts bound $\theta$ (not $d \cdot \theta$), so the discount factor multiplies $\theta$ only in the objective — exactly as in the risk-neutral case.

## 7 Cut Generation with Risk Measures

For each visited state $\hat{x}_{t-1}$, compute the risk-averse cut as follows:

**Step 1: Solve subproblems** for all realizations $\omega \in \Omega_t$:

$$
Q_t(\hat{x}_{t-1}, \omega) = \min_{x_t} \left\{ c_t^\top x_t + d_{t \to t+1} \cdot \theta_t : \text{constraints} \right\}
$$

Extract dual solutions $\pi_t(\omega)$ and compute per-scenario cut coefficients:

- Intercept: $\hat{\alpha}_t(\omega) = Q_t(\hat{x}_{t-1}, \omega) - \beta_t(\omega)^\top \hat{x}_{t-1}$
- Coefficients: $\beta_t(\omega)$ — derived from LP duals (see [Cut Management §2](cut-management.md))

**Step 2: Compute risk-adjusted scenario weights** $\mu^*_\omega$.

Each scenario $\omega$ has a probability upper bound:

$$
\bar{\mu}_\omega = (1 - \lambda_t) \, p_\omega + \frac{\lambda_t \, p_\omega}{\alpha_t}
$$

Since $\bar{\mu}_\omega > p_\omega$ when $\lambda_t > 0$ and $\alpha_t < 1$, the total capacity $\sum_\omega \bar{\mu}_\omega > 1$. The risk-adjusted weights are found by assigning as much weight as possible to the most expensive scenarios:

1. Sort scenarios by cost $Q_\omega$ in **descending** order
2. Walk down the sorted list, assigning $\mu^*_\omega = \bar{\mu}_\omega$ (the upper bound) to each scenario
3. When the cumulative weight reaches 1, the current scenario receives the remaining fraction, and all cheaper scenarios get $\mu^*_\omega = 0$

This is a greedy allocation (continuous knapsack) — it places maximum weight on the worst scenarios and minimum weight on the best.

> **Special cases**:
>
> - **Risk-neutral** ($\lambda_t = 0$): $\bar{\mu}_\omega = p_\omega$ for all $\omega$, so $\mu^* = p$ and this reduces to the standard aggregation from [Cut Management §3](cut-management.md).
> - **Pure CVaR** ($\lambda_t = 1$): $\bar{\mu}_\omega = p_\omega / \alpha_t$. Only the worst $\alpha_t$-fraction of scenarios receive weight; all others get $\mu^*_\omega = 0$.
> - **Convex combination** ($0 < \lambda_t < 1$): All scenarios receive at least some weight (the $(1-\lambda_t) p_\omega$ floor), but the worst scenarios receive up to $\bar{\mu}_\omega$.

> **Equivalence note**: This sorting procedure produces the same $\mu^*$ as solving the dual LP from §4.2, because the LP maximizes a linear objective with per-scenario upper bounds — a structure whose optimal solution is the greedy allocation above.

**Step 3: Compute risk-averse cut coefficients** using $\mu^*$ (justified by the theorem in §5):

$$
\bar{\hat{\alpha}}_{t-1} = \sum_{\omega \in \Omega_t} \mu^*_\omega \cdot \hat{\alpha}_t(\omega)
$$

$$
\bar{\beta}_{t-1} = \sum_{\omega \in \Omega_t} \mu^*_\omega \cdot \beta_t(\omega)
$$

**Step 4: Add cut to stage $t-1$**:

$$
\theta_{t-1} \geq \bar{\hat{\alpha}}_{t-1} + \bar{\beta}_{t-1}^\top x_{t-1}
$$

> **Comparison with risk-neutral aggregation**: The only difference from [Cut Management §3](cut-management.md) is that the scenario probabilities $p(\omega)$ are replaced by the risk-adjusted weights $\mu^*_\omega$.

## 8 Per-Stage Risk Profiles

Risk aversion can vary by stage. The `risk_measure` field in `stages.json` specifies $(\lambda_t, \alpha_t)$ per stage (see [Input Scenarios §1.7](../02-data-model/input-scenarios.md)):

| Option            | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `"expectation"`   | Risk-neutral expected value (default)                                 |
| `{"cvar": {...}}` | CVaR parameters with `alpha` (confidence level) and `lambda` (weight) |

Example with stage-varying risk:

```json
{
  "stages": [
    {
      "id": 0,
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.5 } }
    },
    {
      "id": 1,
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.25 } }
    },
    {
      "id": 2,
      "risk_measure": "expectation"
    }
  ]
}
```

This allows decreasing risk aversion over the horizon (e.g., higher $\lambda$ for near-term stages, lower for distant stages).

## 9 Upper Bound with Risk Measures

Monte Carlo simulation cannot directly estimate the upper bound for CVaR problems because:

1. CVaR is computed over the entire distribution, not sample averages
2. The optimal $\eta$ (VaR threshold) changes with the policy

For risk-averse problems, the inner approximation (SIDP) provides deterministic upper bounds that remain valid regardless of the risk measure. See [Upper Bound Evaluation §1](upper-bound-evaluation.md) for the complete formulation.

## 10 Lower Bound Validity with Risk Measures

> **Critical Warning**: The lower bound computed during SDDP training is **NOT a valid bound** for risk-averse problems.

### Why the Lower Bound Fails

In risk-neutral SDDP, the lower bound $\underline{z} = V_1(x_0)$ is the optimal value of the first-stage LP, which uses cuts that provide valid outer approximations of the expected future cost. This bound converges to the true optimal expected cost.

In risk-averse SDDP, this property **does not hold** because:

1. **Cuts approximate nested risk measures**: Each cut approximates $\rho_t[V_{t+1}(x_t, \omega)]$, where the risk measure $\rho_t$ depends on the _distribution of costs at that stage_.

2. **The LP optimizes under the wrong distribution**: The first-stage LP optimizes $\rho_1[V_2]$, but the risk-adjusted distribution used in the cuts was computed for the _training_ states, not for the optimal first-stage decision.

3. **Nested risk measures are not time-consistent in expectation**: Unlike $\mathbb{E}[\cdot]$, the nested application of CVaR does not satisfy:
   $$
   \rho_1[\rho_2[\ldots]] \neq \rho[\text{total cost}]
   $$

### What the "Lower Bound" Represents

For risk-averse problems, the value $\underline{z} = V_1(x_0)$ computed by SDDP is:

- A **convergence indicator**: It increases monotonically and plateaus when additional cuts provide no improvement
- **NOT a valid lower bound** on the true risk-averse optimal cost

### Recommendations

| Purpose                    | Method                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Convergence monitoring** | Track $\underline{z}$ stabilization (bound stalling rule — see [Stopping Rules §4](stopping-rules.md)) |
| **Valid upper bound**      | Inner approximation (SIDP) — see [Upper Bound Evaluation](upper-bound-evaluation.md)                   |
| **Policy evaluation**      | Monte Carlo simulation with risk-averse policy decisions                                               |

> **Convergence reporting**: When risk measures are enabled, convergence reports should label the "lower bound" as "convergence indicator" or explicitly note that it is not a valid bound.

## 11 References

> Philpott, A.B., de Matos, V.L., & Finardi, E.C. (2013). "On solving multistage stochastic programs with coherent risk measures." _Operations Research_, 61(4), 957-970. https://doi.org/10.1287/opre.2013.1200

> Shapiro, A. (2011). "Analysis of stochastic dual dynamic programming method." _European Journal of Operational Research_, 209(1), 63-72.

## Cross-References

- [Notation Conventions](../00-overview/notation-conventions.md) — Symbol definitions, dual variable notation, and sign conventions
- [SDDP Algorithm](sddp-algorithm.md) — Bellman recursion and forward/backward pass structure modified by risk measures
- [Cut Management](cut-management.md) — Dual extraction and cut coefficient computation; risk-averse aggregation replaces $p(\omega)$ with $\mu^*_\omega$ (§7)
- [Stopping Rules](stopping-rules.md) — Bound stalling recommended for risk-averse convergence monitoring; simulation-based stopping limitations
- [Discount Rate](discount-rate.md) — Discount factor $d$ convention and discounted Bellman equation
- [Infinite Horizon](infinite-horizon.md) — Interaction of risk measures with periodic policy graphs
- [Upper Bound Evaluation](upper-bound-evaluation.md) — SIDP inner approximation for valid upper bounds with CVaR objectives
- [Input Scenarios §1.7](../02-data-model/input-scenarios.md) — JSON schema for `risk_measure` field: `"expectation"` or `{"cvar": {"alpha": ..., "lambda": ...}}`
