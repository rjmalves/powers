---
status: draft
review_priority: 3-medium
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
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Risk Measures

## Purpose

This spec defines the risk-averse SDDP formulation used in POWE.RS, based on Conditional Value-at-Risk (CVaR). It covers the CVaR definition, the convex combination risk measure (SDDP.jl convention), dual representations, the risk-averse subgradient theorem, modified cut generation, per-stage risk profiles, and the critical implications for bound validity.

For notation conventions (index sets, parameters, decision variables, dual variables), see [Notation Conventions](../00-overview/notation-conventions.md).

## 17.1 Motivation

Risk-neutral SDDP minimizes expected cost, which can lead to policies that perform poorly in adverse scenarios. **Risk-averse SDDP** incorporates a coherent risk measure (typically CVaR) to protect against tail risks.

## 17.2 Conditional Value-at-Risk (CVaR)

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

## 17.3 Convex Combination Risk Measure (SDDP.jl Convention)

SDDP.jl uses a convex combination of expectation and CVaR:

$$
\rho^{\lambda, \alpha}[Z] = (1 - \lambda) \mathbb{E}[Z] + \lambda \cdot \text{CVaR}_\alpha[Z]
$$

where:

- $\lambda \in [0, 1]$: Risk aversion weight (0 = risk-neutral, 1 = pure CVaR)
- $\alpha \in (0, 1]$: CVaR confidence level

## 17.4 Dual Representation of Convex Risk Measures

Convex risk measures have a **dual representation** that is essential for computing risk-averse cuts:

$$
\mathbb{F}[Z] = \sup_{q \in \mathcal{M}(p)} \mathbb{E}_q[Z] - \alpha(p, q)
$$

where:

- $\mathcal{M}(p) \subseteq \mathcal{P}$ is a convex subset of the probability simplex
- $\alpha(p, q)$ is a concave penalty function
- $\mathcal{P} = \{p \geq 0 : \sum_{\omega} p_\omega = 1\}$

**Interpretation**: The dual computes the expectation with respect to the **worst** probability vector $q$ within the set $\mathcal{M}$, less a penalty term $\alpha(p, q)$.

### CVaR Dual Representation

For CVaR$_\alpha$, the dual representation is:

$$
\text{CVaR}_\alpha[Z] = \sup_{\mu \in \mathcal{M}_\alpha(p)} \mathbb{E}_\mu[Z]
$$

where the risk set $\mathcal{M}_\alpha(p)$ is:

$$
\mathcal{M}_\alpha(p) = \left\{\mu \geq 0 : \sum_\omega \mu_\omega = 1, \; \mu_\omega \leq \frac{p_\omega}{\alpha} \; \forall \omega \right\}
$$

> **Notation Convention**: We use $\mu$ (Greek mu) for the risk-adjusted probability measure to avoid confusion with turbined flow $q$.

The penalty $\alpha(p, \mu) = 0$ for CVaR (no penalty term).

**Interpretation**: CVaR puts more probability weight on the worst outcomes, with each scenario receiving at most $p_\omega / \alpha$ probability mass. For small $\alpha$, only the worst scenarios receive significant weight.

### EAVaR Dual Representation

For the convex combination $\rho^{\lambda, \alpha}[Z] = (1-\lambda)\mathbb{E}[Z] + \lambda \cdot \text{CVaR}_\alpha[Z]$:

$$
\mathcal{M}^{EAVaR}(p) = \left\{\mu \geq 0 : \sum_\omega \mu_\omega = 1, \; \mu_\omega \leq (1-\lambda) p_\omega + \frac{\lambda p_\omega}{\alpha} \; \forall \omega \right\}
$$

## 17.5 Risk-Averse Subgradient Theorem

The key theorem for computing risk-averse cuts:

> **Theorem (Risk-Averse Subgradient)**: Let $V(x, \omega)$ be convex with respect to $x$ for all fixed $\omega \in \Omega$, and let $\lambda(\tilde{x}, \omega)$ be a subgradient of $V(x, \omega)$ at $x = \tilde{x}$.
>
> If $\mu^* = \text{argmax}_{\mu \in \mathcal{M}(p)} \mathbb{E}_\mu[V(\tilde{x}, \omega)] - \alpha(p, \mu)$, then:
>
> $$\sum_{\omega \in \Omega} \mu^*_\omega \cdot \lambda(\tilde{x}, \omega)$$
>
> is a subgradient of $\mathbb{F}[V(x, \omega)]$ at $\tilde{x}$.

**Application to Cut Generation**:

In SDDP, the subgradients $\lambda(\tilde{x}, \omega)$ are the cut coefficients $\beta_t(\omega)$ obtained from LP duals. The risk-averse cut coefficients are computed as:

$$
\bar{\beta}_{t-1,h} = \sum_{\omega \in \Omega_t} \mu^*_\omega \cdot \beta_{t,h}(\omega)
$$

where $\mu^*$ is the optimal dual probability vector computed from the scenario costs $\{Q_t(\hat{x}, \omega)\}_{\omega \in \Omega_t}$.

## 17.6 Risk-Averse Bellman Equation

The risk-averse value function satisfies:

$$
V_t(x_{t-1}) = \rho^{\lambda, \alpha}\left[\min_{x_t} \left\{ c_t^\top x_t + V_{t+1}(x_t) : (x_t, x_{t-1}) \text{ feasible} \right\}\right]
$$

This replaces the standard expectation in the [Bellman recursion](sddp-algorithm.md) with the nested risk measure $\rho^{\lambda, \alpha}$. Time consistency is guaranteed because the risk measure is applied stage-wise (nested formulation), not to the total cost.

## 17.7 Cut Generation with Risk Measures

For each visited state $\hat{x}_{t-1}$, compute the risk-averse cut as follows:

**Step 1: Solve subproblems** for all realizations $\omega \in \Omega_t$:

$$
Q_t(\hat{x}_{t-1}, \omega) = \min_{x_t} \left\{ c_t^\top x_t + \theta_t : \text{constraints} \right\}
$$

Extract dual solutions $\pi_t(\omega)$ and compute per-scenario cut coefficients:

- Intercept: $\alpha_t(\omega) = Q_t(\hat{x}_{t-1}, \omega) - \beta_t(\omega)^\top \hat{x}_{t-1}$
- Coefficients: $\beta_t(\omega) = -\pi_t(\omega)$ (dual of state constraint)

For details on dual extraction and sign conventions, see [Cut Management §11.1–11.2](cut-management.md).

**Step 2: Find optimal risk-adjusted probability** by solving the dual:

$$
\mu^* = \text{argmax}_{\mu \in \mathcal{M}(p)} \sum_{\omega} \mu_\omega \cdot Q_t(\hat{x}_{t-1}, \omega) - \alpha(p, \mu)
$$

For EAVaR with parameters $(\lambda, \alpha)$, this is a linear program:

$$
\max_{\mu} \sum_\omega \mu_\omega \cdot Q_\omega \quad \text{s.t.} \quad \mu_\omega \leq (1-\lambda)p_\omega + \frac{\lambda p_\omega}{\alpha}, \; \sum_\omega \mu_\omega = 1, \; \mu \geq 0
$$

The solution places maximum weight on the worst (highest-cost) scenarios.

**Step 3: Compute risk-averse cut coefficients** using the theorem from §17.5:

$$
\bar{\alpha}_{t-1} = \sum_{\omega \in \Omega_t} \mu^*_\omega \cdot \alpha_t(\omega)
$$

$$
\bar{\beta}_{t-1} = \sum_{\omega \in \Omega_t} \mu^*_\omega \cdot \beta_t(\omega)
$$

**Step 4: Add cut to stage $t-1$**:

$$
\theta_{t-1} \geq \bar{\alpha}_{t-1} + \bar{\beta}_{t-1}^\top x_{t-1}
$$

> **Note**: For pure CVaR ($\lambda = 1$), the optimal $\mu^*$ assigns weight only to scenarios with costs at or above VaR$_\alpha$. For the convex combination ($0 < \lambda < 1$), all scenarios receive some weight.

## 17.8 Per-Stage Risk Profiles

Risk aversion can vary by stage. The configuration specifies $(\lambda_t, \alpha_t)$ for each stage:

| Scenario         | Stage 1-12                | Stage 13-60               | Stage 61-120              |
| ---------------- | ------------------------- | ------------------------- | ------------------------- |
| **Conservative** | $\lambda=0.5, \alpha=0.2$ | $\lambda=0.3, \alpha=0.3$ | $\lambda=0.1, \alpha=0.5$ |
| **Aggressive**   | $\lambda=0.1, \alpha=0.5$ | $\lambda=0.2, \alpha=0.3$ | $\lambda=0.3, \alpha=0.2$ |

## 17.9 Implementation Notes

| Mathematical Concept                            | Data Model Reference                                                |
| ----------------------------------------------- | ------------------------------------------------------------------- |
| Risk measure $\rho^{\lambda, \alpha}$           | `stages.json` → `risk_measure` field (per stage)                    |
| $\lambda$ parameter                             | `risk_measure.lambda`                                               |
| $\alpha$ parameter                              | `risk_measure.alpha`                                                |
| Risk-adjusted probabilities $\tilde{p}(\omega)$ | Computed at runtime during backward pass                            |
| CVaR cut coefficients                           | Stored in `policy/cuts/stage_XXX.bin` (same format as risk-neutral) |

For the full JSON schema and configuration options, see [Configuration Reference](../05-config/configuration-reference.md).

## 17.10 Upper Bound with Risk Measures

**Important**: Monte Carlo simulation cannot directly estimate the upper bound for CVaR problems because:

1. CVaR is computed over the entire distribution, not sample averages
2. The optimal $\eta$ (VaR threshold) changes with the policy

**Solution**: Use the inner approximation (SIDP) for true upper bounds with CVaR objectives.

## 17.11 Reference

> Philpott, A.B., de Matos, V.L., & Finardi, E.C. (2013). "On solving multistage stochastic programs with coherent risk measures." _Operations Research_, 61(4), 957-970. https://doi.org/10.1287/opre.2013.1200

## 17.12 Lower Bound Validity with Risk Measures

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

| Purpose                    | Method                                                    |
| -------------------------- | --------------------------------------------------------- |
| **Convergence monitoring** | Track $\underline{z}$ stabilization (bound stalling rule) |
| **Valid lower bound**      | Use inner approximation evaluated with risk measures      |
| **Policy evaluation**      | Monte Carlo simulation with risk-averse policy decisions  |

> **Implementation Note**: When risk measures are enabled, convergence reports should label the "lower bound" as "convergence indicator" or explicitly note that it is not a valid bound.

### Reference

> Shapiro, A. (2011). "Analysis of stochastic dual dynamic programming method." _European Journal of Operational Research_, 209(1), 63-72.

## Cross-References

- [Notation Conventions](../00-overview/notation-conventions.md) — Symbol definitions, dual variable notation, and sign conventions
- [SDDP Algorithm](sddp-algorithm.md) — Bellman recursion and forward/backward pass structure modified by risk measures
- [Cut Management](cut-management.md) — Dual extraction and cut coefficient computation reused in risk-averse cut generation (§17.7)
- [Stopping Rules](stopping-rules.md) — Statistical stopping limitations for risk-averse problems; bound stalling recommended instead
- [Configuration Reference](../05-config/configuration-reference.md) — JSON schema for `risk_measure` parameters (`lambda`, `alpha`) per stage
