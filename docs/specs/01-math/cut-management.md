---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §11 (11.1-11.5, 11.7) Cut Generation and Aggregation"
  - "MATHEMATICAL_FORMULATIONS.md §12 (12.1-12.9) Cut Selection Strategies"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Cut Management

## Purpose

This spec defines the complete cut lifecycle in the POWE.RS SDDP solver: dual variable extraction, cut coefficient computation, single-cut aggregation, the cut addition algorithm, cut validity conditions, and cut selection strategies (Level-1, LML1, dominated cut detection). It consolidates §11 (Cut Generation and Aggregation) and §12 (Cut Selection Strategies) from the source formulation.

## 11.1 Dual Variable Extraction

After solving the stage $t$ subproblem for state $\hat{x}_{t-1}$ and scenario $\omega_t$, extract dual variables from the optimal LP solution:

| Constraint                            | Dual Variable        | Notation                           | Units     |
| ------------------------------------- | -------------------- | ---------------------------------- | --------- |
| Water balance (hydro $h$)             | $\pi^{wb}_h$         | Shadow price of storage            | \$/hm³    |
| AR lag fixing (hydro $h$, lag $\ell$) | $\pi^{lag}_{h,\ell}$ | Shadow price of inflow lag         | \$/(m³/s) |
| Generic constraint (constraint $c$)   | $\pi^{gen}_c$        | Shadow price of generic constraint | depends   |

**Sign Convention**: For minimization LPs with $\leq$ constraints, $\pi \geq 0$. For equality constraints (water balance), the sign depends on constraint orientation. See [Notation Conventions §5.4](../00-overview/notation-conventions.md) for the detailed sign convention and cut coefficient derivation.

## 11.2 Cut Coefficient Computation

From the dual variables, compute cut coefficients for state variables:

**Storage coefficient** (marginal value of water):

$$
\beta^v_{t,h} = \pi^{wb}_h
$$

> **Note**: The dual $\pi^{wb}_h$ has units \$/hm³ (shadow price of the water balance constraint). No scaling factor $\zeta$ appears because the incoming state $\hat{v}_h$ is on the RHS with coefficient $+1$. See [Notation Conventions §5.4](../00-overview/notation-conventions.md) for the detailed derivation and sign convention explanation.

**AR lag coefficient** (marginal value of historical inflow information):

$$
\beta^{lag}_{t,h,\ell} = \pi^{lag}_{h,\ell}
$$

**Cut intercept** (computed to make the cut pass through the trial point):

$$
\alpha_t = Q_t(\hat{x}_{t-1}, \omega_t) - \sum_{h \in \mathcal{H}} \beta^v_{t,h} \cdot \hat{v}_h - \sum_{h \in \mathcal{H}} \sum_{\ell=1}^{p_h} \beta^{lag}_{t,h,\ell} \cdot \hat{a}_{h,\ell}
$$

where $Q_t(\hat{x}_{t-1}, \omega_t)$ is the optimal objective value of the stage $t$ subproblem.

## 11.3 Single-Cut Aggregation

For **risk-neutral** problems with single-cut aggregation, compute expectation over scenarios:

**Aggregated intercept**:

$$
\bar{\alpha}_{t-1} = \sum_{\omega \in \Omega_t} p(\omega) \cdot \alpha_t(\omega)
$$

**Aggregated storage coefficients**:

$$
\bar{\beta}^v_{t-1,h} = \sum_{\omega \in \Omega_t} p(\omega) \cdot \beta^v_{t,h}(\omega)
$$

**Aggregated lag coefficients**:

$$
\bar{\beta}^{lag}_{t-1,h,\ell} = \sum_{\omega \in \Omega_t} p(\omega) \cdot \beta^{lag}_{t,h,\ell}(\omega)
$$

where $p(\omega)$ is the probability of scenario $\omega$.

## 11.4 Multi-Cut Formulation (DEFERRED)

The multi-cut variant creates one cut per scenario instead of aggregating:

$$
\theta_{t-1,\omega} \geq \beta_{t-1 \to t} \cdot \left( \alpha_t(\omega) + \sum_h \beta^v_{t,h}(\omega) \cdot v_h + \sum_{h,\ell} \beta^{lag}_{t,h,\ell}(\omega) \cdot a_{h,\ell} \right) \quad \forall \omega \in \Omega_t
$$

with aggregation in the objective:

$$
\theta_{t-1} = \sum_{\omega \in \Omega_t} p(\omega) \cdot \theta_{t-1,\omega}
$$

**Trade-offs**:

- More cuts per iteration (one per scenario vs. one total)
- Potentially faster convergence in early iterations
- Higher LP solve time due to more constraints

See [Deferred Features](../06-deferred/deferred-features.md) for planned implementation details.

## 11.5 Cut Addition Algorithm

**Algorithm: Backward Pass Cut Generation**

1. For each stage $t$ from $T-1$ down to $1$:
   - For each trial point $\hat{x}$ from forward pass:
     - Initialize `cuts_for_this_point = []`
     - For each scenario $\omega \in \Omega_t$:
       - Solve subproblem: $(Q_\omega, \pi_\omega) = \text{solve\_subproblem}(t, \hat{x}, \omega)$
       - Compute per-scenario cut coefficients:
         $$\beta^v_\omega = \text{compute\_storage\_coefficients}(\pi_\omega)$$
         $$\beta^{lag}_\omega = \text{compute\_lag\_coefficients}(\pi_\omega)$$
         $$\alpha_\omega = Q_\omega - \beta^v_\omega \cdot \hat{x}.\text{storage} - \beta^{lag}_\omega \cdot \hat{x}.\text{lags}$$
       - Append to cuts: `cuts_for_this_point.append(`$(\alpha_\omega, \beta^v_\omega, \beta^{lag}_\omega)$`)`
     - Aggregate (single-cut formulation):
       $$(\bar{\alpha}, \bar{\beta}^v, \bar{\beta}^{lag}) = \text{aggregate\_cuts}(\text{cuts\_for\_this\_point}, \text{probabilities})$$
     - Add to stage $t-1$:
       $$\text{add\_cut\_to\_stage}(t-1, \text{intercept}=\bar{\alpha}, \text{coefs}=(\bar{\beta}^v, \bar{\beta}^{lag}))$$

## 11.7 Cut Validity

A cut is **valid** if it provides a lower bound on the true cost-to-go function:

$$
\alpha_k + \beta_k^\top x \leq V_{t+1}(x) \quad \forall x \in \mathcal{X}_t
$$

**Theorem**: The cuts generated by SDDP are valid under:

1. Convexity of stage subproblems
2. Relatively complete recourse (feasibility for all scenarios)
3. Correct dual extraction from optimal LP solutions

## 12.1 Motivation for Cut Selection

As SDDP iterations progress, the number of Benders cuts grows linearly ($\mathcal{O}(\text{iterations} \times \text{forward\_passes})$). Many cuts become redundant (dominated by newer, tighter cuts). Cut selection removes inactive cuts to:

1. Reduce LP solve time (fewer constraints)
2. Improve numerical stability (remove near-parallel constraints)
3. Maintain memory efficiency

## 12.2 Cut Activity Definition

A cut $k$ at stage $t$ is **active** at state $\hat{x}$ if it is binding at the optimal solution:

$$
\theta^* = \alpha_k + \beta_k^\top \hat{x}
$$

Equivalently, the cut constraint has **positive dual multiplier** $\lambda_k > 0$.

A cut is **dominated** if there exists no visited state where it is active.

## 12.3 Level-1 Cut Selection

**Definition**: A cut is **Level-1** if it was active at least once during the entire algorithm execution.

**Algorithm: Level-1 Cut Selection**

1. After each backward pass:
   - For each stage $t$:
     - For each cut $k$ in stage $t$:
       - If cut $k$ was active in current iteration:
         - Mark cut $k$ as "used"

2. Periodically (every $N$ iterations):
   - For each stage $t$:
     - For each cut $k$ in stage $t$:
       - If cut $k$ was never "used":
         - Deactivate cut $k$ (set bound to $-\infty$)

**Properties**:

- Simple to implement (just track "ever active" flag)
- Preserves convergence guarantee
- May retain some dominated cuts (active once, never again)

## 12.4 Limited Memory Level-1 (LML1)

**Definition**: For each visited state, keep only the **most recently active** cut.

**Algorithm: Limited Memory Level-1**

1. After backward pass at iteration $i$:
   - For each visited state $\hat{x}$ in forward pass:
     - For each stage $t$:
       - Identify which cut $k^*$ was active at $\hat{x}$
       - Mark $k^*$ with timestamp $i$

2. Periodically:
   - For each stage $t$:
     - For each cut $k$:
       - If $k.\text{timestamp} < \text{current\_iteration} - \text{memory\_window}$:
         - Deactivate cut $k$

**Properties**:

- More aggressive than Level-1
- Memory window controls retention period
- Still preserves finite convergence (with probability 1)

## 12.5 Dominated Cut Detection

A cut $k$ is **dominated** by a set of cuts $\mathcal{S}$ if:

$$
\alpha_k + \beta_k^\top x \leq \max_{j \in \mathcal{S}} \left\{ \alpha_j + \beta_j^\top x \right\} \quad \forall x \in \mathcal{X}
$$

**Practical Detection** (at visited states):

For each visited state $\hat{x}$, compute:

$$
\Delta_k(\hat{x}) = \max_{j \neq k} \left\{ \alpha_j + \beta_j^\top \hat{x} \right\} - \left( \alpha_k + \beta_k^\top \hat{x} \right)
$$

If $\Delta_k(\hat{x}) > \epsilon$ for all visited states, cut $k$ is **dominated**.

**Algorithm: Dominated Cut Detection**

1. For each stage $t$:
   - Let $\text{states} = \text{visited\_states}[t]$
   - For each cut $k$:
     - Set `dominated = true`
     - For each state $\hat{x}$ in states:
       - Compute: $\text{value}_k = \alpha_k + \beta_k^\top \hat{x}$
       - Compute: $\text{max\_other} = \max_{j \neq k} (\alpha_j + \beta_j^\top \hat{x})$
       - If $\text{value}_k \geq \text{max\_other} - \text{threshold}$:
         - Set `dominated = false`
         - Break
     - If `dominated`:
       - Deactivate cut $k$

## 12.6 Threshold Parameter

The `threshold` parameter in cut selection controls the minimum violation to consider a cut active:

$$
\text{cut } k \text{ is active at } \hat{x} \iff \theta^* - (\alpha_k + \beta_k^\top \hat{x}) < \text{threshold}
$$

| Threshold Value | Behavior                                         |
| --------------- | ------------------------------------------------ |
| 0               | Only strictly binding cuts are active            |
| 1e-6            | Near-binding cuts included (numerical tolerance) |
| 1e-3            | Moderately loose cuts included                   |
| Large           | All cuts considered active (no selection)        |

**Recommended**: `threshold = 0` with numerical tolerance handled separately.

## 12.7 Cut Selection Configuration

```json
{
  "training": {
    "cut_selection": {
      "enabled": true,
      "method": "domination",
      "threshold": 0,
      "check_frequency": 10,
      "memory_window": 50
    }
  }
}
```

| Parameter         | Description                                  |
| ----------------- | -------------------------------------------- |
| `method`          | `"level1"`, `"lml1"`, or `"domination"`      |
| `threshold`       | Minimum violation for activity               |
| `check_frequency` | Iterations between cut selection runs        |
| `memory_window`   | For LML1: iterations to retain inactive cuts |

## 12.8 Convergence Guarantee

**Theorem** (Guigues & Bandarra, 2019): Under Level-1 or LML1 cut selection, SDDP with finitely many scenarios converges to the optimal value function with probability 1.

**Key insight**: Removing cuts that are never active at any visited state does not affect the quality of the outer approximation at those states. As the set of visited states becomes dense, the approximation converges.

## 12.9 Reference

> Guigues, V., & Bandarra, M.P. (2019). "Single cut and multicut SDDP with cut selection for multistage stochastic linear programs: convergence proof and numerical experiments." _arXiv:1902.06757_. https://arxiv.org/abs/1902.06757

## Cross-References

- [Notation Conventions](../00-overview/notation-conventions.md) — Symbol definitions, dual variable notation, and sign convention derivation (§5)
- [SDDP Algorithm](sddp-algorithm.md) — Forward/backward pass structure that drives cut generation
- [Stopping Rules](stopping-rules.md) — Convergence criteria that depend on cut quality
- [Configuration Reference](../05-config/configuration-reference.md) — JSON schema for `cut_selection` parameters
- [Deferred Features](../06-deferred/deferred-features.md) — Multi-cut formulation (§11.4) planned for future implementation
