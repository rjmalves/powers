---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §11 (11.1-11.5, 11.7) Cut Generation and Aggregation"
  - "MATHEMATICAL_FORMULATIONS.md §12 (12.1-12.9) Cut Selection Strategies"
last_reviewed: 2026-02-20
reviewed_by: rogerio
review_notes: "Approved. Full rewrite: definitions → aggregation → validity → selection → convergence. Pseudocode replaced with behavioral definitions. Discount factor applied to θ in objective (not to cut coefficients). FPHA contribution to storage cut coefficient documented. Multi-cut deferred."
change_log:
  - date: 2026-02-14
    description: "Extracted from MATHEMATICAL_FORMULATIONS.md §11-§12"
  - date: 2026-02-20
    description: "Review: renumbered sections sequentially. Restructured into mathematical definitions → aggregation → validity → selection strategies → convergence. Stripped pseudocode algorithms (cut addition, Level-1, LML1, dominated detection) — replaced with behavioral definitions and properties. Reduced multi-cut to brief note referencing deferred-features.md. Moved configuration JSON to reference note. Added discount factor acknowledgment in aggregation. Clarified relatively complete recourse via penalty system. Added cross-references to lp-formulation.md, binary-formats.md, par-inflow-model.md, penalty-system.md."
  - date: 2026-02-20
    description: "Approved. Discount factor note updated: applied to θ in objective, not to cut coefficients. Added FPHA contribution note to §2 dual extraction (storage coefficient includes FPHA hyperplane duals). 6 cross-references added."
---

# Cut Management

## Purpose

This spec defines the complete Benders cut lifecycle in POWE.RS: what cuts represent mathematically, how cut coefficients relate to LP dual variables, how per-scenario cuts are aggregated into a single cut, under what conditions cuts are valid, and what selection strategies are available to control cut pool growth. For cut pool persistence and in-memory layout, see [Binary Formats §3–4](../02-data-model/binary-formats.md).

## 1. Cut Definition

A **Benders cut** at stage $t-1$ is a linear inequality that provides a lower bound on the cost-to-go function $V_t(x)$:

$$
\theta \geq \alpha + \sum_{h \in \mathcal{H}} \beta^v_h \cdot v_h + \sum_{h \in \mathcal{H}} \sum_{\ell=1}^{p_h} \beta^{lag}_{h,\ell} \cdot a_{h,\ell}
$$

where:

- $\theta$ is the future cost variable in the stage $t-1$ LP
- $\alpha$ is the cut intercept
- $\beta^v_h$ is the storage coefficient for hydro $h$ (marginal value of water)
- $\beta^{lag}_{h,\ell}$ is the AR lag coefficient for hydro $h$, lag $\ell$ (marginal value of inflow history)
- $v_h$, $a_{h,\ell}$ are the state variables (see [LP Formulation §4, §5](lp-formulation.md))

The cut coefficients are dense — every state variable (all storage volumes and all AR lags) has a non-zero coefficient in every cut. See [Binary Formats §3.4](../02-data-model/binary-formats.md) for the implications on memory layout.

## 2. Dual Variable Extraction

After solving the stage $t$ subproblem for trial state $\hat{x}_{t-1}$ and scenario $\omega_t$, the cut coefficients are derived from the LP dual variables:

| Constraint                            | Dual Variable        | Cut Coefficient                               | Units     |
| ------------------------------------- | -------------------- | --------------------------------------------- | --------- |
| Water balance (hydro $h$)             | $\pi^{wb}_h$         | $\beta^v_{t,h} = \pi^{wb}_h$                  | \$/hm³    |
| AR lag fixing (hydro $h$, lag $\ell$) | $\pi^{lag}_{h,\ell}$ | $\beta^{lag}_{t,h,\ell} = \pi^{lag}_{h,\ell}$ | \$/(m³/s) |

The cut intercept ensures the cut passes through the trial point:

$$
\alpha_t = Q_t(\hat{x}_{t-1}, \omega_t) - \sum_{h \in \mathcal{H}} \beta^v_{t,h} \cdot \hat{v}_h - \sum_{h \in \mathcal{H}} \sum_{\ell=1}^{p_h} \beta^{lag}_{t,h,\ell} \cdot \hat{a}_{h,\ell}
$$

where $Q_t(\hat{x}_{t-1}, \omega_t)$ is the optimal objective value of the stage $t$ subproblem.

**Sign convention**: For minimization LPs with $\leq$ constraints, $\pi \geq 0$. For equality constraints (water balance), the sign depends on constraint orientation. The incoming state $\hat{v}_h$ appears on the RHS with coefficient $+1$, so no scaling factor is needed. See [Notation Conventions §5.4](../00-overview/notation-conventions.md) for the complete derivation.

> **Note on FPHA contribution**: For hydros using the FPHA production model, the storage cut coefficient has an additional term from the FPHA hyperplane duals. The FPHA constraints involve $v^{avg}_h = (\hat{v}_h + v_h)/2$, so the incoming storage $\hat{v}_h$ contributes with a factor of $\frac{1}{2}$:
>
> $$\beta^v_{t,h} = \pi^{wb}_h + \frac{1}{2} \sum_m \pi_m^{fpha} \cdot \gamma_v^m$$
>
> where $\pi_m^{fpha}$ is the dual of FPHA hyperplane $m$ and $\gamma_v^m$ is its storage coefficient. This term captures how marginal changes in incoming storage affect generation capacity through head variation. See [Hydro Production Models §2.10](hydro-production-models.md) for details.

> **Note on generic constraints**: If generic constraints (see [LP Formulation §10](lp-formulation.md)) involve state variables, their duals $\pi^{gen}_c$ also contribute to cut coefficients. The coefficient for each state variable is the sum of all dual contributions from constraints where that state variable appears on the RHS.

## 3. Single-Cut Aggregation

In the single-cut formulation, per-scenario cuts from the backward pass are aggregated into one cut per trial point by taking the probability-weighted expectation:

$$
\bar{\alpha}_{t-1} = \sum_{\omega \in \Omega_t} p(\omega) \cdot \alpha_t(\omega)
$$

$$
\bar{\beta}^v_{t-1,h} = \sum_{\omega \in \Omega_t} p(\omega) \cdot \beta^v_{t,h}(\omega)
$$

$$
\bar{\beta}^{lag}_{t-1,h,\ell} = \sum_{\omega \in \Omega_t} p(\omega) \cdot \beta^{lag}_{t,h,\ell}(\omega)
$$

where $p(\omega)$ is the probability of scenario $\omega$.

The aggregated cut $(\bar{\alpha}, \bar{\beta}^v, \bar{\beta}^{lag})$ is added to stage $t-1$'s cut pool.

> **Discount factor**: When the problem uses stage-dependent discount rates, discounting is applied to the $\theta$ variable in the stage $t-1$ objective function (i.e., $\beta_{t-1 \to t} \cdot \theta$), rather than scaling the cut coefficients directly. This is simpler — the cuts remain unmodified and the discount factor appears only in the objective. See [Discount Rate](discount-rate.md) for the discounted Bellman equation and how discount factors interact with cut generation.

> **Multi-cut formulation**: An alternative formulation creates one cut per scenario instead of aggregating, with per-scenario future cost variables $\theta_\omega$. This is deferred — see [Deferred Features §C.3](../06-deferred/deferred-features.md).

## 4. Cut Validity

A cut is **valid** if it is a lower bound on the true cost-to-go function everywhere in the feasible state space:

$$
\alpha_k + \beta_k^\top x \leq V_{t+1}(x) \quad \forall x \in \mathcal{X}_t
$$

**Validity conditions**: The cuts generated by SDDP are valid under:

1. **Convexity of stage subproblems** — guaranteed because all subproblems are LPs
2. **Relatively complete recourse** — feasibility for all states and scenarios. In POWE.RS, this is guaranteed by the recourse slack system (Category 1 penalties): every constraint that could be violated by exogenous uncertainty has a penalty slack variable, ensuring the LP is always feasible. See [Penalty System](../02-data-model/penalty-system.md).
3. **Correct dual extraction** — duals must come from an optimal LP solution (not an infeasible or unbounded one)

## 5. Cut Growth and Selection Motivation

The number of cuts grows as $\mathcal{O}(\text{iterations} \times \text{forward\_passes})$. Many older cuts become redundant as newer, tighter cuts are generated. Without selection, LP solve time increases linearly with iteration count.

Cut selection removes inactive cuts to:

1. Reduce LP solve time (fewer constraint rows)
2. Improve numerical stability (remove near-parallel constraints)
3. Control memory consumption

**Deactivation mechanism**: Cuts are not deleted — they are deactivated by relaxing their bound to $-\infty$. This preserves cut indices for reproducibility and allows reactivation if needed. See [Binary Formats §4](../02-data-model/binary-formats.md) for the preallocation strategy.

## 6. Cut Activity

A cut $k$ at stage $t$ is **active** at state $\hat{x}$ if it is binding at the optimal LP solution:

$$
\theta^* = \alpha_k + \beta_k^\top \hat{x}
$$

Equivalently, the dual multiplier $\lambda_k$ of the cut constraint is positive ($\lambda_k > 0$).

**Activity threshold**: A threshold parameter $\epsilon$ relaxes the binding condition:

$$
\text{cut } k \text{ is active at } \hat{x} \iff \theta^* - (\alpha_k + \beta_k^\top \hat{x}) < \epsilon
$$

| Threshold | Effect                                           |
| --------- | ------------------------------------------------ |
| 0         | Only strictly binding cuts are active            |
| 1e-6      | Near-binding cuts included (numerical tolerance) |
| 1e-3      | Moderately loose cuts retained                   |
| Large     | All cuts considered active (no selection)        |

## 7. Selection Strategies

Three cut selection strategies are available, in increasing order of aggressiveness:

### 7.1 Level-1

A cut is **Level-1 active** if it was binding at least once during the entire algorithm execution. Periodically (every $N$ iterations), cuts that have never been active are deactivated.

**Properties**:

- Least aggressive — retains any cut that was ever useful
- May keep some cuts that were active early but are now permanently dominated
- Preserves convergence guarantee (see §8)

### 7.2 Limited Memory Level-1 (LML1)

Each cut is timestamped with the most recent iteration at which it was active. Periodically, cuts whose timestamp is older than a configurable **memory window** $W$ (in iterations) are deactivated.

**Properties**:

- More aggressive than Level-1 — forgets cuts that haven't been active recently
- Memory window $W$ controls the trade-off between retention and aggressiveness
- Preserves convergence guarantee (see §8)

### 7.3 Dominated Cut Detection

A cut $k$ is **dominated** if no other cut in the pool is ever tighter at any visited state. Formally, cut $k$ is dominated by the remaining cuts $\mathcal{S}$ if:

$$
\alpha_k + \beta_k^\top x \leq \max_{j \in \mathcal{S}} \left\{ \alpha_j + \beta_j^\top x \right\} \quad \forall x \in \mathcal{X}
$$

Since checking this globally is intractable, domination is assessed **at visited states only**: a cut is dominated if at every visited state $\hat{x}$, some other cut achieves a higher (or equal within $\epsilon$) value:

$$
\Delta_k(\hat{x}) = \max_{j \neq k} \left\{ \alpha_j + \beta_j^\top \hat{x} \right\} - \left( \alpha_k + \beta_k^\top \hat{x} \right) > \epsilon \quad \forall \hat{x} \in \text{visited states}
$$

**Properties**:

- Most aggressive — directly identifies cuts that provide no value at any known operating point
- Computational cost is $\mathcal{O}(|\text{cuts}| \times |\text{visited states}|)$ per stage per check
- May remove cuts that would be active at unvisited states (acceptable as the visited set grows dense)

## 8. Convergence Guarantee

**Theorem** (Guigues & Bandarra, 2019): Under Level-1 or LML1 cut selection, SDDP with finitely many scenarios converges to the optimal value function with probability 1.

**Key insight**: Removing cuts that are never active at any visited state does not affect the outer approximation quality at those states. As the set of visited states becomes dense over iterations, the approximation converges.

> Guigues, V., & Bandarra, M.P. (2019). "Single cut and multicut SDDP with cut selection for multistage stochastic linear programs: convergence proof and numerical experiments." _arXiv:1902.06757_. https://arxiv.org/abs/1902.06757

## 9. Selection Parameters

The cut selection strategy is configured with the following parameters:

| Parameter         | Description                                                 | Applies to |
| ----------------- | ----------------------------------------------------------- | ---------- |
| `method`          | Selection strategy: `"level1"`, `"lml1"`, or `"domination"` | All        |
| `threshold`       | Activity threshold $\epsilon$ (recommended: 0)              | All        |
| `check_frequency` | Iterations between selection runs                           | All        |
| `memory_window`   | Iterations to retain inactive cuts                          | LML1 only  |

See [Configuration Reference](../05-config/configuration-reference.md) for the JSON schema.

## Cross-References

- [LP Formulation §4, §5, §11](lp-formulation.md) — Water balance and AR lag constraints that produce duals; Benders cut constraints in the LP
- [PAR Inflow Model §3](par-inflow-model.md) — AR lag state variables that appear in cut coefficients
- [Notation Conventions §5.4](../00-overview/notation-conventions.md) — Dual variable sign convention and cut coefficient derivation
- [Penalty System](../02-data-model/penalty-system.md) — Recourse slacks that guarantee relatively complete recourse (cut validity condition)
- [Binary Formats §3–4](../02-data-model/binary-formats.md) — Cut pool FlatBuffers schema, CSR memory layout requirements, preallocation strategy, checkpoint/resume semantics
- [SDDP Algorithm](sddp-algorithm.md) — Forward/backward pass structure that drives cut generation
- [Stopping Rules](stopping-rules.md) — Convergence criteria that depend on cut quality
- [Discount Rate](discount-rate.md) — Discount factor scaling in cut aggregation
- [Risk Measures](risk-measures.md) — Risk-averse cut generation (CVaR modifies aggregation weights)
- [Deferred Features §C.3](../06-deferred/deferred-features.md) — Multi-cut formulation
- [Configuration Reference](../05-config/configuration-reference.md) — JSON schema for `cut_selection` parameters
