---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §10 (10.1-10.7)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from MATHEMATICAL_FORMULATIONS.md §10"
---

# Inflow Non-Negativity Solution Methods

## Purpose

This spec defines the four methods available for handling negative inflow realizations produced by the PAR(p) model, including their LP formulations, objective function modifications, and trade-offs.

## 1. Problem Statement

The PAR(p) model can generate negative inflow realizations:

$$
a_h = \underbrace{\mu_m - \sum_{\ell=1}^{p} \psi_\ell \mu_{m-\ell}}_{\text{deterministic base}} + \underbrace{\sum_{\ell=1}^{p} \psi_\ell \cdot \hat{a}_{h,\ell}}_{\text{lag contribution}} + \underbrace{\sigma_m \cdot \eta}_{\text{noise term}}
$$

When $\eta$ is sufficiently negative (e.g., $\eta < -2$), the total can become negative, which is physically impossible.

## 2. Method 1: None (`sem_relaxacao`)

**Description**: No treatment. Negative inflows pass directly to the LP.

**LP Formulation**: Standard AR constraint (unchanged):

$$
a_h = \text{deterministic\_base} + \sum_{\ell=1}^{p} \psi_\ell \cdot a_{h,\ell} + \sigma_m \cdot \eta
$$

**Implications**:

- LP may become **infeasible** when $a_h < 0$ causes water balance violation
- Useful only for debugging or when AR model guarantees positive outputs
- **Not recommended for production**

## 3. Method 2: Penalty (`penalizacao`)

**Description**: Add a slack variable to ensure LP feasibility, with penalty in objective.

**Additional Variables**:

| Variable         | Domain   | Units | Description                 |
| ---------------- | -------- | ----- | --------------------------- |
| $\sigma^{inf}_h$ | $\geq 0$ | m³/s  | Inflow non-negativity slack |

**Modified AR Constraint**:

$$
a_h + \sigma^{inf}_h = \text{deterministic\_base} + \sum_{\ell=1}^{p} \psi_\ell \cdot a_{h,\ell} + \sigma_m \cdot \eta
$$

**Interpretation**: When the AR model produces negative $a_h$, the slack $\sigma^{inf}_h$ absorbs the violation, making the effective inflow:

$$
a_h^{effective} = a_h + \sigma^{inf}_h \geq 0
$$

**Objective Function Addition**:

$$
+ \sum_{h \in \mathcal{H}} c^{inf} \cdot \sigma^{inf}_h \cdot \zeta
$$

where $c^{inf}$ is the penalty cost (default: 1000 \$/(m³/s·h)) and $\zeta$ is the time conversion factor.

**Advantages**:

- LP always feasible
- Clear cost signal for negative inflow events
- Preserves AR dynamics for positive realizations

**Disadvantages**:

- Adds variables and constraints
- Slightly affects marginal water values

**Recommended for most production cases.**

## 4. Method 3: Truncation (`truncamento`)

**Description**: Hard truncation of negative values to zero during scenario generation.

**Scenario Generation**:

$$
a_h = \max\left(0, \text{deterministic\_base} + \sum_{\ell=1}^{p} \psi_\ell \cdot \hat{a}_{h,\ell} + \sigma_m \cdot \eta\right)
$$

**LP Formulation**: Standard AR constraint with the already-truncated $a_h$ value:

$$
a_h = \text{(truncated value from scenario)}
$$

**Advantages**:

- Simple implementation
- No additional LP variables
- Fast computation

**Disadvantages**:

- **Biases the distribution**: Shifts mean upward
- **Breaks AR dynamics**: When truncation occurs, temporal correlation is disrupted
- May affect long-term storage dynamics

## 5. Method 4: Truncation with Penalty (`truncamento_penalizacao`)

**Description**: Hybrid approach that truncates the final inflow but penalizes the statistical violation in the noise term. Based on the YP_FINF slack in SPARHTACUS/SPTcpp.

**Additional Variables**:

| Variable | Domain   | Units | Description                            |
| -------- | -------- | ----- | -------------------------------------- |
| $\xi_h$  | $\geq 0$ | -     | Noise adjustment slack (dimensionless) |

**Modified AR Constraint** (in two parts):

**Part A — Modified noise term**:

$$
\eta_h^{adj} = \eta_h + \xi_h
$$

where $\eta_h$ is the original (possibly very negative) noise realization.

**Part B — Inflow with adjusted noise**:

$$
a_h = \text{deterministic\_base} + \sum_{\ell=1}^{p} \psi_\ell \cdot a_{h,\ell} + \sigma_m \cdot \eta_h^{adj}
$$

**Non-negativity constraint**:

$$
a_h \geq 0
$$

**Interpretation**: The optimizer chooses $\xi_h$ to be the minimum adjustment needed to make $a_h \geq 0$:

$$
\xi_h = \max\left(0, -\eta_h - \frac{\text{deterministic\_base} + \sum_\ell \psi_\ell \cdot \hat{a}_{h,\ell}}{\sigma_m}\right)
$$

**Objective Function Addition**:

$$
+ \sum_{h \in \mathcal{H}} c^{inf} \cdot \sigma_m \cdot \xi_h \cdot \zeta
$$

The penalty is proportional to $\sigma_m \cdot \xi_h$, which is the actual inflow adjustment in m³/s.

**Advantages**:

- Preserves AR model structure better than pure truncation
- Penalty signals statistical violation severity
- Effective inflow is always non-negative

**Disadvantages**:

- More complex formulation
- Requires careful interaction with noise generation

## 6. Comparison Summary

| Method                    | LP Size    | Bias    | AR Preservation | Feasibility | Recommendation |
| ------------------------- | ---------- | ------- | --------------- | ----------- | -------------- |
| `none`                    | Base       | None    | Full            | May fail    | Debugging only |
| `penalty`                 | +vars/cons | Minimal | Full            | Guaranteed  | **Production** |
| `truncation`              | Base       | Upward  | Partial         | Guaranteed  | Quick studies  |
| `truncation_with_penalty` | +vars/cons | Minimal | Full            | Guaranteed  | Risk-averse    |

## 7. Reference

> Larroyd, P.V., Matos, V.L., Diniz, A.L., & Borges, C.L.T. (2022). "Tackling the Seasonal and Stochastic Components in Hydro-Dominated Power Systems with High Renewable Penetration." _Energies_, 15(3), 1115. https://doi.org/10.3390/en15031115

## Cross-References

- [PAR Inflow Model](par-inflow-model.md) — Defines the PAR(p) model that produces the inflow realizations handled here
- [Notation Conventions](../00-overview/notation-conventions.md) — Defines the inflow slack variable $\sigma^{inf}_h$ and related notation
- [Configuration Reference](../05-config/configuration-reference.md) — Runtime configuration for `modeling.inflow_non_negativity.method` and `penalty_cost`
