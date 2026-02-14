---
status: draft
review_priority: 1-critical
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §5.0 (Cost and Penalty Taxonomy)"
  - "MATHEMATICAL_FORMULATIONS.md §5.1 (Objective Function)"
  - "MATHEMATICAL_FORMULATIONS.md §5.2 (Load Balance Constraint)"
  - "MATHEMATICAL_FORMULATIONS.md §5.3 (Hydro Water Balance)"
  - "MATHEMATICAL_FORMULATIONS.md §5.4 (AR Inflow Dynamics)"
  - "MATHEMATICAL_FORMULATIONS.md §5.5 (Hydro Generation Constraints)"
  - "MATHEMATICAL_FORMULATIONS.md §5.6 (Outflow Constraints)"
  - "MATHEMATICAL_FORMULATIONS.md §5.7 (Minimum Constraints)"
  - "MATHEMATICAL_FORMULATIONS.md §5.8 (Slack Penalties and Soft Constraints)"
  - "MATHEMATICAL_FORMULATIONS.md §5.9 (Generic Constraints)"
  - "MATHEMATICAL_FORMULATIONS.md §5.10 (Benders Cuts)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# LP Formulation

## Purpose

This spec presents the complete stage subproblem LP for the POWE.RS SDDP solver: the objective function with its cost taxonomy, all constraint families, slack/penalty variables, and the Benders cut interface to the future cost function. It uses the **parallel blocks** formulation by default.

**Reading order**: [SDDP algorithm](sddp-algorithm.md) → [system elements](system-elements.md) → **this spec** → [equipment formulations](equipment-formulations.md)

For what each physical element represents and its decision variables, see [system elements](system-elements.md). For variable naming conventions and index sets, see [notation conventions](../00-overview/notation-conventions.md).

## 1. Cost and Penalty Taxonomy

The objective function includes several cost categories with distinct purposes and typical magnitudes. Understanding this taxonomy is essential for setting appropriate parameter values and interpreting solution reports.

### 1.1 Cost Categories Overview

| Category                            | Purpose                                        | Examples                                | Typical Magnitude           |
| ----------------------------------- | ---------------------------------------------- | --------------------------------------- | --------------------------- |
| **Resource Costs**                  | Actual generation/operational costs            | Thermal fuel, contract prices           | \$ 50-500/MWh               |
| **Economic Signals**                | Represent opportunity cost or value            | Deficit (load shedding), export revenue | \$ 1,000-10,000/MWh         |
| **Regularization Costs**            | Avoid degenerate solutions, guide solver       | Spillage, exchange, excess              | \$ 0.001-10/unit            |
| **Operational Violation Penalties** | Discourage undesirable but feasible operations | Minimum outflow, generation minimum     | \$ 500-5,000/unit           |
| **Physical Violation Penalties**    | Discourage physically impossible operations    | Negative inflow, storage beyond limits  | Very high (\$ 10,000+/unit) |

### 1.2 Resource Costs (Actual Operating Expenses)

| Cost               | Symbol         | Units  | Typical Values | Objective Term                                            |
| ------------------ | -------------- | ------ | -------------- | --------------------------------------------------------- |
| Thermal generation | $c^{th}_{j,s}$ | \$/MWh | 50-500         | $\sum_{j,k,s} \tau_k \cdot c^{th}_{j,s} \cdot g_{j,k,s}$  |
| Import contract    | $c^{imp}_c$    | \$/MWh | 100-300        | $\sum_{c,k} \tau_k \cdot c^{imp}_c \cdot \chi^{in}_{c,k}$ |

> **Note on Pumping**: Pumping stations do not have an explicit cost parameter. The cost of pumping is implicitly determined by the marginal cost of energy at the bus where the pump is connected — see [equipment formulations](equipment-formulations.md) for details.

### 1.3 Economic Signals (Opportunity Cost / Value of Lost Load)

| Cost                    | Symbol          | Units  | Typical Values | Purpose                              |
| ----------------------- | --------------- | ------ | -------------- | ------------------------------------ |
| Deficit (load shedding) | $c^{def}_{b,s}$ | \$/MWh | 5,000-10,000   | Represents value of unserved energy  |
| Export revenue          | $c^{out}_c$     | \$/MWh | 50-200         | Revenue from exports (negative cost) |

### 1.4 Regularization Costs (Solution Guidance)

These are small costs that prevent degenerate solutions without significantly affecting the optimal policy:

| Cost              | Symbol          | Units       | Typical Values | Purpose                                         |
| ----------------- | --------------- | ----------- | -------------- | ----------------------------------------------- |
| Spillage          | $c^{spill}_h$   | \$/(m³/s·h) | 0.001-0.01     | Prefer turbining over spilling when indifferent |
| Diversion         | $c^{div}_h$     | \$/(m³/s·h) | 0.001-0.01     | Prefer main channel flow                        |
| Exchange          | $c^{exch}_\ell$ | \$/MWh      | 0.01-1.0       | Prevent unnecessary power flows                 |
| Excess generation | $c^{exc}_b$     | \$/MWh      | 0.001-0.1      | Eliminate slack generation                      |

> **Note**: Regularization costs should be at least 2-3 orders of magnitude smaller than economic costs to avoid distorting the optimal solution.

### 1.5 Operational Violation Penalties (Soft Constraints)

| Penalty               | Symbol   | Units       | Typical Values | Violated Constraint            |
| --------------------- | -------- | ----------- | -------------- | ------------------------------ |
| Turbined flow minimum | $c^{q-}$ | \$/(m³/s·h) | 500-1,000      | $q_{h,k} \geq \underline{Q}_h$ |
| Outflow minimum       | $c^{o-}$ | \$/(m³/s·h) | 500-1,000      | $o_{h,k} \geq \underline{O}_h$ |
| Outflow maximum       | $c^{o+}$ | \$/(m³/s·h) | 500-1,000      | $o_{h,k} \leq \bar{O}_h$       |
| Generation minimum    | $c^{g-}$ | \$/MWh      | 1,000-2,000    | $g_{h,k} \geq \underline{G}_h$ |

### 1.6 Physical Violation Penalties (Infeasibility Avoidance)

| Penalty               | Symbol     | Units       | Typical Values | Purpose                               |
| --------------------- | ---------- | ----------- | -------------- | ------------------------------------- |
| Negative inflow       | $c^{inf}$  | \$/(m³/s·h) | 10,000+        | PAR(p) model produces negative value  |
| Evaporation violation | $c^{evap}$ | \$/(m³/s·h) | 5,000+         | Computed evaporation exceeds capacity |
| Withdrawal violation  | $c^{with}$ | \$/(m³/s·h) | 5,000+         | Committed withdrawal cannot be met    |

### 1.7 Penalty Priority and Hierarchy

When setting penalties, ensure the following ordering (from highest to lowest):

1. **Physical violations** ($c^{inf}$, $c^{evap}$): Must be prohibitively high to ensure LP feasibility represents physical reality
2. **Deficit**: Represents value of lost load; should exceed any generation cost
3. **Operational violations**: Should exceed typical marginal cost but allow violation when physically necessary
4. **Resource costs**: Market-based or fuel-based
5. **Regularization**: Near-zero to avoid solution distortion

**Mathematical Requirement**:

$$c^{inf} > c^{def} > c^{q-}, c^{o-}, c^{g-} > c^{th} > c^{spill}, c^{exch}$$

> **Note on Thermal Plants**: Unlike hydro plants, thermal plants do not have slack variables for minimum generation constraints. Thermal bounds ($\underline{G}_j$, $\bar{G}_j$) are treated as hard constraints. If operational requirements (e.g., minimum take-or-pay contracts) cannot be met, the bounds should be adjusted in `thermal_bounds.parquet`. This design choice reflects that thermal dispatch is directly controllable, whereas hydro constraints may be violated due to exogenous inflow uncertainty.

> **Note on Storage Targets**: The `target_storage_hm3` in filling hydros is not a cost term but a constraint target for the dead-volume filling period. See DATA_MODEL_SPECIFICATION for filling hydro behavior.

### 1.8 Objective Function Structure

The complete stage objective is:

$$
\min \; \underbrace{C^{resource}}_{\text{thermal, contracts}} + \underbrace{C^{deficit}}_{\text{load shedding}} + \underbrace{C^{regularization}}_{\text{spillage, exchange}} + \underbrace{C^{penalty}}_{\text{soft constraints}} + \theta
$$

where each component is summed over blocks with appropriate time weighting:

$$C^{component} = \sum_{k \in \mathcal{K}} \tau_k \cdot (\text{cost terms for component})$$

## 2. Objective Function

$$
\min \sum_{k \in \mathcal{K}} \tau_k \Bigg[
  \underbrace{\sum_{b \in \mathcal{B}} \sum_{s \in \mathcal{S}_b} c^{def}_{b,s} \delta_{b,k,s}}_{\text{Deficit cost}}
  + \underbrace{\sum_{b \in \mathcal{B}} c^{exc}_b \epsilon_{b,k}}_{\text{Excess cost}}
  + \underbrace{\sum_{j \in \mathcal{T}} \sum_s c^{th}_{j,s} g_{j,k,s}}_{\text{Thermal cost}}
$$

$$
  + \underbrace{\sum_{l \in \mathcal{L}} c^{exch}_l (f^+_{l,k} + f^-_{l,k})}_{\text{Exchange cost (regularization)}}
  + \underbrace{\sum_{h \in \mathcal{H}} c^{spill}_h s_{h,k}}_{\text{Spillage cost (regularization)}}
  + \underbrace{\sum_{h \in \mathcal{H}} c^{div}_h u_{h,k}}_{\text{Diversion cost (regularization)}}
$$

$$
  + \underbrace{\sum_{c \in \mathcal{C}^{in}} c^{in}_c c^{in}_{c,k} - \sum_{c \in \mathcal{C}^{out}} c^{out}_c c^{out}_{c,k}}_{\text{Contract cost (import - export revenue)}}
$$

$$
  + \underbrace{\text{Slack penalty terms}}_{\text{See §4}}
\Bigg] + \theta
$$

## 3. Load Balance Constraint

For each bus $b \in \mathcal{B}$ and block $k \in \mathcal{K}$:

$$
\sum_{h \in \mathcal{H}_b} g_{h,k} + \sum_{j \in \mathcal{T}_b} \sum_s g_{j,k,s}
+ \sum_{l: \text{target}=b} \eta_l f^+_{l,k} + \sum_{l: \text{source}=b} \eta_l f^-_{l,k}
+ \sum_{c \in \mathcal{C}^{in}_b} c^{in}_{c,k}
$$

$$
- \sum_{l: \text{source}=b} f^+_{l,k} - \sum_{l: \text{target}=b} f^-_{l,k}
- \sum_{c \in \mathcal{C}^{out}_b} c^{out}_{c,k}
- \sum_{j \in \mathcal{P}_b} \gamma_j p_{j,k}
+ \sum_{s \in \mathcal{S}_b} \delta_{b,k,s} - \epsilon_{b,k} = D_{b,k}
$$

**Dual variable**: $\pi^{lb}_{b,k}$ (marginal cost of energy at bus $b$, block $k$)

For the physical meaning of each element in the balance, see [system elements](system-elements.md).

## 4. Hydro Water Balance

For each hydro $h \in \mathcal{H}$ (parallel blocks formulation):

$$
v_h = \hat{v}_h + \zeta \Bigg[ a_h + \sum_{k \in \mathcal{K}} w_k \Big(
  \underbrace{\sum_{i \in \mathcal{U}_h} (q_{i,k} + s_{i,k} + u_{i,k})}_{\text{Inflow from upstream}}
  + \underbrace{\sum_{i: \text{div}=h} u_{i,k}}_{\text{Diverted inflow}}
  + \underbrace{\sum_{j: \text{dest}=h} p_{j,k}}_{\text{Pumped inflow}}
$$

$$
  - \underbrace{q_{h,k} - s_{h,k} - u_{h,k}}_{\text{Outflows}}
  - \underbrace{e_{h,k}}_{\text{Evaporation}}
  - \underbrace{r_{h,k}}_{\text{Withdrawal}}
  - \underbrace{\sum_{j: \text{src}=h} p_{j,k}}_{\text{Pumped outflow}}
\Big) \Bigg]
$$

where:

- $\hat{v}_h$ = incoming storage (state from previous stage)
- $a_h$ = incremental inflow (from AR model, see [PAR(p) inflow model](par-inflow-model.md))
- $w_k = \tau_k / \sum_j \tau_j$ = block weight
- $\zeta = 0.0036 \times \sum_k \tau_k$ = time conversion factor

> **Dimensional Consistency**:
>
> - LHS: $v_h$ [hm³]
> - RHS: $\hat{v}_h$ [hm³] + $\zeta$ [hm³/(m³/s)] × (flow terms [m³/s])
> - The factor $\zeta$ converts all flow rates (m³/s) to volumes (hm³) accumulated over the stage
> - Block weights $w_k$ are dimensionless and sum to 1
> - The AR inflow $a_h$ is in m³/s (average rate over the stage)

**Dual variable**: $\pi^{wb}_h$ (water value, used for cut coefficients — see [cut management](cut-management.md))

## 5. AR Inflow Dynamics

The incremental inflow $a_h$ is determined by the PAR(p) autoregressive model:

$$
a_h = \underbrace{\left( \mu_t - \sum_{\ell=1}^{P_h} \psi_\ell \mu_{t-\ell} \right)}_{\text{deterministic base}}
+ \underbrace{\sum_{\ell=1}^{P_h} \psi_\ell \cdot a_{h,\ell}}_{\text{lag contribution}}
+ \underbrace{\sigma_t \cdot \eta_t}_{\text{stochastic innovation}}
$$

**State expansion**: To maintain the Markov property, lagged inflows are state variables with fixing constraints:

$$
a_{h,\ell} = \hat{a}_{h,\ell} \quad \forall h \in \mathcal{H}, \; \ell \in \{1, \ldots, P_h\}
$$

**Dual variable**: $\pi^{lag}_{h,\ell}$ (value of inflow history, used for cut coefficients — see [cut management](cut-management.md))

See [PAR(p) inflow model](par-inflow-model.md) for the complete PAR(p) model specification.

## 6. Hydro Generation Constraints

**Constant Productivity Model** (for each hydro $h \in \mathcal{H}^{op}$, block $k$):

$$
g_{h,k} = \rho_h \cdot q_{h,k}
$$

**FPHA Model** (for each plane $m \in \mathcal{M}_h$, hydro $h$, block $k$):

$$
g_{h,k} \leq \gamma^m_0 + \gamma^m_v \cdot v^{avg}_h + \gamma^m_q \cdot q_{h,k} + \gamma^m_s \cdot s_{h,k}
$$

where $v^{avg}_h$ is the average storage during the stage.

For details on the FPHA construction and production function model variants, see [equipment formulations](equipment-formulations.md).

## 7. Outflow Constraints

**Outflow Definition** (per hydro $h$, block $k$):

$$
o_{h,k} = q_{h,k} + s_{h,k}
$$

> **Clarification**: Outflow $o$ represents water released to the downstream channel (affecting tailrace level). It does NOT include:
>
> - **Withdrawal** $r_{h,k}$: Consumptive use removed from the system (irrigation, water supply)
> - **Diversion** $u_{h,k}$: Water bypassed to a separate channel (not affecting main tailrace)
>
> The water balance (§4) accounts for all flows: inflow $-$ $(q + s + u + r)$ $-$ evaporation = storage change.

**Outflow Bounds** (with slacks for soft enforcement):

$$
\underline{O}_h - \sigma^{o-}_{h,k} \leq o_{h,k} \leq \bar{O}_h + \sigma^{o+}_{h,k}
$$

## 8. Minimum Constraints

**Turbined Flow Minimum** (per hydro $h$, block $k$):

$$
q_{h,k} + \sigma^{q-}_{h,k} \geq \underline{Q}_h
$$

**Generation Minimum** (per hydro $h$, block $k$):

$$
g_{h,k} + \sigma^{g-}_{h,k} \geq \underline{G}_h
$$

## 9. Slack Penalties and Soft Constraints

Slack variables allow constraint violations at a cost. The penalty terms in the objective are:

$$
\text{Penalties} = \sum_{h,k} \Big[
  c^{q-} \sigma^{q-}_{h,k} + c^{o-} \sigma^{o-}_{h,k} + c^{o+} \sigma^{o+}_{h,k} + c^{g-} \sigma^{g-}_{h,k}
  + c^{e} (\sigma^{e+}_{h,k} + \sigma^{e-}_{h,k}) + c^{r} \sigma^{r}_{h,k}
\Big]
$$

**Penalty Priority** (highest to lowest):

1. Stage-specific override (from Parquet files)
2. Entity-specific override (from JSON)
3. Global default (from `penalties.json`)

Typical penalty values:

| Constraint       | Typical Penalty | Units       |
| ---------------- | --------------- | ----------- |
| Turbined min     | 500             | \$/(m³/s·h) |
| Outflow min/max  | 500             | \$/(m³/s·h) |
| Generation min   | 1000            | \$/MWh      |
| Evaporation      | 5000            | \$/(m³/s·h) |
| Water withdrawal | 1000            | \$/(m³/s·h) |

## 10. Generic Constraints

User-defined linear constraints (per constraint $g \in \mathcal{G}$):

$$
\sum_{e} \gamma_{g,e} \cdot x_e \quad \{\leq, =, \geq\} \quad b_g
$$

where $x_e$ can reference any LP variable using expression syntax:

- `hydro_storage(id)`, `hydro_turbined(id)`, `hydro_spillage(id)`
- `thermal_generation(id)`, `bus_deficit(id)`, etc.

Generic constraints can have optional slack variables with configurable penalties.

## 11. Benders Cuts

For each active cut $i$ from previous iterations:

$$
\theta \geq \alpha_i + \sum_{h \in \mathcal{H}} \beta^v_{i,h} \cdot v_h + \sum_{h,\ell} \beta^{lag}_{i,h,\ell} \cdot a_{h,\ell}
$$

where:

- $\alpha_i$ = cut intercept (RHS)
- $\beta^v_{i,h}$ = coefficient for storage state variable
- $\beta^{lag}_{i,h,\ell}$ = coefficient for AR lag state variable

Cuts are pre-allocated and toggled active/inactive via bound changes for warm-starting efficiency.

For cut coefficient derivation, aggregation, and selection strategies, see [cut management](cut-management.md).

## Cross-References

- [Notation conventions](../00-overview/notation-conventions.md) — index sets, parameters, decision variable naming
- [System elements](system-elements.md) — physical meaning of each element and its decision variables
- [SDDP algorithm](sddp-algorithm.md) — iterative structure that solves this LP at each stage
- [PAR(p) inflow model](par-inflow-model.md) — complete AR inflow model specification
- [Cut management](cut-management.md) — dual extraction, cut coefficients, aggregation, and selection
- [Equipment formulations](equipment-formulations.md) — FPHA hydro production, pumping, and other element-specific details
