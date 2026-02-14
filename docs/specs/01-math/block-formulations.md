---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §6.1 (Parallel Blocks)"
  - "MATHEMATICAL_FORMULATIONS.md §6.2 (Chronological Blocks)"
  - "MATHEMATICAL_FORMULATIONS.md §6.3 (Comparison Summary)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Block Formulation Variants

## Purpose

This spec defines the two block formulations supported by POWE.RS — parallel and chronological — which determine how intra-stage time periods (e.g., peak, off-peak, or hourly resolution) are handled in the LP. The choice of block formulation affects water balance constraints, LP size, and the ability to model intra-stage storage dynamics.

For the variable and set definitions used here, see [notation conventions](../00-overview/notation-conventions.md). For how blocks integrate into the full LP, see [LP formulation](lp-formulation.md). For the system elements that participate in block constraints, see [system elements](system-elements.md).

## 1. Parallel Blocks (Default)

In parallel blocks mode, all blocks within a stage are **independent** — there is no intra-stage storage dynamics.

### 1.1 Water Balance (Parallel)

A single water balance constraint spans all blocks:

$$
v_h = \hat{v}_h + \zeta \left[ a_h + \sum_{k \in \mathcal{K}} w_k \cdot \text{net\_flows}_{h,k} \right]
$$

where:

- $w_k = \tau_k / \sum_j \tau_j$ is the block weight
- $\text{net\_flows}_{h,k}$ = inflows from upstream − outflows − evaporation − withdrawal

This formulation assumes the reservoir can freely redistribute water across blocks within the stage.

### 1.2 Characteristics

| Aspect           | Description                           |
| ---------------- | ------------------------------------- |
| LP size          | Smaller (one water balance per hydro) |
| Storage dynamics | End-of-stage only                     |
| Use case         | Long-term strategic planning          |
| Configuration    | `modeling.block_mode = "parallel"`    |

## 2. Chronological Blocks

In chronological blocks mode, blocks are **sequential** within each stage, enabling modeling of intra-stage storage dynamics (e.g., daily cycling patterns within a monthly stage).

### 2.1 Additional Variables

| Variable  | Domain                         | Units | Description                 |
| --------- | ------------------------------ | ----- | --------------------------- |
| $v_{h,k}$ | $[\underline{V}_h, \bar{V}_h]$ | hm³   | Storage at end of block $k$ |

The end-of-stage storage (state variable) is: $v_h = v_{h,|\mathcal{K}|}$

### 2.2 Block 1 Water Balance

$$
v_{h,1} = \hat{v}_h + \zeta_1 \left[ a_h \cdot w_1 + \text{net\_flows}_{h,1} \right]
$$

where $\zeta_1 = 0.0036 \times \tau_1$ is the time conversion for block 1.

### 2.3 Subsequent Blocks Water Balance

For $k = 2, \ldots, |\mathcal{K}|$:

$$
v_{h,k} = v_{h,k-1} + \zeta_k \left[ a_h \cdot w_k + \text{net\_flows}_{h,k} \right]
$$

### 2.4 State Variable Definition

Only **end-of-stage storage** is a state variable:

$$
v_h = v_{h,|\mathcal{K}|}
$$

Inter-block storages $v_{h,k}$ for $k < |\mathcal{K}|$ are internal LP variables — not state variables. This ensures:

1. Cuts are computed with respect to end-of-stage storage only
2. State dimension does not increase with number of blocks

### 2.5 Dual Extraction for Cuts

For cut generation, we need the dual of block 1's water balance (containing $\hat{v}_h$):

$$
\beta_h^{storage} = \pi^{wb}_{h,1}
$$

### 2.6 Characteristics

| Aspect           | Description                                                                    |
| ---------------- | ------------------------------------------------------------------------------ |
| LP size          | Larger ($N_{hydro} \times (\lvert\mathcal{K}\rvert - 1)$ additional vars/cons) |
| Storage dynamics | Intra-stage cycling modeled                                                    |
| Use case         | Short-term planning with storage cycling                                       |
| Configuration    | `modeling.block_mode = "chronological"`                                        |

## 3. Comparison Summary

| Aspect               | Parallel Blocks       | Chronological Blocks                          |
| -------------------- | --------------------- | --------------------------------------------- |
| Water balance        | 1 per hydro per stage | $\lvert\mathcal{K}\rvert$ per hydro per stage |
| Inter-block storage  | Not modeled           | Explicit continuity                           |
| State variables      | End-of-stage only     | End-of-stage only                             |
| LP variables         | Fewer                 | More                                          |
| LP constraints       | Fewer                 | More                                          |
| Intra-stage dynamics | None                  | Full                                          |

## Cross-References

- [Notation conventions](../00-overview/notation-conventions.md) — variable and set definitions ($v_h$, $\hat{v}_h$, $\mathcal{K}$, $\tau_k$, $w_k$)
- [System elements](system-elements.md) — hydro plant element description and decision variables
- [LP formulation](lp-formulation.md) — how block formulations integrate into the assembled LP
- [Hydro production models](hydro-production-models.md) — production function constraints that operate within each block
- [Configuration reference](../05-config/configuration-reference.md) — `modeling.block_mode` setting
