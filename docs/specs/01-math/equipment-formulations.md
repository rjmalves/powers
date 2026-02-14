---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §8.1 (Thermal Plants)"
  - "MATHEMATICAL_FORMULATIONS.md §8.2 (Transmission Lines)"
  - "MATHEMATICAL_FORMULATIONS.md §8.3 (Import/Export Contracts)"
  - "MATHEMATICAL_FORMULATIONS.md §8.4 (Pumping Stations)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Equipment-Specific Formulations

## Purpose

This spec details the LP constraints for each equipment type in POWE.RS. While [system elements](system-elements.md) describes _what_ each element is and its decision variables, this spec contains the _detailed mathematical constraints_ governing each equipment type's behavior within the LP. The reading order is: [system elements](system-elements.md) → [LP formulation](lp-formulation.md) → **this spec** (per-equipment deep dives).

For variable definitions and index sets, see [notation conventions](../00-overview/notation-conventions.md). For hydro production constraints specifically, see [hydro production models](hydro-production-models.md).

## 1. Thermal Plants

### 1.1 Standard Thermals

Thermal generation uses piecewise-linear cost functions with segments.

**Decision Variables:**

- $g_{j,k,s}$ = generation at thermal $j$, block $k$, cost segment $s$

**Constraints:**

Total generation:

$$
g_{j,k} = \sum_{s} g_{j,k,s}
$$

Segment bounds:

$$
0 \leq g_{j,k,s} \leq \bar{g}_{j,s} \quad \forall s
$$

**Objective Contribution:**

$$
\sum_{k} \tau_k \sum_{s} c^{th}_{j,s} \cdot g_{j,k,s}
$$

> **Note**: POWE.RS does not include binary commitment variables. The model uses continuous relaxation with min/max bounds. For detailed unit commitment, post-process SDDP results with a commitment model.

### 1.2 GNL Thermals

GNL (Liquefied Natural Gas) plants require dispatch anticipation due to fuel ordering lead times. This feature is **DEFERRED** — see [deferred features](../06-deferred/deferred-features.md) for the planned formulation.

## 2. Transmission Lines

**Decision Variables:**

- $f^+_{l,k}$ = direct flow (source → target)
- $f^-_{l,k}$ = reverse flow (target → source)

**Bounds:**

$$
0 \leq f^+_{l,k} \leq \bar{F}^+_l, \quad 0 \leq f^-_{l,k} \leq \bar{F}^-_l
$$

**Load Balance Contribution:**

At source bus:

$$
-f^+_{l,k} + \eta_l f^-_{l,k}
$$

At target bus:

$$
\eta_l f^+_{l,k} - f^-_{l,k}
$$

where $\eta_l = 1 - \text{losses\_percent}/100$ accounts for transmission losses.

**Objective Contribution:**

$$
\sum_{k} \tau_k \cdot c^{exch}_l (f^+_{l,k} + f^-_{l,k})
$$

> **Note on Exchange Cost**: The cost $c^{exch}_l$ is a **regularization term**, not an actual transmission cost. Its purpose is to:
>
> 1. **Prevent degenerate solutions**: Without this term, multiple equivalent solutions exist with different flow patterns
> 2. **Guide the solver**: Small positive cost encourages minimal power transfers when indifferent
> 3. **Improve numerical stability**: Reduces cycling in LP simplex iterations
>
> Typical values are very small (\$0.01–1.00/MWh), several orders of magnitude below generation costs. If this cost significantly affects dispatch decisions, the value is set too high.
>
> See LP formulation (§5.0.2 Regularization Costs) for the full taxonomy of penalty vs. cost types.

## 3. Import/Export Contracts

**Decision Variables:**

- $\chi^{in}_{c,k}$ = import power from contract $c$
- $\chi^{out}_{c,k}$ = export power to contract $c$

**Bounds:**

$$
0 \leq \chi^{in}_{c,k} \leq \bar{C}_c, \quad 0 \leq \chi^{out}_{c,k} \leq \bar{C}_c
$$

**Load Balance Contribution:**

At connected bus: $+\chi^{in}_{c,k} - \chi^{out}_{c,k}$

**Objective Contribution:**

$$
\sum_{k} \tau_k \left( c^{imp}_c \cdot \chi^{in}_{c,k} - c^{exp}_c \cdot \chi^{out}_{c,k} \right)
$$

Note: Export revenue is typically positive, hence subtracted from cost.

## 4. Pumping Stations

Pumping stations transfer water from a source reservoir (downstream) to a destination reservoir (upstream), consuming electrical power in the process.

**Decision Variables:**

- $p_{j,k}$ = pumped water flow at station $j$, block $k$ (m³/s)

**Power Consumption:**

$$
P^{pump}_{j,k} = \gamma_j \cdot p_{j,k}
$$

where $\gamma_j$ is the power consumption rate (MW per m³/s).

**Water Balance Impact:**

- Source hydro: $-p_{j,k}$ (water removed)
- Destination hydro: $+p_{j,k}$ (water added)

**Load Balance Impact:**

At connected bus: $-P^{pump}_{j,k} = -\gamma_j \cdot p_{j,k}$ (power consumed)

**Objective Contribution:** None

> **Economic Modeling Note**: Pumping stations do not have a direct cost term in the objective function. The cost of pumping is implicitly captured through energy consumption — the marginal cost of energy at the connected bus determines the effective pumping cost. This approach correctly models the economic incentive: pumping is attractive when energy prices are low (e.g., excess hydro/renewable generation) and unattractive when prices are high (e.g., thermal dispatch at margin).

## 5. Deferred Equipment Types

The following equipment types are planned but not yet implemented:

### 5.1 Batteries

Battery energy storage systems with charge/discharge dynamics. **DEFERRED** — see [deferred features](../06-deferred/deferred-features.md) for the planned formulation.

### 5.2 Non-Controllable Sources

Wind and solar generation with stochastic availability. **DEFERRED** — see [deferred features](../06-deferred/deferred-features.md) for the planned formulation.

## Cross-References

- [Notation conventions](../00-overview/notation-conventions.md) — variable and set definitions ($g_j$, $f_l$, $\chi_c$, $p_j$, $\tau_k$)
- [System elements](system-elements.md) — element descriptions, decision variables, and connections
- [LP formulation](lp-formulation.md) — how equipment constraints integrate into the assembled LP
- [Hydro production models](hydro-production-models.md) — hydro-specific production function constraints
- [Block formulations](block-formulations.md) — block structure within which equipment constraints operate
- [Configuration reference](../05-config/configuration-reference.md) — equipment configuration parameters
- [Deferred features](../06-deferred/deferred-features.md) — batteries, non-controllable sources, GNL thermals
