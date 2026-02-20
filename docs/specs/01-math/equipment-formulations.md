---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §8.1 (Thermal Plants)"
  - "MATHEMATICAL_FORMULATIONS.md §8.2 (Transmission Lines)"
  - "MATHEMATICAL_FORMULATIONS.md §8.3 (Import/Export Contracts)"
  - "MATHEMATICAL_FORMULATIONS.md §8.4 (Pumping Stations)"
last_reviewed: 2026-02-20
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-20
    description: "Review approved. Thermal §1.1: clarified piecewise-linear convex cost curve semantics (segment filling order, concrete example), separated continuous relaxation note from cost modeling, added total generation bounds. GNL §1.2: updated to reflect data model fully specified but implementation deferred with validation rejection. Transmission §2: added exchange factor note for stage-varying capacity. Contracts §3: rewritten from bidirectional (χ^in, χ^out) to typed unidirectional (single χ_{c,k} with c^ctr_c). Pumping §4: added minimum flow bound. Added §5 hydro cross-reference section. NCS §6: promoted from DEFERRED to full section with bounds, curtailment cost, load balance; added open question on block distribution of available generation. Added §8 simulation-only constraint enhancements (future): stepped thermal enforcement, storage-dependent hydro bounds, relationship to generic constraints. Updated cross-references throughout."
---

# Equipment-Specific Formulations

## Purpose

This spec details the LP constraints for each equipment type in POWE.RS. While [system elements](system-elements.md) describes _what_ each element is and its decision variables, this spec contains the _detailed mathematical constraints_ governing each equipment type's behavior within the LP. The reading order is: [system elements](system-elements.md) → [LP formulation](lp-formulation.md) → **this spec** (per-equipment deep dives).

For variable definitions and index sets, see [notation conventions](../00-overview/notation-conventions.md). For hydro production constraints specifically, see [hydro production models](hydro-production-models.md).

## 1. Thermal Plants

### 1.1 Standard Thermals

Thermal generation cost is modeled as a **piecewise-linear convex function** of dispatched power. Each cost segment represents a generation tranche with its own marginal cost — for example, a 200 MW plant might have its first 50 MW at \$100/MWh, the next 50 MW at \$150/MWh, and the final 100 MW at \$200/MWh. Because the segment costs are non-decreasing (convex), the LP solver naturally fills cheaper segments first.

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

Total generation bounds:

$$
\underline{G}_j \leq g_{j,k} \leq \bar{G}_j
$$

Both bounds are **hard** constraints with no slack variables — thermal dispatch is directly controllable (unlike hydro, which depends on exogenous inflows).

**Objective Contribution:**

$$
\sum_{k} \tau_k \sum_{s} c^{th}_{j,s} \cdot g_{j,k,s}
$$

> **Continuous relaxation**: POWE.RS does not include binary commitment variables in the training LP. The minimum generation bound $\underline{G}_j$ is enforced as a simple lower bound, meaning the LP may dispatch a thermal at an intermediate level between zero and $\underline{G}_j$ — a region that is physically infeasible for most thermal units (below minimum stable load). This is a deliberate modeling choice: the continuous relaxation is acceptable for the long-term planning horizon targeted by SDDP, and stepped enforcement can be applied during simulation (see §8).

### 1.2 GNL Thermals

GNL (Gas Natural Liquefeito) plants require dispatch anticipation due to fuel ordering lead times. The data model is fully specified (see [Input System Entities §4](../02-data-model/input-system-entities.md)), including state variables for the committed dispatch pipeline. However, implementation is **deferred** — input validation currently rejects GNL thermals with an exceptional validation rule. See [deferred features](../06-deferred/deferred-features.md) for the planned formulation.

## 2. Transmission Lines

**Decision Variables:**

- $f^+_{l,k}$ = direct flow (source → target)
- $f^-_{l,k}$ = reverse flow (target → source)

**Bounds:**

$$
0 \leq f^+_{l,k} \leq \bar{F}^+_l, \quad 0 \leq f^-_{l,k} \leq \bar{F}^-_l
$$

> **Stage-varying capacity**: The capacity limits $\bar{F}^+_l$ and $\bar{F}^-_l$ may vary by stage through exchange factors defined in `constraints/exchange_factors.json`. See [Input Constraints](../02-data-model/input-constraints.md) for the exchange factor specification.

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
> See LP formulation (§1.4 Regularization Costs) for the full taxonomy of penalty vs. cost types.

## 3. Import/Export Contracts

Each contract is **unidirectional** — either an import or an export contract, identified by a `type` field. This matches the data model in [Input System Entities §6](../02-data-model/input-system-entities.md).

**Decision Variables:**

- $\chi_{c,k}$ = dispatched power for contract $c$, block $k$

**Bounds:**

$$
\underline{C}_c \leq \chi_{c,k} \leq \bar{C}_c
$$

**Load Balance Contribution:**

At connected bus:

- Import contracts ($c \in \mathcal{C}^{imp}$): $+\chi_{c,k}$ (power entering the system)
- Export contracts ($c \in \mathcal{C}^{exp}$): $-\chi_{c,k}$ (power leaving the system)

**Objective Contribution:**

$$
\sum_{k} \tau_k \sum_{c \in \mathcal{C}} c^{ctr}_c \cdot \chi_{c,k}
$$

Because import prices ($c^{ctr}_c$) are positive and export prices are negative, this single summation naturally adds import costs and subtracts export revenue.

## 4. Pumping Stations

Pumping stations transfer water from a source reservoir (downstream) to a destination reservoir (upstream), consuming electrical power in the process.

**Decision Variables:**

- $p_{j,k}$ = pumped water flow at station $j$, block $k$ (m³/s)

**Bounds:**

$$
\underline{P}_j \leq p_{j,k} \leq \bar{P}_j
$$

Both bounds are hard constraints.

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

## 5. Hydro Plants

Hydro constraints are the most complex in the system. Rather than duplicating them here, the hydro formulation is split across two specs:

- **Water balance, outflow, storage bounds, and soft constraints**: See [LP formulation](lp-formulation.md) §4 (water balance), §6 (generation constraints), §7 (outflow constraints), §8 (variable bounds), §9 (constraint violation penalties).
- **Production function models** (constant productivity, FPHA, linearized head): See [hydro production models](hydro-production-models.md).

For hydro decision variables and physical meaning, see [system elements](system-elements.md) §5.

## 6. Non-Controllable Generation Sources

Non-controllable sources (wind farms, solar plants, small run-of-river hydros) have stochastic availability determined by the scenario pipeline. The solver can only curtail generation below the available amount — it cannot dispatch upward beyond what nature provides.

**Decision Variables:**

- $g^{nc}_{r,k}$ = generation at non-controllable source $r$, block $k$

**Bounds (hard):**

$$
0 \leq g^{nc}_{r,k} \leq A_r
$$

where $A_r$ is the stochastic available generation for the current (stage, scenario), bounded by $[0, \bar{G}_r]$ (installed capacity).

**Load Balance Contribution:**

At connected bus: $+g^{nc}_{r,k}$ (generation injected)

**Objective Contribution:**

$$
\sum_{k} \tau_k \sum_{r \in \mathcal{R}} c^{curt}_r \cdot (A_r - g^{nc}_{r,k})
$$

The curtailment cost $c^{curt}_r$ is a **regularization** penalty (Category 3 in the [Penalty System](../02-data-model/penalty-system.md)), analogous to `spillage_cost` for hydros — curtailment discards available "free" energy.

> **Note**: Curtailment is **not** a separate LP decision variable — it is derived as $\kappa_{r,k} = A_r - g^{nc}_{r,k}$. Both bounds are hard constraints with no slack variables, since the available generation is always non-negative and output can always be curtailed to zero.

> **Open question — block distribution of available generation**: The scenario pipeline produces a single available generation value $A_r$ per (stage, scenario). Currently, this value is used identically across all blocks within the stage — the available generation does not vary by block. This contrasts with load demand, which has per-block factors (multipliers on the stochastic load value, see [Input Scenarios §8](../02-data-model/input-scenarios.md)), and exchange capacity, which has per-block factors on line limits (see [Input Constraints §4](../02-data-model/input-constraints.md)). For non-controllable sources, the generation profile within a stage is often non-uniform — solar generation concentrates in daylight blocks, wind may peak at night. Without block factors, the model distributes NCS generation proportionally to block duration, which may not fit all applications. Whether to add NCS block factors (analogous to load and exchange) and where to define them (scenario pipeline, entity registry, or constraints) is to be resolved during the architecture review.

## 7. Deferred Equipment Types

### 7.1 Batteries

Battery energy storage systems with charge/discharge dynamics. **DEFERRED** — see [deferred features](../06-deferred/deferred-features.md) for the planned formulation.

## 8. Simulation-Only Constraint Enhancements (Future)

> **Implementation status**: The enhancements described in this section are planned for future implementation. They are documented here to establish the modeling intent and inform architectural decisions. The training LP formulations in §1–§6 are the current scope.

During training (policy construction), all constraints must be linear and stable across iterations to preserve SDDP convergence guarantees. During simulation (single forward pass, no cuts), these restrictions do not apply — the LP is solved once per (stage, scenario) with no convergence concerns. This opens the door for higher-fidelity constraint enforcement that would break SDDP if used during training.

The following enhancements follow the same pattern established for the [linearized head model](hydro-production-models.md): relaxed/linearized during training, exact during simulation via preprocessing and postprocessing.

### 8.1 Stepped Thermal Enforcement

During training, thermal generation uses a continuous LP relaxation where dispatch can take any value in $[0, \bar{G}_j]$ (or $[\underline{G}_j, \bar{G}_j]$). In reality, most thermal plants operate either fully off or above their minimum stable load — the region $(0, \underline{G}_j)$ is physically infeasible.

During simulation, the stepped behavior can be enforced by adjusting bounds before the LP solve: for each thermal, either fix $g_{j,k} = 0$ or set $\underline{G}_j$ as a hard lower bound, based on the dispatch signal from the continuous solution. This is a preprocessing step that does not change the LP structure.

### 8.2 Storage-Dependent Hydro Bounds

Several operational and environmental constraints on hydro plants are defined as **step functions of reservoir storage** by system operators. Common examples:

- **Outflow bounds**: Maximum outflow depends on storage level (e.g., $v < 500$ hm³ → $\bar{O} = 100$ m³/s; $500 \leq v < 1000$ hm³ → $\bar{O} = 200$ m³/s; $v \geq 1000$ hm³ → unbounded)
- **Turbined flow bounds**: Minimum turbined flow varies with reservoir level (environmental flow requirements that depend on downstream conditions affected by reservoir level)

During training, these constraints must be linearized — typically using the most conservative bound or a piecewise-linear relaxation — to keep the LP linear and stage-invariant. During simulation, the exact step function can be enforced by reading the incoming storage at each stage and selecting the appropriate bound before the LP solve.

### 8.3 Relationship to Generic Constraints

POWE.RS already supports [generic constraints](lp-formulation.md) (§10) — user-defined linear constraints on any LP variable. A user could express storage-dependent bounds as generic constraints with appropriate coefficients. However, the piecewise stepped constraints described above have a fundamentally different character:

- **Generic constraints** are linear and active during both training and simulation
- **Stepped constraints** involve discrete breakpoints on state variables, requiring different treatment in each phase (linearized for training, exact for simulation)

Whether stepped constraints should be expressed through the generic constraint mechanism (with a training/simulation mode flag) or defined in a dedicated input format is an **open design question** to be resolved during the architecture review. The key requirement is that the user can specify the step function breakpoints and the system handles the training/simulation duality transparently.

## Cross-References

- [Notation conventions](../00-overview/notation-conventions.md) — variable and set definitions ($g_j$, $f_l$, $\chi_c$, $p_j$, $\tau_k$)
- [System elements](system-elements.md) — element descriptions, decision variables, and connections
- [LP formulation](lp-formulation.md) — how equipment constraints integrate into the assembled LP; hydro water balance (§4), generation constraints (§6), variable bounds (§8)
- [Hydro production models](hydro-production-models.md) — hydro-specific production function constraints (constant, FPHA, linearized head)
- [Penalty System](../02-data-model/penalty-system.md) — penalty taxonomy, regularization vs. violation costs
- [Input System Entities](../02-data-model/input-system-entities.md) — entity registries (contracts §6, NCS §7, GNL §4, pumping §5)
- [Input Constraints](../02-data-model/input-constraints.md) — exchange factors, generic constraints
- [Block formulations](block-formulations.md) — block structure within which equipment constraints operate
- [Deferred features](../06-deferred/deferred-features.md) — batteries, GNL thermals
