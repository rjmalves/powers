---
status: approved
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §3 (System Element Modeling Overview: 3.1-3.9)"
last_reviewed: 2026-02-19
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-19
    description: "Review approved. Round 1: Buses generalized (not just regional subsystems), deficit sum fixed, exchange factor note added, thermal LP relaxation note improved, AR lag state variable clarified, evaporation/withdrawal clarified. Round 2: Contracts rewritten as typed unidirectional model (single χ per contract), linearized_head added as third production model, filling model section added, FPHA turbined cost added, diversion max flow added, hydro slack table expanded, generation bounds clarified as user-defined. Round 3: Non-controllable sources §6 fully defined (was DEFERRED), pumping station min flow bound added, GNL thermal subsection added with state variables, summary table updated (contracts, NCS, pumping, GNL row). Variable Units Convention section added documenting rate-units decision with full trade-off analysis."
---

# System Element Modeling Overview

## Purpose

This spec describes the physical components of a hydrothermal power system as modeled by POWE.RS: what each element represents, its decision variables, how it connects to other elements, and its role in the optimization objective. It serves as the conceptual foundation between the SDDP algorithm description and the full LP formulation — the reader should understand _what_ is being optimized before seeing _how_ the constraints are assembled.

**Reading order**: [SDDP algorithm](sddp-algorithm.md) → **this spec** → [LP formulation](lp-formulation.md) → [equipment formulations](equipment-formulations.md)

For variable naming conventions and index sets, see [notation conventions](../00-overview/notation-conventions.md).

## Variable Units Convention

All decision variables in POWE.RS use **rate units**: electrical quantities in MW, hydraulic flows in m³/s. Storage (hm³) is inherently an absolute quantity. The block duration $\tau_k$ [hours] enters the LP as an **external multiplier** — it appears in the objective function coefficients and in the water balance conversion factor, but not in the variable bounds or constraint matrix.

This is a deliberate design decision with consequences across the formulation:

| Aspect                 | Consequence                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Objective function** | All cost terms are scaled by $\tau_k$: the coefficient for a \$100/MWh thermal is $\tau_k \times 100$          |
| **Variable bounds**    | Block-invariant — a 500 MW capacity bound is the same regardless of block duration                             |
| **Constraint matrix**  | Clean coefficients: load balance has ±1, production function uses $\rho$ [MW/(m³/s)] directly                  |
| **Duals**              | Load balance dual $\pi_{b,k}$ has units \$/MW. To obtain the marginal cost (CMO) in \$/MWh, divide by $\tau_k$ |
| **Cut coefficients**   | Unaffected — coupling state variables (storage in hm³, AR lags in m³/s) are independent of this choice         |

**Why not absolute units (MWh, hm³ per block)?** The alternative — internalizing $\tau_k$ into the variables — was evaluated and rejected for three reasons:

1. **FPHA incompatibility**: The FPHA hyperplane $g \leq \gamma_0 + \gamma_v \cdot v^{avg} + \gamma_q \cdot q + \gamma_s \cdot s$ relates instantaneous rates to storage level. Converting to absolute units does not eliminate $\tau_k$ from the FPHA constant term, and introduces a $10^6/3600 \approx 278$ scaling factor on flow coefficients.

2. **Constraint matrix quality**: With rate units, the constraint matrix (excluding the objective row) spans ~3 orders of magnitude ($\rho \approx 0.05$–$5$, $\eta \approx 0.95$–$1.0$, conversion factors ~$0.01$–$3$). Absolute units inflate this to ~5 orders via the 278 factor in FPHA rows. A tighter constraint matrix improves simplex pivot stability.

3. **Block-varying bounds**: Absolute units make every variable bound depend on $\tau_k$ ($\bar{G} \cdot \tau_k$ instead of $\bar{G}$), complicating LP construction and making the LP structure vary with block configuration.

The objective row does carry large coefficients — up to $\tau_k \times c^{def} \approx 730 \times 10{,}000 = 7.3 \times 10^6$ for deep deficit in monthly problems. Modern LP solvers handle objective scaling effectively through internal prescaling, and this is where the scaling challenge is most tractable.

> **Output convention**: All output reports and user-facing marginal costs must apply the $\div \tau_k$ conversion. See [Output Schemas](../02-data-model/output-schemas.md).

## 1. System Architecture Overview

A hydrothermal power system in POWE.RS consists of interconnected physical elements that work together to meet electricity demand at minimum cost under inflow uncertainty:

![System Element Overview](../../diagrams/exports/svg/sddp/system-element-overview.svg)

The optimizer determines generation and flow decisions at each stage to minimize total expected cost (thermal generation + deficit penalties + regularization costs) while respecting physical constraints and preparing for uncertain future inflows.

## 2. Buses (Regional Subsystems)

### Physical Meaning

A **bus** represents a node in the power network where electrical energy balance must be maintained. The granularity is user-defined: a bus may represent a large regional subsystem, a single substation, or any aggregation level in between. The model scales from a handful of buses to hundreds or thousands without structural changes.

> **Example**: In studies of the Brazilian interconnected system (SIN), buses typically correspond to the four major regions (Southeast/Midwest, South, Northeast, North), but finer decompositions are equally valid.

### Decision Variables

| Variable         | Units | Description                                                       |
| ---------------- | ----- | ----------------------------------------------------------------- |
| $\delta_{b,k,s}$ | MW    | Load deficit (unserved energy) at bus $b$, block $k$, segment $s$ |
| $\epsilon_{b,k}$ | MW    | Excess generation at bus $b$, block $k$                           |

### Connections to Other Elements

Each bus serves as the energy balance node where:

- **Inflows**: Generation from hydro plants, thermal plants, and import contracts connected to the bus
- **Outflows**: Demand, export contracts, pumping station consumption, and transmission to other buses

### Key Parameters

| Parameter       | Units  | Description                                          |
| --------------- | ------ | ---------------------------------------------------- |
| $D_{b,k}$       | MW     | Load demand at bus $b$, block $k$                    |
| $c^{def}_{b,s}$ | \$/MWh | Deficit cost (value of unserved energy), segment $s$ |
| $c^{exc}_b$     | \$/MWh | Excess generation penalty (regularization)           |
| $\bar{d}_{b,s}$ | MW     | Deficit segment depth                                |

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \left[ \sum_{b \in \mathcal{B}} \sum_{s \in \mathcal{S}_b} c^{def}_{b,s} \cdot \delta_{b,k,s} + \sum_{b \in \mathcal{B}} c^{exc}_b \cdot \epsilon_{b,k} \right]
$$

- **Deficit cost**: Very high penalty (\$1,000–10,000/MWh) representing value of lost load
- **Excess cost**: Small regularization term to eliminate spurious slack generation

### LP Constraint Preview

For each bus $b$ and block $k$, the **load balance constraint** enforces:

$$
\text{(generation at } b\text{)} + \text{(imports)} - \text{(exports)} - \text{(pumping)} + \sum_{s} \delta_{b,k,s} - \epsilon_{b,k} = D_{b,k}
$$

For the assembled constraint, see [LP formulation](lp-formulation.md).

## 3. Transmission Lines

### Physical Meaning

A **transmission line** represents the interconnection between two buses, allowing power transfer subject to capacity limits and transmission losses. Lines are bidirectional.

### Decision Variables

| Variable    | Units | Description                                           |
| ----------- | ----- | ----------------------------------------------------- |
| $f^+_{l,k}$ | MW    | Direct flow on line $l$ (source → target), block $k$  |
| $f^-_{l,k}$ | MW    | Reverse flow on line $l$ (target → source), block $k$ |

> **Modeling note**: POWE.RS uses two non-negative variables ($f^+$, $f^-$) rather than a single signed variable. This simplifies bound handling and naturally prevents simultaneous bidirectional flow through regularization costs.

### Connections to Other Elements

Each line connects exactly two buses:

- **Source bus**: Exports $f^+_{l,k}$, receives $\eta_l \cdot f^-_{l,k}$
- **Target bus**: Receives $\eta_l \cdot f^+_{l,k}$, exports $f^-_{l,k}$

### Key Parameters

| Parameter     | Units  | Description                                                                |
| ------------- | ------ | -------------------------------------------------------------------------- |
| $\bar{F}^+_l$ | MW     | Capacity limit (direct direction); may vary by stage via exchange factors  |
| $\bar{F}^-_l$ | MW     | Capacity limit (reverse direction); may vary by stage via exchange factors |
| $\eta_l$      | —      | Transmission efficiency: $\eta_l = 1 - \text{losses}/100$                  |
| $c^{exch}_l$  | \$/MWh | Exchange cost (regularization)                                             |

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \sum_{l \in \mathcal{L}} c^{exch}_l \cdot (f^+_{l,k} + f^-_{l,k})
$$

The exchange cost is a **regularization term** (typically \$0.01–1.00/MWh) that prevents degenerate solutions with unnecessary power circulation and guides the solver toward physically meaningful flow patterns.

### LP Constraint Preview

**Capacity bounds**:

$$
0 \leq f^+_{l,k} \leq \bar{F}^+_l, \quad 0 \leq f^-_{l,k} \leq \bar{F}^-_l
$$

**Load balance contribution** at source bus: $-f^+_{l,k} + \eta_l \cdot f^-_{l,k}$

For detailed constraints, see [equipment formulations](equipment-formulations.md).

## 4. Thermal Plants

### Physical Meaning

A **thermal plant** represents dispatchable generation using fuel (natural gas, coal, oil, biomass, nuclear). Thermal plants have fuel costs modeled with piecewise-linear cost curves.

### Decision Variables

| Variable    | Units | Description                                                  |
| ----------- | ----- | ------------------------------------------------------------ |
| $g_{j,k,s}$ | MW    | Generation at thermal plant $j$, block $k$, cost segment $s$ |

The total generation is $g_{j,k} = \sum_s g_{j,k,s}$.

### Connections to Other Elements

- **Bus connection**: Each thermal plant connects to exactly one bus, contributing to its energy balance
- **No cascade coupling**: Unlike hydro plants, thermals are independent of each other

### Key Parameters

| Parameter                      | Units  | Description                                       |
| ------------------------------ | ------ | ------------------------------------------------- |
| $\bar{G}_j$, $\underline{G}_j$ | MW     | Generation bounds (capacity, minimum stable load) |
| $c^{th}_{j,s}$                 | \$/MWh | Marginal cost for segment $s$ (fuel + O&M)        |
| $\bar{g}_{j,s}$                | MW     | Segment $s$ capacity                              |

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \sum_{j \in \mathcal{T}} \sum_{s} c^{th}_{j,s} \cdot g_{j,k,s}
$$

Thermal costs represent actual operating expenses (\$50–500/MWh depending on fuel type) and constitute the primary controllable cost in the objective function.

### LP Constraint Preview

**Segment bounds** (cost curve linearization): $0 \leq g_{j,k,s} \leq \bar{g}_{j,s}$ for all $s$.

**Total generation bounds**: $\underline{G}_j \leq \sum_s g_{j,k,s} \leq \bar{G}_j$

> **Note**: POWE.RS uses a continuous LP relaxation — there are no binary commitment variables. The minimum generation bound $\underline{G}_j$ represents either "fully off" (0) or "minimum stable" operation. This is a deliberate modeling choice: the LP may dispatch a thermal at an intermediate level between zero and $\underline{G}_j$, which is physically unrealistic but acceptable for the long-term planning horizon targeted by SDDP.

For detailed constraints, see [equipment formulations](equipment-formulations.md).

### GNL Thermal Plants (Deferred)

> **Implementation Status**: The data model is fully specified (see [Input System Entities §4](../02-data-model/input-system-entities.md)); implementation is deferred. GNL thermals are currently rejected by input validation.

GNL (Gas Natural Liquefeito) plants require **dispatch anticipation**: the dispatch decision must be committed $L$ stages ahead due to fuel ordering lead times. This introduces additional **state variables** — a committed dispatch pipeline of $L$ values that shift forward at each stage:

| State Variable    | Description                         |
| ----------------- | ----------------------------------- |
| $g^{gnl}_{j,t+1}$ | Dispatch committed for stage $t+1$  |
| $g^{gnl}_{j,t+2}$ | Dispatch committed for stage $t+2$  |
| $\vdots$          | ... up to $L = $ `lag_stages` ahead |

These state variables link stages through the Bellman recursion alongside hydro storage and AR lags. When implemented, GNL thermals will be the only non-hydro elements with state variables in the SDDP formulation.

## 5. Hydro Plants

Hydro plants are the central elements of the SDDP formulation because:

1. Reservoir storage creates **temporal coupling** (water saved today is available tomorrow)
2. Inflows are **stochastic** (uncertain future rainfall/snowmelt)
3. The **water value** (opportunity cost of using water now vs. saving it) emerges from the optimization

### Physical Meaning

A **hydro plant** converts the potential energy of stored water into electricity. Each plant has a reservoir (storage), turbines (conversion), and spillways (excess water release). Hydro plants are typically arranged in **cascades** where upstream releases become downstream inflows.

### Operating Status

POWE.RS distinguishes two hydro plant subsets based on their operational state:

| Subset        | Symbol               | Description                                                             |
| ------------- | -------------------- | ----------------------------------------------------------------------- |
| **Operating** | $\mathcal{H}^{op}$   | Plants that can generate electricity; subject to generation constraints |
| **Filling**   | $\mathcal{H}^{fill}$ | New plants under commissioning, filling dead volume; no generation      |

Most plants are in $\mathcal{H}^{op}$. Filling hydros have target storage constraints instead of generation constraints. Some plants have negligible storage capacity (**run-of-river**) and must pass all inflows through turbines and spillways within the same stage.

### Decision Variables

| Variable     | Units | Description                                                            |
| ------------ | ----- | ---------------------------------------------------------------------- |
| $v_h$        | hm³   | End-of-stage reservoir storage (**state variable**)                    |
| $a_{h,\ell}$ | m³/s  | AR lag $\ell$ for inflow model (**state variable**, see note below)    |
| $q_{h,k}$    | m³/s  | Turbined flow (through generators), block $k$                          |
| $s_{h,k}$    | m³/s  | Spillage (released without generation), block $k$                      |
| $u_{h,k}$    | m³/s  | Diversion flow (bypassed to separate channel), block $k$               |
| $e_{h,k}$    | m³/s  | Evaporation (see note below), block $k$                                |
| $r_{h,k}$    | m³/s  | Water withdrawal (see note below), block $k$                           |
| $g_{h,k}$    | MW    | Hydro generation, block $k$                                            |
| $o_{h,k}$    | m³/s  | Total outflow: $o_{h,k} = q_{h,k} + s_{h,k}$ (downstream channel flow) |

**State variables** ($v_h$ and $a_{h,\ell}$) link stages through the Bellman recursion. The storage $v_h$ tracks reservoir volume and is a true decision variable within each stage (the optimizer chooses its end-of-stage value). The AR lags $a_{h,\ell}$ carry inflow history for the PAR(p) model: they are _state variables_ in the SDDP sense (passed between stages and subject to Benders cuts), but are **fixed at the beginning of each stage** to the realized inflow values — they are not free for the optimizer to choose. All other variables are **control variables** determined within each stage.

> **Evaporation and withdrawal**: $e_{h,k}$ and $r_{h,k}$ are parameter-driven quantities computed from input data (reservoir surface area, withdrawal schedules). They appear in the LP as fixed contributions to the water balance, with slack variables to ensure feasibility when the computed values cannot be physically sustained (e.g., drought reducing available water below the withdrawal commitment).

### Connections to Other Elements

- **Bus connection**: Each hydro plant connects to one bus for energy delivery
- **Cascade topology**: Upstream plants' outflows ($q + s + u$) become downstream plants' inflows, with optional water travel time delay
- **Diversion targets**: Some plants can divert water to a separate downstream plant (not the immediate cascade successor), bounded by $\bar{U}_h$
- **Pumping stations**: May receive pumped water (increasing storage) or supply water to pumps (decreasing storage)

### Key Parameters

| Parameter                                        | Units     | Description                                             |
| ------------------------------------------------ | --------- | ------------------------------------------------------- |
| $\bar{V}_h$, $\underline{V}_h$                   | hm³       | Storage bounds (useful volume)                          |
| $\bar{Q}_h$, $\underline{Q}_h$                   | m³/s      | Turbined flow bounds (machine limits)                   |
| $\bar{O}_h$, $\underline{O}_h$                   | m³/s      | Outflow bounds (environmental flow, flood control)      |
| $\bar{G}_h$, $\underline{G}_h$                   | MW        | Generation bounds (user-defined, not derived from flow) |
| $\bar{U}_h$                                      | m³/s      | Maximum diversion flow                                  |
| $\rho_h$                                         | MW/(m³/s) | Productivity (constant model)                           |
| $\gamma^m_0, \gamma^m_v, \gamma^m_q, \gamma^m_s$ | —         | FPHA hyperplane coefficients for plane $m$              |
| $a_h$                                            | m³/s      | Incremental inflow (stochastic, from PAR model)         |
| $\hat{v}_h$, $\hat{a}_{h,\ell}$                  | hm³, m³/s | Incoming state (from previous stage)                    |

### Water Balance Phenomena

The reservoir dynamics account for all water flows in and out of the plant:

| Term                                          | Direction | Description                                                                 |
| --------------------------------------------- | --------- | --------------------------------------------------------------------------- |
| $\hat{v}_h$                                   | Initial   | Incoming storage from previous stage                                        |
| $a_h$                                         | Inflow    | Incremental inflow (lateral catchment, stochastic)                          |
| $\sum_{i \in \mathcal{U}_h}(q_i + s_i + u_i)$ | Inflow    | Upstream cascade outflows (with travel time delay)                          |
| $\sum_{i:\text{div}=h} u_i$                   | Inflow    | Diverted water received from other plants                                   |
| $\sum_{j:\text{dest}=h} p_j$                  | Inflow    | Pumped water received from pumping stations                                 |
| $q_h + s_h + u_h$                             | Outflow   | Turbined + spillage + diversion (released downstream)                       |
| $e_h$                                         | Outflow   | Evaporation (reservoir surface loss; can be negative for net precipitation) |
| $r_h$                                         | Outflow   | Water withdrawal (consumptive use, removed from system)                     |
| $\sum_{j:\text{src}=h} p_j$                   | Outflow   | Pumped water extracted by pumping stations                                  |

> **Note**: Outflow $o_h = q_h + s_h$ is the water released to the downstream channel (affects tailrace level). Withdrawal $r_h$ and diversion $u_h$ exit through different paths and do not affect the main tailrace.

### Production Function (Water to Power)

POWE.RS supports three models for converting turbined flow to electrical generation, in increasing order of complexity:

1. **Constant Productivity**: $g_{h,k} = \rho_h \cdot q_{h,k}$ — simple linear relationship with fixed $\rho_h$ [MW/(m³/s)], suitable for plants with stable head.

2. **Linearized Head**: Adjusts productivity based on head variation with storage level. Requires hydro geometry data (Volume-Height-Area curve). See [hydro production models](hydro-production-models.md).

3. **FPHA (Função de Produção Hidrelétrica Aproximada)**: Piecewise-linear approximation via hyperplanes that captures head variation with storage level and accounts for tailrace effects from spillage. Each plane $m$:

$$
g_{h,k} \leq \gamma_0^m + \gamma_v^m \cdot v^{avg}_h + \gamma_q^m \cdot q_{h,k} + \gamma_s^m \cdot s_{h,k}
$$

The production model can vary by stage or season per hydro — see [Input Hydro Extensions §2](../02-data-model/input-hydro-extensions.md) for model selection modes.

For the complete FPHA formulation and model comparison, see [hydro production models](hydro-production-models.md).

### Operational Constraints (Soft)

Several hydro constraints are enforced as **soft constraints** with slack variables and penalties:

| Constraint                                    | Meaning                                  | Slack Variable        |
| --------------------------------------------- | ---------------------------------------- | --------------------- |
| $v_h \geq \underline{V}_h$                    | Minimum storage (dead volume)            | $\sigma^{v-}_{h}$     |
| $q_{h,k} \geq \underline{Q}_h$                | Minimum turbined flow (equipment limits) | $\sigma^{q-}_{h,k}$   |
| $o_{h,k} \geq \underline{O}_h$                | Minimum outflow (environmental flow)     | $\sigma^{o-}_{h,k}$   |
| $o_{h,k} \leq \bar{O}_h$                      | Maximum outflow (flood control)          | $\sigma^{o+}_{h,k}$   |
| $g_{h,k} \geq \underline{G}_h$                | Minimum generation (grid services)       | $\sigma^{g-}_{h,k}$   |
| $e_{h,k}$ feasible                            | Evaporation within physical limits       | $\sigma^{e\pm}_{h,k}$ |
| $r_{h,k}$ met                                 | Water withdrawal commitment              | $\sigma^{r}_{h,k}$    |
| $v_h \geq \underline{V}_h$ (filling terminal) | Filling target at last filling stage     | $\sigma^{fill}_{h}$   |

Soft constraints allow the optimizer to violate bounds when physically necessary (e.g., drought conditions preventing minimum outflow), with high penalty costs signaling undesirable operation. Maximum storage ($\bar{V}_h$) is a **hard** physical limit — excess water is handled by emergency spillage, not a slack variable. For the penalty priority ordering and cost magnitudes, see [Penalty System §2](../02-data-model/penalty-system.md). For the complete constraint formulations, see [equipment formulations](equipment-formulations.md).

### Dead-Volume Filling

POWE.RS models the commissioning of new hydro plants with a **filling period** during which the reservoir accumulates water to reach the dead volume ($\underline{V}_h$). During this period:

- The hydro has **no generation**: $q_{h,k} = 0$, $g_{h,k} = 0$ (hard constraint)
- Storage is allowed below $\underline{V}_h$ (the min storage slack is inactive)
- Outflow is limited to spillage (turbines not operational)
- Environmental flow must be met via spillage, with `outflow_violation_below` slack if impossible
- A **terminal filling constraint** at the last filling stage enforces $v_h \geq \underline{V}_h$ with the highest penalty in the system ($\sigma^{fill}_h$)

For the full filling model description, see [Input System Entities §3](../02-data-model/input-system-entities.md) and [Penalty System §7](../02-data-model/penalty-system.md).

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \sum_{h \in \mathcal{H}} \left[ c^{spill}_h \cdot s_{h,k} + c^{fpha}_h \cdot q_{h,k} + c^{div}_h \cdot u_{h,k} \right] + \text{(slack penalties)}
$$

- **Spillage cost**: Small regularization (\$0.001–0.01 per m³/s·h) to prefer turbining over spilling
- **FPHA turbined cost**: Regularization applied **only** to hydros using the FPHA production model. Must be > `spillage_cost` for the same plant to prevent interior FPHA solutions. See [Penalty System §2](../02-data-model/penalty-system.md).
- **Diversion cost**: Small regularization, typically higher than spillage (water leaves main cascade)
- **Slack penalties**: High costs for constraint violations — storage below dead volume, outflow violations, generation violations, evaporation violations, water withdrawal shortfall. See [Penalty System](../02-data-model/penalty-system.md) for the full penalty taxonomy and priority ordering.
- **No generation cost**: Hydro generation has zero marginal fuel cost — its "cost" is the opportunity cost of depleting storage, captured through the value function $V_{t+1}(v_h)$

### LP Constraint Preview

**Water balance** (reservoir dynamics):

$$
v_h = \hat{v}_h + \zeta \cdot \left[ a_h + \sum_{k} w_k \cdot \text{net\_flows}_{h,k} \right]
$$

where $\text{net\_flows}_{h,k}$ includes all inflow and outflow terms listed above.

**AR lag fixing** (inflow history): $a_{h,\ell} = \hat{a}_{h,\ell}$ for all $\ell \in \{1, \ldots, P_h\}$

**Outflow definition**: $o_{h,k} = q_{h,k} + s_{h,k}$

**Generation constraint** (depends on production model):

- Constant productivity: $g_{h,k} = \rho_h \cdot q_{h,k}$
- FPHA: $g_{h,k} \leq \gamma_0^m + \gamma_v^m \cdot v_h^{avg} + \gamma_q^m \cdot q_{h,k} + \gamma_s^m \cdot s_{h,k}$ for each plane $m$

For the fully assembled constraints, see [LP formulation](lp-formulation.md).

## 6. Non-Controllable Generation Sources

### Physical Meaning

A **non-controllable source** represents intermittent generation (wind farms, solar plants, small run-of-river hydros, etc.) whose available output depends on external conditions (weather, river flow) rather than dispatch decisions. The solver receives a stochastic availability value per scenario from the scenario pipeline, and can only curtail generation below that availability — it cannot dispatch upward beyond what nature provides.

Non-controllable sources have near-zero marginal cost. The cost of curtailing available generation is a **regularization penalty** (Category 3 in the [Penalty System](../02-data-model/penalty-system.md)), analogous to `spillage_cost` for hydros — curtailment discards available "free" energy.

### Operative States

Non-controllable sources have lifecycle stages controlled by `entry_stage_id` and `exit_stage_id`:

| State            | Condition                                    | LP Variables                      |
| ---------------- | -------------------------------------------- | --------------------------------- |
| `non_existing`   | Before `entry_stage_id`                      | None                              |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | generation, curtailment (derived) |
| `decommissioned` | After `exit_stage_id`                        | None                              |

### Decision Variables

| Variable       | Units | Description                                          |
| -------------- | ----- | ---------------------------------------------------- |
| $g^{nc}_{r,k}$ | MW    | Generation at non-controllable source $r$, block $k$ |

**Curtailment** is **not** a separate LP decision variable — it is derived as $\kappa_{r,k} = A_{r} - g^{nc}_{r,k}$, where $A_r$ is the stochastic available generation for the current (stage, scenario). The implementation may equivalently formulate curtailment as a reward for dispatching (negative cost on $g^{nc}_{r,k}$) rather than a penalty on $(A_r - g^{nc}_{r,k})$; either is valid.

### Connections to Other Elements

- **Bus connection**: Each source connects to exactly one bus, contributing generation to its energy balance

### Key Parameters

| Parameter    | Units  | Description                                                                                             |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------- |
| $\bar{G}_r$  | MW     | Installed capacity (hard physical upper bound)                                                          |
| $A_r$        | MW     | Available generation for current (stage, scenario), from scenario pipeline. Bounded by $[0, \bar{G}_r]$ |
| $c^{curt}_r$ | \$/MWh | Curtailment cost (regularization penalty)                                                               |

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \sum_{r \in \mathcal{R}} c^{curt}_r \cdot (A_r - g^{nc}_{r,k})
$$

The curtailment cost is a small regularization term (\$0.001–0.01/MWh) that incentivizes the solver to use all available generation before curtailing.

### LP Constraint Preview

**Generation bounds** (hard constraints, no slack variables):

$$
0 \leq g^{nc}_{r,k} \leq A_r
$$

**Load balance contribution** at connected bus: $+g^{nc}_{r,k}$ (generation injected).

> **Note**: Unlike hydro and thermal plants, non-controllable sources have **no stage-varying operational bounds** beyond the stochastic availability. Their output is entirely determined by the scenario model. See [Input System Entities §7](../02-data-model/input-system-entities.md) for the data model.

For detailed constraints, see [equipment formulations](equipment-formulations.md).

## 7. Pumping Stations

### Physical Meaning

A **pumping station** transfers water from one reservoir (source) to another (destination), consuming electrical power in the process. Pumping enables elevation transfer, basin transfer, and storage arbitrage (pumping during low-demand periods, generating during high-demand).

### Decision Variables

| Variable  | Units | Description                                 |
| --------- | ----- | ------------------------------------------- |
| $p_{j,k}$ | m³/s  | Pumped water flow at station $j$, block $k$ |

### Connections to Other Elements

- **Source hydro**: Water is withdrawn from this reservoir
- **Destination hydro**: Water is added to this reservoir
- **Bus connection**: Pumping consumes power at the connected bus

### Key Parameters

| Parameter         | Units     | Description            |
| ----------------- | --------- | ---------------------- |
| $\underline{P}_j$ | m³/s      | Minimum pumped flow    |
| $\bar{P}_j$       | m³/s      | Maximum pumped flow    |
| $\gamma_j$        | MW/(m³/s) | Power consumption rate |

### Role in Objective Function

> **Important**: Pumping stations do **not** have a direct cost term in the objective function. The cost of pumping is implicitly captured through the energy consumed, which appears as load in the power balance.

### LP Constraint Preview

**Flow bounds**: $\underline{P}_j \leq p_{j,k} \leq \bar{P}_j$

**Water balance impact**: Source hydro: $-p_{j,k}$ (water removed); Destination hydro: $+p_{j,k}$ (water added).

**Load balance impact** at connected bus: $-\gamma_j \cdot p_{j,k}$ (power consumed).

For detailed constraints, see [equipment formulations](equipment-formulations.md).

## 8. Import/Export Contracts

### Physical Meaning

**Contracts** represent agreements to buy (import) or sell (export) electricity with external systems outside the modeled region, providing flexibility during shortages and revenue opportunity for surplus.

Each contract is **unidirectional**: it is either an import contract or an export contract, identified by a `type` field. This matches the data model in [Input System Entities §6](../02-data-model/input-system-entities.md), where each contract has a single `price_per_mwh` and a single `limits.min_mw`/`limits.max_mw` pair.

### Decision Variables

| Variable     | Units | Description                                  |
| ------------ | ----- | -------------------------------------------- |
| $\chi_{c,k}$ | MW    | Dispatched power for contract $c$, block $k$ |

> **Notation**: $\chi$ (Greek chi) is used for contracts to avoid confusion with cost parameter $c$ and flow variable $q$.

### Connections to Other Elements

Each contract connects to exactly one bus, contributing to its energy balance:

- **Import contracts** ($c \in \mathcal{C}^{imp}$): Add $\chi_{c,k}$ to the bus (power entering the system)
- **Export contracts** ($c \in \mathcal{C}^{exp}$): Remove $\chi_{c,k}$ from the bus (power leaving the system)

### Key Parameters

| Parameter                      | Units  | Description                                                                 |
| ------------------------------ | ------ | --------------------------------------------------------------------------- |
| $\underline{C}_c$, $\bar{C}_c$ | MW     | Minimum and maximum contract dispatch limits                                |
| $c^{ctr}_c$                    | \$/MWh | Contract price: positive for imports (cost), negative for exports (revenue) |

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \sum_{c \in \mathcal{C}} c^{ctr}_c \cdot \chi_{c,k}
$$

Because import prices are positive and export prices are negative, this single summation naturally adds import costs and subtracts export revenue.

### LP Constraint Preview

**Capacity bounds**: $\underline{C}_c \leq \chi_{c,k} \leq \bar{C}_c$

For detailed constraints, see [equipment formulations](equipment-formulations.md).

## 9. Summary: Physical Elements to LP Components

The following table maps each physical system element to its LP representation:

| Physical Element      | State Variables      | Control Variables                          | Key Constraints                     | Objective Role                               |
| --------------------- | -------------------- | ------------------------------------------ | ----------------------------------- | -------------------------------------------- |
| **Bus**               | —                    | $\delta_{b,k,s}$, $\epsilon_{b,k}$         | Load balance                        | Deficit penalty (high), Excess penalty (low) |
| **Transmission Line** | —                    | $f^+_{l,k}$, $f^-_{l,k}$                   | Capacity bounds                     | Exchange cost (regularization)               |
| **Thermal Plant**     | —                    | $g_{j,k,s}$                                | Generation bounds, Segment limits   | Fuel cost                                    |
| **Thermal (GNL)**     | $g^{gnl}_{j,t+\ell}$ | $g_{j,k,s}$                                | Generation bounds + commit pipeline | Fuel cost (deferred)                         |
| **Hydro Plant**       | $v_h$, $a_{h,\ell}$  | $q_{h,k}$, $s_{h,k}$, $u_{h,k}$, $g_{h,k}$ | Water balance, Generation function  | Spillage/diversion cost (regularization)     |
| **Non-Controllable**  | —                    | $g^{nc}_{r,k}$                             | Availability bound                  | Curtailment penalty (regularization)         |
| **Pumping Station**   | —                    | $p_{j,k}$                                  | Flow bounds (min/max)               | None (cost via energy consumption)           |
| **Contract**          | —                    | $\chi_{c,k}$                               | Dispatch bounds (min/max)           | Import cost or Export revenue                |

**Key insight**: The hydro storage variables $v_h$ and AR lag variables $a_{h,\ell}$ are the **only state variables** that currently link stages through the Bellman recursion. When GNL support is implemented, the committed dispatch pipeline $g^{gnl}_{j,t+\ell}$ will add thermal state variables as well. All other elements contribute control variables that are determined within each stage. This structure enables SDDP's decomposition: the stage subproblem optimizes all control variables given the incoming state, and Benders cuts approximate the future cost as a function of the outgoing state.

## Cross-References

- [Notation conventions](../00-overview/notation-conventions.md) — variable naming conventions and index sets used throughout
- [LP formulation](lp-formulation.md) — fully assembled LP constraints combining all elements
- [Equipment formulations](equipment-formulations.md) — detailed per-equipment constraint derivations
- [Hydro production models](hydro-production-models.md) — FPHA and linearized head alternatives for the hydro production function
- [Deferred features](../06-deferred/deferred-features.md) — GNL thermal dispatch anticipation and battery storage (planned)
