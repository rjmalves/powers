---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §3 (System Element Modeling Overview: 3.1-3.9)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# System Element Modeling Overview

## Purpose

This spec describes the physical components of a hydrothermal power system as modeled by POWE.RS: what each element represents, its decision variables, how it connects to other elements, and its role in the optimization objective. It serves as the conceptual foundation between the SDDP algorithm description and the full LP formulation — the reader should understand _what_ is being optimized before seeing _how_ the constraints are assembled.

**Reading order**: [SDDP algorithm](sddp-algorithm.md) → **this spec** → [LP formulation](lp-formulation.md) → [equipment formulations](equipment-formulations.md)

For variable naming conventions and index sets, see [notation conventions](../00-overview/notation-conventions.md).

## 1. System Architecture Overview

A hydrothermal power system in POWE.RS consists of interconnected physical elements that work together to meet electricity demand at minimum cost under inflow uncertainty:

![System Element Overview](../../diagrams/exports/svg/sddp/system-element-overview.svg)

The optimizer determines generation and flow decisions at each stage to minimize total expected cost (thermal generation + deficit penalties + regularization costs) while respecting physical constraints and preparing for uncertain future inflows.

## 2. Buses (Regional Subsystems)

### Physical Meaning

A **bus** represents a regional subsystem or load aggregation point where electrical energy balance must be maintained. In the Brazilian interconnected system (SIN), buses typically correspond to major regions (Southeast/Midwest, South, Northeast, North).

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
\text{(generation at } b\text{)} + \text{(imports)} - \text{(exports)} - \text{(pumping)} + \delta_{b,k} - \epsilon_{b,k} = D_{b,k}
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

| Parameter     | Units  | Description                                               |
| ------------- | ------ | --------------------------------------------------------- |
| $\bar{F}^+_l$ | MW     | Capacity limit (direct direction)                         |
| $\bar{F}^-_l$ | MW     | Capacity limit (reverse direction)                        |
| $\eta_l$      | —      | Transmission efficiency: $\eta_l = 1 - \text{losses}/100$ |
| $c^{exch}_l$  | \$/MWh | Exchange cost (regularization)                            |

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

> **Note**: POWE.RS uses a continuous relaxation without binary commitment variables. The minimum generation bound $\underline{G}_j$ represents either "fully off" (0) or "minimum stable" operation — the optimizer may choose intermediate values.

For detailed constraints, see [equipment formulations](equipment-formulations.md).

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
| $a_{h,\ell}$ | m³/s  | AR lag $\ell$ for inflow model (**state variable**)                    |
| $q_{h,k}$    | m³/s  | Turbined flow (through generators), block $k$                          |
| $s_{h,k}$    | m³/s  | Spillage (released without generation), block $k$                      |
| $u_{h,k}$    | m³/s  | Diversion flow (bypassed to separate channel), block $k$               |
| $e_{h,k}$    | m³/s  | Evaporation (water loss from reservoir surface), block $k$             |
| $r_{h,k}$    | m³/s  | Water withdrawal (consumptive use: irrigation, supply), block $k$      |
| $g_{h,k}$    | MW    | Hydro generation, block $k$                                            |
| $o_{h,k}$    | m³/s  | Total outflow: $o_{h,k} = q_{h,k} + s_{h,k}$ (downstream channel flow) |

**State variables** ($v_h$ and $a_{h,\ell}$) link stages through the Bellman recursion. The storage $v_h$ tracks reservoir volume, while the AR lags $a_{h,\ell}$ capture inflow history for the PAR(p) model. All other variables are **control variables** determined within each stage.

### Connections to Other Elements

- **Bus connection**: Each hydro plant connects to one bus for energy delivery
- **Cascade topology**: Upstream plants' outflows ($q + s + u$) become downstream plants' inflows, with optional water travel time delay
- **Diversion targets**: Some plants can divert water to a separate downstream plant (not the immediate cascade successor)
- **Pumping stations**: May receive pumped water (increasing storage) or supply water to pumps (decreasing storage)

### Key Parameters

| Parameter                                        | Units     | Description                                            |
| ------------------------------------------------ | --------- | ------------------------------------------------------ |
| $\bar{V}_h$, $\underline{V}_h$                   | hm³       | Storage bounds (useful volume)                         |
| $\bar{Q}_h$, $\underline{Q}_h$                   | m³/s      | Turbined flow bounds (machine limits)                  |
| $\bar{O}_h$, $\underline{O}_h$                   | m³/s      | Outflow bounds (environmental flow, flood control)     |
| $\bar{G}_h$, $\underline{G}_h$                   | MW        | Generation bounds (installed capacity, minimum stable) |
| $\rho_h$                                         | MW/(m³/s) | Productivity (constant model)                          |
| $\gamma^m_0, \gamma^m_v, \gamma^m_q, \gamma^m_s$ | —         | FPHA hyperplane coefficients for plane $m$             |
| $a_h$                                            | m³/s      | Incremental inflow (stochastic, from PAR model)        |
| $\hat{v}_h$, $\hat{a}_{h,\ell}$                  | hm³, m³/s | Incoming state (from previous stage)                   |

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

POWE.RS supports two models for converting turbined flow to electrical generation:

1. **Constant Productivity**: $g_{h,k} = \rho_h \cdot q_{h,k}$ — simple linear relationship with fixed $\rho_h$ [MW/(m³/s)], suitable for plants with stable head.

2. **FPHA (Função de Produção Hidrelétrica Aproximada)**: Piecewise-linear approximation via hyperplanes that captures head variation with storage level and accounts for tailrace effects from spillage. Each plane $m$:

$$
g_{h,k} \leq \gamma_0^m + \gamma_v^m \cdot v^{avg}_h + \gamma_q^m \cdot q_{h,k} + \gamma_s^m \cdot s_{h,k}
$$

For the complete FPHA formulation, see [hydro production models](hydro-production-models.md).

### Operational Constraints (Soft)

Several hydro constraints are enforced as **soft constraints** with slack variables and penalties:

| Constraint                     | Meaning                                  | Slack Variable        |
| ------------------------------ | ---------------------------------------- | --------------------- |
| $q_{h,k} \geq \underline{Q}_h$ | Minimum turbined flow (equipment limits) | $\sigma^{q-}_{h,k}$   |
| $o_{h,k} \geq \underline{O}_h$ | Minimum outflow (environmental flow)     | $\sigma^{o-}_{h,k}$   |
| $o_{h,k} \leq \bar{O}_h$       | Maximum outflow (flood control)          | $\sigma^{o+}_{h,k}$   |
| $g_{h,k} \geq \underline{G}_h$ | Minimum generation (grid services)       | $\sigma^{g-}_{h,k}$   |
| $e_{h,k}$ feasible             | Evaporation within physical limits       | $\sigma^{e\pm}_{h,k}$ |
| $r_{h,k}$ met                  | Water withdrawal commitment              | $\sigma^{r}_{h,k}$    |

Soft constraints allow the optimizer to violate bounds when physically necessary (e.g., drought conditions preventing minimum outflow), with high penalty costs signaling undesirable operation. For the complete constraint formulations, see [equipment formulations](equipment-formulations.md).

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \sum_{h \in \mathcal{H}} \left[ c^{spill}_h \cdot s_{h,k} + c^{div}_h \cdot u_{h,k} \right] + \text{(slack penalties)}
$$

- **Spillage cost**: Small regularization (\$0.001–0.01 per m³/s·h) to prefer turbining over spilling
- **Diversion cost**: Small regularization, typically higher than spillage (water leaves main cascade)
- **Slack penalties**: High costs for constraint violations
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

> **Status: DEFERRED** — This feature is planned but not yet implemented. See [deferred features](../06-deferred/deferred-features.md) for the planned formulation.

**Non-controllable sources** include wind farms and solar plants whose generation depends on weather conditions rather than dispatch decisions. These sources have stochastic availability, near-zero marginal cost, and a curtailment option.

**Planned decision variables**: generation $g^{nc}_{r,k}$ [MW] and curtailment $\kappa_{r,k}$ [MW], with availability constraint $g^{nc}_{r,k} + \kappa_{r,k} = \bar{G}_r \cdot \alpha_r(\omega)$.

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

| Parameter   | Units     | Description              |
| ----------- | --------- | ------------------------ |
| $\bar{P}_j$ | m³/s      | Maximum pumping capacity |
| $\gamma_j$  | MW/(m³/s) | Power consumption rate   |

### Role in Objective Function

> **Important**: Pumping stations do **not** have a direct cost term in the objective function. The cost of pumping is implicitly captured through the energy consumed, which appears as load in the power balance.

### LP Constraint Preview

**Flow bounds**: $0 \leq p_{j,k} \leq \bar{P}_j$

**Water balance impact**: Source hydro: $-p_{j,k}$ (water removed); Destination hydro: $+p_{j,k}$ (water added).

**Load balance impact** at connected bus: $-\gamma_j \cdot p_{j,k}$ (power consumed).

For detailed constraints, see [equipment formulations](equipment-formulations.md).

## 8. Import/Export Contracts

### Physical Meaning

**Contracts** represent agreements to buy (import) or sell (export) electricity with external systems outside the modeled region, providing flexibility during shortages and revenue opportunity for surplus.

### Decision Variables

| Variable           | Units | Description                               |
| ------------------ | ----- | ----------------------------------------- |
| $\chi^{in}_{c,k}$  | MW    | Import power from contract $c$, block $k$ |
| $\chi^{out}_{c,k}$ | MW    | Export power to contract $c$, block $k$   |

> **Notation**: $\chi$ (Greek chi) is used for contracts to avoid confusion with cost parameter $c$ and flow variable $q$.

### Connections to Other Elements

Each contract connects to exactly one bus, contributing to its energy balance:

- **Import**: Adds $\chi^{in}_{c,k}$ to the bus
- **Export**: Removes $\chi^{out}_{c,k}$ from the bus

### Key Parameters

| Parameter   | Units  | Description                  |
| ----------- | ------ | ---------------------------- |
| $\bar{C}_c$ | MW     | Contract capacity limit      |
| $c^{imp}_c$ | \$/MWh | Import cost (purchase price) |
| $c^{exp}_c$ | \$/MWh | Export revenue (sale price)  |

### Role in Objective Function

$$
\sum_{k \in \mathcal{K}} \tau_k \sum_{c \in \mathcal{C}^{imp}} c^{imp}_c \cdot \chi^{in}_{c,k} - \sum_{k \in \mathcal{K}} \tau_k \sum_{c \in \mathcal{C}^{exp}} c^{exp}_c \cdot \chi^{out}_{c,k}
$$

- **Import cost**: Positive term (actual expense)
- **Export revenue**: Negative term (reduces total cost)

### LP Constraint Preview

**Capacity bounds**: $0 \leq \chi^{in}_{c,k} \leq \bar{C}_c$, $\quad 0 \leq \chi^{out}_{c,k} \leq \bar{C}_c$

For detailed constraints, see [equipment formulations](equipment-formulations.md).

## 9. Summary: Physical Elements to LP Components

The following table maps each physical system element to its LP representation:

| Physical Element      | State Variables     | Control Variables                          | Key Constraints                    | Objective Role                               |
| --------------------- | ------------------- | ------------------------------------------ | ---------------------------------- | -------------------------------------------- |
| **Bus**               | —                   | $\delta_{b,k,s}$, $\epsilon_{b,k}$         | Load balance                       | Deficit penalty (high), Excess penalty (low) |
| **Transmission Line** | —                   | $f^+_{l,k}$, $f^-_{l,k}$                   | Capacity bounds                    | Exchange cost (regularization)               |
| **Thermal Plant**     | —                   | $g_{j,k,s}$                                | Generation bounds, Segment limits  | Fuel cost                                    |
| **Hydro Plant**       | $v_h$, $a_{h,\ell}$ | $q_{h,k}$, $s_{h,k}$, $u_{h,k}$, $g_{h,k}$ | Water balance, Generation function | Spillage/diversion cost (regularization)     |
| **Non-Controllable**  | —                   | $g^{nc}_{r,k}$, $\kappa_{r,k}$             | Availability limit                 | Curtailment penalty (DEFERRED)               |
| **Pumping Station**   | —                   | $p_{j,k}$                                  | Capacity bounds                    | None (cost via energy consumption)           |
| **Contract**          | —                   | $\chi^{in}_{c,k}$, $\chi^{out}_{c,k}$      | Capacity bounds                    | Import cost, Export revenue                  |

**Key insight**: The hydro storage variables $v_h$ and AR lag variables $a_{h,\ell}$ are the **only state variables** that link stages through the Bellman recursion. All other elements contribute control variables that are determined within each stage. This structure enables SDDP's decomposition: the stage subproblem optimizes all control variables given the incoming state, and Benders cuts approximate the future cost as a function of the outgoing state.

## Cross-References

- [Notation conventions](../00-overview/notation-conventions.md) — variable naming conventions and index sets used throughout
- [LP formulation](lp-formulation.md) — fully assembled LP constraints combining all elements
- [Equipment formulations](equipment-formulations.md) — detailed per-equipment constraint derivations
- [Hydro production models](hydro-production-models.md) — FPHA and linearized head alternatives for the hydro production function
- [Deferred features](../06-deferred/deferred-features.md) — non-controllable generation sources and battery storage (planned)
