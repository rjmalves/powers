---
status: draft
review_priority: 4-low
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md Appendix C (C.1-C.7) Deferred Features"
  - "DATA_MODEL_SPECIFICATION.md §3.5.7 Non-Controllable Generation Sources"
  - "DATA_MODEL_SPECIFICATION.md §3.5.8 Battery Storage"
  - "DATA_MODEL_SPECIFICATION.md §3.2 SDDP Algorithm Variants (DEFERRED)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Deferred Features

## Purpose

This spec documents features that are planned but not yet implemented in the POWE.RS SDDP solver. Each feature includes a description, rationale for deferral, prerequisites, and estimated effort. The data model is designed to accommodate these extensions without breaking changes.

## C.1 GNL Thermal Plants

**Status**: DEFERRED

**Description**: Gas Natural Liquefeito (GNL) thermal plants have complex operational constraints including:

- Minimum take-or-pay contracts
- Variable fuel costs based on LNG spot market
- Start-up and shutdown constraints
- Fuel inventory management

**Planned Formulation**:

- Binary variables for unit commitment (requires MIP solver integration)
- Fuel inventory balance constraints
- Contract fulfillment constraints
- Piecewise-linear fuel cost functions

**Why Deferred**: Requires MIP solver integration (SDDiP or Lagrangian relaxation) which is a significant architectural change. Unit commitment is not in the immediate roadmap for medium/long-term planning.

**Prerequisites**:

- Duality handler infrastructure (Lagrangian relaxation for MIP subproblems)
- MIP solver integration (Gurobi, CPLEX, or Cbc)
- GNL-specific data model extensions

**Estimated Effort**: Large (3-4 weeks). Requires SDDiP infrastructure before GNL-specific modeling.

**Reference**: CEPEL NEWAVE/DECOMP GNL modeling documentation.

## C.2 Battery Energy Storage Systems

**Status**: DEFERRED

**Description**: Grid-scale batteries with:

- State-of-charge management
- Charge/discharge efficiency losses
- Degradation modeling (cycle counting)
- Capacity fade over time

**Planned Formulation**:

**Variables**:

| Variable        | Domain           | Units | Description       |
| --------------- | ---------------- | ----- | ----------------- |
| $e_{b,k}$       | $[0, \bar{E}_b]$ | MWh   | State of charge   |
| $p^{ch}_{b,k}$  | $\geq 0$         | MW    | Charging power    |
| $p^{dis}_{b,k}$ | $\geq 0$         | MW    | Discharging power |

**Energy Balance**:

$$
e_{b,k} = e_{b,k-1} + \eta^{ch} \cdot p^{ch}_{b,k} \cdot \Delta t - \frac{p^{dis}_{b,k} \cdot \Delta t}{\eta^{dis}}
$$

**Load Balance Contribution**:

$$
\sum_{b \in \mathcal{B}} (p^{dis}_{b,k} - p^{ch}_{b,k}) \text{ added to generation}
$$

**Data Model** (from DATA_MODEL §3.5.8): The `system/batteries.json` schema is fully specified with fields for capacity (energy/charge/discharge), efficiency, initial SOC, and SOC limits. LP variables include `battery_soc` (state), `battery_charge`, and `battery_discharge` (controls). Output schema in `simulation/batteries/` is also defined.

**Why Deferred**: Batteries are linear storage devices (no integer variables needed), but require:

- New state variable dimension (SOC per battery)
- Integration with bus load balance
- Testing with realistic battery degradation scenarios

**Prerequisites**:

- State variable infrastructure supports dynamic dimension
- Bus balance constraint generation handles battery charge/discharge
- Output schema for battery simulation results

**Estimated Effort**: Medium (2-3 weeks). LP formulation is straightforward; main effort is data pipeline and testing.

## C.3 Multi-Cut Formulation

**Status**: DEFERRED

**Description**: Alternative to single-cut aggregation that creates one cut per scenario.

**Formulation**:

Instead of a single aggregated future cost variable $\theta$, introduce per-scenario variables $\theta_\omega$:

$$
\theta = \sum_{\omega \in \Omega_t} p(\omega) \cdot \theta_\omega
$$

With per-scenario cuts:

$$
\theta_\omega \geq \alpha_k(\omega) + \beta_k(\omega)^\top x \quad \forall k, \omega
$$

**Trade-offs**:

| Aspect                  | Single-Cut                         | Multi-Cut                           |
| ----------------------- | ---------------------------------- | ----------------------------------- | --------- | ------------------------ |
| **Cuts per iteration**  | 1                                  | `                                   | scenarios | `                        |
| **LP size**             | Smaller (1 future cost variable)   | Larger (`                           | scenarios | ` future cost variables) |
| **Convergence rate**    | Slower (more iterations)           | Faster (fewer iterations)           |
| **Time per iteration**  | Faster                             | Slower                              |
| **Memory**              | Lower                              | Higher                              |
| **Numerical stability** | More stable                        | Can have issues with risk measures  |
| **Best for**            | Large scenario counts, risk-averse | Small scenario counts, risk-neutral |

**Why Deferred**: Multi-cut requires significant changes:

- LP construction must handle multiple future cost variables
- Cut storage and selection becomes more complex
- Interaction with CVaR risk measures needs careful implementation
- Performance tuning (when to use which formulation) is problem-dependent

**Prerequisites**:

- LP builder supports variable number of future cost variables
- Cut pool indexed by scenario
- Cut selection adapted for multi-cut pools

**Estimated Effort**: Medium (2-3 weeks). Core algorithm change with wide-reaching effects on cut management.

**Reference**: Birge, J.R. (1985). "Decomposition and partitioning methods for multistage stochastic linear programs." _Operations Research_, 33(5), 989-1007.

## C.4 Markovian Policy Graphs

**Status**: DEFERRED

**Description**: Extension to handle scenario-dependent transitions (e.g., different inflow regimes).

**Current Limitation**: POWE.RS assumes stage-wise independent scenarios. The same scenario tree structure is used regardless of which scenario was realized in the previous stage.

**Planned Extension**:

- Markov chain over "regimes" (e.g., wet/dry/normal)
- Transition probabilities between regimes
- Regime-dependent inflow distributions
- Cut sharing across nodes in same regime

**Formulation**:

Let $M$ be a Markov chain with states $\mathcal{M} = \{1, \ldots, m\}$ and transition matrix $P$.

The policy graph becomes:

- Nodes: $(t, r)$ for stage $t$ and regime $r$
- Edges: $(t, r) \to (t+1, r')$ with probability $P_{r,r'}$

Value function approximation:

$$
V_{t,r}(x) \approx \max_{k \in \mathcal{K}_{t,r}} \{\alpha_k + \beta_k^\top x\}
$$

Cuts are regime-specific and only shared within the same regime.

**Data Model** (from DATA_MODEL §3.2): The `stages.json` schema supports optional `markov_states` fields. Transitions include `source_markov` and `target_markov` fields. Cut files are indexed by `(stage_id, markov_state)` as `stage_XXX_markov_YYY.bin`.

**Why Deferred**: Markovian policy graphs substantially increase algorithm complexity:

- Forward passes must track Markov state transitions
- Backward passes generate cuts for each `(stage, markov_state)` node
- State space grows by factor of `|markov_states|`
- Requires careful handling of stagewise-independent noise within each Markov state

**Prerequisites**:

- Policy graph supports multi-node-per-stage structure
- Cut pool indexed by `(stage, markov_state)` tuples
- Forward/backward pass logic handles Markov transitions
- Scenario generation respects regime-dependent distributions

**Estimated Effort**: Large (3-4 weeks). Fundamental change to policy graph and forward/backward pass logic.

**Reference**: Philpott, A.B., & de Matos, V.L. (2012). "Dynamic sampling algorithms for multi-stage stochastic programs with risk aversion." _European Journal of Operational Research_, 218(2), 470-483.

## C.5 Non-Controllable Sources (Wind/Solar)

**Status**: DEFERRED

**Description**: Stochastic renewable generation with:

- Availability factors correlated with inflows
- Curtailment decisions
- Capacity credit calculations

**Planned Formulation**:

**Variables**:

| Variable       | Domain                                  | Units | Description                 |
| -------------- | --------------------------------------- | ----- | --------------------------- |
| $g^{nc}_{r,k}$ | $[0, \bar{G}_r \cdot \alpha_r(\omega)]$ | MW    | Non-controllable generation |
| $\kappa_{r,k}$ | $\geq 0$                                | MW    | Curtailment                 |

**Generation Constraint**:

$$
g^{nc}_{r,k} + \kappa_{r,k} = \bar{G}_r \cdot \alpha_r(\omega)
$$

where $\alpha_r(\omega) \in [0, 1]$ is the stochastic availability factor.

**Curtailment Penalty**:

$$
+ \sum_{r \in \mathcal{R}} c^{curt} \cdot \kappa_{r,k} \cdot \Delta t_k
$$

**Data Model** (from DATA_MODEL §3.5.7): The `system/non_controllable_sources.json` schema defines source type, bus assignment, capacity, and curtailment settings. Generation models in `scenarios/non_controllable_models.parquet` provide mean and standard deviation per source per stage. Correlation with inflows is supported via `correlation.json` blocks. Output schema in `simulation/non_controllables/` is defined.

**Why Deferred**: Requires:

- Stochastic generation scenario infrastructure (similar to inflow scenarios)
- Correlation structure between renewables and hydro inflows
- Curtailment decision variables in LP
- New output schema for non-controllable results

**Prerequisites**:

- Scenario generation supports correlated non-controllable sources
- LP builder integrates curtailment variables and bus balance
- Output writer handles non-controllable results

**Estimated Effort**: Medium (2-3 weeks). LP formulation is simple; main effort is scenario generation pipeline.

## C.6 FPHA Enhancements

**Status**: DEFERRED (Partial -- core FPHA implemented)

**Description**: Advanced extensions to the FPHA (Four-Point Head Approximation) model for improved accuracy in hydroelectric production function modeling.

### C.6.1 Variable Efficiency Curves

**Current**: Constant turbine-generator efficiency $\eta_{ref}$.

**Enhancement**: Flow-dependent efficiency using characteristic curves:

$$
\eta(q) = \eta_{max} \times f\left(\frac{q}{q_{nom}}\right)
$$

where $f$ is a hill chart approximation, typically:

$$
f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 \quad \text{for } x \in [x_{min}, 1]
$$

**Data Requirements**:

- Efficiency curve coefficients in `hydro_production_data.parquet`
- `efficiency_type = "flow_dependent"`
- `efficiency_coeffs = [a_0, a_1, a_2, a_3]`

**Impact on FPHA**: Hyperplane fitting must use $\eta(q_j)$ at each grid point. Increases nonlinearity captured by the approximation. More planes may be needed for same accuracy.

### C.6.2 Pumped Hydro Production Function

**Current**: Pumping modeled separately from generation.

**Enhancement**: Unified production function for reversible hydro plants:

**Generation Mode** (standard FPHA):

$$
g_{h,k}^{gen} \leq \phi(v, q^{gen}, s)
$$

**Pumping Mode** (reversed inequality -- pumping power increases with head):

$$
p_{h,k}^{pump} \geq \phi_{pump}(v, q^{pump}, s)
$$

**Pumping Production Function**:

$$
p_{pump} = \frac{\rho \times q^{pump} \times h_{net,pump}}{\eta_{pump}}
$$

where $h_{net,pump}$ = pumping head (downstream to upstream) and $\eta_{pump}$ = pumping efficiency (typically 0.85-0.90).

**Hyperplane Form**:

$$
p_{h,k}^{pump} \geq \gamma_{0,pump}^m + \gamma_{v,pump}^m \cdot v_h^{avg} + \gamma_{q,pump}^m \cdot q_{h,k}^{pump}
$$

**Operational Constraints**:

- Mutual exclusion: $q_{h,k}^{gen} \times q_{h,k}^{pump} = 0$ (nonlinear)
- Alternative: Big-M or SOS1 constraints (introduces integer variables)
- POWE.RS approach: Allow simultaneous gen/pump with high penalty (relaxation)

### C.6.3 Dynamic FPHA Recomputation

**Current**: FPHA hyperplanes fixed per stage configuration.

**Enhancement**: Recompute hyperplanes based on expected operating region.

**Approach**:

1. Track volume distribution per stage across forward passes
2. Adjust $[v_{min}, v_{max}]$ to cover observed operation
3. Generate new planes for updated windows
4. Manage cut validity after FPHA updates

**Configuration** (future):

```json
{
  "fpha_config": {
    "dynamic_recomputation": {
      "enabled": true,
      "recompute_every_n_iterations": 50,
      "window_adaptation": "narrow_only",
      "percentile_margin": 5
    }
  }
}
```

> **Warning**: Dynamic recomputation changes the LP structure across iterations. This may affect SDDP convergence guarantees.

**Why Deferred**: Core FPHA is implemented; these are accuracy improvements requiring:

- Variable efficiency: Hill chart data and fitting infrastructure
- Pumped hydro: Unified gen/pump production function with mutual exclusion
- Dynamic recomputation: Online hyperplane refitting with cut validity management

**Prerequisites**:

- Core FPHA operational and validated
- Hyperplane fitting infrastructure supports refitting
- Performance baseline established to measure improvements

**Estimated Effort**: Medium-Large (2-4 weeks total across all three sub-features).

## C.7 Temporal Scope Decoupling

**Status**: DEFERRED

**Description**: Advanced temporal decomposition inspired by SPARHTACUS that decouples the physical time resolution (decision dynamics) from the SDDP stage decomposition (Benders cut generation points). Enables flexible multi-resolution modeling with controlled cut growth.

**Motivation**: In conventional SDDP, three temporal scopes are tightly coupled: stage (SDDP decomposition unit), decision period (physical time resolution), and stochastic process (uncertainty realization base). This forces a trade-off between fine temporal resolution (accurate physics, exponential cut growth) and coarse resolution (manageable cuts, poor short-term dynamics).

**Three Independent Temporal Scopes** (SPARHTACUS nomenclature):

| Scope                         | Portuguese Term                 | POWE.RS Current                   | Purpose                                                |
| ----------------------------- | ------------------------------- | --------------------------------- | ------------------------------------------------------ |
| **Optimization Period**       | Periodo de otimizacao           | `stages[t]`                       | SDDP decomposition unit, Benders cut generation        |
| **Study Period**              | Periodo de estudo               | _(coupled to stage)_              | Physical time resolution for constraints and decisions |
| **Stochastic Process Period** | Periodo do processo estocastico | `inflow_models.parquet` per stage | Base for uncertainty realization                       |

**Extended Multi-Period Stage Subproblem**:

Let stage $t$ contain $K_t$ decision periods indexed by $k = 1, \ldots, K_t$.

$$
V_t(x_t, \{\omega_{t,k}\}_{k=1}^{K_t}) = \min_{\{y_{t,k}\}_{k=1}^{K_t}} \left\{ \sum_{k=1}^{K_t} c_{t,k}^\top y_{t,k} + \mathbb{E} [V_{t+1}(x_{t+1}, \cdot)] \right\}
$$

Subject to:

- **Period 1 constraints**: $A_{t,1} y_{t,1} = b_{t,1}(\omega_{t,1}) - B_t x_t$
- **Period k constraints**: $A_{t,k} y_{t,k} + D_{t,k} y_{t,k-1} = b_{t,k}(\omega_{t,k})$ for $k = 2, \ldots, K_t$
- **State transition**: $x_{t+1} = E_t y_{t,K_t} + F_t x_t$

**Key property**: State variable dimension is unchanged -- cuts reference only $x_t$ at stage boundaries, not intermediate periods. LP size increases by factor $K_t$ but cut count remains constant.

**LP Size Impact**:

| Configuration                    | Stages | Avg LP Size | Cut Pool             |
| -------------------------------- | ------ | ----------- | -------------------- |
| Standard monthly                 | 60     | 1500 vars   | ~1200 cuts @ 20 iter |
| Hybrid (4 weekly + rest monthly) | 60     | ~1600 vars  | ~1200 cuts           |
| All weekly                       | 260    | 1500 vars   | ~5200 cuts @ 20 iter |

**Why Deferred**: Requires significant architectural changes:

- Multi-period LP construction within single stage
- Modified forward/backward pass logic
- Data model changes (periods array within stages)
- Interaction with chronological blocks (nested: Stage → Periods → Blocks)

**Prerequisites**:

- Core SDDP training loop validated and stable
- Chronological block formulation operational
- Performance profiling shows cut growth is the bottleneck

**Estimated Effort**: Large (4-6 weeks). Fundamental change to LP construction, data model, and algorithm.

**References**:

- SPARHTACUS/SPTcpp: [Escopo Temporal](https://github.com/SPARHTACUS/SPTcpp/wiki/Escopo-Temporal)
- Pereira, M.V.F., & Pinto, L.M.V.G. (1991): Original SDDP paper with monthly stages

## C.8 CEPEL PAR(p)-A Variant

**Status**: DEFERRED

**Description**: CEPEL's PAR(p)-A model (referenced in Rel-1941_2021) extends the standard PAR(p) with:

- **Order constraint**: Maximum AR order often fixed at 12 (annual cycle)
- **Stationarity enforcement**: Coefficients adjusted to ensure $\sum_\ell \psi_{m,\ell} < 1$
- **Lognormal transformation**: Working with $\ln(a_{h,t})$ for strictly positive inflows
- **Regional correlation**: Cross-correlation between hydros in the same river basin

**Why Deferred**: The standard PAR(p) model covers most practical use cases. The lognormal variant is primarily relevant for basins with highly skewed inflow distributions where negative synthetic inflows become problematic. The inflow non-negativity handling strategies (see [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md)) provide adequate mitigation for the standard model.

**Prerequisites**:

- Standard PAR(p) fitting and validation operational
- Lognormal transformation infrastructure (log-space fitting, back-transformation)
- Validation suite comparing PAR(p) vs. PAR(p)-A on representative basins

**Data Model Compatibility**: The current input format (`inflow_seasonal_stats.parquet` + `inflow_ar_coefficients.parquet`) supports PAR(p)-A — the lognormal transformation is applied to history before computing seasonal stats, and the resulting μ, s, ψ values are stored in the same schema.

**Estimated Effort**: Small-Medium (1-2 weeks). Mathematical extensions are straightforward; main effort is validation and testing.

**Reference**: CEPEL Rel-1941_2021.

---

## Additional Deferred Algorithm Variants

The following algorithm variants from DATA_MODEL §3.2 are also deferred:

### Pipelined Backward Pass

Overlapped computation/communication: each stage uses $V_{t+1}^{k-1}$ from the previous iteration. Produces looser cuts but may reduce wall-clock time when communication latency dominates.

**Why Deferred**: Should be implemented only when profiling data shows barrier synchronization accounts for >30% of backward pass time.

### Risk-Adjusted Forward Passes

Oversample scenarios from distribution tails, improving exploration of worst-case outcomes for risk-averse policies. Configured via `training.forward_pass.type = "risk_adjusted"` with an `alpha` parameter.

**Why Deferred**: Requires integration with risk measure configuration and performance benchmarking. Default uniform sampling is sufficient for most applications.

### Objective States

Extends SDDP to handle exogenous random processes that affect objective coefficients (e.g., electricity spot prices following an AR process). Uses inner approximation (Lipschitz interpolation) for the value function component that depends on objective states.

**Why Deferred**: Not typically needed for hydrothermal dispatch where prices are marginal cost-based. Can often be approximated by scenario-based approaches.

### Belief States (POMDP)

Extends SDDP to partially observable Markov decision processes where the agent maintains a probability distribution (belief) over hidden states. Useful for modeling phenomena like unobservable climate regimes.

**Why Deferred**: Research-level feature with limited practical benefit for most hydrothermal applications. Can often be approximated by expanding Markov states.

### Duality Handlers (Lagrangian Relaxation)

Methods to generate valid cuts from MIP subproblems when integer variables are present (e.g., unit commitment). Includes Lagrangian relaxation and strengthened Benders cuts.

**Why Deferred**: Significant complexity in Lagrangian multiplier updates. Alternative: solve unit commitment deterministically after SDDP provides marginal values.

## Cross-References

- [Configuration Reference](../05-config/configuration-reference.md) -- Config options for implemented features
- [SDDP Algorithm](../01-math/sddp-algorithm.md) -- Core algorithm that deferred features extend
- [Hydro Production Models](../01-math/hydro-production-models.md) -- Base FPHA that C.6 enhances
- [Cut Management](../01-math/cut-management.md) -- Cut formulation that multi-cut (C.3) modifies
- [Risk Measures](../01-math/risk-measures.md) -- Risk framework that Markovian (C.4) and risk-adjusted passes extend
- [Block Formulations](../01-math/block-formulations.md) -- Block structure that temporal decoupling (C.7) generalizes
- [PAR Inflow Model](../01-math/par-inflow-model.md) -- Standard PAR(p) that CEPEL PAR(p)-A (C.8) extends
- [Equipment Formulations](../01-math/equipment-formulations.md) -- Thermal formulations that GNL (C.1) extends
- [Input System Entities](../02-data-model/input-system-entities.md) -- Entity schemas for batteries (C.2) and non-controllables (C.5)
