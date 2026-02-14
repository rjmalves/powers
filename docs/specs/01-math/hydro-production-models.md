---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §7.1 (Constant Productivity Model)"
  - "MATHEMATICAL_FORMULATIONS.md §7.2 (FPHA — Four-Point Head Approximation)"
  - "MATHEMATICAL_FORMULATIONS.md §7.3 (Linearized Head Model)"
  - "MATHEMATICAL_FORMULATIONS.md §7.4 (Model Selection Guidelines)"
  - "MATHEMATICAL_FORMULATIONS.md §7.5 (FPHA Data Requirements Summary)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Hydro Production Function Models

## Purpose

This spec defines the hydro generation constraint models supported by POWE.RS, which relate turbined flow and reservoir storage to electrical output. Three models of increasing fidelity are provided: constant productivity, linearized head, and FPHA. The choice trades off accuracy vs. computational cost, and can vary by stage. For variable definitions see [notation conventions](../00-overview/notation-conventions.md); for LP integration see [LP formulation](lp-formulation.md); for hydro element descriptions see [system elements](system-elements.md).

## 1. Constant Productivity Model

The simplest model assumes a linear relationship:

$$
g_{h,k} = \rho_h \cdot q_{h,k}
$$

where $\rho_h$ (MW per m³/s) is the hydro productivity, typically:

$$
\rho_h = \frac{9.81 \times \eta_h \times H^{ref}_h}{1000}
$$

with:

- $\eta_h$ = turbine efficiency (typically 0.85–0.92)
- $H^{ref}_h$ = reference net head (meters)

**Characteristics:** 1 equality constraint per hydro per block. Simple and fast, but ignores head variation with storage.

## 2. FPHA (Four-Point Head Approximation)

For accurate modeling of hydroelectric generation, FPHA (Função de Produção Hidrelétrica Aproximada) captures the nonlinear relationship between storage, flow, spillage, and generation through a piecewise-linear approximation.

### 2.1 Notation Mapping

This section uses consistent notation with the LP formulation. The following table maps POWE.RS symbols to equivalent CEPEL/Portuguese terminology for practitioners familiar with DECOMP/DESSEM:

| POWE.RS    | CEPEL/Portuguese               | Description                 | Units |
| ---------- | ------------------------------ | --------------------------- | ----- |
| $\phi$     | FPH                            | Hydro production function   | MW    |
| $v$        | $V$                            | Reservoir storage           | hm³   |
| $q$        | $Q$                            | Turbined flow               | m³/s  |
| $s$        | $S$                            | Spillage                    | m³/s  |
| $g_h$      | GH                             | Hydro generation            | MW    |
| $h_{fore}$ | $h_{mon}$ (montante)           | Forebay (upstream) level    | m     |
| $h_{tail}$ | $h_{jus}$ (jusante)            | Tailrace (downstream) level | m     |
| $h_{net}$  | $h_{liq}$ (líquida)            | Net head                    | m     |
| $h_{loss}$ | $h_{PerdH}$ (perda hidráulica) | Hydraulic losses            | m     |
| $q_{out}$  | $Q_{jus}$                      | Total downstream outflow    | m³/s  |
| $q_{lat}$  | $Q_{lat}$                      | Lateral tributary flow      | m³/s  |

### 2.2 Exact Production Function

The **exact hydroelectric production function** relates generation to the operating state:

$$
\phi(v, q, q_{out}) = \rho(q, h_{net}) \times q \times h_{net}
$$

where:

- $v$ = reservoir storage volume (hm³)
- $q$ = turbined flow (m³/s)
- $q_{out}$ = total downstream outflow affecting tailrace level (m³/s)
- $h_{net}$ = net head (m)
- $\rho$ = specific productivity (MW·s/m⁴)

The **net head** is computed as:

$$
h_{net}(v, q, q_{out}) = h_{fore}(v) - h_{tail}(q_{out}) - h_{loss}(q)
$$

where:

- $h_{fore}(v)$ = forebay (upstream reservoir) level as function of storage
- $h_{tail}(q_{out})$ = tailrace (downstream channel) level as function of total outflow
- $h_{loss}(q)$ = hydraulic head losses in penstock and turbines

**Why linearization is needed**: $\phi$ is nonlinear in $(v, q)$ due to the bilinear product $q \times h_{net}$, nonlinear topology functions $h_{fore}(v)$ and $h_{tail}(q_{out})$, and flow-dependent hydraulic losses. For LP formulation, we approximate $\phi$ with a set of linear hyperplanes.

### 2.3 Topology Functions (POWE.RS Approach)

Unlike CEPEL models which use 4th-degree polynomial fits, POWE.RS uses **tabular data with linear interpolation** for topology functions — more transparent, easier to validate against surveyed data, and flexible for any reservoir geometry.

#### Forebay Level $h_{fore}(v)$

The upstream water level is obtained from `hydro_geometry.parquet`:

| volume_hm3 | height_m | area_km2 |
| ---------- | -------- | -------- |
| $v_1$      | $h_1$    | $A_1$    |
| $v_2$      | $h_2$    | $A_2$    |
| ...        | ...      | ...      |

**Interpolation**: For storage $v$ where $v_i \leq v < v_{i+1}$:

$$
h_{fore}(v) = h_i + \frac{h_{i+1} - h_i}{v_{i+1} - v_i} \times (v - v_i)
$$

#### Tailrace Level $h_{tail}(q_{out})$

The downstream water level depends on total outflow. From `hydro_production_data.parquet`:

**Polynomial model** (CEPEL-compatible):

$$
h_{tail}(q_{out}) = c_0 + c_1 q_{out} + c_2 q_{out}^2 + c_3 q_{out}^3 + c_4 q_{out}^4
$$

**Piecewise-linear model** (POWE.RS native):

| outflow_m3s | tailrace_m   |
| ----------- | ------------ |
| $q_1$       | $h_{tail,1}$ |
| $q_2$       | $h_{tail,2}$ |
| ...         | ...          |

With linear interpolation between points.

**Total downstream flow**: $q_{out} = q + s + q_{lat}$ where:

- $q$ = turbined flow
- $s$ = spillage
- $q_{lat}$ = lateral inflows between reservoir and tailrace (from tributaries or upstream plants)

#### Hydraulic Losses $h_{loss}(q)$

Two models are supported:

**Factor model** (proportional to gross head):

$$
h_{loss}(q) = k_{loss} \times (h_{fore} - h_{tail})
$$

where $k_{loss}$ is typically 0.01–0.05 (1–5% losses).

**Constant model** (fixed head loss):

$$
h_{loss}(q) = \Delta h_{const}
$$

where $\Delta h_{const}$ is in meters (typically 1–5m).

**Flow-dependent model** (future extension):

$$
h_{loss}(q) = k_q \times q^2
$$

This captures the quadratic friction losses in penstocks.

### 2.4 Variable Productivity Model

The **specific productivity** converts hydraulic power to electrical power:

$$
\rho(q, h_{net}) = \frac{g \times \eta(q)}{1000}
$$

where:

- $g = 9.81$ m/s² (gravitational acceleration)
- $\eta(q)$ = turbine-generator efficiency (dimensionless, typically 0.85–0.93)
- Factor 1000 converts W to kW

The full generation formula becomes:

$$
g_h = \frac{9.81 \times \eta \times q \times h_{net}}{1000} \quad \text{[MW]}
$$

**Constant efficiency** (current implementation):

$$
\eta(q) = \eta_{ref} \quad \text{(constant)}
$$

**Variable efficiency** (future extension, see [deferred features](../06-deferred/deferred-features.md)):

$$
\eta(q) = \eta_{max} \times f\left(\frac{q}{q_{nom}}\right)
$$

where $f$ is a characteristic curve peaking near nominal flow.

#### Reference Productivity

For the constant productivity model, the reference value is:

$$
\rho_{ref} = \frac{9.81 \times \eta_{ref} \times h_{ref}}{1000} \quad \text{[MW per m³/s]}
$$

where $h_{ref}$ is the reference net head (typically at 65% storage).

### 2.5 Hyperplane Fitting Algorithm

POWE.RS supports two approaches for obtaining FPHA hyperplanes:

1. **Pre-fitted**: Read coefficients from `fpha_hyperplanes.parquet`
2. **Computed**: Generate from topology data during preprocessing

**Input:**

- Topology data: $h_{fore}(v)$, $h_{tail}(q_{out})$, $h_{loss}(q)$
- Operating bounds: $[v_{min}, v_{max}]$, $[0, q_{max}]$
- Discretization: $n_v$ volume points, $n_q$ turbine flow points
- Reference spillage: $s_{ref}$ (typically 0 or average expected spillage)

**Output:**

- Set of hyperplanes $\{(\gamma_0^m, \gamma_v^m, \gamma_q^m, \gamma_s^m)\}_{m=1}^M$
- Correction factor $\kappa$ (note: we use $\kappa$ to avoid collision with cut intercept $\alpha$)

**Algorithm: FPHA_Fit**

1. **DISCRETIZE operating window**
   - Create volume grid: $v_{grid} = \text{linspace}(v_{min}, v_{max}, n_v)$
   - Create flow grid: $q_{grid} = \text{linspace}(0, q_{max}, n_q)$
   - Stage-dependent configuration:
     - Near-term stages: higher resolution ($n_v = 7$, $n_q = 15$)
     - Far-future stages: lower resolution ($n_v = 3$, $n_q = 5$)

2. **EVALUATE exact production function at grid points**

   For each $(v_i, q_j)$ in grid, compute:
   - Forebay head: $h_{fore} = \text{interpolate}(\text{geometry\_table}, v_i)$
   - Total outflow: $q_{out} = q_j + s_{ref}$
   - Tailrace head: $h_{tail} = \text{interpolate}(\text{tailrace\_table}, q_{out})$
   - Head loss: $h_{loss} = \text{compute\_loss}(q_j, h_{fore}, h_{tail})$
   - Net head: $h_{net} = h_{fore} - h_{tail} - h_{loss}$
   - Generation: If $h_{net} > 0$, then $g_{exact}[i,j] = \rho \times q_j \times h_{net}$, else $g_{exact}[i,j] = 0$

3. **BUILD convex hull of generation surface**
   - Create 3D point cloud: $\text{points} = \{(v_i, q_j, g_{exact}[i,j]) \mid \forall (i,j) \text{ with } g > 0\}$
   - Compute upper convex hull (concave envelope where generation $\leq$ surface):
     - Run qhull: $\text{hull} = \text{qhull}(\text{points}, \text{options} = \text{"Qt Qc"})$
   - Extract facets with downward-facing normals (upper hull):
     - For each facet in `hull.facets`: if `facet.normal[2]` $< 0$ (upward in $g_h$ direction), extract plane coefficients $(\gamma_0, \gamma_v, \gamma_q)$

4. **COMPUTE correction factor $\kappa$**

   Apply correction to ensure FPHA $\leq \phi$ everywhere:
   - Initialize $\kappa = 1.0$
   - For each $(v_i, q_j)$ in grid:
     - Compute $g_{fpha} = \max_m \{\gamma_0^m + \gamma_v^m \cdot v_i + \gamma_q^m \cdot q_j\}$
     - If $g_{exact}[i,j] > 0$ AND $g_{fpha} > 0$: update $\kappa = \min(\kappa, g_{exact}[i,j] / g_{fpha})$
   - Scale all intercepts: $\gamma_0^m = \kappa \times \gamma_0^m$ for each plane $m$

   Optional MSE minimization: $\kappa = \arg\min_\kappa \sum_{i,j} (\kappa \cdot g_{fpha}[i,j] - g_{exact}[i,j])^2$

5. **ADD spillage dimension (secant approximation)**

   Spillage affects tailrace level, reducing net head:
   - Compute tailrace sensitivity:
     $$\frac{dh_{tail}}{ds} = \frac{h_{tail}(q_{ref} + s_{ref} + \Delta s) - h_{tail}(q_{ref} + s_{ref})}{\Delta s}$$
   - For each plane $m$, add spillage coefficient:
     $$\gamma_s^m = -\rho \times q_{ref} \times \frac{dh_{tail}}{ds}$$

6. **RETURN planes and metadata**
   - `planes`: $\{(\gamma_0^m, \gamma_v^m, \gamma_q^m, \gamma_s^m) \mid m = 1, \ldots, M\}$
   - `kappa`: $\kappa$
   - `num_planes`: $M$
   - `fitting_bounds`: $\{v_{min}, v_{max}, q_{max}\}$
   - `grid_resolution`: $\{n_v, n_q\}$

**Qhull implementation options:** Rust `convex_hull` crate, external qhull via FFI/subprocess, or simplified Delaunay triangulation with upper facet filtering.

### 2.6 Correction Factor Calculation

The correction factor $\kappa$ ensures the approximation is conservative (never overestimates generation).

> **Notation note**: We use $\kappa$ (kappa) for the FPHA correction factor to avoid collision with $\alpha$, which is used for Benders cut intercepts (see [cut management](cut-management.md)).

#### Worst-Case Approach (Default)

$$
\kappa = \min_{(v,q) \in \text{grid}} \left\{ \frac{\phi(v, q)}{\max_m (\gamma_0^m + \gamma_v^m v + \gamma_q^m q)} \right\}
$$

This guarantees $g_{h,FPHA} \leq \phi$ everywhere in the operating region.

#### MSE Minimization Approach

$$
\kappa = \arg\min_\kappa \sum_{(v_i, q_j)} \left( \kappa \cdot g_{FPHA}(v_i, q_j) - \phi(v_i, q_j) \right)^2
$$

Closed-form solution:

$$
\kappa = \frac{\sum_{i,j} g_{FPHA} \cdot \phi}{\sum_{i,j} g_{FPHA}^2}
$$

#### Typical Values

| Reservoir Type    | Typical $\kappa$ | Notes                        |
| ----------------- | ---------------- | ---------------------------- |
| High-head storage | 0.97–0.99        | Significant head variation   |
| Medium-head       | 0.98–1.00        | Moderate approximation error |
| Run-of-river      | 0.99–1.00        | Nearly constant head         |

### 2.7 Spillage and Lateral Flow Effects

#### Downstream Level Dependency

The tailrace level depends on total flow through the downstream channel:

$$
q_{out} = q + s + q_{lat}
$$

where $q_{lat}$ includes:

- Lateral tributaries entering between dam and tailrace
- Outflow from upstream plants in cascade
- Return flows from irrigation or other withdrawals

#### Secant Approximation for Spillage

Since spillage $s$ affects tailrace level, it indirectly affects generation. The FPHA constraint incorporates this through $\gamma_s$:

$$
\gamma_s^m = -\rho \times q_{ref} \times \frac{\partial h_{tail}}{\partial q_{out}} \bigg|_{q_{out,ref}}
$$

**Physical interpretation**: Each additional m³/s of spillage raises the tailrace by $\partial h_{tail}/\partial q_{out}$ meters, reducing net head and thus generation.

**Sign convention**: $\gamma_s^m < 0$ because spillage reduces generation capacity.

#### Cascade Effects (Advanced)

In cascade systems, upstream spillage affects downstream tailrace levels with a time delay. This creates cross-plant coupling not captured in standard FPHA. Current assumptions:

- Each plant's FPHA is independent
- Cascade effects are captured through average expected flows
- Future enhancement could add cross-plant correction terms

### 2.8 LP Integration

#### Final FPHA Constraint

For each hydro $h$, block $k$, and plane $m \in \mathcal{M}_h$:

$$
g_{h,k}^{hy} \leq \kappa \times \left( \gamma_0^m + \gamma_v^m \cdot v_h^{avg} + \gamma_q^m \cdot q_{h,k} + \gamma_s^m \cdot s_{h,k} \right)
$$

Or equivalently with pre-scaled coefficients:

$$
g_{h,k}^{hy} \leq \tilde{\gamma}_0^m + \gamma_v^m \cdot v_h^{avg} + \gamma_q^m \cdot q_{h,k} + \gamma_s^m \cdot s_{h,k}
$$

where $\tilde{\gamma}_0^m = \kappa \times \gamma_0^m$.

#### Average Storage Computation

The average storage $v^{avg}_h$ over the stage depends on configuration:

**Option A: Simple Average (Default)**

$$
v^{avg}_h = \frac{\hat{v}_h + v_h}{2}
$$

where $\hat{v}_h$ is incoming storage and $v_h$ is end-of-stage storage.

**Option B: Block-Weighted Average**

$$
v^{avg}_h = \sum_{k} w_k \cdot v_{h,k}^{mid}
$$

where $w_k$ is the block weight (duration fraction) and $v_{h,k}^{mid}$ is the mid-block storage.

#### Generation as Independent Variable

When using FPHA, the generation variable $g_{h,k}^{hy}$ is **not** directly computed from turbined flow. Instead:

1. Generation is a free LP variable bounded by $[0, \bar{G}_h]$
2. FPHA constraints (one per plane) provide upper bounds
3. The optimizer maximizes generation subject to FPHA constraints
4. At optimum, generation "touches" one of the FPHA planes

**Key insight**: Because minimizing cost includes maximizing hydro generation (which has zero fuel cost), the optimizer naturally pushes generation to the FPHA surface boundary.

#### Slack Variables for Soft Constraints

For numerical robustness, FPHA constraints can include slack variables:

$$
g_{h,k}^{hy} - \sigma_{h,k,m}^{fpha} \leq \tilde{\gamma}_0^m + \gamma_v^m \cdot v_h^{avg} + \gamma_q^m \cdot q_{h,k} + \gamma_s^m \cdot s_{h,k}
$$

where $\sigma_{h,k,m}^{fpha} \geq 0$ with high penalty cost. This allows the LP to remain feasible even if operating outside the FPHA validity region.

### 2.9 Water Value and Benders Cuts

The FPHA formulation affects how water values are computed and propagated through Benders cuts.

#### Dual Variables

Let $\pi_m^{fpha}$ be the dual variable for FPHA constraint $m$. At optimum:

$$
\frac{\partial \mathcal{L}}{\partial g_{h,k}^{hy}} = -c_k^{deficit} + \sum_m \pi_m^{fpha} = 0
$$

where $c_k^{deficit}$ is the marginal cost of deficit in block $k$.

#### Water Value Derivation

The marginal value of storage $v_h$ includes the FPHA contribution:

$$
\frac{\partial \text{Cost}}{\partial v_h} = \underbrace{\pi_h^{balance}}_{\text{direct value}} + \underbrace{\sum_m \pi_m^{fpha} \cdot \gamma_v^m}_{\text{FPHA contribution}}
$$

For the hydro balance constraint with dual $\pi_h^{balance}$:

$$
v_h = \hat{v}_h + a_h - q_h - s_h - w_h - e_h
$$

#### Cut Coefficient for Storage

The Benders cut coefficient for storage state variable $\hat{v}_h$ is:

$$
\beta_{\hat{v}_h} = \pi_h^{balance} + \frac{1}{2} \sum_m \pi_m^{fpha} \cdot \gamma_v^m
$$

The factor $\frac{1}{2}$ appears because $v^{avg} = (\hat{v}_h + v_h)/2$, so $\partial v^{avg}/\partial \hat{v}_h = 1/2$.

#### Model Transition Considerations

When a hydro transitions between production models across stages:

| Transition                     | Cut Interpretation                      | Action                            |
| ------------------------------ | --------------------------------------- | --------------------------------- |
| Constant → FPHA                | Cuts at stage $t$ use constant model    | Cut valid but conservative        |
| FPHA → Constant                | Stage $t+1$ backward pass uses constant | May overestimate value            |
| FPHA → FPHA (different params) | Parameters change                       | Cuts remain valid if conservative |

**Recommendation**: When using stage-dependent FPHA configuration, ensure the FPHA at stage $t$ is at least as conservative as stage $t+1$ for cut validity.

### 2.10 Stage-Dependent Configuration

POWE.RS allows different FPHA configurations for different stages, enabling a trade-off between accuracy and computational efficiency.

#### Fitting Window Selection

The operating range $[v_{min}, v_{max}]$ for FPHA fitting can be stage-dependent:

**Near-term stages (0–24):**

- Use full storage range: $[v_{min}^{phys}, v_{max}^{phys}]$
- Higher resolution: $n_v = 7$, $n_q = 15$
- All operating scenarios possible

**Medium-term stages (25–60):**

- Narrower range based on expected operation: $[v_{10\%}, v_{90\%}]$
- Medium resolution: $n_v = 5$, $n_q = 10$
- Focus on likely operating region

**Far-future stages (61+):**

- Conservative range centered on equilibrium: $[v_{25\%}, v_{75\%}]$
- Lower resolution: $n_v = 3$, $n_q = 5$
- Prioritize computational efficiency

#### Configuration Schema

```json
{
  "hydro_id": 42,
  "stage_ranges": [
    {
      "start_stage_id": 0,
      "end_stage_id": 24,
      "model": "fpha",
      "fpha_config": {
        "source": "computed",
        "volume_discretization_points": 7,
        "turbine_discretization_points": 15,
        "fitting_window": {
          "volume_min_hm3": null,
          "volume_max_hm3": null,
          "volume_min_percentile": null,
          "volume_max_percentile": null
        }
      }
    },
    {
      "start_stage_id": 25,
      "end_stage_id": 60,
      "model": "fpha",
      "fpha_config": {
        "source": "computed",
        "volume_discretization_points": 5,
        "turbine_discretization_points": 10,
        "fitting_window": {
          "volume_min_percentile": 10,
          "volume_max_percentile": 90
        }
      }
    },
    {
      "start_stage_id": 61,
      "end_stage_id": null,
      "model": "constant_productivity"
    }
  ]
}
```

#### Training vs Simulation Phases

Future enhancement: different FPHA configurations for training (wider fitting windows for cut validity) vs. simulation (tighter windows based on observed trajectories). This requires careful handling — see [deferred features](../06-deferred/deferred-features.md).

## 3. Linearized Head Model

An intermediate model between constant productivity and full FPHA:

$$
g_{h,k}^{hy} = \rho_{ref} \cdot q_{h,k} \cdot \left( k_0 + k_V \cdot v_h^{avg} \right)
$$

where:

- $k_0, k_V$ are linearization coefficients derived from $h_{mon}(V)$
- $k_0 = 1 - k_V \cdot V_{ref}$ (normalization at reference volume)
- $k_V = \frac{1}{H_{ref}} \cdot \frac{dh_{mon}}{dV}\bigg|_{V_{ref}}$

**Characteristics:** Single constraint (bilinear approximation). Captures first-order head variation with storage but not spillage effects. Suitable for medium-term stages.

## 4. Model Selection Guidelines

| Scenario                        | Recommended Model      | Rationale                          |
| ------------------------------- | ---------------------- | ---------------------------------- |
| High-head storage reservoirs    | FPHA                   | Significant head variation (>20%)  |
| Large storage variation plants  | FPHA                   | Operating across wide volume range |
| Run-of-river plants             | Constant productivity  | Nearly constant head               |
| Initial algorithm testing       | Constant productivity  | Fast iteration, debug focus        |
| Production studies (near-term)  | FPHA                   | Accuracy for operational decisions |
| Production studies (far-future) | Constant or linearized | Computational efficiency           |
| Post-optimization validation    | Compare all models     | Verify approximation quality       |

## 5. FPHA Data Requirements Summary

| Data Source                     | Required Fields                    | Used For                       |
| ------------------------------- | ---------------------------------- | ------------------------------ |
| `hydro_geometry.parquet`        | volume_hm3, height_m               | $h_{mon}(V)$ interpolation     |
| `hydro_production_data.parquet` | tailrace_coeffs or table           | $h_{jus}(Q_{jus})$ computation |
| `hydro_production_data.parquet` | hydraulic_loss_type, value         | $h_{PerdH}(Q)$ computation     |
| `hydros.json`                   | productivity_mw_per_m3s            | Reference $\rho_{ref}$         |
| `fpha_hyperplanes.parquet`      | gamma_0, gamma_v, gamma_q, gamma_s | Pre-fitted planes (optional)   |
| `hydro_production_models.json`  | fpha_config per stage              | Fitting configuration          |

## Cross-References

- [Notation conventions](../00-overview/notation-conventions.md) — variable and set definitions ($g_h$, $q_h$, $v_h$, $s_h$, $\rho_h$)
- [System elements](system-elements.md) — hydro plant element description and decision variables
- [LP formulation](lp-formulation.md) — how production constraints integrate into the assembled LP
- [Block formulations](block-formulations.md) — block-level structure within which production constraints operate
- [Cut management](cut-management.md) — Benders cut generation affected by FPHA dual variables
- [Configuration reference](../05-config/configuration-reference.md) — FPHA and production model configuration settings
- [Deferred features](../06-deferred/deferred-features.md) — variable efficiency, training vs simulation phases
