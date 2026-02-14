---
status: draft
review_priority: 2-high
source_sections:
  - "MATHEMATICAL_FORMULATIONS.md §9 (9.1-9.11)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from MATHEMATICAL_FORMULATIONS.md §9"
---

# PAR(p) Inflow Model

## Purpose

This spec defines the Periodic Autoregressive model of order $p$ (PAR(p)) used to capture temporal correlation in inflow time series, including the complete five-step fitting procedure, parameter set, model order selection, and validation checks.

## 1. PAR(p) Model Definition

The **Periodic Autoregressive model of order p** (PAR(p)) captures temporal correlation in inflow time series while accounting for seasonal variation in parameters. For hydro $h$ at stage $t$ corresponding to season $m(t)$:

$$
a_{h,t} = \mu_{m(t)} + \sum_{\ell=1}^{p} \psi_{m(t),\ell} \left( a_{h,t-\ell} - \mu_{m(t-\ell)} \right) + \sigma_{m(t)} \cdot \varepsilon_t
$$

where:

- $a_{h,t}$: Incremental inflow at stage $t$ (m³/s)
- $\mu_{m(t)}$: Seasonal mean for season $m(t)$
- $\psi_{m(t),\ell}$: Autoregressive coefficient for lag $\ell$ in season $m(t)$
- $\sigma_{m(t)}$: Seasonal standard deviation of residuals
- $\varepsilon_t \sim \mathcal{N}(0, 1)$: Innovation (standardized noise)
- $m(t)$: Season/period index for stage $t$ (e.g., month 1-12)

## 2. Notation for Fitting

Let $Y_m = \{a_{h,t} : m(t) = m\}$ be the historical observations for season $m$. Define:

| Symbol           | Description                                  |
| ---------------- | -------------------------------------------- |
| $N_m$            | Number of observations for season $m$        |
| $\bar{a}_m$      | Sample mean for season $m$                   |
| $s_m$            | Sample standard deviation for season $m$     |
| $\gamma_m(\ell)$ | Autocovariance at lag $\ell$ for season $m$  |
| $\rho_m(\ell)$   | Autocorrelation at lag $\ell$ for season $m$ |

## 3. Step 1: Seasonal Means and Standard Deviations

**Seasonal Mean**:

$$
\hat{\mu}_m = \bar{a}_m = \frac{1}{N_m} \sum_{t: m(t) = m} a_{h,t}
$$

**Seasonal Standard Deviation**:

$$
\hat{s}_m = \sqrt{\frac{1}{N_m - 1} \sum_{t: m(t) = m} (a_{h,t} - \bar{a}_m)^2}
$$

## 4. Step 2: Seasonal Autocorrelations

The autocorrelation at lag $\ell$ for season $m$ is computed from standardized deviations.

**Cross-seasonal autocovariance**:

For observations at season $m$ with lag $\ell$ reaching back to season $m - \ell$ (mod $M$, where $M$ is the cycle length):

$$
\hat{\gamma}_m(\ell) = \frac{1}{N_m - 1} \sum_{t: m(t) = m} \left( a_{h,t} - \bar{a}_m \right) \left( a_{h,t-\ell} - \bar{a}_{m-\ell} \right)
$$

**Autocorrelation**:

$$
\hat{\rho}_m(\ell) = \frac{\hat{\gamma}_m(\ell)}{\hat{s}_m \cdot \hat{s}_{m-\ell}}
$$

where $\hat{s}_{m-\ell}$ is the standard deviation of season $m - \ell$ (cyclically, so season 0 = season $M$).

## 5. Step 3: Yule-Walker Equations

For each season $m$, the PAR(p) coefficients $\psi_{m,1}, \ldots, \psi_{m,p}$ are found by solving the **Yule-Walker system**:

$$
\begin{pmatrix}
1 & \hat{\rho}_{m-1}(1) & \hat{\rho}_{m-2}(2) & \cdots & \hat{\rho}_{m-p+1}(p-1) \\
\hat{\rho}_{m}(1) & 1 & \hat{\rho}_{m-1}(1) & \cdots & \hat{\rho}_{m-p+2}(p-2) \\
\hat{\rho}_{m}(2) & \hat{\rho}_{m-1}(1) & 1 & \cdots & \hat{\rho}_{m-p+3}(p-3) \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\hat{\rho}_{m}(p-1) & \hat{\rho}_{m-1}(p-2) & \hat{\rho}_{m-2}(p-3) & \cdots & 1
\end{pmatrix}
\begin{pmatrix}
\psi_{m,1}^* \\
\psi_{m,2}^* \\
\psi_{m,3}^* \\
\vdots \\
\psi_{m,p}^*
\end{pmatrix}
=
\begin{pmatrix}
\hat{\rho}_{m}(1) \\
\hat{\rho}_{m}(2) \\
\hat{\rho}_{m}(3) \\
\vdots \\
\hat{\rho}_{m}(p)
\end{pmatrix}
$$

In matrix notation: $\mathbf{R}_m \boldsymbol{\psi}_m^* = \boldsymbol{r}_m$

where:

- $\mathbf{R}_m$ is the $p \times p$ correlation matrix (Toeplitz-like but with cross-seasonal correlations)
- $\boldsymbol{r}_m = (\hat{\rho}_m(1), \ldots, \hat{\rho}_m(p))^\top$ is the vector of target autocorrelations

**Solution**:

$$
\hat{\boldsymbol{\psi}}_m^* = \mathbf{R}_m^{-1} \boldsymbol{r}_m
$$

## 6. Step 4: Convert to Original Units

The Yule-Walker solution $\psi_{m,\ell}^*$ is for standardized variables. Convert back to original units:

$$
\hat{\psi}_{m,\ell} = \psi_{m,\ell}^* \cdot \frac{\hat{s}_m}{\hat{s}_{m-\ell}}
$$

## 7. Step 5: Residual Standard Deviation

The residual variance for season $m$ is:

$$
\hat{\sigma}_m^2 = \hat{s}_m^2 \left( 1 - \sum_{\ell=1}^{p} \psi_{m,\ell}^* \cdot \hat{\rho}_m(\ell) \right)
$$

The residual standard deviation:

$$
\hat{\sigma}_m = \hat{s}_m \sqrt{1 - \boldsymbol{r}_m^\top \mathbf{R}_m^{-1} \boldsymbol{r}_m}
$$

## 8. Complete PAR(p) Parameter Set

For each hydro $h$ and each season $m \in \{1, \ldots, M\}$ (e.g., $M=12$ for monthly, $M=52$ for weekly):

| Parameter                        | Formula              | Description                 |
| -------------------------------- | -------------------- | --------------------------- |
| $\mu_m$                          | $\bar{a}_m$          | Seasonal mean               |
| $\psi_{m,1}, \ldots, \psi_{m,p}$ | Yule-Walker solution | AR coefficients             |
| $\sigma_m$                       | $\hat{\sigma}_m$     | Residual standard deviation |

## 9. Model Order Selection

The PAR order $p$ can vary by season. Common selection criteria:

1. **AIC (Akaike Information Criterion)**:

   $$
   \text{AIC}_m(p) = N_m \ln(\hat{\sigma}_m^2) + 2p
   $$

2. **BIC (Bayesian Information Criterion)**:

   $$
   \text{BIC}_m(p) = N_m \ln(\hat{\sigma}_m^2) + p \ln(N_m)
   $$

3. **Coefficient significance**: Include lag $\ell$ only if $|\hat{\psi}_{m,\ell}| > 2 / \sqrt{N_m}$

## 10. CEPEL PAR(p)-A Variant (Future Extension)

CEPEL's PAR(p)-A model (referenced in Rel-1941_2021) uses:

- **Order constraint**: Maximum AR order often fixed at 12 (annual cycle)
- **Stationarity enforcement**: Coefficients adjusted to ensure $\sum_\ell \psi_{m,\ell} < 1$
- **Lognormal transformation**: Working with $\ln(a_{h,t})$ for strictly positive inflows
- **Regional correlation**: Cross-correlation between hydros in same river basin

This variant is not currently implemented but the data model supports it via the `ar_order` and `ar_coef_*` columns in `inflow_models.parquet`.

## 11. Validation Checks

After fitting, verify:

1. **Positive residual variance**: $\hat{\sigma}_m^2 > 0$ for all seasons
2. **Stationarity**: Roots of $1 - \sum_\ell \psi_{m,\ell} z^\ell = 0$ lie outside unit circle
3. **Correlation matrix positive definite**: $\mathbf{R}_m$ is invertible
4. **No systematic bias**: Residuals $\varepsilon_t$ have mean near zero

## Cross-References

- [Notation Conventions](../00-overview/notation-conventions.md) — Defines inflow symbols ($a_{h,t}$, $\mu_m$, $\psi_{m,\ell}$, $\sigma_m$) and unit conventions used here
- [Inflow Non-Negativity](inflow-nonnegativity.md) — Methods for handling negative realizations produced by the PAR(p) model
- [Configuration Reference](../05-config/configuration-reference.md) — PAR(p) data loading from `inflow_models.parquet` and runtime configuration
