---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §8 (8.1-8.3, 8.5)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §9 (9.1-9.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §10 (10.1-10.2, 10.5)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §11 (11.1-11.3)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from ARCHITECTURE §8-11"
---

# Scenario Generation

## Purpose

This spec defines the POWE.RS scenario generation pipeline: PAR model preprocessing, correlated noise sampling, external scenario integration with noise inversion, and the scenario memory layout optimized for the forward pass hot-path. For the mathematical definition of the PAR(p) model, see [PAR(p) Inflow Model](../01-math/par-inflow-model.md).

## 1. PAR Model Preprocessing

### 1.1 Overview

The PAR(p) model generates stochastic inflows during training. Before the training loop begins, the raw PAR coefficients (loaded from `inflow_models.parquet`) are preprocessed into a contiguous, cache-friendly layout that eliminates per-stage season lookups on the hot path.

For the full PAR(p) model definition, parameter set, and fitting theory, see [PAR(p) Inflow Model](../01-math/par-inflow-model.md).

### 1.2 Preprocessing Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       PAR Model Preprocessing Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input: inflow_models.parquet (PAR coefficients per hydro x season)             │
│                                                                                  │
│  Step 1: Load PAR Parameters                                                    │
│  - mu[h][m]: seasonal means (12 x N_hydro)                                     │
│  - psi[h][m][l]: AR coefficients (12 x N_hydro x max_order)                    │
│  - sigma[h][m]: residual std dev (12 x N_hydro)                                │
│  - P[h]: AR order per hydro (N_hydro)                                          │
│                              │                                                   │
│                              ▼                                                   │
│  Step 2: Precompute Stage-Specific Coefficients                                 │
│  For each stage t = 1..T, season m = season(t), hydro h:                        │
│    base[h][t] = mu[h][m] - Sum_l psi[h][m][l] * mu[h][m-l]                     │
│    coeff[h][t][l] = psi[h][m][l]  for l = 1..P[h]                              │
│    scale[h][t] = sigma[h][m]                                                     │
│                              │                                                   │
│                              ▼                                                   │
│  Step 3: Initialize Lag State from History                                       │
│  From inflow_history.parquet:                                                    │
│    lag_state[h][l] = historical_inflow[h][t0 - l]  for l = 1..max_order         │
│                              │                                                   │
│                              ▼                                                   │
│  Output: PrecomputedPar structure (contiguous arrays for hot-path access)       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Memory Layout for Hot-Path Access

The `PrecomputedPar` struct uses struct-of-arrays layout for cache efficiency. All arrays are row-major with stage as the outer dimension so that sequential stage iteration within a scenario touches contiguous memory:

```rust
/// Precomputed PAR data optimized for forward pass access pattern
pub struct PrecomputedPar {
    /// Stage -> Hydro -> base value (deterministic component)
    /// Layout: [stage_0_hydro_0, stage_0_hydro_1, ..., stage_T_hydro_N]
    pub base: Vec<f64>,  // T x N_hydro, row-major

    /// Stage -> Hydro -> Lag -> coefficient
    /// Layout: [s0_h0_l1, s0_h0_l2, ..., s0_h0_lP, s0_h1_l1, ...]
    pub coefficients: Vec<f64>,  // T x N_hydro x max_order

    /// Stage -> Hydro -> noise scale (sigma)
    pub scales: Vec<f64>,  // T x N_hydro

    /// Hydro -> AR order
    pub orders: Vec<u8>,  // N_hydro

    /// Dimensions for indexing
    pub n_stages: usize,
    pub n_hydros: usize,
    pub max_order: usize,
}

impl PrecomputedPar {
    #[inline]
    pub fn base(&self, stage: usize, hydro: usize) -> f64 {
        self.base[stage * self.n_hydros + hydro]
    }

    #[inline]
    pub fn coefficients(&self, stage: usize, hydro: usize) -> &[f64] {
        let start = (stage * self.n_hydros + hydro) * self.max_order;
        let order = self.orders[hydro] as usize;
        &self.coefficients[start..start + order]
    }

    /// Compute inflow given lag state and noise
    #[inline]
    pub fn compute_inflow(
        &self, stage: usize, hydro: usize,
        lag_state: &[f64], noise: f64,
    ) -> f64 {
        let base = self.base(stage, hydro);
        let coeffs = self.coefficients(stage, hydro);
        let scale = self.scales[stage * self.n_hydros + hydro];

        let mut inflow = base;
        for (ell, &coeff) in coeffs.iter().enumerate() {
            inflow += coeff * lag_state[ell];
        }
        inflow += scale * noise;
        inflow
    }
}
```

### 1.4 PAR Model Fitting from Historical Data

When PAR coefficients are not provided in `inflow_models.parquet`, POWE.RS fits PAR models from historical inflow data using the Yule-Walker method with BIC-based order selection. For the mathematical derivation of the fitting procedure, see [PAR(p) Inflow Model — Fitting Procedure](../01-math/par-inflow-model.md).

The implementation uses the Levinson-Durbin algorithm to solve the Yule-Walker equations in O(p²) per season:

```rust
/// Fit PAR model using Yule-Walker method with BIC order selection
pub struct ParFitter {
    max_order: usize,
    min_history_years: usize,
}

impl ParFitter {
    /// Fit PAR model from historical inflows.
    /// Requires at least min_history_years of monthly data.
    /// Fits each of 12 seasons independently, selecting order via BIC.
    pub fn fit(
        &self,
        history: &[f64],  // Chronological monthly inflows
        hydro_id: &str,
    ) -> Result<FittedPar, FitError>;

    /// Solve Yule-Walker equations using Levinson-Durbin recursion
    fn yule_walker_solve(&self, autocorr: &[f64], order: usize) -> (Vec<f64>, f64);
}

/// Fitted PAR model output
pub struct FittedPar {
    pub hydro_id: String,
    pub means: Vec<f64>,             // [12] seasonal means
    pub coefficients: Vec<Vec<f64>>, // [12][max_order] AR coefficients
    pub residual_std: Vec<f64>,      // [12] residual standard deviation
    pub orders: Vec<usize>,          // [12] selected order per season
}
```

#### Fitted Model Validation

| Check                | Criterion                        | Action on Failure          |
| -------------------- | -------------------------------- | -------------------------- |
| Stationarity         | All AR roots outside unit circle | Reduce order or warn       |
| Residual normality   | Shapiro-Wilk p > 0.05            | Warn (non-fatal)           |
| Non-negative inflows | Simulated inflows stay positive  | Truncate at 0 with warning |
| Cross-validation     | Out-of-sample R² > 0.3           | Warn if poor fit           |

## 2. Noise Sampling and Correlation

### 2.1 Correlated Noise Generation

Hydros within the same correlation block share spatially correlated noise. The correlation structure is defined in `correlation.json`.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Correlated Noise Generation                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Correlation Structure:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Block 1 (Southeast):  Hydros [H1, H2, H3, H4]                           │   │
│  │ Block 2 (South):      Hydros [H5, H6, H7]                               │   │
│  │ Block 3 (Northeast):  Hydros [H8, H9]                                   │   │
│  │ Block 4 (North):      Hydros [H10, H11, H12]                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Step 1: Generate independent z[s][t][b] ~ N(0,1) per scenario/stage/block     │
│                              │                                                   │
│                              ▼                                                   │
│  Step 2: Apply Cholesky — L = cholesky(Sigma), correlated_z = L * z            │
│                              │                                                   │
│                              ▼                                                   │
│  Step 3: Map block noise to hydros — all hydros in block b get same noise      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Reproducible Sampling

To ensure reproducibility across MPI ranks and restarts, noise generation uses deterministic seed derivation. Each (iteration, scenario, stage) tuple maps to a unique seed:

```rust
/// Deterministic noise generation with seed management
pub struct NoiseGenerator {
    base_seed: u64,
    rng: Xoshiro256PlusPlus,
    cholesky_l: Vec<f64>,  // Lower triangular, B x B
    n_blocks: usize,
}

impl NoiseGenerator {
    pub fn new(
        base_seed: u64, correlation_matrix: &[f64], n_blocks: usize,
    ) -> Self {
        let cholesky_l = cholesky_decomposition(correlation_matrix, n_blocks);
        Self {
            base_seed,
            rng: Xoshiro256PlusPlus::seed_from_u64(base_seed),
            cholesky_l,
            n_blocks,
        }
    }

    /// Generate noise for a specific (iteration, scenario, stage) tuple.
    /// Deterministic regardless of execution order.
    pub fn generate_noise(
        &mut self, iteration: u32, scenario: u32, stage: u32,
    ) -> Vec<f64> {
        let seed = self.base_seed
            .wrapping_mul(iteration as u64 + 1)
            .wrapping_add(scenario as u64 * 1_000_000)
            .wrapping_add(stage as u64);
        self.rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        let independent: Vec<f64> = (0..self.n_blocks)
            .map(|_| self.standard_normal())
            .collect();
        self.apply_cholesky(&independent)
    }

    fn apply_cholesky(&self, z: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.n_blocks];
        for i in 0..self.n_blocks {
            for j in 0..=i {
                result[i] += self.cholesky_l[i * self.n_blocks + j] * z[j];
            }
        }
        result
    }
}
```

### 2.3 Noise Caching Strategy

For backward pass efficiency, noises are pre-generated and cached in a contiguous array:

```rust
/// Cached noise samples for all scenarios and stages
pub struct NoiseCache {
    /// Layout: [scenario_0_stage_0_block_0, ..., scenario_S_stage_T_block_B]
    pub samples: Vec<f64>,
    pub n_scenarios: usize,
    pub n_stages: usize,
    pub n_blocks: usize,
}

impl NoiseCache {
    /// Pre-generate all noise samples at initialization
    pub fn generate_all(
        generator: &mut NoiseGenerator,
        n_scenarios: usize, n_stages: usize, iteration: u32,
    ) -> Self {
        let n_blocks = generator.n_blocks;
        let mut samples = Vec::with_capacity(n_scenarios * n_stages * n_blocks);
        for scenario in 0..n_scenarios {
            for stage in 0..n_stages {
                samples.extend(generator.generate_noise(
                    iteration, scenario as u32, stage as u32,
                ));
            }
        }
        Self { samples, n_scenarios, n_stages, n_blocks }
    }

    #[inline]
    pub fn get(&self, scenario: usize, stage: usize, block: usize) -> f64 {
        self.samples[(scenario * self.n_stages + stage) * self.n_blocks + block]
    }
}
```

## 3. External Scenario Integration

### 3.1 External Scenario Sources

POWE.RS supports external (deterministic) scenarios for simulation, bypassing stochastic generation:

| Use Case           | Description                                |
| ------------------ | ------------------------------------------ |
| Historical replay  | Use actual historical inflows              |
| Monte Carlo import | Pre-generated scenarios from external tool |
| Stress testing     | Specific drought/flood scenarios           |

**Input file**: `simulation/external_scenarios/inflows.parquet` with columns `(scenario, stage_id, hydro_id, inflow_m3s)`.

**Integration rule**: Training always uses the PAR model. External scenarios are only used during simulation when configured.

### 3.2 Scenario Adapter Interface

```rust
/// Trait for scenario data providers
pub trait ScenarioProvider: Send + Sync {
    fn get_inflow(&self, scenario: usize, stage: usize, hydro: usize) -> f64;
    fn get_load_factor(&self, scenario: usize, stage: usize, bus: usize) -> f64;
    fn n_scenarios(&self) -> usize;
}

/// PAR-based scenario provider for training
pub struct ParScenarioProvider {
    par: Arc<PrecomputedPar>,
    noise_cache: NoiseCache,
    hydro_to_block: Vec<usize>,
}

/// External scenario provider for simulation
pub struct ExternalScenarioProvider {
    inflows: Vec<f64>,  // [scenario][stage][hydro], contiguous
    loads: Vec<f64>,    // [scenario][stage][bus], contiguous
    n_scenarios: usize,
    n_stages: usize,
    n_hydros: usize,
    n_buses: usize,
}
```

### 3.3 Noise Inversion for External Scenarios

When using external scenarios, POWE.RS must compute the **implied noise values** that would have generated those inflows under the PAR model. This is required because the LP constraints embed PAR AR components — RHS updates depend on lag inflows.

Given target inflow $a_t^{\text{target}}$ at stage $t$ for hydro $h$:

$$
\eta_t = \frac{a_t^{\text{target}} - \phi_m - \sum_{\ell=1}^{P} \psi_{m,\ell} \cdot a_{t-\ell}}{\sigma_m}
$$

where $\phi_m = \mu_m - \sum_{\ell=1}^{P} \psi_{m,\ell} \cdot \mu_{m-\ell}$.

The inversion proceeds sequentially through stages (each stage updates the lag buffer for the next):

1. Initialize lag buffer from historical inflows
2. For each stage: compute deterministic PAR component, solve for η, update lag buffer
3. Validate: warn if |η| > 4.0; error if σ ≈ 0 but residual exceeds tolerance

```rust
/// Noise inversion for external scenarios
pub struct NoiseInverter<'a> {
    par: &'a PrecomputedPar,
    hydro_to_block: &'a [usize],
    tolerance: f64,
}

impl<'a> NoiseInverter<'a> {
    /// Invert external inflows to block-level noise values.
    /// Returns per-stage per-block noises and any warnings.
    pub fn invert_scenario(
        &self,
        external_inflows: &[Vec<f64>],  // [stage][hydro]
        initial_lags: &[Vec<f64>],       // [hydro][lag]
    ) -> Result<InvertedNoises, InversionError>;
}

pub struct InvertedNoises {
    pub noises: Vec<Vec<f64>>,            // [stage][block]
    pub warnings: Vec<InversionWarning>,
}

#[derive(Debug)]
pub enum InversionWarning {
    ExtremeNoise { hydro: usize, stage: usize, noise: f64 },
    InconsistentBlockNoises { block: usize, stage: usize, max_diff: f64 },
}

#[derive(Debug)]
pub enum InversionError {
    ZeroVarianceResidual { hydro: usize, stage: usize, residual: f64 },
}
```

After inversion, a JSON validation report is emitted with noise statistics (`mean_noise`, `std_noise`, `max_noise`, `min_noise`, `extreme_count`), warnings, and an overall `status` field.

**Critical**: If AR order mismatches between PAR model and policy, noise inversion produces incorrect values. This is checked during policy loading.

## 4. Scenario Memory Layout

### 4.1 Memory Organization

Scenario data uses **scenario-major ordering** optimized for the forward pass access pattern:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Scenario Memory Layout                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Access Pattern in Forward Pass:                                                │
│  - Outer loop: scenarios (parallel across threads)                              │
│  - Inner loop: stages (sequential within scenario)                              │
│  - Innermost: entities (sequential within stage)                                │
│                                                                                  │
│  Layout: [S0_T0_H0] [S0_T0_H1] ... [S0_TT_HN] [S1_T0_H0] ... [SS_TT_HN]     │
│                                                                                  │
│  Benefits:                                                                       │
│  - Each thread accesses contiguous memory for its scenarios                    │
│  - No false sharing between threads                                             │
│  - Predictable prefetching within stage sequence                                │
│                                                                                  │
│  Size (production scale):                                                        │
│  - 200 scenarios x 120 stages x 160 hydros x 8 bytes = 30.7 MB (inflows)       │
│  - Total per rank: ~50 MB (fits in L3 cache per NUMA domain)                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Per-Rank Scenario Distribution

```rust
/// Scenario distribution across MPI ranks
pub struct ScenarioDistribution {
    pub total_scenarios: usize,
    pub my_range: Range<usize>,    // This rank's assigned scenarios [start, end)
    pub counts: Vec<usize>,        // Scenarios per rank (for gather operations)
    pub displs: Vec<usize>,        // Displacements for MPI_Gatherv
}

impl ScenarioDistribution {
    pub fn new(total_scenarios: usize, rank: usize, world_size: usize) -> Self {
        // Distribute as evenly as possible: first `remainder` ranks get +1
        let base = total_scenarios / world_size;
        let remainder = total_scenarios % world_size;
        let mut counts = vec![base; world_size];
        for i in 0..remainder { counts[i] += 1; }

        let mut displs = vec![0; world_size];
        for i in 1..world_size { displs[i] = displs[i-1] + counts[i-1]; }

        let start = displs[rank];
        Self {
            total_scenarios,
            my_range: start..start + counts[rank],
            counts,
            displs,
        }
    }

    pub fn my_count(&self) -> usize { self.my_range.len() }
    pub fn to_global(&self, local: usize) -> usize { self.my_range.start + local }
}
```

### 4.3 NUMA-Aware Allocation

```rust
/// NUMA-aware scenario data allocation
pub fn allocate_scenario_data(
    distribution: &ScenarioDistribution,
    n_stages: usize,
    n_hydros: usize,
) -> Vec<f64> {
    let size = distribution.my_count() * n_stages * n_hydros;
    let mut data = vec![0.0; size];

    // First-touch initialization in parallel ensures NUMA locality
    #[cfg(feature = "openmp")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(n_stages * n_hydros)
            .for_each(|chunk| chunk.fill(0.0));
    }

    data
}
```

## Cross-References

- [PAR(p) Inflow Model](../01-math/par-inflow-model.md) — Mathematical definition, parameter set, and fitting theory for the PAR(p) model
- [Validation Architecture](./validation-architecture.md) — Input validation rules applied to PAR coefficients and scenario data
- [Input Loading Pipeline](./input-loading-pipeline.md) — How `inflow_models.parquet`, `inflow_history.parquet`, and `correlation.json` are loaded
- [CLI and Lifecycle](./cli-and-lifecycle.md) — Scenario generation phase within the execution lifecycle
