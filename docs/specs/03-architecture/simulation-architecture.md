---
status: approved
review_priority: 3-medium
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §17 (17.1-17.4)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §18 (18.1-18.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §19 (19.1-19.4)"
last_reviewed: 2026-02-23
reviewed_by: rogerio
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §17-19"
  - date: 2026-02-23
    description: "P3 review rewrite: fix review_priority (2→3). Strip all Rust code (8 blocks across §1.2-§3.4) and replace with behavioral descriptions. Add §2 Policy Compatibility Validation (promoted from frontmatter note). Fix stale ScenarioSource enum — replaced with cross-reference to scenario-generation.md sampling scheme abstraction. Add block structure acknowledgment in §3.2. Clarify non-convex features as simulation-only deferred (§5). Remove output schema duplication — cross-reference output-schemas.md. Restructure to: Purpose → Overview → Policy Validation → Execution → Statistics → Non-Convex Extensions → Output Streaming → Cross-References."
  - date: 2026-02-23
    description: "Review fixes: (1) Config field names verified — added config paths, removed non-existent chunk_size/non-convex config rows, clarified sampling_scheme lives in stages.json. (2) Removed §3.3 Scenario Chunking (redundant with §6 streaming + §3.1 distribution) — replaced with §3.3 Memory Management. (3) Fixed std dev formula — use gathered array directly (numerically stable, array already available for CVaR) instead of running sum-of-squares. (4) Added §4.2 Per-Category Cost Statistics with mean/max/frequency per penalty category from penalty-system.md. (5) Softened §5 Non-Convex Extensions — removed prescriptive algorithms (MIP warm-start, iterative fixed-point, convergence criteria), kept what/why, marked all deferred."
---

# Simulation Architecture

## Purpose

This spec defines the simulation phase of the POWE.RS SDDP solver: how trained policies are evaluated on large scenario sets, how simulation statistics are computed, and how results are streamed to Parquet output files across distributed MPI ranks. It also covers the simulation-only non-convex extensions that refine LP solutions during policy evaluation.

For the simulation output Parquet schemas, see [Output Schemas](../02-data-model/output-schemas.md). For the output infrastructure (manifests, MPI partitioning, crash recovery), see [Output Infrastructure](../02-data-model/output-infrastructure.md).

## 1. Simulation Overview

The simulation phase evaluates the trained SDDP policy on a large number of scenarios to assess:

1. **Policy quality** — Expected cost, variance, and risk metrics
2. **Operational behavior** — Storage trajectories, generation mix, deficit frequency
3. **Robustness** — Performance across diverse hydrological conditions

```
┌───────────────────────────────────────────────────────────────────┐
│                    Simulation Architecture                        │
├───────────────────────────────────────────────────────────────────┤
│  Input: Trained FCF (cuts), Simulation scenarios                  │
│                                                                   │
│  SCENARIO SELECTION                                               │
│  Sampling scheme: InSample | External | Historical                │
│  (see scenario-generation.md §3)                                  │
│                                                                   │
│  PARALLEL EXECUTION                                               │
│  Scenarios statically distributed across MPI ranks                │
│  Each rank solves LP sequence for assigned scenarios              │
│  Within each rank: thread-level dynamic work-stealing             │
│                                                                   │
│  PER-SCENARIO (for stage t = 1..T):                               │
│    1. Realize uncertainties via sampling scheme                   │
│    2. Build stage LP with FCF cuts and block structure            │
│    3. Solve LP, extract solution                                  │
│    4. Apply non-convex refinements (if configured)                │
│    5. Stream results to output                                    │
│    6. Propagate state to next stage                               │
│                                                                   │
│  OUTPUT AGGREGATION                                               │
│  Streaming write per rank | Statistics across scenarios           │
└───────────────────────────────────────────────────────────────────┘
```

### 1.1 Simulation Configuration

Simulation behavior is configured via the `simulation` section of `config.json`. Key parameters include:

| Parameter       | Config Path                        | Description                                                           | Reference                                                          |
| --------------- | ---------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `enabled`       | `simulation.enabled`               | Whether the simulation phase executes after training                  | [CLI and Lifecycle §5.3](./cli-and-lifecycle.md)                   |
| `n_scenarios`   | `simulation.n_scenarios`           | Number of scenarios to evaluate                                       | [Configuration Reference](../05-config/configuration-reference.md) |
| Sampling scheme | `scenario_source` in `stages.json` | How scenarios are selected (InSample, External, Historical)           | [Scenario Generation §3](./scenario-generation.md)                 |
| Output detail   | `simulation.output_detail`         | Level of output granularity (summary, stage-level, full per-scenario) | [Output Schemas](../02-data-model/output-schemas.md)               |

The sampling scheme is defined in `stages.json` (not `config.json`) because it is a property of the stochastic model, not the solver. See [Configuration Reference §18.11](../05-config/configuration-reference.md).

For the full `config.json` schema, see [Configuration Reference](../05-config/configuration-reference.md).

### 1.2 Relationship to Training

The simulation phase reuses the same LP construction infrastructure as training. At each stage, the simulation LP includes:

- All constraints from the training LP (water balance, load balance, generation bounds, etc.)
- The trained FCF cuts as lower-bounding constraints on the future cost variable $\theta$
- The same block structure (parallel or chronological) as defined per stage — see [Block Formulations](../01-math/block-formulations.md)

The key differences from training are:

| Aspect              | Training                         | Simulation                                 |
| ------------------- | -------------------------------- | ------------------------------------------ |
| Direction           | Forward + backward passes        | Forward pass only (no cut generation)      |
| Objective           | Build policy (cuts)              | Evaluate policy quality                    |
| Scenario count      | Moderate (1–20 per iteration)    | Large (hundreds to thousands)              |
| Non-convex features | Excluded (LP must remain convex) | Optionally included (post-LP refinement)   |
| Output              | Convergence log, FCF cuts        | Per-scenario results, aggregate statistics |

## 2. Policy Compatibility Validation

Before simulation begins, the system validates that the current input data is compatible with the trained policy. The FCF cuts encode LP structural information (state variable dimensions, constraint structure, dual coefficients) that becomes invalid if the system configuration changes between training and simulation.

**Validation checks include** (non-exhaustive):

| Property                             | Why It Matters                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| Block mode per stage                 | Cut duals come from different water balance structures (parallel vs. chronological) |
| Block count and durations per stage  | LP column/row dimensions change                                                     |
| Number of hydro plants               | State variable dimension changes                                                    |
| AR orders per hydro                  | State variable dimension changes (inflow lags are state)                            |
| Cascade topology                     | Water balance constraint structure changes                                          |
| Number of buses, lines               | Load balance constraint structure changes                                           |
| Number of thermals, contracts        | LP column count changes                                                             |
| Production model per hydro per stage | FPHA vs. constant affects generation constraint and dual structure                  |

Any mismatch results in a **hard error** — the simulation does not proceed with an incompatible policy.

The full policy compatibility validation specification, including the metadata format persisted by training and the validation algorithm, is documented in [Deferred Features §C.9](../06-deferred/deferred-features.md).

## 3. Simulation Execution

### 3.1 Scenario Distribution

Simulation scenarios are distributed across MPI ranks using the same two-level work distribution as training:

1. **MPI rank level — deterministic distribution.** If $S$ total scenarios are distributed across $R$ ranks, the first $S \bmod R$ ranks receive $\lceil S/R \rceil$ scenarios and the remaining ranks receive $\lfloor S/R \rfloor$. Each rank stores only its assigned scenarios in memory.

2. **Thread level — dynamic work-stealing.** Within each rank, scenarios are processed by a thread pool with dynamic work-stealing. Per-scenario costs are not uniform — LP solve time varies with noise realization, active constraints, and warm-start quality — so dynamic scheduling absorbs this variability.

For details on the distribution strategy and NUMA-aware allocation, see [Scenario Generation §5](./scenario-generation.md).

### 3.2 Per-Scenario Forward Pass

For each assigned scenario, the simulation executes a complete forward pass through all stages:

1. **Initialize state** — Set the initial state (storage volumes, inflow lags) from the case data

2. **For each stage $t = 1, \ldots, T$:**

   a. **Realize uncertainties** — Obtain the stage realization from the configured sampling scheme. For InSample, sample from the opening tree. For External or Historical, use the external/historical values with noise inversion to obtain $\varepsilon$ values for the LP's AR dynamics constraint. See [Scenario Generation §3.2](./scenario-generation.md).

   b. **Build stage LP** — Construct the LP with the current state, realized uncertainties, and all FCF cuts accumulated during training. The LP includes the block structure for this stage (parallel or chronological, per-stage definition). See [Block Formulations](../01-math/block-formulations.md).

   c. **Solve LP** — Solve the stage LP. The simulation LP should always be feasible due to recourse slack variables (deficit, excess). If infeasibility occurs, it indicates a system error.

   d. **Apply non-convex refinements** — If non-convex extensions are configured (see §5), refine the LP solution through post-processing.

   e. **Extract results** — Record stage-level outputs: generation, storage, flows, costs, violations, marginal values. See [Output Schemas §5](../02-data-model/output-schemas.md) for column definitions.

   f. **Propagate state** — Extract end-of-stage storage volumes and updated inflow lag buffer as the initial state for stage $t+1$.

3. **Compute scenario cost** — Sum immediate costs across all stages, applying discount factors. See [Discount Rate](../01-math/discount-rate.md).

4. **Stream results** — Send the scenario results to the output writer for streaming to disk (see §6).

### 3.3 Memory Management

Individual scenario results are streamed to the output writer immediately upon completion (see §6). This prevents accumulation of all scenario results in memory, which is critical when simulating thousands of scenarios. Each thread releases the scenario's LP workspace and result buffers before proceeding to its next scenario. Per-scenario scalar costs (total and per-category) are retained in a compact buffer for final statistics computation (see §4.4).

## 4. Simulation Statistics

The simulation phase computes aggregate statistics across all scenarios. Each rank stores per-scenario total costs and per-category cost components in a local buffer during execution. After all ranks complete, costs are gathered to rank 0 for final statistics computation (see §4.4).

### 4.1 Cost Statistics

| Statistic                   | Description                                                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Mean cost                   | $\bar{C} = \frac{1}{S} \sum_{s=1}^{S} C_s$                                                                                              |
| Standard deviation          | $\sigma_C = \sqrt{\frac{1}{S-1} \sum_{s=1}^{S} (C_s - \bar{C})^2}$                                                                      |
| Minimum / Maximum cost      | Range of total scenario costs                                                                                                           |
| CVaR at confidence $\alpha$ | Conditional Value-at-Risk: mean of the worst $(1-\alpha)$ fraction of scenario costs. See [Risk Measures](../01-math/risk-measures.md). |

CVaR computation requires all individual scenario costs (for sorting and tail averaging), which are gathered to rank 0 via `MPI_Gatherv` after all ranks complete (see §4.3). Since the gathered array is already available, the mean and standard deviation are also computed directly from it — avoiding the numerical instability of running sum-of-squares formulas. Min and max are computed via `MPI_Allreduce` without gathering.

### 4.2 Per-Category Cost Statistics

In addition to aggregate cost statistics, the simulation reports mean cost broken down by the cost categories defined in [Penalty System §2](../02-data-model/penalty-system.md). This allows the user to understand the composition of total cost and diagnose policy behavior.

| Category                           | Components Summed                                                                                                            |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Resource costs                     | `thermal_cost` + `contract_cost`                                                                                             |
| Category 1 — Recourse              | `deficit_cost` + `excess_cost`                                                                                               |
| Category 2 — Constraint violations | `storage_violation_cost` + `filling_target_cost` + `hydro_violation_cost` + `inflow_penalty_cost` + `generic_violation_cost` |
| Category 3 — Regularization        | `spillage_cost` + `fpha_turbined_cost` + `curtailment_cost` + `exchange_cost`                                                |
| Imputed costs                      | `pumping_cost`                                                                                                               |

For each category, the simulation computes:

- **Mean** across all scenarios
- **Maximum** across all scenarios (identifies worst-case contributors)
- **Frequency** — fraction of scenarios where the category cost is non-zero (particularly relevant for deficit and constraint violations)

These statistics are computed from the gathered cost arrays on rank 0 alongside the aggregate statistics in §4.1. The per-scenario, per-stage cost breakdown is always available in the detailed output files — see [Output Schemas §5.1](../02-data-model/output-schemas.md) for the full column definitions.

### 4.3 Operational Statistics

| Statistic             | Description                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------- |
| Deficit frequency     | Fraction of scenarios with at least one stage having deficit > 0                             |
| Total deficit energy  | Sum of deficit (MWh) across all scenarios and stages                                         |
| Total spillage energy | Sum of spillage (MWh) across all scenarios and stages                                        |
| Stage-level stats     | Optional per-stage aggregates (mean storage, mean generation, mean cost) when detail ≥ stage |

### 4.4 MPI Aggregation

Each rank stores its local scenario costs in a per-rank buffer. After all ranks complete, results are aggregated:

- **Min/Max** (`min_cost`, `max_cost`) — `MPI_Allreduce` with `MPI_MIN` / `MPI_MAX`
- **Scenario costs** — Individual scenario total costs and per-category cost components gathered to rank 0 via `MPI_Gatherv`
- **Mean, std, CVaR, per-category stats** — Computed on rank 0 from the gathered arrays
- **Deficit/spillage sums** (`deficit_mwh_sum`, `spill_mwh_sum`, `deficit_scenarios`) — `MPI_Allreduce` with `MPI_SUM`

## 5. Non-Convex Extensions (Deferred)

SDDP produces an optimal policy for the convex relaxation. During simulation, the LP solution can optionally be refined with non-convex operational constraints that are excluded from training (because they would break the LP convexity required for valid cut generation).

All non-convex extensions are **deferred**. This section documents _what_ they are and _why_ they are simulation-only, not _how_ they will be implemented.

| Feature                 | Why Simulation-Only                                             | Status   | Reference                                                     |
| ----------------------- | --------------------------------------------------------------- | -------- | ------------------------------------------------------------- |
| Linearized head model   | Bilinear dependency (head × flow) changes LP between iterations | Deferred | [Deferred Features §C.6](../06-deferred/deferred-features.md) |
| Thermal unit commitment | Binary on/off variables incompatible with LP relaxation         | Deferred | [Deferred Features §C.1](../06-deferred/deferred-features.md) |
| Minimum generation      | Big-M or indicator constraints break LP convexity               | Deferred | —                                                             |
| Startup/shutdown costs  | Multi-period linking constraints across blocks                  | Deferred | —                                                             |

**Key invariant**: Non-convex refinements do **not** affect state propagation to the next stage. State transition always uses the original LP solution to maintain consistency with the trained policy. The refined solution replaces the LP solution for output purposes only.

## 6. Output Streaming

### 6.1 Streaming Architecture

With potentially thousands of scenarios, storing all results in memory before writing is impractical. The output writer uses a streaming architecture:

- A **bounded channel** connects the simulation threads to a dedicated **background I/O thread**
- Simulation threads send completed scenario results through the channel as they finish
- The I/O thread writes results to Parquet files asynchronously
- The channel backpressure prevents simulation from running too far ahead of I/O

> **Placeholder** — The output streaming pipeline diagram (`../../diagrams/exports/svg/data/output-streaming-pipeline.svg`) will be revised after the text review is complete.

### 6.2 Output Detail Levels

The amount of data written per scenario depends on the configured output detail level:

| Detail Level | What Is Written                               | Use Case                        |
| ------------ | --------------------------------------------- | ------------------------------- |
| Summary      | Only aggregate cost per scenario              | Quick policy quality assessment |
| Stage-level  | Per-stage aggregates (cost, deficit, storage) | Storage trajectory analysis     |
| Full         | Per-scenario, per-stage, per-entity detail    | Detailed operational analysis   |

For the complete column definitions at each detail level, see [Output Schemas §5](../02-data-model/output-schemas.md).

### 6.3 Distributed Output

Each MPI rank writes its simulation results independently using **per-rank output files**. This avoids MPI coordination during I/O and enables each rank to write at full local disk bandwidth.

The output uses Hive-style partitioning by `scenario_id`, where each rank writes exclusively to the partitions corresponding to its assigned scenarios. No coordination between ranks is needed during writing — the partition assignment is determined by the deterministic scenario distribution (§3.1).

After all ranks complete, rank 0 writes the simulation manifest (`_manifest.json`) with checksums, row counts, and partition listings. The `_SUCCESS` marker file is written atomically on successful completion. See [Output Infrastructure §1.1](../02-data-model/output-infrastructure.md).

## Cross-References

- [CLI and Lifecycle](./cli-and-lifecycle.md) — Execution phases and conditional simulation mode (§5.3)
- [Scenario Generation](./scenario-generation.md) — Sampling scheme abstraction (§3), scenario distribution (§5), noise inversion for external scenarios (§4.3)
- [Training Loop](./training-loop.md) — Training phase that produces the policy evaluated by simulation
- [Block Formulations](../01-math/block-formulations.md) — Block structure (parallel/chronological) within simulation LP stages
- [Risk Measures](../01-math/risk-measures.md) — CVaR computation for simulation cost statistics
- [Discount Rate](../01-math/discount-rate.md) — Discount factors applied to scenario costs
- [LP Formulation](../01-math/lp-formulation.md) — Stage LP construction shared between training and simulation
- [Output Schemas](../02-data-model/output-schemas.md) — Parquet column definitions for all simulation output files
- [Output Infrastructure](../02-data-model/output-infrastructure.md) — Manifests, MPI partitioning, crash recovery
- [Deferred Features §C.1](../06-deferred/deferred-features.md) — GNL / thermal unit commitment (simulation-only MIP)
- [Deferred Features §C.6](../06-deferred/deferred-features.md) — FPHA enhancements including head-dependent refinement
- [Deferred Features §C.9](../06-deferred/deferred-features.md) — Policy compatibility validation specification
- [Deferred Features §C.13](../06-deferred/deferred-features.md) — Alternative forward pass model (simulation-only LP in training)
