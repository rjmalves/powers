---
status: approved
review_priority: 3-medium
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §12 (12.1-12.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §13 (13.1-13.4)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §14 (14.1-14.4)"
last_reviewed: 2026-02-22
reviewed_by: "rogerio"
review_notes: "REVIEW NOTE (from block-formulations.md approval): Training must persist a policy metadata record alongside cut data, capturing all input properties that affect LP structure (block modes, block counts/durations, hydro count, AR orders, topology, etc.). This metadata is used by simulation/warm-start/resume to validate policy compatibility. See Deferred Features §C.9."
change_log:
  - date: 2026-02-14
    description: "Extracted from ARCHITECTURE §12-14"
  - date: 2026-02-22
    description: "Combined P3 review + SDDP.jl-inspired refactoring. Stripped all Rust code (8 blocks across §2-§7) and replaced with behavioral descriptions. Fixed review_priority from 2-high to 3-medium (architecture spec). §2 rewritten: training orchestrator described as configurable components including sampling scheme. §3 rewritten: trait abstractions replaced with behavioral descriptions of four abstraction points (risk measure, cut formulation, horizon mode, sampling scheme). §4 rewritten: forward pass described behaviorally with sampling scheme parameterization and thread-trajectory affinity. §5 rewritten: state management described as behavioral lifecycle (initialization, update, extraction). §6 rewritten: backward pass described with opening tree reference and Complete backward sampling. §7 rewritten: dual extraction math retained, cut structure described behaviorally. Cross-references updated to scenario-generation.md new section numbering."
  - date: 2026-02-22
    description: "P3 review fixes: §4.2 step 2b — clarified that External/Historical always invert to noise terms (LP uses AR dynamics constraint with noise as fixed variables, never raw inflow values). §5.2 item 3 — state deduplication deferred to C.17. §6.1 — clarified trial points come from all scenarios across all ranks (after MPI_Allgatherv)."
---

# Training Loop

## Purpose

This spec defines the POWE.RS SDDP training loop architecture: the core training components, their configurable abstraction points, forward pass execution with sampling scheme parameterization and parallel distribution, backward pass execution with opening tree evaluation and cut generation, state management, and dual extraction for cut coefficients.

## 1. SDDP Algorithm Overview

The training phase implements the Stochastic Dual Dynamic Programming (SDDP) algorithm, iteratively constructing piecewise-linear approximations of the expected future cost function (FCF) through forward simulation and backward cut generation.

Each iteration consists of three phases:

1. **Forward pass** — Sample $M$ scenarios via the configured **sampling scheme**, solve the LP at each stage with the current FCF, record visited states and stage-1 costs for the lower bound
2. **Backward pass** — For each stage $T$ down to 2, evaluate the cost-to-go from each visited state under **all** openings from the fixed opening tree, extract LP duals, and compute new cuts via the risk measure
3. **Convergence check** — Update the upper bound estimate (mean forward cost), compute the gap $(UB - LB) / |UB|$, and test stopping rules (gap tolerance, stable LB, iteration/time limits)

The loop terminates when converged or a limit is reached, outputting the FCF cuts and bound history.

## 2. Training Orchestrator Components

The training orchestrator manages the iterative SDDP loop and coordinates the following components:

| Component               | Responsibility                                                                                            | Configuration Source                               |
| ----------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Risk Measure**        | Determines how backward outcomes are aggregated into cut coefficients (expectation vs CVaR)               | `risk_measure` per stage in `stages.json`          |
| **Cut Formulation**     | Determines cut structure (single-cut; multi-cut is deferred)                                              | Fixed: single-cut                                  |
| **Horizon Mode**        | Determines stage transitions, terminal conditions, and discount factors                                   | `policy_graph` in `stages.json`                    |
| **Sampling Scheme**     | Determines how the forward pass selects scenario realizations at each stage                               | `scenario_source.sampling_scheme` in `stages.json` |
| **FCF**                 | Stores accumulated Benders cuts per stage; queried during LP construction and updated after backward pass | Built incrementally across iterations              |
| **Convergence Monitor** | Tracks lower/upper bounds, gap history, and evaluates stopping rules                                      | `stopping_rules` in `config.json`                  |

### 2.1 Iteration Lifecycle

Each iteration follows a fixed sequence:

1. **Forward pass** — Execute $M$ scenario trajectories (§4)
2. **Forward synchronization** — `MPI_Allreduce` aggregates global statistics (lower bound, visited states) across ranks
3. **Backward pass** — Generate cuts from visited states (§6)
4. **Cut synchronization** — `MPI_Allgatherv` distributes new cuts to all ranks
5. **Convergence update** — Update bound estimates, evaluate stopping rules (see [Convergence Monitoring](./convergence-monitoring.md))
6. **Checkpoint** — If the checkpoint interval has elapsed, persist current FCF and iteration state (see [Checkpointing](../04-hpc/checkpointing.md))
7. **Logging** — Emit iteration summary (bounds, gap, timings)

### 2.2 Termination Conditions

The loop terminates when **any** of the following conditions is met:

| Condition          | Description                                                          | Configuration Parameter           |
| ------------------ | -------------------------------------------------------------------- | --------------------------------- |
| Convergence        | Optimality gap below tolerance (checked only after `min_iterations`) | `gap_tolerance`, `min_iterations` |
| Stable lower bound | Lower bound has not improved for N consecutive iterations            | `stable_iterations`               |
| Iteration limit    | Maximum iteration count reached                                      | `max_iterations`                  |
| Time limit         | Wall-clock time exceeded                                             | `time_limit_seconds`              |
| Graceful shutdown  | External signal received (checkpoints last **completed** iteration)  | OS signal (SIGTERM/SIGINT)        |

For the full stopping rule specification, see [Stopping Rules](../01-math/stopping-rules.md).

## 3. Abstraction Points

The training loop is parameterized by four abstraction points. Each is a behavioral contract — the training loop interacts with each through a defined interface, independent of the specific variant.

### 3.1 Risk Measure

Given a set of backward outcomes (one per opening) with probabilities, the risk measure aggregates them into a single cut. The two variants are:

- **Expectation** (risk-neutral) — Probability-weighted average of outcomes. The cut intercept and gradient are the weighted means of the per-outcome intercepts and gradients.
- **CVaR** (risk-averse) — Convex combination of expectation and conditional value-at-risk: $(1 - \lambda) \cdot \mathbb{E}[\cdot] + \lambda \cdot \text{CVaR}_\alpha[\cdot]$. Cut coefficients are computed via sorting-based greedy weight allocation. See [Risk Measures](../01-math/risk-measures.md).

The risk measure can vary by stage (configured per stage in `stages.json`).

### 3.2 Cut Formulation

Determines the structure of cuts added to the FCF:

- **Single-cut** (current) — One aggregated cut per iteration per stage. The future cost variable $\theta$ receives a single constraint per backward pass evaluation.
- **Multi-cut** (deferred) — One cut per opening per iteration. See [Deferred Features §C.3](../06-deferred/deferred-features.md).

### 3.3 Horizon Mode

Determines stage traversal and terminal conditions:

- **Finite horizon** — Linear chain $1 \to 2 \to \cdots \to T$. Terminal value $V_{T+1} = 0$.
- **Cyclic** — Stage $T$ transitions back to a cycle start stage. Requires discount factor $d < 1$ for convergence. Cuts at equivalent cycle positions are shared.

See [SDDP Algorithm §4](../01-math/sddp-algorithm.md) and [Infinite Horizon](../01-math/infinite-horizon.md).

### 3.4 Sampling Scheme

Determines how the forward pass selects scenario realizations. This is one of three orthogonal SDDP concerns formalized in [Scenario Generation §3](./scenario-generation.md):

| Scheme       | Forward Noise Source                       | Description                                                    |
| ------------ | ------------------------------------------ | -------------------------------------------------------------- |
| `InSample`   | Fixed opening tree                         | Sample random index from pre-generated noise vectors (default) |
| `External`   | User-provided `external_scenarios.parquet` | Draw from external data (random or sequential selection)       |
| `Historical` | `inflow_history.parquet` mapped to stages  | Replay historical inflow sequences in order                    |

The backward pass noise source is **always** the fixed opening tree, regardless of the forward sampling scheme. This separation means the forward and backward passes may use different noise distributions — see [Scenario Generation §3.1](./scenario-generation.md).

## 4. Forward Pass

### 4.1 Overview

The forward pass simulates $M$ independent scenario trajectories through the full stage horizon, solving the stage LP at each step with the current FCF approximation. The purpose is twofold:

1. **Generate trial points** — The visited states $\{\hat{x}_t\}$ at each stage become the evaluation points for the backward pass
2. **Estimate upper bound** — The mean total forward cost across all trajectories provides a statistical upper bound estimate

### 4.2 Scenario Trajectory

For each forward trajectory:

1. **Initialize** — Start from the known initial state $x_0$: initial storage volumes from [Input Constraints §1](../02-data-model/input-constraints.md) and inflow lag values from historical data or pre-study stages
2. **Stage loop** ($t = 1, \ldots, T$):
   a. **Select scenario realization** — The sampling scheme selects the noise vector for this stage:
   - _InSample_: Sample random index $j$ from the opening tree, retrieve noise vector $\eta_{t,j}$
   - _External_: Select scenario from external data (by random sampling or sequential iteration). The external inflow values are **inverted to noise terms** via the PAR model (see [Scenario Generation §3.2](./scenario-generation.md))
   - _Historical_: Look up historical inflow for this stage. The historical values are similarly inverted to noise terms
     b. **Compute inflows and fix noise** — The PAR model evaluates with the selected noise to produce inflow values. The noise terms $\varepsilon_{h,t}$ (whether sampled, inverted from external data, or inverted from historical data) are fixed into the LP via fixing constraints on the AR dynamics equation — the LP always receives noise, never raw inflow values directly (see [Scenario Generation §3.2](./scenario-generation.md))
     c. **Build stage LP** — Construct the stage LP with incoming state $\hat{x}_{t-1}$, scenario realization, and all current FCF cuts as constraints on $\theta$
     d. **Solve** — Solve the LP. Feasibility is guaranteed by the recourse slack system (see [Penalty System](../02-data-model/penalty-system.md))
     e. **Record** — Store the stage cost and the end-of-stage state $\hat{x}_t$ (storage volumes and updated AR lags)
     f. **Transition** — Pass $\hat{x}_t$ as the incoming state to stage $t+1$
3. **Aggregate** — Compute total trajectory cost $\sum_{t=1}^{T} c_t$

### 4.3 Parallel Distribution

Scenarios are distributed across MPI ranks in contiguous blocks. Within each rank, scenarios are parallelized across OpenMP threads with **thread-trajectory affinity**: each thread owns one or more complete trajectories and solves all stages sequentially for its assigned trajectories. This preserves cache locality — the solver basis, scenario data, and LP coefficients remain warm in the thread's cache lines across stages.

When $M > N_{\text{threads}}$, threads process multiple trajectories in batches. Between batches, the thread saves and restores forward pass state (solver basis, visited states, scenario realization) at stage boundaries. This is analogous to context switching, but only occurs at well-defined stage boundaries.

After all ranks complete their trajectories, `MPI_Allreduce` aggregates:

- **Lower bound** — First-stage LP objective value (the deterministic lower bound, monotonically increasing across iterations)
- **Upper bound statistics** — Mean and variance of total forward costs across all trajectories

### 4.4 Warm-Starting

The forward pass LP solution at stage $t$ provides a near-optimal basis for the backward pass solves at the same stage. The solver retains this basis after the forward solve so that the backward pass at stage $t$ can warm-start from it, significantly reducing solve times. See [Solver Workspaces](./solver-workspaces.md).

## 5. State Management

### 5.1 State Vector

The state vector carries all information needed to transition between stages and generate valid cuts. It consists of:

| Component       | Dimension          | Source at Stage $t$                                 |
| --------------- | ------------------ | --------------------------------------------------- |
| Storage volumes | $N_{\text{hydro}}$ | End-of-stage storage from LP solution ($v_{h,T_k}$) |
| AR inflow lags  | $\sum_h P_h$       | Updated lag buffer after inflow computation         |

Future extensions (batteries, GNL pipeline) may add additional state dimensions — see [SDDP Algorithm §5](../01-math/sddp-algorithm.md).

### 5.2 State Lifecycle

1. **Initialization** — The initial state $x_0$ is constructed from:
   - Storage: initial reservoir volumes from `initial_conditions.json` (see [Input Constraints §1](../02-data-model/input-constraints.md))
   - AR lags: historical inflow values from `inflow_history.parquet` or pre-study stages in `stages.json`, ordered newest-first (lag 1 = most recent)

2. **Update** — At each stage, the state is updated in two steps:
   - **Inflow computation**: The PAR model (or external/historical lookup) produces the stage inflow $a_{h,t}$. The lag buffer is shifted: the oldest lag drops off, all remaining lags shift by one position, and $a_{h,t}$ becomes the new lag-1 value
   - **Storage extraction**: End-of-stage storage volumes are read from the LP solution's state variable values

3. **Extraction for backward pass** — After the forward pass, the visited states at each stage are collected and broadcast to all ranks via `MPI_Allgatherv`. State deduplication (merging duplicate visited states to reduce backward pass LP solves) is a potential optimization deferred to [Deferred Features](../06-deferred/deferred-features.md).

## 6. Backward Pass

### 6.1 Overview

The backward pass improves the FCF by generating new Benders cuts. It walks stages in reverse order from $T$ down to 2. The trial points $\{\hat{x}_t\}$ used here are the visited states from **all** forward scenarios across **all** MPI ranks (gathered via `MPI_Allgatherv` in §5.2). At each stage, the cost-to-go from each trial point is evaluated under **all** openings from the fixed opening tree.

### 6.2 Cut Generation per Stage

At each stage $t$, for each trial point $\hat{x}_{t-1}$ collected during the forward pass:

1. **Retrieve openings** — Get all $N_{\text{openings}}$ noise vectors for stage $t$ from the fixed opening tree (see [Scenario Generation §2.3](./scenario-generation.md)). This is the **Complete** backward sampling scheme — all openings are always evaluated. A deferred `MonteCarlo(n)` variant would sample a subset; see [Deferred Features §C.14](../06-deferred/deferred-features.md).

2. **Evaluate each opening** — For each noise vector $\eta_{t,j}$ ($j = 0, \ldots, N_{\text{openings}} - 1$):
   a. Compute realized inflows via the PAR model with the trial state's lag buffer and the opening's noise vector
   b. Build the backward LP at stage $t$: the incoming state $\hat{x}_{t-1}$ is fixed (storage and lag values set as constraints), and the scenario realization uses the computed inflows
   c. Solve the LP and extract:
   - Objective value $Q_t(\hat{x}_{t-1}, \omega_j)$
   - Dual variables of state-linking constraints (water balance for storage, fixing constraints for AR lags)
   - For hydros using FPHA, the FPHA hyperplane duals also contribute to storage cut coefficients (see [Cut Management §2](../01-math/cut-management.md))

3. **Aggregate into cut** — The risk measure aggregates the per-opening outcomes into a single cut:
   - Probabilities are uniform: $p(\omega_j) = 1/N_{\text{openings}}$
   - For Expectation: weighted average of intercepts and gradients
   - For CVaR: sorting-based greedy weight allocation (see [Risk Measures](../01-math/risk-measures.md))

4. **Add cut** — The new cut is added to stage $t-1$'s cut pool in the FCF

### 6.3 Parallel Distribution

Trial states at each stage are distributed across MPI ranks. Within each rank, each thread evaluates its assigned states sequentially, reusing the warm solver basis saved from the forward pass at that stage (§4.4). The branching scenarios (openings) for each state are evaluated sequentially by the same thread, keeping the solver state hot.

**Stage synchronization barrier**: All threads across all ranks must complete cut generation at stage $t$ before any thread proceeds to stage $t-1$. This is because the new cuts at stage $t$ must be available to all ranks before they solve backward LPs at stage $t-1$ (which include stage $t$'s cuts in their FCF approximation).

After processing each stage, `MPI_Allgatherv` collects all new cuts from all ranks and distributes them, so every rank has the complete set of new cuts.

### 6.4 LP Rebuild Considerations

Memory constraints prevent keeping all stage LPs with their full cut sets resident simultaneously. The solver must rebuild LPs when transitioning between stages, which lies on the critical performance path. Strategies to minimize rebuild cost include:

- **Cut preallocation** — Reserve space for expected cut count to avoid LP resizing
- **Basis persistence** — Reuse the forward pass basis as a warm-start for the backward LP
- **Incremental constraint updates** — Add only new cuts (from the current iteration) rather than rebuilding the full LP

See [Solver Abstraction](./solver-abstraction.md) and [Solver Workspaces](./solver-workspaces.md).

## 7. Dual Extraction for Cut Coefficients

### 7.1 Cut Structure

A Benders cut for stage $t-1$ has the form:

$$
\theta_t \geq \alpha + \sum_{h \in \mathcal{H}} \beta^v_h \cdot v_{h,t-1} + \sum_{h \in \mathcal{H}} \sum_{\ell=1}^{P_h} \beta^a_{h,\ell} \cdot a_{h,t-1-\ell}
$$

where:

| Symbol             | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| $\alpha$           | Cut intercept (constant term)                                    |
| $\beta^v_h$        | Cut coefficient for hydro $h$'s storage state variable           |
| $\beta^a_{h,\ell}$ | Cut coefficient for hydro $h$'s inflow lag $\ell$ state variable |
| $v_{h,t-1}$        | End-of-stage storage at stage $t-1$ (state variable)             |
| $a_{h,t-1-\ell}$   | Inflow lag $\ell$ at stage $t-1$ (state variable)                |

### 7.2 Derivation from LP Duality

The cut coefficients are derived from the dual variables of the backward LP's state-linking constraints:

- **Storage**: The water balance constraint links incoming storage $v_{h,t-1}$ to outgoing storage. Its dual $\pi^v_h$ gives $\beta^v_h$, representing the marginal value of an additional unit of storage at the previous stage.
- **AR inflow lags**: The lag fixing constraints bind each lag variable to its incoming value. Their duals $\pi^a_{h,\ell}$ give $\beta^a_{h,\ell}$, representing the marginal value of inflow history.
- **FPHA duals**: For hydros using FPHA, the hyperplane constraint duals contribute additional terms to $\beta^v_h$ because the FPHA planes depend on storage (head). See [Cut Management §2](../01-math/cut-management.md).
- **Generic constraint duals**: When generic constraints involve state variables, their duals also contribute to cut coefficients. The mapping from generic constraint duals to state variable coefficients is static (determined at input loading time) and should be precomputed once. See [Cut Management Implementation](./cut-management-impl.md).

The intercept $\alpha$ is computed from the LP objective value and the state-dependent terms:

$$
\alpha = Q_t(\hat{x}_{t-1}, \omega) - \sum_h \beta^v_h \cdot \hat{v}_{h,t-1} - \sum_h \sum_\ell \beta^a_{h,\ell} \cdot \hat{a}_{h,t-1-\ell}
$$

### 7.3 Cut Metadata

Each cut carries metadata for cut management:

| Field        | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| Stage        | Which stage's FCF this cut belongs to                        |
| Iteration    | The iteration when this cut was generated                    |
| Active count | Number of times this cut was binding in subsequent LP solves |

The active count is used by cut selection strategies to prune dominated or inactive cuts. See [Cut Management Implementation](./cut-management-impl.md).

## Cross-References

- [SDDP Algorithm](../01-math/sddp-algorithm.md) — Mathematical definition of the SDDP algorithm that this training loop implements
- [Cut Management (Math)](../01-math/cut-management.md) — Mathematical foundations for cut coefficients, selection theory, and dominance criteria
- [Cut Management Implementation](./cut-management-impl.md) — FCF structure, cut selection strategies, serialization, and cross-rank cut synchronization
- [Stopping Rules](../01-math/stopping-rules.md) — Convergence criteria and termination conditions
- [Risk Measures](../01-math/risk-measures.md) — CVaR mathematical formulation and cut weight computation
- [Work Distribution](../04-hpc/work-distribution.md) — Detailed MPI+OpenMP parallelism patterns for forward and backward pass distribution
- [Convergence Monitoring](./convergence-monitoring.md) — Convergence criteria, bound computation, and stopping rules applied within this loop
- [Input Loading Pipeline](./input-loading-pipeline.md) — How case data and warm-start policy cuts are loaded before training begins
- [Input Constraints](../02-data-model/input-constraints.md) — Initial conditions (§1) that provide the starting state $x_0$
- [Input Scenarios](../02-data-model/input-scenarios.md) — Scenario source configuration (§2.1), external scenarios (§2.5)
- [Scenario Generation](./scenario-generation.md) — Sampling scheme abstraction (§3), fixed opening tree lifecycle (§2.3), external scenario integration (§4)
- [Penalty System](../02-data-model/penalty-system.md) — Recourse slacks guaranteeing LP feasibility
- [Solver Abstraction](./solver-abstraction.md) — Solver interface and LP construction
- [Solver Workspaces](./solver-workspaces.md) — Solver state management, basis persistence, and warm-starting
- [Checkpointing](../04-hpc/checkpointing.md) — Checkpoint format and graceful shutdown
- [Deferred Features](../06-deferred/deferred-features.md) — Multi-cut (C.3), alternative forward pass (C.13), Monte Carlo backward sampling (C.14), policy compatibility validation (C.9)
