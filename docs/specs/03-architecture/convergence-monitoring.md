---
status: approved
review_priority: 3-medium
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §16 (16.1-16.3)"
last_reviewed: 2026-02-23
reviewed_by: "rogerio"
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from ARCHITECTURE §16"
  - date: 2026-02-23
    description: "P3 review: Fixed priority from 2-high to 3-medium. Stripped all Rust code (~170 lines across §2-§3) and replaced with behavioral descriptions. §2 rewritten as tracked quantities table + convergence check logic. §3 rewritten as behavioral cross-rank aggregation description. Added §3.2 for lower bound properties. Added §3.3 for infinite horizon discount factor. Clarified LB includes future cost approximation θ₂. Added cross-reference to infinite-horizon.md and risk-measures.md. Removed gap_tolerance/min_iterations rule (not in approved stopping-rules.md). Aligned naming with stopping-rules.md: 'bound stalling' (not 'stable bound'), config parameters iterations/tolerance (not stable_iterations/hardcoded 10⁻⁶). Aligned termination reasons with approved rule types. Aligned time_limit parameter name with config (seconds, not time_limit_seconds)."
---

# Convergence Monitoring

## Purpose

This spec defines the POWE.RS SDDP convergence monitoring architecture: the convergence criteria and stopping rules, the convergence monitor's tracked quantities and evaluation logic, bound computation including cross-rank aggregation, and the training log format for progress reporting. For the mathematical foundations, see [Stopping Rules](../01-math/stopping-rules.md) and [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md).

## 1. Convergence Criteria

SDDP convergence is determined by the gap between lower and upper bounds on the optimal objective:

**Lower Bound (LB):**

- The objective value of the stage-1 LP, which includes both the immediate stage-1 cost and the future cost approximation $\theta_2$ from accumulated cuts: $LB = \min_x \{ c_1^\top x_1 + \theta_2 : \text{constraints} \}$
- Deterministic: all scenarios share the same initial state $x_0$, so the stage-1 LP is identical regardless of scenario — only one solve is needed
- Monotonically non-decreasing across iterations, since cuts only tighten the FCF approximation

**Upper Bound (UB):**

- Statistical estimate from forward simulation: the mean total cost across all $N$ scenario trajectories
- $UB_k = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} c_t^{(i)}$
- Includes 95% confidence interval: $UB \pm 1.96 \cdot \sigma / \sqrt{N}$
- Not monotonic — depends on the scenarios sampled in each iteration
- For risk-averse policies (CVaR), the upper bound computation incorporates the risk measure weighting; see [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md)

**Convergence Gap:**

$$
\text{gap} = \frac{UB - LB}{|UB|}
$$

with a guard for near-zero $|UB|$: if $|UB| < 10^{-10}$, the gap is defined as zero. The gap is computed and logged each iteration for progress reporting, but is **not** itself a stopping criterion in standard SDDP — convergence is determined by the stopping rules below.

> **Complete tree mode note**: In [complete tree mode](./scenario-generation.md) (deferred — see [C.12](../06-deferred/deferred-features.md)), all scenarios from the stochastic tree are enumerated exhaustively. The upper bound becomes **deterministic** (exact expected cost, not a statistical estimate), and the gap becomes an exact convergence measure. In this case, a gap-based stopping criterion becomes meaningful and should be added when complete tree mode is implemented. This differs from standard SDDP where the upper bound is a noisy Monte Carlo estimate.

**Stopping Rules:**

The convergence monitor evaluates the stopping rules defined in [Stopping Rules](../01-math/stopping-rules.md). The available rules and their config parameters are:

| Rule              | Condition                                                                     | Config Type       |
| ----------------- | ----------------------------------------------------------------------------- | ----------------- |
| Iteration limit   | Iteration count $\geq$ `limit`                                                | `iteration_limit` |
| Time limit        | Wall-clock time $\geq$ `seconds`                                              | `time_limit`      |
| Bound stalling    | LB relative improvement over `iterations` window $<$ `tolerance`              | `bound_stalling`  |
| Simulation-based  | Bound stable AND simulated policy costs stable (checked every `period` iters) | `simulation`      |
| Graceful shutdown | External signal received (checkpoints last **completed** iteration)           | OS signal         |

Rules combine via `stopping_mode`: `"any"` (default, OR logic) or `"all"` (AND logic). See [Stopping Rules](../01-math/stopping-rules.md) for full mathematical definitions and configuration schema.

## 2. Convergence Monitor

The convergence monitor is updated once per iteration (after forward synchronization) and evaluates all stopping rules.

### 2.1 Tracked Quantities

The monitor maintains the following per-iteration history:

| Quantity                  | Type       | Description                                                                |
| ------------------------- | ---------- | -------------------------------------------------------------------------- |
| Lower bound               | `f64`      | Stage-1 LP objective value                                                 |
| Upper bound               | `f64`      | Mean total forward cost across all scenarios                               |
| Upper bound std           | `f64`      | Standard deviation of total forward costs                                  |
| Gap                       | `f64`      | Relative gap $(UB - LB) / \lvert UB \rvert$                                |
| Iteration wall-clock time | `Duration` | Elapsed time for the full iteration (forward + backward + synchronization) |

### 2.2 Bound Stalling Detection

The bound stalling rule tracks the relative improvement of the lower bound over a configurable window. Given the `bound_stalling` configuration with parameters `iterations` ($\tau$) and `tolerance`:

$$
\Delta_k = \frac{\underline{z}^k - \underline{z}^{k-\tau}}{\max(1, |\underline{z}^k|)}
$$

The rule triggers when $|\Delta_k| < \text{tolerance}$.

This is the same formula defined in [Stopping Rules §4](../01-math/stopping-rules.md). The convergence monitor maintains the bound history needed to evaluate this windowed comparison.

### 2.3 Convergence Evaluation

At each iteration, the monitor evaluates all configured stopping rules:

1. **Iteration limit** — If iteration count $\geq$ `limit`, report terminated
2. **Time limit** — If cumulative wall-clock time $\geq$ `seconds`, report terminated
3. **Bound stalling** — If LB relative improvement over the last `iterations` window is below `tolerance`, report converged
4. **Simulation-based** — If the check period has elapsed: test bound stability, then run Monte Carlo simulations and compare to previous; if both stable, report converged. See [Stopping Rules §5](../01-math/stopping-rules.md)
5. **Graceful shutdown** — If an external signal flag is set, report terminated

The `stopping_mode` determines whether the first satisfied rule terminates (mode `"any"`) or all must be satisfied (mode `"all"`). The termination reason is recorded in the training log and output metadata.

### 2.4 Per-Iteration Output Record

Each iteration produces a record with the following fields, used for logging (§4) and output persistence:

| Field             | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| `iteration`       | Iteration index (1-based)                                           |
| `lower_bound`     | LB value                                                            |
| `upper_bound`     | UB value (mean forward cost)                                        |
| `upper_bound_std` | Standard deviation of forward costs                                 |
| `ci_95`           | 95% confidence interval half-width ($1.96 \cdot \sigma / \sqrt{N}$) |
| `gap`             | Relative gap (informational, not a stopping criterion)              |
| `wall_time`       | Cumulative wall-clock time                                          |
| `iteration_time`  | Wall-clock time for this iteration                                  |

## 3. Bound Computation

### 3.1 Cross-Rank Aggregation

Forward pass scenarios are distributed across MPI ranks (see [Training Loop §4.3](./training-loop.md)). After the forward pass, bound statistics must be aggregated globally. This is done via a single `MPI_Allreduce` operation that collects three sufficient statistics from each rank:

| Statistic        | Per-rank value                                       | Reduction operation |
| ---------------- | ---------------------------------------------------- | ------------------- |
| Scenario count   | Number of trajectories solved by this rank           | Sum                 |
| Cost sum         | Sum of total costs across local trajectories         | Sum                 |
| Cost sum-squares | Sum of squared total costs across local trajectories | Sum                 |

From the global aggregates, the upper bound statistics are computed:

- **Mean**: $\bar{c} = \text{sum} / N$
- **Variance**: $s^2 = (\text{sum\_sq} / N) - \bar{c}^2$
- **Standard deviation**: $\sigma = \sqrt{s^2}$
- **95% CI half-width**: $1.96 \cdot \sigma / \sqrt{N}$

This single allreduce (3 doubles) is sufficient — no per-scenario data needs to be communicated.

### 3.2 Lower Bound Properties

The lower bound is the objective value of the stage-1 LP, which is deterministic (identical across all scenarios and ranks) because all scenarios share the same initial state $x_0$. Only one rank needs to solve and broadcast this value.

Key properties:

- **Monotonically non-decreasing** — Each iteration adds cuts that can only tighten the FCF under-approximation, so $LB_{k+1} \geq LB_k$
- **Valid lower bound** — The stage-1 objective with the current FCF under-approximates the true expected cost, so $LB_k \leq z^*$ for all $k$
- **Includes $\theta_2$** — The LB is not just the stage-1 immediate cost; it includes the future cost variable $\theta_2$ constrained by all accumulated cuts, which represents the best current approximation of the expected cost from stage 2 onward

### 3.3 Infinite Horizon Considerations

In cyclic (infinite horizon) mode, the discount factor $d < 1$ ensures that the infinite sum of stage costs converges. The bound computation must account for this:

- Stage costs in the forward pass are discounted: the cost at stage $t$ is weighted by $d^{t-1}$
- The upper bound is the mean of discounted total trajectory costs
- The lower bound naturally incorporates discounting through the $\theta$ variable, which carries the discounted future cost approximation

See [Infinite Horizon](../01-math/infinite-horizon.md) for the mathematical treatment of discount factors in the SDDP context.

## 4. Training Log Format

The convergence monitor emits a structured log each iteration. The log includes a header (emitted once at training start) and per-iteration lines:

**Header:**

```
═══════════════════════════════════════════════════════════════════
POWE.RS SDDP Training
Case: <case_name>
Started: <timestamp>
Ranks: <R> | Threads/rank: <T> | Stages: <S> | Hydros: <H>
═══════════════════════════════════════════════════════════════════
```

**Per-iteration line:**

```
Iter <n> | LB: <value> | UB: <value> ± <ci_95> | Gap: <value>%
```

**Termination summary:**

```
═══════════════════════════════════════════════════════════════════
<TERMINATION_REASON> after <n> iterations (<reason detail>)
Total time: <elapsed> | Avg iteration: <avg>
Final LB: <value> | Final UB: <value> ± <ci_95>
Total cuts: <count> | Cuts/stage: ~<avg>
═══════════════════════════════════════════════════════════════════
```

The termination reason is one of: `BOUND_STALLING`, `SIMULATION`, `ITERATION_LIMIT`, `TIME_LIMIT`, or `SHUTDOWN` (graceful signal).

## Cross-References

- [Stopping Rules](../01-math/stopping-rules.md) — Mathematical definitions of the stopping rules implemented by the convergence monitor
- [Upper Bound Evaluation](../01-math/upper-bound-evaluation.md) — Statistical upper bound theory, confidence interval construction, and bias corrections
- [Risk Measures](../01-math/risk-measures.md) — CVaR risk measure, which affects upper bound computation for risk-averse policies
- [Infinite Horizon](../01-math/infinite-horizon.md) — Discount factor treatment for cyclic stage graphs
- [Training Loop](./training-loop.md) — The SDDP training loop that invokes this convergence monitor each iteration (§2.1 iteration lifecycle, §4.3 forward synchronization)
- [Configuration Reference](../05-config/configuration-reference.md) — JSON schema for `stopping_rules` and `stopping_mode`
