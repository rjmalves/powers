---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §27 (27.1-27.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §28 (28.1-28.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §29 (29.1-29.4)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §27-29"
---

# Extension Points

## Purpose

This spec defines the extensibility architecture of the POWE.RS SDDP solver: the core trait abstractions that allow algorithm variants (risk measures, cut formulations, horizon modes) to be selected at runtime via configuration, the factory pattern for instantiation, and the concrete implementations of each extension point.

## 1. Extensibility Architecture Overview

POWE.RS uses trait-based polymorphism to support algorithm variants without code duplication:

```
┌───────────────────────────────────────────────────────────────┐
│                  Extension Point Architecture                  │
├───────────────────────────────────────────────────────────────┤
│  TrainingLoop<R, C, H>                                        │
│    R: RiskMeasure    — aggregate outcomes into cuts            │
│    C: CutFormulation — structure of cut constraints            │
│    H: HorizonMode    — finite, infinite, or periodic horizon  │
│  Fixed: forward/backward pass, MPI comms, convergence         │
├───────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐ ┌──────────────────┐ ┌────────────────┐│
│  │  RiskMeasure     │ │ CutFormulation   │ │  HorizonMode   ││
│  │  • ExpectedValue │ │ • SingleCut      │ │  • Finite      ││
│  │  • CVaR          │ │ • MultiCut       │ │  • InfUniform  ││
│  │  • Entropic      │ │ • SDDiP          │ │  • InfPeriodic ││
│  │  • WorstCase     │ │                  │ │                ││
│  └──────────────────┘ └──────────────────┘ └────────────────┘│
└───────────────────────────────────────────────────────────────┘
```

## 2. Core Trait Definitions

### 2.1 RiskMeasure Trait

```rust
/// Risk measure maps the distribution of future costs to a scalar value
/// representing the "risk-adjusted" expected cost.
pub trait RiskMeasure: Send + Sync + Clone + 'static {
    /// Compute cut coefficients from backward pass evaluations
    fn compute_cut(
        &self,
        stage: StageId,
        state: &StatePoint,
        outcomes: &[BackwardOutcome],
        probabilities: &[f64],
    ) -> CutCoefficients;

    /// Risk-adjusted objective for convergence bound computation
    fn evaluate_risk(&self, values: &[f64], probabilities: &[f64]) -> f64;

    /// Display name for logging
    fn name(&self) -> &'static str;

    /// Parameters for serialization/deserialization
    fn parameters(&self) -> RiskMeasureParameters;
}
```

### 2.2 CutFormulation Trait

```rust
/// Cut formulation determines the structure of cuts in the LP
pub trait CutFormulation: Send + Sync + Clone + 'static {
    /// Number of cuts generated per backward state evaluation
    fn cuts_per_state(&self) -> usize;

    /// Build cut constraint(s) for addition to stage LP
    fn build_constraints(
        &self, cut: &CutCoefficients, theta_var: VarId, state_vars: &StateVariables,
    ) -> Vec<Constraint>;

    /// Whether this formulation requires strengthening (default: false)
    fn requires_strengthening(&self) -> bool { false }

    fn name(&self) -> &'static str;
}
```

### 2.3 HorizonMode Trait

```rust
/// Horizon mode determines stage transitions and terminal conditions
pub trait HorizonMode: Send + Sync + Clone + 'static {
    /// Get successor stages with transition probabilities
    fn successors(&self, stage: StageId, stages: &[Stage]) -> Vec<(StageId, f64)>;

    /// Check if stage is terminal (no successors)
    fn is_terminal(&self, stage: StageId, stages: &[Stage]) -> bool;

    /// Discount factor for future costs (1.0 for undiscounted)
    fn discount_factor(&self, from_stage: StageId, to_stage: StageId) -> f64;

    /// Validate stage configuration
    fn validate(&self, stages: &[Stage]) -> Result<(), ValidationError>;

    fn name(&self) -> &'static str;
}
```

## 3. Factory Pattern for Configuration-Driven Selection

```rust
pub struct AlgorithmFactory;

impl AlgorithmFactory {
    pub fn create_risk_measure(config: &RiskConfig) -> Box<dyn RiskMeasure> {
        match config {
            RiskConfig::ExpectedValue => Box::new(ExpectedValueRisk),
            RiskConfig::CVaR { alpha } => Box::new(CVaRRisk::new(*alpha)),
            RiskConfig::Entropic { gamma } => Box::new(EntropicRisk::new(*gamma)),
            RiskConfig::ConvexCombination { lambda, inner } => {
                let inner_risk = Self::create_risk_measure(inner);
                Box::new(ConvexCombinationRisk::new(*lambda, inner_risk))
            }
        }
    }

    pub fn create_horizon_mode(config: &HorizonConfig) -> Box<dyn HorizonMode> {
        match config {
            HorizonConfig::Finite => Box::new(FiniteHorizon),
            HorizonConfig::InfiniteUniform { discount_rate } => {
                Box::new(InfiniteUniformHorizon::new(*discount_rate))
            }
            HorizonConfig::InfinitePeriodic { cycle_start, cycle_length, discount_rate } => {
                Box::new(InfinitePeriodicHorizon::new(*cycle_start, *cycle_length, *discount_rate))
            }
        }
    }

    /// Create complete training loop with configured components (dynamic dispatch)
    pub fn create_training_loop(
        config: &Config, comm: WorldCommunicator,
    ) -> Box<dyn TrainingLoopDyn> {
        let risk = Self::create_risk_measure(&config.risk);
        let horizon = Self::create_horizon_mode(&config.horizon);
        Box::new(TrainingLoopImpl::new(
            risk, SingleCutFormulation, horizon, config.training.clone(), comm
        ))
    }
}
```

## 4. Risk Measure Implementations

All risk measure implementations share a common pattern for computing cut coefficients: compute per-outcome weights, then form weighted sums of objective values and dual multipliers to produce the cut intercept and gradients.

### 4.1 Expected Value (Risk-Neutral)

The standard SDDP risk measure: **E[Q(x, ω)]**. Weights are simply the original probabilities.

```rust
#[derive(Clone)]
pub struct ExpectedValueRisk;

impl RiskMeasure for ExpectedValueRisk {
    fn compute_cut(
        &self, _stage: StageId, _state: &StatePoint,
        outcomes: &[BackwardOutcome], probabilities: &[f64],
    ) -> CutCoefficients {
        let expected_q: f64 = outcomes.iter().zip(probabilities)
            .map(|(o, p)| p * o.objective).sum();

        let n_storage = outcomes[0].dual_storage.len();
        let n_inflow = outcomes[0].dual_inflow.len();
        let mut storage_coef = vec![0.0; n_storage];
        let mut inflow_coef = vec![0.0; n_inflow];

        for (outcome, &prob) in outcomes.iter().zip(probabilities) {
            for (i, &dual) in outcome.dual_storage.iter().enumerate() {
                storage_coef[i] += prob * dual;
            }
            for (i, &dual) in outcome.dual_inflow.iter().enumerate() {
                inflow_coef[i] += prob * dual;
            }
        }
        CutCoefficients { intercept: expected_q, storage_coef, inflow_coef }
    }

    fn evaluate_risk(&self, values: &[f64], probabilities: &[f64]) -> f64 {
        values.iter().zip(probabilities).map(|(v, p)| p * v).sum()
    }

    fn name(&self) -> &'static str { "ExpectedValue" }
    fn parameters(&self) -> RiskMeasureParameters { RiskMeasureParameters::ExpectedValue }
}
```

### 4.2 Conditional Value-at-Risk (CVaR)

**CVaR*α = E[Q | Q ≥ VaR*α]** — focuses on the worst (1−α) fraction of outcomes. For example, α = 0.95 averages over the worst 5% of scenarios.

The key difference from expected value is the weight computation: outcomes are sorted by objective (descending), and probability mass is redistributed to concentrate on the tail.

```rust
#[derive(Clone)]
pub struct CVaRRisk { alpha: f64 }

impl CVaRRisk {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "α must be in (0, 1)");
        Self { alpha }
    }

    /// Compute CVaR weights: sort by value descending, redistribute mass to worst (1-α) tail
    fn cvar_weights(&self, values: &[f64], probabilities: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64, f64)> = values.iter()
            .zip(probabilities).enumerate()
            .map(|(i, (&v, &p))| (i, v, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let tail_prob = 1.0 - self.alpha;
        let mut cumulative = 0.0;
        let mut weights = vec![0.0; values.len()];

        for (idx, _, prob) in &indexed {
            let contribution = if cumulative + prob <= tail_prob {
                *prob
            } else {
                (tail_prob - cumulative).max(0.0)
            };
            weights[*idx] = contribution / tail_prob;
            cumulative += prob;
            if cumulative >= tail_prob { break; }
        }
        weights
    }
}

impl RiskMeasure for CVaRRisk {
    fn compute_cut(
        &self, _stage: StageId, _state: &StatePoint,
        outcomes: &[BackwardOutcome], probabilities: &[f64],
    ) -> CutCoefficients {
        let objectives: Vec<f64> = outcomes.iter().map(|o| o.objective).collect();
        let weights = self.cvar_weights(&objectives, probabilities);

        let cvar_q: f64 = outcomes.iter().zip(&weights)
            .map(|(o, w)| w * o.objective).sum();

        let n_storage = outcomes[0].dual_storage.len();
        let n_inflow = outcomes[0].dual_inflow.len();
        let mut storage_coef = vec![0.0; n_storage];
        let mut inflow_coef = vec![0.0; n_inflow];

        for (outcome, &weight) in outcomes.iter().zip(&weights) {
            for (i, &dual) in outcome.dual_storage.iter().enumerate() {
                storage_coef[i] += weight * dual;
            }
            for (i, &dual) in outcome.dual_inflow.iter().enumerate() {
                inflow_coef[i] += weight * dual;
            }
        }
        CutCoefficients { intercept: cvar_q, storage_coef, inflow_coef }
    }

    fn evaluate_risk(&self, values: &[f64], probabilities: &[f64]) -> f64 {
        let weights = self.cvar_weights(values, probabilities);
        values.iter().zip(&weights).map(|(v, w)| w * v).sum()
    }

    fn name(&self) -> &'static str { "CVaR" }
    fn parameters(&self) -> RiskMeasureParameters {
        RiskMeasureParameters::CVaR { alpha: self.alpha }
    }
}
```

### 4.3 Convex Combination Risk

**ρ(Q) = λ·E[Q] + (1−λ)·ρ_inner(Q)** — common choice: λ=0.5 with CVaR₀.₉₅ gives balanced risk-aversion. Delegates to `ExpectedValueRisk` and `inner`, then linearly combines the results.

```rust
#[derive(Clone)]
pub struct ConvexCombinationRisk {
    lambda: f64,
    inner: Box<dyn RiskMeasure>,
}

impl ConvexCombinationRisk {
    pub fn new(lambda: f64, inner: Box<dyn RiskMeasure>) -> Self {
        assert!(lambda >= 0.0 && lambda <= 1.0, "λ must be in [0, 1]");
        Self { lambda, inner }
    }
}

impl RiskMeasure for ConvexCombinationRisk {
    fn compute_cut(
        &self, stage: StageId, state: &StatePoint,
        outcomes: &[BackwardOutcome], probabilities: &[f64],
    ) -> CutCoefficients {
        let ev = ExpectedValueRisk.compute_cut(stage, state, outcomes, probabilities);
        let ir = self.inner.compute_cut(stage, state, outcomes, probabilities);
        let (l, r) = (self.lambda, 1.0 - self.lambda);
        CutCoefficients {
            intercept: l * ev.intercept + r * ir.intercept,
            storage_coef: ev.storage_coef.iter().zip(&ir.storage_coef)
                .map(|(e, i)| l * e + r * i).collect(),
            inflow_coef: ev.inflow_coef.iter().zip(&ir.inflow_coef)
                .map(|(e, i)| l * e + r * i).collect(),
        }
    }

    fn evaluate_risk(&self, values: &[f64], probabilities: &[f64]) -> f64 {
        let ev = ExpectedValueRisk.evaluate_risk(values, probabilities);
        let ir = self.inner.evaluate_risk(values, probabilities);
        self.lambda * ev + (1.0 - self.lambda) * ir
    }

    fn name(&self) -> &'static str { "ConvexCombination" }
    fn parameters(&self) -> RiskMeasureParameters {
        RiskMeasureParameters::ConvexCombination {
            lambda: self.lambda, inner: Box::new(self.inner.parameters()),
        }
    }
}
```

## 5. Horizon Mode Implementations

### 5.1 Finite Horizon

Standard SDDP: stages 1, 2, …, T with no cycles. Stage T has terminal value function (zero or specified).

```rust
#[derive(Clone)]
pub struct FiniteHorizon;

impl HorizonMode for FiniteHorizon {
    fn successors(&self, stage: StageId, stages: &[Stage]) -> Vec<(StageId, f64)> {
        if stage.0 + 1 < stages.len() {
            vec![(StageId(stage.0 + 1), 1.0)]
        } else {
            vec![]
        }
    }

    fn is_terminal(&self, stage: StageId, stages: &[Stage]) -> bool {
        stage.0 == stages.len() - 1
    }

    fn discount_factor(&self, _from: StageId, _to: StageId) -> f64 { 1.0 }

    fn validate(&self, stages: &[Stage]) -> Result<(), ValidationError> {
        if stages.is_empty() { return Err(ValidationError::new("Must have at least one stage")); }
        Ok(())
    }

    fn name(&self) -> &'static str { "Finite" }
}
```

### 5.2 Infinite Horizon with Uniform Discounting

All stages use the same discount factor δ ∈ (0, 1). Convergence requires δ < 1 for bounded costs. Stages wrap around: after the last stage, transitions back to stage 0.

```rust
#[derive(Clone)]
pub struct InfiniteUniformHorizon { discount_rate: f64 }

impl InfiniteUniformHorizon {
    pub fn new(discount_rate: f64) -> Self {
        assert!(discount_rate > 0.0 && discount_rate < 1.0, "Discount rate must be in (0, 1)");
        Self { discount_rate }
    }
}

impl HorizonMode for InfiniteUniformHorizon {
    fn successors(&self, stage: StageId, stages: &[Stage]) -> Vec<(StageId, f64)> {
        vec![(StageId((stage.0 + 1) % stages.len()), 1.0)]
    }

    fn is_terminal(&self, _stage: StageId, _stages: &[Stage]) -> bool { false }

    fn discount_factor(&self, _from: StageId, _to: StageId) -> f64 { self.discount_rate }

    fn validate(&self, stages: &[Stage]) -> Result<(), ValidationError> {
        if stages.is_empty() { return Err(ValidationError::new("Must have at least one stage")); }
        Ok(())
    }

    fn name(&self) -> &'static str { "InfiniteUniform" }
}
```

### 5.3 Infinite Horizon with Periodic Structure

Structure: `[Initial: 0..cycle_start] → [Cycle: cycle_start..cycle_start+cycle_length]` (repeats). After the last cycle stage, transitions back to `cycle_start` with a discount factor. Useful for systems with seasonal patterns extending to infinity.

```rust
#[derive(Clone)]
pub struct InfinitePeriodicHorizon {
    cycle_start: usize,
    cycle_length: usize,
    discount_rate: f64,
}

impl InfinitePeriodicHorizon {
    pub fn new(cycle_start: usize, cycle_length: usize, discount_rate: f64) -> Self {
        assert!(cycle_length > 0, "Cycle length must be positive");
        assert!(discount_rate > 0.0 && discount_rate < 1.0, "Discount rate must be in (0, 1)");
        Self { cycle_start, cycle_length, discount_rate }
    }
}

impl HorizonMode for InfinitePeriodicHorizon {
    fn successors(&self, stage: StageId, stages: &[Stage]) -> Vec<(StageId, f64)> {
        let cycle_end = self.cycle_start + self.cycle_length;
        if stage.0 + 1 < cycle_end && stage.0 + 1 < stages.len() {
            vec![(StageId(stage.0 + 1), 1.0)]              // Normal progression
        } else if stage.0 + 1 == cycle_end || stage.0 + 1 == stages.len() {
            vec![(StageId(self.cycle_start), 1.0)]          // Wrap to cycle start
        } else {
            vec![]
        }
    }

    fn is_terminal(&self, _stage: StageId, _stages: &[Stage]) -> bool { false }

    fn discount_factor(&self, from: StageId, to: StageId) -> f64 {
        if to.0 < from.0 { self.discount_rate } else { 1.0 }
    }

    fn validate(&self, stages: &[Stage]) -> Result<(), ValidationError> {
        if self.cycle_start >= stages.len() {
            return Err(ValidationError::new(format!(
                "cycle_start ({}) must be < n_stages ({})", self.cycle_start, stages.len()
            )));
        }
        if self.cycle_start + self.cycle_length > stages.len() {
            return Err(ValidationError::new("cycle extends beyond available stages"));
        }
        Ok(())
    }

    fn name(&self) -> &'static str { "InfinitePeriodic" }
}
```

### 5.4 Stage Configuration for Periodic Horizon

Example: 10-year study with 5-year operational cycle (monthly stages: 60 initial + 60 cycle = 120 stages).

```
cycle_start=60, cycle_length=60, discount_rate=0.95

Initial: [0] → [1] → ... → [59]
                                 ↓
  Cycle:  [60] → [61] → ... → [119] ──┐
           ↑                           │ (discount × 0.95)
           └───────────────────────────┘

Cuts at stage 59 reference FCF at stage 60.
Cuts at stage 119 reference FCF at stage 60 (with discount).
```

## Cross-References

- [Risk Measures](../01-math/risk-measures.md) — mathematical definitions of CVaR, convex combination, and coherent risk measures
- [Discount Rate](../01-math/discount-rate.md) — discount factor mathematics for infinite horizon modes
- [Simulation Architecture](./simulation-architecture.md) — how extension points are used during policy evaluation
- [CLI and Lifecycle](./cli-and-lifecycle.md) — execution phases that invoke the training loop with configured extensions
