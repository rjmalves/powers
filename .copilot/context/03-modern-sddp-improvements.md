# Modern SDDP Improvements and Research Directions

## Overview

This document catalogs state-of-the-art improvements to the SDDP algorithm discovered through research over the past decade. These enhancements address convergence speed, numerical stability, scalability, and applicability to broader problem classes. Many are implemented in SDDP.jl and represent proven, production-ready techniques.

## 1. Cut Management and Accelerated Convergence

### 1.1 Multi-Cut vs. Single-Cut

**Single-Cut (Average-Cut)**

- **Current POWE.RS approach**
- Aggregates scenarios: `θ ≥ E[α_ω + β_ω^T x]`
- One cut per stage per iteration
- **Pros**: Numerically stable, fewer constraints
- **Cons**: Slower convergence with many scenarios

**Multi-Cut**

- Separate cuts per scenario: `φ_ω ≥ α_ω + β_ω^T x, θ ≥ Σ p_ω φ_ω`
- More accurate value function representation
- **Pros**: Faster convergence (2-5x for some problems)
- **Cons**: More constraints, potential numerical issues

**Implementation Priority**: **HIGH**  
**Effort**: Moderate (2-3 weeks)  
**Reference**: Birge & Louveaux (1988), "A Multicut Algorithm for Two-Stage Stochastic Programs"

### 1.2 Strengthened Benders Cuts

**Standard cuts**:

```
V(x) ≥ α + β^T x
```

**Strengthened cuts** add valid inequalities from subproblem structure:

```
V(x) ≥ α + β^T x + γ^T x  (where γ from problem-specific structure)
```

Examples:

- Integer subproblems: Add Gomory cuts
- Network flows: Add strong inequalities
- Hydro: Add convex hull of turbining functions

**Implementation Priority**: **LOW** (problem-specific)  
**Effort**: High (requires deep problem knowledge)  
**Reference**: Magnanti & Wong (2011), "Accelerating Benders Decomposition"

### 1.3 Cut Selection Strategies

**Level Method** (Current POWE.RS approach)

- Keep cuts that dominate at visited states
- Track `non_dominated_state_count`
- Remove cuts with low counts

**Alternative: Regularization Method**

- Add trust region around current solution
- Stabilizes cut generation
- Particularly effective for non-convex master problems

**Alternative: Bundle Methods**

- Maintain bundle of subgradients
- Generate aggregate cuts
- Theory from convex optimization

**Alternative: Activity-Based Selection**

- Track how often cuts are active (tight) in solutions
- Remove cuts never active for N iterations
- Simpler than dominance tracking

**Implementation Priority**: **MEDIUM** (alternatives to current approach)  
**Effort**: Low to Moderate  
**Reference**: de Matos, Philpott & Finardi (2015), "Improving the performance of Stochastic Dual Dynamic Programming"

### 1.4 Cut Sharing (Refine at Similar Nodes)

**Current POWE.RS**: Partially implemented  
When two nodes have identical children (e.g., same stage in Markov chain), cuts generated at one can be added to the other.

**Enhancement**: Approximate cut sharing

- Share cuts between "similar" nodes (not just identical)
- Use distance metric on state space
- Particularly effective in Markovian graphs

**Implementation Priority**: **LOW** (already have exact version)  
**Effort**: Moderate  
**Reference**: Shapiro (2011), "Analysis of Stochastic Dual Dynamic Programming Method"

## 2. Risk Measures and Robustness

### 2.1 Coherent Risk Measures

**Risk-neutral** (Current POWE.RS):

```
min E[cost]
```

**Coherent risk measures** satisfy:

1. Monotonicity
2. Subadditivity
3. Positive homogeneity
4. Translation invariance

**Commonly used risk measures**:

#### Conditional Value at Risk (CVaR / AVaR)

```
CVaR_α[Z] = min{t + (1/α)E[(Z - t)^+]}
```

- Focus on worst α% of scenarios
- α = 0.05 means worst 5% of outcomes
- α = 1.0 reduces to expectation

**Dual representation**:
Modify scenario probabilities in backward pass to emphasize worst cases.

#### Convex Combinations

```
ρ[Z] = λ·E[Z] + (1-λ)·CVaR_α[Z]
```

- Balance expected performance and worst-case
- Typical: λ = 0.5, α = 0.1

#### Worst-Case

```
max_ω Z(ω)
```

- Extreme risk aversion
- Conservative policies
- Often too pessimistic

**Implementation Priority**: **HIGH**  
**Effort**: Moderate (CVaR: 1-2 weeks, others: +1 week each)  
**Reference**: Shapiro, Dentcheva & Ruszczyński (2009), "Lectures on Stochastic Programming"

### 2.2 Distributionally Robust Optimization (DRO)

Instead of fixed probability distribution, consider **ambiguity set**:

```
min max_{P ∈ U} E_P[cost]
```

**Modified Chi-Squared DRO** (Philpott et al.):

```
U = {p : (p - q)^T Σ^{-1} (p - q) ≤ ρ}
```

- q = nominal distribution
- ρ = robustness radius
- Larger ρ = more conservative

**Advantages**:

- Protects against distributional uncertainty
- Theory: finite-sample guarantees
- Practical: better out-of-sample performance

**Implementation Priority**: **MEDIUM**  
**Effort**: Moderate (2-3 weeks)  
**Reference**: Philpott, de Matos & Kapelevich (2018), "Distributionally robust SDDP"

### 2.3 Entropic Risk Measure

```
ρ_γ[Z] = (1/γ) log E[exp(γ·Z)]
```

- γ → 0: approaches expectation
- γ → ∞: approaches worst-case
- Smooth interpolation
- Computationally efficient dual form

**Implementation Priority**: **LOW** (less common in practice)  
**Effort**: Moderate  
**Reference**: Dowson, Morton & Pagnoncelli (2022), "Incorporating convex risk measures..."

## 3. Sampling and Scenario Generation

### 3.1 Forward Pass Sampling Schemes

**In-Sample Monte Carlo** (Current POWE.RS)

- Pre-generate scenario tree (SAA)
- Sample from fixed set each iteration
- **Pros**: Deterministic, reproducible
- **Cons**: May miss important states

**Out-of-Sample Monte Carlo**

- Generate fresh scenarios each iteration
- **Pros**: Better exploration, theoretical convergence guarantees
- **Cons**: Non-deterministic, may visit bad states

**Historical Sampling**

- Use historical data as scenarios
- **Pros**: Realistic, includes correlations
- **Cons**: Limited sample size, may not cover tail events

**Implementation Priority**: **MEDIUM**  
**Effort**: Low (1 week for out-of-sample)  
**Reference**: Shapiro & Philpott (2007), "A Tutorial on Stochastic Programming"

### 3.2 Risk-Adjusted Forward Pass

**Idea**: Revisit worst trajectories more often

- Forward pass generates trajectory with cost C
- Store {trajectory, cost} pairs
- With probability p, revisit trajectory (using risk measure to weight)
- With probability 1-p, sample new trajectory

**Benefit**: Refines policy at states that matter most for risk-averse objectives

**Implementation Priority**: **MEDIUM**  
**Effort**: Moderate (2 weeks)  
**Reference**: SDDP.jl implementation (Dowson)

### 3.3 Adaptive Sampling

**Static sampling** (Current): Fixed number of scenarios per stage

**Adaptive sampling**:

- Start with few scenarios
- Increase in stages with high variance
- Decrease in stages with low variance
- **Benefit**: Computational savings while maintaining accuracy

**Implementation Priority**: **LOW**  
**Effort**: Moderate to High  
**Reference**: Guigues & Römisch (2012), "Sampling-Based Decomposition"

### 3.4 Importance Sampling

Weight scenarios by importance for current policy:

```
p_ω' = p_ω · L(ω) / Σ p_j L(j)
```

Where L(ω) is likelihood under current policy.

**Benefit**: Faster convergence by focusing on relevant scenarios

**Implementation Priority**: **LOW**  
**Effort**: High  
**Reference**: Kozmík & Morton (2015), "Evaluating policies in risk-averse multi-stage stochastic programming"

## 4. Parallelization Strategies

### 4.1 Current POWE.RS Parallelism

**Implemented**:

- Parallel forward passes (Rayon)
- Parallel scenario evaluation in backward pass

**Limitations**:

- Single node only
- Shared memory

### 4.2 Distributed SDDP

**Scenario-Based Parallelism**:

- Each worker solves subset of scenarios in backward pass
- Aggregate cuts centrally
- Near-linear speedup (communication minimal)

**Iteration-Based Parallelism**:

- Multiple iterations run simultaneously
- Periodically synchronize cuts
- **Asynchronous variant**: Don't wait for all workers

**Implementation Priority**: **MEDIUM** (if targeting HPC clusters)  
**Effort**: High (4-6 weeks for MPI version)  
**Reference**: Guigues (2018), "Sampling-Based Decomposition Methods for Multistage Stochastic Programs Based on Extended Polyhedral Risk Measures"

### 4.3 GPU Acceleration

**Potential targets**:

- Matrix operations in subproblems (if large)
- Scenario sampling
- Cut evaluation

**Reality check**: Most SDDP problems are too small to benefit from GPUs

- Subproblems: ~100-1000 variables
- Solver is already fast (10-100ms)
- CPU threading usually sufficient

**Implementation Priority**: **VERY LOW**  
**Effort**: Very High

## 5. Advanced Problem Structures

### 5.1 Integer Variables (SDDiP)

**Stochastic Dual Dynamic Integer Programming**:

- Handle binary/integer decisions
- Use Lagrangian relaxation
- Generate two types of cuts:
  - Benders cuts (from LP relaxation)
  - Strengthened cuts (from integer structure)

**Application**: Unit commitment, investment decisions

**Implementation Priority**: **MEDIUM** (if needed for applications)  
**Effort**: Very High (3-4 months)  
**Reference**: Zou, Ahmed & Sun (2019), "Stochastic Dual Dynamic Integer Programming"

### 5.2 Markovian Uncertainty

**Stagewise-independent** (Current POWE.RS): ω*t ⊥ ω*{t-1} | x\_{t-1}

**Markovian**: ω*t ~ P(· | ω*{t-1}, x\_{t-1})

- State includes Markov state
- More realistic for many applications
- Examples: Weather states, price regimes

**Implementation Priority**: **LOW to MEDIUM**  
**Effort**: Moderate (3-4 weeks)  
**Reference**: Philpott & Guan (2008), "On the convergence of stochastic dual dynamic programming"

### 5.3 Stagewise-Dependent Objective

**Standard**: c_t independent of ω_t

**Stagewise-dependent**: c_t(ω_t)

- Objective depends on realization
- Example: Electricity price affects generation value
- Requires objective state variables

**Implementation Priority**: **LOW**  
**Effort**: Moderate  
**Reference**: Downward, Dowson & Baucke (2020), "Stochastic dual dynamic programming with stagewise-dependent objective uncertainty"

### 5.4 Belief State and Bayesian Learning

When decision-maker learns about uncertainty over time:

- Maintain belief distribution over latent parameters
- Belief state evolves via Bayes rule
- Value function depends on both physical and belief states

**Application**: Reservoir inflow forecasting, price prediction

**Implementation Priority**: **LOW**  
**Effort**: High  
**Reference**: Dowson (2020), "The policy graph decomposition of multistage stochastic programming problems"

## 6. Convergence Acceleration Techniques

### 6.1 Regularized Forward Pass

Add regularization term to forward pass:

```
min c^T x + θ + (μ/2)||x - x̄||^2
```

Where x̄ is previous iterate.

**Benefits**:

- Stabilizes forward pass
- Reduces oscillation
- Better for risk-averse problems

**Implementation Priority**: **MEDIUM**  
**Effort**: Low (1 week)  
**Reference**: Philpott, de Matos & Kapelevich (2018)

### 6.2 Strengthened Lower Bounds

**Standard bound**: E[θ_1] (from backward pass)

**Improvements**:

- Solve backward pass at multiple states per iteration
- Use information-relaxation bounds
- Employ perfect information relaxation

**Benefit**: Tighter bounds → better stopping criteria

**Implementation Priority**: **LOW**  
**Effort**: Moderate  
**Reference**: Brown, Smith & Sun (2010), "Information Relaxation and Duality in Stochastic Dynamic Programs"

### 6.3 Dynamic Simplex Basis Selection

**Current**: Save basis from forward pass, use in backward pass

**Enhancement**:

- Maintain library of bases for each node
- Use clustering to select best starting basis
- Particularly effective with cut selection (many cuts inactive)

**Implementation Priority**: **LOW** (current approach already good)  
**Effort**: Moderate

## 7. Reinforcement Learning Connections

### 7.1 SDDP as Approximate Dynamic Programming

SDDP can be viewed through RL lens:

- **State**: x\_{t-1}
- **Action**: x_t
- **Reward**: -c_t^T x_t
- **Value function**: V*t(x*{t-1})
- **Policy**: π*t(x*{t-1}) = argmin{c*t^T x_t + V*{t+1}(x_t)}

**Cuts**: Linear approximation of value function (like linear function approximation in RL)

### 7.2 Policy Gradient Methods

Instead of cutting planes, directly parameterize policy:

```
π_θ(x) = decision function with parameters θ
```

Optimize θ via gradient descent on expected cost.

**Advantages**:

- No curse of dimensionality (potentially)
- Smooth policies
- Exploration built-in

**Disadvantages**:

- No convergence guarantees
- Hyperparameter tuning required
- Less interpretable

**Implementation Priority**: **VERY LOW** (research topic)  
**Effort**: Very High  
**Reference**: Active research area, not yet practical for most applications

### 7.3 Monte Carlo Tree Search (MCTS)

Sample trajectories and build search tree adaptively.

**Advantages**:

- Anytime algorithm
- Explores promising regions

**Disadvantages**:

- Slower than SDDP for problems with structure
- No cut reuse across iterations

**Implementation Priority**: **VERY LOW**  
**Effort**: High

### 7.4 Transfer Learning

Use policy from related problem as starting point:

- Load cuts from similar problem
- Fine-tune with current problem data
- **Benefit**: Faster convergence

**Example**: Train on historical data, transfer to real-time operation

**Implementation Priority**: **MEDIUM** (requires cut serialization first)  
**Effort**: Low (once serialization exists)

## 8. Numerical Improvements

### 8.1 Automatic Scaling

**Problem**: State variables may have very different magnitudes

- Storage: 0-1000 GWh
- Generation: 0-100 MW
- Leads to ill-conditioning

**Solution**: Automatic normalization

```
x_scaled = (x - x_min) / (x_max - x_min)
```

Transform constraints and cuts accordingly.

**Implementation Priority**: **HIGH**  
**Effort**: Low (1 week)  
**Reference**: Standard LP preprocessing

### 8.2 Cut Coefficient Management

Track magnitude of cut coefficients:

```
||β|| = sqrt(Σ β_i^2)
```

**Warning signals**:

- Very large ||β|| (> 10^6): numerical issues likely
- Very small ||β|| (< 10^-6): cut may be redundant

**Actions**:

- Rescale cuts
- Remove problematic cuts
- Adjust tolerances

**Implementation Priority**: **MEDIUM**  
**Effort**: Low

### 8.3 Constraint/Variable Bounds

**Tight bounds improve performance**:

- Reduce feasible region
- Improve simplex performance
- Enable better presolve

**For hydrothermal**:

- Storage bounds: physical capacity
- Generation bounds: turbine limits
- Flow bounds: network capacity

**Implementation Priority**: **LOW** (probably already well-bounded)

### 8.4 Iterative Tolerance Tightening

Instead of fixed tolerances, start loose and tighten:

- Iteration 1-10: tol = 1e-5
- Iteration 11-50: tol = 1e-6
- Iteration 51+: tol = 1e-7

**Benefit**: Early iterations are faster, accuracy when needed

**Implementation Priority**: **LOW**  
**Effort**: Low

## 9. Practical Enhancements

### 9.1 Stopping Rules

**Current**: Fixed iteration count

**Enhancements**:

- **Statistical stopping**: `(UB - LB) / LB < ε` with confidence
- **Stalling detection**: Bound hasn't improved for N iterations
- **Time limit**: Wall-clock time
- **Combined**: Logical OR/AND of multiple rules

**Implementation Priority**: **HIGH**  
**Effort**: Low (1 week)

### 9.2 Checkpointing and Restart

Save algorithm state:

- All cuts
- Visited states
- Iteration count
- Lower bound history

Enables:

- Resuming interrupted training
- Trying different stopping criteria
- Warm-starting related problems

**Implementation Priority**: **MEDIUM**  
**Effort**: Moderate (2 weeks)

### 9.3 Diagnostic Tools

**Convergence diagnostics**:

- Plot lower bound vs. iteration
- Plot upper bound (simulation) vs. iteration
- Optimality gap over time

**Policy diagnostics**:

- Spaghetti plots of simulated trajectories
- State space heatmaps (which states visited)
- Value function visualization

**Cut diagnostics**:

- Number of active cuts per node
- Cut coefficient distribution
- Dominance relationships

**Implementation Priority**: **LOW to MEDIUM**  
**Effort**: Moderate (ongoing)

### 9.4 Warm Starting from Prior Solutions

Given cuts from previous run:

- Load cuts
- Optionally run simulation to test quality
- Continue training with new data

**Use cases**:

- Daily operations (yesterday's policy as starting point)
- Model updates (add new scenarios, retrain)
- Sensitivity analysis (change parameters, retrain)

**Implementation Priority**: **MEDIUM**  
**Effort**: Moderate (depends on cut serialization)

## 10. Research Frontiers (High-Risk, High-Reward)

### 10.1 Deep Learning for Value Function Approximation

Replace cuts with neural network:

```
V_t(x) ≈ NN_θ(x)
```

**Potential advantages**:

- Handles high-dimensional states
- Smooth approximation
- Transfer learning possible

**Challenges**:

- No convergence guarantees
- Requires many samples
- Difficult to train

**Status**: Active research, not production-ready

### 10.2 Quantum Computing for Subproblem Solutions

Solve LP subproblems on quantum computer.

**Reality check**:

- Current quantum computers too small
- Quantum advantage unclear for LP
- Decades away from practical use

**Status**: Speculative research

### 10.3 Hybrid SDDP-RL Algorithms

Combine strengths:

- SDDP for early iterations (fast, guaranteed progress)
- RL for fine-tuning (exploration, nonlinear policies)

**Status**: Emerging research area

## Priority Matrix for POWE.RS

### Immediate High-Value Additions (Next 3-6 Months)

| Feature             | Priority | Effort   | Impact | Dependencies |
| ------------------- | -------- | -------- | ------ | ------------ |
| Multi-Cut Variant   | HIGH     | Moderate | High   | None         |
| CVaR Risk Measure   | HIGH     | Moderate | High   | None         |
| Automated Scaling   | HIGH     | Low      | Medium | None         |
| Stopping Rules      | HIGH     | Low      | Medium | None         |
| Comprehensive Tests | HIGH     | Moderate | High   | None         |
| Benchmarking Suite  | HIGH     | Low      | Medium | None         |

### Medium-Term Enhancements (6-12 Months)

| Feature                    | Priority | Effort   | Impact | Dependencies              |
| -------------------------- | -------- | -------- | ------ | ------------------------- |
| Cut Serialization          | MEDIUM   | Moderate | Medium | None                      |
| DRO Risk Measures          | MEDIUM   | Moderate | Medium | Risk measures             |
| Out-of-Sample Sampling     | MEDIUM   | Low      | Low    | None                      |
| Regularized Forward Pass   | MEDIUM   | Low      | Low    | None                      |
| Risk-Adjusted Forward Pass | MEDIUM   | Moderate | Medium | Risk measures             |
| Distributed Parallel       | MEDIUM   | High     | High   | MPI/distributed framework |

### Long-Term / Research (12+ Months)

| Feature                       | Priority | Effort    | Impact | Dependencies      |
| ----------------------------- | -------- | --------- | ------ | ----------------- |
| SDDiP (Integer Variables)     | MEDIUM   | Very High | High   | Application need  |
| Markovian Graphs              | LOW      | Moderate  | Medium | None              |
| Stagewise-Dependent Objective | LOW      | Moderate  | Low    | Objective states  |
| Transfer Learning             | LOW      | Moderate  | Medium | Cut serialization |

## Recommended Roadmap

### Phase 1: Core Algorithm Completeness (3 months)

**Goal**: Feature parity with basic SDDP.jl capabilities

1. **Multi-cut implementation** (3 weeks)

   - Add local theta variables
   - Modify backward pass
   - Benchmark performance

2. **Risk measures** (3 weeks)

   - Implement CVaR/AVaR
   - Add worst-case
   - Add convex combinations
   - Test convergence

3. **Testing and validation** (3 weeks)

   - Unit tests for all components
   - Numerical validation tests
   - Benchmark problem suite

4. **Stopping rules and diagnostics** (1 week)
   - Statistical stopping
   - Stalling detection
   - Convergence plots

### Phase 2: Production Hardening (3 months)

**Goal**: Robust, production-ready system

5. **Numerical enhancements** (2 weeks)

   - Automatic scaling
   - Cut coefficient management
   - Extended solver retry strategies

6. **Checkpointing** (2 weeks)

   - Save/load cuts
   - Resume training
   - Policy serialization

7. **Performance optimization** (4 weeks)

   - Profile-guided optimization
   - Memory layout improvements
   - Advanced basis selection

8. **Documentation** (2 weeks)
   - API documentation
   - User guide
   - Theory background

### Phase 3: Advanced Features (6 months)

**Goal**: State-of-the-art capabilities

9. **Distributed execution** (6 weeks)

   - MPI-based parallelism
   - Asynchronous cut sharing
   - Load balancing

10. **Advanced risk measures** (4 weeks)

    - DRO (Modified Chi-Squared)
    - Entropic risk
    - Custom risk measures

11. **Advanced sampling** (4 weeks)

    - Out-of-sample Monte Carlo
    - Risk-adjusted forward pass
    - Importance sampling

12. **Policy graph extensions** (4 weeks)
    - Markovian graphs
    - General DAG support

### Phase 4: Specialized Applications (Ongoing)

**Goal**: Domain-specific enhancements

13. **Hydrothermal-specific**

    - Advanced inflow models (AR, Markov)
    - Cascading reservoir modeling
    - Risk-averse operations

14. **Other applications**
    - Supply chain problems
    - Portfolio optimization
    - Network design

## Conclusion

POWE.RS has a solid foundation and excellent performance characteristics. The highest-value improvements for most applications are:

1. **Multi-cut variant**: 2-5x faster convergence for many problems
2. **Risk measures**: Essential for risk-averse decision-making
3. **Testing and validation**: Critical for production use
4. **Checkpointing**: Enables warm-starting and interrupted training

The current architecture can accommodate most enhancements with moderate effort. The codebase is well-positioned to evolve into a state-of-the-art SDDP implementation while maintaining its performance advantages.

Longer-term, distributed parallelism and integer variable support would expand POWE.RS's applicability to larger and more complex problems. However, these are significant undertakings that should only be pursued when needed for specific applications.

The field of multistage stochastic optimization continues to evolve, with connections to reinforcement learning and machine learning providing new research directions. While these are exciting, the classical SDDP enhancements documented here represent proven, production-ready techniques that should be prioritized.
