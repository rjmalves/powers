# SDDP Mathematical Foundations

## Overview

Stochastic Dual Dynamic Programming (SDDP) is a decomposition algorithm for solving large-scale multistage stochastic programming problems. It was originally introduced by Pereira and Pinto (1991) for hydrothermal dispatch optimization and has since become one of the most successful algorithms for this class of problems.

## Problem Formulation

### Multistage Stochastic Linear Program

SDDP solves problems of the following form:

```
min E[Σ_{t=1}^T c_t^T x_t(ω)]
s.t. A_t x_t(ω) + B_t x_{t-1}(ω) = b_t(ω),  ∀t ∈ {1,...,T}, ω ∈ Ω
     x_t(ω) ≥ 0,                              ∀t ∈ {1,...,T}, ω ∈ Ω
     x_0 given
```

Where:

- `T` is the number of stages (time periods)
- `x_t` is the decision vector at stage t
- `c_t` is the cost vector at stage t
- `A_t, B_t` are constraint matrices
- `b_t(ω)` is the right-hand side depending on uncertainty realization ω
- `Ω` is the set of all possible uncertainty realizations
- `E[·]` denotes expected value

### Stagewise Independence

A critical assumption in SDDP is **stagewise independence**: the uncertainty at stage t depends only on the state at the beginning of stage t, not on the entire history. This allows the problem to be reformulated as:

```
V_1(x_0) = min c_1^T x_1 + E_{ω_1}[V_2(x_1, ω_1)]
s.t. A_1 x_1 = b_1 - B_1 x_0
     x_1 ≥ 0
```

Where `V_t(x_{t-1}, ω)` is the **cost-to-go function** (or value function) representing the optimal expected cost from stage t onwards, given state `x_{t-1}` and uncertainty realization `ω`.

## Dynamic Programming Decomposition

### Bellman's Equation

For each stage t, the optimal policy satisfies Bellman's equation:

```
V_t(x_{t-1}, ω_t) = min{c_t^T x_t + E_{ω_{t+1}}[V_{t+1}(x_t, ω_{t+1})] :
                        A_t x_t + B_t x_{t-1} = b_t(ω_t), x_t ≥ 0}
```

With terminal condition `V_{T+1}(x_T) = 0`.

### Convexity of the Value Function

Under the assumption of:

1. Linear constraints
2. Stagewise independent uncertainty
3. Relatively complete recourse (feasibility for all states)

The value function `V_t(x_{t-1})` is **convex** in the state variable `x_{t-1}`. This convexity is fundamental to SDDP's cutting plane approximation.

## Benders Decomposition

SDDP is essentially a stochastic extension of Benders decomposition applied recursively across time stages.

### Dual Variables and Cuts

For a given state `x_{t-1}^k` visited in iteration k, after solving the subproblem at stage t, we obtain:

- Optimal dual variables `π_t^k` (shadow prices of state constraints)
- Optimal objective value `θ_t^k`

These define a **Benders cut** (also called a **cutting plane** or simply **cut**):

```
V_t(x_{t-1}) ≥ θ_t^k + π_t^k^T (x_{t-1} - x_{t-1}^k)
```

Rearranging:

```
V_t(x_{t-1}) ≥ α_t^k + β_t^k^T x_{t-1}
```

Where:

- `α_t^k = θ_t^k - π_t^k^T x_{t-1}^k` (intercept)
- `β_t^k = π_t^k` (slope/gradient)

### Polyhedral Approximation

After K iterations, the value function is approximated by:

```
V_t(x_{t-1}) ≈ max{α_t^k + β_t^k^T x_{t-1} : k = 1,...,K}
```

For minimization problems, we use a lower bounding approximation. As K → ∞, this piecewise linear approximation converges to the true convex value function.

## SDDP Algorithm Structure

### Two-Phase Iteration

Each SDDP iteration consists of two phases:

#### 1. Forward Pass

- Start from the initial state `x_0`
- For t = 1 to T:
  - Sample uncertainty `ω_t` from its distribution
  - Solve the subproblem at stage t with current cut approximation
  - Record the solution `x_t^k` and objective value
  - Move to next stage with state `x_t^k`
- Calculate total cost along the sampled trajectory

#### 2. Backward Pass

- Start from the final stage T
- For t = T down to 1:
  - For each uncertainty realization `ω_{t+1}`:
    - Solve subproblem at stage t+1 with state `x_t^k` from forward pass
    - Extract dual variables `π_{t+1}^k`
    - Compute expected future cost
  - Create and add Benders cut to stage t approximation
  - Update lower bound estimate

### Cut Aggregation Strategies

There are two main approaches for creating cuts:

#### Single-Cut (Average-Cut)

- Aggregate information from all scenarios at stage t+1
- Create one cut per iteration per stage
- Cut uses expected dual variables:
  ```
  β_t^k = E_{ω_{t+1}}[π_{t+1}^k(ω_{t+1})]
  α_t^k = E_{ω_{t+1}}[θ_{t+1}^k(ω_{t+1})] - β_t^k^T x_t^k
  ```
- **Advantages**: Fewer constraints, more stable numerically
- **Disadvantages**: Slower convergence with many scenarios

#### Multi-Cut

- Create one cut per scenario at stage t+1
- Represents cost-to-go as:
  ```
  V_t(x_t) = Σ_ω p_ω · φ_ω(x_t)
  ```
  Where each `φ_ω` is separately approximated
- **Advantages**: Faster convergence, more accurate representation
- **Disadvantages**: More constraints, potential numerical issues

## Convergence Theory

### Lower Bound

The backward pass provides a **lower bound** on the optimal objective:

```
LB^k = E[V_1^k(x_0)]
```

This bound is non-decreasing: `LB^1 ≤ LB^2 ≤ ... ≤ LB^* = V_1^*(x_0)`

### Upper Bound

The forward pass provides an **upper bound** (unbiased estimator) of the optimal objective:

```
UB^k = Σ_{t=1}^T c_t^T x_t^k(ω^k)
```

Where `ω^k` is the sampled scenario in iteration k.

### Statistical Upper Bound

Since forward pass costs are random, we estimate:

```
UB = (1/N) Σ_{i=1}^N UB^i
```

With confidence interval based on sample standard deviation.

### Convergence Criterion

The algorithm converges when:

```
(UB - LB) / |UB| < ε
```

Or equivalently, when the optimality gap is sufficiently small.

### Convergence Rate

Under stagewise independence and bounded noise, SDDP converges:

- **Almost surely** to the optimal policy
- With **geometric rate** in the expected optimality gap (under certain conditions)

However, the number of iterations can grow exponentially with the number of stages in the worst case.

## Mathematical Properties

### Convexity Requirements

SDDP requires:

1. **Convex cost functions**: Linear or convex objective
2. **Convex feasible region**: Linear constraints (or convex constraints with special handling)
3. **Convex value functions**: Ensured by convexity of subproblems

### Relatively Complete Recourse

For every state visited, there must exist a feasible solution in the next stage. This ensures:

- Value functions are finite
- Algorithm doesn't encounter infeasibilities
- Cuts are valid

If recourse is incomplete, penalty terms (deficit variables) are typically added.

### Subgradient Property

The dual variables `π_t` satisfy:

```
π_t ∈ ∂V_t(x_{t-1})
```

That is, they are subgradients of the value function at the visited state. This is the key insight that makes the Benders cuts valid outer approximations.

## State Variable Selection

In hydrothermal dispatch (POWE.RS application):

### Common State Variables

- **Reservoir storage volumes**: Most important state
- **Thermal unit commitment**: Binary states (if considering unit commitment)
- **Line flows**: For multi-area systems with transmission limits

### State Space Dimensionality

- SDDP's effectiveness degrades with state dimension (curse of dimensionality)
- Practical limit: ~10-50 continuous states
- Binary states add significant complexity

### State Aggregation

To manage dimensionality:

- Aggregate reservoirs by cascade or region
- Use representative states
- Employ state space discretization techniques

## Risk Aversion

### Risk-Neutral SDDP

Standard SDDP minimizes expected cost:

```
E_{ω}[Σ_t c_t^T x_t(ω)]
```

### Risk-Averse SDDP

Replace expectation with a **coherent risk measure** ρ:

```
ρ_{ω}[Σ_t c_t^T x_t(ω)]
```

Common risk measures:

- **CVaR (Conditional Value at Risk)**: Expected cost in worst α% of scenarios
- **AV@R (Average Value at Risk)**: Synonym for CVaR
- **Worst-case**: Maximum over all scenarios
- **Convex combinations**: λ·E[·] + (1-λ)·CVaR_α[·]

### Risk Measure Implementation

Risk aversion is implemented by:

1. Modifying probability weights in the backward pass
2. Using dual representation of coherent risk measures
3. Adding "risk cuts" alongside standard Benders cuts

The mathematical framework extends naturally because coherent risk measures preserve convexity.

## Numerical Considerations

### Scaling

- States should be similarly scaled (e.g., normalized to [0,1])
- Objective coefficients should have similar magnitudes
- Poor scaling leads to numerical instability

### Tolerances

- Primal feasibility tolerance: typically 10^-6 to 10^-7
- Dual feasibility tolerance: typically 10^-6 to 10^-7
- Convergence tolerance: typically 0.1% to 1%

### Cut Selection

As iterations progress, the number of cuts grows linearly. To manage this:

- **Cut deletion**: Remove cuts that are never active
- **Cut aggregation**: Combine similar cuts
- **Cut selection**: Keep only the most relevant cuts

Common strategies (inspired by SDDP.jl):

- Track cut activity: remove cuts not active for many iterations
- Level method: prefer cuts that define the value function at visited states
- Dominance: a cut dominates if it's always higher at all visited states

## Extensions and Variants

### Stochastic Dual Dynamic Integer Programming (SDDiP)

- Handles integer decision variables
- Uses Lagrangian relaxation and Benders decomposition
- Significantly more complex than continuous SDDP

### Markovian SDDP

- Relaxes stagewise independence
- Uncertainty follows a Markov chain
- State includes both physical state and Markov state

### Cyclic SDDP

- For infinite-horizon problems with seasonal cycles
- Applies period-by-period with wrap-around
- Convergence to cyclic steady-state policy

### Distributed SDDP

- Parallelize forward passes (embarrassingly parallel)
- Parallelize scenario subproblems in backward pass
- Requires synchronization of cut information

## Key Theoretical Results

### Finite Convergence (Shapiro, 2011)

For problems with:

- Finite scenario tree
- Compact state space
- Linear value functions

SDDP converges in finite iterations.

### Convergence Rate (Guigues & Römisch, 2012)

Under regularity conditions:

- SDDP converges with rate O(1/√K) in expected value
- Faster rates possible with specialized sampling schemes

### Sample Complexity (Philpott & Guan, 2008)

Number of scenarios needed per stage scales with:

- Dimension of uncertainty space
- Required accuracy
- Stage-dependent variance

Practical guidance: 10-100 scenarios per stage often sufficient.

## References

### Foundational Papers

1. Pereira, M. V. F., & Pinto, L. M. V. G. (1991). Multi-stage stochastic optimization applied to energy planning. _Mathematical Programming_, 52(1-3), 359-375.

2. Shapiro, A. (2011). Analysis of stochastic dual dynamic programming method. _European Journal of Operational Research_, 209(1), 63-72.

3. Philpott, A., & de Matos, V. (2012). Dynamic sampling algorithms for multi-stage stochastic programs with risk aversion. _European Journal of Operational Research_, 218(2), 470-483.

### Modern Extensions

4. Downward, A., Dowson, O., & Baucke, R. (2020). Stochastic dual dynamic programming with stagewise-dependent objective uncertainty. _Operations Research Letters_, 48(1), 33-39.

5. Dowson, O., & Morton, D. P., & Pagnoncelli, B. K. (2022). Incorporating convex risk measures into multistage stochastic programming algorithms. _Annals of Operations Research_.

### Practical Implementations

6. SDDP.jl: https://github.com/odow/SDDP.jl (State-of-the-art Julia implementation)
7. Dowson, O., & Kapelevich, L. (2020). SDDP.jl: a Julia package for stochastic dual dynamic programming. _INFORMS Journal on Computing_, 33(1), 27-33.
