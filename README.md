# POWE.RS - Power Optimization for the World of Energy - in pure RuSt

An implementation of the Stochastic Dual Dynamic Programming (SDDP) algorithm in pure Rust, for the hydrothermal dispatch problem.

## Introduction

This repository contains a minimal implementation (methodologically speaking) of the SDDP for the hydrothermal dispatch problem. Although some modeling aspects could be improved, like considering autoregressive models for the hydro inflows on scenario generation and as state variables, the scope of this project goes in another direction.

Instead of obtaining the best model for the physical system, the focus is on implementing the algorithm in a decent way using the Rust programming language, which introduces concepts like ownership and borrowing in exchange for the memory safety.

This code is the result of a learning path on Rust that aimed to close the gap between how performant code for the SDDP algorithm is done in languages like C++, that gives the developer more freedom to mess with memory in exchange for performance, and how performant code for the SDDP algorithm can be done in Rust.

## Main features

### The `powers` system

The power system considered in powers is composed of four basic entities:

1. Buses
2. Lines
3. Thermals
4. Hydros

Each bus will receive some demand, which could vary depending on the stage or scenario, and should be met either by thermal generation, hydro generation, exchange or deficit, which is penalized by a given value for each bus. The lines allow buses to exchange power, with given limits on each direction. Thermals generate power with a given constant cost to the bus they are connected and hydros are able to store and generate power when the evaluated policy decides chooses to.

The `productivity` of each hydro is considered to be constant, for simplicity, and given by the user. Also, the thermal generation, hydro storage and hydro turbined flow can be bounded in the system input data, but only with a single value for the entire study. Defining more complicated constraints, such as seasonal bounds, is a future work. However, hydro cascades can be defined via the `downstream_hydro_id` attribute.

### The `powers` algorithm

The implemented algorithm is the classic SDDP from [Pereira & Pinto, 1991](https://link.springer.com/article/10.1007/BF01582895), in the sense that an Sample Average Approximation (SAA) is made for obtaining a number of inflows from a LogNormal distribution that can be parameterized by the user, for each hydro. These inflows are sampled on each iteration, which are comprised of a `forward` step, that visits viable states, and a `backward` step, that refines the policy.

The main product of this algorithm is a decision-making policy in the form of Benders' Cuts, that are inserted to the optimization problem in the form of constraints. Each iteration produces a new cut for each stage, except for the last one. This is called the `single-cut` or `average-cut` variant of the algorithm. In a scenario that supports parallel computing, each iteration may produce N cuts, where N is the number of allocated threads. Currently, only the serial computing is supported.

### Performance

In exchange for the relavively simple system, some optimizations to the algorithm itself were made. First, the number of memory allocations was minimized when interacting with the underlying solver, [HiGHS](https://github.com/ERGO-Code/HiGHS/). Therefore, the optimization problem is converted in the model form as a pre-processing step in the policy graph construction, and this model is edited through the iterations.

Also, instead of using the more common [highs](https://docs.rs/highs/latest/highs/) crate for interacting with the solver, a different interface was built using the [highs-sys](https://crates.io/crates/highs-sys) crate, which contains the result of applying [bindgen](https://github.com/rust-lang/rust-bindgen) to the solver repository. The developed interface is highly based on the [highs](https://docs.rs/highs/latest/highs/) crate, but differs in some aspects that affected the SDDP in a relevant way.

Given the nature of the `backward` step, it is expected that some info of the solver state in the `forward` step can help improving the solution process. This is mainly known as `basis reuse` and is implemented in `powers` by storing the basis of each solved problem in the `forward` step and initializing each solved problem in the `backward` step, for the same node, with the stored basis.

Also, when the SDDP algorithm continues for a large number of iterations, the number of cuts (which turns into constraints) begins to hurt the performance. For this case, a `cut selection` strategy was implemented, highly based on the existing one from [SDDP.jl](https://github.com/odow/SDDP.jl).

For handling slightly larger problems, it is common for the solver to suffer from numerical issues. Therefore, the `solve` calls consist of a up-to-4 retry steps, which change the solver options in order to continue the iterative process instead of stopping the algorithm with an error state.

### Dependencies

TODO

## How-to and Input Data

TODO

## Output Data and Analysis

TODO

## Example

TODO

## Contributing

TODO
