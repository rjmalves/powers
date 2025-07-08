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

The main product of this algorithm is a decision-making policy in the form of Benders' Cuts, that are inserted to the optimization problem in the form of constraints. Each iteration produces a new cut for each stage, except for the last one. This is called the `single-cut` or `average-cut` variant of the algorithm. In a scenario that supports parallel computing, each iteration may produce N cuts, where N is the number of simultaneous forward passes.

### Performance

In exchange for the relavively simple system, some optimizations to the algorithm itself were made. First, the number of memory allocations was minimized when interacting with the underlying solver, [HiGHS](https://github.com/ERGO-Code/HiGHS/). Therefore, the optimization problem is converted in the model form as a pre-processing step in the policy graph construction, and this model is edited through the iterations.

Also, instead of using the more common [highs](https://docs.rs/highs/latest/highs/) crate for interacting with the solver, a different interface was built using the [highs-sys](https://crates.io/crates/highs-sys) crate, which contains the result of applying [bindgen](https://github.com/rust-lang/rust-bindgen) to the solver repository. The developed interface is highly based on the [highs](https://docs.rs/highs/latest/highs/) crate, but differs in some aspects that affected the SDDP in a relevant way.

Given the nature of the `backward` step, it is expected that some info of the solver state in the `forward` step can help improving the solution process. This is mainly known as `basis reuse` and is implemented in `powers` by storing the basis of each solved problem in the `forward` step and initializing each solved problem in the `backward` step, for the same node, with the stored basis.

Also, when the SDDP algorithm continues for a large number of iterations, the number of cuts (which turns into constraints) begins to hurt the performance. For this case, a `cut selection` strategy was implemented, highly based on the existing one from [SDDP.jl](https://github.com/odow/SDDP.jl), which is inspired in [de Matos, Philpott & Finardi 2015](https://www.sciencedirect.com/science/article/pii/S0377042715002794).

For handling slightly larger problems, it is common for the solver to suffer from numerical issues. Therefore, the `solve` calls consist of a up-to-3 retry steps, which change the solver options in order to continue the iterative process instead of stopping the algorithm with an error state.

Currently there is support for thread-based parallelism, which is capped on the number of logical cores of the running machine. During training, the number of forward passes also limit the parallelism level. During the simulation step, the number of simulated scenarios also defines the maximum number of simultaneous threads. For handling these parallel steps, the [rayon](https://docs.rs/rayon/latest/rayon/) crate is used.

### Dependencies

This implementation was made aiming to minimize the external dependencies whenever possible. The key crates on which it depends are:

1. [highs-sys](https://crates.io/crates/highs-sys): the low-level interface with the HiGHS solver, which is mainly an application of [bindgen](https://github.com/rust-lang/rust-bindgen) to the C-API.
2. [rand](https://docs.rs/rand/latest/rand/), [rand_distr](https://docs.rs/rand_distr/latest/rand_distr/) and [rand_xoshiro](https://docs.rs/rand_xoshiro/latest/rand_xoshiro/): random number generation and probability distributions utilities for the scenario generation and inflow sampling processes.
3. [serde](https://docs.rs/serde/latest/serde/), [serde_json](https://docs.rs/serde_json/latest/serde_json/) and [csv](https://docs.rs/csv/latest/csv/): serializing and deserializing utilities for handling data input and output.
4. [rayon](https://docs.rs/rayon/latest/rayon/): implement parallel iterators for the training and simulation steps.

## How-to and Input Data

### Installing pre-built binaries

Pre-built binaries are available on each release page, for downloading on Linux and Mac architectures. An installation via `curl` is also possible through

```
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/rjmalves/powers/releases/download/<VERSION>/powers-rs-installer.sh | sh
```

where the `<VERSION>` must be replaced by the desired tag, such as `v0.1.1`.

### Building dependencies

Since this crate compiles the solver itself, which includes a whole C++ project, some system dependencies are required. In Ubuntu, the installation can be done with

```
sudo apt update
sudo apt install libclang-dev build-essential cmake
```

### Building locally

The code can be downloaded from the repository and built locally for usage with

```
git clone https://github.com/rjmalves/powers.git
cd powers
cargo build --release
```

### Installing from crates.io

The executable can be installed without cloning the repository with

```
cargo install powers-rs
```

The last line displays the name of the executable to be called, which is `powers` itself.

### Running

Considering the local build scenario, the built executable will be available in the `target/release` path and can be called, resulting in:

```
$ target/release/powers
Problem parsing arguments: Not enough arguments [PATH]
```

This happens because `powers` expect the path with input data as the single argument. Calling with the example data from the repository does:

```
$ target/release/powers example

POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'example'

# Training
- Iterations: 32
- Forward passes: 4

------------------------------------------------------------
iteration  | lower bound ($) | simulation ($) |   time (s)
------------------------------------------------------------
         1 |        176.3589 |      8449.5644 |         0.02
         2 |       2472.6769 |      3359.3435 |         0.02
         3 |       2589.7422 |      4598.5679 |         0.02
         4 |       2777.7344 |      3442.4727 |         0.02
         5 |       2839.5472 |      2454.7594 |         0.02
...
        31 |       3502.7809 |      3646.8279 |         0.03
        32 |       3505.8279 |      3418.2909 |         0.04
------------------------------------------------------------

Training time: 0.96 s

Number of constructed cuts by node: 128

# Simulating
- Scenarios: 128

Expected cost ($): 5230.27 +- 2286.20

Simulation time: 0.35 s

Writing outputs to 'example'

Total running time: 1.37 s
```

### Input Data

Currently, the input data supported by `powers` consists of three `JSON` files:

1. `config.json`: parameters of the SDDP algorithm itself

```json
{
  "num_iterations": 1024,
  "num_stages": 12,
  "num_branchings": 10,
  "num_simulation_scenarios": 1000
}
```

2. `system.json`: definition of the power system underlying the optimization: buses, lines, thermals and hydros.

```json
{
  "buses": [
    {
      "id": 0,
      "deficit_cost": 50.0
    },
    {
      "id": 1,
      "deficit_cost": 50.0
    }
  ],
  "lines": [
    {
      "id": 0,
      "source_bus_id": 0,
      "target_bus_id": 1,
      "direct_capacity": 100.0,
      "reverse_capacity": 50.0,
      "exchange_penalty": 0.005
    }
  ],
  "thermals": [
    {
      "id": 0,
      "bus_id": 0,
      "cost": 5.0,
      "min_generation": 0.0,
      "max_generation": 15.0
    }
  ],
  "hydros": [
    {
      "id": 0,
      "downstream_hydro_id": null,
      "bus_id": 0,
      "productivity": 1.0,
      "min_storage": 0.0,
      "max_storage": 100.0,
      "min_turbined_flow": 0.0,
      "max_turbined_flow": 60.0,
      "spillage_penalty": 0.01
    }
  ]
}
```

3. `recourse.json`: the avaliable resource for the decision-making process, namely the initial state, bus loads and hydro inflows.

```json
{
  "initial_states": [
    {
      "hydro_id": 0,
      "initial_storage": 83.222
    }
  ],
  "loads": [
    {
      "bus_id": 0,
      "value": 50.0
    },
    {
      "bus_id": 1,
      "value": 25.0
    }
  ],
  "inflow_distributions": [
    {
      "hydro_id": 0,
      "lognormal": {
        "mu": 3.6,
        "sigma": 0.6928
      }
    }
  ]
}
```

## Output Data and Analysis

### Execution steps

Running the `powers` executable consists of two steps, which produces different outputs, always in `CSV` format:

1. `train`: the construction of the policy (cuts). This step generates two files: `cuts.csv` and `states.csv`.
2. `simulation`: the evaluation of the policy on different scenarios sampled from the same distributions. This step generates a file for each of the system entities: `simulation_buses.csv`, `simulation_lines.csv`, `simulation_thermals.csv` and `simulation_hydros.csv`.

### Output files

The content of each output file is described below:

#### `cuts.csv`

Containts the Benders' cuts evaluated during the training step. The stages are integers starting from 0 and inside each stage, the cuts are identified also by incremental integers from 0. The cuts contain an RHS term and a multiplier for the storage of each entity. An additional `active` column exists for indicating the result of the cut selection process.

```csv
stage_index, stage_cut_id, active, coefficient_entity, value
          0,            0, false , RHS               , 3355.648056129926
          0,            0, false , 0                 , -23.687301599999994
          0,            1, false , RHS               , 3452.9361506313066
          0,            1, false , 0                 , -20.677596440000002
          0,            2, false , RHS               , 3048.1833071547762
          0,            2, false , 0                 , -15.668791996000001
```

#### `states.csv`

Containts the states sampled during the training step. The stages are integers starting from 0 and inside each stage, each state contain the dominating objective value among all cuts and the storage of each entity that compose the state variables.

```csv
stage_index, dominating_cut_id, coefficient_entity , value
          0,               732, DominatingObjective, 2742.876633160772
          0,               732, 0                  ,   79.84781266283366
          0,              1012, DominatingObjective, 3218.4639098212483
          0,              1012, 0                  ,   58.05526348463894
          0,              1018, DominatingObjective, 2896.4256667197533
          0,              1018, 0                  ,   72.51990252194832
```

#### `simulation_buses.csv`

Containts the simulation results for the variables of each `Bus` defined in the problem.

```csv
stage_index, entity_index, load, deficit              , marginal_cost
          0,            0, 75.0,  0.0                 , 10.0
          1,            0, 75.0,  0.0                 ,  5.0
          2,            0, 75.0,  0.0                 ,  5.0
          3,            0, 75.0,  0.0                 , 12.7562448279
          4,            0, 75.0,  0.0                 , 21.534882317999998
          5,            0, 75.0,  0.0                 ,  5.0
```

#### `simulation_lines.csv`

Containts the simulation results for the variables of each `Line` defined in the problem.

```csv
stage_index, entity_index, exchange
          0,            0, 25.0
          1,            0, 25.0
          2,            0, 25.0
          3,            0, 25.0
          4,            0, 25.0
          5,            0, 25.0
```

#### `simulation_thermals.csv`

Containts the simulation results for the variables of each `Thermal` defined in the problem.

```csv
stage_index, entity_index, generation
          0,            0, 15.0
          0,            1,  5.639951767715928
          1,            0, 15.0
          1,            1,  0.0
          2,            0, 15.0
          2,            1,  0.0
          3,            0, 15.0
          3,            1, 15.0
```

#### `simulation_hydros.csv`

Containts the simulation results for the variables of each `Hydro` defined in the problem.

```csv
stage_index, entity_index, initial_storage       , final_storage         , inflow              , turbined_flow     , spillage               , water_value
          0,            0,  83.222               , 100.0                 ,  71.13804823228408  , 54.36004823228407 ,   0.0                  , -10.0
          1,            0, 100.0                 , 100.0                 ,  78.71364512720558  , 60.0              ,  18.71364512720558     , 0.01
          2,            0, 100.0                 , 100.0                 ,  61.34416007251395  , 60.0              ,   1.3441600725139438   , 0.01
          3,            0, 100.0                 ,  71.79287694552265    ,  16.792876945522647 , 45.0              ,   0.0                  , -12.7562448279
          4,            0,  71.79287694552265    ,  43.27478882572022    ,  16.481911880197575 , 45.0              ,   0.0                  , -21.534882317999998
          5,            0,  43.27478882572022    , 100.0                 , 130.48881003728116  , 60.0              ,  13.763598863001391    , 0.01

```

## Contributing

Contributions are welcome! The formatting should follow the default cargo linter with the `rustfmt.toml` file from the repository and the test routine is done also with the cargo test suite.
