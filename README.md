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
         1 |        150.0000 |      8449.5644 |         0.01
         2 |       1934.4894 |      2982.8026 |         0.01
         3 |       2589.7422 |      4579.9037 |         0.01
         4 |       2747.8503 |      3439.1260 |         0.01
         5 |       2794.7946 |      2447.6383 |         0.01
         6 |       3110.8416 |      3631.8372 |         0.01
         7 |       3215.8914 |      5510.5784 |         0.01
         8 |       3226.9871 |      4692.5430 |         0.01
         9 |       3250.5484 |      3828.3339 |         0.01
        10 |       3287.3970 |      3920.3616 |         0.01
        11 |       3340.4140 |      3102.0091 |         0.01
        12 |       3360.4725 |      3947.1356 |         0.01
        13 |       3362.2850 |      2859.0078 |         0.01
        14 |       3371.9111 |      3490.0555 |         0.01
        15 |       3385.1120 |      3052.7957 |         0.01
        16 |       3392.3852 |      2876.4624 |         0.01
        17 |       3398.9754 |      3237.7560 |         0.01
        18 |       3413.1219 |      2100.9523 |         0.01
        19 |       3422.0440 |      3444.9446 |         0.01
        20 |       3450.6767 |      4584.8400 |         0.01
        21 |       3458.3155 |      2705.4180 |         0.01
        22 |       3463.8145 |      3583.3492 |         0.01
        23 |       3467.2819 |      2288.7754 |         0.01
        24 |       3469.1923 |      2917.0320 |         0.01
        25 |       3474.3462 |      3195.0329 |         0.01
        26 |       3478.9393 |      3516.4845 |         0.01
        27 |       3490.3469 |      3336.5466 |         0.01
        28 |       3492.7897 |      2846.6356 |         0.01
        29 |       3496.2051 |      2771.0824 |         0.01
        30 |       3499.6501 |      3368.4054 |         0.01
        31 |       3502.5246 |      3646.8279 |         0.01
        32 |       3505.5237 |      3418.2909 |         0.01
------------------------------------------------------------

Training time: 0.29 s

Number of constructed cuts by node: 128

# Simulating
- Scenarios: 128

Expected cost ($): 5230.27 +- 2286.20

Simulation time: 0.08 s

Writing outputs to 'example'

Total running time: 0.38 s
```

### Input Data

Currently, the input data supported by `powers` consists of three `JSON` files:

1. `config.json`: parameters of the SDDP algorithm itself

```json
{
  "num_iterations": 32,
  "num_forward_passes": 4,
  "num_simulation_scenarios": 128,
  "seed": 0
}
```

2. `system.json`: definition of the power system underlying the optimization: buses, lines, thermals and hydros.

```json
{
  "buses": [
    {
      "id": 0,
      "deficit_cost": 50.0
    }
  ],
  "lines": [],
  "thermals": [
    {
      "id": 0,
      "bus_id": 0,
      "cost": 5.0,
      "min_generation": 0.0,
      "max_generation": 15.0
    },
    {
      "id": 1,
      "bus_id": 0,
      "cost": 10.0,
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
  "initial_condition": {
    "storage": [
      {
        "hydro_id": 0,
        "value": 83.222
      }
    ],
    "inflow": [
      {
        "hydro_id": 0,
        "lag": 1,
        "value": 50.0
      }
    ]
  },
  "uncertainties": [
    {
      "season_id": 0,
      "num_branchings": 10,
      "distributions": {
        "load": [
          {
            "bus_id": 0,
            "normal": {
              "mu": 75.0,
              "sigma": 0.0
            }
          }
        ],
        "inflow": [
          {
            "hydro_id": 0,
            "lognormal": {
              "mu": 3.6,
              "sigma": 0.6928
            }
          }
        ]
      }
    },
    // ...
    {
      "season_id": 11,
      "num_branchings": 10,
      "distributions": {
        "load": [
          {
            "bus_id": 0,
            "normal": {
              "mu": 75.0,
              "sigma": 0.0
            }
          }
        ],
        "inflow": [
          {
            "hydro_id": 0,
            "lognormal": {
              "mu": 3.6,
              "sigma": 0.6928
            }
          }
        ]
      }
    }
  ]
}
```

4. `graph.json`: the definition of the graph that models the stochastic decomposition problem, with the state definition, risk measure and stochastic processes of each stage.

```json
{
  "nodes": [
    {
      "id": 0,
      "stage_id": 0,
      "season_id": 0,
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-02-01T00:00:00Z",
      "risk_measure": "expectation",
      "load_stochastic_process": "naive",
      "inflow_stochastic_process": "naive",
      "state_variables": "storage"
    },
    // ...
    {
      "id": 11,
      "stage_id": 11,
      "season_id": 11,
      "start_date": "2024-11-01T00:00:00Z",
      "end_date": "2024-12-01T00:00:00Z",
      "risk_measure": "expectation",
      "load_stochastic_process": "naive",
      "inflow_stochastic_process": "naive",
      "state_variables": "storage"
    }
  ],
  "edges": [
    {
      "source_id": 0,
      "target_id": 1,
      "probability": 1.0,
      "discount_rate": 0.0
    },
    {
      "source_id": 10,
      "target_id": 11,
      "probability": 1.0,
      "discount_rate": 0.0
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
          0,            0, false , RHS               , 75.0
          0,            0, false , 0                 , 0.0
          0,            1, true  , RHS               , 556.3289550958654
          0,            1, true  , 0                 , -7.399799999999999
          0,            2, true  , RHS               , 982.7864174441443
          0,            2, true  , 0                 , -10.678600000000001
```

#### `states.csv`

Containts the states sampled during the training step. The stages are integers starting from 0 and inside each stage, each state contain the dominating objective value among all cuts and the storage of each entity that compose the state variables.

```csv
stage_index, dominating_cut_id, coefficient_entity , value
          0,                85, DominatingObjective, 4522.841996737085
          0,                85, 0                  , 20.529909871826362
          0,                85, DominatingObjective, 4439.327063667069
          0,                85, 0                  , 23.799050364700705
          0,                85, DominatingObjective, 4182.479610767903
          0,                85, 0                  , 33.85318533072352
```

#### `simulation_buses.csv`

Containts the simulation results for the variables of each `Bus` defined in the problem.

```csv
stage_index, series_index, entity_index, load, deficit             , marginal_cost
          0,            0,            0,  0.0,  0.0                ,           5.0
          1,            0,            0,  0.0,  0.0                ,           5.0
          2,            0,            0,  0.0,  0.0                ,           5.0
          3,            0,            0,  0.0,  7.460308786652931  ,          50.0
          4,            0,            0,  0.0,  0.0                ,          10.0
          5,            0,            0,  0.0, 22.497932081239966  ,          50.0
          6,            0,            0,  0.0,  0.0                ,           5.0
          7,            0,            0,  0.0,  3.5853689259062236 ,          50.0
          8,            0,            0,  0.0, 24.22378891424749   ,          50.0
          9,            0,            0,  0.0, 17.715927055535314  ,          50.0
         10,            0,            0,  0.0, 33.82354347703064   ,          50.0
         11,            0,            0,  0.0, 26.686549714044883  ,          50.0
```

#### `simulation_lines.csv`

Containts the simulation results for the variables of each `Line` defined in the problem.

```csv
stage_index, series_index, entity_index, exchange
          0,            0,            0, 25.0
          1,            0,            0, 25.0
          2,            0,            0, 25.0
          3,            0,            0, 25.0
          4,            0,            0, 25.0
          5,            0,            0, 25.0
```

#### `simulation_thermals.csv`

Containts the simulation results for the variables of each `Thermal` defined in the problem.

```csv
stage_index, series_index, entity_index, generation
          0,            0,            0, 15.0
          0,            0,            1,  0.0
          1,            0,            0, 15.0
          1,            0,            1,  0.0
          2,            0,            0, 15.0
          2,            0,            1,  0.0
          3,            0,            0, 15.0
          3,            0,            1, 15.0
          4,            0,            0, 15.0
          4,            0,            1,  3.9633896974920972
```

#### `simulation_hydros.csv`

Containts the simulation results for the variables of each `Hydro` defined in the problem.

```csv
stage_index, series_index, entity_index, final_storage        , inflow             , turbined_flow     , spillage            , water_value
          0,            0,            0,  42.82014469170315   ,  19.598144691703155, 60.0              ,   0.0               , -0.0
          1,            0,            0,  20.529909871826362  ,  37.70976518012321 , 60.0              ,   0.0               , -0.0
          2,            0,            0,  26.075621784681616  ,  65.54571191285525 , 60.0              ,   0.0               , -0.0
          3,            0,            0,   0.0                ,  11.464069428665454, 37.53969121334707 ,   0.0               , -50.0
          4,            0,            0,   0.0                ,  56.0366103025079  , 56.0366103025079  ,   0.0               , -10.0
          5,            0,            0,   0.0                ,  22.502067918760034, 22.502067918760034,   0.0               , -50.0
          6,            0,            0,  11.893432441431372  ,  71.89343244143137 , 60.0              ,   0.0               , -0.0
          7,            0,            0,   0.0                ,  29.521198632662404, 41.414631074093776,   0.0               , -50.0

```

## Contributing

Contributions are welcome! The formatting should follow the default cargo linter with the `rustfmt.toml` file from the repository and the test routine is done also with the cargo test suite.
