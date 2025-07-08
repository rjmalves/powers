# v0.2.0

- Implements single-node parallelism (thread-based) for both training and simulation steps
- Applies L1 dominance cut selection while locking the future cost function on each thread
- Number of parallel threads is capped either on the logical core count or number of forward passes / simulated scenarios
- Generalizes the underlying data structure for the SDDP algorithm from a vector to a graph
- Creates submodules for scenario generation, risk measure, state, and stochastic process
- Generalizes recourse input for defining inflow and load scenarios, with branchings per node
- Uses trait objects for dynamic risk measure, state and stochastic process definition

# v0.1.1

- Better handles memory allocation during the SAA generation step in simulation
- Fixes doctests from renaming the crate

# v0.1.0

- Initial release
- Solves a simples hydrothermal dispatch problem with hydro storages as state variables using SDDP
- Inflows are sampled from `LogNormal` distributions, considered the same for all stages
- Loads are constant, given by the user in the input data
- Implements a custom interface to the `HiGHS` solver
- Contains cut selection and basis reuse for improving performance
- Performs simulation by sampling from the same distributions used for training
