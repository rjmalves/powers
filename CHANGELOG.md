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
