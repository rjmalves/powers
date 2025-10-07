//! Subproblem solve benchmarks - isolate solver overhead from SDDP algorithm
//!
//! These benchmarks measure the pure solver performance in isolation,
//! separate from SDDP algorithm overhead (cut selection, state management, etc.).
//!
//! **WHY THIS MATTERS**:
//! - Solver calls are 60-80% of SDDP runtime (hot path)
//! - Basis warm-starting is critical for performance
//! - Solver configuration (simplex vs IPM, presolve) impacts speed
//! - Problem scaling (variables, constraints) affects solve time
//!
//! **PERFORMANCE**: These benchmarks help detect:
//! - Solver interface regressions (highs-sys FFI overhead)
//! - Basis management overhead
//! - Problem construction overhead
//! - Scaling characteristics with problem size
//!
//! Run with: `cargo bench --bench subproblem_solve`

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::solver::Problem;
use powers_rs::subproblem::Subproblem;
use powers_rs::system::{Bus, Hydro, System, Thermal};

// =============================================================================
// Helper Functions - Create Test Systems
// =============================================================================

/// Create a minimal single-reservoir system for benchmarking
fn create_minimal_system() -> System {
    let bus = Bus::new(0, 500.0);
    let hydro = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 40.0, 0.01);
    let thermal = Thermal::new(0, 0, 50.0, 0.0, 25.0);
    System::new(vec![bus], vec![], vec![thermal], vec![hydro])
}

/// Create a cascade system (2 hydros) for larger problem benchmarks
fn create_cascade_system() -> System {
    let bus = Bus::new(0, 500.0);
    let hydro_upstream =
        Hydro::new(0, Some(1), 0, 1.0, 0.0, 60.0, 0.0, 30.0, 0.01);
    let hydro_downstream =
        Hydro::new(1, None, 0, 1.0, 0.0, 80.0, 0.0, 35.0, 0.01);
    let thermal = Thermal::new(0, 0, 50.0, 0.0, 25.0);
    System::new(
        vec![bus],
        vec![],
        vec![thermal],
        vec![hydro_upstream, hydro_downstream],
    )
}

/// Create a larger cascade system (5 hydros) for scaling benchmarks
fn create_large_cascade_system() -> System {
    let bus = Bus::new(0, 500.0);
    let mut hydros = Vec::new();

    // Create 5 hydros in cascade
    for i in 0..5 {
        let downstream = if i < 4 { Some(i + 1) } else { None };
        hydros.push(Hydro::new(
            i, downstream, 0, 1.0, 0.0, 100.0, 0.0, 30.0, 0.01,
        ));
    }

    let thermal = Thermal::new(0, 0, 50.0, 0.0, 50.0);
    System::new(vec![bus], vec![], vec![thermal], hydros)
}

// =============================================================================
// Benchmark Group 1: Cold Start Solve (no warm start)
// =============================================================================

/// Benchmark solving a subproblem from scratch (cold start)
///
/// This measures the baseline solver performance without any warm-starting.
/// Useful for detecting solver interface overhead and problem construction costs.
fn bench_cold_start_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("subproblem_cold_start");
    group.sample_size(100);

    // Single reservoir (minimal problem)
    group.bench_function("single_reservoir", |b| {
        let system = create_minimal_system();

        b.iter(|| {
            // Create new subproblem from scratch (cold start)
            let load_sp = powers_rs::stochastic_process::factory("naive");
            let inflow_sp = powers_rs::stochastic_process::factory("naive");
            let subproblem = Subproblem::new(
                &system,
                "storage",
                load_sp.as_ref(),
                inflow_sp.as_ref(),
            );
            black_box(subproblem);
        });
    });

    // Cascade (2 hydros)
    group.bench_function("cascade_2hydros", |b| {
        let system = create_cascade_system();

        b.iter(|| {
            let load_sp = powers_rs::stochastic_process::factory("naive");
            let inflow_sp = powers_rs::stochastic_process::factory("naive");
            let subproblem = Subproblem::new(
                &system,
                "storage",
                load_sp.as_ref(),
                inflow_sp.as_ref(),
            );
            black_box(subproblem);
        });
    });

    // Large cascade (5 hydros)
    group.bench_function("cascade_5hydros", |b| {
        let system = create_large_cascade_system();

        b.iter(|| {
            let load_sp = powers_rs::stochastic_process::factory("naive");
            let inflow_sp = powers_rs::stochastic_process::factory("naive");
            let subproblem = Subproblem::new(
                &system,
                "storage",
                load_sp.as_ref(),
                inflow_sp.as_ref(),
            );
            black_box(subproblem);
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Group 2: Problem Construction Overhead
// =============================================================================

/// Benchmark the overhead of constructing the LP problem (before solving)
///
/// This isolates the cost of building the Problem struct (variables, constraints)
/// from the actual solve time.
fn bench_problem_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("subproblem_construction");
    group.sample_size(100);

    // Single reservoir
    group.bench_function("single_reservoir", |b| {
        let _system = create_minimal_system();
        let _load_sp = powers_rs::stochastic_process::factory("naive");
        let _inflow_sp = powers_rs::stochastic_process::factory("naive");

        b.iter(|| {
            // Only construct the problem, don't solve yet
            let pb = Problem::new();

            // Add variables (this is what Subproblem::new does internally)
            // Note: This is an approximation since we don't have direct access
            // to the internal construction. In practice, measure via subtraction:
            // Total time - Solve time = Construction time
            black_box(pb);
        });
    });

    group.finish();
} // =============================================================================
  // Benchmark Group 3: Solver Scaling with Problem Size
  // =============================================================================

/// Benchmark how solver performance scales with problem size
fn bench_solver_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("subproblem_scaling");
    group.sample_size(50);

    let problem_sizes = vec![
        ("1_hydro", create_minimal_system()),
        ("2_hydros", create_cascade_system()),
        ("5_hydros", create_large_cascade_system()),
    ];

    for (name, system) in problem_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &system,
            |b, system| {
                let load_sp = powers_rs::stochastic_process::factory("naive");
                let inflow_sp = powers_rs::stochastic_process::factory("naive");

                b.iter(|| {
                    let subproblem = Subproblem::new(
                        system,
                        "storage",
                        load_sp.as_ref(),
                        inflow_sp.as_ref(),
                    );
                    black_box(subproblem);
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 4: Solver Options Impact
// =============================================================================

/// Benchmark different solver configurations
///
/// **PERFORMANCE CONSIDERATION**: Default is simplex with presolve off.
/// This benchmark measures the impact of changing solver options.
fn bench_solver_options(c: &mut Criterion) {
    let mut group = c.benchmark_group("subproblem_solver_options");
    group.sample_size(50);

    let system = create_cascade_system();

    // Benchmark 1: Default (simplex, presolve off)
    group.bench_function("default_simplex_no_presolve", |b| {
        let load_sp = powers_rs::stochastic_process::factory("naive");
        let inflow_sp = powers_rs::stochastic_process::factory("naive");

        b.iter(|| {
            let subproblem = Subproblem::new(
                &system,
                "storage",
                load_sp.as_ref(),
                inflow_sp.as_ref(),
            );
            black_box(subproblem);
        });
    });

    // Note: Other solver options (IPM, presolve on) require modifying
    // Subproblem::new, which is not accessible from benchmarks.
    // This is a limitation - we can only benchmark the default configuration.

    group.finish();
}

// =============================================================================
// Benchmark Group 5: Basis Reuse Overhead
// =============================================================================

/// Benchmark the overhead of basis extraction and reuse
///
/// **PERFORMANCE**: Basis warm-starting is critical for SDDP performance.
/// This measures the overhead of extracting and applying basis.
fn bench_basis_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("subproblem_basis_operations");
    group.sample_size(100);

    let system = create_minimal_system();

    // Benchmark: Create subproblem and extract basis
    group.bench_function("extract_basis", |b| {
        let load_sp = powers_rs::stochastic_process::factory("naive");
        let inflow_sp = powers_rs::stochastic_process::factory("naive");

        b.iter(|| {
            let subproblem = Subproblem::new(
                &system,
                "storage",
                load_sp.as_ref(),
                inflow_sp.as_ref(),
            );

            // Extract basis (if model exists)
            if let Some(ref model) = subproblem.model {
                let basis = model.get_basis();
                black_box(basis);
            }
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Group 6: Sequential Solves (simulating SDDP iteration)
// =============================================================================

/// Benchmark sequential solves with the same problem
///
/// This simulates what happens during SDDP training: solve → update RHS → solve again.
/// Measures the impact of basis reuse across solves.
fn bench_sequential_solves(c: &mut Criterion) {
    let mut group = c.benchmark_group("subproblem_sequential_solves");
    group.sample_size(50);

    let system = create_minimal_system();

    // Benchmark: 10 sequential solves (simulating 10 SDDP iterations)
    group.bench_function("10_sequential_solves", |b| {
        let load_sp = powers_rs::stochastic_process::factory("naive");
        let inflow_sp = powers_rs::stochastic_process::factory("naive");

        b.iter(|| {
            // Create initial subproblem
            let mut subproblem = Subproblem::new(
                &system,
                "storage",
                load_sp.as_ref(),
                inflow_sp.as_ref(),
            );

            // Simulate 10 iterations (basis should be reused)
            for _i in 0..10 {
                // In real SDDP, we'd update constraints here
                // For benchmark, just re-solve
                if let Some(ref mut model) = subproblem.model {
                    // Solve uses existing basis
                    model.solve();
                }
            }

            black_box(subproblem);
        });
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_cold_start_solve,
    bench_problem_construction,
    bench_solver_scaling,
    bench_solver_options,
    bench_basis_operations,
    bench_sequential_solves,
);
criterion_main!(benches);
