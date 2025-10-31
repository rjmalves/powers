//! State operations benchmarks - measure state management overhead
//!
//! These benchmarks measure the performance of state-related operations
//! that are frequently executed during SDDP training:
//! - State construction (initial allocation)
//! - State update (after forward pass)
//! - Coefficient access (for cut evaluation)
//! - State cloning (for FCF management)
//! - Dominating cut tracking (cut selection overhead)
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::state::{State, StorageState};
use powers_rs::stochastic_process;
use powers_rs::system::{Bus, Hydro, System, Thermal};

// =============================================================================
// Helper Functions - Create Test Systems
// =============================================================================

/// Create a system with specified number of hydros (state dimensionality)
fn create_system_with_hydros(num_hydros: usize) -> System {
    let bus = Bus::new(0, 500.0);

    let mut hydros = Vec::new();
    for i in 0..num_hydros {
        // Create cascade: each hydro flows to the next
        let downstream = if i < num_hydros - 1 {
            Some(i + 1)
        } else {
            None
        };

        hydros.push(Hydro::new(
            i, downstream, 0, 1.0,   // productivity
            0.0,   // min_storage
            100.0, // max_storage
            0.0,   // min_turbined_flow
            40.0,  // max_turbined_flow
            0.01,  // spillage_penalty
        ));
    }

    let thermal = Thermal::new(0, 0, 50.0, 0.0, 25.0);
    System::new(vec![bus], vec![], vec![thermal], hydros)
}

/// Create a storage state for the given system
fn create_storage_state(system: &System) -> Box<dyn State> {
    let load_sp = stochastic_process::factory("naive");
    let inflow_sp = stochastic_process::factory("naive");
    let inflow_processes = vec![inflow_sp];
    Box::new(StorageState::new(
        system,
        load_sp.as_ref(),
        &inflow_processes,
    ))
}

// =============================================================================
// Benchmark Group 1: State Construction
// =============================================================================

/// Benchmark state construction overhead
///
/// State construction happens once per stage per iteration.
/// Measures allocation and initialization costs.
fn bench_state_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_construction");
    group.sample_size(100);

    let state_dims = vec![1, 5, 10, 20, 50];

    for &num_hydros in &state_dims {
        group.bench_with_input(
            BenchmarkId::new("storage_state", num_hydros),
            &num_hydros,
            |b, &num_hydros| {
                let system = create_system_with_hydros(num_hydros);
                let load_sp = stochastic_process::factory("naive");
                let inflow_sp = stochastic_process::factory("naive");
                let inflow_processes = vec![inflow_sp];

                b.iter(|| {
                    let state = StorageState::new(
                        &system,
                        load_sp.as_ref(),
                        &inflow_processes,
                    );
                    black_box(state);
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 2: Coefficient Access
// =============================================================================

/// Benchmark coefficient access patterns
///
/// **CRITICAL HOT PATH**: Coefficients are accessed on every cut evaluation.
/// Cache-friendliness is essential.
fn bench_coefficient_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_coefficient_access");
    group.sample_size(100);

    let state_dims = vec![1, 5, 10, 20, 50];

    for &num_hydros in &state_dims {
        group.bench_with_input(
            BenchmarkId::new("read_coefficients", num_hydros),
            &num_hydros,
            |b, &num_hydros| {
                let system = create_system_with_hydros(num_hydros);
                let state = create_storage_state(&system);

                b.iter(|| {
                    // Access coefficients (hot path in cut evaluation)
                    let coeffs = state.coefficients();
                    black_box(coeffs);
                });
            },
        );
    }

    // Benchmark: Sum all coefficients (simulates dot product in cut evaluation)
    for &num_hydros in &state_dims {
        group.bench_with_input(
            BenchmarkId::new("sum_coefficients", num_hydros),
            &num_hydros,
            |b, &num_hydros| {
                let system = create_system_with_hydros(num_hydros);
                let state = create_storage_state(&system);

                b.iter(|| {
                    let coeffs = state.coefficients();
                    let sum: f64 = coeffs.iter().sum();
                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 3: State Update
// =============================================================================

/// Benchmark state update operations
///
/// State updates happen after every forward pass solve.
/// Measures the cost of updating state tracking.
fn bench_state_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_update");
    group.sample_size(100);

    let num_hydros = 5;
    let system = create_system_with_hydros(num_hydros);

    // Benchmark: Set dimension
    group.bench_function("set_dimension", |b| {
        let mut state = create_storage_state(&system);

        b.iter(|| {
            state.set_dimension(num_hydros);
            black_box(&state);
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Group 4: Dominating Cut Tracking
// =============================================================================

/// Benchmark dominating cut tracking operations
///
/// These operations are used during cut selection to track which cut
/// dominates at each state.
fn bench_dominating_cut_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_dominating_cut");
    group.sample_size(100);

    let num_hydros = 5; // Typical problem size
    let system = create_system_with_hydros(num_hydros);

    // Benchmark: Set dominating cut ID
    group.bench_function("set_dominating_cut_id", |b| {
        let mut state = create_storage_state(&system);

        b.iter(|| {
            state.set_dominating_cut_id(42);
            black_box(&state);
        });
    });

    // Benchmark: Set dominating objective
    group.bench_function("set_dominating_objective", |b| {
        let mut state = create_storage_state(&system);

        b.iter(|| {
            state.set_dominating_objective(123.45);
            black_box(&state);
        });
    });

    // Benchmark: Get dominating cut ID
    group.bench_function("get_dominating_cut_id", |b| {
        let state = create_storage_state(&system);

        b.iter(|| {
            let id = state.get_dominating_cut_id();
            black_box(id);
        });
    });

    // Benchmark: Get dominating objective
    group.bench_function("get_dominating_objective", |b| {
        let state = create_storage_state(&system);

        b.iter(|| {
            let obj = state.get_dominating_objective();
            black_box(obj);
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Group 5: State Cloning
// =============================================================================

/// Benchmark state cloning operations
///
/// State cloning may occur in FCF management (though Arc is preferred).
/// Measures allocation + copy overhead.
fn bench_state_cloning(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_cloning");
    group.sample_size(100);

    let state_dims = vec![1, 5, 10, 20, 50];

    for &num_hydros in &state_dims {
        group.bench_with_input(
            BenchmarkId::new("clone_boxed_state", num_hydros),
            &num_hydros,
            |b, &num_hydros| {
                let system = create_system_with_hydros(num_hydros);
                let state = create_storage_state(&system);

                b.iter(|| {
                    // Clone the boxed state using trait method
                    let cloned = state.clone();
                    black_box(cloned);
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 6: Memory Layout Impact
// =============================================================================

/// Benchmark memory access patterns
///
/// Tests if coefficient access is cache-friendly.
/// Compares sequential vs strided access.
fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_memory_patterns");
    group.sample_size(100);

    let num_hydros = 20; // Large enough to show cache effects
    let system = create_system_with_hydros(num_hydros);

    // Benchmark: Sequential access (cache-friendly)
    group.bench_function("sequential_access", |b| {
        let state = create_storage_state(&system);

        b.iter(|| {
            let coeffs = state.coefficients();
            let mut sum = 0.0;
            for &coeff in coeffs {
                sum += coeff;
            }
            black_box(sum);
        });
    });

    // Benchmark: Strided access (cache-unfriendly if large stride)
    group.bench_function("strided_access_step2", |b| {
        let state = create_storage_state(&system);

        b.iter(|| {
            let coeffs = state.coefficients();
            let mut sum = 0.0;
            for i in (0..coeffs.len()).step_by(2) {
                sum += coeffs[i];
            }
            black_box(sum);
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Group 7: Batch Operations
// =============================================================================

/// Benchmark batch state operations
///
/// Simulates operations on multiple states (as in FCF management).
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_batch_operations");
    group.sample_size(50);

    let num_hydros = 5;
    let batch_sizes = vec![10, 50, 100, 500];

    for &batch_size in &batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("create_batch", batch_size),
            &batch_size,
            |b, &batch_size| {
                let system = create_system_with_hydros(num_hydros);

                b.iter(|| {
                    let mut states = Vec::with_capacity(batch_size);
                    for _i in 0..batch_size {
                        states.push(create_storage_state(&system));
                    }
                    black_box(states);
                });
            },
        );
    }

    // Benchmark: Access coefficients from batch of states
    for &batch_size in &batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("access_batch_coefficients", batch_size),
            &batch_size,
            |b, &batch_size| {
                let system = create_system_with_hydros(num_hydros);
                let states: Vec<_> = (0..batch_size)
                    .map(|_| create_storage_state(&system))
                    .collect();

                b.iter(|| {
                    let mut total_sum = 0.0;
                    for state in &states {
                        let coeffs = state.coefficients();
                        let sum: f64 = coeffs.iter().sum();
                        total_sum += sum;
                    }
                    black_box(total_sum);
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_state_construction,
    bench_coefficient_access,
    bench_state_update,
    bench_dominating_cut_tracking,
    bench_state_cloning,
    bench_memory_access_patterns,
    bench_batch_operations,
);
criterion_main!(benches);
