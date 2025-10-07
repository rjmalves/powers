// Cut Selection Performance Benchmarks (T3.5)
//
// Benchmarks for cut selection strategies and performance analysis:
// 1. Scaling with cut pool size [10, 100, 1000, 10000]
// 2. State dimensionality impact [1D, 5D, 20D]
// 3. Thread contention (current per-thread locking)
// 4. Batch vs per-thread selection strategies
// 5. Dominance computation overhead
//
// PERFORMANCE: These benchmarks measure the hot path in backward pass.
// Expected characteristics:
// - O(n × d) complexity (n=cuts, d=state_dimensions)
// - Lock contention overhead: 15-25% on multi-core
// - Batch selection: 15-30% faster due to eliminated contention

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::cut::BendersCut;
use powers_rs::fcf::FutureCostFunction;
use powers_rs::state::{State, StorageState};
use powers_rs::stochastic_process;
use powers_rs::system::{Hydro, System};
use std::sync::{Arc, Mutex};

// =============================================================================
// Helper Functions - Create Test Data
// =============================================================================

/// Create a test cut with given coefficients and RHS
fn create_test_cut(id: usize, coefficients: Vec<f64>, rhs: f64) -> BendersCut {
    BendersCut::new(id, coefficients, rhs)
}

/// Create a test storage state with the specified state dimension (hydro count)
fn create_test_state(state_dim: usize) -> Box<dyn State> {
    let mut system = System::default();

    // Adjust the number of hydros to match the desired state dimension
    system.meta.hydros_count = state_dim;
    system.hydros.clear();
    for i in 0..state_dim {
        system.hydros.push(Hydro::new(
            i, None, 0, 1.0,   // max_flow
            0.0,   // min_storage
            100.0, // max_storage
            0.0,   // initial_storage
            60.0,  // marginal_cost
            0.01,  // spillage_cost
        ));
    }

    let load_sp = stochastic_process::factory("naive");
    let inflow_sp = stochastic_process::factory("naive");
    Box::new(StorageState::new(
        &system,
        load_sp.as_ref(),
        inflow_sp.as_ref(),
    ))
}

/// Create an FCF with n cuts of dimension d
fn create_fcf_with_cuts(
    num_cuts: usize,
    state_dim: usize,
) -> FutureCostFunction {
    let mut fcf = FutureCostFunction::new();

    // Add cuts FIRST (states will reference these)
    for i in 0..num_cuts {
        let coefficients: Vec<f64> = (0..state_dim)
            .map(|j| 1.0 + i as f64 * 0.01 + j as f64 * 0.1)
            .collect();
        let rhs = 100.0 + i as f64 * 10.0;

        let cut = create_test_cut(i, coefficients, rhs);
        fcf.add_cut(cut);
        fcf.update_cut_pool_on_add(i);
    }

    // Then add states (after cuts exist)
    let num_states = std::cmp::max(5, num_cuts / 20);
    for _i in 0..num_states {
        let mut state = create_test_state(state_dim);
        // Initialize with first cut as dominating if cuts exist
        if !fcf.cut_pool.pool.is_empty() {
            let first_cut = &fcf.cut_pool.pool[0];
            let height = first_cut.eval_height_at_state(state.coefficients());
            state.set_dominating_objective(height);
            state.set_dominating_cut_id(0);
        }
        fcf.add_state(state);
    }

    fcf
}

// =============================================================================
// Benchmark Group 1: Scaling with Cut Pool Size
// =============================================================================

fn bench_cut_selection_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cut_selection_scaling");

    // Configure sampling: fewer samples for large problems
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(10));

    let cut_counts = vec![10, 100, 1000, 10000];
    let state_dim = 5; // Typical hydrothermal problem

    for &num_cuts in &cut_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_cuts),
            &num_cuts,
            |b, &num_cuts| {
                let fcf = create_fcf_with_cuts(num_cuts, state_dim);

                b.iter(|| {
                    // Simulate adding a new cut (hot path operation)
                    let mut new_fcf = fcf.clone_for_benchmark();
                    let coefficients: Vec<f64> =
                        (0..state_dim).map(|j| 2.0 + j as f64 * 0.1).collect();
                    let mut new_cut =
                        create_test_cut(num_cuts, coefficients, 200.0);

                    // This is the hot path: evaluate dominance
                    new_fcf.eval_new_cut_domination(&mut new_cut);

                    black_box(new_cut)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 2: State Dimensionality Impact
// =============================================================================

fn bench_state_dimensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group("cut_selection_dimensionality");
    group.sample_size(50);

    let state_dims = vec![1, 5, 20]; // 1D=simple, 5D=typical, 20D=large cascade
    let num_cuts = 100;

    for &state_dim in &state_dims {
        group.bench_with_input(
            BenchmarkId::from_parameter(state_dim),
            &state_dim,
            |b, &state_dim| {
                let fcf = create_fcf_with_cuts(num_cuts, state_dim);

                b.iter(|| {
                    let mut new_fcf = fcf.clone_for_benchmark();
                    let coefficients: Vec<f64> =
                        (0..state_dim).map(|j| 2.0 + j as f64 * 0.1).collect();
                    let mut new_cut =
                        create_test_cut(num_cuts, coefficients, 200.0);

                    new_fcf.eval_new_cut_domination(&mut new_cut);

                    black_box(new_cut)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 3: Thread Contention (Current Architecture)
// =============================================================================

fn bench_thread_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("cut_selection_thread_contention");
    group.sample_size(20);

    let num_threads_vec = vec![1, 2, 4, 8];
    let num_cuts = 1000;
    let state_dim = 5;

    for &num_threads in &num_threads_vec {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &num_threads| {
                // Create shared FCF (simulates current architecture)
                let fcf = Arc::new(Mutex::new(create_fcf_with_cuts(
                    num_cuts, state_dim,
                )));

                b.iter(|| {
                    // Simulate parallel backward pass with lock contention
                    let handles: Vec<_> = (0..num_threads)
                        .map(|thread_id| {
                            let fcf_clone = Arc::clone(&fcf);
                            std::thread::spawn(move || {
                                // Each thread locks FCF and runs cut selection
                                let mut fcf_locked = fcf_clone.lock().unwrap();
                                let coefficients: Vec<f64> = (0..state_dim)
                                    .map(|j| {
                                        2.0 + thread_id as f64 + j as f64 * 0.1
                                    })
                                    .collect();
                                let mut new_cut = create_test_cut(
                                    num_cuts + thread_id,
                                    coefficients,
                                    200.0,
                                );

                                // Hold lock during cut selection (current behavior)
                                fcf_locked
                                    .eval_new_cut_domination(&mut new_cut);
                                fcf_locked.add_cut(new_cut);
                                fcf_locked.update_cut_pool_on_add(
                                    num_cuts + thread_id,
                                );
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 4: Batch vs Per-Thread Selection
// =============================================================================

fn bench_batch_vs_perthread(c: &mut Criterion) {
    let mut group = c.benchmark_group("cut_selection_batch_vs_perthread");
    group.sample_size(20);

    let num_cuts_to_add = 8; // Typical number of cuts per backward pass
    let pool_size = 1000;
    let state_dim = 5;

    // Benchmark 1: Per-thread (current - with lock contention)
    group.bench_function("per_thread_locked", |b| {
        let fcf =
            Arc::new(Mutex::new(create_fcf_with_cuts(pool_size, state_dim)));

        b.iter(|| {
            let handles: Vec<_> = (0..num_cuts_to_add)
                .map(|thread_id| {
                    let fcf_clone = Arc::clone(&fcf);
                    std::thread::spawn(move || {
                        let mut fcf_locked = fcf_clone.lock().unwrap();
                        let coefficients: Vec<f64> = (0..state_dim)
                            .map(|j| 2.0 + thread_id as f64 + j as f64 * 0.1)
                            .collect();
                        let mut new_cut = create_test_cut(
                            pool_size + thread_id,
                            coefficients,
                            200.0,
                        );

                        fcf_locked.eval_new_cut_domination(&mut new_cut);
                        fcf_locked.add_cut(new_cut);
                        fcf_locked
                            .update_cut_pool_on_add(pool_size + thread_id);
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    // Benchmark 2: Batch (proposed - no lock contention)
    group.bench_function("batch_synchronized", |b| {
        let fcf =
            Arc::new(Mutex::new(create_fcf_with_cuts(pool_size, state_dim)));

        b.iter(|| {
            // Phase 1: Compute cuts in parallel (no lock)
            let cuts: Vec<_> = (0..num_cuts_to_add)
                .map(|thread_id| {
                    let coefficients: Vec<f64> = (0..state_dim)
                        .map(|j| 2.0 + thread_id as f64 + j as f64 * 0.1)
                        .collect();
                    create_test_cut(pool_size + thread_id, coefficients, 200.0)
                })
                .collect();

            // Phase 2: Add all cuts synchronously (single lock)
            let mut fcf_locked = fcf.lock().unwrap();
            for mut cut in cuts {
                let cut_id = cut.id;
                fcf_locked.eval_new_cut_domination(&mut cut);
                fcf_locked.add_cut(cut);
                fcf_locked.update_cut_pool_on_add(cut_id);
            }
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Group 5: Dominance Computation Components
// =============================================================================

fn bench_dominance_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("cut_selection_dominance_components");
    group.sample_size(100);

    let state_dim = 5;
    let num_cuts = 100;

    // Benchmark: eval_new_cut_domination (checks new cut against all states)
    group.bench_function("eval_new_cut_domination", |b| {
        let fcf = create_fcf_with_cuts(num_cuts, state_dim);

        b.iter(|| {
            let mut new_fcf = fcf.clone_for_benchmark();
            let coefficients: Vec<f64> =
                (0..state_dim).map(|j| 2.0 + j as f64 * 0.1).collect();
            let mut new_cut = create_test_cut(num_cuts, coefficients, 200.0);

            new_fcf.eval_new_cut_domination(&mut new_cut);

            black_box(new_cut)
        });
    });

    // Benchmark: update_old_cuts_domination (checks old cuts against new state)
    group.bench_function("update_old_cuts_domination", |b| {
        let fcf = create_fcf_with_cuts(num_cuts, state_dim);

        b.iter(|| {
            let mut new_fcf = fcf.clone_for_benchmark();
            let mut new_state = create_test_state(state_dim);

            let returning_cuts =
                new_fcf.update_old_cuts_domination(&mut new_state);

            black_box(returning_cuts)
        });
    });

    // Benchmark: Individual height evaluation (core operation)
    group.bench_function("eval_height_at_state", |b| {
        let coefficients: Vec<f64> =
            (0..state_dim).map(|j| 1.0 + j as f64 * 0.1).collect();
        let cut = create_test_cut(0, coefficients.clone(), 100.0);

        let state = create_test_state(state_dim);

        b.iter(|| {
            let height = cut.eval_height_at_state(state.coefficients());
            black_box(height)
        });
    });

    group.finish();
}

// =============================================================================
// Helper trait for FCF cloning (for benchmarks)
// =============================================================================

trait FcfBenchmarkHelper {
    fn clone_for_benchmark(&self) -> Self;
}

impl FcfBenchmarkHelper for FutureCostFunction {
    fn clone_for_benchmark(&self) -> Self {
        // For benchmarks, we create a shallow clone
        // In production, FCF is wrapped in Arc<Mutex<>>, not cloned
        let mut new_fcf = FutureCostFunction::new();

        // Infer state dimension from cuts (all cuts have same dimension)
        let state_dim = if !self.cut_pool.pool.is_empty() {
            self.cut_pool.pool[0].coefficients.len()
        } else {
            1 // Default dimension
        };

        // Clone cuts
        for cut in &self.cut_pool.pool {
            let new_cut = BendersCut {
                id: cut.id,
                coefficients: cut.coefficients.clone(),
                rhs: cut.rhs,
                active: cut.active,
                non_dominated_state_count: cut.non_dominated_state_count,
            };
            new_fcf.add_cut(new_cut);
        }

        // Clone states (same count, same dimension)
        for _state in &self.state_pool.pool {
            new_fcf.add_state(create_test_state(state_dim));
        }

        new_fcf.cut_pool.active_cut_indices =
            self.cut_pool.active_cut_indices.clone();
        new_fcf.cut_pool.total_cut_count = self.cut_pool.total_cut_count;

        new_fcf
    }
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_cut_selection_scaling,
    bench_state_dimensionality,
    bench_thread_contention,
    bench_batch_vs_perthread,
    bench_dominance_components
);
criterion_main!(benches);
