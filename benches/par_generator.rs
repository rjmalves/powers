//! PAR-008: Performance benchmarks for PAR generator
//!
//! These benchmarks measure the performance of PAR(p) generation under various
//! conditions and compare against stationary AR models.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use powers_rs::par_generator::PeriodicARGenerator;
use powers_rs::seasonal_params::SeasonalParams;

fn benchmark_par1_single_entity(c: &mut Criterion) {
    // Baseline: Single PAR(1) entity, 1000 iterations
    let params = SeasonalParams::new(
        12,
        vec![1; 12],
        vec![vec![0.7]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Pre-generate residuals (don't include in benchmark)
    let residuals: Vec<f64> =
        (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();

    c.bench_function("par1_single_entity_1000_iters", |b| {
        b.iter(|| {
            gen.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen.generate_next(a_t));
            }
        })
    });
}

fn benchmark_par2_single_entity(c: &mut Criterion) {
    // PAR(2) - higher AR order
    let params = SeasonalParams::new(
        12,
        vec![2; 12],
        vec![vec![0.6, 0.3]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    let residuals: Vec<f64> =
        (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();

    c.bench_function("par2_single_entity_1000_iters", |b| {
        b.iter(|| {
            gen.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen.generate_next(a_t));
            }
        })
    });
}

fn benchmark_par_max_order(c: &mut Criterion) {
    // Maximum practical AR order (p=12)
    let coeffs: Vec<f64> = (1..=12).map(|k| 0.05 * k as f64).collect();
    let params = SeasonalParams::new(
        12,
        vec![12; 12],
        vec![coeffs; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    let residuals: Vec<f64> =
        (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();

    c.bench_function("par12_max_order_1000_iters", |b| {
        b.iter(|| {
            gen.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen.generate_next(a_t));
            }
        })
    });
}

fn benchmark_par_varying_orders(c: &mut Criterion) {
    // Mixed AR orders across periods (realistic scenario)
    let params = SeasonalParams::new(
        12,
        vec![1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
        vec![
            vec![0.7],
            vec![0.75],
            vec![0.6, 0.2],
            vec![0.65, 0.25],
            vec![0.7, 0.2],
            vec![0.7, 0.2],
            vec![0.65, 0.25],
            vec![0.6, 0.2],
            vec![0.75],
            vec![0.8],
            vec![0.75],
            vec![0.7],
        ],
        vec![
            80.0, 100.0, 150.0, 200.0, 220.0, 200.0, 150.0, 120.0, 100.0, 80.0,
            70.0, 75.0,
        ],
        vec![
            15.0, 20.0, 30.0, 40.0, 45.0, 40.0, 30.0, 25.0, 20.0, 15.0, 12.0,
            13.0,
        ],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    let residuals: Vec<f64> =
        (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();

    c.bench_function("par_varying_orders_1000_iters", |b| {
        b.iter(|| {
            gen.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen.generate_next(a_t));
            }
        })
    });
}

fn benchmark_par_large_scale_simulation(c: &mut Criterion) {
    // Large scale: 10K iterations (typical SDDP forward pass length)
    let params = SeasonalParams::new(
        12,
        vec![1; 12],
        vec![vec![0.7]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    let residuals: Vec<f64> =
        (0..10000).map(|i| (i as f64 * 0.001).sin()).collect();

    c.bench_function("par1_large_scale_10k_iters", |b| {
        b.iter(|| {
            gen.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen.generate_next(a_t));
            }
        })
    });
}

fn benchmark_par_with_initial_conditions(c: &mut Criterion) {
    // Test performance with pre-filled initial conditions
    let initial_residuals = vec![1.0, 0.5, 0.8, 0.3, -0.2];

    let params = SeasonalParams::new(
        12,
        vec![2; 12],
        vec![vec![0.6, 0.3]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, initial_residuals.clone());

    let residuals: Vec<f64> =
        (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();

    c.bench_function("par2_with_initial_conditions_1000_iters", |b| {
        b.iter(|| {
            gen.reset(initial_residuals.clone());
            for &a_t in &residuals {
                black_box(gen.generate_next(a_t));
            }
        })
    });
}

fn benchmark_par_cache_efficiency(c: &mut Criterion) {
    // PERFORMANCE: Test cache efficiency with tight loop
    // PAR generator should have good cache locality due to VecDeque
    let params = SeasonalParams::new(
        12,
        vec![1; 12],
        vec![vec![0.7]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    c.bench_function("par1_cache_efficiency_tight_loop", |b| {
        b.iter(|| {
            gen.reset(vec![]);
            // Tight loop with constant residual (best case for cache)
            for _ in 0..10000 {
                black_box(gen.generate_next(black_box(0.5)));
            }
        })
    });
}

fn benchmark_par_reset_overhead(c: &mut Criterion) {
    // Measure reset() overhead
    let params = SeasonalParams::new(
        12,
        vec![2; 12],
        vec![vec![0.6, 0.3]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    c.bench_function("par_reset_overhead", |b| {
        b.iter(|| {
            gen.reset(black_box(vec![]));
        })
    });
}

fn benchmark_par_monthly_brazilian_scale(c: &mut Criterion) {
    // Realistic Brazilian system: 50 reservoirs, 120 months (10 years)
    // Each reservoir is a separate PAR generator in practice
    // This benchmarks single entity but at realistic scale

    let params = SeasonalParams::new(
        12,
        vec![2; 12], // Typical AR(2) for Brazilian hydro
        vec![vec![0.65, 0.25]; 12],
        vec![
            // Seasonal pattern: higher inflows Dec-Mar (summer)
            150.0, 200.0, 250.0, 220.0, 180.0, 140.0, 120.0, 110.0, 100.0,
            110.0, 120.0, 140.0,
        ],
        vec![
            30.0, 40.0, 50.0, 45.0, 35.0, 28.0, 24.0, 22.0, 20.0, 22.0, 24.0,
            28.0,
        ],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // 120 months = 10 years
    let residuals: Vec<f64> =
        (0..120).map(|i| (i as f64 * 0.1).sin() * 0.5).collect();

    c.bench_function("par2_brazilian_scale_120_months", |b| {
        b.iter(|| {
            gen.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen.generate_next(a_t));
            }
        })
    });
}

fn benchmark_par_vs_ar0_baseline(c: &mut Criterion) {
    // Compare PAR(1) against AR(0) baseline (white noise)
    // This shows the overhead of AR dynamics

    // AR(0) - no temporal dependence
    let params_ar0 = SeasonalParams::new(
        12,
        vec![0; 12],
        vec![vec![]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen_ar0 = PeriodicARGenerator::new(params_ar0, vec![]);

    // PAR(1)
    let params_par1 = SeasonalParams::new(
        12,
        vec![1; 12],
        vec![vec![0.7]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    )
    .unwrap();

    let mut gen_par1 = PeriodicARGenerator::new(params_par1, vec![]);

    let residuals: Vec<f64> =
        (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();

    c.bench_function("ar0_baseline_1000_iters", |b| {
        b.iter(|| {
            gen_ar0.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen_ar0.generate_next(a_t));
            }
        })
    });

    c.bench_function("par1_vs_ar0_1000_iters", |b| {
        b.iter(|| {
            gen_par1.reset(vec![]);
            for &a_t in &residuals {
                black_box(gen_par1.generate_next(a_t));
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_par1_single_entity,
    benchmark_par2_single_entity,
    benchmark_par_max_order,
    benchmark_par_varying_orders,
    benchmark_par_large_scale_simulation,
    benchmark_par_with_initial_conditions,
    benchmark_par_cache_efficiency,
    benchmark_par_reset_overhead,
    benchmark_par_monthly_brazilian_scale,
    benchmark_par_vs_ar0_baseline,
);
criterion_main!(benches);
