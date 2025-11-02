use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::utils::dot_product;
use powers_rs::utils::simd::{dot_product_kahan_simd, dot_product_simd};

fn benchmark_dot_product_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_comparison");

    // Test different vector sizes relevant to SDDP (AR orders 1-10)
    for size in [1, 3, 5, 10, 50, 100] {
        let a: Vec<f64> = (0..size).map(|i| i as f64 + 0.5).collect();
        let b: Vec<f64> = (0..size).map(|i| (i as f64) * 0.8).collect();

        // Naive implementation
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    black_box(dot_product(black_box(&a), black_box(&b)))
                });
            },
        );

        // SIMD-optimized implementation
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    black_box(dot_product_simd(black_box(&a), black_box(&b)))
                });
            },
        );

        // Kahan SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("kahan_simd", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    black_box(dot_product_kahan_simd(
                        black_box(&a),
                        black_box(&b),
                    ))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_ar_typical_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_ar_typical");

    // AR(1) - most common case
    let ar1_coef = vec![0.8];
    let ar1_lags = vec![100.0];
    group.bench_function("AR1_naive", |b| {
        b.iter(|| {
            black_box(dot_product(black_box(&ar1_coef), black_box(&ar1_lags)))
        });
    });
    group.bench_function("AR1_simd", |b| {
        b.iter(|| {
            black_box(dot_product_simd(
                black_box(&ar1_coef),
                black_box(&ar1_lags),
            ))
        });
    });

    // AR(2) - common case
    let ar2_coef = vec![0.6, 0.3];
    let ar2_lags = vec![100.0, 80.0];
    group.bench_function("AR2_naive", |b| {
        b.iter(|| {
            black_box(dot_product(black_box(&ar2_coef), black_box(&ar2_lags)))
        });
    });
    group.bench_function("AR2_simd", |b| {
        b.iter(|| {
            black_box(dot_product_simd(
                black_box(&ar2_coef),
                black_box(&ar2_lags),
            ))
        });
    });

    // AR(3) - less common but still used
    let ar3_coef = vec![0.5, 0.3, 0.1];
    let ar3_lags = vec![100.0, 80.0, 60.0];
    group.bench_function("AR3_naive", |b| {
        b.iter(|| {
            black_box(dot_product(black_box(&ar3_coef), black_box(&ar3_lags)))
        });
    });
    group.bench_function("AR3_simd", |b| {
        b.iter(|| {
            black_box(dot_product_simd(
                black_box(&ar3_coef),
                black_box(&ar3_lags),
            ))
        });
    });

    group.finish();
}

fn benchmark_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_stability");

    // Pathological case for numerical precision
    let extreme_a = vec![1e10, 1.0, 1.0, 1.0, -1e10];
    let extreme_b = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    group.bench_function("extreme_values_naive", |b| {
        b.iter(|| {
            black_box(dot_product(black_box(&extreme_a), black_box(&extreme_b)))
        });
    });

    group.bench_function("extreme_values_simd", |b| {
        b.iter(|| {
            black_box(dot_product_simd(
                black_box(&extreme_a),
                black_box(&extreme_b),
            ))
        });
    });

    group.bench_function("extreme_values_kahan", |b| {
        b.iter(|| {
            black_box(dot_product_kahan_simd(
                black_box(&extreme_a),
                black_box(&extreme_b),
            ))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_dot_product_sizes,
    benchmark_ar_typical_cases,
    benchmark_numerical_stability
);
criterion_main!(benches);
