use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::initial_condition::InitialCondition;
use powers_rs::input::Recourse;
use powers_rs::scenario::ScenarioGenerator;

/// Benchmark: Independent Normal noise (baseline)
fn benchmark_independent_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("scenario_generation_independent");

    for (num_scenarios, num_entities) in &[(100, 5), (1000, 10), (1000, 20)] {
        let recourse_json = format!(
            r#"{{
            "initial_condition": {{
                "storage": [],
                "inflow": []
            }},
            "noise_models": [
                {}
            ]
        }}"#,
            (0..*num_entities)
                .map(|i| format!(
                    r#"{{
                    "noise_type": "independent",
                    "uncertainty_type": "inflow",
                    "entity_id": {},
                    "season_id": 1,
                    "distribution": {{"type": "normal", "mean": 100.0, "std_dev": 20.0}}
                }}"#,
                    i
                ))
                .collect::<Vec<_>>()
                .join(",\n")
        );

        let recourse: Recourse = serde_json::from_str(&recourse_json).unwrap();
        let initial_condition =
            InitialCondition::new(vec![], vec![vec![]; *num_entities]);
        let seed = 42;

        let generator = ScenarioGenerator::from_recourse_input(
            &recourse,
            &initial_condition,
            seed,
        )
        .unwrap();

        let num_stages = 12;
        let scenarios_per_stage = vec![*num_scenarios; num_stages];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "{}x{}x{}",
                num_scenarios, num_entities, num_stages
            )),
            &(&generator, &scenarios_per_stage),
            |b, (gen, sps)| {
                b.iter(|| {
                    let _saa =
                        gen.generate_saa(black_box(num_stages), black_box(sps));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: AR(1) models (temporal correlation)
fn benchmark_ar1_temporal(c: &mut Criterion) {
    let mut group = c.benchmark_group("scenario_generation_ar1");

    for (num_scenarios, num_entities) in &[(100, 5), (1000, 10)] {
        let recourse_json = format!(
            r#"{{
            "initial_condition": {{
                "storage": [],
                "inflow": [
                    {}
                ]
            }},
            "noise_models": [
                {}
            ]
        }}"#,
            (0..*num_entities)
                .map(|i| format!(
                    r#"{{"hydro_id": {}, "lag": 1, "value": {}}}"#,
                    i,
                    100.0 + i as f64 * 10.0
                ))
                .collect::<Vec<_>>()
                .join(",\n"),
            (0..*num_entities)
                .map(|i| format!(
                    r#"{{
                    "noise_type": "autoregressive",
                    "uncertainty_type": "inflow",
                    "entity_id": {},
                    "season_id": 1,
                    "distribution": {{"type": "normal", "mean": 0.0, "std_dev": 15.0}},
                    "lag_order": 1,
                    "coefficients": [0.7]
                }}"#,
                    i
                ))
                .collect::<Vec<_>>()
                .join(",\n")
        );

        let recourse: Recourse = serde_json::from_str(&recourse_json).unwrap();
        let initial_condition = InitialCondition::new(
            vec![],
            (0..*num_entities)
                .map(|i| vec![100.0 + i as f64 * 10.0])
                .collect(),
        );
        let seed = 42;

        let generator = ScenarioGenerator::from_recourse_input(
            &recourse,
            &initial_condition,
            seed,
        )
        .unwrap();

        let num_stages = 12;
        let scenarios_per_stage = vec![*num_scenarios; num_stages];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "{}x{}x{}",
                num_scenarios, num_entities, num_stages
            )),
            &(&generator, &scenarios_per_stage),
            |b, (gen, sps)| {
                b.iter(|| {
                    let _saa =
                        gen.generate_saa(black_box(num_stages), black_box(sps));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Full pipeline stress test (1000×10×12)
/// Target: <200ms per the architecture spec
fn benchmark_full_pipeline_stress(c: &mut Criterion) {
    let num_scenarios = 1000;
    let num_entities = 10;
    let num_stages = 12;

    let recourse_json = format!(
        r#"{{
        "initial_condition": {{
            "storage": [],
            "inflow": [
                {}
            ]
        }},
        "noise_models": [
            {}
        ]
    }}"#,
        (0..num_entities)
            .map(|i| format!(
                r#"{{"hydro_id": {}, "lag": 1, "value": {}}}"#,
                i,
                100.0 + i as f64 * 10.0
            ))
            .collect::<Vec<_>>()
            .join(",\n"),
        (0..num_entities)
            .map(|i| format!(
                r#"{{
                "noise_type": "autoregressive",
                "uncertainty_type": "inflow",
                "entity_id": {},
                "season_id": 1,
                "distribution": {{"type": "normal", "mean": 0.0, "std_dev": 15.0}},
                "lag_order": 1,
                "coefficients": [0.7]
            }}"#,
                i
            ))
            .collect::<Vec<_>>()
            .join(",\n")
    );

    let recourse: Recourse = serde_json::from_str(&recourse_json).unwrap();
    let initial_condition = InitialCondition::new(
        vec![],
        (0..num_entities)
            .map(|i| vec![100.0 + i as f64 * 10.0])
            .collect(),
    );
    let seed = 42;

    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        seed,
    )
    .unwrap();

    let scenarios_per_stage = vec![num_scenarios; num_stages];

    c.bench_function("full_pipeline_1000x10x12_stress", |b| {
        b.iter(|| {
            let _saa = generator.generate_saa(
                black_box(num_stages),
                black_box(&scenarios_per_stage),
            );
        })
    });
}

/// Benchmark: Memory allocation overhead
/// Measures the cost of SAA structure creation vs computation
fn benchmark_saa_allocation_overhead(c: &mut Criterion) {
    let num_scenarios = 1000;
    let num_entities = 10;
    let num_stages = 12;

    let recourse_json = format!(
        r#"{{
        "initial_condition": {{
            "storage": [],
            "inflow": []
        }},
        "noise_models": [
            {}
        ]
    }}"#,
        (0..num_entities)
            .map(|i| format!(
                r#"{{
                "noise_type": "independent",
                "uncertainty_type": "inflow",
                "entity_id": {},
                "season_id": 1,
                "distribution": {{"type": "normal", "mean": 100.0, "std_dev": 20.0}}
            }}"#,
                i
            ))
            .collect::<Vec<_>>()
            .join(",\n")
    );

    let recourse: Recourse = serde_json::from_str(&recourse_json).unwrap();
    let initial_condition =
        InitialCondition::new(vec![], vec![vec![]; num_entities]);
    let seed = 42;

    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        seed,
    )
    .unwrap();

    let scenarios_per_stage = vec![num_scenarios; num_stages];

    c.bench_function("saa_with_allocation", |b| {
        b.iter(|| {
            let _saa = generator.generate_saa(
                black_box(num_stages),
                black_box(&scenarios_per_stage),
            );
        })
    });
}

criterion_group!(
    benches,
    benchmark_independent_normal,
    benchmark_ar1_temporal,
    benchmark_full_pipeline_stress,
    benchmark_saa_allocation_overhead,
);
criterion_main!(benches);
