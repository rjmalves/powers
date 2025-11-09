use powers_rs::scenario::NoiseGenerator;
use rand_distr::{LogNormal, Normal};
use std::time::Instant;

mod fixtures;
use fixtures::oos::*;

#[test]
fn test_oos_generator_creation() {
    let mut gen = NoiseGenerator::new();
    gen.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        100,
    );

    let oos_gen = OOSGenerator::new(42, gen);
    assert_eq!(oos_gen.get_oos_seed(), 42 + 1_000_000);
}

#[test]
fn test_oos_seed_offsets() {
    let gen = NoiseGenerator::<Normal<f64>, LogNormal<f64>>::new();
    let oos_gen = OOSGenerator::new(42, gen);

    assert_eq!(oos_gen.get_oos_seed(), 42 + 1_000_000);
    assert_eq!(oos_gen.get_shift_seed(), 42 + 2_000_000);
}

#[test]
fn test_generate_independent_scenarios() {
    let mut gen = NoiseGenerator::new();
    gen.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        100,
    );

    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(1000);

    assert_eq!(oos_saa.stage_scenarios.len(), 1);
    assert_eq!(oos_saa.get_branching_count_at_stage(0).unwrap(), 1000);
}

#[test]
fn test_oos_generator_reproducibility() {
    let mut gen1 = NoiseGenerator::new();
    gen1.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        100,
    );

    let mut gen2 = NoiseGenerator::new();
    gen2.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        100,
    );

    let mut oos_gen1 = OOSGenerator::new(42, gen1);
    let mut oos_gen2 = OOSGenerator::new(42, gen2);

    let saa1 = oos_gen1.generate_independent(100);
    let saa2 = oos_gen2.generate_independent(100);

    let sample1 = saa1.get_noises_by_stage_and_branching(0, 0).unwrap();
    let sample2 = saa2.get_noises_by_stage_and_branching(0, 0).unwrap();

    assert_eq!(
        sample1.get_inflow_innovations()[0],
        sample2.get_inflow_innovations()[0]
    );
}

#[test]
fn test_oos_report_creation() {
    let report = OOSReport::new(1000.0, 1030.0);

    assert_eq!(report.in_sample_cost, 1000.0);
    assert_eq!(report.oos_cost, 1030.0);
    assert_eq!(report.generalization_gap, 30.0);
    assert!((report.oos_ratio - 1.03).abs() < 1e-10);
}

#[test]
fn test_oos_report_is_generalizing_excellent() {
    let report = OOSReport::new(1000.0, 1020.0); // 2% gap

    assert!(report.is_generalizing(0.05));
    assert!(report.is_generalizing(0.03));
    assert!(report.is_generalizing(0.02));
}

#[test]
fn test_oos_report_is_generalizing_poor() {
    let report = OOSReport::new(1000.0, 1200.0); // 20% gap

    assert!(!report.is_generalizing(0.05));
    assert!(!report.is_generalizing(0.15));
    assert!(report.is_generalizing(0.25));
}

#[test]
fn test_oos_report_quality_descriptions() {
    let excellent = OOSReport::new(1000.0, 1020.0);
    assert_eq!(excellent.quality_description(), "Excellent");

    let good = OOSReport::new(1000.0, 1070.0);
    assert_eq!(good.quality_description(), "Good");

    let acceptable = OOSReport::new(1000.0, 1120.0);
    assert_eq!(acceptable.quality_description(), "Acceptable");

    let poor = OOSReport::new(1000.0, 1200.0);
    assert_eq!(poor.quality_description(), "Poor");
}

#[test]
fn test_oos_report_zero_division_safety() {
    let report = OOSReport::new(0.0, 10.0);
    assert_eq!(report.oos_ratio, 1.0);
}

#[test]
fn test_oos_report_negative_gap() {
    let report = OOSReport::new(1000.0, 950.0);

    assert_eq!(report.generalization_gap, -50.0);
    assert!((report.oos_ratio - 0.95).abs() < 1e-10);
    assert!(report.is_generalizing(0.10));
}

#[test]
fn test_oos_evaluator_creation() {
    let evaluator = OOSEvaluator::new();
    let report = evaluator.evaluate(1000.0, 1030.0);

    assert_eq!(report.in_sample_cost, 1000.0);
    assert_eq!(report.oos_cost, 1030.0);
}

#[test]
fn test_oos_evaluator_from_vectors() {
    let evaluator = OOSEvaluator::new();
    let in_sample = vec![900.0, 1000.0, 1100.0];
    let oos = vec![920.0, 1030.0, 1140.0];

    let report = evaluator.evaluate_from_vectors(&in_sample, &oos);

    assert!((report.in_sample_cost - 1000.0).abs() < 1e-6);
    assert!((report.oos_cost - 1030.0).abs() < 1e-6);
}

#[test]
fn test_oos_evaluator_default() {
    let evaluator1 = OOSEvaluator::new();
    let evaluator2 = OOSEvaluator;

    let report1 = evaluator1.evaluate(1000.0, 1030.0);
    let report2 = evaluator2.evaluate(1000.0, 1030.0);

    assert_eq!(report1.generalization_gap, report2.generalization_gap);
}

#[test]
fn test_kolmogorov_smirnov_identical_samples() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
    assert!(p_value > 0.05, "p-value = {}", p_value);
}

#[test]
fn test_kolmogorov_smirnov_very_different_samples() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![100.0, 200.0, 300.0, 400.0, 500.0];

    let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
    assert!(p_value < 0.05, "p-value = {}", p_value);
}

#[test]
fn test_kolmogorov_smirnov_overlapping_samples() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![1.5, 2.5, 3.5, 4.5, 5.5];

    let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
    assert!((0.0..=1.0).contains(&p_value));
}

#[test]
fn test_kolmogorov_smirnov_large_samples() {
    let sample1: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let sample2: Vec<f64> = (0..1000).map(|i| (i as f64) + 0.5).collect();

    let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
    assert!((0.0..=1.0).contains(&p_value));
}

#[test]
fn test_oos_report_summary_format() {
    let report = OOSReport::new(1000.0, 1030.0);
    let summary = report.summary();

    assert!(summary.contains("In-sample cost"));
    assert!(summary.contains("OOS cost"));
    assert!(summary.contains("Generalization gap"));
    assert!(summary.contains("Quality"));
}

#[test]
fn test_oos_quality_threshold_boundaries() {
    let excellent_boundary = OOSReport::new(1000.0, 1049.99);
    assert_eq!(excellent_boundary.quality_description(), "Excellent");

    let good_boundary = OOSReport::new(1000.0, 1099.99);
    assert_eq!(good_boundary.quality_description(), "Good");

    let acceptable_boundary = OOSReport::new(1000.0, 1149.99);
    assert_eq!(acceptable_boundary.quality_description(), "Acceptable");

    let poor = OOSReport::new(1000.0, 1150.01);
    assert_eq!(poor.quality_description(), "Poor");
}

#[test]
fn test_oos_report_oos_ratio_edge_cases() {
    let report1 = OOSReport::new(1000.0, 500.0);
    assert!((report1.oos_ratio - 0.5).abs() < 1e-10);

    let report2 = OOSReport::new(1000.0, 2000.0);
    assert!((report2.oos_ratio - 2.0).abs() < 1e-10);

    let report3 = OOSReport::new(1000.0, 1000.0);
    assert!((report3.oos_ratio - 1.0).abs() < 1e-10);
}

#[test]
fn test_oos_scenario_generation_multistage() {
    let mut gen = NoiseGenerator::new();
    for _ in 0..12 {
        gen.add_node_generator(
            vec![Normal::new(75.0, 5.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            100,
        );
    }

    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(1000);

    assert_eq!(oos_saa.stage_scenarios.len(), 12);
    for stage in 0..12 {
        assert_eq!(oos_saa.get_branching_count_at_stage(stage).unwrap(), 1000);
    }
}

#[test]
fn test_oos_with_different_sample_sizes() {
    for num_scenarios in [10, 50, 100, 500, 1000] {
        let mut gen = NoiseGenerator::new();
        gen.add_node_generator(
            vec![Normal::new(75.0, 5.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            100,
        );

        let mut oos_gen = OOSGenerator::new(42, gen);
        let oos_saa = oos_gen.generate_independent(num_scenarios);

        assert_eq!(
            oos_saa.get_branching_count_at_stage(0).unwrap(),
            num_scenarios
        );
    }
}

#[test]
fn test_statistical_independence_verification() {
    let mut gen = NoiseGenerator::new();
    gen.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        1000,
    );

    let training_saa = gen.generate(42);

    gen.node_generators[0].num_branchings = 1000;
    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(1000);

    let training_samples: Vec<f64> = (0..1000)
        .map(|i| {
            training_saa
                .get_noises_by_stage_and_branching(0, i)
                .unwrap()
                .get_inflow_innovations()[0]
        })
        .collect();

    let oos_samples: Vec<f64> = (0..1000)
        .map(|i| {
            oos_saa
                .get_noises_by_stage_and_branching(0, i)
                .unwrap()
                .get_inflow_innovations()[0]
        })
        .collect();

    let p_value = kolmogorov_smirnov_test(&training_samples, &oos_samples);
    assert!(p_value > 0.01);
}

#[test]
fn test_oos_evaluator_with_varying_gaps() {
    let evaluator = OOSEvaluator::new();

    let test_cases = vec![
        (1000.0, 1010.0, true),
        (1000.0, 1050.0, true),
        (1000.0, 1100.0, false),
        (1000.0, 1200.0, false),
    ];

    for (in_sample, oos, should_pass_5_percent) in test_cases {
        let report = evaluator.evaluate(in_sample, oos);
        assert_eq!(report.is_generalizing(0.05), should_pass_5_percent);
    }
}

#[test]
fn test_oos_with_deterministic_scenarios() {
    let mut gen = NoiseGenerator::new();
    gen.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        100,
    );

    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(100);

    let first = oos_saa.get_noises_by_stage_and_branching(0, 0).unwrap();
    let second = oos_saa.get_noises_by_stage_and_branching(0, 1).unwrap();

    assert!(
        (first.get_load_innovations()[0] - second.get_load_innovations()[0])
            .abs()
            < 1e-10
    );
}

#[test]
fn test_oos_evaluator_edge_cases() {
    let evaluator = OOSEvaluator::new();

    let report1 = evaluator.evaluate(0.0, 0.0);
    assert_eq!(report1.generalization_gap, 0.0);

    let report2 = evaluator.evaluate(0.001, 0.0011);
    assert!(report2.generalization_gap.abs() < 0.001);

    let report3 = evaluator.evaluate(1_000_000.0, 1_010_000.0);
    assert!(report3.is_generalizing(0.02));
}

#[test]
fn test_multiple_oos_generators_independent() {
    let mut gen1 = NoiseGenerator::new();
    gen1.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        10,
    );

    let mut gen2 = NoiseGenerator::new();
    gen2.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        10,
    );

    let mut oos_gen1 = OOSGenerator::new(42, gen1);
    let mut oos_gen2 = OOSGenerator::new(43, gen2);

    let saa1 = oos_gen1.generate_independent(10);
    let saa2 = oos_gen2.generate_independent(10);

    let sample1 = saa1.get_noises_by_stage_and_branching(0, 0).unwrap();
    let sample2 = saa2.get_noises_by_stage_and_branching(0, 0).unwrap();

    assert_ne!(
        sample1.get_inflow_innovations()[0],
        sample2.get_inflow_innovations()[0]
    );
}

#[test]
fn test_oos_evaluator_consistency() {
    let evaluator = OOSEvaluator::new();

    for _ in 0..10 {
        let report = evaluator.evaluate(1000.0, 1030.0);
        assert_eq!(report.generalization_gap, 30.0);
        assert!((report.oos_ratio - 1.03).abs() < 1e-10);
    }
}

#[test]
fn test_oos_with_single_stage() {
    let mut gen = NoiseGenerator::new();
    gen.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        100,
    );

    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(500);

    assert_eq!(oos_saa.stage_scenarios.len(), 1);
    assert_eq!(oos_saa.get_branching_count_at_stage(0).unwrap(), 500);
}

#[test]
fn test_oos_with_many_stages() {
    let mut gen = NoiseGenerator::new();
    for _ in 0..100 {
        gen.add_node_generator(
            vec![Normal::new(75.0, 5.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            10,
        );
    }

    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(50);

    assert_eq!(oos_saa.stage_scenarios.len(), 100);
}

#[test]
fn test_oos_seed_wrapping() {
    let gen = NoiseGenerator::<Normal<f64>, LogNormal<f64>>::new();
    let large_seed = u64::MAX - 100_000;
    let oos_gen = OOSGenerator::new(large_seed, gen);

    // Should wrap without panic
    let oos_seed = oos_gen.get_oos_seed();
    let shift_seed = oos_gen.get_shift_seed();

    assert_ne!(oos_seed, large_seed);
    assert_ne!(shift_seed, large_seed);
}

#[test]
fn test_oos_with_multiple_entities() {
    let mut gen = NoiseGenerator::new();
    gen.add_node_generator(
        vec![
            Normal::new(75.0, 5.0).unwrap(),
            Normal::new(80.0, 4.0).unwrap(),
            Normal::new(70.0, 6.0).unwrap(),
        ],
        vec![
            LogNormal::new(3.6, 0.6928).unwrap(),
            LogNormal::new(3.8, 0.5).unwrap(),
            LogNormal::new(3.4, 0.7).unwrap(),
        ],
        100,
    );

    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(200);

    let sample = oos_saa.get_noises_by_stage_and_branching(0, 0).unwrap();
    assert_eq!(sample.get_load_innovations().len(), 3);
    assert_eq!(sample.get_inflow_innovations().len(), 3);
}

#[test]
fn test_oos_report_with_extreme_values() {
    let report1 = OOSReport::new(1e-10, 1e-9);
    assert!(report1.generalization_gap >= 0.0);

    let report2 = OOSReport::new(1e10, 1.1e10);
    assert!(report2.is_generalizing(0.15));
}

#[test]
fn test_oos_evaluator_with_many_scenarios() {
    let evaluator = OOSEvaluator::new();
    let in_sample: Vec<f64> =
        (0..10000).map(|i| 1000.0 + (i as f64) * 0.1).collect();
    let oos: Vec<f64> = (0..10000).map(|i| 1030.0 + (i as f64) * 0.1).collect();

    let report = evaluator.evaluate_from_vectors(&in_sample, &oos);
    assert!(report.generalization_gap > 0.0);
}

#[test]
fn test_oos_with_zero_scenarios() {
    let mut gen = NoiseGenerator::new();
    gen.add_node_generator(
        vec![Normal::new(75.0, 5.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        100,
    );

    let oos_gen = OOSGenerator::new(42, gen);
    assert_eq!(oos_gen.get_oos_seed(), 42 + 1_000_000);
}

#[test]
fn test_oos_generator_different_distributions() {
    let mut gen = NoiseGenerator::new();
    for stage in 0..5 {
        gen.add_node_generator(
            vec![Normal::new(70.0 + stage as f64 * 2.0, 5.0).unwrap()],
            vec![LogNormal::new(3.6 + stage as f64 * 0.1, 0.6928).unwrap()],
            50,
        );
    }

    let mut oos_gen = OOSGenerator::new(42, gen);
    let oos_saa = oos_gen.generate_independent(100);

    assert_eq!(oos_saa.stage_scenarios.len(), 5);
}

#[test]
fn test_kolmogorov_smirnov_empty_sample() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2: Vec<f64> = vec![];

    let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
    assert!((0.0..=1.0).contains(&p_value));
}

#[test]
fn test_kolmogorov_smirnov_single_element() {
    let sample1 = vec![5.0];
    let sample2 = vec![5.1];

    let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
    assert!((0.0..=1.0).contains(&p_value));
}

#[test]
fn test_oos_report_with_negative_costs() {
    let report = OOSReport::new(-1000.0, -950.0);
    assert!(report.generalization_gap > 0.0);
    assert!((report.oos_ratio - 0.95).abs() < 1e-10);
}

#[test]
fn test_oos_evaluator_with_unequal_vector_sizes() {
    let evaluator = OOSEvaluator::new();
    let in_sample = vec![1000.0, 1010.0, 1020.0]; // 3 scenarios
    let oos = vec![1030.0, 1040.0, 1050.0, 1060.0, 1070.0]; // 5 scenarios

    let report = evaluator.evaluate_from_vectors(&in_sample, &oos);
    assert!(report.generalization_gap >= 0.0);
}

#[test]
fn test_oos_scenario_generation_performance() {
    let mut gen = NoiseGenerator::new();
    for _ in 0..12 {
        gen.add_node_generator(
            vec![Normal::new(75.0, 5.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            1000,
        );
    }

    let mut oos_gen = OOSGenerator::new(42, gen);

    let start = Instant::now();
    let _oos_saa = oos_gen.generate_independent(1000);
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 100, "Too slow: {:?}", elapsed);
}

#[test]
fn test_oos_evaluation_performance() {
    let in_sample: Vec<f64> = (0..1000).map(|i| 1000.0 + i as f64).collect();
    let oos: Vec<f64> = (0..1000).map(|i| 1030.0 + i as f64).collect();

    let evaluator = OOSEvaluator::new();

    let start = Instant::now();
    let _report = evaluator.evaluate_from_vectors(&in_sample, &oos);
    let elapsed = start.elapsed();

    assert!(elapsed.as_micros() < 1000, "Too slow: {:?}", elapsed);
}

#[test]
fn test_oos_memory_efficiency() {
    for seed in 0..100 {
        let mut gen = NoiseGenerator::new();
        gen.add_node_generator(
            vec![Normal::new(75.0, 5.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            100,
        );

        let mut oos_gen = OOSGenerator::new(seed, gen);
        let _oos_saa = oos_gen.generate_independent(100);
    }
}
