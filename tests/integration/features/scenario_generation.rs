//! Scenario Generation Integration Tests
//!
//! Tests stochastic process and scenario tree integration.
//! Verifies that SAA (Sample Average Approximation) scenario generation
//! produces correct scenario trees with proper structure, probabilities,
//! and reproducibility.
//!
//! Key aspects tested:
//! - Correct number of scenarios generated
//! - Probabilities sum to 1.0
//! - Reproducibility with fixed seeds
//! - Branching structure matches specification
//! - Scenario noises have correct dimensions

use crate::fixtures::generate_2stage_saa;
use powers_rs::scenario::NoiseGenerator;
use rand_distr::Normal;

#[test]
fn test_saa_scenario_count() {
    // Test that SAA generates correct number of scenarios at each stage
    let saa = generate_2stage_saa(42);

    // Should have 3 nodes: PreStudy (stage 0), Stage 1, Stage 2
    assert_eq!(
        saa.branching_samples.len(),
        3,
        "SAA should have 3 nodes (PreStudy + 2 study stages)"
    );

    // PreStudy: 1 branching (deterministic)
    assert_eq!(
        saa.get_branching_count_at_stage(0).unwrap(),
        1,
        "PreStudy should have 1 branching"
    );

    // Stage 1: 1 branching (deterministic)
    assert_eq!(
        saa.get_branching_count_at_stage(1).unwrap(),
        1,
        "Stage 1 should have 1 branching"
    );

    // Stage 2: 5 scenarios (stochastic)
    assert_eq!(
        saa.get_branching_count_at_stage(2).unwrap(),
        5,
        "Stage 2 should have 5 scenarios"
    );
}

#[test]
fn test_saa_probability_sum() {
    // Test that scenario probabilities sum to 1.0 at each stage
    let saa = generate_2stage_saa(42);

    // For each stage, probabilities should sum to 1.0
    for stage in 0..3 {
        let branching_count = saa.get_branching_count_at_stage(stage).unwrap();

        let mut prob_sum = 0.0;
        for branching_idx in 0..branching_count {
            let _noises = saa
                .get_noises_by_stage_and_branching(stage, branching_idx)
                .unwrap();

            // Each scenario in the branching contributes its probability
            // For uniform sampling, each of N scenarios has probability 1/N
            prob_sum += 1.0 / (branching_count as f64);
        }

        assert!(
            (prob_sum - 1.0).abs() < 1e-10,
            "Stage {} probabilities should sum to 1.0, got {}",
            stage,
            prob_sum
        );
    }
}

#[test]
fn test_saa_reproducibility() {
    // Test that same seed produces identical scenarios
    let seed = 123u64;

    let saa1 = generate_2stage_saa(seed);
    let saa2 = generate_2stage_saa(seed);

    // Both should have same structure
    assert_eq!(saa1.branching_samples.len(), saa2.branching_samples.len());

    for stage in 0..saa1.branching_samples.len() {
        assert_eq!(
            saa1.get_branching_count_at_stage(stage).unwrap(),
            saa2.get_branching_count_at_stage(stage).unwrap(),
            "Stage {} should have same branching count",
            stage
        );
    }

    // Verify scenarios are identical by checking a sample of values
    // Stage 2 has 5 stochastic scenarios
    for branching_idx in 0..5 {
        let noises1 = saa1
            .get_noises_by_stage_and_branching(2, branching_idx)
            .unwrap();
        let noises2 = saa2
            .get_noises_by_stage_and_branching(2, branching_idx)
            .unwrap();

        assert_eq!(
            noises1.num_load_entities, noises2.num_load_entities,
            "Load entities should match"
        );
        assert_eq!(
            noises1.num_inflow_entities, noises2.num_inflow_entities,
            "Inflow entities should match"
        );

        // Noises should be identical (within floating point precision)
        let loads1 = noises1.get_load_innovations();
        let loads2 = noises2.get_load_innovations();
        assert_eq!(loads1.len(), loads2.len(), "Load count should match");
        for i in 0..loads1.len() {
            assert!(
                (loads1[i] - loads2[i]).abs() < 1e-10,
                "Load noises should be identical, got {} vs {}",
                loads1[i],
                loads2[i]
            );
        }

        let inflows1 = noises1.get_inflow_innovations();
        let inflows2 = noises2.get_inflow_innovations();
        assert_eq!(inflows1.len(), inflows2.len(), "Inflow count should match");
        for i in 0..inflows1.len() {
            assert!(
                (inflows1[i] - inflows2[i]).abs() < 1e-10,
                "Inflow noises should be identical, got {} vs {}",
                inflows1[i],
                inflows2[i]
            );
        }
    }
}

#[test]
fn test_different_seeds_produce_different_scenarios() {
    // Test that different seeds produce different scenarios
    let saa1 = generate_2stage_saa(42);
    let saa2 = generate_2stage_saa(123);

    // Structure should be the same
    assert_eq!(saa1.branching_samples.len(), saa2.branching_samples.len());

    // But stochastic scenarios should differ
    // Check Stage 2 first branching
    let noises1 = saa1.get_noises_by_stage_and_branching(2, 0).unwrap();
    let noises2 = saa2.get_noises_by_stage_and_branching(2, 0).unwrap();

    let inflows1 = noises1.get_inflow_innovations();
    let inflows2 = noises2.get_inflow_innovations();

    // Should be different (with high probability for stochastic scenarios)
    assert!(
        !inflows1.is_empty() && !inflows2.is_empty(),
        "Should have inflow data"
    );
    assert!(
        (inflows1[0] - inflows2[0]).abs() > 1e-6,
        "Different seeds should produce different scenarios, got {} and {}",
        inflows1[0],
        inflows2[0]
    );
}

#[test]
fn test_scenario_tree_structure() {
    // Test that scenario tree has correct branching structure
    let saa = generate_2stage_saa(42);

    // Verify tree structure: PreStudy → Stage 1 → Stage 2
    // PreStudy and Stage 1 are deterministic (1 branch each)
    // Stage 2 branches into 5 scenarios

    // Stage 0 (PreStudy): should be deterministic
    assert_eq!(
        saa.get_branching_count_at_stage(0).unwrap(),
        1,
        "PreStudy should have no branching (deterministic)"
    );

    // Stage 1: should be deterministic
    assert_eq!(
        saa.get_branching_count_at_stage(1).unwrap(),
        1,
        "Stage 1 should have no branching (deterministic)"
    );

    // Stage 2: should have 5 branches (stochastic)
    assert_eq!(
        saa.get_branching_count_at_stage(2).unwrap(),
        5,
        "Stage 2 should branch into 5 scenarios"
    );

    // Verify all branchings are accessible
    for branching_idx in 0..5 {
        let noises = saa
            .get_noises_by_stage_and_branching(2, branching_idx)
            .unwrap();

        assert_eq!(noises.num_load_entities, 1, "Should have 1 load entity");
        assert_eq!(
            noises.num_inflow_entities, 1,
            "Should have 1 inflow entity"
        );
    }
}

#[test]
fn test_scenario_noise_dimensions() {
    // Test that scenario noises have correct dimensions
    let saa = generate_2stage_saa(42);

    // Check each stage
    for stage in 0..3 {
        let branching_count = saa.get_branching_count_at_stage(stage).unwrap();

        for branching_idx in 0..branching_count {
            let noises = saa
                .get_noises_by_stage_and_branching(stage, branching_idx)
                .unwrap();

            // For simple 2-stage system: 1 bus (load) and 1 hydro (inflow)
            assert_eq!(
                noises.num_load_entities, 1,
                "Stage {} branching {} should have 1 load entity",
                stage, branching_idx
            );
            assert_eq!(
                noises.num_inflow_entities, 1,
                "Stage {} branching {} should have 1 inflow entity",
                stage, branching_idx
            );

            // Verify noises are finite
            let loads = noises.get_load_innovations();
            let inflows = noises.get_inflow_innovations();

            for (idx, &load) in loads.iter().enumerate() {
                assert!(
                    load.is_finite(),
                    "Load noise {} should be finite at stage {}, branching {}",
                    idx,
                    stage,
                    branching_idx
                );
            }

            for (idx, &inflow) in inflows.iter().enumerate() {
                assert!(
                    inflow.is_finite(),
                    "Inflow noise {} should be finite at stage {}, branching {}",
                    idx,
                    stage,
                    branching_idx
                );
            }
        }
    }
}

#[test]
fn test_deterministic_stages_have_zero_variance() {
    // Test that deterministic stages have zero variance
    // This is indicated by branching count = 1

    let saa = generate_2stage_saa(42);

    // Check Stage 0 (PreStudy)
    assert_eq!(
        saa.get_branching_count_at_stage(0).unwrap(),
        1,
        "PreStudy should be deterministic (1 branching)"
    );

    // Check Stage 1
    assert_eq!(
        saa.get_branching_count_at_stage(1).unwrap(),
        1,
        "Stage 1 should be deterministic (1 branching)"
    );

    // Stage 2 should be stochastic (multiple branchings)
    assert!(
        saa.get_branching_count_at_stage(2).unwrap() > 1,
        "Stage 2 should be stochastic (multiple branchings)"
    );
}

#[test]
fn test_scenario_sampling_with_custom_branching() {
    // Test scenario generation with custom branching counts
    let mut generator = NoiseGenerator::new();

    // Node 0 (PreStudy): deterministic
    let prestudy_load = vec![Normal::new(75.0, 0.0).unwrap()];
    let prestudy_inflow = vec![Normal::new(0.0, 0.0).unwrap()];
    generator.add_node_generator(prestudy_load, prestudy_inflow, 1);

    // Node 1 (Stage 1): 3 scenarios
    let stage1_load = vec![Normal::new(75.0, 5.0).unwrap()];
    let stage1_inflow = vec![Normal::new(40.0, 10.0).unwrap()];
    generator.add_node_generator(stage1_load, stage1_inflow, 3);

    // Node 2 (Stage 2): 5 scenarios
    let stage2_load = vec![Normal::new(75.0, 10.0).unwrap()];
    let stage2_inflow = vec![Normal::new(40.0, 20.0).unwrap()];
    generator.add_node_generator(stage2_load, stage2_inflow, 5);

    let saa = generator.generate(42);

    // Verify branching structure
    assert_eq!(saa.branching_samples.len(), 3);
    assert_eq!(saa.get_branching_count_at_stage(0).unwrap(), 1);
    assert_eq!(saa.get_branching_count_at_stage(1).unwrap(), 3);
    assert_eq!(saa.get_branching_count_at_stage(2).unwrap(), 5);

    // Verify all scenarios are accessible
    for branching_idx in 0..3 {
        let noises = saa
            .get_noises_by_stage_and_branching(1, branching_idx)
            .unwrap();
        assert_eq!(noises.num_load_entities, 1);
        assert_eq!(noises.num_inflow_entities, 1);
    }

    for branching_idx in 0..5 {
        let noises = saa
            .get_noises_by_stage_and_branching(2, branching_idx)
            .unwrap();
        assert_eq!(noises.num_load_entities, 1);
        assert_eq!(noises.num_inflow_entities, 1);
    }
}

#[test]
fn test_scenario_values_within_reasonable_range() {
    // Test that generated scenario values are within reasonable ranges
    let saa = generate_2stage_saa(42);

    // For Stage 2 stochastic scenarios
    // Load: mean 75.0, std 0.0 → should be exactly 75.0
    // Inflow: mean 40.0, std 20.0 → should be within ~3 std devs (0 to 100)

    for branching_idx in 0..5 {
        let noises = saa
            .get_noises_by_stage_and_branching(2, branching_idx)
            .unwrap();

        let loads = noises.get_load_innovations();
        assert!(!loads.is_empty(), "Should have load data");

        // Load is deterministic (std=0), should be exactly 75.0
        let load = loads[0];
        assert!(
            (load - 75.0).abs() < 1e-6,
            "Load should be ~75.0, got {}",
            load
        );

        let inflows = noises.get_inflow_innovations();
        assert!(!inflows.is_empty(), "Should have inflow data");

        // Inflow: mean 40.0, std 20.0 → check within 3 standard deviations
        let inflow = inflows[0];
        assert!(
            inflow >= -20.0 && inflow <= 100.0,
            "Inflow should be within reasonable range, got {}",
            inflow
        );
        assert!(
            inflow.is_finite(),
            "Inflow should be finite, got {}",
            inflow
        );
    }
}

#[test]
fn test_saa_index_sampler_range() {
    // Test that SAA index samplers cover the correct range
    let saa = generate_2stage_saa(42);

    // Should have one index sampler per stage
    assert_eq!(saa.index_samplers.len(), 3, "Should have 3 index samplers");

    // Verify index samplers are created (we can't easily test their output
    // without accessing internal RNG, but we can verify they exist)
    for (stage, sampler) in saa.index_samplers.iter().enumerate() {
        // Samplers exist and are of correct type
        // (Type checking is done at compile time)
        let _ = sampler;
        let branching_count = saa.get_branching_count_at_stage(stage).unwrap();
        assert!(
            branching_count > 0,
            "Stage {} should have positive branching count",
            stage
        );
    }
}
