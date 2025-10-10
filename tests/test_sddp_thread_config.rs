use powers_rs::sddp::{SddpAlgorithm, SddpInstanceBuilder};

#[test]
fn test_builder_with_num_threads() {
    // Test that builder's with_num_threads() modifies config
    let sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_num_threads(2)
    .build()
    .expect("build should succeed");

    // We can't directly inspect the config from SddpInstance,
    // but the build succeeded, which means the config is valid
    // The actual thread configuration happens in train()/simulate()
    drop(sddp);
}

#[test]
fn test_train_with_thread_config() {
    // Test that training works with thread configuration
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_num_iterations(2) // Small for speed
    .with_num_forward_passes(1)
    .with_num_threads(2)
    .build()
    .expect("build should succeed");

    // Training should succeed with thread configuration
    let result = sddp.train();
    assert!(
        result.is_ok(),
        "Training should succeed with thread configuration"
    );
}

#[test]
fn test_simulate_with_thread_config() {
    // Test that simulation works with thread configuration
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_num_iterations(2) // Small for speed
    .with_num_forward_passes(1)
    .with_num_threads(2)
    .build()
    .expect("build should succeed");

    // Train first
    sddp.train().expect("Training should succeed");

    // Simulation should succeed with thread configuration
    let result = sddp.simulate();
    assert!(
        result.is_ok(),
        "Simulation should succeed with thread configuration"
    );
}

#[test]
fn test_from_files_with_example_configs() {
    // Test that from_files() works with updated example configs
    // (all examples now have num_threads field)
    let mut sddp = SddpAlgorithm::from_files(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_files should work with updated config");

    // Training should succeed
    let result = sddp.train();
    assert!(result.is_ok(), "Training should succeed");
}

#[test]
fn test_builder_chain_with_num_threads() {
    // Test chaining with_num_threads() with other modifiers
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_num_iterations(2)
    .with_num_forward_passes(1)
    .with_seed(999)
    .with_num_threads(2)
    .build()
    .expect("build should succeed");

    // All modifications should be applied
    let result = sddp.train();
    assert!(
        result.is_ok(),
        "Training should succeed with all modifications"
    );
}
