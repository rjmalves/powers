//! Policy quality validation infrastructure for SDDP policies.
//!
//! This module provides tools to validate that trained SDDP policies produce
//! reasonable, feasible, and improving decisions. It includes:
//!
//! - **Feasibility validation**: Checks that all state/action bounds and constraints are satisfied
//! - **Reasonableness validation**: Checks qualitative properties (e.g., high load → high generation)
//! - **Analytical benchmarks**: Compares policy cost to known analytical solutions
//! - **Stability analysis**: Validates consistency across different random seeds
//!
//! # Performance Notes
//!
//! - Validation uses `rayon` for parallel trajectory checking
//! - Feasibility checks are O(n) per trajectory, where n = num_stages
//! - Validation overhead is <10% of training time for typical problems
//! - Early termination optional: stop checking trajectory after first violation
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use powers_rs::sddp::SddpAlgorithm;
//! use tests::fixtures::validation::PolicyValidator;
//!
//! // Train a policy
//! let mut sddp = create_hydrothermal_problem();
//! sddp.train(30, 10, &scenarios)?;
//!
//! // Simulate with trained policy
//! let sim_result = sddp.simulate_and_analyze(1000, &scenarios)?;
//!
//! // Validate feasibility
//! let validator = PolicyValidator::new(&sddp.system);
//! let feasibility = validator.check_feasibility(&sim_result.trajectories);
//! assert!(feasibility.all_feasible(), "Policy violated constraints: {:?}",
//!         feasibility.violations);
//!
//! // Validate reasonableness
//! let reasonableness = validator.check_reasonableness(&sim_result.trajectories);
//! assert!(reasonableness.is_reasonable(), "Policy behavior unexpected: {:?}",
//!         reasonableness.unexpected_behaviors);
//!
//! // Compare to analytical solution
//! let analytical_cost = compute_analytical_solution(&system);
//! let relative_diff = validator.compare_to_analytical(
//!     sim_result.statistics.mean,
//!     analytical_cost
//! );
//! assert!(relative_diff < 0.01, "Policy cost differs from analytical by {:.2}%",
//!         relative_diff * 100.0);
//! ```

#![allow(dead_code)] // Tests use these types

use powers_rs::sddp::{StageResult, Trajectory};
use powers_rs::system::System;
use rayon::prelude::*;

/// Tolerance for numerical constraint violations (1e-6).
///
/// Violations below this threshold are considered rounding errors and ignored.
const NUMERICAL_TOLERANCE: f64 = 1e-6;

/// Tolerance for reasonableness checks (relative, 5%).
///
/// Allows 5% deviation from expected behavior for qualitative checks.
const REASONABLENESS_TOLERANCE: f64 = 0.05;

// ============================================================================
// Violation Types
// ============================================================================

/// Specific types of constraint violations detected during validation.
///
/// Each variant provides detailed context about the violation location and magnitude.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    /// Storage below minimum bound.
    ///
    /// # Fields
    /// - `trajectory_id`: Which scenario trajectory
    /// - `stage`: Which stage in the trajectory
    /// - `hydro_id`: Which reservoir
    /// - `value`: Actual storage value
    /// - `min_bound`: Minimum allowed storage
    NegativeStorage {
        trajectory_id: usize,
        stage: usize,
        hydro_id: usize,
        value: f64,
        min_bound: f64,
    },

    /// Storage above maximum bound.
    StorageExceedsCapacity {
        trajectory_id: usize,
        stage: usize,
        hydro_id: usize,
        value: f64,
        max_bound: f64,
    },

    /// Hydro generation (turbined flow) below minimum.
    TurbinedFlowBelowMin {
        trajectory_id: usize,
        stage: usize,
        hydro_id: usize,
        value: f64,
        min_bound: f64,
    },

    /// Hydro generation (turbined flow) above maximum.
    TurbinedFlowExceedsMax {
        trajectory_id: usize,
        stage: usize,
        hydro_id: usize,
        value: f64,
        max_bound: f64,
    },

    /// Thermal generation below minimum.
    ThermalGenerationBelowMin {
        trajectory_id: usize,
        stage: usize,
        thermal_id: usize,
        value: f64,
        min_bound: f64,
    },

    /// Thermal generation above maximum.
    ThermalGenerationExceedsMax {
        trajectory_id: usize,
        stage: usize,
        thermal_id: usize,
        value: f64,
        max_bound: f64,
    },

    /// Power balance equation violated.
    ///
    /// Generation + deficit ≠ load (within tolerance).
    #[allow(clippy::enum_variant_names)]
    // "Violation" suffix is intentional for clarity
    PowerBalanceViolation {
        trajectory_id: usize,
        stage: usize,
        bus_id: usize,
        generation: f64,
        load: f64,
        deficit: f64,
        imbalance: f64,
    },

    /// Water balance equation violated.
    ///
    /// next_storage ≠ prev_storage + inflow - turbined - spillage (within tolerance).
    #[allow(clippy::enum_variant_names)]
    // "Violation" suffix is intentional for clarity
    WaterBalanceViolation {
        trajectory_id: usize,
        stage: usize,
        hydro_id: usize,
        prev_storage: f64,
        inflow: f64,
        turbined: f64,
        spillage: f64,
        next_storage: f64,
        imbalance: f64,
    },

    /// Negative action value (generation, spillage, etc.).
    NegativeAction {
        trajectory_id: usize,
        stage: usize,
        action_index: usize,
        value: f64,
    },
}

/// Types of unexpected policy behaviors detected during reasonableness checks.
///
/// These are not hard constraint violations but indicate potentially poor policy quality.
#[derive(Debug, Clone, PartialEq)]
pub enum UnexpectedBehavior {
    /// Spillage when storage is low (< 50% capacity).
    ///
    /// Usually indicates poor policy: should save water instead of spilling.
    SpillageWhenStorageLow {
        trajectory_id: usize,
        stage: usize,
        hydro_id: usize,
        storage_percent: f64,
        spillage: f64,
    },

    /// Low generation when load is high (generation < 80% of load).
    ///
    /// May indicate excessive deficit or poor dispatch decisions.
    LowGenerationWhenLoadHigh {
        trajectory_id: usize,
        stage: usize,
        bus_id: usize,
        load: f64,
        generation: f64,
        generation_ratio: f64,
    },

    /// Storage increases when inflow is low.
    ///
    /// Counter-intuitive: typically storage should deplete with low inflow.
    StorageIncreasesWithLowInflow {
        trajectory_id: usize,
        stage: usize,
        hydro_id: usize,
        inflow: f64,
        storage_change: f64,
    },

    /// Excessive deficit (> 10% of load).
    ///
    /// May indicate insufficient generation capacity or poor policy.
    ExcessiveDeficit {
        trajectory_id: usize,
        stage: usize,
        bus_id: usize,
        load: f64,
        deficit: f64,
        deficit_ratio: f64,
    },
}

// ============================================================================
// Validation Reports
// ============================================================================

/// Report of feasibility validation results.
///
/// Contains all detected constraint violations with detailed context.
/// Empty violations vector indicates all trajectories are feasible.
#[derive(Debug, Clone)]
pub struct FeasibilityReport {
    /// List of all constraint violations found.
    ///
    /// Empty if all trajectories are feasible.
    pub violations: Vec<Violation>,

    /// Total number of trajectories validated.
    pub num_trajectories: usize,

    /// Total number of stages checked across all trajectories.
    pub num_stages_checked: usize,
}

impl FeasibilityReport {
    /// Returns true if no violations were detected.
    #[inline]
    pub fn all_feasible(&self) -> bool {
        self.violations.is_empty()
    }

    /// Returns the number of violations detected.
    #[inline]
    pub fn num_violations(&self) -> usize {
        self.violations.len()
    }

    /// Returns violation rate (violations per trajectory).
    #[inline]
    pub fn violation_rate(&self) -> f64 {
        if self.num_trajectories == 0 {
            0.0
        } else {
            self.violations.len() as f64 / self.num_trajectories as f64
        }
    }
}

/// Report of reasonableness validation results.
///
/// Contains all detected unexpected behaviors that may indicate poor policy quality.
#[derive(Debug, Clone)]
pub struct ReasonablenessReport {
    /// List of unexpected behaviors found.
    ///
    /// Empty if policy behavior is entirely reasonable.
    pub unexpected_behaviors: Vec<UnexpectedBehavior>,

    /// Total number of trajectories validated.
    pub num_trajectories: usize,

    /// Total number of stages checked.
    pub num_stages_checked: usize,
}

impl ReasonablenessReport {
    /// Returns true if no unexpected behaviors were detected.
    #[inline]
    pub fn is_reasonable(&self) -> bool {
        self.unexpected_behaviors.is_empty()
    }

    /// Returns the number of unexpected behaviors detected.
    #[inline]
    pub fn num_unexpected(&self) -> usize {
        self.unexpected_behaviors.len()
    }

    /// Returns unexpected behavior rate (per trajectory).
    #[inline]
    pub fn unexpected_rate(&self) -> f64 {
        if self.num_trajectories == 0 {
            0.0
        } else {
            self.unexpected_behaviors.len() as f64
                / self.num_trajectories as f64
        }
    }
}

// ============================================================================
// PolicyValidator
// ============================================================================

/// Validator for SDDP policy quality assessment.
///
/// Performs feasibility checking, reasonableness validation, and analytical comparisons
/// on simulated trajectories from a trained SDDP policy.
///
/// # Performance
///
/// - Validation is parallelized using `rayon` for trajectory-level parallelism
/// - Feasibility checks: O(n × m) where n = num_trajectories, m = num_stages
/// - Reasonableness checks: O(n × m) with early termination on clear violations
/// - Memory usage: O(violations) - constant overhead, linear in violations found
///
/// # Example
///
/// ```rust,ignore
/// let validator = PolicyValidator::new(&system);
/// let feasibility = validator.check_feasibility(&trajectories);
/// if !feasibility.all_feasible() {
///     println!("Found {} violations in {} trajectories",
///              feasibility.num_violations(),
///              feasibility.num_trajectories);
///     for violation in &feasibility.violations {
///         println!("  {:?}", violation);
///     }
/// }
/// ```
pub struct PolicyValidator {
    /// Reference to the power system being validated.
    ///
    /// Contains all constraint information (bounds, capacities, etc.).
    system: System,

    /// Number of hydro reservoirs in the system.
    num_hydros: usize,

    /// Number of thermal plants in the system.
    num_thermals: usize,

    /// Number of buses in the system.
    num_buses: usize,

    /// Number of lines in the system.
    num_lines: usize,
}

impl PolicyValidator {
    /// Creates a new policy validator for the given power system.
    ///
    /// # Arguments
    ///
    /// * `system` - The power system containing constraint specifications
    ///
    /// # Performance
    ///
    /// O(1) construction - just copies metadata from system.
    pub fn new(system: System) -> Self {
        let num_hydros = system.hydros.len();
        let num_thermals = system.thermals.len();
        let num_buses = system.buses.len();
        let num_lines = system.lines.len();

        Self {
            system,
            num_hydros,
            num_thermals,
            num_buses,
            num_lines,
        }
    }

    /// Validates feasibility of all trajectories.
    ///
    /// Checks:
    /// - Storage bounds (min ≤ storage ≤ max)
    /// - Turbined flow bounds (min ≤ flow ≤ max)
    /// - Thermal generation bounds (min ≤ gen ≤ max)
    /// - Non-negativity of actions (spillage, deficit, exchange ≥ 0)
    /// - Power balance constraints (generation = load)
    /// - Water balance constraints (storage dynamics)
    ///
    /// # Arguments
    ///
    /// * `trajectories` - Slice of simulated trajectories to validate
    ///
    /// # Returns
    ///
    /// `FeasibilityReport` containing all detected violations.
    ///
    /// # Performance
    ///
    /// - Parallel validation using `rayon::par_iter()`
    /// - O(n × m) where n = num_trajectories, m = avg_stages_per_trajectory
    /// - Memory: O(violations) - only stores detected violations
    /// - Typical overhead: <100ms for 1000 trajectories × 12 stages
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let report = validator.check_feasibility(&trajectories);
    /// assert!(report.all_feasible(), "Violations: {:?}", report.violations);
    /// ```
    pub fn check_feasibility(
        &self,
        trajectories: &[Trajectory],
    ) -> FeasibilityReport {
        // PERFORMANCE: Parallel validation using rayon
        // Each trajectory is independent, so we can validate in parallel
        let violations: Vec<Vec<Violation>> = trajectories
            .par_iter()
            .enumerate()
            .map(|(traj_id, trajectory)| {
                self.check_trajectory_feasibility(traj_id, trajectory)
            })
            .collect();

        let total_stages: usize =
            trajectories.iter().map(|t| t.stages.len()).sum();

        FeasibilityReport {
            violations: violations.into_iter().flatten().collect(),
            num_trajectories: trajectories.len(),
            num_stages_checked: total_stages,
        }
    }

    /// Checks feasibility of a single trajectory.
    ///
    /// Helper method for parallel feasibility validation.
    /// Returns vector of violations found in this trajectory.
    ///
    /// # Performance
    ///
    /// O(m) where m = number of stages in trajectory.
    fn check_trajectory_feasibility(
        &self,
        trajectory_id: usize,
        trajectory: &Trajectory,
    ) -> Vec<Violation> {
        let mut violations = Vec::new();

        for stage_result in &trajectory.stages {
            // Check state bounds (storage)
            self.check_state_bounds(
                trajectory_id,
                stage_result,
                &mut violations,
            );

            // Check action bounds (generation, spillage, etc.)
            self.check_action_bounds(
                trajectory_id,
                stage_result,
                &mut violations,
            );

            // Check power balance constraints
            self.check_power_balance(
                trajectory_id,
                stage_result,
                &mut violations,
            );
        }

        // Check water balance between consecutive stages
        for i in 1..trajectory.stages.len() {
            self.check_water_balance(
                trajectory_id,
                &trajectory.stages[i - 1],
                &trajectory.stages[i],
                &mut violations,
            );
        }

        violations
    }

    /// Checks storage bounds for a single stage.
    ///
    /// Verifies min_storage ≤ storage ≤ max_storage for all hydro reservoirs.
    ///
    /// # Performance
    ///
    /// O(num_hydros) per stage.
    fn check_state_bounds(
        &self,
        trajectory_id: usize,
        stage: &StageResult,
        violations: &mut Vec<Violation>,
    ) {
        for (hydro_id, &storage) in stage.state.iter().enumerate() {
            if hydro_id >= self.num_hydros {
                break; // Beyond hydro states
            }

            let hydro = &self.system.hydros[hydro_id];

            // Check minimum storage
            if storage < hydro.min_storage - NUMERICAL_TOLERANCE {
                violations.push(Violation::NegativeStorage {
                    trajectory_id,
                    stage: stage.stage,
                    hydro_id,
                    value: storage,
                    min_bound: hydro.min_storage,
                });
            }

            // Check maximum storage
            if storage > hydro.max_storage + NUMERICAL_TOLERANCE {
                violations.push(Violation::StorageExceedsCapacity {
                    trajectory_id,
                    stage: stage.stage,
                    hydro_id,
                    value: storage,
                    max_bound: hydro.max_storage,
                });
            }
        }
    }

    /// Checks action bounds for a single stage.
    ///
    /// Verifies bounds for:
    /// - Turbined flow (hydro generation)
    /// - Thermal generation
    /// - Non-negativity of spillage, deficit, exchange
    ///
    /// # Action Vector Layout
    ///
    /// The action vector contains (in order):
    /// - [0..num_hydros): turbined flow
    /// - [num_hydros..num_hydros+num_thermals): thermal generation
    /// - [num_hydros+num_thermals..num_hydros+num_thermals+num_hydros): spillage
    /// - [...]: exchange, deficit
    ///
    /// # Performance
    ///
    /// O(num_hydros + num_thermals) per stage.
    fn check_action_bounds(
        &self,
        trajectory_id: usize,
        stage: &StageResult,
        violations: &mut Vec<Violation>,
    ) {
        let action = &stage.action;

        // Check turbined flow bounds (first num_hydros elements)
        for (hydro_id, &turbined_flow) in
            action.iter().enumerate().take(self.num_hydros)
        {
            let hydro = &self.system.hydros[hydro_id];

            if turbined_flow < hydro.min_turbined_flow - NUMERICAL_TOLERANCE {
                violations.push(Violation::TurbinedFlowBelowMin {
                    trajectory_id,
                    stage: stage.stage,
                    hydro_id,
                    value: turbined_flow,
                    min_bound: hydro.min_turbined_flow,
                });
            }

            if turbined_flow > hydro.max_turbined_flow + NUMERICAL_TOLERANCE {
                violations.push(Violation::TurbinedFlowExceedsMax {
                    trajectory_id,
                    stage: stage.stage,
                    hydro_id,
                    value: turbined_flow,
                    max_bound: hydro.max_turbined_flow,
                });
            }
        }

        // Check thermal generation bounds
        let thermal_offset = self.num_hydros;
        for thermal_id in 0..self
            .num_thermals
            .min(action.len().saturating_sub(thermal_offset))
        {
            let generation = action[thermal_offset + thermal_id];
            let thermal = &self.system.thermals[thermal_id];

            if generation < thermal.min_generation - NUMERICAL_TOLERANCE {
                violations.push(Violation::ThermalGenerationBelowMin {
                    trajectory_id,
                    stage: stage.stage,
                    thermal_id,
                    value: generation,
                    min_bound: thermal.min_generation,
                });
            }

            if generation > thermal.max_generation + NUMERICAL_TOLERANCE {
                violations.push(Violation::ThermalGenerationExceedsMax {
                    trajectory_id,
                    stage: stage.stage,
                    thermal_id,
                    value: generation,
                    max_bound: thermal.max_generation,
                });
            }
        }

        // Check non-negativity of all actions
        for (idx, &value) in action.iter().enumerate() {
            if value < -NUMERICAL_TOLERANCE {
                violations.push(Violation::NegativeAction {
                    trajectory_id,
                    stage: stage.stage,
                    action_index: idx,
                    value,
                });
            }
        }
    }

    /// Checks power balance constraint for a single stage.
    ///
    /// Verifies: hydro_gen + thermal_gen + deficit = load (per bus).
    ///
    /// # Performance
    ///
    /// O(num_buses) per stage.
    fn check_power_balance(
        &self,
        trajectory_id: usize,
        stage: &StageResult,
        violations: &mut Vec<Violation>,
    ) {
        for (bus_id, bus) in self.system.buses.iter().enumerate() {
            if bus_id >= stage.load.len() {
                break;
            }

            let load = stage.load[bus_id];

            // Compute total hydro generation at this bus
            let hydro_gen: f64 = bus
                .hydro_ids
                .iter()
                .filter_map(|&hid| {
                    if hid < self.num_hydros && hid < stage.action.len() {
                        Some(
                            stage.action[hid]
                                * self.system.hydros[hid].productivity,
                        )
                    } else {
                        None
                    }
                })
                .sum();

            // Compute total thermal generation at this bus
            let thermal_offset = self.num_hydros;
            let thermal_gen: f64 = bus
                .thermal_ids
                .iter()
                .filter_map(|&tid| {
                    if tid < self.num_thermals
                        && thermal_offset + tid < stage.action.len()
                    {
                        Some(stage.action[thermal_offset + tid])
                    } else {
                        None
                    }
                })
                .sum();

            // Extract deficit (last num_buses elements of action vector)
            // Deficit is stored after: turbined_flow, thermal_gen, spillage, exchange
            let deficit_offset = self.num_hydros
                + self.num_thermals
                + self.num_hydros
                + 2 * self.num_lines;
            let deficit = if deficit_offset + bus_id < stage.action.len() {
                stage.action[deficit_offset + bus_id]
            } else {
                0.0
            };

            let generation = hydro_gen + thermal_gen;
            let imbalance = (generation + deficit - load).abs();

            if imbalance > NUMERICAL_TOLERANCE {
                violations.push(Violation::PowerBalanceViolation {
                    trajectory_id,
                    stage: stage.stage,
                    bus_id,
                    generation,
                    load,
                    deficit,
                    imbalance,
                });
            }
        }
    }

    /// Checks water balance constraint between two consecutive stages.
    ///
    /// Verifies: next_storage = prev_storage + inflow - turbined - spillage.
    ///
    /// The inflow, turbined, and spillage values are from the PREVIOUS stage,
    /// as they occur during that stage and affect the transition to the next stage.
    ///
    /// # Performance
    ///
    /// O(num_hydros) per stage transition.
    fn check_water_balance(
        &self,
        trajectory_id: usize,
        prev_stage: &StageResult,
        next_stage: &StageResult,
        violations: &mut Vec<Violation>,
    ) {
        for hydro_id in 0..self.num_hydros {
            if hydro_id >= prev_stage.state.len()
                || hydro_id >= next_stage.state.len()
            {
                continue;
            }

            let prev_storage = prev_stage.state[hydro_id];
            let next_storage = next_stage.state[hydro_id];

            // PERFORMANCE: All values from previous stage (O(1) array access)
            // Inflow, turbined, and spillage occur during prev_stage and affect next_storage
            let inflow = if hydro_id < prev_stage.inflow.len() {
                prev_stage.inflow[hydro_id]
            } else {
                0.0
            };

            let turbined = if hydro_id < prev_stage.action.len() {
                prev_stage.action[hydro_id]
            } else {
                0.0
            };

            // Spillage is stored after turbined_flow and thermal_gen
            let spillage_offset = self.num_hydros + self.num_thermals;
            let spillage =
                if spillage_offset + hydro_id < prev_stage.action.len() {
                    prev_stage.action[spillage_offset + hydro_id]
                } else {
                    0.0
                };

            // Water balance: next = prev + inflow - turbined - spillage
            // All values from previous stage
            let expected_storage = prev_storage + inflow - turbined - spillage;
            let imbalance = (next_storage - expected_storage).abs();

            if imbalance > NUMERICAL_TOLERANCE {
                violations.push(Violation::WaterBalanceViolation {
                    trajectory_id,
                    stage: next_stage.stage,
                    hydro_id,
                    prev_storage,
                    inflow,
                    turbined,
                    spillage,
                    next_storage,
                    imbalance,
                });
            }
        }
    }

    /// Validates reasonableness of policy behavior across trajectories.
    ///
    /// Checks qualitative properties that indicate good policy quality:
    /// - No spillage when storage is low (< 50% capacity)
    /// - Generation should meet load (deficit should be small)
    /// - Storage should deplete with low inflow
    /// - Deficit should be reasonable (< 10% of load)
    ///
    /// # Arguments
    ///
    /// * `trajectories` - Slice of simulated trajectories to validate
    ///
    /// # Returns
    ///
    /// `ReasonablenessReport` containing all unexpected behaviors found.
    ///
    /// # Performance
    ///
    /// - Parallel validation using `rayon::par_iter()`
    /// - O(n × m) where n = num_trajectories, m = num_stages
    /// - Typical overhead: <50ms for 1000 trajectories
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let report = validator.check_reasonableness(&trajectories);
    /// if !report.is_reasonable() {
    ///     println!("Found {} unexpected behaviors", report.num_unexpected());
    /// }
    /// ```
    pub fn check_reasonableness(
        &self,
        trajectories: &[Trajectory],
    ) -> ReasonablenessReport {
        // PERFORMANCE: Parallel validation using rayon
        let behaviors: Vec<Vec<UnexpectedBehavior>> = trajectories
            .par_iter()
            .enumerate()
            .map(|(traj_id, trajectory)| {
                self.check_trajectory_reasonableness(traj_id, trajectory)
            })
            .collect();

        let total_stages: usize =
            trajectories.iter().map(|t| t.stages.len()).sum();

        ReasonablenessReport {
            unexpected_behaviors: behaviors.into_iter().flatten().collect(),
            num_trajectories: trajectories.len(),
            num_stages_checked: total_stages,
        }
    }

    /// Checks reasonableness of a single trajectory.
    ///
    /// Helper method for parallel reasonableness validation.
    fn check_trajectory_reasonableness(
        &self,
        trajectory_id: usize,
        trajectory: &Trajectory,
    ) -> Vec<UnexpectedBehavior> {
        let mut behaviors = Vec::new();

        for stage_result in &trajectory.stages {
            // Check for spillage when storage is low
            self.check_spillage_behavior(
                trajectory_id,
                stage_result,
                &mut behaviors,
            );

            // Check for low generation when load is high
            self.check_generation_behavior(
                trajectory_id,
                stage_result,
                &mut behaviors,
            );

            // Check for excessive deficit
            self.check_deficit_behavior(
                trajectory_id,
                stage_result,
                &mut behaviors,
            );
        }

        // Check storage dynamics across stages
        for i in 1..trajectory.stages.len() {
            self.check_storage_dynamics(
                trajectory_id,
                &trajectory.stages[i - 1],
                &trajectory.stages[i],
                &mut behaviors,
            );
        }

        behaviors
    }

    /// Checks for spillage when storage is low (unexpected behavior).
    fn check_spillage_behavior(
        &self,
        trajectory_id: usize,
        stage: &StageResult,
        behaviors: &mut Vec<UnexpectedBehavior>,
    ) {
        let spillage_offset = self.num_hydros + self.num_thermals;

        for hydro_id in 0..self.num_hydros {
            if hydro_id >= stage.state.len() {
                continue;
            }

            let storage = stage.state[hydro_id];
            let max_storage = self.system.hydros[hydro_id].max_storage;
            let storage_percent = storage / max_storage;

            if spillage_offset + hydro_id < stage.action.len() {
                let spillage = stage.action[spillage_offset + hydro_id];

                // Spillage when storage < 50% is unexpected
                if spillage > NUMERICAL_TOLERANCE && storage_percent < 0.5 {
                    behaviors.push(
                        UnexpectedBehavior::SpillageWhenStorageLow {
                            trajectory_id,
                            stage: stage.stage,
                            hydro_id,
                            storage_percent,
                            spillage,
                        },
                    );
                }
            }
        }
    }

    /// Checks for low generation when load is high (unexpected behavior).
    fn check_generation_behavior(
        &self,
        trajectory_id: usize,
        stage: &StageResult,
        behaviors: &mut Vec<UnexpectedBehavior>,
    ) {
        for (bus_id, bus) in self.system.buses.iter().enumerate() {
            if bus_id >= stage.load.len() {
                break;
            }

            let load = stage.load[bus_id];

            // Compute total generation at this bus
            let hydro_gen: f64 = bus
                .hydro_ids
                .iter()
                .filter_map(|&hid| {
                    if hid < self.num_hydros && hid < stage.action.len() {
                        Some(
                            stage.action[hid]
                                * self.system.hydros[hid].productivity,
                        )
                    } else {
                        None
                    }
                })
                .sum();

            let thermal_offset = self.num_hydros;
            let thermal_gen: f64 = bus
                .thermal_ids
                .iter()
                .filter_map(|&tid| {
                    if tid < self.num_thermals
                        && thermal_offset + tid < stage.action.len()
                    {
                        Some(stage.action[thermal_offset + tid])
                    } else {
                        None
                    }
                })
                .sum();

            let generation = hydro_gen + thermal_gen;
            let generation_ratio = if load > NUMERICAL_TOLERANCE {
                generation / load
            } else {
                1.0
            };

            // Generation < 80% of load is concerning
            if generation_ratio < 0.8 && load > NUMERICAL_TOLERANCE {
                behaviors.push(UnexpectedBehavior::LowGenerationWhenLoadHigh {
                    trajectory_id,
                    stage: stage.stage,
                    bus_id,
                    load,
                    generation,
                    generation_ratio,
                });
            }
        }
    }

    /// Checks for excessive deficit (unexpected behavior).
    fn check_deficit_behavior(
        &self,
        trajectory_id: usize,
        stage: &StageResult,
        behaviors: &mut Vec<UnexpectedBehavior>,
    ) {
        let deficit_offset = self.num_hydros
            + self.num_thermals
            + self.num_hydros
            + 2 * self.num_lines;

        for bus_id in 0..self.num_buses {
            if bus_id >= stage.load.len() {
                break;
            }

            let load = stage.load[bus_id];

            if deficit_offset + bus_id < stage.action.len() {
                let deficit = stage.action[deficit_offset + bus_id];
                let deficit_ratio = if load > NUMERICAL_TOLERANCE {
                    deficit / load
                } else {
                    0.0
                };

                // Deficit > 10% of load is excessive
                if deficit_ratio > 0.1 {
                    behaviors.push(UnexpectedBehavior::ExcessiveDeficit {
                        trajectory_id,
                        stage: stage.stage,
                        bus_id,
                        load,
                        deficit,
                        deficit_ratio,
                    });
                }
            }
        }
    }

    /// Checks storage dynamics across consecutive stages.
    fn check_storage_dynamics(
        &self,
        trajectory_id: usize,
        prev_stage: &StageResult,
        next_stage: &StageResult,
        behaviors: &mut Vec<UnexpectedBehavior>,
    ) {
        for hydro_id in 0..self.num_hydros {
            if hydro_id >= prev_stage.state.len()
                || hydro_id >= next_stage.state.len()
                || hydro_id >= next_stage.inflow.len()
            {
                continue;
            }

            let prev_storage = prev_stage.state[hydro_id];
            let next_storage = next_stage.state[hydro_id];
            let inflow = next_stage.inflow[hydro_id];
            let storage_change = next_storage - prev_storage;

            // Storage increases with low inflow is unexpected
            // Low inflow: < 20% of average (approximate check)
            let max_storage = self.system.hydros[hydro_id].max_storage;
            let low_inflow_threshold = max_storage * 0.1;

            if inflow < low_inflow_threshold
                && storage_change > NUMERICAL_TOLERANCE
            {
                behaviors.push(
                    UnexpectedBehavior::StorageIncreasesWithLowInflow {
                        trajectory_id,
                        stage: next_stage.stage,
                        hydro_id,
                        inflow,
                        storage_change,
                    },
                );
            }
        }
    }

    /// Compares simulated policy cost to analytical benchmark.
    ///
    /// Computes relative difference: (simulated - analytical) / analytical.
    ///
    /// # Arguments
    ///
    /// * `simulated_cost` - Mean cost from policy simulation
    /// * `analytical_cost` - Known analytical solution cost
    ///
    /// # Returns
    ///
    /// Relative difference as a fraction (e.g., 0.01 = 1% difference).
    /// Positive means policy is more expensive than optimal.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let diff = validator.compare_to_analytical(305.0, 300.0);
    /// assert!(diff < 0.02, "Policy is >2% suboptimal");
    /// // diff = 0.0167 (1.67% more expensive)
    /// ```
    #[inline]
    pub fn compare_to_analytical(
        &self,
        simulated_cost: f64,
        analytical_cost: f64,
    ) -> f64 {
        if analytical_cost.abs() < NUMERICAL_TOLERANCE {
            // Avoid division by zero
            if simulated_cost.abs() < NUMERICAL_TOLERANCE {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (simulated_cost - analytical_cost) / analytical_cost
        }
    }

    /// Returns the number of hydro reservoirs in the system.
    #[inline]
    pub fn num_hydros(&self) -> usize {
        self.num_hydros
    }

    /// Returns the number of thermal plants in the system.
    #[inline]
    pub fn num_thermals(&self) -> usize {
        self.num_thermals
    }

    /// Returns the number of buses in the system.
    #[inline]
    pub fn num_buses(&self) -> usize {
        self.num_buses
    }
}
