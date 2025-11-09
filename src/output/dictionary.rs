//! Variable dictionary system for indexed output.
//!
//! This module provides a compile-time registry of output variables with runtime
//! dictionary generation for mapping variable indices to names and metadata.
//!
//! The dictionary enables:
//! - Compact indexed output (variable_index instead of variable_name strings)
//! - Type safety through explicit enumeration
//! - Automatic metadata generation (units, descriptions, entity types)
//! - Consistency across output files

use crate::system::System;
use csv::Writer;
use serde::{Deserialize, Serialize};
use std::error::Error;

/// Output variable types with explicit indices for stable mapping.
///
/// Each variant represents a distinct output variable. The discriminant values
/// are explicitly set to ensure stable indices across versions.
///
/// # Index Stability
///
/// Variable indices must remain stable across software versions to ensure
/// compatibility with historical output files. New variables should be added
/// at the end with new indices.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputVariable {
    /// Initial reservoir storage at stage start (MWh)
    InitialStorage = 0,
    /// Historical inflow lag values (m³/s)
    InflowLag = 1,
    /// Sampled load demand realization (MW)
    SampledLoad = 2,
    /// Sampled inflow realization (m³/s)
    SampledInflow = 3,
    /// Final reservoir storage at stage end (MWh)
    FinalStorage = 4,
    /// Turbined water flow for generation (m³/s)
    TurbinedFlow = 5,
    /// Spillage flow (m³/s)
    Spillage = 6,
    /// Water value (shadow price of storage) ($/MWh)
    WaterValue = 7,
    /// Thermal generation output (MW)
    ThermalGeneration = 8,
    /// Load curtailment/deficit (MW)
    Deficit = 9,
    /// Inter-bus power exchange (MW)
    Exchange = 10,
    /// Marginal cost of energy at bus ($/MWh)
    MarginalCost = 11,
    /// Dual variable for inflow lag constraints
    InflowLagDual = 12,
    /// Current stage objective function value ($)
    CurrentStageObjective = 13,
    /// Total stage objective including future cost ($)
    TotalStageObjective = 14,
}

impl OutputVariable {
    /// Returns all output variables in order.
    pub fn all() -> &'static [Self] {
        use OutputVariable::*;
        &[
            InitialStorage,
            InflowLag,
            SampledLoad,
            SampledInflow,
            FinalStorage,
            TurbinedFlow,
            Spillage,
            WaterValue,
            ThermalGeneration,
            Deficit,
            Exchange,
            MarginalCost,
            InflowLagDual,
            CurrentStageObjective,
            TotalStageObjective,
        ]
    }

    /// Returns the variable name as used in output files.
    pub fn name(&self) -> &'static str {
        match self {
            Self::InitialStorage => "initial_storage",
            Self::InflowLag => "inflow_lag",
            Self::SampledLoad => "sampled_load",
            Self::SampledInflow => "sampled_inflow",
            Self::FinalStorage => "final_storage",
            Self::TurbinedFlow => "turbined_flow",
            Self::Spillage => "spillage",
            Self::WaterValue => "water_value",
            Self::ThermalGeneration => "thermal_generation",
            Self::Deficit => "deficit",
            Self::Exchange => "exchange",
            Self::MarginalCost => "marginal_cost",
            Self::InflowLagDual => "inflow_lag_dual",
            Self::CurrentStageObjective => "current_stage_objective",
            Self::TotalStageObjective => "total_stage_objective",
        }
    }

    /// Returns the entity type for this variable.
    pub fn entity_type(&self) -> EntityType {
        match self {
            Self::InitialStorage
            | Self::InflowLag
            | Self::SampledInflow
            | Self::FinalStorage
            | Self::TurbinedFlow
            | Self::Spillage
            | Self::WaterValue
            | Self::InflowLagDual => EntityType::Hydro,
            Self::SampledLoad | Self::Deficit | Self::MarginalCost => {
                EntityType::Bus
            }
            Self::ThermalGeneration => EntityType::Thermal,
            Self::Exchange => EntityType::Line,
            Self::CurrentStageObjective | Self::TotalStageObjective => {
                EntityType::System
            }
        }
    }

    /// Returns whether this variable has an entity ID.
    pub fn has_entity_id(&self) -> bool {
        !matches!(
            self,
            Self::CurrentStageObjective | Self::TotalStageObjective
        )
    }

    /// Returns whether this variable has a lag index.
    pub fn has_lag_index(&self) -> bool {
        matches!(self, Self::InflowLag | Self::InflowLagDual)
    }

    /// Returns the units for this variable.
    pub fn units(&self) -> &'static str {
        match self {
            Self::InitialStorage | Self::FinalStorage => "hm³",
            Self::InflowLag
            | Self::SampledInflow
            | Self::TurbinedFlow
            | Self::Spillage => "m³/s",
            Self::SampledLoad
            | Self::ThermalGeneration
            | Self::Deficit
            | Self::Exchange => "MW",
            Self::WaterValue | Self::MarginalCost => "$/MWh",
            Self::InflowLagDual => "dimensionless",
            Self::CurrentStageObjective | Self::TotalStageObjective => "$",
        }
    }

    /// Returns a description of this variable.
    pub fn description(&self) -> &'static str {
        match self {
            Self::InitialStorage => {
                "Reservoir storage at the beginning of the stage"
            }
            Self::InflowLag => {
                "Historical inflow value for autoregressive modeling"
            }
            Self::SampledLoad => "Load demand realization for this scenario",
            Self::SampledInflow => "Inflow realization for this scenario",
            Self::FinalStorage => "Reservoir storage at the end of the stage",
            Self::TurbinedFlow => "Water flow turbined for generation",
            Self::Spillage => "Water flow spilled (not used for generation)",
            Self::WaterValue => {
                "Shadow price of storage (marginal value of water)"
            }
            Self::ThermalGeneration => "Thermal plant generation output",
            Self::Deficit => "Unmet load demand (load curtailment)",
            Self::Exchange => "Power exchange between buses",
            Self::MarginalCost => "Marginal cost of energy at the bus",
            Self::InflowLagDual => {
                "Dual variable for inflow lag state constraint"
            }
            Self::CurrentStageObjective => {
                "Objective function value for current stage only"
            }
            Self::TotalStageObjective => {
                "Total objective including future cost approximation"
            }
        }
    }

    /// Returns the metadata for this variable.
    pub fn metadata(&self) -> VariableMetadata {
        VariableMetadata {
            variable_index: *self as usize,
            variable_name: self.name().to_string(),
            entity_type: self.entity_type(),
            has_entity_id: self.has_entity_id(),
            has_lag_index: self.has_lag_index(),
            units: self.units().to_string(),
            description: self.description().to_string(),
        }
    }

    /// Returns the cardinality of this variable for the given system.
    ///
    /// This is the number of instances of this variable that will appear
    /// in the output (e.g., one per hydro, one per bus, etc.).
    pub fn cardinality(&self, system: &System) -> usize {
        match self.entity_type() {
            EntityType::Hydro => system.hydros.len(),
            EntityType::Bus => system.buses.len(),
            EntityType::Thermal => system.thermals.len(),
            EntityType::Line => system.lines.len(),
            EntityType::System => 1,
        }
    }

    /// Returns the maximum lag index for this variable.
    ///
    /// Returns `Some(max_lag)` for variables with lag indices, `None` otherwise.
    /// The max_lag parameter should be the maximum AR order across all temporal models.
    pub fn max_lag_index(&self, max_lag: usize) -> Option<usize> {
        if !self.has_lag_index() {
            return None;
        }

        match self {
            Self::InflowLag | Self::InflowLagDual => {
                if max_lag > 0 {
                    Some(max_lag)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Entity type classification for output variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Hydroelectric reservoir
    Hydro,
    /// Electrical bus
    Bus,
    /// Thermal power plant
    Thermal,
    /// Transmission line
    Line,
    /// System-wide variable (no entity)
    System,
}

impl EntityType {
    /// Returns the string representation of the entity type.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hydro => "hydro",
            Self::Bus => "bus",
            Self::Thermal => "thermal",
            Self::Line => "line",
            Self::System => "system",
        }
    }
}

/// Metadata for an output variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableMetadata {
    /// Index of the variable
    pub variable_index: usize,
    /// Name of the variable
    pub variable_name: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Whether this variable has an entity_id dimension
    pub has_entity_id: bool,
    /// Whether this variable has a lag_index dimension
    pub has_lag_index: bool,
    /// Units of measurement
    pub units: String,
    /// Human-readable description
    pub description: String,
}

/// Dictionary entry for a specific variable instance.
///
/// For variables with entity_id and/or lag_index, there will be multiple
/// entries (one per entity/lag combination).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableEntry {
    /// Variable index (from OutputVariable enum)
    pub variable_index: usize,
    /// Variable name
    pub variable_name: String,
    /// Entity type
    pub entity_type: String,
    /// Entity ID (if applicable)
    pub entity_id: Option<usize>,
    /// Lag index (if applicable, 1-indexed)
    pub lag_index: Option<usize>,
    /// Units
    pub units: String,
    /// Description
    pub description: String,
}

/// Complete variable dictionary for a system.
#[derive(Debug, Clone)]
pub struct VariableDictionary {
    /// All variable entries in order
    pub entries: Vec<VariableEntry>,
}

impl VariableDictionary {
    /// Generates a complete variable dictionary for the given system.
    ///
    /// # Arguments
    ///
    /// * `system` - The power system configuration
    /// * `max_ar_order` - Maximum AR order across all temporal models (0 if no AR models)
    ///
    /// # Returns
    ///
    /// A dictionary containing entries for all variable instances.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use powers_rs::output::dictionary::VariableDictionary;
    /// use powers_rs::system::System;
    ///
    /// // Assuming you have a system instance
    /// let system: System = get_system_from_somewhere();
    /// let max_ar_order = 2; // From temporal models
    /// let dict = VariableDictionary::generate(&system, max_ar_order);
    /// println!("Dictionary has {} entries", dict.entries.len());
    /// ```
    pub fn generate(system: &System, max_ar_order: usize) -> Self {
        let mut entries = Vec::new();

        for var in OutputVariable::all() {
            let metadata = var.metadata();
            let cardinality = var.cardinality(system);

            if var.has_lag_index() {
                // Variables with lag indices
                if let Some(max_lag) = var.max_lag_index(max_ar_order) {
                    for entity_id in 0..cardinality {
                        for lag_idx in 1..=max_lag {
                            entries.push(VariableEntry {
                                variable_index: metadata.variable_index,
                                variable_name: metadata.variable_name.clone(),
                                entity_type: metadata
                                    .entity_type
                                    .as_str()
                                    .to_string(),
                                entity_id: Some(entity_id),
                                lag_index: Some(lag_idx),
                                units: metadata.units.clone(),
                                description: metadata.description.clone(),
                            });
                        }
                    }
                }
            } else if var.has_entity_id() {
                // Variables with entity_id but no lag
                for entity_id in 0..cardinality {
                    entries.push(VariableEntry {
                        variable_index: metadata.variable_index,
                        variable_name: metadata.variable_name.clone(),
                        entity_type: metadata.entity_type.as_str().to_string(),
                        entity_id: Some(entity_id),
                        lag_index: None,
                        units: metadata.units.clone(),
                        description: metadata.description.clone(),
                    });
                }
            } else {
                // System-wide variables (no entity_id)
                entries.push(VariableEntry {
                    variable_index: metadata.variable_index,
                    variable_name: metadata.variable_name,
                    entity_type: metadata.entity_type.as_str().to_string(),
                    entity_id: None,
                    lag_index: None,
                    units: metadata.units,
                    description: metadata.description,
                });
            }
        }

        Self { entries }
    }

    /// Writes the dictionary to a CSV file.
    ///
    /// Creates the output directory if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `path` - Output directory path
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error otherwise.
    pub fn write_csv(&self, path: &str) -> Result<(), Box<dyn Error>> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(path)?;

        let mut wtr =
            Writer::from_path(&format!("{}/variable_dictionary.csv", path))?;

        wtr.write_record([
            "variable_index",
            "variable_name",
            "entity_type",
            "entity_id",
            "lag_index",
            "units",
            "description",
        ])?;

        for entry in &self.entries {
            wtr.write_record([
                &entry.variable_index.to_string(),
                &entry.variable_name,
                &entry.entity_type,
                &entry.entity_id.map(|id| id.to_string()).unwrap_or_default(),
                &entry
                    .lag_index
                    .map(|idx| idx.to_string())
                    .unwrap_or_default(),
                &entry.units,
                &entry.description,
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Returns the number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{Bus, Hydro, Line, System, Thermal};

    fn create_test_system() -> System {
        let buses = vec![Bus::new(0, 10000.0), Bus::new(1, 10000.0)];
        let lines = vec![Line::new(0, 0, 1, 200.0, 200.0, 5.0)];
        let thermals = vec![Thermal::new(0, 0, 150.0, 0.0, 500.0)];

        let hydros = vec![
            Hydro::new(0, None, 0, 0.1, 0.0, 1000.0, 0.0, 100.0, 5.0),
            Hydro::new(1, None, 0, 0.1, 0.0, 800.0, 0.0, 80.0, 5.0),
        ];

        System::new(buses, lines, thermals, hydros)
    }

    #[test]
    fn test_output_variable_all() {
        let all = OutputVariable::all();
        assert_eq!(all.len(), 15);
        assert_eq!(all[0], OutputVariable::InitialStorage);
        assert_eq!(all[14], OutputVariable::TotalStageObjective);
    }

    #[test]
    fn test_output_variable_name() {
        assert_eq!(OutputVariable::InitialStorage.name(), "initial_storage");
        assert_eq!(OutputVariable::WaterValue.name(), "water_value");
        assert_eq!(
            OutputVariable::CurrentStageObjective.name(),
            "current_stage_objective"
        );
    }

    #[test]
    fn test_output_variable_entity_type() {
        assert_eq!(
            OutputVariable::InitialStorage.entity_type(),
            EntityType::Hydro
        );
        assert_eq!(OutputVariable::SampledLoad.entity_type(), EntityType::Bus);
        assert_eq!(
            OutputVariable::ThermalGeneration.entity_type(),
            EntityType::Thermal
        );
        assert_eq!(OutputVariable::Exchange.entity_type(), EntityType::Line);
        assert_eq!(
            OutputVariable::CurrentStageObjective.entity_type(),
            EntityType::System
        );
    }

    #[test]
    fn test_output_variable_has_entity_id() {
        assert!(OutputVariable::InitialStorage.has_entity_id());
        assert!(!OutputVariable::CurrentStageObjective.has_entity_id());
    }

    #[test]
    fn test_output_variable_has_lag_index() {
        assert!(OutputVariable::InflowLag.has_lag_index());
        assert!(OutputVariable::InflowLagDual.has_lag_index());
        assert!(!OutputVariable::InitialStorage.has_lag_index());
    }

    #[test]
    fn test_output_variable_cardinality() {
        let system = create_test_system();
        assert_eq!(OutputVariable::InitialStorage.cardinality(&system), 2);
        assert_eq!(OutputVariable::SampledLoad.cardinality(&system), 2);
        assert_eq!(OutputVariable::ThermalGeneration.cardinality(&system), 1);
        assert_eq!(OutputVariable::Exchange.cardinality(&system), 1);
        assert_eq!(
            OutputVariable::CurrentStageObjective.cardinality(&system),
            1
        );
    }

    #[test]
    fn test_output_variable_max_lag_index() {
        assert_eq!(OutputVariable::InflowLag.max_lag_index(2), Some(2));
        assert_eq!(OutputVariable::InflowLag.max_lag_index(1), Some(1));
        assert_eq!(OutputVariable::InflowLag.max_lag_index(0), None);
        assert_eq!(OutputVariable::InitialStorage.max_lag_index(2), None);
    }

    #[test]
    fn test_variable_dictionary_generation() {
        let system = create_test_system();
        let max_ar_order = 2;
        let dict = VariableDictionary::generate(&system, max_ar_order);

        // Check we have entries
        assert!(!dict.is_empty());
        assert!(dict.len() > 15); // More than just the 15 base variables

        // Count entries by variable type
        let initial_storage_count = dict
            .entries
            .iter()
            .filter(|e| e.variable_name == "initial_storage")
            .count();
        assert_eq!(initial_storage_count, 2); // 2 hydros

        let inflow_lag_count = dict
            .entries
            .iter()
            .filter(|e| e.variable_name == "inflow_lag")
            .count();
        assert_eq!(inflow_lag_count, 4); // 2 hydros * 2 lags

        let system_vars = dict
            .entries
            .iter()
            .filter(|e| e.entity_id.is_none())
            .count();
        assert_eq!(system_vars, 2); // current_stage_objective, total_stage_objective
    }

    #[test]
    fn test_variable_entry_structure() {
        let system = create_test_system();
        let dict = VariableDictionary::generate(&system, 2);

        // Find an entry with entity_id
        let storage_entry = dict
            .entries
            .iter()
            .find(|e| e.variable_name == "initial_storage")
            .unwrap();
        assert!(storage_entry.entity_id.is_some());
        assert!(storage_entry.lag_index.is_none());
        assert_eq!(storage_entry.units, "hm³"); // Cubic hectometers for reservoir storage

        // Find an entry with lag_index
        let lag_entry = dict
            .entries
            .iter()
            .find(|e| e.variable_name == "inflow_lag")
            .unwrap();
        assert!(lag_entry.entity_id.is_some());
        assert!(lag_entry.lag_index.is_some());
        assert_eq!(lag_entry.units, "m³/s");

        // Find a system entry
        let system_entry = dict
            .entries
            .iter()
            .find(|e| e.variable_name == "current_stage_objective")
            .unwrap();
        assert!(system_entry.entity_id.is_none());
        assert!(system_entry.lag_index.is_none());
        assert_eq!(system_entry.units, "$");
    }
}
