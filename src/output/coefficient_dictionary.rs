//! Cut coefficient dictionary module.
//!
//! This module provides a dictionary system for indexing cut coefficients,
//! replacing the mixed-type `coefficient_entity` column with integer indices.
//! This provides type safety and handles PAR models with variable lag counts.

use crate::system;
use csv::Writer;
use serde::{Deserialize, Serialize};
use std::error::Error;

/// Type of coefficient in a Benders cut
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CoefficientType {
    /// Right-hand side constant (always index 0)
    Rhs,
    /// Storage state variable coefficient
    Storage,
    /// Inflow lag state variable coefficient (for PAR models)
    Lag,
}

/// Single entry in the coefficient dictionary
///
/// Each entry describes one coefficient position in the Benders cut.
/// For a system with N hydros and PAR(p) models, the coefficient vector is:
/// - Index 0: RHS constant
/// - Indices 1..N+1: Storage coefficients (one per hydro)
/// - Indices N+1..: Lag coefficients (depends on AR order per hydro)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientEntry {
    /// Sequential index in coefficient vector (0 = RHS, 1+ = state variables)
    pub coefficient_index: usize,

    /// Type of this coefficient
    pub coefficient_type: CoefficientType,

    /// Entity ID (hydro ID) for storage and lag coefficients
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<usize>,

    /// Lag index for lag coefficients (1-indexed: 1 = t-1, 2 = t-2, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag_index: Option<usize>,

    /// Human-readable description
    pub description: String,
}

impl CoefficientEntry {
    /// Creates an RHS coefficient entry (always index 0)
    pub fn rhs() -> Self {
        Self {
            coefficient_index: 0,
            coefficient_type: CoefficientType::Rhs,
            entity_id: None,
            lag_index: None,
            description: "Right-hand side constant".to_string(),
        }
    }

    /// Creates a storage coefficient entry
    pub fn storage(index: usize, hydro_id: usize) -> Self {
        Self {
            coefficient_index: index,
            coefficient_type: CoefficientType::Storage,
            entity_id: Some(hydro_id),
            lag_index: None,
            description: format!("Storage coefficient for hydro {}", hydro_id),
        }
    }

    /// Creates a lag coefficient entry
    pub fn lag(index: usize, hydro_id: usize, lag: usize) -> Self {
        Self {
            coefficient_index: index,
            coefficient_type: CoefficientType::Lag,
            entity_id: Some(hydro_id),
            lag_index: Some(lag),
            description: format!(
                "Lag {} coefficient for hydro {} (t-{})",
                lag, hydro_id, lag
            ),
        }
    }
}

/// Dictionary mapping coefficient indices to their metadata
///
/// Provides a complete mapping of the coefficient vector structure for a
/// specific system configuration. The coefficient order matches the state
/// vector construction in the SDDP algorithm.
#[derive(Debug, Clone)]
pub struct CoefficientDictionary {
    /// All coefficient entries in sequential order
    pub entries: Vec<CoefficientEntry>,
}

impl CoefficientDictionary {
    /// Generates a coefficient dictionary for a given system and AR orders
    ///
    /// The coefficient vector structure is:
    /// 1. RHS (index 0)
    /// 2. Storage variables (one per hydro, indices 1..num_hydros+1)
    /// 3. Lag variables (for each hydro with AR model, num_lags entries)
    ///
    /// # Arguments
    ///
    /// * `system` - The power system configuration
    /// * `ar_orders` - AR order for each hydro (0 = no AR model)
    ///
    /// # Returns
    ///
    /// Dictionary with entries for RHS + all state variables
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // System with 2 hydros: hydro 0 has PAR(2), hydro 1 has no AR model
    /// let ar_orders = vec![2, 0];
    /// let dict = CoefficientDictionary::generate(system, &ar_orders);
    /// // Results in:
    /// // Index 0: RHS
    /// // Index 1: Storage hydro 0
    /// // Index 2: Storage hydro 1
    /// // Index 3: Lag 1 hydro 0
    /// // Index 4: Lag 2 hydro 0
    /// ```
    pub fn generate(system: &system::System, ar_orders: &[usize]) -> Self {
        let mut entries = Vec::new();
        let mut index = 0;

        // Index 0: RHS
        entries.push(CoefficientEntry::rhs());
        index += 1;

        // Storage coefficients (one per hydro)
        for hydro_id in 0..system.hydros.len() {
            entries.push(CoefficientEntry::storage(index, hydro_id));
            index += 1;
        }

        // Lag coefficients (depends on AR order per hydro)
        for (hydro_id, &ar_order) in ar_orders.iter().enumerate() {
            for lag in 1..=ar_order {
                entries.push(CoefficientEntry::lag(index, hydro_id, lag));
                index += 1;
            }
        }

        Self { entries }
    }

    /// Returns the total number of coefficients (including RHS)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the expected number of state coefficients (excluding RHS)
    pub fn state_dimension(&self) -> usize {
        self.len().saturating_sub(1)
    }

    /// Writes the dictionary to a CSV file
    ///
    /// Creates `coefficient_dictionary.csv` in the specified directory.
    ///
    /// # Format
    ///
    /// ```csv
    /// coefficient_index,coefficient_type,entity_id,lag_index,description
    /// 0,rhs,,,Right-hand side constant
    /// 1,storage,0,,Storage coefficient for hydro 0
    /// 2,storage,1,,Storage coefficient for hydro 1
    /// 3,lag,0,1,Lag 1 coefficient for hydro 0 (t-1)
    /// ```
    pub fn write_csv(&self, output_dir: &str) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(output_dir)?;

        let path = format!("{}/coefficient_dictionary.csv", output_dir);
        let mut wtr = Writer::from_path(&path)?;

        wtr.write_record([
            "coefficient_index",
            "coefficient_type",
            "entity_id",
            "lag_index",
            "description",
        ])?;

        for entry in &self.entries {
            let coef_type = serde_json::to_string(&entry.coefficient_type)
                .unwrap()
                .trim_matches('"')
                .to_string();

            wtr.write_record([
                &entry.coefficient_index.to_string(),
                &coef_type,
                &entry.entity_id.map_or(String::new(), |id| id.to_string()),
                &entry.lag_index.map_or(String::new(), |lag| lag.to_string()),
                &entry.description,
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Writes state component dictionary to CSV file.
    ///
    /// State components have the same structure as cut coefficients:
    /// [objective, storage_0, storage_1, ..., lag_0_1, lag_1_1, ...]
    ///
    /// This is a convenience method that writes the same dictionary with
    /// a different filename for clarity.
    pub fn write_csv_as_state_components(
        &self,
        output_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!("{}/state_component_dictionary.csv", output_dir);
        let mut wtr = csv::Writer::from_path(&path)?;

        wtr.write_record([
            "component_index",
            "component_type",
            "entity_id",
            "lag_index",
            "description",
        ])?;

        for entry in &self.entries {
            let coef_type = serde_json::to_string(&entry.coefficient_type)?
                .trim_matches('"')
                .to_string();

            wtr.write_record([
                &entry.coefficient_index.to_string(),
                &coef_type,
                &entry.entity_id.map_or(String::new(), |id| id.to_string()),
                &entry.lag_index.map_or(String::new(), |lag| lag.to_string()),
                &entry.description,
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }
}

/// Validates that a cut's coefficient count matches the dictionary
///
/// # Errors
///
/// Returns an error if the coefficient count doesn't match the expected
/// state dimension from the dictionary.
pub fn validate_coefficient_count(
    coefficient_count: usize,
    dictionary: &CoefficientDictionary,
) -> Result<(), String> {
    let expected = dictionary.state_dimension();

    if coefficient_count != expected {
        return Err(format!(
            "Coefficient count mismatch: expected {} state coefficients, got {}",
            expected, coefficient_count
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_system(num_hydros: usize) -> system::System {
        let hydros: Vec<_> = (0..num_hydros)
            .map(|id| system::Hydro {
                id,
                downstream_hydro_id: None,
                bus_id: 0,
                productivity: 1.0,
                min_storage: 0.0,
                max_storage: 100.0,
                min_turbined_flow: 0.0,
                max_turbined_flow: 50.0,
                spillage_penalty: 0.0,
                upstream_hydro_ids: vec![],
            })
            .collect();

        system::System {
            meta: system::SystemMetadata {
                buses_count: 1,
                lines_count: 0,
                thermals_count: 0,
                hydros_count: num_hydros,
            },
            buses: vec![],
            hydros,
            thermals: vec![],
            lines: vec![],
        }
    }

    #[test]
    fn test_dictionary_rhs_only() {
        let system = create_test_system(0);
        let ar_orders = vec![];

        let dict = CoefficientDictionary::generate(&system, &ar_orders);

        assert_eq!(dict.len(), 1);
        assert_eq!(dict.state_dimension(), 0);
        assert_eq!(dict.entries[0].coefficient_type, CoefficientType::Rhs);
    }

    #[test]
    fn test_dictionary_storage_only() {
        let system = create_test_system(3);
        let ar_orders = vec![0, 0, 0]; // No AR models

        let dict = CoefficientDictionary::generate(&system, &ar_orders);

        // RHS + 3 storage coefficients
        assert_eq!(dict.len(), 4);
        assert_eq!(dict.state_dimension(), 3);

        assert_eq!(dict.entries[0].coefficient_type, CoefficientType::Rhs);
        assert_eq!(dict.entries[1].coefficient_type, CoefficientType::Storage);
        assert_eq!(dict.entries[1].entity_id, Some(0));
        assert_eq!(dict.entries[2].coefficient_type, CoefficientType::Storage);
        assert_eq!(dict.entries[2].entity_id, Some(1));
        assert_eq!(dict.entries[3].coefficient_type, CoefficientType::Storage);
        assert_eq!(dict.entries[3].entity_id, Some(2));
    }

    #[test]
    fn test_dictionary_with_ar_model() {
        let system = create_test_system(2);
        let ar_orders = vec![2, 0]; // Hydro 0 has PAR(2), hydro 1 no AR

        let dict = CoefficientDictionary::generate(&system, &ar_orders);

        // RHS + 2 storage + 2 lags for hydro 0 = 5 total
        assert_eq!(dict.len(), 5);
        assert_eq!(dict.state_dimension(), 4);

        assert_eq!(dict.entries[0].coefficient_type, CoefficientType::Rhs);
        assert_eq!(dict.entries[1].coefficient_type, CoefficientType::Storage);
        assert_eq!(dict.entries[1].entity_id, Some(0));
        assert_eq!(dict.entries[2].coefficient_type, CoefficientType::Storage);
        assert_eq!(dict.entries[2].entity_id, Some(1));

        // Lag coefficients for hydro 0
        assert_eq!(dict.entries[3].coefficient_type, CoefficientType::Lag);
        assert_eq!(dict.entries[3].entity_id, Some(0));
        assert_eq!(dict.entries[3].lag_index, Some(1));

        assert_eq!(dict.entries[4].coefficient_type, CoefficientType::Lag);
        assert_eq!(dict.entries[4].entity_id, Some(0));
        assert_eq!(dict.entries[4].lag_index, Some(2));
    }

    #[test]
    fn test_dictionary_mixed_ar_orders() {
        let system = create_test_system(3);
        let ar_orders = vec![1, 3, 0]; // Hydro 0: PAR(1), Hydro 1: PAR(3), Hydro 2: none

        let dict = CoefficientDictionary::generate(&system, &ar_orders);

        // RHS + 3 storage + 1 lag (hydro 0) + 3 lags (hydro 1) = 8 total
        assert_eq!(dict.len(), 8);
        assert_eq!(dict.state_dimension(), 7);
    }

    #[test]
    fn test_validate_coefficient_count_valid() {
        let system = create_test_system(2);
        let ar_orders = vec![1, 0];
        let dict = CoefficientDictionary::generate(&system, &ar_orders);

        // RHS + 2 storage + 1 lag = 4, so state dimension = 3
        let result = validate_coefficient_count(3, &dict);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_coefficient_count_mismatch() {
        let system = create_test_system(2);
        let ar_orders = vec![1, 0];
        let dict = CoefficientDictionary::generate(&system, &ar_orders);

        let result = validate_coefficient_count(5, &dict);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mismatch"));
    }

    #[test]
    fn test_coefficient_entry_constructors() {
        let rhs = CoefficientEntry::rhs();
        assert_eq!(rhs.coefficient_index, 0);
        assert_eq!(rhs.coefficient_type, CoefficientType::Rhs);
        assert!(rhs.entity_id.is_none());
        assert!(rhs.lag_index.is_none());

        let storage = CoefficientEntry::storage(1, 0);
        assert_eq!(storage.coefficient_index, 1);
        assert_eq!(storage.coefficient_type, CoefficientType::Storage);
        assert_eq!(storage.entity_id, Some(0));
        assert!(storage.lag_index.is_none());

        let lag = CoefficientEntry::lag(2, 0, 1);
        assert_eq!(lag.coefficient_index, 2);
        assert_eq!(lag.coefficient_type, CoefficientType::Lag);
        assert_eq!(lag.entity_id, Some(0));
        assert_eq!(lag.lag_index, Some(1));
    }

    #[test]
    fn test_dictionary_csv_write() {
        let system = create_test_system(1);
        let ar_orders = vec![1];
        let dict = CoefficientDictionary::generate(&system, &ar_orders);

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        dict.write_csv(path).unwrap();

        let csv_path = format!("{}/coefficient_dictionary.csv", path);
        assert!(std::path::Path::new(&csv_path).exists());

        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Header + 3 entries (RHS, storage, lag)
        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("coefficient_index"));
        assert!(lines[0].contains("coefficient_type"));

        // Check RHS entry
        assert!(lines[1].contains("0,rhs,"));

        // Check storage entry
        assert!(lines[2].contains("1,storage,0"));

        // Check lag entry
        assert!(lines[3].contains("2,lag,0,1"));
    }
}
