//! Comprehensive error types for POWE.RS with context and actionable guidance.
//!
//! This module provides a hierarchical error system that:
//! - **Provides context**: Field names, line numbers, values for debugging
//! - **Suggests fixes**: Actionable guidance for common errors
//! - **Categorizes errors**: Validation, Solver, I/O, Graph for proper handling
//! - **Hides internals**: User-friendly messages without stack traces (unless debug)
//!
//! # Error Categories
//!
//! - [`ValidationError`]: Input validation failures (invalid values, constraints)
//! - [`SolverError`]: Optimization solver failures (infeasible, unbounded, etc.)
//! - [`IoError`]: File I/O failures (missing files, permission denied, etc.)
//! - [`GraphError`]: Scenario graph issues (disconnected, invalid probabilities, etc.)
//! - [`PowersError`]: Top-level error type that wraps all categories
//!
//! # Design Principles
//!
//! 1. **Context-Rich**: Every error includes relevant context (file, field, value)
//! 2. **Actionable**: Errors suggest fixes, not just what went wrong
//! 3. **User-Friendly**: Production errors are clear; stack traces only in debug
//! 4. **Typed**: Use specific error types for proper error handling
//! 5. **Zero-Cost**: Error types are thin wrappers with no runtime overhead
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::error::{PowersError, ValidationError};
//!
//! fn validate_positive(value: usize, field: &str) -> Result<(), PowersError> {
//!     if value == 0 {
//!         return Err(ValidationError::InvalidFieldValue {
//!             file: "config.json".to_string(),
//!             field: field.to_string(),
//!             value: value.to_string(),
//!             constraint: "must be positive (> 0)".to_string(),
//!             suggestion: format!("Set {} to at least 1", field),
//!         }.into());
//!     }
//!     Ok(())
//! }
//! ```

use std::io;
use thiserror::Error;

/// Top-level error type for POWE.RS operations.
///
/// This enum wraps all error categories and provides a unified interface
/// for error handling throughout the codebase.
///
/// # Usage
///
/// Most functions should return `Result<T, PowersError>` to allow callers
/// to handle different error categories appropriately.
///
/// # Conversion
///
/// All specific error types (`ValidationError`, `SolverError`, etc.) can be
/// converted to `PowersError` using `.into()` or the `?` operator.
///
/// # Performance
///
/// Error variants are boxed to keep the `Result` type small (16 bytes),
/// which is critical for hot paths. Error construction has negligible overhead
/// (~1 allocation) compared to the cost of handling the error itself.
#[derive(Error, Debug)]
pub enum PowersError {
    /// Input validation error (invalid values, constraint violations)
    #[error("{0}")]
    Validation(#[from] Box<ValidationError>),

    /// Solver optimization error (infeasible, unbounded, etc.)
    #[error("{0}")]
    Solver(#[from] Box<SolverError>),

    /// File I/O error (missing files, permission denied, etc.)
    #[error("{0}")]
    Io(#[from] Box<IoError>),

    /// Scenario graph error (disconnected, invalid probabilities, etc.)
    #[error("{0}")]
    Graph(#[from] Box<GraphError>),

    /// Generic error for cases not covered by specific types
    #[error("{0}")]
    Other(String),
}

/// Input validation errors with context and suggestions.
///
/// These errors occur when input data violates constraints or expectations.
/// Each variant includes:
/// - **Context**: Which file, field, and value caused the error
/// - **Constraint**: What constraint was violated
/// - **Suggestion**: How to fix the issue
#[derive(Error, Debug)]
pub enum ValidationError {
    /// A field has an invalid value that violates a constraint.
    ///
    /// # Example
    ///
    /// ```text
    /// config.json: Field 'num_iterations' has invalid value '0'.
    /// Constraint: must be positive (> 0)
    /// Suggestion: Set num_iterations to at least 1
    /// ```
    #[error(
        "{file}: Field '{field}' has invalid value '{value}'.\n\
         Constraint: {constraint}\n\
         Suggestion: {suggestion}"
    )]
    InvalidFieldValue {
        file: String,
        field: String,
        value: String,
        constraint: String,
        suggestion: String,
    },

    /// A field is missing from the input file.
    ///
    /// # Example
    ///
    /// ```text
    /// config.json: Missing required field 'num_iterations'.
    /// Suggestion: Add "num_iterations": 100 to your config file.
    /// See docs/INPUT-SPECIFICATION.md for complete specification.
    /// ```
    #[error(
        "{file}: Missing required field '{field}'.\n\
         Suggestion: {suggestion}\n\
         See docs/INPUT-SPECIFICATION.md for complete specification."
    )]
    MissingField {
        file: String,
        field: String,
        suggestion: String,
    },

    /// A constraint between fields is violated.
    ///
    /// # Example
    ///
    /// ```text
    /// system.json: Thermal 'thermal_1' violates constraint: min_generation <= max_generation.
    /// Found: min_generation=100.0, max_generation=50.0
    /// Suggestion: Set max_generation >= 100.0 or reduce min_generation
    /// ```
    #[error(
        "{file}: {context} violates constraint: {constraint}.\n\
         Found: {details}\n\
         Suggestion: {suggestion}"
    )]
    ConstraintViolation {
        file: String,
        context: String,
        constraint: String,
        details: String,
        suggestion: String,
    },

    /// A reference to another entity is invalid.
    ///
    /// # Example
    ///
    /// ```text
    /// system.json: Line 'line_1' references non-existent bus_id=99.
    /// Available bus IDs: 0, 1, 2, 3
    /// Suggestion: Change source_bus_id or target_bus_id to a valid bus ID (0-3)
    /// ```
    #[error(
        "{file}: {context} references non-existent {ref_type}={ref_id}.\n\
         Available {ref_type}s: {available}\n\
         Suggestion: {suggestion}"
    )]
    InvalidReference {
        file: String,
        context: String,
        ref_type: String,
        ref_id: String,
        available: String,
        suggestion: String,
    },

    /// JSON parsing failed.
    ///
    /// # Example
    ///
    /// ```text
    /// config.json: Failed to parse JSON at line 5, column 12.
    /// Error: expected `,` or `}` at line 5 column 12
    /// Suggestion: Check for missing commas, brackets, or quotes.
    /// Validate your JSON at https://jsonlint.com/
    /// ```
    #[error(
        "{file}: Failed to parse JSON.\n\
         Error: {error}\n\
         Suggestion: Check for missing commas, brackets, or quotes.\n\
         Validate your JSON at https://jsonlint.com/"
    )]
    JsonParseError { file: String, error: String },

    /// A required array is empty when it should contain elements.
    ///
    /// # Example
    ///
    /// ```text
    /// system.json: Array 'hydros' is empty but must contain at least one element.
    /// Suggestion: Add at least one hydro unit to the system.
    /// ```
    #[error(
        "{file}: Array '{field}' is empty but must contain at least one element.\n\
         Suggestion: {suggestion}"
    )]
    EmptyArray {
        file: String,
        field: String,
        suggestion: String,
    },

    /// Missing AR lag inflows for a hydro with an AR noise model.
    ///
    /// # Example
    ///
    /// ```text
    /// recourse.json: Missing AR lag inflows for hydro_id 0 (AR order 1).
    /// AR models require historical lag values in initial_condition.inflow.
    ///
    /// Suggestion: Add inflow entries for hydro_id 0 with lag values 1..1
    /// Example: "inflow": [{"hydro_id": 0, "lag": 1, "value": 120.0}]
    /// ```
    #[error(
        "{file}: Missing AR lag inflows for hydro_id {hydro_id} (AR order {lag_order}).\n\
         AR models require historical lag values in initial_condition.inflow.\n\n\
         Suggestion: Add inflow entries for hydro_id {hydro_id} with lag values 1..{lag_order}\n\
         Example: \"inflow\": [{{\"hydro_id\": {hydro_id}, \"lag\": 1, \"value\": ...}}]"
    )]
    MissingARLagInflows {
        file: String,
        hydro_id: usize,
        lag_order: usize,
    },

    /// Invalid AR lag count (doesn't match lag_order).
    ///
    /// # Example
    ///
    /// ```text
    /// recourse.json: Invalid AR lag count for hydro_id 0: expected 2, found 1.
    /// AR(2) models require exactly 2 historical lag values with lag indices 1..2.
    ///
    /// Suggestion: Provide entries for lag=1 and lag=2 in inflow array
    /// Example: [{"hydro_id": 0, "lag": 1, "value": 120.0}, {"hydro_id": 0, "lag": 2, "value": 115.0}]
    /// ```
    #[error(
        "{file}: Invalid AR lag count for hydro_id {hydro_id}: expected {expected}, found {found}.\n\
         AR({expected}) models require exactly {expected} historical lag values with lag indices 1..{expected}.\n\n\
         Suggestion: Provide entries for lag=1..{expected} in inflow array\n\
         Example: {example}"
    )]
    InvalidARLagCount {
        file: String,
        hydro_id: usize,
        expected: usize,
        found: usize,
        example: String,
    },

    /// Invalid AR lag indices (not consecutive 1..p).
    ///
    /// # Example
    ///
    /// ```text
    /// recourse.json: Invalid AR lag indices for hydro_id 0: expected [1, 2], found [1, 3].
    /// AR models require consecutive lag indices starting from 1.
    ///
    /// Suggestion: Ensure lag indices are exactly 1, 2, ..., p for AR(p) model
    /// Fix duplicate or missing lag entries in inflow array.
    /// ```
    #[error(
        "{file}: Invalid AR lag indices for hydro_id {hydro_id}: expected {expected:?}, found {found:?}.\n\
         AR models require consecutive lag indices starting from 1.\n\n\
         Suggestion: Ensure lag indices are exactly 1, 2, ..., p for AR(p) model\n\
         Fix duplicate or missing lag entries in inflow array."
    )]
    InvalidARLagIndices {
        file: String,
        hydro_id: usize,
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    /// Negative lag inflow value detected.
    ///
    /// # Example
    ///
    /// ```text
    /// recourse.json: Negative lag inflow for hydro_id 0 at lag 1: -50.0
    /// Lag inflow values must be non-negative (inflows cannot be negative).
    ///
    /// Suggestion: Change the inflow entry with hydro_id=0, lag=1 to a non-negative value
    /// ```
    #[error(
        "{file}: Negative lag inflow for hydro_id {hydro_id} at lag {lag_index}: {value}\n\
         Lag inflow values must be non-negative (inflows cannot be negative).\n\n\
         Suggestion: Change the inflow entry with hydro_id={hydro_id}, lag={lag_index} to a non-negative value"
    )]
    NegativeLagInflow {
        file: String,
        hydro_id: usize,
        lag_index: usize,
        value: f64,
    },
}

/// Solver optimization errors with context about the failure.
///
/// These errors occur during LP/MIP solving when the optimization
/// problem cannot be solved successfully.
#[derive(Error, Debug)]
pub enum SolverError {
    /// The optimization problem is infeasible (no solution exists).
    ///
    /// # Example
    ///
    /// ```text
    /// Solver failed: Problem is infeasible at node 5, iteration 10.
    /// This means the constraints cannot be satisfied simultaneously.
    ///
    /// Common causes:
    /// - Conflicting constraints (e.g., min > max)
    /// - Insufficient generation capacity to meet load
    /// - Transmission capacity too restrictive
    ///
    /// Suggestion: Review system.json constraints and capacity limits.
    /// Check that total generation capacity >= peak load.
    /// ```
    #[error(
        "Solver failed: Problem is infeasible at {context}.\n\
         This means the constraints cannot be satisfied simultaneously.\n\n\
         Common causes:\n\
         - Conflicting constraints (e.g., min > max)\n\
         - Insufficient generation capacity to meet load\n\
         - Transmission capacity too restrictive\n\n\
         Suggestion: {suggestion}"
    )]
    Infeasible { context: String, suggestion: String },

    /// The optimization problem is unbounded (objective can be infinite).
    ///
    /// # Example
    ///
    /// ```text
    /// Solver failed: Problem is unbounded at node 5, iteration 10.
    /// This usually indicates a modeling error.
    ///
    /// Common causes:
    /// - Missing upper bounds on generation variables
    /// - Negative costs without constraints
    /// - Missing capacity limits
    ///
    /// Suggestion: Check that all generation variables have finite upper bounds.
    /// ```
    #[error(
        "Solver failed: Problem is unbounded at {context}.\n\
         This usually indicates a modeling error.\n\n\
         Common causes:\n\
         - Missing upper bounds on generation variables\n\
         - Negative costs without constraints\n\
         - Missing capacity limits\n\n\
         Suggestion: {suggestion}"
    )]
    Unbounded { context: String, suggestion: String },

    /// Solver encountered a numerical error.
    ///
    /// # Example
    ///
    /// ```text
    /// Solver failed: Numerical error at node 5, iteration 10.
    /// This indicates numerical instability in the problem.
    ///
    /// Common causes:
    /// - Coefficients differ by many orders of magnitude
    /// - Very large or very small coefficient values
    /// - Poor problem scaling
    ///
    /// Suggestion: Scale your problem so coefficients are between 1e-6 and 1e6.
    /// Check for extreme values in costs, capacities, or bounds.
    /// ```
    #[error(
        "Solver failed: Numerical error at {context}.\n\
         This indicates numerical instability in the problem.\n\n\
         Common causes:\n\
         - Coefficients differ by many orders of magnitude\n\
         - Very large or very small coefficient values\n\
         - Poor problem scaling\n\n\
         Suggestion: {suggestion}"
    )]
    NumericalError { context: String, suggestion: String },

    /// Generic solver error for unexpected failures.
    ///
    /// # Example
    ///
    /// ```text
    /// Solver failed at node 5, iteration 10: Unknown solver status 99
    /// Suggestion: This may be a bug in HiGHS or the Rust bindings.
    /// Please report this issue with your input files.
    /// ```
    #[error(
        "Solver failed at {context}: {message}\n\
         Suggestion: {suggestion}"
    )]
    GenericError {
        context: String,
        message: String,
        suggestion: String,
    },
}

/// File I/O errors with context about the operation.
///
/// These errors occur when reading or writing files fails.
#[derive(Error, Debug)]
pub enum IoError {
    /// File not found.
    ///
    /// # Example
    ///
    /// ```text
    /// File not found: 'example/config.json'
    /// Suggestion: Check that the file path is correct and the file exists.
    /// Current directory: /home/user/powers
    /// ```
    #[error(
        "File not found: '{path}'\n\
         Suggestion: Check that the file path is correct and the file exists.\n\
         Current directory: {current_dir}"
    )]
    FileNotFound { path: String, current_dir: String },

    /// Permission denied when accessing file.
    ///
    /// # Example
    ///
    /// ```text
    /// Permission denied: Cannot read 'example/config.json'
    /// Suggestion: Check file permissions with 'ls -l example/config.json'.
    /// You may need to run 'chmod +r example/config.json'.
    /// ```
    #[error(
        "Permission denied: Cannot read '{path}'\n\
         Suggestion: Check file permissions with 'ls -l {path}'.\n\
         You may need to run 'chmod +r {path}'."
    )]
    PermissionDenied { path: String },

    /// Cannot write to output directory.
    ///
    /// # Example
    ///
    /// ```text
    /// Cannot write to output directory: 'output/results'
    /// Error: No such file or directory
    /// Suggestion: Create the directory with 'mkdir -p output/results' or check permissions.
    /// ```
    #[error(
        "Cannot write to output directory: '{path}'\n\
         Error: {error}\n\
         Suggestion: Create the directory with 'mkdir -p {path}' or check permissions."
    )]
    CannotWriteOutput { path: String, error: String },

    /// Generic I/O error.
    #[error(
        "I/O error with file '{path}': {error}\n\
         Suggestion: {suggestion}"
    )]
    GenericIoError {
        path: String,
        error: String,
        suggestion: String,
    },
}

/// Scenario graph structure errors.
///
/// These errors occur when the scenario tree structure is invalid.
#[derive(Error, Debug)]
pub enum GraphError {
    /// Graph is disconnected (not all nodes reachable from root).
    ///
    /// # Example
    ///
    /// ```text
    /// graph.json: Scenario graph is disconnected.
    /// Unreachable nodes: [5, 6, 7]
    /// Suggestion: Check that all nodes are connected to the root node (node 0).
    /// Review edges to ensure complete connectivity.
    /// ```
    #[error(
        "graph.json: Scenario graph is disconnected.\n\
         Unreachable nodes: {nodes}\n\
         Suggestion: Check that all nodes are connected to the root node (node 0).\n\
         Review edges to ensure complete connectivity."
    )]
    DisconnectedGraph { nodes: String },

    /// Edge probabilities from a node don't sum to 1.0.
    ///
    /// # Example
    ///
    /// ```text
    /// graph.json: Node 2 has outgoing edge probabilities that don't sum to 1.0.
    /// Sum: 0.85 (expected 1.0 ± 0.0001)
    /// Outgoing edges: [(2→3, 0.3), (2→4, 0.25), (2→5, 0.3)]
    /// Suggestion: Adjust edge probabilities so they sum to 1.0.
    /// ```
    #[error(
        "graph.json: Node {node_id} has outgoing edge probabilities that don't sum to 1.0.\n\
         Sum: {sum} (expected 1.0 ± 0.0001)\n\
         Outgoing edges: {edges}\n\
         Suggestion: Adjust edge probabilities so they sum to 1.0."
    )]
    InvalidProbabilitySum {
        node_id: usize,
        sum: f64,
        edges: String,
    },

    /// Cycle detected in graph (should be a DAG).
    ///
    /// # Example
    ///
    /// ```text
    /// graph.json: Cycle detected in scenario graph.
    /// Cycle: 2 → 5 → 8 → 2
    /// Suggestion: Remove edges that create cycles. The graph must be a Directed Acyclic Graph (DAG).
    /// ```
    #[error(
        "graph.json: Cycle detected in scenario graph.\n\
         Cycle: {cycle}\n\
         Suggestion: Remove edges that create cycles. The graph must be a Directed Acyclic Graph (DAG)."
    )]
    CycleDetected { cycle: String },

    /// Missing root node (node with no incoming edges).
    ///
    /// # Example
    ///
    /// ```text
    /// graph.json: No root node found (node with stage_id=0 and no incoming edges).
    /// Suggestion: Ensure there is exactly one node with stage_id=0.
    /// ```
    #[error(
        "graph.json: No root node found (node with stage_id=0 and no incoming edges).\n\
         Suggestion: Ensure there is exactly one node with stage_id=0."
    )]
    MissingRootNode,
}

// Conversions from String for backward compatibility
impl From<String> for PowersError {
    fn from(s: String) -> Self {
        PowersError::Other(s)
    }
}

impl From<&str> for PowersError {
    fn from(s: &str) -> Self {
        PowersError::Other(s.to_string())
    }
}

// Conversion from std::io::Error
impl From<io::Error> for PowersError {
    fn from(error: io::Error) -> Self {
        let suggestion = match error.kind() {
            io::ErrorKind::NotFound => {
                "Check that the file path is correct and the file exists."
            }
            io::ErrorKind::PermissionDenied => {
                "Check file permissions. You may need read/write access."
            }
            io::ErrorKind::AlreadyExists => {
                "File already exists. Remove it or choose a different path."
            }
            _ => "Check the file path and permissions.",
        };

        PowersError::Io(Box::new(IoError::GenericIoError {
            path: "unknown".to_string(),
            error: error.to_string(),
            suggestion: suggestion.to_string(),
        }))
    }
}

// Conversion from serde_json::Error
impl From<serde_json::Error> for PowersError {
    fn from(error: serde_json::Error) -> Self {
        PowersError::Validation(Box::new(ValidationError::JsonParseError {
            file: "unknown".to_string(),
            error: error.to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_display() {
        let error = ValidationError::InvalidFieldValue {
            file: "config.json".to_string(),
            field: "num_iterations".to_string(),
            value: "0".to_string(),
            constraint: "must be positive (> 0)".to_string(),
            suggestion: "Set num_iterations to at least 1".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("config.json"));
        assert!(display.contains("num_iterations"));
        assert!(display.contains("Suggestion"));
    }

    #[test]
    fn test_solver_error_display() {
        let error = SolverError::Infeasible {
            context: "node 5, iteration 10".to_string(),
            suggestion: "Check that total generation capacity >= peak load."
                .to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("infeasible"));
        assert!(display.contains("node 5"));
        assert!(display.contains("Common causes"));
    }

    #[test]
    fn test_io_error_display() {
        let error = IoError::FileNotFound {
            path: "example/config.json".to_string(),
            current_dir: "/home/user/powers".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("File not found"));
        assert!(display.contains("example/config.json"));
        assert!(display.contains("Current directory"));
    }

    #[test]
    fn test_graph_error_display() {
        let error = GraphError::InvalidProbabilitySum {
            node_id: 2,
            sum: 0.85,
            edges: "[(2→3, 0.3), (2→4, 0.25), (2→5, 0.3)]".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Node 2"));
        assert!(display.contains("0.85"));
        assert!(display.contains("sum to 1.0"));
    }

    #[test]
    fn test_powers_error_conversion() {
        let validation_error = ValidationError::InvalidFieldValue {
            file: "test.json".to_string(),
            field: "value".to_string(),
            value: "invalid".to_string(),
            constraint: "must be numeric".to_string(),
            suggestion: "Use a number".to_string(),
        };

        let powers_error: PowersError = Box::new(validation_error).into();
        assert!(matches!(powers_error, PowersError::Validation(_)));
    }

    #[test]
    fn test_string_conversion() {
        let error: PowersError = "Generic error message".into();
        assert!(matches!(error, PowersError::Other(_)));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let powers_error: PowersError = io_err.into();
        assert!(matches!(powers_error, PowersError::Io(_)));
    }

    #[test]
    fn test_json_error_conversion() {
        let json_str = "{ invalid json }";
        let json_err =
            serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let powers_error: PowersError = json_err.into();
        assert!(matches!(powers_error, PowersError::Validation(_)));
    }
}
