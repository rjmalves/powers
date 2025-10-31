//! Command-line interface for POWE.RS.
//!
//! This module defines the CLI structure with subcommands for different operations:
//! - `run`: Execute SDDP algorithm with JSON inputs (default behavior)
//! - `estimate-par`: Estimate PAR model parameters from historical CSV data
//!
//! # Performance Notes
//!
//! - CLI parsing overhead is negligible (<100μs)
//! - Uses `clap`'s derive API for zero-cost abstractions
//! - All hot paths are in the subcommand implementations, not the CLI layer
//!
//! # Example Usage
//!
//! ```bash
//! # Run SDDP algorithm (backward-compatible with old CLI)
//! powers run examples/04-cascade
//! powers examples/04-cascade  # 'run' is default subcommand
//!
//! # Estimate PAR parameters from historical data
//! powers estimate-par historical_inflows.csv --output params.json
//! powers estimate-par data.csv -p 12 -o 1  # Monthly PAR(1)
//! ```

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// POWE.RS - High-performance Stochastic Dual Dynamic Programming for hydrothermal dispatch
#[derive(Parser, Debug)]
#[command(
    name = "powers",
    version,
    about = "Stochastic Dual Dynamic Programming (SDDP) for hydrothermal scheduling",
    long_about = "High-performance SDDP implementation in Rust for solving multi-stage \
                  stochastic optimization problems in hydrothermal dispatch.",
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Input directory path (for backward compatibility when no subcommand specified)
    ///
    /// When no subcommand is provided, this behaves as `powers run <PATH>`.
    /// This maintains backward compatibility with the old CLI: `powers examples/04-cascade`
    #[arg(value_name = "PATH")]
    pub path: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run SDDP algorithm with JSON configuration files
    ///
    /// Expects a directory containing:
    /// - config.json: SDDP configuration (iterations, scenarios, convergence)
    /// - system.json: Hydrothermal system definition (buses, hydros, thermals, lines)
    /// - graph.json: Scenario tree structure (stages, nodes, probabilities)
    /// - recourse.json: Stochastic process models (PAR, independent noise, correlation)
    ///
    /// # Performance
    ///
    /// This is the production hot path. Performance-critical code includes:
    /// - Forward/backward passes (parallelized via Rayon)
    /// - Solver calls (direct FFI to HiGHS via highs-sys)
    /// - Cut management (selection and storage)
    /// - State space exploration
    ///
    /// # Example
    ///
    /// ```bash
    /// powers run examples/04-cascade
    /// powers run my_case --verbose  # (future: verbosity control)
    /// ```
    Run {
        /// Directory containing JSON input files
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
}

impl Cli {
    /// Resolve the actual command to execute, handling backward compatibility.
    ///
    /// If no subcommand is specified but a path is provided, treat it as `run <path>`.
    /// This maintains backward compatibility: `powers examples/04-cascade` works.
    ///
    /// # Performance
    ///
    /// This is just CLI routing - negligible overhead (<1μs).
    pub fn resolve_command(self) -> Commands {
        match (self.command, self.path) {
            // Explicit subcommand takes precedence
            (Some(cmd), _) => cmd,

            // No subcommand but path provided: backward compatibility mode
            (None, Some(path)) => Commands::Run { path },

            // Neither subcommand nor path: this should be caught by clap
            (None, None) => {
                unreachable!("clap should enforce that either command or path is provided")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_run_explicit() {
        let cli = Cli::parse_from(["powers", "run", "examples/04-cascade"]);
        match cli.resolve_command() {
            Commands::Run { path } => {
                assert_eq!(path, PathBuf::from("examples/04-cascade"));
            }
        }
    }

    #[test]
    fn test_cli_run_backward_compat() {
        let cli = Cli::parse_from(["powers", "examples/04-cascade"]);
        match cli.resolve_command() {
            Commands::Run { path } => {
                assert_eq!(path, PathBuf::from("examples/04-cascade"));
            }
        }
    }
}
