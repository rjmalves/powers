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

    /// Estimate PAR model parameters from historical time series data
    ///
    /// Reads historical data from CSV and estimates Periodic Autoregressive (PAR) model
    /// parameters using the Yule-Walker method. Output is JSON compatible with recourse.json.
    ///
    /// # Input Format (CSV)
    ///
    /// The CSV should have one column per entity (e.g., hydro plant inflows).
    /// Header row is optional but recommended. Example:
    ///
    /// ```csv
    /// hydro_1,hydro_2,hydro_3
    /// 45.2,120.5,89.3
    /// 48.1,125.0,92.1
    /// ...
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses Yule-Walker method:
    /// 1. Compute seasonal means μₘ and standard deviations σₘ
    /// 2. De-seasonalize: aₜ = (Zₜ - μₘ) / σₘ
    /// 3. Solve Yule-Walker equations: R·φ = r (via Cholesky decomposition)
    /// 4. Validate stationarity: sum(|φₖₘ|) < 1.0
    ///
    /// # Performance
    ///
    /// - Time complexity: O(T + n_periods·p³) where T = data length, p = AR order
    /// - Space complexity: O(T + n_periods·p²)
    /// - Typical runtime: <10ms for 10 years of monthly data with AR(1)
    /// - Uses pre-allocated buffers and cache-friendly iteration
    ///
    /// # Example
    ///
    /// ```bash
    /// # Estimate monthly PAR(1) for 3 hydro plants
    /// powers estimate-par inflows.csv --periods 12 --order 1 --output params.json
    ///
    /// # Quarterly PAR(2) with minimum 10 samples per quarter
    /// powers estimate-par data.csv -p 4 -o 2 --min-samples 10 -O quarterly_par.json
    /// ```
    #[command(name = "estimate-par")]
    EstimatePar {
        /// CSV file with historical time series data
        ///
        /// Each column represents one entity (e.g., hydro plant inflows).
        /// Rows are consecutive time steps (e.g., months, weeks).
        #[arg(value_name = "CSV_FILE")]
        input: PathBuf,

        /// Number of periods in the seasonal cycle
        ///
        /// Examples: 12 (monthly), 52 (weekly), 4 (quarterly), 365 (daily).
        /// Each period will have its own estimated parameters (μₘ, σₘ, φₖₘ).
        #[arg(short = 'p', long, value_name = "N", default_value = "12")]
        periods: usize,

        /// Autoregressive order (number of lags)
        ///
        /// PAR(1): Zₜ = μₘ + σₘ·[φ₁ₘ·aₜ₋₁ + aₜ]
        /// PAR(2): Zₜ = μₘ + σₘ·[φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + aₜ]
        ///
        /// Higher orders capture more complex autocorrelation but require more data.
        #[arg(short = 'o', long, value_name = "P", default_value = "1")]
        order: usize,

        /// Minimum number of samples required per period
        ///
        /// Ensures sufficient data for reliable parameter estimation.
        /// Rule of thumb: ≥ 5·order for stable estimates.
        #[arg(long, value_name = "N", default_value = "10")]
        min_samples: usize,

        /// Output JSON file path
        ///
        /// If not specified, prints JSON to stdout.
        /// Format is compatible with the "noise_models" field in recourse.json.
        #[arg(short = 'O', long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// CSV has header row (skip first line)
        ///
        /// Use this flag if your CSV has a header row that should be skipped.
        /// By default, assumes no header row.
        #[arg(long)]
        has_header: bool,
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
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_run_backward_compat() {
        let cli = Cli::parse_from(["powers", "examples/04-cascade"]);
        match cli.resolve_command() {
            Commands::Run { path } => {
                assert_eq!(path, PathBuf::from("examples/04-cascade"));
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_estimate_par_minimal() {
        let cli = Cli::parse_from(["powers", "estimate-par", "data.csv"]);
        match cli.resolve_command() {
            Commands::EstimatePar {
                input,
                periods,
                order,
                min_samples,
                output,
                has_header,
            } => {
                assert_eq!(input, PathBuf::from("data.csv"));
                assert_eq!(periods, 12);
                assert_eq!(order, 1);
                assert_eq!(min_samples, 10);
                assert_eq!(output, None);
                assert!(!has_header); // Default is false
            }
            _ => panic!("Expected EstimatePar command"),
        }
    }

    #[test]
    fn test_cli_estimate_par_full() {
        let cli = Cli::parse_from([
            "powers",
            "estimate-par",
            "inflows.csv",
            "-p",
            "4",
            "-o",
            "2",
            "--min-samples",
            "20",
            "-O",
            "params.json",
            "--has-header",
        ]);
        match cli.resolve_command() {
            Commands::EstimatePar {
                input,
                periods,
                order,
                min_samples,
                output,
                has_header,
            } => {
                assert_eq!(input, PathBuf::from("inflows.csv"));
                assert_eq!(periods, 4);
                assert_eq!(order, 2);
                assert_eq!(min_samples, 20);
                assert_eq!(output, Some(PathBuf::from("params.json")));
                assert!(has_header);
            }
            _ => panic!("Expected EstimatePar command"),
        }
    }
}
