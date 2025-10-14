use clap::Parser;
use powers_rs::cli::{Cli, Commands};
use std::process;

fn main() {
    // Parse CLI arguments
    let cli = Cli::parse();
    let command = cli.resolve_command();

    // Execute the appropriate subcommand
    let result = match command {
        Commands::Run { path } => {
            // SDDP algorithm execution (hot path)
            powers_rs::run(&path)
        }
        Commands::EstimatePar {
            input,
            periods,
            order,
            min_samples,
            output,
            has_header,
        } => {
            // PAR parameter estimation from CSV
            powers_rs::estimate_par(
                &input,
                periods,
                order,
                min_samples,
                output.as_deref(),
                has_header,
            )
        }
    };

    // Handle errors with proper exit codes
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
