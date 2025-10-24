use clap::Parser;
use powers_rs::cli::{Cli, Commands};
use std::process;

fn main() {
    let cli = Cli::parse();
    let command = cli.resolve_command();

    let result = match command {
        Commands::Run { path } => powers_rs::run(&path),
        Commands::EstimatePar {
            input,
            periods,
            order,
            min_samples,
            output,
            has_header,
        } => powers_rs::estimate_par(
            &input,
            periods,
            order,
            min_samples,
            output.as_deref(),
            has_header,
        ),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
