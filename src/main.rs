use clap::Parser;
use powers_rs::cli::{Cli, Commands};
use std::process;

fn main() {
    let cli = Cli::parse();

    // Extract CLI options before resolving command
    let log_level = cli.log_level.clone();
    let log_format = cli.log_format.clone();

    let command = cli.resolve_command();

    let result = match command {
        Commands::Run { path } => powers_rs::run(&path, log_level, log_format),
    };

    if let Err(e) = result {
        log::error!("Execution failed: {}", e);
        process::exit(1);
    }
}
