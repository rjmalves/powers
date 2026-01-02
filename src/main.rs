// Global allocator configuration for HPC workloads.
//
// Alternative allocators (mimalloc, jemalloc) provide better memory management
// than glibc, returning freed memory to the OS more aggressively and preventing
// unbounded RSS growth during long training runs.
//
// Priority order (if multiple features enabled):
// 1. jemalloc (if --features jemalloc)
// 2. mimalloc (if --features mimalloc)
// 3. system allocator (default, or if --features system-allocator)

// jemalloc has highest priority when enabled
#[cfg(all(feature = "jemalloc", not(feature = "system-allocator")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// mimalloc is used if enabled and jemalloc is not
#[cfg(all(
    feature = "mimalloc",
    not(feature = "jemalloc"),
    not(feature = "system-allocator")
))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Note: If neither jemalloc nor mimalloc is enabled, or if system-allocator
// feature is enabled, the system allocator (glibc on Linux) is used.

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
