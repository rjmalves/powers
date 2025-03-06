use powers::{run, Config};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = Config::build(&args).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {err}");
        process::exit(1);
    });

    if let Err(e) = run(
        config.num_stages,
        config.num_iterations,
        config.num_branchings,
    ) {
        eprintln!("Application error: {e}");
        process::exit(1);
    }
}
