mod mps;
use highs;
use mps::mps2highs;
use std::error::Error;
use std::fs;
use std::time::Instant;

fn set_model_options(model: &mut highs::Model) {
    model.set_option("presolve", "on");
    model.set_option("solver", "simplex");
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("primal_feasibility_tolerance", 1e-6);
    model.set_option("dual_feasibility_tolerance", 1e-6);
    model.set_option("time_limit", 300);
}

pub fn run(filepath: &str) -> Result<(), Box<dyn Error>> {
    let start_reading_time = Instant::now();
    let file = fs::read_to_string(filepath)?;
    let mut model = mps2highs(file.lines())?;
    set_model_options(&mut model);
    let end_reading_time = start_reading_time.elapsed();
    println!("Time for MPS parsing: {:?}", end_reading_time);
    println!("Beginning to solve...");
    let start_solving_time = Instant::now();
    let solved = model.solve();
    match solved.status() {
        highs::HighsModelStatus::Optimal => {
            println!("Success =)");
        }
        highs::HighsModelStatus::Infeasible => {
            println!("Infeasible =(");
        }
        highs::HighsModelStatus::Unbounded => {
            println!("Unbounded ?!");
        }
        _ => {
            println!("Some other error...");
        }
    }
    let end_solving_time = start_solving_time.elapsed();
    println!("Time for solving: {:?}", end_solving_time);
    Ok(())
}

pub struct Config {
    pub mps_filename: String,
}

impl Config {
    pub fn build(args: &[String]) -> Result<Self, &'static str> {
        if args.len() < 2 {
            return Err("not enough arguments");
        }

        let mps_filename = args[1].clone();

        Ok(Self { mps_filename })
    }
}
