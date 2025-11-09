//! Log formatters for different output formats

pub mod json;
pub mod terminal;

pub use json::JsonFormatter;
pub use terminal::TerminalFormatter;
