//! Display renderer implementations.
//!
//! Provides concrete implementations for each display profile.

pub mod advanced;
pub mod automation;
pub mod minimal;
pub mod standard;

pub use advanced::AdvancedRenderer;
pub use automation::AutomationRenderer;
pub use minimal::MinimalRenderer;
pub use standard::StandardRenderer;
