# T-006: Define DisplayRenderer trait

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-004](./ticket-004-define-display-context.md)
> **Blocks**: [T-007](./ticket-007-automation-renderer.md)

## Files to Read Before Starting

- `src/display/context.rs` - DisplayContext (from T-004)
- `src/display/config.rs` - DisplayConfig (from T-002)
- `src/sddp/mod.rs` - TrainingResult, SimulationTrajectory types

## Context

### Background

The `DisplayRenderer` trait defines the interface for all display profile implementations. Each profile (Advanced, Standard, Minimal, Automation) implements this trait to render output in its specific format.

### Current State

No renderer abstraction exists. Output is directly formatted in the logging module.

## Specification

### DisplayRenderer Trait

```rust
//! Display renderer trait and common utilities.
//!
//! The `DisplayRenderer` trait provides a common interface for all display
//! profile implementations, enabling polymorphic rendering.

use crate::display::config::DisplayConfig;
use crate::display::context::DisplayContext;
use crate::sddp::{TrainingResult, SimulationTrajectory};

/// Trait for rendering display output.
///
/// Implementors produce formatted strings for different phases of execution.
/// The trait is object-safe to allow dynamic dispatch based on profile selection.
///
/// # Thread Safety
///
/// Renderers must be `Send + Sync` to support potential future async logging.
pub trait DisplayRenderer: Send + Sync {
    /// Render the header displayed at program start.
    ///
    /// Includes program name, version, and configuration summary.
    ///
    /// # Arguments
    ///
    /// * `config` - Display configuration
    /// * `iterations` - Total planned iterations
    /// * `forward_passes` - Forward passes per iteration
    /// * `cut_selection` - Whether cut selection is enabled
    ///
    /// # Returns
    ///
    /// Formatted string ready for output.
    fn render_header(
        &self,
        config: &DisplayConfig,
        iterations: usize,
        forward_passes: usize,
        cut_selection: bool,
    ) -> String;
    
    /// Render the table header row (if applicable).
    ///
    /// For table-based renderers, this produces column headers.
    /// For non-table renderers (JSON), this may return empty string.
    fn render_table_header(&self, config: &DisplayConfig) -> String;
    
    /// Render a single iteration's output.
    ///
    /// This is called after each training iteration completes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Complete iteration context with all metrics
    /// * `config` - Display configuration
    ///
    /// # Returns
    ///
    /// Formatted string for this iteration. May be multi-line.
    fn render_iteration(
        &self,
        ctx: &DisplayContext,
        config: &DisplayConfig,
    ) -> String;
    
    /// Render the training completion summary.
    ///
    /// Called once after all iterations complete.
    ///
    /// # Arguments
    ///
    /// * `result` - Complete training result with statistics
    /// * `config` - Display configuration
    ///
    /// # Returns
    ///
    /// Formatted summary string.
    fn render_training_summary(
        &self,
        result: &TrainingResult,
        config: &DisplayConfig,
    ) -> String;
    
    /// Render the simulation start message.
    ///
    /// Called before simulation begins.
    ///
    /// # Arguments
    ///
    /// * `num_scenarios` - Number of simulation scenarios
    /// * `config` - Display configuration
    fn render_simulation_start(
        &self,
        num_scenarios: usize,
        config: &DisplayConfig,
    ) -> String;
    
    /// Render the simulation completion summary.
    ///
    /// Called after all simulation scenarios complete.
    ///
    /// # Arguments
    ///
    /// * `trajectories` - All simulation trajectories
    /// * `elapsed` - Simulation duration
    /// * `config` - Display configuration
    ///
    /// # Returns
    ///
    /// Formatted summary with cost statistics.
    fn render_simulation_summary(
        &self,
        trajectories: &[SimulationTrajectory],
        elapsed: std::time::Duration,
        config: &DisplayConfig,
    ) -> String;
    
    /// Render an error message.
    ///
    /// For styled renderers, this adds error formatting (color, icon).
    ///
    /// # Arguments
    ///
    /// * `message` - Error message text
    /// * `config` - Display configuration
    fn render_error(
        &self,
        message: &str,
        config: &DisplayConfig,
    ) -> String;
    
    /// Render a warning message.
    ///
    /// For styled renderers, this adds warning formatting.
    fn render_warning(
        &self,
        message: &str,
        config: &DisplayConfig,
    ) -> String;
    
    /// Get the display profile this renderer implements.
    fn profile(&self) -> super::config::DisplayProfile;
    
    /// Whether this renderer uses colors.
    ///
    /// Used for testing and documentation.
    fn uses_color(&self) -> bool;
}
```

### Renderer Factory

```rust
use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::renderers::{
    AdvancedRenderer, StandardRenderer, MinimalRenderer, AutomationRenderer
};

/// Create a renderer for the given profile.
///
/// # Arguments
///
/// * `profile` - Display profile to create renderer for
/// * `config` - Display configuration for initialization
///
/// # Returns
///
/// Boxed trait object implementing the selected profile.
pub fn create_renderer(config: &DisplayConfig) -> Box<dyn DisplayRenderer> {
    match config.profile {
        DisplayProfile::Advanced => Box::new(AdvancedRenderer::new(config)),
        DisplayProfile::Standard => Box::new(StandardRenderer::new(config)),
        DisplayProfile::Minimal => Box::new(MinimalRenderer::new(config)),
        DisplayProfile::Automation => Box::new(AutomationRenderer::new()),
    }
}
```

### DisplayManager Struct (Optional Helper)

```rust
/// High-level display manager.
///
/// Wraps a renderer and provides convenient methods for the SDDP algorithm.
pub struct DisplayManager {
    renderer: Box<dyn DisplayRenderer>,
    config: DisplayConfig,
}

impl DisplayManager {
    /// Create a new display manager with the given configuration.
    pub fn new(config: DisplayConfig) -> Self {
        let renderer = create_renderer(&config);
        Self { renderer, config }
    }
    
    /// Output iteration display.
    ///
    /// Checks `should_print` flag before rendering.
    pub fn iteration(&self, ctx: &DisplayContext) {
        if ctx.should_print {
            let output = self.renderer.render_iteration(ctx, &self.config);
            print!("{}", output);
        }
    }
    
    // ... other convenience methods ...
}
```

## Acceptance Criteria

- [ ] `DisplayRenderer` trait defined with all methods
- [ ] Trait is object-safe (`dyn DisplayRenderer` works)
- [ ] Trait requires `Send + Sync` for thread safety
- [ ] `create_renderer()` factory function implemented (with stubs)
- [ ] `DisplayManager` helper struct implemented
- [ ] All methods documented with purpose and arguments
- [ ] Module added to `src/display/mod.rs` exports

## Implementation Guide

### Step 1: Create renderer.rs

Define the trait with all method signatures.

### Step 2: Add stub renderers

Create placeholder implementations in `renderers/` that return empty strings or `todo!()`.

### Step 3: Implement factory

`create_renderer()` function that returns appropriate renderer.

### Step 4: Create DisplayManager

Optional helper for convenient usage from SDDP loop.

## Pitfalls to Avoid

- ⚠️ Trait must be object-safe (no `Self` in return types, no generics on methods)
- ⚠️ All methods need `&self` (not `&mut self`) to allow shared references
- ⚠️ `Send + Sync` bounds are required for `Box<dyn DisplayRenderer>`

## Testing Requirements

### Unit Tests

- [ ] Test that trait is object-safe (create `Box<dyn DisplayRenderer>`)
- [ ] Test factory creates correct renderer type for each profile
- [ ] Test stub renderers don't panic

### Integration Tests

- [ ] Test DisplayManager with mock context

## Documentation Requirements

- [ ] Trait-level documentation explaining purpose
- [ ] Doc comments on every trait method
- [ ] Examples in module docs showing usage pattern
- [ ] Note about thread safety requirements

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Trait definition is straightforward. Implementations are stubs for now.

## Definition of Done

- [ ] Trait defined and documented
- [ ] Factory function implemented with stubs
- [ ] DisplayManager helper created
- [ ] Types exported from `src/display/mod.rs`
- [ ] Compiles without errors
- [ ] PR reviewed and merged
