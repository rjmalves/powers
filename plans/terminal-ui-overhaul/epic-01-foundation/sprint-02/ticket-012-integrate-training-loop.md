# T-012: Integrate display system with training loop

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: [T-010](./ticket-010-build-display-context.md), [T-011](./ticket-011-expose-first-stage-costs.md)
> **Blocks**: [T-013](./ticket-013-integration-tests.md)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Lines 1850-2050: training loop
- `src/lib.rs` - `run()` function where SDDP is called
- `src/display/renderer.rs` - DisplayRenderer trait and factory
- `src/display/context.rs` - DisplayContext and IterationTracker

## Context

### Background

This is the critical integration ticket that connects the display system to the SDDP training loop. After this, running the program will use the new display system instead of the old logging approach.

### Current State

Training loop uses `LogContext` and `::log::info!()` for output:
```rust
crate::logging::LogContext::set(crate::logging::LogContext {
    iteration: Some(index + 1),
    lower_bound: Some(lower_bound),
    // ...
});
::log::info!("Iteration complete");
crate::logging::LogContext::clear();
```

## Specification

### Changes to lib.rs run()

```rust
use crate::display::{
    DisplayConfig, DisplayManager, TerminalCapabilities, create_renderer
};

pub fn run(
    input_path: &Path,
    log_level_override: Option<String>,
    log_format_override: Option<String>,
    // New parameters
    profile_override: Option<String>,
    no_color: bool,
    quiet: bool,
) -> Result<(), Box<dyn Error>> {
    // ... load config ...
    
    // Build display configuration
    let mut display_config = DisplayConfig::from(config.display.clone());
    
    // Apply CLI overrides
    if quiet {
        display_config.profile = DisplayProfile::Minimal;
    } else if let Some(ref profile_str) = profile_override {
        display_config.profile = profile_str.parse()?;
    }
    
    if no_color {
        display_config.color = ColorMode::Never;
    }
    
    // Apply terminal detection
    let caps = TerminalCapabilities::detect();
    display_config.apply_terminal_caps(&caps);
    
    // Create renderer
    let renderer = create_renderer(&display_config);
    
    // ... create SDDP instance ...
    
    // Training with display
    let training_result = sddp.train_with_display(&renderer, &display_config)?;
    
    // ... rest of run ...
}
```

### New train_with_display Method

```rust
impl SddpAlgorithm {
    /// Train with integrated display output.
    pub fn train_with_display(
        &mut self,
        renderer: &dyn DisplayRenderer,
        display_config: &DisplayConfig,
    ) -> Result<TrainingResult, String> {
        // Print header
        let header = renderer.render_header(
            display_config,
            self.num_iterations,
            self.num_forward_passes,
            self.cut_selection_enabled,
        );
        print!("{}", header);
        
        let table_header = renderer.render_table_header(display_config);
        print!("{}", table_header);
        
        // Iteration tracker
        let mut tracker = IterationTracker::new();
        tracker.start();
        
        for index in 0..self.num_iterations {
            // ... existing iteration logic ...
            
            // Build DisplayContext
            let ctx = DisplayContext::from_iteration(
                index + 1,
                self.num_iterations,
                &iteration_result,
                tracker.previous_lower_bound(),
                tracker.elapsed(),
                display_config.target_gap,
            );
            
            // Render and print if should_print
            if ctx.should_print {
                let output = renderer.render_iteration(&ctx, display_config);
                print!("{}", output);
            }
            
            // Update tracker
            tracker.update(iteration_result.lower_bound, ctx.gap_percent);
            
            // ... rest of iteration ...
        }
        
        // Training summary
        let summary = renderer.render_training_summary(&result, display_config);
        print!("{}", summary);
        
        Ok(result)
    }
}
```

### Remove Old Logging

Replace in training loop:
```rust
// Remove these:
crate::logging::LogContext::set(...);
::log::info!("Iteration complete");
crate::logging::LogContext::clear();
::log::info!("{}", "-".repeat(88));
// etc.

// Keep error logging (still uses log crate):
::log::error!("...");
```

### Update main.rs and run() signature

```rust
// main.rs
let result = match command {
    Commands::Run { path } => powers_rs::run(
        &path, 
        cli.log_level, 
        cli.log_format,
        cli.profile,
        cli.no_color,
        cli.quiet,
    ),
};
```

## Acceptance Criteria

- [ ] `run()` signature updated with new parameters
- [ ] Display configuration built and overrides applied
- [ ] Terminal capabilities detected and applied
- [ ] Renderer created based on profile
- [ ] Header rendered at training start
- [ ] Each iteration renders via DisplayRenderer
- [ ] Training summary rendered at end
- [ ] Old LogContext-based output removed
- [ ] `--profile automation` produces JSON output
- [ ] `--no-color` produces plain text
- [ ] Non-interactive terminal produces no ANSI codes

## Implementation Guide

### Step 1: Update run() signature

Add new parameters for display options.

### Step 2: Add display config construction

Build DisplayConfig from config + CLI + terminal detection.

### Step 3: Create new train_with_display method

Extract training loop into method that takes renderer.

### Step 4: Replace logging calls

Remove LogContext usage, use renderer instead.

### Step 5: Update main.rs

Pass new CLI flags to run().

### Step 6: Test all profiles

Verify each profile produces expected output.

## Pitfalls to Avoid

- ⚠️ Don't remove error/debug logging - those still use log crate
- ⚠️ Use `print!` not `println!` - renderer includes newlines
- ⚠️ Flush stdout after each print if needed for real-time display
- ⚠️ Keep backward compatibility - existing tests shouldn't break

## Testing Requirements

### Manual Testing

- [ ] `cargo run -- examples/04-cascade` produces rich output
- [ ] `cargo run -- --profile automation examples/04-cascade` produces JSON
- [ ] `cargo run -- --no-color examples/04-cascade` has no ANSI
- [ ] `cargo run -- -q examples/04-cascade` produces minimal output
- [ ] `cargo run examples/04-cascade | cat` has no ANSI (pipe detection)

### Integration Tests

- [ ] Test that output contains expected sections (header, iterations, summary)
- [ ] Test JSON output is parseable

### Unit Tests

- [ ] Test DisplayConfig construction with various combinations
- [ ] Test CLI override precedence

## Documentation Requirements

- [ ] Update lib.rs `run()` documentation
- [ ] Document new train_with_display method
- [ ] Update CHANGELOG.md with display system addition

## Effort Estimate

**Points**: 4
**Confidence**: Medium
**Rationale**: Core integration touching multiple files. Careful refactoring needed.

## Definition of Done

- [ ] Integration complete
- [ ] All profiles working
- [ ] Old logging removed
- [ ] Manual testing passed
- [ ] No regressions in CI
- [ ] PR reviewed and merged
