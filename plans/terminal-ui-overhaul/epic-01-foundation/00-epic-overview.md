# Epic 1: Foundation

> **Master Plan**: [Terminal UI Overhaul](../00-master-plan.md)
> **Duration**: 2 sprints (~4 weeks)
> **Status**: Not Started

## Summary

Establish the core infrastructure for the new display system: profile configuration, terminal detection, the `DisplayRenderer` trait abstraction, and basic styled output capabilities. This epic creates the foundation that all subsequent display work builds upon.

## Scope

### Included

- `DisplayProfile` enum and configuration parsing (CLI + config.json)
- `DisplayContext` struct with all metrics fields
- `CostStatistics` helper for aggregating cost distributions
- `DisplayRenderer` trait definition
- Terminal capability detection (color support, interactive vs piped)
- `crossterm` integration for styled output
- Basic `AutomationRenderer` (JSON output) - simplest renderer to validate architecture
- CLI flag additions (`--profile`, `--no-color`, `--quiet`)
- Config schema extension for `display` section
- Integration point in SDDP training loop (prepare `DisplayContext`)

### Excluded (Deferred to Epic 2)

- Advanced/Standard/Minimal renderer implementations
- Box-drawing table components
- Progress bars and trend indicators
- First-stage branching cost collection

### Excluded (Deferred to Epic 3)

- Simulation display enhancements
- Error/warning visual treatment
- Documentation and examples

## Dependencies

- **Requires**: None (first epic)
- **Enables**: Epic 2 (Training Display), Epic 3 (Simulation & Polish)

## Acceptance Criteria

- [ ] `DisplayProfile::Automation` produces valid JSON lines for each iteration
- [ ] `--profile automation` CLI flag works and selects JSON output
- [ ] `--no-color` disables ANSI codes even in interactive terminals
- [ ] Non-interactive terminals (pipes, CI) automatically disable colors
- [ ] `DisplayContext` captures all metrics from `IterationResult` and timing
- [ ] Config file `display.profile` setting is respected
- [ ] CLI flags override config file settings
- [ ] Existing tests continue to pass
- [ ] New unit tests for profile parsing, terminal detection, JSON output

## Technical Approach

### Phase 1: Core Types (Sprint 1)

1. Create `src/display/` module structure
2. Define `DisplayProfile`, `DisplayConfig`, `DisplayContext`, `CostStatistics`
3. Implement `CostStatistics::from_slice()` for computing stats
4. Define `DisplayRenderer` trait with all required methods
5. Add `crossterm` dependency

### Phase 2: Terminal & Automation (Sprint 1-2)

1. Implement terminal detection (interactive, color capability)
2. Create `AutomationRenderer` producing JSON lines
3. Wire up to SDDP training loop - build `DisplayContext` each iteration
4. Replace current logging with display system call

### Phase 3: Configuration (Sprint 2)

1. Extend CLI with new flags
2. Extend config schema with `display` section
3. Implement precedence: CLI > config > defaults
4. Add `--quiet` as alias for `--profile minimal`

## Key Files to Create

| File | Purpose |
|------|---------|
| `src/display/mod.rs` | Module root, public exports |
| `src/display/config.rs` | `DisplayProfile`, `DisplayConfig` |
| `src/display/context.rs` | `DisplayContext`, `CostStatistics` |
| `src/display/renderer.rs` | `DisplayRenderer` trait |
| `src/display/terminal.rs` | Terminal detection utilities |
| `src/display/renderers/mod.rs` | Renderer implementations module |
| `src/display/renderers/automation.rs` | JSON renderer |

## Key Files to Modify

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `crossterm` dependency |
| `src/lib.rs` | Add `pub mod display;` |
| `src/cli.rs` | Add `--profile`, `--no-color`, `--quiet` flags |
| `src/input.rs` | Extend config parsing for `display` section |
| `src/sddp/mod.rs` | Build `DisplayContext`, call renderer |

## Estimated Effort

- **Sprint 1**: Core types, terminal detection, automation renderer
- **Sprint 2**: CLI/config integration, SDDP loop integration, testing

**Total**: ~25-30 story points across 2 sprints

## Risks

| Risk | Mitigation |
|------|------------|
| `crossterm` version conflicts | Pin specific version, test on all platforms |
| Terminal detection false positives | Conservative defaults (assume no color if uncertain) |
| Breaking config schema | Make `display` section optional with defaults |

## Definition of Done

- [ ] All tickets in Sprint 1 and Sprint 2 complete
- [ ] `cargo test` passes with no regressions
- [ ] `cargo run -- examples/04-cascade --profile automation` produces JSON
- [ ] Piping output to file produces no ANSI codes
- [ ] Code reviewed and merged to main branch
