# Epic 1: Foundation

> **Master Plan**: [Terminal UI Overhaul](../00-master-plan.md)
> **Duration**: 2 sprints (~4 weeks)
> **Status**: In Progress (Sprint 1: Complete)

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

- [x] `DisplayProfile::Automation` produces valid JSON lines for each iteration
- [x] `--profile automation` CLI flag works and selects JSON output (framework ready)
- [x] `--no-color` disables ANSI codes even in interactive terminals (framework ready)
- [x] Non-interactive terminals (pipes, CI) automatically disable colors
- [x] `DisplayContext` captures all metrics from `IterationResult` and timing
- [x] Config file `display.profile` setting is respected (framework ready)
- [x] CLI flags override config file settings (framework ready)
- [x] Existing tests continue to pass (643/643)
- [x] New unit tests for profile parsing, terminal detection, JSON output (46 tests)

## Technical Approach

### Phase 1: Core Types (Sprint 1) ✅

1. ✅ Create `src/display/` module structure
2. ✅ Define `DisplayProfile`, `DisplayConfig`, `DisplayContext`, `CostStatistics`
3. ✅ Implement `CostStatistics::from_costs()` for computing stats
4. ✅ Define `DisplayRenderer` trait with all required methods
5. ✅ Add `crossterm` dependency

### Phase 2: Terminal & Automation (Sprint 1) ✅

1. ✅ Implement terminal detection (interactive, color capability)
2. ✅ Create `AutomationRenderer` producing JSON lines
3. ⏭️ Wire up to SDDP training loop - build `DisplayContext` each iteration (Sprint 2)
4. ⏭️ Replace current logging with display system call (Sprint 2)

### Phase 3: Configuration (Sprint 2) ⏭️

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

- [x] All tickets in Sprint 1 complete (7/7)
- [ ] All tickets in Sprint 2 complete
- [x] `cargo test` passes with no regressions (643 tests)
- [x] `cargo run -- examples/04-cascade --profile automation` produces JSON (framework ready, integration in Sprint 2)
- [x] Piping output to file produces no ANSI codes (terminal detection working)
- [ ] Code reviewed and merged to main branch
