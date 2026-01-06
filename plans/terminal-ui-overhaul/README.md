# Terminal UI Overhaul - Implementation Plan

A comprehensive plan to transform POWE.RS's terminal output from basic streaming logs into a rich, beautiful terminal display with colors, box-drawing, progress indicators, and real-time statistics.

## 📋 Quick Links

| Document | Description |
|----------|-------------|
| [Master Plan](./00-master-plan.md) | Architecture, phases, sample output, success metrics |
| [Epic 1: Foundation](./epic-01-foundation/00-epic-overview.md) | Core infrastructure (2 sprints, 13 tickets) |
| [Epic 2: Training Display](./epic-02-training-display/00-epic-overview.md) | Rich training iteration UI (2 sprints, 12 tickets) |
| [Epic 3: Simulation & Polish](./epic-03-simulation-and-polish/00-epic-overview.md) | Simulation display & final refinements (1 sprint, 7 tickets) |

## 🎯 Goals

- **Beautiful output** inspired by htop, Python rich library, Claude Code
- **Multiple profiles**: Advanced (default), Standard, Minimal, Automation (JSON)
- **Real-time metrics**: convergence gap, timing breakdown, cut statistics
- **Cross-platform**: works on Linux, macOS, Windows Terminal

## 📊 Implementation Progress

### Epic 1: Foundation (Sprints 1-2)

#### Sprint 1: Core Types and Traits
- [ ] **T-001** Add crossterm dependency and module structure
- [ ] **T-002** Define DisplayProfile enum
- [ ] **T-003** Implement CostStatistics struct
- [ ] **T-004** Define DisplayContext struct
- [ ] **T-005** Terminal detection utilities
- [ ] **T-006** Define DisplayRenderer trait
- [ ] **T-007** Implement AutomationRenderer (JSON)

#### Sprint 2: Configuration and Integration
- [ ] **T-008** Extend CLI with display flags
- [ ] **T-009** Extend config schema
- [ ] **T-010** Build DisplayContext from iteration data
- [ ] **T-011** Expose first-stage branching costs
- [ ] **T-012** Integrate into training loop
- [ ] **T-013** Integration tests

### Epic 2: Training Display (Sprints 3-4)

#### Sprint 3: Display Components and MinimalRenderer
- [ ] **T-014** Implement color utilities with crossterm
- [ ] **T-015** Implement statistics formatter component
- [ ] **T-016** Implement trend indicators component
- [ ] **T-017** Implement progress bar component
- [ ] **T-018** Implement table builder component
- [ ] **T-019** Implement MinimalRenderer

#### Sprint 4: Advanced and Standard Renderers
- [ ] **T-020** Implement AdvancedRenderer header
- [ ] **T-021** Implement AdvancedRenderer iteration row
- [ ] **T-022** Implement AdvancedRenderer training summary
- [ ] **T-023** Implement StandardRenderer
- [ ] **T-024** Add target gap progress visualization
- [ ] **T-025** Polish and visual consistency review

### Epic 3: Simulation & Polish (Sprint 5)
- [ ] **T-026** Implement enhanced simulation summary
- [ ] **T-027** Add percentile calculation to CostStatistics
- [ ] **T-028** Implement error message rendering
- [ ] **T-029** Implement warning message rendering
- [ ] **T-030** Update documentation for display system
- [ ] **T-031** Update examples with display configuration
- [ ] **T-032** Performance validation and optimization

## 🏗️ Architecture Overview

```
src/display/
├── mod.rs              # Module exports
├── config.rs           # DisplayConfig, DisplayProfile
├── context.rs          # DisplayContext, CostStatistics
├── terminal.rs         # Terminal detection, capabilities
├── renderer.rs         # DisplayRenderer trait
├── renderers/
│   ├── mod.rs
│   ├── automation.rs   # JSON output
│   ├── minimal.rs      # Progress bar only
│   ├── standard.rs     # Key metrics with colors
│   └── advanced.rs     # Full rich display
└── components/
    ├── mod.rs
    ├── color.rs        # Semantic color utilities
    ├── progress.rs     # Progress bar
    ├── indicators.rs   # Trend arrows, status icons
    ├── statistics.rs   # Cost/timing formatters
    └── table.rs        # Box-drawing table builder
```

## 🚀 Getting Started

To begin implementation, start with Epic 1, Sprint 1:
1. Read [Sprint 1 Overview](./epic-01-foundation/sprint-01/00-sprint-overview.md)
2. Implement tickets T-001 through T-007 in order
3. Run tests after each ticket

## 📝 Notes

- **Breaking changes accepted**: Output format will change significantly
- **Default profile**: Advanced (richest display)
- **Backward compatibility**: Automation profile provides JSON for scripts
- **Dependencies**: Only `crossterm` added (already widely used in Rust ecosystem)
