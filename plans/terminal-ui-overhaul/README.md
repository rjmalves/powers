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
- [ ] **T-001** Add crossterm dependency
- [ ] **T-002** Define DisplayProfile enum
- [ ] **T-003** Implement CostStatistics struct
- [ ] **T-004** Define DisplayContext struct
- [ ] **T-005** Terminal detection utilities
- [ ] **T-006** Define DisplayRenderer trait
- [ ] **T-007** Implement AutomationRenderer (JSON)
- [ ] **T-008** Extend CLI with display flags
- [ ] **T-009** Extend config schema
- [ ] **T-010** Build DisplayContext from iteration data
- [ ] **T-011** Expose first-stage branching costs
- [ ] **T-012** Integrate into training loop
- [ ] **T-013** Integration tests

### Epic 2: Training Display (Sprints 3-4)
- [ ] **T-014** Header rendering with box-drawing
- [ ] **T-015** Progress bar component
- [ ] **T-016** Convergence metrics panel
- [ ] **T-017** Timing breakdown panel
- [ ] **T-018** Cut statistics panel
- [ ] **T-019** First-stage branching panel
- [ ] **T-020** MinimalRenderer implementation
- [ ] **T-021** StandardRenderer implementation
- [ ] **T-022** AdvancedRenderer implementation
- [ ] **T-023** Color theming system
- [ ] **T-024** Narrow terminal handling
- [ ] **T-025** Visual regression tests

### Epic 3: Simulation & Polish (Sprint 5)
- [ ] **T-026** SimulationDisplayContext
- [ ] **T-027** Simulation results table
- [ ] **T-028** Summary statistics display
- [ ] **T-029** Error/warning styling
- [ ] **T-030** Documentation update
- [ ] **T-031** Performance validation
- [ ] **T-032** Migration guide

## 🏗️ Architecture Overview

```
src/display/
├── mod.rs              # Module exports
├── profile.rs          # DisplayProfile, ColorMode enums
├── context.rs          # DisplayContext, CostStatistics
├── terminal.rs         # Terminal detection, capabilities
├── renderer/
│   ├── mod.rs          # DisplayRenderer trait
│   ├── automation.rs   # JSON output
│   ├── minimal.rs      # Single-line output
│   ├── standard.rs     # Multi-line with key metrics
│   └── advanced.rs     # Full rich display
└── components/
    ├── header.rs       # Box-drawing header
    ├── progress.rs     # Progress bar
    ├── metrics.rs      # Convergence panel
    ├── timing.rs       # Timing breakdown
    └── cuts.rs         # Cut statistics
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
