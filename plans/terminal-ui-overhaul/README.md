# Terminal UI Overhaul - Implementation Plan

A comprehensive plan to transform POWE.RS's terminal output from basic streaming logs into a rich, beautiful terminal display with colors, box-drawing, progress indicators, and real-time statistics.

## 📋 Quick Links

| Document | Description |
|----------|-------------|
| [Master Plan](./00-master-plan.md) | Architecture, phases, sample output, success metrics |
| [Epic 1: Foundation](./epic-01-foundation/00-epic-overview.md) | Core infrastructure (2 sprints) ✅ COMPLETE |
| [Epic 2: Training Display](./epic-02-training-display/00-epic-overview.md) | Rich training iteration UI (2 sprints) ✅ COMPLETE |
| [Epic 2.5: UI Fixes](./epic-02.5-ui-fixes/00-epic-overview.md) | Critical output fixes (1 sprint) ✅ COMPLETE |
| [Epic 2.6: Robust Table Alignment](./epic-02.6-robust-table-alignment/00-epic-overview.md) | Table alignment with guaranteed consistency (1 sprint) ✅ COMPLETE |
| [Epic 3: Simulation & Polish](./epic-03-simulation-and-polish/00-epic-overview.md) | Simulation display & final refinements (1 sprint) |

## 🎯 Goals

- **Beautiful output** inspired by htop, Python rich library, Claude Code
- **Multiple profiles**: Advanced (default), Standard, Minimal, Automation (JSON)
- **Real-time metrics**: convergence gap, timing breakdown, cut statistics
- **Cross-platform**: works on Linux, macOS, Windows Terminal

## 📊 Implementation Progress

### Epic 1: Foundation ✅ COMPLETE

- All 13 tickets implemented
- Core types, traits, and configuration in place
- Display system integrated into training loop

### Epic 2: Training Display ✅ COMPLETE

- All 12 tickets implemented (749 tests passing)
- Four renderers: Advanced, Standard, Minimal, Automation
- Rich components: colors, statistics, progress bars, tables
- See [EPIC02_PROGRESS.md](./EPIC02_PROGRESS.md) for details

### Epic 2.5: UI Fixes ✅ COMPLETE

- Silenced legacy logging
- Fixed duplicate table headers
- Fixed minor display issues

### Epic 2.6: Robust Table Alignment ✅ COMPLETE

**Root cause identified**: Rust's `format!` macro expands fields instead of truncating when content exceeds width. This causes table misalignment.

**Solution**: Centralized column configuration with explicit truncation safety.

| ID | Title | Status | Points |
|----|-------|--------|--------|
| T-031 | Remove tabled dependency and revert related code | ✅ DONE | 2 |
| T-032 | Create robust table formatting utilities | ✅ DONE | 5 |
| T-033 | Migrate AdvancedRenderer to new utilities | ✅ DONE | 3 |
| T-034 | Migrate StandardRenderer to new utilities | ✅ DONE | 2 |
| T-035 | Add comprehensive alignment tests | ✅ DONE | 2 |

**Key changes:**
1. Removed `tabled` dependency (incompatible with progressive rendering)
2. Created `table_format.rs` with ANSI-aware width calculation
3. Use generous column widths with 25-35% slack
4. Explicit truncation prevents format expansion
5. Comprehensive tests prevent regression (19 new alignment tests)

### Epic 3: Simulation & Polish (Ready to Start)

- Enhanced simulation display
- Final documentation
- Performance validation

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
    ├── table.rs        # Box-drawing table builder (BorderStyle)
    └── table_format.rs # Robust cell formatting utilities with alignment guarantees
```

## 🚀 Next Steps

Continue with Epic 3 (Simulation & Polish):

1. Read [Epic 3 Overview](./epic-03-simulation-and-polish/00-epic-overview.md)
2. Use `cargo build -j 1` and `cargo test -j 1 -- --test-threads=1` to avoid RAM issues

## 📝 Notes

- **Breaking changes accepted**: Output format will change significantly
- **Default profile**: Advanced (richest display)
- **Backward compatibility**: Automation profile provides JSON for scripts
- **Dependencies**: Only `crossterm` needed (tabled was removed)
- **Memory-constrained builds**: Use `-j 1` flag for single-threaded compilation
