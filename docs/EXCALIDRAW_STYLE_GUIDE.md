# POWE.RS Excalidraw Style Guide

## Overview

This guide establishes the visual language for all POWE.RS diagrams created in Excalidraw. The goal is to create consistent, professional, yet approachable hand-drawn style diagrams that effectively communicate SDDP concepts.

> **Reference**: Follow [SDDP.jl notation](https://sddp.dev/stable/) for all mathematical symbols and variable naming.

## Directory Structure

```
docs/
├── diagrams/
│   ├── excalidraw/          # Source .excalidraw files (version controlled)
│   │   ├── policy-graphs/
│   │   ├── architecture/
│   │   ├── algorithm/
│   │   └── components/      # Reusable component library
│   └── exports/             # Generated PNG/SVG files
│       ├── light/           # Light theme exports
│       └── dark/            # Dark theme exports (optional)
└── EXCALIDRAW_STYLE_GUIDE.md
```

---

## Color Palette

### Primary Colors (SDDP Domain)

| Purpose | Fill Color | Stroke Color | Hex (Fill) | Hex (Stroke) | Usage |
|---------|------------|--------------|------------|--------------|-------|
| **Stages/Nodes** | Light Blue | Blue | `#a5d8ff` | `#1971c2` | SDDP stages, policy graph nodes |
| **Hydro Plants** | Light Green | Green | `#b2f2bb` | `#2f9e44` | Reservoirs, turbines, hydro elements |
| **Thermal Plants** | Light Yellow | Orange | `#ffec99` | `#f08c00` | Thermal generators, fuel costs |
| **Cycles/Feedback** | Light Red | Red | `#ffc9c9` | `#e03131` | Cycle edges, infinite horizon loops |
| **Parallelization** | Light Purple | Purple | `#d0bfff` | `#7048e8` | MPI ranks, OpenMP threads |
| **Data/State** | Light Cyan | Teal | `#99e9f2` | `#0c8599` | State variables, data flows |
| **Terminal/Null** | Light Gray | Dark Gray | `#e9ecef` | `#495057` | Terminal nodes, inactive states |

### Secondary Colors (Emphasis)

| Purpose | Color | Hex | Usage |
|---------|-------|-----|-------|
| **Highlight** | Warm Yellow | `#fff3bf` | Important annotations, key insights |
| **Warning** | Coral | `#ff8787` | Constraints, barriers, synchronization |
| **Success** | Mint | `#96f2d7` | Convergence, optimal solutions |
| **Background** | Off-white | `#f8f9fa` | Container backgrounds, subgroups |

### Stroke Styles

| Style | Line Type | Width | Usage |
|-------|-----------|-------|-------|
| **Solid** | Continuous | 2px | Normal transitions, direct relationships |
| **Dashed** | `[5, 5]` | 2px | Cycles, feedback loops, optional paths |
| **Dotted** | `[2, 4]` | 1px | Implicit relationships, weak dependencies |
| **Bold** | Continuous | 4px | Critical paths, emphasis |

---

## Typography

### Font Family

Use **Virgil** (Excalidraw's default hand-drawn font) for all text to maintain the sketchy aesthetic.

### Text Sizes

| Element | Size | Style | Example |
|---------|------|-------|---------|
| **Diagram Title** | 32px | Bold | `SDDP Policy Graph` |
| **Section Headers** | 24px | Bold | `Forward Pass` |
| **Node Labels** | 16px | Normal | `Stage t` |
| **Edge Labels** | 14px | Normal | `p(ω) = 0.3` |
| **Annotations** | 12px | Italic | `cuts added here` |
| **Mathematical** | 16px | Normal | `V_t(x_{t-1})` |

### Mathematical Notation

Since Excalidraw doesn't support LaTeX, use these conventions:

| Concept | Representation | Example |
|---------|---------------|---------|
| Subscripts | Underscore + smaller text | `V_t`, `x_{t-1}` |
| Superscripts | Caret notation | `z^k`, `α^k` |
| Greek letters | Unicode | `α β γ δ ε θ λ μ π ρ σ ω Ω` |
| Expectations | E[...] | `E[cost]` |
| Summations | Σ with limits as text | `Σ_{t=1}^T` |

---

## Shape Library

### Stage Nodes (Policy Graph)

```
┌─────────────────────────────────────┐
│  STAGE NODE                         │
│                                     │
│  Shape: Rounded Rectangle           │
│  Corner radius: 8px                 │
│  Fill: #a5d8ff (light blue)         │
│  Stroke: #1971c2 (blue), 2px        │
│  Size: 120×60 px (minimum)          │
│                                     │
│  Label format:                      │
│    Line 1: "Stage t" (bold, 16px)   │
│    Line 2: context (italic, 12px)   │
│                                     │
│  ┌──────────────────┐               │
│  │    Stage 1       │               │
│  │  initial state   │               │
│  └──────────────────┘               │
└─────────────────────────────────────┘
```

### Terminal Node

```
┌─────────────────────────────────────┐
│  TERMINAL NODE                      │
│                                     │
│  Shape: Rounded Rectangle           │
│  Fill: #e9ecef (light gray)         │
│  Stroke: #495057 (dark gray), 2px   │
│  Border: Dashed                     │
│                                     │
│  ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐               │
│  ┊   Terminal       ┊               │
│  ┊   V_{T+1} = 0    ┊               │
│  └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘               │
└─────────────────────────────────────┘
```

### Hydro Plant Icon

```
┌─────────────────────────────────────┐
│  HYDRO PLANT                        │
│                                     │
│  Shape: Custom (reservoir + dam)    │
│  Fill: #b2f2bb (light green)        │
│  Stroke: #2f9e44 (green), 2px       │
│                                     │
│       ~~~~~~~~~~~                   │
│      /           \   ← water level  │
│     |  Reservoir  |                 │
│     |     v_h     |                 │
│     |_____________|                 │
│          |||        ← turbine       │
│          ▼▼▼                        │
└─────────────────────────────────────┘
```

### Thermal Plant Icon

```
┌─────────────────────────────────────┐
│  THERMAL PLANT                      │
│                                     │
│  Shape: Rectangle with chimney      │
│  Fill: #ffec99 (light yellow)       │
│  Stroke: #f08c00 (orange), 2px      │
│                                     │
│       🔥  (or flame sketch)         │
│     ┌─────┐                         │
│     │ g_j │                         │
│     │ MW  │                         │
│     └─────┘                         │
└─────────────────────────────────────┘
```

### MPI Process Box

```
┌─────────────────────────────────────┐
│  MPI PROCESS                        │
│                                     │
│  Shape: Rectangle with header       │
│  Fill: #d0bfff (light purple)       │
│  Stroke: #7048e8 (purple), 2px      │
│  Header: Darker purple band         │
│                                     │
│  ┌──────────────────────────┐       │
│  │ Rank 0 (Master)          │ ← hdr │
│  ├──────────────────────────┤       │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ │       │
│  │  │ T0  │ │ T1  │ │ T2  │ │ ← OMP │
│  │  └─────┘ └─────┘ └─────┘ │       │
│  │     OpenMP Threads       │       │
│  └──────────────────────────┘       │
└─────────────────────────────────────┘
```

### Cut (Hyperplane)

```
┌─────────────────────────────────────┐
│  BENDERS CUT                        │
│                                     │
│  Shape: Angled line with equation   │
│  Stroke: #e03131 (red), 2px         │
│  Style: Solid                       │
│                                     │
│           θ ≥ α^k + β^k · x         │
│          ╱─────────────────         │
│         ╱                           │
│        ╱  ← cut k                   │
│       ╱                             │
└─────────────────────────────────────┘
```

### Scenario/Realization Node

```
┌─────────────────────────────────────┐
│  SCENARIO NODE                      │
│                                     │
│  Shape: Circle or ellipse           │
│  Fill: #99e9f2 (light cyan)         │
│  Stroke: #0c8599 (teal), 2px        │
│                                     │
│       ┌───────┐                     │
│      (   ω_1   )  with p(ω_1) label │
│       └───────┘                     │
└─────────────────────────────────────┘
```

---

## Arrow Styles

### Forward Transition

```
Source ────────────────────▶ Target
        solid, 2px, blue stroke
```

### Cycle/Feedback Loop

```
Source ◁╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ Target
        dashed, 2px, red stroke
        with label "β < 1"
```

### Probabilistic Branch

```
                    ┌──▶ ω_1 (p=0.3)
Source ─────────────┼──▶ ω_2 (p=0.5)
                    └──▶ ω_3 (p=0.2)
        
        Labels on each branch show probability
```

### Cut Addition

```
Subproblem ══════════════════▶ Cut Pool
            double line, bold
            represents "cut generated"
```

---

## Diagram Templates

### Template 1: Policy Graph (Finite Horizon)

**Filename**: `policy-graph-finite.excalidraw`

**Structure**:
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌┄┄┄┄┄┄┄┄┄┄┐       │
│   │ Stage 1│───▶│ Stage 2│───▶│  ...   │───▶┊ Terminal ┊       │
│   │  t=1   │    │  t=2   │    │        │    ┊ V_{T+1}=0┊       │
│   └────────┘    └────────┘    └────────┘    └┄┄┄┄┄┄┄┄┄┄┘       │
│       │              │             │                            │
│       │    ω_1       │    ω_1      │                            │
│       ├────▶         ├────▶        │                            │
│       │    ω_2       │    ω_2      │                            │
│       └────▶         └────▶        │                            │
│                                                                 │
│   ANNOTATIONS:                                                  │
│   • Blue nodes = SDDP stages                                    │
│   • Dashed terminal = V_{T+1}(x) = 0                           │
│   • Branches = stochastic realizations                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Elements**:
1. Horizontal stage progression (left to right)
2. Terminal node with dashed border
3. Scenario branches below each stage
4. Probability labels on branches
5. Annotation box explaining notation

### Template 2: Policy Graph (Infinite/Cyclic Horizon)

**Filename**: `policy-graph-cyclic.excalidraw`

**Structure**:
```
┌─────────────────────────────────────────────────────────────────┐
│              ╭╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╮          │
│              ╎            cycle with β < 1           ╎          │
│              ▼                                       ╎          │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐          │
│   │ Stage 1│───▶│ Stage 2│───▶│  ...   │───▶│Stage 12│──────╯   │
│   │  Jan   │    │  Feb   │    │        │    │  Dec   │          │
│   └────────┘    └────────┘    └────────┘    └────────┘          │
│                                                 ▲               │
│                                                 │               │
│                                          RED stroke             │
│                                          dashed line            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Elements**:
1. Same horizontal progression
2. RED dashed arrow from last stage back to first
3. Discount factor β < 1 labeled on cycle edge
4. Stage 12 has red stroke (cycle endpoint)
5. Stage 1 highlighted as cycle entry point

### Template 3: SDDP Iteration Structure

**Filename**: `sddp-iteration.excalidraw`

**Structure**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    SDDP ITERATION k                             │
│                                                                 │
│   ╔═══════════════════════════════════════════════════════════╗ │
│   ║                    FORWARD PASS                           ║ │
│   ║  Sample M scenarios, simulate policy                      ║ │
│   ║                                                           ║ │
│   ║  ○───▶○───▶○───▶○   scenario 1                           ║ │
│   ║  ○───▶○───▶○───▶○   scenario 2                           ║ │
│   ║  ...                                                      ║ │
│   ║  ○───▶○───▶○───▶○   scenario M                           ║ │
│   ║                        ▼                                  ║ │
│   ║              compute upper bound z̄                       ║ │
│   ╚═══════════════════════════════════════════════════════════╝ │
│                            │                                    │
│                            ▼                                    │
│   ╔═══════════════════════════════════════════════════════════╗ │
│   ║                   BACKWARD PASS                           ║ │
│   ║  Generate cuts at visited states                          ║ │
│   ║                                                           ║ │
│   ║  ○◀───○◀───○◀───○   solve subproblems                    ║ │
│   ║     ↑    ↑    ↑                                          ║ │
│   ║    cut  cut  cut    add to V_t approximation             ║ │
│   ║                        ▼                                  ║ │
│   ║              compute lower bound z̲                       ║ │
│   ╚═══════════════════════════════════════════════════════════╝ │
│                            │                                    │
│                            ▼                                    │
│               gap = (z̄ - z̲) / |z̄| < ε ?                       │
│                     /            \                              │
│                   YES            NO                             │
│                    │              │                             │
│                    ▼              ╰───────────────▶ iteration   │
│                  STOP                              k+1          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Template 4: Value Function Approximation

**Filename**: `value-function-cuts.excalidraw`

**Structure**:
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  V_t(x)                                                         │
│    ▲                                                            │
│    │         ╭──────────────╮  ← true value function V_t(x)     │
│    │        ╱                ╲     (convex, unknown)            │
│    │       ╱                  ╲                                 │
│    │      ╱     ╲               ╲                               │
│    │     ╱       ╲               ╲                              │
│    │    ╱    ●────╲───────────────╲  ← cut k=3                  │
│    │   ╱    ╱      ╲               ╲                            │
│    │  ╱    ╱        ●───────────────╲  ← cut k=2                │
│    │ ╱    ╱        ╱                 ╲                          │
│    │╱    ╱    ●───╱───────────────────╲  ← cut k=1              │
│    │    ╱    ╱   ╱                     ╲                        │
│    │   ╱    ╱   ╱                       ╲                       │
│    │  ╱    ╱   ╱                                                │
│    │ ╱    ╱───╱─────────────────────────  V̲_t(x) = max cuts    │
│    │╱    ╱   ╱                                                  │
│    ├────┼───┼───────────────────────────────────────────▶ x     │
│    │    x̂₁  x̂₂  x̂₃                                             │
│    │                                                            │
│   GAP = V_t(x) - V̲_t(x) → 0 as k → ∞                           │
│                                                                 │
│   LEGEND:                                                       │
│   ● = trial point where cut was generated                       │
│   ─── = cut hyperplane θ ≥ α^k + β^k · x                        │
│   ▬▬▬ = lower approximation (max of all cuts)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Elements**:
1. True value function as smooth convex curve (green, thick)
2. Cut hyperplanes as straight lines (red, thin)
3. Lower approximation as bold piecewise linear (blue, thick)
4. Trial points marked with filled circles
5. Gap annotation showing convergence
6. Legend explaining symbols

### Template 5: MPI+OpenMP Hybrid Architecture

**Filename**: `mpi-openmp-architecture.excalidraw`

**Structure**:
```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPUTE NODE 0                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ MPI Rank 0                                                │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │           Shared Memory (OpenMP)                    │  │  │
│  │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐               │  │  │
│  │  │  │ T0  │  │ T1  │  │ T2  │  │ T3  │   threads     │  │  │
│  │  │  │ ω₁  │  │ ω₂  │  │ ω₃  │  │ ω₄  │   solving     │  │  │
│  │  │  └─────┘  └─────┘  └─────┘  └─────┘   scenarios   │  │  │
│  │  │              │                                      │  │  │
│  │  │              ▼                                      │  │  │
│  │  │        Local Cut Pool                              │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              │
              │ MPI_Allgatherv (cuts)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COMPUTE NODE 1                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ MPI Rank 1                                                │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │           Shared Memory (OpenMP)                    │  │  │
│  │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐               │  │  │
│  │  │  │ T0  │  │ T1  │  │ T2  │  │ T3  │   threads     │  │  │
│  │  │  │ ω₅  │  │ ω₆  │  │ ω₇  │  │ ω₈  │   solving     │  │  │
│  │  │  └─────┘  └─────┘  └─────┘  └─────┘   scenarios   │  │  │
│  │  │              │                                      │  │  │
│  │  │              ▼                                      │  │  │
│  │  │        Local Cut Pool                              │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Annotation Conventions

### Callout Boxes

Use rounded rectangles with light yellow fill (`#fff3bf`) for explanatory callouts:

```
    ┌─────────────────────────────────────┐
    │ 💡 Key Insight                       │
    │                                      │
    │ The cut coefficients β_t are         │
    │ derived from dual multipliers        │
    │ of the state transition constraint   │
    └─────────────────────────────────────┘
```

### Mathematical Equations

Place key equations in boxes with white background and blue border:

```
    ╔══════════════════════════════════════╗
    ║  V_t(x) = E[ min c'x + V_{t+1}(x) ] ║
    ╚══════════════════════════════════════╝
```

### Step Numbers

Use circled numbers for sequential steps:

```
    ① Sample scenarios
    ② Simulate forward pass
    ③ Generate cuts backward
    ④ Update bounds
```

---

## Export Settings

### For Documentation (PNG)

- **Scale**: 2x (for retina displays)
- **Background**: Transparent or white (`#ffffff`)
- **Padding**: 20px
- **Format**: PNG

### For Web (SVG)

- **Scale**: 1x
- **Background**: Transparent
- **Embed fonts**: Yes
- **Format**: SVG

### For Print (PDF embedding)

- **Scale**: 3x
- **Background**: White
- **Format**: PNG (LaTeX compatible)

---

## Migration Checklist

When converting Mermaid/ASCII diagrams to Excalidraw:

### Pre-Migration

- [ ] Identify diagram type and purpose
- [ ] List all elements (nodes, edges, labels)
- [ ] Note any mathematical notation required
- [ ] Choose appropriate template

### During Creation

- [ ] Use correct color palette
- [ ] Apply consistent typography
- [ ] Ensure arrows point in correct direction
- [ ] Add probability labels where applicable
- [ ] Include stage/node labels
- [ ] Add legend if notation is complex

### Post-Creation

- [ ] Verify mathematical notation accuracy
- [ ] Check color contrast for accessibility
- [ ] Export at appropriate resolution
- [ ] Test in documentation context
- [ ] Update references in Markdown files

---

## File Naming Convention

```
<category>-<specific-name>-<version>.excalidraw

Examples:
  policy-graph-finite-v1.excalidraw
  policy-graph-cyclic-v2.excalidraw
  algorithm-sddp-iteration-v1.excalidraw
  architecture-mpi-openmp-v1.excalidraw
  concept-value-function-cuts-v1.excalidraw
```

---

## Reusable Component Library

Create and maintain a component library at `docs/diagrams/excalidraw/components/`:

| Component | Filename | Description |
|-----------|----------|-------------|
| Stage Node | `component-stage-node.excalidraw` | Standard SDDP stage |
| Terminal Node | `component-terminal-node.excalidraw` | V_{T+1} = 0 node |
| Hydro Icon | `component-hydro-plant.excalidraw` | Reservoir + turbine |
| Thermal Icon | `component-thermal-plant.excalidraw` | Generator with flame |
| MPI Rank Box | `component-mpi-rank.excalidraw` | Process container |
| OpenMP Thread | `component-omp-thread.excalidraw` | Thread box |
| Cut Line | `component-cut.excalidraw` | Benders cut with equation |
| Scenario Branch | `component-scenario.excalidraw` | ω node with probability |

---

## Quick Reference Card

### Colors (Copy-Paste)

```
Stages:       Fill=#a5d8ff  Stroke=#1971c2
Hydro:        Fill=#b2f2bb  Stroke=#2f9e44
Thermal:      Fill=#ffec99  Stroke=#f08c00
Cycles:       Fill=#ffc9c9  Stroke=#e03131
Parallel:     Fill=#d0bfff  Stroke=#7048e8
Data:         Fill=#99e9f2  Stroke=#0c8599
Terminal:     Fill=#e9ecef  Stroke=#495057
Highlight:    Fill=#fff3bf  Stroke=#f59f00
```

### Greek Letters (Copy-Paste)

```
α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω
Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
```

### Subscript Numbers (Copy-Paste)

```
₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₊ ₋ ₌ ₍ ₎
```

### Superscript Numbers (Copy-Paste)

```
⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁺ ⁻ ⁼ ⁽ ⁾
```

### Arrows (Copy-Paste)

```
→ ← ↑ ↓ ↔ ↕ ⇒ ⇐ ⇑ ⇓ ⇔ ⇕
▶ ◀ ▲ ▼ ► ◄ △ ▽ ▷ ◁
```

### Mathematical Symbols (Copy-Paste)

```
∈ ∉ ⊂ ⊃ ⊆ ⊇ ∩ ∪ ∅ ∀ ∃ ∄
≤ ≥ ≠ ≈ ≡ ∝ ± ∓ × ÷ √ ∛
Σ Π ∫ ∂ ∇ ∞ ℵ ℓ
𝔼 ℙ ℝ ℤ ℕ ℂ
```
