# POWE.RS Diagram Migration Master Plan

> **Document Purpose**: Production-grade plan for migrating documentation diagrams from ASCII/Mermaid to hand-drawn Excalidraw style.
>
> **Contributors**: SDDP Specialist, HPC Parallel Computing Specialist, Data Model Format Specialist
>
> **Last Updated**: 2026-02-05

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Target State](#3-target-state)
4. [Priority Diagram Inventory](#4-priority-diagram-inventory)
5. [Unified Style Guide](#5-unified-style-guide)
6. [Directory Structure](#6-directory-structure)
7. [Tooling and Automation](#7-tooling-and-automation)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Quality Assurance](#9-quality-assurance)

---

## 1. Executive Summary

### 1.1 Migration Goals

1. **Clarity over formality**: Hand-drawn style improves reader engagement and concept focus
2. **Manual refinement**: Full control to enhance diagram clarity iteratively
3. **Version control**: `.excalidraw` (JSON) source files tracked in Git
4. **Abandon PDF requirement**: Focus on PNG/SVG export for web/markdown rendering
5. **Unified visual language**: Consistent style across SDDP, HPC, and data model diagrams

### 1.2 Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary Tool** | Excalidraw (web + VS Code extension) | Hand-drawn style, open source, JSON format |
| **Source Format** | `.excalidraw` files | Version-controlled, editable |
| **Export Formats** | PNG (2x scale) + SVG | Web embedding, retina displays |
| **Mermaid Migration** | `@excalidraw/mermaid-to-excalidraw` | Automated conversion, then manual refinement |
| **Diagram Location** | `docs/diagrams/` directory tree | Organized by domain and type |

### 1.3 Scope

| Document | Diagram Count | Priority Diagrams |
|----------|---------------|-------------------|
| `MATHEMATICAL_FORMULATIONS.md` | ~8 diagrams | 5 critical (SDDP algorithm) |
| `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` | ~15 diagrams | 7 critical (HPC parallelism) |
| `DATA_MODEL_SPECIFICATION.md` | ~10 diagrams | 6 critical (data structures) |
| **Total** | ~33 diagrams | **18 critical (Tier 1)** |

---

## 2. Current State Analysis

### 2.1 Existing Diagram Types

| Type | Count | Example | Migration Complexity |
|------|-------|---------|---------------------|
| **ASCII Box Diagrams** | ~20 | Node layout, execution flow | Medium (manual recreation) |
| **Mermaid Flowcharts** | ~8 | Policy graphs, system topology | Low (automated + refinement) |
| **Mermaid Sequence** | ~2 | Synchronization patterns | Low (automated) |
| **Text Tables** | ~5 | LP sizing, performance targets | Medium (visual tables) |
| **None (text only)** | ~5 | Complex concepts needing diagrams | High (design from scratch) |

### 2.2 Pain Points with Current Approach

1. **Mermaid limitations**: Limited customization, auto-layout often suboptimal
2. **ASCII fragility**: Alignment breaks on different fonts/renderers
3. **PDF rendering**: Mermaid requires external tooling, inconsistent results
4. **No visual distinction**: All diagrams look the same regardless of domain
5. **Hard to refine**: Both ASCII and Mermaid resist incremental improvements

---

## 3. Target State

### 3.1 Visual Identity

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         POWE.RS Diagram Visual Identity                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Style: Hand-drawn, cartoonish (Excalidraw "architect" roughness)                │
│  Font: Virgil (Excalidraw default) for labels, annotations                       │
│  Colors: Domain-specific palettes (see Section 5)                                │
│                                                                                  │
│  Characteristics:                                                                │
│  • Wobbly, imperfect lines (warmth, approachability)                            │
│  • Rounded corners on containers                                                 │
│  • Hand-drawn arrowheads                                                         │
│  • Soft color fills with clear borders                                           │
│  • Generous whitespace                                                           │
│  • Clear legends and annotations                                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Diagram Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **Algorithm Flow** | SDDP iteration, forward/backward passes | Flowcharts with decision points |
| **Architecture** | System topology, component relationships | Network diagrams with entity cards |
| **Data Structure** | LP matrix, cut storage, memory layout | Grid/block diagrams |
| **Parallel Execution** | MPI/OpenMP hierarchy, work distribution | Nested containers, timelines |
| **Configuration** | File relationships, option effects | Dependency graphs |
| **Pipeline** | Data flow, transformation stages | Swim lanes with process nodes |

---

## 4. Priority Diagram Inventory

### 4.1 Tier 1: Critical (Must-Have) - 18 Diagrams

#### SDDP Algorithm Domain (5 diagrams)

| ID | Name | Document | Section | Description |
|----|------|----------|---------|-------------|
| **SDDP-01** | Policy Graph (Finite Horizon) | MATH | §2.3.1 | Stage sequence with terminal condition |
| **SDDP-02** | Policy Graph (Cyclic/Infinite) | MATH | §2.3.2 | Cyclic graph with discount factor |
| **SDDP-03** | SDDP Iteration Structure | MATH | §2.2 | Forward/backward pass flow |
| **SDDP-04** | Value Function Approximation | MATH | §2.1 | Cuts as outer approximation |
| **SDDP-05** | System Element Overview | MATH | §3.1 | Buses, hydros, thermals, lines |

#### HPC Parallel Computing Domain (7 diagrams)

| ID | Name | Document | Section | Description |
|----|------|----------|---------|-------------|
| **HPC-01** | Hybrid Parallelism Architecture | ARCH | §20.1 | 8 NUMA × 24 threads layout |
| **HPC-02** | Forward Pass Work Distribution | ARCH | §21.1 | MPI static + OpenMP dynamic |
| **HPC-03** | Backward Pass Work Distribution | ARCH | §21.2 | Scenario-based, warm-start |
| **HPC-04** | SDDP Iteration Synchronization | ARCH | §22.1 | Sync points, barriers, data volumes |
| **HPC-05** | Shared Memory Window Architecture | ARCH | §23.4 | MPI windows, NUMA interleaving |
| **HPC-06** | Communication Volume Analysis | ARCH | §23.5 | Bandwidth requirements table |
| **HPC-07** | OpenMP FFI Architecture | ARCH | §20.4 | Rust → C → libgomp stack |

#### Data Model Domain (6 diagrams)

| ID | Name | Document | Section | Description |
|----|------|----------|---------|-------------|
| **DM-01** | Case Directory Structure | DATA | §3.1 | Complete file tree with badges |
| **DM-02** | LP Variable/Constraint Sizing | DATA | §2.2 | Stacked bars with formulas |
| **DM-03** | System Entity Relationships | MATH | §3.1 | Network diagram with entity cards |
| **DM-04** | Penalty Resolution Cascade | DATA | §1.3 | Priority chain flow |
| **DM-05** | State Variable Composition | DATA | §2.1.1 | Storage + AR lags breakdown |
| **DM-06** | File Format Decision Tree | DATA | §1.1 | JSON vs Parquet vs FlatBuffers |

### 4.2 Tier 2: Important - 10 Diagrams

| ID | Name | Domain | Description |
|----|------|--------|-------------|
| **SDDP-06** | Cut Generation Mechanics | SDDP | Dual extraction, coefficient computation |
| **SDDP-07** | Scenario Tree Branching | SDDP | 20-branch structure at each stage |
| **HPC-08** | Execution Phase Diagram | HPC | All phases with responsibilities |
| **HPC-09** | Hierarchical vs Flat Aggregation | HPC | Tree-based cut gathering |
| **HPC-10** | Scaling Efficiency Curve | HPC | Strong scaling plot |
| **DM-07** | Input Loading Pipeline | Data | Parse → Validate → Canonicalize |
| **DM-08** | MPI Broadcast Patterns | Data | Full replication vs shared window |
| **DM-09** | Cut Storage Memory Layout | Data | Preallocation with validity bitmap |
| **DM-10** | Output Streaming Pipeline | Data | Convergence, cuts, states flow |
| **DM-11** | JSON Schema Dependencies | Data | File relationship graph |

### 4.3 Tier 3: Enhancement - 5 Diagrams

| ID | Name | Domain |
|----|------|--------|
| **SDDP-08** | AR Lag State Expansion | SDDP |
| **HPC-11** | Thread Synchronization Pattern | HPC |
| **HPC-12** | Async Communication Overlap | HPC |
| **DM-12** | Config Option → Effect Mapping | Data |
| **DM-13** | Parquet Schema Relationships | Data |

---

## 5. Unified Style Guide

### 5.1 Color Palette

#### Domain Colors

| Domain | Primary | Secondary | Border | Usage |
|--------|---------|-----------|--------|-------|
| **SDDP Algorithm** | `#e1f5ff` | `#dbe4ff` | `#0066cc` | Stages, iterations, cuts |
| **HPC Parallel** | `#d4edda` | `#c3e6cb` | `#28a745` | Ranks, threads, NUMA |
| **Data Model** | `#fff9e6` | `#ffe8cc` | `#ffaa00` | Files, schemas, entities |
| **Stochastic** | `#e2d5f5` | `#d4c5f9` | `#6f42c1` | Scenarios, uncertainty |
| **Warning/Critical** | `#f8d7da` | `#f5c6cb` | `#dc3545` | Bottlenecks, errors |

#### Entity Colors (System Topology)

| Entity | Fill | Border | Icon |
|--------|------|--------|------|
| **Bus (demand)** | `#e1f5ff` | `#0066cc` | Rectangle |
| **Hydro plant** | `#d4edda` | `#28a745` | Hexagon (water drop) |
| **Thermal plant** | `#fff3cd` | `#ffc107` | Circle (flame) |
| **Transmission line** | `#e9ecef` | `#495057` | Diamond or line |
| **Contract** | `#f8d7da` | `#dc3545` | Triangle |
| **Pumping station** | `#e2d5f5` | `#6f42c1` | Pentagon |

#### HPC Hierarchy Colors

| Level | Fill | Border | Stroke Width |
|-------|------|--------|--------------|
| **Cluster** | `#f5f5f5` | `#868e96` | 3px dashed |
| **Node** | `#dbe4ff` | `#1971c2` | 3px solid |
| **NUMA Domain** | `#d3f9d8` | `#2f9e44` | 2px solid |
| **MPI Rank** | `#ffe8cc` | `#fd7e14` | 2px solid |
| **Thread** | `#fcc419` | `#f59f00` | 1px solid |

### 5.2 Typography

| Element | Font | Size | Style |
|---------|------|------|-------|
| **Diagram Title** | Virgil | 24px | Bold |
| **Section Headers** | Virgil | 18px | Bold |
| **Entity Labels** | Virgil | 16px | Regular |
| **Annotations** | Virgil | 14px | Regular |
| **Callouts/Notes** | Virgil | 12px | Italic |
| **Math Notation** | Virgil | 14px | Use Unicode: α, β, θ, Σ |

### 5.3 Shape Standards

| Shape | Usage | Stroke | Fill |
|-------|-------|--------|------|
| **Rounded Rectangle** | Containers, entities, processes | 2px | Solid light |
| **Rectangle** | Data structures, matrices | 2px | Hatched or solid |
| **Diamond** | Decision points, junctions | 2px | Light fill |
| **Circle** | Process nodes, transformations | 2px | Light fill |
| **Hexagon** | Hydro elements | 2px | Green fill |
| **Ellipse** | Terminal nodes | 2px | Gray fill |

### 5.4 Arrow Standards

| Arrow Type | Style | Usage |
|------------|-------|-------|
| **Solid** | Standard arrowhead, 2px | Normal flow, data transfer |
| **Dashed** | Dashed line, standard head | Optional flow, conditional |
| **Dotted** | Dotted line | Reference, weak coupling |
| **Bidirectional** | Arrows on both ends | Exchange, two-way comm |
| **Thick Red Dashed** | 3px, red, dashed | Cycle arrows (infinite horizon) |

### 5.5 Roughness Settings

| Diagram Type | Roughness | Seed | Rationale |
|--------------|-----------|------|-----------|
| **Algorithm/Math** | 1.0 | Fixed | Approachable, conceptual |
| **Architecture** | 0.8 | Fixed | Slightly cleaner for precision |
| **Data Structure** | 0.5 | Fixed | Cleaner for technical detail |

---

## 6. Directory Structure

```
docs/
├── diagrams/
│   ├── README.md                    # Quick reference, contribution guide
│   ├── STYLE_GUIDE.md               # Detailed style specifications
│   │
│   ├── excalidraw/                  # Source files (.excalidraw)
│   │   ├── sddp/                    # SDDP algorithm diagrams
│   │   │   ├── policy-graph-finite.excalidraw
│   │   │   ├── policy-graph-cyclic.excalidraw
│   │   │   ├── sddp-iteration.excalidraw
│   │   │   └── value-function-approximation.excalidraw
│   │   │
│   │   ├── hpc/                     # HPC parallel computing diagrams
│   │   │   ├── hybrid-parallelism.excalidraw
│   │   │   ├── forward-pass-distribution.excalidraw
│   │   │   ├── backward-pass-distribution.excalidraw
│   │   │   ├── synchronization-points.excalidraw
│   │   │   └── shared-memory-architecture.excalidraw
│   │   │
│   │   ├── data/                    # Data model diagrams
│   │   │   ├── directory-structure.excalidraw
│   │   │   ├── lp-sizing.excalidraw
│   │   │   ├── entity-relationships.excalidraw
│   │   │   └── file-format-decision.excalidraw
│   │   │
│   │   └── components/              # Reusable diagram components
│   │       ├── legends.excalidraw
│   │       ├── icons.excalidraw
│   │       └── color-swatches.excalidraw
│   │
│   ├── exports/                     # Generated files (gitignored optional)
│   │   ├── png/                     # PNG exports (2x scale)
│   │   │   ├── sddp/
│   │   │   ├── hpc/
│   │   │   └── data/
│   │   │
│   │   └── svg/                     # SVG exports
│   │       ├── sddp/
│   │       ├── hpc/
│   │       └── data/
│   │
│   └── legacy/                      # Archive of old ASCII/Mermaid (reference only)
│       └── mermaid-extracts/
│
├── DATA_MODEL_SPECIFICATION.md      # Updated to reference exports/png/data/*
├── MATHEMATICAL_FORMULATIONS.md     # Updated to reference exports/png/sddp/*
└── PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md  # Updated to reference exports/png/hpc/*
```

---

## 7. Tooling and Automation

### 7.1 Required Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **Excalidraw Web** | Interactive editing | https://excalidraw.com |
| **VS Code Extension** | IDE integration | `ext install excalidraw.excalidraw-editor` |
| **@excalidraw/mermaid-to-excalidraw** | Mermaid conversion | `npm install` |
| **excalidraw-cli** | CLI export | `npm install -g @excalidraw/cli` |

### 7.2 npm Dependencies

```json
{
  "devDependencies": {
    "@excalidraw/excalidraw": "^0.17.0",
    "@excalidraw/mermaid-to-excalidraw": "^0.3.0"
  },
  "scripts": {
    "diagrams:export": "./scripts/export-diagrams.sh",
    "diagrams:convert": "node scripts/convert-mermaid.mjs",
    "diagrams:list": "grep -rn 'mermaid' docs/*.md | head -50"
  }
}
```

### 7.3 Export Script

```bash
#!/bin/bash
# scripts/export-diagrams.sh

EXCALIDRAW_DIR="docs/diagrams/excalidraw"
EXPORT_DIR="docs/diagrams/exports"
SCALE=2

mkdir -p "$EXPORT_DIR/png" "$EXPORT_DIR/svg"

find "$EXCALIDRAW_DIR" -name "*.excalidraw" | while read -r file; do
    basename="${file##*/}"
    name="${basename%.excalidraw}"
    subdir=$(dirname "${file#$EXCALIDRAW_DIR/}")
    
    mkdir -p "$EXPORT_DIR/png/$subdir"
    mkdir -p "$EXPORT_DIR/svg/$subdir"
    
    excalidraw export "$file" \
        --output "$EXPORT_DIR/png/$subdir/$name.png" \
        --scale $SCALE
    
    excalidraw export "$file" \
        --output "$EXPORT_DIR/svg/$subdir/$name.svg"
done

echo "Exported all diagrams to $EXPORT_DIR"
```

### 7.4 Mermaid Conversion Script

```javascript
// scripts/convert-mermaid.mjs
import { parseMermaidToExcalidraw } from "@excalidraw/mermaid-to-excalidraw";
import { convertToExcalidrawElements } from "@excalidraw/excalidraw";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";

const DOCS = ["docs/MATHEMATICAL_FORMULATIONS.md", "docs/DATA_MODEL_SPECIFICATION.md"];
const OUTPUT_DIR = "docs/diagrams/excalidraw/converted";

mkdirSync(OUTPUT_DIR, { recursive: true });

for (const doc of DOCS) {
    const content = readFileSync(doc, "utf-8");
    const mermaidBlocks = content.match(/```mermaid\n([\s\S]*?)```/g) || [];
    
    for (let i = 0; i < mermaidBlocks.length; i++) {
        const mermaidCode = mermaidBlocks[i]
            .replace(/```mermaid\n/, "")
            .replace(/```$/, "");
        
        try {
            const { elements } = await parseMermaidToExcalidraw(mermaidCode);
            const excalidrawElements = convertToExcalidrawElements(elements);
            
            const output = {
                type: "excalidraw",
                version: 2,
                source: "mermaid-migration",
                elements: excalidrawElements,
                appState: { viewBackgroundColor: "#ffffff" }
            };
            
            const filename = `${doc.split("/").pop().replace(".md", "")}-diagram-${i + 1}.excalidraw`;
            writeFileSync(join(OUTPUT_DIR, filename), JSON.stringify(output, null, 2));
            console.log(`Converted: ${filename}`);
        } catch (e) {
            console.error(`Failed to convert diagram ${i + 1} from ${doc}:`, e.message);
        }
    }
}
```

### 7.5 Makefile Integration

```makefile
# Add to existing Makefile

.PHONY: diagrams diagrams-export diagrams-convert diagrams-clean

diagrams: diagrams-export  ## Export all Excalidraw diagrams to PNG/SVG

diagrams-export:
	@./scripts/export-diagrams.sh

diagrams-convert:  ## Convert existing Mermaid diagrams to Excalidraw
	@node scripts/convert-mermaid.mjs

diagrams-list:  ## List Mermaid code blocks in documentation
	@grep -rn '```mermaid' docs/*.md | head -50

diagrams-clean:  ## Remove exported diagram files
	@rm -rf docs/diagrams/exports/*
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Foundation (Week 1)

**Goal**: Establish infrastructure and create first exemplar diagrams

| Task | Owner | Deliverable |
|------|-------|-------------|
| Create directory structure | Dev | `docs/diagrams/` tree |
| Write STYLE_GUIDE.md | Dev | Detailed style reference |
| Install npm dependencies | Dev | Working conversion script |
| Create 2 template diagrams | Dev | `policy-graph-finite.excalidraw`, `hybrid-parallelism.excalidraw` |
| Validate export pipeline | Dev | PNG exports in correct directories |

**Exit Criteria**:
- Directory structure exists
- At least 2 diagrams exported successfully
- Style guide document complete

### 8.2 Phase 2: Tier 1 SDDP Diagrams (Week 2)

**Goal**: Complete 5 critical SDDP algorithm diagrams

| Diagram | Effort | Notes |
|---------|--------|-------|
| SDDP-01 Policy Graph (Finite) | 2h | Convert from Mermaid, refine |
| SDDP-02 Policy Graph (Cyclic) | 2h | Convert from Mermaid, add cycle arrow |
| SDDP-03 SDDP Iteration | 4h | New design from ASCII |
| SDDP-04 Value Function Approx | 4h | Conceptual, needs careful design |
| SDDP-05 System Elements | 3h | Convert Mermaid, add icons |

**Exit Criteria**: 5 diagrams complete, reviewed for technical accuracy

### 8.3 Phase 3: Tier 1 HPC Diagrams (Week 3)

**Goal**: Complete 7 critical HPC parallel computing diagrams

| Diagram | Effort | Notes |
|---------|--------|-------|
| HPC-01 Hybrid Parallelism | 4h | Detailed NUMA layout |
| HPC-02 Forward Pass Distribution | 2h | Static + dynamic scheduling |
| HPC-03 Backward Pass Distribution | 3h | Warm-start emphasis |
| HPC-04 Synchronization Points | 3h | Timeline format |
| HPC-05 Shared Memory Windows | 4h | NUMA interleaving detail |
| HPC-06 Communication Volume | 2h | Table-style visualization |
| HPC-07 OpenMP FFI | 2h | Three-layer stack |

**Exit Criteria**: 7 diagrams complete, validated by HPC knowledge

### 8.4 Phase 4: Tier 1 Data Model Diagrams (Week 4)

**Goal**: Complete 6 critical data model diagrams

| Diagram | Effort | Notes |
|---------|--------|-------|
| DM-01 Directory Structure | 3h | Complete file tree with icons |
| DM-02 LP Sizing | 3h | Stacked bars with formulas |
| DM-03 Entity Relationships | 4h | Network diagram with cards |
| DM-04 Penalty Resolution | 2h | Priority chain |
| DM-05 State Variables | 2h | Composition breakdown |
| DM-06 File Format Decision | 2h | Decision tree |

**Exit Criteria**: 6 diagrams complete, consistent with style guide

### 8.5 Phase 5: Documentation Update (Week 5)

**Goal**: Replace ASCII/Mermaid with PNG references

| Task | Effort | Notes |
|------|--------|-------|
| Update MATHEMATICAL_FORMULATIONS.md | 2h | Replace 5 diagrams |
| Update PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md | 3h | Replace 7 diagrams |
| Update DATA_MODEL_SPECIFICATION.md | 2h | Replace 6 diagrams |
| Archive legacy diagrams | 1h | Move to `legacy/` |
| Update any cross-references | 1h | Fix broken links |

**Exit Criteria**: All Tier 1 diagrams integrated, no broken images

### 8.6 Phase 6: Tier 2 Diagrams (Weeks 6-7)

**Goal**: Complete 10 additional important diagrams

*Parallel work on all domains*

### 8.7 Phase 7: Polish and CI Integration (Week 8)

**Goal**: Production-ready documentation pipeline

| Task | Effort | Notes |
|------|--------|-------|
| Add CI diagram export step | 2h | GitHub Actions workflow |
| Add diagram validation | 1h | Check for missing exports |
| Create component library | 2h | Reusable icons, legends |
| Documentation review | 2h | Final pass for consistency |

---

## 9. Quality Assurance

### 9.1 Review Checklist (Per Diagram)

#### Technical Accuracy

- [ ] All labels use correct terminology from documentation
- [ ] Numerical values match specifications (cores, memory, timing)
- [ ] Relationships correctly represent documented behavior
- [ ] No misleading simplifications

#### Visual Standards

- [ ] Colors match style guide palette
- [ ] Font sizes follow typography standards
- [ ] Arrow styles are consistent
- [ ] Legend explains all symbols
- [ ] Adequate whitespace

#### Accessibility

- [ ] Text is legible at 50% zoom
- [ ] Color is not the only differentiator (use patterns/shapes)
- [ ] High contrast between text and background
- [ ] Markdown includes alt-text for image embed

#### Export Quality

- [ ] PNG exports at 2x scale are crisp
- [ ] SVG exports render correctly in browser
- [ ] File sizes are reasonable (<500KB per diagram)
- [ ] Filenames follow naming convention

### 9.2 Domain-Specific Validation

#### SDDP Diagrams

- [ ] Stage numbering consistent (1-indexed)
- [ ] Value function notation matches (V_t, θ, α, β)
- [ ] Cut representation shows intercept + coefficients
- [ ] Forward/backward pass directions are clear

#### HPC Diagrams

- [ ] Core counts accurate (192 = 8 NUMA × 24 threads)
- [ ] Memory sizes correct (96 GB per NUMA)
- [ ] Synchronization points correctly placed
- [ ] Communication volumes documented

#### Data Model Diagrams

- [ ] File format badges accurate
- [ ] Required vs optional clearly distinguished
- [ ] Relationships show correct cardinality
- [ ] Sizing formulas match documentation

### 9.3 Regression Testing

After each diagram update:

1. Re-export PNG and SVG
2. Visual diff against previous version
3. Verify markdown references still work
4. Check file sizes haven't bloated

---

## Appendix A: Quick Reference

### A.1 Excalidraw Shortcuts

| Action | Shortcut |
|--------|----------|
| Rectangle | `R` |
| Ellipse | `O` |
| Arrow | `A` |
| Line | `L` |
| Text | `T` |
| Select | `V` |
| Pan | `Space + drag` |
| Duplicate | `Ctrl+D` |
| Group | `Ctrl+G` |
| Export | `Ctrl+Shift+E` |

### A.2 Unicode Math Symbols

```
Greek: α β γ δ ε θ λ μ π ρ σ τ φ ω Σ Ω
Subscripts: ₀₁₂₃₄₅₆₇₈₉ ᵢⱼₖₜₙₘ
Superscripts: ⁰¹²³⁴⁵⁶⁷⁸⁹ ⁺⁻ⁿ
Operators: × ÷ ± ∓ ∑ ∏ ∫ ∂ ∇ √
Relations: ≤ ≥ ≠ ≈ ≡ ∈ ⊂ ⊆
Arrows: → ← ↔ ⇒ ⇐ ⇔ ↦
Sets: ∅ ∩ ∪ ∀ ∃
Other: ∞ ∧ ∨ ¬ ⊤ ⊥
```

### A.3 Common Diagram Dimensions

| Diagram Type | Recommended Size | Aspect Ratio |
|--------------|------------------|--------------|
| Policy Graph | 1200 × 400 px | 3:1 |
| Architecture | 1600 × 900 px | 16:9 |
| Flow Diagram | 1200 × 800 px | 3:2 |
| Timeline | 1600 × 400 px | 4:1 |
| Entity Card | 200 × 150 px | 4:3 |

---

## Appendix B: Migration Tracking

### B.1 Diagram Status

| ID | Name | Status | Assignee | PR |
|----|------|--------|----------|-----|
| SDDP-01 | Policy Graph (Finite) | 🔴 Not Started | - | - |
| SDDP-02 | Policy Graph (Cyclic) | 🔴 Not Started | - | - |
| SDDP-03 | SDDP Iteration | 🔴 Not Started | - | - |
| SDDP-04 | Value Function Approx | 🔴 Not Started | - | - |
| SDDP-05 | System Elements | 🔴 Not Started | - | - |
| HPC-01 | Hybrid Parallelism | 🔴 Not Started | - | - |
| HPC-02 | Forward Pass | 🔴 Not Started | - | - |
| HPC-03 | Backward Pass | 🔴 Not Started | - | - |
| HPC-04 | Synchronization | 🔴 Not Started | - | - |
| HPC-05 | Shared Memory | 🔴 Not Started | - | - |
| HPC-06 | Communication Volume | 🔴 Not Started | - | - |
| HPC-07 | OpenMP FFI | 🔴 Not Started | - | - |
| DM-01 | Directory Structure | 🔴 Not Started | - | - |
| DM-02 | LP Sizing | 🔴 Not Started | - | - |
| DM-03 | Entity Relationships | 🔴 Not Started | - | - |
| DM-04 | Penalty Resolution | 🔴 Not Started | - | - |
| DM-05 | State Variables | 🔴 Not Started | - | - |
| DM-06 | File Format Decision | 🔴 Not Started | - | - |

Legend: 🔴 Not Started | 🟡 In Progress | 🟢 Complete | ✅ Reviewed

---

*Document generated by consolidating recommendations from SDDP, HPC, and Data Model specialists.*
