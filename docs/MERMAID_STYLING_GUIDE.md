# Mermaid Diagram Styling Guide

## Overview

This guide provides best practices for creating readable, professional Mermaid diagrams that convert well to PDF. The key is using proper styling, clear labeling, and appropriate diagram types.

## General Principles

### 1. Use Appropriate Diagram Types

- **Flowcharts (`graph`)**: For stage progressions, workflow flows
- **Sequence Diagrams (`sequenceDiagram`)**: For interactions between components
- **Gantt Charts (`gantt`)**: For timelines and schedules

### 2. Apply Consistent Styling

Always use the theme initialization for better PDF rendering:

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
```

### 3. Use Color Meaningfully

- **Blue tones** (`#e1f5ff`): Start nodes, initial states
- **Yellow/Orange** (`#fff4e1`): Important intermediate nodes
- **Red tones** (`#ffe1e1`): End nodes, cycle points
- **Gray** (`#f0f0f0`): Terminal/null states

## Flowchart Best Practices

### Basic Structure

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
graph LR
    A["Clear<br/>Label"]
    B["Another<br/>Node"]
    
    A --> B
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
```

### Stage Progression Example

**Good:**
```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
graph LR
    S1["Stage 1<br/><i>initialization</i>"]
    S2["Stage 2<br/><i>processing</i>"]
    S3["..."]
    ST["Stage T<br/><i>finalization</i>"]
    
    S1 --> S2
    S2 --> S3
    S3 --> ST
    
    style S1 fill:#e1f5ff
    style ST fill:#fff4e1
```

**Why it's good:**
- Clear labels with context
- Styling distinguishes start/end
- Italics for descriptive text
- Clean, horizontal layout

### Cyclic Graphs

**Good:**
```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
graph LR
    S1["Stage 1<br/><i>month 1</i>"]
    S2["Stage 2"]
    S3["..."]
    S12["Stage 12<br/><i>year end</i>"]
    
    S1 --> S2
    S2 --> S3
    S3 --> S12
    S12 -.->|"cycles with<br/>discount β"| S1
    
    style S1 fill:#e1f5ff
    style S12 fill:#ffe1e1
```

**Features:**
- Dashed line for cycle (`.->`)
- Label on cycle edge
- Different colors for start/cycle point
- Clear temporal context

## Timeline/Gantt Diagrams

**IMPORTANT:** Gantt charts have limitations in Mermaid:
- LaTeX math notation ($...$) doesn't render (shows as literal text)
- Section labels can overlap with content
- Limited styling options

**Better alternative:** Use hierarchical flowcharts with subgraphs for complex temporal relationships.

### Template 4: Hierarchical Temporal Decomposition

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'14px', 'fontFamily':'Arial'}}}%%
graph TB
    subgraph STAGES["<b>SDDP Stages</b> (Benders Cuts)"]
        direction LR
        ST0["<b>Stage 0</b><br/><i>4 weeks</i><br/>Cut boundary"]
        ST1["<b>Stage 1</b><br/><i>1 month</i><br/>Cut boundary"]
        ST0 -.->|"cut"| ST1
    end
    
    subgraph PERIODS["<b>Decision Periods</b> (Physics Resolution)"]
        direction LR
        W1["Week 1"]
        W2["Week 2"]
        W3["Week 3"]
        W4["Week 4"]
        M1["Month 1"]
        W1 --> W2 --> W3 --> W4 --> M1
    end
    
    subgraph STOCH["<b>Stochastic Realizations</b> (Uncertainty)"]
        direction LR
        O1["Inflow ω₁<br/><i>weeks 1-2</i>"]
        O2["Inflow ω₂<br/><i>weeks 3-4</i>"]
        O3["Inflow ω₃<br/><i>month 1</i>"]
        O1 -.-> O2 -.-> O3
    end
    
    STAGES --> PERIODS
    PERIODS --> STOCH
    
    style ST0 fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
    style ST1 fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
    style W1 fill:#fff4e1,stroke:#ffaa00,stroke-width:2px
    style W2 fill:#fff4e1,stroke:#ffaa00,stroke-width:2px
    style W3 fill:#fff4e1,stroke:#ffaa00,stroke-width:2px
    style W4 fill:#fff4e1,stroke:#ffaa00,stroke-width:2px
    style M1 fill:#fff4e1,stroke:#ffaa00,stroke-width:2px
    style O1 fill:#ffe1e1,stroke:#cc0000,stroke-width:2px
    style O2 fill:#ffe1e1,stroke:#cc0000,stroke-width:2px
    style O3 fill:#ffe1e1,stroke:#cc0000,stroke-width:2px
    style STAGES fill:#f9f9f9,stroke:#333,stroke-width:2px
    style PERIODS fill:#f9f9f9,stroke:#333,stroke-width:2px
    style STOCH fill:#f9f9f9,stroke:#333,stroke-width:2px
```

**Key features:**
- Three logical layers (subgraphs) with clear labels
- Unicode Greek letters (ω₁, ω₂) render correctly (unlike $\omega_1$ in Gantt)
- Color coding: blue (stages/cuts), yellow (decisions), red (uncertainty)
- Direction control within subgraphs (`direction LR`)
- Subgraph styling for visual grouping
- Connections between layers show hierarchy

**When to use this:**
- Complex multi-level temporal relationships
- Need to show 3+ different time granularities
- Need mathematical symbols (use Unicode, not LaTeX)
- Gantt chart sections would overlap

## Common Pitfalls

### ❌ Bad: Too Compact

```mermaid
graph LR
    A-->B-->C-->D
```

**Problems:**
- No labels
- No styling
- No context
- Hard to distinguish nodes

### ❌ Bad: ASCII Art in Mermaid

```mermaid
graph TD
    A["+-----+"]
    B["|Node |"]
    C["+-----+"]
```

**Problems:**
- Defeats the purpose of Mermaid
- Renders poorly
- Not semantic

### ❌ Bad: Over-styled

```mermaid
graph LR
    A
    B
    style A fill:#ff0000,stroke:#00ff00,stroke-width:4px,color:#fff,stroke-dasharray: 5 5
    style B fill:#00ff00,stroke:#ff0000,stroke-width:8px,color:#000,font-size:24px
```

**Problems:**
- Visual noise
- Inconsistent
- Distracting colors

## Mermaid vs ASCII Art

### When to Use Mermaid

✅ **Use Mermaid for:**
- State diagrams
- Flow charts
- Sequence diagrams
- Timeline/Gantt charts
- Any diagram with >3 nodes
- Diagrams needing clear visual hierarchy

### When to Use Text/Lists

✅ **Use formatted text for:**
- Simple algorithms (converted to LaTeX bullets)
- Short lists
- Single linear flows
- Mathematical formulas (use `$$...$$`)

## Complete Example: Converting ASCII to Mermaid

### Before (ASCII)

```
     ┌─────────────────────────────────┐
     │                                 │
     ▼                                 │
Stage 1 ──► Stage 2 ──► ... ──► Stage 12 ─┘
  │                                 │
  └─ year 1, month 1              └─ cycles back with discount
```

###  After (Mermaid)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
graph LR
    S1["Stage 1<br/><i>year 1, month 1</i>"]
    S2["Stage 2"]
    S3["..."]
    S12["Stage 12<br/><i>cycles back</i>"]
    
    S1 --> S2
    S2 --> S3
    S3 --> S12
    S12 -.->|"with discount β"| S1
    
    style S1 fill:#e1f5ff
    style S12 fill:#ffe1e1
```

**Improvements:**
- Professional rendering
- Clear labels
- Semantic structure
- PDF-friendly
- Color-coded nodes
- Descriptive edge label

## Style Templates

### Template 1: Linear Progress (SDDP Finite Horizon)

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'16px', 'fontFamily':'Arial'}}}%%
graph LR
    S1(["<b>Stage 1</b><br/><i>initial state</i><br/>t = 1"])
    S2["<b>Stage 2</b><br/><i>forward transitions</i><br/>t = 2"]
    S3["<b>Stage 3</b><br/>.<br/>.<br/>."]
    ST["<b>Stage T</b><br/><i>final decisions</i><br/>t = T"]
    Term(["<b>Terminal</b><br/>V<sub>T+1</sub> = 0<br/><i>no future cost</i>"])
    
    S1 -->|"p = 1"| S2
    S2 -->|"deterministic"| S3
    S3 -->|"acyclic"| ST
    ST -->|"terminate"| Term
    
    style S1 fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
    style S2 fill:#fff9e6,stroke:#ffaa00,stroke-width:2px
    style S3 fill:#fff9e6,stroke:#ffaa00,stroke-width:2px
    style ST fill:#fff4e1,stroke:#ff8800,stroke-width:3px
    style Term fill:#f0f0f0,stroke:#666,stroke-width:2px,stroke-dasharray: 5 5
```

**Key features:**
- Stadium shape `([...])` for start/end nodes
- Bold labels with `<b>...</b>`
- Multiple lines of context
- Edge labels for transition properties
- Gradient coloring from blue (start) to yellow (end)
- Terminal node with dashed border

### Template 2: Branching Flow

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
graph TD
    A["Decision Point"]
    B["Branch 1"]
    C["Branch 2"]
    D["Converge"]
    
    A --> B
    A --> C
    B --> D
    C --> D
    
    style A fill:#fff4e1
    style D fill:#e1f5ff
```

### Template 3: Cycle with Feedback (SDDP Infinite Horizon)

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'16px', 'fontFamily':'Arial'}}}%%
graph LR
    S1(["<b>Stage 1</b><br/><i>year 1, month 1</i><br/>cycle start"])
    S2["<b>Stage 2</b><br/><i>month 2</i>"]
    S3["<b>Stage 3-11</b><br/>.<br/>.<br/>."]
    S12["<b>Stage 12</b><br/><i>December</i><br/>cycle end"]
    
    S1 -->|"forward"| S2
    S2 -->|"months 3-11"| S3
    S3 -->|"final month"| S12
    S12 -.->|"<b>cycle with discount β < 1</b><br/><i>infinite horizon</i>"| S1
    
    style S1 fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
    style S2 fill:#fff9e6,stroke:#ffaa00,stroke-width:2px
    style S3 fill:#fff9e6,stroke:#ffaa00,stroke-width:2px
    style S12 fill:#ffe1e1,stroke:#cc0000,stroke-width:3px
    linkStyle 3 stroke:#cc0000,stroke-width:3px,stroke-dasharray: 5 5
```

**Key features:**
- Red cycle-end node contrasting with blue start
- Dashed red feedback arrow
- Multi-line edge label with bold/italic formatting
- Stadium shape for cycle endpoints
- Clear temporal progression in labels

## Testing Your Diagrams

### In Markdown Preview
- Use GitHub, VS Code, or Typora preview
- Check that labels are readable
- Verify colors aren't too bold

### In PDF
- Generate test PDF: `make mathematical`
- Check node spacing
- Verify text isn't cut off
- Ensure colors print well in grayscale

## Quick Reference

| Element | Syntax | Use Case |
|---------|--------|----------|
| Solid arrow | `A --> B` | Direct flow |
| Dashed arrow | `A -.-> B` | Cycle/feedback |
| Label on edge | `A -->|"label"| B` | Describe transition |
| Line break in node | `A["Line 1<br/>Line 2"]` | Multi-line labels |
| Italic text | `A["Text<br/><i>italic</i>"]` | Descriptive info |
| Bold text | `A["<b>Text</b>"]` | Emphasis |
| Subscript | `A["V<sub>T+1</sub>"]` | Mathematical notation |
| Stadium shape | `A(["Label"])` | Start/end nodes |
| Rectangle | `A["Label"]` | Regular nodes |
| Style node fill | `style A fill:#e1f5ff` | Background color |
| Style node stroke | `style A stroke:#0066cc,stroke-width:3px` | Border |
| Style link | `linkStyle 0 stroke:#cc0000,stroke-width:3px` | Edge styling |
| Subgraph | `subgraph NAME["Label"]...end` | Group nodes |
| Direction in subgraph | `direction LR` | Control layout |
| Theme init | `%%{init: {'theme':'base', 'themeVariables': {'fontSize':'16px'}}}%%` | PDF rendering |
| Unicode symbols | `ω₁ β α` | Math (NOT $\omega_1$) |

### Color Palette Reference

| Color | Hex | Use Case |
|-------|-----|----------|
| Light blue | `#e1f5ff` | Start nodes, stages |
| Blue border | `#0066cc` | Important borders |
| Light yellow | `#fff9e6` | Intermediate nodes |
| Yellow border | `#ffaa00` | Decision nodes |
| Orange border | `#ff8800` | Final decisions |
| Light red | `#ffe1e1` | Cycle/terminal nodes |
| Red border | `#cc0000` | Cycles, uncertainty |
| Light gray | `#f0f0f0` | Null/terminal states |
| Gray border | `#666666` | Inactive states |
| Subgraph background | `#f9f9f9` | Container styling |
| Subgraph border | `#333333` | Container border |

## Summary

**Good Mermaid diagrams:**
- Have clear, descriptive labels
- Use consistent styling
- Choose appropriate diagram types
- Include context (not just node names)
- Render well in PDF
- Are semantically meaningful

**Avoid:**
- ASCII art in Mermaid
- Over-styling
- Unlabeled nodes
- Overly complex diagrams
- Random colors
