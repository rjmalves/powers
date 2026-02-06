# POWE.RS Diagrams

Hand-drawn Excalidraw diagrams for POWE.RS documentation.

## Quick Start

1. **Edit**: Open `.excalidraw` files in [Excalidraw](https://excalidraw.com/) or the VS Code extension (`excalidraw.excalidraw-editor`)
2. **Export**: Run `make -C docs diagrams` or `npm run diagrams:export` from the repo root
3. **Embed**: Reference exported PNGs in markdown: `![Name](diagrams/exports/png/sddp/diagram-name.png)`

## Directory Layout

```
excalidraw/          Source files (version controlled)
  sddp/              SDDP algorithm diagrams
  hpc/               HPC parallel computing diagrams
  data/              Data model diagrams
  components/        Reusable icons, legends, color swatches

exports/             Generated PNG/SVG (gitignored, regenerate with make)
  png/{sddp,hpc,data}/
  svg/{sddp,hpc,data}/

legacy/              Archive of old ASCII/Mermaid diagrams (reference only)
  mermaid-extracts/
```

## Style Guide

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for colors, typography, shapes, and naming conventions.

## Converting Mermaid Diagrams

```bash
# List existing Mermaid blocks in documentation
npm run diagrams:list

# Convert Mermaid blocks to Excalidraw (requires npm install first)
node scripts/convert-mermaid.mjs docs/MATHEMATICAL_FORMULATIONS.md --output-dir docs/diagrams/excalidraw/sddp
```

After conversion, open the generated `.excalidraw` files and apply the POWE.RS color palette and style from the style guide.

## Migration Status

See [DIAGRAM_MIGRATION_PLAN.md](../DIAGRAM_MIGRATION_PLAN.md) for the full inventory and progress tracking.

## Naming Convention

```
<diagram-name>.excalidraw

Examples:
  policy-graph-finite.excalidraw
  hybrid-parallelism.excalidraw
  directory-structure.excalidraw
```

Files are organized by domain directory (`sddp/`, `hpc/`, `data/`), so the filename itself does not need a domain prefix.
