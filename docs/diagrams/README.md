# POWE.RS Diagrams

Hand-drawn Excalidraw diagrams for POWE.RS documentation.

## Quick Start

1. **Edit**: Open `.excalidraw` files in [Excalidraw](https://excalidraw.com/) or the VS Code extension (`excalidraw.excalidraw-editor`)
2. **Export**: Run `make -C docs diagrams` or `npm run diagrams:export` from the repo root
3. **Embed**: Reference exported SVGs in markdown: `![Name](diagrams/exports/svg/sddp/diagram-name.svg)`

## Directory Layout

```
excalidraw/          Source files (version controlled)
  sddp/              SDDP algorithm diagrams
  hpc/               HPC parallel computing diagrams
  data/              Data model diagrams
  components/        Reusable icons, legends, color swatches

exports/             Generated SVG/PNG (gitignored, regenerate with make)
  svg/{sddp,hpc,data}/   ← preferred format for docs
  png/{sddp,hpc,data}/
```

## Style Guide

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for colors, typography, shapes, and naming conventions.

## Naming Convention

```
<diagram-name>.excalidraw

Examples:
  policy-graph-finite.excalidraw
  hybrid-parallelism.excalidraw
  directory-structure.excalidraw
```

Files are organized by domain directory (`sddp/`, `hpc/`, `data/`), so the filename itself does not need a domain prefix.
