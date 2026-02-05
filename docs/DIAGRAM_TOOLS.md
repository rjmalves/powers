Based on my research, I can provide you with a comprehensive analysis of how those beautiful "hand-drawn" diagrams are likely made and what tools you can use to achieve similar results.

---

## Analysis: How Those Diagrams Are Made

The diagrams you see on that blog (and similar technical blogs) with the **cartoonish, hand-drawn, sketchy style** are almost certainly created using **[Excalidraw](https://excalidraw.com/)** or similar tools that use **RoughJS** for rendering. The key characteristics are:

- **Imperfect, wobbly lines** (simulating hand-drawing)
- **Filled shapes with cross-hatching** or simple fills
- **Hand-written-style fonts** (Virgil font in Excalidraw)
- **Warm, muted color palettes**

This style has become extremely popular in technical blogging because it feels approachable, informal, and helps readers focus on concepts rather than being distracted by overly polished graphics.

---

## Recommended Solution: Excalidraw + Mermaid-to-Excalidraw

Great news! There's an **official tool** from the Excalidraw team specifically for your use case:

### 🎯 **@excalidraw/mermaid-to-excalidraw** (Official Package)

This npm package directly converts your existing Mermaid diagrams into Excalidraw's native format with the hand-drawn style.

**Installation:**
```bash
npm i @excalidraw/mermaid-to-excalidraw
```

**Usage Example:**
```javascript name=convert-mermaid.js
import { parseMermaidToExcalidraw } from "@excalidraw/mermaid-to-excalidraw";
import { convertToExcalidrawElements } from "@excalidraw/excalidraw";

const mermaidDiagram = `
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
`;

try {
  const { elements, files } = await parseMermaidToExcalidraw(
    mermaidDiagram,
    {
      themeVariables: { fontSize: "20px" },
    }
  );
  const excalidrawElements = convertToExcalidrawElements(elements);
  // Use these elements in Excalidraw
} catch (e) {
  console.error(e);
}
```

**Supported Diagram Types:**
| Diagram Type | Support Level |
|--------------|---------------|
| Flowcharts | ✅ Full native conversion |
| Sequence diagrams | ✅ Supported |
| Class diagrams | ✅ Supported |
| Other types | ⚠️ Rendered as images |

---

## Complete Tooling Ecosystem

### 1. **For Direct Editing & Creation**

| Tool | Best For | Integration |
|------|----------|-------------|
| **[Excalidraw](https://excalidraw.com/)** | Web-based, collaborative | Export PNG/SVG |
| **[Excalidraw VS Code Extension](https://marketplace.visualstudio.com/items?itemName=excalidraw.excalidraw-editor)** | Edit `.excalidraw` files in VS Code | Native `.excalidraw` files |
| **[tldraw](https://www.tldraw.com/)** | Minimalist, open-source alternative | Similar hand-drawn style |

### 2. **For Automation & CI/CD**

| Tool | Purpose |
|------|---------|
| **[excalidraw-cli](https://github.com/hediet/excalidraw-cli)** | Export `.excalidraw` → PNG/SVG via CLI |
| **[@excalidraw/mermaid-to-excalidraw](https://github.com/excalidraw/mermaid-to-excalidraw)** | Convert Mermaid → Excalidraw |
| **[Kroki](https://kroki.io/)** | Multi-format diagram server (not hand-drawn, but useful for pipeline) |

### 3. **For Note-Taking Integration**

| Tool | Integration |
|------|-------------|
| **[Obsidian Excalidraw Plugin](https://github.com/zsviczian/obsidian-excalidraw-plugin)** | Full Excalidraw editing inside Obsidian |
| **Notion** | Embed Excalidraw via links |

---

## Recommended Workflow for Your Migration

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR CURRENT STATE                        │
│              Mermaid/ASCII diagrams in Markdown              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  OPTION A: Automated Conversion Pipeline                     │
│  ─────────────────────────────────────────                   │
│  1. Use @excalidraw/mermaid-to-excalidraw                   │
│  2. Run in CI or as npm script                               │
│  3. Export to PNG/SVG with excalidraw-cli                   │
│  4. Embed images in documentation                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  OPTION B: Manual + Git-tracked                              │
│  ─────────────────────────────────────                       │
│  1. Open existing Mermaid in Excalidraw web (paste & convert)│
│  2. Refine/beautify the diagram manually                     │
│  3. Save as .excalidraw (JSON, version-controlled)          │
│  4. Export PNG for docs, keep source for edits              │
└─────────────────────────────────────────────────────────────┘
```

---

## My Recommendation

For your use case, I suggest a **hybrid approach**:

1. **For new diagrams**: Create directly in Excalidraw (web or VS Code extension) - it's fast and intuitive
2. **For existing Mermaid diagrams**: Use `@excalidraw/mermaid-to-excalidraw` to convert, then refine in Excalidraw
3. **For automation**: Set up a simple script that:
   - Converts `.mmd` files to `.excalidraw`
   - Exports to PNG for documentation
   - Keeps `.excalidraw` source files in version control

```bash name=export-diagrams.sh
#!/bin/bash
# Export all Excalidraw diagrams to PNG
for file in ./diagrams/*.excalidraw; do
  filename=$(basename "$file" .excalidraw)
  excalidraw-cli render "$file" -o "./docs/images/$filename.png"
done
```

---

## Why This Works Better Than Mermaid for Your Goals

| Aspect | Mermaid | Excalidraw |
|--------|---------|------------|
| **Visual style** | Clean, technical | Hand-drawn, approachable |
| **Customization** | Limited | Full control |
| **PDF export** | Requires renderer | Direct PNG/SVG export |
| **Version control** | ✅ Text-based | ✅ JSON-based |
| **Collaboration** | Limited | Real-time (web) |
| **Reader engagement** | Formal | Casual, blog-friendly |

---

Would you like me to help you set up any specific part of this workflow, such as creating a conversion script or exploring the VS Code extension integration?
