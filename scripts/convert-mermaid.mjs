#!/usr/bin/env node
/**
 * POWE.RS Mermaid to Excalidraw Converter
 * 
 * This script extracts Mermaid diagrams from Markdown files and converts them
 * to Excalidraw format for manual refinement.
 * 
 * Usage:
 *   node scripts/convert-mermaid.mjs <input.md> [--output-dir <dir>]
 * 
 * Run from the repository root after: npm install
 * 
 * Requirements:
 *   @excalidraw/mermaid-to-excalidraw (installed via npm)
 */

import { parseMermaidToExcalidraw } from "@excalidraw/mermaid-to-excalidraw";
import { convertToExcalidrawElements } from "@excalidraw/excalidraw";
import * as fs from "fs";
import * as path from "path";

// POWE.RS color palette (matching docs/diagrams/STYLE_GUIDE.md)
const POWERS_PALETTE = {
  stages: { fill: "#a5d8ff", stroke: "#1971c2" },
  hydro: { fill: "#b2f2bb", stroke: "#2f9e44" },
  thermal: { fill: "#ffec99", stroke: "#f08c00" },
  cycles: { fill: "#ffc9c9", stroke: "#e03131" },
  parallel: { fill: "#d0bfff", stroke: "#7048e8" },
  data: { fill: "#99e9f2", stroke: "#0c8599" },
  terminal: { fill: "#e9ecef", stroke: "#495057" },
};

/**
 * Extract Mermaid code blocks from Markdown content
 */
function extractMermaidBlocks(markdown) {
  const mermaidRegex = /```mermaid\n([\s\S]*?)```/g;
  const blocks = [];
  let match;
  let index = 0;

  while ((match = mermaidRegex.exec(markdown)) !== null) {
    // Find the nearest heading before this block
    const beforeBlock = markdown.substring(0, match.index);
    const headingMatch = beforeBlock.match(/#+\s+([^\n]+)\n[^#]*$/);
    const heading = headingMatch ? headingMatch[1].trim() : `diagram-${index}`;

    blocks.push({
      index: index++,
      heading: heading,
      content: match[1].trim(),
      position: match.index,
    });
  }

  return blocks;
}

/**
 * Generate a valid filename from a heading
 */
function sanitizeFilename(heading) {
  return heading
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .substring(0, 50);
}

/**
 * Create Excalidraw file structure
 */
function createExcalidrawFile(elements, files = {}) {
  return {
    type: "excalidraw",
    version: 2,
    source: "powers-converter",
    elements: elements,
    files: files,
    appState: {
      viewBackgroundColor: "#ffffff",
      gridSize: null,
    },
  };
}

/**
 * Convert a single Mermaid diagram to Excalidraw
 */
async function convertMermaidToExcalidraw(mermaidCode) {
  try {
    const { elements, files } = await parseMermaidToExcalidraw(mermaidCode, {
      themeVariables: {
        fontSize: "16px",
      },
    });

    const excalidrawElements = convertToExcalidrawElements(elements);
    return { success: true, elements: excalidrawElements, files };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Main function
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(`
POWE.RS Mermaid to Excalidraw Converter

Usage:
  node scripts/convert-mermaid.mjs <input.md> [--output-dir <dir>]
  node scripts/convert-mermaid.mjs --list <input.md>

Options:
  --output-dir <dir>   Output directory for .excalidraw files
                       Default: ./docs/diagrams/excalidraw/
  --list               Only list diagrams, don't convert

Examples:
  node scripts/convert-mermaid.mjs docs/MATHEMATICAL_FORMULATIONS.md
  node scripts/convert-mermaid.mjs docs/MATHEMATICAL_FORMULATIONS.md --output-dir docs/diagrams/excalidraw/sddp
`);
    process.exit(0);
  }

  let inputFile = null;
  let outputDir = "./docs/diagrams/excalidraw";
  let listOnly = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--output-dir" && args[i + 1]) {
      outputDir = args[++i];
    } else if (args[i] === "--list") {
      listOnly = true;
    } else if (!args[i].startsWith("--")) {
      inputFile = args[i];
    }
  }

  if (!inputFile) {
    console.error("Error: No input file specified");
    process.exit(1);
  }

  if (!fs.existsSync(inputFile)) {
    console.error(`Error: File not found: ${inputFile}`);
    process.exit(1);
  }

  const markdown = fs.readFileSync(inputFile, "utf-8");
  const blocks = extractMermaidBlocks(markdown);

  console.log(`\nFound ${blocks.length} Mermaid diagrams in ${inputFile}:\n`);

  for (const block of blocks) {
    console.log(`  ${block.index + 1}. ${block.heading}`);
    console.log(`     Type: ${detectDiagramType(block.content)}`);
    console.log(`     Lines: ${block.content.split("\n").length}`);
    console.log();
  }

  if (listOnly) {
    process.exit(0);
  }

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`Created output directory: ${outputDir}`);
  }

  // Convert each diagram
  console.log("\nConverting diagrams...\n");

  let successCount = 0;
  let failCount = 0;

  for (const block of blocks) {
    const filename = `${sanitizeFilename(block.heading)}.excalidraw`;
    const outputPath = path.join(outputDir, filename);

    process.stdout.write(`  Converting "${block.heading}"... `);

    const result = await convertMermaidToExcalidraw(block.content);

    if (result.success) {
      const excalidrawFile = createExcalidrawFile(result.elements, result.files);
      fs.writeFileSync(outputPath, JSON.stringify(excalidrawFile, null, 2));
      console.log(`✓ ${filename}`);
      successCount++;
    } else {
      console.log(`✗ Failed: ${result.error}`);
      failCount++;

      // Save the original Mermaid for manual conversion
      const mermaidPath = path.join(outputDir, `${sanitizeFilename(block.heading)}.mmd`);
      fs.writeFileSync(mermaidPath, block.content);
      console.log(`     Saved Mermaid source to: ${mermaidPath}`);
    }
  }

  console.log(`
Summary:
  ✓ Converted: ${successCount}
  ✗ Failed: ${failCount}
  Total: ${blocks.length}

Next steps:
  1. Open .excalidraw files in Excalidraw (web or VS Code)
  2. Apply POWE.RS color palette from EXCALIDRAW_STYLE_GUIDE.md
  3. Refine labels and positioning
  4. Export to PNG/SVG for documentation
`);
}

/**
 * Detect Mermaid diagram type
 */
function detectDiagramType(content) {
  const firstLine = content.split("\n")[0].toLowerCase();
  if (firstLine.includes("graph") || firstLine.includes("flowchart")) {
    return "Flowchart";
  } else if (firstLine.includes("sequencediagram")) {
    return "Sequence Diagram";
  } else if (firstLine.includes("classdiagram")) {
    return "Class Diagram";
  } else if (firstLine.includes("gantt")) {
    return "Gantt Chart";
  } else if (firstLine.includes("statediagram")) {
    return "State Diagram";
  } else if (firstLine.includes("erdiagram")) {
    return "ER Diagram";
  } else {
    return "Unknown";
  }
}

main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
