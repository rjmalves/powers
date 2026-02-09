#!/usr/bin/env node
/**
 * POWE.RS Excalidraw Diagram Export Script
 *
 * Exports .excalidraw JSON files to PNG and SVG using Playwright (headless Chrome)
 * and @resvg/resvg-js for high-quality SVG-to-PNG rasterization.
 *
 * Architecture:
 *   1. esbuild bundles @excalidraw/excalidraw into a single browser-ready IIFE
 *   2. Playwright launches headless Chromium
 *   3. A minimal HTML page loads the bundle; exportToSvg() is exposed on window
 *   4. For each .excalidraw file: JSON is sent to the page, exportToSvg() is called
 *   5. SVG string is returned to Node.js
 *   6. @resvg/resvg-js converts SVG -> PNG at the requested scale
 *
 * Usage:
 *   node scripts/export-diagrams.mjs [--format png|svg|both] [--scale 2]
 *
 * Run from the repository root after: npm run diagrams:setup
 */

import { chromium } from "playwright";
import { Resvg } from "@resvg/resvg-js";
import { build } from "esbuild";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { createServer } from "http";

// MathJax — server-side TeX-to-SVG rendering
import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG as MathJaxSVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

// Configuration
const EXCALIDRAW_DIR = path.join(REPO_ROOT, "docs/diagrams/excalidraw");
const EXPORT_DIR = path.join(REPO_ROOT, "docs/diagrams/exports");
const BUNDLE_DIR = path.join(REPO_ROOT, "node_modules/.cache/powers-diagrams");

// ---------------------------------------------------------------------------
// MathJax initialisation — TeX → pure-SVG-path rendering (server-side)
// ---------------------------------------------------------------------------
const mjAdaptor = liteAdaptor();
RegisterHTMLHandler(mjAdaptor);

const mjTex = new TeX({ packages: AllPackages });
const mjSvg = new MathJaxSVG({ fontCache: "none" }); // 'none' = fully self-contained paths
const mjDoc = mathjax.document("", { InputJax: mjTex, OutputJax: mjSvg });

/**
 * Convert a LaTeX string to a self-contained SVG string (pure <path> elements,
 * no <foreignObject>, no external fonts).
 *
 * @param {string} tex   - LaTeX source (without delimiters)
 * @param {boolean} display - true for display-math, false for inline
 * @param {number} emPx  - the em-size in pixels (maps to Excalidraw font-size)
 * @returns {string} complete `<svg …>…</svg>` markup
 */
function tex2svg(tex, display = false, emPx = 16) {
  const node = mjDoc.convert(tex, {
    display,
    em: emPx,
    ex: emPx * 0.5, // reasonable ex approximation
  });
  const svgNode = mjAdaptor.firstChild(node); // unwrap <mjx-container>
  return mjAdaptor.outerHTML(svgNode);
}

// ---------------------------------------------------------------------------
// LaTeX post-processing for Excalidraw SVGs
// ---------------------------------------------------------------------------

/**
 * Regex to detect LaTeX delimiters inside text content.
 *
 *   - $$...$$  → display-mode math
 *   - $...$    → inline-mode math
 *
 * A text element is considered a "math element" when its *entire* text
 * (across all <text> children in the group) is wrapped in $ or $$ delimiters.
 * Mixed text-and-math in the same element is NOT supported — that would be
 * extremely fragile with Excalidraw's text model.
 */
const DISPLAY_MATH_RE = /^\$\$([\s\S]+)\$\$$/;
const INLINE_MATH_RE = /^\$([\s\S]+)\$$/;

/**
 * Replace Excalidraw text elements that contain LaTeX delimiters with
 * MathJax-rendered SVG <path> elements.
 *
 * The function scans for groups of the form:
 *   <g transform="translate(X Y) rotate(…)">
 *     <text x="…" y="…" font-size="…" fill="…" text-anchor="…" …>$$\min_x c^T x$$</text>
 *     <!-- possibly more <text> children for multi-line content -->
 *   </g>
 *
 * When the concatenated text content matches $…$ or $$…$$ delimiters the
 * entire <g> is replaced with MathJax output, preserving:
 *   - position (translate)
 *   - colour (fill)
 *   - approximate sizing (based on font-size)
 *
 * @param {string} svgString - the full Excalidraw-exported SVG
 * @returns {string} SVG with LaTeX elements rendered to pure paths
 */
function renderLatexInSvg(svgString) {
  // Match <g> groups that contain one or more <text> elements.
  // Excalidraw wraps every text element in a <g transform="translate(…)…">
  // Some elements have extra attributes (stroke-opacity, fill-opacity) before transform,
  // so we use [^>]* to skip any attributes preceding the transform.
  const textGroupRe =
    /<g\s+[^>]*?transform="(translate\([^)]+\)\s*rotate\([^)]+\))"[^>]*>((?:<text\s[^]*?<\/text>)+)<\/g>/g;

  let result = svgString;
  let mathCount = 0;

  // Collect all matches first (avoid mutation during iteration)
  const matches = [...svgString.matchAll(textGroupRe)];

  for (const match of matches) {
    const fullMatch = match[0];
    const transformAttr = match[1];
    const textBlock = match[2];

    // Extract individual <text> contents and join (multi-line text)
    const textContents = [...textBlock.matchAll(/<text[^>]*>([^<]*)<\/text>/g)];
    // Decode HTML entities (Excalidraw encodes <, >, &, " in SVG text content)
    const decodeEntities = (s) =>
      s
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&amp;/g, "&")
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
    const joinedText = textContents
      .map((m) => decodeEntities(m[1]))
      .join("\n")
      .trim();

    // Check for LaTeX delimiters
    let texSource = null;
    let isDisplay = false;

    const dispMatch = joinedText.match(DISPLAY_MATH_RE);
    if (dispMatch) {
      texSource = dispMatch[1].trim();
      isDisplay = true;
    } else {
      const inlMatch = joinedText.match(INLINE_MATH_RE);
      if (inlMatch) {
        texSource = inlMatch[1].trim();
        isDisplay = false;
      }
    }

    if (texSource === null) continue; // not a math element

    // Extract styling from the first <text> element
    const firstText = textContents[0][0]; // the full <text …>…</text> match string
    // We need to re-match the full <text> tag to get attributes
    const firstTextTag = textBlock.match(/<text\s([^>]*)>/);
    const attrs = firstTextTag ? firstTextTag[1] : "";

    const fillMatch = attrs.match(/fill="([^"]+)"/);
    const fill = fillMatch ? fillMatch[1] : "#1e1e1e";

    const fontSizeMatch = attrs.match(/font-size="([^"]+)"/);
    const fontSizePx = fontSizeMatch ? parseFloat(fontSizeMatch[1]) : 16;

    const textAnchorMatch = attrs.match(/text-anchor="([^"]+)"/);
    const textAnchor = textAnchorMatch ? textAnchorMatch[1] : "start";

    // Compute the width of the original text block (for alignment)
    // From the <g> group's translate we get the position; from the first <text>
    // x attribute we get the offset used for centering.
    const textXMatch = textContents[0][0].match(/<text[^>]*\bx="([^"]+)"/);
    const textX = textXMatch ? parseFloat(textXMatch[1]) : 0;

    // Render LaTeX → SVG
    let mathSvg;
    try {
      mathSvg = tex2svg(texSource, isDisplay, fontSizePx);
    } catch (err) {
      console.warn(
        `  [LaTeX] Failed to render: ${texSource.substring(0, 40)}… — ${err.message}`
      );
      continue;
    }

    // Parse MathJax SVG dimensions — the viewBox gives us the math's bounding box
    // viewBox="X_min Y_min Width Height"  (Y_min is typically negative for ascenders)
    const viewBoxMatch = mathSvg.match(/viewBox="([^"]+)"/);
    if (!viewBoxMatch) continue;
    const [vbX, vbY, vbW, vbH] = viewBoxMatch[1].split(/\s+/).map(Number);

    // MathJax uses "ex" units for width/height; we need absolute px.
    // The width="Nex" and height="Nex" attributes tell us the logical size.
    const widthExMatch = mathSvg.match(/width="([0-9.]+)ex"/);
    const heightExMatch = mathSvg.match(/height="([0-9.]+)ex"/);
    const widthEx = widthExMatch ? parseFloat(widthExMatch[1]) : vbW / 1000;
    const heightEx = heightExMatch ? parseFloat(heightExMatch[1]) : vbH / 1000;

    // 1 ex ≈ 0.5 em; 1 em = fontSizePx
    const exPx = fontSizePx * 0.5;
    const mathWidthPx = widthEx * exPx;
    const mathHeightPx = heightEx * exPx;

    // Vertical alignment: MathJax's vertical-align style tells us the baseline shift
    const vertAlignMatch = mathSvg.match(/vertical-align:\s*(-?[0-9.]+)ex/);
    const vertAlignEx = vertAlignMatch ? parseFloat(vertAlignMatch[1]) : 0;
    const vertAlignPx = vertAlignEx * exPx;

    // Build the replacement <g> with the MathJax SVG embedded as a nested <svg>
    // We need to:
    //   1. Keep the original transform (translate + rotate) — positions the element
    //   2. Offset horizontally based on text-anchor (start/middle/end)
    //   3. Apply the fill colour to the MathJax paths
    //   4. Scale the MathJax SVG to match the font size

    // Horizontal offset for alignment
    let xOffset = 0;
    if (textAnchor === "middle") {
      xOffset = textX - mathWidthPx / 2;
    } else if (textAnchor === "end") {
      xOffset = textX - mathWidthPx;
    } else {
      // "start" — use the text x directly
      xOffset = textX;
    }

    // Vertical offset: Excalidraw uses dominant-baseline="text-before-edge"
    // which means y=0 is the top of the text. MathJax's viewBox starts at the
    // ascender top (negative Y). We want the math vertically centered relative
    // to the original text block height.
    // Original text height: for N lines, it's roughly (N-1)*lineHeight + fontSizePx
    const nLines = textContents.length;
    // Excalidraw line spacing: the y offset between consecutive <text> elements
    let lineHeight = fontSizePx * 1.5; // default
    if (nLines > 1) {
      const y0Match = textContents[0][0].match(/\by="([^"]+)"/);
      const y1Match = textContents[1][0].match(/\by="([^"]+)"/);
      if (y0Match && y1Match) {
        lineHeight = parseFloat(y1Match[1]) - parseFloat(y0Match[1]);
      }
    }
    const origTextHeight = (nLines - 1) * lineHeight + fontSizePx;
    const yOffset = (origTextHeight - mathHeightPx) / 2;

    // Strip MathJax wrapper attributes we don't need and inject our colour
    // Replace the MathJax SVG's own fill with the Excalidraw element's fill
    let innerSvg = mathSvg
      .replace(/^<svg\s/, `<svg x="${xOffset}" y="${yOffset}" `)
      .replace(/style="[^"]*"/, "")
      .replace(/width="[^"]*"/, `width="${mathWidthPx}"`)
      .replace(/height="[^"]*"/, `height="${mathHeightPx}"`)
      .replace(/role="img"/, "")
      .replace(/focusable="false"/, "");

    // Set fill colour on the MathJax content
    // MathJax uses fill="currentColor" and stroke="currentColor" — replace with actual colour
    innerSvg = innerSvg
      .replace(/fill="currentColor"/g, `fill="${fill}"`)
      .replace(/stroke="currentColor"/g, `stroke="${fill}"`);

    const replacement = `<g transform="${transformAttr}">${innerSvg}</g>`;

    result = result.replace(fullMatch, replacement);
    mathCount++;
  }

  if (mathCount > 0) {
    console.log(`  [LaTeX] Rendered ${mathCount} equation(s)`);
  }

  return result;
}

// Parse CLI arguments
function parseArgs() {
  const args = process.argv.slice(2);
  let format = "both";
  let scale = 2;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--format" && args[i + 1]) {
      format = args[++i];
    } else if (args[i] === "--scale" && args[i + 1]) {
      scale = parseInt(args[++i], 10);
    } else if (args[i] === "--help" || args[i] === "-h") {
      console.log(`POWE.RS Excalidraw Diagram Export

Usage: node scripts/export-diagrams.mjs [options]

Options:
  --format <png|svg|both>  Output format (default: both)
  --scale <number>         PNG export scale factor (default: 2)
  --help                   Show this help`);
      process.exit(0);
    }
  }

  return { format, scale };
}

/**
 * Embed Excalidraw fonts in SVG as base64 data URIs.
 * This ensures the SVG renders correctly when viewed standalone.
 */
function embedFontsInSvg(svgString) {
  const fontDir = path.join(
    REPO_ROOT,
    "node_modules/@excalidraw/excalidraw/dist/excalidraw-assets"
  );

  const fonts = {
    Virgil: path.join(fontDir, "Virgil.woff2"),
    Cascadia: path.join(fontDir, "Cascadia.woff2"),
    Assistant: path.join(fontDir, "Assistant-Regular.woff2"),
  };

  let result = svgString;

  for (const [fontFamily, fontPath] of Object.entries(fonts)) {
    if (fs.existsSync(fontPath)) {
      const fontData = fs.readFileSync(fontPath);
      const base64 = fontData.toString("base64");
      const dataUri = `data:font/woff2;base64,${base64}`;

      // Replace unpkg URLs or empty font-face declarations with embedded base64
      // Match patterns like:
      // src: url("https://unpkg.com/@excalidraw/excalidraw@.../*.woff2");
      // or empty declarations
      const fontFacePattern = new RegExp(
        `(font-family:\\s*"${fontFamily}";\\s*)(?:src:\\s*url\\([^)]+\\);)?`,
        "g"
      );
      const replacement = `font-family: "${fontFamily}";\n        src: url("${dataUri}") format("woff2");`;

      result = result.replace(fontFacePattern, replacement);
    }
  }

  return result;
}

// Find all .excalidraw files (excluding components/)
function findExcalidrawFiles(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "components") continue;
      files.push(...findExcalidrawFiles(fullPath));
    } else if (entry.name.endsWith(".excalidraw")) {
      files.push(fullPath);
    }
  }
  return files.sort();
}

/**
 * Bundle @excalidraw/excalidraw into a single browser-ready IIFE using esbuild.
 *
 * Excalidraw 0.18+ ships as unbundled ESM with bare specifiers (jotai, react,
 * roughjs, etc.) that browsers cannot resolve natively. Import maps only cover
 * direct imports and fail on transitive bare specifiers. The correct solution
 * is to pre-bundle everything into a single file.
 */
async function bundleExcalidraw() {
  const bundlePath = path.join(BUNDLE_DIR, "excalidraw-export.js");

  // Skip rebuild if bundle is newer than the installed package
  const pkgJsonPath = path.join(
    REPO_ROOT,
    "node_modules/@excalidraw/excalidraw/package.json"
  );
  if (fs.existsSync(bundlePath) && fs.existsSync(pkgJsonPath)) {
    const bundleStat = fs.statSync(bundlePath);
    const pkgStat = fs.statSync(pkgJsonPath);
    if (bundleStat.mtimeMs > pkgStat.mtimeMs) {
      return bundlePath;
    }
  }

  fs.mkdirSync(BUNDLE_DIR, { recursive: true });

  // Entrypoint: only export the functions we need for SVG generation
  const entryContent = `
    import { exportToSvg } from "@excalidraw/excalidraw";
    window.__exportToSvg = exportToSvg;
    window.__excalidrawReady = true;
    document.title = "READY";
  `;
  const entryPath = path.join(BUNDLE_DIR, "excalidraw-entry.mjs");
  fs.writeFileSync(entryPath, entryContent);

  await build({
    entryPoints: [entryPath],
    bundle: true,
    format: "iife",
    platform: "browser",
    outfile: bundlePath,
    minify: false,
    sourcemap: false,
    // Excalidraw uses JSX — handle .js files containing JSX
    loader: { ".js": "jsx" },
    jsx: "automatic",
    jsxImportSource: "react",
    // Suppress non-critical warnings (e.g. circular deps in Mermaid)
    logLevel: "warning",
    define: {
      "process.env.NODE_ENV": '"production"',
      "process.env.IS_PREACT": '"false"',
    },
  });

  // Cleanup entrypoint
  fs.rmSync(entryPath, { force: true });

  return bundlePath;
}

// Start a local static file server
function startServer(bundlePath) {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const url = decodeURIComponent(req.url);
      let filePath;

      if (url === "/excalidraw-bundle.js") {
        filePath = bundlePath;
      } else {
        filePath = path.join(REPO_ROOT, url);
      }

      // Security: prevent directory traversal
      if (
        !filePath.startsWith(REPO_ROOT) &&
        !filePath.startsWith(BUNDLE_DIR)
      ) {
        res.writeHead(403);
        res.end("Forbidden");
        return;
      }

      if (!fs.existsSync(filePath)) {
        res.writeHead(404);
        res.end("Not found: " + req.url);
        return;
      }

      const ext = path.extname(filePath);
      const mimeTypes = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html",
        ".woff2": "font/woff2",
        ".svg": "image/svg+xml",
        ".png": "image/png",
      };

      res.writeHead(200, {
        "Content-Type": mimeTypes[ext] || "application/octet-stream",
        "Access-Control-Allow-Origin": "*",
      });
      fs.createReadStream(filePath).pipe(res);
    });

    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      resolve({ server, port });
    });
  });
}

// The HTML page that loads the pre-bundled Excalidraw export
function getExportHTML(port) {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>POWE.RS Diagram Export</title>
  <style>
    @font-face {
      font-family: "Virgil";
      src: url("http://127.0.0.1:${port}/node_modules/@excalidraw/excalidraw/dist/excalidraw-assets/Virgil.woff2") format("woff2");
    }
    @font-face {
      font-family: "Cascadia";
      src: url("http://127.0.0.1:${port}/node_modules/@excalidraw/excalidraw/dist/excalidraw-assets/Cascadia.woff2") format("woff2");
    }
    @font-face {
      font-family: "Assistant";
      src: url("http://127.0.0.1:${port}/node_modules/@excalidraw/excalidraw/dist/excalidraw-assets/Assistant-Regular.woff2") format("woff2");
      font-weight: normal;
    }
  </style>
</head>
<body>
  <div id="root"></div>
  <script src="http://127.0.0.1:${port}/excalidraw-bundle.js"></script>
</body>
</html>`;
}

async function main() {
  const { format, scale } = parseArgs();

  console.log("POWE.RS Diagram Export (Playwright + esbuild)");
  console.log("==============================================");
  console.log("");

  // Find files
  const files = findExcalidrawFiles(EXCALIDRAW_DIR);
  if (files.length === 0) {
    console.log("No .excalidraw files found in", EXCALIDRAW_DIR);
    process.exit(0);
  }
  console.log(`Found ${files.length} .excalidraw files`);
  console.log(`Format: ${format}, Scale: ${scale}x`);
  console.log("");

  // Bundle Excalidraw for the browser
  process.stdout.write("Bundling Excalidraw for browser... ");
  const bundlePath = await bundleExcalidraw();
  console.log("ok");

  // Start local server
  const { server, port } = await startServer(bundlePath);
  console.log(`Static server on http://127.0.0.1:${port}`);

  // Launch browser
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
  } catch (err) {
    console.error(
      "Failed to launch Chromium. Run: npx playwright install chromium"
    );
    console.error(err.message);
    server.close();
    process.exit(1);
  }

  const context = await browser.newContext({
    viewport: { width: 4096, height: 4096 },
  });
  const page = await context.newPage();

  // Suppress non-critical console noise from Excalidraw
  page.on("console", () => {});
  page.on("pageerror", () => {});

  // Write the HTML to a temp file and navigate
  const htmlPath = path.join(REPO_ROOT, ".export-diagrams-temp.html");
  fs.writeFileSync(htmlPath, getExportHTML(port));

  try {
    await page.goto(`http://127.0.0.1:${port}/.export-diagrams-temp.html`, {
      waitUntil: "networkidle",
      timeout: 30000,
    });

    // Wait for Excalidraw to load
    await page.waitForFunction(
      () => window.__excalidrawReady === true,
      null,
      { timeout: 30000 }
    );

    // Wait for fonts to load
    await page.evaluate(async () => {
      await Promise.all([
        document.fonts.load("20px Virgil"),
        document.fonts.load("20px Cascadia"),
        document.fonts.load("20px Assistant"),
      ]);
    });

    console.log("Excalidraw loaded successfully");
    console.log("");

    let successCount = 0;
    let failCount = 0;

    for (const file of files) {
      const relPath = path.relative(EXCALIDRAW_DIR, file);
      const subdir = path.dirname(relPath);
      const basename = path.basename(file, ".excalidraw");

      process.stdout.write(`  ${relPath}... `);

      try {
        // Read and parse the .excalidraw JSON
        const json = JSON.parse(fs.readFileSync(file, "utf-8"));
        const elements = json.elements || [];
        const appState = {
          exportBackground: true,
          exportWithDarkMode: false,
          viewBackgroundColor:
            json.appState?.viewBackgroundColor || "#ffffff",
        };
        const files_data = json.files || {};

        // Call exportToSvg in the browser context
        const svgString = await page.evaluate(
          async ({ elements, appState, files }) => {
            try {
              const svg = await window.__exportToSvg({
                elements,
                appState,
                files,
                exportPadding: 20,
              });
              // svg is an SVGSVGElement — serialize it
              const serializer = new XMLSerializer();
              return { ok: true, svg: serializer.serializeToString(svg) };
            } catch (e) {
              return { ok: false, error: e.message || String(e) };
            }
          },
          { elements, appState, files: files_data }
        );

        if (!svgString.ok) {
          process.stdout.write(`FAILED (${svgString.error})\n`);
          failCount++;
          continue;
        }

        // Embed fonts in SVG for standalone viewing
        const svgWithFonts = embedFontsInSvg(svgString.svg);

        // Render LaTeX equations ($..$ / $$..$$) to pure SVG paths
        const svgFinal = renderLatexInSvg(svgWithFonts);

        // Save SVG
        if (format === "svg" || format === "both") {
          const svgDir = path.join(EXPORT_DIR, "svg", subdir);
          fs.mkdirSync(svgDir, { recursive: true });
          const svgPath = path.join(svgDir, `${basename}.svg`);
          fs.writeFileSync(svgPath, svgFinal);
        }

        // Convert SVG to PNG using resvg
        if (format === "png" || format === "both") {
          const pngDir = path.join(EXPORT_DIR, "png", subdir);
          fs.mkdirSync(pngDir, { recursive: true });
          const pngPath = path.join(pngDir, `${basename}.png`);

          const resvg = new Resvg(svgFinal, {
            dpi: 72 * scale,
            shapeRendering: 2, // geometricPrecision
            textRendering: 1, // optimizeLegibility
            imageRendering: 0, // optimizeQuality
            fitTo: {
              mode: "zoom",
              value: scale,
            },
            font: {
              loadSystemFonts: true,
            },
          });
          const pngData = resvg.render();
          const pngBuffer = pngData.asPng();
          fs.writeFileSync(pngPath, pngBuffer);
        }

        const outputs = [];
        if (format === "svg" || format === "both") outputs.push("svg");
        if (format === "png" || format === "both") outputs.push("png");
        process.stdout.write(`ok (${outputs.join("+")})\n`);
        successCount++;
      } catch (err) {
        process.stdout.write(`FAILED (${err.message})\n`);
        failCount++;
      }
    }

    console.log("");
    console.log("Export Complete");
    console.log("===============");
    console.log(`  Success: ${successCount}`);
    if (failCount > 0) console.log(`  Failed:  ${failCount}`);
    console.log(`  Total:   ${files.length}`);
    console.log("");
    if (format === "png" || format === "both") {
      console.log(`  PNG: ${EXPORT_DIR}/png/`);
    }
    if (format === "svg" || format === "both") {
      console.log(`  SVG: ${EXPORT_DIR}/svg/`);
    }

    if (failCount > 0) process.exitCode = 1;
  } finally {
    // Cleanup
    fs.rmSync(htmlPath, { force: true });
    await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
