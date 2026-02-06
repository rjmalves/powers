#!/usr/bin/env node
/**
 * POWE.RS Excalidraw Diagram Export Script
 *
 * Exports .excalidraw JSON files to PNG and SVG using Playwright (headless Chrome)
 * and @resvg/resvg-js for high-quality SVG-to-PNG rasterization.
 *
 * Architecture:
 *   1. Playwright launches headless Chromium
 *   2. A minimal HTML page loads @excalidraw/excalidraw via ESM import map
 *   3. For each .excalidraw file: JSON is sent to the page, exportToSvg() is called
 *   4. SVG string is returned to Node.js
 *   5. @resvg/resvg-js converts SVG -> PNG at the requested scale
 *
 * Usage:
 *   node scripts/export-diagrams.mjs [--format png|svg|both] [--scale 2]
 *
 * Run from the repository root after: npm install && npx playwright install chromium
 */

import { chromium } from "playwright";
import { Resvg } from "@resvg/resvg-js";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { createServer } from "http";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

// Configuration
const EXCALIDRAW_DIR = path.join(REPO_ROOT, "docs/diagrams/excalidraw");
const EXPORT_DIR = path.join(REPO_ROOT, "docs/diagrams/exports");
const EXCALIDRAW_DIST = path.join(
  REPO_ROOT,
  "node_modules/@excalidraw/excalidraw/dist"
);

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

// Start a local static file server for node_modules
function startServer() {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      // Serve files from the repo root
      let filePath = path.join(REPO_ROOT, decodeURIComponent(req.url));

      // Security: prevent directory traversal
      if (!filePath.startsWith(REPO_ROOT)) {
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

// The HTML page that loads Excalidraw and exports SVG
function getExportHTML(port) {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>POWE.RS Diagram Export</title>
  <script type="importmap">
  {
    "imports": {
      "react": "http://127.0.0.1:${port}/node_modules/react/cjs/react.production.min.js",
      "react-dom": "http://127.0.0.1:${port}/node_modules/react-dom/cjs/react-dom.production.min.js"
    }
  }
  </script>
</head>
<body>
  <div id="root"></div>
  <script type="module">
    // Load excalidraw
    const mod = await import("http://127.0.0.1:${port}/node_modules/@excalidraw/excalidraw/dist/prod/index.js");

    // Expose exportToSvg globally
    window.__exportToSvg = mod.exportToSvg;
    window.__excalidrawReady = true;

    // Signal ready
    document.title = "READY";
  </script>
</body>
</html>`;
}

async function main() {
  const { format, scale } = parseArgs();

  console.log("POWE.RS Diagram Export (Playwright)");
  console.log("====================================");
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

  // Start local server
  const { server, port } = await startServer();
  console.log(`Static server on http://127.0.0.1:${port}`);

  // Launch browser
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
  } catch (err) {
    console.error("Failed to launch Chromium. Run: npx playwright install chromium");
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
    await page.waitForFunction(() => window.__excalidrawReady === true, null, {
      timeout: 30000,
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
          viewBackgroundColor: json.appState?.viewBackgroundColor || "#ffffff",
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
              // svg is an SVGSVGElement - serialize it
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

        // Save SVG
        if (format === "svg" || format === "both") {
          const svgDir = path.join(EXPORT_DIR, "svg", subdir);
          fs.mkdirSync(svgDir, { recursive: true });
          const svgPath = path.join(svgDir, `${basename}.svg`);
          fs.writeFileSync(svgPath, svgString.svg);
        }

        // Convert SVG to PNG using resvg
        if (format === "png" || format === "both") {
          const pngDir = path.join(EXPORT_DIR, "png", subdir);
          fs.mkdirSync(pngDir, { recursive: true });
          const pngPath = path.join(pngDir, `${basename}.png`);

          const resvg = new Resvg(svgString.svg, {
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
