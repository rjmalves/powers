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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

// Configuration
const EXCALIDRAW_DIR = path.join(REPO_ROOT, "docs/diagrams/excalidraw");
const EXPORT_DIR = path.join(REPO_ROOT, "docs/diagrams/exports");
const BUNDLE_DIR = path.join(REPO_ROOT, "node_modules/.cache/powers-diagrams");

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

        // Save SVG
        if (format === "svg" || format === "both") {
          const svgDir = path.join(EXPORT_DIR, "svg", subdir);
          fs.mkdirSync(svgDir, { recursive: true });
          const svgPath = path.join(svgDir, `${basename}.svg`);
          fs.writeFileSync(svgPath, svgWithFonts);
        }

        // Convert SVG to PNG using resvg
        if (format === "png" || format === "both") {
          const pngDir = path.join(EXPORT_DIR, "png", subdir);
          fs.mkdirSync(pngDir, { recursive: true });
          const pngPath = path.join(pngDir, `${basename}.png`);

          const resvg = new Resvg(svgWithFonts, {
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
