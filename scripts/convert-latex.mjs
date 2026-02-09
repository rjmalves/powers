#!/usr/bin/env node
/**
 * POWE.RS — Batch LaTeX conversion for Excalidraw diagrams
 *
 * Replaces Unicode math text in .excalidraw files with LaTeX-delimited
 * content ($$...$$ or $...$) that the export pipeline renders via MathJax.
 *
 * Usage:
 *   node scripts/convert-latex.mjs [--dry-run]
 *
 * With --dry-run, prints what would change without modifying files.
 */

import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const EXCALIDRAW_DIR = path.join(REPO_ROOT, "docs/diagrams/excalidraw");

const dryRun = process.argv.includes("--dry-run");

// ---------------------------------------------------------------------------
// Conversion mappings: { [relativePath]: { [elementId]: newText } }
//
// The newText is the literal string that goes into the JSON "text" field.
// Use $$...$$ for display math, $...$ for inline math.
// Multi-line LaTeX uses \n between lines (Excalidraw multi-line text).
// For multi-line math: use \begin{gathered}...\end{gathered} inside $$.
// ---------------------------------------------------------------------------

const CONVERSIONS = {

  // =========================================================================
  // BATCH 1: SDDP high-density files
  // =========================================================================

  "sddp/cut-generation-mechanics.excalidraw": {
    // --- Section 1: Input ---
    "sec1-input-text": "$$\\text{Trial Point}\\\\\\hat{x}_{t-1}$$",
    "sec1-scenario-text": "$$\\text{Scenario }\\omega\\\\\\text{(noise)}$$",

    // --- Section 2: LP Solve ---
    "sec2-lp-obj": "$$\\min \\, c_t^\\top x_t + \\theta_t$$",
    "sec2-lp-constraints": "$$\\begin{gathered}\\text{s.t.}\\\\\\text{constraints}(x_t, \\hat{x}_{t-1}, \\omega)\\\\+ \\text{existing cuts on }\\theta_t\\end{gathered}$$",
    "sec2-output-value-text": "$$Q_t(\\hat{x}, \\omega) = \\text{opt val}$$",

    // --- Arrow labels ---
    "arrow-lp-to-duals-label": "$$\\pi_t(\\omega) = \\text{duals}$$",
    "arrow-value-to-coeff-label": "$$Q_t \\text{ feeds } \\alpha$$",

    // --- Section 3: Dual Extraction ---
    "sec3-dual-wb-text": "$$\\begin{gathered}\\pi^{\\text{wb}}_h\\\\\\text{water balance dual}\\end{gathered}$$",
    "sec3-dual-lag-text": "$$\\begin{gathered}\\pi^{\\text{lag}}_{h,l}\\\\\\text{AR lag dual}\\end{gathered}$$",

    // --- Section 4: Cut Coefficients ---
    "sec4-beta-v-text": "$$\\begin{gathered}\\beta^v_h = \\pi^{\\text{wb}}_h\\\\\\text{(storage coefficients)}\\end{gathered}$$",
    "sec4-beta-lag-text": "$$\\begin{gathered}\\beta^{\\text{lag}}_{h,l} = \\pi^{\\text{lag}}_{h,l}\\\\\\text{(lag coefficients)}\\end{gathered}$$",
    "sec4-alpha-text": "$$\\begin{gathered}\\alpha(\\omega) = Q_t - \\beta^\\top \\cdot \\hat{x}_{t-1}\\\\\\text{(intercept: cut passes}\\\\\\text{through trial point)}\\end{gathered}$$",

    // --- Section 5: Aggregation ---
    "sec5-agg-alpha-text": "$$\\bar{\\alpha} = \\sum p(\\omega) \\cdot \\alpha(\\omega)$$",
    "sec5-agg-beta-text": "$$\\bar{\\beta} = \\sum p(\\omega) \\cdot \\beta(\\omega)$$",
    "sec5-agg-prob-text": "$$\\begin{gathered}\\text{typically 20 scenarios}\\\\p(\\omega) = 1/20\\end{gathered}$$",
    "sec5-loop-label": "$$\\begin{gathered}\\text{repeat for all }\\omega \\in \\Omega_t\\\\\\text{(solve LP once per scenario)}\\end{gathered}$$",
    "sec5-agg-note": "$$\\begin{gathered}\\text{expectation over }\\Omega_t\\\\\\text{risk-neutral aggregation}\\\\\\text{(one cut per trial point)}\\end{gathered}$$",

    // --- Section 6: Output ---
    "sec6-output-cut-text": "$$\\theta_{t-1} \\geq \\bar{\\alpha} + \\bar{\\beta}^\\top \\cdot x_{t-1}$$",
    "sec6-validity-note": "$$\\begin{gathered}\\text{Cut is valid:}\\\\\\bar{\\alpha} + \\bar{\\beta}^\\top x \\leq V_t(x) \\;\\; \\forall x\\\\\\text{(lower bound on cost-to-go)}\\end{gathered}$$",

    // --- Legend: sign convention (complex multi-line mixed text) ---
    "legend-sign-text": "$$\\begin{gathered}\\pi^{\\text{wb}}_h < 0 \\;\\text{ means more water = lower cost}\\\\\\text{(water has value, displaces thermal)}\\\\[4pt]\\beta^v_h < 0 \\;\\text{ so the cut }\\theta \\geq \\alpha + \\beta^v \\cdot v\\\\\\text{correctly penalizes low storage}\\\\\\text{(future is more expensive with less water)}\\\\[4pt]\\text{Incoming state on RHS with coefficient }+1\\\\\\Rightarrow \\beta = \\pi \\text{ directly}\\\\\\text{(no sign flip, no transpose needed)}\\end{gathered}$$",

    // --- Detail: constraint forms ---
    "detail-constraint-wb": "$$\\begin{gathered}\\text{Water balance: } v_h - \\zeta \\cdot (\\text{flows}) = \\hat{v}_h\\\\\\rightarrow \\text{dual }\\pi^{\\text{wb}}_h = \\partial Q^*/\\partial \\hat{v}_h \\;\\; (\\$/\\text{hm}^3)\\end{gathered}$$",
    "detail-constraint-lag": "$$\\begin{gathered}\\text{AR lag fixing: } a_{h,l} = \\hat{a}_{h,l}\\\\\\rightarrow \\text{dual }\\pi^{\\text{lag}}_{h,l} = \\partial Q^*/\\partial \\hat{a}_{h,l} \\;\\; (\\$/(\\text{m}^3/\\text{s}))\\end{gathered}$$",

    // --- Detail: intercept formula ---
    "detail-intercept-formula": "$$\\begin{gathered}\\alpha_t = Q_t(\\hat{x}_{t-1}, \\omega_t)\\\\\\quad - \\sum_h \\beta^v_{t,h} \\cdot \\hat{v}_h\\\\\\quad - \\sum_{h,l} \\beta^{\\text{lag}}_{t,h,l} \\cdot \\hat{a}_{h,l}\\\\[6pt]\\text{Ensures cut is tight at the trial point:}\\\\\\alpha + \\beta^\\top \\cdot \\hat{x} = Q_t(\\hat{x}, \\omega) \\text{ exactly}\\end{gathered}$$",
  },

  "sddp/value-function-approximation.excalidraw": {
    // --- Axis labels ---
    "x-axis-label": "$$x \\;\\text{(state variable)}$$",
    "y-axis-label": "$$V_t(x) \\;\\text{(value function)}$$",

    // --- Trial point labels ---
    "trial-label-1": "$$\\hat{x}_1$$",
    "trial-label-2": "$$\\hat{x}_2$$",
    "trial-label-3": "$$\\hat{x}_3$$",
    "xhat1-axis-label": "$$\\hat{x}_1$$",
    "xhat2-axis-label": "$$\\hat{x}_2$$",
    "xhat3-axis-label": "$$\\hat{x}_3$$",

    // --- Gap ---
    "gap-label": "$$\\begin{gathered}\\text{GAP}\\\\V_t(x) - \\underline{V}_t(x)\\\\\\to 0 \\text{ as } k \\to \\infty\\end{gathered}$$",

    // --- Cut labels ---
    "cut-label-k1": "$$\\text{Cut }k\\!=\\!1{:}\\; \\theta \\geq \\alpha^1 + \\beta^1 \\cdot x$$",
    "cut-label-k2": "$$\\text{Cut }k\\!=\\!2{:}\\; \\theta \\geq \\alpha^2 + \\beta^2 \\cdot x$$",
    "cut-label-k3": "$$\\text{Cut }k\\!=\\!3{:}\\; \\theta \\geq \\alpha^3 + \\beta^3 \\cdot x$$",

    // --- Curve labels ---
    "true-vf-label": "$$\\text{True }V_t(x)$$",
    "lower-approx-label": "$$\\underline{V}_t(x) = \\max\\{\\text{cuts}\\}$$",

    // --- Legend ---
    "legend-green-text": "$$\\text{True }V_t(x)\\text{ (convex, unknown)}$$",
    "legend-red-text": "$$\\text{Benders cuts }\\theta \\geq \\alpha^k + \\beta^k \\cdot x$$",
    "legend-blue-text": "$$\\underline{V}_t(x) = \\max\\{\\text{cuts}\\}\\text{ (lower approx.)}$$",
    "legend-orange-text": "$$\\text{Trial points }\\hat{x}_t\\text{ from forward pass}$$",
    "legend-gap-text": "$$\\text{Gap} \\to 0 \\text{ as } k \\to \\infty$$",

    // --- Equations block ---
    "equations-text": "$$\\begin{gathered}\\text{Cut equation: }\\theta \\geq \\alpha^k + {\\beta^k}^\\top x\\\\[4pt]\\text{where }\\alpha^k = Q_t - \\pi_t^\\top \\hat{x}_{t-1}\\\\\\text{and }\\beta^k = \\pi_t \\;\\text{(dual multiplier)}\\\\[4pt]\\underline{V}_t(x) = \\max_k \\{ \\alpha^k + {\\beta^k}^\\top x \\}\\end{gathered}$$",
  },

  "sddp/sddp-iteration.excalidraw": {
    // --- Forward pass ---
    "fp-upper-bound-text": "$$\\begin{gathered}\\text{compute upper bound:}\\\\\\bar{z}^k = (1/M) \\sum_m \\sum_t c_t^\\top x_t^m\\end{gathered}$$",
    "fp-record-note": "$$\\begin{gathered}\\text{record visited states: }\\hat{x}_1^m, \\hat{x}_2^m, \\ldots, \\hat{x}_T^m\\\\\\text{for each scenario }m = 1, \\ldots, M\\end{gathered}$$",
    "fp-solve-annotation": "$$\\begin{gathered}\\text{solve: }\\min c_t^\\top x_t + \\theta_t\\\\\\text{s.t. }A_t x_t = b_t - E_t x_{t-1},\\; x_t \\in X_t\\end{gathered}$$",

    // --- Backward pass ---
    "bp-lower-bound-text": "$$\\begin{gathered}\\text{compute lower bound:}\\\\\\underline{z}^k = V_1^k(x_0) = c_1^\\top \\hat{x}_1 + \\hat{\\theta}_1\\end{gathered}$$",
    "bp-cut-gen-text": "$$\\begin{gathered}\\text{for each }\\omega{:}\\text{ solve LP, extract }\\pi_t^*(\\omega)\\\\\\beta(\\omega) = W_t^\\top \\pi_t^*(\\omega)\\\\\\alpha(\\omega) = Q_t(\\hat{x}, \\omega) - \\beta(\\omega)^\\top \\hat{x}_{t-1}\\\\\\text{add cut: }\\theta_{t-1} \\geq \\bar{\\alpha} + \\bar{\\beta}^\\top x_{t-1}\\end{gathered}$$",
    "bp-aggregation-note": "$$\\begin{gathered}\\text{single-cut aggregation:}\\\\\\bar{\\beta} = \\sum_\\omega p(\\omega) \\cdot \\beta(\\omega) \\quad \\bar{\\alpha} = \\sum_\\omega p(\\omega) \\cdot \\alpha(\\omega)\\\\\\text{add to stage }t\\!-\\!1\\text{ problem:}\\\\\\theta_{t-1} \\geq \\bar{\\alpha} + \\bar{\\beta}^\\top x_{t-1}\\\\\\text{(one cut per forward pass, per stage)}\\end{gathered}$$",

    // --- Convergence ---
    "conv-diamond-text": "$$\\text{③ gap} < \\varepsilon \\text{ ?}$$",
    "conv-gap-formula": "$$\\text{gap} = \\frac{\\bar{z}^k - \\underline{z}^k}{\\max(1, |\\bar{z}^k|)}$$",
    "stop-annotation": "$$\\text{return }\\underline{V}_1(x_0)\\text{ as policy}$$",
    "z-lower-annotation": "$$\\begin{gathered}\\underline{z}^k \\text{ monotone }\\uparrow\\\\\\text{converges to }z^*\\\\\\text{from below}\\end{gathered}$$",

    // --- Loop ---
    "loop-label": "$$k \\leftarrow k+1$$",

    // --- Equations block ---
    "eq-text": "$$\\begin{gathered}\\text{Lower bound: }\\underline{z}^k = V_1^k(x_0) \\quad \\text{(monotone }\\uparrow\\text{, deterministic)}\\\\\\text{Upper bound: }\\bar{z}^k = (1/M) \\sum_m \\sum_t c_t^\\top x_t^m \\quad \\text{(statistical estimate)}\\\\\\text{Gap: gap}^k = (\\bar{z}^k - \\underline{z}^k) / \\max(1, |\\bar{z}^k|) < \\varepsilon\\\\\\text{Stop when: gap}^k < \\varepsilon \\text{ or } k > k_{\\max} \\text{ or time} > t_{\\max}\\end{gathered}$$",

    // --- Legend (mixed symbols) ---
    "legend-text": "$$\\begin{gathered}\\circ \\;= \\text{ stage LP solve (one per stage per scenario)}\\\\\\to \\;= \\text{ forward simulation (sample }\\omega_t\\text{, solve, advance)}\\\\\\leftarrow \\;= \\text{ backward cut generation (extract duals, add cut)}\\\\\\diamond \\;= \\text{ convergence check (gap test)}\\\\\\dashrightarrow \\;= \\text{ next iteration loop (dashed red, }k \\leftarrow k+1\\text{)}\\end{gathered}$$",
  },

  "sddp/scenario-tree-branching.excalidraw": {
    // --- Tree annotation ---
    "tree-stage-label": "$$\\begin{gathered}\\text{Stage }t\\\\\\text{state }\\hat{x}_{t-1}\\end{gathered}$$",
    "tree-annotation": "$$|\\Omega_t| = 20 \\;\\text{(NEWAVE branching factor)}$$",

    // --- PAR noise ---
    "par-eq1": "$$\\epsilon_t \\sim \\mathcal{N}(0, 1)$$",
    "par-eq2": "$$\\begin{gathered}a_h = \\mu_m + \\sum \\psi \\cdot (a_{\\text{lag}} - \\mu_{\\text{lag}})\\\\\\quad + \\sigma_m \\cdot \\epsilon\\end{gathered}$$",

    // --- Cut formula ---
    "cut-formula": "$$\\bar{\\beta} = \\sum_\\omega p(\\omega) \\cdot \\pi_t(\\omega) \\qquad \\bar{\\alpha} = \\sum_\\omega p(\\omega) \\cdot (Q_t - \\beta(\\omega)^\\top \\hat{x}) \\qquad \\Rightarrow \\; \\theta_{t-1} \\geq \\bar{\\alpha} + \\bar{\\beta}^\\top x_{t-1}$$",

    // --- Comparison ---
    "compare-ratio": "$$\\text{Ratio: 20\\times more LP solves}$$",

    // --- Forward annotations ---
    "fwd-annot2": "$$M \\text{ passes} \\times T \\text{ stages} = M \\cdot T \\text{ LP solves}$$",

    // --- Backward annotations ---
    "bwd-annot2": "$$M \\times T \\times 20 = \\text{total LP solves per iteration}$$",
  },

  // =========================================================================
  // BATCH 2: SDDP medium files
  // =========================================================================

  "sddp/policy-graph-finite.excalidraw": {
    // --- Terminal node ---
    "terminal-label": "$$\\begin{gathered}\\text{Terminal}\\\\V_{T+1} = 0\\\\\\text{no future}\\end{gathered}$$",

    // --- Legend (mixed symbols) ---
    "legend-text": "$$\\begin{gathered}\\bullet \\text{ Blue boxes = SDDP stages (decision nodes)}\\\\\\bullet \\text{ Yellow box = Final stage (}t = T\\text{)}\\\\\\bullet \\text{ Dashed gray = Terminal node }V_{T+1}(x) = 0\\\\\\bullet \\text{ Cyan ellipses = Stochastic realizations }\\omega_t\\\\\\bullet \\text{ Arrows = Stage transitions with probabilities}\\end{gathered}$$",

    // --- Bellman equation ---
    "bellman-eq": "$$\\begin{gathered}V_t(x_{t-1}) = \\mathbb{E}_{\\omega_t}\\!\\left[\\min\\{c_t^\\top x_t + V_{t+1}(x_t)\\}\\right]\\\\[4pt]\\text{where }x_0 = \\bar{x}_0 \\text{ (given initial state)}\\\\\\text{and }V_{T+1}(x) = 0 \\text{ (terminal condition)}\\end{gathered}$$",
  },

  "sddp/policy-graph-cyclic.excalidraw": {
    // --- Cycle label ---
    "cycle-label": "$$\\text{cycle with discount }\\beta < 1$$",

    // --- Legend ---
    "legend-text": "$$\\begin{gathered}\\bullet \\text{ Blue boxes = Monthly stages (12 months/year)}\\\\\\bullet \\text{ Red box = Cycle endpoint (December)}\\\\\\bullet \\text{ Dashed red arrow = Cycle back with discount}\\\\\\bullet \\;\\beta < 1 \\text{ ensures convergence (typically }\\beta \\approx 0.95\\text{)}\\\\\\bullet \\text{ No terminal node: infinite planning horizon}\\\\\\bullet \\text{ Seasonality captured through monthly resolution}\\end{gathered}$$",

    // --- Cyclic Bellman equation ---
    "math-eq": "$$\\begin{gathered}V_{12}(x) = \\mathbb{E}\\!\\left[\\min\\{c_{12}^\\top x_{12} + \\beta \\cdot V_1(x_{12})\\}\\right]\\\\[4pt]\\text{where:}\\\\\\beta \\in (0, 1) = \\text{discount factor}\\\\V_1 \\text{ appears on RHS of }V_{12} \\text{ (cycle!)}\\\\\\text{Solved via fixed-point iteration}\\end{gathered}$$",
  },

  "sddp/system-element-overview.excalidraw": {
    // --- Bus balance equation ---
    "bus-balance-eq": "$$\\begin{gathered}\\sum_i \\rho_i u_i + \\sum_j g_j + \\sum_\\ell f_\\ell + \\delta_s = d_s\\\\\\text{(Power balance at each bus)}\\end{gathered}$$",

    // --- Key constraints block ---
    "reservoir-balance-eq": "$$\\begin{gathered}\\text{Reservoir balance:}\\\\v_{i,t} = v_{i,t-1} + a_{i,t}(\\omega) - u_{i,t} - s_{i,t} + \\sum_k (u_k + s_k)\\\\[4pt]\\text{Power balance at bus }s{:}\\\\\\sum_i \\rho_i u_{i,t} + \\sum_j g_{j,t} + \\sum_\\ell f_{\\ell,t} + \\delta_{s,t} = d_{s,t}(\\omega)\\\\[4pt]\\text{Objective:}\\\\\\min \\sum_j c_j g_{j,t} + \\sum_s c_s^{\\text{def}} \\delta_{s,t} + \\theta_t\\end{gathered}$$",
  },

  // =========================================================================
  // BATCH 3: Data files
  // =========================================================================

  "data/state-variables.excalidraw": {
    "formula_text": "$$N_{\\text{state}} = N_{\\text{hydro}} + \\sum P_h + N_{\\text{battery}} + \\sum L_{\\text{gnl}}$$",
    "impact_text2": "$$\\text{Cut vector }\\beta \\in \\mathbb{R}^{2080}$$",
    "impact_text5": "$$\\begin{gathered}\\text{Total (120 stages): }\\sim\\!20\\text{ GB}\\\\\\theta \\geq \\alpha + \\beta^\\top x \\;\\text{ for every cut}\\end{gathered}$$",
    "card_arlags_body": "$$\\begin{gathered}\\text{Past inflow noise for AR(P) model}\\\\\\epsilon_{t-1}, \\epsilon_{t-2}, \\ldots, \\epsilon_{t-P}\\\\\\text{Determines inflow distribution}\\\\\\text{Count: }\\sum P_h = 160 \\times 12 = 1{,}920\\\\\\text{Dominates state dimension!}\\end{gathered}$$",
  },

  "data/lp-sizing.excalidraw": {
    // These use fontFamily 3 (code font) — convert to LaTeX for consistency
    "formula-nvar-text": "$$\\begin{gathered}N_{\\text{VAR}} = 1 + N_{\\text{bus}} \\cdot N_{\\text{blk}} \\cdot 4 + N_{\\text{link}} \\cdot N_{\\text{blk}} \\cdot 2\\\\\\quad + N_{\\text{hyd}} \\cdot (1 + P_h + N_{\\text{blk}} \\cdot 5 + \\text{slack}) + N_{\\text{thm}} \\cdot N_{\\text{blk}} + \\cdots\\end{gathered}$$",
    "formula-ncon-text": "$$\\begin{gathered}N_{\\text{CON}} = N_{\\text{bus}} \\cdot N_{\\text{blk}} + N_{\\text{hyd}} \\cdot (1 + 1 + P_h + N_{\\text{blk}} \\cdot 6 + \\text{FPHA})\\\\\\quad + N_{\\text{generic}} + N_{\\text{cuts\\_alloc}}\\end{gathered}$$",
    "formula-nstate-text": "$$N_{\\text{STATE}} = N_{\\text{hydro}} + \\sum P_h + N_{\\text{battery}} + \\sum L_{\\text{gnl}} \\qquad \\text{(State dimension determines cut coefficient vector width)}$$",
  },

  "data/cut-storage-layout.excalidraw": {
    // No complex math — mostly code/struct. Only the cut formula label has math:
    // cv-cut-label-formula is in communication-volume, not here.
    // The annotation about β is plain struct text — leave as-is.
  },

  "data/output-streaming-pipeline.excalidraw": {
    // Convergence formula
    "lane1_conv_detail": "$$\\begin{gathered}\\text{LB, UB, gap, stable\\_count}\\\\\\text{gap} = (\\text{UB} - \\text{LB}) / |\\text{UB}|\\end{gathered}$$",
  },

  "data/6-4-hierarchical-cut-aggregation.excalidraw": {
    "annotation-text": "$$\\begin{gathered}\\text{fanout} = 4\\\\\\log_4(16) = 2 \\text{ levels}\\end{gathered}$$",
  },

  // =========================================================================
  // BATCH 4: HPC files
  // =========================================================================

  "hpc/forward-pass-distribution.excalidraw": {
    // The ω subscript labels are already fine with Unicode and render well.
    // Only the formula text has real math:
    "hpc02-formula-text": "$$\\text{base} = \\text{total} / \\text{world\\_size} = 200/8 = 25 \\qquad \\text{remainder} = \\text{total} \\bmod \\text{world\\_size} = 200 \\bmod 8 = 0$$",
  },

  "hpc/communication-volume.excalidraw": {
    // Cut formula label
    "cv-cut-label-formula": "$$\\theta \\geq \\alpha + \\beta^\\top x \\quad \\text{(160 hydros} \\times \\text{12 AR + 1 affine = 2{,}080 states)}$$",
    // Compute:communication ratio
    "cv-timing-ratio": "$$\\text{Compute:Communication ratio} = 18{:}1$$",
    // Key insight text
    "cv-tgt-key-text": "$$\\begin{gathered}\\text{Key insight: 388 MB / iteration over 25 GB/s link = 15.5 ms raw}\\\\\\text{With 119-stage pipelining: communication fully hidden behind LP solves}\\end{gathered}$$",
  },

  "hpc/hierarchical-aggregation.excalidraw": {
    "ha-flat-latency-label": "$$\\text{Latency: }O(N)\\text{ — linear in rank count}$$",
    "ha-hier-stat1": "$$\\log_4(16) = 2 \\text{ levels}$$",
    "ha-hier-stat2": "$$\\text{Latency: }O(\\log N)\\text{ — logarithmic}$$",
    "ha-table-r3-flat": "$$O(N)\\text{ — linear}$$",
    "ha-table-r3-hier": "$$O(\\log N)\\text{ — logarithmic}$$",
  },

  "hpc/scaling-efficiency.excalidraw": {
    "note-hierarchy-text": "$$\\begin{gathered}\\text{Hierarchical aggregation}\\\\\\text{avoids }O(N^2)\\text{ scaling of}\\\\\\text{flat MPI\\_Allgatherv}\\end{gathered}$$",
  },
};

// ---------------------------------------------------------------------------
// Conversion engine
// ---------------------------------------------------------------------------

function applyConversions() {
  let totalFiles = 0;
  let totalElements = 0;
  let totalErrors = 0;

  for (const [relPath, elementMap] of Object.entries(CONVERSIONS)) {
    const entries = Object.entries(elementMap);
    if (entries.length === 0) continue;

    const filePath = path.join(EXCALIDRAW_DIR, relPath);
    if (!fs.existsSync(filePath)) {
      console.error(`  ERROR: File not found: ${relPath}`);
      totalErrors++;
      continue;
    }

    const raw = fs.readFileSync(filePath, "utf-8");
    const json = JSON.parse(raw);
    const elements = json.elements || [];

    // Build id → element index map
    const idMap = new Map();
    elements.forEach((el, idx) => {
      if (el.id) idMap.set(el.id, idx);
    });

    let fileChanges = 0;

    for (const [elementId, newText] of entries) {
      const idx = idMap.get(elementId);
      if (idx === undefined) {
        console.error(`  ERROR: Element "${elementId}" not found in ${relPath}`);
        totalErrors++;
        continue;
      }

      const el = elements[idx];
      if (el.type !== "text") {
        console.error(`  ERROR: Element "${elementId}" in ${relPath} is type "${el.type}", expected "text"`);
        totalErrors++;
        continue;
      }

      const oldText = el.text;
      if (oldText === newText) {
        // Already converted
        continue;
      }

      if (dryRun) {
        console.log(`  [DRY RUN] ${relPath} :: ${elementId}`);
        console.log(`    OLD: ${JSON.stringify(oldText).substring(0, 80)}`);
        console.log(`    NEW: ${JSON.stringify(newText).substring(0, 80)}`);
      }

      el.text = newText;
      fileChanges++;
      totalElements++;
    }

    if (fileChanges > 0) {
      totalFiles++;
      if (!dryRun) {
        // Write back with 2-space indentation (matching Excalidraw format)
        const output = JSON.stringify(json, null, 2) + "\n";
        fs.writeFileSync(filePath, output);
      }
      console.log(`  ${dryRun ? "[DRY RUN] " : ""}${relPath}: ${fileChanges} element(s) updated`);
    }
  }

  console.log("");
  console.log("Summary");
  console.log("=======");
  console.log(`  Files modified: ${totalFiles}`);
  console.log(`  Elements updated: ${totalElements}`);
  if (totalErrors > 0) {
    console.log(`  Errors: ${totalErrors}`);
  }
  if (dryRun) {
    console.log("  (dry run — no files were modified)");
  }

  return totalErrors === 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("POWE.RS LaTeX Conversion for Excalidraw Diagrams");
console.log("=================================================");
console.log("");

if (dryRun) {
  console.log("DRY RUN MODE — no files will be modified\n");
}

const ok = applyConversions();
process.exit(ok ? 0 : 1);
