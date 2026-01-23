# Improved PDF Conversion Workflow

## Overview

This document describes the new, improved approach to converting markdown documentation to PDF format. The key improvements are:

1. **Cleaner code blocks**: Greek letters and math symbols use simple ASCII names (alpha, beta, theta) instead of verbose replacements
2. **Better readability**: Monospace code blocks remain readable with clear variable names
3. **Diagram support**: Optional Mermaid diagram integration for professional-looking flowcharts
4. **Automated conversion**: Scripts handle the tedious Unicode → ASCII conversions

## The Problem We Solved

The original approach had several issues:

- **Greek letters in code blocks** (α, β, θ) were being replaced with verbose names that broke the code aesthetic
- **Subscripts and superscripts** (x₀, x²) weren't rendering properly in monospace fonts
- **ASCII diagrams** with box-drawing characters (─│┌) didn't convert well to PDF
- **Manual conversion** was tedious and error-prone

## The Solution

### 1. Smart Unicode Conversion

The `convert-to-latex-style.py` script converts Unicode to clean ASCII:

**Before (in markdown):**
```
x₀, ω ∈ Ωₜ, θ̂ ≥ α + β'x
```

**After (converted for PDF):**
```
x_0, omega in Omega_t, theta_hat >= alpha + beta'x
```

### Key Conversions:
- Greek letters: `α → alpha`, `β → beta`, `θ → theta`, `ω → omega`
- Subscripts: `x₀ → x_0`, `ωₜ → omega_t`
- Superscripts: `x² → x^2`, `xⁿ → x^n`
- Combining chars: `x̂ → x_hat`, `x̄ → x_bar`
- Operators: `≥ → >=`, `≤ → <=`, `∈ → in`, `∀ → forall`
- Box-drawing: `─│┌ → -|+`

### 2. Diagram Support (Optional)

For even better results, you can convert ASCII diagrams to Mermaid format:

**Before (ASCII):**
```
     ┌──────────────┐
     │              │
     ▼              │
Stage 1 ──► Stage 2 ─┘
```

**After (Mermaid):**
```mermaid
graph LR
    S1[Stage 1] --> S2[Stage 2]
    S2 -.-> S1
```

The `suggest-mermaid-diagrams.py` script identifies diagrams and suggests Mermaid conversions.

## Quick Start

### Step 1: Convert Your Markdown

```bash
cd docs
./convert-to-latex-style.py MATHEMATICAL_FORMULATIONS.md
```

This creates:
- Backup: `MATHEMATICAL_FORMULATIONS.md.bak`
- Converted: `MATHEMATICAL_FORMULATIONS.md` (updated in place)

### Step 2: Generate PDF

```bash
make mathematical
```

Or build all PDFs:

```bash
make all
```

### Step 3: Review Results

View the PDF:
```bash
evince MATHEMATICAL_FORMULATIONS.pdf &
```

Compare changes:
```bash
diff MATHEMATICAL_FORMULATIONS.md.bak MATHEMATICAL_FORMULATIONS.md | less
```

### Step 4: Restore if Needed

If you're not happy with the conversion:
```bash
mv MATHEMATICAL_FORMULATIONS.md.bak MATHEMATICAL_FORMULATIONS.md
```

## Advanced Usage

### Convert All Documentation

```bash
make convert  # Converts both MATHEMATICAL_FORMULATIONS.md and DATA_MODEL_SPECIFICATION.md
```

### Identify Diagrams for Mermaid Conversion

```bash
./suggest-mermaid-diagrams.py MATHEMATICAL_FORMULATIONS.md
```

This will scan your document and suggest Mermaid conversions for ASCII diagrams.

### Install Mermaid Filter (Optional)

For the best diagram rendering:

```bash
npm install -g mermaid-filter
```

The Makefile will automatically detect and use mermaid-filter if available.

## Tools Reference

### 1. convert-to-latex-style.py

**Purpose**: Convert Unicode math symbols in code blocks to ASCII equivalents

**Usage**:
```bash
./convert-to-latex-style.py <input-file> [output-file]
```

**Examples**:
```bash
# Overwrite with backup
./convert-to-latex-style.py MATHEMATICAL_FORMULATIONS.md

# Write to new file
./convert-to-latex-style.py MATHEMATICAL_FORMULATIONS.md converted.md
```

**Features**:
- Automatic backup creation (.bak extension)
- Processes only code blocks (preserves markdown prose)
- Shows modification count
- Handles all common mathematical Unicode characters

### 2. suggest-mermaid-diagrams.py

**Purpose**: Identify ASCII diagrams and suggest Mermaid conversions

**Usage**:
```bash
./suggest-mermaid-diagrams.py <markdown-file>
```

**Example**:
```bash
./suggest-mermaid-diagrams.py MATHEMATICAL_FORMULATIONS.md
```

**Output**:
- Lists all detected diagrams with line numbers
- Provides Mermaid syntax suggestions
- Includes links to Mermaid documentation

### 3. Makefile Targets

**Available commands**:
```bash
make                # Build all PDFs
make mathematical   # Build MATHEMATICAL_FORMULATIONS.pdf
make datamodel      # Build DATA_MODEL_SPECIFICATION.pdf
make convert        # Convert markdown files to ASCII-friendly notation
make clean          # Remove generated PDFs
make help           # Show available targets
```

## Comparison: Before vs. After

### Code Block Example

**Before conversion:**
```
For each ω ∈ Ωₜ:
    β(ω) = -πₜ(ω)
    α(ω) = Qₜ - β(ω)'x̂ₜ₋₁
```

**After conversion:**
```
For each omega in Omega_t:
    beta(omega) = -pi_t(omega)
    alpha(omega) = Q_t - beta(omega)'x_hat_t_-_1
```

**Benefits**:
- ✅ Monospace font renders correctly
- ✅ Clear, readable variable names
- ✅ Matches notation in prose sections
- ✅ Works perfectly in PDF

### Diagram Example

**Before conversion:**
```
     ┌──────────────────┐
     │                  │
     ▼                  │
Stage 1 ──► Stage 2 ──►─┘
```

**After conversion (ASCII):**
```
     +------------------+
     |                  |
     v                  |
Stage 1 --> Stage 2 -->-+
```

**After conversion (Mermaid):**
```mermaid
graph LR
    S1[Stage 1] --> S2[Stage 2]
    S2 -.-> S1
```

## Workflow Recommendations

### For Quick PDF Generation

If you just need a PDF quickly:

```bash
cd docs
make mathematical
```

The current configuration already works well!

### For Best Quality PDFs

If you want the highest quality PDFs with clean code blocks:

```bash
cd docs

# 1. Convert markdown
./convert-to-latex-style.py MATHEMATICAL_FORMULATIONS.md

# 2. (Optional) Review conversion
diff MATHEMATICAL_FORMULATIONS.md.bak MATHEMATICAL_FORMULATIONS.md | less

# 3. Generate PDF
make mathematical

# 4. View result
evince MATHEMATICAL_FORMULATIONS.pdf &
```

### For Production Documentation

If you're preparing final documentation:

```bash
cd docs

# 1. Convert markdown to ASCII-friendly notation
./convert-to-latex-style.py MATHEMATICAL_FORMULATIONS.md

# 2. Identify diagrams for manual Mermaid conversion
./suggest-mermaid-diagrams.py MATHEMATICAL_FORMULATIONS.md

# 3. Manually convert 2-3 key diagrams to Mermaid
#    (follow suggestions from step 2)

# 4. Install mermaid-filter if not already installed
npm install -g mermaid-filter

# 5. Generate final PDF
make clean
make all

# 6. Review PDFs
evince MATHEMATICAL_FORMULATIONS.pdf &
evince DATA_MODEL_SPECIFICATION.pdf &
```

## Configuration Files

### pandoc-header.tex

Contains LaTeX configuration for PDF generation:
- Unicode character support
- Table formatting (longtable, booktabs)
- Narrow margins for wide content
- Better spacing

### Makefile

Build system for PDF generation:
- Pandoc configuration
- XeLaTeX engine
- Automatic mermaid-filter detection
- Convenient targets

## Troubleshooting

### Problem: PDF generation fails

**Check**:
```bash
which pandoc
which xelatex
```

**Install** (Fedora/RHEL):
```bash
sudo dnf install pandoc texlive-xetex texlive-collection-fontsrecommended
```

### Problem: Mermaid diagrams not rendering

**Check**:
```bash
which mermaid-filter
```

**Install**:
```bash
npm install -g mermaid-filter
```

### Problem: Conversion changed too much

**Restore**:
```bash
mv MATHEMATICAL_FORMULATIONS.md.bak MATHEMATICAL_FORMULATIONS.md
```

**Try converting to a new file first**:
```bash
./convert-to-latex-style.py MATHEMATICAL_FORMULATIONS.md test.md
diff MATHEMATICAL_FORMULATIONS.md test.md | less
```

### Problem: Some symbols still don't render

Check the `convert-to-latex-style.py` script and add your symbol to the appropriate dictionary:
- `subscripts`: for subscript characters
- `superscripts`: for superscript characters
- `greek`: for Greek letters
- `operators`: for math operators
- `box_drawing`: for box-drawing characters

## Best Practices

### DO:
- ✅ Keep Greek letters and math in LaTeX math mode (`$\alpha$`) in prose
- ✅ Use the conversion script for code blocks
- ✅ Review the diff before committing changes
- ✅ Keep backup files until you verify the PDF looks good
- ✅ Use Mermaid for complex diagrams

### DON'T:
- ❌ Manually edit hundreds of Unicode characters
- ❌ Use Unicode in code blocks (it breaks monospace rendering)
- ❌ Delete backup files immediately
- ❌ Skip the review step
- ❌ Try to make ASCII diagrams pixel-perfect (use Mermaid instead)

## Future Improvements

Possible enhancements:
- Automatic Mermaid diagram conversion
- Support for more diagram types (sequence, Gantt, etc.)
- Integration with CI/CD for automatic PDF generation
- Custom LaTeX templates for branded PDFs
- Table width optimization

## Getting Help

- Pandoc manual: https://pandoc.org/MANUAL.html
- Mermaid documentation: https://mermaid.js.org/
- LaTeX symbols: https://katex.org/docs/supported.html
- XeLaTeX guide: https://www.overleaf.com/learn/latex/XeLaTeX

## Summary

The new workflow provides:

1. **Better code blocks**: Clean, readable ASCII notation
2. **Professional diagrams**: Optional Mermaid integration
3. **Automated conversion**: Scripts handle the tedious work
4. **Flexible workflow**: Choose the level of quality you need
5. **Easy maintenance**: Simple tools, clear documentation

The conversion has already been run on `MATHEMATICAL_FORMULATIONS.md`, and the PDF has been successfully generated!
