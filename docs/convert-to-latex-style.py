#!/usr/bin/env python3
"""
Convert markdown code blocks to use LaTeX math notation for better PDF rendering.

This script converts Unicode mathematical symbols in code blocks to LaTeX equivalents
while preserving the code block structure. It handles:
- Greek letters: α → \\alpha
- Subscripts: x₀ → x_0
- Superscripts: x² → x^2
- Math operators: ≥ → \\geq
- Box-drawing characters → ASCII
"""

import re
import sys
from pathlib import Path


def process_code_block(text):
    """
    Convert Unicode math symbols in code block to simpler ASCII/LaTeX-friendly versions.

    Strategy: Instead of wrapping everything in $...$ which breaks code blocks,
    we'll just convert to plain ASCII that renders well in monospace fonts.
    """

    # Subscript conversions (Unicode → ASCII underscore notation)
    subscripts = {
        "₀": "_0",
        "₁": "_1",
        "₂": "_2",
        "₃": "_3",
        "₄": "_4",
        "₅": "_5",
        "₆": "_6",
        "₇": "_7",
        "₈": "_8",
        "₉": "_9",
        "ₜ": "_t",
        "ₖ": "_k",
        "ₕ": "_h",
        "ₘ": "_m",
        "ₙ": "_n",
        "ₗ": "_l",
        "ᵢ": "_i",
        "ⱼ": "_j",
        "ₓ": "_x",
        "ₛ": "_s",
        "ᵣ": "_r",
        "₊": "_+",
        "₋": "_-",
        "₌": "_=",
    }

    # Superscript conversions (Unicode → ASCII caret notation)
    superscripts = {
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "ᵏ": "^k",
        "ᵐ": "^m",
        "ⁿ": "^n",
        "ᵀ": "^T",
        "⁺": "^+",
        "⁻": "^-",
    }

    # Greek letter conversions (Unicode → name)
    greek = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "ι": "iota",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "τ": "tau",
        "υ": "upsilon",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
        "Γ": "Gamma",
        "Δ": "Delta",
        "Θ": "Theta",
        "Λ": "Lambda",
        "Ξ": "Xi",
        "Π": "Pi",
        "Σ": "Sigma",
        "Φ": "Phi",
        "Ψ": "Psi",
        "Ω": "Omega",
    }

    # Math operators (Unicode → ASCII/LaTeX operator)
    operators = {
        "≥": ">=",
        "≤": "<=",
        "≠": "!=",
        "≈": "~=",
        "×": "*",
        "÷": "/",
        "·": "*",
        "∈": "in",
        "∉": "not in",
        "⊂": "subset",
        "⊃": "superset",
        "∀": "forall",
        "∃": "exists",
        "∄": "not exists",
        "∧": "and",
        "∨": "or",
        "¬": "not",
        "∑": "sum",
        "∏": "product",
        "∫": "integral",
        "∞": "inf",
        "∅": "empty",
        "→": "->",
        "←": "<-",
        "⇒": "=>",
        "⇐": "<=",
        "⇔": "<=>",
        "↔": "<->",
        "↑": "^",
        "↓": "v",
        "√": "sqrt",
        "∛": "cbrt",
        "∩": "intersect",
        "∪": "union",
    }

    # Box-drawing characters → simple ASCII
    box_drawing = {
        "─": "-",
        "│": "|",
        "┌": "+",
        "┐": "+",
        "└": "+",
        "┘": "+",
        "├": "+",
        "┤": "+",
        "┬": "+",
        "┴": "+",
        "┼": "+",
        "═": "=",
        "║": "|",
        "╔": "+",
        "╗": "+",
        "╚": "+",
        "╝": "+",
        "╠": "+",
        "╣": "+",
        "╦": "+",
        "╩": "+",
        "╬": "+",
        "▼": "v",
        "▶": ">",
        "►": ">",
        "◄": "<",
        "▲": "^",
        "◆": "*",
        "●": "o",
        "○": "o",
    }

    # Combining characters (e.g., x̂ → x_hat, x̄ → x_bar)
    # These need special handling as they're two characters
    text = re.sub(r"([a-zA-Z0-9]+)\u0302", r"\1_hat", text)  # circumflex (hat)
    text = re.sub(r"([a-zA-Z0-9]+)\u0304", r"\1_bar", text)  # macron (bar)
    text = re.sub(r"([a-zA-Z0-9]+)\u0303", r"\1_tilde", text)  # tilde
    text = re.sub(r"([a-zA-Z0-9]+)\u0307", r"\1_dot", text)  # dot above
    text = re.sub(r"([a-zA-Z0-9]+)\u0308", r"\1_ddot", text)  # diaeresis (double dot)

    # Handle specific pre-composed characters with combining marks
    text = text.replace("θ̂", "theta_hat")
    text = text.replace("ω̂", "omega_hat")
    text = text.replace("x̂", "x_hat")
    text = text.replace("V̂", "V_hat")
    text = text.replace("ᾱ", "alpha_bar")
    text = text.replace("β̄", "beta_bar")

    # Apply all character replacements
    for old, new in {
        **subscripts,
        **superscripts,
        **greek,
        **operators,
        **box_drawing,
    }.items():
        text = text.replace(old, new)

    return text


def convert_file(input_path, output_path=None, create_backup=True):
    """Convert a markdown file's code blocks"""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False

    # Read the file
    content = input_path.read_text(encoding="utf-8")

    # Create backup if requested
    if create_backup and not output_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_text(content, encoding="utf-8")
        print(f"✓ Created backup: {backup_path}")

    # Process the content
    lines = content.split("\n")
    result_lines = []
    in_code_block = False
    code_block_lines = []
    code_fence = None

    for line in lines:
        # Check for code fence (with or without language identifier)
        fence_match = re.match(r"^```(\w*)$", line.strip())

        if fence_match:
            if not in_code_block:
                # Starting code block
                in_code_block = True
                code_fence = line
                code_block_lines = []
            else:
                # Ending code block
                # Process the accumulated code block content
                block_content = "\n".join(code_block_lines)
                converted_content = process_code_block(block_content)

                # Add to result with fence markers
                result_lines.append(code_fence)
                result_lines.extend(converted_content.split("\n"))
                result_lines.append(line)

                in_code_block = False
                code_block_lines = []
                code_fence = None
        else:
            if in_code_block:
                code_block_lines.append(line)
            else:
                result_lines.append(line)

    # Handle unclosed code block (shouldn't happen, but be safe)
    if in_code_block:
        result_lines.append(code_fence)
        result_lines.extend(code_block_lines)

    # Join the result
    result = "\n".join(result_lines)

    # Write the output
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = input_path

    output_path.write_text(result, encoding="utf-8")

    # Count changes
    changes = sum(1 for a, b in zip(content.split("\n"), result.split("\n")) if a != b)

    print(f"✓ Converted file written to: {output_path}")
    print(f"  Modified {changes} lines")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: convert-to-latex-style.py <input-file> [output-file]")
        print()
        print("Convert markdown code blocks to use ASCII-friendly notation.")
        print()
        print("Examples:")
        print("  ./convert-to-latex-style.py doc.md          # Overwrite with backup")
        print("  ./convert-to-latex-style.py doc.md new.md   # Write to new file")
        print()
        print("Conversions:")
        print("  - Greek letters: α → alpha, β → beta, etc.")
        print("  - Subscripts: x₀ → x_0, xₜ → x_t")
        print("  - Superscripts: x² → x^2, xⁿ → x^n")
        print("  - Operators: ≥ → >=, ≤ → <=, ∈ → in")
        print("  - Box drawing: ─│┌ → -|+")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_file(input_file, output_file)

    if success and not output_file:
        print()
        print("Next steps:")
        print(f"  1. Review changes: diff {input_file}.bak {input_file} | less")
        print(f"  2. Generate PDF: make -C docs mathematical")
        print(f"  3. Restore if needed: mv {input_file}.bak {input_file}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
