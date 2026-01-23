#!/usr/bin/env python3
"""
Convert algorithm pseudocode blocks to use inline LaTeX math for mathematical symbols.

This replaces Greek letters and mathematical symbols with $...$ wrapped versions
that will render correctly in PDF, while keeping the algorithmic structure.
"""

import re
import sys
from pathlib import Path


def wrap_math_symbols(text):
    """Wrap mathematical symbols in $...$ for LaTeX rendering"""

    # Greek letters - wrap each occurrence
    greek_map = {
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "γ": r"$\gamma$",
        "δ": r"$\delta$",
        "ε": r"$\epsilon$",
        "ζ": r"$\zeta$",
        "η": r"$\eta$",
        "θ": r"$\theta$",
        "λ": r"$\lambda$",
        "μ": r"$\mu$",
        "π": r"$\pi$",
        "ρ": r"$\rho$",
        "σ": r"$\sigma$",
        "τ": r"$\tau$",
        "φ": r"$\phi$",
        "ω": r"$\omega$",
        "Γ": r"$\Gamma$",
        "Δ": r"$\Delta$",
        "Θ": r"$\Theta$",
        "Λ": r"$\Lambda$",
        "Π": r"$\Pi$",
        "Σ": r"$\Sigma$",
        "Φ": r"$\Phi$",
        "Ω": r"$\Omega$",
    }

    # Replace Greek letters
    for greek, latex in greek_map.items():
        text = text.replace(greek, latex)

    # Handle variables with subscripts/superscripts - convert to LaTeX
    # Pattern: letter followed by Unicode subscripts/superscripts
    # x₀ -> $x_0$, V̂ₜᵏ -> $\hat{V}_t^k$

    # First handle combining characters (hat, bar, tilde)
    text = re.sub(r"([a-zA-Z])\u0302", r"$\\hat{\1}$", text)  # circumflex
    text = re.sub(r"([a-zA-Z])\u0304", r"$\\bar{\1}$", text)  # macron
    text = re.sub(r"([a-zA-Z])\u0303", r"$\\tilde{\1}$", text)  # tilde

    # Handle subscripts
    subscript_map = {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "ₜ": "t",
        "ₖ": "k",
        "ₕ": "h",
        "ₘ": "m",
        "ₙ": "n",
        "ₗ": "l",
        "ᵢ": "i",
        "ⱼ": "j",
        "ₓ": "x",
        "ₛ": "s",
        "ᵣ": "r",
        "₌": "=",
        "₊": "+",
        "₋": "-",
    }

    # Handle superscripts
    superscript_map = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "ⁿ": "n",
        "ᵏ": "k",
        "ᵐ": "m",
        "ᵀ": "T",
        "⁺": "+",
        "⁻": "-",
    }

    # Pattern to match: variable (letter/number) + subscripts and/or superscripts
    def convert_subscripted(match):
        var = match.group(1)
        rest = match.group(2)

        # Check if already in math mode
        if var.startswith("$"):
            return match.group(0)

        subscripts = []
        superscripts = []

        for char in rest:
            if char in subscript_map:
                subscripts.append(subscript_map[char])
            elif char in superscript_map:
                superscripts.append(superscript_map[char])

        result = var
        if subscripts:
            result += "_{" + "".join(subscripts) + "}"
        if superscripts:
            result += "^{" + "".join(superscripts) + "}"

        return f"${result}$"

    # Match variables with subscripts/superscripts
    pattern = r"([a-zA-Z][a-zA-Z0-9]*)([₀₁₂₃₄₅₆₇₈₉ₜₖₕₘₙₗᵢⱼₓₛᵣ₌₊₋⁰¹²³⁴⁵⁶⁷⁸⁹ⁿᵏᵐᵀ⁺⁻]+)"
    text = re.sub(pattern, convert_subscripted, text)

    # Handle math operators
    operator_map = {
        "≥": r"$\geq$",
        "≤": r"$\leq$",
        "≠": r"$\neq$",
        "∈": r"$\in$",
        "∉": r"$\notin$",
        "∀": r"$\forall$",
        "∃": r"$\exists$",
        "∑": r"$\sum$",
        "∏": r"$\prod$",
        "→": r"$\to$",
        "←": r"$\leftarrow$",
        "⇒": r"$\Rightarrow$",
        "⊂": r"$\subset$",
        "⊃": r"$\supset$",
        "∪": r"$\cup$",
        "∩": r"$\cap$",
        "∅": r"$\emptyset$",
    }

    for op, latex in operator_map.items():
        text = text.replace(op, latex)

    # Simplify consecutive math mode: $x$ $y$ -> $x$ $y$ (keep as is for spacing)
    # But merge when there's no space: $x$$y$ -> $xy$
    text = re.sub(r"\$\s*\$", " ", text)

    return text


def is_algorithm_block(first_line):
    """Check if this looks like an algorithm block"""
    patterns = [
        r"^\s*Algorithm:",
        r"^\s*ALGORITHM",
        r"^\s*Procedure:",
        r"^\s*Function:",
    ]
    for pattern in patterns:
        if re.search(pattern, first_line, re.IGNORECASE):
            return True
    return False


def process_file(input_path, output_path=None, backup=True):
    """Process markdown file, converting algorithm blocks"""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False

    content = input_path.read_text(encoding="utf-8")

    # Create backup
    if backup and not output_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_text(content, encoding="utf-8")
        print(f"✓ Created backup: {backup_path}")

    lines = content.split("\n")
    result_lines = []
    in_code_block = False
    code_block_lines = []
    fence_start = None
    is_algorithm = False

    for line in lines:
        fence_match = re.match(r"^```(\w*)$", line.strip())

        if fence_match:
            if not in_code_block:
                # Starting code block
                in_code_block = True
                fence_start = line
                code_block_lines = []
                is_algorithm = False
            else:
                # Ending code block
                if is_algorithm and code_block_lines:
                    # Process algorithm block
                    block_content = "\n".join(code_block_lines)
                    converted = wrap_math_symbols(block_content)

                    result_lines.append(fence_start)
                    result_lines.extend(converted.split("\n"))
                    result_lines.append(line)
                else:
                    # Regular code block, keep as is
                    result_lines.append(fence_start)
                    result_lines.extend(code_block_lines)
                    result_lines.append(line)

                in_code_block = False
                code_block_lines = []
                fence_start = None
        else:
            if in_code_block:
                code_block_lines.append(line)
                # Check if this is an algorithm block
                if not is_algorithm and len(code_block_lines) <= 3:
                    if is_algorithm_block(line):
                        is_algorithm = True
            else:
                result_lines.append(line)

    result = "\n".join(result_lines)

    # Write output
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = input_path

    output_path.write_text(result, encoding="utf-8")

    # Count changes
    changes = sum(1 for a, b in zip(content.split("\n"), result.split("\n")) if a != b)

    print(f"✓ Processed file: {output_path}")
    print(f"  Modified {changes} lines")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: convert-algorithms-to-math.py <input-file> [output-file]")
        print()
        print("Convert algorithm pseudocode blocks to use LaTeX math mode.")
        print()
        print("This wraps Greek letters and mathematical symbols in $...$ ")
        print("so they render correctly in PDF while preserving readability.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = process_file(input_file, output_file)

    if success and not output_file:
        print()
        print("Next steps:")
        print(f"  1. Review: diff {input_file}.bak {input_file} | less")
        print(f"  2. Generate PDF: make -C docs mathematical")
        print(f"  3. Restore if needed: mv {input_file}.bak {input_file}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
