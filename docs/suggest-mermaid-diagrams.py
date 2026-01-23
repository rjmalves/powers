#!/usr/bin/env python3
"""
Helper script to identify ASCII diagrams and suggest Mermaid conversions.

This script scans markdown files for code blocks that look like diagrams
and provides suggestions for converting them to Mermaid format.
"""

import re
import sys
from pathlib import Path


def is_likely_diagram(text):
    """Heuristic to detect if a code block contains a diagram"""
    # Count special characters commonly used in diagrams
    box_chars = len(re.findall(r"[─│┌┐└┘├┤┬┴┼▼▶►◄]", text))
    arrow_chars = len(re.findall(r"[→←⇒⇐↔]|-->|<--|->|<-", text))
    ascii_arrows = len(re.findall(r"[+\-|>v<^]", text))

    # If we have many box-drawing or arrow characters, it's likely a diagram
    total_special = box_chars + arrow_chars
    ascii_ratio = ascii_arrows / max(len(text), 1)

    return total_special > 5 or ascii_ratio > 0.1


def suggest_mermaid_conversion(diagram_text, line_num):
    """Suggest a Mermaid conversion for a diagram"""
    suggestions = []

    # Detect flow patterns
    if re.search(r"(Stage|Step|Phase)\s+\d+", diagram_text, re.IGNORECASE):
        suggestions.append("Consider using Mermaid flowchart:")
        suggestions.append("```mermaid")
        suggestions.append("graph LR")
        suggestions.append("    S1[Stage 1] --> S2[Stage 2]")
        suggestions.append("    S2 --> S3[Stage 3]")
        suggestions.append("```")

    # Detect hierarchical structures
    elif re.search(r"├|└|│", diagram_text):
        suggestions.append("Consider using Mermaid tree/hierarchy:")
        suggestions.append("```mermaid")
        suggestions.append("graph TD")
        suggestions.append("    Root --> Child1")
        suggestions.append("    Root --> Child2")
        suggestions.append("    Child1 --> Leaf1")
        suggestions.append("```")

    # Detect sequence/timeline
    elif re.search(r"(Forward|Backward|Iteration)", diagram_text, re.IGNORECASE):
        suggestions.append("Consider using Mermaid sequence diagram:")
        suggestions.append("```mermaid")
        suggestions.append("sequenceDiagram")
        suggestions.append("    participant A as Forward Pass")
        suggestions.append("    participant B as Backward Pass")
        suggestions.append("    A->>B: Generate states")
        suggestions.append("    B->>A: Return cuts")
        suggestions.append("```")

    # Generic flowchart fallback
    else:
        suggestions.append("Consider using Mermaid flowchart:")
        suggestions.append("```mermaid")
        suggestions.append("graph LR")
        suggestions.append("    A[Start] --> B[Process]")
        suggestions.append("    B --> C[End]")
        suggestions.append("```")

    return suggestions


def scan_file(file_path):
    """Scan a markdown file for diagrams"""
    content = Path(file_path).read_text(encoding="utf-8")
    lines = content.split("\n")

    in_code_block = False
    code_block_start = 0
    code_block_lines = []
    diagram_count = 0

    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_block_start = i
                code_block_lines = []
            else:
                in_code_block = False
                block_text = "\n".join(code_block_lines)

                # Check if this looks like a diagram
                if is_likely_diagram(block_text):
                    diagram_count += 1
                    print(f"\n{'=' * 70}")
                    print(f"Diagram #{diagram_count} at line {code_block_start}")
                    print(f"{'=' * 70}")
                    print("Current ASCII diagram:")
                    print("```")
                    print(block_text[:500])  # Show first 500 chars
                    if len(block_text) > 500:
                        print("... (truncated)")
                    print("```")
                    print()

                    suggestions = suggest_mermaid_conversion(
                        block_text, code_block_start
                    )
                    print("\n".join(suggestions))
                    print()
        else:
            if in_code_block:
                code_block_lines.append(line)

    if diagram_count == 0:
        print("No diagrams detected in the file.")
    else:
        print(f"\n{'=' * 70}")
        print(f"Summary: Found {diagram_count} potential diagram(s)")
        print(f"{'=' * 70}")
        print()
        print("To learn more about Mermaid syntax:")
        print("  https://mermaid.js.org/intro/")
        print()
        print("To install mermaid-filter for Pandoc:")
        print("  npm install -g mermaid-filter")


def main():
    if len(sys.argv) < 2:
        print("Usage: suggest-mermaid-diagrams.py <markdown-file>")
        print()
        print("Scans a markdown file for ASCII diagrams and suggests")
        print("Mermaid conversions for better PDF rendering.")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    scan_file(file_path)


if __name__ == "__main__":
    main()
