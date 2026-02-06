#!/bin/bash
#
# POWE.RS Diagram Validation Script
#
# Checks consistency between image references in documentation
# and .excalidraw source files.
#
# Usage:
#   ./scripts/validate-diagrams.sh [--help]
#
# Exit codes:
#   0  All image references have matching .excalidraw sources
#   1  One or more image references have no matching source
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (script lives in scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DOCS_DIR="$REPO_ROOT/docs"
EXCALIDRAW_DIR="$DOCS_DIR/diagrams/excalidraw"

# ---------------------------------------------------------------------------
# Color support — degrade gracefully when stdout is not a terminal
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    GREEN=$'\033[0;32m'
    RED=$'\033[0;31m'
    YELLOW=$'\033[0;33m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    GREEN=""
    RED=""
    YELLOW=""
    BOLD=""
    RESET=""
fi

# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
POWE.RS Diagram Validation Script

Checks consistency between image references in documentation and
.excalidraw source files.

Checks performed:
   1. Every ![...](diagrams/exports/svg/...) reference in docs/*.md has a
      matching .excalidraw source file under docs/diagrams/excalidraw/.
   2. Every .excalidraw file (excluding components/) is referenced by at
      least one ![...] tag in docs/*.md.
   3. Counts remaining ```mermaid blocks in the docs (informational only).

Exit codes:
  0  All image references have matching .excalidraw sources (Check 1 passes)
  1  One or more image references have no matching source (Check 1 fails)

Usage:
  ./scripts/validate-diagrams.sh [--help]

Options:
  --help, -h   Show this help message and exit
EOF
    exit 0
fi

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "${BOLD}POWE.RS Diagram Validation${RESET}"
echo "==========================="
echo ""

# ===================================================================
# Check 1: Image references → .excalidraw sources
# ===================================================================
echo "${BOLD}Check 1: Image references → .excalidraw sources${RESET}"

check1_ok=0
check1_missing=0
check1_total=0

# Collect unique svg references from docs/*.md (skip diagrams/README.md etc.)
# Pattern: ![...](diagrams/exports/svg/<subdir>/<name>.svg)
refs=()
while IFS= read -r ref; do
    refs+=("$ref")
done < <(grep -rohP 'diagrams/exports/svg/\K[^)]+\.svg' "$DOCS_DIR"/*.md 2>/dev/null | sort -u)

for svg_ref in "${refs[@]}"; do
    check1_total=$((check1_total + 1))

    # Derive expected excalidraw path: sddp/policy-graph-finite.svg → sddp/policy-graph-finite.excalidraw
    base_name="${svg_ref%.svg}"
    excalidraw_path="$EXCALIDRAW_DIR/${base_name}.excalidraw"

    if [ -f "$excalidraw_path" ]; then
        echo "  ${GREEN}✓${RESET} ${svg_ref} → $(basename "$excalidraw_path")"
        check1_ok=$((check1_ok + 1))
    else
        echo "  ${RED}✗${RESET} ${svg_ref} → ${RED}NO SOURCE FOUND${RESET}"
        check1_missing=$((check1_missing + 1))
    fi
done

if [ "$check1_total" -eq 0 ]; then
    echo "  (no image references found)"
fi

echo "  ${BOLD}Result: ${check1_ok}/${check1_total} OK, ${check1_missing} MISSING${RESET}"
echo ""

# ===================================================================
# Check 2: .excalidraw files → doc references
# ===================================================================
echo "${BOLD}Check 2: .excalidraw files → doc references${RESET}"

check2_ok=0
check2_unreferenced=0
check2_total=0

while IFS= read -r excalidraw_file; do
    check2_total=$((check2_total + 1))

    # Derive relative path: docs/diagrams/excalidraw/sddp/foo.excalidraw → sddp/foo
    rel_path="${excalidraw_file#"$EXCALIDRAW_DIR/"}"
    base_name="${rel_path%.excalidraw}"
    svg_pattern="diagrams/exports/svg/${base_name}.svg"
    display_name="${rel_path}"

    # Search for the svg reference across docs/*.md
    match_file=""
    match_file="$(grep -rl "$svg_pattern" "$DOCS_DIR"/*.md 2>/dev/null | head -n1 || true)"

    if [ -n "$match_file" ]; then
        echo "  ${GREEN}✓${RESET} ${display_name} → referenced in $(basename "$match_file")"
        check2_ok=$((check2_ok + 1))
    else
        echo "  ${YELLOW}✗${RESET} ${display_name} → ${YELLOW}NOT REFERENCED${RESET}"
        check2_unreferenced=$((check2_unreferenced + 1))
    fi
done < <(find "$EXCALIDRAW_DIR" -name "*.excalidraw" -not -path "*/components/*" 2>/dev/null | sort)

if [ "$check2_total" -eq 0 ]; then
    echo "  (no .excalidraw files found)"
fi

echo "  ${BOLD}Result: ${check2_ok}/${check2_total} OK, ${check2_unreferenced} UNREFERENCED (warning only)${RESET}"
echo ""

# ===================================================================
# Check 3: Remaining Mermaid blocks
# ===================================================================
echo "${BOLD}Check 3: Remaining Mermaid blocks${RESET}"

mermaid_total=0

while IFS= read -r md_file; do
    filename="$(basename "$md_file")"

    # Collect line numbers of ```mermaid occurrences
    line_numbers=()
    while IFS= read -r lineno; do
        line_numbers+=("$lineno")
    done < <(grep -n '```mermaid' "$md_file" 2>/dev/null | grep -oP '^\d+' || true)

    count=${#line_numbers[@]}
    if [ "$count" -gt 0 ]; then
        lines_str=""
        for i in "${!line_numbers[@]}"; do
            if [ "$i" -gt 0 ]; then
                lines_str+=", "
            fi
            lines_str+="${line_numbers[$i]}"
        done
        if [ "$count" -eq 1 ]; then
            echo "  ${filename}: ${count} block (line ${lines_str})"
        else
            echo "  ${filename}: ${count} blocks (lines ${lines_str})"
        fi
        mermaid_total=$((mermaid_total + count))
    fi
done < <(find "$DOCS_DIR" -maxdepth 1 -name "*.md" 2>/dev/null | sort)

if [ "$mermaid_total" -eq 0 ]; then
    echo "  (none found)"
else
    echo "  ${BOLD}Total: ${mermaid_total} Mermaid blocks remaining (Tier 3 scope)${RESET}"
fi
echo ""

# ===================================================================
# Summary
# ===================================================================
echo "${BOLD}Summary${RESET}"
echo "======="

# Check 1 result
if [ "$check1_missing" -gt 0 ]; then
    echo "  References: ${check1_ok}/${check1_total} valid ← ${RED}FAIL${RESET}"
else
    echo "  References: ${check1_ok}/${check1_total} valid ← ${GREEN}PASS${RESET}"
fi

# Check 2 result
if [ "$check2_unreferenced" -gt 0 ]; then
    echo "  Sources: ${check2_ok}/${check2_total} referenced (${check2_unreferenced} unreferenced) ← ${YELLOW}WARNING${RESET}"
else
    echo "  Sources: ${check2_ok}/${check2_total} referenced ← ${GREEN}OK${RESET}"
fi

# Check 3 result
if [ "$mermaid_total" -gt 0 ]; then
    echo "  Mermaid: ${mermaid_total} remaining ← INFO"
else
    echo "  Mermaid: 0 remaining ← INFO"
fi

# Exit with failure if any image references are missing sources
if [ "$check1_missing" -gt 0 ]; then
    exit 1
fi

exit 0
