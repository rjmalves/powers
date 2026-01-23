#!/bin/bash
# Check markdown files for problematic Unicode characters that won't render well in PDF

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <markdown-file>"
    echo "Example: $0 docs/MATHEMATICAL_FORMULATIONS.md"
    exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

echo "=== Checking $FILE for PDF rendering issues ==="
echo

# Check for Unicode in code blocks
echo "1. Unicode characters in code blocks (between triple backticks):"
IN_CODE=0
LINE_NUM=0
FOUND=0

while IFS= read -r line; do
    LINE_NUM=$((LINE_NUM + 1))
    
    if [[ "$line" =~ ^\`\`\` ]]; then
        if [ $IN_CODE -eq 0 ]; then
            IN_CODE=1
            CODE_START=$LINE_NUM
        else
            IN_CODE=0
        fi
    elif [ $IN_CODE -eq 1 ]; then
        # Check for problematic Unicode
        if echo "$line" | grep -qE '[α-ωΑ-Ω₀-₉⁰-⁹ᵃ-ᶻᴬ-ᶻ─│┌┐└┘├┤┬┴┼▼▶►◄≥≤≠∈∉∀∃∑∏√∞≈±×÷]'; then
            echo "  Line $LINE_NUM: $line"
            FOUND=1
        fi
    fi
done < "$FILE"

if [ $FOUND -eq 0 ]; then
    echo "  ✓ No Unicode found in code blocks"
fi
echo

# Check for box-drawing characters anywhere
echo "2. Box-drawing characters (use ASCII instead):"
grep -n '[─│┌┐└┘├┤┬┴┼╭╮╯╰═║╔╗╚╝╠╣╦╩╬▲▼◄►◀▶]' "$FILE" || echo "  ✓ No box-drawing characters"
echo

# Check for wide tables (>100 chars)
echo "3. Tables wider than 100 characters:"
awk '/^\|/ {if (length > 100) print NR ": " substr($0, 1, 100) "..."}' "$FILE" | head -10
if ! awk '/^\|/ {if (length > 100) print}' "$FILE" | grep -q .; then
    echo "  ✓ No excessively wide tables"
fi
echo

# Check for tables with >5 columns
echo "4. Tables with more than 5 columns:"
awk -F'|' '/^\|/ && NF > 6 {print NR ": " NF-1 " columns"}' "$FILE" | head -10
if ! awk -F'|' '/^\|/ && NF > 6' "$FILE" | grep -q .; then
    echo "  ✓ No tables with >5 columns"
fi
echo

# Check for Unicode math in prose (outside $...$)
echo "5. Unicode Greek letters in prose (consider using LaTeX):"
# This is tricky - we want to find Greek outside of code blocks and math mode
grep -n '[α-ωΑ-Ω]' "$FILE" | grep -v '\$' | grep -v '```' | head -10 || echo "  ✓ No Greek letters outside math mode"
echo

echo "=== Summary ==="
echo "Review the items above. For best PDF rendering:"
echo "  - Replace Unicode in code blocks with ASCII equivalents"
echo "  - Use ASCII art instead of box-drawing characters"
echo "  - Split or abbreviate wide tables"
echo "  - Use \$\\alpha\$ instead of 'α' in prose"
echo
echo "See MARKDOWN_BEST_PRACTICES.md for detailed guidance."
