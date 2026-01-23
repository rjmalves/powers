#!/bin/bash
# Convert problematic Unicode to PDF-friendly alternatives in markdown files
# This creates a backup and modifies the file in place

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <markdown-file>"
    echo "Example: $0 docs/MATHEMATICAL_FORMULATIONS.md"
    echo ""
    echo "This script will:"
    echo "  1. Create a backup (.bak)"
    echo "  2. Replace Unicode subscripts/superscripts in code blocks with ASCII"
    echo "  3. Replace box-drawing with ASCII art"
    echo "  4. Report changes made"
    exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

# Create backup
BACKUP="${FILE}.bak"
cp "$FILE" "$BACKUP"
echo "Created backup: $BACKUP"

# Create temporary file
TEMP=$(mktemp)

# Process the file
awk '
BEGIN {
    in_code_block = 0
    changes = 0
}

# Detect code block boundaries
/^```/ {
    if (in_code_block == 0) {
        in_code_block = 1
    } else {
        in_code_block = 0
    }
    print
    next
}

# Process lines inside code blocks
in_code_block == 1 {
    original = $0
    
    # Replace Unicode subscripts with _X notation
    gsub(/₀/, "_0")
    gsub(/₁/, "_1")
    gsub(/₂/, "_2")
    gsub(/₃/, "_3")
    gsub(/₄/, "_4")
    gsub(/₅/, "_5")
    gsub(/₆/, "_6")
    gsub(/₇/, "_7")
    gsub(/₈/, "_8")
    gsub(/₉/, "_9")
    gsub(/ₜ/, "_t")
    gsub(/ₖ/, "_k")
    gsub(/ₕ/, "_h")
    gsub(/ₘ/, "_m")
    gsub(/ₙ/, "_n")
    gsub(/ₗ/, "_l")
    gsub(/ᵢ/, "_i")
    gsub(/ⱼ/, "_j")
    gsub(/ₓ/, "_x")
    gsub(/ₛ/, "_s")
    gsub(/ᵣ/, "_r")
    
    # Replace Unicode superscripts with ^X notation
    gsub(/⁰/, "^0")
    gsub(/¹/, "^1")
    gsub(/²/, "^2")
    gsub(/³/, "^3")
    gsub(/⁴/, "^4")
    gsub(/⁵/, "^5")
    gsub(/⁶/, "^6")
    gsub(/⁷/, "^7")
    gsub(/⁸/, "^8")
    gsub(/⁹/, "^9")
    gsub(/ᵏ/, "^k")
    gsub(/ᵐ/, "^m")
    gsub(/ⁿ/, "^n")
    gsub(/ᵀ/, "^T")
    
    # Replace common Greek letters with names
    gsub(/α/, "alpha")
    gsub(/β/, "beta")
    gsub(/γ/, "gamma")
    gsub(/δ/, "delta")
    gsub(/ε/, "epsilon")
    gsub(/ζ/, "zeta")
    gsub(/η/, "eta")
    gsub(/θ/, "theta")
    gsub(/λ/, "lambda")
    gsub(/μ/, "mu")
    gsub(/π/, "pi")
    gsub(/ρ/, "rho")
    gsub(/σ/, "sigma")
    gsub(/τ/, "tau")
    gsub(/φ/, "phi")
    gsub(/ω/, "omega")
    
    # Replace special Greek
    gsub(/ᾱ/, "alpha_bar")
    gsub(/β̄/, "beta_bar")
    
    # Replace math symbols with ASCII equivalents
    gsub(/≥/, ">=")
    gsub(/≤/, "<=")
    gsub(/≠/, "!=")
    gsub(/×/, "*")
    gsub(/÷/, "/")
    gsub(/∈/, "in")
    gsub(/∉/, "not in")
    gsub(/∀/, "for all")
    gsub(/∃/, "exists")
    gsub(/∑/, "sum")
    gsub(/∏/, "product")
    gsub(/Σ/, "Sum")
    
    # Replace combining characters
    gsub(/x̂/, "x_hat")
    gsub(/V̂/, "V_hat")
    gsub(/θ̂/, "theta_hat")
    gsub(/ω̂/, "omega_hat")
    
    # Replace box-drawing with ASCII
    gsub(/─/, "-")
    gsub(/│/, "|")
    gsub(/┌/, "+")
    gsub(/┐/, "+")
    gsub(/└/, "+")
    gsub(/┘/, "+")
    gsub(/├/, "+")
    gsub(/┤/, "+")
    gsub(/┬/, "+")
    gsub(/┴/, "+")
    gsub(/┼/, "+")
    gsub(/▼/, "v")
    gsub(/▶/, ">")
    gsub(/►/, ">")
    gsub(/◄/, "<")
    
    # Arrow replacements
    gsub(/→/, "->")
    gsub(/←/, "<-")
    gsub(/⇒/, "=>")
    
    if (original != $0) {
        changes++
    }
}

# Process lines outside code blocks
in_code_block == 0 {
    original = $0
    
    # Only replace box-drawing outside code blocks
    gsub(/─/, "-")
    gsub(/│/, "|")
    gsub(/┌/, "+")
    gsub(/┐/, "+")
    gsub(/└/, "+")
    gsub(/┘/, "+")
    gsub(/├/, "+")
    gsub(/┤/, "+")
    gsub(/┬/, "+")
    gsub(/┴/, "+")
    gsub(/┼/, "+")
    gsub(/▼/, "v")
    gsub(/▶/, ">")
    gsub(/►/, ">")
    gsub(/◄/, "<")
    
    # Arrow replacements (common in diagrams)
    gsub(/──►/, "-->")
    gsub(/──/, "--")
    
    if (original != $0) {
        changes++
    }
}

{
    print
}

END {
    print changes > "/tmp/awk_changes_count"
}
' "$FILE" > "$TEMP"

# Read the changes count
CHANGES=$(cat /tmp/awk_changes_count 2>/dev/null || echo "0")
rm -f /tmp/awk_changes_count

# Replace original file
mv "$TEMP" "$FILE"

echo ""
echo "✓ Processing complete"
echo "  Changes made: $CHANGES lines"
echo "  Original file: $BACKUP"
echo "  Updated file: $FILE"
echo ""
echo "Review the changes with:"
echo "  diff $BACKUP $FILE | less"
echo ""
echo "If satisfied, remove backup:"
echo "  rm $BACKUP"
echo ""
echo "If not satisfied, restore:"
echo "  mv $BACKUP $FILE"
