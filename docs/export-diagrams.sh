#!/bin/bash
#
# POWE.RS Diagram Export Script
#
# Exports all .excalidraw files to PNG and SVG formats.
#
# Requirements:
#   - excalidraw-cli: npm install -g @excalidraw/cli
#   - or use: npx @excalidraw/cli
#
# Usage:
#   ./export-diagrams.sh [--format png|svg|both] [--scale 2]
#

set -e

# Configuration
EXCALIDRAW_DIR="docs/diagrams/excalidraw"
EXPORT_DIR="docs/diagrams/exports"
DEFAULT_FORMAT="both"
DEFAULT_SCALE=2

# Parse arguments
FORMAT=$DEFAULT_FORMAT
SCALE=$DEFAULT_SCALE

while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --help|-h)
            echo "POWE.RS Diagram Export Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --format <png|svg|both>  Output format (default: both)"
            echo "  --scale <number>         Export scale (default: 2)"
            echo "  --help                   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for excalidraw-cli
if ! command -v excalidraw &> /dev/null; then
    echo "excalidraw-cli not found. Using npx..."
    EXCALIDRAW_CMD="npx @excalidraw/cli"
else
    EXCALIDRAW_CMD="excalidraw"
fi

# Create export directories
mkdir -p "$EXPORT_DIR/png"
mkdir -p "$EXPORT_DIR/svg"

echo "POWE.RS Diagram Export"
echo "======================"
echo ""
echo "Source: $EXCALIDRAW_DIR"
echo "Output: $EXPORT_DIR"
echo "Format: $FORMAT"
echo "Scale:  ${SCALE}x"
echo ""

# Count files
FILE_COUNT=$(find "$EXCALIDRAW_DIR" -name "*.excalidraw" 2>/dev/null | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No .excalidraw files found in $EXCALIDRAW_DIR"
    echo ""
    echo "To create diagrams:"
    echo "  1. Open https://excalidraw.com/"
    echo "  2. Create your diagram using EXCALIDRAW_STYLE_GUIDE.md"
    echo "  3. Save as .excalidraw to $EXCALIDRAW_DIR/"
    exit 0
fi

echo "Found $FILE_COUNT .excalidraw files"
echo ""

# Export function
export_diagram() {
    local input="$1"
    local format="$2"
    local filename=$(basename "$input" .excalidraw)
    local output="$EXPORT_DIR/$format/$filename.$format"
    
    echo -n "  Exporting $filename.$format... "
    
    if $EXCALIDRAW_CMD export \
        --format "$format" \
        --scale "$SCALE" \
        --output "$output" \
        "$input" 2>/dev/null; then
        echo "✓"
        return 0
    else
        echo "✗"
        return 1
    fi
}

# Export all files
SUCCESS=0
FAILED=0

for file in "$EXCALIDRAW_DIR"/**/*.excalidraw "$EXCALIDRAW_DIR"/*.excalidraw; do
    # Skip if no match
    [ -f "$file" ] || continue
    
    echo "Processing: $(basename "$file")"
    
    if [ "$FORMAT" = "png" ] || [ "$FORMAT" = "both" ]; then
        if export_diagram "$file" "png"; then
            ((SUCCESS++))
        else
            ((FAILED++))
        fi
    fi
    
    if [ "$FORMAT" = "svg" ] || [ "$FORMAT" = "both" ]; then
        if export_diagram "$file" "svg"; then
            ((SUCCESS++))
        else
            ((FAILED++))
        fi
    fi
    
    echo ""
done

echo "Export Complete"
echo "==============="
echo "  Success: $SUCCESS"
echo "  Failed:  $FAILED"
echo ""
echo "Exported files are in:"
echo "  PNG: $EXPORT_DIR/png/"
echo "  SVG: $EXPORT_DIR/svg/"
echo ""
echo "To embed in Markdown:"
echo '  ![Diagram Name](diagrams/exports/png/diagram-name.png)'
