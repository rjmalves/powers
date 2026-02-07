# Makefile for POWE.RS documentation diagrams
#
# Requirements:
#   - Node.js 18+
#   - npm dependencies: npm install (from repo root)
#   - Playwright Chromium: npx playwright install chromium
#   - Run ./scripts/setup-diagrams.sh for full setup
#
# Usage:
#   make diagrams          # Export all Excalidraw diagrams to PNG/SVG
#   make diagrams-clean    # Remove exported diagram files
#   make diagrams-validate # Validate diagram references and sources
#   make help              # Show all targets

.PHONY: all diagrams diagrams-export diagrams-validate diagrams-clean help

# Default target
all: diagrams

# Export all Excalidraw diagrams to PNG/SVG
diagrams: diagrams-export

diagrams-export:
	@node ../scripts/export-diagrams.mjs --format both --scale 2

# Remove exported diagram files
diagrams-clean:
	@echo "Removing exported diagrams..."
	@rm -rf diagrams/exports/png/* diagrams/exports/svg/*
	@echo "Clean complete"

# Validate diagram references and source files
diagrams-validate:
	@../scripts/validate-diagrams.sh

# Help target
help:
	@echo "POWE.RS Documentation Diagram Targets"
	@echo ""
	@echo "  make diagrams          - Export all Excalidraw diagrams to PNG/SVG"
	@echo "  make diagrams-clean    - Remove exported diagram files"
	@echo "  make diagrams-validate - Validate diagram references and sources"
	@echo "  make help              - Show this help"
	@echo ""
	@echo "Setup:    ./scripts/setup-diagrams.sh"
	@echo "Style guide: docs/diagrams/STYLE_GUIDE.md"
