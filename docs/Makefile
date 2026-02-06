# Makefile for POWE.RS documentation diagrams
#
# Requirements:
#   - Node.js (for Mermaid conversion)
#   - excalidraw-cli: npm install -g @excalidraw/cli (or use npx)
#   - npm dependencies: npm install (from repo root)
#
# Usage:
#   make diagrams          # Export all Excalidraw diagrams to PNG/SVG
#   make diagrams-list     # List Mermaid code blocks in documentation
#   make diagrams-convert  # Convert Mermaid diagrams to Excalidraw format
#   make diagrams-clean    # Remove exported diagram files
#   make diagrams-validate # Validate diagram references and sources
#   make help              # Show all targets

.PHONY: all diagrams diagrams-export diagrams-convert diagrams-list diagrams-validate diagrams-clean help

# Default target
all: diagrams

# Export all Excalidraw diagrams to PNG/SVG
diagrams: diagrams-export

diagrams-export:
	@../scripts/export-diagrams.sh --format both --scale 2

# Convert existing Mermaid diagrams to Excalidraw format
diagrams-convert:
	@echo "Converting Mermaid diagrams to Excalidraw format..."
	@node ../scripts/convert-mermaid.mjs MATHEMATICAL_FORMULATIONS.md --output-dir diagrams/excalidraw/sddp
	@node ../scripts/convert-mermaid.mjs DATA_MODEL_SPECIFICATION.md --output-dir diagrams/excalidraw/data
	@node ../scripts/convert-mermaid.mjs PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md --output-dir diagrams/excalidraw/hpc
	@echo "Conversion complete. Review and refine in Excalidraw."

# List Mermaid code blocks in documentation
diagrams-list:
	@echo "Mermaid code blocks in documentation:"
	@echo ""
	@grep -rn '```mermaid' *.md | head -50

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
	@echo "  make diagrams-convert  - Convert Mermaid diagrams to Excalidraw format"
	@echo "  make diagrams-list     - List Mermaid code blocks in documentation"
	@echo "  make diagrams-clean    - Remove exported diagram files"
	@echo "  make diagrams-validate - Validate diagram references and sources"
	@echo "  make help              - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - Node.js"
	@echo "  - npm install (from repo root, for @excalidraw dependencies)"
	@echo "  - excalidraw-cli (npm install -g @excalidraw/cli)"
	@echo ""
	@echo "Style guide: docs/diagrams/STYLE_GUIDE.md"
	@echo "Migration plan: docs/DIAGRAM_MIGRATION_PLAN.md"
