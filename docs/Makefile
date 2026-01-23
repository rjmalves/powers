# Makefile for converting markdown documentation to PDF using pandoc
#
# Requirements:
#   - pandoc
#   - xelatex (from texlive-xetex package)
#   - Noto fonts (optional, for better Unicode support)
#   - mermaid-filter (optional, for Mermaid diagram support)
#     Install with: npm install -g mermaid-filter
#
# Usage:
#   make                    # Build all PDFs
#   make mathematical       # Build MATHEMATICAL_FORMULATIONS.pdf
#   make datamodel          # Build DATA_MODEL_SPECIFICATION.pdf
#   make clean              # Remove generated PDFs
#   make convert            # Convert markdown to use LaTeX-style math in code blocks

.PHONY: all clean mathematical datamodel convert

# Default target
all: MATHEMATICAL_FORMULATIONS.pdf DATA_MODEL_SPECIFICATION.pdf

# Common pandoc options for all PDFs
PANDOC_OPTS = -s \
	--pdf-engine=xelatex \
	-H pandoc-header.tex \
	-V colorlinks=true \
	-V linkcolor=blue \
	-V urlcolor=blue \
	-V toccolor=blue \
	-V fontsize=10pt \
	--wrap=auto

# Check if mermaid-filter is available
MERMAID_FILTER := $(shell which mermaid-filter 2>/dev/null)
ifdef MERMAID_FILTER
	PANDOC_OPTS += --filter=mermaid-filter
endif

# PDF generation rules
MATHEMATICAL_FORMULATIONS.pdf: MATHEMATICAL_FORMULATIONS.md pandoc-header.tex
	@echo "Generating $@..."
	@pandoc $< $(PANDOC_OPTS) -o $@
	@echo "✓ $@ created successfully (size: $$(du -h $@ | cut -f1))"

DATA_MODEL_SPECIFICATION.pdf: DATA_MODEL_SPECIFICATION.md pandoc-header.tex
	@echo "Generating $@..."
	@pandoc $< $(PANDOC_OPTS) -o $@
	@echo "✓ $@ created successfully (size: $$(du -h $@ | cut -f1))"

# Shortcuts
mathematical: MATHEMATICAL_FORMULATIONS.pdf
datamodel: DATA_MODEL_SPECIFICATION.pdf

# Clean generated files
clean:
	@echo "Removing generated PDFs..."
	@rm -f *.pdf
	@echo "✓ Clean complete"

# Convert markdown files to use LaTeX-style math notation in code blocks
convert:
	@echo "Converting MATHEMATICAL_FORMULATIONS.md to use LaTeX-style math..."
	@./convert-to-latex-style.py MATHEMATICAL_FORMULATIONS.md
	@echo ""
	@echo "Converting DATA_MODEL_SPECIFICATION.md to use LaTeX-style math..."
	@./convert-to-latex-style.py DATA_MODEL_SPECIFICATION.md
	@echo ""
	@echo "✓ Conversion complete. Backup files created with .bak extension"
	@echo ""
	@echo "Review changes with:"
	@echo "  diff MATHEMATICAL_FORMULATIONS.md.bak MATHEMATICAL_FORMULATIONS.md | less"
	@echo ""
	@echo "Restore if needed:"
	@echo "  mv MATHEMATICAL_FORMULATIONS.md.bak MATHEMATICAL_FORMULATIONS.md"

# Help target
help:
	@echo "Available targets:"
	@echo "  make              - Build all PDFs"
	@echo "  make mathematical - Build MATHEMATICAL_FORMULATIONS.pdf"
	@echo "  make datamodel    - Build DATA_MODEL_SPECIFICATION.pdf"
	@echo "  make convert      - Convert markdown to use LaTeX-style math"
	@echo "  make clean        - Remove generated PDFs"
	@echo ""
	@echo "Requirements:"
	@echo "  - pandoc"
	@echo "  - xelatex (texlive-xetex package)"
	@echo "  - Noto fonts (optional, for better Unicode support)"
	@echo "  - mermaid-filter (optional, for Mermaid diagram support)"
	@echo "    Install with: npm install -g mermaid-filter"

