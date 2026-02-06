#!/bin/bash
#
# POWE.RS Diagram Tooling Setup
#
# Installs all dependencies needed for the diagram export and conversion
# pipeline: npm packages, esbuild, Playwright Chromium, and system libraries.
#
# Usage:
#   ./scripts/setup-diagrams.sh [--help]
#
# Run from the repository root.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Color support
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
# Help
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
POWE.RS Diagram Tooling Setup

Sets up the complete diagram export and conversion environment:
  1. Verifies Node.js >= 18 is installed
  2. Installs npm dependencies (excalidraw, mermaid-to-excalidraw, esbuild,
     playwright, resvg-js)
  3. Downloads Playwright's Chromium browser
  4. Checks system libraries required by Chromium
  5. Runs a quick smoke test to verify everything works

Usage:
  ./scripts/setup-diagrams.sh [--help]

Options:
  --help, -h   Show this help message and exit

After setup, use:
  npm run diagrams:export     Export .excalidraw files to PNG/SVG
  npm run diagrams:convert    Convert Mermaid blocks to .excalidraw
  npm run diagrams:validate   Validate diagram references
EOF
    exit 0
fi

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "${BOLD}POWE.RS Diagram Tooling Setup${RESET}"
echo "=============================="
echo ""

ERRORS=0

# ---------------------------------------------------------------------------
# Step 1: Check Node.js
# ---------------------------------------------------------------------------
echo "${BOLD}Step 1: Checking Node.js${RESET}"

if ! command -v node &> /dev/null; then
    echo "  ${RED}ERROR: Node.js is not installed${RESET}"
    echo "  Install Node.js 18+ from https://nodejs.org/ or via nvm:"
    echo "    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
    echo "    nvm install 20"
    exit 1
fi

NODE_VERSION=$(node -e "console.log(process.versions.node.split('.')[0])")
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "  ${RED}ERROR: Node.js $NODE_VERSION found, but 18+ is required${RESET}"
    echo "  Current: $(node --version)"
    echo "  Update with: nvm install 20"
    exit 1
fi

echo "  ${GREEN}OK${RESET} Node.js $(node --version)"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Install npm dependencies
# ---------------------------------------------------------------------------
echo "${BOLD}Step 2: Installing npm dependencies${RESET}"

cd "$REPO_ROOT"

# Detect SSL issues and apply workaround if needed
NPM_FLAGS=""
if ! npm ping --registry https://registry.npmjs.org/ &>/dev/null 2>&1; then
    # Test if it's a certificate issue
    if npm ping --registry https://registry.npmjs.org/ 2>&1 | grep -qi "CERT\|SSL\|certificate"; then
        echo "  ${YELLOW}WARNING: SSL certificate issue detected (common in corporate/proxy environments)${RESET}"
        echo "  Temporarily disabling strict SSL for npm install..."
        NPM_FLAGS="--config.strict-ssl=false"
    fi
fi

if npm install $NPM_FLAGS 2>&1 | tail -3; then
    echo "  ${GREEN}OK${RESET} npm dependencies installed"
else
    echo "  ${RED}ERROR: npm install failed${RESET}"
    echo "  If you see SSL/certificate errors, try:"
    echo "    npm config set strict-ssl false"
    echo "    npm install"
    echo "    npm config set strict-ssl true"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Install Playwright Chromium
# ---------------------------------------------------------------------------
echo "${BOLD}Step 3: Installing Playwright Chromium${RESET}"

# Check if Chromium is already installed
CHROMIUM_INSTALLED=false
if npx playwright install --dry-run chromium 2>&1 | grep -q "already installed"; then
    CHROMIUM_INSTALLED=true
fi

if [ "$CHROMIUM_INSTALLED" = false ]; then
    # Playwright downloads may also hit SSL issues
    PW_ENV=""
    if [ -n "$NPM_FLAGS" ]; then
        PW_ENV="NODE_TLS_REJECT_UNAUTHORIZED=0"
    fi

    if eval "$PW_ENV npx playwright install chromium" 2>&1 | tail -3; then
        echo "  ${GREEN}OK${RESET} Chromium downloaded"
    else
        echo "  ${RED}ERROR: Failed to download Chromium${RESET}"
        echo "  Try manually: NODE_TLS_REJECT_UNAUTHORIZED=0 npx playwright install chromium"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ${GREEN}OK${RESET} Chromium already installed"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Check system libraries for Chromium
# ---------------------------------------------------------------------------
echo "${BOLD}Step 4: Checking system libraries${RESET}"

# Find the Chromium binary
CHROMIUM_DIR=$(find "$HOME/.cache/ms-playwright" -name "chrome-headless-shell" -type f 2>/dev/null | head -1)

if [ -n "$CHROMIUM_DIR" ]; then
    MISSING_LIBS=$(ldd "$CHROMIUM_DIR" 2>/dev/null | grep "not found" || true)
    if [ -n "$MISSING_LIBS" ]; then
        echo "  ${RED}ERROR: Missing system libraries:${RESET}"
        echo "$MISSING_LIBS" | while read -r line; do
            LIB=$(echo "$line" | awk '{print $1}')
            echo "    - $LIB"
        done
        echo ""
        echo "  Install them with (requires sudo):"
        echo "    sudo npx playwright install-deps chromium"
        echo ""
        echo "  Or install individually, e.g.:"

        # Map common missing libs to packages
        echo "$MISSING_LIBS" | while read -r line; do
            LIB=$(echo "$line" | awk '{print $1}')
            case "$LIB" in
                libasound.so*)
                    echo "    sudo apt-get install -y libasound2t64  # or libasound2 on older distros"
                    ;;
                libnss3.so*)
                    echo "    sudo apt-get install -y libnss3"
                    ;;
                libatk-1.0.so*|libatk-bridge-2.0.so*)
                    echo "    sudo apt-get install -y libatk1.0-0 libatk-bridge2.0-0"
                    ;;
                libcups.so*)
                    echo "    sudo apt-get install -y libcups2"
                    ;;
                libdrm.so*)
                    echo "    sudo apt-get install -y libdrm2"
                    ;;
                libgbm.so*)
                    echo "    sudo apt-get install -y libgbm1"
                    ;;
                libpango*)
                    echo "    sudo apt-get install -y libpango-1.0-0"
                    ;;
                libxkbcommon.so*)
                    echo "    sudo apt-get install -y libxkbcommon0"
                    ;;
                *)
                    echo "    # $LIB — search: apt-file search $LIB"
                    ;;
            esac
        done
        ERRORS=$((ERRORS + 1))
    else
        echo "  ${GREEN}OK${RESET} All system libraries present"
    fi
else
    echo "  ${YELLOW}SKIP${RESET} Could not locate Chromium binary for library check"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: Smoke test
# ---------------------------------------------------------------------------
echo "${BOLD}Step 5: Smoke test${RESET}"

# Test that Chromium launches
if node -e "
  const { chromium } = require('playwright');
  (async () => {
    const browser = await chromium.launch({ headless: true });
    await browser.close();
    process.exit(0);
  })().catch(() => process.exit(1));
" 2>/dev/null; then
    echo "  ${GREEN}OK${RESET} Playwright Chromium launches successfully"
else
    echo "  ${RED}FAIL${RESET} Chromium failed to launch"
    echo "  Check Step 4 for missing system libraries."
    ERRORS=$((ERRORS + 1))
fi

# Test that esbuild works
if npx esbuild --version &>/dev/null; then
    echo "  ${GREEN}OK${RESET} esbuild $(npx esbuild --version 2>/dev/null)"
else
    echo "  ${RED}FAIL${RESET} esbuild not working"
    ERRORS=$((ERRORS + 1))
fi

# Test that resvg-js loads
if node -e "require('@resvg/resvg-js')" 2>/dev/null; then
    echo "  ${GREEN}OK${RESET} @resvg/resvg-js loads"
else
    echo "  ${RED}FAIL${RESET} @resvg/resvg-js failed to load"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "${BOLD}Setup Complete${RESET}"
echo "=============="

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "  ${RED}$ERRORS error(s) detected.${RESET} Fix the issues above and re-run this script."
    echo ""
    exit 1
fi

echo ""
echo "  ${GREEN}All checks passed.${RESET} The diagram tooling is ready."
echo ""
echo "  Available commands:"
echo "    npm run diagrams:export     Export .excalidraw → PNG/SVG"
echo "    npm run diagrams:convert    Convert Mermaid → .excalidraw"
echo "    npm run diagrams:validate   Validate diagram references"
echo "    npm run diagrams:list       List remaining Mermaid blocks"
echo ""
echo "  Or use Make from docs/:"
echo "    make -C docs diagrams       Export all diagrams"
echo "    make -C docs diagrams-convert   Convert Mermaid diagrams"
echo "    make -C docs help           Show all targets"
echo ""
