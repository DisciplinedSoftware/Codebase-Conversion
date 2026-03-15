#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_auth.sh — Interactive authentication setup for the Fortran-to-Rust
# conversion pipeline.
#
# Run automatically by the devcontainer postCreateCommand, or manually at any
# time with:  bash scripts/setup_auth.sh
#
# When the Python package is installed, delegates to the Python auth_setup
# module so that the shell and conversion pipeline share identical logic.
#
# Two supported auth modes:
#   1. GitHub CLI  — `gh auth login` (recommended; no key to store/rotate)
#   2. .env file   — set LLM_API_KEY in a .env file in the project root
# ---------------------------------------------------------------------------

set -euo pipefail

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say()  { echo -e "${CYAN}[setup]${NC} $*"; }
ok()   { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
err()  { echo -e "${RED}[setup]${NC} $*"; }

hr() { echo -e "${CYAN}──────────────────────────────────────────────────────${NC}"; }

# ---------------------------------------------------------------------------
# Delegate to Python when the package is importable
# ---------------------------------------------------------------------------
# The Python auth_setup module is the canonical implementation; the shell
# script is used as a fallback for early devcontainer bootstrap (before pip
# install completes) or when Python is unavailable.
if python3 -c "from fortran_to_rust.auth_setup import prompt_auth_setup" 2>/dev/null; then
    python3 - <<'PYEOF'
from rich.console import Console
from fortran_to_rust.auth_setup import prompt_auth_setup
from fortran_to_rust.llm_client import LLMClient

console = Console()
console.rule("[bold cyan]Fortran-to-Rust — Authentication Setup[/bold cyan]", style="cyan")

llm = LLMClient()
if llm.is_available:
    console.print(
        f"\n  [green]✓[/green] Authentication already configured "
        f"([bold]{llm.provider}[/bold] / [bold]{llm.model}[/bold]). All set!\n"
    )
else:
    prompt_auth_setup(console)

console.rule(style="cyan")
PYEOF
    exit 0
fi

# ---------------------------------------------------------------------------
# Check if auth is already satisfied (pure-bash fallback)
# ---------------------------------------------------------------------------
already_configured() {
    # 1. LLM_API_KEY in environment or .env file
    if [[ -n "${LLM_API_KEY:-}" ]]; then
        ok "LLM_API_KEY is set in the environment."
        return 0
    fi
    if [[ -f ".env" ]] && grep -qE '^LLM_API_KEY\s*=\s*.+' .env 2>/dev/null; then
        ok "LLM_API_KEY found in .env file."
        return 0
    fi
    # 2. gh CLI already logged in
    if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
        ok "GitHub CLI is already authenticated ($(gh auth status --active 2>&1 | head -1 | sed 's/.*as /as /'))."
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Option A: GitHub CLI login
# ---------------------------------------------------------------------------
setup_gh_login() {
    hr
    say "Launching GitHub CLI login…"
    say "Follow the prompts — choose 'GitHub.com' and 'HTTPS' when asked."
    echo
    gh auth login
    echo
    ok "GitHub CLI authentication complete."
    ok "The pipeline will automatically use your GitHub token via 'gh auth token'."
}

# ---------------------------------------------------------------------------
# Option B: .env file
# ---------------------------------------------------------------------------
setup_env_file() {
    hr
    say "Setting up authentication via .env file."
    echo

    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            say "Created .env from .env.example."
        else
            touch .env
            say "Created empty .env."
        fi
    fi

    if grep -qE '^LLM_API_KEY\s*=\s*.+' .env 2>/dev/null; then
        ok "LLM_API_KEY is already set in .env — nothing to do."
        return
    fi

    # Prompt for the key interactively (only when we have a TTY)
    if [[ -t 0 ]]; then
        echo -e "${YELLOW}Enter your GitHub Personal Access Token (or OpenAI API key).${NC}"
        echo    "  • GitHub PAT — visit https://github.com/settings/tokens and create a"
        echo    "    token with at least 'models:read' scope (or use a classic PAT)."
        echo    "  • OpenAI key — starts with sk-"
        echo
        read -rsp "  API key (input hidden): " api_key
        echo

        if [[ -z "$api_key" ]]; then
            warn "No key entered. You can add it later:"
            warn "  echo 'LLM_API_KEY=<your-key>' >> .env"
            return
        fi

        # Update or append the key
        if grep -qE '^LLM_API_KEY' .env; then
            sed -i "s|^LLM_API_KEY.*|LLM_API_KEY=${api_key}|" .env
        else
            echo "LLM_API_KEY=${api_key}" >> .env
        fi
        ok "LLM_API_KEY written to .env."
    else
        warn "Non-interactive shell detected."
        warn "Edit .env and set:  LLM_API_KEY=<your-token>"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    hr
    echo -e "${CYAN}  Fortran-to-Rust — Authentication Setup${NC}"
    hr
    echo

    # Fast path: already good to go
    if already_configured; then
        echo
        ok "Authentication is already configured. You're all set!"
        hr
        return 0
    fi

    # Interactive choice (requires TTY)
    if [[ ! -t 0 ]]; then
        warn "Non-interactive environment detected."
        warn "Run  bash scripts/setup_auth.sh  in a terminal to complete setup, or:"
        warn "  Option A: run 'gh auth login' to authenticate with GitHub CLI."
        warn "  Option B: copy .env.example to .env and set LLM_API_KEY."
        hr
        return 0
    fi

    echo "How would you like to authenticate?"
    echo
    echo "  1) GitHub CLI  (gh auth login)  — recommended"
    echo "     Uses your GitHub account directly; no key to copy or store."
    echo
    echo "  2) .env file  — manual"
    echo "     Paste a GitHub PAT or OpenAI API key into a local .env file."
    echo
    echo "  s) Skip for now"
    echo

    while true; do
        read -rp "  Your choice [1/2/s]: " choice
        case "$choice" in
            1) setup_gh_login; break ;;
            2) setup_env_file; break ;;
            [sS]) warn "Skipped. Run 'bash scripts/setup_auth.sh' later to set up auth."; break ;;
            *) echo "  Please enter 1, 2, or s." ;;
        esac
    done

    echo
    hr
    say "Setup complete. Start a conversion with:"
    say "  python convert.py <fortran-file.f>"
    hr
}

main "$@"
