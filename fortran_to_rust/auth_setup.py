"""Interactive authentication setup for the LLM client.

Called automatically by the conversion pipeline when no API key or
GitHub CLI token is detected.  Offers two paths:

  1. ``gh auth login`` — logs in via the GitHub CLI (recommended).
  2. API key         — prompts for a token and writes it to ``.env``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gh_available() -> bool:
    """Return True when the ``gh`` CLI is on PATH."""
    try:
        subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _gh_auth_login(console: Console) -> bool:
    """Run ``gh auth login`` interactively. Returns True on success."""
    if not _gh_available():
        console.print(
            "  [red]✗[/red] [bold]gh[/bold] CLI not found. "
            "Install it from [link=https://cli.github.com/]https://cli.github.com/[/link] "
            "or use Option 2 (API key)."
        )
        return False
    try:
        result = subprocess.run(["gh", "auth", "login"], check=False)
        return result.returncode == 0
    except OSError as exc:
        console.print(f"  [red]✗[/red] Failed to run gh: {exc}")
        return False


def _write_api_key_to_env(key: str) -> None:
    """Write or update ``LLM_API_KEY`` in the project ``.env`` file."""
    env_path = Path(".env")
    if not env_path.exists():
        example = Path(".env.example")
        env_path.write_text(example.read_text() if example.exists() else "")

    lines = env_path.read_text().splitlines()
    new_lines, replaced = [], False
    for line in lines:
        if line.strip().startswith("LLM_API_KEY"):
            new_lines.append(f"LLM_API_KEY={key}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"LLM_API_KEY={key}")
    env_path.write_text("\n".join(new_lines) + "\n")

    # Make the key immediately visible to subsequent LLMClient instances
    os.environ["LLM_API_KEY"] = key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prompt_auth_setup(console: Console) -> Optional["LLMClient"]:  # type: ignore[name-defined]  # noqa: F821
    """Interactive auth wizard.

    Presents a choice between ``gh auth login`` and a manual API key, then
    returns a freshly constructed :class:`~fortran_to_rust.llm_client.LLMClient`
    if authentication succeeds, or ``None`` if the user skips or setup fails.
    """
    # Import here to avoid circular imports (auth_setup ← llm_client is fine,
    # but llm_client must not import auth_setup at module level).
    from fortran_to_rust.llm_client import LLMClient, _load_dotenv

    console.print(
        "\n  [yellow]⚠  No LLM authentication found.[/yellow]\n"
        "  How would you like to authenticate?\n"
    )
    console.print("  [cyan]1[/cyan]  GitHub CLI  [dim](gh auth login — recommended)[/dim]")
    console.print("  [cyan]2[/cyan]  API key     [dim](paste a GitHub PAT or OpenAI key into .env)[/dim]")
    console.print("  [cyan]s[/cyan]  Skip        [dim](continue without LLM — rule-based conversion only)[/dim]")

    while True:
        choice = Prompt.ask(
            "\n  [bold cyan]>[/bold cyan] Your choice",
            choices=["1", "2", "s"],
            default="1",
        )

        if choice == "1":
            console.print()
            ok = _gh_auth_login(console)
            if not ok:
                # Let the user try the other option instead of hard-failing
                continue
            _load_dotenv()
            llm = LLMClient()
            if llm.is_available:
                console.print(
                    f"  [green]✓[/green] GitHub CLI authenticated — "
                    f"[bold]{llm.provider}[/bold] / [bold]{llm.model}[/bold]"
                )
                return llm
            console.print(
                "  [yellow]⚠[/yellow] Logged in but token could not be resolved. "
                "Try Option 2 or re-run [bold]gh auth login[/bold]."
            )
            return None

        elif choice == "2":
            console.print()
            console.print(
                "  Enter your GitHub Personal Access Token or OpenAI API key.\n"
                "  [dim]GitHub PAT: https://github.com/settings/tokens  "
                "(no special scopes required — any valid GitHub token works)[/dim]"
            )
            key = Prompt.ask("  [bold cyan]>[/bold cyan] API key", password=True).strip()
            if not key:
                console.print("  [yellow]No key entered.[/yellow]")
                continue
            _write_api_key_to_env(key)
            _load_dotenv()
            llm = LLMClient()
            if llm.is_available:
                console.print(
                    f"  [green]✓[/green] API key saved to .env — "
                    f"[bold]{llm.provider}[/bold] / [bold]{llm.model}[/bold]"
                )
                return llm
            console.print("  [yellow]⚠[/yellow] Key accepted but client not available — check key validity.")
            return None

        else:  # skip
            console.print("  [dim]Skipping authentication setup.[/dim]")
            return None
