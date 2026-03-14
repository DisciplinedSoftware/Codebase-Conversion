"""LLM API client.

Supports:
  - GitHub Models  (https://models.inference.ai.azure.com — default)
  - OpenAI         (https://api.openai.com/v1/chat/completions)
  - Any OpenAI-compatible endpoint (e.g. Ollama, LM Studio)

Configuration is read from environment variables (and from a ``.env`` file
in the current working directory, which is loaded automatically):

    LLM_PROVIDER   = "github_models" | "openai" | "openai_compatible"  (default: github_models)
    LLM_API_KEY    = GitHub PAT / OpenAI API key / custom bearer token
    LLM_BASE_URL   = base URL for openai_compatible provider
    LLM_MODEL      = model name override  (default: gpt-4o)

GitHub Models authentication is resolved in this order:
1. ``LLM_API_KEY`` environment variable (or ``.env`` file in the project root).
2. ``gh auth token`` — the GitHub CLI token (works after ``gh auth login``).
3. ``~/.config/github-copilot/hosts.json`` — VS Code / Copilot credential store.

GitHub Models (``https://models.inference.ai.azure.com``) is the supported
public API for application code.  It accepts a raw GitHub Personal Access
Token or the OAuth token obtained via ``gh auth login`` — no token exchange
or Copilot subscription is required.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com/chat/completions"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL = "gpt-4o"
_REQUEST_TIMEOUT = 120  # seconds
_GH_CLI_TIMEOUT = 8  # seconds — gh auth token should respond quickly


class LLMError(Exception):
    """Raised when the LLM call fails unrecoverably."""


class LLMUnavailableError(LLMError):
    """Raised when no API key / token is configured."""


# ---------------------------------------------------------------------------
# .env file loader (no external dependency)
# ---------------------------------------------------------------------------

def _load_dotenv(path: Optional[Path] = None) -> None:
    """Load key=value pairs from a .env file into os.environ.

    Only sets variables that are not already in the environment (so real
    environment variables always win).  Silently does nothing if the file
    does not exist.
    """
    env_file = path or (Path.cwd() / ".env")
    if not env_file.is_file():
        return
    try:
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Token discovery helpers
# ---------------------------------------------------------------------------

def _load_token_from_gh_cli() -> Optional[str]:
    """Return the active GitHub token from the ``gh`` CLI, or None."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=_GH_CLI_TIMEOUT,
        )
        if result.returncode == 0:
            token = result.stdout.strip()
            if token:
                return token
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _load_token_from_vscode_store() -> Optional[str]:
    """Try to read the GitHub OAuth token from the VS Code / Copilot credential store.

    The file ``~/.config/github-copilot/hosts.json`` is written by both the
    GitHub Copilot VS Code extension and the ``gh`` CLI when you run
    ``gh auth login --web``.  The token stored there is a standard GitHub
    OAuth token that can be used with any GitHub API, including GitHub Models.
    """
    candidates = [
        Path.home() / ".config" / "github-copilot" / "hosts.json",
        Path.home() / "AppData" / "Roaming" / "GitHub Copilot" / "hosts.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for _host, value in data.items():
                    token = value.get("oauth_token")
                    if token:
                        return token
            except Exception:
                pass
    return None


def _resolve_api_key(provider: str) -> Optional[str]:
    """Return an API key/token for *provider*, or None if not configured."""
    # 1. Explicit env / .env variable
    key = os.environ.get("LLM_API_KEY")
    if key:
        return key
    # 2. GitHub CLI (relevant for github_models and the legacy "copilot" alias)
    if provider in ("github_models", "copilot"):
        key = _load_token_from_gh_cli()
        if key:
            return key
        # 3. VS Code / Copilot credential store
        key = _load_token_from_vscode_store()
        if key:
            return key
    return None


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around the chat-completions API."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        # Load .env before reading any env vars
        _load_dotenv()

        raw_provider = (provider or os.environ.get("LLM_PROVIDER", "github_models")).lower()
        # "copilot" is kept as a backward-compatible alias for "github_models":
        # both providers authenticate with a GitHub token and use the same
        # GitHub Models API endpoint, so there is no functional difference.
        self.provider = "github_models" if raw_provider == "copilot" else raw_provider

        self.api_key = api_key or _resolve_api_key(self.provider)
        self.model = model or os.environ.get("LLM_MODEL") or _DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or self._default_url()

    def _default_url(self) -> str:
        if self.provider == "openai":
            return _OPENAI_ENDPOINT
        if self.provider == "openai_compatible":
            return os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1/chat/completions")
        return _GITHUB_MODELS_ENDPOINT

    @property
    def is_available(self) -> bool:
        """Return True when a token/key is configured."""
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        retry: int = 2,
    ) -> str:
        """Send a chat-completions request and return the assistant reply text.

        Raises LLMUnavailableError if no key is configured.
        Raises LLMError on API failure after *retry* attempts.
        """
        if not self.is_available:
            raise LLMUnavailableError(
                "No API key found.  To authenticate:\n"
                "  1. Run `gh auth login` — the pipeline will use your GitHub token\n"
                "     automatically with the GitHub Models API.\n"
                "  OR\n"
                "  2. Set LLM_API_KEY=<your-github-pat> in a .env file (or environment).\n"
                "     For OpenAI: also set LLM_PROVIDER=openai and LLM_API_KEY=<openai-key>.\n"
                "  OR\n"
                "  3. Set LLM_PROVIDER=openai_compatible and LLM_BASE_URL=<endpoint> for\n"
                "     a local model server (e.g. Ollama, LM Studio)."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(retry + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", "5"))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < retry:
                    time.sleep(2 ** attempt)

        raise LLMError(f"LLM request failed after {retry + 1} attempts: {last_exc}") from last_exc

    def translate_fortran_to_rust(
        self,
        fortran_source: str,
        function_name: str,
        extra_context: str = "",
    ) -> str:
        """Single-shot translation prompt."""
        fn_lower = function_name.lower()
        system = (
            "You are an expert Fortran-to-Rust translator. "
            "Produce idiomatic, safe Rust code. "
            f"The output MUST contain a top-level `pub fn {fn_lower}(...)` or "
            f"`pub unsafe fn {fn_lower}(...)` — do NOT wrap it inside any `mod` block. "
            "Use f64 for DOUBLE PRECISION, i32 for INTEGER, bool for LOGICAL, "
            "u8 for CHARACTER*1 (pass ASCII byte literals like b'N'). "
            "Fortran arrays are column-major and 1-indexed; convert to 0-indexed slices. "
            "INOUT array arguments become &mut [f64]. "
            "Replace DO loops with for loops. "
            "Return ONLY the Rust source code — no markdown fences, no explanations."
        )
        user_parts = []
        if extra_context:
            user_parts.append(f"Context:\n{extra_context}\n")
        user_parts.append(
            f"Convert the following Fortran subroutine `{function_name}` to Rust:\n\n"
            f"{fortran_source}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_parts)},
        ]
        return self.chat(messages)

    def repair_rust(
        self,
        rust_source: str,
        compiler_errors: str,
        function_name: str,
    ) -> str:
        """Feed compiler errors back to the LLM and ask for a fix."""
        fn_lower = function_name.lower()
        system = (
            "You are an expert Rust developer. "
            "Fix the compilation errors in the provided Rust code. "
            "Keep all Fortran CHARACTER*1 arguments typed as u8 (not char). "
            f"Ensure the function is declared as a top-level `pub fn {fn_lower}(...)` or "
            f"`pub unsafe fn {fn_lower}(...)` — do NOT move it inside a `mod` block. "
            "Return ONLY the corrected Rust source — no markdown, no explanations."
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"The following Rust code for `{function_name}` does not compile.\n\n"
                    f"Rust code:\n{rust_source}\n\n"
                    f"Compiler errors:\n{compiler_errors}\n\n"
                    "Please fix all errors and return the corrected code."
                ),
            },
        ]
        return self.chat(messages)

    def ask_clarification(
        self,
        fortran_source: str,
        question: str,
    ) -> str:
        """Questioner→Solver: ask a clarifying question about the Fortran code."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Fortran expert helping to clarify semantics "
                    "for a Fortran-to-Rust translation. Answer precisely."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Fortran source:\n{fortran_source}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        return self.chat(messages)

    def polish_unsafe_rust(self, unsafe_rust: str, function_name: str) -> str:
        """Replace unsafe blocks with safe Rust idioms where possible."""
        system = (
            "You are an expert safe-Rust refactorer. "
            "Replace raw pointer arithmetic with slice indexing. "
            "Replace unsafe blocks with safe Rust idioms where semantically equivalent. "
            "Preserve correctness exactly. "
            "Return ONLY the Rust source code."
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Refactor this unsafe Rust code for `{function_name}` to be safe:\n\n"
                    f"{unsafe_rust}"
                ),
            },
        ]
        return self.chat(messages)

