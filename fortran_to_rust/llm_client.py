"""LLM API client.

Supports:
  - GitHub Copilot  (https://api.githubcopilot.com/chat/completions)
  - OpenAI          (https://api.openai.com/v1/chat/completions)
  - Any OpenAI-compatible endpoint (e.g. Ollama, LM Studio)

Configuration is read from environment variables (and from a ``.env`` file
in the current working directory, which is loaded automatically):

    LLM_PROVIDER   = "copilot" | "openai" | "openai_compatible"  (default: copilot)
    LLM_API_KEY    = bearer token / API key
    LLM_BASE_URL   = base URL for openai_compatible provider
    LLM_MODEL      = model name override

GitHub Copilot authentication is resolved in this order:
1. ``LLM_API_KEY`` environment variable (or ``.env`` file).
2. ``gh auth token`` — the GitHub CLI token (works after ``gh auth login``).
3. ``~/.config/github-copilot/hosts.json`` — VS Code credential store.

When using the Copilot provider, a raw GitHub OAuth / PAT token (formats
``gho_``, ``ghp_``, ``github_pat_``) is automatically exchanged for a
short-lived Copilot API token using the public exchange endpoint
``https://api.github.com/copilot_internal/v2/token``.  The exchanged token
is cached in-process and refreshed before it expires.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COPILOT_ENDPOINT = "https://api.githubcopilot.com/chat/completions"
_COPILOT_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL_COPILOT = "gpt-4o"
_DEFAULT_MODEL_OPENAI = "gpt-4o"
_REQUEST_TIMEOUT = 120  # seconds
_GH_CLI_TIMEOUT = 8  # seconds — gh auth token should respond quickly
_TOKEN_REFRESH_BUFFER = 90  # seconds — refresh Copilot token this early before expiry
_DEFAULT_TOKEN_EXPIRY_SECONDS = 25 * 60  # fallback when exchange response has no expires_at

# GitHub token prefixes that need to be exchanged for a Copilot API token
_GITHUB_TOKEN_PREFIXES = ("gho_", "ghp_", "ghu_", "ghs_", "github_pat_", "v1.")


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


def _load_copilot_token_from_vscode() -> Optional[str]:
    """Try to read the GitHub OAuth token from the VS Code / Copilot store."""
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


def _is_raw_github_token(token: str) -> bool:
    """Return True if *token* looks like a GitHub OAuth / PAT that needs exchange."""
    return any(token.startswith(p) for p in _GITHUB_TOKEN_PREFIXES)


# ---------------------------------------------------------------------------
# Copilot token exchange + cache
# ---------------------------------------------------------------------------

@dataclass
class _CopilotToken:
    """A short-lived Copilot API token with its expiry timestamp."""
    token: str
    expires_at: float  # Unix timestamp (seconds)

    @property
    def is_valid(self) -> bool:
        # Refresh this many seconds before expiry so we never hit the API with a stale token
        return time.time() < self.expires_at - _TOKEN_REFRESH_BUFFER


# Module-level cache: github_token → exchanged CopilotToken
_copilot_token_cache: Dict[str, _CopilotToken] = {}


def _exchange_for_copilot_token(github_token: str) -> Optional[str]:
    """Exchange a GitHub OAuth / PAT for a short-lived Copilot API token.

    Returns the Copilot API token string, or *None* on failure.
    The result is cached in-process and reused until near expiry.
    """
    cached = _copilot_token_cache.get(github_token)
    if cached and cached.is_valid:
        return cached.token

    try:
        resp = requests.get(
            _COPILOT_EXCHANGE_URL,
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/json",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            copilot_token = data.get("token")
            # expires_at is an ISO-8601 string; fall back to 25 minutes from now
            raw_exp = data.get("expires_at", "")
            try:
                from datetime import datetime, timezone
                expires_at = datetime.fromisoformat(
                    raw_exp.replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                expires_at = time.time() + _DEFAULT_TOKEN_EXPIRY_SECONDS

            if copilot_token:
                _copilot_token_cache[github_token] = _CopilotToken(
                    token=copilot_token, expires_at=expires_at
                )
                return copilot_token
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Unified token resolver
# ---------------------------------------------------------------------------

def _resolve_api_key(provider: str) -> Optional[str]:
    """Return a ready-to-use API token for *provider*, or None."""
    # 1. Explicit env / .env var (may be a Copilot token or raw GitHub token)
    key = os.environ.get("LLM_API_KEY")
    if not key and provider == "copilot":
        # 2. GitHub CLI token
        key = _load_token_from_gh_cli()
    if not key and provider == "copilot":
        # 3. VS Code credential store
        key = _load_copilot_token_from_vscode()

    if not key:
        return None

    # For the Copilot provider, a raw GitHub token must be exchanged
    if provider == "copilot" and _is_raw_github_token(key):
        exchanged = _exchange_for_copilot_token(key)
        if exchanged:
            return exchanged
        # Exchange failed — the raw GitHub token probably won't work directly,
        # but return it anyway so the caller gets a meaningful error from the API
        return key

    return key


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

        self.provider = (provider or os.environ.get("LLM_PROVIDER", "copilot")).lower()
        self.api_key = api_key or _resolve_api_key(self.provider)
        self.model = model or os.environ.get("LLM_MODEL") or self._default_model()
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or self._default_url()

        # Store the raw GitHub token so we can refresh the Copilot token later
        self._github_token: Optional[str] = None
        if self.provider == "copilot" and self.api_key:
            # If the resolved key is a Copilot token, store the source GitHub token
            # so _refresh_copilot_token() can re-exchange it.
            raw = (
                os.environ.get("LLM_API_KEY")
                or _load_token_from_gh_cli()
                or _load_copilot_token_from_vscode()
            )
            if raw and _is_raw_github_token(raw):
                self._github_token = raw

    # ------------------------------------------------------------------

    def _default_model(self) -> str:
        if self.provider == "openai":
            return _DEFAULT_MODEL_OPENAI
        return _DEFAULT_MODEL_COPILOT

    def _default_url(self) -> str:
        if self.provider == "openai":
            return _OPENAI_ENDPOINT
        if self.provider == "openai_compatible":
            return os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1/chat/completions")
        return _COPILOT_ENDPOINT

    @property
    def is_available(self) -> bool:
        """Return True when a token/key is configured."""
        return bool(self.api_key)

    def _refresh_copilot_token(self) -> None:
        """Re-exchange the GitHub token for a fresh Copilot API token if needed."""
        if self._github_token:
            fresh = _exchange_for_copilot_token(self._github_token)
            if fresh:
                self.api_key = fresh

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "copilot":
            headers["Copilot-Integration-Id"] = "vscode-chat"
            headers["Editor-Version"] = "vscode/1.85.0"
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
                "No LLM API key found.  To enable GitHub Copilot:\n"
                "  1. Run `gh auth login` and authenticate with GitHub.\n"
                "     The pipeline will automatically exchange your GitHub token\n"
                "     for a Copilot API token.\n"
                "  OR\n"
                "  2. Set LLM_API_KEY=<your-token> in a .env file (or environment).\n"
                "     For OpenAI: also set LLM_PROVIDER=openai."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(retry + 1):
            # Refresh the Copilot token before each attempt in case it expired
            if self.provider == "copilot":
                self._refresh_copilot_token()

            try:
                resp = requests.post(
                    self.base_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
                )
                if resp.status_code == 401 and self.provider == "copilot" and self._github_token:
                    # Token was rejected — invalidate the cache and retry once
                    _copilot_token_cache.pop(self._github_token, None)
                    self.api_key = _exchange_for_copilot_token(self._github_token) or self.api_key
                    continue
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
        system = (
            "You are an expert Fortran-to-Rust translator. "
            "Produce idiomatic, safe Rust code. "
            "Use f64 for DOUBLE PRECISION, i32 for INTEGER, bool for LOGICAL. "
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
        system = (
            "You are an expert Rust developer. "
            "Fix the compilation errors in the provided Rust code. "
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

