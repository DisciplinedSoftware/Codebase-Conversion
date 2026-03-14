"""LLM API client.

Supports:
  - GitHub Copilot  (https://api.githubcopilot.com/chat/completions)
  - OpenAI          (https://api.openai.com/v1/chat/completions)
  - Any OpenAI-compatible endpoint (e.g. Ollama, LM Studio)

Configuration is read from environment variables:

    LLM_PROVIDER   = "copilot" | "openai" | "openai_compatible"  (default: copilot)
    LLM_API_KEY    = bearer token / API key
    LLM_BASE_URL   = base URL for openai_compatible provider
    LLM_MODEL      = model name override

GitHub Copilot token can also be read from the VS Code credential store
(~/.config/github-copilot/hosts.json) so the devcontainer works out of
the box after `gh auth login`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COPILOT_ENDPOINT = "https://api.githubcopilot.com/chat/completions"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL_COPILOT = "gpt-4o"
_DEFAULT_MODEL_OPENAI = "gpt-4o"
_REQUEST_TIMEOUT = 120  # seconds


class LLMError(Exception):
    """Raised when the LLM call fails unrecoverably."""


class LLMUnavailableError(LLMError):
    """Raised when no API key / token is configured."""


# ---------------------------------------------------------------------------
# Token discovery helpers
# ---------------------------------------------------------------------------

def _load_copilot_token_from_vscode() -> Optional[str]:
    """Try to read the GitHub Copilot OAuth token from the VS Code store."""
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
    """Return the API key for *provider*, or None if not configured."""
    key = os.environ.get("LLM_API_KEY")
    if key:
        return key
    if provider == "copilot":
        return _load_copilot_token_from_vscode()
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
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "copilot")).lower()
        self.api_key = api_key or _resolve_api_key(self.provider)
        self.model = model or os.environ.get("LLM_MODEL") or self._default_model()
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or self._default_url()

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
                "No LLM API key found. "
                "Set LLM_API_KEY in your environment (or .env file), "
                "or log in with `gh auth login` for GitHub Copilot."
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
