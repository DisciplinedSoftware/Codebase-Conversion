"""LLM API client.

Supports:
  - GitHub Copilot  (https://api.githubcopilot.com — default, for Copilot subscribers)
  - GitHub Models   (https://models.inference.ai.azure.com — free tier, separate product)
  - OpenAI          (https://api.openai.com/v1/chat/completions)
  - Any OpenAI-compatible endpoint (e.g. Ollama, LM Studio)

Configuration is read from environment variables (and from a ``.env`` file
in the current working directory, which is loaded automatically):

    LLM_PROVIDER   = "github_models" | "copilot" | "openai" | "openai_compatible"
                     (default: github_models)
    LLM_API_KEY    = GitHub PAT / OpenAI API key / custom bearer token
    LLM_BASE_URL   = base URL for openai_compatible provider
    LLM_MODEL      = model name override  (default: gpt-4o)

GitHub Copilot provider (recommended for Copilot subscribers):
  Uses ``https://api.githubcopilot.com/chat/completions``.  Automatically
  exchanges your GitHub token for a short-lived Copilot API token.  Token
  discovery order:
    1. ``LLM_API_KEY`` env / ``.env`` file.
    2. ``gh auth token`` — the GitHub CLI token (works after ``gh auth login``).
    3. ``~/.config/github-copilot/hosts.json`` — VS Code / Copilot credential store.

GitHub Models provider:
  Uses ``https://models.inference.ai.azure.com``.  Separate free-tier product
  with its own rate limits — not linked to a Copilot subscription.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COPILOT_ENDPOINT = "https://api.githubcopilot.com/chat/completions"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com/chat/completions"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL = "gpt-4o"
_REQUEST_TIMEOUT = 120  # seconds
_GH_CLI_TIMEOUT = 8  # seconds — gh auth token should respond quickly


def _warn(msg: str) -> None:
    """Print a warning to stderr immediately (visible even mid-stream)."""
    import sys
    print(f"\n[LLM] {msg}", file=sys.stderr, flush=True)


class LLMError(Exception):
    """Raised when the LLM call fails unrecoverably."""


class LLMUnavailableError(LLMError):
    """Raised when no API key / token is configured."""


# ---------------------------------------------------------------------------
# Copilot token exchange (GitHub token → short-lived Copilot API token)
# ---------------------------------------------------------------------------

class _CopilotTokenCache:
    """In-memory cache for a single short-lived Copilot API token."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._expires_at: Optional[datetime] = None

    def valid(self) -> bool:
        return bool(
            self._token
            and self._expires_at
            and datetime.now(timezone.utc) < self._expires_at
        )

    def get(self) -> Optional[str]:
        return self._token if self.valid() else None

    def set(self, token: str, expires_at: str) -> None:
        self._token = token
        try:
            self._expires_at = datetime.fromisoformat(
                expires_at.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            self._expires_at = datetime.fromtimestamp(
                time.time() + 25 * 60, tz=timezone.utc
            )


_copilot_cache = _CopilotTokenCache()


def _exchange_token(github_token: str) -> Optional[str]:
    """Try to exchange *github_token* for a Copilot API token via GET.

    Returns the Copilot bearer token string, or None on non-fatal failure.
    Prints a diagnostic line so the caller can see why it failed.
    """
    try:
        resp = requests.get(
            _COPILOT_TOKEN_URL,
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.90.0",
                "Editor-Plugin-Version": "copilot/1.0",
                "User-Agent": "GithubCopilot/1.0.0",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            tok = data.get("token")
            if tok:
                return tok
            _warn(f"Copilot exchange 200 but no 'token' in response: {data}")
            return None
        # Surface the failure so the user (and developer) can see why
        _warn(
            f"Copilot token exchange failed: HTTP {resp.status_code} — {resp.text[:200]}"
        )
        return None
    except requests.RequestException as exc:
        _warn(f"Copilot token exchange network error: {exc}")
        return None


def _get_copilot_bearer(github_token: str) -> str:
    """Return a short-lived Copilot API bearer token.

    Token discovery order:
    1. In-memory cache (reuse until expiry).
    2. VS Code / Copilot extension credential store
       (~/.config/github-copilot/hosts.json).
    3. GITHUB_TOKEN env var (GitHub Codespaces).
    4. The supplied *github_token* (gh auth token / LLM_API_KEY).

    Raises LLMError if no exchange succeeds.
    """
    cached = _copilot_cache.get()
    if cached:
        return cached

    vscode_token = _load_token_from_vscode_store()
    codespaces_token = os.environ.get("GITHUB_TOKEN")
    candidates = [t for t in [vscode_token, codespaces_token, github_token] if t]

    for candidate in candidates:
        token = _exchange_token(candidate)
        if token:
            _copilot_cache.set(token, "")
            return token

    raise LLMError("Copilot token exchange failed for all candidates (see warnings above).")


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
    # 2. GitHub CLI — relevant for copilot and github_models
    if provider in ("copilot", "github_models"):
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
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        # Load .env before reading any env vars
        _load_dotenv()

        # Default to "github_models" — the deprecated github.copilot VS Code
        # extension no longer writes ~/.config/github-copilot/hosts.json, so
        # the Copilot token exchange no longer works in most environments.
        # Override with LLM_PROVIDER=copilot in .env to attempt the exchange.
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "github_models")).lower()

        self.api_key = api_key or _resolve_api_key(self.provider)
        self.model = model or os.environ.get("LLM_MODEL") or _DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or self._default_url()

        # Optional callback receiving each streamed text chunk as it arrives.
        # When set, chat() uses SSE streaming so the caller sees output live.
        self.stream_callback = stream_callback

    def _default_url(self) -> str:
        if self.provider == "copilot":
            return _COPILOT_ENDPOINT
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
            if self.provider == "copilot":
                # Try the Copilot token exchange; fall back to using the raw
                # GitHub token directly (github_models-style) if it fails so
                # that `gh auth token` always works regardless of provider setting.
                try:
                    bearer = _get_copilot_bearer(self.api_key)
                    headers["Authorization"] = f"Bearer {bearer}"
                    headers["Editor-Version"] = "vscode/1.90.0"
                    headers["Copilot-Integration-Id"] = "vscode-chat"
                    # Also switch the endpoint to github_models on fallback
                except LLMError:
                    _warn(
                        "Copilot token exchange failed — falling back to GitHub Models "
                        "(models.inference.ai.azure.com) with raw token."
                    )
                    self.provider = "github_models"
                    self.base_url = _GITHUB_MODELS_ENDPOINT
                    headers["Authorization"] = f"Bearer {self.api_key}"
            else:
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

        When ``self.stream_callback`` is set the request uses SSE streaming:
        each text chunk is passed to the callback as it arrives, so the caller
        can display output live instead of waiting for the full response.

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

        streaming = self.stream_callback is not None
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": streaming,
        }

        _MAX_RATE_LIMIT_WAIT = 30  # seconds — never sleep longer than this per attempt
        _MAX_RATE_LIMIT_RETRIES = 5

        last_exc: Optional[Exception] = None
        rate_limit_hits = 0
        for attempt in range(retry + 1):
            try:
                # Evaluate headers first — _headers() may update self.base_url
                # (e.g. copilot → github_models fallback).
                hdrs = self._headers()
                resp = requests.post(
                    self.base_url,
                    headers=hdrs,
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
                    stream=streaming,
                )
                if resp.status_code == 429:
                    rate_limit_hits += 1
                    raw_wait = resp.headers.get("Retry-After", "5")
                    wait = min(int(raw_wait), _MAX_RATE_LIMIT_WAIT)
                    # Always drain/close the response body before sleeping so
                    # the connection is returned to the pool immediately.
                    resp.close()
                    _warn(
                        f"Rate limited (429) — waiting {wait}s "
                        f"[attempt {rate_limit_hits}/{_MAX_RATE_LIMIT_RETRIES}] …"
                    )
                    if rate_limit_hits >= _MAX_RATE_LIMIT_RETRIES:
                        raise LLMError(
                            f"Rate limited {rate_limit_hits} times in a row. "
                            "Check your API quota or try again later."
                        )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

                if streaming:
                    return self._consume_stream(resp)
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except LLMError:
                raise
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                _warn(f"Request error (attempt {attempt + 1}/{retry + 1}): {exc}")
                if attempt < retry:
                    time.sleep(2 ** attempt)

        raise LLMError(f"LLM request failed after {retry + 1} attempts: {last_exc}") from last_exc

    def _consume_stream(self, resp: "requests.Response") -> str:
        """Parse an SSE streaming response, calling ``stream_callback`` per chunk.

        Returns the full concatenated text when the stream ends.
        """
        chunks: List[str] = []
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)
                content = delta["choices"][0]["delta"].get("content")
                if content:
                    chunks.append(content)
                    if self.stream_callback:
                        self.stream_callback(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return "".join(chunks)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove markdown code fences that LLMs sometimes add despite instructions."""
        text = text.strip()
        # Remove opening fence: ```rust or ``` or ```python etc.
        if text.startswith("```"):
            text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[: text.rfind("```")].rstrip()
        return text

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
            "Fortran arrays are column-major: A(i,j) in Fortran (1-indexed) is stored as "
            "a[(j-1)*lda + (i-1)] in Rust (0-indexed). "
            "NEVER convert to row-major — preserve the Fortran column-major memory layout. "
            "INOUT array arguments become &mut [f64], IN-only arrays become &[f64]. "
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
        return self._strip_code_fence(self.chat(messages))

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
        return self._strip_code_fence(self.chat(messages))

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
            "Fortran arrays are column-major: A(i,j) in Fortran (1-indexed) is stored as "
            "a[(j-1)*lda + (i-1)] in Rust (0-indexed). "
            "NEVER convert to row-major — preserve the Fortran column-major memory layout. "
            "Preserve all other logic exactly. "
            "Return ONLY the Rust source code — no markdown fences, no explanations."
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
        return self._strip_code_fence(self.chat(messages))

    def repair_accuracy(
        self,
        rust_source: str,
        fortran_source: str,
        function_name: str,
        max_abs_error: float,
    ) -> str:
        """Re-translate Fortran→Rust after a numerical accuracy failure.

        Passes the failing Rust alongside the original Fortran and an explicit
        explanation of the most common mistake (row-major vs column-major indexing).
        """
        fn_lower = function_name.lower()
        system = (
            "You are an expert Fortran-to-Rust translator. "
            "A previous translation produced numerically WRONG results due to incorrect "
            "array indexing. You must fix this by strictly following the rule below.\n\n"
            "CRITICAL: Fortran arrays are COLUMN-MAJOR (column index varies slowest in memory). "
            "For a 2-D Fortran array A with leading dimension LDA:\n"
            "  A(i, j)  in Fortran (1-indexed)  ==  a[(j-1)*lda + (i-1)]  in Rust (0-indexed)\n"
            "Example for DGEMM (no-transpose): A[i,l] = a[l*lda + i], "
            "B[l,j] = b[j*ldb + l], C[i,j] = c[j*ldc + i]. "
            "DO NOT use row-major indexing like a[i*lda + l].\n\n"
            f"The output MUST contain a top-level `pub fn {fn_lower}(...)` or "
            f"`pub unsafe fn {fn_lower}(...)` — do NOT wrap it inside any `mod` block. "
            "Use f64 for DOUBLE PRECISION, i32 for INTEGER, bool for LOGICAL, "
            "u8 for CHARACTER*1 (pass ASCII byte literals like b'N'). "
            "Return ONLY the corrected Rust source code — no markdown fences, no explanations."
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"The Rust translation of `{function_name}` produced "
                    f"max_abs_error={max_abs_error:.2e} vs the Fortran reference — "
                    f"this indicates wrong array indexing.\n\n"
                    f"Incorrect Rust code:\n{rust_source}\n\n"
                    f"Original Fortran source:\n{fortran_source}\n\n"
                    "Produce a CORRECT Rust translation using strict column-major indexing."
                ),
            },
        ]
        return self._strip_code_fence(self.chat(messages))

