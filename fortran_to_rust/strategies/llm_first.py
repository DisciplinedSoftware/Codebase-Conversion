"""Strategy 1 — LLM-First with Rule Fallback.

Pipeline
--------
1. Pre-process: strip comments, build call graph context.
2. Feed each function chunk to the LLM for translation.
3. Compile with ``cargo check``; on failure feed errors back (≤ 2 repair rounds).
4. If the LLM is unavailable, fall back to the Hybrid rule-based strategy.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fortran_to_rust.llm_client import LLMClient, LLMUnavailableError
from fortran_to_rust.parser import FortranRoutine, strip_comments
from fortran_to_rust.strategies.base import ConversionResult, ConversionStrategy
from fortran_to_rust.strategies.hybrid import HybridStrategy

_MAX_REPAIR_ROUNDS = 2


class LLMFirstStrategy(ConversionStrategy):
    name = "LLM-First with Rule Fallback"

    def __init__(self, output_dir: Path, llm: Optional[LLMClient] = None, **kwargs) -> None:
        super().__init__(output_dir, **kwargs)
        self.llm = llm or LLMClient()
        self._fallback = HybridStrategy(output_dir, llm=self.llm)

    # ------------------------------------------------------------------
    def convert(
        self,
        routine: FortranRoutine,
        *,
        progress_callback=None,
    ) -> ConversionResult:
        cb = progress_callback or (lambda msg: None)

        if not self.llm.is_available:
            cb(
                "  [yellow]⚠  No LLM API key — falling back to Hybrid rule-based strategy.[/yellow]"
            )
            result = self._fallback.convert(routine, progress_callback=cb)
            result.strategy_used = f"Hybrid fallback (LLM unavailable): {result.strategy_used}"
            result.warnings.append(
                "LLM unavailable; result is a rule-based skeleton requiring manual review."
            )
            return result

        cb(f"[1/llm-first] Sending {routine.name} to LLM ({self.llm.model})…")

        cleaned_source = strip_comments(routine.source)
        context = self._build_context(routine)

        try:
            rust_source = self.llm.translate_fortran_to_rust(
                cleaned_source, routine.name, extra_context=context
            )
        except Exception as exc:
            cb(f"  [red]LLM translation failed: {exc}[/red]")
            result = self._fallback.convert(routine, progress_callback=cb)
            result.strategy_used = f"Hybrid fallback (LLM error): {result.strategy_used}"
            result.error = str(exc)
            return result

        rust_source = _strip_fences(rust_source)

        # Compile + repair loop
        errors: list[str] = []
        repair_rounds = 0
        from fortran_to_rust.rust_project import quick_check

        for attempt in range(_MAX_REPAIR_ROUNDS + 1):
            ok, compiler_output = quick_check(rust_source, self.output_dir)
            if ok:
                break
            errors.append(compiler_output)
            if attempt < _MAX_REPAIR_ROUNDS:
                repair_rounds += 1
                cb(
                    f"  [yellow]Compile errors (round {attempt + 1}), asking LLM to repair…[/yellow]"
                )
                try:
                    rust_source = _strip_fences(
                        self.llm.repair_rust(rust_source, compiler_output, routine.name)
                    )
                except Exception as exc:
                    cb(f"  [red]LLM repair failed: {exc}[/red]")
                    break
            else:
                cb(
                    f"  [yellow]Could not fully compile after {_MAX_REPAIR_ROUNDS} repair rounds.[/yellow]"
                )

        return ConversionResult(
            routine_name=routine.name,
            rust_source=rust_source,
            success=True,
            strategy_used=f"LLM-First ({self.llm.provider}/{self.llm.model})",
            repair_rounds=repair_rounds,
            compiler_errors=errors,
        )

    # ------------------------------------------------------------------
    def _build_context(self, routine: FortranRoutine) -> str:
        """Provide the LLM with helper context: called functions, arg types."""
        parts = []
        if routine.calls:
            parts.append(
                "This routine calls: " + ", ".join(routine.calls) + ".\n"
                "Common BLAS helpers: LSAME (character equality), XERBLA (error handler)."
            )
        if routine.args:
            parts.append("Arguments: " + ", ".join(routine.args))
        parts.append(
            "Array indexing: Fortran uses 1-based column-major. "
            "Convert to 0-based row-major: A(i,j) with LDA leading dimension "
            "maps to a[j * lda + i] in C/Rust (column-major kept) or "
            "a[(i-1) + (j-1) * lda] after 1→0 index shift."
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences that the LLM sometimes adds."""
    text = re.sub(r"^```(?:rust)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()
