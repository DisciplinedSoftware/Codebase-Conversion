"""Strategy 2 — Agentic Multi-Turn Dialogue (Questioner-Solver).

Pipeline
--------
1. *Solver* LLM proposes an initial Rust translation.
2. *Questioner* LLM (same model, different prompt) asks up to
   ``max_questions`` clarifying questions about Fortran intrinsics,
   implicit typing, or module interfaces.
3. Solver revises based on each answer.
4. Compile-test loop: same as Strategy 1 (≤ 2 repair rounds).
5. Idiomisation pass: run clippy suggestions via LLM polish.
6. If LLM unavailable, fall back to Hybrid.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fortran_to_rust.llm_client import LLMClient, LLMUnavailableError
from fortran_to_rust.parser import FortranRoutine, strip_comments
from fortran_to_rust.strategies.base import ConversionResult, ConversionStrategy
from fortran_to_rust.strategies.hybrid import HybridStrategy
from fortran_to_rust.strategies.llm_first import _strip_fences

_MAX_QUESTIONS = 3
_MAX_REPAIR_ROUNDS = 2


class AgenticStrategy(ConversionStrategy):
    name = "Agentic Multi-Turn Dialogue"

    def __init__(
        self,
        output_dir: Path,
        llm: Optional[LLMClient] = None,
        max_questions: int = _MAX_QUESTIONS,
        **kwargs,
    ) -> None:
        super().__init__(output_dir, **kwargs)
        self.llm = llm or LLMClient()
        self.max_questions = max_questions
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

        cleaned = strip_comments(routine.source)

        # ---- Phase 1: Solver — initial translation -------------------------
        cb(f"[2/agentic] Solver: initial translation of {routine.name}…")
        try:
            rust_source = _strip_fences(
                self.llm.translate_fortran_to_rust(cleaned, routine.name)
            )
        except Exception as exc:
            cb(f"  [red]Solver failed: {exc}[/red]")
            result = self._fallback.convert(routine, progress_callback=cb)
            result.error = str(exc)
            return result

        # ---- Phase 2: Questioner-Solver dialogue ---------------------------
        questions = self._generate_questions(routine, rust_source)
        for qi, question in enumerate(questions[: self.max_questions], 1):
            cb(f"  [cyan]Questioner Q{qi}:[/cyan] {question}")
            try:
                answer = self.llm.ask_clarification(cleaned, question)
                cb(f"  [green]Answer:[/green] {answer[:120]}{'…' if len(answer) > 120 else ''}")
                # Solver revises based on the answer
                cb(f"  [2/agentic] Solver: revising translation…")
                revision_prompt = (
                    f"Revise the following Rust translation of `{routine.name}` "
                    f"based on this clarification:\n\nQ: {question}\nA: {answer}\n\n"
                    f"Current Rust:\n{rust_source}"
                )
                revised = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Fortran-to-Rust translator. "
                                "Revise only what the clarification affects. "
                                "Return ONLY Rust source code."
                            ),
                        },
                        {"role": "user", "content": revision_prompt},
                    ]
                )
                rust_source = _strip_fences(revised)
            except Exception as exc:
                cb(f"  [yellow]Dialogue round {qi} failed: {exc}[/yellow]")

        # ---- Phase 3: Compile + repair loop --------------------------------
        errors: List[str] = []
        repair_rounds = 0
        from fortran_to_rust.rust_project import quick_check

        for attempt in range(_MAX_REPAIR_ROUNDS + 1):
            ok, compiler_output = quick_check(rust_source, self.output_dir)
            if ok:
                break
            errors.append(compiler_output)
            if attempt < _MAX_REPAIR_ROUNDS:
                repair_rounds += 1
                cb(f"  [yellow]Compile errors (round {attempt + 1}), asking LLM to repair…[/yellow]")
                try:
                    rust_source = _strip_fences(
                        self.llm.repair_rust(rust_source, compiler_output, routine.name)
                    )
                except Exception as exc:
                    cb(f"  [red]LLM repair failed: {exc}[/red]")
                    break

        # ---- Phase 4: Idiomisation (clippy-guided) -------------------------
        cb(f"[2/agentic] Idiomisation pass…")
        try:
            polished = self.llm.polish_unsafe_rust(rust_source, routine.name)
            if polished.strip():
                rust_source = _strip_fences(polished)
        except Exception:
            pass

        return ConversionResult(
            routine_name=routine.name,
            rust_source=rust_source,
            success=True,
            strategy_used=f"Agentic ({self.llm.provider}/{self.llm.model}, "
                          f"{min(len(questions), self.max_questions)} Q&A rounds)",
            repair_rounds=repair_rounds,
            compiler_errors=errors,
        )

    # ------------------------------------------------------------------
    def _generate_questions(
        self, routine: FortranRoutine, rust_draft: str
    ) -> List[str]:
        """Return clarifying questions the Questioner should ask."""
        questions: List[str] = []

        # Static questions based on common Fortran pitfalls
        if "LSAME" in routine.source.upper():
            questions.append(
                "The Fortran code uses LSAME for case-insensitive character comparison. "
                "How should the Rust translation handle this — inline char comparison or a helper?"
            )
        if "IMPLICIT" not in routine.source.upper():
            questions.append(
                "There is no IMPLICIT NONE statement. Which variables rely on implicit "
                "Fortran-77 typing (I–N = INTEGER, rest = REAL)? List each implicit variable."
            )
        if any(kw in routine.source.upper() for kw in ["GOTO", "GO TO"]):
            questions.append(
                "The code contains GOTO statements. What is the best structured-Rust "
                "equivalent (loop+break, match, or explicit state machine)?"
            )
        if "XERBLA" in routine.source.upper():
            questions.append(
                "XERBLA is a BLAS error handler that stops the program. "
                "Should the Rust translation panic, return Result::Err, or use a custom error type?"
            )
        if "EXTERNAL" in routine.source.upper():
            questions.append(
                "There are EXTERNAL declarations. Which of these are callback function pointers "
                "that should become Rust function parameters with fn(...) types?"
            )

        # If we still need more, ask a generic question
        if len(questions) < _MAX_QUESTIONS:
            questions.append(
                "Are there any Fortran-specific numerical semantics (e.g. array storage order, "
                "IEEE rounding) that must be preserved exactly in the Rust translation?"
            )

        return questions
