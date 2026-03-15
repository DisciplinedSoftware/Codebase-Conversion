"""Shared post-conversion pipeline loop.

After initial Fortran→Rust conversion, this module handles the
scaffold → build → repair → test → accuracy → benchmark cycle and
retries the full cycle (re-converting failing functions) whenever:

  * A numerical accuracy check fails, or
  * The Rust benchmark binary cannot be compiled or executed.

Usage::

    from fortran_to_rust.pipeline import run_post_conversion_loop

    state = run_post_conversion_loop(
        run_dir=run_dir,
        crate_name=crate_name,
        functions_to_convert=functions_to_convert,
        rust_sources=rust_sources,
        conversion_results=conversion_results,
        source_map=source_map,
        routine_map=routine_map,
        llm=llm,
        strategy=strategy,
        log=console.print,          # callable(str) for progress messages
    )
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from fortran_to_rust.benchmarker import BenchResult
    from fortran_to_rust.llm_client import LLMClient
    from fortran_to_rust.parser import FortranRoutine
    from fortran_to_rust.strategies.base import ConversionResult, ConversionStrategy
    from fortran_to_rust.test_harness import AccuracyResult

_MAX_PIPELINE_RETRIES = 3


@dataclass
class PostConversionState:
    """All outputs from the post-conversion pipeline loop."""

    crate_dir: Path
    build_ok: bool
    test_ok: bool
    accuracy_results: List["AccuracyResult"]
    bench_results: List["BenchResult"]
    rust_sources: Dict[str, str]
    conversion_results: List["ConversionResult"]
    retry_count: int = 0


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:rust)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _failed_accuracy_fns(accuracy_results: List["AccuracyResult"]) -> List[str]:
    """Return lower-cased names of functions whose accuracy check failed."""
    return [
        acc.function_name.lower()
        for acc in accuracy_results
        if not acc.passed
    ]


def _failed_bench_fns(bench_results: List["BenchResult"]) -> List[str]:
    """Return lower-cased names of functions whose Rust benchmark could not run.

    A *slow* Rust result is not treated as a failure — only results where the
    Rust binary failed to compile or execute (``rust_time_ms is None``).
    """
    return [
        b.function_name.lower()
        for b in bench_results
        if b.rust_time_ms is None and not b.error_message
    ]


def _repair_functions(
    failing_fns: List[str],
    rust_sources: Dict[str, str],
    conversion_results: List["ConversionResult"],
    source_map: Dict[str, Path],
    routine_map: Dict[str, "FortranRoutine"],
    accuracy_map: Dict[str, "AccuracyResult"],
    llm: "LLMClient",
    strategy: "ConversionStrategy",
    log: Callable[[str], None],
) -> Tuple[Dict[str, str], List["ConversionResult"]]:
    """Re-convert *failing_fns*, returning updated rust_sources and conversion_results.

    For each failing function:
    - If LLM is available and accuracy data exists: use ``llm.repair_accuracy()``
      so the LLM gets the specific numerical-error context.
    - Otherwise: re-run ``strategy.convert()`` from scratch.
    """
    from fortran_to_rust.rust_project import _ensure_top_level_pub_fn

    sources = dict(rust_sources)
    results = list(conversion_results)
    result_index = {r.routine_name.lower(): i for i, r in enumerate(results)}

    for fn_lower in failing_fns:
        routine = routine_map.get(fn_lower.upper())
        if not routine:
            log(f"  [yellow]⚠[/yellow]  {fn_lower}: no parsed routine — cannot re-convert.")
            continue

        acc = accuracy_map.get(fn_lower)
        current_source = sources.get(fn_lower, "")

        repaired_source: Optional[str] = None

        if llm.is_available and acc is not None and acc.max_abs_error is not None:
            log(f"  [cyan]♻[/cyan]  {fn_lower}: accuracy repair (max_err={acc.max_abs_error:.2e})…")
            try:
                repaired_source = _strip_fences(
                    llm.repair_accuracy(
                        current_source,
                        routine.source,
                        fn_lower,
                        acc.max_abs_error,
                    )
                )
                repaired_source = _ensure_top_level_pub_fn(repaired_source, fn_lower)
            except Exception as exc:
                log(f"  [red]✗[/red]  LLM accuracy repair failed for {fn_lower}: {exc}")
                repaired_source = None

        if repaired_source is None:
            log(f"  [cyan]♻[/cyan]  {fn_lower}: re-converting from scratch…")
            try:
                new_result = strategy.convert(
                    routine,
                    progress_callback=lambda msg: log(f"    {msg}"),
                )
                if new_result.rust_source:
                    repaired_source = new_result.rust_source
                    idx = result_index.get(fn_lower)
                    if idx is not None:
                        results[idx] = new_result
                    else:
                        results.append(new_result)
                        result_index[fn_lower] = len(results) - 1
            except Exception as exc:
                log(f"  [red]✗[/red]  Re-conversion failed for {fn_lower}: {exc}")

        if repaired_source:
            sources[fn_lower] = repaired_source
            # Update matching ConversionResult so it reflects the new source
            idx = result_index.get(fn_lower)
            if idx is not None:
                old = results[idx]
                from fortran_to_rust.strategies.base import ConversionResult
                results[idx] = ConversionResult(
                    routine_name=old.routine_name,
                    rust_source=repaired_source,
                    success=True,
                    strategy_used=old.strategy_used + " (retry)",
                    repair_rounds=old.repair_rounds + 1,
                    compiler_errors=old.compiler_errors,
                    warnings=old.warnings,
                    output_path=old.output_path,
                )

    return sources, results


def run_post_conversion_loop(
    run_dir: Path,
    crate_name: str,
    functions_to_convert: List[str],
    rust_sources: Dict[str, str],
    conversion_results: List["ConversionResult"],
    source_map: Dict[str, Path],
    routine_map: Dict[str, "FortranRoutine"],
    llm: "LLMClient",
    strategy: "ConversionStrategy",
    *,
    max_retries: int = _MAX_PIPELINE_RETRIES,
    log: Optional[Callable[[str], None]] = None,
    datasets_dir: Optional[Path] = None,
    fortran_ref_dir: Optional[Path] = None,
    run_scaffold_build: Optional[Callable[..., Tuple[bool, str, Dict[str, str]]]] = None,
) -> PostConversionState:
    """Scaffold, build, test, check accuracy and benchmark with automatic retries.

    On each attempt the full sub-pipeline runs:
      1. Scaffold Cargo crate from *rust_sources*.
      2. ``cargo build --release`` (+ LLM repair if it fails).
      3. ``cargo test``.
      4. Numerical accuracy check for every function.
      5. Performance benchmark for every function.

    After each attempt, any functions whose accuracy check failed **or** whose
    Rust benchmark binary could not run are re-converted (via LLM accuracy
    repair or a full strategy re-conversion), and the loop repeats up to
    *max_retries* additional times.

    Parameters
    ----------
    run_dir:
        Timestamped snapshot directory where the crate is scaffolded.
    crate_name:
        Name for the generated Cargo crate.
    functions_to_convert:
        Ordered list of (lower-cased) function names to convert.
    rust_sources:
        Initial mapping of function name → Rust source code.
    conversion_results:
        Initial list of :class:`ConversionResult` objects (one per function).
    source_map:
        Mapping of function name → Path to Fortran source file.
    routine_map:
        Mapping of uppercase function name → :class:`FortranRoutine` object.
    llm:
        Configured :class:`LLMClient` instance (may be unavailable).
    strategy:
        The active :class:`ConversionStrategy` instance used to re-convert.
    max_retries:
        Maximum number of *additional* attempts after the first.  Default: 3.
    log:
        Optional ``callable(str)`` for progress messages.  Defaults to a no-op.
    run_scaffold_build:
        Optional override for the scaffold+build+repair step (used in tests).

    Returns
    -------
    PostConversionState
        Dataclass containing all pipeline outputs: crate_dir, build_ok,
        test_ok, accuracy_results, bench_results, updated rust_sources,
        updated conversion_results, and the number of retries performed.
    """
    from fortran_to_rust.benchmarker import run_benchmark
    from fortran_to_rust.rust_project import (
        build_crate,
        repair_crate_with_llm,
        scaffold_crate,
        test_crate,
    )
    from fortran_to_rust.test_harness import run_accuracy_check

    _log = log or (lambda msg: None)

    sources = dict(rust_sources)
    conv_results = list(conversion_results)
    crate_dir: Optional[Path] = None
    build_ok = False
    test_ok = False
    accuracy_results: List["AccuracyResult"] = []
    bench_results: List["BenchResult"] = []
    retry_count = 0

    for attempt in range(max_retries + 1):
        is_retry = attempt > 0
        if is_retry:
            _log(
                f"\n[bold yellow]↩  Pipeline retry {attempt}/{max_retries} — "
                f"re-scaffolding and re-checking…[/bold yellow]"
            )

        # ── Step: Scaffold ────────────────────────────────────────────────────
        crate_dir = scaffold_crate(run_dir, crate_name, sources)
        if not is_retry:
            _log(f"  Crate: {crate_dir}")

        # ── Step: Build ───────────────────────────────────────────────────────
        if run_scaffold_build is not None:
            build_ok, build_out, sources = run_scaffold_build(crate_dir, sources)
        else:
            build_ok, build_out = build_crate(crate_dir)
            _log(f"  cargo build: {'✅ passed' if build_ok else '⚠ finished with errors'}")

            if not build_ok and llm.is_available and "cargo not found" not in build_out:
                _log("  [yellow]⚙  Build failed — starting LLM repair loop…[/yellow]")
                build_ok, build_out, sources = repair_crate_with_llm(
                    crate_dir,
                    sources,
                    llm,
                    progress_callback=_log,
                )
                _log(
                    f"  cargo build (after LLM repair): "
                    f"{'✅ passed' if build_ok else '⚠ finished with errors'}"
                )

        # ── Step: Test ────────────────────────────────────────────────────────
        test_ok, _test_out = test_crate(crate_dir)
        if not is_retry:
            _log(f"  cargo test:  {'✅ passed' if test_ok else '⚠ finished with errors'}")

        # ── Step: Accuracy ────────────────────────────────────────────────────
        accuracy_results = []
        for fn in functions_to_convert:
            src_path = source_map.get(fn)
            routine = routine_map.get(fn.upper())
            acc = run_accuracy_check(
                fn, src_path, crate_dir if build_ok else None, routine=routine,
                fortran_ref_dir=fortran_ref_dir,
                datasets_dir=datasets_dir,
            )
            accuracy_results.append(acc)
            status = "✅" if acc.passed else "⚠"
            err_str = f"  max_err={acc.max_abs_error:.2e}" if acc.max_abs_error else ""
            _log(f"  {status} {fn}{err_str}")
            for d in acc.details:
                _log(f"  {d}")

        # ── Step: Benchmark ───────────────────────────────────────────────────
        bench_results = []
        for fn in functions_to_convert:
            src_path = source_map.get(fn)
            routine = routine_map.get(fn.upper())
            bench = run_benchmark(
                fn, src_path, crate_dir if build_ok else None, routine=routine,
                fortran_ref_dir=fortran_ref_dir,
                datasets_dir=datasets_dir,
            )
            bench_results.append(bench)
            _log(f"  {fn}: {bench.summary}")
            for d in bench.details:
                _log(f"  {d}")

        # ── Decide whether to retry ───────────────────────────────────────────
        failed_acc = _failed_accuracy_fns(accuracy_results)
        failed_bench = _failed_bench_fns(bench_results)
        failing_fns = list(dict.fromkeys(failed_acc + failed_bench))

        if not failing_fns:
            break  # All checks passed — no need to retry.

        if attempt >= max_retries:
            _log(
                f"\n[yellow]⚠  Max retries ({max_retries}) reached. "
                f"Remaining failures: {', '.join(failing_fns)}[/yellow]"
            )
            break

        retry_count += 1
        _log(
            f"\n[bold yellow]⚠  Checks failed for: {', '.join(failing_fns)}. "
            f"Re-converting and retrying (attempt {attempt + 1}/{max_retries})…[/bold yellow]"
        )

        accuracy_map = {acc.function_name.lower(): acc for acc in accuracy_results}
        sources, conv_results = _repair_functions(
            failing_fns=failing_fns,
            rust_sources=sources,
            conversion_results=conv_results,
            source_map=source_map,
            routine_map=routine_map,
            accuracy_map=accuracy_map,
            llm=llm,
            strategy=strategy,
            log=_log,
        )

    return PostConversionState(
        crate_dir=crate_dir,
        build_ok=build_ok,
        test_ok=test_ok,
        accuracy_results=accuracy_results,
        bench_results=bench_results,
        rust_sources=sources,
        conversion_results=conv_results,
        retry_count=retry_count,
    )
