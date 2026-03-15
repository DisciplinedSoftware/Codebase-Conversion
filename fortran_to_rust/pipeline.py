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
    from fortran_to_rust.strategies.base import ConversionResult
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


def _failed_accuracy_fns(
    accuracy_results: List["AccuracyResult"],
    exhausted_fns: "set[str]",
) -> List[str]:
    """Return lower-cased names of functions whose accuracy check failed.

    Functions in *exhausted_fns* are excluded — their conversion step already
    exhausted all repair retries and they should not be retried by the pipeline.
    """
    return [
        acc.function_name.lower()
        for acc in accuracy_results
        if not acc.passed and acc.function_name.lower() not in exhausted_fns
    ]


def _failed_bench_fns(
    bench_results: List["BenchResult"],
    exhausted_fns: "set[str]",
) -> List[str]:
    """Return lower-cased names of functions whose Rust benchmark could not run.

    A *slow* Rust result is not treated as a failure — only results where the
    Rust binary failed to compile or execute (``rust_time_ms is None``).
    Functions in *exhausted_fns* are excluded for the same reason as above.
    """
    return [
        b.function_name.lower()
        for b in bench_results
        if b.rust_time_ms is None and not b.error_message
        and b.function_name.lower() not in exhausted_fns
    ]


def _repair_functions(
    failing_fns: List[str],
    rust_sources: Dict[str, str],
    conversion_results: List["ConversionResult"],
    source_map: Dict[str, Path],
    routine_map: Dict[str, "FortranRoutine"],
    accuracy_map: Dict[str, "AccuracyResult"],
    llm: "LLMClient",
    log: Callable[[str], None],
) -> Tuple[Dict[str, str], List["ConversionResult"]]:
    """Apply targeted accuracy repair to *failing_fns*.

    Each failing function gets one ``llm.repair_accuracy()`` call using the
    specific numerical-error context from its last accuracy check.  If the LLM
    is unavailable or repair fails, the function's source is left unchanged so
    the pipeline can re-validate and report the failure.

    Correction is kept *step-specific*: this function never re-runs
    ``strategy.convert()`` — that would conflate the conversion step's
    correction loop with the pipeline's revalidation pass.
    """
    from fortran_to_rust.rust_project import _ensure_top_level_pub_fn

    sources = dict(rust_sources)
    results = list(conversion_results)
    result_index = {r.routine_name.lower(): i for i, r in enumerate(results)}

    for fn_lower in failing_fns:
        routine = routine_map.get(fn_lower.upper())
        if not routine:
            log(f"  [yellow]⚠[/yellow]  {fn_lower}: no parsed routine — skipping repair.")
            continue

        acc = accuracy_map.get(fn_lower)
        current_source = sources.get(fn_lower, "")

        if not llm.is_available:
            log(f"  [yellow]⚠[/yellow]  {fn_lower}: LLM unavailable — cannot repair accuracy.")
            continue

        if acc is None or acc.max_abs_error is None:
            log(f"  [yellow]⚠[/yellow]  {fn_lower}: no accuracy error data — cannot repair.")
            continue

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
            continue

        if repaired_source:
            sources[fn_lower] = repaired_source
            idx = result_index.get(fn_lower)
            if idx is not None:
                old = results[idx]
                from fortran_to_rust.strategies.base import ConversionResult
                results[idx] = ConversionResult(
                    routine_name=old.routine_name,
                    rust_source=repaired_source,
                    success=False,  # pipeline re-validates; success is set after re-check
                    strategy_used=old.strategy_used + " (accuracy-repair)",
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
    # Kept for backward compatibility; no longer used internally.
    # Each pipeline step owns its own correction loop; re-conversion is not
    # triggered during the revalidation pass.
    strategy: object = None,
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
      2. ``cargo build --release`` (+ LLM build repair — the build step's correction).
      3. ``cargo test``.
      4. Numerical accuracy check for every function.
      5. Performance benchmark for every function.

    When accuracy or benchmark failures are detected, the pipeline applies a
    *targeted* accuracy repair (``llm.repair_accuracy()``) for each failing
    function — the accuracy step's own correction loop — and then re-runs ALL
    validation steps from the top.  Re-conversion via ``strategy.convert()`` is
    intentionally not performed here: the conversion step's correction loop
    belongs inside the strategy, not in this revalidation pass.

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
        Deprecated / ignored.  Kept for backward-compatible call-sites.
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

    # Functions whose conversion step already exhausted all accuracy repair
    # retries (success=False).  These are permanently failed and must not be
    # fed back into the pipeline's revalidation loop.
    exhausted_fns: set = {
        r.routine_name.lower()
        for r in conv_results
        if not r.success
    }
    if exhausted_fns:
        _log(
            f"  [yellow]⚠  Conversion failed (accuracy exhausted) for: "
            f"{', '.join(sorted(exhausted_fns))} — skipping pipeline retry for these.[/yellow]"
        )

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

                # After build repair changes code, immediately validate accuracy and
                # apply LLM correction for any failures before moving to the next step.
                if build_ok:
                    _log("  [yellow]⚙  Validating accuracy after build repair…[/yellow]")
                    inline_acc = [
                        run_accuracy_check(
                            fn, source_map.get(fn), crate_dir,
                            routine=routine_map.get(fn.upper()),
                            fortran_ref_dir=fortran_ref_dir,
                            datasets_dir=datasets_dir,
                        )
                        for fn in functions_to_convert
                        if fn not in exhausted_fns
                    ]
                    inline_failed = [acc.function_name.lower() for acc in inline_acc if not acc.passed]
                    if inline_failed:
                        _log(
                            f"  [yellow]⚙  Accuracy failed after build repair for: "
                            f"{', '.join(inline_failed)} — applying LLM correction…[/yellow]"
                        )
                        inline_acc_map = {acc.function_name.lower(): acc for acc in inline_acc}
                        sources, conv_results = _repair_functions(
                            failing_fns=inline_failed,
                            rust_sources=sources,
                            conversion_results=conv_results,
                            source_map=source_map,
                            routine_map=routine_map,
                            accuracy_map=inline_acc_map,
                            llm=llm,
                            log=_log,
                        )
                        # Re-scaffold and re-build with the accuracy-corrected sources
                        crate_dir = scaffold_crate(run_dir, crate_name, sources)
                        build_ok, build_out = build_crate(crate_dir)
                        _log(
                            f"  cargo build (after accuracy correction): "
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
        failed_acc = _failed_accuracy_fns(accuracy_results, exhausted_fns)
        failed_bench = _failed_bench_fns(bench_results, exhausted_fns)
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
            f"Applying targeted accuracy repair and re-validating "
            f"(attempt {attempt + 1}/{max_retries})…[/bold yellow]"
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
