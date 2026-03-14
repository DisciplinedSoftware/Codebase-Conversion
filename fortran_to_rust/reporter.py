"""Post-conversion report generator.

Produces a Markdown report at ``output_dir/reports/<timestamp>_report.md``
summarising:
- Which functions were converted and by which strategy
- Accuracy results (max / mean absolute error)
- Performance comparison
- Compiler warnings and any LLM repair rounds needed
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fortran_to_rust.benchmarker import BenchResult
from fortran_to_rust.strategies.base import ConversionResult
from fortran_to_rust.test_harness import AccuracyResult


def generate_report(
    output_dir: Path,
    library: str,
    strategy_name: str,
    conversion_results: List[ConversionResult],
    accuracy_results: List[AccuracyResult],
    bench_results: List[BenchResult],
    crate_dir: Optional[Path] = None,
    build_ok: bool = False,
    test_ok: bool = False,
) -> Path:
    """Write a Markdown report and return its path."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"{ts}_report.md"

    lines: List[str] = []
    _section = lambda title: lines.extend([f"\n## {title}\n"])

    # ---- Header ----
    lines.append(f"# Fortran-to-Rust Conversion Report")
    lines.append(f"\n**Library:** {library}  ")
    lines.append(f"**Strategy:** {strategy_name}  ")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    if crate_dir:
        lines.append(f"**Crate:** `{crate_dir}`  ")

    # ---- Summary table ----
    _section("Conversion Summary")
    lines.append("| Function | Strategy Used | Lines | Repair Rounds | Status |")
    lines.append("|----------|--------------|-------|---------------|--------|")
    for r in conversion_results:
        status = "✅ OK" if r.success else "❌ Failed"
        lines.append(
            f"| `{r.routine_name}` | {r.strategy_used} "
            f"| {_get_line_count(r)} | {r.repair_rounds} | {status} |"
        )

    # ---- Build / test ----
    _section("Build & Test")
    lines.append(f"- `cargo build --release`: {'✅ passed' if build_ok else '❌ failed / skipped'}")
    lines.append(f"- `cargo test`:            {'✅ passed' if test_ok else '❌ failed / skipped'}")

    # ---- Accuracy ----
    _section("Numerical Accuracy")
    if accuracy_results:
        lines.append("| Function | Tests | Failed | Max Abs Error | Mean Abs Error | Result |")
        lines.append("|----------|-------|--------|---------------|----------------|--------|")
        for a in accuracy_results:
            max_e = f"{a.max_abs_error:.2e}" if a.max_abs_error is not None else "N/A"
            mean_e = f"{a.mean_abs_error:.2e}" if a.mean_abs_error is not None else "N/A"
            result_sym = "✅" if a.passed else "❌"
            lines.append(
                f"| `{a.function_name}` | {a.num_test_cases} | {a.failed_cases} "
                f"| {max_e} | {mean_e} | {result_sym} |"
            )
        for a in accuracy_results:
            if a.details:
                lines.append(f"\n**{a.function_name} details:**")
                for d in a.details:
                    lines.append(d)
        if any(a.error_message for a in accuracy_results):
            lines.append("\n**Notes:**")
            for a in accuracy_results:
                if a.error_message:
                    lines.append(f"- `{a.function_name}`: {a.error_message}")
    else:
        lines.append("_No accuracy results available._")

    # ---- Performance ----
    _section("Performance")
    if bench_results:
        lines.append("| Function | Fortran (ms/call) | Rust (ms/call) | Speedup |")
        lines.append("|----------|-------------------|----------------|---------|")
        for b in bench_results:
            f_ms = f"{b.fortran_time_ms:.3f}" if b.fortran_time_ms else "N/A"
            r_ms = f"{b.rust_time_ms:.3f}" if b.rust_time_ms else "N/A"
            sp = f"{b.speedup:.2f}×" if b.speedup else "N/A"
            lines.append(f"| `{b.function_name}` | {f_ms} | {r_ms} | {sp} |")
        for b in bench_results:
            if b.details:
                lines.append(f"\n**{b.function_name} details:**")
                for d in b.details:
                    lines.append(d)
    else:
        lines.append("_No benchmark results available._")

    # ---- Warnings ----
    all_warnings = [
        (r.routine_name, w)
        for r in conversion_results
        for w in r.warnings
    ]
    if all_warnings:
        _section("Warnings")
        for fn, w in all_warnings:
            lines.append(f"- **{fn}**: {w}")

    # ---- Compiler errors (if any) ----
    compiler_issues = [r for r in conversion_results if r.compiler_errors]
    if compiler_issues:
        _section("Compiler Diagnostics")
        for r in compiler_issues:
            lines.append(f"\n### `{r.routine_name}`")
            for i, err in enumerate(r.compiler_errors, 1):
                lines.append(f"\n_Round {i}:_\n```\n{err[:600]}\n```")

    # ---- Generated code snippets ----
    _section("Generated Code")
    for r in conversion_results:
        if r.rust_source:
            snippet = r.rust_source[:1500]
            truncated = "…" if len(r.rust_source) > 1500 else ""
            lines.append(f"\n### `{r.routine_name}`\n")
            lines.append(f"```rust\n{snippet}{truncated}\n```")

    report_path.write_text("\n".join(lines))
    return report_path


def _get_line_count(r: ConversionResult) -> str:
    if r.rust_source:
        return str(len(r.rust_source.splitlines()))
    return "—"
