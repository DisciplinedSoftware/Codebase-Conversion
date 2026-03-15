"""Parallel strategy comparison utilities for the Fortran-to-Rust pipeline.

Provides ``run_strategy_worker`` (runs the full pipeline for one strategy),
``run_all_parallel`` (executes all three strategies concurrently), and helpers
for writing/printing comparison reports.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def make_compare_dir(base: Path) -> Path:
    """Create and return a timestamped compare directory inside *base*.

    Structure::

        <base>/compare_YYYYMMDD_HHMMSS/
            strategy_1/
            strategy_2/
            strategy_3/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_dir = base / f"compare_{ts}"
    compare_dir.mkdir(parents=True, exist_ok=True)
    return compare_dir


def run_strategy_worker(
    output_dir: Path,
    run_dir: Path,
    functions_to_convert: List[str],
    source_map: Dict,
    routine_map: Dict,
    strategy_key: str,
    step_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run convert → scaffold → build → test → accuracy → benchmark → report.

    Each invocation is self-contained: it creates its own ``LLMClient``,
    scaffolds a Cargo crate inside *run_dir*, and generates an HTML/MD report.

    Returns a result dict with all artifacts and metrics, or a dict with an
    ``"error"`` key if an unrecoverable failure occurs.
    """
    from fortran_to_rust.benchmarker import run_benchmark
    from fortran_to_rust.llm_client import LLMClient
    from fortran_to_rust.reporter import generate_report
    from fortran_to_rust.rust_project import build_crate, scaffold_crate, test_crate
    from fortran_to_rust.strategies import STRATEGY_MAP, STRATEGY_NAMES
    from fortran_to_rust.test_harness import run_accuracy_check

    def step(msg: str) -> None:
        if step_callback:
            step_callback(msg)

    try:
        step("Converting…")
        StrategyClass = STRATEGY_MAP[strategy_key]
        llm = LLMClient()
        strategy_obj = StrategyClass(output_dir, llm=llm)
        conversion_results = []
        for fn in functions_to_convert:
            routine = routine_map.get(fn.upper())
            if not routine:
                continue
            step(f"Converting {fn}…")
            result = strategy_obj.convert(routine)
            conversion_results.append(result)

        step("Scaffolding crate…")
        rust_sources = {
            r.routine_name: r.rust_source
            for r in conversion_results
            if r.rust_source
        }
        crate_dir = scaffold_crate(run_dir, "blas_converted", rust_sources)

        step("cargo build…")
        build_ok, _build_out = build_crate(crate_dir)

        step("cargo test…")
        test_ok, _test_out = test_crate(crate_dir)

        step("Accuracy checks…")
        accuracy_results = []
        for fn in functions_to_convert:
            src_path = source_map.get(fn)
            routine = routine_map.get(fn.upper())
            acc = run_accuracy_check(
                fn, src_path, crate_dir if build_ok else None, routine=routine
            )
            accuracy_results.append(acc)

        step("Benchmarking…")
        bench_results = []
        for fn in functions_to_convert:
            src_path = source_map.get(fn)
            routine = routine_map.get(fn.upper())
            bench = run_benchmark(
                fn, src_path, crate_dir if build_ok else None, routine=routine
            )
            bench_results.append(bench)

        step("Generating report…")
        md_path, html_path = generate_report(
            output_dir=run_dir,
            library="BLAS",
            strategy_name=STRATEGY_NAMES[strategy_key],
            conversion_results=conversion_results,
            accuracy_results=accuracy_results,
            bench_results=bench_results,
            crate_dir=crate_dir,
            build_ok=build_ok,
            test_ok=test_ok,
            open_browser=False,
        )

        step("Done ✓")
        return {
            "strategy_key": strategy_key,
            "strategy_name": STRATEGY_NAMES[strategy_key],
            "run_dir": run_dir,
            "conversion_results": conversion_results,
            "accuracy_results": accuracy_results,
            "bench_results": bench_results,
            "build_ok": build_ok,
            "test_ok": test_ok,
            "html_path": html_path,
            "md_path": md_path,
            "error": None,
        }

    except Exception as exc:
        step(f"Failed: {exc}")
        from fortran_to_rust.strategies import STRATEGY_NAMES

        return {
            "strategy_key": strategy_key,
            "strategy_name": STRATEGY_NAMES.get(strategy_key, strategy_key),
            "run_dir": run_dir,
            "conversion_results": [],
            "accuracy_results": [],
            "bench_results": [],
            "build_ok": False,
            "test_ok": False,
            "html_path": None,
            "md_path": None,
            "error": str(exc),
        }


def run_all_parallel(
    output_dir: Path,
    compare_dir: Path,
    functions_to_convert: List[str],
    source_map: Dict,
    routine_map: Dict,
    progress_update: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, dict]:
    """Run all three conversion strategies concurrently.

    Each strategy gets its own subdirectory under *compare_dir*::

        compare_dir/strategy_1/
        compare_dir/strategy_2/
        compare_dir/strategy_3/

    Args:
        progress_update: ``callable(strategy_key, message)`` invoked on every
            pipeline step change so callers can update a live progress display.

    Returns:
        Mapping of strategy_key → result dict (see ``run_strategy_worker``).
    """
    strategy_dirs = {
        key: compare_dir / f"strategy_{key}" for key in ("1", "2", "3")
    }
    for d in strategy_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, dict] = {}

    def run_one(key: str) -> None:
        def step_cb(msg: str) -> None:
            if progress_update:
                progress_update(key, msg)

        all_results[key] = run_strategy_worker(
            output_dir=output_dir,
            run_dir=strategy_dirs[key],
            functions_to_convert=functions_to_convert,
            source_map=source_map,
            routine_map=routine_map,
            strategy_key=key,
            step_callback=step_cb,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_one, key) for key in ("1", "2", "3")]
        concurrent.futures.wait(futures)

    return all_results


def print_comparison_table(console, all_results: Dict[str, dict]) -> None:
    """Render a Rich summary table of all three strategy results to *console*."""
    from rich.table import Table

    from fortran_to_rust.strategies import STRATEGY_NAMES

    table = Table(title="Strategy Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Strategy", min_width=35)
    table.add_column("Build", justify="center")
    table.add_column("Tests", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Max Error", justify="right")
    table.add_column("Avg Speedup", justify="right")

    for key in ("1", "2", "3"):
        res = all_results.get(key, {})
        name = STRATEGY_NAMES.get(key, key)
        if res.get("error") and not res.get("conversion_results"):
            table.add_row(
                f"[cyan]{key}[/cyan] {name}",
                "❌", "❌", "❌", "—", "—",
            )
            continue
        build = "✅" if res.get("build_ok") else "❌"
        tests = "✅" if res.get("test_ok") else "❌"
        acc_results = res.get("accuracy_results", [])
        acc_passed = all(a.passed for a in acc_results if a.max_abs_error is not None)
        accuracy = "✅" if (acc_results and acc_passed) else ("❌" if acc_results else "—")
        max_err = max(
            (a.max_abs_error for a in acc_results if a.max_abs_error is not None),
            default=None,
        )
        max_err_str = f"{max_err:.2e}" if max_err is not None else "—"
        bench_results = res.get("bench_results", [])
        speedups = [b.speedup for b in bench_results if b.speedup]
        speedup_str = f"{sum(speedups) / len(speedups):.2f}×" if speedups else "—"
        table.add_row(
            f"[cyan]{key}[/cyan] {name}",
            build, tests, accuracy, max_err_str, speedup_str,
        )

    console.print(table)


def write_comparison_report(compare_dir: Path, all_results: Dict[str, dict]) -> Path:
    """Write ``compare_dir/comparison.md`` summarising all three strategies.

    Returns the path to the written file.
    """
    from fortran_to_rust.strategies import STRATEGY_NAMES

    lines = [
        "# Strategy Comparison Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nCompare directory: `{compare_dir}`",
        "\n## Results\n",
        "| Strategy | Build | Tests | Accuracy | Max Error | Avg Speedup |",
        "|----------|-------|-------|----------|-----------|-------------|",
    ]
    for key in ("1", "2", "3"):
        res = all_results.get(key, {})
        name = STRATEGY_NAMES.get(key, key)
        if res.get("error") and not res.get("conversion_results"):
            lines.append(f"| **{key}** {name} | ❌ | ❌ | ❌ | — | — |")
            continue
        build = "✅" if res.get("build_ok") else "❌"
        tests = "✅" if res.get("test_ok") else "❌"
        acc_results = res.get("accuracy_results", [])
        acc_passed = all(a.passed for a in acc_results if a.max_abs_error is not None)
        accuracy = "✅" if (acc_results and acc_passed) else ("❌" if acc_results else "—")
        max_err = max(
            (a.max_abs_error for a in acc_results if a.max_abs_error is not None),
            default=None,
        )
        max_err_str = f"{max_err:.2e}" if max_err is not None else "—"
        bench_results = res.get("bench_results", [])
        speedups = [b.speedup for b in bench_results if b.speedup]
        speedup_str = f"{sum(speedups) / len(speedups):.2f}×" if speedups else "—"
        lines.append(
            f"| **{key}** {name} | {build} | {tests} | {accuracy} | {max_err_str} | {speedup_str} |"
        )

    lines.append("\n## Individual Reports\n")
    for key in ("1", "2", "3"):
        res = all_results.get(key, {})
        name = STRATEGY_NAMES.get(key, key)
        lines.append(f"### Strategy {key}: {name}\n")
        lines.append(f"- **Directory:** `{res.get('run_dir', '—')}`")
        html = res.get("html_path")
        md = res.get("md_path")
        if html:
            lines.append(f"- **HTML Report:** `{html}`")
        if md:
            lines.append(f"- **Markdown Report:** `{md}`")
        if res.get("error"):
            lines.append(f"- **Error:** {res['error']}")
        lines.append("")

    report_path = compare_dir / "comparison.md"
    report_path.write_text("\n".join(lines))
    return report_path
