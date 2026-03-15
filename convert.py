#!/usr/bin/env python3
"""fortran-to-rust — Automated Fortran→Rust conversion pipeline.

Usage
-----
    python convert.py [options]

Options
-------
  --output-dir PATH     Directory for downloaded sources, generated Rust code,
                        and reports.  Defaults to ./output.
  --non-interactive     Run conversion without prompts.  Useful for CI/demos.
  --functions NAMES     Comma-separated function names to convert, OR a positive
                        integer for a random sample of that size.
                        Defaults to "dgemm" when --non-interactive is used.
  --strategy {1,2,3}   Conversion strategy (default: 3 = Hybrid Rule-Based).
  --compare             Run all 3 strategies in parallel, each in its own
                        sub-directory, then emit a side-by-side comparison
                        report.  Implies --non-interactive.

Environment variables
---------------------
  LLM_PROVIDER    copilot | openai | openai_compatible  (default: copilot)
  LLM_API_KEY     Bearer token / API key
  LLM_MODEL       Model override (default: gpt-4o)
  LLM_BASE_URL    Base URL for openai_compatible provider
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env if present (best-effort; no hard dependency on python-dotenv)
# ---------------------------------------------------------------------------
_env_file = Path(".env")
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())


def _make_run_dir(base: Path) -> Path:
    """Create and return a timestamped run directory inside *base*.

    Each invocation produces a fresh snapshot directory:
        <base>/report_YYYYMMDD_HHMMSS/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"report_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convert",
        description="Automated Fortran-to-Rust conversion pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        metavar="PATH",
        help=(
            "Base directory for caches and run snapshots (default: ./output). "
            "Each run creates output/<base>/report_YYYYMMDD_HHMMSS/ "
            "so previous runs are preserved."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run conversion without prompts.",
    )
    parser.add_argument(
        "--functions",
        default="dgemm",
        metavar="NAMES_OR_SIZE",
        help=(
            "Comma-separated list of Fortran function names to convert, "
            "OR a positive integer to pick that many functions from the BLAS list. "
            "Default: 'dgemm'."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["1", "2", "3"],
        default="3",
        help="Conversion strategy: 1=LLM-First, 2=Agentic, 3=Hybrid (default: 3).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run all 3 strategies in parallel, each in its own sub-directory "
            "under output/compare_YYYYMMDD_HHMMSS/, then emit a side-by-side "
            "comparison report.  Implies --non-interactive."
        ),
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        _run_compare(output_dir, args.functions)
    elif args.non_interactive:
        _run_non_interactive(output_dir, args.functions, args.strategy)
    else:
        from fortran_to_rust.cli import run
        run(output_dir)


def _parse_functions_arg(raw: str) -> list[str]:
    """Parse the --functions argument.

    Accepts either:
    - a plain positive integer → return the first N functions from BLAS_FUNCTIONS
    - a comma-separated list of names → return those names (lower-cased)
    """
    from fortran_to_rust.fetcher import BLAS_FUNCTIONS

    raw = raw.strip()
    try:
        n = int(raw)
        if n > len(BLAS_FUNCTIONS):
            print(
                f"Warning: requested {n} functions but only "
                f"{len(BLAS_FUNCTIONS)} are known; using all {len(BLAS_FUNCTIONS)}."
            )
            n = len(BLAS_FUNCTIONS)
        n = max(1, n)
        return [f.lower() for f in BLAS_FUNCTIONS[:n]]
    except ValueError:
        pass
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _make_progress_callback(fn_name: str, messages: list[str], console):  # type: ignore[type-arg]
    """Return a progress callback that tags messages with *fn_name*."""
    def cb(msg: str) -> None:
        console.print(f"  [{fn_name}] {msg}")
        messages.append(msg)
    return cb


def _make_stream_cb(fn_name: str, console):  # type: ignore[type-arg]
    """Return a (callback, state) pair for an LLM Progress spinner.

    The callback increments the token counter and updates the spinner
    description on each arriving chunk — no raw LLM output is printed.
    Callers must set state['progress'] and state['task'] before the LLM call.
    """
    state: dict = {"progress": None, "task": None, "tokens": 0}

    def cb(chunk: str) -> None:
        state["tokens"] += 1
        p, t = state.get("progress"), state.get("task")
        if p is not None and t is not None:
            p.update(
                t,
                description=(
                    f"  [{fn_name}] [dim]LLM thinking… ({state['tokens']} tokens)[/dim]"
                ),
            )

    return cb, state


def _run_with_spinner(label: str, console, fn, *args, **kwargs):  # type: ignore[type-arg]
    """Run *fn(*args, **kwargs)* with a transient Rich spinner showing *label*."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    with Progress(
        SpinnerColumn(),
        TextColumn(f"  [dim]{label}…[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("")
        return fn(*args, **kwargs)


def _run_non_interactive(output_dir: Path, functions_arg: str, strategy_key: str) -> None:
    """Headless pipeline: convert the requested functions and produce a report."""
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table

    from fortran_to_rust.benchmarker import run_benchmark
    from fortran_to_rust.call_graph import build_call_graph, render_graph
    from fortran_to_rust.fetcher import BLAS_FUNCTIONS, fetch_blas
    from fortran_to_rust.llm_client import LLMClient
    from fortran_to_rust.parser import parse_file
    from fortran_to_rust.reporter import generate_report
    from fortran_to_rust.rust_project import build_crate, scaffold_crate, test_crate
    from fortran_to_rust.strategies import STRATEGY_MAP, STRATEGY_NAMES
    from fortran_to_rust.test_harness import run_accuracy_check

    console = Console()
    console.print(Rule("[bold cyan]Fortran-to-Rust  (non-interactive)[/bold cyan]"))

    # Each run gets its own snapshot directory; BLAS sources are cached at output_dir.
    run_dir = _make_run_dir(output_dir)
    console.print(f"  [dim]Run directory: {run_dir}[/dim]")

    # --- 1. Resolve function list ---
    functions_to_convert = _parse_functions_arg(functions_arg)
    # Always pull in support routines required by most BLAS functions
    support_fns = ["lsame", "xerbla"]
    fetch_list = list(dict.fromkeys(functions_to_convert + support_fns))

    console.print(
        f"\n[bold]Converting:[/bold] {', '.join(functions_to_convert)}  "
        f"[dim](strategy {strategy_key} — {STRATEGY_NAMES[strategy_key]})[/dim]"
    )

    # --- 2. Fetch sources ---
    console.print("\n[bold]Fetching BLAS sources…[/bold]")
    source_map = fetch_blas(output_dir, functions=fetch_list)

    missing = [fn for fn in functions_to_convert if fn not in source_map]
    if missing:
        console.print(f"[yellow]⚠[/yellow] Could not fetch: {', '.join(missing)}")
        functions_to_convert = [fn for fn in functions_to_convert if fn in source_map]
    if not functions_to_convert:
        console.print("[red]No fetchable functions — aborting.[/red]")
        sys.exit(1)

    # --- 3. Parse ---
    console.print("[bold]Parsing…[/bold]")
    all_routines = []
    for path in source_map.values():
        all_routines.extend(parse_file(path))

    routine_map = {r.name.upper(): r for r in all_routines}
    for fn in functions_to_convert:
        r = routine_map.get(fn.upper())
        if r:
            console.print(f"  Parsed {r.name} ({r.line_count} lines)")
        else:
            console.print(f"  [yellow]⚠[/yellow] {fn}: routine not found in parsed source")

    graph = build_call_graph(all_routines)
    console.print(f"  [dim]Call graph:[/dim]\n{render_graph(graph)}")

    # --- 4. Convert ---
    console.print(f"\n[bold]Converting with strategy {strategy_key}…[/bold]")
    StrategyClass = STRATEGY_MAP[strategy_key]
    llm = LLMClient()
    strategy = StrategyClass(output_dir, llm=llm)

    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    conversion_results = []
    for fn in functions_to_convert:
        routine = routine_map.get(fn.upper())
        if not routine:
            console.print(f"  [yellow]⚠[/yellow] {fn}: skipping (no parsed routine).")
            continue

        messages: list[str] = []
        stream_cb, stream_state = _make_stream_cb(fn, console)
        llm.stream_callback = stream_cb
        cb = _make_progress_callback(fn, messages, console)

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"  [{fn}] [dim]LLM thinking…[/dim]")
            stream_state["progress"] = progress
            stream_state["task"] = task
            result = strategy.convert(routine, progress_callback=cb)

        conversion_results.append(result)
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        console.print(f"  {status} {fn} — {result.strategy_used}")

    # --- 5. Scaffold crate ---
    console.print("\n[bold]Scaffolding Rust crate…[/bold]")
    rust_sources = {
        r.routine_name: r.rust_source
        for r in conversion_results
        if r.rust_source
    }
    crate_name = "blas_converted"
    crate_dir = scaffold_crate(run_dir, crate_name, rust_sources)
    console.print(f"  Crate: {crate_dir}")

    build_ok, build_out = _run_with_spinner("cargo build --release", console, build_crate, crate_dir)
    console.print(f"  cargo build: {'✅ passed' if build_ok else '⚠ finished with errors'}")

    test_ok, test_out = _run_with_spinner("cargo test", console, test_crate, crate_dir)
    console.print(f"  cargo test:  {'✅ passed' if test_ok else '⚠ finished with errors'}")

    # --- 6. Accuracy ---
    console.print("\n[bold]Accuracy checks…[/bold]")
    accuracy_results = []
    for fn in functions_to_convert:
        src_path = source_map.get(fn)
        routine = routine_map.get(fn.upper())
        acc = run_accuracy_check(
            fn, src_path, crate_dir if build_ok else None, routine=routine
        )
        accuracy_results.append(acc)
        status = "✅" if acc.passed else "⚠"
        err_str = f"  max_err={acc.max_abs_error:.2e}" if acc.max_abs_error else ""
        console.print(f"  {status} {fn}{err_str}")
        for d in acc.details:
            console.print(f"  {d}")

    # --- 7. Benchmark ---
    console.print("\n[bold]Benchmarks…[/bold]")
    bench_results = []
    for fn in functions_to_convert:
        src_path = source_map.get(fn)
        routine = routine_map.get(fn.upper())
        bench = run_benchmark(
            fn, src_path, crate_dir if build_ok else None, routine=routine
        )
        bench_results.append(bench)
        console.print(f"  {fn}: {bench.summary}")
        for d in bench.details:
            console.print(f"  {d}")

    # --- 8. Report ---
    console.print("\n[bold]Generating report…[/bold]")
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
    console.print(f"  [green]Markdown:[/green]  {md_path}")
    console.print(f"  [green]HTML:[/green]      {html_path}")
    console.print(Rule(style="cyan"))


def _run_compare(output_dir: Path, functions_arg: str) -> None:
    """Run all 3 strategies in parallel and produce a side-by-side comparison."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.rule import Rule

    from fortran_to_rust.call_graph import build_call_graph, render_graph
    from fortran_to_rust.compare import (
        print_comparison_table,
        run_all_parallel,
    )
    from fortran_to_rust.fetcher import fetch_blas
    from fortran_to_rust.parser import parse_file
    from fortran_to_rust.strategies import STRATEGY_NAMES

    console = Console()
    console.print(Rule("[bold cyan]Fortran-to-Rust  (compare all strategies)[/bold cyan]"))

    run_dir = _make_run_dir(output_dir)
    console.print(f"  [dim]Run directory: {run_dir}[/dim]")

    # --- 1. Resolve function list ---
    functions_to_convert = _parse_functions_arg(functions_arg)
    support_fns = ["lsame", "xerbla"]
    fetch_list = list(dict.fromkeys(functions_to_convert + support_fns))

    console.print(
        f"\n[bold]Converting:[/bold] {', '.join(functions_to_convert)}  "
        "[dim](all 3 strategies in parallel)[/dim]"
    )

    # --- 2. Fetch sources (shared across all strategies) ---
    console.print("\n[bold]Fetching BLAS sources…[/bold]")
    source_map = fetch_blas(output_dir, functions=fetch_list)

    missing = [fn for fn in functions_to_convert if fn not in source_map]
    if missing:
        console.print(f"[yellow]⚠[/yellow] Could not fetch: {', '.join(missing)}")
        functions_to_convert = [fn for fn in functions_to_convert if fn in source_map]
    if not functions_to_convert:
        console.print("[red]No fetchable functions — aborting.[/red]")
        sys.exit(1)

    # --- 3. Parse (shared) ---
    console.print("[bold]Parsing…[/bold]")
    all_routines = []
    for path in source_map.values():
        all_routines.extend(parse_file(path))
    routine_map = {r.name.upper(): r for r in all_routines}

    graph = build_call_graph(all_routines)
    console.print(f"  [dim]Call graph:[/dim]\n{render_graph(graph)}")

    # --- 4. Run all 3 strategies in parallel ---
    console.print("\n[bold]Running all 3 strategies in parallel…[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        tasks = {
            key: progress.add_task(
                f"  [cyan]Strategy {key}[/cyan] ({STRATEGY_NAMES[key]}): Starting…"
            )
            for key in ("1", "2", "3")
        }

        def _make_update(key: str):
            def _update(msg: str) -> None:
                progress.update(
                    tasks[key],
                    description=(
                        f"  [cyan]Strategy {key}[/cyan] ({STRATEGY_NAMES[key]}): {msg}"
                    ),
                )
            return _update

        all_results, md_path, html_path = run_all_parallel(
            output_dir=output_dir,
            run_dir=run_dir,
            functions_to_convert=functions_to_convert,
            source_map=source_map,
            routine_map=routine_map,
            library="BLAS",
            progress_update=lambda key, msg: _make_update(key)(msg),
        )

    # --- 5. Print comparison table ---
    console.print(Rule("[bold cyan]Comparison Summary[/bold cyan]"))
    print_comparison_table(console, all_results)

    # --- 6. Show report locations ---
    console.print(f"\n  [green]Markdown:[/green]  {md_path}")
    console.print(f"  [green]HTML:[/green]      {html_path}")
    console.print(Rule(style="cyan"))


if __name__ == "__main__":
    main()
