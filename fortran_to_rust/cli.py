"""Interactive CLI for the Fortran-to-Rust conversion pipeline.

Modelled on the GitHub Copilot CLI experience: rich prompts, live
progress feedback, inline Q&A during the agentic strategy, and a
final report printed to the terminal.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint

from fortran_to_rust.benchmarker import run_benchmark
from fortran_to_rust.call_graph import build_call_graph, render_graph
from fortran_to_rust.fetcher import (
    BLAS_FUNCTIONS,
    fetch_blas,
)
from fortran_to_rust.llm_client import LLMClient, LLMUnavailableError
from fortran_to_rust.parser import parse_file
from fortran_to_rust.reporter import generate_report
from fortran_to_rust.rust_project import build_crate, scaffold_crate, test_crate
from fortran_to_rust.strategies import STRATEGY_MAP, STRATEGY_NAMES
from fortran_to_rust.test_harness import run_accuracy_check

console = Console()


def _print_banner() -> None:
    """Print the application banner in a clean, reliably-rendered Panel."""
    from rich.panel import Panel
    from rich.text import Text

    title = Text.assemble(("Fortran", "bold cyan"), (" → ", "bold white"), ("Rust", "bold cyan"))
    subtitle = Text("Automated Conversion Pipeline  •  v0.1.0", style="dim")
    content = Text.assemble(title, "\n", subtitle)
    console.print(Panel(content, border_style="cyan", padding=(1, 4), expand=False))

_STRATEGY_DESCRIPTIONS = {
    "1": (
        "[bold]LLM-First with Rule Fallback[/bold]\n"
        "  Let an LLM do the heavy lifting; invoke rule-based transpiler only when "
        "the LLM fails to compile.  Requires an LLM API key."
    ),
    "2": (
        "[bold]Agentic Multi-Turn Dialogue[/bold]\n"
        "  A Questioner–Solver agent pair clarifies Fortran semantics before the "
        "Solver finalises the Rust code.  Requires an LLM API key."
    ),
    "3": (
        "[bold]Hybrid Rule-Based + LLM Polish[/bold]\n"
        "  Deterministic f2c skeleton first, then optional LLM polishing pass.  "
        "Works fully offline; LLM polish is applied if a key is available."
    ),
}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(output_dir: Path) -> None:
    """Launch the interactive conversion wizard."""
    _print_banner()
    console.print(Rule(style="cyan"))

    # --- 1. Library selection -----------------------------------------------
    library = _ask_library()

    # --- 2. Fetch source -------------------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 1 — Fetching source code[/bold]", style="cyan"))
    functions_to_convert, source_map = _fetch_and_select(library, output_dir)
    if not functions_to_convert:
        console.print("[red]No functions selected. Exiting.[/red]")
        sys.exit(0)

    # --- 3. Strategy selection -------------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 2 — Choose a conversion strategy[/bold]", style="cyan"))
    strategy_key = _ask_strategy()

    # --- 4. LLM configuration (if needed) -------------------------------------
    llm = _configure_llm(strategy_key)

    # --- 5. Call graph --------------------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 3 — Analysing call graph[/bold]", style="cyan"))
    all_routines = []
    for fn, path in source_map.items():
        all_routines.extend(parse_file(path))
    graph = build_call_graph(all_routines)
    console.print(f"  [dim]Call graph:[/dim]\n{render_graph(graph)}")

    # Build a lookup map: fn_name (upper) → FortranRoutine
    routine_map = {r.name.upper(): r for r in all_routines}

    # --- 6. Convert -----------------------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 4 — Converting Fortran → Rust[/bold]", style="cyan"))
    StrategyClass = STRATEGY_MAP[strategy_key]
    strategy = StrategyClass(output_dir, llm=llm)
    conversion_results = []

    for fn_name in functions_to_convert:
        routine = routine_map.get(fn_name.upper())
        if not routine:
            console.print(f"  [yellow]⚠[/yellow] {fn_name}: no parsed routine found, skipping.")
            continue

        console.print()
        console.print(f"  [bold cyan]Converting:[/bold cyan] {fn_name}  "
                      f"[dim]({routine.line_count} lines)[/dim]")
        console.print(f"  [dim]LLM output ↓[/dim]")

        result = strategy.convert(routine, progress_callback=_make_cb())
        console.print()  # newline after streamed output
        conversion_results.append(result)

        if result.success:
            console.print(f"  [green]✓[/green] {fn_name} converted via {result.strategy_used}")
        else:
            console.print(f"  [red]✗[/red] {fn_name} conversion failed: {result.error}")

    # --- 7. Package into Cargo crate ------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 5 — Packaging Rust crate[/bold]", style="cyan"))
    rust_sources: Dict[str, str] = {
        r.routine_name: r.rust_source
        for r in conversion_results
        if r.rust_source
    }
    crate_name = f"{library.lower()}_converted"
    crate_dir = scaffold_crate(output_dir, crate_name, rust_sources)
    console.print(f"  [green]✓[/green] Crate created at {crate_dir}")

    build_ok, build_out = build_crate(crate_dir)
    _show_build_result("cargo build --release", build_ok, build_out)

    test_ok, test_out = test_crate(crate_dir)
    _show_build_result("cargo test", test_ok, test_out)

    # --- 8. Accuracy -----------------------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 6 — Numerical accuracy check[/bold]", style="cyan"))
    accuracy_results = []
    for fn_name in functions_to_convert:
        src_path = source_map.get(fn_name.lower())
        routine = routine_map.get(fn_name.upper())
        acc = run_accuracy_check(
            fn_name, src_path, crate_dir if build_ok else None, routine=routine
        )
        accuracy_results.append(acc)
        _show_accuracy(acc)

    # --- 9. Benchmark ----------------------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 7 — Performance benchmark[/bold]", style="cyan"))
    bench_results = []
    for fn_name in functions_to_convert:
        src_path = source_map.get(fn_name.lower())
        routine = routine_map.get(fn_name.upper())
        br = run_benchmark(fn_name, src_path, crate_dir if build_ok else None, routine=routine)
        bench_results.append(br)
        console.print(f"  {br.summary}")
        for d in br.details:
            console.print(d)

    # --- 10. Report -----------------------------------------------------------
    console.print()
    console.print(Panel("[bold]Step 8 — Generating report[/bold]", style="cyan"))
    report_path = generate_report(
        output_dir=output_dir,
        library=library,
        strategy_name=STRATEGY_NAMES[strategy_key],
        conversion_results=conversion_results,
        accuracy_results=accuracy_results,
        bench_results=bench_results,
        crate_dir=crate_dir,
        build_ok=build_ok,
        test_ok=test_ok,
    )
    md_path, html_path = report_path
    console.print(f"  [green]✓[/green] Report written to [bold]{html_path}[/bold]")

    # --- 11. Final summary ----------------------------------------------------
    _print_final_summary(conversion_results, accuracy_results, bench_results, html_path)


# ---------------------------------------------------------------------------
# Wizard helpers
# ---------------------------------------------------------------------------

def _ask_library() -> str:
    console.print("\n[bold]Which library would you like to convert?[/bold]")
    console.print("  [cyan]1[/cyan]  BLAS (Basic Linear Algebra Subprograms) — Fortran reference implementation")
    console.print("  [cyan]q[/cyan]  Quit")
    while True:
        choice = Prompt.ask("\n[bold cyan]>[/bold cyan] Your choice", default="1")
        if choice.lower() == "q":
            sys.exit(0)
        if choice == "1":
            return "BLAS"
        console.print("[yellow]Please enter 1 or q.[/yellow]")


def _fetch_and_select(
    library: str, output_dir: Path
) -> Tuple[List[str], Dict[str, Path]]:
    """Download sources and ask the user what to convert."""
    console.print("\n[bold]What would you like to convert?[/bold]")
    console.print("  [cyan]1[/cyan]  Single function  (you choose which one)")
    console.print("  [cyan]2[/cyan]  Custom sample  (comma-separated names OR a sample size, default 10)")
    console.print(f"  [cyan]3[/cyan]  Full library  ({len(BLAS_FUNCTIONS)} functions)")

    scope = Prompt.ask("\n[bold cyan]>[/bold cyan] Scope", choices=["1", "2", "3"], default="1")

    if scope == "1":
        fn = Prompt.ask(
            "\n[bold cyan]>[/bold cyan] Function name",
            default="dgemm",
        ).strip().lower()
        wanted = [fn]
    elif scope == "2":
        raw = Prompt.ask(
            "\n[bold cyan]>[/bold cyan] "
            "Function names (comma-separated) OR sample size",
            default="10",
        ).strip()
        wanted = _parse_sample_input(raw)
        console.print(f"  Selected: {', '.join(wanted)}")
    else:
        wanted = [f.lower() for f in BLAS_FUNCTIONS]
        console.print(f"  Will convert all {len(wanted)} BLAS functions.")

    # Always fetch the BLAS support routines so that the Fortran compiler
    # can link functions that call LSAME / XERBLA (e.g. DGEMM).
    _SUPPORT_FNS = ["lsame", "xerbla"]
    fetch_list = list(dict.fromkeys(wanted + _SUPPORT_FNS))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching source files…", total=None)
        source_map = fetch_blas(
            output_dir,
            functions=fetch_list,
            progress_callback=lambda msg: progress.update(task, description=msg),
        )

    # Only expose user-selected functions as conversion targets
    wanted_set = set(wanted)
    found = sorted(k for k in source_map if k in wanted_set)
    missing = [f for f in wanted if f not in source_map]
    console.print(f"\n  [green]✓[/green] {len(found)} source file(s) ready.")
    if missing:
        console.print(f"  [yellow]⚠[/yellow] Could not fetch: {', '.join(missing)}")

    return found, source_map


def _parse_sample_input(raw: str) -> List[str]:
    """Parse the sample scope input.

    Accepts either:
    - a plain integer → pick that many functions from the default list
    - a comma-separated list of names → use exactly those functions
    """
    raw = raw.strip()
    # If it looks like an integer, treat it as a sample size
    try:
        n = int(raw)
        n = max(1, min(n, len(BLAS_FUNCTIONS)))
        return [f.lower() for f in BLAS_FUNCTIONS[:n]]
    except ValueError:
        pass
    # Otherwise treat as a comma-separated list of function names
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    # Validate: warn about unknown names but keep them so the user sees the error
    known = {f.lower() for f in BLAS_FUNCTIONS}
    for name in names:
        if name not in known:
            console.print(
                f"  [yellow]⚠[/yellow] '{name}' is not in the known BLAS function list — "
                "will attempt to fetch anyway."
            )
    return names if names else [f.lower() for f in BLAS_FUNCTIONS[:10]]


def _ask_strategy() -> str:
    console.print()
    for key, desc in _STRATEGY_DESCRIPTIONS.items():
        console.print(f"  [cyan]{key}[/cyan]  {desc}\n")
    return Prompt.ask(
        "[bold cyan]>[/bold cyan] Strategy", choices=["1", "2", "3"], default="3"
    )


def _configure_llm(strategy_key: str) -> LLMClient:
    llm = LLMClient(stream_callback=_make_stream_cb())
    if llm.is_available:
        console.print(
            f"\n  [green]✓[/green] LLM configured: "
            f"[bold]{llm.provider}[/bold] / [bold]{llm.model}[/bold]"
        )
        return llm

    # No auth found — offer the interactive setup wizard
    from fortran_to_rust.auth_setup import prompt_auth_setup

    configured = prompt_auth_setup(console)
    if configured is not None and configured.is_available:
        configured.stream_callback = _make_stream_cb()
        return configured

    # Still no auth after setup (user skipped or setup failed)
    if strategy_key in ("1", "2"):
        console.print(
            "\n  [yellow]⚠[/yellow] No LLM available — "
            "switching to Strategy 3 (Hybrid rule-based only)."
        )
    else:
        console.print(
            "\n  [dim]No LLM available — Strategy 3 will use rule-based conversion only.[/dim]"
        )
    return llm


def _make_stream_cb():
    """Return a stream callback that writes LLM output chunks to the console live."""
    import sys
    def cb(chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    return cb


def _make_cb():
    """Return a progress_callback that pretty-prints to the console."""
    def cb(msg: str) -> None:
        # Strip rich markup when the message already contains it
        console.print(f"    {msg}")
    return cb


def _show_build_result(label: str, ok: bool, output: str) -> None:
    sym = "[green]✓[/green]" if ok else "[yellow]⚠[/yellow]"
    console.print(f"  {sym} {label}: {'passed' if ok else 'finished with warnings/errors'}")
    if not ok and output:
        # Show only the first few lines of errors
        snippet = "\n".join(output.splitlines()[:15])
        console.print(f"[dim]{snippet}[/dim]")


def _show_accuracy(acc) -> None:
    if acc.max_abs_error is not None:
        sym = "[green]✓[/green]" if acc.passed else "[red]✗[/red]"
        console.print(
            f"  {sym} {acc.function_name}: "
            f"max_abs_error={acc.max_abs_error:.2e}, "
            f"mean_abs_error={acc.mean_abs_error:.2e}"
        )
    else:
        console.print(f"  [dim]  {acc.function_name}: {acc.error_message or 'no result'}[/dim]")
    for d in acc.details:
        console.print(f"[dim]{d}[/dim]")


def _print_final_summary(
    conversion_results, accuracy_results, bench_results, report_path
) -> None:
    console.print()
    console.print(Rule("[bold cyan]Conversion Complete[/bold cyan]", style="cyan"))

    table = Table(title="Summary", show_header=True, header_style="bold cyan")
    table.add_column("Function", style="bold")
    table.add_column("Converted", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Speedup", justify="center")

    acc_map = {a.function_name.upper(): a for a in accuracy_results}
    bench_map = {b.function_name.upper(): b for b in bench_results}

    for r in conversion_results:
        fn = r.routine_name.upper()
        converted = "✅" if r.success else "❌"
        acc = acc_map.get(fn)
        accuracy = (
            ("✅" if acc.passed else "❌") if acc and acc.max_abs_error is not None else "—"
        )
        bench = bench_map.get(fn)
        speedup = f"{bench.speedup:.2f}×" if bench and bench.speedup else "—"
        table.add_row(fn, converted, accuracy, speedup)

    console.print(table)
    console.print(f"\n  Full report: [bold green]{report_path}[/bold green]")
    console.print()
