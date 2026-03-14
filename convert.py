#!/usr/bin/env python3
"""fortran-to-rust — Automated Fortran→Rust conversion pipeline.

Usage
-----
    python convert.py [--output-dir PATH] [--non-interactive]

Options
-------
  --output-dir PATH   Directory for downloaded sources, generated Rust code,
                      and reports.  Defaults to ./output.
  --non-interactive   Run dgemm conversion with the Hybrid strategy without
                      prompting.  Useful for CI or quick demos.

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
        help="Directory for output files (default: ./output).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run dgemm/Hybrid conversion without prompts.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.non_interactive:
        _run_non_interactive(output_dir)
    else:
        from fortran_to_rust.cli import run
        run(output_dir)


def _run_non_interactive(output_dir: Path) -> None:
    """Headless demo: convert dgemm with the Hybrid strategy."""
    from rich.console import Console
    from rich.rule import Rule

    from fortran_to_rust.benchmarker import run_benchmark
    from fortran_to_rust.call_graph import build_call_graph
    from fortran_to_rust.fetcher import fetch_blas
    from fortran_to_rust.llm_client import LLMClient
    from fortran_to_rust.parser import parse_file
    from fortran_to_rust.reporter import generate_report
    from fortran_to_rust.rust_project import build_crate, scaffold_crate, test_crate
    from fortran_to_rust.strategies.hybrid import HybridStrategy
    from fortran_to_rust.test_harness import run_accuracy_check

    console = Console()
    console.print(Rule("[bold cyan]Fortran-to-Rust  (non-interactive demo)[/bold cyan]"))

    # 1. Fetch dgemm source
    console.print("\n[bold]Fetching BLAS sources…[/bold]")
    source_map = fetch_blas(output_dir, functions=["dgemm", "lsame", "xerbla"])
    if "dgemm" not in source_map:
        console.print("[red]Could not obtain dgemm.f — aborting.[/red]")
        sys.exit(1)

    # 2. Parse
    console.print("[bold]Parsing…[/bold]")
    all_routines = []
    for path in source_map.values():
        all_routines.extend(parse_file(path))
    dgemm_routines = [r for r in all_routines if r.name.upper() == "DGEMM"]
    if not dgemm_routines:
        console.print("[red]dgemm routine not found in parsed source.[/red]")
        sys.exit(1)
    routine = dgemm_routines[0]
    console.print(f"  Parsed {routine.name} ({routine.line_count} lines)")

    # 3. Convert
    console.print("[bold]Converting with Hybrid strategy…[/bold]")
    llm = LLMClient()
    strategy = HybridStrategy(output_dir, llm=llm)

    messages: list[str] = []
    def cb(msg: str) -> None:
        console.print(f"  {msg}")
        messages.append(msg)

    result = strategy.convert(routine, progress_callback=cb)
    console.print(f"  [green]✓[/green] Strategy used: {result.strategy_used}")

    # 4. Scaffold crate
    console.print("[bold]Scaffolding Rust crate…[/bold]")
    crate_dir = scaffold_crate(output_dir, "blas_converted", {"dgemm": result.rust_source or ""})
    console.print(f"  Crate: {crate_dir}")

    build_ok, build_out = build_crate(crate_dir)
    console.print(f"  cargo build: {'✅ passed' if build_ok else '⚠ finished with errors'}")

    # 5. Accuracy
    console.print("[bold]Accuracy check…[/bold]")
    acc = run_accuracy_check(
        "dgemm", source_map["dgemm"], crate_dir if build_ok else None, routine=routine
    )
    console.print(
        f"  Accuracy: {'✅ passed' if acc.passed else '⚠ see report'}"
        + (f"  max_err={acc.max_abs_error:.2e}" if acc.max_abs_error else "")
    )

    # 6. Benchmark
    console.print("[bold]Benchmarking…[/bold]")
    bench = run_benchmark(
        "dgemm", source_map["dgemm"], crate_dir if build_ok else None, routine=routine
    )
    console.print(f"  {bench.summary}")

    # 7. Report
    console.print("[bold]Generating report…[/bold]")
    report_path = generate_report(
        output_dir=output_dir,
        library="BLAS",
        strategy_name="Hybrid Rule-Based + LLM Polish",
        conversion_results=[result],
        accuracy_results=[acc],
        bench_results=[bench],
        crate_dir=crate_dir,
        build_ok=build_ok,
        test_ok=False,
    )
    console.print(f"  [green]Report:[/green] {report_path}")
    console.print(Rule(style="cyan"))


if __name__ == "__main__":
    main()
