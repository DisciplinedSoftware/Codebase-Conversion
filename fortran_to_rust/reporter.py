"""Post-conversion report generator.

Produces a Markdown and HTML report at ``output_dir/reports/<timestamp>_report.{md,html}``
summarising:
- Which functions were converted and by which strategy
- Accuracy results (max / mean absolute error)
- Performance comparison
- Compiler warnings and any LLM repair rounds needed
"""

from __future__ import annotations

import datetime
import html as html_lib
import http.server
import os
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fortran_to_rust.benchmarker import BenchResult
from fortran_to_rust.strategies.base import ConversionResult
from fortran_to_rust.test_harness import AccuracyResult


def generate_comparison_report(
    run_dir: Path,
    library: str,
    all_results: Dict,
    open_browser: bool = False,
) -> Tuple[Path, Path]:
    """Write a single MD+HTML report covering all three strategies.

    Files land at ``run_dir/reports/<ts>_report.{md,html}`` reusing the
    timestamp embedded in *run_dir*'s name so all filenames stay consistent.
    Returns ``(md_path, html_path)``.
    """
    from fortran_to_rust.strategies import STRATEGY_NAMES

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    dir_ts = run_dir.name
    if dir_ts.startswith("report_") and len(dir_ts) == len("report_YYYYMMDD_HHMMSS"):
        ts = dir_ts[len("report_"):]
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        generated_str = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        generated_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_path = reports_dir / f"{ts}_report.md"
    lines: List[str] = []
    _section = lambda title: lines.extend([f"\n## {title}\n"])

    lines.append("# Fortran-to-Rust Comparison Report")
    lines.append(f"\n**Library:** {library}  ")
    lines.append("**Strategies:** All (1 · LLM-First, 2 · Agentic, 3 · Hybrid)  ")
    lines.append(f"**Generated:** {generated_str}  ")

    _section("Overview")
    lines.append("| Strategy | Build | Tests | Accuracy | Max Abs Error | Avg Speedup |")
    lines.append("|----------|-------|-------|----------|---------------|-------------|")
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
        bench_results = res.get("bench_results", [])
        speedups = [b.speedup for b in bench_results if b.speedup]
        lines.append(
            f"| **{key}** {name} | {build} | {tests} | {accuracy} "
            f"| {f'{max_err:.2e}' if max_err is not None else '—'} "
            f"| {f'{sum(speedups)/len(speedups):.2f}×' if speedups else '—'} |"
        )

    for key in ("1", "2", "3"):
        res = all_results.get(key, {})
        name = STRATEGY_NAMES.get(key, key)
        _section(f"Strategy {key} — {name}")
        if res.get("error") and not res.get("conversion_results"):
            lines.append(f"**Error:** {res['error']}")
            continue
        crate_dir = res.get("crate_dir")
        if crate_dir:
            lines.append(f"**Crate:** `{crate_dir}`  ")
        conversion_results = res.get("conversion_results", [])
        lines.append("| Function | Strategy Used | Lines | Repair Rounds | Status |")
        lines.append("|----------|--------------|-------|---------------|--------|")
        for r in conversion_results:
            status = "✅ OK" if r.success else "❌ Failed"
            lines.append(
                f"| `{r.routine_name}` | {r.strategy_used} "
                f"| {_get_line_count(r)} | {r.repair_rounds} | {status} |"
            )
        build_ok = res.get("build_ok", False)
        test_ok = res.get("test_ok", False)
        lines.append(
            f"\n- `cargo build --release`: {'✅ passed' if build_ok else '❌ failed'}"
        )
        lines.append(f"- `cargo test`: {'✅ passed' if test_ok else '❌ failed'}")
        accuracy_results = res.get("accuracy_results", [])
        if accuracy_results:
            lines.append("\n| Function | Max Abs Error | Mean Abs Error | Result |")
            lines.append("|----------|---------------|----------------|--------|")
            for a in accuracy_results:
                max_e = f"{a.max_abs_error:.2e}" if a.max_abs_error is not None else "N/A"
                mean_e = f"{a.mean_abs_error:.2e}" if a.mean_abs_error is not None else "N/A"
                lines.append(
                    f"| `{a.function_name}` | {max_e} | {mean_e} | {'✅' if a.passed else '❌'} |"
                )
        bench_results = res.get("bench_results", [])
        if bench_results:
            lines.append("\n| Function | Fortran (ms/call) | Rust (ms/call) | Speedup |")
            lines.append("|----------|-------------------|----------------|---------|")
            for b in bench_results:
                f_ms = f"{b.fortran_time_ms:.3f}" if b.fortran_time_ms else "N/A"
                r_ms = f"{b.rust_time_ms:.3f}" if b.rust_time_ms else "N/A"
                sp = f"{b.speedup:.2f}×" if b.speedup else "N/A"
                lines.append(f"| `{b.function_name}` | {f_ms} | {r_ms} | {sp} |")
        for r in conversion_results:
            if r.rust_source:
                snippet = r.rust_source[:1500]
                truncated = "…" if len(r.rust_source) > 1500 else ""
                lines.append(f"\n**`{r.routine_name}` generated code:**\n")
                lines.append(f"```rust\n{snippet}{truncated}\n```")

    report_path.write_text("\n".join(lines))

    html_path = _write_comparison_html_report(
        report_path.with_suffix(".html"),
        library=library,
        all_results=all_results,
        generated_str=generated_str,
    )

    if open_browser:
        _serve_report(html_path)

    return report_path, html_path


def _write_comparison_html_report(
    html_path: Path,
    library: str,
    all_results: Dict,
    generated_str: Optional[str] = None,
) -> Path:
    from fortran_to_rust.strategies import STRATEGY_NAMES

    ts = generated_str or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def section(title: str) -> str:
        return f"<h2>{_esc(title)}</h2>\n"

    parts: List[str] = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1.0'>",
        f"<title>Fortran→Rust Comparison — {_esc(library)}</title>",
        f"<style>{_HTML_CSS}</style>",
        "</head><body>",
        "<h1>Fortran→Rust Comparison Report</h1>",
        "<div class='summary-grid'>",
        f"<div class='card'><div class='card-label'>Library</div><div class='card-value'>{_esc(library)}</div></div>",
        "<div class='card'><div class='card-label'>Strategies</div><div class='card-value' style='font-size:16px'>All 3</div></div>",
        f"<div class='card'><div class='card-label'>Generated</div><div class='card-value' style='font-size:14px'>{_esc(ts)}</div></div>",
        "</div>",
    ]

    parts.append(section("Overview"))
    parts.append(
        "<table><tr>"
        "<th>Strategy</th><th>Build</th><th>Tests</th>"
        "<th>Accuracy</th><th>Max Error</th><th>Avg Speedup</th>"
        "</tr>"
    )
    for key in ("1", "2", "3"):
        res = all_results.get(key, {})
        name = STRATEGY_NAMES.get(key, key)
        if res.get("error") and not res.get("conversion_results"):
            parts.append(
                f"<tr><td><strong>{_esc(key)}</strong> {_esc(name)}</td>"
                "<td>❌</td><td>❌</td><td>❌</td><td>—</td><td>—</td></tr>"
            )
            continue
        build = _status_badge(res.get("build_ok", False))
        tests = _status_badge(res.get("test_ok", False))
        acc_results = res.get("accuracy_results", [])
        acc_passed = all(a.passed for a in acc_results if a.max_abs_error is not None)
        acc_badge = (
            "<span class='ok'>✅</span>"
            if (acc_results and acc_passed)
            else ("<span class='fail'>❌</span>" if acc_results else "—")
        )
        max_err = max(
            (a.max_abs_error for a in acc_results if a.max_abs_error is not None),
            default=None,
        )
        max_err_str = f"{max_err:.2e}" if max_err is not None else "—"
        bench_results = res.get("bench_results", [])
        speedups = [b.speedup for b in bench_results if b.speedup]
        speedup_str = f"{sum(speedups)/len(speedups):.2f}×" if speedups else "—"
        parts.append(
            f"<tr><td><strong>{_esc(key)}</strong> {_esc(name)}</td>"
            f"<td style='text-align:center'>{build}</td>"
            f"<td style='text-align:center'>{tests}</td>"
            f"<td style='text-align:center'>{acc_badge}</td>"
            f"<td style='text-align:right'>{_esc(max_err_str)}</td>"
            f"<td style='text-align:right'>{_esc(speedup_str)}</td></tr>"
        )
    parts.append("</table>")

    for key in ("1", "2", "3"):
        res = all_results.get(key, {})
        name = STRATEGY_NAMES.get(key, key)
        parts.append(f"<h2>Strategy {_esc(key)} — {_esc(name)}</h2>")
        if res.get("error") and not res.get("conversion_results"):
            parts.append(f"<p class='fail'>Error: {_esc(res['error'])}</p>")
            continue
        crate_dir = res.get("crate_dir")
        if crate_dir:
            parts.append(f"<p class='dim'>Crate: <code>{_esc(str(crate_dir))}</code></p>")
        conversion_results = res.get("conversion_results", [])
        if conversion_results:
            parts.append("<table><tr><th>Function</th><th>Strategy Used</th><th>Lines</th><th>Repair Rounds</th><th>Status</th></tr>")
            for r in conversion_results:
                status = "<span class='ok'>✅ OK</span>" if r.success else "<span class='fail'>❌ Failed</span>"
                parts.append(
                    f"<tr><td><code>{_esc(r.routine_name)}</code></td>"
                    f"<td>{_esc(r.strategy_used)}</td>"
                    f"<td>{_get_line_count(r)}</td>"
                    f"<td>{r.repair_rounds}</td>"
                    f"<td>{status}</td></tr>"
                )
            parts.append("</table>")
        build_ok = res.get("build_ok", False)
        test_ok = res.get("test_ok", False)
        parts.append(f"<p>cargo build --release: {_status_badge(build_ok)}</p>")
        parts.append(f"<p>cargo test: {_status_badge(test_ok)}</p>")
        accuracy_results = res.get("accuracy_results", [])
        if accuracy_results:
            parts.append("<table><tr><th>Function</th><th>Max Abs Error</th><th>Mean Abs Error</th><th>Result</th></tr>")
            for a in accuracy_results:
                max_e = f"{a.max_abs_error:.2e}" if a.max_abs_error is not None else "N/A"
                mean_e = f"{a.mean_abs_error:.2e}" if a.mean_abs_error is not None else "N/A"
                badge = "<span class='ok'>✅</span>" if a.passed else "<span class='fail'>❌</span>"
                parts.append(
                    f"<tr><td><code>{_esc(a.function_name)}</code></td>"
                    f"<td>{_esc(max_e)}</td><td>{_esc(mean_e)}</td><td>{badge}</td></tr>"
                )
            parts.append("</table>")
            for a in accuracy_results:
                if a.details:
                    details_html = "<br>".join(_esc(d) for d in a.details)
                    parts.append(f"<details><summary><code>{_esc(a.function_name)}</code> test details</summary><p class='dim'>{details_html}</p></details>")
        bench_results = res.get("bench_results", [])
        if bench_results:
            parts.append("<table><tr><th>Function</th><th>Fortran (ms/call)</th><th>Rust (ms/call)</th><th>Speedup</th></tr>")
            for b in bench_results:
                f_ms = f"{b.fortran_time_ms:.3f}" if b.fortran_time_ms else "N/A"
                r_ms = f"{b.rust_time_ms:.3f}" if b.rust_time_ms else "N/A"
                sp = f"{b.speedup:.2f}×" if b.speedup else "N/A"
                parts.append(
                    f"<tr><td><code>{_esc(b.function_name)}</code></td>"
                    f"<td>{_esc(f_ms)}</td><td>{_esc(r_ms)}</td><td>{_esc(sp)}</td></tr>"
                )
            parts.append("</table>")
        for r in conversion_results:
            if r.rust_source:
                snippet = _esc(r.rust_source[:2000])
                trunc = "<span class='dim'>… (truncated)</span>" if len(r.rust_source) > 2000 else ""
                parts.append(f"<h3><code>{_esc(r.routine_name)}</code></h3>")
                parts.append(f"<pre><code>{snippet}{trunc}</code></pre>")

    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


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
    open_browser: bool = True,
) -> Tuple[Path, Path]:
    """Write Markdown and HTML reports; return ``(md_path, html_path)``.

    The HTML report is automatically opened in the default browser unless
    *open_browser* is ``False``.
    """
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the timestamp embedded in the run directory name (report_YYYYMMDD_HHMMSS)
    # so the report filenames match the folder name exactly.
    dir_ts = output_dir.name  # e.g. "report_20260315_030100"
    if dir_ts.startswith("report_") and len(dir_ts) == len("report_YYYYMMDD_HHMMSS"):
        ts = dir_ts[len("report_"):]  # "20260315_030100"
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = reports_dir / f"{ts}_report.md"

    # Human-readable form for the report header
    try:
        generated_str = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        generated_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    _section = lambda title: lines.extend([f"\n## {title}\n"])

    # ---- Header ----
    lines.append(f"# Fortran-to-Rust Conversion Report")
    lines.append(f"\n**Library:** {library}  ")
    lines.append(f"**Strategy:** {strategy_name}  ")
    lines.append(f"**Generated:** {generated_str}  ")
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

    # ---- HTML report ----
    html_path = _write_html_report(
        report_path.with_suffix(".html"),
        library=library,
        strategy_name=strategy_name,
        conversion_results=conversion_results,
        accuracy_results=accuracy_results,
        bench_results=bench_results,
        crate_dir=crate_dir,
        build_ok=build_ok,
        test_ok=test_ok,
        generated_str=generated_str,
    )

    if open_browser:
        _serve_report(html_path)

    return report_path, html_path


# ---------------------------------------------------------------------------
# HTTP report server (devcontainer / headless-friendly)
# ---------------------------------------------------------------------------

_SERVE_PORT = 8080


def _serve_report(html_path: Path, port: int = _SERVE_PORT) -> None:
    """Serve the reports directory over HTTP on *port* and block until Ctrl+C.

    VS Code automatically detects the open port and offers to forward it to
    the host browser, so this works seamlessly inside devcontainers and
    GitHub Codespaces.  Falls back to ``webbrowser.open`` on regular desktops.
    """
    reports_dir = html_path.parent

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(reports_dir), **kwargs)

        def log_message(self, fmt, *args):  # silence access log noise
            pass

    # Try the HTTP server first; if the port is busy fall back to webbrowser.
    try:
        server = http.server.HTTPServer(("0.0.0.0", port), _Handler)
    except OSError:
        # Port already in use — try to open the browser directly.
        try:
            webbrowser.open(html_path.as_uri())
        except Exception:
            pass
        return

    url = f"http://localhost:{port}/{html_path.name}"
    print(f"\n  📄 Report: {url}")
    print(f"     (VS Code will auto-forward port {port} — or open the URL above)")
    print("     Press Ctrl+C to stop the report server.\n")

    # In VS Code the port-forwarding notification appears as soon as the
    # socket is bound.  Serve until the user interrupts.
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_CSS = """\
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  max-width:1100px;margin:40px auto;padding:0 24px;background:#0d1117;color:#c9d1d9;line-height:1.6}
h1{color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:12px}
h2{color:#79c0ff;margin-top:32px;border-bottom:1px solid #21262d;padding-bottom:6px}
h3{color:#d2a8ff}
table{width:100%;border-collapse:collapse;margin:16px 0}
th{background:#161b22;color:#58a6ff;text-align:left;padding:8px 12px;border:1px solid #21262d}
td{padding:8px 12px;border:1px solid #21262d}
tr:nth-child(even){background:#161b22}
code,pre{background:#161b22;border:1px solid #21262d;border-radius:6px;font-family:'SFMono-Regular',Consolas,monospace;font-size:13px}
code{padding:2px 6px}
pre{padding:16px;overflow-x:auto}
.ok{color:#3fb950}.fail{color:#f85149}.dim{color:#8b949e}
.badge-ok{background:#1f6feb;color:#fff;padding:2px 8px;border-radius:20px}
.badge-fail{background:#da3633;color:#fff;padding:2px 8px;border-radius:20px}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:16px 0}
.card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px}
.card-label{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
.card-value{font-size:24px;font-weight:700;margin-top:4px}
"""


def _esc(s: str) -> str:
    return html_lib.escape(str(s))


def _status_badge(ok: bool) -> str:
    cls = "badge-ok" if ok else "badge-fail"
    label = "✅ passed" if ok else "❌ failed"
    return f'<span class="{cls}">{label}</span>'


def _write_html_report(
    html_path: Path,
    library: str,
    strategy_name: str,
    conversion_results: List[ConversionResult],
    accuracy_results: List[AccuracyResult],
    bench_results: List[BenchResult],
    crate_dir: Optional[Path],
    build_ok: bool,
    test_ok: bool,
    generated_str: Optional[str] = None,
) -> Path:
    ts = generated_str or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def section(title: str) -> str:
        return f"<h2>{_esc(title)}</h2>\n"

    parts: List[str] = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1.0'>",
        f"<title>Fortran→Rust Report — {_esc(library)}</title>",
        f"<style>{_HTML_CSS}</style>",
        "</head><body>",
        f"<h1>Fortran→Rust Conversion Report</h1>",
        "<div class='summary-grid'>",
        f"<div class='card'><div class='card-label'>Library</div><div class='card-value'>{_esc(library)}</div></div>",
        f"<div class='card'><div class='card-label'>Strategy</div><div class='card-value' style='font-size:16px'>{_esc(strategy_name)}</div></div>",
        f"<div class='card'><div class='card-label'>Functions</div><div class='card-value'>{len(conversion_results)}</div></div>",
        f"<div class='card'><div class='card-label'>Generated</div><div class='card-value' style='font-size:14px'>{_esc(ts)}</div></div>",
        "</div>",
    ]
    if crate_dir:
        parts.append(f"<p class='dim'>Crate: <code>{_esc(str(crate_dir))}</code></p>")

    # ---- Conversion summary ----
    parts.append(section("Conversion Summary"))
    parts.append("<table><tr><th>Function</th><th>Strategy Used</th><th>Lines</th><th>Repair Rounds</th><th>Status</th></tr>")
    for r in conversion_results:
        status = "<span class='ok'>✅ OK</span>" if r.success else "<span class='fail'>❌ Failed</span>"
        parts.append(
            f"<tr><td><code>{_esc(r.routine_name)}</code></td>"
            f"<td>{_esc(r.strategy_used)}</td>"
            f"<td>{_get_line_count(r)}</td>"
            f"<td>{r.repair_rounds}</td>"
            f"<td>{status}</td></tr>"
        )
    parts.append("</table>")

    # ---- Build / test ----
    parts.append(section("Build & Test"))
    parts.append(f"<p>cargo build --release: {_status_badge(build_ok)}</p>")
    parts.append(f"<p>cargo test: {_status_badge(test_ok)}</p>")

    # ---- Accuracy ----
    parts.append(section("Numerical Accuracy"))
    if accuracy_results:
        parts.append("<table><tr><th>Function</th><th>Tests</th><th>Failed</th><th>Max Abs Error</th><th>Mean Abs Error</th><th>Result</th></tr>")
        for a in accuracy_results:
            max_e  = f"{a.max_abs_error:.2e}"  if a.max_abs_error  is not None else "N/A"
            mean_e = f"{a.mean_abs_error:.2e}" if a.mean_abs_error is not None else "N/A"
            badge  = "<span class='ok'>✅</span>" if a.passed else "<span class='fail'>❌</span>"
            parts.append(
                f"<tr><td><code>{_esc(a.function_name)}</code></td>"
                f"<td>{a.num_test_cases}</td><td>{a.failed_cases}</td>"
                f"<td>{_esc(max_e)}</td><td>{_esc(mean_e)}</td><td>{badge}</td></tr>"
            )
        parts.append("</table>")
        for a in accuracy_results:
            if a.error_message:
                parts.append(f"<p class='dim'><code>{_esc(a.function_name)}</code>: {_esc(a.error_message)}</p>")
            if a.details:
                details_html = "<br>".join(_esc(d) for d in a.details)
                parts.append(f"<details><summary><code>{_esc(a.function_name)}</code> test details</summary><p class='dim'>{details_html}</p></details>")
    else:
        parts.append("<p class='dim'>No accuracy results available.</p>")

    # ---- Performance ----
    parts.append(section("Performance"))
    if bench_results:
        parts.append("<table><tr><th>Function</th><th>Fortran (ms/call)</th><th>Rust (ms/call)</th><th>Speedup</th></tr>")
        for b in bench_results:
            f_ms = f"{b.fortran_time_ms:.3f}" if b.fortran_time_ms else "N/A"
            r_ms = f"{b.rust_time_ms:.3f}"    if b.rust_time_ms    else "N/A"
            sp   = f"{b.speedup:.2f}×"        if b.speedup         else "N/A"
            parts.append(
                f"<tr><td><code>{_esc(b.function_name)}</code></td>"
                f"<td>{_esc(f_ms)}</td><td>{_esc(r_ms)}</td><td>{_esc(sp)}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p class='dim'>No benchmark results available.</p>")

    # ---- Warnings ----
    all_warnings = [(r.routine_name, w) for r in conversion_results for w in r.warnings]
    if all_warnings:
        parts.append(section("Warnings"))
        parts.append("<ul>")
        for fn, w in all_warnings:
            parts.append(f"<li><strong>{_esc(fn)}</strong>: {_esc(w)}</li>")
        parts.append("</ul>")

    # ---- Generated code ----
    parts.append(section("Generated Rust Code"))
    for r in conversion_results:
        if r.rust_source:
            snippet = _esc(r.rust_source[:2000])
            trunc   = "<span class='dim'>… (truncated)</span>" if len(r.rust_source) > 2000 else ""
            parts.append(f"<h3><code>{_esc(r.routine_name)}</code></h3>")
            parts.append(f"<pre><code>{snippet}{trunc}</code></pre>")

    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def _get_line_count(r: ConversionResult) -> str:
    if r.rust_source:
        return str(len(r.rust_source.splitlines()))
    return "—"

