"""Performance benchmarker.

For each converted function, this module:

1. **Generates** a Fortran benchmark program on the fly based on the parsed
   argument declarations (types, array shapes).
2. **Compiles and runs** it with ``gfortran -O2`` to measure wall-clock time.
3. **Generates** a Rust ``examples/bench_<fn>.rs`` binary using the same input
   shapes.
4. **Compiles and runs** it with ``cargo build --release`` to measure Rust
   wall-clock time.
5. Reports a speedup ratio.

No function-specific knowledge is needed — everything is derived from the
:class:`~fortran_to_rust.parser.FortranRoutine` object.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fortran_to_rust.test_harness import (
    ArgDecl,
    _array_size,
    _assign_dims,
    _find_support_files,
    _fortran_call,
    parse_arg_declarations,
)

_REPS = 10   # number of timed repetitions


@dataclass
class BenchResult:
    function_name: str
    fortran_time_ms: Optional[float] = None
    rust_time_ms: Optional[float] = None
    speedup: Optional[float] = None
    error_message: Optional[str] = None
    details: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.error_message:
            return f"Benchmark failed: {self.error_message}"
        if self.fortran_time_ms and self.rust_time_ms:
            direction = "faster" if self.speedup and self.speedup >= 1.0 else "slower"
            ratio = abs(self.speedup or 1.0)
            return (
                f"Fortran: {self.fortran_time_ms:.3f} ms  |  "
                f"Rust: {self.rust_time_ms:.3f} ms  |  "
                f"Rust is {ratio:.2f}x {direction} than Fortran"
            )
        if self.fortran_time_ms:
            return f"Fortran: {self.fortran_time_ms:.3f} ms  |  Rust: N/A"
        return "Incomplete benchmark data."


_FORTRAN_BENCH_TEMPLATE = """\
      PROGRAM BENCH_{name}
      IMPLICIT NONE
      INTEGER BENCH_R, INIT_I
      DOUBLE PRECISION BENCH_T1, BENCH_T2, BENCH_ELAPSED
{declarations}
{assignments}
      CALL CPU_TIME(BENCH_T1)
      DO BENCH_R = 1, {reps}
{call_stmt}
      END DO
      CALL CPU_TIME(BENCH_T2)
      BENCH_ELAPSED = (BENCH_T2 - BENCH_T1) * 1000.0D0 / DBLE({reps})
      WRITE(*,'(F30.15)') BENCH_ELAPSED
      END
"""


def _generate_fortran_bench(
    routine_name: str,
    arg_names: List[str],
    arg_decls: Dict[str, ArgDecl],
    assigned_dims: Dict[str, int],
    reps: int = _REPS,
) -> str:
    decl_lines: List[str] = []
    assign_lines: List[str] = []

    int_scalars = [n for n, d in arg_decls.items() if d.is_integer and not d.is_array]
    if int_scalars:
        decl_lines.append("      INTEGER " + ", ".join(int_scalars))
        for n in int_scalars:
            assign_lines.append(f"      {n} = {assigned_dims.get(n, 4)}")

    dp_scalars = [n for n, d in arg_decls.items() if d.is_real and not d.is_array]
    if dp_scalars:
        decl_lines.append("      DOUBLE PRECISION " + ", ".join(dp_scalars))
        for n in dp_scalars:
            assign_lines.append(f"      {n} = 1.0D0")

    # Array init — use INIT_I, not BENCH_R, to avoid variable conflict
    for arr_name, decl in [(n, d) for n, d in arg_decls.items() if d.is_array and d.is_real]:
        sizes = _array_size(decl, assigned_dims)
        if not sizes:
            continue
        dim_str = ", ".join(str(s) for s in sizes)
        decl_lines.append(f"      DOUBLE PRECISION {arr_name}({dim_str})")
        total = 1
        for s in sizes:
            total *= s
        if len(sizes) == 1:
            assign_lines.append(f"      DO INIT_I=1,{sizes[0]}")
            assign_lines.append(f"      {arr_name}(INIT_I) = 0.5D0")
            assign_lines.append(f"      END DO")
        elif len(sizes) >= 2:
            n_rows = sizes[0]
            assign_lines.append(f"      DO INIT_I=1,{total}")
            # Convert 1-based linear index INIT_I to column-major (row, col):
            #   row = MOD(INIT_I-1, n_rows) + 1
            #   col = (INIT_I-1) / n_rows   + 1
            assign_lines.append(
                f"      {arr_name}(MOD(INIT_I-1,{n_rows})+1,"
                f"(INIT_I-1)/{n_rows}+1) = 0.5D0"
            )
            assign_lines.append(f"      END DO")

    char_scalars = [n for n, d in arg_decls.items() if d.is_char and not d.is_array]
    if char_scalars:
        decl_lines.append("      CHARACTER*1 " + ", ".join(char_scalars))
        for n in char_scalars:
            assign_lines.append(f"      {n} = 'N'")

    logical_scalars = [n for n, d in arg_decls.items() if d.is_logical and not d.is_array]
    if logical_scalars:
        decl_lines.append("      LOGICAL " + ", ".join(logical_scalars))
        for n in logical_scalars:
            assign_lines.append(f"      {n} = .FALSE.")

    # Use _fortran_call to handle long argument lists with Fortran-77 continuation
    call_stmt = _fortran_call(routine_name.upper(), arg_names)

    return _FORTRAN_BENCH_TEMPLATE.format(
        name=routine_name.upper(),
        declarations="\n".join(decl_lines),
        assignments="\n".join(assign_lines),
        call_stmt=call_stmt,
        reps=reps,
    )


def _run_fortran_bench(bench_src: str, extra_sources: List[Path]) -> Optional[float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bench_f = tmp / "bench.f"
        bench_f.write_text(bench_src)
        exe = tmp / "bench"
        cmd = ["gfortran", "-O2", "-o", str(exe), str(bench_f)]
        cmd += [str(s) for s in extra_sources]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        if run.returncode != 0:
            return None
        try:
            return float(run.stdout.strip())
        except ValueError:
            return None


_RUST_BENCH_TEMPLATE = """\
// Auto-generated benchmark binary for `{fn_name}`
#![allow(unused_mut, unused_variables, dead_code)]
use std::time::Instant;
use {crate_name}::*;

fn main() {{
    const REPS: usize = {reps};
{rust_inputs}

    let start = Instant::now();
    for _ in 0..REPS {{
        unsafe {{ std::hint::black_box({fn_lower}({call_args})); }}
    }}
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    println!("{{:.6}}", elapsed_ms);
}}
"""


def _generate_rust_bench(
    routine_name: str,
    arg_names: List[str],
    arg_decls: Dict[str, ArgDecl],
    assigned_dims: Dict[str, int],
    crate_dir: Path,
    reps: int = _REPS,
) -> bool:
    fn_lower = routine_name.lower()
    examples_dir = crate_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    inputs: List[str] = []
    call_args: List[str] = []

    for name in arg_names:
        decl = arg_decls.get(name.upper(), ArgDecl(name=name.upper(), ftype="DOUBLE PRECISION", dims=[]))
        rname = name.lower()
        if decl.is_char:
            inputs.append(f"    let {rname}: u8 = b'N';")
            call_args.append(rname)
        elif decl.is_integer and not decl.is_array:
            inputs.append(f"    let {rname}: i32 = {assigned_dims.get(name.upper(), 4)};")
            call_args.append(rname)
        elif decl.is_real and not decl.is_array:
            inputs.append(f"    let {rname}: f64 = 1.0;")
            call_args.append(rname)
        elif decl.is_real and decl.is_array:
            sizes = _array_size(decl, assigned_dims)
            total = 1
            for s in (sizes or [4]):
                total *= s
            inputs.append(f"    let mut {rname} = vec![0.5_f64; {total}];")
            call_args.append(f"&mut {rname}")
        elif decl.is_logical:
            inputs.append(f"    let {rname}: bool = false;")
            call_args.append(rname)
        else:
            inputs.append(f"    let mut {rname}: f64 = 0.0;")
            call_args.append(f"&mut {rname}")

    crate_name = crate_dir.name
    rust_src = _RUST_BENCH_TEMPLATE.format(
        fn_name=routine_name,
        fn_lower=fn_lower,
        crate_name=crate_name,
        reps=reps,
        rust_inputs="\n".join(inputs),
        call_args=", ".join(call_args),
    )
    (examples_dir / f"bench_{fn_lower}.rs").write_text(rust_src)
    return True


def _run_rust_bench(crate_dir: Path, routine_name: str) -> Optional[float]:
    fn_lower = routine_name.lower()
    result = subprocess.run(
        ["cargo", "build", "--release", "--example", f"bench_{fn_lower}"],
        capture_output=True, text=True, cwd=crate_dir, timeout=120,
    )
    if result.returncode != 0:
        return None
    exe = crate_dir / "target" / "release" / "examples" / f"bench_{fn_lower}"
    if not exe.exists():
        return None
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    if run.returncode != 0:
        return None
    try:
        return float(run.stdout.strip())
    except ValueError:
        return None


def run_benchmark(
    function_name: str,
    fortran_source_path: Optional[Path],
    crate_dir: Optional[Path],
    *,
    routine=None,
    reps: int = _REPS,
) -> BenchResult:
    """Benchmark *function_name* in both Fortran and Rust.

    Works for any function: generates all benchmark drivers on the fly from
    the argument declarations parsed from the Fortran source.
    """
    fn = function_name.upper()
    details: List[str] = []

    if routine is not None:
        arg_names = routine.args
        arg_decls = parse_arg_declarations(routine.source, arg_names)
    elif fortran_source_path is not None and fortran_source_path.exists():
        from fortran_to_rust.parser import parse_file
        routines = parse_file(fortran_source_path)
        matched = [r for r in routines if r.name.upper() == fn]
        if matched:
            arg_names = matched[0].args
            arg_decls = parse_arg_declarations(matched[0].source, arg_names)
        else:
            return BenchResult(function_name=fn, error_message=f"Could not parse routine {fn}.")
    else:
        return BenchResult(function_name=fn, error_message="No routine or source path provided.")

    assigned_dims = _assign_dims(arg_decls)

    fortran_ms: Optional[float] = None
    if fortran_source_path and fortran_source_path.exists():
        extra = [fortran_source_path] + _find_support_files(fortran_source_path.parent)
        bench_src = _generate_fortran_bench(fn, arg_names, arg_decls, assigned_dims, reps)
        fortran_ms = _run_fortran_bench(bench_src, extra)
        if fortran_ms is not None:
            dim_info = ", ".join(f"{k}={v}" for k, v in sorted(assigned_dims.items()))
            details.append(f"  Fortran (gfortran -O2): {fortran_ms:.3f} ms/call  [{dim_info}]")
        else:
            details.append("  Fortran benchmark failed to compile/run.")

    rust_ms: Optional[float] = None
    if crate_dir and crate_dir.exists():
        _generate_rust_bench(fn, arg_names, arg_decls, assigned_dims, crate_dir, reps)
        rust_ms = _run_rust_bench(crate_dir, fn)
        if rust_ms is not None:
            details.append(f"  Rust   (--release):       {rust_ms:.3f} ms/call")
        else:
            details.append(
                "  Rust benchmark skipped "
                "(function not yet callable from the generated skeleton)."
            )

    speedup: Optional[float] = None
    if fortran_ms and rust_ms and rust_ms > 0:
        speedup = fortran_ms / rust_ms

    return BenchResult(
        function_name=fn,
        fortran_time_ms=fortran_ms,
        rust_time_ms=rust_ms,
        speedup=speedup,
        details=details,
    )
