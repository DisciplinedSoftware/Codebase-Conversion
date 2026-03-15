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

from fortran_to_rust.rust_project import get_crate_lib_name
from fortran_to_rust.test_harness import (
    ArgDecl,
    TestDataset,
    _array_size,
    _assign_dims,
    _find_support_files,
    _fortran_call,
    _get_fortran_lock,
    generate_dataset,
    parse_arg_declarations,
    write_dataset_file,
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
      INTEGER BENCH_R, RDSI, RDSJ
      DOUBLE PRECISION BENCH_T1, BENCH_T2, BENCH_ELAPSED
{declarations}
      ! Load shared dataset from file (excluded from timing).
      OPEN(UNIT=99, FILE='{dataset_path}', STATUS='OLD')
{read_stmts}
      CLOSE(99)
{const_stmts}
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
    dataset_path: Path,
    reps: int = _REPS,
) -> str:
    """Generate a Fortran benchmark program that loads inputs from *dataset_path*.

    The dataset file is read **before** the timed ``DO`` loop so that I/O does
    not inflate the benchmark measurement.
    """
    decl_lines: List[str] = []
    read_lines: List[str] = []
    const_lines: List[str] = []

    # Iterate arg_names in order to match the dataset file layout.
    int_scalars  = [n.upper() for n in arg_names
                    if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_integer
                    and not arg_decls[n.upper()].is_array]
    dp_scalars   = [n.upper() for n in arg_names
                    if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_real
                    and not arg_decls[n.upper()].is_array]
    char_scalars = [n.upper() for n in arg_names
                    if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_char
                    and not arg_decls[n.upper()].is_array]
    log_scalars  = [n.upper() for n in arg_names
                    if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_logical
                    and not arg_decls[n.upper()].is_array]
    array_args   = [(n.upper(), arg_decls[n.upper()])
                    for n in arg_names
                    if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_array
                    and arg_decls[n.upper()].is_real]

    if int_scalars:
        decl_lines.append("      INTEGER " + ", ".join(int_scalars))
    if dp_scalars:
        decl_lines.append("      DOUBLE PRECISION " + ", ".join(dp_scalars))
    for arr_name, decl in array_args:
        sizes = _array_size(decl, assigned_dims)
        if sizes:
            dim_str = ", ".join(str(s) for s in sizes)
            decl_lines.append(f"      DOUBLE PRECISION {arr_name}({dim_str})")
    if char_scalars:
        decl_lines.append("      CHARACTER*1 " + ", ".join(char_scalars))
    if log_scalars:
        decl_lines.append("      LOGICAL " + ", ".join(log_scalars))

    # Build READ statements for real args (same order as the dataset file).
    for name in arg_names:
        upper = name.upper()
        decl = arg_decls.get(upper)
        if decl is None or not decl.is_real:
            continue
        if not decl.is_array:
            read_lines.append(f"      READ(99,*) {upper}")
        else:
            sizes = _array_size(decl, assigned_dims)
            if not sizes:
                continue
            if len(sizes) == 1:
                read_lines += [
                    f"      DO RDSI=1,{sizes[0]}",
                    f"        READ(99,*) {upper}(RDSI)",
                    f"      END DO",
                ]
            elif len(sizes) == 2:
                read_lines += [
                    f"      DO RDSJ=1,{sizes[1]}",
                    f"        DO RDSI=1,{sizes[0]}",
                    f"          READ(99,*) {upper}(RDSI,RDSJ)",
                    f"        END DO",
                    f"      END DO",
                ]

    # Hardcode non-real constants.
    for n in int_scalars:
        const_lines.append(f"      {n} = {assigned_dims.get(n, 4)}")
    for n in char_scalars:
        const_lines.append(f"      {n} = 'N'")
    for n in log_scalars:
        const_lines.append(f"      {n} = .FALSE.")

    call_stmt = _fortran_call(routine_name.upper(), arg_names)

    return _FORTRAN_BENCH_TEMPLATE.format(
        name=routine_name.upper(),
        dataset_path=dataset_path,
        declarations="\n".join(decl_lines),
        read_stmts="\n".join(read_lines),
        const_stmts="\n".join(const_lines),
        call_stmt=call_stmt,
        reps=reps,
    )


def _run_fortran_bench(
    bench_src: str,
    extra_sources: List[Path],
    keep_dir: Optional[Path] = None,
    fn_stem: str = "bench",
) -> Optional[float]:
    """Compile *bench_src* and return measured ms/call.

    When *keep_dir* is provided the Fortran source and compiled executable are
    written there and **not** deleted afterwards, so they can be inspected.
    Otherwise a temporary directory is used and cleaned up automatically.
    *fn_stem* is used as the filename base so that different functions in the
    same *keep_dir* each get their own cached binary.
    """
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        bench_f = keep_dir / f"{fn_stem}.f"
        exe = keep_dir / fn_stem
        lock = _get_fortran_lock(str(exe))
        with lock:
            if not exe.exists():
                bench_f.write_text(bench_src)
                cmd = ["gfortran", "-O2", "-ffixed-line-length-none",
                       "-o", str(exe), str(bench_f)]
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bench_f = tmp / "bench.f"
        bench_f.write_text(bench_src)
        exe = tmp / "bench"
        cmd = ["gfortran", "-O2", "-ffixed-line-length-none",
               "-o", str(exe), str(bench_f)]
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

    // Load shared dataset from file (excluded from timing).
    let _ds_str = std::fs::read_to_string("{dataset_path}")
        .expect("dataset file not found");
    let mut _ds = _ds_str.split_ascii_whitespace()
        .map(|s| s.parse::<f64>().expect("bad float in dataset"));

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
    dataset_path: Path,
    reps: int = _REPS,
) -> bool:
    """Generate a Rust benchmark binary that loads inputs from *dataset_path*.

    The dataset file is read **before** ``Instant::now()`` so that I/O does not
    inflate the benchmark measurement.
    """
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
            inputs.append(f"    let {rname}: f64 = _ds.next().unwrap();")
            call_args.append(rname)
        elif decl.is_real and decl.is_array:
            sizes = _array_size(decl, assigned_dims)
            total = 1
            for s in (sizes or [4]):
                total *= s
            inputs.append(
                f"    let mut {rname}: Vec<f64> = (0..{total}).map(|_| _ds.next().unwrap()).collect();"
            )
            call_args.append(f"&mut {rname}")
        elif decl.is_logical:
            inputs.append(f"    let {rname}: bool = false;")
            call_args.append(rname)
        else:
            inputs.append(f"    let mut {rname}: f64 = 0.0;")
            call_args.append(f"&mut {rname}")

    crate_name = get_crate_lib_name(crate_dir)
    rust_src = _RUST_BENCH_TEMPLATE.format(
        fn_name=routine_name,
        fn_lower=fn_lower,
        crate_name=crate_name,
        dataset_path=dataset_path,
        reps=reps,
        rust_inputs="\n".join(inputs),
        call_args=", ".join(call_args),
    )
    (examples_dir / f"bench_{fn_lower}.rs").write_text(rust_src)
    return True


def _run_rust_bench(crate_dir: Path, routine_name: str) -> Optional[float]:
    import shutil

    if not shutil.which("cargo"):
        return None
    fn_lower = routine_name.lower()
    try:
        result = subprocess.run(
            ["cargo", "build", "--release", "--example", f"bench_{fn_lower}"],
            capture_output=True, text=True, cwd=crate_dir, timeout=120,
        )
    except FileNotFoundError:
        return None
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
    fortran_ref_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
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

    # Generate a single dataset (test_index=0) and write it to a shared file
    # that both Fortran and Rust benchmark drivers will read at runtime.
    fortran_keep_dir = (
        fortran_ref_dir
        if fortran_ref_dir is not None
        else ((crate_dir.parent / "fortran") if crate_dir else None)
    )
    if datasets_dir is None:
        if crate_dir:
            datasets_dir = crate_dir.parent / "datasets"
        elif fortran_keep_dir:
            datasets_dir = fortran_keep_dir / "datasets"
        else:
            import tempfile as _tf
            datasets_dir = Path(_tf.mkdtemp())
    datasets_dir.mkdir(parents=True, exist_ok=True)

    fn_lower = fn.lower()
    dataset = generate_dataset(arg_names, arg_decls, assigned_dims, test_index=0)
    dataset_path = datasets_dir / f"{fn_lower}_bench.txt"
    write_dataset_file(dataset, dataset_path)

    fortran_ms: Optional[float] = None
    if fortran_source_path and fortran_source_path.exists():
        extra = [fortran_source_path] + _find_support_files(fortran_source_path.parent)
        bench_src = _generate_fortran_bench(fn, arg_names, arg_decls, assigned_dims, dataset_path, reps)
        fortran_ms = _run_fortran_bench(bench_src, extra, keep_dir=fortran_keep_dir, fn_stem=f"{fn_lower}_bench")
        if fortran_ms is not None:
            dim_info = ", ".join(f"{k}={v}" for k, v in sorted(assigned_dims.items()))
            details.append(f"  Fortran (gfortran -O2): {fortran_ms:.3f} ms/call  [{dim_info}]")
        else:
            details.append("  Fortran benchmark failed to compile/run.")

    rust_ms: Optional[float] = None
    if crate_dir and crate_dir.exists():
        _generate_rust_bench(fn, arg_names, arg_decls, assigned_dims, crate_dir, dataset_path, reps)
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
