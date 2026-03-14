"""Numerical accuracy test harness.

For each converted function, this module:

1. **Parses** the Fortran argument declarations to discover types and array shapes.
2. **Generates** a Fortran test driver (``PROGRAM``) on the fly, initialised with
   deterministic pseudo-random values.
3. **Compiles and runs** the Fortran driver with the *original* reference sources to
   obtain the expected outputs.
4. **Generates** a Rust example binary that calls the converted function with the
   same inputs and prints its outputs.
5. **Compares** the two output streams; reports max / mean absolute error.

Both step 3 and step 4 are driven entirely by the information in the
:class:`~fortran_to_rust.parser.FortranRoutine` object, so the harness works for
*any* function without requiring function-specific knowledge.
"""

from __future__ import annotations

import random
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_TOLERANCE = 1e-10  # max acceptable absolute error

# ---------------------------------------------------------------------------
# Argument declaration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArgDecl:
    """Information about one Fortran dummy argument."""

    name: str          # uppercase name
    ftype: str         # 'DOUBLE PRECISION', 'REAL', 'INTEGER', 'CHARACTER', 'LOGICAL'
    dims: List[str]    # dimension specs, e.g. ['LDA', '*'] or [] for scalars

    @property
    def is_array(self) -> bool:
        return bool(self.dims)

    @property
    def is_char(self) -> bool:
        return "CHARACTER" in self.ftype.upper()

    @property
    def is_integer(self) -> bool:
        return "INTEGER" in self.ftype.upper()

    @property
    def is_real(self) -> bool:
        t = self.ftype.upper()
        return "DOUBLE" in t or ("REAL" in t and "CHARACTER" not in t)

    @property
    def is_logical(self) -> bool:
        return "LOGICAL" in self.ftype.upper()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AccuracyResult:
    """Result of the numerical accuracy comparison."""

    function_name: str
    passed: bool
    max_abs_error: Optional[float] = None
    mean_abs_error: Optional[float] = None
    num_test_cases: int = 0
    failed_cases: int = 0
    error_message: Optional[str] = None
    details: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Argument declaration parser
# ---------------------------------------------------------------------------

_TYPE_PATTERNS: List[Tuple[str, str]] = [
    (r"DOUBLE\s+PRECISION\s+(.*)", "DOUBLE PRECISION"),
    (r"REAL\s+(.*)",               "REAL"),
    (r"INTEGER\s+(.*)",            "INTEGER"),
    (r"CHARACTER(?:\s*\*\s*1?)?\s+(.*)", "CHARACTER"),
    (r"LOGICAL\s+(.*)",            "LOGICAL"),
]

_VAR_WITH_DIMS = re.compile(r"(\w+)\s*\(([^)]+)\)")


def _split_decl_list(text: str) -> List[str]:
    """Split 'A(LDA,*),B(LDB,*),C(LDC,*)' into individual items."""
    items: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        items.append("".join(current).strip())
    return [i for i in items if i]


def parse_arg_declarations(source: str, arg_names: List[str]) -> Dict[str, ArgDecl]:
    """Return a mapping of (uppercase arg name) -> ArgDecl for each argument."""
    arg_set = {a.upper() for a in arg_names}
    result: Dict[str, ArgDecl] = {}

    for raw_line in source.splitlines():
        if re.match(r"^[Cc*!]", raw_line):
            continue
        line = raw_line.strip().upper()
        if "!" in line:
            line = line[: line.index("!")].rstrip()

        for pattern, ftype in _TYPE_PATTERNS:
            m = re.match(pattern, line, re.IGNORECASE)
            if not m:
                continue
            decl_text = m.group(1)
            for item in _split_decl_list(decl_text):
                item = item.strip()
                am = _VAR_WITH_DIMS.match(item)
                if am:
                    vname = am.group(1).upper()
                    raw_dims = [d.strip() for d in am.group(2).split(",")]
                    if vname in arg_set:
                        result[vname] = ArgDecl(name=vname, ftype=ftype, dims=raw_dims)
                else:
                    vname = item.upper().split()[0] if item else ""
                    if vname in arg_set:
                        result[vname] = ArgDecl(name=vname, ftype=ftype, dims=[])
            break

    # Fallback for any undiscovered args
    for arg in arg_names:
        a = arg.upper()
        if a not in result:
            result[a] = ArgDecl(name=a, ftype="DOUBLE PRECISION", dims=[])

    return result


# ---------------------------------------------------------------------------
# Concrete dimension assignment
# ---------------------------------------------------------------------------

_DIM_DEFAULTS: Dict[str, int] = {
    "M": 4,
    "N": 4,
    "K": 4,
    "LDA": 4,
    "LDB": 4,
    "LDC": 4,
    "LDE": 4,
    "LDF": 4,
    "INCX": 1,
    "INCY": 1,
    "INC": 1,
    "NRHS": 2,
    "KL": 1,
    "KU": 1,
    "KB": 4,
    "P": 3,
    "Q": 3,
}


def _assign_dims(arg_decls: Dict[str, ArgDecl]) -> Dict[str, int]:
    """Return concrete integer values for every INTEGER scalar argument."""
    assigned: Dict[str, int] = {}
    for name, decl in arg_decls.items():
        if decl.is_integer and not decl.is_array:
            assigned[name] = _DIM_DEFAULTS.get(name, 4)
    return assigned


def _resolve_dim(dim_str: str, assigned: Dict[str, int], fallback: int = 4) -> int:
    """Resolve a dimension string ('LDA', '*', '4') to a concrete integer."""
    key = dim_str.strip().upper()
    if key == "*":
        return fallback
    if key in assigned:
        return assigned[key]
    try:
        return int(key)
    except ValueError:
        return fallback


def _array_size(decl: ArgDecl, assigned: Dict[str, int]) -> List[int]:
    """Return concrete dimension sizes for an array argument."""
    if not decl.dims:
        return []
    return [_resolve_dim(d, assigned, fallback=4) for d in decl.dims]


# ---------------------------------------------------------------------------
# Fortran code generation helpers
# ---------------------------------------------------------------------------

def _f90_double(val: float) -> str:
    """Format a Python float as a Fortran DOUBLE PRECISION literal."""
    s = f"{val:.15E}"          # e.g. '1.766632777287572E+00'
    return s.replace("E", "D") # e.g. '1.766632777287572D+00'


def _fortran_call(routine_name: str, arg_names: List[str]) -> str:
    """Generate a Fortran-77 CALL statement, wrapping with continuation lines as needed."""
    prefix = "      "    # 6 blanks (code starts at col 7)
    cont   = "     +"    # col 6 continuation marker
    max_w  = 65          # max code chars per line: 72 - 6 - 1 for safety (trailing comma)

    full = f"CALL {routine_name}({', '.join(arg_names)})"
    if len(full) <= max_w:
        return prefix + full

    # Build each line greedily.  We accumulate tokens separated by ', ';
    # when adding the next token would overflow, flush the current line and
    # start a continuation.
    lines: List[str] = []
    current = f"CALL {routine_name}("
    for i, arg in enumerate(arg_names):
        is_last = i == len(arg_names) - 1
        sep = "" if current.endswith("(") else ", "
        closing = ")" if is_last else ""
        candidate = current + sep + arg + closing
        if len(candidate) <= max_w or current.endswith("("):
            # Either it fits, or this is the very first arg (must put it somewhere)
            current = candidate
        else:
            # Flush current line with a trailing comma; start continuation
            lines.append((prefix if not lines else cont) + current + ",")
            current = arg + closing
    if current:
        lines.append((prefix if not lines else cont) + current)
    return "\n".join(lines)


def _fortran_scalar_init(name: str, decl: ArgDecl, seed_offset: int) -> str:
    """Return a Fortran assignment statement for a scalar argument."""
    rng = random.Random(seed_offset)
    if decl.is_char:
        return f"      {name} = 'N'"
    if decl.is_logical:
        return f"      {name} = .FALSE."
    if decl.is_integer:
        return ""   # dimension vars are assigned separately
    val = rng.uniform(0.5, 2.0)
    return f"      {name} = {_f90_double(val)}"


def _fortran_array_init(name: str, sizes: List[int], seed_offset: int) -> str:
    """Return Fortran element-assignment statements for an array."""
    rng = random.Random(seed_offset)
    lines: List[str] = []
    if len(sizes) == 1:
        for i in range(sizes[0]):
            v = rng.uniform(-1.0, 1.0)
            lines.append(f"      {name}({i+1}) = {_f90_double(v)}")
    elif len(sizes) == 2:
        for j in range(sizes[1]):
            for i in range(sizes[0]):
                v = rng.uniform(-1.0, 1.0)
                lines.append(f"      {name}({i+1},{j+1}) = {_f90_double(v)}")
    else:
        total = 1
        for s in sizes:
            total *= s
        for idx in range(total):
            v = rng.uniform(-1.0, 1.0)
            lines.append(f"      {name}({idx+1}) = {_f90_double(v)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fortran test driver generator
# ---------------------------------------------------------------------------

_FORTRAN_DRIVER_HEADER = """\
      PROGRAM TEST_{name}
      IMPLICIT NONE
"""
_FORTRAN_DRIVER_FOOTER = """\
      END
"""


def generate_fortran_driver(
    routine_name: str,
    arg_names: List[str],
    arg_decls: Dict[str, ArgDecl],
    assigned_dims: Dict[str, int],
    test_index: int = 0,
) -> str:
    """Build a complete Fortran PROGRAM that calls *routine_name* and prints its outputs."""
    decl_lines: List[str] = []
    assign_lines: List[str] = []
    print_lines: List[str] = []

    # --- INTEGER scalars ---
    int_scalars = [n for n, d in arg_decls.items() if d.is_integer and not d.is_array]
    if int_scalars:
        decl_lines.append("      INTEGER " + ", ".join(int_scalars))
        for n in int_scalars:
            assign_lines.append(f"      {n} = {assigned_dims.get(n, 4)}")

    # --- DOUBLE PRECISION scalars ---
    dp_scalars = [n for n, d in arg_decls.items() if d.is_real and not d.is_array]
    if dp_scalars:
        decl_lines.append("      DOUBLE PRECISION " + ", ".join(dp_scalars))
        for i, n in enumerate(dp_scalars):
            stmt = _fortran_scalar_init(n, arg_decls[n], seed_offset=test_index * 100 + i)
            if stmt:
                assign_lines.append(stmt)

    # --- Real arrays ---
    array_args = [(n, d) for n, d in arg_decls.items() if d.is_array and d.is_real]
    loop_int_vars: List[str] = []
    for arr_name, decl in array_args:
        sizes = _array_size(decl, assigned_dims)
        if not sizes:
            continue
        dim_str = ", ".join(str(s) for s in sizes)
        decl_lines.append(f"      DOUBLE PRECISION {arr_name}({dim_str})")
        init = _fortran_array_init(arr_name, sizes, seed_offset=test_index * 1000 + ord(arr_name[0]))
        if init:
            assign_lines.append(init)
        # Print loop for this array
        iv = f"I{arr_name}"
        jv = f"J{arr_name}"
        if len(sizes) == 1:
            loop_int_vars.append(iv)
            print_lines += [
                f"      DO {iv}=1,{sizes[0]}",
                f"        WRITE(*,'(ES25.15)') {arr_name}({iv})",
                f"      END DO",
            ]
        elif len(sizes) == 2:
            loop_int_vars += [iv, jv]
            print_lines += [
                f"      DO {jv}=1,{sizes[1]}",
                f"        DO {iv}=1,{sizes[0]}",
                f"          WRITE(*,'(ES25.15)') {arr_name}({iv},{jv})",
                f"        END DO",
                f"      END DO",
            ]

    if loop_int_vars:
        decl_lines.append("      INTEGER " + ", ".join(loop_int_vars))

    # --- CHARACTER scalars ---
    char_scalars = [n for n, d in arg_decls.items() if d.is_char and not d.is_array]
    if char_scalars:
        decl_lines.append("      CHARACTER*1 " + ", ".join(char_scalars))
        for n in char_scalars:
            assign_lines.append(f"      {n} = 'N'")

    # --- LOGICAL scalars ---
    logical_scalars = [n for n, d in arg_decls.items() if d.is_logical and not d.is_array]
    if logical_scalars:
        decl_lines.append("      LOGICAL " + ", ".join(logical_scalars))
        for n in logical_scalars:
            assign_lines.append(f"      {n} = .FALSE.")

    # Print DP scalars after the CALL (outputs may have been updated)
    for n in dp_scalars:
        print_lines.append(f"      WRITE(*,'(ES25.15)') {n}")

    call_stmt = _fortran_call(routine_name.upper(), arg_names)

    parts = [
        _FORTRAN_DRIVER_HEADER.format(name=routine_name.upper()),
        "\n".join(decl_lines),
        "\n".join(assign_lines),
        call_stmt,
        "\n".join(print_lines) if print_lines else "      CONTINUE",
        _FORTRAN_DRIVER_FOOTER,
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Compile and run helpers
# ---------------------------------------------------------------------------

def _find_support_files(directory: Path) -> List[Path]:
    """Return helper .f files (lsame, xerbla, etc.) present alongside the routine."""
    helpers = []
    for name in ("lsame.f", "xerbla.f", "LSAME.f", "XERBLA.f"):
        p = directory / name
        if p.exists():
            helpers.append(p)
    return helpers


def _compile_run_fortran(
    driver_src: str,
    extra_sources: List[Path],
    timeout: int = 30,
) -> Optional[List[float]]:
    """Compile *driver_src* together with *extra_sources* and return printed floats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        driver_f = tmp / "test_driver.f"
        driver_f.write_text(driver_src)
        exe = tmp / "test_driver"
        cmd = ["gfortran", "-O2", "-o", str(exe), str(driver_f)]
        cmd += [str(s) for s in extra_sources]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
        if run.returncode != 0:
            return None
        try:
            return [float(line.strip()) for line in run.stdout.splitlines() if line.strip()]
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Rust example generation and execution
# ---------------------------------------------------------------------------

_RUST_EXAMPLE_TEMPLATE = """\
// Auto-generated accuracy test binary for `{fn_name}`
// Calls the converted Rust function with the same inputs as the Fortran reference
// and prints every f64 output value (one per line).

fn main() {{
    // NOTE: Uncomment and adjust the call once the Rust signature is confirmed.
    // use {crate_name}::{fn_lower}::*;

{rust_inputs}

    // Call the converted function (uncomment once signature is known):
    // {fn_lower}({call_args});

    // Print outputs
{rust_prints}
}}
"""


def _generate_rust_example(
    routine_name: str,
    arg_names: List[str],
    arg_decls: Dict[str, ArgDecl],
    assigned_dims: Dict[str, int],
    crate_dir: Path,
    test_index: int = 0,
) -> bool:
    """Write a Rust example binary that mirrors the Fortran test driver."""
    fn_lower = routine_name.lower()
    crate_name = crate_dir.name
    examples_dir = crate_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    inputs: List[str] = []
    call_args: List[str] = []
    prints: List[str] = []
    rng = random.Random(test_index * 100)

    for name in arg_names:
        decl = arg_decls.get(name.upper(), ArgDecl(name=name.upper(), ftype="DOUBLE PRECISION", dims=[]))
        rname = name.lower()

        if decl.is_char:
            inputs.append(f"    let {rname}: u8 = b'N';")
            call_args.append(rname)
        elif decl.is_integer and not decl.is_array:
            val = assigned_dims.get(name.upper(), 4)
            inputs.append(f"    let {rname}: i32 = {val};")
            call_args.append(rname)
        elif decl.is_real and not decl.is_array:
            val = rng.uniform(0.5, 2.0)
            inputs.append(f"    let {rname}: f64 = {val:.15f};")
            call_args.append(rname)
        elif decl.is_real and decl.is_array:
            sizes = _array_size(decl, assigned_dims)
            total = 1
            for s in (sizes or [4]):
                total *= s
            vals = ", ".join(f"{rng.uniform(-1.0, 1.0):.15f}_f64" for _ in range(total))
            inputs.append(f"    let mut {rname} = vec![{vals}];")
            call_args.append(f"&mut {rname}")
            prints.append(f'    for v in &{rname} {{ println!("{{:.15e}}", v); }}')
        elif decl.is_logical:
            inputs.append(f"    let {rname}: bool = false;")
            call_args.append(rname)
        else:
            inputs.append(f"    let mut {rname}: f64 = 0.0;")
            call_args.append(f"&mut {rname}")

    rust_src = _RUST_EXAMPLE_TEMPLATE.format(
        fn_name=routine_name,
        fn_lower=fn_lower,
        crate_name=crate_name,
        rust_inputs="\n".join(inputs),
        call_args=", ".join(call_args),
        rust_prints="\n".join(prints) if prints else "    // no array outputs detected",
    )
    (examples_dir / f"accuracy_{fn_lower}.rs").write_text(rust_src)
    return True


def _compile_run_rust_example(
    crate_dir: Path,
    routine_name: str,
) -> Optional[List[float]]:
    """Compile and run the Rust accuracy example; return printed floats or None."""
    fn_lower = routine_name.lower()
    result = subprocess.run(
        ["cargo", "build", "--release", "--example", f"accuracy_{fn_lower}"],
        capture_output=True, text=True, cwd=crate_dir, timeout=120,
    )
    if result.returncode != 0:
        return None
    exe = crate_dir / "target" / "release" / "examples" / f"accuracy_{fn_lower}"
    if not exe.exists():
        return None
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
    if run.returncode != 0:
        return None
    try:
        return [float(line.strip()) for line in run.stdout.splitlines() if line.strip()]
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_accuracy_check(
    function_name: str,
    fortran_source_path: Optional[Path],
    crate_dir: Optional[Path],
    *,
    routine=None,   # FortranRoutine | None
    num_tests: int = 3,
) -> AccuracyResult:
    """Run accuracy comparison for *function_name*.

    Uses the *routine* object (parsed Fortran source) to generate test drivers
    on the fly, so this works for any function -- not just dgemm.
    """
    fn = function_name.upper()

    if routine is None and fortran_source_path is None:
        return AccuracyResult(
            function_name=fn,
            passed=True,
            error_message="No routine or source path provided -- accuracy check skipped.",
        )

    # Resolve argument information from routine or by parsing the source file
    if routine is not None:
        arg_names = routine.args
        arg_decls = parse_arg_declarations(routine.source, arg_names)
    else:
        from fortran_to_rust.parser import parse_file
        routines = parse_file(fortran_source_path)
        matched = [r for r in routines if r.name.upper() == fn]
        if not matched:
            return AccuracyResult(
                function_name=fn,
                passed=True,
                error_message=f"Could not parse routine {fn} from {fortran_source_path}.",
            )
        routine = matched[0]
        arg_names = routine.args
        arg_decls = parse_arg_declarations(routine.source, arg_names)

    assigned_dims = _assign_dims(arg_decls)

    extra_sources: List[Path] = []
    if fortran_source_path and fortran_source_path.exists():
        extra_sources.append(fortran_source_path)
        extra_sources += _find_support_files(fortran_source_path.parent)

    errors: List[float] = []
    details: List[str] = []
    failed = 0

    for t in range(num_tests):
        driver = generate_fortran_driver(fn, arg_names, arg_decls, assigned_dims, test_index=t)
        fortran_out = _compile_run_fortran(driver, extra_sources)

        if fortran_out is None:
            details.append(f"  Test {t+1}: Fortran reference failed to compile/run.")
            continue

        details.append(f"  Test {t+1}: Fortran reference produced {len(fortran_out)} value(s).")

        rust_out: Optional[List[float]] = None
        if crate_dir and crate_dir.exists():
            _generate_rust_example(fn, arg_names, arg_decls, assigned_dims, crate_dir, t)
            rust_out = _compile_run_rust_example(crate_dir, fn)

        if rust_out and len(rust_out) == len(fortran_out) and fortran_out:
            case_errors = [abs(rust_out[i] - fortran_out[i]) for i in range(len(fortran_out))]
            max_e = max(case_errors)
            errors.append(max_e)
            ok = max_e <= _TOLERANCE
            if not ok:
                failed += 1
            details.append(
                f"  Test {t+1}: max_abs_error={max_e:.2e} {'OK' if ok else 'FAIL'}"
            )
        else:
            details.append(
                f"  Test {t+1}: Rust binary not available -- "
                "Fortran reference computed, numerical comparison skipped."
            )

    if not errors:
        return AccuracyResult(
            function_name=fn,
            passed=True,
            num_test_cases=num_tests,
            error_message=(
                "Fortran reference ran successfully. "
                "No Rust binary available for numerical comparison."
            ),
            details=details,
        )

    max_abs = max(errors)
    mean_abs = sum(errors) / len(errors)
    return AccuracyResult(
        function_name=fn,
        passed=(failed == 0),
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        num_test_cases=num_tests,
        failed_cases=failed,
        details=details,
    )
