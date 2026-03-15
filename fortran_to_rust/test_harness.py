"""Numerical accuracy test harness.

For each converted function, this module:

1. **Parses** the Fortran argument declarations to discover types and array shapes.
2. **Generates** a :class:`TestDataset` and writes it to a plain-text file
   (one real value per line) via :func:`write_dataset_file`.
3. **Generates** a Fortran test driver (``PROGRAM``) that opens and reads that
   file at runtime to obtain its inputs.
4. **Compiles and runs** the Fortran driver with the *original* reference sources to
   obtain the expected outputs.
5. **Generates** a Rust example binary that reads the same file at runtime.
6. **Compares** the two output streams; reports max / mean absolute error.

Because both drivers read from the same on-disk file (steps 3 and 5), it is
structurally impossible for them to receive different numerical inputs.
Both are driven entirely by the information in the
:class:`~fortran_to_rust.parser.FortranRoutine` object, so the harness works for
*any* function without requiring function-specific knowledge.
"""

from __future__ import annotations

import random
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from fortran_to_rust.rust_project import get_crate_lib_name

_TOLERANCE = 1e-10  # max acceptable absolute error

# Per-file-stem locks prevent multiple threads from compiling the same Fortran
# driver simultaneously when a shared fortran_ref_dir is used across strategies.
_fortran_locks: Dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()


def _get_fortran_lock(key: str) -> threading.Lock:
    with _locks_mutex:
        if key not in _fortran_locks:
            _fortran_locks[key] = threading.Lock()
        return _fortran_locks[key]

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
# Test dataset dataclass
# ---------------------------------------------------------------------------

@dataclass
class TestDataset:
    """Concrete input values for a single test case, shared by Fortran and Rust drivers.

    ``values`` maps each uppercase argument name to its concrete value:
    - scalar real args  → ``float``
    - array real args   → ``List[float]`` (flat, column-major order)
    - integer scalars   → ``int``
    - char scalars      → ``str`` (single character, e.g. ``'N'``)
    - logical scalars   → ``bool``
    """

    test_index: int
    arg_names: List[str]
    values: Dict[str, Union[float, int, bool, str, List[float]]]


def generate_dataset(
    arg_names: List[str],
    arg_decls: Dict[str, "ArgDecl"],
    assigned_dims: Dict[str, int],
    test_index: int = 0,
) -> TestDataset:
    """Generate a single ``TestDataset`` for *test_index* from the given argument info.

    A single ``random.Random(test_index)`` RNG is iterated over ``arg_names``
    in order.  The same seed and iteration order are used in both
    ``generate_fortran_driver`` and ``_generate_rust_example`` so that both
    drivers are guaranteed to receive identical numerical inputs.
    """
    rng = random.Random(test_index)
    values: Dict[str, Union[float, int, bool, str, List[float]]] = {}

    for name in arg_names:
        upper = name.upper()
        decl = arg_decls.get(upper)
        if decl is None:
            continue
        if decl.is_char and not decl.is_array:
            values[upper] = "N"
        elif decl.is_logical and not decl.is_array:
            values[upper] = False
        elif decl.is_integer and not decl.is_array:
            values[upper] = assigned_dims.get(upper, 4)
        elif decl.is_real and not decl.is_array:
            values[upper] = rng.uniform(0.5, 2.0)
        elif decl.is_real and decl.is_array:
            sizes = _array_size(decl, assigned_dims)
            total = 1
            for s in (sizes or [4]):
                total *= s
            values[upper] = [rng.uniform(-1.0, 1.0) for _ in range(total)]

    return TestDataset(test_index=test_index, arg_names=arg_names, values=values)


def write_dataset_file(dataset: TestDataset, path: Path) -> None:
    """Write the real-valued entries of *dataset* to *path*, one value per line.

    Only DOUBLE PRECISION / REAL arguments are written (scalar → one line,
    array → one line per element in flat column-major order).  INTEGER,
    CHARACTER and LOGICAL arguments keep hardcoded constants in the generated
    source and are **not** included in the file.

    Both the Fortran driver and the Rust example binary read this file at
    runtime, guaranteeing identical numerical inputs without any re-seeding.
    """
    lines: List[str] = []
    for name in dataset.arg_names:
        val = dataset.values.get(name.upper())
        if isinstance(val, float):
            lines.append(repr(val))
        elif isinstance(val, list):
            for v in val:
                lines.append(repr(v))
        # int / bool / str (INTEGER, LOGICAL, CHARACTER) → hardcoded in source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


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


def _fortran_assign_call(result_var: str, routine_name: str, arg_names: List[str]) -> str:
    """Generate a Fortran-77 assignment call: RESULT = FUNC(args)."""
    prefix = "      "
    cont   = "     +"
    max_w  = 65

    full = f"{result_var} = {routine_name}({', '.join(arg_names)})"
    if len(full) <= max_w:
        return prefix + full

    lines: List[str] = []
    current = f"{result_var} = {routine_name}("
    for i, arg in enumerate(arg_names):
        is_last = i == len(arg_names) - 1
        sep = "" if current.endswith("(") else ", "
        closing = ")" if is_last else ""
        candidate = current + sep + arg + closing
        if len(candidate) <= max_w or current.endswith("("):
            current = candidate
        else:
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
    dataset_path: Path,
    routine_kind: str = "subroutine",
    return_ftype: str = "DOUBLE PRECISION",
) -> str:
    """Build a complete Fortran PROGRAM that calls *routine_name* and prints its outputs.

    For subroutines the output is the modified arrays/scalars.
    For functions (``routine_kind='function'``) the return value is captured and printed.

    Real-valued inputs are loaded at runtime from *dataset_path* (written by
    ``write_dataset_file``), so both the Fortran driver and the Rust example
    binary read from the same file and are guaranteed identical numerical inputs.
    INTEGER, CHARACTER and LOGICAL arguments are hardcoded as constants.
    """
    # --- Collect grouped types for Fortran declarations ---
    int_scalars = [n.upper() for n in arg_names
                   if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_integer
                   and not arg_decls[n.upper()].is_array]
    dp_scalars = [n.upper() for n in arg_names
                  if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_real
                  and not arg_decls[n.upper()].is_array]
    char_scalars = [n.upper() for n in arg_names
                    if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_char
                    and not arg_decls[n.upper()].is_array]
    logical_scalars = [n.upper() for n in arg_names
                       if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_logical
                       and not arg_decls[n.upper()].is_array]
    array_args = [(n.upper(), arg_decls[n.upper()])
                  for n in arg_names
                  if arg_decls.get(n.upper()) and arg_decls[n.upper()].is_array
                  and arg_decls[n.upper()].is_real]

    decl_lines: List[str] = []
    loop_int_vars: List[str] = []

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
    if logical_scalars:
        decl_lines.append("      LOGICAL " + ", ".join(logical_scalars))

    # --- Build file-reading section (OPEN / READ / CLOSE) ---
    # Real args are read from the shared dataset file in arg_names order.
    # RDSI / RDSJ are loop variables for array reads; add them to declarations.
    need_rdsi = any(
        arg_decls.get(n.upper()) and arg_decls[n.upper()].is_real
        and arg_decls[n.upper()].is_array
        for n in arg_names
    )
    need_rdsj = any(
        arg_decls.get(n.upper()) and arg_decls[n.upper()].is_real
        and arg_decls[n.upper()].is_array
        and len(_array_size(arg_decls[n.upper()], assigned_dims) or []) >= 2
        for n in arg_names
    )
    read_int_vars: List[str] = []
    if need_rdsi:
        read_int_vars.append("RDSI")
    if need_rdsj:
        read_int_vars.append("RDSJ")

    read_lines: List[str] = [
        f"      OPEN(UNIT=99, FILE='{dataset_path}', STATUS='OLD')",
    ]
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
    read_lines.append("      CLOSE(99)")

    # --- Constant assignments for non-real args ---
    const_lines: List[str] = []
    for name in arg_names:
        upper = name.upper()
        decl = arg_decls.get(upper)
        if decl is None or decl.is_real:
            continue
        if decl.is_char and not decl.is_array:
            const_lines.append(f"      {upper} = 'N'")
        elif decl.is_logical and not decl.is_array:
            const_lines.append(f"      {upper} = .FALSE.")
        elif decl.is_integer and not decl.is_array:
            const_lines.append(f"      {upper} = {assigned_dims.get(upper, 4)}")

    # --- Print loops for array outputs ---
    print_lines: List[str] = []
    for name in arg_names:
        upper = name.upper()
        decl = arg_decls.get(upper)
        if decl is None or not decl.is_real or not decl.is_array:
            continue
        sizes = _array_size(decl, assigned_dims)
        if not sizes:
            continue
        if len(sizes) == 1:
            iv = f"I{upper}"
            loop_int_vars.append(iv)
            print_lines += [
                f"      DO {iv}=1,{sizes[0]}",
                f"        WRITE(*,'(ES25.15)') {upper}({iv})",
                f"      END DO",
            ]
        elif len(sizes) == 2:
            iv = f"I{upper}"
            jv = f"J{upper}"
            loop_int_vars += [iv, jv]
            print_lines += [
                f"      DO {jv}=1,{sizes[1]}",
                f"        DO {iv}=1,{sizes[0]}",
                f"          WRITE(*,'(ES25.15)') {upper}({iv},{jv})",
                f"        END DO",
                f"      END DO",
            ]

    all_int_loop_vars = read_int_vars + loop_int_vars
    if all_int_loop_vars:
        decl_lines.append("      INTEGER " + ", ".join(all_int_loop_vars))

    # DP scalars (ALPHA, BETA, …) are read-only inputs in BLAS; printing them
    # after CALL would cause length mismatches in the Rust comparison without
    # testing the conversion. For functions the return value is printed via
    # result_var (handled below).

    # --- Handle FUNCTION vs SUBROUTINE call ---
    if routine_kind == "function":
        result_var = f"RES_{routine_name.upper()[:6]}"
        ftype_upper = return_ftype.strip().upper()
        if "INTEGER" in ftype_upper:
            ftype_decl = "INTEGER"
            print_fmt  = "(I20)"
        elif "LOGICAL" in ftype_upper:
            ftype_decl = "LOGICAL"
            print_fmt  = "(L5)"
        else:
            ftype_decl = "DOUBLE PRECISION"
            print_fmt  = "(ES25.15)"
        decl_lines.append(f"      {ftype_decl} {result_var}")
        decl_lines.append(f"      {ftype_decl} {routine_name.upper()}")
        decl_lines.append(f"      EXTERNAL {routine_name.upper()}")
        call_stmt = _fortran_assign_call(result_var, routine_name.upper(), arg_names)
        print_lines.append(f"      WRITE(*,'{print_fmt}') {result_var}")
    else:
        call_stmt = _fortran_call(routine_name.upper(), arg_names)

    parts = [
        _FORTRAN_DRIVER_HEADER.format(name=routine_name.upper()),
        "\n".join(decl_lines),
        "\n".join(read_lines),
        "\n".join(const_lines),
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
    keep_dir: Optional[Path] = None,
    file_stem: str = "test_driver",
) -> Optional[List[float]]:
    """Compile *driver_src* together with *extra_sources* and return printed floats.

    When *keep_dir* is provided the Fortran source and compiled executable are
    written there and **not** deleted afterwards, so they can be inspected.
    Otherwise a temporary directory is used and cleaned up automatically.
    """
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        driver_f = keep_dir / f"{file_stem}.f"
        exe = keep_dir / file_stem
        lock = _get_fortran_lock(str(keep_dir / file_stem))
        with lock:
            if not exe.exists():
                driver_f.write_text(driver_src)
                cmd = ["gfortran", "-O2", "-ffixed-line-length-none",
                       "-o", str(exe), str(driver_f)]
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        driver_f = tmp / "test_driver.f"
        driver_f.write_text(driver_src)
        exe = tmp / "test_driver"
        cmd = ["gfortran", "-O2", "-ffixed-line-length-none",
               "-o", str(exe), str(driver_f)]
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

_RUST_EXAMPLE_TEMPLATE_SUB = """\
// Auto-generated accuracy test binary for `{fn_name}`
#[allow(unused_imports)]
use {crate_name}::*;

fn main() {{
    // Load shared dataset from file (one real value per line, in argument order).
    let _ds_str = std::fs::read_to_string("{dataset_path}")
        .expect("dataset file not found");
    let mut _ds = _ds_str.split_ascii_whitespace()
        .map(|s| s.parse::<f64>().expect("bad float in dataset"));

{rust_inputs}

    // Safety: the generated function may be declared unsafe if the LLM used raw pointers.
    unsafe {{ {fn_lower}({call_args}); }}

    // Print outputs
{rust_prints}
}}
"""

_RUST_EXAMPLE_TEMPLATE_FN = """\
// Auto-generated accuracy test binary for `{fn_name}`
#[allow(unused_imports)]
use {crate_name}::*;

fn main() {{
    // Load shared dataset from file (one real value per line, in argument order).
    let _ds_str = std::fs::read_to_string("{dataset_path}")
        .expect("dataset file not found");
    let mut _ds = _ds_str.split_ascii_whitespace()
        .map(|s| s.parse::<f64>().expect("bad float in dataset"));

{rust_inputs}

    // Safety: the generated function may be declared unsafe if the LLM used raw pointers.
    let _result = unsafe {{ {fn_lower}({call_args}) }};

    // Print return value
    println!("{{:.15e}}", _result as f64);

    // Print any array outputs
{rust_prints}
}}
"""


def _generate_rust_example(
    routine_name: str,
    arg_names: List[str],
    arg_decls: Dict[str, ArgDecl],
    assigned_dims: Dict[str, int],
    crate_dir: Path,
    dataset_path: Path,
    routine_kind: str = "subroutine",
    return_ftype: str = "DOUBLE PRECISION",
) -> bool:
    """Write a Rust example binary that mirrors the Fortran test driver.

    Real inputs are loaded at runtime from *dataset_path* (written by
    ``write_dataset_file``), so the Rust example and the Fortran driver both
    read from the same on-disk file and are guaranteed identical numerical inputs.
    """
    fn_lower = routine_name.lower()
    crate_name = get_crate_lib_name(crate_dir)
    examples_dir = crate_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    inputs: List[str] = []
    call_args: List[str] = []
    prints: List[str] = []

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
            prints.append(f'    for v in &{rname} {{ println!("{{:.15e}}", v); }}')
        elif decl.is_logical:
            inputs.append(f"    let {rname}: bool = false;")
            call_args.append(rname)
        else:
            inputs.append(f"    let mut {rname}: f64 = 0.0;")
            call_args.append(f"&mut {rname}")

    template = _RUST_EXAMPLE_TEMPLATE_FN if routine_kind == "function" else _RUST_EXAMPLE_TEMPLATE_SUB
    rust_src = template.format(
        fn_name=routine_name,
        fn_lower=fn_lower,
        crate_name=crate_name,
        dataset_path=dataset_path,
        rust_inputs="\n".join(inputs),
        call_args=", ".join(call_args),
        rust_prints="\n".join(prints) if prints else "    // no array outputs",
    )
    (examples_dir / f"accuracy_{fn_lower}.rs").write_text(rust_src)
    return True


def _compile_run_rust_example(
    crate_dir: Path,
    routine_name: str,
) -> Optional[List[float]]:
    """Compile and run the Rust accuracy example; return printed floats or None."""
    import shutil

    if not shutil.which("cargo"):
        return None
    fn_lower = routine_name.lower()
    try:
        result = subprocess.run(
            ["cargo", "build", "--release", "--example", f"accuracy_{fn_lower}"],
            capture_output=True, text=True, cwd=crate_dir, timeout=120,
        )
    except FileNotFoundError:
        return None
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
    fortran_ref_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
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
    fortran_ok_count = 0  # explicit counter — avoids fragile string matching

    routine_kind   = getattr(routine, "kind", "subroutine")
    return_ftype   = getattr(routine, "return_type", None) or "DOUBLE PRECISION"

    fortran_keep_dir = (
        fortran_ref_dir
        if fortran_ref_dir is not None
        else ((crate_dir.parent / "fortran") if crate_dir else None)
    )

    # Resolve a stable directory for dataset files so both the Fortran driver
    # and the Rust example binary can find them by absolute path.
    # Default to the parent of the crate (= run_dir), keeping datasets at the
    # report root rather than buried inside the Rust crate directory.
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

    for t in range(num_tests):
        dataset = generate_dataset(arg_names, arg_decls, assigned_dims, test_index=t)
        dataset_path = datasets_dir / f"{fn_lower}_t{t}.txt"
        write_dataset_file(dataset, dataset_path)

        driver = generate_fortran_driver(
            fn, arg_names, arg_decls, assigned_dims, dataset_path,
            routine_kind=routine_kind, return_ftype=return_ftype,
        )
        fortran_out = _compile_run_fortran(
            driver, extra_sources,
            keep_dir=fortran_keep_dir,
            file_stem=f"{fn_lower}_test_driver_{t}",
        )

        if fortran_out is None:
            details.append(f"  Test {t+1}: Fortran reference failed to compile/run.")
            continue

        fortran_ok_count += 1
        details.append(f"  Test {t+1}: Fortran reference produced {len(fortran_out)} value(s).")

        rust_out: Optional[List[float]] = None
        if crate_dir and crate_dir.exists():
            _generate_rust_example(
                fn, arg_names, arg_decls, assigned_dims, crate_dir, dataset_path,
                routine_kind=routine_kind, return_ftype=return_ftype,
            )
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
        elif rust_out is not None:
            details.append(
                f"  Test {t+1}: Rust produced {len(rust_out)} value(s) vs "
                f"Fortran {len(fortran_out)} — length mismatch, comparison skipped."
            )
        else:
            details.append(
                f"  Test {t+1}: Rust binary not available -- "
                "Fortran reference computed, numerical comparison skipped."
            )

    if not errors:
        msg = (
            "Fortran reference ran successfully. "
            "No Rust binary available for numerical comparison."
            if fortran_ok_count > 0
            else "No tests completed successfully."
        )
        return AccuracyResult(
            function_name=fn,
            passed=fortran_ok_count > 0,
            num_test_cases=num_tests,
            error_message=msg,
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
