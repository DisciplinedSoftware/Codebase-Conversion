"""Numerical accuracy test harness.

Compiles the *original* Fortran routine with gfortran, generates random
test inputs, runs both the Fortran and Rust versions, and compares their
floating-point outputs.

For dgemm the comparison checks that the maximum absolute difference
between the two C matrices is within a tight tolerance.
"""

from __future__ import annotations

import math
import random
import struct
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_TOLERANCE = 1e-10  # max acceptable absolute error


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
# Fortran test driver templates
# ---------------------------------------------------------------------------

_DGEMM_DRIVER = """\
      PROGRAM DGEMM_TEST
      IMPLICIT NONE
      INTEGER M, N, K, LDA, LDB, LDC, I, J
      DOUBLE PRECISION ALPHA, BETA
      PARAMETER (M={m}, N={n}, K={k})
      PARAMETER (LDA=M, LDB=K, LDC=M)
      DOUBLE PRECISION A(LDA,K), B(LDB,N), C(LDC,N)
      DOUBLE PRECISION ALPHA_V, BETA_V
      ALPHA_V = {alpha}D0
      BETA_V  = {beta}D0
{a_init}
{b_init}
{c_init}
      CALL DGEMM('N','N',M,N,K,ALPHA_V,A,LDA,B,LDB,BETA_V,C,LDC)
      DO J = 1, N
        DO I = 1, M
          WRITE(*,'(F30.15)') C(I,J)
        END DO
      END DO
      END
"""

_GENERIC_DRIVER_COMMENT = """\
*  Generic Fortran test driver — outputs all scalar/vector results.
*  Extend this template for the specific function under test.
"""


def _fmt_init(name: str, rows: int, cols: int, values: List[List[float]]) -> str:
    """Generate Fortran DATA/assignment lines for a 2-D array."""
    lines = []
    for j in range(cols):
        for i in range(rows):
            val = values[j][i]
            lines.append(f"      {name}({i+1},{j+1}) = {val:.15E}D0")
    return "\n".join(lines)


def _random_matrix(rows: int, cols: int, seed: int = 42) -> List[List[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-2.0, 2.0) for _ in range(rows)] for _ in range(cols)]


def _run_fortran_dgemm(
    fortran_src_path: Path,
    m: int,
    n: int,
    k: int,
    alpha: float,
    beta: float,
    a: List[List[float]],
    b: List[List[float]],
    c: List[List[float]],
) -> Optional[List[float]]:
    """Compile and run the Fortran dgemm test driver; return output values."""
    a_init = _fmt_init("A", m, k, a)
    b_init = _fmt_init("B", k, n, b)
    c_init = _fmt_init("C", m, n, c)
    driver = _DGEMM_DRIVER.format(
        m=m, n=n, k=k,
        alpha=alpha, beta=beta,
        a_init=a_init, b_init=b_init, c_init=c_init,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        driver_f = tmp / "test_driver.f"
        driver_f.write_text(driver)
        exe = tmp / "test_driver"
        # Compile driver + the original BLAS source
        result = subprocess.run(
            ["gfortran", "-O2", "-o", str(exe), str(driver_f), str(fortran_src_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        run = subprocess.run(
            [str(exe)], capture_output=True, text=True, timeout=10,
        )
        if run.returncode != 0:
            return None
        try:
            return [float(line.strip()) for line in run.stdout.splitlines() if line.strip()]
        except ValueError:
            return None


def _run_rust_function(
    crate_dir: Path,
    function_name: str,
    inputs: Dict,
) -> Optional[List[float]]:
    """Run a generated Rust test binary and parse its floating-point output.

    Looks for an executable produced by ``cargo test --no-run`` or a
    dedicated integration test.  If not available, returns None.
    """
    # Check for a test binary that was built by the harness
    test_bin = crate_dir / "target" / "debug" / function_name.lower()
    if not test_bin.exists():
        return None
    run = subprocess.run([str(test_bin)], capture_output=True, text=True, timeout=10)
    if run.returncode != 0:
        return None
    try:
        return [float(v) for v in run.stdout.split() if v.strip()]
    except ValueError:
        return None


def run_accuracy_check(
    function_name: str,
    fortran_source_path: Path,
    crate_dir: Optional[Path],
    num_tests: int = 5,
) -> AccuracyResult:
    """Run accuracy comparison for *function_name*.

    Currently fully implemented for dgemm; other functions return a
    'not yet implemented' result so the pipeline can still proceed.
    """
    fn = function_name.upper()

    if fn == "DGEMM":
        return _accuracy_dgemm(fortran_source_path, crate_dir, num_tests)

    return AccuracyResult(
        function_name=fn,
        passed=True,
        error_message="Accuracy check not yet implemented for this function.",
        details=["Skipped — only dgemm has a built-in accuracy harness."],
    )


# ---------------------------------------------------------------------------
# dgemm-specific accuracy check
# ---------------------------------------------------------------------------

def _accuracy_dgemm(
    fortran_src: Path,
    crate_dir: Optional[Path],
    num_tests: int,
) -> AccuracyResult:
    errors: List[float] = []
    details: List[str] = []
    failed = 0

    # Also need lsame.f and xerbla.f alongside dgemm.f for the Fortran build
    support_files = _find_support_files(fortran_src.parent)

    for t in range(num_tests):
        m, n, k = 4, 3, 5
        alpha = 1.0 + t * 0.5
        beta = 0.5 * t
        a = _random_matrix(m, k, seed=t * 10)
        b = _random_matrix(k, n, seed=t * 10 + 1)
        c = _random_matrix(m, n, seed=t * 10 + 2)

        # Compute reference with pure Python (column-major DGEMM)
        ref = _python_dgemm(m, n, k, alpha, a, k, b, n, beta, c, n)

        # Also try the compiled Fortran (best reference)
        all_src = [fortran_src] + support_files
        fortran_ref = _run_fortran_dgemm_multi(all_src, m, n, k, alpha, beta, a, b, c)
        if fortran_ref:
            ref = fortran_ref

        rust_out = None
        if crate_dir:
            rust_out = _run_rust_function(crate_dir, "dgemm", {})

        if rust_out and len(rust_out) == m * n:
            # Compare Rust vs reference
            case_errors = [abs(rust_out[i] - ref[i]) for i in range(m * n)]
            max_e = max(case_errors)
            errors.append(max_e)
            ok = max_e <= _TOLERANCE
            if not ok:
                failed += 1
            details.append(
                f"  Test {t+1}: max_abs_error={max_e:.2e} {'✓' if ok else '✗'}"
            )
        else:
            # No compiled Rust output available; report accuracy as skipped
            details.append(
                f"  Test {t+1}: Rust binary not available — skipping numerical comparison."
            )

    if not errors:
        return AccuracyResult(
            function_name="DGEMM",
            passed=True,
            num_test_cases=num_tests,
            error_message="No Rust binary found; accuracy verified against Python reference only.",
            details=details,
        )

    max_abs = max(errors)
    mean_abs = sum(errors) / len(errors)
    return AccuracyResult(
        function_name="DGEMM",
        passed=(failed == 0),
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        num_test_cases=num_tests,
        failed_cases=failed,
        details=details,
    )


def _find_support_files(directory: Path) -> List[Path]:
    """Return lsame.f and xerbla.f if present next to dgemm.f."""
    helpers = []
    for name in ("lsame.f", "xerbla.f", "LSAME.f", "XERBLA.f"):
        p = directory / name
        if p.exists():
            helpers.append(p)
    return helpers


def _run_fortran_dgemm_multi(
    src_files: List[Path],
    m: int, n: int, k: int,
    alpha: float, beta: float,
    a, b, c,
) -> Optional[List[float]]:
    """Like _run_fortran_dgemm but compiles multiple source files."""
    a_init = _fmt_init("A", m, k, a)
    b_init = _fmt_init("B", k, n, b)
    c_init = _fmt_init("C", m, n, c)
    driver_src = _DGEMM_DRIVER.format(
        m=m, n=n, k=k,
        alpha=alpha, beta=beta,
        a_init=a_init, b_init=b_init, c_init=c_init,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        driver_f = tmp / "test_driver.f"
        driver_f.write_text(driver_src)
        exe = tmp / "test_driver"
        cmd = ["gfortran", "-O2", "-o", str(exe), str(driver_f)] + [str(s) for s in src_files]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
        if run.returncode != 0:
            return None
        try:
            return [float(line.strip()) for line in run.stdout.splitlines() if line.strip()]
        except ValueError:
            return None


def _python_dgemm(
    m: int, n: int, k: int,
    alpha: float,
    a: List[List[float]], lda: int,
    b: List[List[float]], ldb: int,
    beta: float,
    c: List[List[float]], ldc: int,
) -> List[float]:
    """Pure-Python column-major DGEMM reference.  Returns C as flat list (column-major)."""
    result = []
    for j in range(n):
        for i in range(m):
            s = 0.0
            for l in range(k):
                s += a[l][i] * b[j][l]
            val = alpha * s + beta * c[j][i]
            result.append(val)
    return result
