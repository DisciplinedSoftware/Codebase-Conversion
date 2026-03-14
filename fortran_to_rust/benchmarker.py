"""Performance benchmarker.

Measures wall-clock time for both the Fortran and Rust implementations
of a function and reports a speedup ratio.

Uses ``gfortran`` for Fortran timing and ``cargo bench`` (or a simple
timing loop) for Rust timing.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BenchResult:
    """Performance comparison result."""

    function_name: str
    fortran_time_ms: Optional[float] = None   # median wall-clock ms
    rust_time_ms: Optional[float] = None       # median wall-clock ms
    speedup: Optional[float] = None            # fortran / rust  (>1 = Rust faster)
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
                f"Fortran: {self.fortran_time_ms:.2f} ms  |  "
                f"Rust: {self.rust_time_ms:.2f} ms  |  "
                f"Rust is {ratio:.2f}× {direction} than Fortran"
            )
        return "Incomplete benchmark data."


# ---------------------------------------------------------------------------
# Fortran timing driver (dgemm)
# ---------------------------------------------------------------------------

_DGEMM_BENCH_F = """\
      PROGRAM DGEMM_BENCH
      IMPLICIT NONE
      INTEGER M, N, K, LDA, LDB, LDC, I, J, REP, R
      DOUBLE PRECISION ALPHA, BETA
      PARAMETER (M=256, N=256, K=256)
      PARAMETER (LDA=M, LDB=K, LDC=M)
      PARAMETER (REP={reps})
      DOUBLE PRECISION A(LDA,K), B(LDB,N), C(LDC,N)
      DOUBLE PRECISION T1, T2, ELAPSED
      DO J = 1, K
        DO I = 1, M
          A(I,J) = DBLE(I+J) / DBLE(M*K)
        END DO
      END DO
      DO J = 1, N
        DO I = 1, K
          B(I,J) = DBLE(I-J+1) / DBLE(K*N)
        END DO
      END DO
      DO J = 1, N
        DO I = 1, M
          C(I,J) = 0.0D0
        END DO
      END DO
      ALPHA = 1.0D0
      BETA  = 0.0D0
      CALL CPU_TIME(T1)
      DO R = 1, REP
        CALL DGEMM('N','N',M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = (T2 - T1) * 1000.0D0 / DBLE(REP)
      WRITE(*,'(F20.6)') ELAPSED
      END
"""

_REPS = 10  # warmup + measurement repetitions


def _bench_fortran_dgemm(fortran_src_path: Path) -> Optional[float]:
    """Return median ms per dgemm call, or None on error."""
    support = _find_support_files(fortran_src_path.parent)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bench_f = tmp / "bench.f"
        bench_f.write_text(_DGEMM_BENCH_F.format(reps=_REPS))
        exe = tmp / "bench"
        srcs = [str(bench_f), str(fortran_src_path)] + [str(s) for s in support]
        result = subprocess.run(
            ["gfortran", "-O2", "-o", str(exe)] + srcs,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        if run.returncode != 0:
            return None
        try:
            return float(run.stdout.strip())
        except ValueError:
            return None


def _find_support_files(directory: Path) -> List[Path]:
    helpers = []
    for name in ("lsame.f", "xerbla.f", "LSAME.f", "XERBLA.f"):
        p = directory / name
        if p.exists():
            helpers.append(p)
    return helpers


# ---------------------------------------------------------------------------
# Rust timing (via cargo bench or a simple timing binary)
# ---------------------------------------------------------------------------

_RUST_BENCH_MAIN = """\
use std::time::Instant;

fn main() {{
    const M: usize = 256;
    const N: usize = 256;
    const K: usize = 256;
    const REPS: usize = {reps};

    let mut a = vec![0.0f64; M * K];
    let mut b = vec![0.0f64; K * N];
    let mut c = vec![0.0f64; M * N];
    for i in 0..M*K {{ a[i] = (i as f64 + 1.0) / (M * K) as f64; }}
    for i in 0..K*N {{ b[i] = (i as f64 + 1.0) / (K * N) as f64; }}

    let start = Instant::now();
    for _ in 0..REPS {{
        {crate_name}::dgemm::dgemm(
            b'N', b'N', M, N, K,
            1.0, &a, M, &b, K,
            0.0, &mut c, M,
        );
    }}
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    println!("{{:.6}}", elapsed_ms);
}}
"""


def _bench_rust_dgemm(crate_dir: Path) -> Optional[float]:
    """Return median ms per Rust dgemm call, or None if not available."""
    # Try to find a timing binary already built
    examples_dir = crate_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    bench_rs = examples_dir / "bench_dgemm.rs"
    crate_name = crate_dir.name

    bench_rs.write_text(_RUST_BENCH_MAIN.format(reps=_REPS, crate_name=crate_name))
    result = subprocess.run(
        ["cargo", "build", "--release", "--example", "bench_dgemm"],
        capture_output=True, text=True, cwd=crate_dir, timeout=120,
    )
    if result.returncode != 0:
        return None
    exe = crate_dir / "target" / "release" / "examples" / "bench_dgemm"
    if not exe.exists():
        return None
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    if run.returncode != 0:
        return None
    try:
        return float(run.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(
    function_name: str,
    fortran_source_path: Optional[Path],
    crate_dir: Optional[Path],
) -> BenchResult:
    """Benchmark *function_name* in both Fortran and Rust."""
    fn = function_name.upper()

    fortran_ms: Optional[float] = None
    rust_ms: Optional[float] = None
    details: List[str] = []

    if fn == "DGEMM":
        if fortran_source_path and fortran_source_path.exists():
            fortran_ms = _bench_fortran_dgemm(fortran_source_path)
            if fortran_ms is not None:
                details.append(f"  Fortran (gfortran -O2): {fortran_ms:.3f} ms/call (256×256×256)")
            else:
                details.append("  Fortran benchmark failed to compile/run.")

        if crate_dir and crate_dir.exists():
            rust_ms = _bench_rust_dgemm(crate_dir)
            if rust_ms is not None:
                details.append(f"  Rust (--release):        {rust_ms:.3f} ms/call (256×256×256)")
            else:
                details.append("  Rust benchmark skipped (dgemm function not yet callable).")
    else:
        details.append(f"  Benchmark not yet implemented for {fn}.")

    speedup: Optional[float] = None
    if fortran_ms and rust_ms and fortran_ms > 0:
        speedup = fortran_ms / rust_ms

    return BenchResult(
        function_name=fn,
        fortran_time_ms=fortran_ms,
        rust_time_ms=rust_ms,
        speedup=speedup,
        details=details,
    )
