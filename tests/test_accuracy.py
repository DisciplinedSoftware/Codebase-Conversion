"""Scenario-driven accuracy tests for converted Fortran→Rust functions.

For every function that the pipeline has converted (i.e. that appears in both
``output/fortran/blas/`` and the Rust crate ``output/rust/blas_converted/``),
this module runs the full :data:`~fortran_to_rust.test_scenarios.BLAS_*_SCENARIOS`
suite against the **original Fortran reference binary** and the **generated
Rust binary** in parallel, then asserts that their numerical outputs match
within :data:`~fortran_to_rust.test_harness._TOLERANCE`.

Running the tests
-----------------
You need a completed conversion run first::

    python convert.py --functions dgemm,daxpy,...

Then run the accuracy suite::

    pytest tests/test_accuracy.py -v

Marks
-----
``slow``
    Applied automatically to every test; use ``-m "not slow"`` to skip during
    quick CI checks.

Skip behaviour
--------------
* If ``output/`` is absent the whole module is skipped (see conftest.py).
* If a specific function is missing a Fortran source it is skipped per test.
* If ``cargo`` or ``gfortran`` are not installed, the relevant half of the
  comparison is skipped gracefully and the test still validates that the
  Fortran reference (or Rust side) ran without error.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from fortran_to_rust.test_harness import (
    AccuracyResult,
    _TOLERANCE,
    run_scenario_suite,
)
from fortran_to_rust.test_scenarios import (
    TestScenario,
    get_scenarios_for_function,
)

# ---------------------------------------------------------------------------
# Path constants (mirrors conftest.py; used at collection time before fixtures)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _REPO_ROOT / "output"
_FORTRAN_SRC_DIR = _OUTPUT_DIR / "fortran" / "blas"
_RUST_CRATE_DIR = _OUTPUT_DIR / "rust" / "blas_converted"
_DATASETS_DIR = _OUTPUT_DIR / "datasets"
_FORTRAN_REF_DIR = _OUTPUT_DIR / "fortran_drivers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_converted_functions() -> List[str]:
    """Return function names present in both Fortran source and Rust crate."""
    if not _FORTRAN_SRC_DIR.exists() or not _RUST_CRATE_DIR.exists():
        return []
    rust_src_dir = _RUST_CRATE_DIR / "src"
    fortran_stems = {p.stem.lower() for p in _FORTRAN_SRC_DIR.glob("*.f")}
    rust_stems = (
        {p.stem.lower() for p in rust_src_dir.glob("*.rs")
         if p.stem not in ("lib", "main")}
        if rust_src_dir.exists()
        else set()
    )
    return sorted(fortran_stems & rust_stems)


def _parse_routine(fortran_path: Path, fn_name: str):
    """Parse and return the FortranRoutine for *fn_name*, or None."""
    from fortran_to_rust.parser import parse_file

    routines = parse_file(fortran_path)
    matched = [r for r in routines if r.name.upper() == fn_name.upper()]
    return matched[0] if matched else None


def _all_scenario_params() -> List[Tuple[str, TestScenario]]:
    """Build the full (fn_name, scenario) parametrisation list at collection time."""
    params = []
    for fn in _discover_converted_functions():
        for sc in get_scenarios_for_function(fn):
            params.append((fn, sc))
    return params


_SCENARIO_PARAMS = _all_scenario_params()
_SCENARIO_IDS = [f"{fn}::{sc.name}" for fn, sc in _SCENARIO_PARAMS]


# ---------------------------------------------------------------------------
# Parametric accuracy test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize(
    "fn_name,scenario",
    _SCENARIO_PARAMS,
    ids=_SCENARIO_IDS,
)
def test_scenario_accuracy(
    fn_name: str,
    scenario: TestScenario,
    fortran_src_dir: Path,
    rust_crate_dir: Path,
    datasets_dir: Path,
    fortran_ref_dir: Path,
):
    """One scenario for one function: Fortran and Rust must agree numerically.

    The test is considered passing when:

    * Both implementations produce matching output within ``_TOLERANCE``.
    * OR both produce empty output for a documented no-op scenario
      (``scenario.expects_noop=True``).
    * OR the Rust binary is not yet available (compilation skipped) and the
      Fortran reference ran without error — in which case the test is a
      partial pass that documents Fortran behaviour for later comparison.
    """
    fortran_path = fortran_src_dir / f"{fn_name.lower()}.f"
    if not fortran_path.exists():
        pytest.skip(f"Fortran source not found: {fortran_path}")

    routine = _parse_routine(fortran_path, fn_name)
    if routine is None:
        pytest.skip(f"Could not parse routine '{fn_name}' from {fortran_path}")

    results: List[AccuracyResult] = run_scenario_suite(
        function_name=fn_name,
        fortran_source_path=fortran_path,
        crate_dir=rust_crate_dir,
        scenarios=[scenario],
        routine=routine,
        fortran_ref_dir=fortran_ref_dir,
        datasets_dir=datasets_dir,
    )

    assert results, "run_scenario_suite returned an empty result list"
    result = results[0]

    detail_text = "\n".join(result.details or [])
    msg = (
        f"Scenario '{scenario.name}' FAILED for '{fn_name}'.\n"
        f"  Description: {scenario.description}\n"
        f"  Error: {result.error_message or '(none)'}\n"
        f"  Details:\n{detail_text}"
    )

    assert result.passed, msg


# ---------------------------------------------------------------------------
# Session-level test: verify that every converted function compiles its
# Fortran reference driver without error (baseline health check).
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not _FORTRAN_SRC_DIR.exists(),
    reason="output/ not found — run python convert.py first",
)
def test_fortran_reference_compiles(
    converted_function_names: List[str],
    fortran_src_dir: Path,
    datasets_dir: Path,
    fortran_ref_dir: Path,
):
    """All converted functions must have a compilable Fortran reference driver.

    Uses the first random-seed scenario from the Level-3 scenario set (the
    most conservative default) to verify that the Fortran reference source
    compiles and produces output without error.  This is the minimal
    correctness bar: the original Fortran code must be runnable before we
    compare it against any Rust output.
    """
    from fortran_to_rust.test_scenarios import BLAS_LEVEL3_SCENARIOS

    first_scenario = BLAS_LEVEL3_SCENARIOS[0]  # random_seed_0
    failures: List[str] = []

    for fn_name in converted_function_names:
        fortran_path = fortran_src_dir / f"{fn_name.lower()}.f"
        if not fortran_path.exists():
            continue
        routine = _parse_routine(fortran_path, fn_name)
        if routine is None:
            continue

        results = run_scenario_suite(
            function_name=fn_name,
            fortran_source_path=fortran_path,
            crate_dir=None,          # Fortran-only check
            scenarios=[first_scenario],
            routine=routine,
            fortran_ref_dir=fortran_ref_dir,
            datasets_dir=datasets_dir,
        )
        if results and not results[0].passed:
            failures.append(
                f"  {fn_name}: {results[0].error_message or 'unknown failure'}"
            )

    assert not failures, (
        "The following functions failed their Fortran reference check:\n"
        + "\n".join(failures)
    )

